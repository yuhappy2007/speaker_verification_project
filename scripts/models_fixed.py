# -*- coding: utf-8 -*-
"""
修复版模型 - 添加约束防止embedding collapse
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLossFixed(nn.Module):
    """
    修复版Triplet Loss

    问题：标准Triplet Loss可能导致模型把所有embedding映射到同一点
    解决：添加约束确保负样本距离足够大
    """

    def __init__(self, margin=0.25, min_neg_dist=0.4):
        super().__init__()
        self.margin = margin
        self.min_neg_dist = min_neg_dist  # 不同说话人最小距离

    def cosine_distance(self, x, y):
        """计算cosine distance: d = 1 - cos(θ)"""
        cos_sim = F.cosine_similarity(x, y, dim=1)
        return 1 - cos_sim

    def forward(self, anchor, positive, negative):
        """
        Args:
            anchor: [batch_size, embedding_dim]
            positive: [batch_size, embedding_dim] (same speaker as anchor)
            negative: [batch_size, embedding_dim] (different speaker)
        """
        # 计算距离
        pos_dist = self.cosine_distance(anchor, positive)
        neg_dist = self.cosine_distance(anchor, negative)

        # 原始triplet loss
        # L = max(0, d(A,P) - d(A,N) + margin)
        triplet_loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)

        # ✅ 添加约束：确保负样本距离足够大
        # 如果neg_dist < min_neg_dist，则惩罚
        neg_constraint = torch.clamp(self.min_neg_dist - neg_dist, min=0.0)

        # ✅ 可选：确保正样本距离足够小（通常triplet loss已经做了）
        # max_pos_dist = 0.3
        # pos_constraint = torch.clamp(pos_dist - max_pos_dist, min=0.0)

        # 组合loss
        # neg_constraint的权重可以调整（0.5是经验值）
        total_loss = triplet_loss + 0.5 * neg_constraint

        return total_loss.mean()


class RobustEmbeddingMLP(nn.Module):
    """
    3层MLP用于融合noisy和enhanced embeddings

    架构：
    - Layer 1: 2N × 1 → N × 1 (ReLU)
    - Layer 2: N × 1 → N × 1 (ReLU)
    - Layer 3: N × 1 → N × 1 (L2 normalize)
    """

    def __init__(self, embedding_dim=192):
        super().__init__()

        # 三层全连接
        self.fc1 = nn.Linear(embedding_dim * 2, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, embedding_dim)
        self.fc3 = nn.Linear(embedding_dim, embedding_dim)

        # 激活函数
        self.relu = nn.ReLU()

        # Xavier初始化（帮助训练稳定）
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, noisy_emb, enhanced_emb):
        """
        Args:
            noisy_emb: [batch_size, embedding_dim]
            enhanced_emb: [batch_size, embedding_dim]

        Returns:
            robust_emb: [batch_size, embedding_dim] (L2 normalized)
        """
        # 拼接两个embeddings
        x = torch.cat([noisy_emb, enhanced_emb], dim=1)  # [batch, 2*192]

        # 三层MLP
        x = self.relu(self.fc1(x))  # [batch, 192]
        x = self.relu(self.fc2(x))  # [batch, 192]
        x = self.fc3(x)  # [batch, 192]

        # L2归一化（用于cosine distance）
        x = F.normalize(x, p=2, dim=1)

        return x


class SiameseTrainerFixed:
    """
    修复版Siamese训练器
    使用修复后的Triplet Loss
    """

    def __init__(self, embedding_dim=192, device='cuda',
                 triplet_margin=0.25, min_neg_dist=0.4):
        self.device = device
        self.embedding_dim = embedding_dim

        # 创建MLP
        self.mlp = RobustEmbeddingMLP(embedding_dim).to(device)

        # 使用修复版的Triplet Loss
        self.criterion = TripletLossFixed(
            margin=triplet_margin,
            min_neg_dist=min_neg_dist
        )

        # 优化器（稍后在训练脚本中设置）
        self.optimizer = None

        print(f'Trainer initialized on {device}')
        print(f'MLP parameters: {sum(p.numel() for p in self.mlp.parameters())}')

    def train_step(self, anchor_robust, positive_robust, negative_robust):
        """执行一次训练步骤"""
        self.optimizer.zero_grad()
        loss = self.criterion(anchor_robust, positive_robust, negative_robust)
        loss.backward()
        self.optimizer.step()
        return loss.item()