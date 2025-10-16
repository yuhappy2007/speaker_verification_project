# -*- coding: utf-8 -*-
"""
阶段1：实现SupCon损失替换Triplet Loss
基于原有的models_fixed.py进行改进
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (SupCon)
    参考论文: "Supervised Contrastive Learning" (NeurIPS 2020)

    优势：
    1. 同时利用batch内所有正负样本对
    2. 训练更稳定，不需要复杂的难样本挖掘
    3. 对噪声更鲁棒
    """

    def __init__(self, temperature=0.07, base_temperature=0.07):
        """
        Args:
            temperature: 温度参数，控制分布的尖锐程度
                        - 较小的值(0.05-0.1)使模型更关注困难样本
                        - 较大的值(0.5-1.0)使训练更平滑
            base_temperature: 基础温度，用于归一化
        """
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels):
        """
        计算Supervised Contrastive Loss

        Args:
            features: [batch_size, embedding_dim] - L2归一化的嵌入向量
            labels: [batch_size] - 说话人标签

        Returns:
            loss: 标量张量
        """
        device = features.device
        batch_size = features.shape[0]

        # 确保features已经L2归一化
        features = F.normalize(features, p=2, dim=1)

        # 计算余弦相似度矩阵: [batch_size, batch_size]
        similarity_matrix = torch.matmul(features, features.T)

        # 创建mask：标记哪些样本对是同一说话人（正样本对）
        # labels: [batch_size] -> [batch_size, 1] -> [batch_size, batch_size]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # 去除对角线（自己和自己不算正样本对）
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # 计算log_prob
        # 除以温度参数
        similarity_matrix = similarity_matrix / self.temperature

        # 为了数值稳定性，减去最大值
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()

        # 计算exp(similarity)
        exp_logits = torch.exp(logits) * logits_mask

        # 计算log(sum(exp(similarity)))用于分母
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # 计算每个anchor的正样本对的平均log_prob
        # mask.sum(1)是每个anchor的正样本数量
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)

        # 损失是负的平均log probability
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss


class RobustEmbeddingMLP(nn.Module):
    """
    保持与原论文一致的MLP结构
    3层MLP用于融合noisy和enhanced embeddings
    """

    def __init__(self, embedding_dim=192):
        super().__init__()

        self.fc1 = nn.Linear(embedding_dim * 2, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, embedding_dim)
        self.fc3 = nn.Linear(embedding_dim, embedding_dim)

        self.relu = nn.ReLU()

        # Xavier初始化
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
        x = torch.cat([noisy_emb, enhanced_emb], dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        # L2归一化（对SupCon很重要！）
        x = F.normalize(x, p=2, dim=1)

        return x


class SupConTrainer:
    """
    使用SupCon损失的训练器
    """

    def __init__(self, embedding_dim=192, device='cuda', temperature=0.07):
        self.device = device
        self.embedding_dim = embedding_dim

        # 创建MLP
        self.mlp = RobustEmbeddingMLP(embedding_dim).to(device)

        # 使用SupCon损失
        self.criterion = SupConLoss(temperature=temperature)

        self.optimizer = None

        print(f'✅ SupCon Trainer initialized on {device}')
        print(f'📊 MLP parameters: {sum(p.numel() for p in self.mlp.parameters()):,}')
        print(f'🌡️  Temperature: {temperature}')

    def train_step(self, noisy_embs, enhanced_embs, labels):
        """
        执行一次训练步骤

        Args:
            noisy_embs: [batch_size, embedding_dim] - 噪声嵌入
            enhanced_embs: [batch_size, embedding_dim] - 增强嵌入
            labels: [batch_size] - 说话人标签

        Returns:
            loss: float - 损失值
        """
        self.optimizer.zero_grad()

        # 通过MLP融合得到鲁棒嵌入
        robust_embs = self.mlp(noisy_embs, enhanced_embs)

        # 计算SupCon损失
        loss = self.criterion(robust_embs, labels)

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def extract_embedding(self, noisy_emb, enhanced_emb):
        """
        推理时提取鲁棒嵌入

        Args:
            noisy_emb: [1, embedding_dim] or [embedding_dim]
            enhanced_emb: [1, embedding_dim] or [embedding_dim]

        Returns:
            robust_emb: [1, embedding_dim] 或 [embedding_dim]
        """
        self.mlp.eval()
        with torch.no_grad():
            # 确保是2D张量
            if noisy_emb.dim() == 1:
                noisy_emb = noisy_emb.unsqueeze(0)
            if enhanced_emb.dim() == 1:
                enhanced_emb = enhanced_emb.unsqueeze(0)

            robust_emb = self.mlp(noisy_emb, enhanced_emb)

        return robust_emb


# ============ 对比：Triplet Loss版本（用于baseline对比）============

class TripletLossFixed(nn.Module):
    """原论文的Triplet Loss（修复版）"""

    def __init__(self, margin=0.25, min_neg_dist=0.4):
        super().__init__()
        self.margin = margin
        self.min_neg_dist = min_neg_dist

    def cosine_distance(self, x, y):
        cos_sim = F.cosine_similarity(x, y, dim=1)
        return 1 - cos_sim

    def forward(self, anchor, positive, negative):
        pos_dist = self.cosine_distance(anchor, positive)
        neg_dist = self.cosine_distance(anchor, negative)

        triplet_loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)
        neg_constraint = torch.clamp(self.min_neg_dist - neg_dist, min=0.0)

        total_loss = triplet_loss + 0.5 * neg_constraint
        return total_loss.mean()


class TripletTrainer:
    """原论文的Triplet Loss训练器"""

    def __init__(self, embedding_dim=192, device='cuda',
                 triplet_margin=0.25, min_neg_dist=0.4):
        self.device = device
        self.embedding_dim = embedding_dim

        self.mlp = RobustEmbeddingMLP(embedding_dim).to(device)
        self.criterion = TripletLossFixed(
            margin=triplet_margin,
            min_neg_dist=min_neg_dist
        )
        self.optimizer = None

        print(f'✅ Triplet Trainer initialized on {device}')
        print(f'📊 MLP parameters: {sum(p.numel() for p in self.mlp.parameters()):,}')

    def train_step(self, anchor_robust, positive_robust, negative_robust):
        self.optimizer.zero_grad()
        loss = self.criterion(anchor_robust, positive_robust, negative_robust)
        loss.backward()
        self.optimizer.step()
        return loss.item()


# ============ 使用示例 ============

def example_usage():
    """使用示例代码"""

    # 假设参数
    batch_size = 32
    embedding_dim = 192
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. 创建SupCon训练器
    supcon_trainer = SupConTrainer(
        embedding_dim=embedding_dim,
        device=device,
        temperature=0.07  # 可调整：0.05-0.1
    )

    # 2. 设置优化器
    supcon_trainer.optimizer = torch.optim.AdamW(
        supcon_trainer.mlp.parameters(),
        lr=1e-3,
        weight_decay=1e-4
    )

    # 3. 模拟训练数据
    noisy_embs = torch.randn(batch_size, embedding_dim).to(device)
    enhanced_embs = torch.randn(batch_size, embedding_dim).to(device)
    labels = torch.randint(0, 10, (batch_size,)).to(device)  # 10个说话人

    # 4. 训练一步
    loss = supcon_trainer.train_step(noisy_embs, enhanced_embs, labels)
    print(f'SupCon Loss: {loss:.4f}')

    # 5. 推理
    test_noisy = torch.randn(1, embedding_dim).to(device)
    test_enhanced = torch.randn(1, embedding_dim).to(device)
    robust_emb = supcon_trainer.extract_embedding(test_noisy, test_enhanced)
    print(f'Robust embedding shape: {robust_emb.shape}')


if __name__ == '__main__':
    example_usage()