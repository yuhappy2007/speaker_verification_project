# """
# 对比损失函数 (Contrastive Loss)
# 用于置信度网络训练
# """
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class ContrastiveLoss(nn.Module):
#     """
#     对比损失函数
#
#     用途: 训练置信度网络，使同一说话人的嵌入距离近，不同说话人的嵌入距离远
#
#     公式:
#     - 正样本对: L_pos = d(x1, x2)^2
#     - 负样本对: L_neg = max(0, margin - d(x1, x2))^2
#     - 总损失: L = (1-y) * L_pos + y * L_neg
#
#     其中:
#     - y=0表示同一说话人(正样本对)
#     - y=1表示不同说话人(负样本对)
#     - margin是间隔超参数
#     """
#
#     def __init__(self, margin=1.0, distance_type='euclidean'):
#         """
#         Args:
#             margin: 负样本对的最小距离间隔
#             distance_type: 'euclidean' 或 'cosine'
#         """
#         super().__init__()
#         self.margin = margin
#         self.distance_type = distance_type
#
#         print(f'📐 Contrastive Loss initialized')
#         print(f'   Margin: {margin}')
#         print(f'   Distance: {distance_type}')
#
#     def compute_distance(self, x1, x2):
#         """
#         计算两个嵌入之间的距离
#
#         Args:
#             x1, x2: [batch, embedding_dim]
#
#         Returns:
#             distance: [batch]
#         """
#         if self.distance_type == 'euclidean':
#             # 欧氏距离
#             distance = torch.sqrt(torch.sum((x1 - x2) ** 2, dim=1) + 1e-8)
#         elif self.distance_type == 'cosine':
#             # 余弦距离 = 1 - 余弦相似度
#             cos_sim = F.cosine_similarity(x1, x2, dim=1)
#             distance = 1 - cos_sim
#         else:
#             raise ValueError(f'Unknown distance type: {self.distance_type}')
#
#         return distance
#
#     def forward(self, x1, x2, labels):
#         """
#         计算对比损失
#
#         Args:
#             x1: [batch, embedding_dim]，第一个嵌入
#             x2: [batch, embedding_dim]，第二个嵌入
#             labels: [batch]，0表示同一说话人，1表示不同说话人
#
#         Returns:
#             loss: scalar
#             stats: dict，统计信息
#         """
#         # 计算距离
#         distance = self.compute_distance(x1, x2)
#
#         # 分离正负样本
#         positive_mask = (labels == 0).float()  # 同一说话人
#         negative_mask = (labels == 1).float()  # 不同说话人
#
#         # 正样本损失：距离越小越好
#         positive_loss = positive_mask * distance ** 2
#
#         # 负样本损失：距离大于margin才好
#         negative_loss = negative_mask * torch.clamp(self.margin - distance, min=0) ** 2
#
#         # 总损失
#         total_loss = (positive_loss + negative_loss).mean()
#
#         # 统计信息
#         with torch.no_grad():
#             num_positives = positive_mask.sum().item()
#             num_negatives = negative_mask.sum().item()
#
#             if num_positives > 0:
#                 avg_pos_dist = (distance * positive_mask).sum().item() / num_positives
#             else:
#                 avg_pos_dist = 0.0
#
#             if num_negatives > 0:
#                 avg_neg_dist = (distance * negative_mask).sum().item() / num_negatives
#             else:
#                 avg_neg_dist = 0.0
#
#         stats = {
#             'loss': total_loss.item(),
#             'avg_positive_distance': avg_pos_dist,
#             'avg_negative_distance': avg_neg_dist,
#             'num_positives': num_positives,
#             'num_negatives': num_negatives
#         }
#
#         return total_loss, stats
#
#
# class TripletContrastiveLoss(nn.Module):
#     """
#     三元组对比损失
#
#     适用于triplet数据：anchor, positive, negative
#
#     公式:
#     L = max(0, d(anchor, positive) - d(anchor, negative) + margin)
#
#     目标：使d(anchor, positive) + margin < d(anchor, negative)
#     """
#
#     def __init__(self, margin=0.5, distance_type='euclidean'):
#         super().__init__()
#         self.margin = margin
#         self.distance_type = distance_type
#
#         print(f'📐 Triplet Contrastive Loss initialized')
#         print(f'   Margin: {margin}')
#         print(f'   Distance: {distance_type}')
#
#     def compute_distance(self, x1, x2):
#         """计算距离"""
#         if self.distance_type == 'euclidean':
#             distance = torch.sqrt(torch.sum((x1 - x2) ** 2, dim=1) + 1e-8)
#         elif self.distance_type == 'cosine':
#             cos_sim = F.cosine_similarity(x1, x2, dim=1)
#             distance = 1 - cos_sim
#         else:
#             raise ValueError(f'Unknown distance type: {self.distance_type}')
#
#         return distance
#
#     def forward(self, anchor, positive, negative):
#         """
#         计算三元组损失
#
#         Args:
#             anchor: [batch, embedding_dim]
#             positive: [batch, embedding_dim]，同一说话人
#             negative: [batch, embedding_dim]，不同说话人
#
#         Returns:
#             loss: scalar
#             stats: dict
#         """
#         # 计算距离
#         pos_distance = self.compute_distance(anchor, positive)
#         neg_distance = self.compute_distance(anchor, negative)
#
#         # Triplet loss
#         loss = torch.clamp(pos_distance - neg_distance + self.margin, min=0)
#         total_loss = loss.mean()
#
#         # 统计信息
#         with torch.no_grad():
#             avg_pos_dist = pos_distance.mean().item()
#             avg_neg_dist = neg_distance.mean().item()
#             num_hard_triplets = (loss > 0).sum().item()
#
#         stats = {
#             'loss': total_loss.item(),
#             'avg_positive_distance': avg_pos_dist,
#             'avg_negative_distance': avg_neg_dist,
#             'num_hard_triplets': num_hard_triplets,
#             'total_triplets': anchor.shape[0]
#         }
#
#         return total_loss, stats
#
#
# def test_contrastive_loss():
#     """测试对比损失"""
#     print('=' * 80)
#     print('🧪 Testing Contrastive Loss')
#     print('=' * 80)
#
#     batch_size = 16
#     embedding_dim = 192
#
#     # === 测试1: 标准对比损失 ===
#     print('\n[Test 1] Standard Contrastive Loss')
#     contrastive = ContrastiveLoss(margin=1.0, distance_type='euclidean')
#
#     x1 = torch.randn(batch_size, embedding_dim)
#     x2 = torch.randn(batch_size, embedding_dim)
#     labels = torch.randint(0, 2, (batch_size,))  # 随机0或1
#
#     loss, stats = contrastive(x1, x2, labels)
#
#     print(f'Loss: {loss.item():.4f}')
#     print(f'Avg positive distance: {stats["avg_positive_distance"]:.4f}')
#     print(f'Avg negative distance: {stats["avg_negative_distance"]:.4f}')
#     print(f'Num positives: {stats["num_positives"]}')
#     print(f'Num negatives: {stats["num_negatives"]}')
#
#     # === 测试2: 三元组损失 ===
#     print('\n[Test 2] Triplet Contrastive Loss')
#     triplet_loss = TripletContrastiveLoss(margin=0.5, distance_type='cosine')
#
#     anchor = F.normalize(torch.randn(batch_size, embedding_dim), p=2, dim=1)
#     positive = F.normalize(torch.randn(batch_size, embedding_dim), p=2, dim=1)
#     negative = F.normalize(torch.randn(batch_size, embedding_dim), p=2, dim=1)
#
#     loss, stats = triplet_loss(anchor, positive, negative)
#
#     print(f'Loss: {loss.item():.4f}')
#     print(f'Avg positive distance: {stats["avg_positive_distance"]:.4f}')
#     print(f'Avg negative distance: {stats["avg_negative_distance"]:.4f}')
#     print(f'Hard triplets: {stats["num_hard_triplets"]}/{stats["total_triplets"]}')
#
#     # === 测试3: 梯度测试 ===
#     print('\n[Test 3] Gradient Flow Test')
#     x1_grad = torch.randn(batch_size, embedding_dim, requires_grad=True)
#     x2_grad = torch.randn(batch_size, embedding_dim, requires_grad=True)
#     labels_grad = torch.randint(0, 2, (batch_size,))
#
#     loss, _ = contrastive(x1_grad, x2_grad, labels_grad)
#     loss.backward()
#
#     print(f'x1 gradient: {x1_grad.grad is not None}')
#     print(f'x1 gradient norm: {x1_grad.grad.norm().item():.6f}')
#     print(f'x2 gradient: {x2_grad.grad is not None}')
#     print(f'x2 gradient norm: {x2_grad.grad.norm().item():.6f}')
#
#     print('\n' + '=' * 80)
#     print('✅ All contrastive loss tests passed!')
#     print('=' * 80)
#
#
# if __name__ == '__main__':
#     test_contrastive_loss()
"""
Contrastive Loss for Speaker Verification

实现监督对比学习损失，用于替代 Triplet Loss。
适用于说话人验证任务，支持多正样本和多负样本。

参考文献：
- Supervised Contrastive Learning (Khosla et al., NeurIPS 2020)
- 适配为说话人验证场景

使用方法：
    loss_fn = ContrastiveLoss(temperature=0.07)
    embeddings = model(audio)  # [batch, embedding_dim]
    labels = speaker_labels    # [batch]
    loss = loss_fn(embeddings, labels)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    监督对比学习损失（Supervised Contrastive Loss）

    用于说话人验证任务，鼓励同一说话人的嵌入相似，
    不同说话人的嵌入分离。

    公式：
        L = -1/|P(i)| * Σ_{p∈P(i)} log[ exp(z_i·z_p/τ) / Σ_{a∈A(i)} exp(z_i·z_a/τ) ]

    其中：
        - P(i): 与样本 i 同类的样本集合（不包括 i 自己）
        - A(i): 除了 i 之外的所有样本
        - τ: 温度参数
        - z_i·z_p: 归一化后的余弦相似度

    Args:
        temperature: 温度参数，控制分布的平滑度（默认 0.07）
        contrast_mode: 对比模式
            - 'all': 使用所有样本作为对比
            - 'one': 只使用一个正样本（类似 triplet loss）
        base_temperature: 基础温度（默认 0.07）

    输入：
        embeddings: [batch_size, embedding_dim] - 嵌入向量
        labels: [batch_size] - 类别标签（说话人 ID）
        mask: [batch_size, batch_size] - 可选，手动指定正样本对

    输出：
        loss: 标量张量
    """

    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, embeddings, labels=None, mask=None):
        """
        计算 Contrastive Loss

        Args:
            embeddings: [batch_size, embedding_dim] 或 [batch_size, n_views, embedding_dim]
            labels: [batch_size] - 类别标签
            mask: [batch_size, batch_size] - 可选的正样本对掩码

        Returns:
            loss: 标量张量
        """
        device = embeddings.device

        # 处理多视图情况（例如：noisy + enhanced）
        if len(embeddings.shape) == 3:
            # [batch_size, n_views, embedding_dim]
            batch_size, n_views, embedding_dim = embeddings.shape
            embeddings = embeddings.view(batch_size * n_views, embedding_dim)

            if labels is not None:
                labels = labels.contiguous().view(-1, 1)
                labels = labels.repeat(n_views, 1).view(-1)
        else:
            # [batch_size, embedding_dim]
            batch_size = embeddings.shape[0]
            n_views = 1

        # L2 归一化（余弦相似度）
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # 计算相似度矩阵: [batch_size, batch_size]
        similarity_matrix = torch.matmul(embeddings, embeddings.T)

        # 创建正样本对掩码
        if mask is None:
            if labels is None:
                raise ValueError("Either labels or mask must be provided")

            # 基于标签创建掩码
            labels = labels.contiguous().view(-1, 1)
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        # 移除对角线（自己与自己的相似度）
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * n_views).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # 计算 log_prob
        # 分子：exp(sim(z_i, z_p) / τ) for all positive pairs
        # 分母：Σ exp(sim(z_i, z_a) / τ) for all a ≠ i

        # 缩放相似度
        logits = similarity_matrix / self.temperature

        # 数值稳定性：减去最大值
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # 计算 exp
        exp_logits = torch.exp(logits) * logits_mask

        # 分母：所有样本的 exp 之和
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # 计算每个样本的正样本对数量
        mask_sum = mask.sum(1)

        # 避免除以0
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)

        # 计算平均 log-likelihood
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum

        # 损失：负 log-likelihood
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos

        # 对所有样本求平均
        loss = loss.view(n_views, batch_size).mean()

        return loss


class TripletContrastiveLoss(nn.Module):
    """
    结合 Triplet Loss 和 Contrastive Loss 的混合损失

    L = α * L_triplet + (1-α) * L_contrastive

    Args:
        margin: Triplet loss 的 margin
        temperature: Contrastive loss 的温度
        alpha: 权重系数 (0-1)，alpha=1 时只有 triplet，alpha=0 时只有 contrastive
    """

    def __init__(self, margin=0.3, temperature=0.07, alpha=0.5):
        super().__init__()
        self.margin = margin
        self.contrastive = ContrastiveLoss(temperature=temperature)
        self.alpha = alpha

    def forward(self, embeddings, labels):
        """
        计算混合损失

        Args:
            embeddings: [batch_size, embedding_dim]
            labels: [batch_size]

        Returns:
            loss: 标量
            losses_dict: 包含各部分损失的字典
        """
        # Contrastive Loss
        loss_contrastive = self.contrastive(embeddings, labels)

        # Triplet Loss (硬负样本挖掘)
        loss_triplet = self._compute_triplet_loss(embeddings, labels)

        # 混合
        loss = self.alpha * loss_triplet + (1 - self.alpha) * loss_contrastive

        losses_dict = {
            'total': loss.item(),
            'triplet': loss_triplet.item(),
            'contrastive': loss_contrastive.item()
        }

        return loss, losses_dict

    def _compute_triplet_loss(self, embeddings, labels):
        """计算 Triplet Loss（硬负样本挖掘）"""
        # L2 归一化
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # 计算距离矩阵
        dist_matrix = torch.cdist(embeddings, embeddings, p=2)

        batch_size = embeddings.shape[0]
        losses = []

        for i in range(batch_size):
            # 锚点
            anchor_label = labels[i]

            # 正样本：同类且不是自己
            pos_mask = (labels == anchor_label) & (torch.arange(batch_size).to(labels.device) != i)
            if pos_mask.sum() == 0:
                continue

            # 负样本：不同类
            neg_mask = labels != anchor_label
            if neg_mask.sum() == 0:
                continue

            # 硬正样本（最远的正样本）
            pos_dists = dist_matrix[i][pos_mask]
            hardest_pos_dist = pos_dists.max()

            # 硬负样本（最近的负样本）
            neg_dists = dist_matrix[i][neg_mask]
            hardest_neg_dist = neg_dists.min()

            # Triplet Loss
            loss = torch.clamp(hardest_pos_dist - hardest_neg_dist + self.margin, min=0.0)
            losses.append(loss)

        if len(losses) == 0:
            return torch.tensor(0.0).to(embeddings.device)

        return torch.stack(losses).mean()


def test_contrastive_loss():
    """测试 Contrastive Loss"""
    print('=' * 80)
    print('🧪 Testing Contrastive Loss')
    print('=' * 80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 测试配置
    batch_size = 32
    embedding_dim = 192
    num_speakers = 8

    print(f'\n[1/4] Creating test data...')
    print(f'   Batch size: {batch_size}')
    print(f'   Embedding dim: {embedding_dim}')
    print(f'   Num speakers: {num_speakers}')

    # 创建测试数据
    embeddings = torch.randn(batch_size, embedding_dim).to(device)
    labels = torch.randint(0, num_speakers, (batch_size,)).to(device)

    print(f'   Embeddings shape: {embeddings.shape}')
    print(f'   Labels shape: {labels.shape}')
    print(f'   Unique speakers: {labels.unique().numel()}')

    # 测试 Contrastive Loss
    print(f'\n[2/4] Testing ContrastiveLoss...')
    loss_fn = ContrastiveLoss(temperature=0.07).to(device)

    loss = loss_fn(embeddings, labels)
    print(f'   Loss: {loss.item():.6f}')
    print(f'   ✅ ContrastiveLoss works!')

    # 测试梯度
    print(f'\n[3/4] Testing gradient flow...')
    embeddings.requires_grad = True
    loss = loss_fn(embeddings, labels)
    loss.backward()

    print(f'   Embeddings has gradient: {embeddings.grad is not None}')
    if embeddings.grad is not None:
        print(f'   Gradient norm: {embeddings.grad.norm().item():.6f}')
        print(f'   ✅ Gradient flows correctly!')

    # 测试 TripletContrastiveLoss
    print(f'\n[4/4] Testing TripletContrastiveLoss...')
    embeddings = torch.randn(batch_size, embedding_dim).to(device)
    mixed_loss_fn = TripletContrastiveLoss(margin=0.3, temperature=0.07, alpha=0.5).to(device)

    loss, losses_dict = mixed_loss_fn(embeddings, labels)
    print(f'   Total loss: {losses_dict["total"]:.6f}')
    print(f'   Triplet loss: {losses_dict["triplet"]:.6f}')
    print(f'   Contrastive loss: {losses_dict["contrastive"]:.6f}')
    print(f'   ✅ TripletContrastiveLoss works!')

    print('\n' + '=' * 80)
    print('✅ All tests passed!')
    print('=' * 80)

    print('\n💡 Usage examples:')
    print('\n1. Pure Contrastive Loss:')
    print('   loss_fn = ContrastiveLoss(temperature=0.07)')
    print('   loss = loss_fn(embeddings, labels)')

    print('\n2. Mixed Loss:')
    print('   loss_fn = TripletContrastiveLoss(margin=0.3, temperature=0.07, alpha=0.5)')
    print('   loss, losses_dict = loss_fn(embeddings, labels)')

    print('\n3. Multi-view (noisy + enhanced):')
    print('   embeddings = torch.stack([emb_noisy, emb_enhanced], dim=1)  # [batch, 2, dim]')
    print('   loss = loss_fn(embeddings, labels)')


if __name__ == '__main__':
    test_contrastive_loss()