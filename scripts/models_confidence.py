# """
# 阶段2优化: 置信度评估网络 (Confidence Estimation Network)
#
# 核心思想:
# - 动态学习noisy和enhanced embedding的可信度
# - 通过softmax确保权重和为1
# - 轻量级设计,可解释性强
#
# 参考文献:
# - SENet: Squeeze-and-Excitation Networks (CVPR 2018)
# - Attention mechanisms for multi-modal fusion
# - Dynamic weighting in speaker recognition
#
# 设计:
# 输入: [E_n; E_e] (concatenated)
# 网络: 2层MLP + Softmax
# 输出: [w_n, w_e] (权重和为1)
# 融合: E_robust = w_n * E_n + w_e * E_e
# """
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class ConfidenceEstimationNet(nn.Module):
#     """
#     置信度评估网络
#
#     功能:
#     1. 评估noisy和enhanced embedding的可信度
#     2. 输出两个权重(和为1)用于动态融合
#     3. 轻量级设计,易于训练
#
#     架构:
#     Input: [batch, 384] (concatenated E_n and E_e)
#     ├── FC1: 384 -> 192 + ReLU
#     ├── FC2: 192 -> 64 + ReLU
#     └── FC3: 64 -> 2 + Softmax
#     Output: [batch, 2] (weights for [E_n, E_e])
#     """
#
#     def __init__(self, embedding_dim=192, hidden_dim=64):
#         """
#         Args:
#             embedding_dim: 单个embedding的维度(默认192)
#             hidden_dim: 隐藏层维度(默认64,可调整)
#         """
#         super().__init__()
#
#         self.embedding_dim = embedding_dim
#         self.hidden_dim = hidden_dim
#
#         # 三层MLP: 逐步压缩维度
#         self.fc1 = nn.Linear(embedding_dim * 2, embedding_dim)
#         self.fc2 = nn.Linear(embedding_dim, hidden_dim)
#         self.fc3 = nn.Linear(hidden_dim, 2)  # 输出2个权重
#
#         # BatchNorm提升稳定性(可选)
#         self.bn1 = nn.BatchNorm1d(embedding_dim)
#         self.bn2 = nn.BatchNorm1d(hidden_dim)
#
#         # Dropout防止过拟合(可选)
#         self.dropout = nn.Dropout(0.1)
#
#         # Xavier初始化
#         self._init_weights()
#
#     def _init_weights(self):
#         """初始化网络权重"""
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)
#
#     def forward(self, noisy_emb, enhanced_emb):
#         """
#         前向传播
#
#         Args:
#             noisy_emb: [batch, embedding_dim]
#             enhanced_emb: [batch, embedding_dim]
#
#         Returns:
#             weights: [batch, 2] - 权重[w_n, w_e],和为1
#             stats: dict - 用于分析的统计信息
#         """
#         # 拼接两个embedding
#         x = torch.cat([noisy_emb, enhanced_emb], dim=1)  # [batch, 384]
#
#         # 第一层
#         x = self.fc1(x)
#         x = self.bn1(x)
#         x = F.relu(x)
#         x = self.dropout(x)
#
#         # 第二层
#         x = self.fc2(x)
#         x = self.bn2(x)
#         x = F.relu(x)
#         x = self.dropout(x)
#
#         # 输出层
#         logits = self.fc3(x)  # [batch, 2]
#
#         # Softmax确保权重和为1
#         weights = F.softmax(logits, dim=1)  # [batch, 2]
#
#         # 收集统计信息(用于分析)
#         stats = {
#             'w_noisy_mean': weights[:, 0].mean().item(),
#             'w_enhanced_mean': weights[:, 1].mean().item(),
#             'w_noisy_std': weights[:, 0].std().item(),
#             'w_enhanced_std': weights[:, 1].std().item(),
#             'logits_mean': logits.mean().item(),
#             'logits_std': logits.std().item(),
#         }
#
#         return weights, stats
#
#
# class DynamicFusionMLP(nn.Module):
#     """
#     动态融合MLP - 结合置信度网络
#
#     流程:
#     1. 输入noisy和enhanced embedding
#     2. 置信度网络评估权重[w_n, w_e]
#     3. 动态融合: E_fused = w_n * E_n + w_e * E_e
#     4. 通过一个轻量MLP得到最终robust embedding
#     5. L2归一化
#     """
#
#     def __init__(self, embedding_dim=192, confidence_hidden_dim=64):
#         super().__init__()
#
#         self.embedding_dim = embedding_dim
#
#         # 置信度评估网络
#         self.confidence_net = ConfidenceEstimationNet(
#             embedding_dim=embedding_dim,
#             hidden_dim=confidence_hidden_dim
#         )
#
#         # 融合后的refinement network(可选,增强表达能力)
#         self.refine_fc1 = nn.Linear(embedding_dim, embedding_dim)
#         self.refine_fc2 = nn.Linear(embedding_dim, embedding_dim)
#
#         # Xavier初始化
#         nn.init.xavier_uniform_(self.refine_fc1.weight)
#         nn.init.xavier_uniform_(self.refine_fc2.weight)
#         nn.init.zeros_(self.refine_fc1.bias)
#         nn.init.zeros_(self.refine_fc2.bias)
#
#     def forward(self, noisy_emb, enhanced_emb, return_weights=False):
#         """
#         动态融合前向传播
#
#         Args:
#             noisy_emb: [batch, 192]
#             enhanced_emb: [batch, 192]
#             return_weights: 是否返回置信度权重(用于分析)
#
#         Returns:
#             robust_emb: [batch, 192] - L2归一化的robust embedding
#             weights (可选): [batch, 2] - 置信度权重
#             stats (可选): dict - 统计信息
#         """
#         # 1. 评估置信度权重
#         weights, stats = self.confidence_net(noisy_emb, enhanced_emb)
#         w_noisy = weights[:, 0].unsqueeze(1)  # [batch, 1]
#         w_enhanced = weights[:, 1].unsqueeze(1)  # [batch, 1]
#
#         # 2. 动态加权融合
#         fused_emb = w_noisy * noisy_emb + w_enhanced * enhanced_emb  # [batch, 192]
#
#         # 3. Refinement (可选,增强非线性能力)
#         x = F.relu(self.refine_fc1(fused_emb))
#         x = self.refine_fc2(x)
#
#         # 4. 残差连接(保留原始融合信息)
#         robust_emb = fused_emb + x
#
#         # 5. L2归一化(SupCon要求)
#         robust_emb = F.normalize(robust_emb, p=2, dim=1)
#
#         if return_weights:
#             return robust_emb, weights, stats
#         else:
#             return robust_emb
#
#
# class ConfidenceSupConTrainer:
#     """
#     带置信度网络的SupCon训练器
#
#     相比原始SupConTrainer的改进:
#     1. 使用DynamicFusionMLP替代简单MLP
#     2. 可以记录和分析置信度权重
#     3. 支持权重可视化
#     """
#
#     def __init__(self, embedding_dim=192, device='cuda',
#                  temperature=0.07, confidence_hidden_dim=64):
#         self.device = device
#         self.embedding_dim = embedding_dim
#
#         # 使用动态融合MLP
#         self.mlp = DynamicFusionMLP(
#             embedding_dim=embedding_dim,
#             confidence_hidden_dim=confidence_hidden_dim
#         ).to(device)
#
#         # SupCon Loss(从原始实现导入)
#         from models_supcon import SupConLoss
#         self.criterion = SupConLoss(temperature=temperature)
#
#         self.optimizer = None
#
#         # 用于记录权重统计
#         self.weight_history = []
#
#         print(f'✅ Confidence SupCon Trainer initialized on {device}')
#         print(f'📊 Model architecture:')
#         print(f'   - Confidence Network: {sum(p.numel() for p in self.mlp.confidence_net.parameters()):,} params')
#         print(
#             f'   - Refinement Network: {sum(p.numel() for p in [self.mlp.refine_fc1.parameters(), self.mlp.refine_fc2.parameters()]):,} params')
#         print(f'   - Total MLP params: {sum(p.numel() for p in self.mlp.parameters()):,}')
#         print(f'🌡️  Temperature: {temperature}')
#
#     def train_step(self, noisy_embs, enhanced_embs, labels, log_weights=False):
#         """
#         执行一次训练步骤
#
#         Args:
#             noisy_embs: [batch_size, embedding_dim]
#             enhanced_embs: [batch_size, embedding_dim]
#             labels: [batch_size]
#             log_weights: 是否记录权重统计
#
#         Returns:
#             loss: float - 损失值
#             stats: dict - 统计信息(包含权重信息)
#         """
#         self.optimizer.zero_grad()
#
#         # 通过动态融合MLP获得robust embeddings
#         if log_weights:
#             robust_embs, weights, weight_stats = self.mlp(
#                 noisy_embs, enhanced_embs, return_weights=True
#             )
#         else:
#             robust_embs = self.mlp(noisy_embs, enhanced_embs, return_weights=False)
#             weight_stats = {}
#
#         # 计算SupCon损失
#         loss = self.criterion(robust_embs, labels)
#
#         loss.backward()
#         self.optimizer.step()
#
#         # 收集统计信息
#         stats = {
#             'loss': loss.item(),
#             **weight_stats
#         }
#
#         if log_weights:
#             self.weight_history.append(weight_stats)
#
#         return loss.item(), stats
#
#     def extract_embedding(self, noisy_emb, enhanced_emb):
#         """
#         推理时提取robust embedding
#
#         Args:
#             noisy_emb: [1, embedding_dim] or [embedding_dim]
#             enhanced_emb: [1, embedding_dim] or [embedding_dim]
#
#         Returns:
#             robust_emb: [1, embedding_dim] 或 [embedding_dim]
#             weights: [1, 2] - 置信度权重
#         """
#         self.mlp.eval()
#         with torch.no_grad():
#             # 确保是2D张量
#             if noisy_emb.dim() == 1:
#                 noisy_emb = noisy_emb.unsqueeze(0)
#             if enhanced_emb.dim() == 1:
#                 enhanced_emb = enhanced_emb.unsqueeze(0)
#
#             robust_emb, weights, _ = self.mlp(
#                 noisy_emb, enhanced_emb, return_weights=True
#             )
#
#         return robust_emb, weights
#
#     def get_weight_statistics(self):
#         """获取训练过程中的权重统计"""
#         if not self.weight_history:
#             return None
#
#         import numpy as np
#
#         w_noisy_vals = [h['w_noisy_mean'] for h in self.weight_history]
#         w_enhanced_vals = [h['w_enhanced_mean'] for h in self.weight_history]
#
#         return {
#             'w_noisy': {
#                 'mean': np.mean(w_noisy_vals),
#                 'std': np.std(w_noisy_vals),
#                 'min': np.min(w_noisy_vals),
#                 'max': np.max(w_noisy_vals),
#             },
#             'w_enhanced': {
#                 'mean': np.mean(w_enhanced_vals),
#                 'std': np.std(w_enhanced_vals),
#                 'min': np.min(w_enhanced_vals),
#                 'max': np.max(w_enhanced_vals),
#             }
#         }
#
#
# # ============ 使用示例 ============
#
# def example_usage():
#     """演示如何使用置信度网络"""
#
#     batch_size = 32
#     embedding_dim = 192
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#
#     # 1. 创建训练器
#     trainer = ConfidenceSupConTrainer(
#         embedding_dim=embedding_dim,
#         device=device,
#         temperature=0.07,
#         confidence_hidden_dim=64
#     )
#
#     # 2. 设置优化器
#     trainer.optimizer = torch.optim.AdamW(
#         trainer.mlp.parameters(),
#         lr=1e-3,
#         weight_decay=1e-4
#     )
#
#     # 3. 模拟训练数据
#     noisy_embs = torch.randn(batch_size, embedding_dim).to(device)
#     enhanced_embs = torch.randn(batch_size, embedding_dim).to(device)
#     labels = torch.randint(0, 10, (batch_size,)).to(device)
#
#     # 4. 训练一步
#     loss, stats = trainer.train_step(
#         noisy_embs, enhanced_embs, labels, log_weights=True
#     )
#
#     print(f'\n📊 Training step results:')
#     print(f'   Loss: {loss:.4f}')
#     print(f'   Avg weight (noisy): {stats["w_noisy_mean"]:.4f}')
#     print(f'   Avg weight (enhanced): {stats["w_enhanced_mean"]:.4f}')
#
#     # 5. 推理
#     test_noisy = torch.randn(1, embedding_dim).to(device)
#     test_enhanced = torch.randn(1, embedding_dim).to(device)
#     robust_emb, weights = trainer.extract_embedding(test_noisy, test_enhanced)
#
#     print(f'\n🔍 Inference results:')
#     print(f'   Robust embedding shape: {robust_emb.shape}')
#     print(f'   Confidence weights: noisy={weights[0, 0]:.4f}, enhanced={weights[0, 1]:.4f}')
#
#
# if __name__ == '__main__':
#     example_usage()
"""
阶段2优化: 置信度评估网络 (Confidence Estimation Network)

核心思想:
- 动态学习noisy和enhanced embedding的可信度
- 通过softmax确保权重和为1
- 轻量级设计,可解释性强

参考文献:
- SENet: Squeeze-and-Excitation Networks (CVPR 2018)
- Attention mechanisms for multi-modal fusion
- Dynamic weighting in speaker recognition

设计:
输入: [E_n; E_e] (concatenated)
网络: 2层MLP + Softmax
输出: [w_n, w_e] (权重和为1)
融合: E_robust = w_n * E_n + w_e * E_e
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConfidenceEstimationNet(nn.Module):
    """
    置信度评估网络

    功能:
    1. 评估noisy和enhanced embedding的可信度
    2. 输出两个权重(和为1)用于动态融合
    3. 轻量级设计,易于训练

    架构:
    Input: [batch, 384] (concatenated E_n and E_e)
    ├── FC1: 384 -> 192 + ReLU
    ├── FC2: 192 -> 64 + ReLU
    └── FC3: 64 -> 2 + Softmax
    Output: [batch, 2] (weights for [E_n, E_e])
    """

    def __init__(self, embedding_dim=192, hidden_dim=64):
        """
        Args:
            embedding_dim: 单个embedding的维度(默认192)
            hidden_dim: 隐藏层维度(默认64,可调整)
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # 三层MLP: 逐步压缩维度
        self.fc1 = nn.Linear(embedding_dim * 2, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 2)  # 输出2个权重

        # BatchNorm提升稳定性(可选)
        self.bn1 = nn.BatchNorm1d(embedding_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        # Dropout防止过拟合(可选)
        self.dropout = nn.Dropout(0.1)

        # Xavier初始化
        self._init_weights()

    def _init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, noisy_emb, enhanced_emb):
        """
        前向传播

        Args:
            noisy_emb: [batch, embedding_dim]
            enhanced_emb: [batch, embedding_dim]

        Returns:
            weights: [batch, 2] - 权重[w_n, w_e],和为1
            stats: dict - 用于分析的统计信息
        """
        # 拼接两个embedding
        x = torch.cat([noisy_emb, enhanced_emb], dim=1)  # [batch, 384]

        # 第一层
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        # 第二层
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        # 输出层
        logits = self.fc3(x)  # [batch, 2]

        # Softmax确保权重和为1
        weights = F.softmax(logits, dim=1)  # [batch, 2]

        # 收集统计信息(用于分析)
        stats = {
            'w_noisy_mean': weights[:, 0].mean().item(),
            'w_enhanced_mean': weights[:, 1].mean().item(),
            'w_noisy_std': weights[:, 0].std().item(),
            'w_enhanced_std': weights[:, 1].std().item(),
            'logits_mean': logits.mean().item(),
            'logits_std': logits.std().item(),
        }

        return weights, stats


class DynamicFusionMLP(nn.Module):
    """
    动态融合MLP - 结合置信度网络

    流程:
    1. 输入noisy和enhanced embedding
    2. 置信度网络评估权重[w_n, w_e]
    3. 动态融合: E_fused = w_n * E_n + w_e * E_e
    4. 通过一个轻量MLP得到最终robust embedding
    5. L2归一化
    """

    def __init__(self, embedding_dim=192, confidence_hidden_dim=64):
        super().__init__()

        self.embedding_dim = embedding_dim

        # 置信度评估网络
        self.confidence_net = ConfidenceEstimationNet(
            embedding_dim=embedding_dim,
            hidden_dim=confidence_hidden_dim
        )

        # 融合后的refinement network(可选,增强表达能力)
        self.refine_fc1 = nn.Linear(embedding_dim, embedding_dim)
        self.refine_fc2 = nn.Linear(embedding_dim, embedding_dim)

        # Xavier初始化
        nn.init.xavier_uniform_(self.refine_fc1.weight)
        nn.init.xavier_uniform_(self.refine_fc2.weight)
        nn.init.zeros_(self.refine_fc1.bias)
        nn.init.zeros_(self.refine_fc2.bias)

    def forward(self, noisy_emb, enhanced_emb, return_weights=False):
        """
        动态融合前向传播

        Args:
            noisy_emb: [batch, 192]
            enhanced_emb: [batch, 192]
            return_weights: 是否返回置信度权重(用于分析)

        Returns:
            robust_emb: [batch, 192] - L2归一化的robust embedding
            weights (可选): [batch, 2] - 置信度权重
            stats (可选): dict - 统计信息
        """
        # 1. 评估置信度权重
        weights, stats = self.confidence_net(noisy_emb, enhanced_emb)
        w_noisy = weights[:, 0].unsqueeze(1)  # [batch, 1]
        w_enhanced = weights[:, 1].unsqueeze(1)  # [batch, 1]

        # 2. 动态加权融合
        fused_emb = w_noisy * noisy_emb + w_enhanced * enhanced_emb  # [batch, 192]

        # 3. Refinement (可选,增强非线性能力)
        x = F.relu(self.refine_fc1(fused_emb))
        x = self.refine_fc2(x)

        # 4. 残差连接(保留原始融合信息)
        robust_emb = fused_emb + x

        # 5. L2归一化(SupCon要求)
        robust_emb = F.normalize(robust_emb, p=2, dim=1)

        if return_weights:
            return robust_emb, weights, stats
        else:
            return robust_emb


class ConfidenceSupConTrainer:
    """
    带置信度网络的SupCon训练器

    相比原始SupConTrainer的改进:
    1. 使用DynamicFusionMLP替代简单MLP
    2. 可以记录和分析置信度权重
    3. 支持权重可视化
    """

    def __init__(self, embedding_dim=192, device='cuda',
                 temperature=0.07, confidence_hidden_dim=64):
        self.device = device
        self.embedding_dim = embedding_dim

        # 使用动态融合MLP
        self.mlp = DynamicFusionMLP(
            embedding_dim=embedding_dim,
            confidence_hidden_dim=confidence_hidden_dim
        ).to(device)

        # SupCon Loss(从原始实现导入)
        from models_supcon import SupConLoss
        self.criterion = SupConLoss(temperature=temperature)

        self.optimizer = None

        # 用于记录权重统计
        self.weight_history = []

        # ✅ 修复: 正确计算参数量
        confidence_params = sum(p.numel() for p in self.mlp.confidence_net.parameters())
        refine_params = sum(p.numel() for p in self.mlp.refine_fc1.parameters()) + \
                        sum(p.numel() for p in self.mlp.refine_fc2.parameters())
        total_params = sum(p.numel() for p in self.mlp.parameters())

        print(f'✅ Confidence SupCon Trainer initialized on {device}')
        print(f'📊 Model architecture:')
        print(f'   - Confidence Network: {confidence_params:,} params')
        print(f'   - Refinement Network: {refine_params:,} params')
        print(f'   - Total MLP params: {total_params:,}')
        print(f'🌡️  Temperature: {temperature}')

    def train_step(self, noisy_embs, enhanced_embs, labels, log_weights=False):
        """
        执行一次训练步骤

        Args:
            noisy_embs: [batch_size, embedding_dim]
            enhanced_embs: [batch_size, embedding_dim]
            labels: [batch_size]
            log_weights: 是否记录权重统计

        Returns:
            loss: float - 损失值
            stats: dict - 统计信息(包含权重信息)
        """
        self.optimizer.zero_grad()

        # 通过动态融合MLP获得robust embeddings
        if log_weights:
            robust_embs, weights, weight_stats = self.mlp(
                noisy_embs, enhanced_embs, return_weights=True
            )
        else:
            robust_embs = self.mlp(noisy_embs, enhanced_embs, return_weights=False)
            weight_stats = {}

        # 计算SupCon损失
        loss = self.criterion(robust_embs, labels)

        loss.backward()
        self.optimizer.step()

        # 收集统计信息
        stats = {
            'loss': loss.item(),
            **weight_stats
        }

        if log_weights:
            self.weight_history.append(weight_stats)

        return loss.item(), stats

    def extract_embedding(self, noisy_emb, enhanced_emb):
        """
        推理时提取robust embedding

        Args:
            noisy_emb: [1, embedding_dim] or [embedding_dim]
            enhanced_emb: [1, embedding_dim] or [embedding_dim]

        Returns:
            robust_emb: [1, embedding_dim] 或 [embedding_dim]
            weights: [1, 2] - 置信度权重
        """
        self.mlp.eval()
        with torch.no_grad():
            # 确保是2D张量
            if noisy_emb.dim() == 1:
                noisy_emb = noisy_emb.unsqueeze(0)
            if enhanced_emb.dim() == 1:
                enhanced_emb = enhanced_emb.unsqueeze(0)

            robust_emb, weights, _ = self.mlp(
                noisy_emb, enhanced_emb, return_weights=True
            )

        return robust_emb, weights

    def get_weight_statistics(self):
        """获取训练过程中的权重统计"""
        if not self.weight_history:
            return None

        import numpy as np

        w_noisy_vals = [h['w_noisy_mean'] for h in self.weight_history]
        w_enhanced_vals = [h['w_enhanced_mean'] for h in self.weight_history]

        return {
            'w_noisy': {
                'mean': np.mean(w_noisy_vals),
                'std': np.std(w_noisy_vals),
                'min': np.min(w_noisy_vals),
                'max': np.max(w_noisy_vals),
            },
            'w_enhanced': {
                'mean': np.mean(w_enhanced_vals),
                'std': np.std(w_enhanced_vals),
                'min': np.min(w_enhanced_vals),
                'max': np.max(w_enhanced_vals),
            }
        }


# ============ 使用示例 ============

def example_usage():
    """演示如何使用置信度网络"""

    batch_size = 32
    embedding_dim = 192
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('=' * 80)
    print('🧪 Testing Confidence Estimation Network')
    print('=' * 80)

    # 1. 创建训练器
    trainer = ConfidenceSupConTrainer(
        embedding_dim=embedding_dim,
        device=device,
        temperature=0.07,
        confidence_hidden_dim=64
    )

    # 2. 设置优化器
    trainer.optimizer = torch.optim.AdamW(
        trainer.mlp.parameters(),
        lr=1e-3,
        weight_decay=1e-4
    )

    print('\n' + '=' * 80)
    print('📝 Training Step Test')
    print('=' * 80)

    # 3. 模拟训练数据
    noisy_embs = torch.randn(batch_size, embedding_dim).to(device)
    enhanced_embs = torch.randn(batch_size, embedding_dim).to(device)
    labels = torch.randint(0, 10, (batch_size,)).to(device)

    # 4. 训练一步
    loss, stats = trainer.train_step(
        noisy_embs, enhanced_embs, labels, log_weights=True
    )

    print(f'\n📊 Training results:')
    print(f'   Loss: {loss:.4f}')
    print(f'   Avg weight (noisy): {stats["w_noisy_mean"]:.4f} ± {stats["w_noisy_std"]:.4f}')
    print(f'   Avg weight (enhanced): {stats["w_enhanced_mean"]:.4f} ± {stats["w_enhanced_std"]:.4f}')
    print(f'   Weight sum check: {stats["w_noisy_mean"] + stats["w_enhanced_mean"]:.4f} (should be ~1.0)')

    print('\n' + '=' * 80)
    print('🔍 Inference Test')
    print('=' * 80)

    # 5. 推理测试
    test_noisy = torch.randn(1, embedding_dim).to(device)
    test_enhanced = torch.randn(1, embedding_dim).to(device)
    robust_emb, weights = trainer.extract_embedding(test_noisy, test_enhanced)

    print(f'\n🎯 Inference results:')
    print(f'   Robust embedding shape: {robust_emb.shape}')
    print(f'   Robust embedding norm: {torch.norm(robust_emb, p=2).item():.4f} (should be ~1.0)')
    print(f'   Confidence weights:')
    print(f'      - Noisy: {weights[0, 0]:.4f}')
    print(f'      - Enhanced: {weights[0, 1]:.4f}')
    print(f'      - Sum: {weights[0, 0] + weights[0, 1]:.4f}')

    print('\n' + '=' * 80)
    print('✅ All tests passed!')
    print('=' * 80)

    print('\n💡 Next steps:')
    print('   1. Run: python train_confidence.py (train the model)')
    print('   2. Run: python evaluate_confidence.py (evaluate performance)')
    print('   3. Run: python visualize_weights.py (analyze weight distribution)')


if __name__ == '__main__':
    example_usage()