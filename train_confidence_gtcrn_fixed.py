# """
# 置信度网络训练脚本（集成 GTCRN）- 已修复 Windows 多进程问题
#
# 训练流程：
# 1. 加载训练好的 GTCRN（冻结）
# 2. 加载预训练的说话人模型 ECAPA-TDNN（冻结）
# 3. 训练置信度网络融合 noisy 和 enhanced 嵌入
# 4. 使用 Contrastive Loss 优化
#
# SNR 配置：
# - 训练：从 [-5, 0, 5, 10, 15] dB 中随机选择
# - 测试：在每个 SNR 上分别评估
#
# 用法：
#     python train_confidence_gtcrn_fixed.py --voxceleb_dir data/voxceleb1 --musan_dir data/musan
# """
#
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from pathlib import Path
# import sys
# import argparse
# from tqdm import tqdm
# import json
# from datetime import datetime
# import logging
#
# # 添加项目路径
# project_root = Path(__file__).parent
# sys.path.insert(0, str(project_root))
# sys.path.insert(0, str(project_root / 'scripts'))
#
# from gtcrn_wrapper_fixed import GTCRNWrapper
# from contrastive_loss import ContrastiveLoss
# from dataset_fixed import VoxCelebMusanDataset
# from fixed_snr_dataset import FixedSNRDataset, collate_fn_fixed_length
#
#
# # ============= Windows 多进程修复 =============
# def custom_collate_fn(batch):
#     """全局 collate 函数，避免 pickle 错误"""
#     return collate_fn_fixed_length(batch, target_length=48000)
#
#
# # ============================================
#
#
# class ConfidenceNetwork(nn.Module):
#     """
#     置信度网络：融合 noisy 和 enhanced 嵌入
#
#     输入：
#         emb_noisy: [batch, embedding_dim] - 来自 noisy 音频
#         emb_enhanced: [batch, embedding_dim] - 来自 GTCRN 增强音频
#
#     输出：
#         emb_fused: [batch, embedding_dim] - 融合后的嵌入
#     """
#
#     def __init__(self, embedding_dim=192, hidden_dim=256):
#         super().__init__()
#
#         # 融合网络
#         self.fusion = nn.Sequential(
#             nn.Linear(embedding_dim * 2, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(hidden_dim, embedding_dim)
#         )
#
#         # 注意力权重（学习每个嵌入的重要性）
#         self.attention = nn.Sequential(
#             nn.Linear(embedding_dim * 2, 64),
#             nn.ReLU(),
#             nn.Linear(64, 2),
#             nn.Softmax(dim=1)
#         )
#
#     def forward(self, emb_noisy, emb_enhanced):
#         """
#         融合两个嵌入
#
#         Args:
#             emb_noisy: [batch, embedding_dim]
#             emb_enhanced: [batch, embedding_dim]
#
#         Returns:
#             emb_fused: [batch, embedding_dim]
#         """
#         # 拼接
#         concat = torch.cat([emb_noisy, emb_enhanced], dim=1)  # [batch, embedding_dim*2]
#
#         # 计算注意力权重
#         weights = self.attention(concat)  # [batch, 2]
#         w_noisy = weights[:, 0:1]  # [batch, 1]
#         w_enhanced = weights[:, 1:2]  # [batch, 1]
#
#         # 加权融合
#         weighted = w_noisy * emb_noisy + w_enhanced * emb_enhanced
#
#         # 通过融合网络
#         fused = self.fusion(concat)
#
#         # 残差连接
#         output = weighted + fused
#
#         # L2 归一化
#         output = nn.functional.normalize(output, p=2, dim=1)
#
#         return output
#
#
# def setup_logger(log_dir):
#     """设置日志"""
#     log_dir = Path(log_dir)
#     log_dir.mkdir(parents=True, exist_ok=True)
#
#     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#     log_file = log_dir / f'train_confidence_{timestamp}.log'
#
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s [%(levelname)s] %(message)s',
#         handlers=[
#             logging.FileHandler(log_file),
#             logging.StreamHandler()
#         ]
#     )
#
#     return logging.getLogger(__name__)
#
#
# def train_epoch(gtcrn, speaker_model, confidence_net, dataloader, criterion,
#                 optimizer, device, epoch):
#     """训练一个 epoch"""
#     # GTCRN 和说话人模型冻结
#
#     speaker_model.eval()
#
#     # 置信度网络训练
#     confidence_net.train()
#
#     total_loss = 0
#     num_batches = 0
#     num_errors = 0
#
#     pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
#
#     for batch_idx, batch in enumerate(pbar):
#         try:
#             # 获取数据
#             noisy = batch['anchor_noisy'].to(device)  # [batch, 1, samples]
#             speaker_ids = batch['speaker_id']
#
#             # 构建标签（说话人 ID 转换为连续索引）
#             unique_speakers = list(set(speaker_ids))
#             speaker_to_idx = {spk: idx for idx, spk in enumerate(unique_speakers)}
#             labels = torch.tensor([speaker_to_idx[spk] for spk in speaker_ids]).to(device)
#
#             # GTCRN 增强（冻结）
#             with torch.no_grad():
#                 enhanced = gtcrn.enhance(noisy)  # [batch, 1, samples]
#
#             # 提取嵌入（冻结）
#             with torch.no_grad():
#                 emb_noisy = speaker_model(noisy)  # [batch, embedding_dim]
#                 emb_enhanced = speaker_model(enhanced)  # [batch, embedding_dim]
#
#             # 置信度网络融合
#             emb_fused = confidence_net(emb_noisy, emb_enhanced)
#
#             # 计算损失
#             loss = criterion(emb_fused, labels)
#
#             # 反向传播
#             optimizer.zero_grad()
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(confidence_net.parameters(), max_norm=5.0)
#             optimizer.step()
#
#             # 统计
#             total_loss += loss.item()
#             num_batches += 1
#
#             # 更新进度条
#             pbar.set_postfix({
#                 'loss': f"{loss.item():.4f}",
#                 'errors': num_errors
#             })
#
#         except Exception as e:
#             num_errors += 1
#             print(f"\nError in batch {batch_idx}: {e}")
#             if num_errors > 10:
#                 print("Too many errors, stopping epoch")
#                 break
#             continue
#
#     if num_batches == 0:
#         raise RuntimeError("No batches processed successfully!")
#
#     avg_loss = total_loss / num_batches
#     return {'loss': avg_loss, 'num_errors': num_errors}
#
#
# def validate(gtcrn, speaker_model, confidence_net, dataloader, criterion, device):
#     """验证"""
#
#     speaker_model.eval()
#     confidence_net.eval()
#
#     total_loss = 0
#     num_batches = 0
#
#     with torch.no_grad():
#         for batch in tqdm(dataloader, desc='Validation'):
#             try:
#                 noisy = batch['anchor_noisy'].to(device)
#                 speaker_ids = batch['speaker_id']
#
#                 # 构建标签
#                 unique_speakers = list(set(speaker_ids))
#                 speaker_to_idx = {spk: idx for idx, spk in enumerate(unique_speakers)}
#                 labels = torch.tensor([speaker_to_idx[spk] for spk in speaker_ids]).to(device)
#
#                 # GTCRN 增强
#                 enhanced = gtcrn.enhance(noisy)
#
#                 # 提取嵌入
#                 emb_noisy = speaker_model(noisy)
#                 emb_enhanced = speaker_model(enhanced)
#
#                 # 融合
#                 emb_fused = confidence_net(emb_noisy, emb_enhanced)
#
#                 # 计算损失
#                 loss = criterion(emb_fused, labels)
#                 total_loss += loss.item()
#                 num_batches += 1
#
#             except Exception as e:
#                 print(f"Validation error: {e}")
#                 continue
#
#     if num_batches == 0:
#         raise RuntimeError("No validation batches processed!")
#
#     avg_loss = total_loss / num_batches
#     return {'loss': avg_loss}
#
#
# def main(args):
#     # 设置日志
#     logger = setup_logger(args.log_dir)
#     logger.info('=' * 80)
#     logger.info('Confidence Network Training with GTCRN')
#     logger.info('=' * 80)
#     logger.info(f'SNR values: {args.snr_values}')
#     logger.info(f'Arguments: {vars(args)}')
#
#     # 设备
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     logger.info(f'Device: {device}')
#
#     # 检查 GTCRN checkpoint
#     gtcrn_path = Path(args.gtcrn_checkpoint)
#     if not gtcrn_path.exists():
#         logger.error(f"GTCRN checkpoint not found: {gtcrn_path}")
#         logger.error("Please train GTCRN first or specify correct path")
#         return 1
#
#     # 创建数据集
#     logger.info('\nLoading datasets...')
#
#     try:
#         # 训练集
#         base_train_dataset = VoxCelebMusanDataset(
#             voxceleb_dir=args.voxceleb_dir,
#             musan_dir=args.musan_dir,
#             split='train',
#             snr_range=(-5, 15),
#             return_clean=False  # 不需要 clean 音频
#         )
#
#         train_dataset = FixedSNRDataset(base_train_dataset, snr_values=args.snr_values)
#
#         # 验证集
#         val_dataset = VoxCelebMusanDataset(
#             voxceleb_dir=args.voxceleb_dir,
#             musan_dir=args.musan_dir,
#             split='test',
#             test_snr=0,
#             test_noise_type='noise',
#             return_clean=False
#         )
#
#         logger.info(f'[OK] Train samples: {len(train_dataset)}')
#         logger.info(f'[OK] Val samples: {len(val_dataset)}')
#
#     except Exception as e:
#         logger.error(f"Failed to load datasets: {e}")
#         return 1
#
#     # DataLoader
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=args.batch_size,
#         shuffle=True,
#         num_workers=args.num_workers,
#         pin_memory=True,
#         collate_fn=custom_collate_fn  # 使用全局函数
#     )
#
#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=args.batch_size,
#         shuffle=False,
#         num_workers=args.num_workers,
#         pin_memory=True,
#         collate_fn=custom_collate_fn  # 使用全局函数
#     )
#
#     # 加载 GTCRN（冻结）
#     logger.info('\nLoading GTCRN (frozen)...')
#     gtcrn = GTCRNWrapper(
#         checkpoint_path=str(gtcrn_path),
#         device=device,
#         freeze=True  # 冻结
#     )
#     logger.info('[OK] GTCRN loaded and frozen')
#
#     # 加载说话人模型（冻结）
#     logger.info('\nLoading speaker model (frozen)...')
#
#     if args.use_dummy_speaker_model:
#         # 使用占位符模型（测试用）
#         logger.warning('[WARNING] Using dummy speaker model for testing!')
#         logger.warning('Replace with actual ECAPA-TDNN for real training')
#
#         class DummySpeakerModel(nn.Module):
#             def __init__(self, embedding_dim=192):
#                 super().__init__()
#                 self.conv = nn.Conv1d(1, 64, 3, padding=1)
#                 self.pool = nn.AdaptiveAvgPool1d(1)
#                 self.fc = nn.Linear(64, embedding_dim)
#
#             def forward(self, x):
#                 # x: [batch, 1, samples]
#                 batch_size = x.size(0)
#
#                 x = self.conv(x)  # [batch, 64, samples]
#                 x = self.pool(x)  # [batch, 64, 1]
#                 x = x.view(batch_size, 64)  # [batch, 64]
#                 x = self.fc(x)  # [batch, embedding_dim]
#
#                 # L2 normalize
#                 x = nn.functional.normalize(x, p=2, dim=1)
#                 return x
#
#         speaker_model = DummySpeakerModel(embedding_dim=args.embedding_dim).to(device)
#         speaker_model.eval()
#
#     else:
#         # 加载实际的说话人模型
#         try:
#             # TODO: 替换为你的实际加载代码
#             # 示例：
#             # from ecapa_tdnn import ECAPA_TDNN
#             # speaker_model = ECAPA_TDNN(C=1024, embedding_dim=192)
#             # checkpoint = torch.load(args.speaker_model_path)
#             # speaker_model.load_state_dict(checkpoint['model_state_dict'])
#             # speaker_model = speaker_model.to(device)
#             # speaker_model.eval()
#
#             raise NotImplementedError(
#                 "Please implement actual speaker model loading. "
#                 "Use --use_dummy_speaker_model for testing."
#             )
#
#         except Exception as e:
#             logger.error(f"Failed to load speaker model: {e}")
#             logger.error("Use --use_dummy_speaker_model for testing")
#             return 1
#
#     logger.info('[OK] Speaker model loaded and frozen')
#
#     # 初始化置信度网络
#     logger.info('\nInitializing confidence network...')
#     confidence_net = ConfidenceNetwork(
#         embedding_dim=args.embedding_dim,
#         hidden_dim=args.hidden_dim
#     ).to(device)
#     logger.info('[OK] Confidence network initialized')
#
#     # 损失函数
#     criterion = ContrastiveLoss(temperature=args.temperature)
#
#     # 优化器（只优化置信度网络）
#     optimizer = optim.AdamW(
#         confidence_net.parameters(),
#         lr=args.lr,
#         weight_decay=args.weight_decay
#     )
#
#     # 学习率调度
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, mode='min', factor=0.5, patience=3, verbose=True
#     )
#
#     # 训练历史
#     history = {
#         'train_loss': [],
#         'val_loss': []
#     }
#
#     best_val_loss = float('inf')
#     checkpoint_dir = Path(args.checkpoint_dir)
#     checkpoint_dir.mkdir(parents=True, exist_ok=True)
#
#     # 训练循环
#     logger.info('\nStarting training...')
#     logger.info('=' * 80)
#
#     for epoch in range(1, args.num_epochs + 1):
#         logger.info(f'\nEpoch {epoch}/{args.num_epochs}')
#         logger.info('-' * 80)
#
#         try:
#             # 训练
#             train_metrics = train_epoch(
#                 gtcrn, speaker_model, confidence_net, train_loader,
#                 criterion, optimizer, device, epoch
#             )
#
#             logger.info(f'Train - Loss: {train_metrics["loss"]:.4f}, '
#                         f'Errors: {train_metrics["num_errors"]}')
#
#             # 验证
#             val_metrics = validate(
#                 gtcrn, speaker_model, confidence_net, val_loader,
#                 criterion, device
#             )
#
#             logger.info(f'Val   - Loss: {val_metrics["loss"]:.4f}')
#
#             # 更新历史
#             history['train_loss'].append(train_metrics['loss'])
#             history['val_loss'].append(val_metrics['loss'])
#
#             # 学习率调整
#             scheduler.step(val_metrics['loss'])
#
#             # 保存最佳模型
#             if val_metrics['loss'] < best_val_loss:
#                 best_val_loss = val_metrics['loss']
#                 checkpoint_path = checkpoint_dir / 'confidence_net_best.pth'
#                 torch.save({
#                     'epoch': epoch,
#                     'model_state_dict': confidence_net.state_dict(),
#                     'optimizer_state_dict': optimizer.state_dict(),
#                     'val_loss': best_val_loss,
#                     'history': history
#                 }, checkpoint_path)
#                 logger.info(f'[OK] Saved best model: {checkpoint_path}')
#
#             # 定期保存
#             if epoch % args.save_interval == 0:
#                 checkpoint_path = checkpoint_dir / f'confidence_net_epoch{epoch}.pth'
#                 torch.save({
#                     'epoch': epoch,
#                     'model_state_dict': confidence_net.state_dict(),
#                     'optimizer_state_dict': optimizer.state_dict(),
#                     'val_loss': val_metrics['loss'],
#                     'history': history
#                 }, checkpoint_path)
#                 logger.info(f'[SAVE] Saved checkpoint: {checkpoint_path}')
#
#         except Exception as e:
#             logger.error(f"Error in epoch {epoch}: {e}")
#             import traceback
#             traceback.print_exc()
#             continue
#
#     # 保存最终模型
#     final_checkpoint = checkpoint_dir / 'confidence_net_final.pth'
#     torch.save({
#         'epoch': args.num_epochs,
#         'model_state_dict': confidence_net.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'history': history
#     }, final_checkpoint)
#     logger.info(f'\n[OK] Training complete! Final model saved: {final_checkpoint}')
#
#     # 保存历史
#     history_file = checkpoint_dir / 'training_history.json'
#     with open(history_file, 'w') as f:
#         json.dump(history, f, indent=2)
#     logger.info(f'[INFO] Training history saved: {history_file}')
#
#     return 0
#
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Train Confidence Network with GTCRN')
#
#     # 数据
#     parser.add_argument('--voxceleb_dir', type=str, required=True)
#     parser.add_argument('--musan_dir', type=str, required=True)
#     parser.add_argument('--snr_values', type=int, nargs='+',
#                         default=[-5, 0, 5, 10, 15])
#
#     # 模型
#     parser.add_argument('--gtcrn_checkpoint', type=str,
#                         default='checkpoints/gtcrn/gtcrn_best.pth',
#                         help='Path to trained GTCRN checkpoint')
#     parser.add_argument('--speaker_model_path', type=str,
#                         default='checkpoints/speaker_model.pth',
#                         help='Path to pretrained speaker model')
#     parser.add_argument('--use_dummy_speaker_model', action='store_true',
#                         help='Use dummy speaker model for testing')
#     parser.add_argument('--embedding_dim', type=int, default=192)
#     parser.add_argument('--hidden_dim', type=int, default=256)
#
#     # 损失
#     parser.add_argument('--temperature', type=float, default=0.07)
#
#     # 训练
#     parser.add_argument('--batch_size', type=int, default=32)
#     parser.add_argument('--num_epochs', type=int, default=30)
#     parser.add_argument('--lr', type=float, default=1e-3)
#     parser.add_argument('--weight_decay', type=float, default=1e-5)
#     parser.add_argument('--num_workers', type=int, default=4)
#
#     # 输出
#     parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/confidence')
#     parser.add_argument('--log_dir', type=str, default='logs')
#     parser.add_argument('--save_interval', type=int, default=5)
#
#     args = parser.parse_args()
#     exit(main(args))
"""
置信度网络训练脚本（集成 GTCRN + ECAPA-TDNN）

训练流程：
1. 加载训练好的 GTCRN（冻结）
2. 加载预训练的说话人模型 ECAPA-TDNN（冻结）
3. 训练置信度网络融合 noisy 和 enhanced 嵌入
4. 使用 Contrastive Loss 优化

SNR 配置：
- 训练：从 [-5, 0, 5, 10, 15] dB 中随机选择
- 测试：在每个 SNR 上分别评估

用法：
    python train_confidence_gtcrn_fixed.py --voxceleb_dir data/voxceleb1 --musan_dir data/musan
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import argparse
from tqdm import tqdm
import json
from datetime import datetime
import logging

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'scripts'))

from gtcrn_wrapper_fixed import GTCRNWrapper
from ecapa_tdnn_wrapper import ECAPATDNNWrapper  # 新增：ECAPA-TDNN wrapper
from contrastive_loss import ContrastiveLoss
from dataset_fixed import VoxCelebMusanDataset
from fixed_snr_dataset import FixedSNRDataset, collate_fn_fixed_length


# ============= Windows 多进程修复 =============
def custom_collate_fn(batch):
    """全局 collate 函数，避免 pickle 错误"""
    return collate_fn_fixed_length(batch, target_length=48000)


# ============================================


class ConfidenceNetwork(nn.Module):
    """
    置信度网络：融合 noisy 和 enhanced 嵌入

    输入：
        emb_noisy: [batch, embedding_dim] - 来自 noisy 音频
        emb_enhanced: [batch, embedding_dim] - 来自 GTCRN 增强音频

    输出：
        emb_fused: [batch, embedding_dim] - 融合后的嵌入
    """

    def __init__(self, embedding_dim=192, hidden_dim=256):
        super().__init__()

        # 融合网络
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim)
        )

        # 注意力权重（学习每个嵌入的重要性）
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, emb_noisy, emb_enhanced):
        """
        融合两个嵌入

        Args:
            emb_noisy: [batch, embedding_dim]
            emb_enhanced: [batch, embedding_dim]

        Returns:
            emb_fused: [batch, embedding_dim]
        """
        # 拼接
        concat = torch.cat([emb_noisy, emb_enhanced], dim=1)  # [batch, embedding_dim*2]

        # 计算注意力权重
        weights = self.attention(concat)  # [batch, 2]
        w_noisy = weights[:, 0:1]  # [batch, 1]
        w_enhanced = weights[:, 1:2]  # [batch, 1]

        # 加权融合
        weighted = w_noisy * emb_noisy + w_enhanced * emb_enhanced

        # 通过融合网络
        fused = self.fusion(concat)

        # 残差连接
        output = weighted + fused

        # L2 归一化
        output = nn.functional.normalize(output, p=2, dim=1)

        return output


def setup_logger(log_dir):
    """设置日志"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'train_confidence_{timestamp}.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__)


def train_epoch(gtcrn, speaker_model, confidence_net, dataloader, criterion,
                optimizer, device, epoch):
    """训练一个 epoch"""
    # GTCRN 和说话人模型冻结
    speaker_model.eval()

    # 置信度网络训练
    confidence_net.train()

    total_loss = 0
    num_batches = 0
    num_errors = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')

    for batch_idx, batch in enumerate(pbar):
        try:
            # 获取数据
            noisy = batch['anchor_noisy'].to(device)  # [batch, 1, samples]
            speaker_ids = batch['speaker_id']

            # 构建标签（说话人 ID 转换为连续索引）
            unique_speakers = list(set(speaker_ids))
            speaker_to_idx = {spk: idx for idx, spk in enumerate(unique_speakers)}
            labels = torch.tensor([speaker_to_idx[spk] for spk in speaker_ids]).to(device)

            # GTCRN 增强（冻结）
            with torch.no_grad():
                enhanced = gtcrn.enhance(noisy)  # [batch, 1, samples]

            # 提取嵌入（冻结）
            with torch.no_grad():
                emb_noisy = speaker_model(noisy)  # [batch, embedding_dim]
                emb_enhanced = speaker_model(enhanced)  # [batch, embedding_dim]

            # 置信度网络融合
            emb_fused = confidence_net(emb_noisy, emb_enhanced)

            # 计算损失
            loss = criterion(emb_fused, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(confidence_net.parameters(), max_norm=5.0)
            optimizer.step()

            # 统计
            total_loss += loss.item()
            num_batches += 1

            # 更新进度条
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'errors': num_errors
            })

        except Exception as e:
            num_errors += 1
            print(f"\nError in batch {batch_idx}: {e}")
            if num_errors > 10:
                print("Too many errors, stopping epoch")
                break
            continue

    if num_batches == 0:
        raise RuntimeError("No batches processed successfully!")

    avg_loss = total_loss / num_batches
    return {'loss': avg_loss, 'num_errors': num_errors}


def validate(gtcrn, speaker_model, confidence_net, dataloader, criterion, device):
    """验证"""
    speaker_model.eval()
    confidence_net.eval()

    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            try:
                noisy = batch['anchor_noisy'].to(device)
                speaker_ids = batch['speaker_id']

                # 构建标签
                unique_speakers = list(set(speaker_ids))
                speaker_to_idx = {spk: idx for idx, spk in enumerate(unique_speakers)}
                labels = torch.tensor([speaker_to_idx[spk] for spk in speaker_ids]).to(device)

                # GTCRN 增强
                enhanced = gtcrn.enhance(noisy)

                # 提取嵌入
                emb_noisy = speaker_model(noisy)
                emb_enhanced = speaker_model(enhanced)

                # 融合
                emb_fused = confidence_net(emb_noisy, emb_enhanced)

                # 计算损失
                loss = criterion(emb_fused, labels)
                total_loss += loss.item()
                num_batches += 1

            except Exception as e:
                print(f"Validation error: {e}")
                continue

    if num_batches == 0:
        raise RuntimeError("No validation batches processed!")

    avg_loss = total_loss / num_batches
    return {'loss': avg_loss}


def main(args):
    # 设置日志
    logger = setup_logger(args.log_dir)
    logger.info('=' * 80)
    logger.info('Confidence Network Training with GTCRN + ECAPA-TDNN')
    logger.info('=' * 80)
    logger.info(f'SNR values: {args.snr_values}')
    logger.info(f'Arguments: {vars(args)}')

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')

    # 检查 GTCRN checkpoint
    gtcrn_path = Path(args.gtcrn_checkpoint)
    if not gtcrn_path.exists():
        logger.error(f"GTCRN checkpoint not found: {gtcrn_path}")
        logger.error("Please train GTCRN first or specify correct path")
        return 1

    # 创建数据集
    logger.info('\nLoading datasets...')

    try:
        # 训练集
        base_train_dataset = VoxCelebMusanDataset(
            voxceleb_dir=args.voxceleb_dir,
            musan_dir=args.musan_dir,
            split='train',
            snr_range=(-5, 15),
            return_clean=False  # 不需要 clean 音频
        )

        train_dataset = FixedSNRDataset(base_train_dataset, snr_values=args.snr_values)

        # 验证集
        val_dataset = VoxCelebMusanDataset(
            voxceleb_dir=args.voxceleb_dir,
            musan_dir=args.musan_dir,
            split='test',
            test_snr=0,
            test_noise_type='noise',
            return_clean=False
        )

        logger.info(f'[OK] Train samples: {len(train_dataset)}')
        logger.info(f'[OK] Val samples: {len(val_dataset)}')

    except Exception as e:
        logger.error(f"Failed to load datasets: {e}")
        return 1

    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn  # 使用全局函数
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn  # 使用全局函数
    )

    # 加载 GTCRN（冻结）
    logger.info('\nLoading GTCRN (frozen)...')
    gtcrn = GTCRNWrapper(
        checkpoint_path=str(gtcrn_path),
        device=device,
        freeze=True  # 冻结
    )
    logger.info('[OK] GTCRN loaded and frozen')

    # ==================== 加载说话人模型（ECAPA-TDNN） ====================
    logger.info('\nLoading ECAPA-TDNN speaker model (frozen)...')

    try:
        speaker_model = ECAPATDNNWrapper(
            model_path=args.speaker_model_path,
            device=device,
            freeze=True  # 冻结参数
        )
        logger.info('[OK] ECAPA-TDNN loaded and frozen')

    except Exception as e:
        logger.error(f"Failed to load ECAPA-TDNN: {e}")
        logger.error("Please ensure:")
        logger.error("  1. SpeechBrain is installed: pip install speechbrain")
        logger.error(f"  2. Model path exists: {args.speaker_model_path}")
        return 1
    # =====================================================================

    # 初始化置信度网络
    logger.info('\nInitializing confidence network...')
    confidence_net = ConfidenceNetwork(
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim
    ).to(device)
    logger.info('[OK] Confidence network initialized')

    # 损失函数
    criterion = ContrastiveLoss(temperature=args.temperature)

    # 优化器（只优化置信度网络）
    optimizer = optim.AdamW(
        confidence_net.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # 学习率调度
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    # 训练历史
    history = {
        'train_loss': [],
        'val_loss': []
    }

    best_val_loss = float('inf')
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 训练循环
    logger.info('\nStarting training...')
    logger.info('=' * 80)

    for epoch in range(1, args.num_epochs + 1):
        logger.info(f'\nEpoch {epoch}/{args.num_epochs}')
        logger.info('-' * 80)

        try:
            # 训练
            train_metrics = train_epoch(
                gtcrn, speaker_model, confidence_net, train_loader,
                criterion, optimizer, device, epoch
            )

            logger.info(f'Train - Loss: {train_metrics["loss"]:.4f}, '
                        f'Errors: {train_metrics["num_errors"]}')

            # 验证
            val_metrics = validate(
                gtcrn, speaker_model, confidence_net, val_loader,
                criterion, device
            )

            logger.info(f'Val   - Loss: {val_metrics["loss"]:.4f}')

            # 更新历史
            history['train_loss'].append(train_metrics['loss'])
            history['val_loss'].append(val_metrics['loss'])

            # 学习率调整
            scheduler.step(val_metrics['loss'])

            # 保存最佳模型
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                checkpoint_path = checkpoint_dir / 'confidence_net_best.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': confidence_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': best_val_loss,
                    'history': history
                }, checkpoint_path)
                logger.info(f'[OK] Saved best model: {checkpoint_path}')

            # 定期保存
            if epoch % args.save_interval == 0:
                checkpoint_path = checkpoint_dir / f'confidence_net_epoch{epoch}.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': confidence_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_metrics['loss'],
                    'history': history
                }, checkpoint_path)
                logger.info(f'[SAVE] Saved checkpoint: {checkpoint_path}')

        except Exception as e:
            logger.error(f"Error in epoch {epoch}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 保存最终模型
    final_checkpoint = checkpoint_dir / 'confidence_net_final.pth'
    torch.save({
        'epoch': args.num_epochs,
        'model_state_dict': confidence_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history
    }, final_checkpoint)
    logger.info(f'\n[OK] Training complete! Final model saved: {final_checkpoint}')

    # 保存历史
    history_file = checkpoint_dir / 'training_history.json'
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)
    logger.info(f'[INFO] Training history saved: {history_file}')

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Confidence Network with GTCRN + ECAPA-TDNN')

    # 数据
    parser.add_argument('--voxceleb_dir', type=str, required=True)
    parser.add_argument('--musan_dir', type=str, required=True)
    parser.add_argument('--snr_values', type=int, nargs='+',
                        default=[-5, 0, 5, 10, 15])

    # 模型
    parser.add_argument('--gtcrn_checkpoint', type=str,
                        default='checkpoints/gtcrn/gtcrn_best.pth',
                        help='Path to trained GTCRN checkpoint')
    parser.add_argument('--speaker_model_path', type=str,
                        default='pretrained_models/spkrec-ecapa-voxceleb',
                        help='Path to ECAPA-TDNN pretrained model')
    parser.add_argument('--embedding_dim', type=int, default=192)
    parser.add_argument('--hidden_dim', type=int, default=256)

    # 损失
    parser.add_argument('--temperature', type=float, default=0.07)

    # 训练
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--num_workers', type=int, default=4)

    # 输出
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/confidence')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--save_interval', type=int, default=5)

    args = parser.parse_args()
    exit(main(args))