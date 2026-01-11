"""
置信度网络训练脚本（集成 GTCRN）

训练流程：
1. 加载训练好的 GTCRN（冻结）
2. 加载预训练的说话人模型 ECAPA-TDNN（冻结）
3. 训练置信度网络融合 noisy 和 enhanced 嵌入
4. 使用 Contrastive Loss 优化

SNR 配置：
- 训练：从 [-5, 0, 5, 10, 15] dB 中随机选择
- 测试：在每个 SNR 上分别评估

用法：
    python train_confidence_gtcrn.py --voxceleb_dir data/voxceleb1 --musan_dir data/musan
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
from contrastive_loss import ContrastiveLoss
from dataset_fixed import VoxCelebMusanDataset
from fixed_snr_dataset import FixedSNRDataset, collate_fn_fixed_length


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
    gtcrn.eval()
    speaker_model.eval()

    # 置信度网络训练
    confidence_net.train()

    total_loss = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')

    for batch_idx, batch in enumerate(pbar):
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

        # 更新进度条
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    avg_loss = total_loss / len(dataloader)
    return {'loss': avg_loss}


def validate(gtcrn, speaker_model, confidence_net, dataloader, criterion, device):
    """验证"""
    gtcrn.eval()
    speaker_model.eval()
    confidence_net.eval()

    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
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

    avg_loss = total_loss / len(dataloader)
    return {'loss': avg_loss}


def main(args):
    # 设置日志
    logger = setup_logger(args.log_dir)
    logger.info('=' * 80)
    logger.info('Confidence Network Training with GTCRN')
    logger.info('=' * 80)
    logger.info(f'SNR values: {args.snr_values}')
    logger.info(f'Arguments: {vars(args)}')

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')

    # 创建数据集
    logger.info('\nLoading datasets...')

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

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=lambda batch: collate_fn_fixed_length(batch, target_length=48000)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=lambda batch: collate_fn_fixed_length(batch, target_length=48000)
    )

    logger.info(f'Train samples: {len(train_dataset)}')
    logger.info(f'Val samples: {len(val_dataset)}')

    # 加载 GTCRN（冻结）
    logger.info('\nLoading GTCRN (frozen)...')
    gtcrn = GTCRNWrapper(
        checkpoint_path=args.gtcrn_checkpoint,
        device=device,
        freeze=True  # 冻结
    )

    # 加载说话人模型（冻结）
    logger.info('Loading speaker model (frozen)...')
    # TODO: 替换为你的实际说话人模型
    # speaker_model = YourSpeakerModel()
    # speaker_model.load_state_dict(torch.load(args.speaker_model_path))
    # speaker_model = speaker_model.to(device)
    # speaker_model.eval()

    # 占位符（你需要替换为实际模型）
    logger.warning('⚠️  Using dummy speaker model! Replace with actual ECAPA-TDNN')

    class DummySpeakerModel(nn.Module):
        def __init__(self, embedding_dim=192):
            super().__init__()
            self.conv = nn.Conv1d(1, 64, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Linear(64, embedding_dim)

        def forward(self, x):
            # x: [batch, 1, samples]
            x = self.conv(x)
            x = self.pool(x).squeeze(-1)
            x = self.fc(x)
            return nn.functional.normalize(x, p=2, dim=1)

    speaker_model = DummySpeakerModel(embedding_dim=args.embedding_dim).to(device)
    speaker_model.eval()

    # 初始化置信度网络
    logger.info('Initializing confidence network...')
    confidence_net = ConfidenceNetwork(
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim
    ).to(device)

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
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
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

        # 训练
        train_metrics = train_epoch(
            gtcrn, speaker_model, confidence_net, train_loader,
            criterion, optimizer, device, epoch
        )

        logger.info(f'Train - Loss: {train_metrics["loss"]:.4f}')

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
            logger.info(f'✅ Saved best model: {checkpoint_path}')

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
            logger.info(f'💾 Saved checkpoint: {checkpoint_path}')

    # 保存最终模型
    final_checkpoint = checkpoint_dir / 'confidence_net_final.pth'
    torch.save({
        'epoch': args.num_epochs,
        'model_state_dict': confidence_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history
    }, final_checkpoint)
    logger.info(f'\n✅ Training complete! Final model saved: {final_checkpoint}')

    # 保存历史
    history_file = checkpoint_dir / 'training_history.json'
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)
    logger.info(f'📊 Training history saved: {history_file}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Confidence Network with GTCRN')

    # 数据
    parser.add_argument('--voxceleb_dir', type=str, required=True)
    parser.add_argument('--musan_dir', type=str, required=True)
    parser.add_argument('--snr_values', type=int, nargs='+',
                        default=[-5, 0, 5, 10, 15])

    # 模型
    parser.add_argument('--gtcrn_checkpoint', type=str,
                        default='checkpoints/gtcrn/gtcrn_best.pth')
    parser.add_argument('--speaker_model_path', type=str,
                        default='checkpoints/speaker_model.pth')
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
    main(args)