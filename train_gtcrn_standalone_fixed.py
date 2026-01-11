"""
GTCRN 独立训练脚本 - 已修复版本

修复内容：
1. 添加 torchaudio backend 配置
2. 改进错误处理
3. 添加环境检查

用法：
    python train_gtcrn_standalone_fixed.py --voxceleb_dir data/voxceleb1 --musan_dir data/musan
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import argparse
from tqdm import tqdm
import json
from datetime import datetime
import logging

# ============= torchaudio 音频后端配置 =============
import torchaudio
import os

# 新版本 torchaudio 使用 dispatcher，会自动选择 soundfile
# 旧的 set_audio_backend 已弃用，不需要手动设置
print("[OK] torchaudio will automatically use soundfile if available")

# 验证 soundfile 是否可用
try:
    import soundfile

    print(f"[OK] soundfile version: {soundfile.__version__}")
except ImportError:
    print("❌ soundfile not installed!")
    print("Please install: pip install soundfile")
    sys.exit(1)
# ============================================================

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'scripts'))

from gtcrn_wrapper_fixed import GTCRNWrapper
from perceptual_loss_fixed import WavLMPerceptualLoss
from dataset_fixed import VoxCelebMusanDataset
from fixed_snr_dataset import FixedSNRDataset, collate_fn_fixed_length


# ============= Windows 多进程修复 =============
# 将 collate_fn 定义为全局函数，避免 pickle 错误
def custom_collate_fn(batch):
    """全局 collate 函数，用于 Windows 多进程"""
    return collate_fn_fixed_length(batch, target_length=48000)


# ============================================


def check_environment():
    """检查运行环境"""
    print("\n" + "=" * 80)
    print("环境检查")
    print("=" * 80)

    # Python 版本
    print(f"Python: {sys.version}")

    # PyTorch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # torchaudio
    print(f"torchaudio: {torchaudio.__version__}")
    print(f"Audio backend: {torchaudio.get_audio_backend()}")

    # soundfile
    try:
        import soundfile
        print(f"soundfile: {soundfile.__version__}")
    except ImportError:
        print("❌ soundfile not installed!")
        return False

    print("=" * 80 + "\n")
    return True


class GTCRNLoss(nn.Module):
    """GTCRN 训练损失：MSE + SI-SNR"""

    def __init__(self, mse_weight=0.5, sisnr_weight=0.5):
        super().__init__()
        self.mse_weight = mse_weight
        self.sisnr_weight = sisnr_weight

    def si_snr_loss(self, estimate, target, eps=1e-8):
        """Scale-Invariant SNR Loss"""
        target = target - torch.mean(target, dim=-1, keepdim=True)
        estimate = estimate - torch.mean(estimate, dim=-1, keepdim=True)

        dot = torch.sum(estimate * target, dim=-1, keepdim=True)
        s_target_norm = torch.sum(target ** 2, dim=-1, keepdim=True)
        s_target = (dot / (s_target_norm + 1e-8)) * target

        e_noise = estimate - s_target

        si_snr = 10 * torch.log10(
            (torch.sum(s_target ** 2, dim=-1) + 1e-8) /
            (torch.sum(e_noise ** 2, dim=-1) + 1e-8)
        )

        return -torch.mean(si_snr)

    def forward(self, estimate, target):
        if estimate.dim() == 3:
            estimate = estimate.squeeze(1)
        if target.dim() == 3:
            target = target.squeeze(1)

        loss_mse = F.mse_loss(estimate, target)
        loss_sisnr = self.si_snr_loss(estimate, target)

        loss = self.mse_weight * loss_mse + self.sisnr_weight * loss_sisnr

        return loss, {
            'total': loss.item(),
            'mse': loss_mse.item(),
            'si_snr': loss_sisnr.item()
        }


def setup_logger(log_dir):
    """设置日志"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'train_gtcrn_{timestamp}.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__)


def train_epoch(model, dataloader, criterion, perceptual_loss, optimizer,
                device, epoch, use_perceptual=False, perceptual_weight=0.1):
    """训练一个 epoch"""
    model.train()

    total_loss = 0
    total_mse = 0
    total_sisnr = 0
    total_perceptual = 0

    num_batches = 0
    num_errors = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')

    for batch_idx, batch in enumerate(pbar):
        try:
            # 获取数据
            noisy = batch['anchor_noisy'].to(device)
            clean = batch['anchor_clean'].to(device)

            # 前向传播
            enhanced = model.enhance(noisy)

            # 计算主损失
            loss, loss_dict = criterion(enhanced, clean)

            # 可选：感知损失
            if use_perceptual and perceptual_loss is not None:
                loss_perc, _ = perceptual_loss(enhanced, clean)
                loss = loss + perceptual_weight * loss_perc
                loss_dict['perceptual'] = loss_perc.item()

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.get_trainable_params(), max_norm=5.0)
            optimizer.step()

            # 统计
            total_loss += loss_dict['total']
            total_mse += loss_dict['mse']
            total_sisnr += loss_dict['si_snr']
            if 'perceptual' in loss_dict:
                total_perceptual += loss_dict['perceptual']

            num_batches += 1

            # 更新进度条
            pbar.set_postfix({
                'loss': f"{loss_dict['total']:.4f}",
                'mse': f"{loss_dict['mse']:.4f}",
                'sisnr': f"{loss_dict['si_snr']:.2f}dB",
                'errors': num_errors
            })

        except Exception as e:
            num_errors += 1
            print(f"\n⚠️ Error in batch {batch_idx}: {e}")
            if num_errors > 10:
                print("Too many errors, stopping training")
                break
            continue

    if num_batches == 0:
        raise RuntimeError("No batches processed successfully!")

    return {
        'loss': total_loss / num_batches,
        'mse': total_mse / num_batches,
        'si_snr': total_sisnr / num_batches,
        'perceptual': total_perceptual / num_batches if use_perceptual else 0,
        'num_errors': num_errors
    }


def validate(model, dataloader, criterion, device):
    """验证"""
    model.eval()

    total_loss = 0
    total_mse = 0
    total_sisnr = 0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            try:
                noisy = batch['anchor_noisy'].to(device)
                clean = batch['anchor_clean'].to(device)

                enhanced = model.enhance(noisy)
                loss, loss_dict = criterion(enhanced, clean)

                total_loss += loss_dict['total']
                total_mse += loss_dict['mse']
                total_sisnr += loss_dict['si_snr']
                num_batches += 1
            except Exception as e:
                print(f"Validation error: {e}")
                continue

    if num_batches == 0:
        raise RuntimeError("No validation batches processed!")

    return {
        'loss': total_loss / num_batches,
        'mse': total_mse / num_batches,
        'si_snr': total_sisnr / num_batches
    }


def main(args):
    # 环境检查
    if not check_environment():
        print("\n❌ Environment check failed. Please install required packages.")
        print("Run: pip install soundfile")
        return 1

    # 设置日志
    logger = setup_logger(args.log_dir)
    logger.info('=' * 80)
    logger.info('GTCRN Training with Fixed SNR Values')
    logger.info('=' * 80)
    logger.info(f'SNR values: {args.snr_values}')
    logger.info(f'Arguments: {vars(args)}')

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')

    # 检查数据集路径
    voxceleb_dir = Path(args.voxceleb_dir)
    musan_dir = Path(args.musan_dir)

    if not voxceleb_dir.exists():
        logger.error(f"VoxCeleb directory not found: {voxceleb_dir}")
        return 1
    if not musan_dir.exists():
        logger.error(f"MUSAN directory not found: {musan_dir}")
        return 1

    # 创建数据集
    logger.info('\nLoading datasets...')

    try:
        # 基础训练数据集
        base_train_dataset = VoxCelebMusanDataset(
            voxceleb_dir=str(voxceleb_dir),
            musan_dir=str(musan_dir),
            split='train',
            snr_range=(-5, 15),
            return_clean=True
        )

        # 包装为固定 SNR 采样
        train_dataset = FixedSNRDataset(base_train_dataset, snr_values=args.snr_values)

        # 验证数据集
        val_dataset = VoxCelebMusanDataset(
            voxceleb_dir=str(voxceleb_dir),
            musan_dir=str(musan_dir),
            split='test',
            test_snr=0,
            test_noise_type='noise',
            return_clean=True
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

    # 初始化模型
    logger.info('\nInitializing GTCRN...')
    model = GTCRNWrapper(
        checkpoint_path=args.gtcrn_checkpoint,
        device=device,
        freeze=False
    )

    # 损失函数
    criterion = GTCRNLoss(mse_weight=0.5, sisnr_weight=0.5)

    # 感知损失（可选）
    perceptual_loss = None
    if args.use_perceptual:
        logger.info('Loading WavLM for perceptual loss...')
        perceptual_loss = WavLMPerceptualLoss(args.wavlm_path).to(device)

    # 优化器
    optimizer = optim.AdamW(
        model.get_trainable_params(),
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
        'train_mse': [],
        'train_sisnr': [],
        'val_loss': [],
        'val_mse': [],
        'val_sisnr': []
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
        try:
            train_metrics = train_epoch(
                model, train_loader, criterion, perceptual_loss, optimizer,
                device, epoch,
                use_perceptual=args.use_perceptual,
                perceptual_weight=args.perceptual_weight
            )

            logger.info(f'Train - Loss: {train_metrics["loss"]:.4f}, '
                        f'MSE: {train_metrics["mse"]:.4f}, '
                        f'SI-SNR: {train_metrics["si_snr"]:.2f}dB, '
                        f'Errors: {train_metrics["num_errors"]}')

            # 验证
            val_metrics = validate(model, val_loader, criterion, device)

            logger.info(f'Val   - Loss: {val_metrics["loss"]:.4f}, '
                        f'MSE: {val_metrics["mse"]:.4f}, '
                        f'SI-SNR: {val_metrics["si_snr"]:.2f}dB')

            # 更新历史
            history['train_loss'].append(train_metrics['loss'])
            history['train_mse'].append(train_metrics['mse'])
            history['train_sisnr'].append(train_metrics['si_snr'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_mse'].append(val_metrics['mse'])
            history['val_sisnr'].append(val_metrics['si_snr'])

            # 学习率调整
            scheduler.step(val_metrics['loss'])

            # 保存最佳模型
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                checkpoint_path = checkpoint_dir / 'gtcrn_best.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': best_val_loss,
                    'history': history
                }, checkpoint_path)
                logger.info(f'[OK] Saved best model: {checkpoint_path}')

            # 定期保存
            if epoch % args.save_interval == 0:
                checkpoint_path = checkpoint_dir / f'gtcrn_epoch{epoch}.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.model.state_dict(),
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
    final_checkpoint = checkpoint_dir / 'gtcrn_final.pth'
    torch.save({
        'epoch': args.num_epochs,
        'model_state_dict': model.model.state_dict(),
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
    parser = argparse.ArgumentParser(description='Train GTCRN for Speech Enhancement')

    # 数据
    parser.add_argument('--voxceleb_dir', type=str, required=True,
                        help='VoxCeleb1 directory')
    parser.add_argument('--musan_dir', type=str, required=True,
                        help='MUSAN directory')
    parser.add_argument('--snr_values', type=int, nargs='+',
                        default=[-5, 0, 5, 10, 15],
                        help='Fixed SNR values for training (default: -5 0 5 10 15)')

    # 模型
    parser.add_argument('--gtcrn_checkpoint', type=str,
                        default='gtcrn/checkpoints/model_trained_on_vctk.tar',
                        help='GTCRN checkpoint path')
    parser.add_argument('--wavlm_path', type=str, default='D:/WavLM',
                        help='WavLM model path')
    parser.add_argument('--use_perceptual', action='store_true',
                        help='Use WavLM perceptual loss')
    parser.add_argument('--perceptual_weight', type=float, default=0.1,
                        help='Weight for perceptual loss')

    # 训练
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for dataloader')

    # 输出
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/gtcrn',
                        help='Checkpoint directory')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Log directory')
    parser.add_argument('--save_interval', type=int, default=5,
                        help='Save checkpoint every N epochs')

    args = parser.parse_args()
    exit(main(args))