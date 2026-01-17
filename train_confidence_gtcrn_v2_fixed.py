#!/usr/bin/env python3
"""
置信度网络训练脚本 v2 - 修复版
修复: 不使用遍历数据集的BalancedBatchSampler，使用普通DataLoader
参考: 之前能正常运行的 train_confidence_gtcrn_fixed.py
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import argparse
import logging
import json

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'scripts'))

from gtcrn_wrapper_fixed import GTCRNWrapper
from dataset_fixed import VoxCelebMusanDataset
from fixed_snr_dataset import FixedSNRDataset, collate_fn_fixed_length

try:
    from speechbrain.inference.speaker import EncoderClassifier
except ImportError:
    from speechbrain.pretrained import EncoderClassifier


class ECAPATDNNWrapper(nn.Module):
    """ECAPA-TDNN wrapper for speaker embedding extraction"""

    def __init__(self, model_path, device='cuda', freeze=True):
        super().__init__()
        self.device = device
        self.freeze = freeze
        self.model_path = Path(model_path).resolve().as_posix()

        self.classifier = EncoderClassifier.from_hparams(
            source=self.model_path,
            savedir=self.model_path,
            run_opts={"device": self.device}
        )
        if self.freeze:
            self.eval()

    def forward(self, audio):
        if audio.dim() == 3:
            audio = audio.squeeze(1)

        with torch.no_grad():
            embeddings = self.classifier.encode_batch(audio)
            if isinstance(embeddings, tuple):
                embeddings = embeddings[0]
            while embeddings.dim() > 2:
                embeddings = embeddings.squeeze(1)

        return F.normalize(embeddings, p=2, dim=1)


class ConfidenceNetworkV2(nn.Module):
    """改进的置信度网络 - 所有输出都受attention控制"""

    def __init__(self, embedding_dim=192, hidden_dim=256):
        super().__init__()

        self.noisy_transform = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim)
        )

        self.enhanced_transform = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim)
        )

        self.attention = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, emb_noisy, emb_enhanced):
        noisy_feat = self.noisy_transform(emb_noisy)
        enhanced_feat = self.enhanced_transform(emb_enhanced)

        concat = torch.cat([emb_noisy, emb_enhanced], dim=1)
        weights = self.attention(concat)
        w_noisy = weights[:, 0:1]
        w_enhanced = weights[:, 1:2]

        # 所有输出都受attention控制
        output = w_noisy * (emb_noisy + noisy_feat) + w_enhanced * (emb_enhanced + enhanced_feat)
        return F.normalize(output, p=2, dim=1)


class ContrastiveLoss(nn.Module):
    """Supervised Contrastive Loss"""

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, labels):
        device = embeddings.device
        batch_size = embeddings.shape[0]

        embeddings = F.normalize(embeddings, p=2, dim=1)
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        logits_mask = torch.ones_like(mask) - torch.eye(batch_size).to(device)
        mask = mask * logits_mask

        # For numerical stability
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        mask_sum = mask.sum(1)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum
        return -mean_log_prob_pos.mean()


def setup_logger(log_dir):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'training_v2_{timestamp}.log'

    logger = logging.getLogger('confidence_training_v2')
    logger.setLevel(logging.INFO)
    logger.handlers = []

    fh = logging.FileHandler(log_file)
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def train_epoch(gtcrn, speaker_model, confidence_net, dataloader, criterion, optimizer, device, epoch, logger):
    """训练一个epoch"""
    confidence_net.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')

    for batch_idx, batch in enumerate(pbar):
        try:
            noisy = batch['anchor_noisy'].to(device)
            speaker_ids = batch['speaker_id']

            # 构建标签 - 每个batch内独立映射
            unique_speakers = list(set(speaker_ids))
            speaker_to_idx = {spk: idx for idx, spk in enumerate(unique_speakers)}
            labels = torch.tensor([speaker_to_idx[spk] for spk in speaker_ids]).to(device)

            # 跳过说话人太少的batch
            if len(unique_speakers) < 2:
                continue

            with torch.no_grad():
                enhanced = gtcrn.enhance(noisy)
                emb_noisy = speaker_model(noisy)
                emb_enhanced = speaker_model(enhanced)

            emb_fused = confidence_net(emb_noisy, emb_enhanced)
            loss = criterion(emb_fused, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(confidence_net.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if batch_idx % 100 == 0:
                n_spk = len(unique_speakers)
                logger.info(
                    f"[Epoch {epoch}][Batch {batch_idx}] Loss={loss.item():.4f}, Speakers={n_spk}/{len(speaker_ids)}")

            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'spk': len(unique_speakers)})

        except Exception as e:
            print(f"\nError in batch {batch_idx}: {e}")
            continue

    return {'loss': total_loss / max(num_batches, 1)}


def validate(gtcrn, speaker_model, confidence_net, dataloader, criterion, device):
    """验证"""
    confidence_net.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation', leave=False):
            try:
                noisy = batch['anchor_noisy'].to(device)
                speaker_ids = batch['speaker_id']

                unique_speakers = list(set(speaker_ids))
                speaker_to_idx = {spk: idx for idx, spk in enumerate(unique_speakers)}
                labels = torch.tensor([speaker_to_idx[spk] for spk in speaker_ids]).to(device)

                if len(unique_speakers) < 2:
                    continue

                enhanced = gtcrn.enhance(noisy)
                emb_noisy = speaker_model(noisy)
                emb_enhanced = speaker_model(enhanced)
                emb_fused = confidence_net(emb_noisy, emb_enhanced)

                loss = criterion(emb_fused, labels)
                total_loss += loss.item()
                num_batches += 1

            except Exception as e:
                continue

    return {'loss': total_loss / max(num_batches, 1)}


def main(args):
    # Setup
    logger = setup_logger(args.log_dir)
    logger.info('=' * 80)
    logger.info('Confidence Network Training V2 (Fixed)')
    logger.info('=' * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load GTCRN
    logger.info('\n[1/4] Loading GTCRN...')
    gtcrn = GTCRNWrapper(
        checkpoint_path=args.gtcrn_checkpoint,
        device=device,
        freeze=True
    )

    # 2. Load ECAPA-TDNN
    logger.info('\n[2/4] Loading ECAPA-TDNN...')
    speaker_model = ECAPATDNNWrapper(
        model_path=args.speaker_model_path,
        device=device,
        freeze=True
    )

    # 3. Create ConfidenceNetwork
    logger.info('\n[3/4] Creating ConfidenceNetworkV2...')
    confidence_net = ConfidenceNetworkV2(
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim
    ).to(device)
    logger.info(f'   Parameters: {sum(p.numel() for p in confidence_net.parameters())}')

    # 4. Load datasets - 使用普通DataLoader，不用BalancedBatchSampler
    logger.info('\n[4/4] Loading datasets...')

    print("Loading train dataset...")
    train_base = VoxCelebMusanDataset(
        voxceleb_dir=args.voxceleb_dir,
        musan_dir=args.musan_dir,
        split='train',
        snr_range=(-5, 15),
        return_clean=False
    )
    train_dataset = FixedSNRDataset(train_base, snr_values=args.snr_values)

    # 使用shuffle=True的普通DataLoader，不需要BalancedBatchSampler
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,  # 随机采样
        num_workers=args.num_workers,
        collate_fn=collate_fn_fixed_length,
        pin_memory=True,
        drop_last=True  # 丢弃最后一个不完整的batch
    )

    print("Loading test dataset...")
    test_base = VoxCelebMusanDataset(
        voxceleb_dir=args.voxceleb_dir,
        musan_dir=args.musan_dir,
        split='test',
        snr_range=(-5, 15),
        return_clean=False
    )
    test_dataset = FixedSNRDataset(test_base, snr_values=args.snr_values)

    val_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn_fixed_length,
        pin_memory=True
    )

    logger.info(f'   Train batches: {len(train_loader)}')
    logger.info(f'   Val batches: {len(val_loader)}')

    # Criterion and optimizer
    criterion = ContrastiveLoss(temperature=args.temperature)
    optimizer = torch.optim.AdamW(
        confidence_net.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': []
    }
    best_val_loss = float('inf')

    logger.info('\n' + '=' * 80)
    logger.info('Starting training...')
    logger.info('=' * 80)

    for epoch in range(1, args.num_epochs + 1):
        logger.info(f'\n--- Epoch {epoch}/{args.num_epochs} ---')

        # Train
        train_stats = train_epoch(
            gtcrn, speaker_model, confidence_net,
            train_loader, criterion, optimizer, device, epoch, logger
        )

        # Validate
        val_stats = validate(
            gtcrn, speaker_model, confidence_net,
            val_loader, criterion, device
        )

        # Update scheduler
        scheduler.step(val_stats['loss'])

        # Log
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f'Train Loss: {train_stats["loss"]:.4f}')
        logger.info(f'Val Loss: {val_stats["loss"]:.4f}')
        logger.info(f'Learning Rate: {current_lr:.6f}')

        # Save history
        history['train_loss'].append(train_stats['loss'])
        history['val_loss'].append(val_stats['loss'])
        history['learning_rate'].append(current_lr)

        # Save best model
        if val_stats['loss'] < best_val_loss:
            best_val_loss = val_stats['loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': confidence_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_stats['loss'],
            }, checkpoint_dir / 'confidence_net_best.pth')
            logger.info(f'   Saved best model (val_loss={val_stats["loss"]:.4f})')

        # Save checkpoint periodically
        if epoch % args.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': confidence_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_dir / f'confidence_net_epoch{epoch}.pth')

    # Save final model
    torch.save({
        'epoch': args.num_epochs,
        'model_state_dict': confidence_net.state_dict(),
    }, checkpoint_dir / 'confidence_net_final.pth')

    # Save history
    with open(checkpoint_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    logger.info('\n' + '=' * 80)
    logger.info('Training complete!')
    logger.info(f'Best validation loss: {best_val_loss:.4f}')
    logger.info(f'Final model saved: {checkpoint_dir / "confidence_net_final.pth"}')
    logger.info('=' * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--voxceleb_dir', type=str, required=True)
    parser.add_argument('--musan_dir', type=str, required=True)
    parser.add_argument('--gtcrn_checkpoint', type=str, default='checkpoints/gtcrn/gtcrn_best.pth')
    parser.add_argument('--speaker_model_path', type=str, default='pretrained_models/spkrec-ecapa-voxceleb')
    parser.add_argument('--snr_values', type=int, nargs='+', default=[-5, 0, 5, 10, 15])
    parser.add_argument('--embedding_dim', type=int, default=192)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--temperature', type=float, default=0.05)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/confidence_v2')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--save_interval', type=int, default=5)

    args = parser.parse_args()
    main(args)