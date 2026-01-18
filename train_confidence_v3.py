#!/usr/bin/env python3
"""
置信度网络V3 - 修复版本
================================================================================
核心改进:
1. 不需要显式SNR输入，网络从嵌入差异中自动推断
2. 添加辅助损失，防止权重坍塌到只用noisy
3. 嵌入相似度引导：noisy和enhanced差异大时，更多使用enhanced

使用方法 (用cat >复制到服务器):
--------------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=2 python train_confidence_v3.py \
    --voxceleb_dir data/voxceleb1 \
    --musan_dir data/musan \
    --gtcrn_checkpoint checkpoints/gtcrn/gtcrn_best.pth \
    --speaker_model_path pretrained_models/spkrec-ecapa-voxceleb \
    --speakers_per_batch 32 \
    --samples_per_speaker 2 \
    --num_epochs 30 \
    --num_workers 24
================================================================================
"""

import os
import sys
import argparse
import logging
import random
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler
import numpy as np
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# ==============================================================================
# 改进的置信度网络V3
# ==============================================================================
class ConfidenceNetworkV3(nn.Module):
    """
    改进的置信度网络 - 不需要SNR输入

    核心思路:
    1. 计算noisy和enhanced嵌入的相似度
    2. 相似度低 → SNR低 → 更多使用enhanced
    3. 相似度高 → SNR高 → 两者都可用
    """

    def __init__(self, embedding_dim=192, hidden_dim=256):
        super().__init__()
        self.embedding_dim = embedding_dim

        # 嵌入差异编码器
        # 输入: |emb_noisy - emb_enhanced| 和 emb_noisy * emb_enhanced
        self.diff_encoder = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # Attention机制
        # 输入: emb_noisy(192) + emb_enhanced(192) + diff_feat(32) = 416
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim * 2 + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

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

        # 初始化
        self._init_weights()

    def _init_weights(self):
        """初始化权重，让初始attention权重接近均匀"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # 特别处理attention最后一层，初始输出接近[0, 0]
        # softmax([0,0]) = [0.5, 0.5]，初始权重均匀
        nn.init.zeros_(self.attention[-1].weight)
        nn.init.zeros_(self.attention[-1].bias)

    def forward(self, emb_noisy, emb_enhanced):
        """
        Args:
            emb_noisy: [batch, 192] noisy音频的嵌入
            emb_enhanced: [batch, 192] enhanced音频的嵌入

        Returns:
            fused: [batch, 192] 融合后的嵌入
            weights: [batch, 2] attention权重 [w_noisy, w_enhanced]
        """
        # 计算嵌入差异特征
        diff = torch.abs(emb_noisy - emb_enhanced)  # 绝对差
        prod = emb_noisy * emb_enhanced  # 元素积
        diff_feat = self.diff_encoder(torch.cat([diff, prod], dim=1))

        # 拼接所有特征
        concat = torch.cat([emb_noisy, emb_enhanced], dim=1)
        concat_with_diff = torch.cat([concat, diff_feat], dim=1)

        # 计算attention权重
        logits = self.attention(concat_with_diff)
        weights = F.softmax(logits, dim=1)  # [batch, 2]

        # 加权融合
        weighted = weights[:, 0:1] * emb_noisy + weights[:, 1:2] * emb_enhanced

        # 额外的非线性融合
        fusion_out = self.fusion(concat)

        # 残差连接
        output = fusion_out + weighted

        # 归一化输出
        output = F.normalize(output, p=2, dim=1)

        return output, weights


# ==============================================================================
# 改进的损失函数
# ==============================================================================
class ImprovedContrastiveLoss(nn.Module):
    """
    改进的对比损失

    包含:
    1. 主损失: InfoNCE对比损失
    2. 辅助损失1: 防止权重坍塌（熵正则化）
    3. 辅助损失2: 基于嵌入差异的权重引导
    """

    def __init__(self, temperature=0.07, entropy_weight=0.1, guidance_weight=0.1):
        super().__init__()
        self.temperature = temperature
        self.entropy_weight = entropy_weight
        self.guidance_weight = guidance_weight

    def forward(self, embeddings, labels, weights=None, emb_noisy=None, emb_enhanced=None):
        """
        Args:
            embeddings: [batch, dim] 融合后的嵌入
            labels: [batch] 说话人标签
            weights: [batch, 2] attention权重（可选）
            emb_noisy: [batch, dim] noisy嵌入（可选，用于引导损失）
            emb_enhanced: [batch, dim] enhanced嵌入（可选）
        """
        device = embeddings.device
        batch_size = embeddings.shape[0]

        # 1. 主损失: InfoNCE
        similarity_matrix = torch.mm(embeddings, embeddings.T) / self.temperature

        # 创建正样本mask
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # 移除对角线（自己和自己）
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
        mask = mask * logits_mask

        # 计算损失
        exp_logits = torch.exp(similarity_matrix) * logits_mask
        log_prob = similarity_matrix - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

        # 只计算正样本对的损失
        mask_sum = mask.sum(dim=1)
        mask_sum = torch.clamp(mask_sum, min=1)
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask_sum
        main_loss = -mean_log_prob_pos.mean()

        # 2. 辅助损失: 熵正则化（防止权重坍塌）
        entropy_loss = torch.tensor(0.0, device=device)
        if weights is not None:
            # 计算权重的熵，熵越高表示权重越均匀
            entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=1)
            # 我们希望熵不要太低（避免坍塌到[1,0]或[0,1]）
            min_entropy = 0.5  # 目标最小熵
            entropy_loss = F.relu(min_entropy - entropy).mean()

        # 3. 辅助损失: 嵌入差异引导
        guidance_loss = torch.tensor(0.0, device=device)
        if weights is not None and emb_noisy is not None and emb_enhanced is not None:
            # 计算noisy和enhanced的余弦相似度
            cos_sim = F.cosine_similarity(emb_noisy, emb_enhanced, dim=1)

            # 相似度低 → 应该更多用enhanced（因为noisy质量差）
            # 相似度高 → 两者差不多，可以均匀使用
            # 目标enhanced权重 = 0.3 + 0.4 * (1 - cos_sim)
            # 当cos_sim=1时，目标=0.3；当cos_sim=0时，目标=0.7
            target_enhanced = 0.3 + 0.4 * (1 - cos_sim)

            # 引导损失
            guidance_loss = F.mse_loss(weights[:, 1], target_enhanced)

        # 总损失
        total_loss = main_loss + self.entropy_weight * entropy_loss + self.guidance_weight * guidance_loss

        return total_loss, {
            'main': main_loss.item(),
            'entropy': entropy_loss.item(),
            'guidance': guidance_loss.item()
        }


# ==============================================================================
# 数据采样器
# ==============================================================================
class EfficientBalancedBatchSampler(Sampler):
    """高效的平衡批次采样器"""

    def __init__(self, dataset, speakers_per_batch=32, samples_per_speaker=2, n_batches=2000):
        self.dataset = dataset
        self.speakers_per_batch = speakers_per_batch
        self.samples_per_speaker = samples_per_speaker
        self.n_batches = n_batches

        # 构建speaker到indices的映射
        self.speaker_to_indices = {}
        for idx, item in enumerate(dataset.file_list):
            speaker_id = item['speaker_id']
            if speaker_id not in self.speaker_to_indices:
                self.speaker_to_indices[speaker_id] = []
            self.speaker_to_indices[speaker_id].append(idx)

        # 过滤掉样本不足的说话人
        self.valid_speakers = [
            spk for spk, indices in self.speaker_to_indices.items()
            if len(indices) >= samples_per_speaker
        ]

        logger.info(f"Total speakers: {len(self.valid_speakers)}")
        logger.info(
            f"Batch size: {speakers_per_batch} speakers × {samples_per_speaker} samples = {speakers_per_batch * samples_per_speaker}")
        logger.info(f"Total batches per epoch: {self.n_batches}")

    def __iter__(self):
        for _ in range(self.n_batches):
            batch_indices = []
            selected_speakers = random.sample(self.valid_speakers,
                                              min(self.speakers_per_batch, len(self.valid_speakers)))

            for speaker in selected_speakers:
                indices = self.speaker_to_indices[speaker]
                if len(indices) >= self.samples_per_speaker:
                    selected = random.sample(indices, self.samples_per_speaker)
                    batch_indices.extend(selected)

            yield batch_indices

    def __len__(self):
        return self.n_batches


def collate_fn(batch):
    """整理batch数据"""
    keys = batch[0].keys()
    result = {}
    for key in keys:
        if key == 'speaker_id':
            result[key] = [item[key] for item in batch]
        elif isinstance(batch[0][key], torch.Tensor):
            result[key] = torch.stack([item[key] for item in batch])
        else:
            result[key] = [item[key] for item in batch]
    return result


# ==============================================================================
# 训练函数
# ==============================================================================
def train_epoch(model, gtcrn, speaker_model, dataloader, criterion, optimizer, device, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_main = 0
    total_entropy = 0
    total_guidance = 0
    valid_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(pbar):
        try:
            noisy = batch['anchor_noisy'].to(device)
            speaker_ids = batch['speaker_id']

            # 构建标签 - 每个batch内独立编号
            unique_speakers = list(set(speaker_ids))
            speaker_to_idx = {spk: idx for idx, spk in enumerate(unique_speakers)}
            labels = torch.tensor([speaker_to_idx[spk] for spk in speaker_ids]).to(device)

            # GTCRN增强
            with torch.no_grad():
                enhanced = gtcrn.enhance(noisy)

            # 提取嵌入
            with torch.no_grad():
                noisy_2d = noisy.squeeze(1) if noisy.dim() == 3 else noisy
                enhanced_2d = enhanced.squeeze(1) if enhanced.dim() == 3 else enhanced

                emb_noisy = speaker_model.encode_batch(noisy_2d)
                emb_enhanced = speaker_model.encode_batch(enhanced_2d)

                # 处理SpeechBrain返回格式
                if isinstance(emb_noisy, tuple):
                    emb_noisy = emb_noisy[0]
                if isinstance(emb_enhanced, tuple):
                    emb_enhanced = emb_enhanced[0]

                # 确保是2D
                while emb_noisy.dim() > 2:
                    emb_noisy = emb_noisy.squeeze(1)
                while emb_enhanced.dim() > 2:
                    emb_enhanced = emb_enhanced.squeeze(1)

            # 前向传播 - 置信度网络
            fused_emb, weights = model(emb_noisy, emb_enhanced)

            # 计算损失
            loss, loss_dict = criterion(
                fused_emb, labels, weights,
                emb_noisy, emb_enhanced
            )

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # 统计
            total_loss += loss.item()
            total_main += loss_dict['main']
            total_entropy += loss_dict['entropy']
            total_guidance += loss_dict['guidance']
            valid_batches += 1

            # 更新进度条
            avg_noisy_w = weights[:, 0].mean().item()
            avg_enhanced_w = weights[:, 1].mean().item()
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'w_n': f'{avg_noisy_w:.2f}',
                'w_e': f'{avg_enhanced_w:.2f}',
                'spk': len(unique_speakers)
            })

            # 定期日志
            if batch_idx % 50 == 0:
                logger.info(
                    f"[Epoch {epoch}][Batch {batch_idx}] "
                    f"Loss={loss.item():.4f} (main={loss_dict['main']:.4f}, "
                    f"ent={loss_dict['entropy']:.4f}, guide={loss_dict['guidance']:.4f}) "
                    f"W_noisy={avg_noisy_w:.3f}, W_enhanced={avg_enhanced_w:.3f}"
                )

        except Exception as e:
            logger.warning(f"Batch {batch_idx} error: {e}")
            continue

    if valid_batches > 0:
        avg_loss = total_loss / valid_batches
        logger.info(f"[Epoch {epoch}] Avg Loss: {avg_loss:.4f}, Valid Batches: {valid_batches}")
        return {
            'loss': avg_loss,
            'main': total_main / valid_batches,
            'entropy': total_entropy / valid_batches,
            'guidance': total_guidance / valid_batches
        }
    return {'loss': float('inf')}


def validate(model, gtcrn, speaker_model, dataloader, criterion, device):
    """验证"""
    model.eval()
    total_loss = 0
    total_weights_noisy = 0
    total_weights_enhanced = 0
    valid_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", leave=False):
            try:
                noisy = batch['anchor_noisy'].to(device)
                speaker_ids = batch['speaker_id']

                unique_speakers = list(set(speaker_ids))
                speaker_to_idx = {spk: idx for idx, spk in enumerate(unique_speakers)}
                labels = torch.tensor([speaker_to_idx[spk] for spk in speaker_ids]).to(device)

                enhanced = gtcrn.enhance(noisy)

                noisy_2d = noisy.squeeze(1) if noisy.dim() == 3 else noisy
                enhanced_2d = enhanced.squeeze(1) if enhanced.dim() == 3 else enhanced

                emb_noisy = speaker_model.encode_batch(noisy_2d)
                emb_enhanced = speaker_model.encode_batch(enhanced_2d)

                if isinstance(emb_noisy, tuple):
                    emb_noisy = emb_noisy[0]
                if isinstance(emb_enhanced, tuple):
                    emb_enhanced = emb_enhanced[0]

                while emb_noisy.dim() > 2:
                    emb_noisy = emb_noisy.squeeze(1)
                while emb_enhanced.dim() > 2:
                    emb_enhanced = emb_enhanced.squeeze(1)

                fused_emb, weights = model(emb_noisy, emb_enhanced)
                loss, _ = criterion(fused_emb, labels, weights, emb_noisy, emb_enhanced)

                total_loss += loss.item()
                total_weights_noisy += weights[:, 0].mean().item()
                total_weights_enhanced += weights[:, 1].mean().item()
                valid_batches += 1

            except Exception as e:
                continue

    if valid_batches > 0:
        return {
            'loss': total_loss / valid_batches,
            'w_noisy': total_weights_noisy / valid_batches,
            'w_enhanced': total_weights_enhanced / valid_batches
        }
    return {'loss': float('inf')}


# ==============================================================================
# 主函数
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description='Train Confidence Network V3')
    parser.add_argument('--voxceleb_dir', type=str, required=True)
    parser.add_argument('--musan_dir', type=str, required=True)
    parser.add_argument('--gtcrn_checkpoint', type=str, required=True)
    parser.add_argument('--speaker_model_path', type=str, required=True)
    parser.add_argument('--embedding_dim', type=int, default=192)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--entropy_weight', type=float, default=0.1)
    parser.add_argument('--guidance_weight', type=float, default=0.1)
    parser.add_argument('--speakers_per_batch', type=int, default=32)
    parser.add_argument('--samples_per_speaker', type=int, default=2)
    parser.add_argument('--n_batches', type=int, default=2000)
    parser.add_argument('--val_batches', type=int, default=100)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/confidence_v3')
    parser.add_argument('--log_dir', type=str, default='logs')
    args = parser.parse_args()

    # 创建目录
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # 加载GTCRN
    logger.info("Loading GTCRN...")
    from gtcrn_wrapper_fixed import GTCRNWrapper
    gtcrn = GTCRNWrapper(checkpoint_path=args.gtcrn_checkpoint, device=device, freeze=True)

    # 加载说话人模型
    logger.info("Loading speaker model...")
    from speechbrain.pretrained import EncoderClassifier
    speaker_model = EncoderClassifier.from_hparams(
        source=args.speaker_model_path,
        savedir=args.speaker_model_path,
        run_opts={"device": device}
    )

    # 创建数据集
    logger.info("Creating datasets...")
    from dataset_fixed import VoxCelebMusanDataset

    train_dataset = VoxCelebMusanDataset(
        voxceleb_dir=args.voxceleb_dir,
        musan_dir=args.musan_dir,
        split='train',
        snr_range=(-5, 20),  # 宽SNR范围
        return_clean=False
    )

    val_dataset = VoxCelebMusanDataset(
        voxceleb_dir=args.voxceleb_dir,
        musan_dir=args.musan_dir,
        split='test',
        snr_range=(-5, 20),
        return_clean=False
    )

    # 创建采样器和数据加载器
    train_sampler = EfficientBalancedBatchSampler(
        train_dataset,
        speakers_per_batch=args.speakers_per_batch,
        samples_per_speaker=args.samples_per_speaker,
        n_batches=args.n_batches
    )

    val_sampler = EfficientBalancedBatchSampler(
        val_dataset,
        speakers_per_batch=args.speakers_per_batch,
        samples_per_speaker=args.samples_per_speaker,
        n_batches=args.val_batches
    )

    train_loader = DataLoader(
        train_dataset, batch_sampler=train_sampler,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, batch_sampler=val_sampler,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True
    )

    # 创建模型
    logger.info("Creating model...")
    model = ConfidenceNetworkV3(
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim
    ).to(device)

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 损失函数和优化器
    criterion = ImprovedContrastiveLoss(
        temperature=args.temperature,
        entropy_weight=args.entropy_weight,
        guidance_weight=args.guidance_weight
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )

    # 训练循环
    best_val_loss = float('inf')

    logger.info("=" * 70)
    logger.info("Starting training...")
    logger.info("=" * 70)

    for epoch in range(1, args.num_epochs + 1):
        logger.info(f"\n{'=' * 70}")
        logger.info(f"Epoch {epoch}/{args.num_epochs}")
        logger.info(f"{'=' * 70}")

        # 训练
        train_stats = train_epoch(
            model, gtcrn, speaker_model, train_loader,
            criterion, optimizer, device, epoch
        )

        # 验证
        val_stats = validate(
            model, gtcrn, speaker_model, val_loader,
            criterion, device
        )

        # 更新学习率
        scheduler.step(val_stats['loss'])
        current_lr = optimizer.param_groups[0]['lr']

        # 日志
        logger.info(f"Train Loss: {train_stats['loss']:.4f}")
        logger.info(f"Val Loss: {val_stats['loss']:.4f}")
        logger.info(
            f"Val Weights - Noisy: {val_stats.get('w_noisy', 0):.3f}, Enhanced: {val_stats.get('w_enhanced', 0):.3f}")
        logger.info(f"Learning Rate: {current_lr:.6f}")

        # 保存最佳模型
        if val_stats['loss'] < best_val_loss:
            best_val_loss = val_stats['loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_stats['loss'],
                'train_loss': train_stats['loss']
            }, f"{args.checkpoint_dir}/confidence_net_best.pth")
            logger.info(f"   Saved best model (val_loss={val_stats['loss']:.4f})")

        # 定期保存
        if epoch % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_stats['loss']
            }, f"{args.checkpoint_dir}/confidence_net_epoch{epoch}.pth")

    # 保存最终模型
    torch.save({
        'epoch': args.num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, f"{args.checkpoint_dir}/confidence_net_final.pth")

    logger.info("\n" + "=" * 70)
    logger.info("Training complete!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Final model saved: {args.checkpoint_dir}/confidence_net_final.pth")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()