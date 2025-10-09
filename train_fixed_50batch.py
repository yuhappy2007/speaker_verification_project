import torch
import torch.nn.functional as F
import os

os.environ['DF_DISABLE_LOGGING'] = '1'

import subprocess

original_check_output = subprocess.check_output
subprocess.check_output = lambda *args, **kwargs: b'unknown' if args and 'git' in str(
    args[0]) else original_check_output(*args, **kwargs)

from torch.utils.data import DataLoader
from pathlib import Path
import sys
import time
import random
import numpy as np

sys.path.append('scripts')
from models_fixed import SiameseTrainerFixed
from speaker_embedding import SpeakerEmbeddingExtractor
from speech_enhancer import SpeechEnhancer
from dataset import VoxCelebMusanDataset


class Config:
    """训练配置"""
    # 数据路径
    voxceleb_dir = 'data/voxceleb1'
    musan_dir = 'data/musan'

    # 训练参数（严格遵循论文）
    batch_size = 32
    learning_rate = 1e-3
    triplet_margin = 0.25
    snr_range = (-20, 0)

    # ✅ 修复参数
    min_neg_dist = 0.4  # 不同说话人最小距离（防止collapse）

    # 训练控制
    max_batches = 200
    checkpoint_interval = 25

    # 输出
    checkpoint_dir = 'checkpoints_fixed_200batch'

    # 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 随机种子
    seed = 42


def set_seed(seed):
    """设置所有随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pad_collate(batch):
    """处理变长音频的collate函数"""
    max_anchor = max([item['anchor_noisy'].shape[1] for item in batch])
    max_positive = max([item['positive_noisy'].shape[1] for item in batch])
    max_negative = max([item['negative_noisy'].shape[1] for item in batch])

    padded_batch = {
        'anchor_noisy': [],
        'positive_noisy': [],
        'negative_noisy': [],
        'snr': [],
        'speaker_id': []
    }

    for item in batch:
        # Anchor
        audio = item['anchor_noisy']
        if audio.shape[1] < max_anchor:
            padding = torch.zeros(1, max_anchor - audio.shape[1])
            audio = torch.cat([audio, padding], dim=1)
        padded_batch['anchor_noisy'].append(audio)

        # Positive
        audio = item['positive_noisy']
        if audio.shape[1] < max_positive:
            padding = torch.zeros(1, max_positive - audio.shape[1])
            audio = torch.cat([audio, padding], dim=1)
        padded_batch['positive_noisy'].append(audio)

        # Negative
        audio = item['negative_noisy']
        if audio.shape[1] < max_negative:
            padding = torch.zeros(1, max_negative - audio.shape[1])
            audio = torch.cat([audio, padding], dim=1)
        padded_batch['negative_noisy'].append(audio)

        padded_batch['snr'].append(item['snr'])
        padded_batch['speaker_id'].append(item['speaker_id'])

    for key in ['anchor_noisy', 'positive_noisy', 'negative_noisy']:
        padded_batch[key] = torch.stack(padded_batch[key])

    return padded_batch


def extract_embeddings_batch(audio_batch, speaker_model, enhancer, device):
    """批量提取embeddings"""
    batch_size = len(audio_batch['anchor_noisy'])

    embeddings = {
        'anchor_noisy': [],
        'anchor_enhanced': [],
        'positive_noisy': [],
        'positive_enhanced': [],
        'negative_noisy': [],
        'negative_enhanced': [],
    }

    for i in range(batch_size):
        # Anchor
        anchor_noisy = audio_batch['anchor_noisy'][i]
        anchor_enhanced = enhancer.enhance_audio(anchor_noisy, sr=16000)

        emb_noisy = speaker_model.extract_embedding(audio_tensor=anchor_noisy)
        emb_enhanced = speaker_model.extract_embedding(audio_tensor=anchor_enhanced)

        emb_noisy = emb_noisy.squeeze()
        emb_enhanced = emb_enhanced.squeeze()

        emb_noisy = F.normalize(emb_noisy.unsqueeze(0), p=2, dim=1).squeeze(0)
        emb_enhanced = F.normalize(emb_enhanced.unsqueeze(0), p=2, dim=1).squeeze(0)

        embeddings['anchor_noisy'].append(emb_noisy)
        embeddings['anchor_enhanced'].append(emb_enhanced)

        # Positive
        pos_noisy = audio_batch['positive_noisy'][i]
        pos_enhanced = enhancer.enhance_audio(pos_noisy, sr=16000)

        emb_noisy = speaker_model.extract_embedding(audio_tensor=pos_noisy)
        emb_enhanced = speaker_model.extract_embedding(audio_tensor=pos_enhanced)

        emb_noisy = emb_noisy.squeeze()
        emb_enhanced = emb_enhanced.squeeze()

        emb_noisy = F.normalize(emb_noisy.unsqueeze(0), p=2, dim=1).squeeze(0)
        emb_enhanced = F.normalize(emb_enhanced.unsqueeze(0), p=2, dim=1).squeeze(0)

        embeddings['positive_noisy'].append(emb_noisy)
        embeddings['positive_enhanced'].append(emb_enhanced)

        # Negative
        neg_noisy = audio_batch['negative_noisy'][i]
        neg_enhanced = enhancer.enhance_audio(neg_noisy, sr=16000)

        emb_noisy = speaker_model.extract_embedding(audio_tensor=neg_noisy)
        emb_enhanced = speaker_model.extract_embedding(audio_tensor=neg_enhanced)

        emb_noisy = emb_noisy.squeeze()
        emb_enhanced = emb_enhanced.squeeze()

        emb_noisy = F.normalize(emb_noisy.unsqueeze(0), p=2, dim=1).squeeze(0)
        emb_enhanced = F.normalize(emb_enhanced.unsqueeze(0), p=2, dim=1).squeeze(0)

        embeddings['negative_noisy'].append(emb_noisy)
        embeddings['negative_enhanced'].append(emb_enhanced)

    for key in embeddings:
        embeddings[key] = torch.stack(embeddings[key])

    return embeddings


def save_checkpoint(trainer, loss, batch_idx, total_time, config, is_final=False):
    """保存checkpoint"""
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)

    if is_final:
        filename = 'final_model.pth'
    else:
        filename = f'checkpoint_batch_{batch_idx}.pth'

    checkpoint_path = checkpoint_dir / filename

    torch.save({
        'model_state_dict': trainer.mlp.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'loss': loss,
        'batch': batch_idx,
        'training_minutes': total_time,
        'config': {
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate,
            'optimizer': 'AdamW',
            'triplet_margin': config.triplet_margin,
            'min_neg_dist': config.min_neg_dist,  # ✅ 保存新参数
            'snr_range': config.snr_range,
            'max_batches': config.max_batches,
            'seed': config.seed
        }
    }, checkpoint_path)

    return checkpoint_path


def main():
    config = Config()
    set_seed(config.seed)

    print('=' * 70)
    print('FIXED TRAINING - 50 BATCHES WITH COLLAPSE PREVENTION')
    print('=' * 70)
    print(f'Device: {config.device}')
    print(f'Random seed: {config.seed}')
    print(f'Batch size: {config.batch_size}')
    print(f'Learning rate: {config.learning_rate}')
    print(f'Triplet margin: {config.triplet_margin}')
    print(f'Min negative distance: {config.min_neg_dist} ✅ NEW')
    print(f'Max batches: {config.max_batches}')
    print('=' * 70)

    # [1/3] 加载模型
    print('\\n[1/3] Loading models...')
    speaker_model = SpeakerEmbeddingExtractor('ecapa')
    enhancer = SpeechEnhancer()

    # ✅ 使用修复版trainer
    trainer = SiameseTrainerFixed(
        embedding_dim=192,
        device=config.device,
        triplet_margin=config.triplet_margin,
        min_neg_dist=config.min_neg_dist
    )

    trainer.optimizer = torch.optim.AdamW(
        trainer.mlp.parameters(),
        lr=config.learning_rate
    )

    print(f'✓ Using FIXED Triplet Loss (min_neg_dist={config.min_neg_dist})')

    # [2/3] 加载数据集
    print('\\n[2/3] Loading dataset...')
    train_dataset = VoxCelebMusanDataset(
        config.voxceleb_dir,
        config.musan_dir,
        split='train',
        snr_range=config.snr_range
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=pad_collate
    )

    print(f'✓ Training samples: {len(train_dataset)}')

    # [3/3] 训练
    print('\\n[3/3] Training...')
    print('=' * 70)

    start_time = time.time()
    total_loss = 0.0
    loss_history = []

    trainer.mlp.train()

    for batch_idx, audio_batch in enumerate(train_loader, 1):
        batch_start = time.time()

        # 提取embeddings
        embedding_batch = extract_embeddings_batch(
            audio_batch, speaker_model, enhancer, config.device
        )

        # 移到GPU
        for key in embedding_batch:
            embedding_batch[key] = embedding_batch[key].to(config.device)

        # MLP处理
        anchor_robust = trainer.mlp(
            embedding_batch['anchor_noisy'],
            embedding_batch['anchor_enhanced']
        )
        pos_robust = trainer.mlp(
            embedding_batch['positive_noisy'],
            embedding_batch['positive_enhanced']
        )
        neg_robust = trainer.mlp(
            embedding_batch['negative_noisy'],
            embedding_batch['negative_enhanced']
        )

        # 计算loss（使用修复版）
        loss = trainer.criterion(anchor_robust, pos_robust, neg_robust)

        # 反向传播
        trainer.optimizer.zero_grad()
        loss.backward()
        trainer.optimizer.step()

        # 记录
        loss_val = loss.item()
        total_loss += loss_val
        loss_history.append(loss_val)

        batch_time = time.time() - batch_start
        elapsed_total = (time.time() - start_time) / 60

        # 打印进度
        if batch_idx % 5 == 0 or batch_idx == 1:
            avg_loss = total_loss / batch_idx
            print(f'Batch {batch_idx:3d}/{config.max_batches} | '
                  f'Loss: {loss_val:.4f} (avg: {avg_loss:.4f}) | '
                  f'Time: {batch_time:.1f}s | '
                  f'Total: {elapsed_total:.1f}min')

        # 保存checkpoint
        if batch_idx % config.checkpoint_interval == 0:
            checkpoint_path = save_checkpoint(
                trainer,
                total_loss / batch_idx,
                batch_idx,
                elapsed_total,
                config,
                is_final=False
            )
            print(f'  ✓ Checkpoint saved: {checkpoint_path.name}')

        if batch_idx >= config.max_batches:
            break

    # 训练完成
    total_time = (time.time() - start_time) / 60
    final_loss = total_loss / config.max_batches

    print('\\n' + '=' * 70)
    print('TRAINING COMPLETED')
    print('=' * 70)
    print(f'Final loss: {final_loss:.4f}')
    print(f'Total time: {total_time:.1f} minutes')
    print('=' * 70)

    # 保存最终模型
    final_path = save_checkpoint(
        trainer,
        final_loss,
        config.max_batches,
        total_time,
        config,
        is_final=True
    )

    print(f'\\n✓ Final model saved: {final_path}')

    # 保存loss历史
    loss_file = Path(config.checkpoint_dir) / 'loss_history.txt'
    with open(loss_file, 'w') as f:
        for i, loss in enumerate(loss_history, 1):
            f.write(f'{i}\\t{loss:.6f}\\n')
    print(f'✓ Loss history saved: {loss_file}')

    print('\\n' + '=' * 70)



if __name__ == '__main__':
    main()