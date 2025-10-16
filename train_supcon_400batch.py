"""
阶段2：SupCon扩展训练(400 batches)
基于200 batch的成功结果，继续训练以获得更好性能

目标：验证训练稳定性和收敛性，与Triplet Loss对比
"""

import torch
import torch.nn.functional as F
import os
import subprocess
import sys
import time
import random
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

# 禁用DeepFilter日志
os.environ['DF_DISABLE_LOGGING'] = '1'
original_check_output = subprocess.check_output
subprocess.check_output = lambda *args, **kwargs: b'unknown' if args and 'git' in str(
    args[0]) else original_check_output(*args, **kwargs)

sys.path.append('scripts')
from models_supcon import SupConTrainer
from speaker_embedding import SpeakerEmbeddingExtractor
from speech_enhancer import SpeechEnhancer
from dataset import VoxCelebMusanDataset


class Config:
    """SupCon扩展训练配置"""
    # 数据路径
    voxceleb_dir = 'data/voxceleb1'
    musan_dir = 'data/musan'

    # 训练参数
    batch_size = 32
    learning_rate = 1e-3
    snr_range = (-20, 0)

    # SupCon特有参数
    temperature = 0.07

    # ✅ 关键修改：扩展训练批次
    max_batches = 400  # 从200扩展到400
    checkpoint_interval = 50  # 每50个batch保存一次

    # ✅ 新的输出目录
    checkpoint_dir = 'checkpoints_supcon_400batch'

    # 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 随机种子
    seed = 42


def set_seed(seed):
    """设置所有随机种子确保可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def pad_collate(batch):
    """
    处理不同长度音频的collate函数
    """
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
        # Anchor padding
        audio = item['anchor_noisy']
        if audio.shape[1] < max_anchor:
            padding = torch.zeros(1, max_anchor - audio.shape[1])
            audio = torch.cat([audio, padding], dim=1)
        padded_batch['anchor_noisy'].append(audio)

        # Positive padding
        audio = item['positive_noisy']
        if audio.shape[1] < max_positive:
            padding = torch.zeros(1, max_positive - audio.shape[1])
            audio = torch.cat([audio, padding], dim=1)
        padded_batch['positive_noisy'].append(audio)

        # Negative padding
        audio = item['negative_noisy']
        if audio.shape[1] < max_negative:
            padding = torch.zeros(1, max_negative - audio.shape[1])
            audio = torch.cat([audio, padding], dim=1)
        padded_batch['negative_noisy'].append(audio)

        padded_batch['snr'].append(item['snr'])
        padded_batch['speaker_id'].append(item['speaker_id'])

    # Stack音频数据
    for key in ['anchor_noisy', 'positive_noisy', 'negative_noisy']:
        padded_batch[key] = torch.stack(padded_batch[key])

    return padded_batch


def extract_and_normalize_embeddings(audio_batch, speaker_model, enhancer, device, speaker_id_map):
    """
    提取embeddings并确保L2归一化
    """
    batch_size = len(audio_batch['anchor_noisy'])

    all_noisy_embs = []
    all_enhanced_embs = []
    all_labels = []

    for i in range(batch_size):
        speaker_id_str = audio_batch['speaker_id'][i]

        # 字符串映射为整数
        if speaker_id_str not in speaker_id_map:
            speaker_id_map[speaker_id_str] = len(speaker_id_map)
        speaker_id_int = speaker_id_map[speaker_id_str]

        # === 处理 Anchor（说话人A）===
        anchor_noisy = audio_batch['anchor_noisy'][i]
        anchor_enhanced = enhancer.enhance_audio(anchor_noisy, sr=16000)

        emb_noisy = speaker_model.extract_embedding(audio_tensor=anchor_noisy)
        emb_enhanced = speaker_model.extract_embedding(audio_tensor=anchor_enhanced)

        # L2归一化（SupCon论文要求）
        emb_noisy = emb_noisy.squeeze()
        emb_enhanced = emb_enhanced.squeeze()
        emb_noisy = F.normalize(emb_noisy.unsqueeze(0), p=2, dim=1).squeeze(0)
        emb_enhanced = F.normalize(emb_enhanced.unsqueeze(0), p=2, dim=1).squeeze(0)

        all_noisy_embs.append(emb_noisy)
        all_enhanced_embs.append(emb_enhanced)
        all_labels.append(speaker_id_int)

        # === 处理 Positive（说话人A，不同utterance）===
        pos_noisy = audio_batch['positive_noisy'][i]
        pos_enhanced = enhancer.enhance_audio(pos_noisy, sr=16000)

        emb_noisy = speaker_model.extract_embedding(audio_tensor=pos_noisy)
        emb_enhanced = speaker_model.extract_embedding(audio_tensor=pos_enhanced)

        emb_noisy = F.normalize(emb_noisy.squeeze().unsqueeze(0), p=2, dim=1).squeeze(0)
        emb_enhanced = F.normalize(emb_enhanced.squeeze().unsqueeze(0), p=2, dim=1).squeeze(0)

        all_noisy_embs.append(emb_noisy)
        all_enhanced_embs.append(emb_enhanced)
        all_labels.append(speaker_id_int)

        # === 处理 Negative（说话人B，不同人）===
        neg_noisy = audio_batch['negative_noisy'][i]
        neg_enhanced = enhancer.enhance_audio(neg_noisy, sr=16000)

        emb_noisy = speaker_model.extract_embedding(audio_tensor=neg_noisy)
        emb_enhanced = speaker_model.extract_embedding(audio_tensor=neg_enhanced)

        emb_noisy = F.normalize(emb_noisy.squeeze().unsqueeze(0), p=2, dim=1).squeeze(0)
        emb_enhanced = F.normalize(emb_enhanced.squeeze().unsqueeze(0), p=2, dim=1).squeeze(0)

        all_noisy_embs.append(emb_noisy)
        all_enhanced_embs.append(emb_enhanced)
        all_labels.append(speaker_id_int + 100000)

    # 堆叠成batch
    noisy_embeddings = torch.stack(all_noisy_embs)
    enhanced_embeddings = torch.stack(all_enhanced_embs)
    labels = torch.tensor(all_labels, dtype=torch.long)

    return noisy_embeddings, enhanced_embeddings, labels


def save_checkpoint(trainer, loss, batch_idx, total_time, config, is_final=False):
    """保存checkpoint"""
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)

    if is_final:
        filename = 'final_model_supcon_400batch.pth'
    else:
        filename = f'checkpoint_supcon_batch_{batch_idx}.pth'

    checkpoint_path = checkpoint_dir / filename

    torch.save({
        'model_state_dict': trainer.mlp.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'loss': loss,
        'batch': batch_idx,
        'training_minutes': total_time,
        'config': {
            'loss_type': 'SupCon',
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate,
            'temperature': config.temperature,
            'optimizer': 'AdamW',
            'snr_range': config.snr_range,
            'max_batches': config.max_batches,
            'seed': config.seed
        }
    }, checkpoint_path)

    return checkpoint_path


def main():
    config = Config()
    set_seed(config.seed)

    print('=' * 80)
    print('🚀 SUPCON EXTENDED TRAINING - 400 Batches')
    print('=' * 80)
    print(f'📌 Loss Function: SupCon')
    print(f'🎯 Goal: Extended training for better convergence')
    print(f'📚 Reference: Supervised Contrastive Learning (NeurIPS 2020)')
    print(f'💻 Official Code: https://github.com/HobbitLong/SupContrast')
    print('=' * 80)
    print(f'Device: {config.device}')
    print(f'Random seed: {config.seed}')
    print(f'Batch size: {config.batch_size}')
    print(f'Learning rate: {config.learning_rate}')
    print(f'Temperature: {config.temperature} 🌡️')
    print(f'Max batches: {config.max_batches} ⬆️ (extended from 200)')
    print(f'Checkpoint dir: {config.checkpoint_dir} ✨ (new directory)')
    print('=' * 80)

    # [1/3] 加载模型
    print('\n[1/3] Loading models...')
    speaker_model = SpeakerEmbeddingExtractor('ecapa')
    enhancer = SpeechEnhancer()

    trainer = SupConTrainer(
        embedding_dim=192,
        device=config.device,
        temperature=config.temperature
    )

    trainer.optimizer = torch.optim.AdamW(
        trainer.mlp.parameters(),
        lr=config.learning_rate,
        weight_decay=1e-4
    )

    print(f'✅ SupCon Trainer initialized')
    print(f'   - Temperature: {config.temperature}')
    print(f'   - MLP parameters: {sum(p.numel() for p in trainer.mlp.parameters()):,}')
    print(f'   - Optimizer: AdamW (lr={config.learning_rate}, wd=1e-4)')

    # [2/3] 加载数据集
    print('\n[2/3] Loading dataset...')
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

    print(f'✅ Training samples: {len(train_dataset)}')
    print(f'   - SNR range: {config.snr_range} dB')

    # [3/3] 训练
    print('\n[3/3] Training with SupCon Loss (400 batches)...')
    print('=' * 80)

    start_time = time.time()
    total_loss = 0.0
    loss_history = []

    speaker_id_map = {}

    trainer.mlp.train()

    for batch_idx, audio_batch in enumerate(train_loader, 1):
        batch_start = time.time()

        noisy_embs, enhanced_embs, labels = extract_and_normalize_embeddings(
            audio_batch, speaker_model, enhancer, config.device, speaker_id_map
        )

        # 移到GPU
        noisy_embs = noisy_embs.to(config.device)
        enhanced_embs = enhanced_embs.to(config.device)
        labels = labels.to(config.device)

        # 通过MLP获得
        robust_embs = trainer.mlp(noisy_embs, enhanced_embs)

        # 检查归一化（第一个batch）
        if batch_idx == 1:
            norms = torch.norm(robust_embs, p=2, dim=1)
            print(f'\n🔍 Checking normalization (batch 1):')
            print(f'   - Robust emb norms: min={norms.min():.4f}, max={norms.max():.4f}, mean={norms.mean():.4f}')
            if not torch.allclose(norms, torch.ones_like(norms), atol=1e-4):
                print(f'   ⚠️  Warning: Features not perfectly normalized!')
            else:
                print(f'   ✅ Features properly normalized')
            print(f'   - Unique speakers in batch: {len(speaker_id_map)}')
            print()

        # 计算SupCon损失
        loss = trainer.criterion(robust_embs, labels)

        # 反向传播
        trainer.optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(trainer.mlp.parameters(), max_norm=1.0)

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
            print(f'  💾 Checkpoint saved: {checkpoint_path.name}')

        if batch_idx >= config.max_batches:
            break

    # 训练完成
    total_time = (time.time() - start_time) / 60
    final_loss = total_loss / config.max_batches

    print('\n' + '=' * 80)
    print('✅ TRAINING COMPLETED')
    print('=' * 80)
    print(f'📊 Final loss: {final_loss:.4f}')
    print(f'⏱️  Total time: {total_time:.1f} minutes')
    print(f'📈 Avg loss: {sum(loss_history) / len(loss_history):.4f}')
    print(f'📉 Min loss: {min(loss_history):.4f}')
    print(f'📈 Max loss: {max(loss_history):.4f}')
    print(f'👥 Total unique speakers: {len(speaker_id_map)}')
    print('=' * 80)

    # 保存最终模型
    final_path = save_checkpoint(
        trainer,
        final_loss,
        config.max_batches,
        total_time,
        config,
        is_final=True
    )

    print(f'\n💾 Final model saved: {final_path}')

    # 保存loss历史
    loss_file = Path(config.checkpoint_dir) / 'loss_history_supcon_400batch.txt'
    with open(loss_file, 'w') as f:
        f.write('# SupCon Loss History (400 batches)\n')
        f.write(f'# Temperature: {config.temperature}\n')
        f.write(f'# Batch_size: {config.batch_size}\n')
        f.write(f'# Total_speakers: {len(speaker_id_map)}\n')
        f.write(f'# Reference: https://github.com/HobbitLong/SupContrast\n')
        f.write('# Batch\tLoss\n')
        for i, loss in enumerate(loss_history, 1):
            f.write(f'{i}\t{loss:.6f}\n')
    print(f'📄 Loss history saved: {loss_file}')

    # 分析训练稳定性
    print('\n' + '=' * 80)
    print('📊 TRAINING STABILITY ANALYSIS')
    print('=' * 80)

    loss_std = np.std(loss_history)
    loss_mean = np.mean(loss_history)
    cv = loss_std / loss_mean

    print(f'Loss Mean: {loss_mean:.4f}')
    print(f'Loss Std: {loss_std:.4f}')
    print(f'Coefficient of Variation: {cv:.4f}')

    last_10_avg = np.mean(loss_history[-10:])
    first_10_avg = np.mean(loss_history[:10])
    improvement = (first_10_avg - last_10_avg) / first_10_avg * 100

    print(f'\nFirst 10 batches avg loss: {first_10_avg:.4f}')
    print(f'Last 10 batches avg loss: {last_10_avg:.4f}')
    print(f'Improvement: {improvement:.2f}%')

    # 与200 batch对比
    print('\n📊 Comparison with 200-batch training:')
    print(f'   200-batch final loss: 2.7775 (from previous run)')
    print(f'   400-batch final loss: {final_loss:.4f}')
    if final_loss < 2.7775:
        improvement_pct = (2.7775 - final_loss) / 2.7775 * 100
        print(f'   ✅ Improvement: {improvement_pct:.2f}%')
    else:
        print(f'   ➖ Similar performance')

    if improvement > 0:
        print('\n✅ Model is converging')
    else:
        print('\n⚠️  Model may need more training')

    print('=' * 80)

    # 保存分析结果
    analysis_file = Path(config.checkpoint_dir) / 'training_analysis_400batch.txt'
    with open(analysis_file, 'w') as f:
        f.write('SupCon Training Analysis (400 batches)\n')
        f.write('=' * 80 + '\n')
        f.write(f'Loss Mean: {loss_mean:.4f}\n')
        f.write(f'Loss Std: {loss_std:.4f}\n')
        f.write(f'CV: {cv:.4f}\n')
        f.write(f'Improvement: {improvement:.2f}%\n')
        f.write(f'Total time: {total_time:.1f} min\n')
        f.write(f'Temperature: {config.temperature}\n')
        f.write(f'Unique speakers: {len(speaker_id_map)}\n')
        f.write(f'Comparison with 200-batch: {final_loss:.4f} vs 2.7775\n')
        f.write('Reference: https://github.com/HobbitLong/SupContrast\n')

    print(f'\n📄 Analysis saved: {analysis_file}')
    print('\n🎉 All done! Next: Test with evaluate_supcon.py --num_pairs 10000')


if __name__ == '__main__':
    main()