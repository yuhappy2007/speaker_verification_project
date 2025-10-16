"""
阶段2: 置信度网络训练
- 使用动态加权融合替代固定concat
- 记录权重变化历史
- 与SupCon baseline对比

用法:
    python train_confidence.py
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
import json

# 禁用DeepFilter日志
os.environ['DF_DISABLE_LOGGING'] = '1'
original_check_output = subprocess.check_output
subprocess.check_output = lambda *args, **kwargs: b'unknown' if args and 'git' in str(
    args[0]) else original_check_output(*args, **kwargs)

sys.path.append('scripts')
from models_confidence import ConfidenceSupConTrainer
from speaker_embedding import SpeakerEmbeddingExtractor
from speech_enhancer import SpeechEnhancer
from dataset import VoxCelebMusanDataset


class Config:
    """训练配置"""
    # 数据路径
    voxceleb_dir = 'data/voxceleb1'
    musan_dir = 'data/musan'

    # 训练参数(与SupCon保持一致以便对比)
    batch_size = 32
    learning_rate = 1e-3
    snr_range = (-20, 0)

    # 置信度网络特有参数
    temperature = 0.07
    confidence_hidden_dim = 64  # 可调整: 32, 64, 128

    # 训练控制
    max_batches = 200  # 与之前的400batch实验对齐
    checkpoint_interval = 25
    log_weights_interval = 5  # 每5个batch记录一次权重

    # 输出目录
    checkpoint_dir = 'checkpoints_confidence_400batch'

    # 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 随机种子
    seed = 42


def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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

    # 保持speaker_id为字符串列表
    return padded_batch


def extract_and_normalize_embeddings(audio_batch, speaker_model, enhancer, device, speaker_id_map):
    """
    提取并归一化embeddings

    返回:
        noisy_embeddings: [batch_size*3, 192]
        enhanced_embeddings: [batch_size*3, 192]
        labels: [batch_size*3] (整数)
    """
    batch_size = len(audio_batch['anchor_noisy'])

    all_noisy_embs = []
    all_enhanced_embs = []
    all_labels = []

    for i in range(batch_size):
        speaker_id_str = audio_batch['speaker_id'][i]

        # 映射speaker_id
        if speaker_id_str not in speaker_id_map:
            speaker_id_map[speaker_id_str] = len(speaker_id_map)
        speaker_id_int = speaker_id_map[speaker_id_str]

        # === 处理 Anchor(说话人A) ===
        anchor_noisy = audio_batch['anchor_noisy'][i]
        anchor_enhanced = enhancer.enhance_audio(anchor_noisy, sr=16000)

        emb_noisy = speaker_model.extract_embedding(audio_tensor=anchor_noisy)
        emb_enhanced = speaker_model.extract_embedding(audio_tensor=anchor_enhanced)

        # L2归一化
        emb_noisy = emb_noisy.squeeze()
        emb_enhanced = emb_enhanced.squeeze()
        emb_noisy = F.normalize(emb_noisy.unsqueeze(0), p=2, dim=1).squeeze(0)
        emb_enhanced = F.normalize(emb_enhanced.unsqueeze(0), p=2, dim=1).squeeze(0)

        all_noisy_embs.append(emb_noisy)
        all_enhanced_embs.append(emb_enhanced)
        all_labels.append(speaker_id_int)

        # === 处理 Positive(说话人A,不同utterance) ===
        pos_noisy = audio_batch['positive_noisy'][i]
        pos_enhanced = enhancer.enhance_audio(pos_noisy, sr=16000)

        emb_noisy = speaker_model.extract_embedding(audio_tensor=pos_noisy)
        emb_enhanced = speaker_model.extract_embedding(audio_tensor=pos_enhanced)

        emb_noisy = F.normalize(emb_noisy.squeeze().unsqueeze(0), p=2, dim=1).squeeze(0)
        emb_enhanced = F.normalize(emb_enhanced.squeeze().unsqueeze(0), p=2, dim=1).squeeze(0)

        all_noisy_embs.append(emb_noisy)
        all_enhanced_embs.append(emb_enhanced)
        all_labels.append(speaker_id_int)

        # === 处理 Negative(说话人B,不同人) ===
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


def save_checkpoint(trainer, loss, batch_idx, total_time, config, weight_stats=None, is_final=False):
    """保存checkpoint"""
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)

    if is_final:
        filename = 'final_model_confidence.pth'
    else:
        filename = f'checkpoint_confidence_batch_{batch_idx}.pth'

    checkpoint_path = checkpoint_dir / filename

    checkpoint_data = {
        'model_state_dict': trainer.mlp.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'loss': loss,
        'batch': batch_idx,
        'training_minutes': total_time,
        'config': {
            'loss_type': 'SupCon + Confidence',
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate,
            'temperature': config.temperature,
            'confidence_hidden_dim': config.confidence_hidden_dim,
            'optimizer': 'AdamW',
            'snr_range': config.snr_range,
            'max_batches': config.max_batches,
            'seed': config.seed
        }
    }

    # 保存权重统计
    if weight_stats:
        checkpoint_data['weight_stats'] = weight_stats

    torch.save(checkpoint_data, checkpoint_path)

    return checkpoint_path


def main():
    config = Config()
    set_seed(config.seed)

    print('=' * 80)
    print('🚀 CONFIDENCE NETWORK TRAINING - Dynamic Weighting')
    print('=' * 80)
    print(f'📊 Innovation: Learnable weights for noisy/enhanced fusion')
    print(f'📚 Architecture: Confidence Net + SupCon Loss')
    print(f'🎯 Goal: Adaptive fusion based on embedding quality')
    print('=' * 80)
    print(f'Device: {config.device}')
    print(f'Random seed: {config.seed}')
    print(f'Batch size: {config.batch_size}')
    print(f'Learning rate: {config.learning_rate}')
    print(f'Temperature: {config.temperature} 🌡️')
    print(f'Confidence hidden dim: {config.confidence_hidden_dim}')
    print(f'Max batches: {config.max_batches}')
    print('=' * 80)

    # [1/3] 加载模型
    print('\n[1/3] Loading models...')
    speaker_model = SpeakerEmbeddingExtractor('ecapa')
    enhancer = SpeechEnhancer()

    trainer = ConfidenceSupConTrainer(
        embedding_dim=192,
        device=config.device,
        temperature=config.temperature,
        confidence_hidden_dim=config.confidence_hidden_dim
    )

    trainer.optimizer = torch.optim.AdamW(
        trainer.mlp.parameters(),
        lr=config.learning_rate,
        weight_decay=1e-4
    )

    print(f'✅ All models loaded')
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
    print('\n[3/3] Training with Confidence Network...')
    print('=' * 80)

    start_time = time.time()
    total_loss = 0.0
    loss_history = []
    weight_history = []  # 记录权重变化

    speaker_id_map = {}

    trainer.mlp.train()

    for batch_idx, audio_batch in enumerate(train_loader, 1):
        batch_start = time.time()

        # 提取并归一化embeddings
        noisy_embs, enhanced_embs, labels = extract_and_normalize_embeddings(
            audio_batch, speaker_model, enhancer, config.device, speaker_id_map
        )

        # 移到GPU
        noisy_embs = noisy_embs.to(config.device)
        enhanced_embs = enhanced_embs.to(config.device)
        labels = labels.to(config.device)

        # 训练一步(记录权重)
        log_weights = (batch_idx % config.log_weights_interval == 0) or (batch_idx == 1)
        loss, stats = trainer.train_step(
            noisy_embs, enhanced_embs, labels, log_weights=log_weights
        )

        # 记录
        loss_history.append(loss)
        total_loss += loss

        if log_weights:
            weight_history.append({
                'batch': batch_idx,
                'w_noisy': stats['w_noisy_mean'],
                'w_enhanced': stats['w_enhanced_mean'],
                'w_noisy_std': stats['w_noisy_std'],
                'w_enhanced_std': stats['w_enhanced_std']
            })

        batch_time = time.time() - batch_start
        elapsed_total = (time.time() - start_time) / 60

        # 打印进度
        if batch_idx % 5 == 0 or batch_idx == 1:
            avg_loss = total_loss / batch_idx
            msg = f'Batch {batch_idx:3d}/{config.max_batches} | Loss: {loss:.4f} (avg: {avg_loss:.4f})'

            if log_weights:
                msg += f' | w_n={stats["w_noisy_mean"]:.3f} w_e={stats["w_enhanced_mean"]:.3f}'

            msg += f' | Time: {batch_time:.1f}s | Total: {elapsed_total:.1f}min'
            print(msg)

        # 保存checkpoint
        if batch_idx % config.checkpoint_interval == 0:
            current_weight_stats = trainer.get_weight_statistics()
            checkpoint_path = save_checkpoint(
                trainer,
                total_loss / batch_idx,
                batch_idx,
                elapsed_total,
                config,
                weight_stats=current_weight_stats,
                is_final=False
            )
            print(f'  💾 Checkpoint saved: {checkpoint_path.name}')

        if batch_idx >= config.max_batches:
            break

    # === 训练完成 ===
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
    final_weight_stats = trainer.get_weight_statistics()
    final_path = save_checkpoint(
        trainer,
        final_loss,
        config.max_batches,
        total_time,
        config,
        weight_stats=final_weight_stats,
        is_final=True
    )

    print(f'\n💾 Final model saved: {final_path}')

    # 保存loss历史
    loss_file = Path(config.checkpoint_dir) / 'loss_history_confidence.txt'
    with open(loss_file, 'w') as f:
        f.write('# Confidence Network Loss History\n')
        f.write(f'# Temperature: {config.temperature}\n')
        f.write(f'# Confidence hidden dim: {config.confidence_hidden_dim}\n')
        f.write(f'# Batch_size: {config.batch_size}\n')
        f.write(f'# Total_speakers: {len(speaker_id_map)}\n')
        f.write('# Batch\tLoss\n')
        for i, loss in enumerate(loss_history, 1):
            f.write(f'{i}\t{loss:.6f}\n')
    print(f'📄 Loss history saved: {loss_file}')

    # 保存权重历史
    weight_file = Path(config.checkpoint_dir) / 'weight_history.json'
    with open(weight_file, 'w') as f:
        json.dump(weight_history, f, indent=2)
    print(f'📊 Weight history saved: {weight_file}')

    # 分析权重统计
    print('\n' + '=' * 80)
    print('📊 WEIGHT STATISTICS ANALYSIS')
    print('=' * 80)

    if final_weight_stats:
        print(f'\n🔍 Noisy Embedding Weights:')
        print(f'   Mean: {final_weight_stats["w_noisy"]["mean"]:.4f}')
        print(f'   Std:  {final_weight_stats["w_noisy"]["std"]:.4f}')
        print(f'   Range: [{final_weight_stats["w_noisy"]["min"]:.4f}, {final_weight_stats["w_noisy"]["max"]:.4f}]')

        print(f'\n🔍 Enhanced Embedding Weights:')
        print(f'   Mean: {final_weight_stats["w_enhanced"]["mean"]:.4f}')
        print(f'   Std:  {final_weight_stats["w_enhanced"]["std"]:.4f}')
        print(
            f'   Range: [{final_weight_stats["w_enhanced"]["min"]:.4f}, {final_weight_stats["w_enhanced"]["max"]:.4f}]')

        # 分析倾向
        avg_noisy = final_weight_stats["w_noisy"]["mean"]
        avg_enhanced = final_weight_stats["w_enhanced"]["mean"]

        print(f'\n💡 Weight Preference Analysis:')
        if avg_noisy > avg_enhanced:
            diff = (avg_noisy - avg_enhanced) * 100
            print(f'   ⚠️  Network prefers NOISY embeddings by {diff:.1f}%')
            print(f'   Possible reasons:')
            print(f'      - Enhanced embeddings may introduce artifacts')
            print(f'      - Network learned noisy is more reliable')
        elif avg_enhanced > avg_noisy:
            diff = (avg_enhanced - avg_noisy) * 100
            print(f'   ✅ Network prefers ENHANCED embeddings by {diff:.1f}%')
            print(f'   Good sign: enhancement is helping!')
        else:
            print(f'   ➖ Balanced weights (50-50)')

    print('=' * 80)

    # 训练稳定性分析
    print('\n' + '=' * 80)
    print('📈 TRAINING STABILITY ANALYSIS')
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

    if improvement > 0:
        print('✅ Model is converging')
    else:
        print('⚠️  Model may need more training')

    print('=' * 80)

    # 保存分析结果
    analysis_file = Path(config.checkpoint_dir) / 'training_analysis.txt'
    with open(analysis_file, 'w') as f:
        f.write('Confidence Network Training Analysis\n')
        f.write('=' * 80 + '\n')
        f.write(f'Loss Mean: {loss_mean:.4f}\n')
        f.write(f'Loss Std: {loss_std:.4f}\n')
        f.write(f'CV: {cv:.4f}\n')
        f.write(f'Improvement: {improvement:.2f}%\n')
        f.write(f'Total time: {total_time:.1f} min\n')
        f.write(f'Temperature: {config.temperature}\n')
        f.write(f'Confidence hidden dim: {config.confidence_hidden_dim}\n')
        f.write(f'Unique speakers: {len(speaker_id_map)}\n')
        if final_weight_stats:
            f.write(f'\nWeight Statistics:\n')
            f.write(f'  Noisy mean: {final_weight_stats["w_noisy"]["mean"]:.4f}\n')
            f.write(f'  Enhanced mean: {final_weight_stats["w_enhanced"]["mean"]:.4f}\n')

    print(f'\n📄 Analysis saved: {analysis_file}')
    print('\n🎯 All done! Next steps:')
    print('   1. Run: python evaluate_confidence.py --num_pairs 10000')
    print('   2. Run: python visualize_weights.py')
    print('   3. Compare with SupCon baseline')


if __name__ == '__main__':
    main()