# """
# 阶段1：SupCon损失完整实现
# ✅ 参考官方源码: https://github.com/HobbitLong/SupContrast
# ✅ 正确处理归一化
# ✅ 完整的训练流程
#
# 目标：验证训练稳定性和收敛性，与Triplet Loss对比
# """
#
# import torch
# import torch.nn.functional as F
# import os
# import subprocess
# import sys
# import time
# import random
# import numpy as np
# from pathlib import Path
# from torch.utils.data import DataLoader
#
# # 禁用DeepFilter日志
# os.environ['DF_DISABLE_LOGGING'] = '1'
# original_check_output = subprocess.check_output
# subprocess.check_output = lambda *args, **kwargs: b'unknown' if args and 'git' in str(
#     args[0]) else original_check_output(*args, **kwargs)
#
# sys.path.append('scripts')
# from models_supcon import SupConTrainer  # 使用SupCon模型
# from speaker_embedding import SpeakerEmbeddingExtractor
# from speech_enhancer import SpeechEnhancer
# from dataset import VoxCelebMusanDataset
#
#
# class Config:
#     """SupCon训练配置"""
#     # 数据路径
#     voxceleb_dir = 'data/voxceleb1'
#     musan_dir = 'data/musan'
#
#     # ===== 训练参数（与原论文保持一致以便对比）=====
#     batch_size = 32
#     learning_rate = 1e-3
#     snr_range = (-20, 0)
#
#     # ===== SupCon特有参数 =====
#     temperature = 0.07  # 🌡️ 温度参数（官方推荐0.07）
#
#     # 训练控制
#     max_batches = 200
#     checkpoint_interval = 25
#
#     # 输出目录
#     checkpoint_dir = 'checkpoints_supcon_200batch'
#
#     # 设备
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#
#     # 随机种子
#     seed = 42
#
#
# def set_seed(seed):
#     """设置所有随机种子确保可复现"""
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False
#
#
# def pad_collate(batch):
#     """
#     处理变长音频的collate函数
#
#     ✅ 修复：处理字符串类型的speaker_id
#     """
#     max_anchor = max([item['anchor_noisy'].shape[1] for item in batch])
#     max_positive = max([item['positive_noisy'].shape[1] for item in batch])
#     max_negative = max([item['negative_noisy'].shape[1] for item in batch])
#
#     padded_batch = {
#         'anchor_noisy': [],
#         'positive_noisy': [],
#         'negative_noisy': [],
#         'snr': [],
#         'speaker_id': []
#     }
#
#     for item in batch:
#         # Anchor padding
#         audio = item['anchor_noisy']
#         if audio.shape[1] < max_anchor:
#             padding = torch.zeros(1, max_anchor - audio.shape[1])
#             audio = torch.cat([audio, padding], dim=1)
#         padded_batch['anchor_noisy'].append(audio)
#
#         # Positive padding
#         audio = item['positive_noisy']
#         if audio.shape[1] < max_positive:
#             padding = torch.zeros(1, max_positive - audio.shape[1])
#             audio = torch.cat([audio, padding], dim=1)
#         padded_batch['positive_noisy'].append(audio)
#
#         # Negative padding
#         audio = item['negative_noisy']
#         if audio.shape[1] < max_negative:
#             padding = torch.zeros(1, max_negative - audio.shape[1])
#             audio = torch.cat([audio, padding], dim=1)
#         padded_batch['negative_noisy'].append(audio)
#
#         padded_batch['snr'].append(item['snr'])
#         padded_batch['speaker_id'].append(item['speaker_id'])
#
#     # Stack所有样本
#     for key in ['anchor_noisy', 'positive_noisy', 'negative_noisy']:
#         padded_batch[key] = torch.stack(padded_batch[key])
#
#     # ✅ 关键修复：speaker_id是字符串，需要保持原样或映射为整数
#     # 这里保持为列表，在extract函数中处理
#     # padded_batch['speaker_id'] 保持为字符串列表
#
#     return padded_batch
#
#
# def extract_and_normalize_embeddings(audio_batch, speaker_model, enhancer, device):
#     """
#     提取embeddings并确保L2归一化
#
#     🔑 关键改进：
#     1. 提取embedding后立即归一化（在speaker_embedding.py中已经做了）
#     2. 再次确认归一化（防止数值问题）
#     3. 组织成SupCon需要的格式
#
#     返回格式：
#     - noisy_embeddings: [batch_size*3, embedding_dim]
#     - enhanced_embeddings: [batch_size*3, embedding_dim]
#     - labels: [batch_size*3]
#     """
#     batch_size = len(audio_batch['anchor_noisy'])
#
#     all_noisy_embs = []
#     all_enhanced_embs = []
#     all_labels = []
#
#     for i in range(batch_size):
#         speaker_id = audio_batch['speaker_id'][i].item()
#
#         # === 处理Anchor（说话人A）===
#         anchor_noisy = audio_batch['anchor_noisy'][i]
#         anchor_enhanced = enhancer.enhance_audio(anchor_noisy, sr=16000)
#
#         # 提取embedding（speaker_model内部已归一化，但我们再确认一次）
#         emb_noisy = speaker_model.extract_embedding(audio_tensor=anchor_noisy)
#         emb_enhanced = speaker_model.extract_embedding(audio_tensor=anchor_enhanced)
#
#         # ✅ 关键：确保L2归一化（论文要求！）
#         emb_noisy = emb_noisy.squeeze()
#         emb_enhanced = emb_enhanced.squeeze()
#
#         # 归一化（即使speaker_model已经做了，再做一次以防万一）
#         emb_noisy = F.normalize(emb_noisy.unsqueeze(0), p=2, dim=1).squeeze(0)
#         emb_enhanced = F.normalize(emb_enhanced.unsqueeze(0), p=2, dim=1).squeeze(0)
#
#         all_noisy_embs.append(emb_noisy)
#         all_enhanced_embs.append(emb_enhanced)
#         all_labels.append(speaker_id)
#
#         # === 处理Positive（说话人A，不同utterance）===
#         pos_noisy = audio_batch['positive_noisy'][i]
#         pos_enhanced = enhancer.enhance_audio(pos_noisy, sr=16000)
#
#         emb_noisy = speaker_model.extract_embedding(audio_tensor=pos_noisy)
#         emb_enhanced = speaker_model.extract_embedding(audio_tensor=pos_enhanced)
#
#         emb_noisy = F.normalize(emb_noisy.squeeze().unsqueeze(0), p=2, dim=1).squeeze(0)
#         emb_enhanced = F.normalize(emb_enhanced.squeeze().unsqueeze(0), p=2, dim=1).squeeze(0)
#
#         all_noisy_embs.append(emb_noisy)
#         all_enhanced_embs.append(emb_enhanced)
#         all_labels.append(speaker_id)  # 与anchor相同
#
#         # === 处理Negative（说话人B，不同人）===
#         neg_noisy = audio_batch['negative_noisy'][i]
#         neg_enhanced = enhancer.enhance_audio(neg_noisy, sr=16000)
#
#         emb_noisy = speaker_model.extract_embedding(audio_tensor=neg_noisy)
#         emb_enhanced = speaker_model.extract_embedding(audio_tensor=neg_enhanced)
#
#         emb_noisy = F.normalize(emb_noisy.squeeze().unsqueeze(0), p=2, dim=1).squeeze(0)
#         emb_enhanced = F.normalize(emb_enhanced.squeeze().unsqueeze(0), p=2, dim=1).squeeze(0)
#
#         all_noisy_embs.append(emb_noisy)
#         all_enhanced_embs.append(emb_enhanced)
#         # ⚠️ 注意：这里需要negative的真实speaker_id
#         # 由于dataset返回的negative没有speaker_id，我们用一个不同的标记
#         # 实际中应该从dataset获取negative的speaker_id
#         all_labels.append(speaker_id + 100000)  # 临时方案：确保与anchor/positive不同
#
#     # 堆叠成batch
#     noisy_embeddings = torch.stack(all_noisy_embs)
#     enhanced_embeddings = torch.stack(all_enhanced_embs)
#     labels = torch.tensor(all_labels)
#
#     return noisy_embeddings, enhanced_embeddings, labels
#
#
# def save_checkpoint(trainer, loss, batch_idx, total_time, config, is_final=False):
#     """保存checkpoint"""
#     checkpoint_dir = Path(config.checkpoint_dir)
#     checkpoint_dir.mkdir(exist_ok=True)
#
#     if is_final:
#         filename = 'final_model_supcon.pth'
#     else:
#         filename = f'checkpoint_supcon_batch_{batch_idx}.pth'
#
#     checkpoint_path = checkpoint_dir / filename
#
#     torch.save({
#         'model_state_dict': trainer.mlp.state_dict(),
#         'optimizer_state_dict': trainer.optimizer.state_dict(),
#         'loss': loss,
#         'batch': batch_idx,
#         'training_minutes': total_time,
#         'config': {
#             'loss_type': 'SupCon',
#             'batch_size': config.batch_size,
#             'learning_rate': config.learning_rate,
#             'temperature': config.temperature,
#             'optimizer': 'AdamW',
#             'snr_range': config.snr_range,
#             'max_batches': config.max_batches,
#             'seed': config.seed
#         }
#     }, checkpoint_path)
#
#     return checkpoint_path
#
#
# def main():
#     config = Config()
#     set_seed(config.seed)
#
#     print('=' * 80)
#     print('🚀 SUPCON TRAINING - Supervised Contrastive Learning')
#     print('=' * 80)
#     print(f'📌 Loss Function: SupCon (replacing Triplet Loss)')
#     print(f'🎯 Goal: Verify training stability and convergence')
#     print(f'📚 Reference: Supervised Contrastive Learning (NeurIPS 2020)')
#     print(f'💻 Official Code: https://github.com/HobbitLong/SupContrast')
#     print('=' * 80)
#     print(f'Device: {config.device}')
#     print(f'Random seed: {config.seed}')
#     print(f'Batch size: {config.batch_size}')
#     print(f'Learning rate: {config.learning_rate}')
#     print(f'Temperature: {config.temperature} 🌡️')
#     print(f'Max batches: {config.max_batches}')
#     print('=' * 80)
#
#     # [1/3] 加载模型
#     print('\n[1/3] Loading models...')
#     speaker_model = SpeakerEmbeddingExtractor('ecapa')
#     enhancer = SpeechEnhancer()
#
#     # 创建SupCon训练器
#     trainer = SupConTrainer(
#         embedding_dim=192,
#         device=config.device,
#         temperature=config.temperature
#     )
#
#     trainer.optimizer = torch.optim.AdamW(
#         trainer.mlp.parameters(),
#         lr=config.learning_rate,
#         weight_decay=1e-4
#     )
#
#     print(f'✅ SupCon Trainer initialized')
#     print(f'   - Temperature: {config.temperature}')
#     print(f'   - MLP parameters: {sum(p.numel() for p in trainer.mlp.parameters()):,}')
#     print(f'   - Optimizer: AdamW (lr={config.learning_rate}, wd=1e-4)')
#
#     # [2/3] 加载数据集
#     print('\n[2/3] Loading dataset...')
#     train_dataset = VoxCelebMusanDataset(
#         config.voxceleb_dir,
#         config.musan_dir,
#         split='train',
#         snr_range=config.snr_range
#     )
#
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=config.batch_size,
#         shuffle=True,
#         num_workers=0,
#         collate_fn=pad_collate
#     )
#
#     print(f'✅ Training samples: {len(train_dataset)}')
#     print(f'   - SNR range: {config.snr_range} dB')
#
#     # [3/3] 训练
#     print('\n[3/3] Training with SupCon Loss...')
#     print('=' * 80)
#
#     start_time = time.time()
#     total_loss = 0.0
#     loss_history = []
#
#     trainer.mlp.train()
#
#     for batch_idx, audio_batch in enumerate(train_loader, 1):
#         batch_start = time.time()
#
#         # === 提取并归一化embeddings ===
#         noisy_embs, enhanced_embs, labels = extract_and_normalize_embeddings(
#             audio_batch, speaker_model, enhancer, config.device
#         )
#
#         # 移到GPU
#         noisy_embs = noisy_embs.to(config.device)
#         enhanced_embs = enhanced_embs.to(config.device)
#         labels = labels.to(config.device)
#
#         # === 通过MLP融合 ===
#         robust_embs = trainer.mlp(noisy_embs, enhanced_embs)
#
#         # ✅ 检查归一化（调试用）
#         if batch_idx == 1:
#             norms = torch.norm(robust_embs, p=2, dim=1)
#             print(f'\n🔍 Checking normalization (batch 1):')
#             print(f'   - Robust emb norms: min={norms.min():.4f}, max={norms.max():.4f}, mean={norms.mean():.4f}')
#             if not torch.allclose(norms, torch.ones_like(norms), atol=1e-4):
#                 print(f'   ⚠️ Warning: Features not perfectly normalized!')
#             else:
#                 print(f'   ✅ Features properly normalized\n')
#
#         # === 计算SupCon损失 ===
#         loss = trainer.criterion(robust_embs, labels)
#
#         # === 反向传播 ===
#         trainer.optimizer.zero_grad()
#         loss.backward()
#
#         # 梯度裁剪（提高稳定性）
#         torch.nn.utils.clip_grad_norm_(trainer.mlp.parameters(), max_norm=1.0)
#
#         trainer.optimizer.step()
#
#         # === 记录 ===
#         loss_val = loss.item()
#         total_loss += loss_val
#         loss_history.append(loss_val)
#
#         batch_time = time.time() - batch_start
#         elapsed_total = (time.time() - start_time) / 60
#
#         # 打印进度
#         if batch_idx % 5 == 0 or batch_idx == 1:
#             avg_loss = total_loss / batch_idx
#             print(f'Batch {batch_idx:3d}/{config.max_batches} | '
#                   f'Loss: {loss_val:.4f} (avg: {avg_loss:.4f}) | '
#                   f'Time: {batch_time:.1f}s | '
#                   f'Total: {elapsed_total:.1f}min')
#
#         # 保存checkpoint
#         if batch_idx % config.checkpoint_interval == 0:
#             checkpoint_path = save_checkpoint(
#                 trainer,
#                 total_loss / batch_idx,
#                 batch_idx,
#                 elapsed_total,
#                 config,
#                 is_final=False
#             )
#             print(f'  💾 Checkpoint saved: {checkpoint_path.name}')
#
#         if batch_idx >= config.max_batches:
#             break
#
#     # === 训练完成 ===
#     total_time = (time.time() - start_time) / 60
#     final_loss = total_loss / config.max_batches
#
#     print('\n' + '=' * 80)
#     print('✅ TRAINING COMPLETED')
#     print('=' * 80)
#     print(f'📊 Final loss: {final_loss:.4f}')
#     print(f'⏱️  Total time: {total_time:.1f} minutes')
#     print(f'📈 Avg loss: {sum(loss_history) / len(loss_history):.4f}')
#     print(f'📉 Min loss: {min(loss_history):.4f}')
#     print(f'📈 Max loss: {max(loss_history):.4f}')
#     print('=' * 80)
#
#     # 保存最终模型
#     final_path = save_checkpoint(
#         trainer,
#         final_loss,
#         config.max_batches,
#         total_time,
#         config,
#         is_final=True
#     )
#
#     print(f'\n💾 Final model saved: {final_path}')
#
#     # 保存loss历史
#     loss_file = Path(config.checkpoint_dir) / 'loss_history_supcon.txt'
#     with open(loss_file, 'w') as f:
#         f.write('# SupCon Loss History\n')
#         f.write(f'# Temperature: {config.temperature}\n')
#         f.write(f'# Batch_size: {config.batch_size}\n')
#         f.write(f'# Reference: https://github.com/HobbitLong/SupContrast\n')
#         f.write('# Batch\tLoss\n')
#         for i, loss in enumerate(loss_history, 1):
#             f.write(f'{i}\t{loss:.6f}\n')
#     print(f'📄 Loss history saved: {loss_file}')
#
#     # 分析训练稳定性
#     print('\n' + '=' * 80)
#     print('📊 TRAINING STABILITY ANALYSIS')
#     print('=' * 80)
#
#     loss_std = np.std(loss_history)
#     loss_mean = np.mean(loss_history)
#     cv = loss_std / loss_mean
#
#     print(f'Loss Mean: {loss_mean:.4f}')
#     print(f'Loss Std: {loss_std:.4f}')
#     print(f'Coefficient of Variation: {cv:.4f}')
#
#     last_10_avg = np.mean(loss_history[-10:])
#     first_10_avg = np.mean(loss_history[:10])
#     improvement = (first_10_avg - last_10_avg) / first_10_avg * 100
#
#     print(f'\nFirst 10 batches avg loss: {first_10_avg:.4f}')
#     print(f'Last 10 batches avg loss: {last_10_avg:.4f}')
#     print(f'Improvement: {improvement:.2f}%')
#
#     if improvement > 0:
#         print('✅ Model is converging')
#     else:
#         print('⚠️ Model may need more training')
#
#     print('=' * 80)
#
#     # 保存分析结果
#     analysis_file = Path(config.checkpoint_dir) / 'training_analysis.txt'
#     with open(analysis_file, 'w') as f:
#         f.write('SupCon Training Analysis\n')
#         f.write('=' * 80 + '\n')
#         f.write(f'Loss Mean: {loss_mean:.4f}\n')
#         f.write(f'Loss Std: {loss_std:.4f}\n')
#         f.write(f'CV: {cv:.4f}\n')
#         f.write(f'Improvement: {improvement:.2f}%\n')
#         f.write(f'Total time: {total_time:.1f} min\n')
#         f.write(f'Temperature: {config.temperature}\n')
#         f.write('Reference: https://github.com/HobbitLong/SupContrast\n')
#
#     print(f'\n📄 Analysis saved: {analysis_file}')
#     print('\n🎉 All done! Next: Compare with Triplet Loss using compare_training.py')
#
#
# if __name__ == '__main__':
#     main()
"""
阶段1：SupCon损失完整实现（修复speaker_id字符串问题）
✅ 参考官方源码: https://github.com/HobbitLong/SupContrast
✅ 正确处理归一化
✅ 修复speaker_id字符串->整数映射

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
    """SupCon训练配置"""
    # 数据路径
    voxceleb_dir = 'data/voxceleb1'
    musan_dir = 'data/musan'

    # 训练参数
    batch_size = 32
    learning_rate = 1e-3
    snr_range = (-20, 0)

    # SupCon特有参数
    temperature = 0.07

    # 训练控制
    max_batches = 200
    checkpoint_interval = 25

    # 输出目录
    checkpoint_dir = 'checkpoints_supcon_200batch'

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
    处理变长音频的collate函数

    ✅ 修复点1：不转换speaker_id为tensor，保持为字符串列表
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

    # ✅ 关键修复：speaker_id保持为字符串列表，不转换为tensor
    # 原来的代码：padded_batch['speaker_id'] = torch.tensor(padded_batch['speaker_id'])
    # 这会报错：ValueError: too many dimensions 'str'

    return padded_batch


def extract_and_normalize_embeddings(audio_batch, speaker_model, enhancer, device, speaker_id_map):
    """
    提取embeddings并确保L2归一化

    ✅ 修复点2：将字符串speaker_id映射为整数

    参数:
        speaker_id_map: dict, 字符串speaker_id -> 整数映射

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
        # ✅ 修复点3：直接获取字符串，不调用.item()
        # 原来：speaker_id = audio_batch['speaker_id'][i].item()  ❌ 会报错
        # 现在：speaker_id_str = audio_batch['speaker_id'][i]     ✅ 正确
        speaker_id_str = audio_batch['speaker_id'][i]

        # ✅ 修复点4：将字符串映射为整数
        if speaker_id_str not in speaker_id_map:
            speaker_id_map[speaker_id_str] = len(speaker_id_map)
        speaker_id_int = speaker_id_map[speaker_id_str]

        # === 处理Anchor（说话人A）===
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

        # === 处理Positive（说话人A，不同utterance）===
        pos_noisy = audio_batch['positive_noisy'][i]
        pos_enhanced = enhancer.enhance_audio(pos_noisy, sr=16000)

        emb_noisy = speaker_model.extract_embedding(audio_tensor=pos_noisy)
        emb_enhanced = speaker_model.extract_embedding(audio_tensor=pos_enhanced)

        emb_noisy = F.normalize(emb_noisy.squeeze().unsqueeze(0), p=2, dim=1).squeeze(0)
        emb_enhanced = F.normalize(emb_enhanced.squeeze().unsqueeze(0), p=2, dim=1).squeeze(0)

        all_noisy_embs.append(emb_noisy)
        all_enhanced_embs.append(emb_enhanced)
        all_labels.append(speaker_id_int)  # 与anchor相同

        # === 处理Negative（说话人B，不同人）===
        neg_noisy = audio_batch['negative_noisy'][i]
        neg_enhanced = enhancer.enhance_audio(neg_noisy, sr=16000)

        emb_noisy = speaker_model.extract_embedding(audio_tensor=neg_noisy)
        emb_enhanced = speaker_model.extract_embedding(audio_tensor=neg_enhanced)

        emb_noisy = F.normalize(emb_noisy.squeeze().unsqueeze(0), p=2, dim=1).squeeze(0)
        emb_enhanced = F.normalize(emb_enhanced.squeeze().unsqueeze(0), p=2, dim=1).squeeze(0)

        all_noisy_embs.append(emb_noisy)
        all_enhanced_embs.append(emb_enhanced)
        # Negative的标签应该不同，用大偏移量确保不冲突
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
        filename = 'final_model_supcon.pth'
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
    print('🚀 SUPCON TRAINING - Supervised Contrastive Learning')
    print('=' * 80)
    print(f'📌 Loss Function: SupCon (replacing Triplet Loss)')
    print(f'🎯 Goal: Verify training stability and convergence')
    print(f'📚 Reference: Supervised Contrastive Learning (NeurIPS 2020)')
    print(f'💻 Official Code: https://github.com/HobbitLong/SupContrast')
    print('=' * 80)
    print(f'Device: {config.device}')
    print(f'Random seed: {config.seed}')
    print(f'Batch size: {config.batch_size}')
    print(f'Learning rate: {config.learning_rate}')
    print(f'Temperature: {config.temperature} 🌡️')
    print(f'Max batches: {config.max_batches}')
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
    print('\n[3/3] Training with SupCon Loss...')
    print('=' * 80)

    start_time = time.time()
    total_loss = 0.0
    loss_history = []

    # ✅ 修复点5：创建speaker_id映射字典
    speaker_id_map = {}

    trainer.mlp.train()

    for batch_idx, audio_batch in enumerate(train_loader, 1):
        batch_start = time.time()

        # ✅ 修复点6：传递speaker_id_map
        noisy_embs, enhanced_embs, labels = extract_and_normalize_embeddings(
            audio_batch, speaker_model, enhancer, config.device, speaker_id_map
        )

        # 移到GPU
        noisy_embs = noisy_embs.to(config.device)
        enhanced_embs = enhanced_embs.to(config.device)
        labels = labels.to(config.device)

        # 通过MLP融合
        robust_embs = trainer.mlp(noisy_embs, enhanced_embs)

        # 检查归一化（第一个batch）
        if batch_idx == 1:
            norms = torch.norm(robust_embs, p=2, dim=1)
            print(f'\n🔍 Checking normalization (batch 1):')
            print(f'   - Robust emb norms: min={norms.min():.4f}, max={norms.max():.4f}, mean={norms.mean():.4f}')
            if not torch.allclose(norms, torch.ones_like(norms), atol=1e-4):
                print(f'   ⚠️ Warning: Features not perfectly normalized!')
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
    loss_file = Path(config.checkpoint_dir) / 'loss_history_supcon.txt'
    with open(loss_file, 'w') as f:
        f.write('# SupCon Loss History\n')
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

    if improvement > 0:
        print('✅ Model is converging')
    else:
        print('⚠️ Model may need more training')

    print('=' * 80)

    # 保存分析结果
    analysis_file = Path(config.checkpoint_dir) / 'training_analysis.txt'
    with open(analysis_file, 'w') as f:
        f.write('SupCon Training Analysis\n')
        f.write('=' * 80 + '\n')
        f.write(f'Loss Mean: {loss_mean:.4f}\n')
        f.write(f'Loss Std: {loss_std:.4f}\n')
        f.write(f'CV: {cv:.4f}\n')
        f.write(f'Improvement: {improvement:.2f}%\n')
        f.write(f'Total time: {total_time:.1f} min\n')
        f.write(f'Temperature: {config.temperature}\n')
        f.write(f'Unique speakers: {len(speaker_id_map)}\n')
        f.write('Reference: https://github.com/HobbitLong/SupContrast\n')

    print(f'\n📄 Analysis saved: {analysis_file}')
    print('\n🎉 All done! Next: Compare with Triplet Loss using compare_training.py')


if __name__ == '__main__':
    main()