# # # # """
# # # # 阶段2：安全的Speech性能优化（两阶段渐进训练）
# # # #
# # # # 🛡️ 安全机制：
# # # # 1. 阶段2.1：冻结MLP主干，只训练Confidence Head（50 batches）
# # # # 2. 阶段2.2：解冻全部，极小lr微调（150 batches）
# # # # 3. Speech适度提升至40%（不是60%）
# # # # 4. 保持完整SNR范围(-20, 0)
# # # # 5. 每25 batch保存checkpoint，可随时回退
# # # # 6. 监控权重变化，及时发现异常
# # # #
# # # # 预期效果：
# # # # - Noise/Music：保持或轻微下降1-2%
# # # # - Speech @ 0dB：保持
# # # # - Speech @ -5~-20dB：提升3-5%
# # # # - 总体胜率：从11/15提升到12-13/15
# # # #
# # # # 用法：
# # # #     python train_confidence_stage2_safe.py
# # # # """
# # # #
# # # # import torch
# # # # import torch.nn.functional as F
# # # # import os
# # # # import subprocess
# # # # import sys
# # # # import time
# # # # import random
# # # # import numpy as np
# # # # from pathlib import Path
# # # # from torch.utils.data import DataLoader
# # # # import json
# # # #
# # # # # 禁用DeepFilter日志
# # # # os.environ['DF_DISABLE_LOGGING'] = '1'
# # # # original_check_output = subprocess.check_output
# # # # subprocess.check_output = lambda *args, **kwargs: b'unknown' if args and 'git' in str(
# # # #     args[0]) else original_check_output(*args, **kwargs)
# # # #
# # # # sys.path.append('scripts')
# # # # from models_confidence import ConfidenceSupConTrainer
# # # # from speaker_embedding import SpeakerEmbeddingExtractor
# # # # from speech_enhancer import SpeechEnhancer
# # # # from dataset import VoxCelebMusanDataset
# # # #
# # # #
# # # # class ConfigStage2:
# # # #     """阶段2训练配置 - 安全优化"""
# # # #     # 数据路径
# # # #     voxceleb_dir = 'data/voxceleb1'
# # # #     musan_dir = 'data/musan'
# # # #
# # # #     # ===== 阶段1 checkpoint =====
# # # #     stage1_checkpoint = 'checkpoints_confidence_400batch/final_model_confidence.pth'
# # # #
# # # #     # ===== 两个子阶段的参数 =====
# # # #
# # # #     # 子阶段2.1：冻结MLP主干
# # # #     stage_2_1_config = {
# # # #         'name': 'freeze_mlp',
# # # #         'batches': 50,
# # # #         'learning_rate': 1e-3,
# # # #         'freeze_mlp': True,  # ⭐ 关键：冻结MLP
# # # #         'speech_boost': 2.0,  # Speech占40%
# # # #         'description': 'Train only Confidence Head with frozen MLP'
# # # #     }
# # # #
# # # #     # 子阶段2.2：解冻全部，精细调整
# # # #     stage_2_2_config = {
# # # #         'name': 'finetune_all',
# # # #         'batches': 150,
# # # #         'learning_rate': 1e-4,  # ⭐ 极小学习率
# # # #         'freeze_mlp': False,
# # # #         'speech_boost': 1.5,  # Speech占37.5%
# # # #         'description': 'Fine-tune all parameters with very small lr'
# # # #     }
# # # #
# # # #     # ===== 通用参数 =====
# # # #     batch_size = 32
# # # #     snr_range = (-20, 0)  # ⭐ 保持完整范围
# # # #     temperature = 0.07
# # # #     confidence_hidden_dim = 64
# # # #
# # # #     # 输出目录
# # # #     checkpoint_dir = 'checkpoints_confidence_stage2'
# # # #
# # # #     # 控制参数
# # # #     checkpoint_interval = 25
# # # #     log_weights_interval = 5
# # # #
# # # #     # 设备
# # # #     device = 'cuda' if torch.cuda.is_available() else 'cpu'
# # # #     seed = 42
# # # #
# # # #
# # # # class SpeechBoostedDataset(VoxCelebMusanDataset):
# # # #     """适度增强Speech噪声比例的数据集"""
# # # #
# # # #     def __init__(self, *args, speech_boost_factor=1.5, **kwargs):
# # # #         super().__init__(*args, **kwargs)
# # # #         self.speech_boost_factor = speech_boost_factor
# # # #
# # # #         # 统计原始分布
# # # #         original_counts = {
# # # #             'noise': len(self.musan_files['noise']),
# # # #             'music': len(self.musan_files['music']),
# # # #             'speech': len(self.musan_files['speech'])
# # # #         }
# # # #
# # # #         print(f'📊 Original MUSAN distribution:')
# # # #         for noise_type, count in original_counts.items():
# # # #             print(f'   {noise_type}: {count} files')
# # # #
# # # #         # 适度扩充Speech文件列表
# # # #         if speech_boost_factor > 1.0:
# # # #             original_speech = self.musan_files['speech'].copy()
# # # #             repeats = int(speech_boost_factor) - 1
# # # #             fractional = speech_boost_factor - int(speech_boost_factor)
# # # #
# # # #             for _ in range(repeats):
# # # #                 self.musan_files['speech'].extend(original_speech)
# # # #
# # # #             if fractional > 0:
# # # #                 n_extra = int(len(original_speech) * fractional)
# # # #                 self.musan_files['speech'].extend(
# # # #                     random.sample(original_speech, n_extra)
# # # #                 )
# # # #
# # # #         new_counts = {
# # # #             'noise': len(self.musan_files['noise']),
# # # #             'music': len(self.musan_files['music']),
# # # #             'speech': len(self.musan_files['speech'])
# # # #         }
# # # #
# # # #         total = sum(new_counts.values())
# # # #         print(f'\n✅ Adjusted MUSAN distribution (boost={speech_boost_factor:.1f}x):')
# # # #         for noise_type, count in new_counts.items():
# # # #             pct = count / total * 100
# # # #             print(f'   {noise_type}: {count} files ({pct:.1f}%)')
# # # #
# # # #
# # # # def set_seed(seed):
# # # #     """设置随机种子"""
# # # #     random.seed(seed)
# # # #     np.random.seed(seed)
# # # #     torch.manual_seed(seed)
# # # #     if torch.cuda.is_available():
# # # #         torch.cuda.manual_seed_all(seed)
# # # #         torch.backends.cudnn.deterministic = True
# # # #         torch.backends.cudnn.benchmark = False
# # # #
# # # #
# # # # def pad_collate(batch):
# # # #     """处理变长音频的collate函数"""
# # # #     max_anchor = max([item['anchor_noisy'].shape[1] for item in batch])
# # # #     max_positive = max([item['positive_noisy'].shape[1] for item in batch])
# # # #     max_negative = max([item['negative_noisy'].shape[1] for item in batch])
# # # #
# # # #     padded_batch = {
# # # #         'anchor_noisy': [],
# # # #         'positive_noisy': [],
# # # #         'negative_noisy': [],
# # # #         'snr': [],
# # # #         'speaker_id': []
# # # #     }
# # # #
# # # #     for item in batch:
# # # #         # Anchor padding
# # # #         audio = item['anchor_noisy']
# # # #         if audio.shape[1] < max_anchor:
# # # #             padding = torch.zeros(1, max_anchor - audio.shape[1])
# # # #             audio = torch.cat([audio, padding], dim=1)
# # # #         padded_batch['anchor_noisy'].append(audio)
# # # #
# # # #         # Positive padding
# # # #         audio = item['positive_noisy']
# # # #         if audio.shape[1] < max_positive:
# # # #             padding = torch.zeros(1, max_positive - audio.shape[1])
# # # #             audio = torch.cat([audio, padding], dim=1)
# # # #         padded_batch['positive_noisy'].append(audio)
# # # #
# # # #         # Negative padding
# # # #         audio = item['negative_noisy']
# # # #         if audio.shape[1] < max_negative:
# # # #             padding = torch.zeros(1, max_negative - audio.shape[1])
# # # #             audio = torch.cat([audio, padding], dim=1)
# # # #         padded_batch['negative_noisy'].append(audio)
# # # #
# # # #         padded_batch['snr'].append(item['snr'])
# # # #         padded_batch['speaker_id'].append(item['speaker_id'])
# # # #
# # # #     # Stack音频数据
# # # #     for key in ['anchor_noisy', 'positive_noisy', 'negative_noisy']:
# # # #         padded_batch[key] = torch.stack(padded_batch[key])
# # # #
# # # #     return padded_batch
# # # #
# # # #
# # # # def extract_and_normalize_embeddings(audio_batch, speaker_model, enhancer, device, speaker_id_map):
# # # #     """提取并归一化embeddings"""
# # # #     batch_size = len(audio_batch['anchor_noisy'])
# # # #
# # # #     all_noisy_embs = []
# # # #     all_enhanced_embs = []
# # # #     all_labels = []
# # # #
# # # #     for i in range(batch_size):
# # # #         speaker_id_str = audio_batch['speaker_id'][i]
# # # #
# # # #         if speaker_id_str not in speaker_id_map:
# # # #             speaker_id_map[speaker_id_str] = len(speaker_id_map)
# # # #         speaker_id_int = speaker_id_map[speaker_id_str]
# # # #
# # # #         # === Anchor ===
# # # #         anchor_noisy = audio_batch['anchor_noisy'][i]
# # # #         anchor_enhanced = enhancer.enhance_audio(anchor_noisy, sr=16000)
# # # #
# # # #         emb_noisy = speaker_model.extract_embedding(audio_tensor=anchor_noisy)
# # # #         emb_enhanced = speaker_model.extract_embedding(audio_tensor=anchor_enhanced)
# # # #
# # # #         emb_noisy = F.normalize(emb_noisy.squeeze().unsqueeze(0), p=2, dim=1).squeeze(0)
# # # #         emb_enhanced = F.normalize(emb_enhanced.squeeze().unsqueeze(0), p=2, dim=1).squeeze(0)
# # # #
# # # #         all_noisy_embs.append(emb_noisy)
# # # #         all_enhanced_embs.append(emb_enhanced)
# # # #         all_labels.append(speaker_id_int)
# # # #
# # # #         # === Positive ===
# # # #         pos_noisy = audio_batch['positive_noisy'][i]
# # # #         pos_enhanced = enhancer.enhance_audio(pos_noisy, sr=16000)
# # # #
# # # #         emb_noisy = speaker_model.extract_embedding(audio_tensor=pos_noisy)
# # # #         emb_enhanced = speaker_model.extract_embedding(audio_tensor=pos_enhanced)
# # # #
# # # #         emb_noisy = F.normalize(emb_noisy.squeeze().unsqueeze(0), p=2, dim=1).squeeze(0)
# # # #         emb_enhanced = F.normalize(emb_enhanced.squeeze().unsqueeze(0), p=2, dim=1).squeeze(0)
# # # #
# # # #         all_noisy_embs.append(emb_noisy)
# # # #         all_enhanced_embs.append(emb_enhanced)
# # # #         all_labels.append(speaker_id_int)
# # # #
# # # #         # === Negative ===
# # # #         neg_noisy = audio_batch['negative_noisy'][i]
# # # #         neg_enhanced = enhancer.enhance_audio(neg_noisy, sr=16000)
# # # #
# # # #         emb_noisy = speaker_model.extract_embedding(audio_tensor=neg_noisy)
# # # #         emb_enhanced = speaker_model.extract_embedding(audio_tensor=neg_enhanced)
# # # #
# # # #         emb_noisy = F.normalize(emb_noisy.squeeze().unsqueeze(0), p=2, dim=1).squeeze(0)
# # # #         emb_enhanced = F.normalize(emb_enhanced.squeeze().unsqueeze(0), p=2, dim=1).squeeze(0)
# # # #
# # # #         all_noisy_embs.append(emb_noisy)
# # # #         all_enhanced_embs.append(emb_enhanced)
# # # #         all_labels.append(speaker_id_int + 100000)
# # # #
# # # #     noisy_embeddings = torch.stack(all_noisy_embs)
# # # #     enhanced_embeddings = torch.stack(all_enhanced_embs)
# # # #     labels = torch.tensor(all_labels, dtype=torch.long)
# # # #
# # # #     return noisy_embeddings, enhanced_embs, labels
# # # #
# # # #
# # # # def freeze_mlp_backbone(trainer):
# # # #     """冻结MLP主干（fc1, fc2, fc3）"""
# # # #     for param in trainer.mlp.fc1.parameters():
# # # #         param.requires_grad = False
# # # #     for param in trainer.mlp.fc2.parameters():
# # # #         param.requires_grad = False
# # # #     for param in trainer.mlp.fc3.parameters():
# # # #         param.requires_grad = False
# # # #
# # # #     print('🧊 Frozen MLP backbone (fc1, fc2, fc3)')
# # # #     print('   - Only training Confidence Head')
# # # #
# # # #
# # # # def unfreeze_all(trainer):
# # # #     """解冻所有参数"""
# # # #     for param in trainer.mlp.parameters():
# # # #         param.requires_grad = True
# # # #
# # # #     print('🔓 Unfrozen all parameters')
# # # #     print('   - Training entire network')
# # # #
# # # #
# # # # def save_checkpoint(trainer, loss, batch_idx, total_time, config, stage_name, weight_stats=None, is_final=False):
# # # #     """保存checkpoint"""
# # # #     checkpoint_dir = Path(config.checkpoint_dir)
# # # #     checkpoint_dir.mkdir(exist_ok=True)
# # # #
# # # #     if is_final:
# # # #         filename = f'final_model_stage2_{stage_name}.pth'
# # # #     else:
# # # #         filename = f'checkpoint_stage2_{stage_name}_batch_{batch_idx}.pth'
# # # #
# # # #     checkpoint_path = checkpoint_dir / filename
# # # #
# # # #     checkpoint_data = {
# # # #         'model_state_dict': trainer.mlp.state_dict(),
# # # #         'optimizer_state_dict': trainer.optimizer.state_dict(),
# # # #         'loss': loss,
# # # #         'batch': batch_idx,
# # # #         'training_minutes': total_time,
# # # #         'stage': f'stage2_{stage_name}',
# # # #         'config': {
# # # #             'model_type': 'Confidence Stage 2',
# # # #             'batch_size': config.batch_size,
# # # #             'temperature': config.temperature,
# # # #             'confidence_hidden_dim': config.confidence_hidden_dim,
# # # #             'snr_range': config.snr_range,
# # # #             'seed': config.seed,
# # # #             'stage1_checkpoint': config.stage1_checkpoint
# # # #         }
# # # #     }
# # # #
# # # #     if weight_stats:
# # # #         checkpoint_data['weight_stats'] = weight_stats
# # # #
# # # #     torch.save(checkpoint_data, checkpoint_path)
# # # #
# # # #     return checkpoint_path
# # # #
# # # #
# # # # def train_substage(trainer, speaker_model, enhancer, train_loader,
# # # #                    config, substage_config, speaker_id_map, stage_start_time):
# # # #     """训练一个子阶段"""
# # # #
# # # #     stage_name = substage_config['name']
# # # #     max_batches = substage_config['batches']
# # # #     lr = substage_config['learning_rate']
# # # #
# # # #     print('\n' + '=' * 80)
# # # #     print(f'🎯 SUB-STAGE: {substage_config["description"]}')
# # # #     print('=' * 80)
# # # #     print(f'Batches: {max_batches}')
# # # #     print(f'Learning rate: {lr}')
# # # #     print(f'Freeze MLP: {substage_config["freeze_mlp"]}')
# # # #     print(f'Speech boost: {substage_config["speech_boost"]}x')
# # # #     print('=' * 80)
# # # #
# # # #     # 设置冻结状态
# # # #     if substage_config['freeze_mlp']:
# # # #         freeze_mlp_backbone(trainer)
# # # #     else:
# # # #         unfreeze_all(trainer)
# # # #
# # # #     # 更新优化器（使用新的学习率）
# # # #     trainer.optimizer = torch.optim.AdamW(
# # # #         filter(lambda p: p.requires_grad, trainer.mlp.parameters()),
# # # #         lr=lr,
# # # #         weight_decay=1e-4
# # # #     )
# # # #
# # # #     print(f'✅ Optimizer updated with lr={lr}')
# # # #
# # # #     # 训练循环
# # # #     total_loss = 0.0
# # # #     loss_history = []
# # # #     weight_history = []
# # # #
# # # #     trainer.mlp.train()
# # # #
# # # #     batch_count = 0
# # # #     data_iterator = iter(train_loader)
# # # #
# # # #     while batch_count < max_batches:
# # # #         try:
# # # #             audio_batch = next(data_iterator)
# # # #         except StopIteration:
# # # #             data_iterator = iter(train_loader)
# # # #             audio_batch = next(data_iterator)
# # # #
# # # #         batch_count += 1
# # # #         batch_start = time.time()
# # # #
# # # #         # 提取embeddings
# # # #         noisy_embs, enhanced_embs, labels = extract_and_normalize_embeddings(
# # # #             audio_batch, speaker_model, enhancer, config.device, speaker_id_map
# # # #         )
# # # #
# # # #         noisy_embs = noisy_embs.to(config.device)
# # # #         enhanced_embs = enhanced_embs.to(config.device)
# # # #         labels = labels.to(config.device)
# # # #
# # # #         # 训练步骤
# # # #         log_weights = (batch_count % config.log_weights_interval == 0) or (batch_count == 1)
# # # #         loss, stats = trainer.train_step(
# # # #             noisy_embs, enhanced_embs, labels, log_weights=log_weights
# # # #         )
# # # #
# # # #         # 记录
# # # #         loss_history.append(loss)
# # # #         total_loss += loss
# # # #
# # # #         if log_weights:
# # # #             weight_history.append({
# # # #                 'batch': batch_count,
# # # #                 'w_noisy': stats['w_noisy_mean'],
# # # #                 'w_enhanced': stats['w_enhanced_mean'],
# # # #                 'w_noisy_std': stats['w_noisy_std'],
# # # #                 'w_enhanced_std': stats['w_enhanced_std']
# # # #             })
# # # #
# # # #         batch_time = time.time() - batch_start
# # # #         elapsed_total = (time.time() - stage_start_time) / 60
# # # #
# # # #         # 打印进度
# # # #         if batch_count % 5 == 0 or batch_count == 1:
# # # #             avg_loss = total_loss / batch_count
# # # #             msg = f'[{stage_name}] Batch {batch_count:3d}/{max_batches} | Loss: {loss:.4f} (avg: {avg_loss:.4f})'
# # # #
# # # #             if log_weights:
# # # #                 msg += f' | w_n={stats["w_noisy_mean"]:.3f} w_e={stats["w_enhanced_mean"]:.3f}'
# # # #
# # # #             msg += f' | Time: {batch_time:.1f}s | Total: {elapsed_total:.1f}min'
# # # #             print(msg)
# # # #
# # # #         # 保存checkpoint
# # # #         if batch_count % config.checkpoint_interval == 0:
# # # #             current_weight_stats = trainer.get_weight_statistics()
# # # #             checkpoint_path = save_checkpoint(
# # # #                 trainer,
# # # #                 total_loss / batch_count,
# # # #                 batch_count,
# # # #                 elapsed_total,
# # # #                 config,
# # # #                 stage_name,
# # # #                 weight_stats=current_weight_stats,
# # # #                 is_final=False
# # # #             )
# # # #             print(f'  💾 Checkpoint saved: {checkpoint_path.name}')
# # # #
# # # #     # 子阶段完成
# # # #     avg_loss = total_loss / max_batches
# # # #
# # # #     print(f'\n✅ Sub-stage "{stage_name}" completed')
# # # #     print(f'   Average loss: {avg_loss:.4f}')
# # # #     print(f'   Batches trained: {max_batches}')
# # # #
# # # #     # 保存子阶段最终模型
# # # #     final_weight_stats = trainer.get_weight_statistics()
# # # #     final_path = save_checkpoint(
# # # #         trainer,
# # # #         avg_loss,
# # # #         max_batches,
# # # #         (time.time() - stage_start_time) / 60,
# # # #         config,
# # # #         stage_name,
# # # #         weight_stats=final_weight_stats,
# # # #         is_final=True
# # # #     )
# # # #
# # # #     print(f'💾 Sub-stage final model saved: {final_path}')
# # # #
# # # #     return loss_history, weight_history
# # # #
# # # #
# # # # def main():
# # # #     config = ConfigStage2()
# # # #     set_seed(config.seed)
# # # #
# # # #     print('=' * 80)
# # # #     print('🛡️ CONFIDENCE STAGE 2: SAFE SPEECH OPTIMIZATION')
# # # #     print('=' * 80)
# # # #     print('Strategy: Two-phase progressive training')
# # # #     print(f'Base model: {config.stage1_checkpoint}')
# # # #     print('=' * 80)
# # # #     print('Phase 2.1: Freeze MLP + Train Confidence Head (50 batches)')
# # # #     print('  - Speech boost: 2.0x (40%)')
# # # #     print('  - Learning rate: 1e-3')
# # # #     print('  - Goal: Adapt confidence network to Speech')
# # # #     print('')
# # # #     print('Phase 2.2: Unfreeze All + Fine-tune (150 batches)')
# # # #     print('  - Speech boost: 1.5x (37.5%)')
# # # #     print('  - Learning rate: 1e-4 (very small!)')
# # # #     print('  - Goal: Gentle adjustment without forgetting')
# # # #     print('=' * 80)
# # # #     print(f'Device: {config.device}')
# # # #     print(f'Random seed: {config.seed}')
# # # #     print(f'SNR range: {config.snr_range} (full coverage)')
# # # #     print(f'Total expected time: ~2.5-3 hours')
# # # #     print('=' * 80)
# # # #
# # # #     # [1/4] 加载模型
# # # #     print('\n[1/4] Loading models and Stage 1 checkpoint...')
# # # #     speaker_model = SpeakerEmbeddingExtractor('ecapa')
# # # #     enhancer = SpeechEnhancer()
# # # #
# # # #     # 加载Stage 1模型
# # # #     stage1_checkpoint = torch.load(config.stage1_checkpoint, map_location=config.device, weights_only=False)
# # # #
# # # #     trainer = ConfidenceSupConTrainer(
# # # #         embedding_dim=192,
# # # #         device=config.device,
# # # #         temperature=config.temperature,
# # # #         confidence_hidden_dim=config.confidence_hidden_dim
# # # #     )
# # # #
# # # #     trainer.mlp.load_state_dict(stage1_checkpoint['model_state_dict'])
# # # #
# # # #     print(f'✅ Loaded Stage 1 checkpoint')
# # # #     print(f'   - Stage 1 loss: {stage1_checkpoint["loss"]:.4f}')
# # # #     print(f'   - Stage 1 batches: {stage1_checkpoint["batch"]}')
# # # #
# # # #     # [2/4] 准备数据集
# # # #     print('\n[2/4] Preparing datasets for two sub-stages...')
# # # #
# # # #     speaker_id_map = {}
# # # #
# # # #     # 阶段2.1数据集（Speech占40%）
# # # #     print('\n📦 Dataset for Sub-stage 2.1:')
# # # #     dataset_2_1 = SpeechBoostedDataset(
# # # #         config.voxceleb_dir,
# # # #         config.musan_dir,
# # # #         split='train',
# # # #         snr_range=config.snr_range,
# # # #         speech_boost_factor=config.stage_2_1_config['speech_boost']
# # # #     )
# # # #
# # # #     loader_2_1 = DataLoader(
# # # #         dataset_2_1,
# # # #         batch_size=config.batch_size,
# # # #         shuffle=True,
# # # #         num_workers=0,
# # # #         collate_fn=pad_collate
# # # #     )
# # # #
# # # #     # 阶段2.2数据集（Speech占37.5%）
# # # #     print('\n📦 Dataset for Sub-stage 2.2:')
# # # #     dataset_2_2 = SpeechBoostedDataset(
# # # #         config.voxceleb_dir,
# # # #         config.musan_dir,
# # # #         split='train',
# # # #         snr_range=config.snr_range,
# # # #         speech_boost_factor=config.stage_2_2_config['speech_boost']
# # # #     )
# # # #
# # # #     loader_2_2 = DataLoader(
# # # #         dataset_2_2,
# # # #         batch_size=config.batch_size,
# # # #         shuffle=True,
# # # #         num_workers=0,
# # # #         collate_fn=pad_collate
# # # #     )
# # # #
# # # #     # [3/4] 训练阶段2.1
# # # #     print('\n[3/4] Training Sub-stage 2.1...')
# # # #     stage_start_time = time.time()
# # # #
# # # #     loss_history_2_1, weight_history_2_1 = train_substage(
# # # #         trainer, speaker_model, enhancer, loader_2_1,
# # # #         config, config.stage_2_1_config, speaker_id_map, stage_start_time
# # # #     )
# # # #
# # # #     # [4/4] 训练阶段2.2
# # # #     print('\n[4/4] Training Sub-stage 2.2...')
# # # #
# # # #     loss_history_2_2, weight_history_2_2 = train_substage(
# # # #         trainer, speaker_model, enhancer, loader_2_2,
# # # #         config, config.stage_2_2_config, speaker_id_map, stage_start_time
# # # #     )
# # # #
# # # #     # === 全部完成 ===
# # # #     total_time = (time.time() - stage_start_time) / 60
# # # #
# # # #     print('\n' + '=' * 80)
# # # #     print('🎉 STAGE 2 TRAINING COMPLETED')
# # # #     print('=' * 80)
# # # #     print(f'⏱️  Total time: {total_time:.1f} minutes')
# # # #     print(f'📊 Sub-stage 2.1: {len(loss_history_2_1)} batches')
# # # #     print(f'📊 Sub-stage 2.2: {len(loss_history_2_2)} batches')
# # # #     print(f'👥 Total unique speakers: {len(speaker_id_map)}')
# # # #     print('=' * 80)
# # # #
# # # #     # 保存最终模型
# # # #     final_checkpoint_dir = Path(config.checkpoint_dir)
# # # #     final_checkpoint_dir.mkdir(exist_ok=True)
# # # #
# # # #     final_path = final_checkpoint_dir / 'final_model_stage2_complete.pth'
# # # #     final_weight_stats = trainer.get_weight_statistics()
# # # #
# # # #     torch.save({
# # # #         'model_state_dict': trainer.mlp.state_dict(),
# # # #         'optimizer_state_dict': trainer.optimizer.state_dict(),
# # # #         'stage': 'stage2_complete',
# # # #         'config': {
# # # #             'model_type': 'Confidence Stage 2 Complete',
# # # #             'stage1_checkpoint': config.stage1_checkpoint,
# # # #             'substage_2_1': config.stage_2_1_config,
# # # #             'substage_2_2': config.stage_2_2_config,
# # # #             'total_batches': len(loss_history_2_1) + len(loss_history_2_2),
# # # #             'training_minutes': total_time
# # # #         },
# # # #         'weight_stats': final_weight_stats,
# # # #         'loss_history': {
# # # #             'stage_2_1': loss_history_2_1,
# # # #             'stage_2_2': loss_history_2_2
# # # #         },
# # # #         'weight_history': {
# # # #             'stage_2_1': weight_history_2_1,
# # # #             'stage_2_2': weight_history_2_2
# # # #         }
# # # #     }, final_path)
# # # #
# # # #     print(f'\n💾 Final Stage 2 model saved: {final_path}')
# # # #
# # # #     # 保存详细报告
# # # #     report_path = final_checkpoint_dir / 'stage2_training_report.txt'
# # # #     with open(report_path, 'w') as f:
# # # #         f.write('=' * 80 + '\n')
# # # #         f.write('CONFIDENCE STAGE 2 TRAINING REPORT\n')
# # # #         f.write('=' * 80 + '\n\n')
# # # #
# # # #         f.write('TRAINING STRATEGY:\n')
# # # #         f.write('- Two-phase progressive training\n')
# # # #         f.write('- Phase 2.1: Freeze MLP + Train Confidence (50 batches)\n')
# # # #         f.write('- Phase 2.2: Fine-tune all (150 batches)\n\n')
# # # #
# # # #         f.write('STAGE 2.1 RESULTS:\n')
# # # #         f.write(f'  Batches: {len(loss_history_2_1)}\n')
# # # #         f.write(f'  Avg loss: {np.mean(loss_history_2_1):.4f}\n')
# # # #         f.write(f'  Min loss: {np.min(loss_history_2_1):.4f}\n')
# # # #         f.write(f'  Max loss: {np.max(loss_history_2_1):.4f}\n\n')
# # # #
# # # #         f.write('STAGE 2.2 RESULTS:\n')
# # # #         f.write(f'  Batches: {len(loss_history_2_2)}\n')
# # # #         f.write(f'  Avg loss: {np.mean(loss_history_2_2):.4f}\n')
# # # #         f.write(f'  Min loss: {np.min(loss_history_2_2):.4f}\n')
# # # #         f.write(f'  Max loss: {np.max(loss_history_2_2):.4f}\n\n')
# # # #
# # # #         f.write('FINAL WEIGHT STATISTICS:\n')
# # # #         if final_weight_stats:
# # # #             f.write(
# # # #                 f'  Noisy weight: {final_weight_stats["w_noisy"]["mean"]:.4f} ± {final_weight_stats["w_noisy"]["std"]:.4f}\n')
# # # #             f.write(
# # # #                 f'  Enhanced weight: {final_weight_stats["w_enhanced"]["mean"]:.4f} ± {final_weight_stats["w_enhanced"]["std"]:.4f}\n\n')
# # # #
# # # #         f.write(f'Total time: {total_time:.1f} minutes\n')
# # # #         f.write(f'Total speakers: {len(speaker_id_map)}\n')
# # # #
# # # #     print(f'📄 Training report saved: {report_path}')
# # # #
# # # #     # 最终建议
# # # #     print('\n' + '=' * 80)
# # # #     print('🎯 NEXT STEPS')
# # # #     print('=' * 80)
# # # #     print('1. Test Stage 2 model on all 15 conditions:')
# # # #     print(f'   python evaluate_confidence_paper_15conditions.py \\')
# # # #     print(f'       --checkpoint {final_path} \\')
# # # #     print(f'       --output_dir results_confidence_stage2 \\')
# # # #     print(f'       --num_pairs 1000')
# # # #     print('')
# # # #     print('2. Compare with Stage 1:')
# # # #     print('   python view_results.py results_confidence_stage2')
# # # #     print('')
# # # #     print('3. If Stage 2 is worse than Stage 1:')
# # # #     print('   - Use Stage 1 model for paper')
# # # #     print('   - Mention Stage 2 attempt in discussion')
# # # #     print('')
# # # #     print('4. If Stage 2 improves Speech without hurting Noise/Music:')
# # # #     print('   - Great! Use Stage 2 model')
# # # #     print('   - Emphasize adaptive training strategy')
# # # #     print('=' * 80)
# # # #
# # # #
# # # # if __name__ == '__main__':
# # # #     main()
# # """
# # 阶段2：安全的Speech性能优化（两阶段渐进训练）- Bug修复版
# #
# # 🛡️ 安全机制：
# # 1. 阶段2.1：冻结MLP主干，只训练Confidence Head（50 batches）
# # 2. 阶段2.2：解冻全部，极小lr微调（150 batches）
# # 3. Speech适度提升至40%（不是60%）
# # 4. 保持完整SNR范围(-20, 0)
# # 5. 每25 batch保存checkpoint，可随时回退
# # 6. 监控权重变化，及时发现异常
# #
# # ✅ 修复：musan_files属性访问顺序问题
# #
# # 用法：
# #     python train_confidence_stage2_safe.py
# # """
# #
# # import torch
# # import torch.nn.functional as F
# # import os
# # import subprocess
# # import sys
# # import time
# # import random
# # import numpy as np
# # from pathlib import Path
# # from torch.utils.data import DataLoader
# # import json
# #
# # # 禁用DeepFilter日志
# # os.environ['DF_DISABLE_LOGGING'] = '1'
# # original_check_output = subprocess.check_output
# # subprocess.check_output = lambda *args, **kwargs: b'unknown' if args and 'git' in str(
# #     args[0]) else original_check_output(*args, **kwargs)
# #
# # sys.path.append('scripts')
# # from models_confidence import ConfidenceSupConTrainer
# # from speaker_embedding import SpeakerEmbeddingExtractor
# # from speech_enhancer import SpeechEnhancer
# # from dataset import VoxCelebMusanDataset
# #
# #
# # class ConfigStage2:
# #     """阶段2训练配置 - 安全优化"""
# #     # 数据路径
# #     voxceleb_dir = 'data/voxceleb1'
# #     musan_dir = 'data/musan'
# #
# #     # ===== 阶段1 checkpoint =====
# #     stage1_checkpoint = 'checkpoints_confidence_400batch/final_model_confidence.pth'
# #
# #     # ===== 两个子阶段的参数 =====
# #
# #     # 子阶段2.1：冻结MLP主干
# #     stage_2_1_config = {
# #         'name': 'freeze_mlp',
# #         'batches': 50,
# #         'learning_rate': 1e-3,
# #         'freeze_mlp': True,
# #         'speech_boost': 2.0,  # Speech占40%
# #         'description': 'Train only Confidence Head with frozen MLP'
# #     }
# #
# #     # 子阶段2.2：解冻全部，精细调整
# #     stage_2_2_config = {
# #         'name': 'finetune_all',
# #         'batches': 150,
# #         'learning_rate': 1e-4,
# #         'freeze_mlp': False,
# #         'speech_boost': 1.5,  # Speech占37.5%
# #         'description': 'Fine-tune all parameters with very small lr'
# #     }
# #
# #     # ===== 通用参数 =====
# #     batch_size = 32
# #     snr_range = (-20, 0)
# #     temperature = 0.07
# #     confidence_hidden_dim = 64
# #
# #     # 输出目录
# #     checkpoint_dir = 'checkpoints_confidence_stage2'
# #
# #     # 控制参数
# #     checkpoint_interval = 25
# #     log_weights_interval = 5
# #
# #     # 设备
# #     device = 'cuda' if torch.cuda.is_available() else 'cpu'
# #     seed = 42
# #
# #
# # class SpeechBoostedDataset(VoxCelebMusanDataset):
# #     """
# #     适度增强Speech噪声比例的数据集
# #
# #     ✅ 修复：确保在父类初始化后再访问musan_files
# #     """
# #
# #     def __init__(self, *args, speech_boost_factor=1.5, **kwargs):
# #         self.speech_boost_factor = speech_boost_factor
# #
# #         # ✅ 先调用父类初始化
# #         super().__init__(*args, **kwargs)
# #
# #         # ✅ 现在可以安全访问self.musan_files了
# #         # 统计原始分布
# #         original_counts = {
# #             'noise': len(self.musan_files['noise']),
# #             'music': len(self.musan_files['music']),
# #             'speech': len(self.musan_files['speech'])
# #         }
# #
# #         print(f'\n📊 Original MUSAN distribution:')
# #         for noise_type, count in original_counts.items():
# #             print(f'   {noise_type}: {count} files')
# #
# #         # 适度扩充Speech文件列表
# #         if speech_boost_factor > 1.0:
# #             original_speech = self.musan_files['speech'].copy()
# #             repeats = int(speech_boost_factor) - 1
# #             fractional = speech_boost_factor - int(speech_boost_factor)
# #
# #             # 整数倍复制
# #             for _ in range(repeats):
# #                 self.musan_files['speech'].extend(original_speech)
# #
# #             # 小数部分复制
# #             if fractional > 0:
# #                 n_extra = int(len(original_speech) * fractional)
# #                 self.musan_files['speech'].extend(
# #                     random.sample(original_speech, n_extra)
# #                 )
# #
# #         new_counts = {
# #             'noise': len(self.musan_files['noise']),
# #             'music': len(self.musan_files['music']),
# #             'speech': len(self.musan_files['speech'])
# #         }
# #
# #         total = sum(new_counts.values())
# #         print(f'\n✅ Adjusted MUSAN distribution (boost={speech_boost_factor:.1f}x):')
# #         for noise_type, count in new_counts.items():
# #             pct = count / total * 100
# #             print(f'   {noise_type}: {count} files ({pct:.1f}%)')
# #
# #
# # def set_seed(seed):
# #     """设置随机种子"""
# #     random.seed(seed)
# #     np.random.seed(seed)
# #     torch.manual_seed(seed)
# #     if torch.cuda.is_available():
# #         torch.cuda.manual_seed_all(seed)
# #         torch.backends.cudnn.deterministic = True
# #         torch.backends.cudnn.benchmark = False
# #
# #
# # def pad_collate(batch):
# #     """处理变长音频的collate函数"""
# #     max_anchor = max([item['anchor_noisy'].shape[1] for item in batch])
# #     max_positive = max([item['positive_noisy'].shape[1] for item in batch])
# #     max_negative = max([item['negative_noisy'].shape[1] for item in batch])
# #
# #     padded_batch = {
# #         'anchor_noisy': [],
# #         'positive_noisy': [],
# #         'negative_noisy': [],
# #         'snr': [],
# #         'speaker_id': []
# #     }
# #
# #     for item in batch:
# #         # Anchor padding
# #         audio = item['anchor_noisy']
# #         if audio.shape[1] < max_anchor:
# #             padding = torch.zeros(1, max_anchor - audio.shape[1])
# #             audio = torch.cat([audio, padding], dim=1)
# #         padded_batch['anchor_noisy'].append(audio)
# #
# #         # Positive padding
# #         audio = item['positive_noisy']
# #         if audio.shape[1] < max_positive:
# #             padding = torch.zeros(1, max_positive - audio.shape[1])
# #             audio = torch.cat([audio, padding], dim=1)
# #         padded_batch['positive_noisy'].append(audio)
# #
# #         # Negative padding
# #         audio = item['negative_noisy']
# #         if audio.shape[1] < max_negative:
# #             padding = torch.zeros(1, max_negative - audio.shape[1])
# #             audio = torch.cat([audio, padding], dim=1)
# #         padded_batch['negative_noisy'].append(audio)
# #
# #         padded_batch['snr'].append(item['snr'])
# #         padded_batch['speaker_id'].append(item['speaker_id'])
# #
# #     # Stack音频数据
# #     for key in ['anchor_noisy', 'positive_noisy', 'negative_noisy']:
# #         padded_batch[key] = torch.stack(padded_batch[key])
# #
# #     return padded_batch
# #
# #
# # def extract_and_normalize_embeddings(audio_batch, speaker_model, enhancer, device, speaker_id_map):
# #     """提取并归一化embeddings"""
# #     batch_size = len(audio_batch['anchor_noisy'])
# #
# #     all_noisy_embs = []
# #     all_enhanced_embs = []
# #     all_labels = []
# #
# #     for i in range(batch_size):
# #         speaker_id_str = audio_batch['speaker_id'][i]
# #
# #         if speaker_id_str not in speaker_id_map:
# #             speaker_id_map[speaker_id_str] = len(speaker_id_map)
# #         speaker_id_int = speaker_id_map[speaker_id_str]
# #
# #         # === Anchor ===
# #         anchor_noisy = audio_batch['anchor_noisy'][i]
# #         anchor_enhanced = enhancer.enhance_audio(anchor_noisy, sr=16000)
# #
# #         emb_noisy = speaker_model.extract_embedding(audio_tensor=anchor_noisy)
# #         emb_enhanced = speaker_model.extract_embedding(audio_tensor=anchor_enhanced)
# #
# #         emb_noisy = F.normalize(emb_noisy.squeeze().unsqueeze(0), p=2, dim=1).squeeze(0)
# #         emb_enhanced = F.normalize(emb_enhanced.squeeze().unsqueeze(0), p=2, dim=1).squeeze(0)
# #
# #         all_noisy_embs.append(emb_noisy)
# #         all_enhanced_embs.append(emb_enhanced)
# #         all_labels.append(speaker_id_int)
# #
# #         # === Positive ===
# #         pos_noisy = audio_batch['positive_noisy'][i]
# #         pos_enhanced = enhancer.enhance_audio(pos_noisy, sr=16000)
# #
# #         emb_noisy = speaker_model.extract_embedding(audio_tensor=pos_noisy)
# #         emb_enhanced = speaker_model.extract_embedding(audio_tensor=pos_enhanced)
# #
# #         emb_noisy = F.normalize(emb_noisy.squeeze().unsqueeze(0), p=2, dim=1).squeeze(0)
# #         emb_enhanced = F.normalize(emb_enhanced.squeeze().unsqueeze(0), p=2, dim=1).squeeze(0)
# #
# #         all_noisy_embs.append(emb_noisy)
# #         all_enhanced_embs.append(emb_enhanced)
# #         all_labels.append(speaker_id_int)
# #
# #         # === Negative ===
# #         neg_noisy = audio_batch['negative_noisy'][i]
# #         neg_enhanced = enhancer.enhance_audio(neg_noisy, sr=16000)
# #
# #         emb_noisy = speaker_model.extract_embedding(audio_tensor=neg_noisy)
# #         emb_enhanced = speaker_model.extract_embedding(audio_tensor=neg_enhanced)
# #
# #         emb_noisy = F.normalize(emb_noisy.squeeze().unsqueeze(0), p=2, dim=1).squeeze(0)
# #         emb_enhanced = F.normalize(emb_enhanced.squeeze().unsqueeze(0), p=2, dim=1).squeeze(0)
# #
# #         all_noisy_embs.append(emb_noisy)
# #         all_enhanced_embs.append(emb_enhanced)
# #         all_labels.append(speaker_id_int + 100000)
# #
# #     noisy_embeddings = torch.stack(all_noisy_embs)
# #     enhanced_embeddings = torch.stack(all_enhanced_embs)
# #     labels = torch.tensor(all_labels, dtype=torch.long)
# #
# #     return noisy_embeddings, enhanced_embeddings, labels
# #
# #
# # def freeze_mlp_backbone(trainer):
# #     """冻结MLP主干（fc1, fc2, fc3）"""
# #     for param in trainer.mlp.fc1.parameters():
# #         param.requires_grad = False
# #     for param in trainer.mlp.fc2.parameters():
# #         param.requires_grad = False
# #     for param in trainer.mlp.fc3.parameters():
# #         param.requires_grad = False
# #
# #     print('🧊 Frozen MLP backbone (fc1, fc2, fc3)')
# #     print('   - Only training Confidence Head')
# #
# #     # 统计可训练参数
# #     trainable_params = sum(p.numel() for p in trainer.mlp.parameters() if p.requires_grad)
# #     total_params = sum(p.numel() for p in trainer.mlp.parameters())
# #     print(
# #         f'   - Trainable params: {trainable_params:,} / {total_params:,} ({trainable_params / total_params * 100:.1f}%)')
# #
# #
# # def unfreeze_all(trainer):
# #     """解冻所有参数"""
# #     for param in trainer.mlp.parameters():
# #         param.requires_grad = True
# #
# #     print('🔓 Unfrozen all parameters')
# #     print('   - Training entire network')
# #
# #     trainable_params = sum(p.numel() for p in trainer.mlp.parameters() if p.requires_grad)
# #     print(f'   - Trainable params: {trainable_params:,}')
# #
# #
# # def save_checkpoint(trainer, loss, batch_idx, total_time, config, stage_name, weight_stats=None, is_final=False):
# #     """保存checkpoint"""
# #     checkpoint_dir = Path(config.checkpoint_dir)
# #     checkpoint_dir.mkdir(exist_ok=True)
# #
# #     if is_final:
# #         filename = f'final_model_stage2_{stage_name}.pth'
# #     else:
# #         filename = f'checkpoint_stage2_{stage_name}_batch_{batch_idx}.pth'
# #
# #     checkpoint_path = checkpoint_dir / filename
# #
# #     checkpoint_data = {
# #         'model_state_dict': trainer.mlp.state_dict(),
# #         'optimizer_state_dict': trainer.optimizer.state_dict(),
# #         'loss': loss,
# #         'batch': batch_idx,
# #         'training_minutes': total_time,
# #         'stage': f'stage2_{stage_name}',
# #         'config': {
# #             'model_type': 'Confidence Stage 2',
# #             'batch_size': config.batch_size,
# #             'temperature': config.temperature,
# #             'confidence_hidden_dim': config.confidence_hidden_dim,
# #             'snr_range': config.snr_range,
# #             'seed': config.seed,
# #             'stage1_checkpoint': config.stage1_checkpoint
# #         }
# #     }
# #
# #     if weight_stats:
# #         checkpoint_data['weight_stats'] = weight_stats
# #
# #     torch.save(checkpoint_data, checkpoint_path)
# #
# #     return checkpoint_path
# #
# #
# # def train_substage(trainer, speaker_model, enhancer, train_loader,
# #                    config, substage_config, speaker_id_map, stage_start_time):
# #     """训练一个子阶段"""
# #
# #     stage_name = substage_config['name']
# #     max_batches = substage_config['batches']
# #     lr = substage_config['learning_rate']
# #
# #     print('\n' + '=' * 80)
# #     print(f'🎯 SUB-STAGE: {substage_config["description"]}')
# #     print('=' * 80)
# #     print(f'Batches: {max_batches}')
# #     print(f'Learning rate: {lr}')
# #     print(f'Freeze MLP: {substage_config["freeze_mlp"]}')
# #     print(f'Speech boost: {substage_config["speech_boost"]}x')
# #     print('=' * 80)
# #
# #     # 设置冻结状态
# #     if substage_config['freeze_mlp']:
# #         freeze_mlp_backbone(trainer)
# #     else:
# #         unfreeze_all(trainer)
# #
# #     # 更新优化器（使用新的学习率）
# #     trainer.optimizer = torch.optim.AdamW(
# #         filter(lambda p: p.requires_grad, trainer.mlp.parameters()),
# #         lr=lr,
# #         weight_decay=1e-4
# #     )
# #
# #     print(f'✅ Optimizer updated with lr={lr}')
# #
# #     # 训练循环
# #     total_loss = 0.0
# #     loss_history = []
# #     weight_history = []
# #
# #     trainer.mlp.train()
# #
# #     batch_count = 0
# #     data_iterator = iter(train_loader)
# #
# #     while batch_count < max_batches:
# #         try:
# #             audio_batch = next(data_iterator)
# #         except StopIteration:
# #             data_iterator = iter(train_loader)
# #             audio_batch = next(data_iterator)
# #
# #         batch_count += 1
# #         batch_start = time.time()
# #
# #         # 提取embeddings
# #         noisy_embs, enhanced_embs, labels = extract_and_normalize_embeddings(
# #             audio_batch, speaker_model, enhancer, config.device, speaker_id_map
# #         )
# #
# #         noisy_embs = noisy_embs.to(config.device)
# #         enhanced_embs = enhanced_embs.to(config.device)
# #         labels = labels.to(config.device)
# #
# #         # 训练步骤
# #         log_weights = (batch_count % config.log_weights_interval == 0) or (batch_count == 1)
# #         loss, stats = trainer.train_step(
# #             noisy_embs, enhanced_embs, labels, log_weights=log_weights
# #         )
# #
# #         # 记录
# #         loss_history.append(loss)
# #         total_loss += loss
# #
# #         if log_weights:
# #             weight_history.append({
# #                 'batch': batch_count,
# #                 'w_noisy': stats['w_noisy_mean'],
# #                 'w_enhanced': stats['w_enhanced_mean'],
# #                 'w_noisy_std': stats['w_noisy_std'],
# #                 'w_enhanced_std': stats['w_enhanced_std']
# #             })
# #
# #         batch_time = time.time() - batch_start
# #         elapsed_total = (time.time() - stage_start_time) / 60
# #
# #         # 打印进度
# #         if batch_count % 5 == 0 or batch_count == 1:
# #             avg_loss = total_loss / batch_count
# #             msg = f'[{stage_name}] Batch {batch_count:3d}/{max_batches} | Loss: {loss:.4f} (avg: {avg_loss:.4f})'
# #
# #             if log_weights:
# #                 msg += f' | w_n={stats["w_noisy_mean"]:.3f} w_e={stats["w_enhanced_mean"]:.3f}'
# #
# #             msg += f' | Time: {batch_time:.1f}s | Total: {elapsed_total:.1f}min'
# #             print(msg)
# #
# #         # 保存checkpoint
# #         if batch_count % config.checkpoint_interval == 0:
# #             current_weight_stats = trainer.get_weight_statistics()
# #             checkpoint_path = save_checkpoint(
# #                 trainer,
# #                 total_loss / batch_count,
# #                 batch_count,
# #                 elapsed_total,
# #                 config,
# #                 stage_name,
# #                 weight_stats=current_weight_stats,
# #                 is_final=False
# #             )
# #             print(f'  💾 Checkpoint saved: {checkpoint_path.name}')
# #
# #     # 子阶段完成
# #     avg_loss = total_loss / max_batches
# #
# #     print(f'\n✅ Sub-stage "{stage_name}" completed')
# #     print(f'   Average loss: {avg_loss:.4f}')
# #     print(f'   Batches trained: {max_batches}')
# #
# #     # 保存子阶段最终模型
# #     final_weight_stats = trainer.get_weight_statistics()
# #     final_path = save_checkpoint(
# #         trainer,
# #         avg_loss,
# #         max_batches,
# #         (time.time() - stage_start_time) / 60,
# #         config,
# #         stage_name,
# #         weight_stats=final_weight_stats,
# #         is_final=True
# #     )
# #
# #     print(f'💾 Sub-stage final model saved: {final_path}')
# #
# #     return loss_history, weight_history
# #
# #
# # def main():
# #     config = ConfigStage2()
# #     set_seed(config.seed)
# #
# #     print('=' * 80)
# #     print('🛡️ CONFIDENCE STAGE 2: SAFE SPEECH OPTIMIZATION')
# #     print('=' * 80)
# #     print('Strategy: Two-phase progressive training')
# #     print(f'Base model: {config.stage1_checkpoint}')
# #     print('=' * 80)
# #     print('Phase 2.1: Freeze MLP + Train Confidence Head (50 batches)')
# #     print('  - Speech boost: 2.0x (40%)')
# #     print('  - Learning rate: 1e-3')
# #     print('  - Goal: Adapt confidence network to Speech')
# #     print('')
# #     print('Phase 2.2: Unfreeze All + Fine-tune (150 batches)')
# #     print('  - Speech boost: 1.5x (37.5%)')
# #     print('  - Learning rate: 1e-4 (very small!)')
# #     print('  - Goal: Gentle adjustment without forgetting')
# #     print('=' * 80)
# #     print(f'Device: {config.device}')
# #     print(f'Random seed: {config.seed}')
# #     print(f'SNR range: {config.snr_range} (full coverage)')
# #     print(f'Total expected time: ~2.5-3 hours')
# #     print('=' * 80)
# #
# #     # [1/4] 加载模型
# #     print('\n[1/4] Loading models and Stage 1 checkpoint...')
# #     speaker_model = SpeakerEmbeddingExtractor('ecapa')
# #     enhancer = SpeechEnhancer()
# #
# #     # 加载Stage 1模型
# #     stage1_checkpoint = torch.load(config.stage1_checkpoint, map_location=config.device, weights_only=False)
# #
# #     trainer = ConfidenceSupConTrainer(
# #         embedding_dim=192,
# #         device=config.device,
# #         temperature=config.temperature,
# #         confidence_hidden_dim=config.confidence_hidden_dim
# #     )
# #
# #     trainer.mlp.load_state_dict(stage1_checkpoint['model_state_dict'])
# #
# #     print(f'✅ Loaded Stage 1 checkpoint')
# #     print(f'   - Stage 1 loss: {stage1_checkpoint["loss"]:.4f}')
# #     print(f'   - Stage 1 batches: {stage1_checkpoint["batch"]}')
# #
# #     # [2/4] 准备数据集
# #     print('\n[2/4] Preparing datasets for two sub-stages...')
# #
# #     speaker_id_map = {}
# #
# #     # 阶段2.1数据集（Speech占40%）
# #     print('\n📦 Dataset for Sub-stage 2.1:')
# #     dataset_2_1 = SpeechBoostedDataset(
# #         config.voxceleb_dir,
# #         config.musan_dir,
# #         split='train',
# #         snr_range=config.snr_range,
# #         speech_boost_factor=config.stage_2_1_config['speech_boost']
# #     )
# #
# #     loader_2_1 = DataLoader(
# #         dataset_2_1,
# #         batch_size=config.batch_size,
# #         shuffle=True,
# #         num_workers=0,
# #         collate_fn=pad_collate
# #     )
# #
# #     # 阶段2.2数据集（Speech占37.5%）
# #     print('\n📦 Dataset for Sub-stage 2.2:')
# #     dataset_2_2 = SpeechBoostedDataset(
# #         config.voxceleb_dir,
# #         config.musan_dir,
# #         split='train',
# #         snr_range=config.snr_range,
# #         speech_boost_factor=config.stage_2_2_config['speech_boost']
# #     )
# #
# #     loader_2_2 = DataLoader(
# #         dataset_2_2,
# #         batch_size=config.batch_size,
# #         shuffle=True,
# #         num_workers=0,
# #         collate_fn=pad_collate
# #     )
# #
# #     # [3/4] 训练阶段2.1
# #     print('\n[3/4] Training Sub-stage 2.1...')
# #     stage_start_time = time.time()
# #
# #     loss_history_2_1, weight_history_2_1 = train_substage(
# #         trainer, speaker_model, enhancer, loader_2_1,
# #         config, config.stage_2_1_config, speaker_id_map, stage_start_time
# #     )
# #
# #     # [4/4] 训练阶段2.2
# #     print('\n[4/4] Training Sub-stage 2.2...')
# #
# #     loss_history_2_2, weight_history_2_2 = train_substage(
# #         trainer, speaker_model, enhancer, loader_2_2,
# #         config, config.stage_2_2_config, speaker_id_map, stage_start_time
# #     )
# #
# #     # === 全部完成 ===
# #     total_time = (time.time() - stage_start_time) / 60
# #
# #     print('\n' + '=' * 80)
# #     print('🎉 STAGE 2 TRAINING COMPLETED')
# #     print('=' * 80)
# #     print(f'⏱️  Total time: {total_time:.1f} minutes')
# #     print(f'📊 Sub-stage 2.1: {len(loss_history_2_1)} batches')
# #     print(f'📊 Sub-stage 2.2: {len(loss_history_2_2)} batches')
# #     print(f'👥 Total unique speakers: {len(speaker_id_map)}')
# #     print('=' * 80)
# #
# #     # 保存最终模型
# #     final_checkpoint_dir = Path(config.checkpoint_dir)
# #     final_checkpoint_dir.mkdir(exist_ok=True)
# #
# #     final_path = final_checkpoint_dir / 'final_model_stage2_complete.pth'
# #     final_weight_stats = trainer.get_weight_statistics()
# #
# #     torch.save({
# #         'model_state_dict': trainer.mlp.state_dict(),
# #         'optimizer_state_dict': trainer.optimizer.state_dict(),
# #         'stage': 'stage2_complete',
# #         'config': {
# #             'model_type': 'Confidence Stage 2 Complete',
# #             'stage1_checkpoint': config.stage1_checkpoint,
# #             'substage_2_1': config.stage_2_1_config,
# #             'substage_2_2': config.stage_2_2_config,
# #             'total_batches': len(loss_history_2_1) + len(loss_history_2_2),
# #             'training_minutes': total_time
# #         },
# #         'weight_stats': final_weight_stats,
# #         'loss_history': {
# #             'stage_2_1': loss_history_2_1,
# #             'stage_2_2': loss_history_2_2
# #         },
# #         'weight_history': {
# #             'stage_2_1': weight_history_2_1,
# #             'stage_2_2': weight_history_2_2
# #         }
# #     }, final_path)
# #
# #     print(f'\n💾 Final Stage 2 model saved: {final_path}')
# #
# #     # 保存详细报告
# #     report_path = final_checkpoint_dir / 'stage2_training_report.txt'
# #     with open(report_path, 'w') as f:
# #         f.write('=' * 80 + '\n')
# #         f.write('CONFIDENCE STAGE 2 TRAINING REPORT\n')
# #         f.write('=' * 80 + '\n\n')
# #
# #         f.write('TRAINING STRATEGY:\n')
# #         f.write('- Two-phase progressive training\n')
# #         f.write('- Phase 2.1: Freeze MLP + Train Confidence (50 batches)\n')
# #         f.write('- Phase 2.2: Fine-tune all (150 batches)\n\n')
# #
# #         f.write('STAGE 2.1 RESULTS:\n')
# #         f.write(f'  Batches: {len(loss_history_2_1)}\n')
# #         f.write(f'  Avg loss: {np.mean(loss_history_2_1):.4f}\n')
# #         f.write(f'  Min loss: {np.min(loss_history_2_1):.4f}\n')
# #         f.write(f'  Max loss: {np.max(loss_history_2_1):.4f}\n\n')
# #
# #         f.write('STAGE 2.2 RESULTS:\n')
# #         f.write(f'  Batches: {len(loss_history_2_2)}\n')
# #         f.write(f'  Avg loss: {np.mean(loss_history_2_2):.4f}\n')
# #         f.write(f'  Min loss: {np.min(loss_history_2_2):.4f}\n')
# #         f.write(f'  Max loss: {np.max(loss_history_2_2):.4f}\n\n')
# #
# #         f.write('FINAL WEIGHT STATISTICS:\n')
# #         if final_weight_stats:
# #             f.write(
# #                 f'  Noisy weight: {final_weight_stats["w_noisy"]["mean"]:.4f} ± {final_weight_stats["w_noisy"]["std"]:.4f}\n')
# #             f.write(
# #                 f'  Enhanced weight: {final_weight_stats["w_enhanced"]["mean"]:.4f} ± {final_weight_stats["w_enhanced"]["std"]:.4f}\n\n')
# #
# #         f.write(f'Total time: {total_time:.1f} minutes\n')
# #         f.write(f'Total speakers: {len(speaker_id_map)}\n')
# #
# #     print(f'📄 Training report saved: {report_path}')
# #
# #     # 最终建议
# #     print('\n' + '=' * 80)
# #     print('🎯 NEXT STEPS')
# #     print('=' * 80)
# #     print('1. Test Stage 2 model on all 15 conditions:')
# #     print(f'   python evaluate_confidence_paper_15conditions.py \\')
# #     print(f'       --checkpoint {final_path} \\')
# #     print(f'       --output_dir results_confidence_stage2 \\')
# #     print(f'       --num_pairs 1000')
# #     print('')
# #     print('2. Compare with Stage 1:')
# #     print('   python view_results.py results_confidence_stage2')
# #     print('')
# #     print('3. If Stage 2 is worse than Stage 1:')
# #     print('   - Use Stage 1 model for paper')
# #     print('   - Mention Stage 2 attempt in discussion')
# #     print('')
# #     print('4. If Stage 2 improves Speech without hurting Noise/Music:')
# #     print('   - Great! Use Stage 2 model')
# #     print('   - Emphasize adaptive training strategy')
# #     print('=' * 80)
# #
# #
# # if __name__ == '__main__':
# #     main()
# """
# 阶段2：安全的Speech性能优化（两阶段渐进训练）- 简化版
#
# 🛡️ 安全机制：
# 1. 阶段2.1：冻结MLP主干，只训练Confidence Head（75 batches）
# 2. 阶段2.2：解冻全部，极小lr微调（175 batches）
# 3. 使用标准数据集分布（不修改MUSAN比例）
# 4. 通过增加batch数来获得足够训练
# 5. 每25 batch保存checkpoint，可随时回退
#
# ✅ 简化：直接使用VoxCelebMusanDataset，避免继承问题
#
# 用法：
#     python train_confidence_stage2_safe.py
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
# import json
#
# # 禁用DeepFilter日志
# os.environ['DF_DISABLE_LOGGING'] = '1'
# original_check_output = subprocess.check_output
# subprocess.check_output = lambda *args, **kwargs: b'unknown' if args and 'git' in str(
#     args[0]) else original_check_output(*args, **kwargs)
#
# sys.path.append('scripts')
# from models_confidence import ConfidenceSupConTrainer
# from speaker_embedding import SpeakerEmbeddingExtractor
# from speech_enhancer import SpeechEnhancer
# from dataset import VoxCelebMusanDataset
#
#
# class ConfigStage2:
#     """阶段2训练配置 - 安全优化"""
#     # 数据路径
#     voxceleb_dir = 'data/voxceleb1'
#     musan_dir = 'data/musan'
#
#     # ===== 阶段1 checkpoint =====
#     stage1_checkpoint = 'checkpoints_confidence_400batch/final_model_confidence.pth'
#
#     # ===== 两个子阶段的参数 =====
#     # 子阶段2.1：冻结MLP主干
#     stage_2_1_config = {
#         'name': 'freeze_mlp',
#         'batches': 75,
#         'learning_rate': 1e-3,
#         'freeze_mlp': True,
#         'description': 'Train only Confidence Head with frozen MLP'
#     }
#
#     # 子阶段2.2：解冻全部，精细调整
#     stage_2_2_config = {
#         'name': 'finetune_all',
#         'batches': 175,
#         'learning_rate': 1e-4,
#         'freeze_mlp': False,
#         'description': 'Fine-tune all parameters with very small lr'
#     }
#
#     # ===== 通用参数 =====
#     batch_size = 32
#     snr_range = (-20, 0)
#     temperature = 0.07
#     confidence_hidden_dim = 64
#
#     # 输出目录
#     checkpoint_dir = 'checkpoints_confidence_stage2'
#
#     # 控制参数
#     checkpoint_interval = 25
#     log_weights_interval = 5
#
#     # 设备
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     seed = 42
#
#
# def set_seed(seed):
#     """设置随机种子"""
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
#     """处理变长音频的collate函数"""
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
#     # Stack音频数据
#     for key in ['anchor_noisy', 'positive_noisy', 'negative_noisy']:
#         padded_batch[key] = torch.stack(padded_batch[key])
#
#     return padded_batch
#
#
# def extract_and_normalize_embeddings(audio_batch, speaker_model, enhancer, device, speaker_id_map):
#     """提取并归一化embeddings"""
#     batch_size = len(audio_batch['anchor_noisy'])
#
#     all_noisy_embs = []
#     all_enhanced_embs = []
#     all_labels = []
#
#     for i in range(batch_size):
#         speaker_id_str = audio_batch['speaker_id'][i]
#
#         if speaker_id_str not in speaker_id_map:
#             speaker_id_map[speaker_id_str] = len(speaker_id_map)
#         speaker_id_int = speaker_id_map[speaker_id_str]
#
#         # === Anchor ===
#         anchor_noisy = audio_batch['anchor_noisy'][i]
#         anchor_enhanced = enhancer.enhance_audio(anchor_noisy, sr=16000)
#
#         emb_noisy = speaker_model.extract_embedding(audio_tensor=anchor_noisy)
#         emb_enhanced = speaker_model.extract_embedding(audio_tensor=anchor_enhanced)
#
#         emb_noisy = F.normalize(emb_noisy.squeeze().unsqueeze(0), p=2, dim=1).squeeze(0)
#         emb_enhanced = F.normalize(emb_enhanced.squeeze().unsqueeze(0), p=2, dim=1).squeeze(0)
#
#         all_noisy_embs.append(emb_noisy)
#         all_enhanced_embs.append(emb_enhanced)
#         all_labels.append(speaker_id_int)
#
#         # === Positive ===
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
#         all_labels.append(speaker_id_int)
#
#         # === Negative ===
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
#         all_labels.append(speaker_id_int + 100000)
#
#     noisy_embeddings = torch.stack(all_noisy_embs)
#     enhanced_embeddings = torch.stack(all_enhanced_embs)
#     labels = torch.tensor(all_labels, dtype=torch.long)
#
#     return noisy_embeddings, enhanced_embeddings, labels
#
#
# def freeze_mlp_backbone(trainer):
#     """冻结MLP主干（fc1, fc2, fc3）"""
#     for param in trainer.mlp.fc1.parameters():
#         param.requires_grad = False
#     for param in trainer.mlp.fc2.parameters():
#         param.requires_grad = False
#     for param in trainer.mlp.fc3.parameters():
#         param.requires_grad = False
#
#     print('🧊 Frozen MLP backbone (fc1, fc2, fc3)')
#     print('   - Only training Confidence Head')
#
#     # 统计可训练参数
#     trainable_params = sum(p.numel() for p in trainer.mlp.parameters() if p.requires_grad)
#     total_params = sum(p.numel() for p in trainer.mlp.parameters())
#     print(
#         f'   - Trainable params: {trainable_params:,} / {total_params:,} ({trainable_params / total_params * 100:.1f}%)')
#
#
# def unfreeze_all(trainer):
#     """解冻所有参数"""
#     for param in trainer.mlp.parameters():
#         param.requires_grad = True
#
#     print('🔓 Unfrozen all parameters')
#     print('   - Training entire network')
#
#     trainable_params = sum(p.numel() for p in trainer.mlp.parameters() if p.requires_grad)
#     print(f'   - Trainable params: {trainable_params:,}')
#
#
# def save_checkpoint(trainer, loss, batch_idx, total_time, config, stage_name, weight_stats=None, is_final=False):
#     """保存checkpoint"""
#     checkpoint_dir = Path(config.checkpoint_dir)
#     checkpoint_dir.mkdir(exist_ok=True)
#
#     if is_final:
#         filename = f'final_model_stage2_{stage_name}.pth'
#     else:
#         filename = f'checkpoint_stage2_{stage_name}_batch_{batch_idx}.pth'
#
#     checkpoint_path = checkpoint_dir / filename
#
#     checkpoint_data = {
#         'model_state_dict': trainer.mlp.state_dict(),
#         'optimizer_state_dict': trainer.optimizer.state_dict(),
#         'loss': loss,
#         'batch': batch_idx,
#         'training_minutes': total_time,
#         'stage': f'stage2_{stage_name}',
#         'config': {
#             'model_type': 'Confidence Stage 2',
#             'batch_size': config.batch_size,
#             'temperature': config.temperature,
#             'confidence_hidden_dim': config.confidence_hidden_dim,
#             'snr_range': config.snr_range,
#             'seed': config.seed,
#             'stage1_checkpoint': config.stage1_checkpoint
#         }
#     }
#
#     if weight_stats:
#         checkpoint_data['weight_stats'] = weight_stats
#
#     torch.save(checkpoint_data, checkpoint_path)
#
#     return checkpoint_path
#
#
# def train_substage(trainer, speaker_model, enhancer, train_loader,
#                    config, substage_config, speaker_id_map, stage_start_time):
#     """训练一个子阶段"""
#     stage_name = substage_config['name']
#     max_batches = substage_config['batches']
#     lr = substage_config['learning_rate']
#
#     print('\n' + '=' * 80)
#     print(f'🎯 SUB-STAGE: {substage_config["description"]}')
#     print('=' * 80)
#     print(f'Batches: {max_batches}')
#     print(f'Learning rate: {lr}')
#     print(f'Freeze MLP: {substage_config["freeze_mlp"]}')
#     print('=' * 80)
#
#     # 设置冻结状态
#     if substage_config['freeze_mlp']:
#         freeze_mlp_backbone(trainer)
#     else:
#         unfreeze_all(trainer)
#
#     # 更新优化器（使用新的学习率）
#     trainer.optimizer = torch.optim.AdamW(
#         filter(lambda p: p.requires_grad, trainer.mlp.parameters()),
#         lr=lr,
#         weight_decay=1e-4
#     )
#
#     print(f'✅ Optimizer updated with lr={lr}')
#
#     # 训练循环
#     total_loss = 0.0
#     loss_history = []
#     weight_history = []
#
#     trainer.mlp.train()
#
#     batch_count = 0
#     data_iterator = iter(train_loader)
#
#     while batch_count < max_batches:
#         try:
#             audio_batch = next(data_iterator)
#         except StopIteration:
#             data_iterator = iter(train_loader)
#             audio_batch = next(data_iterator)
#
#         batch_count += 1
#         batch_start = time.time()
#
#         # 提取embeddings
#         noisy_embs, enhanced_embs, labels = extract_and_normalize_embeddings(
#             audio_batch, speaker_model, enhancer, config.device, speaker_id_map
#         )
#
#         noisy_embs = noisy_embs.to(config.device)
#         enhanced_embs = enhanced_embs.to(config.device)
#         labels = labels.to(config.device)
#
#         # 训练步骤
#         log_weights = (batch_count % config.log_weights_interval == 0) or (batch_count == 1)
#         loss, stats = trainer.train_step(
#             noisy_embs, enhanced_embs, labels, log_weights=log_weights
#         )
#
#         # 记录
#         loss_history.append(loss)
#         total_loss += loss
#
#         if log_weights:
#             weight_history.append({
#                 'batch': batch_count,
#                 'w_noisy': stats['w_noisy_mean'],
#                 'w_enhanced': stats['w_enhanced_mean'],
#                 'w_noisy_std': stats['w_noisy_std'],
#                 'w_enhanced_std': stats['w_enhanced_std']
#             })
#
#         batch_time = time.time() - batch_start
#         elapsed_total = (time.time() - stage_start_time) / 60
#
#         # 打印进度
#         if batch_count % 5 == 0 or batch_count == 1:
#             avg_loss = total_loss / batch_count
#             msg = f'[{stage_name}] Batch {batch_count:3d}/{max_batches} | Loss: {loss:.4f} (avg: {avg_loss:.4f})'
#
#             if log_weights:
#                 msg += f' | w_n={stats["w_noisy_mean"]:.3f} w_e={stats["w_enhanced_mean"]:.3f}'
#
#             msg += f' | Time: {batch_time:.1f}s | Total: {elapsed_total:.1f}min'
#             print(msg)
#
#         # 保存checkpoint
#         if batch_count % config.checkpoint_interval == 0:
#             current_weight_stats = trainer.get_weight_statistics()
#             checkpoint_path = save_checkpoint(
#                 trainer,
#                 total_loss / batch_count,
#                 batch_count,
#                 elapsed_total,
#                 config,
#                 stage_name,
#                 weight_stats=current_weight_stats,
#                 is_final=False
#             )
#             print(f'  💾 Checkpoint saved: {checkpoint_path.name}')
#
#     # 子阶段完成
#     avg_loss = total_loss / max_batches
#     print(f'\n✅ Sub-stage "{stage_name}" completed')
#     print(f'   Average loss: {avg_loss:.4f}')
#     print(f'   Batches trained: {max_batches}')
#
#     # 保存子阶段最终模型
#     final_weight_stats = trainer.get_weight_statistics()
#     final_path = save_checkpoint(
#         trainer,
#         avg_loss,
#         max_batches,
#         (time.time() - stage_start_time) / 60,
#         config,
#         stage_name,
#         weight_stats=final_weight_stats,
#         is_final=True
#     )
#     print(f'💾 Sub-stage final model saved: {final_path}')
#
#     return loss_history, weight_history
#
#
# def main():
#     config = ConfigStage2()
#     set_seed(config.seed)
#
#     print('=' * 80)
#     print('🛡️ CONFIDENCE STAGE 2: SAFE SPEECH OPTIMIZATION')
#     print('=' * 80)
#     print('Strategy: Two-phase progressive training (SIMPLIFIED)')
#     print(f'Base model: {config.stage1_checkpoint}')
#     print('=' * 80)
#     print('Phase 2.1: Freeze MLP + Train Confidence Head (75 batches)')
#     print('  - Learning rate: 1e-3')
#     print('  - Dataset: Standard distribution (33% each noise type)')
#     print('  - Goal: Adapt confidence network without forgetting')
#     print('')
#     print('Phase 2.2: Unfreeze All + Fine-tune (175 batches)')
#     print('  - Learning rate: 1e-4 (very small!)')
#     print('  - Dataset: Standard distribution')
#     print('  - Goal: Gentle full-network adjustment')
#     print('=' * 80)
#     print(f'Device: {config.device}')
#     print(f'Random seed: {config.seed}')
#     print(f'SNR range: {config.snr_range} (full coverage)')
#     print(f'Total batches: 250 (75 + 175)')
#     print(f'Expected time: ~3-3.5 hours')
#     print('=' * 80)
#
#     # [1/4] 加载模型
#     print('\n[1/4] Loading models and Stage 1 checkpoint...')
#     speaker_model = SpeakerEmbeddingExtractor('ecapa')
#     enhancer = SpeechEnhancer()
#
#     # 加载Stage 1模型
#     stage1_checkpoint = torch.load(config.stage1_checkpoint, map_location=config.device, weights_only=False)
#
#     trainer = ConfidenceSupConTrainer(
#         embedding_dim=192,
#         device=config.device,
#         temperature=config.temperature,
#         confidence_hidden_dim=config.confidence_hidden_dim
#     )
#
#     trainer.mlp.load_state_dict(stage1_checkpoint['model_state_dict'])
#
#     print(f'✅ Loaded Stage 1 checkpoint')
#     print(f'   - Stage 1 loss: {stage1_checkpoint["loss"]:.4f}')
#     print(f'   - Stage 1 batches: {stage1_checkpoint["batch"]}')
#
#     # [2/4] 准备数据集
#     print('\n[2/4] Preparing dataset (standard distribution)...')
#     speaker_id_map = {}
#
#     # ✅ 直接使用原始数据集，不做任何修改
#     print('\n📦 Dataset configuration:')
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
#     print(f'✅ Dataset loaded: {len(train_dataset)} samples')
#     print(f'   - Using standard MUSAN distribution (33% each)')
#     print(f'   - Total batches (2.1 + 2.2): 250')
#
#     # [3/4] 训练阶段2.1
#     print('\n[3/4] Training Sub-stage 2.1...')
#     stage_start_time = time.time()
#     loss_history_2_1, weight_history_2_1 = train_substage(
#         trainer, speaker_model, enhancer, train_loader,
#         config, config.stage_2_1_config, speaker_id_map, stage_start_time
#     )
#
#     # [4/4] 训练阶段2.2
#     print('\n[4/4] Training Sub-stage 2.2...')
#     loss_history_2_2, weight_history_2_2 = train_substage(
#         trainer, speaker_model, enhancer, train_loader,
#         config, config.stage_2_2_config, speaker_id_map, stage_start_time
#     )
#
#     # === 全部完成 ===
#     total_time = (time.time() - stage_start_time) / 60
#
#     print('\n' + '=' * 80)
#     print('🎉 STAGE 2 TRAINING COMPLETED')
#     print('=' * 80)
#     print(f'⏱️  Total time: {total_time:.1f} minutes')
#     print(f'📊 Sub-stage 2.1: {len(loss_history_2_1)} batches')
#     print(f'📊 Sub-stage 2.2: {len(loss_history_2_2)} batches')
#     print(f'👥 Total unique speakers: {len(speaker_id_map)}')
#     print('=' * 80)
#
#     # 保存最终模型
#     final_checkpoint_dir = Path(config.checkpoint_dir)
#     final_checkpoint_dir.mkdir(exist_ok=True)
#
#     final_path = final_checkpoint_dir / 'final_model_stage2_complete.pth'
#     final_weight_stats = trainer.get_weight_statistics()
#
#     torch.save({
#         'model_state_dict': trainer.mlp.state_dict(),
#         'optimizer_state_dict': trainer.optimizer.state_dict(),
#         'stage': 'stage2_complete',
#         'config': {
#             'model_type': 'Confidence Stage 2 Complete',
#             'stage1_checkpoint': config.stage1_checkpoint,
#             'substage_2_1': config.stage_2_1_config,
#             'substage_2_2': config.stage_2_2_config,
#             'total_batches': len(loss_history_2_1) + len(loss_history_2_2),
#             'training_minutes': total_time
#         },
#         'weight_stats': final_weight_stats,
#         'loss_history': {
#             'stage_2_1': loss_history_2_1,
#             'stage_2_2': loss_history_2_2
#         },
#         'weight_history': {
#             'stage_2_1': weight_history_2_1,
#             'stage_2_2': weight_history_2_2
#         }
#     }, final_path)
#
#     print(f'\n💾 Final Stage 2 model saved: {final_path}')
#
#     # 保存详细报告
#     report_path = final_checkpoint_dir / 'stage2_training_report.txt'
#     with open(report_path, 'w') as f:
#         f.write('=' * 80 + '\n')
#         f.write('CONFIDENCE STAGE 2 TRAINING REPORT\n')
#         f.write('=' * 80 + '\n\n')
#
#         f.write('TRAINING STRATEGY:\n')
#         f.write('- Two-phase progressive training\n')
#         f.write('- Phase 2.1: Freeze MLP + Train Confidence (75 batches)\n')
#         f.write('- Phase 2.2: Fine-tune all (175 batches)\n')
#         f.write('- Standard MUSAN distribution (33% each)\n\n')
#
#         f.write('STAGE 2.1 RESULTS:\n')
#         f.write(f'  Batches: {len(loss_history_2_1)}\n')
#         f.write(f'  Avg loss: {np.mean(loss_history_2_1):.4f}\n')
#         f.write(f'  Min loss: {np.min(loss_history_2_1):.4f}\n')
#         f.write(f'  Max loss: {np.max(loss_history_2_1):.4f}\n\n')
#
#         f.write('STAGE 2.2 RESULTS:\n')
#         f.write(f'  Batches: {len(loss_history_2_2)}\n')
#         f.write(f'  Avg loss: {np.mean(loss_history_2_2):.4f}\n')
#         f.write(f'  Min loss: {np.min(loss_history_2_2):.4f}\n')
#         f.write(f'  Max loss: {np.max(loss_history_2_2):.4f}\n\n')
#
#         f.write('FINAL WEIGHT STATISTICS:\n')
#         if final_weight_stats:
#             f.write(
#                 f'  Noisy weight: {final_weight_stats["w_noisy"]["mean"]:.4f} ± {final_weight_stats["w_noisy"]["std"]:.4f}\n')
#             f.write(
#                 f'  Enhanced weight: {final_weight_stats["w_enhanced"]["mean"]:.4f} ± {final_weight_stats["w_enhanced"]["std"]:.4f}\n\n')
#
#         f.write(f'Total time: {total_time:.1f} minutes\n')
#         f.write(f'Total speakers: {len(speaker_id_map)}\n')
#
#     print(f'📄 Training report saved: {report_path}')
#
#     # 最终建议
#     print('\n' + '=' * 80)
#     print('🎯 NEXT STEPS')
#     print('=' * 80)
#     print('1. Test Stage 2 model on all 15 conditions:')
#     print(f'   python evaluate_confidence_paper_15conditions.py \\')
#     print(f'       --checkpoint {final_path} \\')
#     print(f'       --output_dir results_confidence_stage2 \\')
#     print(f'       --num_pairs 1000')
#     print('')
#     print('2. Compare with Stage 1:')
#     print('   python compare_stage1_stage2.py')
#     print('')
#     print('3. If Stage 2 is worse than Stage 1:')
#     print('   - Use Stage 1 model for paper')
#     print('   - Mention Stage 2 attempt in discussion')
#     print('')
#     print('4. If Stage 2 improves Speech without hurting Noise/Music:')
#     print('   - Great! Use Stage 2 model')
#     print('   - Emphasize adaptive training strategy')
#     print('=' * 80)
#
#
# if __name__ == '__main__':
#     main()
"""
阶段2：安全的Speech性能优化（两阶段渐进训练）- 完整修复版

🛡️ 安全机制：
1. 阶段2.1：冻结MLP主干，只训练Confidence Head（75 batches）
2. 阶段2.2：解冻全部，极小lr微调（175 batches）
3. 使用标准数据集分布（不修改MUSAN比例）
4. 通过增加batch数来获得足够训练
5. 每25 batch保存checkpoint，可随时回退

✅ 修复：
- 通用的参数冻结机制（根据名称判断）
- 正确的import路径处理

用法：
    python train_confidence_stage2_safe.py
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

# 添加scripts路径
sys.path.append('scripts')

# 导入模型
from models_confidence import ConfidenceSupConTrainer
from speaker_embedding import SpeakerEmbeddingExtractor
from speech_enhancer import SpeechEnhancer
from dataset import VoxCelebMusanDataset


class ConfigStage2:
    """阶段2训练配置 - 安全优化"""
    # 数据路径
    voxceleb_dir = 'data/voxceleb1'
    musan_dir = 'data/musan'

    # ===== 阶段1 checkpoint =====
    stage1_checkpoint = 'checkpoints_confidence_400batch/final_model_confidence.pth'

    # ===== 两个子阶段的参数 =====
    # 子阶段2.1：冻结MLP主干
    stage_2_1_config = {
        'name': 'freeze_mlp',
        'batches': 75,
        'learning_rate': 1e-3,
        'freeze_mlp': True,
        'description': 'Train only Confidence Head with frozen MLP'
    }

    # 子阶段2.2：解冻全部，精细调整
    stage_2_2_config = {
        'name': 'finetune_all',
        'batches': 175,
        'learning_rate': 1e-4,
        'freeze_mlp': False,
        'description': 'Fine-tune all parameters with very small lr'
    }

    # ===== 通用参数 =====
    batch_size = 32
    snr_range = (-20, 0)
    temperature = 0.07
    confidence_hidden_dim = 64

    # 输出目录
    checkpoint_dir = 'checkpoints_confidence_stage2'

    # 控制参数
    checkpoint_interval = 25
    log_weights_interval = 5

    # 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
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

    return padded_batch


def extract_and_normalize_embeddings(audio_batch, speaker_model, enhancer, device, speaker_id_map):
    """提取并归一化embeddings"""
    batch_size = len(audio_batch['anchor_noisy'])

    all_noisy_embs = []
    all_enhanced_embs = []
    all_labels = []

    for i in range(batch_size):
        speaker_id_str = audio_batch['speaker_id'][i]

        if speaker_id_str not in speaker_id_map:
            speaker_id_map[speaker_id_str] = len(speaker_id_map)
        speaker_id_int = speaker_id_map[speaker_id_str]

        # === Anchor ===
        anchor_noisy = audio_batch['anchor_noisy'][i]
        anchor_enhanced = enhancer.enhance_audio(anchor_noisy, sr=16000)

        emb_noisy = speaker_model.extract_embedding(audio_tensor=anchor_noisy)
        emb_enhanced = speaker_model.extract_embedding(audio_tensor=anchor_enhanced)

        emb_noisy = F.normalize(emb_noisy.squeeze().unsqueeze(0), p=2, dim=1).squeeze(0)
        emb_enhanced = F.normalize(emb_enhanced.squeeze().unsqueeze(0), p=2, dim=1).squeeze(0)

        all_noisy_embs.append(emb_noisy)
        all_enhanced_embs.append(emb_enhanced)
        all_labels.append(speaker_id_int)

        # === Positive ===
        pos_noisy = audio_batch['positive_noisy'][i]
        pos_enhanced = enhancer.enhance_audio(pos_noisy, sr=16000)

        emb_noisy = speaker_model.extract_embedding(audio_tensor=pos_noisy)
        emb_enhanced = speaker_model.extract_embedding(audio_tensor=pos_enhanced)

        emb_noisy = F.normalize(emb_noisy.squeeze().unsqueeze(0), p=2, dim=1).squeeze(0)
        emb_enhanced = F.normalize(emb_enhanced.squeeze().unsqueeze(0), p=2, dim=1).squeeze(0)

        all_noisy_embs.append(emb_noisy)
        all_enhanced_embs.append(emb_enhanced)
        all_labels.append(speaker_id_int)

        # === Negative ===
        neg_noisy = audio_batch['negative_noisy'][i]
        neg_enhanced = enhancer.enhance_audio(neg_noisy, sr=16000)

        emb_noisy = speaker_model.extract_embedding(audio_tensor=neg_noisy)
        emb_enhanced = speaker_model.extract_embedding(audio_tensor=neg_enhanced)

        emb_noisy = F.normalize(emb_noisy.squeeze().unsqueeze(0), p=2, dim=1).squeeze(0)
        emb_enhanced = F.normalize(emb_enhanced.squeeze().unsqueeze(0), p=2, dim=1).squeeze(0)

        all_noisy_embs.append(emb_noisy)
        all_enhanced_embs.append(emb_enhanced)
        all_labels.append(speaker_id_int + 100000)

    noisy_embeddings = torch.stack(all_noisy_embs)
    enhanced_embeddings = torch.stack(all_enhanced_embs)
    labels = torch.tensor(all_labels, dtype=torch.long)

    return noisy_embeddings, enhanced_embeddings, labels


def freeze_mlp_backbone(trainer):
    """
    冻结MLP主干，只保留Confidence Network可训练

    ✅ 通用方法：根据参数名称判断
    - 包含'confidence'的参数 → 可训练
    - 其他所有参数 → 冻结
    """
    print('\n🔍 Analyzing model parameters...')

    frozen_params = []
    trainable_params = []

    for name, param in trainer.mlp.named_parameters():
        # 只保留confidence相关的参数可训练
        if 'confidence' in name.lower():
            param.requires_grad = True
            trainable_params.append(name)
        else:
            param.requires_grad = False
            frozen_params.append(name)

    print(f'\n🧊 Frozen MLP backbone')
    print(f'   - Frozen parameters: {len(frozen_params)}')
    if len(frozen_params) > 0:
        print('   Frozen layers:')
        for name in frozen_params[:5]:  # 只显示前5个
            print(f'     ❄️  {name}')
        if len(frozen_params) > 5:
            print(f'     ... and {len(frozen_params) - 5} more')

    print(f'\n   - Trainable parameters: {len(trainable_params)}')
    if len(trainable_params) > 0:
        print('   Trainable layers:')
        for name in trainable_params:
            print(f'     ✅ {name}')

    # 统计总参数量
    trainable_count = sum(p.numel() for p in trainer.mlp.parameters() if p.requires_grad)
    total_count = sum(p.numel() for p in trainer.mlp.parameters())

    print(
        f'\n   - Trainable param count: {trainable_count:,} / {total_count:,} ({trainable_count / total_count * 100:.1f}%)')

    if trainable_count == 0:
        print('\n⚠️  WARNING: No trainable parameters found!')
        print('   This might cause training to fail.')
        print('   Model structure might be different than expected.')

    return trainable_count > 0  # 返回是否有可训练参数


def unfreeze_all(trainer):
    """解冻所有参数"""
    for param in trainer.mlp.parameters():
        param.requires_grad = True

    print('\n🔓 Unfrozen all parameters')
    print('   - Training entire network')

    trainable_params = sum(p.numel() for p in trainer.mlp.parameters() if p.requires_grad)
    print(f'   - Trainable params: {trainable_params:,}')


def save_checkpoint(trainer, loss, batch_idx, total_time, config, stage_name, weight_stats=None, is_final=False):
    """保存checkpoint"""
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)

    if is_final:
        filename = f'final_model_stage2_{stage_name}.pth'
    else:
        filename = f'checkpoint_stage2_{stage_name}_batch_{batch_idx}.pth'

    checkpoint_path = checkpoint_dir / filename

    checkpoint_data = {
        'model_state_dict': trainer.mlp.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'loss': loss,
        'batch': batch_idx,
        'training_minutes': total_time,
        'stage': f'stage2_{stage_name}',
        'config': {
            'model_type': 'Confidence Stage 2',
            'batch_size': config.batch_size,
            'temperature': config.temperature,
            'confidence_hidden_dim': config.confidence_hidden_dim,
            'snr_range': config.snr_range,
            'seed': config.seed,
            'stage1_checkpoint': config.stage1_checkpoint
        }
    }

    if weight_stats:
        checkpoint_data['weight_stats'] = weight_stats

    torch.save(checkpoint_data, checkpoint_path)

    return checkpoint_path


def train_substage(trainer, speaker_model, enhancer, train_loader,
                   config, substage_config, speaker_id_map, stage_start_time):
    """训练一个子阶段"""
    stage_name = substage_config['name']
    max_batches = substage_config['batches']
    lr = substage_config['learning_rate']

    print('\n' + '=' * 80)
    print(f'🎯 SUB-STAGE: {substage_config["description"]}')
    print('=' * 80)
    print(f'Batches: {max_batches}')
    print(f'Learning rate: {lr}')
    print(f'Freeze MLP: {substage_config["freeze_mlp"]}')
    print('=' * 80)

    # 设置冻结状态
    has_trainable = True
    if substage_config['freeze_mlp']:
        has_trainable = freeze_mlp_backbone(trainer)
        if not has_trainable:
            print('\n❌ ERROR: No trainable parameters after freezing!')
            print('Skipping this stage...')
            return [], []
    else:
        unfreeze_all(trainer)

    # 更新优化器（使用新的学习率）
    trainable_params = filter(lambda p: p.requires_grad, trainer.mlp.parameters())
    trainer.optimizer = torch.optim.AdamW(
        trainable_params,
        lr=lr,
        weight_decay=1e-4
    )

    print(f'\n✅ Optimizer updated with lr={lr}')

    # 训练循环
    total_loss = 0.0
    loss_history = []
    weight_history = []

    trainer.mlp.train()

    batch_count = 0
    data_iterator = iter(train_loader)

    while batch_count < max_batches:
        try:
            audio_batch = next(data_iterator)
        except StopIteration:
            data_iterator = iter(train_loader)
            audio_batch = next(data_iterator)

        batch_count += 1
        batch_start = time.time()

        # 提取embeddings
        noisy_embs, enhanced_embs, labels = extract_and_normalize_embeddings(
            audio_batch, speaker_model, enhancer, config.device, speaker_id_map
        )

        noisy_embs = noisy_embs.to(config.device)
        enhanced_embs = enhanced_embs.to(config.device)
        labels = labels.to(config.device)

        # 训练步骤
        log_weights = (batch_count % config.log_weights_interval == 0) or (batch_count == 1)
        loss, stats = trainer.train_step(
            noisy_embs, enhanced_embs, labels, log_weights=log_weights
        )

        # 记录
        loss_history.append(loss)
        total_loss += loss

        if log_weights:
            weight_history.append({
                'batch': batch_count,
                'w_noisy': stats['w_noisy_mean'],
                'w_enhanced': stats['w_enhanced_mean'],
                'w_noisy_std': stats['w_noisy_std'],
                'w_enhanced_std': stats['w_enhanced_std']
            })

        batch_time = time.time() - batch_start
        elapsed_total = (time.time() - stage_start_time) / 60

        # 打印进度
        if batch_count % 5 == 0 or batch_count == 1:
            avg_loss = total_loss / batch_count
            msg = f'[{stage_name}] Batch {batch_count:3d}/{max_batches} | Loss: {loss:.4f} (avg: {avg_loss:.4f})'

            if log_weights:
                msg += f' | w_n={stats["w_noisy_mean"]:.3f} w_e={stats["w_enhanced_mean"]:.3f}'

            msg += f' | Time: {batch_time:.1f}s | Total: {elapsed_total:.1f}min'
            print(msg)

        # 保存checkpoint
        if batch_count % config.checkpoint_interval == 0:
            current_weight_stats = trainer.get_weight_statistics()
            checkpoint_path = save_checkpoint(
                trainer,
                total_loss / batch_count,
                batch_count,
                elapsed_total,
                config,
                stage_name,
                weight_stats=current_weight_stats,
                is_final=False
            )
            print(f'  💾 Checkpoint saved: {checkpoint_path.name}')

    # 子阶段完成
    avg_loss = total_loss / max_batches
    print(f'\n✅ Sub-stage "{stage_name}" completed')
    print(f'   Average loss: {avg_loss:.4f}')
    print(f'   Batches trained: {max_batches}')

    # 保存子阶段最终模型
    final_weight_stats = trainer.get_weight_statistics()
    final_path = save_checkpoint(
        trainer,
        avg_loss,
        max_batches,
        (time.time() - stage_start_time) / 60,
        config,
        stage_name,
        weight_stats=final_weight_stats,
        is_final=True
    )
    print(f'💾 Sub-stage final model saved: {final_path}')

    return loss_history, weight_history


def main():
    config = ConfigStage2()
    set_seed(config.seed)

    print('=' * 80)
    print('🛡️ CONFIDENCE STAGE 2: SAFE SPEECH OPTIMIZATION')
    print('=' * 80)
    print('Strategy: Two-phase progressive training (SIMPLIFIED)')
    print(f'Base model: {config.stage1_checkpoint}')
    print('=' * 80)
    print('Phase 2.1: Freeze MLP + Train Confidence Head (75 batches)')
    print('  - Learning rate: 1e-3')
    print('  - Dataset: Standard distribution (33% each noise type)')
    print('  - Goal: Adapt confidence network without forgetting')
    print('')
    print('Phase 2.2: Unfreeze All + Fine-tune (175 batches)')
    print('  - Learning rate: 1e-4 (very small!)')
    print('  - Dataset: Standard distribution')
    print('  - Goal: Gentle full-network adjustment')
    print('=' * 80)
    print(f'Device: {config.device}')
    print(f'Random seed: {config.seed}')
    print(f'SNR range: {config.snr_range} (full coverage)')
    print(f'Total batches: 250 (75 + 175)')
    print(f'Expected time: ~3-3.5 hours')
    print('=' * 80)

    # [1/4] 加载模型
    print('\n[1/4] Loading models and Stage 1 checkpoint...')
    speaker_model = SpeakerEmbeddingExtractor('ecapa')
    enhancer = SpeechEnhancer()

    # 加载Stage 1模型
    print(f'\nLoading Stage 1 checkpoint: {config.stage1_checkpoint}')
    stage1_checkpoint = torch.load(config.stage1_checkpoint, map_location=config.device, weights_only=False)

    trainer = ConfidenceSupConTrainer(
        embedding_dim=192,
        device=config.device,
        temperature=config.temperature,
        confidence_hidden_dim=config.confidence_hidden_dim
    )

    trainer.mlp.load_state_dict(stage1_checkpoint['model_state_dict'])

    print(f'\n✅ Loaded Stage 1 checkpoint')
    print(f'   - Stage 1 loss: {stage1_checkpoint["loss"]:.4f}')
    print(f'   - Stage 1 batches: {stage1_checkpoint["batch"]}')

    # 打印模型结构（用于调试）
    print(f'\n📊 Model structure:')
    print(f'   Total parameters: {sum(p.numel() for p in trainer.mlp.parameters()):,}')
    print(f'   Model type: {type(trainer.mlp).__name__}')

    # [2/4] 准备数据集
    print('\n[2/4] Preparing dataset (standard distribution)...')
    speaker_id_map = {}

    print('\n📦 Dataset configuration:')
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

    print(f'\n✅ Dataset loaded: {len(train_dataset)} samples')
    print(f'   - Using standard MUSAN distribution (33% each)')
    print(f'   - Total batches (2.1 + 2.2): 250')

    # [3/4] 训练阶段2.1
    print('\n[3/4] Training Sub-stage 2.1...')
    stage_start_time = time.time()
    loss_history_2_1, weight_history_2_1 = train_substage(
        trainer, speaker_model, enhancer, train_loader,
        config, config.stage_2_1_config, speaker_id_map, stage_start_time
    )

    # [4/4] 训练阶段2.2
    print('\n[4/4] Training Sub-stage 2.2...')
    loss_history_2_2, weight_history_2_2 = train_substage(
        trainer, speaker_model, enhancer, train_loader,
        config, config.stage_2_2_config, speaker_id_map, stage_start_time
    )

    # === 全部完成 ===
    total_time = (time.time() - stage_start_time) / 60

    print('\n' + '=' * 80)
    print('🎉 STAGE 2 TRAINING COMPLETED')
    print('=' * 80)
    print(f'⏱️  Total time: {total_time:.1f} minutes')
    print(f'📊 Sub-stage 2.1: {len(loss_history_2_1)} batches')
    print(f'📊 Sub-stage 2.2: {len(loss_history_2_2)} batches')
    print(f'👥 Total unique speakers: {len(speaker_id_map)}')
    print('=' * 80)

    # 保存最终模型
    final_checkpoint_dir = Path(config.checkpoint_dir)
    final_checkpoint_dir.mkdir(exist_ok=True)

    final_path = final_checkpoint_dir / 'final_model_stage2_complete.pth'
    final_weight_stats = trainer.get_weight_statistics()

    torch.save({
        'model_state_dict': trainer.mlp.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'stage': 'stage2_complete',
        'config': {
            'model_type': 'Confidence Stage 2 Complete',
            'stage1_checkpoint': config.stage1_checkpoint,
            'substage_2_1': config.stage_2_1_config,
            'substage_2_2': config.stage_2_2_config,
            'total_batches': len(loss_history_2_1) + len(loss_history_2_2),
            'training_minutes': total_time
        },
        'weight_stats': final_weight_stats,
        'loss_history': {
            'stage_2_1': loss_history_2_1,
            'stage_2_2': loss_history_2_2
        },
        'weight_history': {
            'stage_2_1': weight_history_2_1,
            'stage_2_2': weight_history_2_2
        }
    }, final_path)

    print(f'\n💾 Final Stage 2 model saved: {final_path}')

    # 保存详细报告
    report_path = final_checkpoint_dir / 'stage2_training_report.txt'
    with open(report_path, 'w') as f:
        f.write('=' * 80 + '\n')
        f.write('CONFIDENCE STAGE 2 TRAINING REPORT\n')
        f.write('=' * 80 + '\n\n')

        f.write('TRAINING STRATEGY:\n')
        f.write('- Two-phase progressive training\n')
        f.write('- Phase 2.1: Freeze MLP + Train Confidence (75 batches)\n')
        f.write('- Phase 2.2: Fine-tune all (175 batches)\n')
        f.write('- Standard MUSAN distribution (33% each)\n\n')

        if len(loss_history_2_1) > 0:
            f.write('STAGE 2.1 RESULTS:\n')
            f.write(f'  Batches: {len(loss_history_2_1)}\n')
            f.write(f'  Avg loss: {np.mean(loss_history_2_1):.4f}\n')
            f.write(f'  Min loss: {np.min(loss_history_2_1):.4f}\n')
            f.write(f'  Max loss: {np.max(loss_history_2_1):.4f}\n\n')

        if len(loss_history_2_2) > 0:
            f.write('STAGE 2.2 RESULTS:\n')
            f.write(f'  Batches: {len(loss_history_2_2)}\n')
            f.write(f'  Avg loss: {np.mean(loss_history_2_2):.4f}\n')
            f.write(f'  Min loss: {np.min(loss_history_2_2):.4f}\n')
            f.write(f'  Max loss: {np.max(loss_history_2_2):.4f}\n\n')

        f.write('FINAL WEIGHT STATISTICS:\n')
        if final_weight_stats:
            f.write(
                f'  Noisy weight: {final_weight_stats["w_noisy"]["mean"]:.4f} ± {final_weight_stats["w_noisy"]["std"]:.4f}\n')
            f.write(
                f'  Enhanced weight: {final_weight_stats["w_enhanced"]["mean"]:.4f} ± {final_weight_stats["w_enhanced"]["std"]:.4f}\n\n')

        f.write(f'Total time: {total_time:.1f} minutes\n')
        f.write(f'Total speakers: {len(speaker_id_map)}\n')

    print(f'📄 Training report saved: {report_path}')

    # 最终建议
    print('\n' + '=' * 80)
    print('🎯 NEXT STEPS')
    print('=' * 80)
    print('1. Test Stage 2 model on all 15 conditions:')
    print(f'   python evaluate_confidence_paper_15conditions.py \\')
    print(f'       --checkpoint {final_path} \\')
    print(f'       --output_dir results_confidence_stage2 \\')
    print(f'       --num_pairs 1000')
    print('')
    print('2. Compare with Stage 1:')
    print('   python compare_stage1_stage2.py')
    print('')
    print('3. Decision making:')
    print('   - If Stage 2 improves Speech: Use Stage 2 ✅')
    print('   - If Stage 2 hurts Noise/Music: Use Stage 1 ⚠️')
    print('   - Either way, you have great results!')
    print('=' * 80)


if __name__ == '__main__':
    main()