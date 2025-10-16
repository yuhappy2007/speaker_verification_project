# """
# SupCon模型测试脚本
# - 支持1000 pairs快速测试
# - 与原论文baseline对比
# - 生成对比表格
#
# 用法:
#     python evaluate_supcon.py --checkpoint checkpoints_supcon_200batch/final_model_supcon.pth
# """
#
# import torch
# import torch.nn.functional as F
# import numpy as np
# from pathlib import Path
# import argparse
# from tqdm import tqdm
# import json
#
# # 添加scripts路径
# import sys
#
# sys.path.append('scripts')
#
# from models_supcon import RobustEmbeddingMLP
# from speaker_embedding import SpeakerEmbeddingExtractor
# from speech_enhancer import SpeechEnhancer
# from dataset import VoxCelebMusanDataset
#
# # 原论文baseline结果(从论文Table 1复制)
# PAPER_BASELINE = {
#     'Noise_-15dB': {
#         'Noisy': 34.71,
#         'Enhanced': 32.77,
#         'Triplet(Paper)': 25.21
#     },
#     'Music_-15dB': {
#         'Noisy': 44.46,
#         'Enhanced': 41.86,
#         'Triplet(Paper)': 36.37
#     },
#     'Babble_-15dB': {
#         'Noisy': 46.72,
#         'Enhanced': 47.48,
#         'Triplet(Paper)': 37.21
#     }
# }
#
#
# def load_supcon_model(checkpoint_path, device):
#     """加载SupCon模型"""
#     print(f'Loading SupCon model from: {checkpoint_path}')
#
#     # 创建MLP
#     mlp = RobustEmbeddingMLP(embedding_dim=192).to(device)
#
#     # 加载权重
#     checkpoint = torch.load(checkpoint_path, map_location=device)
#     mlp.load_state_dict(checkpoint['model_state_dict'])
#     mlp.eval()
#
#     print(f"✅ Model loaded")
#     print(f"   - Training loss: {checkpoint['loss']:.4f}")
#     print(f"   - Batch: {checkpoint['batch']}")
#     print(f"   - Temperature: {checkpoint['config']['temperature']}")
#
#     return mlp, checkpoint
#
#
# def extract_supcon_embedding(audio, speaker_model, enhancer, mlp, device):
#     """
#     使用SupCon模型提取robust embedding
#
#     流程:
#     1. 提取noisy embedding
#     2. 语音增强
#     3. 提取enhanced embedding
#     4. 通过MLP融合
#     """
#     with torch.no_grad():
#         # 提取noisy embedding
#         noisy_emb = speaker_model.extract_embedding(audio_tensor=audio)
#         noisy_emb = F.normalize(noisy_emb.squeeze(), p=2, dim=0)
#
#         # 语音增强
#         enhanced_audio = enhancer.enhance_audio(audio, sr=16000)
#
#         # 提取enhanced embedding
#         enhanced_emb = speaker_model.extract_embedding(audio_tensor=enhanced_audio)
#         enhanced_emb = F.normalize(enhanced_emb.squeeze(), p=2, dim=0)
#
#         # 通过MLP融合(会自动L2归一化)
#         noisy_emb = noisy_emb.unsqueeze(0).to(device)
#         enhanced_emb = enhanced_emb.unsqueeze(0).to(device)
#         robust_emb = mlp(noisy_emb, enhanced_emb)
#
#         return robust_emb.squeeze().cpu()
#
#
# def compute_eer(scores, labels):
#     """
#     计算EER(Equal Error Rate)
#
#     参数:
#         scores: 相似度分数列表
#         labels: 标签列表(1=same speaker, 0=different speaker)
#     """
#     # 转换为numpy
#     scores = np.array(scores)
#     labels = np.array(labels)
#
#     # 按分数排序
#     sorted_indices = np.argsort(scores)
#     scores = scores[sorted_indices]
#     labels = labels[sorted_indices]
#
#     # 计算FAR和FRR
#     n_positive = np.sum(labels == 1)
#     n_negative = np.sum(labels == 0)
#
#     best_eer = 1.0
#     best_threshold = 0.0
#
#     for i, threshold in enumerate(scores):
#         # False Accept Rate: 负样本中得分>=threshold的比例
#         far = np.sum((scores >= threshold) & (labels == 0)) / n_negative
#
#         # False Reject Rate: 正样本中得分<threshold的比例
#         frr = np.sum((scores < threshold) & (labels == 1)) / n_positive
#
#         # EER是FAR和FRR相等的点
#         eer = (far + frr) / 2
#
#         if abs(far - frr) < abs(best_eer - 0.5):
#             best_eer = eer
#             best_threshold = threshold
#
#     return best_eer * 100, best_threshold  # 返回百分比
#
#
# def evaluate_supcon(mlp, speaker_model, enhancer, test_dataset, num_pairs=1000, device='cuda'):
#     """
#     评估SupCon模型
#
#     参数:
#         num_pairs: 测试pair数量(默认1000用于快速测试)
#
#     返回:
#         eer: Equal Error Rate (%)
#     """
#     print(f'\n📊 Evaluating SupCon model on {num_pairs} pairs...')
#
#     mlp.eval()
#     scores = []
#     labels = []
#
#     # 生成测试pairs
#     test_pairs = []
#     for i in range(num_pairs // 2):
#         # 正样本对(同一说话人)
#         idx1 = np.random.randint(len(test_dataset))
#         sample1 = test_dataset[idx1]
#
#         # 找同一说话人的另一个样本
#         same_speaker_samples = [j for j, s in enumerate(test_dataset.samples)
#                                 if s['speaker_id'] == sample1['speaker_id'] and j != idx1]
#         if same_speaker_samples:
#             idx2 = np.random.choice(same_speaker_samples)
#             test_pairs.append((idx1, idx2, 1))
#
#         # 负样本对(不同说话人)
#         idx3 = np.random.randint(len(test_dataset))
#         sample3 = test_dataset[idx3]
#
#         # 找不同说话人
#         diff_speaker_samples = [j for j, s in enumerate(test_dataset.samples)
#                                 if s['speaker_id'] != sample3['speaker_id']]
#         if diff_speaker_samples:
#             idx4 = np.random.choice(diff_speaker_samples)
#             test_pairs.append((idx3, idx4, 0))
#
#     # 评估每个pair
#     for idx1, idx2, label in tqdm(test_pairs, desc='Testing'):
#         sample1 = test_dataset[idx1]
#         sample2 = test_dataset[idx2]
#
#         # 提取robust embeddings
#         emb1 = extract_supcon_embedding(
#             sample1['anchor_noisy'], speaker_model, enhancer, mlp, device
#         )
#         emb2 = extract_supcon_embedding(
#             sample2['anchor_noisy'], speaker_model, enhancer, mlp, device
#         )
#
#         # 计算余弦相似度
#         score = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
#
#         scores.append(score)
#         labels.append(label)
#
#     # 计算EER
#     eer, threshold = compute_eer(scores, labels)
#
#     print(f'✅ Evaluation complete')
#     print(f'   - EER: {eer:.2f}%')
#     print(f'   - Threshold: {threshold:.4f}')
#     print(f'   - Avg score (same): {np.mean([s for s, l in zip(scores, labels) if l == 1]):.4f}')
#     print(f'   - Avg score (diff): {np.mean([s for s, l in zip(scores, labels) if l == 0]):.4f}')
#
#     return eer, threshold
#
#
# def print_comparison_table(supcon_eer, noise_type='-15dB'):
#     """
#     打印与原论文baseline的对比表格
#
#     参数:
#         supcon_eer: SupCon模型的EER
#         noise_type: 噪声类型(用于从baseline中查找)
#     """
#     print('\n' + '=' * 80)
#     print('📊 COMPARISON WITH PAPER BASELINE')
#     print('=' * 80)
#
#     # 获取对应噪声类型的baseline
#     baseline_key = f'Noise_{noise_type}'
#     if baseline_key in PAPER_BASELINE:
#         baseline = PAPER_BASELINE[baseline_key]
#
#         print(f'\n{"Method":<25} {"EER (%)":<15} {"vs Triplet":<15} {"vs Noisy":<15}')
#         print('-' * 80)
#
#         # Noisy baseline
#         print(f'{"Noisy (Paper)":<25} {baseline["Noisy"]:<15.2f} {"":<15} {"":<15}')
#
#         # Enhanced baseline
#         print(f'{"Enhanced (Paper)":<25} {baseline["Enhanced"]:<15.2f} {"":<15} {"":<15}')
#
#         # Triplet Loss baseline
#         triplet_eer = baseline['Triplet(Paper)']
#         print(f'{"Triplet Loss (Paper)":<25} {triplet_eer:<15.2f} {"":<15} {"":<15}')
#
#         # SupCon (你的结果)
#         vs_triplet = supcon_eer - triplet_eer
#         vs_noisy = supcon_eer - baseline['Noisy']
#
#         status = '✅' if supcon_eer < triplet_eer else '⚠️'
#         print(f'{f"SupCon (Yours) {status}":<25} {supcon_eer:<15.2f} {vs_triplet:+.2f} {"":<6} {vs_noisy:+.2f}')
#
#         print('-' * 80)
#
#         # 分析
#         print('\n🔍 ANALYSIS:')
#         if supcon_eer < triplet_eer:
#             improvement = (triplet_eer - supcon_eer) / triplet_eer * 100
#             print(f'✅ SupCon is BETTER than Triplet Loss by {improvement:.1f}%')
#             print(f'   - Absolute improvement: {triplet_eer - supcon_eer:.2f}% EER')
#         elif supcon_eer > triplet_eer:
#             degradation = (supcon_eer - triplet_eer) / triplet_eer * 100
#             print(f'⚠️  SupCon is worse than Triplet Loss by {degradation:.1f}%')
#             print(f'   - Consider: adjusting temperature, more training epochs')
#         else:
#             print(f'➖ SupCon performs similarly to Triplet Loss')
#
#         if supcon_eer < baseline['Noisy']:
#             improvement_noisy = (baseline['Noisy'] - supcon_eer) / baseline['Noisy'] * 100
#             print(f'✅ SupCon improves Noisy baseline by {improvement_noisy:.1f}%')
#
#     print('=' * 80)
#
#
# def main():
#     parser = argparse.ArgumentParser(description='Evaluate SupCon model')
#     parser.add_argument('--checkpoint', type=str,
#                         default='checkpoints_supcon_200batch/final_model_supcon.pth',
#                         help='Path to SupCon checkpoint')
#     parser.add_argument('--num_pairs', type=int, default=1000,
#                         help='Number of test pairs (default: 1000 for fast testing)')
#     parser.add_argument('--noise_type', type=str, default='-15dB',
#                         help='Noise type for comparison (default: -15dB)')
#     parser.add_argument('--voxceleb_dir', type=str, default='data/voxceleb1',
#                         help='VoxCeleb dataset directory')
#     parser.add_argument('--musan_dir', type=str, default='data/musan',
#                         help='MUSAN dataset directory')
#     args = parser.parse_args()
#
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#
#     print('=' * 80)
#     print('🧪 SUPCON MODEL EVALUATION')
#     print('=' * 80)
#     print(f'Checkpoint: {args.checkpoint}')
#     print(f'Test pairs: {args.num_pairs}')
#     print(f'Device: {device}')
#     print('=' * 80)
#
#     # 加载模型
#     print('\n[1/4] Loading models...')
#     mlp, checkpoint = load_supcon_model(args.checkpoint, device)
#     speaker_model = SpeakerEmbeddingExtractor('ecapa')
#     enhancer = SpeechEnhancer()
#
#     # 加载测试数据
#     print('\n[2/4] Loading test dataset...')
#     test_dataset = VoxCelebMusanDataset(
#         args.voxceleb_dir,
#         args.musan_dir,
#         split='test',
#         snr_range=(-15, -15)  # 固定-15dB用于与论文对比
#     )
#     print(f'✅ Test samples: {len(test_dataset)}')
#
#     # 评估
#     print('\n[3/4] Evaluating...')
#     supcon_eer, threshold = evaluate_supcon(
#         mlp, speaker_model, enhancer, test_dataset,
#         num_pairs=args.num_pairs, device=device
#     )
#
#     # 对比
#     print('\n[4/4] Comparing with baseline...')
#     print_comparison_table(supcon_eer, noise_type=args.noise_type)
#
#     # 保存结果
#     results_file = Path(args.checkpoint).parent / 'evaluation_results.json'
#     results = {
#         'supcon_eer': supcon_eer,
#         'threshold': threshold,
#         'num_pairs': args.num_pairs,
#         'paper_baseline': PAPER_BASELINE,
#         'checkpoint': str(args.checkpoint)
#     }
#
#     with open(results_file, 'w') as f:
#         json.dump(results, f, indent=2)
#
#     print(f'\n💾 Results saved: {results_file}')
#     print('\n✅ Evaluation complete!')
#
#
# if __name__ == '__main__':
#     main()
"""
SupCon模型测试脚本 (修复版)
- 支持1000 pairs快速测试
- 与原论文baseline对比
- 生成对比表格

用法:
    python evaluate_supcon.py --checkpoint checkpoints_supcon_200batch/final_model_supcon.pth
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import json

# 添加scripts路径
import sys

sys.path.append('scripts')

from models_supcon import RobustEmbeddingMLP
from speaker_embedding import SpeakerEmbeddingExtractor
from speech_enhancer import SpeechEnhancer
from dataset import VoxCelebMusanDataset

# 原论文baseline结果(从论文Table 1复制)
PAPER_BASELINE = {
    'Noise_-15dB': {
        'Noisy': 34.71,
        'Enhanced': 32.77,
        'Triplet(Paper)': 25.21
    },
    'Music_-15dB': {
        'Noisy': 44.46,
        'Enhanced': 41.86,
        'Triplet(Paper)': 36.37
    },
    'Babble_-15dB': {
        'Noisy': 46.72,
        'Enhanced': 47.48,
        'Triplet(Paper)': 37.21
    }
}


def load_supcon_model(checkpoint_path, device):
    """加载SupCon模型"""
    print(f'Loading SupCon model from: {checkpoint_path}')

    # 创建MLP
    mlp = RobustEmbeddingMLP(embedding_dim=192).to(device)

    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    mlp.load_state_dict(checkpoint['model_state_dict'])
    mlp.eval()

    print(f"✅ Model loaded")
    print(f"   - Training loss: {checkpoint['loss']:.4f}")
    print(f"   - Batch: {checkpoint['batch']}")
    print(f"   - Temperature: {checkpoint['config']['temperature']}")

    return mlp, checkpoint


def extract_supcon_embedding(audio, speaker_model, enhancer, mlp, device):
    """
    使用SupCon模型提取robust embedding

    流程:
    1. 提取noisy embedding
    2. 语音增强
    3. 提取enhanced embedding
    4. 通过MLP融合
    """
    with torch.no_grad():
        # 提取noisy embedding
        noisy_emb = speaker_model.extract_embedding(audio_tensor=audio)
        noisy_emb = F.normalize(noisy_emb.squeeze(), p=2, dim=0)

        # 语音增强
        enhanced_audio = enhancer.enhance_audio(audio, sr=16000)

        # 提取enhanced embedding
        enhanced_emb = speaker_model.extract_embedding(audio_tensor=enhanced_audio)
        enhanced_emb = F.normalize(enhanced_emb.squeeze(), p=2, dim=0)

        # 通过MLP融合(会自动L2归一化)
        noisy_emb = noisy_emb.unsqueeze(0).to(device)
        enhanced_emb = enhanced_emb.unsqueeze(0).to(device)
        robust_emb = mlp(noisy_emb, enhanced_emb)

        return robust_emb.squeeze().cpu()


def compute_eer(scores, labels):
    """
    计算EER(Equal Error Rate)

    参数:
        scores: 相似度分数列表
        labels: 标签列表(1=same speaker, 0=different speaker)
    """
    # 转换为numpy
    scores = np.array(scores)
    labels = np.array(labels)

    # 按分数排序
    sorted_indices = np.argsort(scores)
    scores = scores[sorted_indices]
    labels = labels[sorted_indices]

    # 计算FAR和FRR
    n_positive = np.sum(labels == 1)
    n_negative = np.sum(labels == 0)

    best_eer = 1.0
    best_threshold = 0.0

    for i, threshold in enumerate(scores):
        # False Accept Rate: 负样本中得分>=threshold的比例
        far = np.sum((scores >= threshold) & (labels == 0)) / n_negative

        # False Reject Rate: 正样本中得分<threshold的比例
        frr = np.sum((scores < threshold) & (labels == 1)) / n_positive

        # EER是FAR和FRR相等的点
        eer = (far + frr) / 2

        if abs(far - frr) < abs(best_eer - 0.5):
            best_eer = eer
            best_threshold = threshold

    return best_eer * 100, best_threshold  # 返回百分比


def evaluate_supcon(mlp, speaker_model, enhancer, test_dataset, num_pairs=1000, device='cuda'):
    """
    评估SupCon模型

    参数:
        num_pairs: 测试pair数量(默认1000用于快速测试)

    返回:
        eer: Equal Error Rate (%)
    """
    print(f'\n📊 Evaluating SupCon model on {num_pairs} pairs...')

    mlp.eval()
    scores = []
    labels = []

    # ✅ 修复: 构建speaker_id到样本索引的映射
    print('Building speaker index...')
    speaker_to_samples = {}
    for idx in range(len(test_dataset)):
        # 获取speaker_id(通过临时加载)
        sample = test_dataset[idx]
        speaker_id = sample['speaker_id']

        if speaker_id not in speaker_to_samples:
            speaker_to_samples[speaker_id] = []
        speaker_to_samples[speaker_id].append(idx)

    print(f'Found {len(speaker_to_samples)} unique speakers in test set')

    # 生成测试pairs
    print('Generating test pairs...')
    test_pairs = []
    speakers = list(speaker_to_samples.keys())

    for i in range(num_pairs // 2):
        # 正样本对(同一说话人)
        speaker1 = np.random.choice(speakers)
        samples1 = speaker_to_samples[speaker1]

        if len(samples1) >= 2:
            idx1, idx2 = np.random.choice(samples1, size=2, replace=False)
            test_pairs.append((idx1, idx2, 1))

        # 负样本对(不同说话人)
        speaker2, speaker3 = np.random.choice(speakers, size=2, replace=False)
        idx3 = np.random.choice(speaker_to_samples[speaker2])
        idx4 = np.random.choice(speaker_to_samples[speaker3])
        test_pairs.append((idx3, idx4, 0))

    print(f'Generated {len(test_pairs)} test pairs')

    # 评估每个pair
    for idx1, idx2, label in tqdm(test_pairs, desc='Testing'):
        sample1 = test_dataset[idx1]
        sample2 = test_dataset[idx2]

        # 提取robust embeddings
        emb1 = extract_supcon_embedding(
            sample1['anchor_noisy'], speaker_model, enhancer, mlp, device
        )
        emb2 = extract_supcon_embedding(
            sample2['anchor_noisy'], speaker_model, enhancer, mlp, device
        )

        # 计算余弦相似度
        score = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()

        scores.append(score)
        labels.append(label)

    # 计算EER
    eer, threshold = compute_eer(scores, labels)

    print(f'✅ Evaluation complete')
    print(f'   - EER: {eer:.2f}%')
    print(f'   - Threshold: {threshold:.4f}')
    print(f'   - Avg score (same): {np.mean([s for s, l in zip(scores, labels) if l == 1]):.4f}')
    print(f'   - Avg score (diff): {np.mean([s for s, l in zip(scores, labels) if l == 0]):.4f}')

    return eer, threshold


def print_comparison_table(supcon_eer, noise_type='-15dB'):
    """
    打印与原论文baseline的对比表格

    参数:
        supcon_eer: SupCon模型的EER
        noise_type: 噪声类型(用于从baseline中查找)
    """
    print('\n' + '=' * 80)
    print('📊 COMPARISON WITH PAPER BASELINE')
    print('=' * 80)

    # 获取对应噪声类型的baseline
    baseline_key = f'Noise_{noise_type}'
    if baseline_key in PAPER_BASELINE:
        baseline = PAPER_BASELINE[baseline_key]

        print(f'\n{"Method":<25} {"EER (%)":<15} {"vs Triplet":<15} {"vs Noisy":<15}')
        print('-' * 80)

        # Noisy baseline
        print(f'{"Noisy (Paper)":<25} {baseline["Noisy"]:<15.2f} {"":<15} {"":<15}')

        # Enhanced baseline
        print(f'{"Enhanced (Paper)":<25} {baseline["Enhanced"]:<15.2f} {"":<15} {"":<15}')

        # Triplet Loss baseline
        triplet_eer = baseline['Triplet(Paper)']
        print(f'{"Triplet Loss (Paper)":<25} {triplet_eer:<15.2f} {"":<15} {"":<15}')

        # SupCon (你的结果)
        vs_triplet = supcon_eer - triplet_eer
        vs_noisy = supcon_eer - baseline['Noisy']

        status = '✅' if supcon_eer < triplet_eer else '⚠️'
        print(f'{f"SupCon (Yours) {status}":<25} {supcon_eer:<15.2f} {vs_triplet:+.2f} {"":<6} {vs_noisy:+.2f}')

        print('-' * 80)

        # 分析
        print('\n🔍 ANALYSIS:')
        if supcon_eer < triplet_eer:
            improvement = (triplet_eer - supcon_eer) / triplet_eer * 100
            print(f'✅ SupCon is BETTER than Triplet Loss by {improvement:.1f}%')
            print(f'   - Absolute improvement: {triplet_eer - supcon_eer:.2f}% EER')
        elif supcon_eer > triplet_eer:
            degradation = (supcon_eer - triplet_eer) / triplet_eer * 100
            print(f'⚠️  SupCon is worse than Triplet Loss by {degradation:.1f}%')
            print(f'   - Consider: adjusting temperature, more training epochs')
        else:
            print(f'➖ SupCon performs similarly to Triplet Loss')

        if supcon_eer < baseline['Noisy']:
            improvement_noisy = (baseline['Noisy'] - supcon_eer) / baseline['Noisy'] * 100
            print(f'✅ SupCon improves Noisy baseline by {improvement_noisy:.1f}%')

    print('=' * 80)


def main():
    parser = argparse.ArgumentParser(description='Evaluate SupCon model')
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints_supcon_200batch/final_model_supcon.pth',
                        help='Path to SupCon checkpoint')
    parser.add_argument('--num_pairs', type=int, default=1000,
                        help='Number of test pairs (default: 1000 for fast testing)')
    parser.add_argument('--noise_type', type=str, default='-15dB',
                        help='Noise type for comparison (default: -15dB)')
    parser.add_argument('--voxceleb_dir', type=str, default='data/voxceleb1',
                        help='VoxCeleb dataset directory')
    parser.add_argument('--musan_dir', type=str, default='data/musan',
                        help='MUSAN dataset directory')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('=' * 80)
    print('🧪 SUPCON MODEL EVALUATION')
    print('=' * 80)
    print(f'Checkpoint: {args.checkpoint}')
    print(f'Test pairs: {args.num_pairs}')
    print(f'Device: {device}')
    print('=' * 80)

    # 加载模型
    print('\n[1/4] Loading models...')
    mlp, checkpoint = load_supcon_model(args.checkpoint, device)
    speaker_model = SpeakerEmbeddingExtractor('ecapa')
    enhancer = SpeechEnhancer()

    # 加载测试数据
    print('\n[2/4] Loading test dataset...')
    test_dataset = VoxCelebMusanDataset(
        args.voxceleb_dir,
        args.musan_dir,
        split='test',
        snr_range=(-15, -15)  # 固定-15dB用于与论文对比
    )
    print(f'✅ Test samples: {len(test_dataset)}')

    # 评估
    print('\n[3/4] Evaluating...')
    supcon_eer, threshold = evaluate_supcon(
        mlp, speaker_model, enhancer, test_dataset,
        num_pairs=args.num_pairs, device=device
    )

    # 对比
    print('\n[4/4] Comparing with baseline...')
    print_comparison_table(supcon_eer, noise_type=args.noise_type)

    # 保存结果
    results_file = Path(args.checkpoint).parent / 'evaluation_results.json'
    results = {
        'supcon_eer': supcon_eer,
        'threshold': threshold,
        'num_pairs': args.num_pairs,
        'paper_baseline': PAPER_BASELINE,
        'checkpoint': str(args.checkpoint)
    }

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f'\n💾 Results saved: {results_file}')
    print('\n✅ Evaluation complete!')


if __name__ == '__main__':
    main()