"""
SupCon 400-batch模型完整测试脚本
- 支持10,000 pairs完整测试
- 与原论文baseline对比
- 与200-batch结果对比
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import json

import sys

sys.path.append('scripts')

from models_supcon import RobustEmbeddingMLP
from speaker_embedding import SpeakerEmbeddingExtractor
from speech_enhancer import SpeechEnhancer
from dataset import VoxCelebMusanDataset

# 原论文baseline结果
PAPER_BASELINE = {
    'Noise_-15dB': {
        'Noisy': 34.71,
        'Enhanced': 32.77,
        'Triplet(Paper)': 25.21
    }
}

# 之前200-batch的结果
BASELINE_200BATCH = {
    'SupCon_200batch': 21.70
}


def load_supcon_model(checkpoint_path, device):
    """加载SupCon模型"""
    print(f'Loading SupCon model from: {checkpoint_path}')

    mlp = RobustEmbeddingMLP(embedding_dim=192).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    mlp.load_state_dict(checkpoint['model_state_dict'])
    mlp.eval()

    print(f"✅ Model loaded")
    print(f"   - Training loss: {checkpoint['loss']:.4f}")
    print(f"   - Batch: {checkpoint['batch']}")
    print(f"   - Temperature: {checkpoint['config']['temperature']}")

    return mlp, checkpoint


def extract_supcon_embedding(audio, speaker_model, enhancer, mlp, device):
    """使用SupCon模型提取robust embedding"""
    with torch.no_grad():
        noisy_emb = speaker_model.extract_embedding(audio_tensor=audio)
        noisy_emb = F.normalize(noisy_emb.squeeze(), p=2, dim=0)

        enhanced_audio = enhancer.enhance_audio(audio, sr=16000)

        enhanced_emb = speaker_model.extract_embedding(audio_tensor=enhanced_audio)
        enhanced_emb = F.normalize(enhanced_emb.squeeze(), p=2, dim=0)

        noisy_emb = noisy_emb.unsqueeze(0).to(device)
        enhanced_emb = enhanced_emb.unsqueeze(0).to(device)
        robust_emb = mlp(noisy_emb, enhanced_emb)

        return robust_emb.squeeze().cpu()


def compute_eer(scores, labels):
    """计算EER"""
    scores = np.array(scores)
    labels = np.array(labels)

    sorted_indices = np.argsort(scores)
    scores = scores[sorted_indices]
    labels = labels[sorted_indices]

    n_positive = np.sum(labels == 1)
    n_negative = np.sum(labels == 0)

    best_eer = 1.0
    best_threshold = 0.0

    for i, threshold in enumerate(scores):
        far = np.sum((scores >= threshold) & (labels == 0)) / n_negative
        frr = np.sum((scores < threshold) & (labels == 1)) / n_positive
        eer = (far + frr) / 2

        if abs(far - frr) < abs(best_eer - 0.5):
            best_eer = eer
            best_threshold = threshold

    return best_eer * 100, best_threshold


def evaluate_supcon(mlp, speaker_model, enhancer, test_dataset, num_pairs=10000, device='cuda'):
    """评估SupCon模型"""
    print(f'\n📊 Evaluating SupCon model on {num_pairs} pairs...')

    mlp.eval()
    scores = []
    labels = []

    # 构建说话人索引
    print('Building speaker index...')
    speaker_to_samples = {}
    for idx, sample_info in enumerate(test_dataset.samples):
        speaker_id = sample_info['speaker_id']
        if speaker_id not in speaker_to_samples:
            speaker_to_samples[speaker_id] = []
        speaker_to_samples[speaker_id].append(idx)

    print(f'Found {len(speaker_to_samples)} unique speakers in test set')

    # 生成测试pairs
    print('Generating test pairs...')
    test_pairs = []
    speakers = list(speaker_to_samples.keys())

    for i in range(num_pairs // 2):
        # 正样本对
        speaker = np.random.choice(speakers)
        if len(speaker_to_samples[speaker]) >= 2:
            idx1, idx2 = np.random.choice(speaker_to_samples[speaker], 2, replace=False)
            test_pairs.append((idx1, idx2, 1))

        # 负样本对
        speaker1, speaker2 = np.random.choice(speakers, 2, replace=False)
        idx1 = np.random.choice(speaker_to_samples[speaker1])
        idx2 = np.random.choice(speaker_to_samples[speaker2])
        test_pairs.append((idx1, idx2, 0))

    print(f'Generated {len(test_pairs)} test pairs')

    # 评估每个pair
    for idx1, idx2, label in tqdm(test_pairs, desc='Testing'):
        sample1 = test_dataset[idx1]
        sample2 = test_dataset[idx2]

        emb1 = extract_supcon_embedding(
            sample1['anchor_noisy'], speaker_model, enhancer, mlp, device
        )
        emb2 = extract_supcon_embedding(
            sample2['anchor_noisy'], speaker_model, enhancer, mlp, device
        )

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


def print_comprehensive_comparison(eer_400batch, eer_200batch=21.70):
    """打印完整对比表格"""
    print('\n' + '=' * 80)
    print('📊 COMPREHENSIVE COMPARISON')
    print('=' * 80)

    baseline = PAPER_BASELINE['Noise_-15dB']

    print(f'\n{"Method":<30} {"EER (%)":<15} {"vs Paper Triplet":<20} {"vs 200-batch":<15}')
    print('-' * 85)

    # 论文baseline
    print(f'{"Noisy (Paper)":<30} {baseline["Noisy"]:<15.2f}')
    print(f'{"Enhanced (Paper)":<30} {baseline["Enhanced"]:<15.2f}')
    triplet_eer = baseline['Triplet(Paper)']
    print(f'{"Triplet Loss (Paper)":<30} {triplet_eer:<15.2f}')

    print('-' * 85)

    # 200-batch结果
    vs_triplet_200 = eer_200batch - triplet_eer
    print(
        f'{"SupCon 200-batch ✅":<30} {eer_200batch:<15.2f} {vs_triplet_200:+.2f} ({(abs(vs_triplet_200) / triplet_eer * 100):.1f}%)')

    # 400-batch结果
    vs_triplet_400 = eer_400batch - triplet_eer
    vs_200batch = eer_400batch - eer_200batch
    status = '✅' if eer_400batch < eer_200batch else '➖'
    print(f'{f"SupCon 400-batch {status}":<30} {eer_400batch:<15.2f} {vs_triplet_400:+.2f} ({(abs(vs_triplet_400) / triplet_eer * 100):.1f}%)    {vs_200batch:+.2f}')

    print('-' * 85)

    # 分析
    print('\n🔍 ANALYSIS:')

    # vs Triplet Loss
    if eer_400batch < triplet_eer:
        improvement = (triplet_eer - eer_400batch) / triplet_eer * 100
        print(f'✅ SupCon 400-batch is BETTER than Triplet Loss by {improvement:.1f}%')
        print(f'   - Absolute improvement: {triplet_eer - eer_400batch:.2f}% EER')

    # vs 200-batch
    if eer_400batch < eer_200batch:
        improvement = (eer_200batch - eer_400batch) / eer_200batch * 100
        print(f'✅ Extended training improved performance by {improvement:.1f}%')
        print(f'   - 400-batch is {eer_200batch - eer_400batch:.2f}% better than 200-batch')
    elif eer_400batch > eer_200batch:
        degradation = (eer_400batch - eer_200batch) / eer_200batch * 100
        print(f'➖ 400-batch is {degradation:.1f}% worse than 200-batch')
        print(f'   - This might indicate overfitting or random variation')
    else:
        print(f'➖ 400-batch performs similarly to 200-batch')

    # 统计显著性
    print(f'\n📊 Statistical Analysis (assuming 10,000 pairs):')
    se = np.sqrt(eer_400batch / 100 * (1 - eer_400batch / 100) / 10000) * 100
    ci_lower = eer_400batch - 1.96 * se
    ci_upper = eer_400batch + 1.96 * se
    print(f'   - 95% Confidence Interval: [{ci_lower:.2f}%, {ci_upper:.2f}%]')
    print(f'   - Standard Error: {se:.2f}%')

    if ci_upper < triplet_eer:
        print(f'   ✅ Statistically significant improvement over Triplet Loss')

    print('=' * 80)


def main():
    parser = argparse.ArgumentParser(description='Evaluate SupCon 400-batch model')
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints_supcon_400batch/final_model_supcon_400batch.pth',
                        help='Path to SupCon 400-batch checkpoint')
    parser.add_argument('--num_pairs', type=int, default=10000,
                        help='Number of test pairs (default: 10000 for comprehensive testing)')
    parser.add_argument('--compare_200batch', type=float, default=21.70,
                        help='EER from 200-batch training for comparison')
    parser.add_argument('--voxceleb_dir', type=str, default='data/voxceleb1')
    parser.add_argument('--musan_dir', type=str, default='data/musan')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('=' * 80)
    print('🧪 SUPCON 400-BATCH COMPREHENSIVE EVALUATION')
    print('=' * 80)
    print(f'Checkpoint: {args.checkpoint}')
    print(f'Test pairs: {args.num_pairs}')
    print(f'Device: {device}')
    print(f'Comparison baseline: 200-batch EER = {args.compare_200batch}%')
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
        snr_range=(-15, -15)
    )
    print(f'✅ Test samples: {len(test_dataset)}')

    # 评估
    print('\n[3/4] Evaluating...')
    eer_400batch, threshold = evaluate_supcon(
        mlp, speaker_model, enhancer, test_dataset,
        num_pairs=args.num_pairs, device=device
    )

    # 对比
    print('\n[4/4] Comprehensive comparison...')
    print_comprehensive_comparison(eer_400batch, args.compare_200batch)

    # 保存结果
    results_file = Path(args.checkpoint).parent / 'evaluation_results_400batch.json'
    results = {
        'eer_400batch': eer_400batch,
        'eer_200batch': args.compare_200batch,
        'threshold': threshold,
        'num_pairs': args.num_pairs,
        'improvement_vs_200batch': args.compare_200batch - eer_400batch,
        'improvement_vs_triplet': 25.21 - eer_400batch,
        'paper_baseline': PAPER_BASELINE,
        'checkpoint': str(args.checkpoint)
    }

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f'\n💾 Results saved: {results_file}')

    # 生成论文表格
    print('\n' + '=' * 80)
    print('📝 PAPER-READY RESULTS TABLE')
    print('=' * 80)
    print('\nLaTeX format:')
    print('\\begin{table}[h]')
    print('\\centering')
    print('\\begin{tabular}{lc}')
    print('\\hline')
    print('Method & EER (\\%) \\\\')
    print('\\hline')
    print(f'Noisy Baseline & 34.71 \\\\')
    print(f'Enhanced Baseline & 32.77 \\\\')
    print(f'Triplet Loss (Paper) & 25.21 \\\\')
    print(f'\\textbf{{SupCon (Ours)}} & \\textbf{{{eer_400batch:.2f}}} \\\\')
    print('\\hline')
    print('\\end{tabular}')
    print('\\end{table}')

    print('\n✅ Evaluation complete!')


if __name__ == '__main__':
    main()