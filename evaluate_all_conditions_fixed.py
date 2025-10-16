"""
完整的15条件评估脚本 (基于VoxCeleb官方pairs)
- 使用veri_test2.txt官方验证集
- 与论文Table 1精确对应
- 测试3种噪声 × 5个SNR = 15个条件
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import json
import time
from datetime import datetime
import sys
import random
import torchaudio

sys.path.append('scripts')

from models_confidence import DynamicFusionMLP
from speaker_embedding import SpeakerEmbeddingExtractor
from speech_enhancer import SpeechEnhancer

# 📊 论文Table 1的baseline结果
PAPER_BASELINE = {
    'noise': {
        0: {'Noisy': 9.70, 'Enhanced': 13.45, 'Triplet': 13.17},
        -5: {'Noisy': 16.39, 'Enhanced': 18.18, 'Triplet': 15.67},
        -10: {'Noisy': 26.19, 'Enhanced': 25.12, 'Triplet': 19.31},
        -15: {'Noisy': 34.71, 'Enhanced': 32.77, 'Triplet': 25.21},
        -20: {'Noisy': 41.66, 'Enhanced': 39.69, 'Triplet': 33.00}
    },
    'music': {
        0: {'Noisy': 12.42, 'Enhanced': 14.62, 'Triplet': 15.33},
        -5: {'Noisy': 23.73, 'Enhanced': 22.80, 'Triplet': 19.48},
        -10: {'Noisy': 36.82, 'Enhanced': 33.72, 'Triplet': 27.37},
        -15: {'Noisy': 44.46, 'Enhanced': 41.86, 'Triplet': 36.37},
        -20: {'Noisy': 48.69, 'Enhanced': 47.39, 'Triplet': 44.05}
    },
    'speech': {  # Babble
        0: {'Noisy': 20.24, 'Enhanced': 24.79, 'Triplet': 21.89},
        -5: {'Noisy': 32.64, 'Enhanced': 36.59, 'Triplet': 26.82},
        -10: {'Noisy': 43.85, 'Enhanced': 44.77, 'Triplet': 32.30},
        -15: {'Noisy': 46.72, 'Enhanced': 47.48, 'Triplet': 37.21},
        -20: {'Noisy': 48.31, 'Enhanced': 48.38, 'Triplet': 41.73}
    }
}


class VoxCelebPairsLoader:
    """加载VoxCeleb官方验证对"""

    def __init__(self, voxceleb_dir, musan_dir, pairs_file):
        self.voxceleb_dir = Path(voxceleb_dir)
        self.musan_dir = Path(musan_dir)
        self.pairs = self.load_pairs(pairs_file)
        self.musan_files = self.load_musan_files()

    def load_pairs(self, pairs_file):
        """加载veri_test2.txt"""
        pairs = []
        with open(pairs_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    label = int(parts[0])
                    path1 = parts[1]
                    path2 = parts[2]
                    pairs.append((label, path1, path2))
        print(f"✅ Loaded {len(pairs)} verification pairs")
        return pairs

    def load_musan_files(self):
        """加载MUSAN测试集"""
        split_file = self.musan_dir / 'musan_split.json'

        if not split_file.exists():
            raise FileNotFoundError(f"MUSAN split file not found: {split_file}")

        with open(split_file, 'r') as f:
            split_data = json.load(f)

        musan_files = {'noise': [], 'music': [], 'speech': []}
        for noise_type in ['noise', 'music', 'speech']:
            files = split_data[noise_type]['test']
            for rel_path in files:
                full_path = self.musan_dir / rel_path
                if full_path.exists():
                    musan_files[noise_type].append(str(full_path))

        print(f"✅ MUSAN test files loaded:")
        for noise_type, files in musan_files.items():
            print(f"   - {noise_type}: {len(files)} files")

        return musan_files

    def load_audio(self, rel_path):
        """加载音频文件"""
        audio_path = self.voxceleb_dir / 'voxceleb1_complete' / 'vox1_test_wav' / 'wav' / rel_path

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio not found: {audio_path}")

        audio, sr = torchaudio.load(audio_path)

        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        return audio, sr

    def add_noise(self, audio, sr, noise_type, snr_db):
        """添加指定类型和SNR的噪声"""
        noise_files = self.musan_files[noise_type]
        if not noise_files:
            return audio

        noise_file = random.choice(noise_files)
        noise, noise_sr = torchaudio.load(noise_file)

        if noise_sr != sr:
            resampler = torchaudio.transforms.Resample(noise_sr, sr)
            noise = resampler(noise)

        if noise.shape[0] > 1:
            noise = noise.mean(dim=0, keepdim=True)

        # 重复噪声以匹配音频长度
        if noise.shape[1] < audio.shape[1]:
            repeats = int(np.ceil(audio.shape[1] / noise.shape[1]))
            noise = noise.repeat(1, repeats)
        noise = noise[:, :audio.shape[1]]

        # 计算SNR
        signal_power = audio.pow(2).mean()
        noise_power = noise.pow(2).mean()
        snr_linear = 10 ** (snr_db / 10)
        scale = torch.sqrt(signal_power / (noise_power * snr_linear))

        noisy_audio = audio + scale * noise
        return noisy_audio


def load_confidence_model(checkpoint_path, device):
    """加载Confidence模型"""
    print(f'📦 Loading model from: {checkpoint_path}')

    mlp = DynamicFusionMLP(
        embedding_dim=192,
        confidence_hidden_dim=64
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    mlp.load_state_dict(checkpoint['model_state_dict'])
    mlp.eval()

    print(f"✅ Model loaded successfully")
    print(f"   - Training loss: {checkpoint['loss']:.4f}")
    print(f"   - Trained batches: {checkpoint['batch']}")

    return mlp, checkpoint


def extract_confidence_embedding(audio, speaker_model, enhancer, mlp, device):
    """
    使用Confidence模型提取embedding
    返回: robust_emb, w_noisy, w_enhanced
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

        # 通过MLP融合
        noisy_emb = noisy_emb.unsqueeze(0).to(device)
        enhanced_emb = enhanced_emb.unsqueeze(0).to(device)

        robust_emb, weights, _ = mlp(noisy_emb, enhanced_emb, return_weights=True)

        w_noisy = weights[0, 0].item()
        w_enhanced = weights[0, 1].item()

        return robust_emb.squeeze().cpu(), w_noisy, w_enhanced


def compute_eer(scores, labels):
    """计算EER"""
    from sklearn.metrics import roc_curve

    # 转换为距离（越小越相似）
    distances = [1 - s for s in scores]

    # 计算ROC曲线
    fpr, tpr, thresholds = roc_curve(labels, [-d for d in distances])
    fnr = 1 - tpr

    # 找到EER点
    idx = np.nanargmin(np.absolute(fnr - fpr))
    eer = (fpr[idx] + fnr[idx]) / 2

    return eer * 100


def evaluate_single_condition(mlp, speaker_model, enhancer, pairs_loader,
                              noise_type, snr_db, device, max_pairs=None, seed=42):
    """评估单个条件"""
    # 设置随机种子（确保可重复）
    condition_seed = seed + hash(f"{noise_type}_{snr_db}") % 10000
    random.seed(condition_seed)
    torch.manual_seed(condition_seed)
    np.random.seed(condition_seed)

    mlp.eval()
    scores = []
    labels = []
    weights_same = []
    weights_diff = []

    # 选择要测试的pairs
    pairs = pairs_loader.pairs
    if max_pairs:
        pairs = pairs[:max_pairs]

    # 评估每个pair
    desc = f'{noise_type.capitalize()} {snr_db}dB'
    for label, path1, path2 in tqdm(pairs, desc=desc, leave=False):
        # 加载音频
        audio1, sr1 = pairs_loader.load_audio(path1)
        audio2, sr2 = pairs_loader.load_audio(path2)

        # 添加噪声
        noisy1 = pairs_loader.add_noise(audio1, sr1, noise_type, snr_db)
        noisy2 = pairs_loader.add_noise(audio2, sr2, noise_type, snr_db)

        noisy1 = noisy1.to(device)
        noisy2 = noisy2.to(device)

        # 提取embeddings
        emb1, w1_noisy, w1_enh = extract_confidence_embedding(
            noisy1, speaker_model, enhancer, mlp, device
        )
        emb2, w2_noisy, w2_enh = extract_confidence_embedding(
            noisy2, speaker_model, enhancer, mlp, device
        )

        # 计算相似度
        score = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()

        scores.append(score)
        labels.append(label)

        # 记录权重
        avg_noisy = (w1_noisy + w2_noisy) / 2
        avg_enh = (w1_enh + w2_enh) / 2

        if label == 1:
            weights_same.append({'noisy': avg_noisy, 'enhanced': avg_enh})
        else:
            weights_diff.append({'noisy': avg_noisy, 'enhanced': avg_enh})

    # 计算EER
    eer = compute_eer(scores, labels)

    # 计算平均权重
    avg_weights = {
        'same': {
            'noisy': np.mean([w['noisy'] for w in weights_same]) if weights_same else 0,
            'enhanced': np.mean([w['enhanced'] for w in weights_same]) if weights_same else 0
        },
        'diff': {
            'noisy': np.mean([w['noisy'] for w in weights_diff]) if weights_diff else 0,
            'enhanced': np.mean([w['enhanced'] for w in weights_diff]) if weights_diff else 0
        }
    }

    return eer, avg_weights


def print_single_condition_result(noise_type, snr, eer, baseline):
    """打印单个条件的结果"""
    triplet_eer = baseline['Triplet']
    improvement = triplet_eer - eer
    improvement_pct = (improvement / triplet_eer) * 100 if triplet_eer > 0 else 0

    status = '✅' if eer < triplet_eer else '⚠️' if eer > triplet_eer else '➖'

    print(f"\n{status} {noise_type.capitalize()} @ {snr}dB:")
    print(f"   Noisy (Paper):    {baseline['Noisy']:6.2f}%")
    print(f"   Enhanced (Paper): {baseline['Enhanced']:6.2f}%")
    print(f"   Triplet (Paper):  {baseline['Triplet']:6.2f}%")
    print(f"   Confidence:       {eer:6.2f}%  (Δ {improvement:+.2f}%, {improvement_pct:+.1f}%)")


def generate_comparison_table(results, output_dir):
    """生成对比表格"""
    output_dir = Path(output_dir)

    # Markdown表格
    md_path = output_dir / 'comparison_table.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('# Complete Evaluation Results\n\n')
        f.write('## Comparison with Paper Baseline (Table 1)\n\n')

        noise_names = {'noise': 'Noise', 'music': 'Music', 'speech': 'Speech (Babble)'}

        for noise_type in ['noise', 'music', 'speech']:
            f.write(f'\n### {noise_names[noise_type]}\n\n')
            f.write(
                '| SNR | Noisy (Paper) | Enhanced (Paper) | Triplet (Paper) | Confidence (Ours) | vs Triplet | Improvement |\n')
            f.write(
                '|-----|---------------|------------------|-----------------|-------------------|------------|-------------|\n')

            for snr in [0, -5, -10, -15, -20]:
                result = results[noise_type][snr]
                baseline = PAPER_BASELINE[noise_type][snr]

                eer = result['eer']
                diff = baseline['Triplet'] - eer
                improv = (diff / baseline['Triplet'] * 100) if baseline['Triplet'] > 0 else 0

                status = '✅' if eer < baseline['Triplet'] else '⚠️'

                f.write(f'| {snr:3d} | {baseline["Noisy"]:6.2f}% | {baseline["Enhanced"]:6.2f}% | '
                        f'{baseline["Triplet"]:6.2f}% | {eer:6.2f}% {status} | {diff:+.2f}% | {improv:+.1f}% |\n')

    print(f'📄 Markdown table saved: {md_path}')

    # CSV文件
    csv_path = output_dir / 'results.csv'
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write(
            'Noise Type,SNR,Noisy(Paper),Enhanced(Paper),Triplet(Paper),Confidence(Ours),vs Triplet,Improvement(%)\n')

        for noise_type in ['noise', 'music', 'speech']:
            for snr in [0, -5, -10, -15, -20]:
                result = results[noise_type][snr]
                baseline = PAPER_BASELINE[noise_type][snr]

                eer = result['eer']
                diff = baseline['Triplet'] - eer
                improv = (diff / baseline['Triplet'] * 100) if baseline['Triplet'] > 0 else 0

                f.write(f'{noise_type},{snr},{baseline["Noisy"]:.2f},{baseline["Enhanced"]:.2f},'
                        f'{baseline["Triplet"]:.2f},{eer:.2f},{diff:+.2f},{improv:+.1f}\n')

    print(f'📊 CSV file saved: {csv_path}')


def generate_summary_statistics(results, output_dir):
    """生成汇总统计"""
    total_conditions = 0
    better_count = 0
    total_improvement = 0
    best_improvement = {'value': -999, 'condition': ''}
    worst_case = {'value': 999, 'condition': ''}

    for noise_type in ['noise', 'music', 'speech']:
        for snr in [0, -5, -10, -15, -20]:
            total_conditions += 1

            result = results[noise_type][snr]
            baseline = PAPER_BASELINE[noise_type][snr]

            eer = result['eer']
            triplet_eer = baseline['Triplet']
            diff = triplet_eer - eer

            if eer < triplet_eer:
                better_count += 1

            total_improvement += diff

            if diff > best_improvement['value']:
                best_improvement = {'value': diff, 'condition': f'{noise_type} {snr}dB'}

            if diff < worst_case['value']:
                worst_case = {'value': diff, 'condition': f'{noise_type} {snr}dB'}

    avg_improvement = total_improvement / total_conditions
    win_rate = (better_count / total_conditions) * 100

    # 打印统计
    print('\n' + '=' * 80)
    print('📊 OVERALL STATISTICS')
    print('=' * 80)
    print(f'Total conditions tested:              {total_conditions}')
    print(f'Conditions where Confidence > Triplet: {better_count}/{total_conditions} ({win_rate:.1f}%)')
    print(f'Average EER improvement:              {avg_improvement:+.2f}%')
    print(f'Best improvement:                     {best_improvement["value"]:+.2f}% @ {best_improvement["condition"]}')
    print(f'Worst case:                           {worst_case["value"]:+.2f}% @ {worst_case["condition"]}')
    print('=' * 80)

    # 保存统计
    stats = {
        'total_conditions': total_conditions,
        'win_rate': win_rate,
        'average_improvement': avg_improvement,
        'best_improvement': best_improvement,
        'worst_case': worst_case
    }

    stats_path = Path(output_dir) / 'summary_statistics.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f'\n💾 Statistics saved: {stats_path}')

    return stats


def main():
    parser = argparse.ArgumentParser(description='Evaluate all 15 conditions')
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints_confidence_400batch/final_model_confidence.pth',
                        help='Confidence model checkpoint')
    parser.add_argument('--num_pairs', type=int, default=None,
                        help='Number of pairs to test (default: all pairs in veri_test2.txt)')
    parser.add_argument('--output_dir', type=str, default='results_all_conditions',
                        help='Output directory for results')
    parser.add_argument('--voxceleb_dir', type=str, default='data/voxceleb1')
    parser.add_argument('--musan_dir', type=str, default='data/musan')
    parser.add_argument('--pairs_file', type=str, default='data/voxceleb1/veri_test2.txt',
                        help='VoxCeleb verification pairs file')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # 设置随机种子
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 估算总时间
    pairs_info = "all" if args.num_pairs is None else str(args.num_pairs)
    estimated_time_per_condition = 30 if args.num_pairs is None else (args.num_pairs / 1000 * 30)
    total_estimated_time = estimated_time_per_condition * 15 / 60

    print('=' * 80)
    print('🔬 COMPREHENSIVE EVALUATION: All 15 Conditions')
    print('=' * 80)
    print(f'Checkpoint:       {args.checkpoint}')
    print(f'Pairs per condition: {pairs_info}')
    print(f'Total conditions: 15 (3 noise types × 5 SNR levels)')
    print(f'Output directory: {args.output_dir}')
    print(f'Random seed:      {args.seed}')
    print(f'Estimated time:   ~{total_estimated_time:.1f} hours')
    print('=' * 80)

    # 加载模型
    print('\n[1/3] Loading models...')
    mlp, checkpoint = load_confidence_model(args.checkpoint, device)
    speaker_model = SpeakerEmbeddingExtractor('ecapa')
    enhancer = SpeechEnhancer()

    print('✅ All models loaded')

    # 加载验证对
    print('\n[2/3] Loading verification pairs...')
    pairs_loader = VoxCelebPairsLoader(
        args.voxceleb_dir,
        args.musan_dir,
        args.pairs_file
    )

    # 测试所有条件
    print('\n[3/3] Testing all 15 conditions...')
    print('=' * 80)

    results = {}
    start_time = time.time()
    condition_count = 0

    for noise_type in ['noise', 'music', 'speech']:
        results[noise_type] = {}

        for snr in [0, -5, -10, -15, -20]:
            condition_count += 1

            print(f'\n[{condition_count}/15] Testing {noise_type.capitalize()} @ {snr}dB...')

            # 评估
            eer, weights = evaluate_single_condition(
                mlp, speaker_model, enhancer, pairs_loader,
                noise_type, snr, device,
                max_pairs=args.num_pairs,
                seed=args.seed
            )

            # 保存结果
            results[noise_type][snr] = {
                'eer': eer,
                'weights': weights
            }

            # 打印结果
            baseline = PAPER_BASELINE[noise_type][snr]
            print_single_condition_result(noise_type, snr, eer, baseline)

            # 预估剩余时间
            elapsed = (time.time() - start_time) / 60
            avg_time_per_condition = elapsed / condition_count
            remaining_conditions = 15 - condition_count
            estimated_remaining = avg_time_per_condition * remaining_conditions

            print(f'   ⏱️  Elapsed: {elapsed:.1f}min | Remaining: ~{estimated_remaining:.1f}min')

    # 生成报告
    print('\n[4/4] Generating reports...')

    # 保存完整结果
    results_path = output_dir / 'complete_results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'results': results,
            'baseline': PAPER_BASELINE,
            'config': {
                'checkpoint': args.checkpoint,
                'num_pairs': args.num_pairs,
                'seed': args.seed,
                'timestamp': datetime.now().isoformat()
            }
        }, f, indent=2)

    print(f'💾 Complete results saved: {results_path}')

    # 生成对比表格
    generate_comparison_table(results, output_dir)

    # 生成统计摘要
    generate_summary_statistics(results, output_dir)

    total_time = (time.time() - start_time) / 60
    print(f'\n✅ All done! Total time: {total_time:.1f} minutes')
    print(f'📁 Results saved in: {output_dir}')

    print('\n🎯 Next steps:')
    print('   1. Check comparison_table.md for detailed results')
    print('   2. Review summary_statistics.json for overview')
    print('   3. Use results.csv for Excel analysis')


if __name__ == '__main__':
    main()