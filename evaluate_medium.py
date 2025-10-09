import os

os.environ['SPEECHBRAIN_LOCAL_STRATEGY'] = 'copy'
os.environ['HF_HUB_OFFLINE'] = '1'

import torch
import numpy as np
from pathlib import Path
import sys

sys.path.append('scripts')

from models import RobustEmbeddingMLP
import torch.nn.functional as F
from sklearn.metrics import roc_curve
from tqdm import tqdm
import time
import json
import torchaudio
import random

from speechbrain.inference import EncoderClassifier
from df.enhance import enhance, init_df


def compute_eer(distances, labels):
    """计算 Equal Error Rate"""
    scores = [-d for d in distances]
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.absolute(fnr - fpr))
    eer = (fpr[idx] + fnr[idx]) / 2
    return eer * 100


class VoxCelebTestPairsEvaluator:
    """使用官方verification pairs的评估器"""

    def __init__(self, voxceleb_dir, musan_dir, pairs_file):
        self.voxceleb_dir = Path(voxceleb_dir)
        self.musan_dir = Path(musan_dir)
        self.pairs = self.load_pairs(pairs_file)
        print(f"Loaded {len(self.pairs)} verification pairs")
        self.musan_files = self.load_musan_test_files()

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
        return pairs

    def load_musan_test_files(self):
        """加载MUSAN测试集(20%)"""
        split_file = self.musan_dir / 'musan_split.json'

        if not split_file.exists():
            raise FileNotFoundError(
                f"MUSAN split file not found: {split_file}\n"
                "Please run: python scripts/split_musan.py data/musan"
            )

        with open(split_file, 'r') as f:
            split_data = json.load(f)

        musan_files = {'noise': [], 'music': [], 'speech': []}
        for noise_type in ['noise', 'music', 'speech']:
            files = split_data[noise_type]['test']
            for rel_path in files:
                full_path = self.musan_dir / rel_path
                if full_path.exists():
                    musan_files[noise_type].append(str(full_path))

        print(f"MUSAN test files loaded:")
        print(f"  Noise: {len(musan_files['noise'])} files")
        print(f"  Music: {len(musan_files['music'])} files")
        print(f"  Speech: {len(musan_files['speech'])} files")

        return musan_files

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

        if noise.shape[1] < audio.shape[1]:
            repeats = int(np.ceil(audio.shape[1] / noise.shape[1]))
            noise = noise.repeat(1, repeats)
        noise = noise[:, :audio.shape[1]]

        signal_power = audio.pow(2).mean()
        noise_power = noise.pow(2).mean()
        snr_linear = 10 ** (snr_db / 10)
        scale = torch.sqrt(signal_power / (noise_power * snr_linear))

        noisy_audio = audio + scale * noise
        return noisy_audio

    def load_audio(self, rel_path):
        """加载音频文件"""
        audio_path = self.voxceleb_dir / 'voxceleb1_complete' / 'vox1_test_wav' / 'wav' / rel_path

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio not found: {audio_path}")

        audio, sr = torchaudio.load(audio_path)

        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        return audio, sr


class PaperEvaluator:
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = device

        print("=" * 70)
        print("LOADING MODELS FOR EVALUATION (OPTIMIZED VERSION)")
        print("=" * 70)

        # 加载ECAPA-TDNN说话人识别模型
        print("Loading speaker model...")
        self.speaker_model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device": device}
        )

        # 加载DeepFilterNet3
        print("Loading DeepFilterNet3...")

        # 补丁：避免DeepFilterNet调用git
        import subprocess
        original_check_output = subprocess.check_output
        subprocess.check_output = lambda *args, **kwargs: b'unknown' if 'git' in str(args) else original_check_output(
            *args, **kwargs)

        self.enhancer_model, self.df_state, _ = init_df()
        self.enhancer_model = self.enhancer_model.to(device)

        subprocess.check_output = original_check_output  # 恢复


        # ✅ 关键优化：预创建resampler（避免重复创建）
        print("Creating resamplers (optimization)...")
        self.resampler_to_48k = torchaudio.transforms.Resample(16000, 48000).to(device)
        self.resampler_to_16k = torchaudio.transforms.Resample(48000, 16000).to(device)

        # 加载训练好的MLP
        print("Loading trained MLP...")
        self.mlp = RobustEmbeddingMLP(embedding_dim=192).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.mlp.load_state_dict(checkpoint['model_state_dict'])
        self.mlp.eval()

        print(f"✓ All models loaded (OPTIMIZED)")
        print(f"  MLP: {checkpoint_path}")
        print(f"  Batches: {checkpoint.get('batch', 'N/A')}")
        print(f"  Loss: {checkpoint.get('loss', 'N/A'):.4f}")
        print(f"  Optimization: Pre-created resamplers")
        print("=" * 70)

    def extract_speaker_embedding(self, audio, sr=16000):
        """提取说话人embedding"""
        with torch.no_grad():
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            embeddings = self.speaker_model.encode_batch(audio)
            return embeddings

    def enhance_audio(self, audio, sr=16000):
        """
        使用DeepFilterNet3增强音频 - 优化版本
        ✅ 使用预创建的resampler，避免每次都创建新对象
        """
        TARGET_SR = 48000

        with torch.no_grad():
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)

            # 使用预创建的resampler（关键优化！）
            if sr != TARGET_SR:
                audio_48k = self.resampler_to_48k(audio)
            else:
                audio_48k = audio

            # 增强（DeepFilterNet3在CPU上运行）
            enhanced_48k = enhance(self.enhancer_model, self.df_state, audio_48k.cpu())

            # 使用预创建的resampler转回16kHz
            if sr != TARGET_SR:
                enhanced = self.resampler_to_16k(enhanced_48k.to(self.device))
            else:
                enhanced = enhanced_48k.to(self.device)

            return enhanced

    def extract_embeddings(self, audio, sr=16000):
        """提取noisy, enhanced和robust embeddings"""
        with torch.no_grad():
            noisy_emb = self.extract_speaker_embedding(audio, sr)
            enhanced_audio = self.enhance_audio(audio, sr)
            enhanced_emb = self.extract_speaker_embedding(enhanced_audio, sr)

            if noisy_emb.dim() == 3:
                noisy_emb = noisy_emb.squeeze(1)
            if enhanced_emb.dim() == 3:
                enhanced_emb = enhanced_emb.squeeze(1)

            # ✅ 现在要归一化输入（与训练一致）
            noisy_emb = F.normalize(noisy_emb, p=2, dim=1)
            enhanced_emb = F.normalize(enhanced_emb, p=2, dim=1)

            # MLP内部会再归一化输出
            robust_emb = self.mlp(noisy_emb, enhanced_emb)

        return noisy_emb, enhanced_emb, robust_emb

    def evaluate_condition(self, test_pairs_eval, noise_type, snr_db, max_pairs=None, seed=42):
        """评估单个条件"""
        condition_seed = seed + hash(f"{noise_type}_{snr_db}") % 10000
        random.seed(condition_seed)
        torch.manual_seed(condition_seed)
        np.random.seed(condition_seed)

        pairs = test_pairs_eval.pairs
        if max_pairs:
            pairs = pairs[:max_pairs]

        noisy_scores = []
        enhanced_scores = []
        robust_scores = []
        labels = []

        print(f"  Evaluating {len(pairs)} pairs (seed={condition_seed})...")

        for label, path1, path2 in tqdm(pairs, desc=f"{noise_type} {snr_db}dB", leave=False):
            audio1, sr1 = test_pairs_eval.load_audio(path1)
            audio2, sr2 = test_pairs_eval.load_audio(path2)

            noisy1 = test_pairs_eval.add_noise(audio1, sr1, noise_type, snr_db)
            noisy2 = test_pairs_eval.add_noise(audio2, sr2, noise_type, snr_db)

            noisy1 = noisy1.to(self.device)
            noisy2 = noisy2.to(self.device)

            emb1_noisy, emb1_enh, emb1_robust = self.extract_embeddings(noisy1, sr1)
            emb2_noisy, emb2_enh, emb2_robust = self.extract_embeddings(noisy2, sr2)

            noisy_dist = (1 - F.cosine_similarity(emb1_noisy, emb2_noisy, dim=1)).item()
            enhanced_dist = (1 - F.cosine_similarity(emb1_enh, emb2_enh, dim=1)).item()
            robust_dist = (1 - F.cosine_similarity(emb1_robust, emb2_robust, dim=1)).item()

            noisy_scores.append(noisy_dist)
            enhanced_scores.append(enhanced_dist)
            robust_scores.append(robust_dist)
            labels.append(label)

        noisy_eer = compute_eer(noisy_scores, labels)
        enhanced_eer = compute_eer(enhanced_scores, labels)
        robust_eer = compute_eer(robust_scores, labels)

        return noisy_eer, enhanced_eer, robust_eer

    def evaluate_full(self, voxceleb_dir, musan_dir, pairs_file, quick_test=False, seed=42):
        """完整评估Table 1"""
        print("\n" + "=" * 70)
        print("PAPER TABLE 1 EVALUATION - OPTIMIZED VERSION")
        print(f"Random seed: {seed} (for reproducibility)")
        print("=" * 70)

        test_pairs_eval = VoxCelebTestPairsEvaluator(voxceleb_dir, musan_dir, pairs_file)

        max_pairs = 1000 if quick_test else None
        if quick_test:
            print(f"⚠️  QUICK TEST: Only 100 pairs per condition")

        print("=" * 70)

        noise_types = ['noise', 'music', 'speech']
        snr_levels = [0, -5, -10, -15, -20]

        results = {}
        start_time = time.time()

        for noise_type in noise_types:
            results[noise_type] = {}
            display_name = {'noise': 'NOISE', 'music': 'MUSIC', 'speech': 'SPEECH (Babble)'}
            print(f"\n{'=' * 70}")
            print(f"Noise Type: {display_name[noise_type]}")
            print(f"{'=' * 70}")

            for snr in snr_levels:
                condition_start = time.time()

                noisy_eer, enhanced_eer, robust_eer = self.evaluate_condition(
                    test_pairs_eval=test_pairs_eval,
                    noise_type=noise_type,
                    snr_db=snr,
                    max_pairs=max_pairs,
                    seed=seed
                )

                results[noise_type][snr] = {
                    'noisy': noisy_eer,
                    'enhanced': enhanced_eer,
                    'robust': robust_eer
                }

                elapsed = time.time() - condition_start
                best = min(noisy_eer, enhanced_eer, robust_eer)
                noisy_mark = "★" if noisy_eer == best else " "
                enhanced_mark = "★" if enhanced_eer == best else " "
                robust_mark = "★" if robust_eer == best else " "

                print(f"  SNR={snr:3d}dB: "
                      f"Noisy={noisy_eer:5.2f}%{noisy_mark} | "
                      f"Enhanced={enhanced_eer:5.2f}%{enhanced_mark} | "
                      f"Ours={robust_eer:5.2f}%{robust_mark} | "
                      f"Time={elapsed:.1f}s")

        total_time = time.time() - start_time
        print(f"\n{'=' * 70}")
        print(f"Total time: {total_time / 60:.1f} minutes")
        print(f"{'=' * 70}")

        self.print_table(results)
        return results

    def print_table(self, results):
        """打印论文Table 1格式的结果"""
        print("\n" + "=" * 70)
        print("RESULTS TABLE (ECAPA-TDNN)")
        print("=" * 70)
        print(f"{'Type':<8} {'SNR':>4} | {'Noisy':>6} {'Enhc':>6} {'Ours':>6} | {'Best':<6}")
        print("-" * 70)

        for noise_type in ['noise', 'music', 'speech']:
            display_name = {'noise': 'Noise', 'music': 'Music', 'speech': 'Speech'}
            print(f"\n{display_name[noise_type]}")

            for snr in [0, -5, -10, -15, -20]:
                r = results[noise_type][snr]

                best_val = min(r['noisy'], r['enhanced'], r['robust'])
                if r['noisy'] == best_val:
                    best_method = "Noisy"
                elif r['enhanced'] == best_val:
                    best_method = "Enhc"
                else:
                    best_method = "Ours"

                print(f"         {snr:>4} | {r['noisy']:>6.2f} {r['enhanced']:>6.2f} "
                      f"{r['robust']:>6.2f} | {best_method:<6}")

        print("\n" + "-" * 70)
        print("Note: Results may differ from paper values due to random")
        print("      initialization and noise sampling. Focus on trends.")
        print("=" * 70)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Paper evaluation - OPTIMIZED')
    parser.add_argument('--checkpoint', default='checkpoints_paper_final/paper_exact_model.pth')
    parser.add_argument('--voxceleb', default='data/voxceleb1')
    parser.add_argument('--musan', default='data/musan')
    parser.add_argument('--pairs', default='data/voxceleb1/veri_test2.txt')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda')

    args = parser.parse_args()

    if not Path(args.checkpoint).exists():
        print(f"✗ Checkpoint not found: {args.checkpoint}")
        return

    if not Path(args.pairs).exists():
        print(f"✗ Pairs file not found: {args.pairs}")
        return

    test_dir = Path(args.voxceleb) / 'voxceleb1_complete' / 'vox1_test_wav' / 'wav'
    if not test_dir.exists():
        print(f"✗ Test set not found: {test_dir}")
        return

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"Global random seed set to: {args.seed}")

    evaluator = PaperEvaluator(args.checkpoint, device=args.device)
    results = evaluator.evaluate_full(
        voxceleb_dir=args.voxceleb,
        musan_dir=args.musan,
        pairs_file=args.pairs,
        quick_test=args.quick,
        seed=args.seed
    )

    output_file = f'results_paper_table1_v2_seed{args.seed}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to: {output_file}")

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    improvements = []
    for noise_type in ['noise', 'music', 'speech']:
        for snr in [-10, -15, -20]:
            r = results[noise_type][snr]
            improvement = r['noisy'] - r['robust']
            improvements.append(improvement)

    avg_improvement = np.mean(improvements)
    print(f"Average improvement (Ours vs Noisy) at low SNR: {avg_improvement:.2f}%")
    print(f"Optimization: Pre-created resamplers for faster processing")
    print("=" * 70)


if __name__ == '__main__':
    main()