#!/usr/bin/env python3
"""
简化的评估脚本 - 评估置信度网络V3效果
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'scripts'))


class ConfidenceNetworkV3(nn.Module):
    """与训练脚本完全一致的网络定义"""

    def __init__(self, embedding_dim=192, hidden_dim=256):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.diff_encoder = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        self.attention = nn.Sequential(
            nn.Linear(embedding_dim * 2 + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim)
        )

    def forward(self, emb_noisy, emb_enhanced):
        diff = torch.abs(emb_noisy - emb_enhanced)
        prod = emb_noisy * emb_enhanced
        diff_input = torch.cat([diff, prod], dim=1)
        diff_feat = self.diff_encoder(diff_input)

        concat = torch.cat([emb_noisy, emb_enhanced, diff_feat], dim=1)
        attn_logits = self.attention(concat)
        weights = F.softmax(attn_logits, dim=1)
        w_noisy = weights[:, 0:1]
        w_enhanced = weights[:, 1:2]

        weighted = w_noisy * emb_noisy + w_enhanced * emb_enhanced
        fused = self.fusion(torch.cat([emb_noisy, emb_enhanced], dim=1))
        output = weighted + 0.1 * fused

        return F.normalize(output, p=2, dim=1), weights


def compute_eer(scores, labels):
    """计算EER"""
    from scipy.optimize import brentq
    from scipy.interpolate import interp1d

    # 简单实现
    scores = np.array(scores)
    labels = np.array(labels)

    # 排序
    sorted_idx = np.argsort(scores)[::-1]
    scores_sorted = scores[sorted_idx]
    labels_sorted = labels[sorted_idx]

    # 计算累积
    total_pos = np.sum(labels == 1)
    total_neg = np.sum(labels == 0)

    if total_pos == 0 or total_neg == 0:
        return 50.0, 0.0

    tp = np.cumsum(labels_sorted == 1)
    fp = np.cumsum(labels_sorted == 0)

    tpr = tp / total_pos
    fpr = fp / total_neg
    fnr = 1 - tpr

    # 找EER点
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2 * 100

    return eer, scores_sorted[eer_idx]


def collate_fn_fixed(batch, target_length=48000):
    """固定长度collate"""
    result = {}
    for key in batch[0].keys():
        if isinstance(batch[0][key], torch.Tensor):
            tensors = [item[key] for item in batch]
            if tensors[0].dim() >= 1 and tensors[0].shape[-1] > 1000:
                padded = []
                for t in tensors:
                    if t.dim() == 1:
                        t = t.unsqueeze(0)
                    length = t.shape[-1]
                    if length > target_length:
                        t = t[..., :target_length]
                    elif length < target_length:
                        t = F.pad(t, (0, target_length - length))
                    padded.append(t)
                result[key] = torch.stack(padded)
            else:
                result[key] = torch.stack(tensors)
        else:
            result[key] = [item[key] for item in batch]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--voxceleb_dir', type=str, required=True)
    parser.add_argument('--musan_dir', type=str, required=True)
    parser.add_argument('--gtcrn_checkpoint', type=str, required=True)
    parser.add_argument('--speaker_model_path', type=str, required=True)
    parser.add_argument('--confidence_checkpoint', type=str, required=True)
    parser.add_argument('--snr_values', type=int, nargs='+', default=[-5, 0, 5, 10, 15, 20])
    parser.add_argument('--noise_types', type=str, nargs='+', default=['noise', 'music', 'speech'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--max_batches', type=int, default=50, help='每个条件最多评估多少个batch')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ========== 加载模型 ==========
    print("\n[1/3] Loading GTCRN...")
    from gtcrn_wrapper_fixed import GTCRNWrapper
    gtcrn = GTCRNWrapper(checkpoint_path=args.gtcrn_checkpoint, device=device)

    print("\n[2/3] Loading ECAPA-TDNN...")
    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError:
        from speechbrain.pretrained import EncoderClassifier

    speaker_model = EncoderClassifier.from_hparams(
        source=args.speaker_model_path,
        savedir=args.speaker_model_path,
        run_opts={"device": device}
    )

    print("\n[3/3] Loading ConfidenceNetworkV3...")
    confidence_net = ConfidenceNetworkV3(embedding_dim=192, hidden_dim=256).to(device)
    checkpoint = torch.load(args.confidence_checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        confidence_net.load_state_dict(checkpoint['model_state_dict'])
    else:
        confidence_net.load_state_dict(checkpoint)
    confidence_net.eval()
    print(f"   Loaded from {args.confidence_checkpoint}")

    # ========== 数据集 ==========
    from dataset_fixed import VoxCelebMusanDataset

    results = {}

    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)

    for noise_type in args.noise_types:
        for snr in args.snr_values:
            condition = f"{noise_type}_{snr}dB"
            print(f"\n>>> {condition}")

            test_dataset = VoxCelebMusanDataset(
                voxceleb_dir=args.voxceleb_dir,
                musan_dir=args.musan_dir,
                split='test',
                snr_range=(snr, snr),
                test_noise_type=noise_type,
                return_clean=False
            )

            test_loader = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=collate_fn_fixed,
                pin_memory=True
            )

            # 评估
            all_scores = {'noisy': [], 'enhanced': [], 'proposed': []}
            all_labels = []
            all_weights = []

            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(test_loader, desc=condition, leave=False)):
                    if batch_idx >= args.max_batches:
                        break

                    anchor_noisy = batch['anchor_noisy'].to(device)
                    speaker_ids = batch['speaker_id']
                    batch_size = anchor_noisy.shape[0]

                    if anchor_noisy.dim() == 3:
                        anchor_noisy = anchor_noisy.squeeze(1)

                    # Noisy嵌入
                    emb_noisy = speaker_model.encode_batch(anchor_noisy)
                    if isinstance(emb_noisy, tuple):
                        emb_noisy = emb_noisy[0]
                    while emb_noisy.dim() > 2:
                        emb_noisy = emb_noisy.squeeze(1)
                    emb_noisy = F.normalize(emb_noisy, p=2, dim=1)

                    # Enhanced嵌入
                    noisy_3d = anchor_noisy.unsqueeze(1) if anchor_noisy.dim() == 2 else anchor_noisy
                    enhanced = gtcrn.enhance(noisy_3d)
                    if enhanced.dim() == 3:
                        enhanced = enhanced.squeeze(1)
                    emb_enhanced = speaker_model.encode_batch(enhanced)
                    if isinstance(emb_enhanced, tuple):
                        emb_enhanced = emb_enhanced[0]
                    while emb_enhanced.dim() > 2:
                        emb_enhanced = emb_enhanced.squeeze(1)
                    emb_enhanced = F.normalize(emb_enhanced, p=2, dim=1)

                    # Proposed嵌入
                    emb_proposed, weights = confidence_net(emb_noisy, emb_enhanced)
                    all_weights.append(weights.cpu().numpy())

                    # 计算相似度
                    scores_noisy = torch.mm(emb_noisy.cpu(), emb_noisy.cpu().t())
                    scores_enhanced = torch.mm(emb_enhanced.cpu(), emb_enhanced.cpu().t())
                    scores_proposed = torch.mm(emb_proposed.cpu(), emb_proposed.cpu().t())

                    # 标签矩阵
                    labels_matrix = torch.zeros(batch_size, batch_size, dtype=torch.long)
                    for i in range(batch_size):
                        for j in range(batch_size):
                            if speaker_ids[i] == speaker_ids[j]:
                                labels_matrix[i, j] = 1

                    # 上三角
                    mask = torch.triu(torch.ones_like(labels_matrix, dtype=torch.bool), diagonal=1)
                    all_scores['noisy'].extend(scores_noisy[mask].numpy().tolist())
                    all_scores['enhanced'].extend(scores_enhanced[mask].numpy().tolist())
                    all_scores['proposed'].extend(scores_proposed[mask].numpy().tolist())
                    all_labels.extend(labels_matrix[mask].numpy().tolist())

            # 计算EER
            eer_noisy, _ = compute_eer(all_scores['noisy'], all_labels)
            eer_enhanced, _ = compute_eer(all_scores['enhanced'], all_labels)
            eer_proposed, _ = compute_eer(all_scores['proposed'], all_labels)

            avg_weights = np.concatenate(all_weights, axis=0).mean(axis=0)

            results[condition] = {
                'noisy': eer_noisy,
                'enhanced': eer_enhanced,
                'proposed': eer_proposed,
                'w_noisy': float(avg_weights[0]),
                'w_enhanced': float(avg_weights[1])
            }

            print(f"   Noisy:    {eer_noisy:6.2f}%")
            print(f"   Enhanced: {eer_enhanced:6.2f}% ({eer_enhanced - eer_noisy:+.2f}%)")
            print(f"   Proposed: {eer_proposed:6.2f}% ({eer_proposed - eer_noisy:+.2f}%)")
            print(f"   Weights:  noisy={avg_weights[0]:.3f}, enhanced={avg_weights[1]:.3f}")

    # ========== 汇总 ==========
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    avg_noisy = np.mean([r['noisy'] for r in results.values()])
    avg_enhanced = np.mean([r['enhanced'] for r in results.values()])
    avg_proposed = np.mean([r['proposed'] for r in results.values()])

    print(f"\nAverage EER:")
    print(f"   Noisy:    {avg_noisy:.2f}%")
    print(f"   Enhanced: {avg_enhanced:.2f}% ({avg_enhanced - avg_noisy:+.2f}%)")
    print(f"   Proposed: {avg_proposed:.2f}% ({avg_proposed - avg_noisy:+.2f}%)")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    if avg_proposed < avg_noisy and avg_proposed < avg_enhanced:
        print("✅ 置信度网络有效！融合结果优于单独使用noisy或enhanced")
    elif avg_proposed < avg_noisy:
        print("⚠️ 部分有效：融合优于noisy，但不如enhanced")
    elif avg_proposed < avg_enhanced:
        print("⚠️ 部分有效：融合优于enhanced，但不如noisy")
    else:
        print("❌ 置信度网络未有效学习")

    # 保存结果
    import json
    from datetime import datetime
    output_file = f"eval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存: {output_file}")


if __name__ == '__main__':
    main()