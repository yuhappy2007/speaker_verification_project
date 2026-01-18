#!/usr/bin/env python3
"""
评估置信度网络V3效果
比较三种方法的EER：
1. 纯noisy嵌入
2. 纯enhanced嵌入
3. 置信度网络融合嵌入
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

try:
    from sklearn.metrics import roc_curve
except ImportError:
    # 如果sklearn有问题，使用简单实现
    def roc_curve(y_true, y_score, pos_label=1):
        """简单的ROC曲线计算"""
        y_true = np.array(y_true)
        y_score = np.array(y_score)

        # 排序
        sorted_indices = np.argsort(y_score)[::-1]
        y_true_sorted = y_true[sorted_indices]
        y_score_sorted = y_score[sorted_indices]

        # 计算TPR和FPR
        thresholds = np.unique(y_score_sorted)
        tpr_list, fpr_list, thresh_list = [], [], []

        total_pos = np.sum(y_true == pos_label)
        total_neg = np.sum(y_true != pos_label)

        for thresh in thresholds:
            pred_pos = y_score >= thresh
            tp = np.sum((pred_pos) & (y_true == pos_label))
            fp = np.sum((pred_pos) & (y_true != pos_label))

            tpr = tp / total_pos if total_pos > 0 else 0
            fpr = fp / total_neg if total_neg > 0 else 0

            tpr_list.append(tpr)
            fpr_list.append(fpr)
            thresh_list.append(thresh)

        return np.array(fpr_list), np.array(tpr_list), np.array(thresh_list)
from pathlib import Path
from tqdm import tqdm
import argparse

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'scripts'))


class ConfidenceNetworkV3(nn.Module):
    """
    置信度网络V3 - 与训练脚本完全一致
    """

    def __init__(self, embedding_dim=192, hidden_dim=256):
        super().__init__()
        self.embedding_dim = embedding_dim

        # 嵌入差异编码器
        # 输入: |emb_noisy - emb_enhanced| 和 emb_noisy * emb_enhanced
        self.diff_encoder = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # Attention机制
        # 输入: emb_noisy(192) + emb_enhanced(192) + diff_feat(32) = 416
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim * 2 + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

        # 融合网络
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
        # 计算差异特征
        diff = torch.abs(emb_noisy - emb_enhanced)
        prod = emb_noisy * emb_enhanced
        diff_input = torch.cat([diff, prod], dim=1)
        diff_feat = self.diff_encoder(diff_input)

        # 计算attention权重
        concat = torch.cat([emb_noisy, emb_enhanced, diff_feat], dim=1)
        attn_logits = self.attention(concat)
        weights = F.softmax(attn_logits, dim=1)
        w_noisy = weights[:, 0:1]
        w_enhanced = weights[:, 1:2]

        # 加权融合
        weighted = w_noisy * emb_noisy + w_enhanced * emb_enhanced

        # 残差融合
        fused = self.fusion(torch.cat([emb_noisy, emb_enhanced], dim=1))
        output = weighted + 0.1 * fused

        return F.normalize(output, p=2, dim=1), weights


def compute_eer(scores, labels):
    """计算EER"""
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    eer = fpr[eer_idx] * 100
    return eer, thresholds[eer_idx]


def collate_fn_fixed(batch, target_length=48000):
    """固定长度的collate函数"""
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


def evaluate_eer(gtcrn, speaker_model, confidence_net, dataloader, device, mode='noisy'):
    """评估指定模式的EER"""
    all_scores = []
    all_labels = []
    all_weights = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {mode}", leave=False):
            anchor_noisy = batch['anchor_noisy'].to(device)
            speaker_ids = batch['speaker_id']
            batch_size = anchor_noisy.shape[0]

            # 处理音频维度
            if anchor_noisy.dim() == 3:
                anchor_noisy = anchor_noisy.squeeze(1)

            if mode == 'noisy':
                # 只用noisy
                embeddings = speaker_model.encode_batch(anchor_noisy)
                if isinstance(embeddings, tuple):
                    embeddings = embeddings[0]
                while embeddings.dim() > 2:
                    embeddings = embeddings.squeeze(1)
                embeddings = F.normalize(embeddings, p=2, dim=1)

            elif mode == 'enhanced':
                # 只用enhanced
                enhanced = gtcrn.enhance(anchor_noisy.unsqueeze(1) if anchor_noisy.dim() == 2 else anchor_noisy)
                if enhanced.dim() == 3:
                    enhanced = enhanced.squeeze(1)
                embeddings = speaker_model.encode_batch(enhanced)
                if isinstance(embeddings, tuple):
                    embeddings = embeddings[0]
                while embeddings.dim() > 2:
                    embeddings = embeddings.squeeze(1)
                embeddings = F.normalize(embeddings, p=2, dim=1)

            elif mode == 'proposed':
                # 置信度网络融合
                # 获取noisy嵌入
                emb_noisy = speaker_model.encode_batch(anchor_noisy)
                if isinstance(emb_noisy, tuple):
                    emb_noisy = emb_noisy[0]
                while emb_noisy.dim() > 2:
                    emb_noisy = emb_noisy.squeeze(1)
                emb_noisy = F.normalize(emb_noisy, p=2, dim=1)

                # 获取enhanced嵌入
                enhanced = gtcrn.enhance(anchor_noisy.unsqueeze(1) if anchor_noisy.dim() == 2 else anchor_noisy)
                if enhanced.dim() == 3:
                    enhanced = enhanced.squeeze(1)
                emb_enhanced = speaker_model.encode_batch(enhanced)
                if isinstance(emb_enhanced, tuple):
                    emb_enhanced = emb_enhanced[0]
                while emb_enhanced.dim() > 2:
                    emb_enhanced = emb_enhanced.squeeze(1)
                emb_enhanced = F.normalize(emb_enhanced, p=2, dim=1)

                # 置信度网络融合
                embeddings, weights = confidence_net(emb_noisy, emb_enhanced)
                all_weights.append(weights.cpu().numpy())

            # 计算相似度矩阵
            emb_cpu = embeddings.cpu()
            scores_matrix = torch.mm(emb_cpu, emb_cpu.t())

            # 构建标签矩阵
            labels_matrix = torch.zeros(batch_size, batch_size, dtype=torch.long)
            for i in range(batch_size):
                for j in range(batch_size):
                    if speaker_ids[i] == speaker_ids[j]:
                        labels_matrix[i, j] = 1

            # 取上三角（不包含对角线）
            mask = torch.triu(torch.ones_like(labels_matrix, dtype=torch.bool), diagonal=1)
            all_scores.extend(scores_matrix[mask].numpy().tolist())
            all_labels.extend(labels_matrix[mask].numpy().tolist())

    eer, threshold = compute_eer(np.array(all_scores), np.array(all_labels))

    avg_weights = None
    if all_weights:
        avg_weights = np.concatenate(all_weights, axis=0).mean(axis=0)

    return eer, threshold, avg_weights


def main():
    parser = argparse.ArgumentParser(description='Evaluate Confidence Network V3')
    parser.add_argument('--voxceleb_dir', type=str, required=True)
    parser.add_argument('--musan_dir', type=str, required=True)
    parser.add_argument('--gtcrn_checkpoint', type=str, required=True)
    parser.add_argument('--speaker_model_path', type=str, required=True)
    parser.add_argument('--confidence_checkpoint', type=str, required=True)
    parser.add_argument('--snr_values', type=int, nargs='+', default=[-5, 0, 5, 10, 15, 20])
    parser.add_argument('--noise_types', type=str, nargs='+', default=['noise', 'music', 'speech'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 加载模型
    print("\n" + "=" * 70)
    print("Loading models...")
    print("=" * 70)

    # 1. GTCRN
    from gtcrn_wrapper_fixed import GTCRNWrapper
    gtcrn = GTCRNWrapper(checkpoint_path=args.gtcrn_checkpoint, device=device)

    # 2. ECAPA-TDNN
    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError:
        from speechbrain.pretrained import EncoderClassifier

    speaker_model = EncoderClassifier.from_hparams(
        source=args.speaker_model_path,
        savedir=args.speaker_model_path,
        run_opts={"device": device}
    )

    # 3. 置信度网络V3
    confidence_net = ConfidenceNetworkV3(embedding_dim=192, hidden_dim=256).to(device)
    checkpoint = torch.load(args.confidence_checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        confidence_net.load_state_dict(checkpoint['model_state_dict'])
    else:
        confidence_net.load_state_dict(checkpoint)
    confidence_net.eval()
    print(f"Loaded confidence network from {args.confidence_checkpoint}")

    # 数据集
    from dataset_fixed import VoxCelebMusanDataset

    # 结果存储
    results = {}

    print("\n" + "=" * 70)
    print("Starting Evaluation")
    print("=" * 70)

    for noise_type in args.noise_types:
        for snr in args.snr_values:
            condition = f"{noise_type}_{snr}dB"
            print(f"\n>>> {condition}")

            # 创建测试数据集
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

            # 评估三种方法
            eer_noisy, _, _ = evaluate_eer(gtcrn, speaker_model, confidence_net, test_loader, device, 'noisy')
            eer_enhanced, _, _ = evaluate_eer(gtcrn, speaker_model, confidence_net, test_loader, device, 'enhanced')
            eer_proposed, _, weights = evaluate_eer(gtcrn, speaker_model, confidence_net, test_loader, device,
                                                    'proposed')

            results[condition] = {
                'noisy': eer_noisy,
                'enhanced': eer_enhanced,
                'proposed': eer_proposed,
                'weights': weights.tolist() if weights is not None else None
            }

            # 打印结果
            print(f"   Noisy:    {eer_noisy:6.2f}%")
            print(f"   Enhanced: {eer_enhanced:6.2f}% (vs noisy: {eer_enhanced - eer_noisy:+.2f}%)")
            print(f"   Proposed: {eer_proposed:6.2f}% (vs noisy: {eer_proposed - eer_noisy:+.2f}%)")
            if weights is not None:
                print(f"   Weights:  noisy={weights[0]:.3f}, enhanced={weights[1]:.3f}")

    # 汇总
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    avg_noisy = np.mean([r['noisy'] for r in results.values()])
    avg_enhanced = np.mean([r['enhanced'] for r in results.values()])
    avg_proposed = np.mean([r['proposed'] for r in results.values()])

    print(f"\nAverage EER across all conditions:")
    print(f"   Noisy:    {avg_noisy:.2f}%")
    print(f"   Enhanced: {avg_enhanced:.2f}% (vs noisy: {avg_enhanced - avg_noisy:+.2f}%)")
    print(f"   Proposed: {avg_proposed:.2f}% (vs noisy: {avg_proposed - avg_noisy:+.2f}%)")

    # 判断置信度网络是否有效
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    if avg_proposed < avg_noisy and avg_proposed < avg_enhanced:
        print("✅ 置信度网络有效！融合结果优于单独使用noisy或enhanced")
    elif avg_proposed < avg_noisy:
        print("⚠️ 置信度网络部分有效：融合结果优于noisy，但不如enhanced")
    elif avg_proposed < avg_enhanced:
        print("⚠️ 置信度网络部分有效：融合结果优于enhanced，但不如noisy")
    else:
        print("❌ 置信度网络未有效学习：融合结果不如单独使用noisy或enhanced")

    # 保存结果
    import json
    from datetime import datetime
    output_file = f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存到: {output_file}")


if __name__ == '__main__':
    main()