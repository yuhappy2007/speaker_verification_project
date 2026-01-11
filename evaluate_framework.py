"""
评估脚本 - 15种测试条件

测试配置：
- SNR: -5, 0, 5, 10, 15 dB（5个值）
- 噪声: noise, music, speech（3种）
- 总条件: 5 × 3 = 15 种

评估3种方案：
1. Noisy - 直接使用 noisy 音频
2. GTCRN - 使用 GTCRN 增强音频
3. Proposed - 使用置信度网络融合

输出：
- EER (Equal Error Rate) 对比表
- 性能提升百分比
- 可视化图表

用法：
    python evaluate_framework.py --voxceleb_dir data/voxceleb1 --musan_dir data/musan
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import argparse
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'scripts'))

from gtcrn_wrapper_fixed import GTCRNWrapper
from dataset_fixed import VoxCelebMusanDataset
from fixed_snr_dataset import collate_fn_fixed_length


def compute_eer(labels, scores):
    """
    计算 Equal Error Rate (EER)

    Args:
        labels: 真实标签 (1=same speaker, 0=different speaker)
        scores: 相似度分数（越高越相似）

    Returns:
        eer: Equal Error Rate (百分比)
        threshold: EER 对应的阈值
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr

    # 找到 FPR = FNR 的点
    eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]

    return eer * 100, eer_threshold


def extract_embeddings(model, audio, device):
    """
    提取嵌入向量

    Args:
        model: 说话人模型
        audio: [batch, 1, samples]

    Returns:
        embeddings: [batch, embedding_dim]
    """
    with torch.no_grad():
        audio = audio.to(device)
        embeddings = model(audio)
    return embeddings


def compute_similarity_scores(emb1, emb2):
    """
    计算余弦相似度

    Args:
        emb1, emb2: [batch, embedding_dim]

    Returns:
        scores: [batch]
    """
    # L2 归一化
    emb1 = torch.nn.functional.normalize(emb1, p=2, dim=1)
    emb2 = torch.nn.functional.normalize(emb2, p=2, dim=1)

    # 余弦相似度
    scores = torch.sum(emb1 * emb2, dim=1)

    return scores.cpu().numpy()


def evaluate_condition(gtcrn, speaker_model, confidence_net, dataloader, device, mode='noisy'):
    """
    在单一条件下评估

    Args:
        mode: 'noisy', 'gtcrn', 'proposed'

    Returns:
        eer: Equal Error Rate
    """
    all_scores = []
    all_labels = []

    for batch in tqdm(dataloader, desc=f'Evaluating {mode}'):
        # 获取数据
        anchor_noisy = batch['anchor_noisy'].to(device)
        positive_noisy = batch['positive_noisy'].to(device)
        negative_noisy = batch['negative_noisy'].to(device)

        # 根据模式选择音频
        if mode == 'noisy':
            # 直接使用 noisy 音频
            anchor = anchor_noisy
            positive = positive_noisy
            negative = negative_noisy

        elif mode == 'gtcrn':
            # 使用 GTCRN 增强
            with torch.no_grad():
                anchor = gtcrn.enhance(anchor_noisy)
                positive = gtcrn.enhance(positive_noisy)
                negative = gtcrn.enhance(negative_noisy)

        elif mode == 'proposed':
            # 使用置信度网络
            with torch.no_grad():
                # GTCRN 增强
                anchor_enh = gtcrn.enhance(anchor_noisy)
                positive_enh = gtcrn.enhance(positive_noisy)
                negative_enh = gtcrn.enhance(negative_noisy)

                # 提取两种嵌入
                emb_anchor_noisy = speaker_model(anchor_noisy)
                emb_anchor_enh = speaker_model(anchor_enh)
                emb_pos_noisy = speaker_model(positive_noisy)
                emb_pos_enh = speaker_model(positive_enh)
                emb_neg_noisy = speaker_model(negative_noisy)
                emb_neg_enh = speaker_model(negative_enh)

                # 融合
                emb_anchor = confidence_net(emb_anchor_noisy, emb_anchor_enh)
                emb_positive = confidence_net(emb_pos_noisy, emb_pos_enh)
                emb_negative = confidence_net(emb_neg_noisy, emb_neg_enh)

                # 计算相似度
                pos_scores = compute_similarity_scores(emb_anchor, emb_positive)
                neg_scores = compute_similarity_scores(emb_anchor, emb_negative)

                all_scores.extend(pos_scores.tolist())
                all_scores.extend(neg_scores.tolist())
                all_labels.extend([1] * len(pos_scores))  # Positive pairs
                all_labels.extend([0] * len(neg_scores))  # Negative pairs

                continue  # 跳过下面的嵌入提取

        # 对于 noisy 和 gtcrn 模式
        if mode in ['noisy', 'gtcrn']:
            # 提取嵌入
            emb_anchor = extract_embeddings(speaker_model, anchor, device)
            emb_positive = extract_embeddings(speaker_model, positive, device)
            emb_negative = extract_embeddings(speaker_model, negative, device)

            # 计算相似度
            pos_scores = compute_similarity_scores(emb_anchor, emb_positive)
            neg_scores = compute_similarity_scores(emb_anchor, emb_negative)

            all_scores.extend(pos_scores.tolist())
            all_scores.extend(neg_scores.tolist())
            all_labels.extend([1] * len(pos_scores))
            all_labels.extend([0] * len(neg_scores))

    # 计算 EER
    eer, _ = compute_eer(np.array(all_labels), np.array(all_scores))

    return eer


def main(args):
    print('=' * 80)
    print('Evaluation on 15 Test Conditions')
    print('=' * 80)
    print(f'SNR values: {args.snr_values}')
    print(f'Noise types: {args.noise_types}')
    print(f'Total conditions: {len(args.snr_values) * len(args.noise_types)}')

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # 加载模型
    print('\nLoading models...')

    # GTCRN
    gtcrn = GTCRNWrapper(
        checkpoint_path=args.gtcrn_checkpoint,
        device=device,
        freeze=True
    )
    print(f'✅ Loaded GTCRN: {args.gtcrn_checkpoint}')

    # 说话人模型
    # TODO: 替换为实际模型
    print('⚠️  Using dummy speaker model! Replace with actual ECAPA-TDNN')

    class DummySpeakerModel(nn.Module):
        def __init__(self, embedding_dim=192):
            super().__init__()
            self.conv = nn.Conv1d(1, 64, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Linear(64, embedding_dim)

        def forward(self, x):
            x = self.conv(x)
            x = self.pool(x).squeeze(-1)
            x = self.fc(x)
            return nn.functional.normalize(x, p=2, dim=1)

    speaker_model = DummySpeakerModel(embedding_dim=args.embedding_dim).to(device)
    speaker_model.eval()

    # 置信度网络（如果评估 proposed）
    confidence_net = None
    if args.evaluate_proposed:
        from train_confidence_gtcrn import ConfidenceNetwork
        confidence_net = ConfidenceNetwork(
            embedding_dim=args.embedding_dim,
            hidden_dim=256
        ).to(device)

        if Path(args.confidence_checkpoint).exists():
            checkpoint = torch.load(args.confidence_checkpoint, map_location=device)
            confidence_net.load_state_dict(checkpoint['model_state_dict'])
            confidence_net.eval()
            print(f'✅ Loaded confidence network: {args.confidence_checkpoint}')
        else:
            print(f'⚠️  Confidence checkpoint not found: {args.confidence_checkpoint}')
            print('   Will only evaluate Noisy and GTCRN')
            args.evaluate_proposed = False

    # 评估所有条件
    results = {
        'noisy': {},
        'gtcrn': {},
        'proposed': {}
    }

    print('\nStarting evaluation...')
    print('=' * 80)

    for snr in args.snr_values:
        for noise_type in args.noise_types:
            condition = f'{noise_type}_{snr}dB'
            print(f'\n[{condition}]')
            print('-' * 80)

            # 创建测试集
            test_dataset = VoxCelebMusanDataset(
                voxceleb_dir=args.voxceleb_dir,
                musan_dir=args.musan_dir,
                split='test',
                test_snr=snr,
                test_noise_type=noise_type,
                return_clean=False
            )

            test_loader = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                collate_fn=lambda batch: collate_fn_fixed_length(batch, target_length=48000)
            )

            print(f'Test samples: {len(test_dataset)}')

            # 评估 Noisy
            eer_noisy = evaluate_condition(
                gtcrn, speaker_model, None, test_loader, device, mode='noisy'
            )
            results['noisy'][condition] = eer_noisy
            print(f'Noisy:    EER = {eer_noisy:.2f}%')

            # 评估 GTCRN
            eer_gtcrn = evaluate_condition(
                gtcrn, speaker_model, None, test_loader, device, mode='gtcrn'
            )
            results['gtcrn'][condition] = eer_gtcrn
            improvement_gtcrn = ((eer_noisy - eer_gtcrn) / eer_noisy) * 100
            print(f'GTCRN:    EER = {eer_gtcrn:.2f}% (↓{improvement_gtcrn:.1f}%)')

            # 评估 Proposed
            if args.evaluate_proposed:
                eer_proposed = evaluate_condition(
                    gtcrn, speaker_model, confidence_net, test_loader, device, mode='proposed'
                )
                results['proposed'][condition] = eer_proposed
                improvement_proposed = ((eer_noisy - eer_proposed) / eer_noisy) * 100
                print(f'Proposed: EER = {eer_proposed:.2f}% (↓{improvement_proposed:.1f}%)')

    # 保存结果
    print('\n' + '=' * 80)
    print('Saving results...')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = output_dir / f'evaluation_results_{timestamp}.json'

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'✅ Results saved: {results_file}')

    # 创建对比表
    print('\n' + '=' * 80)
    print('Results Summary')
    print('=' * 80)
    print(f'{"Condition":<20} {"Noisy":<12} {"GTCRN":<12} {"Proposed":<12}')
    print('-' * 80)

    for snr in args.snr_values:
        for noise_type in args.noise_types:
            condition = f'{noise_type}_{snr}dB'
            noisy_eer = results['noisy'][condition]
            gtcrn_eer = results['gtcrn'][condition]

            row = f'{condition:<20} {noisy_eer:>10.2f}% {gtcrn_eer:>10.2f}%'

            if args.evaluate_proposed:
                proposed_eer = results['proposed'][condition]
                row += f' {proposed_eer:>10.2f}%'

            print(row)

    # 计算平均
    print('-' * 80)
    avg_noisy = np.mean(list(results['noisy'].values()))
    avg_gtcrn = np.mean(list(results['gtcrn'].values()))

    row = f'{"Average":<20} {avg_noisy:>10.2f}% {avg_gtcrn:>10.2f}%'

    if args.evaluate_proposed:
        avg_proposed = np.mean(list(results['proposed'].values()))
        row += f' {avg_proposed:>10.2f}%'

    print(row)
    print('=' * 80)

    print('\n✅ Evaluation complete!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate on 15 Test Conditions')

    # 数据
    parser.add_argument('--voxceleb_dir', type=str, required=True)
    parser.add_argument('--musan_dir', type=str, required=True)
    parser.add_argument('--snr_values', type=int, nargs='+',
                        default=[-5, 0, 5, 10, 15])
    parser.add_argument('--noise_types', type=str, nargs='+',
                        default=['noise', 'music', 'speech'])

    # 模型
    parser.add_argument('--gtcrn_checkpoint', type=str,
                        default='checkpoints/gtcrn/gtcrn_best.pth')
    parser.add_argument('--speaker_model_path', type=str,
                        default='checkpoints/speaker_model.pth')
    parser.add_argument('--confidence_checkpoint', type=str,
                        default='checkpoints/confidence/confidence_net_best.pth')
    parser.add_argument('--embedding_dim', type=int, default=192)
    parser.add_argument('--evaluate_proposed', action='store_true',
                        help='Evaluate proposed method (requires trained confidence network)')

    # 评估
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)

    # 输出
    parser.add_argument('--output_dir', type=str, default='evaluation_results')

    args = parser.parse_args()
    main(args)