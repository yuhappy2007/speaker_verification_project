"""
评估脚本 - 使用预训练ECAPA-TDNN（无可视化版本）

使用与训练时相同的ECAPA-TDNN加载方式
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
from sklearn.metrics import roc_curve

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'scripts'))

from gtcrn_wrapper_fixed import GTCRNWrapper
from dataset_fixed import VoxCelebMusanDataset
from fixed_snr_dataset import collate_fn_fixed_length


def load_ecapa_tdnn(model_path, device):
    """
    加载预训练的ECAPA-TDNN模型

    使用与训练脚本相同的加载方式
    """
    print(f'Loading ECAPA-TDNN from: {model_path}')

    try:
        # 尝试使用 SpeechBrain 的方式加载
        from speechbrain.pretrained import EncoderClassifier

        classifier = EncoderClassifier.from_hparams(
            source=model_path,
            savedir=model_path,
            run_opts={"device": str(device)}
        )

        # 提取encoder
        speaker_model = classifier.encode_batch
        print('✅ Loaded ECAPA-TDNN using SpeechBrain')
        return speaker_model, 'speechbrain'

    except Exception as e:
        print(f'⚠️  SpeechBrain加载失败: {e}')
        print('   尝试直接加载PyTorch模型...')

        try:
            # 尝试直接加载 .pth 文件
            checkpoint = torch.load(Path(model_path) / 'embedding_model.ckpt',
                                    map_location=device)

            from ecapa_tdnn import ECAPA_TDNN
            speaker_model = ECAPA_TDNN(
                C=1024,
                input_size=80,
                channels=[512, 512, 512, 512, 1536],
                emb_size=192
            ).to(device)

            speaker_model.load_state_dict(checkpoint)
            speaker_model.eval()
            print('✅ Loaded ECAPA-TDNN from checkpoint')
            return speaker_model, 'pytorch'

        except Exception as e2:
            print(f'❌ 无法加载ECAPA-TDNN: {e2}')
            raise


def compute_eer(labels, scores):
    """计算 Equal Error Rate (EER)"""
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    return eer * 100, eer_threshold


def extract_embeddings_speechbrain(encode_fn, audio, device):
    """使用 SpeechBrain 的方式提取嵌入"""
    with torch.no_grad():
        # SpeechBrain 需要 [batch, samples] 格式
        if audio.dim() == 3:  # [batch, 1, samples]
            audio = audio.squeeze(1)  # → [batch, samples]

        # 确保在正确设备上
        audio = audio.to(device)

        # 使用 encode_batch
        embeddings = encode_fn(audio)

        # 结果可能是tuple，取第一个
        if isinstance(embeddings, tuple):
            embeddings = embeddings[0]

        return embeddings


def extract_embeddings_pytorch(model, audio, device):
    """使用 PyTorch 模型提取嵌入"""
    with torch.no_grad():
        audio = audio.to(device)
        embeddings = model(audio)
    return embeddings


def compute_similarity_scores(emb1, emb2):
    """计算余弦相似度"""
    emb1 = torch.nn.functional.normalize(emb1, p=2, dim=1)
    emb2 = torch.nn.functional.normalize(emb2, p=2, dim=1)
    scores = torch.sum(emb1 * emb2, dim=1)
    return scores.cpu().numpy()


def evaluate_condition(gtcrn, speaker_model, model_type, confidence_net,
                       dataloader, device, mode='noisy'):
    """
    在单一条件下评估

    Args:
        mode: 'noisy', 'gtcrn', 'proposed'
    """
    all_scores = []
    all_labels = []

    # 选择正确的embedding提取函数
    if model_type == 'speechbrain':
        extract_emb = lambda x: extract_embeddings_speechbrain(speaker_model, x, device)
    else:
        extract_emb = lambda x: extract_embeddings_pytorch(speaker_model, x, device)

    for batch in tqdm(dataloader, desc=f'Evaluating {mode}'):
        anchor_noisy = batch['anchor_noisy'].to(device)
        positive_noisy = batch['positive_noisy'].to(device)
        negative_noisy = batch['negative_noisy'].to(device)

        if mode == 'noisy':
            anchor = anchor_noisy
            positive = positive_noisy
            negative = negative_noisy

        elif mode == 'gtcrn':
            with torch.no_grad():
                anchor = gtcrn.enhance(anchor_noisy)
                positive = gtcrn.enhance(positive_noisy)
                negative = gtcrn.enhance(negative_noisy)

        elif mode == 'proposed':
            with torch.no_grad():
                # GTCRN 增强
                anchor_enh = gtcrn.enhance(anchor_noisy)
                positive_enh = gtcrn.enhance(positive_noisy)
                negative_enh = gtcrn.enhance(negative_noisy)

                # 提取两种嵌入
                emb_anchor_noisy = extract_emb(anchor_noisy)
                emb_anchor_enh = extract_emb(anchor_enh)
                emb_pos_noisy = extract_emb(positive_noisy)
                emb_pos_enh = extract_emb(positive_enh)
                emb_neg_noisy = extract_emb(negative_noisy)
                emb_neg_enh = extract_emb(negative_enh)

                # 置信度网络融合
                emb_anchor = confidence_net(emb_anchor_noisy, emb_anchor_enh)
                emb_positive = confidence_net(emb_pos_noisy, emb_pos_enh)
                emb_negative = confidence_net(emb_neg_noisy, emb_neg_enh)

                # 计算相似度
                pos_scores = compute_similarity_scores(emb_anchor, emb_positive)
                neg_scores = compute_similarity_scores(emb_anchor, emb_negative)

                all_scores.extend(pos_scores.tolist())
                all_scores.extend(neg_scores.tolist())
                all_labels.extend([1] * len(pos_scores))
                all_labels.extend([0] * len(neg_scores))
                continue

        # noisy 和 gtcrn 模式
        if mode in ['noisy', 'gtcrn']:
            emb_anchor = extract_emb(anchor)
            emb_positive = extract_emb(positive)
            emb_negative = extract_emb(negative)

            pos_scores = compute_similarity_scores(emb_anchor, emb_positive)
            neg_scores = compute_similarity_scores(emb_anchor, emb_negative)

            all_scores.extend(pos_scores.tolist())
            all_scores.extend(neg_scores.tolist())
            all_labels.extend([1] * len(pos_scores))
            all_labels.extend([0] * len(neg_scores))

    eer, _ = compute_eer(np.array(all_labels), np.array(all_scores))
    return eer


def main(args):
    print('=' * 80)
    print('Evaluation on 15 Test Conditions')
    print('=' * 80)
    print(f'SNR values: {args.snr_values}')
    print(f'Noise types: {args.noise_types}')
    print(f'Total conditions: {len(args.snr_values) * len(args.noise_types)}')

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

    # ECAPA-TDNN（使用与训练时相同的方式）
    speaker_model, model_type = load_ecapa_tdnn(args.speaker_model_path, device)

    # 置信度网络
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
            print(f'⚠️  Confidence checkpoint not found!')
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
                gtcrn, speaker_model, model_type, None, test_loader, device, mode='noisy'
            )
            results['noisy'][condition] = eer_noisy
            print(f'Noisy:    EER = {eer_noisy:.2f}%')

            # 评估 GTCRN
            eer_gtcrn = evaluate_condition(
                gtcrn, speaker_model, model_type, None, test_loader, device, mode='gtcrn'
            )
            results['gtcrn'][condition] = eer_gtcrn
            improvement_gtcrn = ((eer_noisy - eer_gtcrn) / eer_noisy) * 100
            print(f'GTCRN:    EER = {eer_gtcrn:.2f}% (↓{improvement_gtcrn:.1f}%)')

            # 评估 Proposed
            if args.evaluate_proposed:
                eer_proposed = evaluate_condition(
                    gtcrn, speaker_model, model_type, confidence_net, test_loader, device, mode='proposed'
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

    # 打印结果表
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
                        default='pretrained_models/spkrec-ecapa-voxceleb',
                        help='Path to pretrained ECAPA-TDNN model')
    parser.add_argument('--confidence_checkpoint', type=str,
                        default='checkpoints/confidence/confidence_net_final.pth')
    parser.add_argument('--embedding_dim', type=int, default=192)
    parser.add_argument('--evaluate_proposed', action='store_true')

    # 评估
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)

    # 输出
    parser.add_argument('--output_dir', type=str, default='evaluation_results')

    args = parser.parse_args()
    main(args)