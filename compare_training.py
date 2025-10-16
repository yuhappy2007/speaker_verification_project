"""
对比SupCon和Triplet Loss的训练过程
- 加载两个训练的loss历史
- 生成对比图表
- 分析训练稳定性

用法:
    python compare_training.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json


def load_loss_history(file_path):
    """加载loss历史"""
    losses = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('#') or line.strip() == '':
                continue
            parts = line.strip().split('\t')
            if len(parts) == 2:
                losses.append(float(parts[1]))
    return losses


def analyze_training_stability(losses, name):
    """分析训练稳定性"""
    losses = np.array(losses)

    analysis = {
        'name': name,
        'mean': np.mean(losses),
        'std': np.std(losses),
        'cv': np.std(losses) / np.mean(losses),  # 变异系数(越小越稳定)
        'min': np.min(losses),
        'max': np.max(losses),
        'final_10_avg': np.mean(losses[-10:]),
        'first_10_avg': np.mean(losses[:10]),
    }

    # 计算改进率
    analysis['improvement'] = (
            (analysis['first_10_avg'] - analysis['final_10_avg']) /
            analysis['first_10_avg'] * 100
    )

    return analysis


def plot_loss_comparison(triplet_losses, supcon_losses, save_path):
    """绘制loss对比图"""
    plt.figure(figsize=(14, 5))

    # 子图1: 原始loss曲线
    plt.subplot(1, 2, 1)
    plt.plot(triplet_losses, label='Triplet Loss', alpha=0.7, linewidth=1)
    plt.plot(supcon_losses, label='SupCon Loss', alpha=0.7, linewidth=1)
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 子图2: 平滑后的loss曲线(移动平均)
    plt.subplot(1, 2, 2)
    window = 10
    triplet_smooth = np.convolve(
        triplet_losses,
        np.ones(window) / window,
        mode='valid'
    )
    supcon_smooth = np.convolve(
        supcon_losses,
        np.ones(window) / window,
        mode='valid'
    )

    plt.plot(triplet_smooth, label='Triplet Loss (smoothed)', linewidth=2)
    plt.plot(supcon_smooth, label='SupCon Loss (smoothed)', linewidth=2)
    plt.xlabel('Batch')
    plt.ylabel('Loss (Moving Average)')
    plt.title(f'Smoothed Loss (window={window})')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'📊 Plot saved: {save_path}')


def main():
    print('=' * 80)
    print('📊 TRAINING COMPARISON: SupCon vs Triplet Loss')
    print('=' * 80)

    # 加载loss历史
    triplet_file = Path('checkpoints_fixed_200batch/loss_history.txt')
    supcon_file = Path('checkpoints_supcon_200batch/loss_history_supcon.txt')

    if not triplet_file.exists():
        print(f'❌ Triplet loss file not found: {triplet_file}')
        print('   Please run train_fixed_50batch.py first')
        return

    if not supcon_file.exists():
        print(f'❌ SupCon loss file not found: {supcon_file}')
        print('   Please run train_supcon.py first')
        return

    triplet_losses = load_loss_history(triplet_file)
    supcon_losses = load_loss_history(supcon_file)

    print(f'✅ Loaded Triplet losses: {len(triplet_losses)} batches')
    print(f'✅ Loaded SupCon losses: {len(supcon_losses)} batches')

    # 分析
    print('\n' + '=' * 80)
    print('📈 TRAINING STABILITY ANALYSIS')
    print('=' * 80)

    triplet_analysis = analyze_training_stability(triplet_losses, 'Triplet')
    supcon_analysis = analyze_training_stability(supcon_losses, 'SupCon')

    # 打印对比表格
    print(f'\n{"Metric":<20} {"Triplet Loss":<15} {"SupCon Loss":<15} {"Winner":<10}')
    print('-' * 70)

    metrics = [
        ('Mean Loss', 'mean', 'lower'),
        ('Std Dev', 'std', 'lower'),
        ('CV (stability)', 'cv', 'lower'),
        ('Min Loss', 'min', 'lower'),
        ('Max Loss', 'max', 'lower'),
        ('Final 10 Avg', 'final_10_avg', 'lower'),
        ('Improvement %', 'improvement', 'higher'),
    ]

    wins = {'Triplet': 0, 'SupCon': 0}

    for metric_name, key, better in metrics:
        triplet_val = triplet_analysis[key]
        supcon_val = supcon_analysis[key]

        if better == 'lower':
            winner = 'SupCon ✅' if supcon_val < triplet_val else 'Triplet ✅'
            if supcon_val < triplet_val:
                wins['SupCon'] += 1
            else:
                wins['Triplet'] += 1
        else:
            winner = 'SupCon ✅' if supcon_val > triplet_val else 'Triplet ✅'
            if supcon_val > triplet_val:
                wins['SupCon'] += 1
            else:
                wins['Triplet'] += 1

        print(f'{metric_name:<20} {triplet_val:<15.4f} {supcon_val:<15.4f} {winner:<10}')

    print('-' * 70)
    print(f'{"TOTAL WINS":<20} {wins["Triplet"]:<15} {wins["SupCon"]:<15}')

    # 判断哪个更好
    print('\n' + '=' * 80)
    print('🏆 CONCLUSION')
    print('=' * 80)

    if wins['SupCon'] > wins['Triplet']:
        print('✅ SupCon Loss shows BETTER performance:')
        if supcon_analysis['cv'] < triplet_analysis['cv']:
            print('   - More stable training (lower CV)')
        if supcon_analysis['improvement'] > triplet_analysis['improvement']:
            print('   - Better convergence (higher improvement)')
        if supcon_analysis['final_10_avg'] < triplet_analysis['final_10_avg']:
            print('   - Lower final loss')
    elif wins['Triplet'] > wins['SupCon']:
        print('⚠️  Triplet Loss shows better performance')
        print('   Consider tuning SupCon temperature parameter')
    else:
        print('🤝 Both methods show similar performance')

    # 绘制对比图
    print('\n' + '=' * 80)
    print('📊 Generating comparison plots...')

    plot_path = Path('checkpoints_supcon_200batch/training_comparison.png')
    plot_loss_comparison(triplet_losses, supcon_losses, plot_path)

    # 保存详细报告
    report_path = Path('checkpoints_supcon_200batch/comparison_report.txt')
    with open(report_path, 'w') as f:
        f.write('=' * 80 + '\n')
        f.write('TRAINING COMPARISON REPORT\n')
        f.write('SupCon vs Triplet Loss\n')
        f.write('=' * 80 + '\n\n')

        f.write('TRIPLET LOSS:\n')
        for key, val in triplet_analysis.items():
            f.write(f'  {key}: {val}\n')

        f.write('\nSUPCON LOSS:\n')
        for key, val in supcon_analysis.items():
            f.write(f'  {key}: {val}\n')

        f.write(f'\nWINNER: {"SupCon" if wins["SupCon"] > wins["Triplet"] else "Triplet"}\n')
        f.write(f'Score: SupCon {wins["SupCon"]} vs Triplet {wins["Triplet"]}\n')

    print(f'📄 Report saved: {report_path}')

    print('\n' + '=' * 80)
    print('✅ Analysis complete!')
    print('=' * 80)


if __name__ == '__main__':
    main()