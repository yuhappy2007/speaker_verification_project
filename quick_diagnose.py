#!/usr/bin/env python3
"""
快速诊断脚本：检查训练脚本和当前训练状态
"""

import os
import re

print("=" * 70)
print(" 1. 检查训练脚本中的归一化")
print("=" * 70)

train_script = os.path.expanduser('~/speaker_verification_project/train_confidence_v3.py')

if os.path.exists(train_script):
    with open(train_script, 'r') as f:
        content = f.read()
        lines = content.split('\n')

    # 查找所有F.normalize调用
    print("\n找到的F.normalize调用:")
    print("-" * 50)
    for i, line in enumerate(lines):
        if 'F.normalize' in line and not line.strip().startswith('#'):
            print(f"第{i + 1:4d}行: {line.strip()}")

    # 检查train_epoch函数
    print("\n检查train_epoch函数:")
    print("-" * 50)
    in_train_epoch = False
    squeeze_count = 0
    normalize_after_squeeze = False

    for i, line in enumerate(lines):
        if 'def train_epoch' in line:
            in_train_epoch = True
        if in_train_epoch and 'def ' in line and 'train_epoch' not in line:
            in_train_epoch = False

        if in_train_epoch:
            if 'squeeze' in line and 'emb' in line:
                squeeze_count += 1
                print(f"第{i + 1:4d}行 [squeeze]: {line.strip()}")
            if 'F.normalize' in line and 'emb' in line:
                normalize_after_squeeze = True
                print(f"第{i + 1:4d}行 [normalize]: {line.strip()}")

    if normalize_after_squeeze:
        print("\n✅ train_epoch中有归一化")
    else:
        print("\n❌ train_epoch中缺少归一化!")

    # 检查validate函数
    print("\n检查validate函数:")
    print("-" * 50)
    in_validate = False
    normalize_in_validate = False

    for i, line in enumerate(lines):
        if 'def validate' in line:
            in_validate = True
        if in_validate and 'def ' in line and 'validate' not in line:
            in_validate = False

        if in_validate:
            if 'squeeze' in line and 'emb' in line:
                print(f"第{i + 1:4d}行 [squeeze]: {line.strip()}")
            if 'F.normalize' in line and 'emb' in line:
                normalize_in_validate = True
                print(f"第{i + 1:4d}行 [normalize]: {line.strip()}")

    if normalize_in_validate:
        print("\n✅ validate中有归一化")
    else:
        print("\n❌ validate中缺少归一化!")
else:
    print(f"❌ 找不到: {train_script}")

print("\n" + "=" * 70)
print(" 2. 查看最新训练日志中的权重变化")
print("=" * 70)

import glob

log_files = sorted(glob.glob(os.path.expanduser('~/speaker_verification_project/training_v3_*.log')))

if log_files:
    latest_log = log_files[-1]
    print(f"\n最新日志: {latest_log}")
    print("-" * 50)

    with open(latest_log, 'r') as f:
        lines = f.readlines()

    # 提取权重信息
    weight_data = []
    for line in lines:
        if 'Val Weights' in line:
            match = re.search(r'Noisy: ([\d.]+), Enhanced: ([\d.]+)', line)
            if match:
                weight_data.append({
                    'noisy': float(match.group(1)),
                    'enhanced': float(match.group(2))
                })

    if weight_data:
        print(f"\n权重变化趋势 (共{len(weight_data)}个epoch):")
        print("-" * 30)
        for i, w in enumerate(weight_data):
            bar_noisy = '█' * int(w['noisy'] * 20)
            bar_enhanced = '█' * int(w['enhanced'] * 20)
            print(f"Epoch {i + 1:2d}: Noisy={w['noisy']:.3f} {bar_noisy}")
            print(f"          Enhanced={w['enhanced']:.3f} {bar_enhanced}")
            print()

        # 分析趋势
        if len(weight_data) >= 3:
            first_enhanced = weight_data[0]['enhanced']
            last_enhanced = weight_data[-1]['enhanced']
            change = last_enhanced - first_enhanced

            print(f"\nEnhanced权重变化: {first_enhanced:.3f} → {last_enhanced:.3f} (变化: {change:+.3f})")

            if abs(change) < 0.02:
                print("⚠️ 权重几乎没变化，网络可能没有学到有效的融合策略")
            elif change > 0:
                print("✅ Enhanced权重在增加，网络在学习使用enhanced")
            else:
                print("⚠️ Enhanced权重在减少")
else:
    print("没有找到训练日志")

print("\n" + "=" * 70)
print(" 3. 检查模型checkpoint")
print("=" * 70)

checkpoint_dirs = [
    '~/speaker_verification_project/checkpoints/confidence_net_v3',
    '~/speaker_verification_project/checkpoints/confidence_net',
]

for dir_path in checkpoint_dirs:
    dir_path = os.path.expanduser(dir_path)
    if os.path.exists(dir_path):
        print(f"\n{dir_path}:")
        files = os.listdir(dir_path)
        for f in files:
            full_path = os.path.join(dir_path, f)
            size = os.path.getsize(full_path) / 1024
            print(f"  {f} ({size:.1f} KB)")

print("\n" + "=" * 70)
print(" 结论和建议")
print("=" * 70)
print("""
如果权重稳定在 Noisy:0.80, Enhanced:0.20 左右，可能是因为:

1. 【正常情况】
   - 对于你训练数据的SNR分布，noisy嵌入质量足够好
   - 网络学到了"大部分情况用noisy就行"

2. 【需要检查】
   - 确认归一化已正确添加（见上面的检查结果）
   - 检查GTCRN是否正常工作

3. 【判断是否有效学习】
   最重要的指标是：在评估时，融合后的嵌入是否比单独用noisy或enhanced更好

   运行评估脚本后，比较:
   - baseline (只用noisy): EER = ?
   - enhanced only: EER = ?  
   - fusion (融合后): EER = ?

   如果 fusion EER < baseline EER，说明融合有效！
""")