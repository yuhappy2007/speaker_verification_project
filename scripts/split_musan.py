# -*- coding: utf-8 -*-
"""
Split MUSAN corpus into train/test sets (80/20)
论文要求: train和test使用不同的噪声文件
"""

import json
import random
from pathlib import Path

def split_musan(musan_root, split_ratio=0.8, seed=42):
    """
    Split MUSAN into disjoint train and test sets
    
    Args:
        musan_root: MUSAN数据集根目录
        split_ratio: 训练集比例(默认0.8)
        seed: 随机种子(默认42,确保可复现)
    """
    random.seed(seed)
    musan_root = Path(musan_root)
    output_file = musan_root / 'musan_split.json'
    
    split_data = {
        'noise': {'train': [], 'test': []},
        'music': {'train': [], 'test': []},
        'speech': {'train': [], 'test': []}
    }
    
    print("="*60)
    print("SPLITTING MUSAN CORPUS")
    print("="*60)
    print(f"Split ratio: {split_ratio:.0%} train / {1-split_ratio:.0%} test")
    print(f"Random seed: {seed}")
    print("="*60)
    
    for noise_type in ['noise', 'music', 'speech']:
        type_dir = musan_root / noise_type
        
        if not type_dir.exists():
            print(f"⚠️  {noise_type} directory not found: {type_dir}")
            continue
        
        # 收集所有wav文件
        all_files = []
        for wav_file in type_dir.rglob('*.wav'):
            rel_path = wav_file.relative_to(musan_root).as_posix()
            all_files.append(rel_path)
        
        # 随机打乱
        random.shuffle(all_files)
        
        # 划分
        split_idx = int(len(all_files) * split_ratio)
        train_files = sorted(all_files[:split_idx])
        test_files = sorted(all_files[split_idx:])
        
        split_data[noise_type]['train'] = train_files
        split_data[noise_type]['test'] = test_files
        
        print(f"\n{noise_type.upper()}:")
        print(f"  Total:  {len(all_files):4d} files")
        print(f"  Train:  {len(train_files):4d} files ({len(train_files)/len(all_files)*100:.1f}%)")
        print(f"  Test:   {len(test_files):4d} files ({len(test_files)/len(all_files)*100:.1f}%)")
    
    # 保存划分结果
    with open(output_file, 'w') as f:
        json.dump(split_data, f, indent=2)
    
    print("\n" + "="*60)
    print(f"✓ Split saved to: {output_file}")
    
    # 总结
    total_train = sum(len(v['train']) for v in split_data.values())
    total_test = sum(len(v['test']) for v in split_data.values())
    print(f"\nSummary:")
    print(f"  Total train files: {total_train}")
    print(f"  Total test files:  {total_test}")
    print(f"  Actual ratio: {total_train/(total_train+total_test):.1%} train")
    print("="*60)

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        musan_path = sys.argv[1]
    else:
        musan_path = 'data/musan'
    
    split_musan(musan_path)
