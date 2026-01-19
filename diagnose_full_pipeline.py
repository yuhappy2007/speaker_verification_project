#!/usr/bin/env python3
"""
全面诊断脚本：检查GTCRN和置信度网络的完整流程
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
import os


def print_section(title):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # ============================================================
    # 1. 检查GTCRN输出
    # ============================================================
    print_section("1. 检查GTCRN Wrapper输出")

    try:
        sys.path.insert(0, os.path.expanduser('~/speaker_verification_project'))
        from gtcrn_wrapper import GTCRNWrapper

        gtcrn = GTCRNWrapper(
            checkpoint_path='checkpoints/gtcrn/gtcrn_best.pth',
            device=device
        )

        # 生成测试音频
        sample_rate = 16000
        duration = 3.0
        t = torch.linspace(0, duration, int(sample_rate * duration))

        # 干净语音（模拟）
        clean = torch.sin(2 * np.pi * 200 * t) * 0.3
        clean += torch.sin(2 * np.pi * 400 * t) * 0.2

        # 添加噪声（不同SNR）
        for snr in [20, 10, 0, -5]:
            noise_power = 10 ** (-snr / 10) * (clean ** 2).mean()
            noise = torch.randn_like(clean) * torch.sqrt(noise_power)
            noisy = clean + noise

            # 增强
            noisy_input = noisy.unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, T]
            with torch.no_grad():
                enhanced = gtcrn(noisy_input)

            # 检查输出
            noisy_np = noisy.cpu().numpy()
            enhanced_np = enhanced.squeeze().cpu().numpy()

            print(f"\nSNR={snr}dB:")
            print(f"  Noisy:    min={noisy_np.min():.4f}, max={noisy_np.max():.4f}, "
                  f"std={noisy_np.std():.4f}, mean={noisy_np.mean():.4f}")
            print(f"  Enhanced: min={enhanced_np.min():.4f}, max={enhanced_np.max():.4f}, "
                  f"std={enhanced_np.std():.4f}, mean={enhanced_np.mean():.4f}")

            # 检查enhanced是否全零或异常
            if enhanced_np.std() < 0.001:
                print(f"  ⚠️ 警告: Enhanced输出几乎为零！")
            elif enhanced_np.std() < noisy_np.std() * 0.1:
                print(f"  ⚠️ 警告: Enhanced输出幅度过小！")
            else:
                print(f"  ✅ Enhanced输出正常")

    except Exception as e:
        print(f"❌ GTCRN检查失败: {e}")
        import traceback
        traceback.print_exc()

    # ============================================================
    # 2. 检查ECAPA-TDNN嵌入
    # ============================================================
    print_section("2. 检查ECAPA-TDNN嵌入和归一化")

    try:
        from speechbrain.inference.speaker import EncoderClassifier

        speaker_model = EncoderClassifier.from_hparams(
            source="pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device": str(device)}
        )

        # 用不同的音频测试
        for snr in [20, 0, -5]:
            noise_power = 10 ** (-snr / 10) * (clean ** 2).mean()
            noise = torch.randn_like(clean) * torch.sqrt(noise_power)
            noisy = clean + noise

            noisy_input = noisy.unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                enhanced = gtcrn(noisy_input)

            # 提取嵌入
            with torch.no_grad():
                # Noisy嵌入
                noisy_2d = noisy_input.squeeze(1)
                emb_noisy = speaker_model.encode_batch(noisy_2d)
                if isinstance(emb_noisy, tuple):
                    emb_noisy = emb_noisy[0]
                while emb_noisy.dim() > 2:
                    emb_noisy = emb_noisy.squeeze(1)

                # Enhanced嵌入
                enhanced_2d = enhanced.squeeze(1)
                emb_enhanced = speaker_model.encode_batch(enhanced_2d)
                if isinstance(emb_enhanced, tuple):
                    emb_enhanced = emb_enhanced[0]
                while emb_enhanced.dim() > 2:
                    emb_enhanced = emb_enhanced.squeeze(1)

            # 检查归一化前
            print(f"\nSNR={snr}dB (归一化前):")
            print(f"  emb_noisy norm: {emb_noisy.norm().item():.4f}")
            print(f"  emb_enhanced norm: {emb_enhanced.norm().item():.4f}")

            # 归一化
            emb_noisy_norm = F.normalize(emb_noisy, p=2, dim=1)
            emb_enhanced_norm = F.normalize(emb_enhanced, p=2, dim=1)

            print(f"  (归一化后)")
            print(f"  emb_noisy norm: {emb_noisy_norm.norm().item():.4f}")
            print(f"  emb_enhanced norm: {emb_enhanced_norm.norm().item():.4f}")

            # 计算相似度
            cos_sim = F.cosine_similarity(emb_noisy_norm, emb_enhanced_norm).item()
            print(f"  余弦相似度: {cos_sim:.4f}")

            if cos_sim > 0.95:
                print(f"  ⚠️ 相似度很高，enhanced和noisy嵌入几乎相同")
            elif cos_sim < 0.3:
                print(f"  ⚠️ 相似度很低，可能有问题")
            else:
                print(f"  ✅ 相似度正常范围")

    except Exception as e:
        print(f"❌ 嵌入检查失败: {e}")
        import traceback
        traceback.print_exc()

    # ============================================================
    # 3. 检查训练脚本中的归一化
    # ============================================================
    print_section("3. 检查训练脚本中是否有归一化")

    train_script = os.path.expanduser('~/speaker_verification_project/train_confidence_v3.py')
    if os.path.exists(train_script):
        with open(train_script, 'r') as f:
            content = f.read()

        # 查找squeeze后面是否有normalize
        lines = content.split('\n')
        found_normalize_after_squeeze = False

        for i, line in enumerate(lines):
            if 'emb_enhanced = emb_enhanced.squeeze' in line:
                # 检查后面几行是否有normalize
                for j in range(1, 5):
                    if i + j < len(lines):
                        if 'F.normalize' in lines[i + j]:
                            found_normalize_after_squeeze = True
                            print(f"✅ 找到归一化 (第{i + j + 1}行): {lines[i + j].strip()}")
                            break

        if not found_normalize_after_squeeze:
            print("❌ 警告: 在squeeze后没有找到F.normalize!")
            print("   请添加以下代码:")
            print("   emb_noisy = F.normalize(emb_noisy, p=2, dim=1)")
            print("   emb_enhanced = F.normalize(emb_enhanced, p=2, dim=1)")

        # 检查validate函数
        if 'def validate' in content:
            validate_start = content.find('def validate')
            validate_section = content[validate_start:validate_start + 3000]
            if 'F.normalize' in validate_section:
                print("✅ validate函数中有归一化")
            else:
                print("⚠️ validate函数中可能缺少归一化")
    else:
        print(f"❌ 找不到训练脚本: {train_script}")

    # ============================================================
    # 4. 检查已保存的模型权重
    # ============================================================
    print_section("4. 检查已训练模型的权重分布")

    checkpoint_path = os.path.expanduser('~/speaker_verification_project/checkpoints/confidence_net_v3/best_model.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)

        print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"Best val loss: {checkpoint.get('best_val_loss', 'N/A')}")

        # 检查attention层的权重
        state_dict = checkpoint.get('model_state_dict', checkpoint)

        for name, param in state_dict.items():
            if 'attention' in name.lower() and 'weight' in name.lower():
                print(f"\n{name}:")
                print(f"  shape: {param.shape}")
                print(f"  mean: {param.mean().item():.6f}")
                print(f"  std: {param.std().item():.6f}")
                print(f"  min: {param.min().item():.6f}")
                print(f"  max: {param.max().item():.6f}")

                if param.std().item() < 0.001:
                    print(f"  ⚠️ 权重几乎没有变化（可能没学到东西）")
    else:
        print(f"模型文件不存在: {checkpoint_path}")
        # 尝试其他路径
        alt_paths = [
            '~/speaker_verification_project/checkpoints/confidence_net/best_model.pth',
            '~/speaker_verification_project/checkpoints/best_model.pth'
        ]
        for p in alt_paths:
            p = os.path.expanduser(p)
            if os.path.exists(p):
                print(f"找到替代模型: {p}")
                break

    # ============================================================
    # 5. 用真实数据测试置信度网络
    # ============================================================
    print_section("5. 用真实数据测试完整流程")

    try:
        from dataset_fixed import VoxCelebMusanDataset

        # 加载几个样本
        test_dataset = VoxCelebMusanDataset(
            voxceleb_dir='data/voxceleb1',
            musan_dir='data/musan',
            split='test',
            snr_range=(-5, 20),  # 宽SNR范围
            return_clean=False
        )

        print(f"数据集大小: {len(test_dataset)}")

        # 测试不同SNR的样本
        results = []
        for i in range(min(10, len(test_dataset))):
            sample = test_dataset[i]
            noisy = sample['noisy'].unsqueeze(0).to(device)  # [1, 1, T]
            snr = sample.get('snr', 'unknown')

            with torch.no_grad():
                # GTCRN增强
                enhanced = gtcrn(noisy)

                # 提取嵌入
                noisy_2d = noisy.squeeze(1)
                enhanced_2d = enhanced.squeeze(1)

                emb_noisy = speaker_model.encode_batch(noisy_2d)
                emb_enhanced = speaker_model.encode_batch(enhanced_2d)

                if isinstance(emb_noisy, tuple):
                    emb_noisy = emb_noisy[0]
                if isinstance(emb_enhanced, tuple):
                    emb_enhanced = emb_enhanced[0]

                while emb_noisy.dim() > 2:
                    emb_noisy = emb_noisy.squeeze(1)
                while emb_enhanced.dim() > 2:
                    emb_enhanced = emb_enhanced.squeeze(1)

                # 归一化
                emb_noisy = F.normalize(emb_noisy, p=2, dim=1)
                emb_enhanced = F.normalize(emb_enhanced, p=2, dim=1)

                cos_sim = F.cosine_similarity(emb_noisy, emb_enhanced).item()

                results.append({
                    'snr': snr,
                    'cos_sim': cos_sim,
                    'noisy_std': noisy.std().item(),
                    'enhanced_std': enhanced.std().item()
                })

        print("\n样本分析:")
        print("-" * 60)
        for r in results:
            print(f"SNR={r['snr']:>5}, cos_sim={r['cos_sim']:.4f}, "
                  f"noisy_std={r['noisy_std']:.4f}, enhanced_std={r['enhanced_std']:.4f}")

        avg_sim = np.mean([r['cos_sim'] for r in results])
        print(f"\n平均余弦相似度: {avg_sim:.4f}")

        if avg_sim > 0.9:
            print("⚠️ 相似度很高 - enhanced和noisy嵌入非常相似")
            print("   这可能解释了为什么权重偏向noisy")

    except Exception as e:
        print(f"❌ 真实数据测试失败: {e}")
        import traceback
        traceback.print_exc()

    # ============================================================
    # 总结
    # ============================================================
    print_section("诊断总结")
    print("""
关于权重偏向noisy (0.80 vs 0.20) 的可能原因:

1. ✅ 正常情况: 
   - 在高SNR (>10dB)时，noisy本身质量就不错
   - 网络正确地学到了"高SNR时不需要太多enhanced"

2. ⚠️ 可能的问题:
   - GTCRN增强效果不够显著
   - enhanced嵌入和noisy嵌入太相似
   - 训练数据SNR分布不均衡

3. 📊 如何判断是否正常学习:
   - 观察不同SNR下的权重变化
   - 低SNR应该有更高的enhanced权重
   - 权重应该随SNR变化而变化

建议: 在评估时按SNR分组统计权重，看是否有自适应性。
""")


if __name__ == '__main__':
    main()