"""
测试配置 - 15种条件
用于评估说话人验证系统在不同噪声环境下的性能
"""

# 测试 SNR 水平（dB）
TEST_SNRS = [0, 5, 10, 15, 20]  # 你的要求：正值 SNR

# 噪声类型
TEST_NOISE_TYPES = ['noise', 'music', 'speech']

# 总测试条件数
TOTAL_CONDITIONS = len(TEST_SNRS) * len(TEST_NOISE_TYPES)  # 15

# 数据路径
VOXCELEB_DIR = 'data/voxceleb1'
MUSAN_DIR = 'data/musan'

# 模型路径
GTCRN_CHECKPOINT = 'gtcrn/checkpoints/model_trained_on_vctk.tar'
WAVLM_PATH = 'D:/WavLM'
SPEAKER_MODEL_PATH = 'checkpoints/speaker_model.pth'
CONFIDENCE_NET_PATH = 'checkpoints/confidence_net.pth'

# 训练配置
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 10

# 说明
"""
测试 SNR 含义：
- 0 dB: 信号和噪声功率相等
- 5 dB: 信号功率是噪声的 ~3.2 倍
- 10 dB: 信号功率是噪声的 10 倍
- 15 dB: 信号功率是噪声的 ~31.6 倍
- 20 dB: 信号功率是噪声的 100 倍

SNR 越高，噪声越小，任务越容易。
0 dB 是最困难的条件。

dataset_fixed.py 使用方法：
```python
from dataset_fixed import VoxCelebMusanDataset

# 训练集（随机 SNR）
train_dataset = VoxCelebMusanDataset(
    voxceleb_dir=VOXCELEB_DIR,
    musan_dir=MUSAN_DIR,
    split='train',
    snr_range=(-20, 0),  # 训练时使用 -20 到 0 dB
    return_clean=True
)

# 测试集（固定 SNR 和噪声类型）
for snr in TEST_SNRS:
    for noise_type in TEST_NOISE_TYPES:
        test_dataset = VoxCelebMusanDataset(
            voxceleb_dir=VOXCELEB_DIR,
            musan_dir=MUSAN_DIR,
            split='test',
            test_snr=snr,  # 固定 SNR
            test_noise_type=noise_type,  # 固定噪声类型
            return_clean=True
        )

        # 评估...
        eer = evaluate(model, test_dataset)
        print(f'{noise_type} @ {snr}dB: EER = {eer:.2f}%')
```
"""