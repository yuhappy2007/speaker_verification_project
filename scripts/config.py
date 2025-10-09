# # config.py
# # 项目配置文件
#
# import os
#
# # 项目路径
# PROJECT_ROOT = r"D:\speaker_verification_project"
#
# # 数据路径
# VOXCELEB1_ROOT = os.path.join(PROJECT_ROOT, "data", "voxceleb1", "voxceleb1_complete")
# VOXCELEB1_TRAIN = os.path.join(VOXCELEB1_ROOT, "vox1_dev_wav", "wav")
# VOXCELEB1_TEST = os.path.join(VOXCELEB1_ROOT, "vox1_test_wav", "wav")
# VOXCELEB1_TEST_LIST = os.path.join(VOXCELEB1_ROOT, "veri_test2.txt")
#
# MUSAN_ROOT = os.path.join(PROJECT_ROOT, "data", "musan")
# MUSAN_NOISE = os.path.join(MUSAN_ROOT, "noise")
# MUSAN_MUSIC = os.path.join(MUSAN_ROOT, "music")
# MUSAN_SPEECH = os.path.join(MUSAN_ROOT, "speech")
#
# # 模型保存路径
# CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
# RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
#
# # 创建必要的目录
# os.makedirs(CHECKPOINT_DIR, exist_ok=True)
# os.makedirs(RESULTS_DIR, exist_ok=True)
#
# # 训练参数
# BATCH_SIZE = 32
# LEARNING_RATE = 1e-3
# NUM_EPOCHS = 10
# TRIPLET_MARGIN = 0.25
#
# # 音频参数
# SAMPLE_RATE = 16000
# MAX_AUDIO_LENGTH = 3  # 秒
#
# # SNR范围
# SNR_TRAIN_MIN = -20
# SNR_TRAIN_MAX = 0
# SNR_TEST_LEVELS = [0, -5, -10, -15, -20]
#
# print("Configuration loaded successfully!")
# print(f"Project root: {PROJECT_ROOT}")
# 项目配置文件

import os

# 项目路径
PROJECT_ROOT = r"D:\speaker_verification_project"

# 数据路径
VOXCELEB1_ROOT = os.path.join(PROJECT_ROOT, "data", "voxceleb1", "voxceleb1_complete")
VOXCELEB1_TRAIN = os.path.join(VOXCELEB1_ROOT, "vox1_dev_wav", "wav")
VOXCELEB1_TEST = os.path.join(VOXCELEB1_ROOT, "vox1_test_wav", "wav")
VOXCELEB1_TEST_LIST = os.path.join(VOXCELEB1_ROOT, "veri_test2.txt")

MUSAN_ROOT = os.path.join(PROJECT_ROOT, "data", "musan")
MUSAN_NOISE = os.path.join(MUSAN_ROOT, "noise")
MUSAN_MUSIC = os.path.join(MUSAN_ROOT, "music")
MUSAN_SPEECH = os.path.join(MUSAN_ROOT, "speech")

# 模型保存路径
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# 创建必要的目录
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# 训练参数
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 10
TRIPLET_MARGIN = 0.25

# 音频参数
SAMPLE_RATE = 16000
MAX_AUDIO_LENGTH = 3  # 秒

# SNR范围
SNR_TRAIN_MIN = -20
SNR_TRAIN_MAX = 0
SNR_TEST_MIN = -20  # 添加最小 SNR
SNR_TEST_MAX = 0    # 添加最大 SNR
SNR_TEST_LEVELS = [0, -5, -10, -15, -20]

print("Configuration loaded successfully!")
print(f"Project root: {PROJECT_ROOT}")
