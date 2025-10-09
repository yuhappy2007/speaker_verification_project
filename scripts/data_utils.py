# data_utils.py
# 数据加载和预处理工具

import os
import random
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import config


class AudioProcessor:
    """音频处理工具类"""

    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def load_audio(self, audio_path, max_length=None):
        """加载音频文件"""
        try:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)

            # 如果指定了最大长度，进行截断或填充
            if max_length is not None:
                max_samples = int(max_length * self.sample_rate)
                if len(audio) > max_samples:
                    # 随机截取
                    start = random.randint(0, len(audio) - max_samples)
                    audio = audio[start:start + max_samples]
                elif len(audio) < max_samples:
                    # 填充零
                    audio = np.pad(audio, (0, max_samples - len(audio)), mode='constant')

            return audio
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return None

    def add_noise(self, audio, noise, snr_db):
        """向音频添加噪声"""
        # 计算音频和噪声的能量
        audio_power = np.mean(audio ** 2)
        noise_power = np.mean(noise ** 2)

        # 根据SNR计算噪声缩放因子
        snr_linear = 10 ** (snr_db / 10)
        scale = np.sqrt(audio_power / (noise_power * snr_linear))

        # 添加缩放后的噪声
        noisy_audio = audio + scale * noise

        return noisy_audio


class MusanNoiseLoader:
    """MUSAN噪声数据加载器"""

    def __init__(self, musan_root, sample_rate=16000):
        self.musan_root = musan_root
        self.sample_rate = sample_rate

        # 收集所有噪声文件
        self.noise_files = self._collect_files(os.path.join(musan_root, "noise"))
        self.music_files = self._collect_files(os.path.join(musan_root, "music"))
        self.speech_files = self._collect_files(os.path.join(musan_root, "speech"))

        print(f"Loaded {len(self.noise_files)} noise files")
        print(f"Loaded {len(self.music_files)} music files")
        print(f"Loaded {len(self.speech_files)} speech files")

    def _collect_files(self, folder):
        """收集文件夹中的所有wav文件"""
        files = []
        for root, dirs, filenames in os.walk(folder):
            for filename in filenames:
                if filename.endswith('.wav'):
                    files.append(os.path.join(root, filename))
        return files

    def get_random_noise(self, noise_type='noise', duration=None):
        """随机获取一段噪声"""
        if noise_type == 'noise':
            files = self.noise_files
        elif noise_type == 'music':
            files = self.music_files
        elif noise_type == 'speech':
            files = self.speech_files
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")

        if len(files) == 0:
            raise ValueError(f"No files found for noise type: {noise_type}")

        # 随机选择一个文件
        noise_file = random.choice(files)

        # 加载噪声
        noise, sr = librosa.load(noise_file, sr=self.sample_rate)

        # 如果指定了时长，随机截取或重复
        if duration is not None:
            required_samples = int(duration * self.sample_rate)
            if len(noise) > required_samples:
                start = random.randint(0, len(noise) - required_samples)
                noise = noise[start:start + required_samples]
            else:
                # 重复噪声直到达到所需长度
                repeats = int(np.ceil(required_samples / len(noise)))
                noise = np.tile(noise, repeats)[:required_samples]

        return noise


class VoxCelebDataset:
    """VoxCeleb数据集加载器"""

    def __init__(self, data_root, is_train=True):
        self.data_root = data_root
        self.is_train = is_train

        # 收集所有说话人和音频文件
        self.speakers = {}
        self._collect_speakers()

        self.speaker_list = list(self.speakers.keys())
        print(f"Loaded {len(self.speaker_list)} speakers")
        print(f"Total utterances: {sum(len(files) for files in self.speakers.values())}")

    def _collect_speakers(self):
        """收集所有说话人的音频文件"""
        for speaker_id in os.listdir(self.data_root):
            speaker_path = os.path.join(self.data_root, speaker_id)
            if not os.path.isdir(speaker_path):
                continue

            audio_files = []
            for root, dirs, files in os.walk(speaker_path):
                for file in files:
                    if file.endswith('.wav'):
                        audio_files.append(os.path.join(root, file))

            if len(audio_files) > 0:
                self.speakers[speaker_id] = audio_files

    def get_random_utterance(self, speaker_id=None):
        """随机获取一个说话人的音频"""
        if speaker_id is None:
            speaker_id = random.choice(self.speaker_list)

        if speaker_id not in self.speakers:
            raise ValueError(f"Speaker {speaker_id} not found")

        audio_file = random.choice(self.speakers[speaker_id])
        return audio_file, speaker_id

    def get_triplet(self):
        """获取一个三元组：anchor, positive, negative"""
        # 随机选择一个说话人作为anchor和positive
        anchor_speaker = random.choice(self.speaker_list)
        anchor_file, _ = self.get_random_utterance(anchor_speaker)
        positive_file, _ = self.get_random_utterance(anchor_speaker)

        # 选择不同的说话人作为negative
        negative_speaker = random.choice([s for s in self.speaker_list if s != anchor_speaker])
        negative_file, _ = self.get_random_utterance(negative_speaker)

        return anchor_file, positive_file, negative_file


# 测试代码
if __name__ == "__main__":
    print("Testing data utilities...")

    # 测试MUSAN加载器
    print("\n1. Testing MUSAN loader...")
    musan_loader = MusanNoiseLoader(config.MUSAN_ROOT)

    # 测试VoxCeleb加载器
    print("\n2. Testing VoxCeleb loader...")
    voxceleb_train = VoxCelebDataset(config.VOXCELEB1_TRAIN, is_train=True)

    # 只有当说话人数量>=2时才测试三元组生成
    if len(voxceleb_train.speaker_list) >= 2:
        print("\n3. Testing triplet generation...")
        anchor, positive, negative = voxceleb_train.get_triplet()
        print(f"Anchor: {os.path.basename(anchor)}")
        print(f"Positive: {os.path.basename(positive)}")
        print(f"Negative: {os.path.basename(negative)}")

        # 测试音频处理
        print("\n4. Testing audio processing...")
        processor = AudioProcessor()
        audio = processor.load_audio(anchor, max_length=3)
        print(f"Audio shape: {audio.shape}")

        # 测试添加噪声
        noise = musan_loader.get_random_noise('noise', duration=3)
        noisy_audio = processor.add_noise(audio, noise, snr_db=-10)
        print(f"Noisy audio shape: {noisy_audio.shape}")
    else:
        print(f"\n⚠ Warning: Only {len(voxceleb_train.speaker_list)} speakers found, need at least 2 for triplet")
        print("Skipping triplet and audio processing tests")

    print("\nBasic tests completed!")