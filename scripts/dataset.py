"""
VoxCeleb1 + MUSAN 数据集 - 严格按照论文实现
支持：
1. MUSAN Train/Test划分（不相交）- 使用预生成的split文件
2. 测试集15种条件（3噪声 × 5 SNR）
3. 修复：处理单文件说话人问题
"""
import os
import random
import numpy as np
import torch
import torchaudio
from pathlib import Path
from torch.utils.data import Dataset
import json


class VoxCelebMusanDataset(Dataset):
    """
    论文数据集实现

    参数:
    - split: 'train' 或 'test'
    - snr_range: 训练时的SNR范围，默认(-20, 0)
    - test_snr: 测试时的SNR水平，默认None（使用所有5个SNR）
    - test_noise_type: 测试时的噪声类型，默认None（使用所有3种）
    """

    def __init__(self, voxceleb_dir, musan_dir, split='train',
                 snr_range=(-20, 0), test_snr=None, test_noise_type=None):
        self.voxceleb_dir = Path(voxceleb_dir)
        self.musan_dir = Path(musan_dir)
        self.split = split
        self.snr_range = snr_range
        self.test_snr = test_snr
        self.test_noise_type = test_noise_type

        print(f'Loading {split} dataset...')
        self.audio_files, self.speaker_to_files = self._load_file_list()
        self.speakers = list(self.speaker_to_files.keys())

        if len(self.speakers) < 2:
            raise ValueError(f'Need at least 2 speakers, found {len(self.speakers)}')

        # 过滤掉只有单个文件的说话人
        self._filter_single_file_speakers()

        self.noise_files = self._load_musan_files()

        # 打印数据集统计信息
        self._print_statistics()

    def _load_file_list(self):
        """加载音频文件列表"""
        if self.split == 'train':
            search_dir = self.voxceleb_dir / 'voxceleb1_complete' / 'vox1_dev_wav' / 'wav'
        else:
            search_dir = self.voxceleb_dir / 'voxceleb1_complete' / 'vox1_test_wav' / 'wav'

        # 备用路径
        if not search_dir.exists():
            if self.split == 'train':
                search_dir = self.voxceleb_dir / 'voxceleb1_complete' / 'vox1_dev_wav'
            else:
                search_dir = self.voxceleb_dir / 'voxceleb1_complete' / 'vox1_test_wav'

        audio_files = []
        speaker_to_files = {}
        file_to_speaker = {}

        print(f'Scanning: {search_dir}')

        if not search_dir.exists():
            raise FileNotFoundError(f'Not found: {search_dir}')

        speaker_dirs = [d for d in search_dir.iterdir() if d.is_dir() and d.name.startswith('id')]
        print(f'Found {len(speaker_dirs)} speaker directories')

        for spk_dir in speaker_dirs:
            speaker_id = spk_dir.name
            spk_files = list(spk_dir.rglob('*.wav'))

            if len(spk_files) > 0:
                speaker_to_files[speaker_id] = spk_files
                audio_files.extend(spk_files)
                for f in spk_files:
                    file_to_speaker[str(f)] = speaker_id

        self.file_to_speaker = file_to_speaker
        return audio_files, speaker_to_files

    def _filter_single_file_speakers(self):
        """
        过滤掉只有一个文件的说话人
        论文使用triplet loss，需要每个说话人至少有2个utterance
        """
        original_speaker_count = len(self.speakers)
        original_file_count = len(self.audio_files)

        # 找出至少有2个文件的说话人
        valid_speakers = {
            spk: files for spk, files in self.speaker_to_files.items()
            if len(files) >= 2
        }

        if len(valid_speakers) < 2:
            raise ValueError(
                f'After filtering, only {len(valid_speakers)} speakers have >=2 files. '
                'Need at least 2 speakers for triplet sampling.'
            )

        # 更新数据结构
        self.speaker_to_files = valid_speakers
        self.speakers = list(valid_speakers.keys())

        # 重建audio_files和file_to_speaker
        self.audio_files = []
        self.file_to_speaker = {}
        for spk, files in valid_speakers.items():
            self.audio_files.extend(files)
            for f in files:
                self.file_to_speaker[str(f)] = spk

        filtered_speakers = original_speaker_count - len(self.speakers)
        filtered_files = original_file_count - len(self.audio_files)

        if filtered_speakers > 0:
            print(f'Filtered {filtered_speakers} speakers with <2 files '
                  f'({filtered_files} files removed)')

    def _print_statistics(self):
        """打印数据集统计信息"""
        files_per_speaker = [len(files) for files in self.speaker_to_files.values()]

        print(f'\n{"=" * 70}')
        print(f'DATASET STATISTICS ({self.split.upper()})')
        print(f'{"=" * 70}')
        print(f'Total speakers: {len(self.speakers)}')
        print(f'Total files: {len(self.audio_files)}')
        print(f'Files per speaker:')
        print(f'  Min: {min(files_per_speaker)}')
        print(f'  Max: {max(files_per_speaker)}')
        print(f'  Mean: {np.mean(files_per_speaker):.1f}')
        print(f'  Median: {np.median(files_per_speaker):.1f}')
        print(f'SNR range: {self.snr_range}')
        print(f'{"=" * 70}\n')

    def _load_musan_files(self):
        """
        加载MUSAN噪声，优先使用预生成的划分文件
        论文: "We divided the MUSAN corpus into two disjoint sets"
        """
        split_file = self.musan_dir / 'musan_split.json'

        # 方法1：使用预生成的划分文件（推荐）
        if split_file.exists():
            print(f'Loading MUSAN from pre-split file: {split_file}')
            with open(split_file, 'r') as f:
                split_data = json.load(f)

            noise_files = {}
            for noise_type in ['noise', 'music', 'speech']:
                if noise_type not in split_data:
                    print(f'Warning: {noise_type} not in split file')
                    continue

                rel_paths = split_data[noise_type][self.split]
                # 转换为绝对路径
                abs_paths = [str(self.musan_dir / rel_path) for rel_path in rel_paths]
                noise_files[noise_type] = abs_paths

                print(f'  MUSAN {noise_type} ({self.split}): {len(abs_paths)} files')

            return noise_files

        # 方法2：动态划分（回退方案）
        else:
            print(f'Pre-split file not found: {split_file}')
            print('Falling back to dynamic splitting (less reliable)')
            print('Recommend running: python scripts/split_musan.py data/musan')

            noise_files = {}
            for noise_type in ['noise', 'music', 'speech']:
                noise_dir = self.musan_dir / noise_type
                if not noise_dir.exists():
                    print(f'Warning: {noise_dir} not found')
                    continue

                all_files = list(noise_dir.rglob('*.wav'))

                if len(all_files) == 0:
                    continue

                # 固定种子确保可复现
                random.seed(42)
                random.shuffle(all_files)

                # 80/20划分
                split_idx = int(len(all_files) * 0.8)

                if self.split == 'train':
                    files = all_files[:split_idx]
                else:
                    files = all_files[split_idx:]

                noise_files[noise_type] = [str(f) for f in files]
                print(f'  MUSAN {noise_type} ({self.split}): {len(files)} files')

            return noise_files

    def add_noise(self, audio, sr, snr_db, noise_type=None):
        """
        添加噪声

        参数:
        - noise_type: 指定噪声类型，None则随机选择
        """
        if not self.noise_files:
            return audio

        # 选择噪声类型
        if noise_type is None:
            available_types = list(self.noise_files.keys())
            noise_type = random.choice(available_types)

        if noise_type not in self.noise_files or len(self.noise_files[noise_type]) == 0:
            return audio

        noise_file = random.choice(self.noise_files[noise_type])

        try:
            noise, noise_sr = torchaudio.load(noise_file)
            if noise_sr != sr:
                noise = torchaudio.transforms.Resample(noise_sr, sr)(noise)
            if noise.shape[0] > 1:
                noise = noise.mean(dim=0, keepdim=True)
            if noise.shape[1] < audio.shape[1]:
                repeats = int(np.ceil(audio.shape[1] / noise.shape[1]))
                noise = noise.repeat(1, repeats)
            noise = noise[:, :audio.shape[1]]

            # 计算噪声增益
            signal_power = torch.mean(audio ** 2)
            noise_power = torch.mean(noise ** 2)
            snr_linear = 10 ** (snr_db / 10)
            noise_gain = torch.sqrt(signal_power / (noise_power * snr_linear + 1e-10))

            return audio + noise_gain * noise
        except Exception as e:
            print(f'Error adding noise from {noise_file}: {e}')
            return audio

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        """
        修复：确保positive不会与anchor相同
        """
        # 获取anchor
        anchor_file = self.audio_files[idx]
        speaker_id = self.file_to_speaker[str(anchor_file)]
        speaker_files = self.speaker_to_files[speaker_id]

        # 由于已经过滤了单文件说话人，这里保证len(speaker_files) >= 2
        assert len(speaker_files) >= 2, f"Speaker {speaker_id} has <2 files"

        # Positive: 同一说话人的不同音频（确保不是同一个文件）
        positive_candidates = [f for f in speaker_files if f != anchor_file]
        positive_file = random.choice(positive_candidates)

        # Negative: 不同说话人
        negative_speaker_candidates = [s for s in self.speakers if s != speaker_id]
        neg_speaker = random.choice(negative_speaker_candidates)
        negative_file = random.choice(self.speaker_to_files[neg_speaker])

        try:
            # 加载音频
            anchor_audio, sr = torchaudio.load(anchor_file)
            positive_audio, _ = torchaudio.load(positive_file)
            negative_audio, _ = torchaudio.load(negative_file)

            # 转单声道
            if anchor_audio.shape[0] > 1:
                anchor_audio = anchor_audio.mean(dim=0, keepdim=True)
            if positive_audio.shape[0] > 1:
                positive_audio = positive_audio.mean(dim=0, keepdim=True)
            if negative_audio.shape[0] > 1:
                negative_audio = negative_audio.mean(dim=0, keepdim=True)

            # 确定SNR和噪声类型
            if self.split == 'train':
                # 训练：随机SNR [-20, 0]，随机噪声类型
                snr = random.uniform(self.snr_range[0], self.snr_range[1])
                noise_type = None  # 随机选择
            else:
                # 测试：固定SNR和噪声类型（如果指定）
                snr = self.test_snr if self.test_snr is not None else 0
                noise_type = self.test_noise_type

            return {
                'anchor_noisy': self.add_noise(anchor_audio, sr, snr, noise_type),
                'positive_noisy': self.add_noise(positive_audio, sr, snr, noise_type),
                'negative_noisy': self.add_noise(negative_audio, sr, snr, noise_type),
                'snr': snr,
                'noise_type': noise_type if noise_type else 'mixed',
                'speaker_id': speaker_id
            }
        except Exception as e:
            print(f'Error loading audio: {e}')
            # 加载失败时尝试下一个样本
            return self.__getitem__((idx + 1) % len(self))


# VoxCelebTestPairs类保持不变（备用）
class VoxCelebTestPairs(Dataset):
    """
    VoxCeleb1测试集的配对数据集（备用实现）

    注意：当前评估脚本使用VoxCelebTestPairsEvaluator，
    这个类保留作为备用。
    """

    def __init__(self, voxceleb_dir, musan_dir, pairs_file,
                 noise_type='noise', snr_db=0):
        self.voxceleb_dir = Path(voxceleb_dir)
        self.musan_dir = Path(musan_dir)
        self.noise_type = noise_type
        self.snr_db = snr_db

        self.pairs = self._load_pairs(pairs_file)
        self.noise_files = self._load_musan_test_noise()

        print(f'Loaded {len(self.pairs)} pairs for {noise_type} at {snr_db} dB')

    def _load_pairs(self, pairs_file):
        """加载验证对"""
        pairs = []
        with open(pairs_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    label = int(parts[0])
                    file1 = self.voxceleb_dir / 'voxceleb1_complete' / 'vox1_test_wav' / 'wav' / parts[1]
                    file2 = self.voxceleb_dir / 'voxceleb1_complete' / 'vox1_test_wav' / 'wav' / parts[2]
                    pairs.append((label, file1, file2))
        return pairs

    def _load_musan_test_noise(self):
        """加载MUSAN测试集噪声（优先使用预生成划分）"""
        split_file = self.musan_dir / 'musan_split.json'

        # 使用预生成的划分
        if split_file.exists():
            with open(split_file, 'r') as f:
                split_data = json.load(f)

            if self.noise_type in split_data:
                rel_paths = split_data[self.noise_type]['test']
                return [str(self.musan_dir / rp) for rp in rel_paths]

        # 回退方案：动态划分
        noise_dir = self.musan_dir / self.noise_type
        if not noise_dir.exists():
            return []

        all_files = list(noise_dir.rglob('*.wav'))
        random.seed(42)
        random.shuffle(all_files)

        split_idx = int(len(all_files) * 0.8)
        return [str(f) for f in all_files[split_idx:]]

    def add_noise(self, audio, sr):
        """添加噪声"""
        if not self.noise_files:
            return audio

        noise_file = random.choice(self.noise_files)

        try:
            noise, noise_sr = torchaudio.load(noise_file)
            if noise_sr != sr:
                noise = torchaudio.transforms.Resample(noise_sr, sr)(noise)
            if noise.shape[0] > 1:
                noise = noise.mean(dim=0, keepdim=True)
            if noise.shape[1] < audio.shape[1]:
                repeats = int(np.ceil(audio.shape[1] / noise.shape[1]))
                noise = noise.repeat(1, repeats)
            noise = noise[:, :audio.shape[1]]

            signal_power = torch.mean(audio ** 2)
            noise_power = torch.mean(noise ** 2)
            snr_linear = 10 ** (self.snr_db / 10)
            noise_gain = torch.sqrt(signal_power / (noise_power * snr_linear + 1e-10))

            return audio + noise_gain * noise
        except:
            return audio

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        label, file1, file2 = self.pairs[idx]

        try:
            audio1, sr1 = torchaudio.load(file1)
            audio2, sr2 = torchaudio.load(file2)

            if audio1.shape[0] > 1:
                audio1 = audio1.mean(dim=0, keepdim=True)
            if audio2.shape[0] > 1:
                audio2 = audio2.mean(dim=0, keepdim=True)

            audio1_noisy = self.add_noise(audio1, sr1)
            audio2_noisy = self.add_noise(audio2, sr2)

            return {
                'audio1_noisy': audio1_noisy,
                'audio2_noisy': audio2_noisy,
                'label': label,
                'snr': self.snr_db,
                'noise_type': self.noise_type
            }
        except Exception as e:
            print(f'Error: {e}')
            return self.__getitem__((idx + 1) % len(self))
