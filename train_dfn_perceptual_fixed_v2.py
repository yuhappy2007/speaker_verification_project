"""
DeepFilterNet Perceptual Loss Training Script (FIXED VERSION)
=============================================================

Multi-stage training with perceptual loss from WavLM.

FIXED ISSUES:
- DeepFilterNet model device mismatch (CPU vs CUDA)
- Enhanced audio processing flow
- Proper model device management

Author: Claude
Date: 2025-11-02
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional
import numpy as np
from pathlib import Path

# Import project modules
sys.path.append('scripts')
# NOTE: We define VoxCelebNoiseDataset here instead of importing
# from speaker_embedding import EcapaTDNNSpeakerEmbedding  # Not needed for Stage 1
# from speech_enhancer import DeepFilterNetEnhancer  # We implement our own wrapper

# WavLM imports
from transformers import Wav2Vec2FeatureExtractor, WavLMModel

# DeepFilterNet imports
from df import enhance, init_df
from df.enhance import init_df, load_audio, save_audio
from df.io import resample
import torchaudio


# ==================== Simple Dataset for DFN Training ====================
class VoxCelebNoiseDataset:
    """
    简单的VoxCeleb + MUSAN数据集，用于DeepFilterNet训练

    每个样本返回：
    - noisy: 带噪声的音频
    - clean: 对应的干净音频
    - speaker_id: 说话人ID
    - snr: 信噪比
    """

    def __init__(self, voxceleb_dir, musan_dir, snr_levels=[-5, 0, 5, 10, 15],
                 target_length=3.0, augment=True, split='train'):
        """
        参数:
        - voxceleb_dir: VoxCeleb1数据集目录
        - musan_dir: MUSAN噪声数据集目录
        - snr_levels: SNR水平列表（dB）
        - target_length: 目标音频长度（秒）
        - augment: 是否数据增强
        - split: 'train' 或 'test'
        """
        self.voxceleb_dir = Path(voxceleb_dir)
        self.musan_dir = Path(musan_dir)
        self.snr_levels = snr_levels
        self.target_length = target_length
        self.augment = augment
        self.split = split
        self.target_sr = 16000  # VoxCeleb标准采样率

        print(f'\n{"=" * 70}')
        print(f'Loading VoxCeleb Noise Dataset ({split.upper()})')
        print(f'{"=" * 70}')

        # 加载音频文件列表
        self.audio_files, self.speaker_to_files = self._load_file_list()
        self.speakers = list(self.speaker_to_files.keys())

        # 加载噪声文件
        self.noise_files = self._load_musan_files()

        # 打印统计信息
        self._print_statistics()

    def _load_file_list(self):
        """加载VoxCeleb音频文件列表"""
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

        if not search_dir.exists():
            raise FileNotFoundError(f'VoxCeleb directory not found: {search_dir}')

        print(f'Scanning: {search_dir}')

        audio_files = []
        speaker_to_files = {}
        file_to_speaker = {}

        # 扫描说话人目录
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

    def _load_musan_files(self):
        """加载MUSAN噪声文件"""
        import json
        split_file = self.musan_dir / 'musan_split.json'

        # 优先使用预生成的划分文件
        if split_file.exists():
            print(f'Loading MUSAN from: {split_file}')
            with open(split_file, 'r') as f:
                split_data = json.load(f)

            noise_files = {}
            for noise_type in ['noise', 'music', 'speech']:
                if noise_type not in split_data:
                    continue

                rel_paths = split_data[noise_type][self.split]
                abs_paths = [str(self.musan_dir / rel_path) for rel_path in rel_paths]
                noise_files[noise_type] = abs_paths

                print(f'  {noise_type}: {len(abs_paths)} files')

            return noise_files

        # 回退：动态划分
        print('Warning: musan_split.json not found, using dynamic split')
        noise_files = {}

        for noise_type in ['noise', 'music', 'speech']:
            noise_dir = self.musan_dir / noise_type
            if not noise_dir.exists():
                continue

            all_files = list(noise_dir.rglob('*.wav'))
            if len(all_files) == 0:
                continue

            # 固定种子
            random.seed(42)
            random.shuffle(all_files)

            # 80/20划分
            split_idx = int(len(all_files) * 0.8)
            if self.split == 'train':
                files = all_files[:split_idx]
            else:
                files = all_files[split_idx:]

            noise_files[noise_type] = [str(f) for f in files]
            print(f'  {noise_type}: {len(files)} files')

        return noise_files

    def _print_statistics(self):
        """打印数据集统计"""
        print(f'\nDataset Statistics:')
        print(f'  Speakers: {len(self.speakers)}')
        print(f'  Audio files: {len(self.audio_files)}')
        print(f'  SNR levels: {self.snr_levels}')
        print(f'  Target length: {self.target_length}s')
        print(f'  Sample rate: {self.target_sr}Hz')
        print(f'{"=" * 70}\n')

    def add_noise(self, audio, sr, snr_db):
        """
        添加噪声到音频

        参数:
        - audio: [1, samples] tensor
        - sr: 采样率
        - snr_db: 信噪比（dB）

        返回:
        - noisy_audio: [1, samples] tensor
        """
        if not self.noise_files:
            return audio

        # 随机选择噪声类型
        available_types = list(self.noise_files.keys())
        if not available_types:
            return audio

        noise_type = random.choice(available_types)
        noise_file = random.choice(self.noise_files[noise_type])

        try:
            # 加载噪声
            noise, noise_sr = torchaudio.load(noise_file)

            # 重采样
            if noise_sr != sr:
                resampler = torchaudio.transforms.Resample(noise_sr, sr)
                noise = resampler(noise)

            # 转单声道
            if noise.shape[0] > 1:
                noise = noise.mean(dim=0, keepdim=True)

            # 循环噪声以匹配音频长度
            if noise.shape[1] < audio.shape[1]:
                repeats = int(np.ceil(audio.shape[1] / noise.shape[1]))
                noise = noise.repeat(1, repeats)

            # 裁剪到相同长度
            noise = noise[:, :audio.shape[1]]

            # 计算噪声增益
            signal_power = torch.mean(audio ** 2)
            noise_power = torch.mean(noise ** 2)
            snr_linear = 10 ** (snr_db / 10)
            noise_gain = torch.sqrt(signal_power / (noise_power * snr_linear + 1e-10))

            # 添加噪声
            noisy = audio + noise_gain * noise

            return noisy

        except Exception as e:
            print(f'Error adding noise from {noise_file}: {e}')
            return audio

    def _process_audio(self, audio_path):
        """
        加载并处理音频

        返回:
        - audio: [1, samples] tensor
        - sr: 采样率
        """
        # 加载音频
        audio, sr = torchaudio.load(audio_path)

        # 转单声道
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        # 重采样到目标采样率
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            audio = resampler(audio)
            sr = self.target_sr

        # 调整长度
        target_samples = int(self.target_length * sr)
        current_samples = audio.shape[1]

        if current_samples > target_samples:
            # 随机裁剪
            if self.augment and self.split == 'train':
                start = random.randint(0, current_samples - target_samples)
            else:
                start = 0
            audio = audio[:, start:start + target_samples]
        elif current_samples < target_samples:
            # 填充
            pad_length = target_samples - current_samples
            audio = torch.nn.functional.pad(audio, (0, pad_length))

        return audio, sr

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        """
        获取一个样本

        返回:
        {
            'noisy': noisy音频 [samples],
            'clean': clean音频 [samples],
            'speaker_id': 说话人ID,
            'snr': 信噪比
        }
        """
        # 获取音频文件
        audio_path = self.audio_files[idx]
        speaker_id = self.file_to_speaker[str(audio_path)]

        try:
            # 加载并处理音频
            clean_audio, sr = self._process_audio(audio_path)

            # 随机选择SNR
            if self.augment and self.split == 'train':
                snr = random.choice(self.snr_levels)
            else:
                snr = self.snr_levels[0] if self.snr_levels else 10

            # 添加噪声
            noisy_audio = self.add_noise(clean_audio, sr, snr)

            return {
                'noisy': noisy_audio.squeeze(0),  # [samples]
                'clean': clean_audio.squeeze(0),  # [samples]
                'speaker_id': speaker_id,
                'snr': float(snr)
            }

        except Exception as e:
            print(f'Error loading {audio_path}: {e}')
            # 返回下一个样本
            return self.__getitem__((idx + 1) % len(self))


# ==================== Configuration ====================
class Config:
    """Training configuration"""
    # Paths
    voxceleb_dir = 'data/voxceleb1'
    musan_dir = 'data/musan'
    wavlm_path = 'D:/WavLM'  # ⚠️ MODIFY THIS TO YOUR ACTUAL PATH

    checkpoint_dir = 'checkpoints_dfn_perceptual'
    log_dir = 'logs_dfn_perceptual'

    # Training parameters
    batch_size = 8
    num_epochs = 15
    save_interval = 500  # Save every N batches

    # Learning rates for each stage
    lr_stage1 = 1e-4  # Stage 1: warm-up
    lr_stage2 = 5e-5  # Stage 2: fine-tune with perceptual loss
    lr_stage3 = 1e-4  # Stage 3: full training

    # Stage boundaries (epochs)
    stage1_end = 5  # Epochs 1-5: warm-up with original loss only
    stage2_end = 10  # Epochs 6-10: add perceptual loss with small LR
    stage3_end = 15  # Epochs 11-15: full training

    # Perceptual loss weights
    lambda_perceptual_stage2 = 0.1  # Small weight during fine-tuning
    lambda_perceptual_stage3 = 0.3  # Larger weight during full training

    # SNR levels for training
    snr_levels = [-5, 0, 5, 10, 15]

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ==================== Perceptual Loss Module ====================
class WavLMPerceptualLoss(nn.Module):
    """
    Perceptual loss using frozen WavLM model.

    Computes cosine similarity loss between WavLM embeddings
    of enhanced and clean speech.
    """

    def __init__(self, wavlm_path: str, device: str = 'cuda'):
        super().__init__()
        self.device = device

        print(f"Loading WavLM from {wavlm_path}...")
        # Load WavLM model and feature extractor
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(wavlm_path)
        self.wavlm = WavLMModel.from_pretrained(wavlm_path).to(device)

        # Freeze WavLM
        for param in self.wavlm.parameters():
            param.requires_grad = False
        self.wavlm.eval()

        print("✓ WavLM loaded and frozen")

    def extract_embedding(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Extract WavLM embedding from audio.

        Args:
            audio: [batch_size, samples] or [samples]

        Returns:
            embedding: [batch_size, hidden_dim]
        """
        # Ensure 2D
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        # Move to CPU for feature extraction (transformers requirement)
        audio_cpu = audio.cpu().numpy()

        # Extract features
        inputs = self.feature_extractor(
            audio_cpu,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )

        # Move to device
        input_values = inputs.input_values.to(self.device)

        # Extract WavLM features
        with torch.no_grad():
            outputs = self.wavlm(input_values)
            # Use mean of last hidden state as embedding
            embedding = outputs.last_hidden_state.mean(dim=1)  # [B, hidden_dim]

        return embedding

    def forward(self, enhanced: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss.

        Args:
            enhanced: Enhanced audio [batch_size, samples]
            clean: Clean audio [batch_size, samples]

        Returns:
            loss: Cosine similarity loss (0 = identical, 2 = opposite)
        """
        # Extract embeddings
        emb_enhanced = self.extract_embedding(enhanced)
        emb_clean = self.extract_embedding(clean)

        # Compute cosine similarity loss
        # 1 - cosine_sim maps: [1, -1] -> [0, 2]
        cos_sim = nn.functional.cosine_similarity(emb_enhanced, emb_clean, dim=1)
        loss = (1 - cos_sim).mean()

        return loss


# ==================== Enhanced DeepFilterNet Wrapper ====================
class TrainableDeepFilterNet:
    """
    Wrapper for DeepFilterNet that handles device management properly.

    CRITICAL FIX V3:
    - Use model.forward() directly (not enhance() which has no_grad)
    - Keep gradients flowing
    - Handle frequency domain transforms manually
    """

    def __init__(self, device: str = 'cuda'):
        self.device = device

        # Initialize DeepFilterNet
        print("Initializing DeepFilterNet3...")
        self.model, self.df_state, _ = init_df()

        # Move model to device
        self.model.to(device)
        self.model.train()

        # Make trainable
        for param in self.model.parameters():
            param.requires_grad = True

        # Get model parameters
        self.sr_model = self.df_state.sr()  # 48000 Hz
        self.sr_input = 16000  # VoxCeleb standard

        print(f"Creating resamplers ({self.sr_input}Hz <-> {self.sr_model}Hz)...")
        self.resampler_up = torchaudio.transforms.Resample(
            self.sr_input, self.sr_model
        ).to(device)
        self.resampler_down = torchaudio.transforms.Resample(
            self.sr_model, self.sr_input
        ).to(device)

        print(f"✓ DeepFilterNet model on {device} (trainable)")
        print(f"  Model target SR: {self.sr_model} Hz")
        print(f"  Input/Output SR: {self.sr_input} Hz (VoxCeleb standard)")

    def enhance_audio(self, noisy: torch.Tensor, sr: int = 16000) -> torch.Tensor:
        """
        Enhance audio using trainable forward pass (preserves gradients).

        Args:
            noisy: Noisy audio [batch_size, samples] on ANY device
            sr: Sample rate (should be 16000 for VoxCeleb)

        Returns:
            enhanced: Enhanced audio [batch_size, samples] on SAME device as input
        """
        input_device = noisy.device
        batch_size = noisy.shape[0]

        # Resample to model SR (48kHz)
        if sr != self.sr_model:
            noisy_48k = self.resampler_up(noisy)  # [B, samples_48k]
        else:
            noisy_48k = noisy

        # Ensure on correct device
        noisy_48k = noisy_48k.to(self.device)

        # Add channel dimension if needed: [B, samples] -> [B, 1, samples]
        if noisy_48k.dim() == 2:
            noisy_48k = noisy_48k.unsqueeze(1)

        # Process with DeepFilterNet encoder-decoder
        # We'll do a simplified version that preserves gradients
        enhanced_48k = self._forward_enhance(noisy_48k)

        # Remove channel dimension: [B, 1, samples] -> [B, samples]
        if enhanced_48k.dim() == 3:
            enhanced_48k = enhanced_48k.squeeze(1)

        # Resample back to input SR (16kHz)
        if sr != self.sr_model:
            enhanced_16k = self.resampler_down(enhanced_48k)
        else:
            enhanced_16k = enhanced_48k

        return enhanced_16k

    def _forward_enhance(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through DeepFilterNet (preserves gradients).

        Uses PyTorch native STFT/iSTFT to maintain gradient flow.

        Args:
            audio: [B, C, samples] audio on self.device

        Returns:
            enhanced: [B, C, samples] enhanced audio
        """
        batch_size, channels, samples = audio.shape

        # Get STFT parameters from df_state
        hop_length = self.df_state.hop_size()
        n_fft = self.df_state.fft_size()
        window = torch.hann_window(n_fft, device=self.device)

        # Flatten batch and channels for STFT
        audio_flat = audio.view(-1, samples)  # [B*C, samples]

        # PyTorch STFT (maintains gradients!)
        spec = torch.stft(
            audio_flat,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=window,
            return_complex=True,
            center=True
        )  # [B*C, F, T]

        # Reshape to [B, C, F, T]
        freq_bins, time_frames = spec.shape[1], spec.shape[2]
        spec = spec.view(batch_size, channels, freq_bins, time_frames)

        # Keep 4D format [B, C, F, T] - model expects this!
        # Don't squeeze channel dimension

        # Get magnitude spectrogram
        spec_abs = spec.abs()

        # erb_widths() returns ERB band definitions, doesn't take arguments
        # For training, we can use the model's internal feature extraction
        # Just pass the magnitude spec, and the model will handle it
        erb_feat = spec_abs  # The model will convert this internally
        spec_feat = spec_abs

        # Forward pass through model (preserves gradients!)
        # Model expects [B, C, F, T] for all inputs
        enhanced_spec, *_ = self.model(spec, erb_feat, spec_feat)

        # enhanced_spec is [B, C, F, T]

        # Flatten for iSTFT
        enhanced_spec_flat = enhanced_spec.view(-1, freq_bins, time_frames)  # [B*C, F, T]

        # PyTorch iSTFT (maintains gradients!)
        enhanced_flat = torch.istft(
            enhanced_spec_flat,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=window,
            center=True,
            length=samples  # Ensure same length as input
        )  # [B*C, samples]

        # Reshape back to [B, C, samples]
        enhanced = enhanced_flat.view(batch_size, channels, samples)

        return enhanced

    def get_model(self):
        """Get the underlying model for optimization"""
        return self.model


# ==================== Trainer ====================
class DFNPerceptualTrainer:
    """
    Main trainer for DeepFilterNet with perceptual loss.

    Training stages:
    - Stage 1 (epoch 1-5): Warm-up with original loss only
    - Stage 2 (epoch 6-10): Add perceptual loss, small LR
    - Stage 3 (epoch 11-15): Full training, normal LR
    """

    def __init__(self, config: Config):
        self.config = config
        self.device = config.device

        # Create directories
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)

        print("\n" + "=" * 80)
        print("LOADING MODELS")
        print("=" * 80)

        # Load DeepFilterNet (trainable)
        print("\n[1/2] Loading DeepFilterNet3...")
        self.enhancer = TrainableDeepFilterNet(device=self.device)
        self.dfn_model = self.enhancer.get_model()

        # Load WavLM for perceptual loss (frozen)
        print("\n[2/2] Loading WavLM for perceptual loss...")
        self.perceptual_loss_fn = WavLMPerceptualLoss(
            config.wavlm_path,
            device=self.device
        )
        print("✓ WavLM loaded and frozen")

        # Original loss (MSE)
        self.mse_loss = nn.MSELoss()

        # Optimizer (will be recreated for each stage)
        self.optimizer = None

        # Training history
        self.history = {
            'epochs': [],
            'batches': [],
            'mse_loss': [],
            'perceptual_loss': [],
            'total_loss': [],
            'stage': []
        }

        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.dfn_model.parameters() if p.requires_grad)
        print(f"\n📊 Trainable parameters (DeepFilterNet): {trainable_params:,}")

    def get_stage_params(self, epoch: int) -> Tuple[int, float, float]:
        """
        Get training stage parameters based on epoch.

        Returns:
            (stage_number, learning_rate, lambda_perceptual)
        """
        if epoch <= self.config.stage1_end:
            return 1, self.config.lr_stage1, 0.0
        elif epoch <= self.config.stage2_end:
            return 2, self.config.lr_stage2, self.config.lambda_perceptual_stage2
        else:
            return 3, self.config.lr_stage3, self.config.lambda_perceptual_stage3

    def setup_optimizer(self, lr: float):
        """Setup optimizer with given learning rate"""
        self.optimizer = optim.Adam(self.dfn_model.parameters(), lr=lr)

    def train_batch(self, batch: Dict, lambda_perceptual: float) -> Dict[str, float]:
        """
        Train on a single batch.

        Args:
            batch: Dict with 'noisy', 'clean', 'speaker_id', 'snr'
            lambda_perceptual: Weight for perceptual loss

        Returns:
            Dict with loss values
        """
        noisy = batch['noisy'].to(self.device)  # [B, samples]
        clean = batch['clean'].to(self.device)  # [B, samples]

        # Zero gradients
        self.optimizer.zero_grad()

        # Enhance audio
        enhanced = self.enhancer.enhance_audio(noisy, sr=16000)  # [B, samples]

        # Compute MSE loss (original DeepFilterNet loss)
        mse_loss = self.mse_loss(enhanced, clean)

        # Compute perceptual loss (if lambda > 0)
        if lambda_perceptual > 0:
            perceptual_loss = self.perceptual_loss_fn(enhanced, clean)
            total_loss = mse_loss + lambda_perceptual * perceptual_loss
        else:
            perceptual_loss = torch.tensor(0.0)
            total_loss = mse_loss

        # Backward pass
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.dfn_model.parameters(), max_norm=1.0)

        # Update weights
        self.optimizer.step()

        return {
            'mse_loss': mse_loss.item(),
            'perceptual_loss': perceptual_loss.item() if isinstance(perceptual_loss, torch.Tensor) else perceptual_loss,
            'total_loss': total_loss.item()
        }

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        stage, lr, lambda_perceptual = self.get_stage_params(epoch)

        # Setup optimizer if LR changed
        if self.optimizer is None or self.optimizer.param_groups[0]['lr'] != lr:
            self.setup_optimizer(lr)
            print(f"✓ Optimizer updated: LR={lr}, λ={lambda_perceptual}")

        self.dfn_model.train()

        epoch_losses = {
            'mse_loss': [],
            'perceptual_loss': [],
            'total_loss': []
        }

        print(f"\n{'=' * 80}")
        print(f"EPOCH {epoch}/{self.config.num_epochs} - Stage {stage}")
        print(f"LR: {lr}, λ_perceptual: {lambda_perceptual}")
        print(f"{'=' * 80}")

        for batch_idx, batch in enumerate(dataloader):
            losses = self.train_batch(batch, lambda_perceptual)

            # Accumulate losses
            for key in epoch_losses:
                epoch_losses[key].append(losses[key])

            # Log progress
            if (batch_idx + 1) % 10 == 0:
                print(f"Batch {batch_idx + 1}/{len(dataloader)}: "
                      f"MSE={losses['mse_loss']:.4f}, "
                      f"Perceptual={losses['perceptual_loss']:.4f}, "
                      f"Total={losses['total_loss']:.4f}")

            # Save checkpoint periodically
            if (batch_idx + 1) % self.config.save_interval == 0:
                self.save_checkpoint(epoch, batch_idx + 1, is_final=False)

            # Record history
            self.history['epochs'].append(epoch)
            self.history['batches'].append(batch_idx)
            self.history['mse_loss'].append(losses['mse_loss'])
            self.history['perceptual_loss'].append(losses['perceptual_loss'])
            self.history['total_loss'].append(losses['total_loss'])
            self.history['stage'].append(stage)

        # Compute epoch average
        avg_losses = {key: np.mean(values) for key, values in epoch_losses.items()}

        print(f"\n📊 Epoch {epoch} Average:")
        print(f"   MSE Loss: {avg_losses['mse_loss']:.4f}")
        print(f"   Perceptual Loss: {avg_losses['perceptual_loss']:.4f}")
        print(f"   Total Loss: {avg_losses['total_loss']:.4f}")

        return avg_losses

    def save_checkpoint(self, epoch: int, batch: int, is_final: bool = False):
        """Save model checkpoint"""
        if is_final:
            filename = f"dfn_perceptual_epoch{epoch}_final.pth"
        else:
            filename = f"dfn_perceptual_epoch{epoch}_batch{batch}.pth"

        path = os.path.join(self.config.checkpoint_dir, filename)

        torch.save({
            'epoch': epoch,
            'batch': batch,
            'model_state_dict': self.dfn_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }, path)

        print(f"✓ Checkpoint saved: {path}")

    def save_history(self):
        """Save training history to JSON and text"""
        # JSON
        json_path = os.path.join(self.config.log_dir, 'loss_history.json')
        with open(json_path, 'w') as f:
            json.dump(self.history, f, indent=2)

        # Text (readable)
        txt_path = os.path.join(self.config.log_dir, 'loss_history.txt')
        with open(txt_path, 'w') as f:
            f.write("Epoch\tBatch\tStage\tMSE_Loss\tPerceptual_Loss\tTotal_Loss\n")
            for i in range(len(self.history['epochs'])):
                f.write(f"{self.history['epochs'][i]}\t"
                        f"{self.history['batches'][i]}\t"
                        f"{self.history['stage'][i]}\t"
                        f"{self.history['mse_loss'][i]:.6f}\t"
                        f"{self.history['perceptual_loss'][i]:.6f}\t"
                        f"{self.history['total_loss'][i]:.6f}\n")

        print(f"✓ History saved: {json_path}, {txt_path}")

    def train(self, dataloader: DataLoader):
        """Full training loop"""
        print("\n" + "=" * 80)
        print("STARTING TRAINING")
        print("=" * 80)
        print(f"Total epochs: {self.config.num_epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Device: {self.device}")
        print("\nTraining stages:")
        print(f"  Stage 1 (Epoch 1-{self.config.stage1_end}): Warm-up with MSE only")
        print(f"  Stage 2 (Epoch {self.config.stage1_end + 1}-{self.config.stage2_end}): Add perceptual loss, small LR")
        print(f"  Stage 3 (Epoch {self.config.stage2_end + 1}-{self.config.stage3_end}): Full training")

        for epoch in range(1, self.config.num_epochs + 1):
            avg_losses = self.train_epoch(dataloader, epoch)

            # Save checkpoint at end of epoch
            self.save_checkpoint(epoch, len(dataloader), is_final=True)

            # Save history
            self.save_history()

        print("\n" + "=" * 80)
        print("✅ TRAINING COMPLETE!")
        print("=" * 80)
        print(f"Checkpoints saved in: {self.config.checkpoint_dir}")
        print(f"Logs saved in: {self.config.log_dir}")


# ==================== Main ====================
def main():
    """Main training function"""
    # Configuration
    config = Config()

    print("=" * 80)
    print("DEEPFILTERNET PERCEPTUAL LOSS TRAINING")
    print("=" * 80)
    print(f"Device: {config.device}")
    print(f"Batch size: {config.batch_size}")
    print(f"Epochs: {config.num_epochs}")
    print(f"SNR levels: {config.snr_levels}")

    # Create dataset
    print("\nCreating dataset...")
    dataset = VoxCelebNoiseDataset(
        voxceleb_dir=config.voxceleb_dir,
        musan_dir=config.musan_dir,
        snr_levels=config.snr_levels,
        target_length=3.0,  # 3 seconds
        augment=True,
        split='train'
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    print(f"✓ Dataset created: {len(dataset)} samples")
    print(f"✓ Batches per epoch: {len(dataloader)}")

    # Create trainer
    trainer = DFNPerceptualTrainer(config)

    # Train
    trainer.train(dataloader)


if __name__ == '__main__':
    main()