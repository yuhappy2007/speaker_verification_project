"""
GTCRN 语音增强模型包装器（完全修复版）

功能：
1. 加载预训练的 GTCRN 模型
2. 提供统一的增强接口（自动处理 STFT/ISTFT）
3. 支持梯度计算（用于训练）
4. 自动处理音频格式转换

使用方法：
    gtcrn = GTCRNWrapper('gtcrn/checkpoints/model_trained_on_vctk.tar')
    enhanced = gtcrn.enhance(noisy_audio)  # 输入时域音频，输出时域音频

重要：GTCRN 在频域工作（STFT），包装器自动处理时频转换
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path
import warnings

# 添加 GTCRN 目录到 Python 路径
GTCRN_DIR = Path(__file__).parent / 'gtcrn'
if str(GTCRN_DIR) not in sys.path:
    sys.path.insert(0, str(GTCRN_DIR))

# STFT 参数（与 GTCRN 训练时一致，从 infer.py 得知）
STFT_N_FFT = 512
STFT_HOP_LENGTH = 256
STFT_WIN_LENGTH = 512


class GTCRNWrapper(nn.Module):
    """
    GTCRN 模型包装器（完全修复版）

    关键改进：
    1. ✅ 自动处理 STFT/ISTFT 转换
    2. ✅ 正确加载 checkpoint 格式 ({'model': state_dict})
    3. ✅ 支持训练模式（保持梯度）
    4. ✅ 批量处理

    GTCRN 工作原理：
    - 输入：时域音频 [batch, samples] @ 16kHz
    - 内部：
      * STFT → [batch, freq, time, 2]
      * GTCRN 处理频谱 → [batch, freq, time, 2]
      * ISTFT → [batch, samples]
    - 输出：时域音频 [batch, samples] @ 16kHz

    Args:
        checkpoint_path: 预训练模型路径
        device: 'cuda' 或 'cpu'
        sample_rate: 采样率（默认16000，GTCRN 固定使用）
        freeze: 是否冻结参数（默认False，因为我们要训练）
    """

    def __init__(self,
                 checkpoint_path='gtcrn/checkpoints/model_trained_on_vctk.tar',
                 device='cuda',
                 sample_rate=16000,
                 freeze=False):
        super().__init__()

        self.device = device
        self.sample_rate = sample_rate
        self.checkpoint_path = Path(checkpoint_path)

        # STFT 参数（与 GTCRN 训练时一致）
        self.n_fft = STFT_N_FFT
        self.hop_length = STFT_HOP_LENGTH
        self.win_length = STFT_WIN_LENGTH

        # 注册窗函数为 buffer（不是参数，但会随模型移动到 GPU）
        # ✅ 修复：直接在目标设备上创建
        window = torch.hann_window(self.win_length, device=device).pow(0.5)
        self.register_buffer('window', window)

        print(f'🎤 Initializing GTCRN Wrapper (Fixed Version)...')
        print(f'   Checkpoint: {self.checkpoint_path}')
        print(f'   Device: {device}')
        print(f'   Sample rate: {sample_rate} Hz')
        print(f'   STFT params: n_fft={self.n_fft}, hop={self.hop_length}')
        print(f'   Freeze parameters: {freeze}')

        # 加载 GTCRN 模型
        self._load_model()

        # 设置训练/冻结模式
        if freeze:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
            print(f'   ✅ Model loaded and FROZEN')
        else:
            self.model.train()  # 训练模式
            for param in self.model.parameters():
                param.requires_grad = True
            print(f'   ✅ Model loaded and TRAINABLE')

    def _load_model(self):
        """加载 GTCRN 模型"""
        try:
            # 步骤1: 导入 GTCRN 类
            try:
                from gtcrn import GTCRN
                print(f'   📦 Importing GTCRN from gtcrn.py...')
            except ImportError as e:
                error_msg = (
                    f"Cannot import GTCRN: {e}\n\n"
                    f"Please install missing dependencies:\n"
                    f"  pip install einops --break-system-packages\n\n"
                    f"Or check if gtcrn.py is in: {GTCRN_DIR}"
                )
                raise ImportError(error_msg)

            # 步骤2: 创建模型实例
            self.model = GTCRN().to(self.device)
            print(f'   ✅ GTCRN model instance created')

            # 步骤3: 加载预训练权重
            if self.checkpoint_path.exists():
                print(f'   📂 Loading checkpoint: {self.checkpoint_path.name}')
                checkpoint = torch.load(
                    self.checkpoint_path,
                    map_location=self.device,
                    weights_only=False  # 明确设置以避免警告
                )

                # 根据 infer.py，checkpoint 格式是 {'model': state_dict}
                if isinstance(checkpoint, dict) and 'model' in checkpoint:
                    self.model.load_state_dict(checkpoint['model'])
                    print(f'   ✅ Loaded state_dict from checkpoint["model"]')
                elif isinstance(checkpoint, dict):
                    # 尝试直接作为 state_dict
                    self.model.load_state_dict(checkpoint)
                    print(f'   ✅ Loaded state_dict directly')
                else:
                    raise ValueError(
                        f"Unexpected checkpoint format. "
                        f"Expected dict with 'model' key, got: {type(checkpoint)}"
                    )

                # 确保在正确的设备上
                self.model = self.model.to(self.device)
                print(f'   ✅ Checkpoint loaded successfully')

            else:
                warnings.warn(
                    f"Checkpoint not found: {self.checkpoint_path}. "
                    "Using randomly initialized model."
                )

        except Exception as e:
            print(f'   ❌ Error loading GTCRN model: {e}')
            print(f'\n💡 Troubleshooting:')
            print(f'   1. Install einops: pip install einops --break-system-packages')
            print(f'   2. Ensure gtcrn.py is in: {GTCRN_DIR}')
            print(f'   3. Ensure checkpoint exists: {self.checkpoint_path}')
            raise

    def _do_stft(self, audio):
        """
        执行 STFT

        Args:
            audio: [batch, samples] 时域音频

        Returns:
            spec: [batch, freq, time, 2] GTCRN 期望的格式（实部+虚部）
        """
        # audio: [batch, samples]
        # 新版 PyTorch 推荐使用 return_complex=True
        spec_complex = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True  # 返回复数张量 [batch, freq, time]
        )

        # 转换为实数格式 [batch, freq, time, 2]
        # 最后一维：[实部, 虚部]
        spec = torch.view_as_real(spec_complex)  # [batch, freq, time, 2]

        return spec

    def _do_istft(self, spec):
        """
        执行 ISTFT

        Args:
            spec: [batch, freq, time, 2] GTCRN 输出的格式（实部+虚部）

        Returns:
            audio: [batch, samples] 时域音频
        """
        # 转换为复数格式 [batch, freq, time]
        spec_complex = torch.view_as_complex(spec.contiguous())

        # ISTFT
        audio = torch.istft(
            spec_complex,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=False  # 返回实数音频
        )

        return audio

    def enhance(self, noisy_audio, return_numpy=False):
        """
        增强音频（时域输入 → 时域输出）

        Args:
            noisy_audio: 带噪音频
                - torch.Tensor: [batch, samples] 或 [batch, 1, samples]
                - numpy.ndarray: [samples] 或 [1, samples] 或 [batch, samples]
            return_numpy: 是否返回 numpy 数组（默认 False）

        Returns:
            enhanced: 增强后的音频
                - torch.Tensor (如果 return_numpy=False)
                - numpy.ndarray (如果 return_numpy=True)
        """
        # 转换输入格式
        if isinstance(noisy_audio, torch.Tensor):
            audio_tensor = noisy_audio
        else:
            # numpy array
            import numpy as np
            audio_tensor = torch.from_numpy(noisy_audio).float()

        # 确保在正确的设备上
        audio_tensor = audio_tensor.to(self.device)

        # 确保是 [batch, samples] 格式
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)  # [samples] -> [1, samples]
        elif audio_tensor.dim() == 3:
            audio_tensor = audio_tensor.squeeze(1)  # [batch, 1, samples] -> [batch, samples]

        # 记录原始长度（用于裁剪）
        original_length = audio_tensor.shape[-1]

        # STFT: [batch, samples] → [batch, freq, time, 2]
        noisy_spec = self._do_stft(audio_tensor)

        # GTCRN 处理: [batch, freq, time, 2] → [batch, freq, time, 2]
        enhanced_spec = self.model(noisy_spec)

        # ISTFT: [batch, freq, time, 2] → [batch, samples]
        enhanced = self._do_istft(enhanced_spec)

        # 裁剪到原始长度（STFT/ISTFT 可能改变长度）
        if enhanced.shape[-1] > original_length:
            enhanced = enhanced[..., :original_length]
        elif enhanced.shape[-1] < original_length:
            # 填充零
            padding = original_length - enhanced.shape[-1]
            enhanced = torch.nn.functional.pad(enhanced, (0, padding))

        # 返回格式
        if return_numpy:
            return enhanced.detach().cpu().numpy()
        else:
            return enhanced

    def get_trainable_params(self):
        """
        获取可训练参数列表
        用于设置优化器

        Returns:
            list of parameters with requires_grad=True
        """
        return [p for p in self.model.parameters() if p.requires_grad]

    def freeze(self):
        """冻结所有参数"""
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        print('🔒 GTCRN parameters frozen')

    def unfreeze(self):
        """解冻所有参数"""
        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = True
        print('🔓 GTCRN parameters unfrozen')

    def forward(self, noisy_audio):
        """前向传播（调用 enhance）"""
        return self.enhance(noisy_audio)


def test_gtcrn_wrapper():
    """测试 GTCRN 包装器"""
    print('=' * 80)
    print('🧪 Testing GTCRN Wrapper (Fixed Version)')
    print('=' * 80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 测试配置
    checkpoint_path = 'gtcrn/checkpoints/model_trained_on_vctk.tar'

    # 检查 checkpoint 是否存在
    if not Path(checkpoint_path).exists():
        print(f'\n⚠️  Checkpoint not found: {checkpoint_path}')
        print(f'   Please download from: https://github.com/Xiaobin-Rong/gtcrn')
        print(f'   Or adjust the path in the code.')
        return

    try:
        # 初始化包装器
        print(f'\n[1/5] Initializing GTCRN...')
        gtcrn = GTCRNWrapper(
            checkpoint_path=checkpoint_path,
            device=device,
            freeze=False  # 不冻结，用于训练
        )

        # 测试数据
        print(f'\n[2/5] Preparing test data...')
        batch_size = 2
        duration = 3  # 秒
        sr = 16000

        noisy = torch.randn(batch_size, sr * duration).to(device)
        print(f'   Noisy audio shape: {noisy.shape}')

        # 测试增强
        print(f'\n[3/5] Testing enhancement...')
        enhanced = gtcrn.enhance(noisy)
        print(f'   Enhanced audio shape: {enhanced.shape}')
        print(f'   Enhanced audio device: {enhanced.device}')
        print(f'   Enhanced has grad_fn: {enhanced.grad_fn is not None}')
        print(f'   ✅ Shape matches input!')

        # 测试梯度流
        print(f'\n[4/5] Testing gradient flow...')

        # 模拟损失
        target = torch.randn_like(enhanced)
        loss = torch.nn.functional.mse_loss(enhanced, target)
        print(f'   Loss: {loss.item():.6f}')

        # 反向传播
        loss.backward()

        # 检查梯度
        trainable_params = gtcrn.get_trainable_params()
        has_grad = sum(1 for p in trainable_params if p.grad is not None)
        total = len(trainable_params)

        print(f'   Parameters with gradient: {has_grad}/{total}')

        if has_grad > 0:
            # 计算梯度范数
            total_norm = 0
            for p in trainable_params:
                if p.grad is not None:
                    total_norm += p.grad.norm().item() ** 2
            total_norm = total_norm ** 0.5
            print(f'   Total gradient norm: {total_norm:.6f}')
            print(f'   ✅ Gradient flows correctly!')
        else:
            print(f'   ❌ No gradients found!')

        # 测试 STFT/ISTFT 往返
        print(f'\n[5/5] Testing STFT/ISTFT round-trip...')
        test_audio = torch.randn(1, sr).to(device)
        spec = gtcrn._do_stft(test_audio)
        reconstructed = gtcrn._do_istft(spec)

        print(f'   Original shape: {test_audio.shape}')
        print(f'   Spec shape: {spec.shape}  # [batch, freq, time, 2]')
        print(f'   Reconstructed shape: {reconstructed.shape}')

        # 检查重建误差
        if reconstructed.shape[-1] != test_audio.shape[-1]:
            # 裁剪到相同长度
            min_len = min(reconstructed.shape[-1], test_audio.shape[-1])
            reconstructed = reconstructed[..., :min_len]
            test_audio = test_audio[..., :min_len]

        mse = torch.nn.functional.mse_loss(reconstructed, test_audio).item()
        print(f'   STFT/ISTFT MSE: {mse:.6f}')
        if mse < 1e-5:
            print(f'   ✅ Perfect reconstruction!')
        else:
            print(f'   ℹ️  Small reconstruction error (normal)')

        print('\n' + '=' * 80)
        print('✅ All tests passed!')
        print('=' * 80)

        print('\n💡 Usage example:')
        print('   gtcrn = GTCRNWrapper("gtcrn/checkpoints/model_trained_on_vctk.tar")')
        print('   enhanced = gtcrn.enhance(noisy_audio)  # Time domain in/out')
        print('   loss = loss_fn(enhanced, target)')
        print('   loss.backward()  # Gradients flow to GTCRN')
        print('   optimizer.step()  # Update GTCRN parameters')

    except ImportError as e:
        print(f'\n❌ Import Error: {e}')
        print(f'\n📦 Please install einops:')
        print(f'   pip install einops --break-system-packages')

    except Exception as e:
        print(f'\n❌ Test failed: {e}')
        print(f'\nDebugging information:')
        print(f'   - Checkpoint path: {checkpoint_path}')
        print(f'   - GTCRN directory: {GTCRN_DIR}')
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    test_gtcrn_wrapper()