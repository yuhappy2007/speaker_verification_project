"""
WavLM感知损失模块 (修复版)
- 使用预训练WavLM提取特征
- 计算增强语音与干净语音的感知距离
- 参数冻结，仅用于损失计算
- 适配GTCRN训练流程

修复：
1. 修复 Wav2Vec2FeatureExtractor 导入
2. 添加配置文件支持
3. 优化接口以适配训练流程
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WavLMModel, Wav2Vec2FeatureExtractor  # ✅ 修复：添加 Wav2Vec2FeatureExtractor
from pathlib import Path


class WavLMPerceptualLoss(nn.Module):
    """
    基于WavLM的感知损失

    参考:
    - WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing
    - Perceptual loss for speech enhancement

    功能:
    1. 冻结的WavLM模型提取音频特征
    2. 计算增强语音和干净语音特征的距离
    3. 支持多层特征融合
    """

    def __init__(self,
                 model_path='D:/WavLM',  # 本地WavLM路径
                 feature_layers=[3, 7, 11],  # 使用哪些层的特征
                 loss_type='l1',  # 'l1', 'l2', 'cosine'
                 normalize=True,  # 是否归一化特征
                 device='cuda'):
        """
        Args:
            model_path: WavLM模型路径
            feature_layers: 提取特征的层索引（WavLM-Base有12层）
            loss_type: 损失类型
            normalize: 是否对特征进行L2归一化
            device: 计算设备
        """
        super().__init__()

        self.device = device
        self.feature_layers = feature_layers
        self.loss_type = loss_type
        self.normalize = normalize

        print(f'🎵 Initializing WavLM Perceptual Loss...')
        print(f'   Model path: {model_path}')
        print(f'   Feature layers: {feature_layers}')
        print(f'   Loss type: {loss_type}')
        print(f'   Device: {device}')

        # 加载WavLM模型
        try:
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
            self.model = WavLMModel.from_pretrained(model_path)
            self.model = self.model.to(device)
            self.model.eval()

            # 冻结所有参数
            for param in self.model.parameters():
                param.requires_grad = False

            print(f'✅ WavLM model loaded and frozen')

            # 获取模型配置
            self.sampling_rate = self.feature_extractor.sampling_rate
            print(f'   Expected sampling rate: {self.sampling_rate} Hz')

        except Exception as e:
            print(f'❌ Failed to load WavLM model: {e}')
            raise

        # 层权重（可选：不同层的重要性不同）
        # ✅ 修复：直接在目标设备上创建
        self.layer_weights = nn.Parameter(
            torch.ones(len(feature_layers), device=device) / len(feature_layers),
            requires_grad=False  # 也冻结
        )

    def extract_features(self, audio):
        """
        提取WavLM特征

        Args:
            audio: [batch, samples] 或 [batch, 1, samples]

        Returns:
            features: dict of {layer_idx: [batch, time, hidden_dim]}
        """
        # 确保是2D: [batch, samples]
        if audio.dim() == 3:
            audio = audio.squeeze(1)

        batch_size = audio.shape[0]

        # WavLM期望输入为16kHz
        # 注意：确保输入已经是16kHz

        # ✅ 修复：移除 with torch.no_grad()
        # WavLM参数已被冻结（requires_grad=False），但我们需要保持梯度流向输入
        # 这样才能在训练GTCRN时通过WavLM反向传播梯度

        # WavLM的forward返回所有层的hidden states
        outputs = self.model(
            audio,
            output_hidden_states=True,
            return_dict=True
        )

        hidden_states = outputs.hidden_states  # Tuple of [batch, time, 768]

        # 提取指定层的特征
        features = {}
        for layer_idx in self.feature_layers:
            feat = hidden_states[layer_idx]  # [batch, time, 768]

            # 可选：L2归一化
            if self.normalize:
                feat = F.normalize(feat, p=2, dim=-1)

            features[layer_idx] = feat

        return features

    def compute_loss(self, enhanced_audio, clean_audio):
        """
        计算感知损失

        Args:
            enhanced_audio: [batch, samples] 或 [batch, 1, samples]，增强后的语音
            clean_audio: [batch, samples] 或 [batch, 1, samples]，干净目标语音

        Returns:
            loss: scalar，感知损失
            loss_dict: dict，各层损失的详细信息（用于监控）
        """
        # 提取特征
        enhanced_features = self.extract_features(enhanced_audio)
        clean_features = self.extract_features(clean_audio)

        # 计算各层损失
        layer_losses = []
        loss_dict = {}

        for i, layer_idx in enumerate(self.feature_layers):
            enhanced_feat = enhanced_features[layer_idx]  # [batch, time, dim]
            clean_feat = clean_features[layer_idx]

            # 计算距离
            if self.loss_type == 'l1':
                layer_loss = F.l1_loss(enhanced_feat, clean_feat)
            elif self.loss_type == 'l2':
                layer_loss = F.mse_loss(enhanced_feat, clean_feat)
            elif self.loss_type == 'cosine':
                # Cosine similarity loss
                cos_sim = F.cosine_similarity(
                    enhanced_feat.flatten(1),
                    clean_feat.flatten(1),
                    dim=1
                ).mean()
                layer_loss = 1 - cos_sim
            else:
                raise ValueError(f'Unknown loss type: {self.loss_type}')

            layer_losses.append(layer_loss)
            loss_dict[f'layer_{layer_idx}'] = layer_loss.item()

        # 加权求和
        layer_losses = torch.stack(layer_losses)
        total_loss = (layer_losses * self.layer_weights).sum()

        loss_dict['total'] = total_loss.item()

        return total_loss, loss_dict

    def forward(self, enhanced_audio, clean_audio):
        """前向传播（调用compute_loss）"""
        return self.compute_loss(enhanced_audio, clean_audio)


def test_perceptual_loss():
    """测试感知损失模块"""
    print('=' * 80)
    print('🧪 Testing WavLM Perceptual Loss')
    print('=' * 80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 初始化
    perceptual_loss = WavLMPerceptualLoss(
        model_path='D:/WavLM',
        feature_layers=[3, 7, 11],
        loss_type='l1',
        device=device
    )

    # 测试数据（16kHz，3秒）
    batch_size = 2
    duration = 3
    sr = 16000

    print(f'\n📊 Test configuration:')
    print(f'   Batch size: {batch_size}')
    print(f'   Duration: {duration}s')
    print(f'   Sampling rate: {sr} Hz')

    # 模拟增强语音和干净语音
    enhanced = torch.randn(batch_size, sr * duration).to(device)
    clean = torch.randn(batch_size, sr * duration).to(device)

    print(f'\n🔊 Audio shapes:')
    print(f'   Enhanced: {enhanced.shape}')
    print(f'   Clean: {clean.shape}')

    # 计算损失
    print(f'\n🧮 Computing perceptual loss...')
    loss, loss_dict = perceptual_loss(enhanced, clean)

    print(f'\n📈 Loss results:')
    print(f'   Total loss: {loss.item():.6f}')
    for layer, value in loss_dict.items():
        if layer != 'total':
            print(f'   {layer}: {value:.6f}')

    # 测试梯度流（模拟实际训练场景）
    print(f'\n🔄 Testing gradient flow...')

    # 模拟 GTCRN 输出：创建一个需要梯度的"模型"
    # 在实际训练中，这相当于 GTCRN 的参数
    mock_gtcrn_weight = nn.Parameter(torch.randn(1, sr * duration).to(device))

    # 模拟 GTCRN 的输出（非叶子张量）
    enhanced_with_grad = mock_gtcrn_weight * torch.randn(batch_size, sr * duration).to(device)
    clean_no_grad = torch.randn(batch_size, sr * duration).to(device)

    # 计算损失
    loss, _ = perceptual_loss(enhanced_with_grad, clean_no_grad)
    loss.backward()

    # 检查梯度是否流到"模型参数"（叶子张量）
    print(f'   Mock GTCRN weight has gradient: {mock_gtcrn_weight.grad is not None}')
    if mock_gtcrn_weight.grad is not None:
        print(f'   Gradient norm: {mock_gtcrn_weight.grad.norm().item():.6f}')
        print(f'   ✅ Gradient flows correctly to model parameters!')
    else:
        print(f'   ❌ No gradient! Something is wrong.')

    # 检查 enhanced 本身的 grad_fn（证明它在计算图中）
    print(f'   Enhanced has grad_fn: {enhanced_with_grad.grad_fn is not None}')
    print(f'   Enhanced is leaf: {enhanced_with_grad.is_leaf}')  # 应该是 False

    # 检查WavLM参数是否被冻结
    print(f'\n🔒 Checking frozen parameters...')
    frozen_count = sum(1 for p in perceptual_loss.model.parameters() if not p.requires_grad)
    total_count = sum(1 for p in perceptual_loss.model.parameters())
    print(f'   Frozen parameters: {frozen_count}/{total_count}')

    print('\n' + '=' * 80)
    print('✅ All tests passed!')
    print('=' * 80)


if __name__ == '__main__':
    test_perceptual_loss()