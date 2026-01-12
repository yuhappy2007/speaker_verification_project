"""
ECAPA-TDNN Wrapper for Speaker Embedding Extraction
使用 SpeechBrain 预训练模型
"""

import torch
import torch.nn as nn
from pathlib import Path


class ECAPATDNNWrapper(nn.Module):
    """
    ECAPA-TDNN 说话人模型 Wrapper

    使用 SpeechBrain 的预训练模型提取说话人嵌入
    """

    def __init__(self, model_path='pretrained_models/spkrec-ecapa-voxceleb',
                 device='cuda', freeze=True):
        """
        初始化 ECAPA-TDNN

        Args:
            model_path: 预训练模型路径
            device: 设备 ('cuda' 或 'cpu')
            freeze: 是否冻结模型参数
        """
        super().__init__()

        self.device = device
        self.model_path = Path(model_path)
        self.freeze = freeze

        print(f"\n🎤 Initializing ECAPA-TDNN Speaker Model...")
        print(f"   Model path: {self.model_path}")
        print(f"   Device: {self.device}")
        print(f"   Freeze: {self.freeze}")

        # 加载模型
        self._load_model()

        # 冻结参数
        if self.freeze:
            for param in self.parameters():
                param.requires_grad = False
            self.eval()
            print(f"   ✅ Model loaded and FROZEN")
        else:
            print(f"   ✅ Model loaded (trainable)")

    def _load_model(self):
        """加载 SpeechBrain 预训练模型"""
        try:
            from speechbrain.inference.speaker import EncoderClassifier

            # 加载预训练模型
            self.classifier = EncoderClassifier.from_hparams(
                source=str(self.model_path),
                run_opts={"device": self.device}
            )

            print(f"   ✅ SpeechBrain ECAPA-TDNN loaded successfully")

        except ImportError:
            raise ImportError(
                "SpeechBrain not installed! Please run:\n"
                "pip install speechbrain"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load ECAPA-TDNN: {e}")

    def forward(self, audio):
        """
        提取说话人嵌入

        Args:
            audio: 音频张量
                - [batch, samples] 或
                - [batch, 1, samples]

        Returns:
            embeddings: [batch, embedding_dim] (通常是 192)
        """
        # 确保输入格式正确
        if audio.dim() == 3:
            # [batch, 1, samples] -> [batch, samples]
            audio = audio.squeeze(1)

        batch_size = audio.size(0)

        # 提取嵌入（逐个样本，因为SpeechBrain的批处理可能有问题）
        embeddings_list = []

        for i in range(batch_size):
            # 获取单个音频样本 [samples]
            audio_sample = audio[i]

            # SpeechBrain encode_batch需要 [batch, samples]
            audio_batch = audio_sample.unsqueeze(0)  # [1, samples]

            # 提取嵌入
            with torch.no_grad() if self.freeze else torch.enable_grad():
                # encode_batch返回的是一个张量
                embedding = self.classifier.encode_batch(audio_batch)
                # embedding shape: [1, 1, embedding_dim] 需要squeeze
                embedding = embedding.squeeze()  # [embedding_dim]

            embeddings_list.append(embedding)

        # 堆叠成批次
        embeddings = torch.stack(embeddings_list)  # [batch, embedding_dim]

        # L2 归一化
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings

    def extract_embedding(self, audio):
        """
        便捷方法：提取单个音频的嵌入

        Args:
            audio: [samples] 或 [1, samples]

        Returns:
            embedding: [embedding_dim]
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)  # [samples] -> [1, samples]

        with torch.no_grad():
            embedding = self.forward(audio)

        return embedding.squeeze(0)  # [embedding_dim]


def test_ecapa_wrapper():
    """测试 ECAPA-TDNN Wrapper"""
    print("\n" + "=" * 60)
    print("Testing ECAPA-TDNN Wrapper")
    print("=" * 60)

    # 初始化模型
    model = ECAPATDNNWrapper(
        model_path='pretrained_models/spkrec-ecapa-voxceleb',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        freeze=True
    )

    # 测试不同输入格式
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    test_cases = [
        ("Single sample", torch.randn(1, 16000).to(device)),
        ("Batch [2, samples]", torch.randn(2, 16000).to(device)),
        ("Batch [2, 1, samples]", torch.randn(2, 1, 16000).to(device)),
    ]

    for name, test_input in test_cases:
        print(f"\n{name}:")
        print(f"  Input shape: {test_input.shape}")

        try:
            output = model(test_input)
            print(f"  Output shape: {output.shape}")
            print(f"  Output dtype: {output.dtype}")
            print(f"  Output norm: {torch.norm(output, dim=1).mean():.4f} (should be ~1.0)")
            print(f"  ✅ Success")
        except Exception as e:
            print(f"  ❌ Error: {e}")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    test_ecapa_wrapper()