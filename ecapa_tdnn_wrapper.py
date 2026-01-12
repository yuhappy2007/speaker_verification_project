# # """
# # ECAPA-TDNN Wrapper for Speaker Embedding Extraction
# # 使用 SpeechBrain 预训练模型
# # """
# #
# # import torch
# # import torch.nn as nn
# # from pathlib import Path
# #
# #
# # class ECAPATDNNWrapper(nn.Module):
# #     """
# #     ECAPA-TDNN 说话人模型 Wrapper
# #
# #     使用 SpeechBrain 的预训练模型提取说话人嵌入
# #     """
# #
# #     def __init__(self, model_path='pretrained_models/spkrec-ecapa-voxceleb',
# #                  device='cuda', freeze=True):
# #         """
# #         初始化 ECAPA-TDNN
# #
# #         Args:
# #             model_path: 预训练模型路径
# #             device: 设备 ('cuda' 或 'cpu')
# #             freeze: 是否冻结模型参数
# #         """
# #         super().__init__()
# #
# #         self.device = device
# #         self.model_path = Path(model_path)
# #         self.freeze = freeze
# #
# #         print(f"\n🎤 Initializing ECAPA-TDNN Speaker Model...")
# #         print(f"   Model path: {self.model_path}")
# #         print(f"   Device: {self.device}")
# #         print(f"   Freeze: {self.freeze}")
# #
# #         # 加载模型
# #         self._load_model()
# #
# #         # 冻结参数
# #         if self.freeze:
# #             for param in self.parameters():
# #                 param.requires_grad = False
# #             self.eval()
# #             print(f"   ✅ Model loaded and FROZEN")
# #         else:
# #             print(f"   ✅ Model loaded (trainable)")
# #
# #     def _load_model(self):
# #         """加载 SpeechBrain 预训练模型"""
# #         try:
# #             from speechbrain.inference.speaker import EncoderClassifier
# #
# #             # 加载预训练模型
# #             self.classifier = EncoderClassifier.from_hparams(
# #                 source=str(self.model_path),
# #                 run_opts={"device": self.device}
# #             )
# #
# #             print(f"   ✅ SpeechBrain ECAPA-TDNN loaded successfully")
# #
# #         except ImportError:
# #             raise ImportError(
# #                 "SpeechBrain not installed! Please run:\n"
# #                 "pip install speechbrain"
# #             )
# #         except Exception as e:
# #             raise RuntimeError(f"Failed to load ECAPA-TDNN: {e}")
# #
# #     def forward(self, audio):
# #         """
# #         提取说话人嵌入
# #
# #         Args:
# #             audio: 音频张量
# #                 - [batch, samples] 或
# #                 - [batch, 1, samples]
# #
# #         Returns:
# #             embeddings: [batch, embedding_dim] (通常是 192)
# #         """
# #         # 确保输入格式正确
# #         if audio.dim() == 3:
# #             # [batch, 1, samples] -> [batch, samples]
# #             audio = audio.squeeze(1)
# #
# #         batch_size = audio.size(0)
# #
# #         # 提取嵌入（逐个样本，因为SpeechBrain的批处理可能有问题）
# #         embeddings_list = []
# #
# #         for i in range(batch_size):
# #             # 获取单个音频样本 [samples]
# #             audio_sample = audio[i]
# #
# #             # SpeechBrain encode_batch需要 [batch, samples]
# #             audio_batch = audio_sample.unsqueeze(0)  # [1, samples]
# #
# #             # 提取嵌入
# #             with torch.no_grad() if self.freeze else torch.enable_grad():
# #                 # encode_batch返回的是一个张量
# #                 embedding = self.classifier.encode_batch(audio_batch)
# #                 # embedding shape: [1, 1, embedding_dim] 需要squeeze
# #                 embedding = embedding.squeeze()  # [embedding_dim]
# #
# #             embeddings_list.append(embedding)
# #
# #         # 堆叠成批次
# #         embeddings = torch.stack(embeddings_list)  # [batch, embedding_dim]
# #
# #         # L2 归一化
# #         embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
# #
# #         return embeddings
# #
# #     def extract_embedding(self, audio):
# #         """
# #         便捷方法：提取单个音频的嵌入
# #
# #         Args:
# #             audio: [samples] 或 [1, samples]
# #
# #         Returns:
# #             embedding: [embedding_dim]
# #         """
# #         if audio.dim() == 1:
# #             audio = audio.unsqueeze(0)  # [samples] -> [1, samples]
# #
# #         with torch.no_grad():
# #             embedding = self.forward(audio)
# #
# #         return embedding.squeeze(0)  # [embedding_dim]
# #
# #
# # def test_ecapa_wrapper():
# #     """测试 ECAPA-TDNN Wrapper"""
# #     print("\n" + "=" * 60)
# #     print("Testing ECAPA-TDNN Wrapper")
# #     print("=" * 60)
# #
# #     # 初始化模型
# #     model = ECAPATDNNWrapper(
# #         model_path='pretrained_models/spkrec-ecapa-voxceleb',
# #         device='cuda' if torch.cuda.is_available() else 'cpu',
# #         freeze=True
# #     )
# #
# #     # 测试不同输入格式
# #     device = 'cuda' if torch.cuda.is_available() else 'cpu'
# #
# #     test_cases = [
# #         ("Single sample", torch.randn(1, 16000).to(device)),
# #         ("Batch [2, samples]", torch.randn(2, 16000).to(device)),
# #         ("Batch [2, 1, samples]", torch.randn(2, 1, 16000).to(device)),
# #     ]
# #
# #     for name, test_input in test_cases:
# #         print(f"\n{name}:")
# #         print(f"  Input shape: {test_input.shape}")
# #
# #         try:
# #             output = model(test_input)
# #             print(f"  Output shape: {output.shape}")
# #             print(f"  Output dtype: {output.dtype}")
# #             print(f"  Output norm: {torch.norm(output, dim=1).mean():.4f} (should be ~1.0)")
# #             print(f"  ✅ Success")
# #         except Exception as e:
# #             print(f"  ❌ Error: {e}")
# #
# #     print("\n" + "=" * 60)
# #
# #
# # if __name__ == '__main__':
# #     test_ecapa_wrapper()
# """
# ECAPA-TDNN Wrapper for Speaker Embedding Extraction
# 使用 SpeechBrain 预训练模型
#
# 修复：
# - Windows 路径兼容性（使用 as_posix() 转换为正斜杠）
# - 确保路径格式正确传递给 HuggingFace
# """
#
# import torch
# import torch.nn as nn
# from pathlib import Path
#
#
# class ECAPATDNNWrapper(nn.Module):
#     """
#     ECAPA-TDNN 说话人模型 Wrapper
#
#     使用 SpeechBrain 的预训练模型提取说话人嵌入
#     """
#
#     def __init__(self, model_path='pretrained_models/spkrec-ecapa-voxceleb',
#                  device='cuda', freeze=True):
#         """
#         初始化 ECAPA-TDNN
#
#         Args:
#             model_path: 预训练模型路径
#             device: 设备 ('cuda' 或 'cpu')
#             freeze: 是否冻结模型参数
#         """
#         super().__init__()
#
#         self.device = device
#         self.freeze = freeze
#
#         # ✅ 修复：使用 Path 对象但转换为 POSIX 格式（正斜杠）
#         # 这样在 Windows 和 Linux 上都能正常工作
#         model_path_obj = Path(model_path)
#
#         # 如果是相对路径，转换为绝对路径
#         if not model_path_obj.is_absolute():
#             model_path_obj = model_path_obj.resolve()
#
#         # 转换为 POSIX 格式（使用正斜杠）
#         self.model_path_str = model_path_obj.as_posix()
#
#         print(f"\n🎤 Initializing ECAPA-TDNN Speaker Model...")
#         print(f"   Model path: {self.model_path_str}")
#         print(f"   Device: {self.device}")
#         print(f"   Freeze: {self.freeze}")
#
#         # 加载模型
#         self._load_model()
#
#         # 冻结参数
#         if self.freeze:
#             for param in self.parameters():
#                 param.requires_grad = False
#             self.eval()
#             print(f"   ✅ Model loaded and FROZEN")
#         else:
#             print(f"   ✅ Model loaded (trainable)")
#
#     def _load_model(self):
#         """加载 SpeechBrain 预训练模型"""
#         try:
#             from speechbrain.inference.speaker import EncoderClassifier
#
#             # ✅ 修复：使用 POSIX 格式路径（正斜杠）
#             # HuggingFace 不接受反斜杠路径
#             self.classifier = EncoderClassifier.from_hparams(
#                 source=self.model_path_str,  # 使用 POSIX 格式
#                 run_opts={"device": self.device}
#             )
#
#             print(f"   ✅ SpeechBrain ECAPA-TDNN loaded successfully")
#
#         except ImportError:
#             raise ImportError(
#                 "SpeechBrain not installed! Please run:\n"
#                 "pip install speechbrain"
#             )
#         except Exception as e:
#             print(f"   ❌ Error details: {e}")
#             print(f"   Model path used: {self.model_path_str}")
#             raise RuntimeError(f"Failed to load ECAPA-TDNN: {e}")
#
#     def forward(self, audio):
#         """
#         提取说话人嵌入
#
#         Args:
#             audio: 音频张量
#                 - [batch, samples] 或
#                 - [batch, 1, samples]
#
#         Returns:
#             embeddings: [batch, embedding_dim] (通常是 192)
#         """
#         # 确保输入格式正确
#         if audio.dim() == 3:
#             # [batch, 1, samples] -> [batch, samples]
#             audio = audio.squeeze(1)
#
#         batch_size = audio.size(0)
#
#         # 提取嵌入（逐个样本，因为SpeechBrain的批处理可能有问题）
#         embeddings_list = []
#
#         for i in range(batch_size):
#             # 获取单个音频样本 [samples]
#             audio_sample = audio[i]
#
#             # SpeechBrain encode_batch需要 [batch, samples]
#             audio_batch = audio_sample.unsqueeze(0)  # [1, samples]
#
#             # 提取嵌入
#             with torch.no_grad() if self.freeze else torch.enable_grad():
#                 # encode_batch返回的是一个张量
#                 embedding = self.classifier.encode_batch(audio_batch)
#                 # embedding shape: [1, 1, embedding_dim] 需要squeeze
#                 embedding = embedding.squeeze()  # [embedding_dim]
#
#             embeddings_list.append(embedding)
#
#         # 堆叠成批次
#         embeddings = torch.stack(embeddings_list)  # [batch, embedding_dim]
#
#         # L2 归一化
#         embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
#
#         return embeddings
#
#     def extract_embedding(self, audio):
#         """
#         便捷方法：提取单个音频的嵌入
#
#         Args:
#             audio: [samples] 或 [1, samples]
#
#         Returns:
#             embedding: [embedding_dim]
#         """
#         if audio.dim() == 1:
#             audio = audio.unsqueeze(0)  # [samples] -> [1, samples]
#
#         with torch.no_grad():
#             embedding = self.forward(audio)
#
#         return embedding.squeeze(0)  # [embedding_dim]
#
#
# def test_ecapa_wrapper():
#     """测试 ECAPA-TDNN Wrapper"""
#     print("\n" + "=" * 60)
#     print("Testing ECAPA-TDNN Wrapper")
#     print("=" * 60)
#
#     # 初始化模型
#     model = ECAPATDNNWrapper(
#         model_path='pretrained_models/spkrec-ecapa-voxceleb',
#         device='cuda' if torch.cuda.is_available() else 'cpu',
#         freeze=True
#     )
#
#     # 测试不同输入格式
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#
#     test_cases = [
#         ("Single sample", torch.randn(1, 16000).to(device)),
#         ("Batch [2, samples]", torch.randn(2, 16000).to(device)),
#         ("Batch [2, 1, samples]", torch.randn(2, 1, 16000).to(device)),
#     ]
#
#     for name, test_input in test_cases:
#         print(f"\n{name}:")
#         print(f"  Input shape: {test_input.shape}")
#
#         try:
#             output = model(test_input)
#             print(f"  Output shape: {output.shape}")
#             print(f"  Output dtype: {output.dtype}")
#             print(f"  Output norm: {torch.norm(output, dim=1).mean():.4f} (should be ~1.0)")
#             print(f"  ✅ Success")
#         except Exception as e:
#             print(f"  ❌ Error: {e}")
#
#     print("\n" + "=" * 60)
#
#
# if __name__ == '__main__':
#     test_ecapa_wrapper()
"""
ECAPA-TDNN Wrapper for Speaker Embedding Extraction
使用 SpeechBrain 预训练模型（离线加载版本）

修复：
- Windows 路径兼容性（使用 as_posix() 转换为正斜杠）
- 完全离线加载，不需要网络连接
- 直接从本地文件加载模型
"""

import torch
import torch.nn as nn
from pathlib import Path
import os


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
        self.freeze = freeze

        # 使用 Path 对象但转换为 POSIX 格式（正斜杠）
        model_path_obj = Path(model_path)

        # 如果是相对路径，转换为绝对路径
        if not model_path_obj.is_absolute():
            model_path_obj = model_path_obj.resolve()

        # 转换为 POSIX 格式（使用正斜杠）
        self.model_path_str = model_path_obj.as_posix()
        self.model_path = model_path_obj

        print(f"\n🎤 Initializing ECAPA-TDNN Speaker Model...")
        print(f"   Model path: {self.model_path_str}")
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
        """加载 SpeechBrain 预训练模型（完全离线）"""
        try:
            from speechbrain.inference.speaker import EncoderClassifier

            # ✅ 关键：设置环境变量强制离线模式
            os.environ['HF_HUB_OFFLINE'] = '1'
            os.environ['TRANSFORMERS_OFFLINE'] = '1'

            print(f"   🔒 Offline mode enabled")

            # 方法1：尝试使用 local_files_only（推荐）
            try:
                self.classifier = EncoderClassifier.from_hparams(
                    source=self.model_path_str,
                    run_opts={"device": self.device},
                    local_files_only=True  # ✅ 强制只使用本地文件
                )
                print(f"   ✅ Loaded with local_files_only=True")

            except Exception as e1:
                print(f"   ⚠️ Method 1 failed: {e1}")
                print(f"   📂 Trying alternative loading method...")

                # 方法2：直接指定 savedir（备选方案）
                try:
                    # 检查必要文件是否存在
                    required_files = [
                        'hyperparams.yaml',
                        'embedding_model.ckpt',
                        'classifier.ckpt',
                        'mean_var_norm_emb.ckpt'
                    ]

                    missing_files = []
                    for fname in required_files:
                        fpath = self.model_path / fname
                        if not fpath.exists():
                            missing_files.append(fname)

                    if missing_files:
                        raise FileNotFoundError(
                            f"Missing required files in {self.model_path}: {missing_files}\n"
                            f"Please ensure the model is properly downloaded."
                        )

                    # 使用 savedir 参数（告诉它文件已经在这里了）
                    self.classifier = EncoderClassifier.from_hparams(
                        source=self.model_path_str,
                        savedir=self.model_path_str,  # ✅ 指定文件位置
                        run_opts={"device": self.device}
                    )
                    print(f"   ✅ Loaded with savedir parameter")

                except Exception as e2:
                    raise RuntimeError(
                        f"Failed to load ECAPA-TDNN with both methods.\n"
                        f"Method 1 error: {e1}\n"
                        f"Method 2 error: {e2}\n"
                        f"Model path: {self.model_path_str}\n"
                        f"\nTroubleshooting:\n"
                        f"1. Check if all model files exist in {self.model_path}\n"
                        f"2. Verify symlinks are valid (if using symlinks)\n"
                        f"3. Try copying actual files instead of symlinks"
                    )

            print(f"   ✅ SpeechBrain ECAPA-TDNN loaded successfully")

        except ImportError:
            raise ImportError(
                "SpeechBrain not installed! Please run:\n"
                "pip install speechbrain"
            )
        except Exception as e:
            print(f"   ❌ Error details: {e}")
            print(f"   Model path used: {self.model_path_str}")
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