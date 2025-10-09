# '''
# 说话人嵌入提取器 - 避免符号链接问题
# '''
# import torch
# import torchaudio
# from pathlib import Path
#
# class SpeakerEmbeddingExtractor:
#     def __init__(self, model_name='ecapa'):
#         self.model_name = model_name
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self._load_model()
#
#     def _load_model(self):
#         '''加载预训练模型 - 直接从缓存使用'''
#         print(f'Loading {self.model_name} model...')
#
#         if self.model_name == 'ecapa':
#             from speechbrain.inference.speaker import EncoderClassifier
#
#             # 使用缓存中的模型
#             cache_path = Path.home() / '.cache' / 'huggingface' / 'hub' / 'models--speechbrain--spkrec-ecapa-voxceleb' / 'snapshots'
#             snapshots = list(cache_path.iterdir())
#
#             if snapshots:
#                 model_dir = str(snapshots[0])
#                 print(f'Using cached model: {model_dir}')
#
#                 self.model = EncoderClassifier.from_hparams(
#                     source=model_dir,
#                     savedir=model_dir,
#                     run_opts={'device': str(self.device)}
#                 )
#             else:
#                 raise FileNotFoundError('Model not found in cache.')
#
#         self.model.eval()
#         print(f'Model loaded on {self.device}')
#
#     def extract_embedding(self, audio_path=None, audio_tensor=None, sr=16000):
#         if audio_path is not None:
#             audio, file_sr = torchaudio.load(audio_path)
#             if file_sr != 16000:
#                 resampler = torchaudio.transforms.Resample(file_sr, 16000)
#                 audio = resampler(audio)
#         elif audio_tensor is not None:
#             audio = audio_tensor
#             if sr != 16000:
#                 resampler = torchaudio.transforms.Resample(sr, 16000)
#                 audio = resampler(audio)
#         else:
#             raise ValueError('Either audio_path or audio_tensor must be provided')
#
#         if audio.shape[0] > 1:
#             audio = audio.mean(dim=0, keepdim=True)
#
#         with torch.no_grad():
#             audio = audio.to(self.device)
#             embeddings = self.model.encode_batch(audio)
#
#         return embeddings.squeeze().cpu()
#
# if __name__ == '__main__':
#     print('Testing speaker embedding extractor...')
#     try:
#         extractor = SpeakerEmbeddingExtractor('ecapa')
#         dummy_audio = torch.randn(1, 16000 * 3)
#         embedding = extractor.extract_embedding(audio_tensor=dummy_audio)
#         print(f'Embedding shape: {embedding.shape}')
#         print('Speaker embedding extractor working!')
#     except Exception as e:
#         print(f'Error: {e}')
#         import traceback
#         traceback.print_exc()

'''
说话人嵌入提取器 - 避免符号链接问题
'''
import torch
import torchaudio
from pathlib import Path


class SpeakerEmbeddingExtractor:
    def __init__(self, model_name='ecapa'):
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._load_model()

    def _load_model(self):
        '''加载预训练模型 - 直接从缓存使用'''
        print(f'Loading {self.model_name} model...')

        if self.model_name == 'ecapa':
            from speechbrain.inference.speaker import EncoderClassifier

            # 使用缓存中的模型
            cache_path = Path.home() / '.cache' / 'huggingface' / 'hub' / 'models--speechbrain--spkrec-ecapa-voxceleb' / 'snapshots'
            snapshots = list(cache_path.iterdir())

            if snapshots:
                model_dir = str(snapshots[0])
                print(f'Using cached model: {model_dir}')

                self.model = EncoderClassifier.from_hparams(
                    source=model_dir,
                    savedir=model_dir,
                    run_opts={'device': str(self.device)}
                )
            else:
                raise FileNotFoundError('Model not found in cache.')

        self.model.eval()
        print(f'Model loaded on {self.device}')
        print(f'Expected embedding dim: 192')

    def extract_embedding(self, audio_path=None, audio_tensor=None, sr=16000):
        """
        提取说话人嵌入

        返回: torch.Tensor, shape [embedding_dim] 或 [batch, embedding_dim]
        """
        if audio_path is not None:
            audio, file_sr = torchaudio.load(audio_path)
            if file_sr != 16000:
                resampler = torchaudio.transforms.Resample(file_sr, 16000)
                audio = resampler(audio)
        elif audio_tensor is not None:
            audio = audio_tensor
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                audio = resampler(audio)
        else:
            raise ValueError('Either audio_path or audio_tensor must be provided')

        # 确保是单声道
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        with torch.no_grad():
            audio = audio.to(self.device)
            embeddings = self.model.encode_batch(audio)

            # ✅ 修复：正确处理维度
            # encode_batch返回 [batch, 1, dim] 或 [batch, dim]
            if embeddings.dim() == 3:
                embeddings = embeddings.squeeze(1)  # [batch, 1, 192] -> [batch, 192]

        return embeddings.cpu()


if __name__ == '__main__':
    print('Testing speaker embedding extractor...')
    try:
        extractor = SpeakerEmbeddingExtractor('ecapa')
        dummy_audio = torch.randn(1, 16000 * 3)
        embedding = extractor.extract_embedding(audio_tensor=dummy_audio)

        print(f'Embedding shape: {embedding.shape}')
        print(f'Expected shape: [1, 192] or [192]')
        print(f'Embedding norm: {torch.norm(embedding).item():.4f}')
        print('✓ Speaker embedding extractor working!')
    except Exception as e:
        print(f'Error: {e}')
        import traceback

        traceback.print_exc()