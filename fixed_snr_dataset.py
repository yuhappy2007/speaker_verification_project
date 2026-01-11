"""
固定SNR采样的数据集包装器

用于训练时从固定的SNR值中随机选择
"""
import random
import torch
from torch.utils.data import Dataset


class FixedSNRDataset(Dataset):
    """
    包装 VoxCelebMusanDataset，使其从固定的 SNR 值中采样

    用法：
        base_dataset = VoxCelebMusanDataset(...)
        fixed_snr_dataset = FixedSNRDataset(base_dataset, snr_values=[-5, 0, 5, 10, 15])
    """

    def __init__(self, base_dataset, snr_values=[-5, 0, 5, 10, 15]):
        """
        Args:
            base_dataset: VoxCelebMusanDataset 实例
            snr_values: 固定的 SNR 值列表
        """
        self.base_dataset = base_dataset
        self.snr_values = snr_values

        print(f'FixedSNRDataset: Using SNR values: {snr_values}')

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        """
        获取样本，从固定的 SNR 值中随机选择
        """
        # 随机选择一个 SNR
        snr = random.choice(self.snr_values)

        # 临时设置 base_dataset 的 SNR 范围为这个固定值
        # 保存原始值
        original_snr_range = self.base_dataset.snr_range

        # 设置为固定值（range 设为相同值）
        self.base_dataset.snr_range = (snr, snr)

        # 获取样本
        sample = self.base_dataset[idx]

        # 恢复原始值
        self.base_dataset.snr_range = original_snr_range

        # 更新 SNR 信息（确保记录正确）
        sample['snr'] = snr

        return sample


def collate_fn_fixed_length(batch, target_length=48000):
    """
    自定义 collate function，确保所有音频长度一致

    Args:
        batch: 样本列表
        target_length: 目标长度（默认3秒 @ 16kHz）

    Returns:
        批次字典
    """
    import torch.nn.functional as F

    # 提取所有音频
    anchor_noisy_list = []
    positive_noisy_list = []
    negative_noisy_list = []
    anchor_clean_list = []
    positive_clean_list = []
    negative_clean_list = []

    snrs = []
    noise_types = []
    speaker_ids = []

    for sample in batch:
        # 处理每个音频：裁剪或填充到目标长度
        def process_audio(audio, target_len):
            if audio.shape[-1] > target_len:
                # 裁剪
                start = random.randint(0, audio.shape[-1] - target_len)
                return audio[..., start:start + target_len]
            elif audio.shape[-1] < target_len:
                # 填充
                pad_len = target_len - audio.shape[-1]
                return F.pad(audio, (0, pad_len))
            else:
                return audio

        anchor_noisy_list.append(process_audio(sample['anchor_noisy'], target_length))
        positive_noisy_list.append(process_audio(sample['positive_noisy'], target_length))
        negative_noisy_list.append(process_audio(sample['negative_noisy'], target_length))

        if 'anchor_clean' in sample:
            anchor_clean_list.append(process_audio(sample['anchor_clean'], target_length))
            positive_clean_list.append(process_audio(sample['positive_clean'], target_length))
            negative_clean_list.append(process_audio(sample['negative_clean'], target_length))

        snrs.append(sample['snr'])
        noise_types.append(sample['noise_type'])
        speaker_ids.append(sample['speaker_id'])

    # 堆叠成批次
    result = {
        'anchor_noisy': torch.stack(anchor_noisy_list),
        'positive_noisy': torch.stack(positive_noisy_list),
        'negative_noisy': torch.stack(negative_noisy_list),
        'snr': torch.tensor(snrs),
        'noise_type': noise_types,
        'speaker_id': speaker_ids
    }

    if anchor_clean_list:
        result['anchor_clean'] = torch.stack(anchor_clean_list)
        result['positive_clean'] = torch.stack(positive_clean_list)
        result['negative_clean'] = torch.stack(negative_clean_list)

    return result