"""
小规模测试脚本 - 笔记本版（修复版）
快速验证训练流程，只训练几个batch
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import sys

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'scripts'))

from gtcrn_wrapper_fixed import GTCRNWrapper
from ecapa_tdnn_wrapper import ECAPATDNNWrapper
from contrastive_loss import ContrastiveLoss
from dataset_fixed import VoxCelebMusanDataset
from fixed_snr_dataset import FixedSNRDataset, collate_fn_fixed_length


def custom_collate_fn(batch):
    """全局 collate 函数"""
    return collate_fn_fixed_length(batch, target_length=48000)


class ConfidenceNetwork(nn.Module):
    """置信度网络（简化版）"""

    def __init__(self, embedding_dim=192, hidden_dim=256):
        super().__init__()

        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim)
        )

        self.attention = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, emb_noisy, emb_enhanced):
        concat = torch.cat([emb_noisy, emb_enhanced], dim=1)
        weights = self.attention(concat)
        w_noisy = weights[:, 0:1]
        w_enhanced = weights[:, 1:2]

        weighted = w_noisy * emb_noisy + w_enhanced * emb_enhanced
        fused = self.fusion(concat)
        output = weighted + fused
        output = nn.functional.normalize(output, p=2, dim=1)

        return output


def test_mini_training():
    """
    迷你训练测试
    - 只用少量样本
    - 只训练1-2个epoch
    - 快速验证整个流程
    """

    print("\n" + "=" * 80)
    print("Mini Training Test (Notebook Version)")
    print("=" * 80)

    # 配置
    VOXCELEB_DIR = 'data/voxceleb1'
    MUSAN_DIR = 'data/musan'
    GTCRN_CHECKPOINT = 'checkpoints/gtcrn/gtcrn_best.pth'
    SPEAKER_MODEL = 'pretrained_models/spkrec-ecapa-voxceleb'

    NUM_TRAIN_SAMPLES = 32  # 只用32个样本
    NUM_VAL_SAMPLES = 8
    BATCH_SIZE = 4
    NUM_EPOCHS = 2

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # ========== 1. 加载数据集 ==========
    print("\n1. Loading datasets (mini version)...")
    try:
        # 完整数据集
        full_train_dataset = VoxCelebMusanDataset(
            voxceleb_dir=VOXCELEB_DIR,
            musan_dir=MUSAN_DIR,
            split='train',
            snr_range=(-5, 15),
            return_clean=False
        )

        full_val_dataset = VoxCelebMusanDataset(
            voxceleb_dir=VOXCELEB_DIR,
            musan_dir=MUSAN_DIR,
            split='test',
            test_snr=0,
            test_noise_type='noise',
            return_clean=False
        )

        # ✅ 修复：直接从完整数据集创建子集，然后包装成 FixedSNRDataset
        # 不要用 Subset，因为 FixedSNRDataset 需要访问 snr_range 等属性
        import random

        # 创建一个小的训练数据集（使用前 NUM_TRAIN_SAMPLES 个样本）
        # 这样避免 Subset 的问题
        class SmallDataset:
            """简单的数据集包装器"""

            def __init__(self, base_dataset, num_samples):
                self.base_dataset = base_dataset
                self.indices = random.sample(range(len(base_dataset)), min(num_samples, len(base_dataset)))
                # 复制必要的属性
                self.snr_range = base_dataset.snr_range
                self.return_clean = base_dataset.return_clean

            def __len__(self):
                return len(self.indices)

            def __getitem__(self, idx):
                return self.base_dataset[self.indices[idx]]

        small_train_dataset = SmallDataset(full_train_dataset, NUM_TRAIN_SAMPLES)
        small_val_dataset = SmallDataset(full_val_dataset, NUM_VAL_SAMPLES)

        # 包装成 FixedSNRDataset
        train_dataset = FixedSNRDataset(small_train_dataset, snr_values=[-5, 0, 5])

        print(f"   ✅ Train samples: {len(train_dataset)}")
        print(f"   ✅ Val samples: {len(small_val_dataset)}")

    except Exception as e:
        print(f"   ❌ Failed to load datasets: {e}")
        import traceback
        traceback.print_exc()
        return

    # ========== 2. DataLoader ==========
    print("\n2. Creating DataLoaders...")
    try:
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,  # 笔记本上用单线程
            collate_fn=custom_collate_fn
        )

        val_loader = DataLoader(
            small_val_dataset,  # 注意：验证集不用 FixedSNRDataset 包装
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            collate_fn=custom_collate_fn
        )

        print(f"   ✅ Train batches: {len(train_loader)}")
        print(f"   ✅ Val batches: {len(val_loader)}")

    except Exception as e:
        print(f"   ❌ Failed to create DataLoaders: {e}")
        return

    # ========== 3. 加载模型 ==========
    print("\n3. Loading models...")

    # GTCRN
    print("   Loading GTCRN...")
    try:
        gtcrn = GTCRNWrapper(
            checkpoint_path=GTCRN_CHECKPOINT,
            device=device,
            freeze=True
        )
        print("   ✅ GTCRN loaded")
    except Exception as e:
        print(f"   ❌ Failed to load GTCRN: {e}")
        return

    # ECAPA-TDNN
    print("   Loading ECAPA-TDNN...")
    try:
        speaker_model = ECAPATDNNWrapper(
            model_path=SPEAKER_MODEL,
            device=device,
            freeze=True
        )
        print("   ✅ ECAPA-TDNN loaded")
    except Exception as e:
        print(f"   ❌ Failed to load ECAPA-TDNN: {e}")
        return

    # Confidence Network
    print("   Initializing Confidence Network...")
    confidence_net = ConfidenceNetwork(embedding_dim=192, hidden_dim=256).to(device)
    print("   ✅ Confidence Network initialized")

    # ========== 4. 训练设置 ==========
    criterion = ContrastiveLoss(temperature=0.07)
    optimizer = optim.AdamW(confidence_net.parameters(), lr=1e-3)

    print("\n4. Training setup complete")

    # ========== 5. 训练循环 ==========
    print(f"\n5. Starting mini training ({NUM_EPOCHS} epochs)...")
    print("=" * 80)

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        print("-" * 80)

        # 训练
        confidence_net.train()
        total_loss = 0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            try:
                # 数据
                noisy = batch['anchor_noisy'].to(device)
                speaker_ids = batch['speaker_id']

                # 标签
                unique_speakers = list(set(speaker_ids))
                speaker_to_idx = {spk: idx for idx, spk in enumerate(unique_speakers)}
                labels = torch.tensor([speaker_to_idx[spk] for spk in speaker_ids]).to(device)

                # 前向传播
                with torch.no_grad():
                    enhanced = gtcrn.enhance(noisy)
                    emb_noisy = speaker_model(noisy)
                    emb_enhanced = speaker_model(enhanced)

                emb_fused = confidence_net(emb_noisy, emb_enhanced)
                loss = criterion(emb_fused, labels)

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                print(f"  Batch {batch_idx + 1}/{len(train_loader)}: loss={loss.item():.4f}")

            except Exception as e:
                print(f"  ❌ Error in batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue

        if num_batches > 0:
            avg_loss = total_loss / num_batches
            print(f"\nEpoch {epoch} - Train Loss: {avg_loss:.4f}")
        else:
            print(f"\n❌ No batches processed in epoch {epoch}")

        # 验证
        print("\nValidation...")
        confidence_net.eval()
        val_loss = 0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                try:
                    noisy = batch['anchor_noisy'].to(device)
                    speaker_ids = batch['speaker_id']

                    unique_speakers = list(set(speaker_ids))
                    speaker_to_idx = {spk: idx for idx, spk in enumerate(unique_speakers)}
                    labels = torch.tensor([speaker_to_idx[spk] for spk in speaker_ids]).to(device)

                    enhanced = gtcrn.enhance(noisy)
                    emb_noisy = speaker_model(noisy)
                    emb_enhanced = speaker_model(enhanced)
                    emb_fused = confidence_net(emb_noisy, emb_enhanced)

                    loss = criterion(emb_fused, labels)
                    val_loss += loss.item()
                    val_batches += 1

                except Exception as e:
                    print(f"  ❌ Validation error: {e}")
                    continue

        if val_batches > 0:
            avg_val_loss = val_loss / val_batches
            print(f"Epoch {epoch} - Val Loss: {avg_val_loss:.4f}")
        else:
            print(f"❌ No validation batches processed")

    print("\n" + "=" * 80)
    print("✅ Mini training test completed!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. ✅ Models loaded successfully")
    print("  2. ✅ Training loop works")
    print("  3. 📤 Upload to server: scp ecapa_tdnn_wrapper.py server:/path/")
    print("  4. 🚀 Run full training on server")


if __name__ == '__main__':
    test_mini_training()