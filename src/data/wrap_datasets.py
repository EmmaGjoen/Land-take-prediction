import random
import torch
from torch.utils.data import Dataset
import numpy as np


class FusedDataset(Dataset):
    """Fuse Sentinel imagery with AlphaEarth embeddings. 
    """
    DATASET_NAME = "fused_sentinel_alpha"

    def __init__(self, sentinel_ds, alpha_ds):
        self.sentinel_ds = sentinel_ds
        self.alpha_ds = alpha_ds
        
    def __len__(self):
        return len(self.sentinel_ds)

    def __getitem__(self, idx):
        # 1. Create a deterministic seed based on the index and the current epoch
        # Using just 'idx' ensures consistency within an epoch, but 
        # identical flips across different epochs. That's usually fine!
        sample_seed = idx 

        # 2. Transform Sentinel
        random.seed(sample_seed)
        torch.manual_seed(sample_seed)
        # This will trigger RandomFlipTS/RandomRotate90TS using sample_seed
        sen_img, mask = self.sentinel_ds[idx]

        # 3. Transform AlphaEarth
        random.seed(sample_seed)
        torch.manual_seed(sample_seed)
        # This will trigger the SAME flips/rotations because the seed reset
        alpha_img, _ = self.alpha_ds[idx]

        # 4. Fuse
        fused_img = torch.cat([sen_img, alpha_img], dim=1)
        return fused_img, mask


class FusedSentinelTesseraDataset(Dataset):
    """Fuse Sentinel imagery with GeoTessera embeddings.

    Identical fusion strategy to ``FusedDataset`` but paired with Tessera
    embeddings. The two streams are concatenated along the channel axis.
    """

    DATASET_NAME = "fused_sentinel_tessera"

    def __init__(self, sentinel_ds, tessera_ds):
        self.sentinel_ds = sentinel_ds
        self.tessera_ds = tessera_ds

    def __len__(self):
        return len(self.sentinel_ds)

    def __getitem__(self, idx):
        sample_seed = idx

        random.seed(sample_seed)
        torch.manual_seed(sample_seed)
        sen_img, mask = self.sentinel_ds[idx]

        random.seed(sample_seed)
        torch.manual_seed(sample_seed)
        tessera_img, _ = self.tessera_ds[idx]

        fused_img = torch.cat([sen_img, tessera_img], dim=1)
        return fused_img, mask