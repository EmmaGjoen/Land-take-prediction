"""
Shared data transformation utilities for fair model comparison.

This module provides consistent normalization and augmentation transforms
for both U-Net and FCEF baselines, ensuring identical preprocessing.
"""

import torch
import torch.nn.functional as F
import random
from typing import Tuple, Sequence


def compute_normalization_stats(
    dataset,
    num_samples: int = 2000,
) -> Tuple[list[float], list[float]]:
    """
    Compute mathematically exact global per-channel mean and std.
    """

    num_samples = min(num_samples, len(dataset))
    indices = random.sample(range(len(dataset)), num_samples)

    first_sample = dataset[indices[0]]
    first_chip = first_sample[0]
    if first_chip.dim() == 4:
        C = first_chip.shape[1]
    elif first_chip.dim() == 3:
        C = first_chip.shape[0]
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got shape {first_chip.shape}")
    
    # Initialize running sums for the exact global calculation
    pixel_count = 0
    channel_sum = torch.zeros(C, dtype=torch.float64)
    channel_sum_sq = torch.zeros(C, dtype=torch.float64)
    
    for idx in indices:
        img_tensor = dataset[idx][0]
        
        if img_tensor.dim() == 4:  # (T, C, H, W)
            # Rearrange to (C, T*H*W) to sum across all pixels per channel
            reshaped = img_tensor.view(C, -1).double() 
            pixels_in_tensor = reshaped.shape[1]
        elif img_tensor.dim() == 3:  # (C, H, W)
            # Rearrange to (C, H*W)
            reshaped = img_tensor.view(C, -1).double()
            pixels_in_tensor = reshaped.shape[1]
        
        # Update running totals
        pixel_count += pixels_in_tensor
        channel_sum += reshaped.sum(dim=1)
        channel_sum_sq += (reshaped ** 2).sum(dim=1)

    # Calculate global mean and std using exact formulas
    global_mean = channel_sum / pixel_count
    
    # Variance = E[X^2] - (E[X])^2
    global_var = (channel_sum_sq / pixel_count) - (global_mean ** 2)
    
    # Clamp to prevent negative variance due to floating point inaccuracies
    global_var = torch.clamp(global_var, min=0.0)
    global_std = torch.sqrt(global_var)
    
    return global_mean.float().tolist(), global_std.float().tolist()


class Normalize:
    """
    Apply per-channel standardization using precomputed mean and std.
    
    This transform should be applied AFTER scaling (e.g., dividing by 10000).
    Use the same mean/std values computed from the training set for all splits.
    
    Args:
        mean: Sequence of per-channel mean values
        std: Sequence of per-channel std values
        
    Example:
        >>> # For flattened Sentinel data (B, C, H, W)
        >>> transform = Normalize(mean=train_mean, std=train_std)
        >>> img_normalized, mask = transform(img_scaled, mask)
    """
    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)
    
    def __call__(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Handle different input shapes
        if x.dim() == 4:  # (T, C, H, W) for time series
            T, C, H, W = x.shape
            mean = self.mean.view(1, C, 1, 1)
            std = self.std.view(1, C, 1, 1)
        elif x.dim() == 3:  # (C, H, W) for standard images
            C, H, W = x.shape
            mean = self.mean.view(C, 1, 1)
            std = self.std.view(C, 1, 1)
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got shape {x.shape}")
        
        x = (x - mean) / (std + 1e-6)
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
        return x, mask


class RandomCropTS:
    """returns cropped image + mask with default size H,W => 64,64"""

    def __init__(self, size=64):
        self.size = size

    def __call__(self, x, mask):
        # x: (T, C, H, W)
        T, C, H, W = x.shape
        s = self.size

        top = 0 if H <= s else random.randint(0, H - s)
        left = 0 if W <= s else random.randint(0, W - s)

        x = x[:, :, top:top+s, left:left+s]
        mask = mask[top:top+s, left:left+s]

        return x, mask 
    
class CenterCropTS:
    """Center crop (or pad) time series data to a fixed size.
    
    Handles variable-sized inputs by:
    - Padding with zeros if smaller than target size
    - Center cropping if larger than target size
    
    Works deterministically on all splits for reproducibility.
    """
    def __init__(self, size):
        self.size = size 
    def __call__(self, x, mask):
        T, C, H, W = x.shape
        s = self.size
        
        # Pad if smaller than target size
        if H < s or W < s:
            pad_h = max(0, s - H)
            pad_w = max(0, s - W)
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0)
            mask = F.pad(mask, (0, pad_w, 0, pad_h), mode="constant", value=0)
            T, C, H, W = x.shape
        
        # Center crop if larger than target size
        if H > s or W > s:
            top = (H - s) // 2
            left = (W - s) // 2
            x = x[:, :, top:top+s, left:left+s]
            mask = mask[top:top+s, left:left+s]
        
        return x, mask
    
class NormalizeBy:
    """Divide by a constant (Sentinel is TOAx10000)."""
    def __init__(self, denom=10000.0):
        self.denom = denom
    def __call__(self, x, mask):
        x = x / self.denom
        # replace NaNs and infinities with 0 (or some sensible default)
        x = torch.nan_to_num(x, nan=0.0, posinf=5.0, neginf=-5.0)
        return x, mask


class ComposeTS:
    def __init__(self, ops):
        self.ops = ops
    def __call__(self, x, mask):
        for op in self.ops:
            x, mask = op(x, mask)
        return x, mask


class RandomFlipTS:
    """Random horizontal and vertical flips for time series data (T, C, H, W).
    Applies the same flip to all timesteps and the mask.
    Works with 64x64 pre-cropped chips.
    """
    def __init__(self, p_horizontal=0.5, p_vertical=0.5):
        self.p_horizontal = p_horizontal
        self.p_vertical = p_vertical
    
    def __call__(self, x, mask):
        # x: (T, C, H, W), mask: (H, W)
        if random.random() < self.p_horizontal:
            x = x.flip(-1)  # flip width (last dimension)
            mask = mask.flip(-1)
        
        if random.random() < self.p_vertical:
            x = x.flip(-2)  # flip height (second to last dimension)
            mask = mask.flip(-2)
        
        return x, mask


class RandomRotate90TS:
    """Random 90-degree rotations for time series data (T, C, H, W).
    Applies the same rotation to all timesteps and the mask.
    Works with 64x64 pre-cropped chips on both CPU and GPU.
    """
    def __init__(self):
        pass
    
    def __call__(self, x, mask):
        # x: (T, C, H, W), mask: (H, W)
        # Sample k in {0, 1, 2, 3} for k * 90° rotation
        k = random.randint(0, 3)
        if k > 0:
            x = torch.rot90(x, k=k, dims=(-2, -1))
            mask = torch.rot90(mask, k=k, dims=(-2, -1))
        
        return x, mask
