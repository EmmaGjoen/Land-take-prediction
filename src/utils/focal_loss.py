"""Focal loss for class-imbalanced binary segmentation."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal loss for class-imbalanced segmentation (Lin et al., 2017).

    Down-weights easy pixels by (1 - p_t)^gamma, focusing on hard examples.
    gamma=0 is equivalent to standard cross-entropy.
    """

    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # inputs: (B, C, H, W), targets: (B, H, W)
        ce = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()
