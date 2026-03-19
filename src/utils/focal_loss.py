import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal loss for class-imbalanced segmentation (Lin et al., 2017).

    Reduces the loss contribution of easy-to-classify pixels by a factor of
    (1 - p_t)^gamma, focusing training on hard examples. With gamma=0 this
    reduces to standard cross-entropy.

    Args:
        gamma: focusing parameter (default 2.0, as in the original paper)
        weight: per-class weight tensor, same as in nn.CrossEntropyLoss
    """

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # inputs: (B, C, H, W), targets: (B, H, W)
        ce = F.cross_entropy(inputs, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()
