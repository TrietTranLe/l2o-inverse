"""
L1 Norm and L1 Distance Losses implementations.
------------------------------
Provides two mathematically consistent variants:

1. L1NormLoss:      |x|
2. L1DistanceLoss:  |x - y|
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class L1NormLoss(nn.Module):
    """
    Computes the L1 norm of a single tensor: |x|.
    Commonly used as a sparsity or regularization term.
    """

    def __init__(self, reduction: str = "mean"):
        """
        Args:
            reduction (str): 'none' | 'mean' | 'sum'
        """
        super().__init__()
        self.reduction = reduction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        abs_x = x.abs()
        if self.reduction == "none":
            return abs_x
        elif self.reduction == "sum":
            return abs_x.sum()
        else:
            return abs_x.mean()


class L1DistanceLoss(nn.Module):
    """
    Computes the L1 distance between two tensors: |x - y|.
    Equivalent to Mean Absolute Error when reduction='mean'.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(pred, target, reduction=self.reduction)
