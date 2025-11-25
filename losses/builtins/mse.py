"""
Mean Squared Error (MSE) loss implementation.
--------------------------------------------
This module wraps PyTorch's built-in 'nn.MSELoss' class
"""

import torch
import torch.nn as nn


class MSE(nn.Module):
    """
    Wrapper around torch.nn.MSELoss with automatic registry.

    Attributes:
        loss_fn (nn.MSELoss): The underlying PyTorch loss function.
    """

    def __init__(self, reduction: str = 'mean'):
        """
        Initialize the MSE loss module.

        Args:
            reduction (str): Specifies the reduction to apply to the output.
                             Options: 'none' | 'mean' | 'sum'.
        """
        super().__init__()
        self.loss_fn = nn.MSELoss(reduction=reduction)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the mean squared error between predictions and targets.

        Args:
            pred (torch.Tensor): Model predictions.
            target (torch.Tensor): Ground truth targets.

        Returns:
            torch.Tensor: The computed MSE loss.
        """
        return self.loss_fn(pred, target)

