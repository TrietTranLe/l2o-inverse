import torch
import torch.nn as nn


class L2(nn.Module):
    """
    Computes the L2 norm (squared Euclidean distance)
    """

    def __init__(self, reduction: str = 'mean'):
        """
        Args:
            reduction (str): Specifies the reduction to apply to the output.
                                Options: 'none' | 'mean' | 'sum'.
        """
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the L2 norm loss between predictions and targets.

        Args:
            pred (torch.Tensor): Model predictions.
            target (torch.Tensor): Ground truth targets.

        Returns:
            torch.Tensor: The computed L2 norm loss.
        """

        loss = torch.sum((pred - target) ** 2, dim=list(range(1, pred.dim())))
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss