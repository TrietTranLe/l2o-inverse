"""
Cosine Similarity Loss implementations.
---------------------------------------
Provides two variants of cosine similarity–based losses:
1. Flatten all dimensions except the batch before computing similarity.
2. Compute cosine similarity over feature dimensions, then take mean.
"""

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineSimilarityFlatLoss(nn.Module):
    """
    Cosine similarity loss where all dimensions except the batch
    are flattened before computing the cosine similarity.

    Example:
        pred:  [B, C, H, W] → flattened to [B, C*H*W]
        target:[B, C, H, W] → flattened to [B, C*H*W]

    The loss is defined as:  1 - mean(cosine_similarity(pred, target))
    """

    def __init__(self, eps: float = 1e-8):
        """
        Args:
            eps (float): Small constant to avoid division by zero.
        """
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity loss after flattening.

        Args:
            pred (torch.Tensor): Predicted tensor of shape [B, ...].
            target (torch.Tensor): Ground truth tensor of same shape.

        Returns:
            torch.Tensor: Scalar cosine similarity loss (1 - mean(similarity)).
        """
        pred = einops.rearrange(pred, 'b c h -> b (c h)')
        target = einops.rearrange(target, 'b c h -> b (c h)')
        # cosine_similarity returns [B] (one value per batch)
        sim = F.cosine_similarity(pred, target, dim=1, eps=self.eps)
        loss = 1.0 - sim.mean()
        return loss


class CosineSimilarityMeanLoss(nn.Module):
    """
    Cosine similarity loss where cosine is computed across all
    feature dimensions except the batch, then averaged.

    Example:
        For 4D input [B, C, H, W]:
        -> cosine similarity is computed along dim=1 (channels)
        -> then averaged over H and W and across the batch.

    The loss is defined as:  1 - mean(cosine_similarity(pred, target))
    """

    def __init__(self, dim: int = 1, eps: float = 1e-8):
        """
        Args:
            dim (int): Dimension along which to compute cosine similarity.
                       Typically the feature/channel dimension.
            eps (float): Small constant to avoid division by zero.
        """
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity loss with mean reduction.

        Args:
            pred (torch.Tensor): Predicted tensor of shape [B, ...].
            target (torch.Tensor): Ground truth tensor of same shape.

        Returns:
            torch.Tensor: Scalar cosine similarity loss (1 - mean(similarity)).
        """
        sim = F.cosine_similarity(pred, target, dim=self.dim, eps=self.eps)
        loss = 1.0 - sim.mean()
        return loss

