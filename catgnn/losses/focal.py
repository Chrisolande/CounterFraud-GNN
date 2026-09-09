import torch
import torch.nn.functional as F
from torch import nn


class FocalLoss(nn.Module):
    """Binary Focal Loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        assert reduction in ("mean", "sum", "none")
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        assert logits.shape == targets.shape
        targets = targets.float()
        p = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = p * targets + (1.0 - p) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        loss = alpha_t * (1.0 - p_t).pow(self.gamma) * ce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
