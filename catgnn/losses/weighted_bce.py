import torch
import torch.nn.functional as F
from torch import nn


class WeightedBCEWithLogitsLoss(nn.Module):
    """BCE with positive class weighting."""

    def __init__(self, pos_weight: float | None = None):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        assert logits.shape == targets.shape
        targets = targets.float()
        pw = None
        if self.pos_weight is not None:
            pw = torch.as_tensor(self.pos_weight, device=logits.device, dtype=logits.dtype)
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw)
