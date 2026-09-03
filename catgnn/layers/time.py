import torch
from torch import nn


class HarmonicTimeEncoder(nn.Module):
    """Time2Vec harmonic encoding: Phi(dt) = cos(w * dt + b)."""

    def __init__(self, time_dim: int):
        super().__init__()
        self.time_dim = time_dim
        init_w = 1.0 / (10.0 ** torch.linspace(0.0, 9.0, time_dim))
        self.w = nn.Parameter(init_w.float())
        self.b = nn.Parameter(torch.zeros(time_dim))

    def forward(self, delta_t: torch.Tensor) -> torch.Tensor:
        # delta_t: [N, M] -> [N, M, time_dim]
        phase = delta_t.unsqueeze(-1) * self.w + self.b
        return torch.cos(phase)
