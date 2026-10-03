import torch
from torch import nn


class CausalInspector(nn.Module):
    """Split neighborhood into causal vs environment nodes via attention weights."""

    def __init__(self, env_ratio: float = 0.2):
        super().__init__()
        assert 0.0 < env_ratio < 1.0, "env_ratio must be in (0, 1)"
        self.env_ratio = env_ratio

    def forward(
        self,
        alpha: torch.Tensor,  # [N, M, H]
        valid_mask: torch.Tensor,  # [N, M] bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        N, M, _H = alpha.shape
        assert valid_mask.shape == (N, M)

        # Importance score: mean attention across heads
        importance = alpha.mean(dim=2)  # [N, M]
        valid_counts = valid_mask.sum(dim=1)  # [N]

        # Sort masked padding slots to end
        masked_scores = importance.masked_fill(~valid_mask, float("inf"))
        order = torch.argsort(masked_scores, dim=1)
        rank = torch.argsort(order, dim=1)

        # env_count = min(ceil(env_ratio * |N_i|), |N_i| - 1)
        raw_env = torch.ceil(self.env_ratio * valid_counts.float()).long()  # [N]
        max_allowed_env = torch.clamp(valid_counts - 1, min=0)  # [N]
        env_count = torch.min(raw_env, max_allowed_env)  # [N]

        env_mask = (rank < env_count.unsqueeze(1)) & valid_mask  # [N, M]
        causal_mask = valid_mask & (~env_mask)  # [N, M]

        return env_mask, causal_mask, importance
