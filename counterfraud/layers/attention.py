import math

import torch
from torch import nn

from counterfraud.layers.time import HarmonicTimeEncoder


class TemporalAttentionLayer(nn.Module):
    """Multi-head relational temporal attention.

    e_ij^h = LeakyReLU(a_h^T [q_i^h ; k_j^h ; Phi(dt_ij)] * scale) + rel_bias(r)^h
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        heads: int = 4,
        time_dim: int = 16,
        num_relations: int = 4,
        dropout: float = 0.1,
        negative_slope: float = 0.2,
    ):
        super().__init__()
        assert hidden_dim % heads == 0, (
            f"hidden_dim ({hidden_dim}) must be divisible by heads ({heads})"
        )
        self.heads = heads
        self.d_h = hidden_dim // heads
        self.hidden_dim = hidden_dim
        self.time_dim = time_dim
        self.num_relations = num_relations

        self.time_encoder = HarmonicTimeEncoder(time_dim)
        self.W_q = nn.Linear(in_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(in_dim, hidden_dim, bias=False)
        self.W_v = nn.Linear(in_dim, hidden_dim, bias=False)

        self.rel_embed = nn.Embedding(num_relations + 1, heads)
        self.attn = nn.Parameter(torch.empty(heads, 2 * self.d_h + time_dim))
        nn.init.xavier_uniform_(self.attn)

        self.scale = 1.0 / math.sqrt(2 * self.d_h + time_dim)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_target: torch.Tensor,  # [N, in_dim]
        x_neighbor: torch.Tensor,  # [N, M, in_dim]
        t_gap: torch.Tensor,  # [N, M]
        valid_mask: torch.Tensor,  # [N, M] bool
        rel_ids: torch.Tensor | None = None,  # [N, M] long
    ) -> tuple[torch.Tensor, torch.Tensor]:
        N, M, in_dim = x_neighbor.shape
        assert x_target.shape == (N, in_dim)
        assert t_gap.shape == (N, M)
        assert valid_mask.shape == (N, M)
        H, d_h = self.heads, self.d_h

        t_emb = self.time_encoder(t_gap)  # [N, M, time_dim]

        Q = self.W_q(x_target).view(N, 1, H, d_h)
        K = self.W_k(x_neighbor).view(N, M, H, d_h)
        V = self.W_v(x_neighbor).view(N, M, H, d_h)

        Q_exp = Q.expand(-1, M, -1, -1)  # [N, M, H, d_h]
        t_emb_exp = t_emb.unsqueeze(2).expand(-1, -1, H, -1)  # [N, M, H, time_dim]
        concat = torch.cat([Q_exp, K, t_emb_exp], dim=-1)  # [N, M, H, 2*d_h+time_dim]

        # e_ij^h logits
        e = self.leaky_relu((concat * self.attn).sum(dim=-1) * self.scale)  # [N, M, H]

        if rel_ids is not None:
            e = e + self.rel_embed(rel_ids)  # [N, M, H]

        e = e.masked_fill(~valid_mask.unsqueeze(-1), float("-inf"))  # [N, M, H]
        alpha = torch.softmax(e, dim=1)  # [N, M, H]
        alpha = torch.nan_to_num(alpha, nan=0.0)
        alpha = self.dropout(alpha)

        return alpha, V

    def aggregate(self, alpha: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        # alpha: [N, M, H], V: [N, M, H, d_h] -> z: [N, hidden_dim]
        N, M, H = alpha.shape
        assert V.shape == (N, M, H, self.d_h)
        z = (alpha.unsqueeze(-1) * V).sum(dim=1)  # [N, H, d_h]
        return z.reshape(N, H * self.d_h)  # [N, hidden_dim]
