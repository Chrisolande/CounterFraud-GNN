import math
import torch
import torch.nn as nn


class HarmonicTimeEncoder(nn.Module):
    """Time2Vec-style harmonic encoding of a temporal gap, as used in TGAT.
    Phi(dt) = cos(w * dt + b), with w initialized to a log-spaced frequency bank so
    different channels resolve different time scales.
    """
    def __init__(self, time_dim: int):
        super().__init__()
        self.time_dim = time_dim
        init_w = 1.0 / (10.0 ** torch.linspace(0.0, 9.0, time_dim))
        self.w = nn.Parameter(init_w.float())
        self.b = nn.Parameter(torch.zeros(time_dim))

    def forward(self, delta_t: torch.Tensor) -> torch.Tensor:
        # delta_t: [N, M]
        phase = delta_t.unsqueeze(-1) * self.w + self.b  # [N, M, time_dim]
        return torch.cos(phase)  # [N, M, time_dim]


class TemporalAttentionLayer(nn.Module):
    """Multi-head relational temporal GAT attention.

    Score for head h, neighbor j of target i with relation r:
        e_ij^h = LeakyReLU( a_h . [ q_i^h ; k_j^h ; phi(dt_ij) ] * scale ) + rel_bias(r)^h
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        heads: int = 4,
        time_dim: int = 16,
        num_relations: int = 4,  # 0: Source, 1: Target, 2: Location, 3: Type, 4: Padding
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

        # Relation embedding to differentiate Source, Target, Location, Type edges
        self.rel_embed = nn.Embedding(num_relations + 1, heads)

        # Attention vector per head over the concatenated [q; k; t_emb]
        self.attn = nn.Parameter(torch.empty(heads, 2 * self.d_h + time_dim))
        nn.init.xavier_uniform_(self.attn)

        self.scale = 1.0 / math.sqrt(2 * self.d_h + time_dim)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_target: torch.Tensor, # [N, in_dim]
        x_neighbor: torch.Tensor, # [N, M, in_dim]
        t_gap: torch.Tensor, # [N, M]
        valid_mask: torch.Tensor, # [N, M] bool, True = real neighbor
        rel_ids: torch.Tensor | None = None, # [N, M] long, relation types
    ) -> tuple[torch.Tensor, torch.Tensor]:
        N, M, in_dim = x_neighbor.shape
        assert x_target.shape == (N, in_dim), (
            f"x_target {tuple(x_target.shape)} inconsistent with x_neighbor {tuple(x_neighbor.shape)}"
        )
        assert t_gap.shape == (N, M)
        assert valid_mask.shape == (N, M)
        H, d_h = self.heads, self.d_h

        t_emb = self.time_encoder(t_gap) # [N, M, time_dim]

        Q = self.W_q(x_target).view(N, 1, H, d_h)
        K = self.W_k(x_neighbor).view(N, M, H, d_h)
        V = self.W_v(x_neighbor).view(N, M, H, d_h)

        Q_exp = Q.expand(-1, M, -1, -1) # [N, M, H, d_h]
        t_emb_exp = t_emb.unsqueeze(2).expand(-1, -1, H, -1) # [N, M, H, time_dim]
        concat = torch.cat([Q_exp, K, t_emb_exp], dim=-1) # [N, M, H, 2*d_h+time_dim]

        # Scaled attention logits
        e = self.leaky_relu((concat * self.attn).sum(dim=-1) * self.scale) # [N, M, H]

        if rel_ids is not None:
            rel_bias = self.rel_embed(rel_ids) # [N, M, H]
            e = e + rel_bias

        e = e.masked_fill(~valid_mask.unsqueeze(-1), float("-inf")) # [N, M, H]
        alpha = torch.softmax(e, dim=1) # [N, M, H]
        alpha = torch.nan_to_num(alpha, nan=0.0)
        alpha = self.dropout(alpha)

        return alpha, V

    def aggregate(self, alpha: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        # alpha: [N, M, H], V: [N, M, H, d_h] -> z: [N, hidden_dim]
        N, M, H = alpha.shape
        assert V.shape == (N, M, H, self.d_h)
        z = (alpha.unsqueeze(-1) * V).sum(dim=1) # [N, H, d_h]
        z = z.reshape(N, H * self.d_h) # [N, hidden_dim]
        return z
