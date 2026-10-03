import torch
from torch import nn

from counterfraud.causal import CausalInspector, CausalIntervener
from counterfraud.layers import TemporalAttentionLayer


class CaTGNN(nn.Module):
    def __init__(
        self,
        cont_dim: int,
        cat_dims: list[int],
        emb_dim: int = 16,
        hidden_dim: int = 64,
        heads: int = 4,
        time_dim: int = 16,
        num_relations: int = 4,
        env_ratio: float = 0.2,
        top_k: int = 5,
        beta_alpha: float = 2.0,
        beta_beta: float = 2.0,
        mlp_hidden: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embeddings = nn.ModuleList([
            nn.Embedding(num_classes, emb_dim) for num_classes in cat_dims
        ])

        in_dim = cont_dim + (len(cat_dims) * emb_dim)

        self.target_proj = nn.Linear(in_dim, hidden_dim)

        self.attn_layer = TemporalAttentionLayer(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            heads=heads,
            time_dim=time_dim,
            num_relations=num_relations,
            dropout=dropout,
        )
        self.inspector = CausalInspector(env_ratio=env_ratio)
        self.intervener = CausalIntervener(
            top_k=top_k,
            beta_alpha=beta_alpha,
            beta_beta=beta_beta,
        )

        # Classifier: [h_target ; z_neighbor] -> [2 * hidden_dim] -> 1
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def encode_inputs(self, x_cont: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        if len(self.embeddings) > 0 and x_cat.numel() > 0:
            embs = [emb(x_cat[..., i]) for i, emb in enumerate(self.embeddings)]
            return torch.cat([x_cont] + embs, dim=-1)
        return x_cont

    def forward(
        self,
        x_target_cont: torch.Tensor,
        x_target_cat: torch.Tensor,
        x_neighbor_cont: torch.Tensor,
        x_neighbor_cat: torch.Tensor,
        t_gap: torch.Tensor,
        valid_mask: torch.Tensor,
        rel_ids: torch.Tensor | None = None,
        intervene: bool = True,
    ) -> dict[str, torch.Tensor]:
        x_target = self.encode_inputs(x_target_cont, x_target_cat)
        x_neighbor = self.encode_inputs(x_neighbor_cont, x_neighbor_cat)

        h_target = torch.relu(self.target_proj(x_target))  # [N, hidden_dim]

        # Temporal attention aggregation: z_neigh = Sum_j alpha_ij * V_ij
        alpha, V = self.attn_layer(x_target, x_neighbor, t_gap, valid_mask, rel_ids=rel_ids)
        z_neigh = self.attn_layer.aggregate(alpha, V)  # [N, hidden_dim]

        # Fused representation: h_fused = [h_target ; z_neigh]
        h_fused = torch.cat([h_target, z_neigh], dim=-1)  # [N, 2 * hidden_dim]
        logits = self.classifier(h_fused).squeeze(-1)

        out = {"logits": logits, "z": h_fused, "alpha": alpha}

        if intervene:
            env_mask, causal_mask, importance = self.inspector(alpha, valid_mask)
            V_int, lam = self.intervener(V, importance, env_mask, causal_mask)
            z_int = self.attn_layer.aggregate(alpha, V_int)
            h_fused_int = torch.cat([h_target, z_int], dim=-1)
            logits_int = self.classifier(h_fused_int).squeeze(-1)

            out.update({
                "logits_intervened": logits_int,
                "z_intervened": h_fused_int,
                "env_mask": env_mask,
                "causal_mask": causal_mask,
                "lambda": lam,
            })
        return out