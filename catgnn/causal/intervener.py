import torch
from torch import nn


class CausalIntervener(nn.Module):
    """Counterfactual intervention: do(V_env) = lambda * V_env + (1 - lambda) * V_causal."""

    def __init__(self, top_k: int = 5, beta_alpha: float = 2.0, beta_beta: float = 2.0):
        super().__init__()
        assert top_k >= 1, "top_k must be >= 1"
        self.top_k = top_k
        self.beta_alpha = beta_alpha
        self.beta_beta = beta_beta

    def forward(
        self,
        V: torch.Tensor,  # [N, M, H, d_h]
        importance: torch.Tensor,  # [N, M]
        env_mask: torch.Tensor,  # [N, M] bool
        causal_mask: torch.Tensor,  # [N, M] bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        N, M, H, d_h = V.shape
        assert importance.shape == (N, M)
        assert env_mask.shape == (N, M) and causal_mask.shape == (N, M)
        k = min(self.top_k, M)

        # Top-k causal donors
        causal_scores = importance.masked_fill(~causal_mask, float("-inf"))
        topk_scores, topk_idx = torch.topk(causal_scores, k=k, dim=1)  # [N, k]
        valid_topk = topk_scores > float("-inf")  # [N, k]

        no_causal_row = ~causal_mask.any(dim=1)  # [N]
        col_probs = valid_topk.float()
        col_probs = torch.where(
            no_causal_row.unsqueeze(1),
            torch.full_like(col_probs, 1.0 / k),
            col_probs,
        )
        col_probs = col_probs / col_probs.sum(dim=1, keepdim=True).clamp(min=1e-12)

        # Sample causal donor per neighbor slot
        samples = torch.multinomial(col_probs, num_samples=M, replacement=True)  # [N, M]
        selected_idx = torch.gather(topk_idx, 1, samples)  # [N, M]
        x_c = torch.gather(
            V, 1, selected_idx.view(N, M, 1, 1).expand(-1, -1, H, d_h)
        )  # [N, M, H, d_h]

        # lambda ~ Beta(alpha, beta)
        beta_dist = torch.distributions.Beta(self.beta_alpha, self.beta_beta)
        lam = beta_dist.sample((N, M)).to(device=V.device, dtype=V.dtype)  # [N, M]
        lam_b = lam.view(N, M, 1, 1)

        # do(V_env) = lambda * V_env + (1 - lambda) * x_c
        mixed = lam_b * V + (1.0 - lam_b) * x_c  # [N, M, H, d_h]
        V_intervened = torch.where(env_mask.view(N, M, 1, 1), mixed, V)
        V_intervened = torch.where(no_causal_row.view(N, 1, 1, 1), V, V_intervened)

        return V_intervened, lam
