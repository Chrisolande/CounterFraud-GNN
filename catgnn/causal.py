import torch
import torch.nn as nn


class CausalInspector(nn.Module):
    """Parameter-free split of each neighborhood into causal vs environment
    nodes, using the temporal attention weights as an importance score.
    """

    def __init__(self, env_ratio: float = 0.2):
        super().__init__()
        assert 0.0 < env_ratio < 1.0, "env_ratio must be in (0, 1)"
        self.env_ratio = env_ratio

    def forward(
        self,
        alpha: torch.Tensor, # [N, M, H]
        valid_mask: torch.Tensor, # [N, M] bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        N, M, H = alpha.shape
        assert valid_mask.shape == (N, M)

        # Avg attention weights across attention heads
        importance = alpha.mean(dim=2)  # [N, M]

        valid_counts = valid_mask.sum(dim=1)  # [N]

        # Push padding slots to +inf so they sort to the very end
        # and are never selected as low-importance environment nodes
        masked_scores = importance.masked_fill(~valid_mask, float("inf"))
        order = torch.argsort(masked_scores, dim=1) # Ascending: lowest importance first
        rank = torch.argsort(order, dim=1) # rank[i, j] = slot j's position in ascending order


        # if valid_counts >= 1, causal_count must be at least 1.
        raw_env = torch.ceil(self.env_ratio * valid_counts.float()).long() # [N]
        max_allowed_env = torch.clamp(valid_counts - 1, min=0) # [N]
        env_count = torch.min(raw_env, max_allowed_env) # [N]

        env_mask = (rank < env_count.unsqueeze(1)) & valid_mask # [N, M]
        causal_mask = valid_mask & (~env_mask) # [N, M]

        return env_mask, causal_mask, importance


class CausalIntervener(nn.Module):
    """Backdoor adjustment: replace each environment node's value vector with
    a Beta-mixup of itself and a randomly sampled top-k causal node's value
    vector, producing a counterfactual V under the same attention weights.
    """

    def __init__(self, top_k: int = 5, beta_alpha: float = 2.0, beta_beta: float = 2.0):
        super().__init__()
        assert top_k >= 1, "top_k must be >= 1"
        self.top_k = top_k
        self.beta_alpha = beta_alpha
        self.beta_beta = beta_beta

    def forward(
        self,
        V: torch.Tensor, # [N, M, H, d_h]
        importance: torch.Tensor, # [N, M]
        env_mask: torch.Tensor, # [N, M] bool
        causal_mask: torch.Tensor, # [N, M] bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        N, M, H, d_h = V.shape
        assert importance.shape == (N, M)
        assert env_mask.shape == (N, M) and causal_mask.shape == (N, M)
        k = min(self.top_k, M)

        # Mask out non-causal nodes with -inf to retrieve the top-k causal nodes
        causal_scores = importance.masked_fill(~causal_mask, float("-inf"))
        topk_scores, topk_idx = torch.topk(causal_scores, k=k, dim=1)  # [N, k] each
        valid_topk = topk_scores > float("-inf") # [N, k]

        # Identify rows that have no valid causal neighbors (eg isolated nodes)
        no_causal_row = ~causal_mask.any(dim=1)  # [N]

        col_probs = valid_topk.float()

        col_probs = torch.where(
            no_causal_row.unsqueeze(1),
            torch.full_like(col_probs, 1.0 / k),
            col_probs,
        ) # Fallback to uniform distr
        col_probs = col_probs / col_probs.sum(dim=1, keepdim=True).clamp(min=1e-12)

        # Categorical draw over the k candidate causal donors for each of the M slots
        samples = torch.multinomial(col_probs, num_samples=M, replacement=True) # [N, M]
        selected_idx = torch.gather(topk_idx, 1, samples) # [N, M]

        # donor causal representations
        x_c = torch.gather(
            V, 1, selected_idx.view(N, M, 1, 1).expand(-1, -1, H, d_h)
        )  # [N, M, H, d_h]

        # Sample mixup weights from Beta distribution
        beta_dist = torch.distributions.Beta(self.beta_alpha, self.beta_beta)
        lam = beta_dist.sample((N, M)).to(device=V.device, dtype=V.dtype)  # [N, M]
        lam_b = lam.view(N, M, 1, 1) # [N, M, 1, 1]

        # Intervene only on environment slots: do(V_env) = lambda * V_env + (
        # 1-lambda) * x_c
        mixed = lam_b * V + (1.0 - lam_b) * x_c  # [N, M, H, d_h]
        env_mask_b = env_mask.view(N, M, 1, 1)
        V_intervened = torch.where(env_mask_b, mixed, V)

        # For rows with zero causal donors, preserve original V untouched
        V_intervened = torch.where(no_causal_row.view(N, 1, 1, 1), V, V_intervened)

        return V_intervened, lam