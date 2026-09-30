import numpy as np
import torch

from catgnn.causal import CausalInspector, CausalIntervener
from catgnn.metrics import compute_metrics
from catgnn.model import CaTGNN


def test_causal_inspector():
    inspector = CausalInspector(env_ratio=0.25)
    N, M, H = 4, 8, 2
    alpha = torch.rand(N, M, H)
    valid_mask = torch.ones(N, M, dtype=torch.bool)
    valid_mask[:, 6:] = False  # slots 6,7 are padding

    env_mask, causal_mask, importance = inspector(alpha, valid_mask)

    assert env_mask.shape == (N, M)
    assert causal_mask.shape == (N, M)
    assert importance.shape == (N, M)
    # Valid slots must be partitioned into either env or causal
    assert torch.all((env_mask | causal_mask) == valid_mask)
    # No overlap
    assert not torch.any(env_mask & causal_mask)


def test_causal_intervener():
    intervener = CausalIntervener(top_k=3)
    N, M, H, d_h = 4, 8, 2, 16
    V = torch.randn(N, M, H, d_h)
    importance = torch.rand(N, M)
    valid_mask = torch.ones(N, M, dtype=torch.bool)
    env_mask = torch.zeros(N, M, dtype=torch.bool)
    env_mask[:, :2] = True
    causal_mask = valid_mask & (~env_mask)

    V_intervened, lam = intervener(V, importance, env_mask, causal_mask)
    assert V_intervened.shape == V.shape
    assert lam.shape == (N, M)
    # Non-environment slots should remain untouched
    assert torch.allclose(V_intervened[:, 2:], V[:, 2:])


def test_compute_metrics():
    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3])
    m = compute_metrics(y_true, y_prob, threshold=0.5)

    assert "auroc" in m
    assert "auprc" in m
    assert "macro_f1" in m
    assert m["auroc"] > 0.9
    assert m["macro_f1"] == 1.0


def test_model_forward():
    model = CaTGNN(
        cont_dim=10,
        cat_dims=[5, 8],
        emb_dim=4,
        hidden_dim=16,
        heads=2,
        num_relations=3,
        env_ratio=0.2,
        top_k=2,
    )
    N, M = 4, 6
    x_target_cont = torch.randn(N, 10)
    x_target_cat = torch.tensor([[1, 2], [0, 3], [2, 1], [4, 0]], dtype=torch.long)
    x_neighbor_cont = torch.randn(N, M, 10)
    x_neighbor_cat = torch.zeros(N, M, 2, dtype=torch.long)
    t_gap = torch.rand(N, M)
    valid_mask = torch.ones(N, M, dtype=torch.bool)
    rel_ids = torch.zeros(N, M, dtype=torch.long)

    out = model(
        x_target_cont=x_target_cont,
        x_target_cat=x_target_cat,
        x_neighbor_cont=x_neighbor_cont,
        x_neighbor_cat=x_neighbor_cat,
        t_gap=t_gap,
        valid_mask=valid_mask,
        rel_ids=rel_ids,
        intervene=True,
    )

    assert "logits" in out
    assert "logits_intervened" in out
    assert out["logits"].shape == (N,)
