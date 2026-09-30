import torch

from cat_gnn_v2 import CausalInspector, CausalIntervener, weighted_bce_loss


def test_causal_inspector_verbatim_eq1_eq2_eq3():
    """
    Test CausalInspector for local relative neighborhood importance calculation
    and environment/causal mask partitioning.
    """
    inspector = CausalInspector(re=0.5)
    
    edge_index = torch.tensor([[1, 2, 3, 2], [0, 0, 0, 1]], dtype=torch.long)
    num_nodes = 4
    
    alpha = torch.tensor([
        [0.2, 0.4],
        [0.6, 0.8],
        [0.1, 0.1],
        [0.5, 0.5],
    ], dtype=torch.float32)
    
    importance = inspector(alpha, edge_index, num_nodes)
    
    assert importance.shape[0] == num_nodes
    
    env_mask, causal_mask = inspector.split_masks(importance, re=0.5)
    
    assert env_mask.shape[0] == num_nodes
    assert causal_mask.shape[0] == num_nodes
    assert torch.all(env_mask | causal_mask)
    assert not torch.any(env_mask & causal_mask)


def test_causal_intervener_verbatim_eq4():
    """
    Test CausalIntervener for linear Mixup weighting layer.
    """
    dim = 8
    intervener = CausalIntervener(dim=dim, rc=0.5, max_k=2)
    
    num_nodes = 6
    x = torch.randn(num_nodes, dim)
    
    env_mask = torch.tensor([True, False, True, False, False, False], dtype=torch.bool)
    causal_mask = ~env_mask
    importance = torch.tensor([0.1, 0.9, 0.2, 0.8, 0.7, 0.6], dtype=torch.float32)
    
    x_out = intervener(x, env_mask, causal_mask, importance)
    
    assert x_out.shape == x.shape
    assert torch.allclose(x_out[~env_mask], x[~env_mask])
    assert not torch.allclose(x_out[env_mask], x[env_mask])


def test_full_model_verbatim_forward_loss():
    """
    Test binary cross-entropy loss computation.
    """
    logits = torch.tensor([1.5, -2.0, 0.5, -0.5], dtype=torch.float32)
    labels = torch.tensor([1, 0, 1, 0], dtype=torch.long)
    mask = torch.tensor([True, True, True, True], dtype=torch.bool)
    
    loss = weighted_bce_loss(logits, labels, mask)
    
    assert loss.item() > 0
    assert torch.isfinite(loss)
