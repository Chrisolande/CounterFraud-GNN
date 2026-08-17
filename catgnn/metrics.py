import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """y_true: {0,1} labels. y_prob: predicted fraud probability in [0,1]."""
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob).astype(float).ravel()
    assert y_true.shape == y_prob.shape

    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "auroc": float("nan"),
        "auprc": float("nan"),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "positive_rate": float(y_true.mean()),
        "n": int(y_true.shape[0]),
    }

    if len(np.unique(y_true)) == 2:
        metrics["auroc"] = float(roc_auc_score(y_true, y_prob))
        metrics["auprc"] = float(average_precision_score(y_true, y_prob))

    return metrics
