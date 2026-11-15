import numpy as np


def change_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Binary change metrics over flattened masks (1 = changed)."""
    p, t = pred.astype(bool).ravel(), target.astype(bool).ravel()
    tp = float((p & t).sum())
    fp = float((p & ~t).sum())
    fn = float((~p & t).sum())
    eps = 1e-9
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall + eps),
        "iou": tp / (tp + fp + fn + eps),
    }
