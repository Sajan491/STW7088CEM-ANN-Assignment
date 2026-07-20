"""Shared binary-classification metrics used by training and evaluation."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> dict:
    """Accuracy / precision / recall / F1 (positive class) at a threshold,
    plus threshold-free ROC-AUC."""
    y_pred = (y_prob >= threshold).astype(int)
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    # AUC is undefined when only one class is present (tiny smoke runs)
    if len(np.unique(y_true)) == 2:
        out["roc_auc"] = roc_auc_score(y_true, y_prob)
    return out


def confusion(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """2x2 confusion matrix [[TN, FP], [FN, TP]] at a threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    return confusion_matrix(y_true, y_pred, labels=[0, 1])
