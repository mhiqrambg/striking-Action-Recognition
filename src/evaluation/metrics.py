from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    labels = list(range(num_classes))
    return confusion_matrix(y_true, y_pred, labels=labels)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
) -> Dict[str, float]:
    """
    y_prob: (N, C) probabilitas softmax.
    Mengembalikan accuracy, precision/recall/f1 (weighted), dan pr_auc per kelas (one-vs-rest).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)
    n_classes = len(class_names)

    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

    for ci in range(n_classes):
        binary_true = (y_true == ci).astype(int)
        if binary_true.sum() == 0:
            metrics[f"pr_auc_{class_names[ci]}"] = float("nan")
        else:
            metrics[f"pr_auc_{class_names[ci]}"] = float(
                average_precision_score(binary_true, y_prob[:, ci])
            )
    return metrics
