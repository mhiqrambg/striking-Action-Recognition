import numpy as np
from src.evaluation.metrics import compute_metrics, confusion_matrix_np


def test_metrics_keys():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 2, 2])
    y_prob = np.eye(3)[y_pred] * 0.8 + 0.1
    m = compute_metrics(y_true, y_pred, y_prob, ["normal", "light", "heavy"])
    assert "accuracy" in m
    assert "f1_weighted" in m
    assert "pr_auc_normal" in m
    assert "pr_auc_light" in m
    assert "pr_auc_heavy" in m


def test_perfect_prediction():
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 1, 2])
    y_prob = np.eye(3)
    m = compute_metrics(y_true, y_pred, y_prob, ["normal", "light", "heavy"])
    assert abs(m["accuracy"] - 1.0) < 1e-9


def test_confusion_matrix_shape():
    cm = confusion_matrix_np(np.array([0, 1]), np.array([0, 1]), 3)
    assert cm.shape == (3, 3)
