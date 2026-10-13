import numpy as np
import pytest
import torch
from src.config import load_config
from src.training.checkpoint import build_model, save_checkpoint
from src.inference.predictor import predict_from_pose


def _make_ckpt(tmp_path):
    cfg = load_config("configs/default.yaml")
    model = build_model(cfg, num_classes=3)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    path = str(tmp_path / "ckpt.pt")
    save_checkpoint(path, model, opt, cfg,
                    {"normal": 0, "light": 1, "heavy": 2}, 0, {})
    return path


def test_predict_from_pose_rejects_wrong_feature_dim(tmp_path):
    ckpt = _make_ckpt(tmp_path)
    stale = np.random.randn(30, 51).astype(np.float32)
    with pytest.raises(ValueError, match="Expected 38 features per frame, got 51"):
        predict_from_pose(stale, ckpt)


def test_predict_from_pose_valid_window(tmp_path):
    ckpt = _make_ckpt(tmp_path)
    window = np.random.randn(30, 38).astype(np.float32)
    res = predict_from_pose(window, ckpt)
    assert isinstance(res, dict)
    assert res["label"] in ["normal", "light", "heavy"]
