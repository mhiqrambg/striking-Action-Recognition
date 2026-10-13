import numpy as np
from pathlib import Path
import torch.nn as nn

from src.config import load_config
from src.models.embedding import (
    LearnablePositionalEncoding,
    SinusoidalPositionalEncoding,
)
from src.training.checkpoint import build_model
from src.training.losses import FocalLoss
from src.training.trainer import train, _make_criterion


def _make_dataset(tmp_path):
    rng = np.random.default_rng(0)
    for cls in ["normal", "light", "heavy"]:
        for v in range(4):
            np.save(Path(tmp_path) / f"{cls}_video{v}.npy",
                    rng.standard_normal((90, 38)).astype(np.float32))


def test_train_smoke(tmp_path):
    _make_dataset(tmp_path)
    cfg = load_config("configs/default.yaml")
    cfg.paths.processed_dir = str(tmp_path)
    cfg.paths.results_dir = str(tmp_path / "results")
    cfg.paths.checkpoint_path = str(tmp_path / "ckpt.pt")
    cfg.training.epochs = 1
    cfg.data.num_folds = 2
    summary = train(cfg, scenario_name="B", use_learnable_pe=True, use_focal_loss=True)
    assert "accuracy_mean" in summary
    assert (Path(cfg.paths.results_dir) / "B.csv").exists()


def test_make_criterion_switches_on_focal_flag():
    cfg = load_config("configs/default.yaml")
    assert isinstance(_make_criterion(cfg, use_focal_loss=False), nn.CrossEntropyLoss)
    assert isinstance(_make_criterion(cfg, use_focal_loss=True), FocalLoss)


def test_build_model_switches_positional_encoding():
    cfg = load_config("configs/default.yaml")
    cfg.model.use_learnable_pe = False
    assert isinstance(build_model(cfg).position, SinusoidalPositionalEncoding)
    cfg.model.use_learnable_pe = True
    assert isinstance(build_model(cfg).position, LearnablePositionalEncoding)
