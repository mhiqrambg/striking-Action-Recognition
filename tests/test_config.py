from pathlib import Path

from src.config import load_config, config_from_dict

CONFIG = Path(__file__).parents[1] / "configs" / "default.yaml"


def test_defaults():
    cfg = load_config(str(CONFIG))
    assert cfg.model.input_dim == 38
    assert cfg.model.dim_model == 128
    assert cfg.model.ffn_dim == 256
    assert cfg.model.num_classes == 3
    assert cfg.data.window_size == 30
    assert cfg.data.stride == 15
    assert cfg.data.num_folds == 5
    assert cfg.training.focal_gamma == 2.0
    assert list(cfg.training.focal_alpha) == [0.25, 0.40, 0.35]
    assert cfg.class_names == ["normal", "light", "heavy"]


def test_roundtrip():
    cfg = load_config(str(CONFIG))
    cfg2 = config_from_dict(cfg.to_dict())
    assert cfg2.model.input_dim == cfg.model.input_dim
    assert cfg2.data.window_size == cfg.data.window_size
