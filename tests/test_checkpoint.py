import torch
from src.config import load_config
from src.training.checkpoint import build_model, save_checkpoint, build_model_from_checkpoint


def test_build_model(tmp_path):
    cfg = load_config("configs/default.yaml")
    model = build_model(cfg, num_classes=3)
    x = torch.randn(2, 30, 38)
    assert model(x).shape == (2, 3)


def test_save_load(tmp_path):
    cfg = load_config("configs/default.yaml")
    model = build_model(cfg, num_classes=3)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    path = str(tmp_path / "ckpt.pt")
    save_checkpoint(path, model, opt, cfg, {"normal": 0, "light": 1, "heavy": 2}, 0, {"val_f1": 0.5})
    m2, ck = build_model_from_checkpoint(path, torch.device("cpu"))
    assert ck["label_map"]["light"] == 1
