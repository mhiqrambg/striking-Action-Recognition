import numpy as np
import torch
from src.config import load_config
from src.training.checkpoint import build_model, save_checkpoint
from src.realtime.predictor import RealtimePredictor


def _make_ckpt(tmp_path):
    cfg = load_config("configs/default.yaml")
    model = build_model(cfg, num_classes=3)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    path = str(tmp_path / "ckpt.pt")
    save_checkpoint(path, model, opt, cfg,
                    {"normal": 0, "light": 1, "heavy": 2}, 0, {})
    return path


def test_predict_window(tmp_path):
    ckpt = _make_ckpt(tmp_path)
    rp = RealtimePredictor(checkpoint_path=ckpt, yolo_model_path=None)
    window = np.random.randn(30, 38).astype(np.float32)
    res = rp.predict_window(window)
    assert res["label"] in ["normal", "light", "heavy"]
    assert set(res["probabilities"].keys()) == {"normal", "light", "heavy"}


def test_push_frame_skips_no_detection(tmp_path, monkeypatch):
    ckpt = _make_ckpt(tmp_path)
    rp = RealtimePredictor(checkpoint_path=ckpt, yolo_model_path=None)

    class _FakeYolo:
        def predict(self, frame, verbose=False, conf=0.7):
            return ["fake_results"]

    rp.yolo = _FakeYolo()  # non-None stub; pose_from_prediction is monkeypatched below
    monkeypatch.setattr("src.realtime.predictor.pose_from_prediction",
                        lambda results: np.full((38,), -1.0, dtype=np.float32))
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for _ in range(rp.window_size + 5):
        assert rp.push_frame(frame) is None
    assert len(rp.buffer) == 0

