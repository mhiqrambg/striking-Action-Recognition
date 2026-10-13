import numpy as np
import pytest

from src.sliding_window import classify_long_video as clv


def test_trailing_frames_forward_filled(monkeypatch, tmp_path):
    """[I7] Frame trailing tanpa coverage harus mewarisi prob window terakhir."""
    n_frames = 100
    max_len, stride = 30, 15
    pose = np.random.randn(n_frames, 38).astype(np.float32)
    class_names = ["normal", "light", "heavy"]

    dummy = tmp_path / "dummy.mp4"
    dummy.write_bytes(b"x")

    def fake_predict(batch, ckpt):
        return [{"probabilities": {"normal": 0.5, "light": 0.3, "heavy": 0.2}}
                for _ in range(len(batch))]

    monkeypatch.setattr(clv, "load_checkpoint",
                        lambda p: {"config": {"data": {"window_size": max_len}}})
    monkeypatch.setattr(clv, "video_to_pose_array", lambda *a, **k: pose)
    monkeypatch.setattr(clv, "predict_from_pose_batch", fake_predict)

    results = clv.process_long_video(str(dummy), "ckpt.pt", "yolo.pt", stride=stride)
    assert len(results) == n_frames
    # Semua probabilitas terisi & ternormalisasi (trailing ikut terisi)
    for r in results:
        probs = r["probabilities"]
        assert set(probs.keys()) == set(class_names)
        assert abs(sum(probs.values()) - 1.0) < 1e-4
    # trailing frames tidak all-zero -> label konsisten dengan probabilitas
    for r in results[-5:]:
        best = max(r["probabilities"], key=r["probabilities"].get)
        assert r["label"] == best
