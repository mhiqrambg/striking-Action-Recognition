import numpy as np
from src.realtime.run_realtime import draw_overlay, LatencyMeter


def test_draw_overlay_shape():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    out = draw_overlay(frame, "light", 0.87, warning=True, fps=20.0, latency_ms=45.0, cpu=42.0)
    assert out.shape == (480, 640, 3)


def test_draw_overlay_cpu_defaults():
    # [I5] parameter cpu opsional (default 0.0)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    out = draw_overlay(frame, "normal", 0.5, warning=False, fps=10.0, latency_ms=100.0)
    assert out.shape == (480, 640, 3)


def test_latency_meter():
    lm = LatencyMeter()
    lm.tick(0.050)
    lm.tick(0.050)
    assert abs(lm.fps - 20.0) < 1e-6
    assert abs(lm.avg_latency_ms - 50.0) < 1e-6
