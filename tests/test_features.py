import numpy as np
from src.data.features import compute_geometric_features


def _make_kpts():
    # 11 keypoint, koordinat ternormalisasi origin bahu
    k = np.zeros((11, 3), dtype=np.float32)
    k[:, 2] = 1.0  # confidence
    k[0] = [0.0, -0.30, 1.0]     # nose
    k[5] = [-0.10, 0.0, 1.0]     # left shoulder
    k[6] = [0.10, 0.0, 1.0]      # right shoulder
    k[9] = [-0.15, -0.05, 1.0]   # left wrist
    k[10] = [0.15, -0.05, 1.0]   # right wrist
    return k


def test_output_shape():
    k = _make_kpts()
    f = compute_geometric_features(k)
    assert f.shape == (5,)


def test_batch_shape():
    k = np.stack([_make_kpts()] * 30, axis=0)  # (30, 11, 3)
    f = compute_geometric_features(k)
    assert f.shape == (30, 5)


def test_d_forward_equals_nose_x():
    k = _make_kpts()
    f = compute_geometric_features(k)
    # d_forward = x_nose - x_midShoulder ; midShoulder.x = 0
    assert abs(f[1] - 0.0) < 1e-6


def test_shoulder_asymmetry():
    k = _make_kpts()
    k[5] = [-0.10, -0.05, 1.0]  # left shoulder lebih tinggi
    k[6] = [0.10, 0.05, 1.0]
    f = compute_geometric_features(k)
    assert abs(f[2] - 0.10) < 1e-6
