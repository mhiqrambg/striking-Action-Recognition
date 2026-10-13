import numpy as np

from src.data.preprocessing import (
    _gaussian_noise,
    _random_scale,
    augment_window,
)
from src.data.features import compute_geometric_features


def _make_window(T=30, seed=0):
    rng = np.random.default_rng(seed)
    window = rng.standard_normal((T, 38)).astype(np.float32)
    # Set kolom confidence (idx 2,5,...,32) ke nilai valid [0,1]
    conf_idx = np.arange(2, 33, 3)
    window[:, conf_idx] = rng.uniform(0.0, 1.0, size=(T, len(conf_idx))).astype(np.float32)
    return window


def test_augment_preserves_shape_and_dtype():
    window = _make_window()
    out = augment_window(window)
    assert out.shape == (30, 38)
    assert out.dtype == np.float32


def test_scale_and_noise_leave_confidence_untouched():
    # [I6] scale/noise HANYA x/y; confidence (kolom 2) tidak diubah
    window = _make_window()
    conf_idx = np.arange(2, 33, 3)
    scaled = _random_scale(window[:, :33].reshape(-1, 11, 3).copy())
    noised = _gaussian_noise(scaled)
    flat = noised.reshape(-1, 33)
    assert np.array_equal(flat[:, conf_idx], window[:, conf_idx])


def test_augment_confidence_stays_in_unit_range():
    # [I6] setelah augmentasi penuh, confidence tetap [0, 1]
    window = _make_window()
    conf_idx = np.arange(2, 33, 3)
    out = augment_window(window)
    assert np.all(out[:, conf_idx] >= 0.0)
    assert np.all(out[:, conf_idx] <= 1.0)


def test_augment_recomputes_geometric_features(monkeypatch):
    # [I6] 5 kolom geometris (33:38) dihitung ulang dari keypoint hasil
    # perturbasi. Time-warp di-patch ke identity agar konsistensi bisa diuji
    # langsung (time-warp memang interpolasi temporal pada array penuh).
    import src.data.preprocessing as prep
    monkeypatch.setattr(prep, "_time_warp", lambda a, **kw: a.astype(a.dtype))
    window = _make_window()
    out = augment_window(window)
    kpts = out[:, :33].reshape(-1, 11, 3)
    expected_geo = compute_geometric_features(kpts)
    assert np.allclose(out[:, 33:38], expected_geo, atol=1e-5)
