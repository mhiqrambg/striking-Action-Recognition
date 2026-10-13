from __future__ import annotations

import numpy as np

from src.data.features import compute_geometric_features

# Keypoint untuk augmentasi flip (tukar kiri-kanan), indeks 0-10
FLIP_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)]

NUM_KEYPOINTS = 11
KEYPOINT_COLS = NUM_KEYPOINTS * 3      # 33
NUM_GEOMETRIC = 5                      # 5 fitur geometris
FEATURE_DIM = KEYPOINT_COLS + NUM_GEOMETRIC   # 38


def _random_scale(kpts: np.ndarray, low: float = 0.85, high: float = 1.15) -> np.ndarray:
    """[I6] Skala HANYA x/y. Confidence (kolom 2) tidak boleh diubah."""
    scale = np.random.uniform(low, high)
    kpts[:, :, :2] *= scale
    return kpts


def _gaussian_noise(kpts: np.ndarray, sigma: float = 0.01) -> np.ndarray:
    """[I6] Noise HANYA pada x/y. Confidence (kolom 2) tetap [0, 1]."""
    noise = np.random.normal(0, sigma, size=kpts[:, :, :2].shape).astype(kpts.dtype)
    kpts[:, :, :2] += noise
    return kpts


def _time_warp(array: np.ndarray, low: float = 0.8, high: float = 1.2) -> np.ndarray:
    T = array.shape[0]
    factor = np.random.uniform(low, high)
    new_T = max(2, int(T * factor))
    old_idx = np.linspace(0, T - 1, new_T)
    lo = np.floor(old_idx).astype(int)
    hi = np.clip(lo + 1, 0, T - 1)
    alpha = (old_idx - lo).reshape(-1, 1)
    warped = array[lo] * (1 - alpha) + array[hi] * alpha
    # Resample kembali ke panjang T: window harus tetap fixed-length (window, 38)
    # agar default DataLoader collate (torch.stack) bisa membentuk batch.
    if new_T != T:
        idx = np.linspace(0, new_T - 1, T)
        lo2 = np.floor(idx).astype(int)
        hi2 = np.clip(lo2 + 1, 0, new_T - 1)
        a2 = (idx - lo2).reshape(-1, 1)
        warped = warped[lo2] * (1 - a2) + warped[hi2] * a2
    return warped.astype(array.dtype)


def augment_window(array: np.ndarray) -> np.ndarray:
    """[I6] Augmentasi window fitur 38-dim secara semantik-konsisten.

    Struktur vektor fitur: 33 kolom keypoint (x, y, conf) x 11 + 5 fitur
    geometris. Scale/noise hanya diterapkan ke x/y (kolom 0,1 tiap keypoint);
    kolom confidence TIDAK diubah supaya tetap [0, 1]. Setelah perturbasi,
    5 fitur geometris DIHITUNG ULANG dari keypoint yang sudah diubah sehingga
    tetap konsisten (tidak stale). Time-warp diterapkan terakhir.

    array shape (window, 38) -> return shape (window, 38) dtype float32.
    """
    out = array.copy().astype(np.float32)
    # View keypoint (window, 11, 3) dari 33 kolom pertama
    kpts = out[:, :KEYPOINT_COLS].reshape(-1, NUM_KEYPOINTS, 3)
    kpts = _random_scale(kpts)
    kpts = _gaussian_noise(kpts)
    # Recompute fitur geometris dari keypoint yang sudah diperturbasi
    geo = compute_geometric_features(kpts)                       # (window, 5)
    out = np.concatenate(
        [kpts.reshape(kpts.shape[0], KEYPOINT_COLS), geo], axis=1
    )                                                            # (window, 38)
    out = _time_warp(out)
    return out.astype(np.float32)
