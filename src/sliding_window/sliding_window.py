from __future__ import annotations

import numpy as np
import numpy.typing as npt


def create_sliding_windows(
    pose_array: npt.NDArray[np.float32],
    max_len: int = 30,
    stride: int = 15,
) -> npt.NDArray[np.float32]:
    """
    pose_array: (N_frames, 38)
    max_len: panjang window
    stride: langkah geser
    return: (N_windows, max_len, 38)
    """
    n_frames = pose_array.shape[0]
    windows = []
    for start in range(0, n_frames - max_len + 1, stride):
        windows.append(pose_array[start:start + max_len])
    if not windows:
        raise ValueError(f"Video terlalu pendek: {n_frames} frame < window {max_len}")
    return np.stack(windows).astype(np.float32)
