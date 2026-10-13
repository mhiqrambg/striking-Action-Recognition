from __future__ import annotations

import numpy as np

# Indeks keypoint COCO (0-10, upper-body)
NOSE = 0
L_EAR = 3
R_EAR = 4
L_SHOULDER = 5
R_SHOULDER = 6
L_ELBOW = 7
R_ELBOW = 8
L_WRIST = 9
R_WRIST = 10

_EPS = 1e-6


def compute_geometric_features(kpts: np.ndarray) -> np.ndarray:
    """
    Menghitung 5 fitur geometris gesture kelelahan dari keypoint ternormalisasi.

    kpts : np.ndarray shape (..., 11, 3) -> (x, y, confidence)
    return: np.ndarray shape (..., 5)
        [theta_head, d_forward, delta_y_shoulder, d_wrist_nose, theta_torso]

    Rumus persis mengikuti thesis (Tabel 2.4):
        theta_head      = arctan( |y_nose - y_midShoulder| / |x_nose - x_midShoulder| )
        d_forward       = x_nose - x_midShoulder
        delta_y_shoulder= |y_leftShoulder - y_rightShoulder|
        d_wrist_nose    = min( |y_wrist - y_nose| )
        theta_torso     = arctan( |y_ls - y_rs| / |x_ls - x_rs| )
    """
    kpts = np.asarray(kpts, dtype=np.float32)

    shoulder = kpts[..., [L_SHOULDER, R_SHOULDER], :2]        # (..., 2, 2)
    mid_shoulder = shoulder.mean(axis=-2)                     # (..., 2)
    nose = kpts[..., NOSE, :2]                                # (..., 2)
    ls = kpts[..., L_SHOULDER, :2]
    rs = kpts[..., R_SHOULDER, :2]
    lw = kpts[..., L_WRIST, :2]
    rw = kpts[..., R_WRIST, :2]

    dx_head = nose[..., 0] - mid_shoulder[..., 0]
    dy_head = nose[..., 1] - mid_shoulder[..., 1]
    theta_head = np.arctan(np.abs(dy_head) / (np.abs(dx_head) + _EPS))

    d_forward = dx_head

    delta_y_shoulder = np.abs(ls[..., 1] - rs[..., 1])

    dy_lw = np.abs(lw[..., 1] - nose[..., 1])
    dy_rw = np.abs(rw[..., 1] - nose[..., 1])
    d_wrist_nose = np.minimum(dy_lw, dy_rw)

    dx_sh = ls[..., 0] - rs[..., 0]
    dy_sh = ls[..., 1] - rs[..., 1]
    theta_torso = np.arctan(np.abs(dy_sh) / (np.abs(dx_sh) + _EPS))

    out = np.stack(
        [theta_head, d_forward, delta_y_shoulder, d_wrist_nose, theta_torso],
        axis=-1,
    )
    return out.astype(np.float32)
