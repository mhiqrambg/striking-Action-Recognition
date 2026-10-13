from __future__ import annotations

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from src.data.features import compute_geometric_features

NUM_KEYPOINTS = 11          # COCO idx 0-10 (upper-body)
FEATURE_DIM = NUM_KEYPOINTS * 3 + 5   # 33 + 5 = 38
L_SHOULDER = 5
R_SHOULDER = 6

# Konvensi sentinel: vektor fitur yang SEMUA elemennya -1.0 berarti YOLO tidak
# mendeteksi orang pada frame tersebut. Frame seperti ini DIBUANG di ekstraksi
# (offline) agar konsisten dengan RealtimePredictor.push_frame yang juga
# membuang frame tanpa deteksi.
NO_DETECTION_SENTINEL = -1.0


def is_no_detection(feat: np.ndarray) -> bool:
    """True jika vektor fitur adalah sentinel 'tidak ada deteksi' (semua -1.0)."""
    return bool(np.all(np.asarray(feat) == NO_DETECTION_SENTINEL))


def load_yolo_pose_model(model_path: str) -> YOLO:
    model = YOLO(model_path)
    if torch.cuda.is_available():
        model = model.to("cuda")
    return model


def normalize_and_featurize(keypoints: np.ndarray) -> np.ndarray:
    """
    keypoints : (17, 3) koordinat COCO mentah (x, y, conf)
    return    : (38,) vektor = 11 keypoint ternormalisasi origin bahu (x,y,conf)
                + 5 fitur geometris
    """
    k = np.asarray(keypoints, dtype=np.float32)
    k = k[:NUM_KEYPOINTS]                       # ambil idx 0-10

    origin = (k[L_SHOULDER, :2] + k[R_SHOULDER, :2]) * 0.5
    norm = k.copy()
    norm[:, :2] = norm[:, :2] - origin          # normalisasi origin midpoint bahu

    geo = compute_geometric_features(norm)      # (5,)
    return np.concatenate([norm.reshape(-1), geo]).astype(np.float32)


def pose_from_prediction(prediction) -> np.ndarray:
    if len(prediction[0].keypoints.data) == 0:
        return np.full((FEATURE_DIM,), NO_DETECTION_SENTINEL, dtype=np.float32)
    kpts = prediction[0].keypoints.data[0].cpu().numpy()   # (17, 3) x,y dalam pixel
    xy = prediction[0].keypoints.xyn[0].cpu().numpy()      # (17, 2) ternormalisasi 0-1
    conf = kpts[:, 2:3]
    merged = np.hstack([xy, conf]).astype(np.float32)      # (17, 3)
    return normalize_and_featurize(merged)


def video_to_pose_array(video_path: str, yolo_model_path: str, conf: float = 0.7) -> np.ndarray:
    model = load_yolo_pose_model(yolo_model_path)
    capture = cv2.VideoCapture(video_path)
    frames = []

    while capture.isOpened():
        ok, frame = capture.read()
        if not ok:
            break
        prediction = model.predict(frame, verbose=False, conf=conf)
        feat = pose_from_prediction(prediction)
        # Buang frame tanpa deteksi (sentinel semua -1.0) agar distribusi
        # training sama dengan inference realtime (lihat NO_DETECTION_SENTINEL).
        if is_no_detection(feat):
            continue
        frames.append(feat)

    capture.release()
    if not frames:
        raise RuntimeError(f"No pose detections (or frames) in video: {video_path}")
    return np.array(frames, dtype=np.float32)
