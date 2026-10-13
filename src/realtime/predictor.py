from __future__ import annotations

from collections import deque
from typing import Dict, Optional

import numpy as np
import torch

from src.data.video_pose import load_yolo_pose_model, pose_from_prediction
from src.training.checkpoint import build_model_from_checkpoint, index_to_class_from_label_map


class RealtimePredictor:
    def __init__(self, checkpoint_path: str, yolo_model_path: Optional[str] = None,
                 window_size: int = 30, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.checkpoint = build_model_from_checkpoint(checkpoint_path, self.device)
        self.model.eval()
        self.window_size = window_size
        self.buffer: deque = deque(maxlen=window_size)
        self.labels = index_to_class_from_label_map(self.checkpoint["label_map"])
        self.yolo = load_yolo_pose_model(yolo_model_path) if yolo_model_path else None
        # keypoint deteksi terakhir (untuk digambar di overlay realtime)
        self.last_xy: Optional[np.ndarray] = None    # (17, 2) ternormalisasi 0-1
        self.last_conf: Optional[np.ndarray] = None  # (17,)

    @torch.no_grad()
    def predict_window(self, window: np.ndarray) -> Dict[str, object]:
        x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(self.device)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        idx = int(np.argmax(probs))
        return {
            "label": self.labels[idx],
            "index": idx,
            "probabilities": {self.labels[i]: float(probs[i]) for i in range(len(self.labels))},
        }

    def push_frame(self, frame: np.ndarray) -> Optional[Dict[str, object]]:
        """frame: BGR image dari OpenCV."""
        if self.yolo is None:
            raise ValueError("yolo_model_path tidak di-set; tidak bisa memproses frame.")
        results = self.yolo.predict(frame, verbose=False, conf=0.7)
        feat = pose_from_prediction(results)         # (38,)
        if np.all(feat == -1.0):
            self.last_xy = None                      # no detection
            self.last_conf = None
            return None                              # no detection: hold buffer, no prediction
        # simpan keypoint mentah untuk visualisasi
        self.last_xy = results[0].keypoints.xyn[0].cpu().numpy()
        self.last_conf = results[0].keypoints.data[0].cpu().numpy()[:, 2]
        self.buffer.append(feat)
        if len(self.buffer) < self.window_size:
            return None
        window = np.stack(self.buffer, axis=0)       # (window_size, 38)
        return self.predict_window(window)
