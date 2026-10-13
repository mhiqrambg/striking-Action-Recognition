from __future__ import annotations
import torch
import numpy as np
from pathlib import Path

from src.data.video_pose import video_to_pose_array
from src.inference.predictor import predict_from_pose_batch
from src.sliding_window.sliding_window import create_sliding_windows
from src.training.checkpoint import load_checkpoint

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}


def process_long_video(file_path, checkpoint_path, yolo_model_path, stride=15):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(checkpoint_path)
    max_len = int(checkpoint["config"]["data"]["window_size"])

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")
    if path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError(f"Unsupported video format: {path.suffix}")

    pose_array = video_to_pose_array(str(path), yolo_model_path=yolo_model_path)  # (T, 38)
    n_frames = pose_array.shape[0]

    windows = create_sliding_windows(pose_array, max_len, stride)  # (N, max_len, 38)
    tensor_batch = torch.tensor(windows, dtype=torch.float32).to(device)
    window_results = predict_from_pose_batch(tensor_batch, checkpoint_path)

    class_names = list(window_results[0]["probabilities"].keys())
    n_classes = len(class_names)
    prob_sum = np.zeros((n_frames, n_classes))
    cover_count = np.zeros(n_frames)

    for win_idx, result in enumerate(window_results):
        start = win_idx * stride
        end = start + max_len
        for ci, name in enumerate(class_names):
            prob_sum[start:end, ci] += result["probabilities"][name]
        cover_count[start:end] += 1

    covered = cover_count > 0                      # [I7] mask sebelum dinolkan
    cover_count[cover_count == 0] = 1
    prob_avg = prob_sum / cover_count[:, None]

    # [I7] Forward-fill frame trailing yang tidak tercakup window mana pun.
    # Window hanya menutupi hingga (N-1)*stride+max_len; sisa frame punya
    # cover_count == 0. Untuk frame itu, salin probabilitas dari frame terakhir
    # yang punya coverage, agar tidak diisi nol (yang membuat max() memilih
    # kelas pertama secara arbitrer).
    last_covered = -1
    for i in range(n_frames):
        if covered[i]:
            last_covered = i
        elif last_covered >= 0:
            prob_avg[i] = prob_avg[last_covered]

    frame_results = []
    for i in range(n_frames):
        probs = {name: float(prob_avg[i, ci]) for ci, name in enumerate(class_names)}
        best_label = max(probs, key=probs.get)
        frame_results.append({"label": best_label, "probabilities": probs})
    return frame_results
