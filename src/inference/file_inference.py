from __future__ import annotations

from pathlib import Path
from typing import Dict

from src.data.video_pose import video_to_pose_array
from src.inference.predictor import predict_from_pose, predict_from_npy

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}


def predict_from_file(file_path: str, checkpoint_path: str, yolo_model_path: str | None = None) -> Dict[str, object]:
    """
    Предсказывает класс либо .npy файла, либо video файла
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    suffix = path.suffix.lower()
    if suffix == ".npy":
        return predict_from_npy(str(path), checkpoint_path=checkpoint_path)

    if suffix in VIDEO_EXTENSIONS:
        if yolo_model_path is None:
            raise ValueError("yolo_model_path is required for video inference")
        pose = video_to_pose_array(str(path), yolo_model_path=yolo_model_path)
        result = predict_from_pose(pose, checkpoint_path=checkpoint_path)
        result["source"] = str(path)
        return result

    raise ValueError(f"Unsupported file type: {suffix}. Use .npy or video file.")
