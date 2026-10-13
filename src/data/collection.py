from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from tqdm.auto import tqdm

from src.config import ProjectConfig
from src.data.video_pose import is_no_detection, pose_from_prediction, load_yolo_pose_model

VIDEO_EXTENSIONS = [".mp4"]


def logger(message: str) -> None:
    tqdm.write(message)


def save_pose_array(pose_array: np.ndarray, processed_dir: str, class_name: str, stem: str) -> str:
    out_dir = Path(processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{class_name}_{stem}.npy"
    np.save(out_path, pose_array.astype(np.float32))
    return str(out_path)


def process_all_videos(config: ProjectConfig, save_skeleton_video: bool = False) -> None:
    raw_dir = Path(config.paths.raw_dir)
    processed_dir = Path(config.paths.processed_dir)
    skeleton_dir = Path(config.paths.skeleton_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    if save_skeleton_video:
        skeleton_dir.mkdir(parents=True, exist_ok=True)

    model = load_yolo_pose_model(config.paths.yolo_model_path)

    class_dirs = [p for p in sorted(raw_dir.iterdir()) if p.is_dir()]
    all_videos = []
    for class_dir in class_dirs:
        vids = [p for p in sorted(class_dir.iterdir())
                if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS]
        all_videos.extend((class_dir, v) for v in vids)

    if not all_videos:
        logger(f"No video files found in '{raw_dir}'.")
        return

    logger(f"Found classes: {[d.name for d in class_dirs]}")
    logger(f"Total videos to process: {len(all_videos)}")

    pbar = tqdm(all_videos, desc="Processing videos", unit="video")
    for class_dir, video_path in pbar:
        pbar.set_postfix_str(f"{class_dir.name}/{video_path.name}")
        capture = cv2.VideoCapture(str(video_path))
        fps = int(capture.get(cv2.CAP_PROP_FPS)) or 30
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if save_skeleton_video:
            class_skel = skeleton_dir / class_dir.name
            class_skel.mkdir(parents=True, exist_ok=True)
            out_vid = class_skel / f"{video_path.stem}_skeleton.mp4"
            writer = cv2.VideoWriter(str(out_vid),
                                     cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps, (width, height))

        sequence = []
        while capture.isOpened():
            ok, frame = capture.read()
            if not ok:
                break
            results = model.predict(frame, verbose=False, conf=0.7)
            if writer is not None:
                writer.write(results[0].plot())
            feat = pose_from_prediction(results)
            # Buang frame tanpa deteksi (sentinel semua -1.0) agar konsisten
            # dengan RealtimePredictor.push_frame di inference.
            if is_no_detection(feat):
                continue
            sequence.append(feat)

        capture.release()
        if writer is not None:
            writer.release()

        # Jangan simpan array malformed bila tidak ada frame yang tersisa.
        if len(sequence) == 0:
            logger(f"Skip '{video_path.name}': no pose detections after filtering.")
            continue

        save_pose_array(np.array(sequence), str(processed_dir),
                        class_dir.name, video_path.stem)

    logger(f"Done: processed {len(all_videos)} videos.")
