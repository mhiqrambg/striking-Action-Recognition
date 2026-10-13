from __future__ import annotations

import cv2
import os
import subprocess

from tqdm import tqdm
from pathlib import Path
from typing import Dict, List


def _reencode_h264(input_path: str) -> str:
    """
    Re-encode video dari codec mp4v/mpeg4 (tidak universal) ke H.264 agar bisa
    diputar di QuickTime/browser/player umum. Mengembalikan path video H.264.
    """
    h264_path = str(Path(input_path).with_name(f"{Path(input_path).stem}_h264.mp4"))

    ffmpeg_exe = None
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ffmpeg_exe = None

    if ffmpeg_exe is None:
        import shutil
        ffmpeg_exe = shutil.which("ffmpeg")

    if ffmpeg_exe is None:
        # Tidak ada ffmpeg: kembalikan file apa adanya (mungkin tak bisa diputar).
        return input_path

    subprocess.run(
        [
            ffmpeg_exe, "-y", "-i", input_path,
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",   # kompatibilitas maksimal (QuickTime/Safari)
            "-preset", "fast",
            "-crf", "23",
            h264_path,
        ],
        check=True,
        capture_output=True,
    )
    os.remove(input_path)
    return h264_path


def create_video_overlay(file_path: str, frame_results: List[Dict[str, object]]):
    """
    Menggambar prediksi per-frame pada video lalu re-encode ke H.264.

    frame_results : list per-frame (label + probabilities)
    return        : path video overlay H.264 (siap diputar)
    """
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {file_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Tulis dulu dengan codec mp4v (didukung OpenCV), lalu re-encode ke H.264.
    raw_path = str(Path(file_path).with_name(f"{Path(file_path).stem}_overlay_raw.mp4"))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(raw_path, fourcc, fps, (w, h))

    print(f"Merender overlay ke: {raw_path}")

    frame_idx = 0
    pbar = tqdm(total=total_frames)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_label = "normal"
        current_conf = 0.0

        if frame_idx < len(frame_results):
            res = frame_results[frame_idx]
            current_label = res["label"]
            current_conf = res["probabilities"][current_label]

        is_fatigue = current_label != "normal"
        overlay = frame.copy()
        alpha = current_conf if is_fatigue else 0.2
        if is_fatigue:
            cv2.rectangle(overlay, (10, 10), (450, 90), (0, 0, 180), -1)
        else:
            cv2.rectangle(overlay, (10, 10), (450, 90), (0, 0, 0), -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        text = f"Label: {current_label}"
        conf_text = f"Conf: {current_conf:.2%}"

        if is_fatigue:
            color = (0, 255, 255)
            thickness = 3
        else:
            color = (180, 180, 180)
            thickness = 1

        cv2.putText(frame, text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, thickness)
        cv2.putText(frame, conf_text, (20, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)

        out.write(frame)
        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    out.release()

    # Re-encode ke H.264 agar bisa diputar di semua player.
    return _reencode_h264(raw_path)
