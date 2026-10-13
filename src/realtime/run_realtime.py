from __future__ import annotations

import argparse
import time
from collections import deque
from typing import Optional

import cv2
import psutil

from src import load_config
from src.data.visualize_keypoints import draw_upper_body
from src.realtime.early_warning import EarlyWarning
from src.realtime.predictor import RealtimePredictor


class LatencyMeter:
    def __init__(self, window: int = 30):
        self.window = window
        self.latencies = deque(maxlen=window)

    def tick(self, latency_s: float) -> None:
        self.latencies.append(latency_s)

    @property
    def fps(self) -> float:
        if not self.latencies:
            return 0.0
        mean = sum(self.latencies) / len(self.latencies)
        return 1.0 / mean if mean > 0 else 0.0

    @property
    def avg_latency_ms(self) -> float:
        if not self.latencies:
            return 0.0
        return 1000.0 * sum(self.latencies) / len(self.latencies)


def draw_overlay(frame, label, confidence, warning, fps, latency_ms, cpu=0.0):
    color = (0, 0, 255) if warning else (0, 200, 0)
    cv2.rectangle(frame, (10, 10), (520, 155), (0, 0, 0), -1)
    cv2.putText(frame, f"Label: {label}", (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.putText(frame, f"Conf: {confidence:.2%}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    cv2.putText(frame, f"FPS: {fps:.1f}  Latency: {latency_ms:.1f}ms",
                (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    # [I5] utilizations CPU (spec §7.1/§7.3)
    cv2.putText(frame, f"CPU: {cpu:.0f}%", (20, 145),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    if warning:
        cv2.putText(frame, "EARLY WARNING: FATIGUE DETECTED", (20, 185),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return frame


def run(camera_index: int = 0, checkpoint_path: Optional[str] = None,
        yolo_model_path: Optional[str] = None, window_size: int = 30,
        show_keypoints: bool = True) -> None:
    cfg = load_config("configs/default.yaml")
    ckpt = checkpoint_path or cfg.paths.checkpoint_path
    yolo = yolo_model_path or cfg.paths.yolo_model_path

    predictor = RealtimePredictor(ckpt, yolo, window_size=window_size)
    warning = EarlyWarning(threshold=5)
    meter = LatencyMeter()
    # [I5] primer call untuk inisialisasi; pengukuran aktual pakai interval=None
    psutil.cpu_percent(interval=None)
    cpu_percent = 0.0

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise IOError(f"Cannot open camera index {camera_index}")

    try:
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t0 = time.perf_counter()
            result = predictor.push_frame(frame)
            latency = time.perf_counter() - t0
            meter.tick(latency)

            # gambar 11 keypoint upper-body + skeleton (jika ada deteksi)
            if show_keypoints and predictor.last_xy is not None:
                frame = draw_upper_body(frame, predictor.last_xy, predictor.last_conf)

            # [I5] sampling CPU% tiap N frame (non-blocking, interval=None)
            if frame_idx % 10 == 0:
                cpu_percent = psutil.cpu_percent(interval=None)
            frame_idx += 1

            if result is not None:
                label = result["label"]
                conf = result["probabilities"][label]
                warn = warning.update(label)
                frame = draw_overlay(frame, label, conf, warn, meter.fps,
                                     meter.avg_latency_ms, cpu_percent)
            else:
                cv2.putText(frame, "Buffering...", (20, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

            cv2.imshow("Fatigue Detection (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time fatigue detection")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--yolo", default=None)
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--no-keypoints", action="store_true",
                        help="Matikan visualisasi keypoint di overlay")
    args = parser.parse_args()
    run(args.camera, args.checkpoint, args.yolo, args.window,
        show_keypoints=not args.no_keypoints)


if __name__ == "__main__":
    main()
