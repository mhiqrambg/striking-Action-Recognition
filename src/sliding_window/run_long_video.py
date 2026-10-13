from __future__ import annotations

import argparse
import json
from pathlib import Path

from src import load_config
from src.sliding_window.classify_long_video import process_long_video
from src.sliding_window.video_overlay import create_video_overlay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run long-video inference pipeline.")
    parser.add_argument("video_path", help="Path to input video file")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--yolo", default=None)
    parser.add_argument("--stride", type=int, default=15)
    parser.add_argument("--no-overlay", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    checkpoint_path = args.checkpoint or cfg.paths.checkpoint_path
    yolo_model_path = args.yolo or cfg.paths.yolo_model_path

    frame_results = process_long_video(
        file_path=args.video_path,
        checkpoint_path=checkpoint_path,
        yolo_model_path=yolo_model_path,
        stride=args.stride,
    )

    result_path = Path(args.video_path).with_suffix(".predictions.json")
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(frame_results, f, ensure_ascii=False, indent=2)
    print(f"Predictions saved: {result_path}")

    if args.no_overlay:
        return

    overlay_path = create_video_overlay(args.video_path, frame_results)
    print(f"Overlay saved: {overlay_path}")


if __name__ == "__main__":
    main()
