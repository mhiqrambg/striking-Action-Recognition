"""
Visualisasi keypoint sesuai thesis.

Menampilkan apa yang sebenarnya dideteksi oleh YOLOv11n-Pose dan bagaimana
fitur thesis dihitung:
  - 11 keypoint upper-body (COCO idx 0-10) — sisanya (11-16) tidak dipakai
  - origin normalisasi = midpoint bahu (idx 5 & 6)
  - 5 fitur geometris gesture kelelahan (theta_head, d_forward,
    delta_y_shoulder, d_wrist_nose, theta_torso)

Pemakaian:
    # gambar / video tunggal -> simpan overlay
    python -m src.data.visualize_keypoints path/ke/video.mp4
    python -m src.data.visualize_keypoints path/ke/gambar.jpg --output out.jpg

    # tampilkan jendela (butuh GUI)
    python -m src.data.visualize_keypoints path/ke/video.mp4 --show

    # cetak nilai fitur ke terminal (setiap N frame)
    python -m src.data.visualize_keypoints path/ke/video.mp4 --print-features
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.data.features import (
    L_EAR,
    L_ELBOW,
    L_SHOULDER,
    L_WRIST,
    NOSE,
    R_EAR,
    R_ELBOW,
    R_SHOULDER,
    R_WRIST,
    compute_geometric_features,
)

# 11 keypoint upper-body (COCO idx 0-10) — sesuai thesis
KEYPOINT_NAMES = {
    0: "nose",
    1: "left_eye",
    2: "right_eye",
    3: "left_ear",
    4: "right_ear",
    5: "left_shoulder",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    9: "left_wrist",
    10: "right_wrist",
}

# Koneksi kerangka HANYA di antara idx 0-10 (upper-body)
SKELETON = [
    (0, 1), (0, 2),          # hidung - mata
    (1, 3), (2, 4),          # mata - telinga
    (5, 6),                  # bahu kiri - bahu kanan
    (5, 7), (7, 9),          # lengan kiri
    (6, 8), (8, 10),         # lengan kanan
    (5, 0), (6, 0),          # bahu - hidung (garis leher)
]

# Warna BGR per keypoint
COLORS = {
    0: (0, 255, 255),    # nose
    1: (255, 128, 0), 2: (255, 128, 0),
    3: (0, 128, 255), 4: (0, 128, 255),
    5: (0, 255, 0), 6: (0, 200, 0),      # bahu (origin) - hijau
    7: (255, 0, 0), 8: (200, 0, 0),
    9: (255, 0, 255), 10: (200, 0, 200),
}

FEATURE_NAMES = [
    "theta_head", "d_forward", "delta_y_shoulder", "d_wrist_nose", "theta_torso",
]


def draw_upper_body(frame: np.ndarray, kpts_xy: np.ndarray, kpts_conf: np.ndarray) -> np.ndarray:
    """
    Gambar 11 keypoint upper-body + skeleton + origin bahu.

    kpts_xy   : (17, 2) koordinat ternormalisasi 0-1 (xyn)
    kpts_conf : (17,) confidence
    """
    h, w = frame.shape[:2]
    vis = frame.copy()

    def px(i):
        return int(kpts_xy[i, 0] * w), int(kpts_xy[i, 1] * h)

    # skeleton
    for a, b in SKELETON:
        cv2.line(vis, px(a), px(b), (0, 255, 255), 2, cv2.LINE_AA)

    # keypoint + label
    for i in range(11):
        x, y = px(i)
        color = COLORS.get(i, (255, 255, 255))
        cv2.circle(vis, (x, y), 5, color, -1, cv2.LINE_AA)
        cv2.circle(vis, (x, y), 5, (0, 0, 0), 1, cv2.LINE_AA)
        label = f"{i}:{KEYPOINT_NAMES[i]}({kpts_conf[i]:.2f})"
        cv2.putText(vis, label, (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, color, 1, cv2.LINE_AA)

    # origin = midpoint bahu
    mx = (kpts_xy[L_SHOULDER, 0] + kpts_xy[R_SHOULDER, 0]) / 2 * w
    my = (kpts_xy[L_SHOULDER, 1] + kpts_xy[R_SHOULDER, 1]) / 2 * h
    cv2.drawMarker(vis, (int(mx), int(my)), (255, 255, 255),
                   cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
    cv2.putText(vis, "origin (mid-shoulder)", (int(mx) + 10, int(my) + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # tandai keypoint bawah (11-16) yang DIBUANG thesis, sebagai abu-abu
    for i in range(11, min(17, kpts_xy.shape[0])):
        x, y = px(i)
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(vis, (x, y), 3, (90, 90, 90), -1, cv2.LINE_AA)

    return vis


def draw_feature_panel(frame: np.ndarray, feat: np.ndarray) -> np.ndarray:
    """Panel teks 5 fitur geometris di kiri atas."""
    vis = frame.copy()
    lines = ["5 fitur geometris (thesis):"]
    for name, val in zip(FEATURE_NAMES, feat):
        lines.append(f"  {name:18s} = {val:+.4f}")
    y0 = 24
    cv2.rectangle(vis, (8, 8), (360, y0 + 20 * len(lines) + 6), (0, 0, 0), -1)
    for k, line in enumerate(lines):
        color = (0, 255, 255) if k == 0 else (255, 255, 255)
        cv2.putText(vis, line, (16, y0 + 20 * k), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1, cv2.LINE_AA)
    return vis


def _extract_raw(prediction):
    """Ambil (17,2) xyn + (17,) conf dari hasil YOLO; None jika tak ada deteksi."""
    if len(prediction[0].keypoints.data) == 0:
        return None, None
    xy = prediction[0].keypoints.xyn[0].cpu().numpy()            # (17,2)
    conf = prediction[0].keypoints.data[0].cpu().numpy()[:, 2]   # (17,)
    return xy, conf


def process_image(model, path: str, output: str, show_features: bool) -> str:
    frame = cv2.imread(path)
    if frame is None:
        raise IOError(f"Cannot read image: {path}")
    pred = model.predict(frame, verbose=False, conf=0.7)
    xy, conf = _extract_raw(pred)
    if xy is None:
        print("Tidak ada deteksi pada gambar ini.")
        vis = frame
    else:
        vis = draw_upper_body(frame, xy, conf)
        if show_features:
            feat = compute_geometric_features(
                np.hstack([xy, conf.reshape(-1, 1)]).astype(np.float32)[:11]
            )
            vis = draw_feature_panel(vis, feat)
    cv2.imwrite(output, vis)
    return output


def process_video(model, path: str, output: str, show: bool,
                  show_features: bool, print_features: bool, stride: int) -> None:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = None
    if output:
        writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    idx = 0
    n_det = 0
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx % stride != 0:
            if writer is not None:
                writer.write(frame)
            continue
        pred = model.predict(frame, verbose=False, conf=0.7)
        xy, conf = _extract_raw(pred)
        if xy is None:
            if writer is not None:
                writer.write(frame)
            if show:
                cv2.imshow("keypoints (q=quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            continue
        n_det += 1
        vis = draw_upper_body(frame, xy, conf)
        if show_features:
            feat = compute_geometric_features(
                np.hstack([xy, conf.reshape(-1, 1)]).astype(np.float32)[:11]
            )
            vis = draw_feature_panel(vis, feat)
            if print_features:
                print(f"frame {idx}: " + ", ".join(
                    f"{n}={v:+.4f}" for n, v in zip(FEATURE_NAMES, feat)))
        if writer is not None:
            writer.write(vis)
        if show:
            cv2.imshow("keypoints (q=quit)", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer is not None:
        writer.release()
    if show:
        cv2.destroyAllWindows()
    print(f"Selesai. Frame diproses: {idx}, frame dengan deteksi: {n_det}")
    if output:
        print(f"Output: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualisasi 11 keypoint upper-body + 5 fitur (thesis).")
    parser.add_argument("input", help="Path ke video atau gambar")
    parser.add_argument("--output", default=None,
                        help="Path output (default: <nama>_keypoints.mp4 / .jpg)")
    parser.add_argument("--yolo", default="models/yolo11n-pose.pt", help="Model YOLO pose")
    parser.add_argument("--stride", type=int, default=1, help="Proses tiap N frame (video)")
    parser.add_argument("--show", action="store_true", help="Tampilkan jendela (butuh GUI)")
    parser.add_argument("--features", action="store_true", help="Tampilkan panel 5 fitur geometris")
    parser.add_argument("--print-features", action="store_true", help="Cetak nilai fitur ke terminal")
    args = parser.parse_args()

    from src.data.video_pose import load_yolo_pose_model
    model = load_yolo_pose_model(args.yolo)

    p = Path(args.input)
    is_image = p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    if is_image:
        out = args.output or str(p.with_name(f"{p.stem}_keypoints.jpg"))
        res = process_image(model, str(p), out, show_features=args.features or True)
        print(f"Output: {res}")
    else:
        out = args.output or str(p.with_name(f"{p.stem}_keypoints.mp4"))
        process_video(model, str(p), out, args.show,
                      args.features or args.print_features, args.print_features, args.stride)


if __name__ == "__main__":
    main()
