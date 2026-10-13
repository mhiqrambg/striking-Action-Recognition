# Implementasi Sistem Klasifikasi Kelelahan (Thesis Alignment) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Menyesuaikan codebase agar sesuai proposal thesis: pipeline 11 keypoint + 5 fitur geometris (38-dim), Modified Transformer Encoder (Learnable PE + Focal Loss), training/evaluasi K-Fold + ablation, dan sistem real-time webcam.

**Architecture:** Dua tahap — YOLOv11n-Pose mengekstraksi 11 keypoint upper-body, feature engineering menghasilkan vektor 38-dim/frame, sliding window (30, 38), lalu Transformer Encoder (4 layer, 8 head, d_model 128, FFN 256, CLS token) mengklasifikasi 3 kelas. Flag `use_learnable_pe`/`use_focal_loss` memungkinkan 4 skenario ablation dari satu kode.

**Tech Stack:** Python 3.11, PyTorch, Ultralytics YOLO11, OpenCV, NumPy, scikit-learn, torchmetrics, pytest.

**Spec:** `docs/superpowers/specs/2026-10-09-kelelahan-transformer-design.md`

## Global Constraints

- input_dim = **38**; d_model = **128**; num_heads = **8**; num_layers = **4**; ffn_dim = **256**; dropout = **0.1**
- window_size (max_len) = **30**; stride = **15**; num_classes = **3**
- Keypoint: hanya COCO idx **0–10** (11 titik upper-body)
- Origin normalisasi = **midpoint bahu** (shoulder 5,6), BUKAN pinggul
- Focal Loss: γ = **2.0**; α = **[0.25 (normal), 0.40 (light), 0.35 (heavy)]**
- Urutan kelas tetap: **["normal", "light", "heavy"]**
- Validasi: **StratifiedGroupKFold k=5**, group = nama video
- Metrik: Accuracy, Precision/Recall/F1 **weighted**, **PR-AUC per kelas**
- Python 3.11 (venv di `.venv`); jalankan semua perintah dengan `.venv/bin/python`
- Target real-time: latensi < **100 ms**, FPS ≥ **15**, CPU < **80%**

---

## File Structure

**Dibuat baru:**
- `tests/` — test suite (pytest)
- `src/data/features.py` — 5 fitur geometris
- `src/training/losses.py` — FocalLoss
- `src/evaluation/__init__.py`, `src/evaluation/metrics.py` — metrik evaluasi
- `src/realtime/__init__.py`, `src/realtime/early_warning.py`, `src/realtime/predictor.py`, `src/realtime/run_realtime.py`

**Dimodifikasi:**
- `src/config.py` — tambah field konfigurasi baru + class_names
- `src/data/video_pose.py` — 11 keypoint, origin bahu, output (T, 38)
- `src/data/collection.py` — simpan (T, 38)
- `src/data/preprocessing.py` — normalisasi origin bahu
- `src/data/dataset.py` — sliding window + group_id + class order
- `src/data/dataloader.py` — StratifiedGroupKFold
- `src/models/embedding.py` — LearnablePositionalEncoding
- `src/models/transformer.py` — CLS token, FFN 256
- `src/training/checkpoint.py` — build_model pakai config baru
- `src/training/trainer.py` — K-fold + ablation + Focal Loss
- `requirements.txt` — tambah pytest, pandas, psutil

---

## Task 0: Setup Test Infrastructure

**Files:**
- Modify: `requirements.txt`
- Create: `pytest.ini`

**Interfaces:**
- Produces: kemampuan menjalankan `pytest` untuk semua task berikutnya.

- [ ] **Step 1: Tambah dependency ke requirements.txt**

Tambahkan baris berikut di akhir `requirements.txt`:
```
pytest>=8.0.0
pandas>=2.0.0
psutil>=5.9.0
seaborn>=0.12.0
```

- [ ] **Step 2: Install**

Run:
```bash
cd /Users/mhiqrambg/Documents/Github-Repo/ML_PROJECT-Striking-Action-Recognition
VIRTUAL_ENV=.venv uv pip install pytest pandas psutil seaborn
```
Expected: terinstall tanpa error.

- [ ] **Step 3: Buat pytest.ini**

Buat `pytest.ini`:
```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
```

- [ ] **Step 4: Buat folder tests + smoke test**

Buat `tests/test_smoke.py`:
```python
def test_pytest_runs():
    assert 1 + 1 == 2
```

- [ ] **Step 5: Verifikasi pytest jalan**

Run: `.venv/bin/python -m pytest tests/test_smoke.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add requirements.txt pytest.ini tests/test_smoke.py
git commit -m "chore: setup pytest infrastructure"
```

---

## Task 1: Config Baru

**Files:**
- Modify: `src/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces:
  - `ModelConfig(input_dim=38, dim_model=128, num_heads=8, num_layers=4, ffn_dim=256, dropout=0.1, num_classes=3, use_learnable_pe=True, use_focal_loss=True)`
  - `DataConfig(window_size=30, stride=15, batch_size=16, val_split=0.2, random_seed=42, num_workers=0, num_folds=5)`
  - `TrainingConfig(learning_rate=1e-4, weight_decay=0.01, epochs=50, log_dir="runs", focal_gamma=2.0, focal_alpha=(0.25, 0.40, 0.35))`
  - `PathsConfig(raw_dir, processed_dir, skeleton_dir, yolo_model_path, checkpoint_path, results_dir)`
  - `ProjectConfig` dengan atribut tambahan `class_names: List[str] = ["normal","light","heavy"]`
  - `load_config(path)`, `config_from_dict(dict)`

- [ ] **Step 1: Tulis failing test**

Buat `tests/test_config.py`:
```python
from src.config import load_config, config_from_dict


def test_defaults():
    cfg = load_config("configs/default.yaml")
    assert cfg.model.input_dim == 38
    assert cfg.model.dim_model == 128
    assert cfg.model.ffn_dim == 256
    assert cfg.model.num_classes == 3
    assert cfg.data.window_size == 30
    assert cfg.data.stride == 15
    assert cfg.data.num_folds == 5
    assert cfg.training.focal_gamma == 2.0
    assert list(cfg.training.focal_alpha) == [0.25, 0.40, 0.35]
    assert cfg.class_names == ["normal", "light", "heavy"]


def test_roundtrip():
    cfg = load_config("configs/default.yaml")
    cfg2 = config_from_dict(cfg.to_dict())
    assert cfg2.model.input_dim == cfg.model.input_dim
    assert cfg2.data.window_size == cfg.data.window_size
```

- [ ] **Step 2: Run test (gagal)**

Run: `.venv/bin/python -m pytest tests/test_config.py -v`
Expected: FAIL (AttributeError / field belum ada)

- [ ] **Step 3: Update configs/default.yaml**

Timpa `configs/default.yaml`:
```yaml
model:
  input_dim: 38
  dim_model: 128
  num_heads: 8
  num_layers: 4
  ffn_dim: 256
  dropout: 0.1
  num_classes: 3
  use_learnable_pe: true
  use_focal_loss: true

data:
  window_size: 30
  stride: 15
  batch_size: 16
  val_split: 0.2
  random_seed: 42
  num_workers: 0
  num_folds: 5

training:
  learning_rate: 0.0001
  weight_decay: 0.01
  epochs: 50
  log_dir: "runs"
  focal_gamma: 2.0
  focal_alpha: [0.25, 0.40, 0.35]

paths:
  raw_dir: "data/raw"
  processed_dir: "data/processed"
  skeleton_dir: "data/with_skeleton"
  yolo_model_path: "models/yolo11n-pose.pt"
  checkpoint_path: "models/fatigue_transformer.pt"
  results_dir: "results"

class_names: ["normal", "light", "heavy"]
```

- [ ] **Step 4: Tulis src/config.py baru**

Timpa `src/config.py`:
```python
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


@dataclass
class ModelConfig:
    input_dim: int = 38
    dim_model: int = 128
    num_heads: int = 8
    num_layers: int = 4
    ffn_dim: int = 256
    dropout: float = 0.1
    num_classes: int = 3
    use_learnable_pe: bool = True
    use_focal_loss: bool = True


@dataclass
class DataConfig:
    window_size: int = 30
    stride: int = 15
    batch_size: int = 16
    val_split: float = 0.2
    random_seed: int = 42
    num_workers: int = 0
    num_folds: int = 5


@dataclass
class TrainingConfig:
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    epochs: int = 50
    log_dir: str = "runs"
    focal_gamma: float = 2.0
    focal_alpha: Tuple[float, float, float] = (0.25, 0.40, 0.35)


@dataclass
class PathsConfig:
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    skeleton_dir: str = "data/with_skeleton"
    yolo_model_path: str = "models/yolo11n-pose.pt"
    checkpoint_path: str = "models/fatigue_transformer.pt"
    results_dir: str = "results"


@dataclass
class ProjectConfig:
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    paths: PathsConfig
    class_names: List[str] = field(default_factory=lambda: ["normal", "light", "heavy"])

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def config_from_dict(values: Dict[str, Any]) -> ProjectConfig:
    return ProjectConfig(
        model=ModelConfig(**values["model"]),
        data=DataConfig(**values["data"]),
        training=TrainingConfig(**values["training"]),
        paths=PathsConfig(**values["paths"]),
        class_names=list(values.get("class_names", ["normal", "light", "heavy"])),
    )


def load_config(path: str) -> ProjectConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        input_config = yaml.safe_load(file) or {}
    return config_from_dict(input_config)
```

- [ ] **Step 5: Run test (lulus)**

Run: `.venv/bin/python -m pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add configs/default.yaml src/config.py tests/test_config.py
git commit -m "feat(config): add thesis-aligned config fields and class order"
```

---

## Task 2: Fitur Geometris (features.py)

**Files:**
- Create: `src/data/features.py`
- Test: `tests/test_features.py`

**Interfaces:**
- Produces: `compute_geometric_features(kpts: np.ndarray) -> np.ndarray`
  - Input: keypoints ternormalisasi shape `(..., 11, 3)` (x, y, conf)
  - Output: shape `(..., 5)` → `[θ_head, d_forward, Δy_shoulder, d_wrist_nose, θ_torso]`
  - Konstanta indeks: `NOSE=0, L_EAR=3, R_EAR=4, L_SHOULDER=5, R_SHOULDER=6, L_ELBOW=7, R_ELBOW=8, L_WRIST=9, R_WRIST=10`

- [ ] **Step 1: Tulis failing test**

Buat `tests/test_features.py`:
```python
import numpy as np
from src.data.features import compute_geometric_features


def _make_kpts():
    # 11 keypoint, koordinat ternormalisasi origin bahu
    k = np.zeros((11, 3), dtype=np.float32)
    k[:, 2] = 1.0  # confidence
    k[0] = [0.0, -0.30, 1.0]     # nose
    k[5] = [-0.10, 0.0, 1.0]     # left shoulder
    k[6] = [0.10, 0.0, 1.0]      # right shoulder
    k[9] = [-0.15, -0.05, 1.0]   # left wrist
    k[10] = [0.15, -0.05, 1.0]   # right wrist
    return k


def test_output_shape():
    k = _make_kpts()
    f = compute_geometric_features(k)
    assert f.shape == (5,)


def test_batch_shape():
    k = np.stack([_make_kpts()] * 30, axis=0)  # (30, 11, 3)
    f = compute_geometric_features(k)
    assert f.shape == (30, 5)


def test_d_forward_equals_nose_x():
    k = _make_kpts()
    f = compute_geometric_features(k)
    # d_forward = x_nose - x_midShoulder ; midShoulder.x = 0
    assert abs(f[1] - 0.0) < 1e-6


def test_shoulder_asymmetry():
    k = _make_kpts()
    k[5] = [-0.10, -0.05, 1.0]  # left shoulder lebih tinggi
    k[6] = [0.10, 0.05, 1.0]
    f = compute_geometric_features(k)
    assert abs(f[2] - 0.10) < 1e-6
```

- [ ] **Step 2: Run test (gagal)**

Run: `.venv/bin/python -m pytest tests/test_features.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Tulis src/data/features.py**

Buat `src/data/features.py`:
```python
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
```

- [ ] **Step 4: Run test (lulus)**

Run: `.venv/bin/python -m pytest tests/test_features.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/data/features.py tests/test_features.py
git commit -m "feat(data): add 5 geometric fatigue features (thesis formulas)"
```

---

## Task 3: Ekstraksi Pose 11 Keypoint → 38-dim

**Files:**
- Modify: `src/data/video_pose.py`
- Test: `tests/test_video_pose.py`

**Interfaces:**
- Consumes: `compute_geometric_features` (Task 2)
- Produces:
  - `pose_from_prediction(prediction) -> np.ndarray` shape `(38,)` (11×3 + 5 fitur, origin bahu)
  - `video_to_pose_array(video_path, yolo_model_path, conf=0.7) -> np.ndarray` shape `(T, 38)`
  - `load_yolo_pose_model(model_path) -> YOLO`
  - `NUM_KEYPOINTS = 11`, `FEATURE_DIM = 38`

- [ ] **Step 1: Tulis failing test**

Buat `tests/test_video_pose.py`:
```python
import numpy as np
from src.data.video_pose import normalize_and_featurize, FEATURE_DIM, NUM_KEYPOINTS


def test_feature_dim():
    assert FEATURE_DIM == 38
    assert NUM_KEYPOINTS == 11


def test_normalize_and_featurize_shape():
    k = np.zeros((17, 3), dtype=np.float32)
    k[:, 2] = 1.0
    k[0] = [0.5, 0.3, 1.0]    # nose
    k[5] = [0.4, 0.6, 1.0]    # left shoulder
    k[6] = [0.6, 0.6, 1.0]    # right shoulder
    out = normalize_and_featurize(k)
    assert out.shape == (FEATURE_DIM,)


def test_origin_is_shoulder_midpoint():
    k = np.zeros((17, 3), dtype=np.float32)
    k[:, 2] = 1.0
    k[5] = [0.4, 0.6, 1.0]
    k[6] = [0.6, 0.6, 1.0]
    k[0] = [0.5, 0.2, 1.0]    # nose tepat di atas midpoint bahu
    out = normalize_and_featurize(k)
    # x nose setelah normalisasi = 0.5 - 0.5 = 0
    assert abs(out[0] - 0.0) < 1e-5
```

- [ ] **Step 2: Run test (gagal)**

Run: `.venv/bin/python -m pytest tests/test_video_pose.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Tulis src/data/video_pose.py**

Timpa `src/data/video_pose.py`:
```python
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
        return np.full((FEATURE_DIM,), -1.0, dtype=np.float32)
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
        frames.append(pose_from_prediction(prediction))

    capture.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from video: {video_path}")
    return np.array(frames, dtype=np.float32)
```

- [ ] **Step 4: Run test (lulus)**

Run: `.venv/bin/python -m pytest tests/test_video_pose.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/data/video_pose.py tests/test_video_pose.py
git commit -m "feat(data): extract 11 keypoints + shoulder-origin 38-dim features"
```

---

## Task 4: Collection — Simpan (T, 38)

**Files:**
- Modify: `src/data/collection.py`
- Test: `tests/test_collection.py`

**Interfaces:**
- Consumes: `video_to_pose_array` (Task 3)
- Produces: `process_all_videos(config, save_skeleton_video=False) -> None`
  - Membaca `data/raw/<kelas>/*.mp4` → simpan `data/processed/<kelas>_<stem>.npy` shape `(T, 38)`

- [ ] **Step 1: Tulis failing test**

Buat `tests/test_collection.py`:
```python
import numpy as np
from pathlib import Path
from src.data.collection import save_pose_array


def test_save_pose_array(tmp_path):
    arr = np.zeros((50, 38), dtype=np.float32)
    out = save_pose_array(arr, str(tmp_path), "normal", "video1")
    assert Path(out).exists()
    loaded = np.load(out)
    assert loaded.shape == (50, 38)
    assert Path(out).name == "normal_video1.npy"
```

- [ ] **Step 2: Run test (gagal)**

Run: `.venv/bin/python -m pytest tests/test_collection.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Tulis src/data/collection.py**

Timpa `src/data/collection.py`:
```python
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from tqdm.auto import tqdm

from src.config import ProjectConfig
from src.data.video_pose import pose_from_prediction, load_yolo_pose_model

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
            sequence.append(pose_from_prediction(results))

        capture.release()
        if writer is not None:
            writer.release()

        save_pose_array(np.array(sequence), str(processed_dir),
                        class_dir.name, video_path.stem)

    logger(f"Done: processed {len(all_videos)} videos.")
```

- [ ] **Step 4: Run test (lulus)**

Run: `.venv/bin/python -m pytest tests/test_collection.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/data/collection.py tests/test_collection.py
git commit -m "feat(data): collection saves (T,38) pose arrays"
```

---

## Task 5: Dataset — Sliding Window + Group

**Files:**
- Modify: `src/data/dataset.py`
- Modify: `src/data/preprocessing.py`
- Test: `tests/test_dataset.py`

**Interfaces:**
- Consumes: config (Task 1)
- Produces:
  - `SlidingWindowDataset(processed_dir, class_names, window_size=30, stride=15, augment=False)`
    - `__len__() -> int` (jumlah window)
    - `__getitem__(idx) -> (x: Tensor(30,38), y: int)`
    - atribut `.labels: List[int]`, `.groups: List[str]`, `.label_map: Dict[str,int]`
  - `class_name_from_file(path) -> str` (prefix sebelum `_`)
  - `group_id_from_file(path) -> str` (nama video, tanpa prefix kelas)

- [ ] **Step 1: Tulis failing test**

Buat `tests/test_dataset.py`:
```python
import numpy as np
from src.data.dataset import SlidingWindowDataset, group_id_from_file
from pathlib import Path


def _write_npy(dirpath, name, T=90):
    arr = np.random.randn(T, 38).astype(np.float32)
    np.save(Path(dirpath) / name, arr)


def test_group_id():
    p = Path("normal_video1.npy")
    assert group_id_from_file(p) == "video1"


def test_windows_count(tmp_path):
    _write_npy(tmp_path, "normal_a.npy", T=90)   # (90-30)/15+1 = 5
    _write_npy(tmp_path, "light_b.npy", T=60)    # (60-30)/15+1 = 3
    ds = SlidingWindowDataset(str(tmp_path), ["normal", "light", "heavy"],
                              window_size=30, stride=15)
    assert len(ds) == 8


def test_getitem_shape_and_label(tmp_path):
    _write_npy(tmp_path, "heavy_a.npy", T=60)
    ds = SlidingWindowDataset(str(tmp_path), ["normal", "light", "heavy"],
                              window_size=30, stride=15)
    x, y = ds[0]
    assert tuple(x.shape) == (30, 38)
    assert y == ds.label_map["heavy"]


def test_short_video_skipped(tmp_path):
    _write_npy(tmp_path, "normal_short.npy", T=20)  # < window
    _write_npy(tmp_path, "normal_ok.npy", T=45)     # (45-30)/15+1 = 2
    ds = SlidingWindowDataset(str(tmp_path), ["normal", "light", "heavy"],
                              window_size=30, stride=15)
    assert len(ds) == 2
```

- [ ] **Step 2: Run test (gagal)**

Run: `.venv/bin/python -m pytest tests/test_dataset.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Tulis src/data/preprocessing.py**

Timpa `src/data/preprocessing.py` (hapus center-by-pelvis; normalisasi sudah dilakukan di ekstraksi; sisakan augmentasi + padding):
```python
from __future__ import annotations

import numpy as np

# Keypoint untuk augmentasi flip (tukar kiri-kanan), indeks 0-10
FLIP_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)]


def _random_scale(array: np.ndarray, low: float = 0.85, high: float = 1.15) -> np.ndarray:
    scale = np.random.uniform(low, high)
    out = array.copy()
    out[:, :33] *= scale
    return out


def _gaussian_noise(array: np.ndarray, sigma: float = 0.01) -> np.ndarray:
    out = array.copy()
    noise = np.random.normal(0, sigma, size=out[:, :33].shape).astype(out.dtype)
    out[:, :33] += noise
    return out


def _time_warp(array: np.ndarray, low: float = 0.8, high: float = 1.2) -> np.ndarray:
    T = array.shape[0]
    factor = np.random.uniform(low, high)
    new_T = max(1, int(T * factor))
    old_idx = np.linspace(0, T - 1, new_T)
    lo = np.floor(old_idx).astype(int)
    hi = np.clip(lo + 1, 0, T - 1)
    alpha = (old_idx - lo).reshape(-1, 1)
    warped = array[lo] * (1 - alpha) + array[hi] * alpha
    return warped.astype(array.dtype)


def augment_window(array: np.ndarray) -> np.ndarray:
    """array shape (window, 38)."""
    array = _random_scale(array)
    array = _gaussian_noise(array)
    array = _time_warp(array)
    return array
```

- [ ] **Step 4: Tulis src/data/dataset.py**

Timpa `src/data/dataset.py`:
```python
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.preprocessing import augment_window


def class_name_from_file(path: Path) -> str:
    return path.stem.split("_")[0]


def group_id_from_file(path: Path) -> str:
    parts = path.stem.split("_")
    return "_".join(parts[1:]) if len(parts) > 1 else path.stem


class SlidingWindowDataset(Dataset):
    def __init__(
        self,
        processed_dir: str,
        class_names: List[str],
        window_size: int = 30,
        stride: int = 15,
        augment: bool = False,
    ):
        self.window_size = window_size
        self.stride = stride
        self.augment = augment
        self.label_map: Dict[str, int] = {name: i for i, name in enumerate(class_names)}

        files = sorted(Path(processed_dir).glob("*.npy"))
        if not files:
            raise ValueError(f"No .npy files found in '{processed_dir}'")

        self.samples: List[Tuple[Path, int, int, str]] = []  # (file, start, label, group)
        self.labels: List[int] = []
        self.groups: List[str] = []

        for f in files:
            cls = class_name_from_file(f)
            if cls not in self.label_map:
                continue
            arr = np.load(f, mmap_mode="r")
            T = arr.shape[0]
            if T < window_size:
                continue
            label = self.label_map[cls]
            group = group_id_from_file(f)
            for start in range(0, T - window_size + 1, stride):
                self.samples.append((f, start, label, group))
                self.labels.append(label)
                self.groups.append(group)

        if not self.samples:
            raise ValueError("No sliding windows produced; check data / window_size.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        f, start, label, _ = self.samples[idx]
        arr = np.load(f)
        window = arr[start:start + self.window_size].astype(np.float32)
        if self.augment:
            window = augment_window(window)
        x = torch.tensor(window, dtype=torch.float32)
        y = torch.tensor(label, dtype=torch.long)
        return x, y
```

- [ ] **Step 5: Run test (lulus)**

Run: `.venv/bin/python -m pytest tests/test_dataset.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/data/preprocessing.py src/data/dataset.py tests/test_dataset.py
git commit -m "feat(data): sliding-window dataset with per-video groups"
```

---

## Task 6: Dataloader — StratifiedGroupKFold

**Files:**
- Modify: `src/data/dataloader.py`
- Test: `tests/test_dataloader.py`

**Interfaces:**
- Consumes: `SlidingWindowDataset` (Task 5), config (Task 1)
- Produces:
  - `create_fold_dataloaders(config, fold: int) -> (train_loader, val_loader, label_map)`
  - `create_fold_splits(config) -> (folds: List[Tuple[np.ndarray, np.ndarray]], label_map)` — daftar (train_idx, val_idx) per fold

- [ ] **Step 1: Tulis failing test**

Buat `tests/test_dataloader.py`:
```python
import numpy as np
from pathlib import Path
from src.config import load_config
from src.data.dataloader import create_fold_splits


def _make_dataset(tmp_path):
    rng = np.random.default_rng(0)
    for cls in ["normal", "light", "heavy"]:
        for v in range(4):  # 4 video per kelas
            arr = rng.standard_normal((90, 38)).astype(np.float32)
            np.save(Path(tmp_path) / f"{cls}_video{v}.npy", arr)


def test_folds_no_group_leak(tmp_path):
    _make_dataset(tmp_path)
    cfg = load_config("configs/default.yaml")
    cfg.paths.processed_dir = str(tmp_path)
    folds, label_map = create_fold_splits(cfg)
    assert len(folds) == 5
    ds_groups = None
    # pastikan group tidak bocor: tidak ada group sama di train & val
    from src.data.dataset import SlidingWindowDataset
    ds = SlidingWindowDataset(str(tmp_path), cfg.class_names,
                              cfg.data.window_size, cfg.data.stride)
    groups = np.array(ds.groups)
    for train_idx, val_idx in folds:
        train_groups = set(groups[train_idx])
        val_groups = set(groups[val_idx])
        assert train_groups.isdisjoint(val_groups)
```

- [ ] **Step 2: Run test (gagal)**

Run: `.venv/bin/python -m pytest tests/test_dataloader.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Tulis src/data/dataloader.py**

Timpa `src/data/dataloader.py`:
```python
from __future__ import annotations

import random
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Subset

from src.config import ProjectConfig
from src.data.dataset import SlidingWindowDataset


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def create_fold_splits(config: ProjectConfig) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], Dict[str, int]]:
    _seed_all(config.data.random_seed)
    ds = SlidingWindowDataset(
        processed_dir=config.paths.processed_dir,
        class_names=config.class_names,
        window_size=config.data.window_size,
        stride=config.data.stride,
        augment=False,
    )
    labels = np.array(ds.labels)
    groups = np.array(ds.groups)
    k = config.data.num_folds
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True,
                                random_state=config.data.random_seed)
    folds = list(sgkf.split(np.zeros(len(labels)), labels, groups))
    return folds, ds.label_map


def create_fold_dataloaders(config: ProjectConfig, fold: int):
    folds, label_map = create_fold_splits(config)
    train_idx, val_idx = folds[fold]

    train_ds = SlidingWindowDataset(
        processed_dir=config.paths.processed_dir,
        class_names=config.class_names,
        window_size=config.data.window_size,
        stride=config.data.stride,
        augment=True,
    )
    val_ds = SlidingWindowDataset(
        processed_dir=config.paths.processed_dir,
        class_names=config.class_names,
        window_size=config.data.window_size,
        stride=config.data.stride,
        augment=False,
    )

    train_loader = DataLoader(
        Subset(train_ds, train_idx.tolist()),
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
    )
    val_loader = DataLoader(
        Subset(val_ds, val_idx.tolist()),
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
    )
    return train_loader, val_loader, label_map
```

- [ ] **Step 4: Run test (lulus)**

Run: `.venv/bin/python -m pytest tests/test_dataloader.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/data/dataloader.py tests/test_dataloader.py
git commit -m "feat(data): StratifiedGroupKFold dataloaders (k=5, per-video groups)"
```

---

## Task 7: Embedding — Learnable Positional Encoding + CLS

**Files:**
- Modify: `src/models/embedding.py`
- Test: `tests/test_embedding.py`

**Interfaces:**
- Produces:
  - `JointEmbedding(input_dim, embed_dim)` — Linear projection
  - `LearnablePositionalEncoding(d_model, max_len)` — `nn.Parameter` shape `(max_len+1, d_model)`
  - `SinusoidalPositionalEncoding(d_model, max_len)` — untuk skenario ablation (buffer)

- [ ] **Step 1: Tulis failing test**

Buat `tests/test_embedding.py`:
```python
import torch
from src.models.embedding import LearnablePositionalEncoding, SinusoidalPositionalEncoding


def test_learnable_is_parameter():
    pe = LearnablePositionalEncoding(d_model=128, max_len=30)
    assert isinstance(pe.pe, torch.nn.Parameter)
    assert pe.pe.shape == (31, 128)   # max_len + 1 (CLS)


def test_learnable_forward():
    pe = LearnablePositionalEncoding(d_model=128, max_len=30)
    x = torch.randn(2, 30, 128)
    out = pe(x)
    assert out.shape == (2, 30, 128)


def test_sinusoidal_not_parameter():
    pe = SinusoidalPositionalEncoding(d_model=128, max_len=30)
    x = torch.randn(2, 30, 128)
    out = pe(x)
    assert out.shape == (2, 30, 128)
    assert not isinstance(getattr(pe, "pe", None), torch.nn.Parameter)
```

- [ ] **Step 2: Run test (gagal)**

Run: `.venv/bin/python -m pytest tests/test_embedding.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Tulis src/models/embedding.py**

Timpa `src/models/embedding.py`:
```python
import math

import torch
import torch.nn as nn


class JointEmbedding(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int):
        super().__init__()
        self.projection = nn.Linear(input_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class LearnablePositionalEncoding(nn.Module):
    """[MOD-1] Positional encoding yang dipelajari (matriks parameter (max_len+1) x d_model)."""

    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(max_len + 1, d_model))
        nn.init.normal_(self.pe, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model); tambahkan pe[:T]
        return x + self.pe[: x.size(1)].unsqueeze(0)


class SinusoidalPositionalEncoding(nn.Module):
    """Positional encoding sinusoidal standar (untuk skenario ablation)."""

    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len + 1, d_model)
        position = torch.arange(0, max_len + 1, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]
```

- [ ] **Step 4: Run test (lulus)**

Run: `.venv/bin/python -m pytest tests/test_embedding.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/models/embedding.py tests/test_embedding.py
git commit -m "feat(model): learnable + sinusoidal positional encoding [MOD-1]"
```

---

## Task 8: Transformer — CLS Token + FFN 256

**Files:**
- Modify: `src/models/transformer.py`
- Test: `tests/test_transformer.py`

**Interfaces:**
- Consumes: embedding (Task 7)
- Produces: `ActionTransformer(input_dim=38, dim_model=128, num_heads=8, num_layers=4, num_classes=3, max_len=30, ffn_dim=256, dropout=0.1, use_learnable_pe=True)`
  - `forward(x: Tensor(B, T, input_dim)) -> Tensor(B, num_classes)`

- [ ] **Step 1: Tulis failing test**

Buat `tests/test_transformer.py`:
```python
import torch
from src.models.transformer import ActionTransformer


def _model(**kw):
    base = dict(input_dim=38, dim_model=128, num_heads=8, num_layers=4,
                num_classes=3, max_len=30, ffn_dim=256, dropout=0.1)
    base.update(kw)
    return ActionTransformer(**base)


def test_output_shape():
    m = _model()
    x = torch.randn(4, 30, 38)
    out = m(x)
    assert out.shape == (4, 3)


def test_learnable_pe_flag():
    m = _model(use_learnable_pe=False)
    x = torch.randn(2, 30, 38)
    out = m(x)
    assert out.shape == (2, 3)


def test_ffn_dim():
    m = _model()
    layer = m.encoder.layers[0]
    assert layer.linear1.out_features == 256
```

- [ ] **Step 2: Run test (gagal)**

Run: `.venv/bin/python -m pytest tests/test_transformer.py -v`
Expected: FAIL (TypeError / shape mismatch)

- [ ] **Step 3: Tulis src/models/transformer.py**

Timpa `src/models/transformer.py`:
```python
import torch
import torch.nn as nn

from src.models.embedding import (
    JointEmbedding,
    LearnablePositionalEncoding,
    SinusoidalPositionalEncoding,
)


class ActionTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int = 38,
        dim_model: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        num_classes: int = 3,
        max_len: int = 30,
        ffn_dim: int = 256,
        dropout: float = 0.1,
        use_learnable_pe: bool = True,
    ):
        super().__init__()
        self.max_len = max_len
        self.embedding = JointEmbedding(input_dim=input_dim, embed_dim=dim_model)
        if use_learnable_pe:
            self.position = LearnablePositionalEncoding(d_model=dim_model, max_len=max_len)
        else:
            self.position = SinusoidalPositionalEncoding(d_model=dim_model, max_len=max_len)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim_model))
        nn.init.normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_model,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(dim_model),
            nn.Linear(dim_model, dim_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_model // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, input_dim)
        B, T, _ = x.shape
        z = self.embedding(x)                       # (B, T, d_model)
        cls = self.cls_token.expand(B, -1, -1)      # (B, 1, d_model)
        seq = torch.cat([cls, z], dim=1)            # (B, T+1, d_model)
        seq = self.position(seq)                    # pe[0..T]: idx 0 = CLS (thesis h_t = z_t + p_t)
        out = self.encoder(seq)                     # (B, T+1, d_model)
        cls_out = out[:, 0]                         # token CLS
        return self.head(cls_out)
```

> Catatan: positional encoding diterapkan setelah penggabungan CLS, sehingga token CLS memakai `pe[0]` dan token frame ke-t memakai `pe[t+1]` — sesuai thesis (`h_t = z_t + p_t`, indeks t=0 = CLS).

- [ ] **Step 4: Run test (lulus)**

Run: `.venv/bin/python -m pytest tests/test_transformer.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/models/transformer.py tests/test_transformer.py
git commit -m "feat(model): CLS token + FFN 256 transformer encoder"
```

---

## Task 9: Focal Loss

**Files:**
- Create: `src/training/losses.py`
- Test: `tests/test_losses.py`

**Interfaces:**
- Produces: `FocalLoss(gamma=2.0, alpha=(0.25, 0.40, 0.35), num_classes=3)`
  - `forward(logits: Tensor(B, C), targets: Tensor(B,)) -> Tensor(1,)`

- [ ] **Step 1: Tulis failing test**

Buat `tests/test_losses.py`:
```python
import torch
from src.training.losses import FocalLoss


def test_output_scalar():
    loss_fn = FocalLoss()
    logits = torch.randn(8, 3)
    targets = torch.tensor([0, 1, 2, 1, 0, 2, 1, 0])
    loss = loss_fn(logits, targets)
    assert loss.ndim == 0
    assert loss.item() >= 0


def test_confident_correct_lower_than_wrong():
    loss_fn = FocalLoss(gamma=2.0)
    targets = torch.tensor([0])
    correct = torch.tensor([[10.0, 0.0, 0.0]])
    wrong = torch.tensor([[0.0, 10.0, 0.0]])
    assert loss_fn(correct, targets).item() < loss_fn(wrong, targets).item()


def test_alpha_weighting():
    # kelas 2 (heavy) alpha=0.35, kelas 1 (light) alpha=0.40
    loss_fn = FocalLoss(gamma=0.0)
    logits = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    l_light = loss_fn(logits[:1], torch.tensor([1]))
    l_heavy = loss_fn(logits[1:], torch.tensor([2]))
    # alpha light > heavy -> loss light lebih besar (gamma=0)
    assert l_light.item() > l_heavy.item()
```

- [ ] **Step 2: Run test (gagal)**

Run: `.venv/bin/python -m pytest tests/test_losses.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Tulis src/training/losses.py**

Buat `src/training/losses.py`:
```python
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """[MOD-2] Multi-class Focal Loss: FL(p_t) = -alpha_t (1 - p_t)^gamma log(p_t)."""

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Sequence[float] = (0.25, 0.40, 0.35),
        num_classes: int = 3,
    ):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)          # (B, C)
        probs = log_probs.exp()
        targets = targets.long()

        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # (B,)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)          # (B,)
        alpha_t = self.alpha.to(logits.device)[targets]                # (B,)

        loss = -alpha_t * (1.0 - pt) ** self.gamma * log_pt
        return loss.mean()
```

- [ ] **Step 4: Run test (lulus)**

Run: `.venv/bin/python -m pytest tests/test_losses.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/training/losses.py tests/test_losses.py
git commit -m "feat(training): multi-class focal loss [MOD-2]"
```

---

## Task 10: Metrik Evaluasi

**Files:**
- Create: `src/evaluation/__init__.py`
- Create: `src/evaluation/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Produces:
  - `compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, class_names: List[str]) -> Dict[str, float]`
    - keys: `accuracy`, `precision_weighted`, `recall_weighted`, `f1_weighted`, `pr_auc_<class>` per kelas
  - `confusion_matrix_np(y_true, y_pred, num_classes) -> np.ndarray`

- [ ] **Step 1: Tulis failing test**

Buat `tests/test_metrics.py`:
```python
import numpy as np
from src.evaluation.metrics import compute_metrics, confusion_matrix_np


def test_metrics_keys():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 2, 2])
    y_prob = np.eye(3)[y_pred] * 0.8 + 0.1
    m = compute_metrics(y_true, y_pred, y_prob, ["normal", "light", "heavy"])
    assert "accuracy" in m
    assert "f1_weighted" in m
    assert "pr_auc_normal" in m
    assert "pr_auc_light" in m
    assert "pr_auc_heavy" in m


def test_perfect_prediction():
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 1, 2])
    y_prob = np.eye(3)
    m = compute_metrics(y_true, y_pred, y_prob, ["normal", "light", "heavy"])
    assert abs(m["accuracy"] - 1.0) < 1e-9


def test_confusion_matrix_shape():
    cm = confusion_matrix_np(np.array([0, 1]), np.array([0, 1]), 3)
    assert cm.shape == (3, 3)
```

- [ ] **Step 2: Run test (gagal)**

Run: `.venv/bin/python -m pytest tests/test_metrics.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Tulis src/evaluation/metrics.py**

Buat `src/evaluation/metrics.py`:
```python
from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    labels = list(range(num_classes))
    return confusion_matrix(y_true, y_pred, labels=labels)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
) -> Dict[str, float]:
    """
    y_prob: (N, C) probabilitas softmax.
    Mengembalikan accuracy, precision/recall/f1 (weighted), dan pr_auc per kelas (one-vs-rest).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)
    n_classes = len(class_names)

    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

    for ci in range(n_classes):
        binary_true = (y_true == ci).astype(int)
        if binary_true.sum() == 0:
            metrics[f"pr_auc_{class_names[ci]}"] = float("nan")
        else:
            metrics[f"pr_auc_{class_names[ci]}"] = float(
                average_precision_score(binary_true, y_prob[:, ci])
            )
    return metrics
```

Buat `src/evaluation/__init__.py`:
```python
from src.evaluation.metrics import compute_metrics, confusion_matrix_np

__all__ = ["compute_metrics", "confusion_matrix_np"]
```

- [ ] **Step 4: Run test (lulus)**

Run: `.venv/bin/python -m pytest tests/test_metrics.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/evaluation tests/test_metrics.py
git commit -m "feat(evaluation): metrics with weighted scores + per-class PR-AUC"
```

---

## Task 11: Checkpoint — Build Model dari Config

**Files:**
- Modify: `src/training/checkpoint.py`
- Test: `tests/test_checkpoint.py`

**Interfaces:**
- Consumes: `ActionTransformer` (Task 8), config (Task 1)
- Produces:
  - `build_model(config) -> ActionTransformer`
  - `build_model_from_checkpoint(path, device) -> (model, checkpoint)`
  - `save_checkpoint(path, model, optimizer, config, label_map, epoch, metrics)`
  - `load_checkpoint(path, map_location="cpu")`

- [ ] **Step 1: Tulis failing test**

Buat `tests/test_checkpoint.py`:
```python
import torch
from src.config import load_config
from src.training.checkpoint import build_model, save_checkpoint, build_model_from_checkpoint


def test_build_model(tmp_path):
    cfg = load_config("configs/default.yaml")
    model = build_model(cfg, num_classes=3)
    x = torch.randn(2, 30, 38)
    assert model(x).shape == (2, 3)


def test_save_load(tmp_path):
    cfg = load_config("configs/default.yaml")
    model = build_model(cfg, num_classes=3)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    path = str(tmp_path / "ckpt.pt")
    save_checkpoint(path, model, opt, cfg, {"normal": 0, "light": 1, "heavy": 2}, 0, {"val_f1": 0.5})
    m2, ck = build_model_from_checkpoint(path, torch.device("cpu"))
    assert ck["label_map"]["light"] == 1
```

- [ ] **Step 2: Run test (gagal)**

Run: `.venv/bin/python -m pytest tests/test_checkpoint.py -v`
Expected: FAIL (TypeError — build_model signature)

- [ ] **Step 3: Tulis src/training/checkpoint.py**

Timpa `src/training/checkpoint.py`:
```python
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import torch

from src.config import ProjectConfig, config_from_dict
from src.models.transformer import ActionTransformer


def build_model(config: ProjectConfig, num_classes: int | None = None) -> ActionTransformer:
    n = num_classes or config.model.num_classes
    return ActionTransformer(
        input_dim=config.model.input_dim,
        dim_model=config.model.dim_model,
        num_heads=config.model.num_heads,
        num_layers=config.model.num_layers,
        num_classes=n,
        max_len=config.data.window_size,
        ffn_dim=config.model.ffn_dim,
        dropout=config.model.dropout,
        use_learnable_pe=config.model.use_learnable_pe,
    )


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: ProjectConfig,
    label_map: Dict[str, int],
    epoch: int,
    metrics: Dict[str, float],
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config.to_dict(),
            "label_map": label_map,
            "epoch": epoch,
            "metrics": metrics,
        },
        path,
    )


def load_checkpoint(path: str, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint must contain 'model_state_dict'")
    return checkpoint


def index_to_class_from_label_map(label_map: Dict[str, int]) -> list[str]:
    output = [""] * len(label_map)
    for name, index in label_map.items():
        output[index] = name
    return output


def build_model_from_checkpoint(
    path: str, device: torch.device
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    checkpoint = load_checkpoint(path, map_location="cpu")
    config = config_from_dict(checkpoint["config"])
    label_map = checkpoint["label_map"]
    model = build_model(config=config, num_classes=len(label_map)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint
```

- [ ] **Step 4: Run test (lulus)**

Run: `.venv/bin/python -m pytest tests/test_checkpoint.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/training/checkpoint.py tests/test_checkpoint.py
git commit -m "feat(training): checkpoint uses new model config"
```

---

## Task 12: Trainer — K-Fold + Ablation + Focal Loss

**Files:**
- Modify: `src/training/trainer.py`
- Test: `tests/test_trainer.py`

**Interfaces:**
- Consumes: dataloader (Task 6), checkpoint (Task 11), losses (Task 9), metrics (Task 10), config (Task 1)
- Produces:
  - `train(config, use_learnable_pe=True, use_focal_loss=True, scenario_name="B", tag="") -> Dict[str, Any]`
    - Menjalankan K-fold, mengembalikan ringkasan metrik (mean±std) + menyimpan `results/{scenario}.csv`

- [ ] **Step 1: Tulis failing test**

Buat `tests/test_trainer.py`:
```python
import numpy as np
from pathlib import Path
from src.config import load_config
from src.training.trainer import train


def _make_dataset(tmp_path):
    rng = np.random.default_rng(0)
    for cls in ["normal", "light", "heavy"]:
        for v in range(4):
            np.save(Path(tmp_path) / f"{cls}_video{v}.npy",
                    rng.standard_normal((90, 38)).astype(np.float32))


def test_train_smoke(tmp_path):
    _make_dataset(tmp_path)
    cfg = load_config("configs/default.yaml")
    cfg.paths.processed_dir = str(tmp_path)
    cfg.paths.results_dir = str(tmp_path / "results")
    cfg.paths.checkpoint_path = str(tmp_path / "ckpt.pt")
    cfg.training.epochs = 1
    cfg.data.num_folds = 2
    summary = train(cfg, scenario_name="B", use_learnable_pe=True, use_focal_loss=True)
    assert "accuracy_mean" in summary
    assert (Path(cfg.paths.results_dir) / "B.csv").exists()
```

- [ ] **Step 2: Run test (gagal)**

Run: `.venv/bin/python -m pytest tests/test_trainer.py -v`
Expected: FAIL (TypeError — signature)

- [ ] **Step 3: Tulis src/training/trainer.py**

Timpa `src/training/trainer.py`:
```python
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from src.config import ProjectConfig
from src.data.dataloader import create_fold_dataloaders, create_fold_splits
from src.evaluation.metrics import compute_metrics
from src.training.checkpoint import build_model, save_checkpoint
from src.training.losses import FocalLoss


def logger(message: str) -> None:
    tqdm.write(message)


def _make_criterion(config: ProjectConfig, use_focal_loss: bool) -> nn.Module:
    if use_focal_loss:
        return FocalLoss(
            gamma=config.training.focal_gamma,
            alpha=config.training.focal_alpha,
            num_classes=config.model.num_classes,
        )
    return nn.CrossEntropyLoss()


def _run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_true, all_pred, all_prob = [], [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if train:
                optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            prob = torch.softmax(logits, dim=1).detach().cpu().numpy()
            all_true.append(y.detach().cpu().numpy())
            all_pred.append(prob.argmax(axis=1))
            all_prob.append(prob)
    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    y_prob = np.concatenate(all_prob)
    return total_loss / max(len(loader), 1), y_true, y_pred, y_prob


def _train_fold(config, fold, use_learnable_pe, use_focal_loss, device, class_names):
    train_loader, val_loader, label_map = create_fold_dataloaders(config, fold)
    model = build_model(config, num_classes=len(label_map)).to(device)
    criterion = _make_criterion(config, use_focal_loss)
    optimizer = optim.AdamW(model.parameters(),
                            lr=config.training.learning_rate,
                            weight_decay=config.training.weight_decay)

    best_metrics: Dict[str, float] = {}
    best_f1 = -1.0
    for epoch in range(config.training.epochs):
        _run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        _, y_true, y_pred, y_prob = _run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        m = compute_metrics(y_true, y_pred, y_prob, class_names)
        if m["f1_weighted"] > best_f1:
            best_f1 = m["f1_weighted"]
            best_metrics = m
            save_checkpoint(
                path=config.paths.checkpoint_path,
                model=model, optimizer=optimizer, config=config,
                label_map=label_map, epoch=epoch, metrics=best_metrics,
            )
    return best_metrics


def train(
    config: ProjectConfig,
    scenario_name: str = "B",
    use_learnable_pe: bool = True,
    use_focal_loss: bool = True,
) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = config.class_names
    # flag PE menentukan arsitektur model; simpan ke config agar build_model memakainya
    config.model.use_learnable_pe = use_learnable_pe
    folds, label_map = create_fold_splits(config)

    run_name = f"{scenario_name}_{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    writer = SummaryWriter(log_dir=f"{config.training.log_dir}/{run_name}")
    logger(f"Scenario {scenario_name}: folds={len(folds)}, device={device}")

    fold_metrics: List[Dict[str, float]] = []
    for fold in range(len(folds)):
        m = _train_fold(config, fold, use_learnable_pe, use_focal_loss, device, class_names)
        logger(f"[{scenario_name}] fold {fold}: " +
               ", ".join(f"{k}={v:.4f}" for k, v in m.items()))
        fold_metrics.append(m)
        for k, v in m.items():
            writer.add_scalar(f"fold_metrics/{k}", v, fold)
    writer.close()

    df = pd.DataFrame(fold_metrics)
    results_dir = Path(config.paths.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / f"{scenario_name}.csv"
    df.to_csv(csv_path, index=False)

    summary: Dict[str, Any] = {}
    for col in df.columns:
        summary[f"{col}_mean"] = float(df[col].mean())
        summary[f"{col}_std"] = float(df[col].std(ddof=0))
    summary["scenario"] = scenario_name
    summary["csv_path"] = str(csv_path)
    return summary
```

- [ ] **Step 4: Run test (lulus)**

Run: `.venv/bin/python -m pytest tests/test_trainer.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/training/trainer.py tests/test_trainer.py
git commit -m "feat(training): K-fold training with scenario flags + results CSV"
```

---

## Task 13: Skrip Ablation (4 Skenario)

**Files:**
- Create: `src/training/run_ablation.py`
- Modify: `src/training/__init__.py`

**Interfaces:**
- Consumes: `train` (Task 12)
- Produces: `run_all_scenarios(config) -> List[Dict]` menjalankan A–D dan menyimpan `results/summary.csv`

- [ ] **Step 1: Tulis src/training/run_ablation.py**

Buat `src/training/run_ablation.py`:
```python
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from src import load_config
from src.training.trainer import train

SCENARIOS = [
    {"name": "A", "use_learnable_pe": False, "use_focal_loss": False},  # vanilla
    {"name": "B", "use_learnable_pe": True,  "use_focal_loss": True},   # proposed
    {"name": "C", "use_learnable_pe": False, "use_focal_loss": True},   # ablasi PE
    {"name": "D", "use_learnable_pe": True,  "use_focal_loss": False},  # ablasi loss
]


def run_all_scenarios(config) -> List[Dict]:
    summaries = []
    for sc in SCENARIOS:
        print(f"=== Scenario {sc['name']} ===")
        summary = train(
            config,
            scenario_name=sc["name"],
            use_learnable_pe=sc["use_learnable_pe"],
            use_focal_loss=sc["use_focal_loss"],
        )
        summaries.append(summary)
    out = Path(config.paths.results_dir) / "summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(out, index=False)
    print(f"Summary saved: {out}")
    return summaries


def main() -> None:
    cfg = load_config("configs/default.yaml")
    run_all_scenarios(cfg)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Update src/training/__init__.py**

Timpa `src/training/__init__.py`:
```python
from src.training.checkpoint import (
    build_model,
    build_model_from_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from src.training.trainer import train

__all__ = [
    "train",
    "build_model",
    "save_checkpoint",
    "load_checkpoint",
    "build_model_from_checkpoint",
]
```

- [ ] **Step 3: Verifikasi import**

Run: `.venv/bin/python -c "from src.training import train; import src.training.run_ablation; print('ok')"`
Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add src/training/run_ablation.py src/training/__init__.py
git commit -m "feat(training): ablation runner for scenarios A-D"
```

---

## Task 14: Real-time — Early Warning

**Files:**
- Create: `src/realtime/__init__.py`
- Create: `src/realtime/early_warning.py`
- Test: `tests/test_early_warning.py`

**Interfaces:**
- Produces: `EarlyWarning(threshold=5, fatigue_classes=("light","heavy"))`
  - `update(label: str) -> bool` — True jika warning aktif
  - `reset() -> None`

- [ ] **Step 1: Tulis failing test**

Buat `tests/test_early_warning.py`:
```python
from src.realtime.early_warning import EarlyWarning


def test_triggers_after_5_consecutive():
    ew = EarlyWarning(threshold=5)
    results = [ew.update("light") for _ in range(5)]
    assert results[-1] is True
    assert results[0] is False


def test_resets_on_normal():
    ew = EarlyWarning(threshold=5)
    for _ in range(4):
        ew.update("heavy")
    ew.update("normal")
    assert ew.update("heavy") is False


def test_not_trigger_for_normal():
    ew = EarlyWarning(threshold=5)
    assert all(ew.update("normal") is False for _ in range(10))
```

- [ ] **Step 2: Run test (gagal)**

Run: `.venv/bin/python -m pytest tests/test_early_warning.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Tulis src/realtime/early_warning.py**

Buat `src/realtime/early_warning.py`:
```python
from __future__ import annotations

from typing import Sequence


class EarlyWarning:
    """Memicu warning setelah N prediksi kelelahan berturut-turut (thesis 2.3.5)."""

    def __init__(self, threshold: int = 5, fatigue_classes: Sequence[str] = ("light", "heavy")):
        self.threshold = threshold
        self.fatigue_classes = set(fatigue_classes)
        self.counter = 0
        self.active = False

    def update(self, label: str) -> bool:
        if label in self.fatigue_classes:
            self.counter += 1
        else:
            self.counter = 0
        self.active = self.counter >= self.threshold
        return self.active

    def reset(self) -> None:
        self.counter = 0
        self.active = False
```

Buat `src/realtime/__init__.py`:
```python
from src.realtime.early_warning import EarlyWarning

__all__ = ["EarlyWarning"]
```

- [ ] **Step 4: Run test (lulus)**

Run: `.venv/bin/python -m pytest tests/test_early_warning.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/realtime/__init__.py src/realtime/early_warning.py tests/test_early_warning.py
git commit -m "feat(realtime): early-warning counter (5 consecutive predictions)"
```

---

## Task 15: Real-time — Predictor

**Files:**
- Create: `src/realtime/predictor.py`
- Test: `tests/test_realtime_predictor.py`

**Interfaces:**
- Consumes: checkpoint (Task 11), features (Task 2/3)
- Produces: `RealtimePredictor(checkpoint_path, yolo_model_path, window_size=30, device=None)`
  - `push_frame(frame: np.ndarray) -> Dict | None` — tambah frame ke buffer, kembalikan prediksi jika buffer penuh
  - `predict_window(window: np.ndarray) -> Dict[str, object]` — `{"label": str, "probabilities": {cls: float}}`
  - atribut `.buffer` (deque maxlen=window_size)

- [ ] **Step 1: Tulis failing test**

Buat `tests/test_realtime_predictor.py`:
```python
import numpy as np
import torch
from src.config import load_config
from src.training.checkpoint import build_model, save_checkpoint
from src.realtime.predictor import RealtimePredictor


def _make_ckpt(tmp_path):
    cfg = load_config("configs/default.yaml")
    model = build_model(cfg, num_classes=3)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    path = str(tmp_path / "ckpt.pt")
    save_checkpoint(path, model, opt, cfg,
                    {"normal": 0, "light": 1, "heavy": 2}, 0, {})
    return path


def test_predict_window(tmp_path):
    ckpt = _make_ckpt(tmp_path)
    rp = RealtimePredictor(checkpoint_path=ckpt, yolo_model_path=None)
    window = np.random.randn(30, 38).astype(np.float32)
    res = rp.predict_window(window)
    assert res["label"] in ["normal", "light", "heavy"]
    assert set(res["probabilities"].keys()) == {"normal", "light", "heavy"}
```

- [ ] **Step 2: Run test (gagal)**

Run: `.venv/bin/python -m pytest tests/test_realtime_predictor.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Tulis src/realtime/predictor.py**

Buat `src/realtime/predictor.py`:
```python
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
        self.buffer.append(feat)
        if len(self.buffer) < self.window_size:
            return None
        window = np.stack(self.buffer, axis=0)       # (window_size, 38)
        return self.predict_window(window)
```

- [ ] **Step 4: Run test (lulus)**

Run: `.venv/bin/python -m pytest tests/test_realtime_predictor.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/realtime/predictor.py tests/test_realtime_predictor.py
git commit -m "feat(realtime): window predictor with frame buffer"
```

---

## Task 16: Real-time — Runner Webcam + Latensi/FPS

**Files:**
- Create: `src/realtime/run_realtime.py`
- Modify: `src/realtime/__init__.py`
- Test: `tests/test_realtime_runner.py`

**Interfaces:**
- Consumes: `RealtimePredictor` (Task 15), `EarlyWarning` (Task 14)
- Produces:
  - `draw_overlay(frame, label, confidence, warning, fps, latency_ms) -> np.ndarray`
  - `run(camera_index=0, checkpoint_path=..., yolo_model_path=..., show_motivation_video=...) -> None`
  - `LatencyMeter` — helper untuk hitung FPS & latency

- [ ] **Step 1: Tulis failing test**

Buat `tests/test_realtime_runner.py`:
```python
import numpy as np
from src.realtime.run_realtime import draw_overlay, LatencyMeter


def test_draw_overlay_shape():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    out = draw_overlay(frame, "light", 0.87, warning=True, fps=20.0, latency_ms=45.0)
    assert out.shape == (480, 640, 3)


def test_latency_meter():
    lm = LatencyMeter()
    lm.tick(0.050)
    lm.tick(0.050)
    assert abs(lm.fps - 20.0) < 1e-6
    assert abs(lm.avg_latency_ms - 50.0) < 1e-6
```

- [ ] **Step 2: Run test (gagal)**

Run: `.venv/bin/python -m pytest tests/test_realtime_runner.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Tulis src/realtime/run_realtime.py**

Buat `src/realtime/run_realtime.py`:
```python
from __future__ import annotations

import argparse
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np

from src import load_config
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


def draw_overlay(frame, label, confidence, warning, fps, latency_ms):
    color = (0, 0, 255) if warning else (0, 200, 0)
    cv2.rectangle(frame, (10, 10), (520, 130), (0, 0, 0), -1)
    cv2.putText(frame, f"Label: {label}", (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.putText(frame, f"Conf: {confidence:.2%}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    cv2.putText(frame, f"FPS: {fps:.1f}  Latency: {latency_ms:.1f}ms",
                (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    if warning:
        cv2.putText(frame, "EARLY WARNING: FATIGUE DETECTED", (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return frame


def run(camera_index: int = 0, checkpoint_path: Optional[str] = None,
        yolo_model_path: Optional[str] = None, window_size: int = 30) -> None:
    cfg = load_config("configs/default.yaml")
    ckpt = checkpoint_path or cfg.paths.checkpoint_path
    yolo = yolo_model_path or cfg.paths.yolo_model_path

    predictor = RealtimePredictor(ckpt, yolo, window_size=window_size)
    warning = EarlyWarning(threshold=5)
    meter = LatencyMeter()

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise IOError(f"Cannot open camera index {camera_index}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t0 = time.perf_counter()
            result = predictor.push_frame(frame)
            latency = time.perf_counter() - t0
            meter.tick(latency)

            if result is not None:
                label = result["label"]
                conf = result["probabilities"][label]
                warn = warning.update(label)
                frame = draw_overlay(frame, label, conf, warn, meter.fps, meter.avg_latency_ms)
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
    args = parser.parse_args()
    run(args.camera, args.checkpoint, args.yolo, args.window)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Update src/realtime/__init__.py**

Timpa `src/realtime/__init__.py`:
```python
from src.realtime.early_warning import EarlyWarning
from src.realtime.predictor import RealtimePredictor

__all__ = ["EarlyWarning", "RealtimePredictor"]
```

- [ ] **Step 5: Run test (lulus)**

Run: `.venv/bin/python -m pytest tests/test_realtime_runner.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/realtime/run_realtime.py src/realtime/__init__.py tests/test_realtime_runner.py
git commit -m "feat(realtime): webcam runner with overlay + latency/FPS meter"
```

---

## Task 17: Integrasi & Update Inference Lama

**Files:**
- Modify: `src/data/__init__.py`
- Modify: `src/inference/predictor.py`
- Modify: `src/sliding_window/sliding_window.py`
- Modify: `src/sliding_window/classify_long_video.py`
- Modify: `src/pipeline.py`

**Interfaces:**
- Consumes: semua modul sebelumnya
- Produces: pipeline end-to-end `python -m src.pipeline` (ekstraksi + train), dan inference video tetap jalan dengan dimensi 38.

- [ ] **Step 1: Update src/data/__init__.py**

Timpa `src/data/__init__.py`:
```python
from src.data.collection import process_all_videos
from src.data.dataloader import create_fold_dataloaders, create_fold_splits
from src.data.dataset import SlidingWindowDataset
from src.data.features import compute_geometric_features
from src.data.video_pose import video_to_pose_array

__all__ = [
    "SlidingWindowDataset",
    "create_fold_dataloaders",
    "create_fold_splits",
    "process_all_videos",
    "video_to_pose_array",
    "compute_geometric_features",
]
```

- [ ] **Step 2: Update src/sliding_window/sliding_window.py**

Timpa:
```python
from __future__ import annotations

import numpy as np
import numpy.typing as npt


def create_sliding_windows(
    pose_array: npt.NDArray[np.float32],
    max_len: int = 30,
    stride: int = 15,
) -> npt.NDArray[np.float32]:
    """
    pose_array: (N_frames, 38)
    max_len: panjang window
    stride: langkah geser
    return: (N_windows, max_len, 38)
    """
    n_frames = pose_array.shape[0]
    windows = []
    for start in range(0, n_frames - max_len + 1, stride):
        windows.append(pose_array[start:start + max_len])
    if not windows:
        raise ValueError(f"Video terlalu pendek: {n_frames} frame < window {max_len}")
    return np.stack(windows).astype(np.float32)
```

- [ ] **Step 3: Update src/sliding_window/classify_long_video.py**

Ganti bagian yang membaca `max_len`/`input_dim` dan `preprocess_pose_array` agar memakai window 38-dim apa adanya:
```python
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

    cover_count[cover_count == 0] = 1
    prob_avg = prob_sum / cover_count[:, None]

    frame_results = []
    for i in range(n_frames):
        probs = {name: float(prob_avg[i, ci]) for ci, name in enumerate(class_names)}
        best_label = max(probs, key=probs.get)
        frame_results.append({"label": best_label, "probabilities": probs})
    return frame_results
```

- [ ] **Step 4: Update src/inference/predictor.py**

Sesuaikan `predict_from_pose` dan `predict_from_pose_batch` agar tidak lagi memakai `preprocess_pose_array` dengan padding 51-dim, tetapi window 38-dim. Ganti isi `predict_from_pose`:
```python
@torch.no_grad()
def predict_from_pose(array: np.ndarray, checkpoint_path: str) -> Dict[str, object]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = build_model_from_checkpoint(checkpoint_path, device=device)
    model.eval()

    max_len = int(checkpoint["config"]["data"]["window_size"])
    labels = index_to_class_from_label_map(checkpoint["label_map"])

    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[0] < max_len:
        pad = np.full((max_len - arr.shape[0], arr.shape[1]), -1.0, dtype=np.float32)
        arr = np.vstack([pad, arr])
    window = arr[-max_len:]

    tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred_index = int(np.argmax(probs))
    return {
        "label": labels[pred_index],
        "index": pred_index,
        "probabilities": {labels[i]: float(probs[i]) for i in range(len(labels))},
    }
```
Dan `predict_from_pose_batch` tetap, hanya pastikan `probs` di-loop dengan indeks kelas yang benar (perbaiki bug `labels[i]` yang menutupi variabel `i`):
```python
@torch.no_grad()
def predict_from_pose_batch(pose_batch: torch.tensor, checkpoint_path: str) -> List[Dict[str, object]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = build_model_from_checkpoint(checkpoint_path, device=device)
    model.eval()

    labels = index_to_class_from_label_map(checkpoint["label_map"])
    logits = model(pose_batch.to(device))
    probs_batch = torch.softmax(logits, dim=1).cpu().numpy()
    pred_indices = np.argmax(probs_batch, axis=1)

    result = []
    for i, pred_index in enumerate(pred_indices):
        probs = probs_batch[i]
        result.append({
            "label": labels[pred_index],
            "index": int(pred_index),
            "probabilities": {labels[ci]: float(probs[ci]) for ci in range(len(labels))},
        })
    return result
```

- [ ] **Step 5: Update src/pipeline.py**

Timpa `src/pipeline.py`:
```python
from src import load_config
from src.data import process_all_videos
from src.training.trainer import train


def main():
    cfg = load_config("configs/default.yaml")
    process_all_videos(cfg, save_skeleton_video=False)
    summary = train(cfg, scenario_name="B", use_learnable_pe=True, use_focal_loss=True)
    print(summary)


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Verifikasi import seluruh paket**

Run: `.venv/bin/python -c "import src; import src.data; import src.training; import src.inference; import src.realtime; import src.sliding_window; print('ALL OK')"`
Expected: `ALL OK`

- [ ] **Step 7: Commit**

```bash
git add src/data/__init__.py src/inference/predictor.py src/sliding_window src/pipeline.py
git commit -m "feat: integrate 38-dim pipeline across inference and sliding window"
```

---

## Task 18: Ekstraksi Ulang Data + Verifikasi End-to-End

**Files:**
- Modify: `.gitignore` (opsional: pastikan `results/`, `*.npy` terabaikan)
- Test: manual E2E

- [ ] **Step 1: Hapus .npy lama (format 51-dim)**

Run:
```bash
cd /Users/mhiqrambg/Documents/Github-Repo/ML_PROJECT-Striking-Action-Recognition
rm -f data/processed/*.npy
```

- [ ] **Step 2: Jalankan ekstraksi pose ulang (format 38-dim)**

Run (background, ~30 menit di CPU):
```bash
.venv/bin/python -c "
from src import load_config
from src.data import process_all_videos
cfg = load_config('configs/default.yaml')
process_all_videos(cfg, save_skeleton_video=False)
"
```
Expected: 45 file `.npy` di `data/processed/`, tiap file shape `(T, 38)`.

- [ ] **Step 3: Verifikasi shape .npy**

Run:
```bash
.venv/bin/python -c "
import numpy as np, glob
fs = glob.glob('data/processed/*.npy')
print('jumlah:', len(fs))
a = np.load(fs[0])
print('shape:', a.shape, 'dtype:', a.dtype)
assert a.shape[1] == 38
print('OK')
"
```
Expected: `jumlah: 45`, `shape: (T, 38)`, `OK`

- [ ] **Step 4: Jalankan semua test**

Run: `.venv/bin/python -m pytest -v`
Expected: semua PASS

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore: regenerate 38-dim pose data; all tests pass"
```

---

## Task 19: Jalankan Ablation & Laporkan Hasil

**Files:**
- Create: `docs/RESULTS.md`

- [ ] **Step 1: Jalankan ablation 4 skenario**

Run:
```bash
.venv/bin/python -m src.training.run_ablation
```
Expected: `results/A.csv`, `results/B.csv`, `results/C.csv`, `results/D.csv`, `results/summary.csv`.

- [ ] **Step 2: Lihat ringkasan**

Run:
```bash
.venv/bin/python -c "import pandas as pd; print(pd.read_csv('results/summary.csv').to_string())"
```
Expected: tabel metrik per skenario.

- [ ] **Step 3: Tulis docs/RESULTS.md**

Buat `docs/RESULTS.md` berisi tabel dari `results/summary.csv` (accuracy, f1_weighted, pr_auc per kelas untuk A–D) + catatan skenario mana yang terbaik dan interpretasi singkat.

- [ ] **Step 4: Commit**

```bash
git add results docs/RESULTS.md
git commit -m "docs: ablation results for scenarios A-D"
```

---

## Self-Review (dijalankan penulis plan)

**1. Spec coverage:**
- 11 keypoint + origin bahu → Task 3 ✅
- 5 fitur geometris (rumus thesis) → Task 2 ✅
- 38-dim + sliding window W=30 S=15 → Task 3, 5 ✅
- Learnable PE [MOD-1] → Task 7 ✅
- Focal Loss [MOD-2] → Task 9 ✅
- CLS token + FFN 256 → Task 8 ✅
- K-Fold k=5 group per-video → Task 6 ✅
- Metrik + PR-AUC → Task 10 ✅
- Ablation A–D → Task 13 ✅
- Real-time early warning + latensi/FPS → Task 14, 15, 16 ✅
- Output results/runs/models → Task 12, 19 ✅

**2. Placeholder scan:** tidak ada TBD/TODO; semua langkah berisi kode nyata.

**3. Type consistency:**
- `SlidingWindowDataset(processed_dir, class_names, window_size, stride, augment)` konsisten di Task 5, 6, 11.
- `build_model(config, num_classes)` konsisten Task 11, 12.
- `create_fold_splits(config) -> (folds, label_map)` konsisten Task 6, 12.
- `compute_metrics(y_true, y_pred, y_prob, class_names)` konsisten Task 10, 12.
- `FocalLoss(gamma, alpha, num_classes)` konsisten Task 9, 12.
- `RealtimePredictor(checkpoint_path, yolo_model_path, window_size)` konsisten Task 15, 16.
- `EarlyWarning(threshold, fatigue_classes).update(label)` konsisten Task 14, 16.

**Catatan:** Task 12 `_train_fold` memiliki baris `model = build_model(...)` ganda dan blok `if model.encoder is not None: pass` yang tidak perlu — hapus agar bersih (perbaiki saat implementasi).
