# Cara Menjalankan Project

**Sistem Klasifikasi Multi-Kelas Kelelahan** — deteksi kelelahan siswa (Normal / Kelelahan Ringan / Kelelahan Berat) dari gestur postur tubuh pada video/webcam.

Pipeline: **Pose Estimation** (YOLOv11n-Pose, 11 keypoint upper-body) → **Feature Engineering** (normalisasi origin bahu + 5 fitur geometris → vektor 38-dim/frame) → **Sliding Window** (W=30, S=15) → **Modified Transformer Encoder** (Learnable Positional Encoding + Focal Loss) → 3 kelas.

Acuan: proposal thesis "Klasifikasi Multi-Kelas Kelelahan Berbasis Pemodelan Temporal Gestur Menggunakan YOLOv11-Pose dan Modified Transformer Encoder pada Pembelajaran Online".

> **Catatan OS:** Perintah di bawah ditulis untuk **macOS/Linux** dan **Windows**. Pilih blok sesuai sistem operasi Anda. Perbedaan utama hanya pada aktivasi virtual environment dan pemanggilan Python.

---

## Prasyarat

- **Python 3.11** (JANGAN 3.14 — PyTorch belum support)
- Dependency Python: lihat `requirements.txt` (`torch`, `numpy<2`, `opencv-python`, `ultralytics`, `pyyaml`, `tqdm`, `imageio-ffmpeg`, `scikit-learn`, `tensorboard`, `torchmetrics`, `pandas`, `psutil`, `seaborn`, `pytest`)
- (Opsional) GPU NVIDIA + CUDA untuk training cepat (CPU juga jalan)

---

## 1. Setup Environment

### macOS / Linux

```bash
cd ML_PROJECT-Striking-Action-Recognition

# dengan uv (direkomendasikan)
uv venv --python 3.11 .venv
VIRTUAL_ENV=.venv uv pip install -r requirements.txt

# atau dengan venv + pip
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### Windows (PowerShell)

```powershell
cd ML_PROJECT-Striking-Action-Recognition

# dengan uv (direkomendasikan)
uv venv --python 3.11 .venv
$env:VIRTUAL_ENV = ".venv"
uv pip install -r requirements.txt

# atau dengan venv + pip (jalankan di PowerShell)
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Windows (Command Prompt / cmd)

```bat
cd ML_PROJECT-Striking-Action-Recognition
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
```

> Jika muncul error *"running scripts is disabled"* di PowerShell, jalankan sekali:
> `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`

**Cara memanggil Python setelah venv aktif:**

| macOS / Linux | Windows (venv aktif) | Windows (tanpa aktivasi) |
|---|---|---|
| `.venv/bin/python` | `python` | `.venv\Scripts\python.exe` |
| `.venv/bin/pytest` | `pytest` | `.venv\Scripts\pytest.exe` |

---

## 2. Menyiapkan Dataset

Letakkan video `.mp4` di folder per kelas (nama folder = label, **tanpa underscore**):

```
data/raw/
├── normal/*.mp4   → label "normal"   (KSS 1–4)
├── light/*.mp4    → label "light"    (Kelelahan Ringan, KSS 5–6)
└── heavy/*.mp4    → label "heavy"    (Kelelahan Berat, KSS 7–9)
```

---

## 3. Training

### Ekstraksi pose (11 keypoint → 38-dim .npy)

**macOS / Linux**
```bash
.venv/bin/python -c "
from src import load_config
from src.data import process_all_videos
cfg = load_config('configs/default.yaml')
process_all_videos(cfg, save_skeleton_video=False)
"
```

**Windows (PowerShell, venv aktif)**
```powershell
python -c "
from src import load_config
from src.data import process_all_videos
cfg = load_config('configs/default.yaml')
process_all_videos(cfg, save_skeleton_video=False)
"
```

Menghasilkan `data/processed/<kelas>_<nama>.npy` dengan shape `(T, 38)`.

### Latih model (K-Fold + ablation + model final)

**macOS / Linux**
```bash
# pipeline lengkap: ekstraksi + ablation A-D + model final (skenario B, seluruh data)
.venv/bin/python -m src.pipeline

# atau hanya ablation 4 skenario
.venv/bin/python -m src.training.run_ablation
```

**Windows (PowerShell, venv aktif)**
```powershell
# pipeline lengkap: ekstraksi + ablation A-D + model final (skenario B, seluruh data)
python -m src.pipeline

# atau hanya ablation 4 skenario
python -m src.training.run_ablation
```

- **Ablation 4 skenario** (Tabel 2.6 thesis): A=vanilla, B=proposed (Learnable PE + Focal Loss), C=ablasi PE, D=ablasi Loss.
- Hasil: `results/A.csv` … `results/D.csv`, `results/summary.csv`, confusion matrix per fold (`results/<scenario>_fold<N>_cm.csv`).
- Log TensorBoard: `runs/`.
- **Model final (deploy)**: `models/fatigue_transformer.pt` (skenario B dilatih pada seluruh data).
- Checkpoint per-fold: `models/fatigue_transformer_<scenario>_fold<N>.pt`.

### Lihat hasil training

**macOS / Linux**
```bash
tensorboard --logdir runs   # buka http://localhost:6006
.venv/bin/python -c "import pandas as pd; print(pd.read_csv('results/summary.csv').to_string())"
```

**Windows (PowerShell, venv aktif)**
```powershell
tensorboard --logdir runs   # buka http://localhost:6006
python -c "import pandas as pd; print(pd.read_csv('results/summary.csv').to_string())"
```

---

## 4. Inference

### Video panjang (sliding window + overlay)

**macOS / Linux**
```bash
.venv/bin/python -m src.sliding_window.run_long_video path/ke/video.mp4 --stride 15
```

**Windows (PowerShell, venv aktif)**
```powershell
python -m src.sliding_window.run_long_video path\ke\video.mp4 --stride 15
```

Menghasilkan `*.predictions.json` (prediksi per frame) + video overlay `*_h264.mp4`.

### Prediksi 1 file (video atau .npy)

```python
from src.inference import predict_from_file
predict_from_file("data/raw/heavy/test.mp4", "models/fatigue_transformer.pt",
                  yolo_model_path="models/yolo11n-pose.pt")
```

### Real-time webcam (early warning + latensi/FPS)

**macOS / Linux**
```bash
.venv/bin/python -m src.realtime.run_realtime --camera 0
```

**Windows (PowerShell, venv aktif)**
```powershell
python -m src.realtime.run_realtime --camera 0
```

Menampilkan overlay label + confidence, latensi (ms), FPS, dan CPU%. **Early warning** aktif setelah 5 prediksi kelelahan (light/heavy) berturut-turut.

> **Windows:** jika kamera tidak terbuka, coba `--camera 1` atau ubah backend OpenCV ke `cv2.CAP_DSHOW` bila perlu.

---

## 5. Konfigurasi (`configs/default.yaml`)

| Parameter | Nilai | Keterangan |
|-----------|-------|-----------|
| `model.input_dim` | 38 | 11 keypoint × 3 + 5 fitur geometris |
| `model.dim_model` / `num_heads` / `num_layers` / `ffn_dim` | 128 / 8 / 4 / 256 | sesuai thesis |
| `model.use_learnable_pe` / `use_focal_loss` | true | modifikasi [MOD-1] & [MOD-2] |
| `data.window_size` / `stride` | 30 / 15 | sliding window (overlap 50%) |
| `data.num_folds` | 5 | StratifiedGroupKFold |
| `training.focal_gamma` / `focal_alpha` | 2.0 / [0.25, 0.40, 0.35] | Focal Loss |
| `paths.yolo_model_path` | `models/yolo11n-pose.pt` | gesture detector |
| `paths.checkpoint_path` | `models/fatigue_transformer.pt` | model deploy |

---

## 6. Menjalankan Test

**macOS / Linux**
```bash
.venv/bin/pytest -v
```

**Windows (PowerShell, venv aktif)**
```powershell
pytest -v
```

---

## Struktur Project

```text
ML_PROJECT-Striking-Action-Recognition/
├── configs/default.yaml
├── src/
│   ├── config.py
│   ├── pipeline.py
│   ├── data/                 # features.py (5 fitur geometris), video_pose.py (11 kpt→38-dim),
│   │                         # collection.py, dataset.py (sliding window), dataloader.py (K-fold)
│   ├── models/               # embedding.py (Learnable PE), transformer.py (CLS + FFN 256)
│   ├── training/             # losses.py (Focal Loss), trainer.py, run_ablation.py, checkpoint.py
│   ├── evaluation/           # metrics.py (acc, weighted P/R/F1, PR-AUC)
│   ├── inference/            # predictor.py, file_inference.py
│   ├── sliding_window/       # sliding_window.py, classify_long_video.py, run_long_video.py, video_overlay.py
│   └── realtime/             # early_warning.py, predictor.py, run_realtime.py
├── models/                   # yolo11*-pose.pt, fatigue_transformer.pt (hasil training)
├── data/                     # raw/ (video), processed/ (.npy 38-dim)
├── results/                  # metrik ablation + confusion matrix
├── runs/                     # log TensorBoard
├── tests/                    # pytest
└── docs/                     # spec + plan + hasil
```
