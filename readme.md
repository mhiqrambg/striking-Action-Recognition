# Sistem Klasifikasi Multi-Kelas Kelelahan (Fatigue Classification)

Klasifikasi kelelahan siswa (Normal / Kelelahan Ringan / Kelelahan Berat) dari gestur postur tubuh pada pembelajaran online.

> **Catatan nama folder:** direktori repo masih bernama `ML_PROJECT-Striking-Action-Recognition`
> (nama lama dari template), tetapi seluruh isi proyek ini adalah sistem klasifikasi **kelelahan**.
> Artefak lama yang tidak relevan (web demo tinju, notebook lama) sudah dihapus.

**Pipeline:** YOLOv11n-Pose (11 keypoint upper-body) → normalisasi origin bahu + 5 fitur geometris (vektor 38-dim/frame) → sliding window (W=30, S=15) → Modified Transformer Encoder (Learnable Positional Encoding + Focal Loss) → 3 kelas.

## Fitur Utama

- **11 keypoint** COCO idx 0–10 (upper-body), origin normalisasi di **midpoint bahu**.
- **5 fitur geometris** gesture kelelahan: θ_head, d_forward, Δy_shoulder, d_wrist,nose, θ_torso.
- **Modified Transformer Encoder** [MOD-1] Learnable Positional Encoding, [MOD-2] Focal Loss (γ=2, α=[0.25,0.40,0.35]), token CLS, FFN 256.
- **Evaluasi** Group Stratified K-Fold (k=5), metrik accuracy/precision/recall/F1 (weighted) + PR-AUC per kelas.
- **Ablation study** 4 skenario (A–D).
- **Real-time** webcam dengan early warning + pengukuran latensi/FPS.

## Struktur

```text
├── configs/
│   └── default.yaml
├── src/
│   ├── config.py
│   ├── pipeline.py
│   ├── data/
│   │   ├── features.py
│   │   ├── video_pose.py
│   │   ├── collection.py
│   │   ├── preprocessing.py
│   │   ├── dataset.py
│   │   └── dataloader.py
│   ├── models/
│   │   ├── embedding.py
│   │   └── transformer.py
│   ├── training/
│   │   ├── losses.py
│   │   ├── trainer.py
│   │   ├── run_ablation.py
│   │   └── checkpoint.py
│   ├── evaluation/
│   │   └── metrics.py
│   ├── inference/
│   │   ├── predictor.py
│   │   └── file_inference.py
│   ├── sliding_window/
│   │   ├── sliding_window.py
│   │   ├── classify_long_video.py
│   │   ├── run_long_video.py
│   │   └── video_overlay.py
│   └── realtime/
│       ├── early_warning.py
│       ├── predictor.py
│       └── run_realtime.py
├── tests/
├── docs/
├── requirements.txt
└── readme.md
```

## Cara Menjalankan

Lihat **[CARA_MENJALANKAN.md](CARA_MENJALANKAN.md)** dan **[HASIL_TRAINING.md](HASIL_TRAINING.md)**.

```bash
uv venv --python 3.11 .venv
VIRTUAL_ENV=.venv uv pip install -r requirements.txt

# ekstraksi + ablation + model final
.venv/bin/python -m src.pipeline
```

## Penggunaan API

```python
from src import load_config
from src.training import train, train_final_model
from src.inference import predict_npy

cfg = load_config("configs/default.yaml")
summary = train(cfg, scenario_name="B", use_learnable_pe=True, use_focal_loss=True)
deploy_path = train_final_model(cfg)   # model final (skenario B, seluruh data)
prediction = predict_npy("data/processed/normal_sample.npy", deploy_path)
```
