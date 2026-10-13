# Hasil Training & Inference

Dokumen ini menjelaskan output setelah melatih model klasifikasi kelelahan (3 kelas: `normal` / `light` / `heavy`) sesuai thesis.

---

## 1. Output Saat Training (terminal)

Ablation menjalankan 4 skenario × 5 fold. Tiap fold melaporkan metrik ke terminal:

```
Scenario A: folds=5, device=cpu
[A] fold 0: accuracy=0.8..., precision_weighted=..., recall_weighted=..., f1_weighted=..., pr_auc_normal=..., pr_auc_light=..., pr_auc_heavy=...
...
```

Setelah semua skenario selesai:
```
Summary saved: results/summary.csv
```

---

## 2. File yang Dihasilkan

| Lokasi | Isi |
|--------|-----|
| `models/fatigue_transformer.pt` | **Model deploy** (skenario B, dilatih pada seluruh data) |
| `models/fatigue_transformer_<scenario>_fold<N>.pt` | Checkpoint per-fold per-skenario |
| `results/<scenario>.csv` | Metrik per fold untuk tiap skenario (A–D) |
| `results/<scenario>_fold<N>_cm.csv` | Confusion matrix 3×3 per fold |
| `results/summary.csv` | Ringkasan mean±std semua skenario |
| `runs/<scenario>_<timestamp>/` | Log TensorBoard |
| `data/processed/*.npy` | Array pose `(T, 38)` per video |

### Isi file checkpoint `*.pt`
```python
{
  "model_state_dict":     # bobot model
  "optimizer_state_dict": # state optimizer
  "config":               # config saat training (schema baru: data.window_size)
  "label_map":            # {"normal":0, "light":1, "heavy":2}
  "epoch":                # epoch terbaik
  "metrics":              # metrik validasi terbaik
}
```

---

## 3. Visualisasi TensorBoard

```bash
tensorboard --logdir runs   # http://localhost:6006
```

---

## 4. Metrik Evaluasi

Sesuai Tabel 2.5 thesis:

| Metrik | Keterangan |
|--------|-----------|
| Accuracy | akurasi keseluruhan |
| Precision / Recall / F1 (weighted) | rata-rata berbobot per kelas |
| **PR-AUC per kelas** | metrik utama (data imbalance), fokus kelas `light` |
| Confusion Matrix | 3×3 per fold |

---

## 5. Ablation Study (Tabel 2.6)

| Skenario | Positional Encoding | Loss | Tujuan |
|----------|--------------------|------|--------|
| **A** (Baseline Vanilla) | Sinusoidal | Cross-Entropy | baseline pembanding |
| **B** (Proposed) | Learnable [MOD-1] | Focal Loss [MOD-2] | model usulan |
| **C** (Ablasi PE) | Sinusoidal | Focal Loss | isolasi kontribusi Learnable PE |
| **D** (Ablasi Loss) | Learnable | Cross-Entropy | isolasi kontribusi Focal Loss |

---

## 6. Hasil Inference

### Prediksi 1 file → JSON
```python
from src.inference import predict_from_file
predict_from_file("data/raw/heavy/test.mp4", "models/fatigue_transformer.pt",
                  yolo_model_path="models/yolo11n-pose.pt")
```
Output:
```json
{
  "label": "heavy",
  "index": 2,
  "probabilities": {"normal": 0.04, "light": 0.09, "heavy": 0.87},
  "source": "data/raw/heavy/test.mp4"
}
```

### Video panjang → overlay + JSON
```bash
.venv/bin/python -m src.sliding_window.run_long_video data/raw/normal/test.mp4 --stride 15
```
Menghasilkan `test.predictions.json` (per frame) + `*_h264.mp4` (overlay label + confidence).

### Real-time webcam
```bash
.venv/bin/python -m src.realtime.run_realtime --camera 0
```
Menampilkan overlay + latensi/FPS/CPU%. **Early warning** muncul setelah 5 prediksi kelelahan berturut-turut.

---

## 7. Catatan

1. **Kualitas hasil bergantung pada data** — gesture normal/light/heavy yang bedanya halus → akurasi bisa rendah, confusion matrix menunjukkan kelas tertukar.
2. **`use_learnable_pe` & `use_focal_loss`** menentukan skenario ablation; keduanya benar-benar mengubah arsitektur/loss.
3. **Model deploy = skenario B** (Learnable PE + Focal Loss) dilatih pada seluruh data, bukan hasil salah satu fold.
4. Dataset saat ini 45 video (15/kelas); thesis merancang dataset jauh lebih besar (50 subjek). Untuk hasil optimal, tambah data.
