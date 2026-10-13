# Design Spec — Sistem Klasifikasi Multi-Kelas Kelelahan Berbasis Pemodelan Temporal Gestur

**Tanggal:** 2026-10-09
**Sumber acuan:** `PROPOSAL THESIS b5.docx` (HIKMAYANTI, D082252016 — Universitas Hasanuddin)
**Status:** Disetujui untuk implementasi

---

## 1. Ringkasan

Menyesuaikan codebase `ML_PROJECT-Striking-Action-Recognition` agar sesuai dengan proposal thesis:
sistem deteksi kelelahan siswa selama pembelajaran online, yang mengintegrasikan **YOLOv11n-Pose**
(ekstraksi gesture postur tubuh bagian atas) dengan **Modified Transformer Encoder** (klasifikasi
temporal). Tiga kelas: **Normal**, **Kelelahan Ringan (Light)**, **Kelelahan Berat (Heavy)**.

Sistem dua tahap:

```
Video / Webcam
  → YOLOv11n-Pose: ekstraksi 11 keypoint upper-body (COCO idx 0–10)
  → Feature engineering: normalisasi origin bahu + 5 fitur geometris → vektor 38-dim/frame
  → Temporal buffer: sliding window W=30, S=15 → tensor (30, 38)
  → Modified Transformer Encoder → 3 kelas
  → (real-time) overlay + early warning + video motivasi
```

---

## 2. Perbedaan Kode Sekarang vs Thesis (Gap Analysis)

| Aspek | Kode Sekarang | Target Thesis |
|-------|---------------|---------------|
| Keypoint | 17 (seluruh tubuh) | **11** (upper-body, COCO 0–10) |
| Fitur per frame | 51 (17×3) | **38** (11×3=33 + 5 geometris) |
| Origin normalisasi | midpoint pinggul (hip 11,12) | **midpoint bahu (shoulder 5,6)** |
| Fitur geometris | tidak ada | **5**: θ_head, d_forward, Δy_shoulder, d_wrist,nose, θ_torso |
| Window (max_len) | 20 | **30** |
| YOLO model | yolo11m-pose | **yolo11n-pose** |
| Positional encoding | sinusoidal (fixed) | **Learnable** [MOD-1] |
| Loss | CrossEntropy | **Focal Loss** γ=2, α=[0.25, 0.40, 0.35] [MOD-2] |
| CLS token | tidak ada | **ada** (idx t=0) |
| FFN hidden | dim_model×4 = 512 | **256** |
| Validasi | train_test_split | **Group Stratified K-Fold k=5** |
| Metrik | accuracy | Acc, Precision, Recall, F1 (weighted), **PR-AUC per kelas**, confusion matrix |
| Real-time | tidak ada | latensi <100ms, FPS ≥15, early warning 5 prediksi |

---

## 3. Arsitektur & Struktur Modul

```
src/
├── config.py                 [UBAH] window_size=30, stride=15, num_classes=3,
│                                    focal_gamma=2.0, focal_alpha=[0.25,0.40,0.35],
│                                    ffn_dim=256, num_folds=5, use_learnable_pe, use_focal_loss
├── data/
│   ├── features.py           [BARU] 5 fitur geometris (rumus dari thesis)
│   ├── video_pose.py         [UBAH] 11 keypoint, origin midpoint bahu, output (T, 38)
│   ├── collection.py         [UBAH] simpan (T, 38) per video ke .npy
│   ├── preprocessing.py      [UBAH] normalisasi origin bahu + augmentasi
│   ├── dataset.py            [UBAH] sliding window W=30 S=15, group_id per video
│   └── dataloader.py         [UBAH] StratifiedGroupKFold k=5
├── models/
│   ├── embedding.py          [UBAH] LearnablePositionalEncoding + CLS token
│   └── transformer.py        [UBAH] FFN=256, GELU, CLS-token pooling, input_dim=38
├── training/
│   ├── losses.py             [BARU] FocalLoss multi-kelas
│   ├── trainer.py            [UBAH] loop K-fold + ablation 4 skenario
│   └── checkpoint.py         [tetap, sesuaikan config]
├── evaluation/
│   └── metrics.py            [BARU] Acc, Precision, Recall, F1 (weighted), PR-AUC
└── realtime/                 [BARU]
    ├── predictor.py          inferensi per-window
    ├── early_warning.py      logika 5 prediksi berturut-turut
    └── run_realtime.py       entry point webcam + overlay + latensi/FPS
```

**Keputusan desain kunci:**
- Konfigurasi model di `config.py` memakai flag `use_learnable_pe` dan `use_focal_loss` agar
  satu arsitektur bisa mengekspresikan keempat skenario ablation (A–D) tanpa duplikasi kode.
- `features.py` terpisah dari `video_pose.py` agar rumus geometris bisa diuji independen.

---

## 4. Pipeline Data

### 4.1 Ekstraksi 11 Keypoint
YOLOv11n-Pose menghasilkan 17 keypoint COCO; hanya **idx 0–10** yang diambil:
nose (0), mata (1–2), telinga (3–4), bahu (5–6), siku (7–8), pergelangan (9–10).
Keypoint 11–16 (pinggul, lutut, kaki) dibuang (sering terhalang meja — Ding et al., 2019).

### 4.2 Normalisasi Origin
Origin = **midpoint bahu** `((x5+x6)/2, (y5+y6)/2)`. Semua koordinat dikurangi origin.

### 4.3 Vektor 38-dim per Frame
- 33 fitur: 11 keypoint × (x, y, confidence) — koordinat sudah ternormalisasi origin.
- 5 fitur geometris (rumus persis dari thesis, diekstrak dari equation OMML):

| Indeks | Fitur | Gesture | Rumus |
|--------|-------|---------|-------|
| 33 | θ_head | Head Drop | `arctan( \|y_nose − y_midShoulder\| / \|x_nose − x_midShoulder\| )` |
| 34 | d_forward | Forward Head | `x_nose − x_midShoulder` |
| 35 | Δy_shoulder | Shoulder Asymmetry | `\|y_leftShoulder − y_rightShoulder\|` |
| 36 | d_wrist,nose | Elbow on Desk | `min( \|y_wrist − y_nose\| )` |
| 37 | θ_torso | Torso Lateral Lean | `arctan( \|y_leftShoulder − y_rightShoulder\| / \|x_leftShoulder − x_rightShoulder\| )` |

> Catatan: `midShoulder` = midpoint bahu. `y_wrist` pada fitur 36 = jarak vertikal wrist
> terdekat (kiri/kanan) ke hidung; diambil nilai minimum.

### 4.4 Sliding Window
- W = 30 frame, S = 15 (overlap 50%).
- Setiap video `(T, 38)` → `N = ⌊(T − W) / S⌋ + 1` window `(30, 38)`.
- Label tiap window = kelas video; `group_id` = nama video (untuk K-Fold anti-leakage).

### 4.5 Struktur Dataset (data/raw)
```
data/raw/
├── normal/*.mp4   → label "normal"
├── light/*.mp4    → label "light"
└── heavy/*.mp4    → label "heavy"
```
Hasil ekstraksi → `data/processed/{kelas}_{nama}.npy` (shape `(T, 38)`).
Label dibaca dari `filename.split("_")[0]` (nama folder tanpa underscore).

---

## 5. Model

### 5.1 Konfigurasi
- input_dim = **38**, d_model = **128**, num_heads = **8**, num_layers = **4**
- FFN hidden = **256**, aktivasi **GELU**, dropout = 0.1
- max_len (window) = **30**, num_classes = **3**

### 5.2 Forward Pass
```
(30, 38)
  → Linear(38 → 128)                    # proyeksi embedding z_t = x_t·We + be
  → + Learnable Positional Encoding     # [MOD-1] matriks (W+1)×d_model = 31×128
  → prepend token CLS (idx 0)           # 31 token
  → 4× TransformerEncoderLayer(8 head, FFN 256, GELU)
  → ambil output token CLS
  → LayerNorm → Linear(128→64) → GELU → Dropout → Linear(64→3)
  → logits (3)
```

### 5.3 MOD-1: Learnable Positional Encoding
- `P` = `nn.Parameter` berukuran **(W+1) × d_model = 31 × 128**, inisialisasi acak.
- `h_t = z_t + p_t`; indeks `t=0` = token CLS.
- Menggantikan sinusoidal `register_buffer` pada kode lama.

### 5.4 MOD-2: Focal Loss (multi-kelas)
- `FL(p_t) = −α_t · (1 − p_t)^γ · log(p_t)`
- `γ = 2.0`; `α = [0.25 (normal), 0.40 (light), 0.35 (heavy)]`
- Menggantikan `nn.CrossEntropyLoss`.

---

## 6. Training & Evaluasi

### 6.1 Validasi
- **StratifiedGroupKFold k=5**, group = identitas video (mencegah window dari video yang sama
  tersebar di train & test). Stratified agar proporsi 3 kelas terjaga tiap fold.
- (Thesis memakai group = identitas subjek; dataset saat ini tidak punya label subjek,
  sehingga dipakai group = video sebagai pendekatan setara.)

### 6.2 Hyperparameter
- Optimizer **AdamW**, lr = 1e-4, weight decay = 0.01
- batch_size = 16, epochs = 50 (dapat disesuaikan), seed = 42
- Simpan model terbaik per fold berdasarkan metrik validasi.

### 6.3 Metrik (Tabel 2.5)
- Accuracy; Precision / Recall / F1-Score **weighted average** (sklearn)
- **PR-AUC per kelas** (metrik utama, fokus kelas Light) — one-vs-rest
- Confusion matrix 3×3
- Uji signifikansi: paired t-test / Wilcoxon signed-rank (α = 0.05) antar skenario

### 6.4 Ablation Study (Tabel 2.6) — 4 skenario
| Skenario | Positional Encoding | Loss |
|----------|--------------------|------|
| A (Baseline Vanilla) | Sinusoidal | Cross-Entropy |
| B (Proposed) | Learnable [MOD-1] | Focal Loss [MOD-2] |
| C (Ablasi PE) | Sinusoidal | Focal Loss |
| D (Ablasi Loss) | Learnable | Cross-Entropy |

> Catatan: bagian 2.7 thesis menyebut "skenario A–I" dan "skenario E", tetapi Tabel 2.6
> hanya mendefinisikan A–D. Ini dianggap typo dokumen; implementasi mengikuti Tabel 2.6 (A–D).

### 6.5 Output
- `runs/<timestamp>/` — log TensorBoard (loss, accuracy, confusion matrix)
- `results/` — CSV metrik per fold & per skenario, confusion matrix, kurva PR-AUC
- `models/` — checkpoint (model final: skenario B dilatih di seluruh data)

---

## 7. Sistem Real-Time

### 7.1 Alur
```
webcam 720p 30fps (OpenCV)
  → YOLOv11n-Pose per frame → 11 keypoint → 5 fitur geometris → vektor 38-dim
  → temporal buffer (deque 30 frame) → tensor (30, 38)
  → Transformer (skenario B) → 3 kelas + confidence
  → overlay: label + confidence
  → early warning: Light/Heavy ≥5 prediksi berturut-turut → tampilkan video motivasi
  → ukur latensi (perf_counter) & FPS + utilisasi CPU (psutil)
```

### 7.2 Komponen
- `realtime/predictor.py` — inferensi satu window
- `realtime/early_warning.py` — counter 5 prediksi berturut-turut
- `realtime/run_realtime.py` — loop webcam, overlay, metrik latensi/FPS

### 7.3 Target (Tabel 2.5)
- Latensi inferensi end-to-end < **100 ms**
- FPS ≥ **15**
- Utilisasi CPU < **80%**

---

## 8. Kriteria Sukses

1. Pipeline data menghasilkan `.npy` shape `(T, 38)` dengan 11 keypoint + 5 fitur geometris.
2. Model Transformer (skenario B) dapat dilatih dan dievaluasi dengan Group Stratified K-Fold k=5.
3. Empat skenario ablation (A–D) dapat dijalankan dan hasilnya dibandingkan (tabel + grafik).
4. Metrik lengkap (Accuracy, Precision, Recall, F1 weighted, PR-AUC per kelas) terlaporkan.
5. Skrip real-time berjalan pada webcam, menampilkan overlay + early warning, dan melaporkan latensi/FPS.

---

## 9. Di Luar Scope (untuk saat ini)

- Ground truth KSS aktual dari 50 subjek (dataset saat ini hanya 45 video berlabel folder).
- Optimasi threshold gesture via grid search + kurva ROC (thesis 2.3.4) — dapat menyusul.
- Video motivasi final (dipakai placeholder dulu).
