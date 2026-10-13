# Hasil Ablation Study — Klasifikasi Kelelahan 3 Kelas

**Tanggal:** 2026-10-09
**Dataset:** 45 video (15 `normal`, 15 `light`, 15 `heavy`) → 853 sliding-window sampel `(30, 38)`
**Validasi:** StratifiedGroupKFold k=5 (group = per-video)
**Device:** CPU
**Konfigurasi:** d_model=128, heads=8, layers=4, ffn=256, window=30, stride=15, 50 epoch, AdamW lr=1e-4

---

## 1. Skenario (Tabel 2.6 thesis)

| Skenario | Positional Encoding | Loss |
|----------|--------------------|------|
| **A** (Baseline Vanilla) | Sinusoidal | Cross-Entropy |
| **B** (Proposed) | Learnable [MOD-1] | Focal Loss [MOD-2] |
| **C** (Ablasi PE) | Sinusoidal | Focal Loss |
| **D** (Ablasi Loss) | Learnable | Cross-Entropy |

---

## 2. Ringkasan Metrik (mean across 5 fold)

| Skenario | Accuracy | F1 (weighted) | PR-AUC normal | PR-AUC light | PR-AUC heavy |
|----------|:--------:|:-------------:|:-------------:|:------------:|:------------:|
| **A** (Vanilla) | 0.8469 | 0.8441 | 0.9009 | 0.8909 | 0.8480 |
| **B** (Proposed) | 0.8480 | 0.8453 | 0.9427 | 0.8680 | 0.8237 |
| **C** (Ablasi PE) | 0.8445 | 0.8402 | 0.9176 | 0.8582 | 0.8554 |
| **D** (Ablasi Loss) | **0.8658** | **0.8621** | 0.9108 | 0.8735 | 0.8299 |

> Sumber lengkap: `results/summary.csv`, metrik per fold: `results/<scenario>.csv`, confusion matrix: `results/<scenario>_fold<N>_cm.csv`.

---

## 3. Observasi

1. **Keempat skenario berperforma setara** (accuracy 0.845–0.866, F1 0.840–0.862). Selisih antar-skenario kecil (< 2.5 poin), menunjukkan bahwa dengan dataset kecil ini, kedua modifikasi (Learnable PE & Focal Loss) belum menghasilkan keunggulan yang tegas dibanding baseline.

2. **Skenario D (Learnable PE + Cross-Entropy) mencapai accuracy & F1 tertinggi** (0.8658 / 0.8621). Ini menyiratkan Learnable PE memberi kontribusi sedikit positif pada akurasi, sementara Focal Loss (pada data ini yang relatif seimbang 15/15/15) belum memberi tambahan.

3. **Skenario B (Proposed) unggul pada PR-AUC kelas `normal`** (0.9427, tertinggi) — kelas mayoritas terdeteksi paling baik.

4. **Kelas `light` paling sulit** (PR-AUC terendah di B: 0.8680). Ini wajar — kelelahan ringan adalah kondisi paling ambigu secara visual (transisi normal→lelah), sesuai ekspektasi thesis bahwa kelas ini adalah titik deteksi dini yang menantang.

5. **Variabilitas antar-fold tinggi** (mis. skenario B: fold 1 accuracy 0.9941 vs fold 3 accuracy 0.7456). Dengan hanya 9 video per fold, satu video yang sulit berdampak besar — indikasi dataset terlalu kecil untuk hasil stabil.

---

## 4. Interpretasi & Keterbatasan

- **Focal Loss** dirancang untuk data **tidak seimbang** (thesis: kelas normal jauh lebih banyak). Dataset saat ini **seimbang** (15/15/15), sehingga keunggulan Focal Loss belum terlihat. Bila dataset diperbesar dengan proporsi alami (normal dominan), Focal Loss diharapkan lebih unggul.
- **Ukuran dataset** (45 video) jauh di bawah rancangan thesis (50 subjek × 2 sesi). Variabilitas antar-fold yang tinggi dan selisih skenario yang tipis adalah konsekuensi langsung dari keterbatasan ini.
- **Confusion matrix** per fold tersedia di `results/` untuk analisis pola kesalahan (khususnya kelas `light`).

---

## 5. Model Deploy

Model final = **skenario B (Proposed: Learnable PE + Focal Loss)** dilatih pada **seluruh data** (853 window), disimpan di `models/fatigue_transformer.pt`.

Smoke test inference (sampel data):

| Input | Prediksi | Probabilitas (normal/light/heavy) |
|-------|----------|-----------------------------------|
| `normal_*.npy` | normal | 0.97 / 0.01 / 0.02 |
| `light_*.npy` | light | 0.05 / 0.94 / 0.00 |
| `heavy_*.npy` | heavy | 0.01 / 0.01 / 0.99 |

---

## 6. Reproduksi

```bash
# ablation 4 skenario
.venv/bin/python -m src.training.run_ablation

# model deploy final
.venv/bin/python -c "from src import load_config; from src.training import train_final_model; train_final_model(load_config('configs/default.yaml'))"
```
