# Notebook Tesis

Jalankan notebook secara berurutan dari `00_index.ipynb` sampai `12_realtime_demo.ipynb`.

## Status label saat ini

`data/metadata/dataset_manifest_dummy.csv` memakai skor KSS placeholder yang konsisten dengan label folder: normal=3, light=5, heavy=8. Semua baris secara eksplisit bertanda `DUMMY_FOR_DEVELOPMENT`. File ini dipakai untuk mendemonstrasikan pipeline dan **bukan ground truth tesis final**.

Saat metadata peserta dan KSS asli tersedia, buat manifest baru dengan `subject_id`, `session_id`, waktu pengisian, dan skor KSS asli. Jangan mengganti atau menghapus penanda dummy pada manifest ini.

## Urutan

| Notebook | Tujuan |
|---|---|
| `00_index.ipynb` | Peta alur reproduksi penelitian |
| `01_environment_and_reproducibility.ipynb` | Verifikasi environment dan artefak |
| `02_data_inventory_and_labels.ipynb` | Audit video dan manifest dummy |
| `03_pose_and_feature_extraction.ipynb` | Video ke fitur pose 38 dimensi |
| `04_sequence_dataset_creation.ipynb` | Sliding window `(30, 38)` |
| `05_model_architecture.ipynb` | Transformer dan dua modifikasi |
| `06_train_A_baseline.ipynb` | Training baseline A |
| `07_train_B_proposed.ipynb` | Training model usulan B |
| `08_train_C_D_ablation.ipynb` | Training ablasi C dan D |
| `09_cross_validation_and_evaluation.ipynb` | Evaluasi fold dan confusion matrix |
| `10_results_and_hypothesis_testing.ipynb` | Komparasi hasil A-D |
| `11_realtime_benchmark.ipynb` | Benchmark latency, FPS, CPU |
| `12_realtime_demo.ipynb` | Demo inference video/webcam |

Notebook training dapat memerlukan waktu lama. Hasil yang sudah tersedia ada di `results/` dan checkpoint di `models/`.
