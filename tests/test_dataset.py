import numpy as np
from src.data.dataset import SlidingWindowDataset, group_id_from_file
from pathlib import Path


def _write_npy(dirpath, name, T=90):
    arr = np.random.randn(T, 38).astype(np.float32)
    np.save(Path(dirpath) / name, arr)


def test_group_id():
    p = Path("normal_video1.npy")
    # [I3] group id = stem penuh (unik per video sumber, termasuk prefix kelas)
    assert group_id_from_file(p) == "normal_video1"


def test_group_id_no_cross_class_collision():
    # [I3] normal_video1 dan light_video1 tidak boleh collapse ke group sama
    a = group_id_from_file(Path("normal_video1.npy"))
    b = group_id_from_file(Path("light_video1.npy"))
    assert a != b


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


def test_augment_preserves_window_shape(tmp_path):
    _write_npy(tmp_path, "normal_a.npy", T=90)
    ds = SlidingWindowDataset(str(tmp_path), ["normal", "light", "heavy"],
                              window_size=30, stride=15, augment=True)
    for idx in range(len(ds)):
        x, y = ds[idx]
        # augmentasi tidak boleh mengubah panjang window, kalau tidak
        # default DataLoader collate (torch.stack) akan gagal.
        assert tuple(x.shape) == (30, 38)


def test_short_video_skipped(tmp_path):
    _write_npy(tmp_path, "normal_short.npy", T=20)  # < window
    _write_npy(tmp_path, "normal_ok.npy", T=45)     # (45-30)/15+1 = 2
    ds = SlidingWindowDataset(str(tmp_path), ["normal", "light", "heavy"],
                              window_size=30, stride=15)
    assert len(ds) == 2
