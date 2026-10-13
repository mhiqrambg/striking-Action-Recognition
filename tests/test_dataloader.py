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
