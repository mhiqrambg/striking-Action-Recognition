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
