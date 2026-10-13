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
    """[I3] Group id unik per video sumber.

    Memakai STEM penuh (termasuk prefix kelas) supaya `normal_video1` dan
    `light_video1` tidak dianggap satu group yang sama -> mencegah kebocoran
    lintas kelas pada pembagian fold.
    """
    return path.stem


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
