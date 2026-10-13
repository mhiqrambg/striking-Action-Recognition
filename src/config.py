from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


@dataclass
class ModelConfig:
    input_dim: int = 38
    dim_model: int = 128
    num_heads: int = 8
    num_layers: int = 4
    ffn_dim: int = 256
    dropout: float = 0.1
    num_classes: int = 3
    use_learnable_pe: bool = True
    use_focal_loss: bool = True


@dataclass
class DataConfig:
    window_size: int = 30
    stride: int = 15
    batch_size: int = 16
    val_split: float = 0.2
    random_seed: int = 42
    num_workers: int = 0
    num_folds: int = 5


@dataclass
class TrainingConfig:
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    epochs: int = 50
    log_dir: str = "runs"
    focal_gamma: float = 2.0
    focal_alpha: Tuple[float, float, float] = (0.25, 0.40, 0.35)


@dataclass
class PathsConfig:
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    skeleton_dir: str = "data/with_skeleton"
    yolo_model_path: str = "models/yolo11n-pose.pt"
    checkpoint_path: str = "models/fatigue_transformer.pt"
    results_dir: str = "results"


@dataclass
class ProjectConfig:
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    paths: PathsConfig
    class_names: List[str] = field(default_factory=lambda: ["normal", "light", "heavy"])

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def config_from_dict(values: Dict[str, Any]) -> ProjectConfig:
    return ProjectConfig(
        model=ModelConfig(**values["model"]),
        data=DataConfig(**values["data"]),
        training=TrainingConfig(**values["training"]),
        paths=PathsConfig(**values["paths"]),
        class_names=list(values.get("class_names", ["normal", "light", "heavy"])),
    )


def load_config(path: str) -> ProjectConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        input_config = yaml.safe_load(file) or {}
    return config_from_dict(input_config)
