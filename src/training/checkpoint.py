from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import torch

from src.config import ProjectConfig, config_from_dict
from src.models.transformer import ActionTransformer


def build_model(config: ProjectConfig, num_classes: int | None = None) -> ActionTransformer:
    n = num_classes or config.model.num_classes
    return ActionTransformer(
        input_dim=config.model.input_dim,
        dim_model=config.model.dim_model,
        num_heads=config.model.num_heads,
        num_layers=config.model.num_layers,
        num_classes=n,
        max_len=config.data.window_size,
        ffn_dim=config.model.ffn_dim,
        dropout=config.model.dropout,
        use_learnable_pe=config.model.use_learnable_pe,
    )


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: ProjectConfig,
    label_map: Dict[str, int],
    epoch: int,
    metrics: Dict[str, float],
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config.to_dict(),
            "label_map": label_map,
            "epoch": epoch,
            "metrics": metrics,
        },
        path,
    )


def load_checkpoint(path: str, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint must contain 'model_state_dict'")
    return checkpoint


def index_to_class_from_label_map(label_map: Dict[str, int]) -> list[str]:
    output = [""] * len(label_map)
    for name, index in label_map.items():
        output[index] = name
    return output


def build_model_from_checkpoint(
    path: str, device: torch.device
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    checkpoint = load_checkpoint(path, map_location="cpu")
    config = config_from_dict(checkpoint["config"])
    label_map = checkpoint["label_map"]
    model = build_model(config=config, num_classes=len(label_map)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint
