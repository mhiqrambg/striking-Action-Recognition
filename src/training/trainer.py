from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from src.config import ProjectConfig
from src.data.dataloader import create_fold_dataloaders, create_fold_splits
from src.data.dataset import SlidingWindowDataset
from src.evaluation.metrics import compute_metrics, confusion_matrix_np
from src.training.checkpoint import build_model, save_checkpoint
from src.training.losses import FocalLoss


def logger(message: str) -> None:
    tqdm.write(message)


def _make_criterion(config: ProjectConfig, use_focal_loss: bool) -> nn.Module:
    if use_focal_loss:
        return FocalLoss(
            gamma=config.training.focal_gamma,
            alpha=config.training.focal_alpha,
            num_classes=config.model.num_classes,
        )
    return nn.CrossEntropyLoss()


def _run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_true, all_pred, all_prob = [], [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if train:
                optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            prob = torch.softmax(logits, dim=1).detach().cpu().numpy()
            all_true.append(y.detach().cpu().numpy())
            all_pred.append(prob.argmax(axis=1))
            all_prob.append(prob)
    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    y_prob = np.concatenate(all_prob)
    return total_loss / max(len(loader), 1), y_true, y_pred, y_prob


def _fold_checkpoint_path(config, scenario_name, fold) -> Path:
    """Checkpoint per-fold dengan path unik.

    Tidak boleh menimpa `config.paths.checkpoint_path` (yang merupakan artefak
    deploy final dari `train_final_model`) selama loop K-fold.
    """
    shared = Path(config.paths.checkpoint_path)
    return shared.with_name(f"{shared.stem}_{scenario_name}_fold{fold}.pt")


def _train_fold(config, scenario_name, fold, use_focal_loss, device, class_names):
    train_loader, val_loader, label_map = create_fold_dataloaders(config, fold)
    model = build_model(config, num_classes=len(label_map)).to(device)
    criterion = _make_criterion(config, use_focal_loss)
    optimizer = optim.AdamW(model.parameters(),
                            lr=config.training.learning_rate,
                            weight_decay=config.training.weight_decay)

    checkpoint_path = _fold_checkpoint_path(config, scenario_name, fold)
    best_metrics: Dict[str, float] = {}
    best_true = best_pred = None
    best_f1 = -1.0
    for epoch in range(config.training.epochs):
        _run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        _, y_true, y_pred, y_prob = _run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        m = compute_metrics(y_true, y_pred, y_prob, class_names)
        if m["f1_weighted"] > best_f1:
            best_f1 = m["f1_weighted"]
            best_metrics = m
            best_true, best_pred = y_true, y_pred
            save_checkpoint(
                path=str(checkpoint_path),
                model=model, optimizer=optimizer, config=config,
                label_map=label_map, epoch=epoch, metrics=best_metrics,
            )

    # [I4] Simpan confusion matrix fold (spec §6.3/§6.5) ke results_dir.
    if best_true is not None and best_pred is not None:
        cm = confusion_matrix_np(best_true, best_pred, num_classes=len(label_map))
        results_dir = Path(config.paths.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        cm_path = results_dir / f"{scenario_name}_fold{fold}_cm.csv"
        pd.DataFrame(cm).to_csv(cm_path, index=False)
        logger(f"[{scenario_name}] fold {fold}: confusion matrix -> {cm_path}")
    return best_metrics


def train(
    config: ProjectConfig,
    scenario_name: str = "B",
    use_learnable_pe: bool = True,
    use_focal_loss: bool = True,
) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = config.class_names
    # config.model.use_learnable_pe adalah sumber otoritatif yang dibaca build_model;
    # simpan flag PE di sini sebelum build_model dipanggil di _train_fold.
    config.model.use_learnable_pe = use_learnable_pe
    folds, label_map = create_fold_splits(config)

    run_name = f"{scenario_name}_{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    writer = SummaryWriter(log_dir=f"{config.training.log_dir}/{run_name}")
    logger(f"Scenario {scenario_name}: folds={len(folds)}, device={device}")

    fold_metrics: List[Dict[str, float]] = []
    for fold in range(len(folds)):
        m = _train_fold(config, scenario_name, fold, use_focal_loss, device, class_names)
        logger(f"[{scenario_name}] fold {fold}: " +
               ", ".join(f"{k}={v:.4f}" for k, v in m.items()))
        fold_metrics.append(m)
        for k, v in m.items():
            writer.add_scalar(f"fold_metrics/{k}", v, fold)
    writer.close()

    df = pd.DataFrame(fold_metrics)
    results_dir = Path(config.paths.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / f"{scenario_name}.csv"
    df.to_csv(csv_path, index=False)

    summary: Dict[str, Any] = {}
    for col in df.columns:
        summary[f"{col}_mean"] = float(df[col].mean())
        summary[f"{col}_std"] = float(df[col].std(ddof=0))
    summary["scenario"] = scenario_name
    summary["csv_path"] = str(csv_path)
    return summary


def train_final_model(
    config: ProjectConfig,
    scenario_name: str = "B",
    use_learnable_pe: bool = True,
    use_focal_loss: bool = True,
) -> str:
    """[C1] Latih model final skenario B pada SELURUH data (tanpa K-fold).

    Ini adalah artefak deploy yang disimpan ke `config.paths.checkpoint_path`.
    Berbeda dengan `train()` (validasi K-fold, checkpoint per-fold), fungsi ini
    memakai seluruh dataset sebagai train loader dan menyimpan model akhir.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # build_model membaca config.model.use_learnable_pe sebagai sumber otoritatif.
    config.model.use_learnable_pe = use_learnable_pe

    full_ds = SlidingWindowDataset(
        processed_dir=config.paths.processed_dir,
        class_names=config.class_names,
        window_size=config.data.window_size,
        stride=config.data.stride,
        augment=True,
    )
    loader = DataLoader(
        full_ds,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
    )
    label_map = full_ds.label_map

    model = build_model(config, num_classes=len(label_map)).to(device)
    criterion = _make_criterion(config, use_focal_loss)
    optimizer = optim.AdamW(model.parameters(),
                            lr=config.training.learning_rate,
                            weight_decay=config.training.weight_decay)

    logger(f"[{scenario_name}] final training on ALL data: "
           f"{len(full_ds)} windows, device={device}")
    for epoch in range(config.training.epochs):
        loss, _, _, _ = _run_epoch(model, loader, criterion, optimizer, device, train=True)
        logger(f"[{scenario_name}] final epoch {epoch + 1}/{config.training.epochs}: loss={loss:.4f}")

    save_checkpoint(
        path=config.paths.checkpoint_path,
        model=model, optimizer=optimizer, config=config,
        label_map=label_map, epoch=config.training.epochs - 1,
        metrics={"final_training": True},
    )
    logger(f"[{scenario_name}] final model saved to {config.paths.checkpoint_path}")
    return config.paths.checkpoint_path

