from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from src.training.checkpoint import build_model_from_checkpoint, index_to_class_from_label_map


@torch.no_grad()
def predict_from_pose(array: np.ndarray, checkpoint_path: str) -> Dict[str, object]:
    """
    NumPy array на входе
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = build_model_from_checkpoint(checkpoint_path, device=device)
    model.eval()

    max_len = int(checkpoint["config"]["data"]["window_size"])
    input_dim = int(checkpoint["config"]["model"]["input_dim"])
    labels = index_to_class_from_label_map(checkpoint["label_map"])

    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] != input_dim:
        raise ValueError(f"Expected {input_dim} features per frame, got {arr.shape[1]}")
    if arr.shape[0] < max_len:
        pad = np.full((max_len - arr.shape[0], arr.shape[1]), -1.0, dtype=np.float32)
        arr = np.vstack([pad, arr])
    window = arr[-max_len:]

    tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred_index = int(np.argmax(probs))
    return {
        "label": labels[pred_index],
        "index": pred_index,
        "probabilities": {labels[i]: float(probs[i]) for i in range(len(labels))},
    }


@torch.no_grad()
def predict_from_pose_batch(pose_batch: torch.tensor, checkpoint_path: str) -> List[Dict[str, object]]:
    """
    готовый тензор батч на входе
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = build_model_from_checkpoint(checkpoint_path, device=device)
    model.eval()

    labels = index_to_class_from_label_map(checkpoint["label_map"])
    logits = model(pose_batch.to(device)) # [B, Num_Classes]

    probs_batch = torch.softmax(logits, dim=1).cpu().numpy() # [B, Num_Classes]
    pred_indices = np.argmax(probs_batch, axis=1) # [B]
    

    result = []
    for i, pred_index in enumerate(pred_indices):
        probs = probs_batch[i]

        result.append({
        "label": labels[pred_index],
        "index": int(pred_index),
        "probabilities": {labels[ci]: float(probs[ci]) for ci in range(len(labels))},
    })
    return result


def predict_from_npy(npy_path: str, checkpoint_path: str) -> Dict[str, object]:
    """
    Путь к NumPy array на входе
    """
    array = np.load(npy_path)
    result = predict_from_pose(array=array, checkpoint_path=checkpoint_path)
    result["source"] = npy_path
    return result


def predict_batch_from_folder(folder_path: str, checkpoint_path: str) -> List[Dict[str, object]]:
    """
    Путь к папке с NumPy arrays на входе
    """
    files = sorted(Path(folder_path).glob("*.npy"))
    return [predict_from_npy(str(path), checkpoint_path=checkpoint_path) for path in files]
