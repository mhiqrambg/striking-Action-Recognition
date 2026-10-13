import torch
from src.training.losses import FocalLoss


def test_output_scalar():
    loss_fn = FocalLoss()
    logits = torch.randn(8, 3)
    targets = torch.tensor([0, 1, 2, 1, 0, 2, 1, 0])
    loss = loss_fn(logits, targets)
    assert loss.ndim == 0
    assert loss.item() >= 0


def test_confident_correct_lower_than_wrong():
    loss_fn = FocalLoss(gamma=2.0)
    targets = torch.tensor([0])
    correct = torch.tensor([[10.0, 0.0, 0.0]])
    wrong = torch.tensor([[0.0, 10.0, 0.0]])
    assert loss_fn(correct, targets).item() < loss_fn(wrong, targets).item()


def test_alpha_weighting():
    # kelas 2 (heavy) alpha=0.35, kelas 1 (light) alpha=0.40
    loss_fn = FocalLoss(gamma=0.0)
    logits = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    l_light = loss_fn(logits[:1], torch.tensor([1]))
    l_heavy = loss_fn(logits[1:], torch.tensor([2]))
    # alpha light > heavy -> loss light lebih besar (gamma=0)
    assert l_light.item() > l_heavy.item()
