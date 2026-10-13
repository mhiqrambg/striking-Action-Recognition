import math

import torch
import torch.nn as nn


class JointEmbedding(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int):
        super().__init__()
        self.projection = nn.Linear(input_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class LearnablePositionalEncoding(nn.Module):
    """[MOD-1] Positional encoding yang dipelajari (matriks parameter (max_len+1) x d_model)."""

    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(max_len + 1, d_model))
        nn.init.normal_(self.pe, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model); tambahkan pe[:T]
        return x + self.pe[: x.size(1)].unsqueeze(0)


class SinusoidalPositionalEncoding(nn.Module):
    """Positional encoding sinusoidal standar (untuk skenario ablation)."""

    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        assert d_model % 2 == 0, f"d_model must be even, got {d_model}"
        pe = torch.zeros(max_len + 1, d_model)
        position = torch.arange(0, max_len + 1, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]
