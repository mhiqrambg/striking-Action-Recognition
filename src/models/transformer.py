import torch
import torch.nn as nn

from src.models.embedding import (
    JointEmbedding,
    LearnablePositionalEncoding,
    SinusoidalPositionalEncoding,
)


class ActionTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int = 38,
        dim_model: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        num_classes: int = 3,
        max_len: int = 30,
        ffn_dim: int = 256,
        dropout: float = 0.1,
        use_learnable_pe: bool = True,
    ):
        super().__init__()
        self.max_len = max_len
        self.embedding = JointEmbedding(input_dim=input_dim, embed_dim=dim_model)
        if use_learnable_pe:
            self.position = LearnablePositionalEncoding(d_model=dim_model, max_len=max_len)
        else:
            self.position = SinusoidalPositionalEncoding(d_model=dim_model, max_len=max_len)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim_model))
        nn.init.normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_model,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(dim_model),
            nn.Linear(dim_model, dim_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_model // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, input_dim)
        B, T, _ = x.shape
        z = self.embedding(x)                       # (B, T, d_model)
        cls = self.cls_token.expand(B, -1, -1)      # (B, 1, d_model)
        seq = torch.cat([cls, z], dim=1)            # (B, T+1, d_model)
        seq = self.position(seq)                    # pe[0..T]: idx 0 = CLS (thesis h_t = z_t + p_t)
        out = self.encoder(seq)                     # (B, T+1, d_model)
        cls_out = out[:, 0]                         # token CLS
        return self.head(cls_out)
