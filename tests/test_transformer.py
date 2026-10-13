import torch
from src.models.transformer import ActionTransformer


def _model(**kw):
    base = dict(input_dim=38, dim_model=128, num_heads=8, num_layers=4,
                num_classes=3, max_len=30, ffn_dim=256, dropout=0.1)
    base.update(kw)
    return ActionTransformer(**base)


def test_output_shape():
    m = _model()
    x = torch.randn(4, 30, 38)
    out = m(x)
    assert out.shape == (4, 3)


def test_learnable_pe_flag():
    m = _model(use_learnable_pe=False)
    x = torch.randn(2, 30, 38)
    out = m(x)
    assert out.shape == (2, 3)


def test_ffn_dim():
    m = _model()
    layer = m.encoder.layers[0]
    assert layer.linear1.out_features == 256
