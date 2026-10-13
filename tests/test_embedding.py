import torch
from src.models.embedding import LearnablePositionalEncoding, SinusoidalPositionalEncoding


def test_learnable_is_parameter():
    pe = LearnablePositionalEncoding(d_model=128, max_len=30)
    assert isinstance(pe.pe, torch.nn.Parameter)
    assert pe.pe.shape == (31, 128)   # max_len + 1 (CLS)


def test_learnable_forward():
    pe = LearnablePositionalEncoding(d_model=128, max_len=30)
    x = torch.randn(2, 30, 128)
    out = pe(x)
    assert out.shape == (2, 30, 128)


def test_sinusoidal_not_parameter():
    pe = SinusoidalPositionalEncoding(d_model=128, max_len=30)
    x = torch.randn(2, 30, 128)
    out = pe(x)
    assert out.shape == (2, 30, 128)
    assert not isinstance(getattr(pe, "pe", None), torch.nn.Parameter)
