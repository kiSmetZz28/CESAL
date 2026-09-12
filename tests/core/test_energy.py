import numpy as np
import pytest
import torch
import torch.nn as nn

from cesal_core.utils.energy import compute_energy_batch, my_kl_loss


class _TinyModel(nn.Module):
    """Minimal EM-AT stub: returns (output, [series], [prior], None)."""

    def __init__(self, win_size: int, n_features: int):
        super().__init__()
        self.win_size = win_size
        self.n_features = n_features
        self.linear = nn.Linear(n_features, n_features)

    def forward(self, x):
        B, W, F = x.shape
        output = self.linear(x)
        attn = torch.softmax(torch.randn(B, 1, W, W), dim=-1)
        return output, [attn], [attn], None


@pytest.fixture
def tiny_model():
    m = _TinyModel(win_size=10, n_features=4)
    m.eval()
    return m


def test_my_kl_loss_nonnegative():
    p = torch.softmax(torch.randn(2, 3, 5), dim=-1)
    q = torch.softmax(torch.randn(2, 3, 5), dim=-1)
    loss = my_kl_loss(p, q)
    assert (loss >= 0).all()


def test_my_kl_loss_shape():
    p = torch.softmax(torch.randn(4, 3, 8), dim=-1)
    q = torch.softmax(torch.randn(4, 3, 8), dim=-1)
    loss = my_kl_loss(p, q)
    assert loss.shape == (4,)


def test_compute_energy_batch_shape(tiny_model):
    # Energy is per time step, not per window: [B, win_size]. The pipeline
    # flattens it per line (lad_bat_cloud) or averages it per window (bat_predict).
    x = torch.randn(6, 10, 4)
    energy = compute_energy_batch(tiny_model, x, win_size=10)
    assert energy.shape == (6, 10)


def test_compute_energy_batch_nonnegative(tiny_model):
    x = torch.randn(3, 10, 4)
    energy = compute_energy_batch(tiny_model, x, win_size=10)
    assert (energy >= 0).all()


def test_compute_energy_batch_dtype(tiny_model):
    x = torch.randn(2, 10, 4)
    energy = compute_energy_batch(tiny_model, x, win_size=10)
    assert isinstance(energy, np.ndarray)
