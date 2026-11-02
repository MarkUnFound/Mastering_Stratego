"""SpatialAttention in RainbowDQN transforms its input, mixes positions and passes gradients.

(Converted from the check_spatial_attention.py print script; same five checks.)
"""

import math

import pytest
import torch
import torch.nn.functional as F

from networks.rainbow_dqn import SpatialAttention

C = 64


@pytest.fixture
def attn():
    torch.manual_seed(0)
    layer = SpatialAttention(channels=C, num_heads=4, dropout=0.0)
    layer.eval()
    return layer


def _weights(layer, x):
    flat = x.flatten(2).permute(0, 2, 1)
    with torch.no_grad():
        _, w = layer.attn(flat, flat, flat, need_weights=True)
    return w[0]


def test_output_is_not_the_input(attn):
    x = torch.randn(1, C, 10, 10)
    with torch.no_grad():
        y = attn(x)
    assert y.shape == x.shape
    assert (y - x).abs().mean() > 0.01
    assert F.cosine_similarity(x.flatten()[None], y.flatten()[None]).item() < 0.999


def test_attention_weights_are_neither_collapsed_nor_one_hot(attn):
    w = _weights(attn, torch.randn(1, C, 10, 10))
    assert w.shape == (100, 100)
    assert torch.allclose(w.sum(-1), torch.ones(100), atol=1e-5)
    entropy = -(w * (w + 1e-10).log()).sum(-1).mean().item()
    assert entropy > 0.1 * math.log(100)
    assert w.std() > 1e-8
    assert w.max(-1).values.mean() < 0.5


def test_positions_get_different_outputs(attn):
    with torch.no_grad():
        y = attn(torch.randn(1, C, 10, 10))
    vecs = y[0].flatten(1).T  # (100, C)
    picks = [0, 9, 90, 99, 44]
    sims = [F.cosine_similarity(vecs[a][None], vecs[b][None]).item()
            for i, a in enumerate(picks) for b in picks[i + 1:]]
    assert sum(sims) / len(sims) < 0.99
    assert vecs.var(0).mean() > 1e-4


def test_gradients_reach_input_and_attention_parameters():
    torch.manual_seed(0)
    layer = SpatialAttention(channels=C, num_heads=4, dropout=0.0).train()
    x = torch.randn(1, C, 10, 10, requires_grad=True)
    layer(x).sum().backward()
    assert x.grad is not None and x.grad.norm() > 0
    assert layer.attn.in_proj_weight.grad.norm() > 0
    assert layer.attn.out_proj.weight.grad.norm() > 0


def test_attention_responds_to_board_content(attn):
    torch.manual_seed(1)
    x_uniform = torch.randn(1, C, 10, 10) * 0.1
    x_cluster = x_uniform.clone()
    x_cluster[:, :, 0:3, 7:10] = torch.randn(1, C, 3, 3) * 5.0
    with torch.no_grad():
        out_diff = (attn(x_cluster) - attn(x_uniform)).abs().mean()
    w_diff = (_weights(attn, x_cluster) - _weights(attn, x_uniform)).abs().mean()
    assert w_diff > 1e-3
    assert out_diff > 0.01
