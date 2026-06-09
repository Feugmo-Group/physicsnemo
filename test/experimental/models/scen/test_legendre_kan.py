# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for LegendreKAN and LegendreKANLayer.

Covers MOD-008a (constructor/attributes), MOD-008b (non-regression),
MOD-008c (checkpoint round-trip).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from physicsnemo.experimental.models.scen.legendre_kan import (
    LegendreKAN,
    LegendreKANLayer,
    legendre_basis,
    make_legendre_dleg,
    make_vandermonde,
)

_DATA_DIR = Path(__file__).parent / "data"
_GOLDEN_KAN = _DATA_DIR / "legendre_kan_v1.pth"
_SEED = 42


# ---------------------------------------------------------------------------
# Basis utilities
# ---------------------------------------------------------------------------


def test_legendre_basis_values():
    """P_0=1, P_1=x, P_2=(3x²-1)/2 at a few points."""
    x = torch.tensor([-1.0, 0.0, 1.0])
    P = legendre_basis(x, K=2)
    assert P.shape == (3, 3)
    torch.testing.assert_close(P[:, 0], torch.ones(3), atol=1e-7, rtol=0)
    torch.testing.assert_close(P[:, 1], x, atol=1e-7, rtol=0)
    p2_expected = (3 * x**2 - 1) / 2
    torch.testing.assert_close(P[:, 2], p2_expected, atol=1e-7, rtol=0)


def test_make_legendre_dleg_shape():
    D = make_legendre_dleg(4)
    assert D.shape == (5, 5)


def test_make_vandermonde_shape():
    xi = torch.linspace(-1, 1, 10)
    V = make_vandermonde(xi, K=3)
    assert V.shape == (10, 4)


# ---------------------------------------------------------------------------
# LegendreKANLayer — MOD-008a: constructor and attributes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n_in, n_out, poly_degree",
    [
        (1, 1, 2),   # minimal
        (4, 8, 4),   # wider
        (1, 16, 6),  # higher degree
    ],
)
def test_legendre_kan_layer_constructor(n_in, n_out, poly_degree):
    """LegendreKANLayer constructs and has correct parameter shape."""
    layer = LegendreKANLayer(n_in, n_out, poly_degree)
    assert layer.coeff.shape == (n_in, n_out, poly_degree + 1)
    assert layer.n_in == n_in
    assert layer.n_out == n_out
    assert layer.poly_degree == poly_degree


def test_legendre_kan_layer_forward_shape():
    layer = LegendreKANLayer(4, 8, 3)
    x = torch.randn(16, 4).clamp(-1, 1)
    out = layer(x)
    assert out.shape == (16, 8)


def test_legendre_kan_layer_invalid_input():
    layer = LegendreKANLayer(4, 8, 3)
    with pytest.raises(ValueError, match="Expected input shape"):
        layer(torch.randn(16, 5))


# ---------------------------------------------------------------------------
# LegendreKAN — MOD-008a: constructor and attributes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "hidden_dim, n_layers, poly_degree",
    [
        (8, 2, 2),    # default-like
        (16, 3, 4),   # standard
        (32, 4, 6),   # deeper
    ],
)
def test_legendre_kan_constructor(hidden_dim, n_layers, poly_degree):
    """LegendreKAN constructs with correct depth and stored hyperparams."""
    net = LegendreKAN(hidden_dim=hidden_dim, n_layers=n_layers, poly_degree=poly_degree)
    assert net.hidden_dim == hidden_dim
    assert net.n_layers == n_layers
    assert net.poly_degree == poly_degree
    assert len(net.layers) == n_layers


def test_legendre_kan_forward_shape():
    net = LegendreKAN(hidden_dim=8, n_layers=2, poly_degree=3)
    xi = torch.linspace(-1, 1, 32).unsqueeze(1)
    out = net(xi)
    assert out.shape == (32, 1)


def test_legendre_kan_invalid_input():
    net = LegendreKAN()
    with pytest.raises(ValueError, match="Expected input shape"):
        net(torch.randn(16, 2))


def test_legendre_kan_gradients_flow():
    """Backward pass must propagate gradients to coeff parameters."""
    torch.manual_seed(_SEED)
    net = LegendreKAN(hidden_dim=8, n_layers=2, poly_degree=3)
    xi = torch.linspace(-1, 1, 16).unsqueeze(1)
    loss = net(xi).sum()
    loss.backward()
    for name, p in net.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"
        assert p.grad.abs().sum() > 0, f"Zero gradient for {name}"


# ---------------------------------------------------------------------------
# MOD-008b: non-regression test
# ---------------------------------------------------------------------------


def _make_kan_for_regression():
    torch.manual_seed(_SEED)
    return LegendreKAN(hidden_dim=8, n_layers=2, poly_degree=3, dtype="float32")


def test_legendre_kan_non_regression():
    """Forward output matches committed golden fixture."""
    net = _make_kan_for_regression()
    torch.manual_seed(_SEED)
    xi = torch.linspace(-1, 1, 16).unsqueeze(1)
    out = net(xi).detach()

    if not _GOLDEN_KAN.exists():
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        torch.save({"output": out}, _GOLDEN_KAN)
        pytest.skip("Golden fixture created — re-run to validate.")

    golden = torch.load(_GOLDEN_KAN, weights_only=True)
    torch.testing.assert_close(out, golden["output"], atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# MOD-008c: checkpoint round-trip
# ---------------------------------------------------------------------------


def test_legendre_kan_checkpoint(tmp_path):
    """save() → from_checkpoint() preserves forward output."""
    torch.manual_seed(_SEED)
    net = LegendreKAN(hidden_dim=8, n_layers=2, poly_degree=3)
    xi = torch.linspace(-1, 1, 16).unsqueeze(1)
    out_before = net(xi).detach()

    ckpt_path = tmp_path / "legendre_kan.mdlus"
    net.save(str(ckpt_path))

    from physicsnemo import Module
    net_loaded = Module.from_checkpoint(str(ckpt_path))
    out_after = net_loaded(xi).detach()

    torch.testing.assert_close(out_before, out_after, atol=1e-6, rtol=1e-6)
    assert net_loaded.hidden_dim == net.hidden_dim
    assert net_loaded.n_layers == net.n_layers
    assert net_loaded.poly_degree == net.poly_degree
