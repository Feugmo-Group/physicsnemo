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

"""Tests for DVRMapper, DVRMapper2D, DVRMapper3D.

Covers MOD-008a (constructor/attributes) and spectral accuracy.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from physicsnemo.experimental.models.scen.dvr_mapper import DVRMapper
from physicsnemo.experimental.models.scen.dvr_mapper_2d import DVRMapper2D
from physicsnemo.experimental.models.scen.dvr_mapper_3d import DVRMapper3D
from physicsnemo.experimental.models.scen.mortar import compute_mortar_projection
from physicsnemo.experimental.models.scen.quadrature import get_quadrature_data

_DATA_DIR = Path(__file__).parent / "data"
_DTYPE = torch.float64  # spectral accuracy tests require float64


# ---------------------------------------------------------------------------
# get_quadrature_data
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rule", ["lgl", "chebyshev", "clenshaw_curtis"])
@pytest.mark.parametrize("N", [4, 8, 16])
def test_quadrature_weight_sum(rule, N):
    """Quadrature weights must sum to 2 on [-1, 1]."""
    _, w, _ = get_quadrature_data(N, rule=rule, dtype=_DTYPE)
    assert abs(w.sum().item() - 2.0) < 1e-12, f"Weight sum={w.sum().item()} for {rule}, N={N}"


@pytest.mark.parametrize("rule", ["lgl", "chebyshev", "clenshaw_curtis"])
def test_quadrature_ascending_nodes(rule):
    """Nodes must be in strictly ascending order."""
    xi, _, _ = get_quadrature_data(8, rule=rule, dtype=_DTYPE)
    diffs = xi[1:] - xi[:-1]
    assert (diffs > 0).all(), f"Nodes not ascending for {rule}"


def test_quadrature_invalid_rule():
    with pytest.raises(ValueError, match="rule must be one of"):
        get_quadrature_data(8, rule="invalid")


# ---------------------------------------------------------------------------
# DVRMapper — MOD-008a: constructor and attributes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "N, a, b, alpha, mapping",
    [
        (8, -1.0, 1.0, 0.0, "kte"),   # uniform
        (16, 0.0, 1.0, 0.5, "kte"),   # KTE-stretched
        (12, 0.0, 1.0, 1.5, "log"),   # log-compressed
    ],
)
def test_dvr_mapper_constructor(N, a, b, alpha, mapping):
    """DVRMapper constructs successfully and exposes expected attributes."""
    m = DVRMapper(N, a, b, alpha, mapping=mapping, dtype=_DTYPE)
    assert m.nodes.shape == (N,)
    assert m.weights.shape == (N,)
    assert m.D1.shape == (N, N)
    assert m.D2.shape == (N, N)
    assert m.D4.shape == (N, N)


def test_dvr_mapper_invalid_mapping():
    with pytest.raises(ValueError, match="mapping must be one of"):
        DVRMapper(8, 0.0, 1.0, mapping="invalid")


# ---------------------------------------------------------------------------
# DVRMapper — spectral derivative accuracy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "N, a, b, alpha, mapping",
    [
        (8, -1.0, 1.0, 0.0, "kte"),
        (16, 0.0, 1.0, 0.5, "kte"),
        (12, 0.0, 1.0, 1.5, "log"),
    ],
)
def test_dvr_mapper_d1_accuracy(N, a, b, alpha, mapping):
    """D1 @ f ≈ f' at LGL nodes: test with f = sin(πx)."""
    m = DVRMapper(N, a, b, alpha, mapping=mapping, dtype=_DTYPE)
    x = m.nodes
    f = torch.sin(math.pi * x)
    fp_exact = math.pi * torch.cos(math.pi * x)
    fp_num = m.D1 @ f
    err = (fp_num - fp_exact).abs().max().item()
    # Interior accuracy is spectral; endpoints may have O(1) roundoff for N=8
    assert err < 1e-6, f"D1 max error={err:.2e} for N={N}, mapping={mapping}"


@pytest.mark.parametrize("N", [8, 12, 16])
def test_dvr_mapper_d2_accuracy_uniform(N):
    """D2 @ sin(πx) ≈ -π² sin(πx) on a uniform grid to spectral accuracy."""
    m = DVRMapper(N, -1.0, 1.0, alpha=0.0, dtype=_DTYPE)
    x = m.nodes
    f = torch.sin(math.pi * x)
    fpp_exact = -(math.pi**2) * torch.sin(math.pi * x)
    fpp_num = m.D2 @ f
    err = (fpp_num - fpp_exact).abs().max().item()
    assert err < 1e-8, f"D2 max error={err:.2e} for N={N}"


def test_dvr_mapper_weight_sum():
    """Physical weights must sum to domain length (b - a)."""
    for a, b in [(-1.0, 1.0), (0.0, 5.0), (-3.0, 2.0)]:
        m = DVRMapper(12, a, b, dtype=_DTYPE)
        assert abs(m.weights.sum().item() - (b - a)) < 1e-12


# ---------------------------------------------------------------------------
# DVRMapper2D
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "Nx, Ny",
    [(4, 4), (6, 8)],
)
def test_dvr_mapper_2d_constructor(Nx, Ny):
    """DVRMapper2D constructs and has correct operator shapes."""
    m = DVRMapper2D(Nx, -1.0, 1.0, Ny=Ny, ay=-1.0, by=1.0, dtype=_DTYPE)
    total = Nx * Ny
    assert m.xy_nodes.shape == (total, 2)
    assert m.D1x.shape == (total, total)
    assert m.D1y.shape == (total, total)
    assert m.D2x.shape == (total, total)
    assert m.D2y.shape == (total, total)
    assert m.laplacian.shape == (total, total)


def test_dvr_mapper_2d_square_shorthand():
    """DVRMapper2D(N, a, b) builds an N² grid on [a,b]²."""
    m = DVRMapper2D(6, -1.0, 1.0, dtype=_DTYPE)
    assert m.shape == (6, 6)
    assert m.xy_nodes.shape == (36, 2)


def test_dvr_mapper_2d_weight_sum():
    """2D weights must sum to (bx-ax)·(by-ay)."""
    m = DVRMapper2D(8, -1.0, 1.0, Ny=6, ay=0.0, by=2.0, dtype=_DTYPE)
    expected = 2.0 * 2.0
    assert abs(m.weights.sum().item() - expected) < 1e-10


# ---------------------------------------------------------------------------
# DVRMapper3D
# ---------------------------------------------------------------------------


def test_dvr_mapper_3d_constructor():
    """DVRMapper3D constructs and has correct shapes."""
    m = DVRMapper3D(4, -1.0, 1.0, dtype=_DTYPE)
    total = 4**3
    assert m.xyz_nodes.shape == (total, 3)
    assert m.D1x.shape == (total, total)
    assert m.laplacian.shape == (total, total)
    assert m.shape == (4, 4, 4)


def test_dvr_mapper_3d_weight_sum():
    """3D weights sum to Lx·Ly·Lz."""
    m = DVRMapper3D(4, -1.0, 1.0, dtype=_DTYPE)
    assert abs(m.weights.sum().item() - 8.0) < 1e-10


# ---------------------------------------------------------------------------
# Mortar projection
# ---------------------------------------------------------------------------


def test_mortar_projection_shape():
    """compute_mortar_projection returns correct shape."""
    P = compute_mortar_projection(8, 5, dtype=_DTYPE)
    assert P.shape == (5, 8)


def test_mortar_projection_constant():
    """Projection of a constant function should give the same constant."""
    N_high, N_low = 12, 6
    P = compute_mortar_projection(N_high, N_low, dtype=_DTYPE)
    f_high = torch.ones(N_high, dtype=_DTYPE)
    f_low_proj = P @ f_high
    assert (f_low_proj - 1.0).abs().max().item() < 1e-12
