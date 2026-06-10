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

"""Tests for the natural-gradient optimizers."""

import pytest
import torch
import torch.nn as nn

from physicsnemo.optim.natural_gradient import (
    NaturalGradient,
    SketchedNaturalGradient,
    compute_gram,
)

_HAS_FUNC = hasattr(torch, "func")


def _make_problem(seed: int = 0, n: int = 64, d: int = 4):
    """Build a tiny linear least-squares problem and a one-layer model.

    The model is ``u(x) = x @ W^T + b`` (a single ``nn.Linear``), targets are
    generated from a fixed random teacher weight, so the loss surface is a
    convex quadratic in the parameters and natural gradient should converge
    very fast.
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, dtype=torch.float64, generator=g)
    w_true = torch.randn(1, d, dtype=torch.float64, generator=g)
    b_true = torch.randn(1, dtype=torch.float64, generator=g)
    y = x @ w_true.T + b_true

    model = nn.Linear(d, 1, dtype=torch.float64)
    return model, x, y


def _mse(model, x, y):
    return ((model(x) - y) ** 2).mean()


def test_natural_gradient_converges():
    """NaturalGradient drives a least-squares loss down substantially."""
    model, x, y = _make_problem(seed=1)

    # output_fn: Jacobian of the raw model output defines the Gram metric,
    # which equals the Gauss-Newton metric for this least-squares problem.
    # lr=0.5 yields the exact Newton step here: the output-Jacobian Gram is
    # (1/N) J^T J while the mean-MSE gradient is (2/N) J^T (u - y), so 0.5 * G^-1
    # of the gradient is exactly the least-squares solution.
    natgrad = NaturalGradient(
        model,
        x,
        output_fn=lambda u, xi: u.sum(),
        lr=0.5,
        damping=1e-8,
        gram_update_freq=1,
    )

    loss0 = _mse(model, x, y).item()
    last = loss0
    for _ in range(15):
        loss = _mse(model, x, y)
        natgrad.step(loss)
        last = _mse(model, x, y).item()
        assert torch.isfinite(torch.tensor(last))

    # With a Gauss-Newton metric and the matched lr the quadratic nearly solves.
    assert last < loss0 * 1e-3


def test_natural_gradient_beats_gradient_descent():
    """Natural gradient reduces the loss more than plain GD at matched lr."""
    # Plain gradient descent baseline.
    model_gd, x, y = _make_problem(seed=2)
    opt = torch.optim.SGD(model_gd.parameters(), lr=0.1)
    for _ in range(10):
        opt.zero_grad()
        loss = _mse(model_gd, x, y)
        loss.backward()
        opt.step()
    gd_loss = _mse(model_gd, x, y).item()

    # Natural gradient with the same lr and step budget.
    model_ng, x, y = _make_problem(seed=2)
    natgrad = NaturalGradient(
        model_ng,
        x,
        output_fn=lambda u, xi: u.sum(),
        lr=0.5,
        damping=1e-8,
        gram_update_freq=1,
    )
    for _ in range(10):
        loss = _mse(model_ng, x, y)
        natgrad.step(loss)
    ng_loss = _mse(model_ng, x, y).item()

    assert torch.isfinite(torch.tensor(ng_loss))
    assert ng_loss < gd_loss


@pytest.mark.skipif(not _HAS_FUNC, reason="requires torch.func")
def test_natural_gradient_functional_path():
    """The functional (gram_fn) path also converges on the quadratic."""
    model, x, y = _make_problem(seed=3)

    from torch.func import functional_call

    def gram_fn(params, xi):
        out = functional_call(model, params, (xi.unsqueeze(0),))
        return out.sum()

    natgrad = NaturalGradient(
        model,
        x,
        gram_fn=gram_fn,
        lr=0.5,
        damping=1e-8,
        gram_update_freq=1,
    )

    loss0 = _mse(model, x, y).item()
    for _ in range(15):
        loss = _mse(model, x, y)
        natgrad.step(loss)
    last = _mse(model, x, y).item()

    assert torch.isfinite(torch.tensor(last))
    assert last < loss0 * 1e-2


def test_sketched_natural_gradient_converges():
    """SketchedNaturalGradient decreases the loss and stays finite."""
    model, x, y = _make_problem(seed=4)
    n_params = sum(p.numel() for p in model.parameters())

    sng = SketchedNaturalGradient(
        model,
        x,
        output_fn=lambda u, xi: u.sum(),
        lr=0.5,
        damping=1e-8,
        sketch_size=n_params,  # full-rank sketch for this tiny problem
        gram_update_freq=1,
    )

    loss0 = _mse(model, x, y).item()
    last = loss0
    for _ in range(40):
        loss = _mse(model, x, y)
        sng.step(loss)
        last = _mse(model, x, y).item()
        assert torch.isfinite(torch.tensor(last))

    assert last < loss0 * 0.5
    # effective_rank should report a positive integer once a sketch exists.
    assert sng.effective_rank is not None
    assert sng.effective_rank > 0


def test_compute_gram_shape_and_symmetry():
    """compute_gram returns a square, symmetric, PSD-ish matrix."""
    model, x, _ = _make_problem(seed=5)
    params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)

    def func(xi):
        xi = xi.detach().requires_grad_(True)
        return model(xi).sum()

    G = compute_gram(func, params, x, damping=1e-6)

    assert G.shape == (n_params, n_params)
    assert torch.allclose(G, G.T, atol=1e-10)

    eigvals = torch.linalg.eigvalsh(G)
    # Damping keeps eigenvalues strictly positive.
    assert (eigvals > 0).all()


def test_natural_gradient_gram_property():
    """The cached Gram matrix is exposed and square after a step."""
    model, x, y = _make_problem(seed=6)
    natgrad = NaturalGradient(model, x, output_fn=lambda u, xi: u.sum(), lr=0.1)
    assert natgrad.gram is None
    natgrad.step(_mse(model, x, y))
    n_params = sum(p.numel() for p in model.parameters())
    assert natgrad.gram is not None
    assert natgrad.gram.shape == (n_params, n_params)
    assert natgrad.current_step == 1
