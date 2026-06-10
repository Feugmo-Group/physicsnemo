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

"""Tests for the adaptive loss aggregator suite."""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from physicsnemo.optim.aggregators import (
    AGGREGATOR_NAMES,
    Aggregator,
    BalancedResidualDecayRate,
    build_aggregator,
)

NUM_LOSSES = 3
TERM_NAMES = ["pde", "bc", "ic"]

# Stateful schemes whose internal balancing state must freeze in eval mode.
STATEFUL_NAMES = [
    "res_norm",
    "ema",
    "soft_adapt",
    "brdr",
    "relobralo",
    "grad_norm",
    "lr_annealing",
    "ntk",
]


def _make_model() -> nn.Module:
    """Tiny linear model supplying real parameters/gradients."""
    torch.manual_seed(0)
    return nn.Linear(4, 1)


def _losses_from_model(model: nn.Module) -> dict[str, torch.Tensor]:
    """Build a synthetic losses dict with real gradients w.r.t. ``model``."""
    x = torch.randn(8, 4)
    out = model(x)
    return {
        "pde": (out**2).mean(),
        "bc": (out - 1.0).pow(2).mean() * 10.0,
        "ic": (out + 0.5).pow(2).mean() * 1e-2,
    }


def _state_snapshot(agg: Aggregator):
    """Capture a comparable snapshot of an aggregator's balancing state."""
    snap = {}
    for name in ("_ema", "_w", "_prev", "_lambdas", "_initial", "_cache"):
        if hasattr(agg, name):
            val = getattr(agg, name)
            snap[name] = copy.deepcopy(val)
    for buf_name in ("residual_4th_ema", "weights_ema"):
        if hasattr(agg, buf_name) and getattr(agg, buf_name) is not None:
            buf = getattr(agg, buf_name)
            if isinstance(buf, torch.Tensor):
                snap[buf_name] = buf.detach().clone()
    cw = agg.current_weights
    snap["current_weights"] = list(cw) if cw is not None else None
    return snap


@pytest.mark.parametrize("name", AGGREGATOR_NAMES)
def test_build_aggregator_constructs_every_scheme(name):
    """build_aggregator constructs each registered scheme."""
    model = _make_model()
    agg = build_aggregator(
        name,
        model.parameters(),
        num_losses=NUM_LOSSES,
        weights=[1.0, 10.0, 1.0],
    )
    assert isinstance(agg, Aggregator)
    assert agg.num_losses == NUM_LOSSES


def test_build_aggregator_unknown_name_raises():
    """An unknown aggregator name raises ValueError."""
    with pytest.raises(ValueError):
        build_aggregator("does_not_exist", [], num_losses=NUM_LOSSES)


@pytest.mark.parametrize("name", AGGREGATOR_NAMES)
def test_finite_scalar_and_backward(name):
    """Each aggregator returns a finite scalar and backward() works."""
    model = _make_model()
    agg = build_aggregator(name, model.parameters(), num_losses=NUM_LOSSES)
    agg.train()

    # Two steps so that step-1 (non-init) branches execute.
    for step in range(2):
        model.zero_grad(set_to_none=True)
        losses = _losses_from_model(model)
        total = agg(losses, step)
        assert total.dim() == 0
        assert torch.isfinite(total)

    # The final total should be differentiable w.r.t. the model parameters.
    assert total.requires_grad
    total.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "expected at least one parameter gradient"
    assert all(torch.isfinite(g).all() for g in grads)


@pytest.mark.parametrize("name", AGGREGATOR_NAMES)
def test_current_weights_is_list_or_none(name):
    """current_weights returns a list (or None) per scheme."""
    model = _make_model()
    agg = build_aggregator(name, model.parameters(), num_losses=NUM_LOSSES)
    agg.train()
    for step in range(2):
        model.zero_grad(set_to_none=True)
        agg(_losses_from_model(model), step)
    cw = agg.current_weights
    assert cw is None or (isinstance(cw, list) and len(cw) == NUM_LOSSES)


@pytest.mark.parametrize("name", STATEFUL_NAMES)
def test_eval_freezes_state(name):
    """Stateful schemes freeze their balancing state in eval mode."""
    model = _make_model()
    # NTK needs frequent recompute to have any cached state to compare; use a
    # small run_per_step so traces populate during the warm-up.
    kwargs = {"run_per_step": 1} if name == "ntk" else {}
    agg = build_aggregator(name, model.parameters(), num_losses=NUM_LOSSES, **kwargs)

    # Warm up in training mode so state is populated/non-trivial.
    agg.train()
    for step in range(5):
        model.zero_grad(set_to_none=True)
        agg(_losses_from_model(model), step)

    # Freeze and run several eval steps; state must not change.
    agg.eval()
    before = _state_snapshot(agg)
    for step in range(5, 10):
        model.zero_grad(set_to_none=True)
        # Markedly different losses to provoke any unguarded update.
        x = torch.randn(8, 4) * 5.0
        out = model(x)
        losses = {
            "pde": (out**2).mean() * 100.0,
            "bc": (out - 3.0).pow(2).mean(),
            "ic": (out + 2.0).pow(2).mean() * 1e3,
        }
        agg(losses, step)
    after = _state_snapshot(agg)

    for key in before:
        b, a = before[key], after[key]
        if isinstance(b, torch.Tensor):
            assert torch.equal(b, a), f"{name}: buffer {key} changed in eval mode"
        else:
            assert b == a, f"{name}: state {key} changed in eval mode"


def test_brdr_balanced_decay_upweights_slow_term():
    """BRDR up-weights the slowly-decaying term relative to a fast one."""
    model = _make_model()
    agg = BalancedResidualDecayRate(
        list(model.parameters()), num_losses=2, weights=[1.0, 1.0]
    )
    agg.train()

    # term 0 decays steadily; term 1 stays roughly constant (slow decay).
    fast = 1.0
    slow = 1.0
    for step in range(200):
        f = torch.tensor(fast)
        s = torch.tensor(slow)
        agg({"fast": f, "slow": s}, step)
        fast *= 0.97  # decaying steadily; slow term barely moves

    w = agg.current_weights
    assert w[1] > w[0], f"slow term should be up-weighted, got {w}"


def test_homoscedastic_logsigma_in_optimizer_path():
    """Homoscedastic uncertainty exposes learnable log_sigma parameters."""
    model = _make_model()
    agg = build_aggregator("homoscedastic", model.parameters(), num_losses=NUM_LOSSES)
    params = list(agg.parameters())
    assert any(p.requires_grad for p in params), "log_sigma must be learnable"

    agg.train()
    losses = _losses_from_model(model)
    total = agg(losses, 0)
    total.backward()
    assert agg.log_sigma.grad is not None
    assert torch.isfinite(agg.log_sigma.grad).all()
