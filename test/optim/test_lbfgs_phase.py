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

"""Tests for TwoPhaseOptimizer."""

from __future__ import annotations

import pytest
import torch

from physicsnemo.optim import TwoPhaseOptimizer


def _make_quadratic_problem():
    """Return (params, closure_fn) for 1D quadratic minimization: f(x) = (x-3)²."""
    x = torch.tensor([0.0], requires_grad=True)
    target = torch.tensor([3.0])

    def closure():
        return (x - target).pow(2).sum()

    return x, closure


def test_two_phase_history_length():
    """History length equals n_adam + actual L-BFGS steps (≤ n_adam + n_lbfgs)."""
    x, closure = _make_quadratic_problem()
    adam = torch.optim.Adam([x], lr=0.1)
    lbfgs = torch.optim.LBFGS([x], line_search_fn="strong_wolfe")
    opt = TwoPhaseOptimizer(adam, lbfgs)
    history = opt.run(closure, n_adam_steps=50, n_lbfgs_steps=20)
    assert len(history) <= 70
    assert len(history) >= 50  # at least Adam steps


def test_two_phase_phases_labelled():
    """History entries contain 'phase' key with 'adam' and 'lbfgs' values."""
    x, closure = _make_quadratic_problem()
    adam = torch.optim.Adam([x], lr=0.1)
    lbfgs = torch.optim.LBFGS([x], line_search_fn="strong_wolfe")
    opt = TwoPhaseOptimizer(adam, lbfgs)
    history = opt.run(closure, n_adam_steps=10, n_lbfgs_steps=5)
    phases = [h["phase"] for h in history]
    assert "adam" in phases
    assert "lbfgs" in phases
    assert all(p in ("adam", "lbfgs") for p in phases)


def test_two_phase_loss_decreases():
    """Final loss is significantly lower than initial loss."""
    x, closure = _make_quadratic_problem()
    initial_loss = closure().item()
    adam = torch.optim.Adam([x], lr=0.1)
    lbfgs = torch.optim.LBFGS([x], line_search_fn="strong_wolfe")
    opt = TwoPhaseOptimizer(adam, lbfgs)
    history = opt.run(closure, n_adam_steps=100, n_lbfgs_steps=20)
    final_loss = history[-1]["loss"]
    assert final_loss < initial_loss * 1e-3, f"Loss didn't decrease: {initial_loss:.4f} → {final_loss:.4f}"


def test_two_phase_lbfgs_phase_lower_than_adam():
    """L-BFGS phase should achieve lower loss than Adam-only."""
    x_adam, closure_adam = _make_quadratic_problem()
    x_both, closure_both = _make_quadratic_problem()

    adam_only = torch.optim.Adam([x_adam], lr=0.1)
    opt_adam = TwoPhaseOptimizer(adam_only, torch.optim.LBFGS([x_adam]))
    h_adam = opt_adam.run(closure_adam, n_adam_steps=100, n_lbfgs_steps=0)

    adam2 = torch.optim.Adam([x_both], lr=0.1)
    lbfgs = torch.optim.LBFGS([x_both], line_search_fn="strong_wolfe")
    opt_both = TwoPhaseOptimizer(adam2, lbfgs)
    h_both = opt_both.run(closure_both, n_adam_steps=100, n_lbfgs_steps=20)

    assert h_both[-1]["loss"] < h_adam[-1]["loss"]


def test_two_phase_adam_only():
    """n_lbfgs_steps=0 runs only Adam and returns history without 'lbfgs' phase."""
    x, closure = _make_quadratic_problem()
    adam = torch.optim.Adam([x], lr=0.1)
    lbfgs = torch.optim.LBFGS([x])
    opt = TwoPhaseOptimizer(adam, lbfgs)
    history = opt.run(closure, n_adam_steps=20, n_lbfgs_steps=0)
    assert len(history) == 20
    assert all(h["phase"] == "adam" for h in history)


def test_two_phase_aggregator_set_eval():
    """Aggregator .eval() is called before the L-BFGS phase."""
    import torch.nn as nn

    class MockAggregator(nn.Module):
        def __init__(self):
            super().__init__()
            self.eval_called = False

        def eval(self):
            self.eval_called = True
            return super().eval()

    x, closure = _make_quadratic_problem()
    agg = MockAggregator()
    adam = torch.optim.Adam([x], lr=0.1)
    lbfgs = torch.optim.LBFGS([x], line_search_fn="strong_wolfe")
    opt = TwoPhaseOptimizer(adam, lbfgs, aggregator=agg)
    opt.run(closure, n_adam_steps=5, n_lbfgs_steps=3)
    assert agg.eval_called, "aggregator.eval() was not called before L-BFGS phase"
