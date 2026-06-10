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

"""Tests for the BRDR adaptive loss weighting module."""

from __future__ import annotations

import pytest
import torch

from physicsnemo.optim.loss_weighting import BalancedResidualDecayRate


def test_balances_differential_decay_rates():
    """BRDR up-weights slow-decaying (stiff) terms to equalize decay rates.

    BRDR balances *relative decay rates*, not absolute magnitudes: a term
    that decays slowly relative to its own fourth-power EMA history is
    up-weighted so its gradient contribution is not drowned out, while a
    term that decays quickly is down-weighted.  This is the defining BRDR
    property — terms that already converge fast yield to the stiff terms
    that the optimizer is still struggling with.

    Three terms start equal but decay at different per-step rates.  The
    slowest term must end up with the largest weight, the fastest with
    the smallest, weights must average to one, and the convergence of the
    weight ordering must hold over a long horizon.
    """
    brdr = BalancedResidualDecayRate(num_losses=3, beta_c=0.9, beta_w=0.9)

    residual_sq = torch.tensor([1.0, 1.0, 1.0])
    decay = torch.tensor([0.90, 0.95, 0.999])  # fast, medium, slow (stiff)

    for _ in range(300):
        residual_sq = torch.clamp(residual_sq * decay, min=0.0)
        brdr(residual_sq.clone())

    w = brdr.weights_ema.detach()

    # Monotone: slowest-decaying (stiffest) term gets the most weight,
    # fastest-decaying term gets the least.
    assert w[2] > w[1] > w[0]

    # Weights average to one (sum to num_losses) by construction.
    assert float(w.sum()) == pytest.approx(3.0, rel=1e-5)

    # The stiff term is boosted above the uniform weight and the easy term
    # suppressed below it — i.e. the imbalance between the per-term decay
    # rates is actively corrected rather than ignored.
    assert w[2] > 1.0
    assert w[0] < 1.0


def test_stiff_term_weight_grows_monotonically():
    """The stiff term's weight is pushed progressively above one over training.

    As the stiff (slow-decaying) term increasingly dominates the residual
    budget relative to its own history, BRDR raises its weight monotone-
    ically (up to EMA smoothing) above the uniform value, demonstrating
    the adaptive concentration of effort onto the unconverged term.
    """
    brdr = BalancedResidualDecayRate(num_losses=2, beta_c=0.9, beta_w=0.9)

    residual_sq = torch.tensor([1.0, 1.0])
    decay = torch.tensor([0.85, 0.99])  # fast, slow (stiff)

    stiff_weights = []
    for _ in range(200):
        residual_sq = torch.clamp(residual_sq * decay, min=0.0)
        brdr(residual_sq.clone())
        stiff_weights.append(float(brdr.weights_ema[1]))

    # The stiff term's weight ends well above the uniform value of one and
    # well above where it started.
    assert stiff_weights[-1] > 1.0
    assert stiff_weights[-1] > stiff_weights[10]


def test_ema_updates_in_train_mode():
    """EMA buffers change across training steps."""
    brdr = BalancedResidualDecayRate(num_losses=2, beta_c=0.9, beta_w=0.9)
    brdr.train()

    # First call seeds the fourth-power EMA.
    brdr(torch.tensor([4.0, 0.25]))
    after_first = brdr.residual_4th_ema.clone()
    assert torch.all(after_first > 0)

    # Feed varying residuals so the EMA keeps moving (a constant input
    # would reach the EMA fixed point and stop changing).
    prev = after_first
    changed = False
    for k in range(5):
        brdr(torch.tensor([4.0 + k, 0.25 + 0.1 * k]))
        if not torch.equal(prev, brdr.residual_4th_ema):
            changed = True
        prev = brdr.residual_4th_ema.clone()
    assert changed, "EMA did not update in train mode"
    assert brdr.num_updates.item() == 6


def test_ema_frozen_in_eval_mode():
    """EMA buffers and weights stay fixed across many eval steps."""
    brdr = BalancedResidualDecayRate(num_losses=2, beta_c=0.9, beta_w=0.9)
    brdr.train()
    for _ in range(10):
        brdr(torch.tensor([4.0, 0.25]) * (1.0 + 0.05 * torch.randn(2)).abs())

    brdr.eval()
    frozen_r4 = brdr.residual_4th_ema.clone()
    frozen_w = brdr.weights_ema.clone()
    frozen_steps = brdr.num_updates.clone()

    for _ in range(20):
        # Feed wildly different magnitudes; nothing should move.
        brdr(torch.tensor([1.0e6, 1.0e-6]))

    assert torch.equal(frozen_r4, brdr.residual_4th_ema)
    assert torch.equal(frozen_w, brdr.weights_ema)
    assert torch.equal(frozen_steps, brdr.num_updates)


def test_forward_accepts_tensor_and_dict_returns_scalar():
    """forward() handles a 1D tensor and an ordered dict, returning a scalar."""
    brdr = BalancedResidualDecayRate(num_losses=2)

    t_out = brdr(torch.tensor([1.0, 2.0]))
    assert t_out.ndim == 0
    assert t_out.requires_grad is False

    brdr2 = BalancedResidualDecayRate(num_losses=2)
    d_out = brdr2({"pde": torch.tensor(1.0), "bc": torch.tensor(2.0)})
    assert d_out.ndim == 0

    # First (seeding) step emits uniform weights -> plain sum of residuals.
    assert t_out.item() == pytest.approx(3.0)
    assert d_out.item() == pytest.approx(3.0)


def test_forward_preserves_gradient_through_residuals():
    """Gradients flow through the residuals while weights are detached."""
    brdr = BalancedResidualDecayRate(num_losses=2)
    x = torch.tensor([2.0, 3.0], requires_grad=True)
    # advance past the seeding step so non-trivial weights are applied
    brdr(x.detach())
    out = brdr(x)
    out.backward()
    assert x.grad is not None
    assert torch.all(torch.isfinite(x.grad))


def test_state_dict_round_trips_buffers():
    """state_dict captures and restores the EMA buffers exactly."""
    brdr = BalancedResidualDecayRate(num_losses=3, beta_c=0.95, beta_w=0.95)
    brdr.train()
    torch.manual_seed(1)
    for _ in range(15):
        brdr(torch.tensor([10.0, 1.0, 0.1]) * (1.0 + 0.1 * torch.randn(3)).abs())

    sd = brdr.state_dict()
    assert "residual_4th_ema" in sd
    assert "weights_ema" in sd
    assert "num_updates" in sd

    restored = BalancedResidualDecayRate(num_losses=3, beta_c=0.95, beta_w=0.95)
    restored.load_state_dict(sd)

    assert torch.equal(restored.residual_4th_ema, brdr.residual_4th_ema)
    assert torch.equal(restored.weights_ema, brdr.weights_ema)
    assert torch.equal(restored.num_updates, brdr.num_updates)

    # The restored module produces identical aggregation.
    probe = torch.tensor([5.0, 5.0, 5.0])
    brdr.eval()
    restored.eval()
    assert torch.equal(brdr(probe), restored(probe))


def test_eval_then_train_resumes_updates():
    """Switching back to train() resumes EMA updates after an eval freeze."""
    brdr = BalancedResidualDecayRate(num_losses=2, beta_c=0.9, beta_w=0.9)
    brdr.train()
    brdr(torch.tensor([2.0, 0.5]))
    brdr(torch.tensor([2.0, 0.5]))

    brdr.eval()
    frozen = brdr.weights_ema.clone()
    brdr(torch.tensor([100.0, 0.01]))
    assert torch.equal(frozen, brdr.weights_ema)

    brdr.train()
    brdr(torch.tensor([100.0, 0.01]))
    assert not torch.equal(frozen, brdr.weights_ema)


def test_composes_with_two_phase_optimizer_eval_call():
    """TwoPhaseOptimizer freezes BRDR by calling aggregator.eval()."""
    from physicsnemo.optim import TwoPhaseOptimizer

    brdr = BalancedResidualDecayRate(num_losses=2, beta_c=0.9, beta_w=0.9)
    x = torch.zeros(2, requires_grad=True)
    target = torch.tensor([3.0, -2.0])

    def closure():
        residuals_squared = (x - target).pow(2)
        return brdr(residuals_squared)

    adam = torch.optim.Adam([x], lr=1e-1)
    lbfgs = torch.optim.LBFGS([x], line_search_fn="strong_wolfe")
    opt = TwoPhaseOptimizer(adam, lbfgs, aggregator=brdr)

    opt.run(closure, n_adam_steps=20, n_lbfgs_steps=5)

    # After the run the optimizer must have left BRDR in eval (frozen) mode.
    assert brdr.training is False
