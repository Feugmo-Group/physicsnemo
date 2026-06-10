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

"""Smoke tests for the experimental reference PINN baselines.

Each baseline ships its own ``train()`` loop with a built-in set of supported
PDEs (the residual/PDE selection lives in the config dataclass via the ``pde``
field, plus the convection PDE for the causal baseline). The tests below build
each baseline with a tiny config (small net, few collocation points, a handful
of Adam steps) on CPU and assert that training runs, produces a finite loss
history, and reduces the loss. Deeper convergence/accuracy is out of scope.
"""

import math

import pytest
import torch

from physicsnemo.experimental.pinn_baselines import (
    PIRBN,
    CausalPINN,
    CausalPINNConfig,
    PIRBNConfig,
    VanillaPINN,
    VanillaPINNConfig,
)


def _finite(history) -> bool:
    return all(math.isfinite(v) for v in history)


@pytest.mark.parametrize("pde", ["helmholtz", "cd", "gc_linear"])
def test_vanilla_pinn_trains(pde):
    """VanillaPINN constructs and reduces the loss over a short Adam run."""
    cfg = VanillaPINNConfig(
        n_pts=16,
        hidden_dim=8,
        n_layers=3,
        n_adam=50,
        pde=pde,
        dtype=torch.float64,
        device="cpu",
        seed=0,
    )
    model = VanillaPINN(cfg)
    model.train(verbose=False)

    assert len(model.history) == cfg.n_adam
    assert _finite(model.history)
    # loss should not blow up; it should drop relative to the start
    assert model.history[-1] < model.history[0]
    # predict / error helpers run without error
    assert math.isfinite(model.max_abs_error())


def test_vanilla_pinn_resample_single_step():
    """A single resampling training step runs without error."""
    cfg = VanillaPINNConfig(
        n_pts=16,
        hidden_dim=8,
        n_layers=3,
        n_adam=1,
        resample=True,
        pde="helmholtz",
        dtype=torch.float64,
        device="cpu",
        seed=0,
    )
    model = VanillaPINN(cfg)
    model.train()
    assert len(model.history) == 1
    assert _finite(model.history)


def test_causal_pinn_trains():
    """CausalPINN constructs and reduces the loss over a short Adam run."""
    cfg = CausalPINNConfig(
        n_pts=128,
        n_t_slabs=4,
        hidden_dim=8,
        n_layers=3,
        n_adam=50,
        dtype=torch.float64,
        device="cpu",
        seed=0,
    )
    model = CausalPINN(cfg)
    history = model.train(log_every=1000)

    assert len(history) == cfg.n_adam
    assert _finite(history)
    assert history[-1] < history[0]
    assert math.isfinite(model.evaluate(nx=8, nt=8))


def test_causal_pinn_reduces_to_vanilla_when_eps_zero():
    """eps_causal=0 yields all-ones causal weights (vanilla PINN limit)."""
    cfg = CausalPINNConfig(
        n_pts=128,
        n_t_slabs=4,
        hidden_dim=8,
        n_layers=3,
        n_adam=1,
        eps_causal=0.0,
        dtype=torch.float64,
        device="cpu",
        seed=0,
    )
    model = CausalPINN(cfg)
    slab_losses = model._pde_residual_per_slab()
    w = model._causal_weights(slab_losses.detach())
    assert torch.allclose(w, torch.ones_like(w))


@pytest.mark.parametrize("pde", ["helmholtz", "cd", "gc_linear"])
def test_pirbn_trains_with_refinement(pde):
    """PIRBN constructs, refines collocation, and reduces the loss."""
    cfg = PIRBNConfig(
        n_pts=32,
        n_candidate=128,
        refine_every=10,
        refine_frac=0.5,
        hidden_dim=8,
        n_layers=3,
        n_adam=50,
        pde=pde,
        dtype=torch.float64,
        device="cpu",
        seed=0,
    )
    model = PIRBN(cfg)
    history = model.train(log_every=1000)

    assert len(history) == cfg.n_adam
    assert _finite(history)
    assert history[-1] < history[0]
    max_err, mean_err = model.evaluate(n_eval=64)
    assert math.isfinite(max_err)
    assert math.isfinite(mean_err)
