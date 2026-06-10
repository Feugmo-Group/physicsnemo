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

"""PIRBN baseline: residual-based adaptive collocation PINN.

A physics-informed neural network with residual-based adaptive refinement. The
key difference from a vanilla PINN is that every ``refine_every`` steps a
fraction of the collocation points are resampled from a candidate pool with
probability proportional to the pointwise PDE-residual magnitude, so the network
concentrates capacity on high-error regions. The network, residuals, and exact
solutions are shared with the vanilla PINN baseline.

Supported PDEs (same set as the vanilla PINN baseline)
------------------------------------------------------
helmholtz
    ``u'' + k**2 u = 0``, with ``u(0) = 0`` and ``u(1) = sin(k)``.
cd
    Convection-diffusion ``eps u'' + a u' = 0``, with ``u(0) = 0`` and
    ``u(1) = 1``.
gc_linear
    Linearized Gouy-Chapman ``psi'' - kappa**2 psi = 0``, with
    ``psi(0) = psi_left`` and ``psi(1) = 0``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch

from physicsnemo.experimental.pinn_baselines.vanilla_pinn import (
    _make_mlp,
    _u_and_derivatives,
    cd_exact,
    gc_linear_exact,
    helmholtz_exact,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class PIRBNConfig:
    """Configuration for :class:`PIRBN`.

    Parameters
    ----------
    a, b : float
        Left/right endpoints of the 1-D domain ``[a, b]``.
    n_pts : int
        Number of active interior collocation points.
    n_candidate : int
        Size of the candidate pool drawn at each refinement.
    refine_every : int
        Number of steps between adaptive collocation refinements.
    refine_frac : float
        Fraction of active collocation points replaced at each refinement.
    hidden_dim : int
        Width of each hidden layer.
    n_layers : int
        Total number of linear layers (including input and output).
    n_adam : int
        Number of Adam optimization steps.
    lr : float
        Adam learning rate.
    lambda_bc : float
        Weight on the boundary-condition loss term.
    pde : {"helmholtz", "cd", "gc_linear"}
        Which supported PDE to solve.
    k : float
        Helmholtz wavenumber.
    eps : float
        Convection-diffusion diffusion coefficient.
    a_conv : float
        Convection-diffusion convection speed.
    kappa_sq : float
        ``kappa**2`` parameter for the linearized Gouy-Chapman PDE.
    psi_left : float
        Left boundary value for the linearized Gouy-Chapman PDE.
    dtype : torch.dtype
        Floating-point precision used for all tensors.
    device : str
        Torch device string (e.g. ``"cpu"`` or ``"cuda"``).
    seed : int
        Random seed for reproducibility.
    """

    a: float = 0.0
    b: float = 1.0
    n_pts: int = 512
    n_candidate: int = 2048
    refine_every: int = 1000
    refine_frac: float = 0.5
    hidden_dim: int = 64
    n_layers: int = 3
    n_adam: int = 10_000
    lr: float = 1e-3
    lambda_bc: float = 100.0
    pde: Literal["helmholtz", "cd", "gc_linear"] = "helmholtz"
    k: float = 1.0
    eps: float = 0.01
    a_conv: float = 1.0
    kappa_sq: float = 9.0
    psi_left: float = 1.0
    dtype: torch.dtype = torch.float64
    device: str = "cpu"
    seed: int = 42


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class PIRBN:
    """PINN with residual-based adaptive collocation refinement.

    Build with a :class:`PIRBNConfig`, then call :meth:`train` to fit the
    network. Use :meth:`evaluate` to compute the max and mean absolute errors
    against the known exact solution.
    """

    def __init__(self, cfg: PIRBNConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.dtype = cfg.dtype

        torch.manual_seed(cfg.seed)
        self.net = _make_mlp(cfg.hidden_dim, cfg.n_layers, cfg.dtype, self.device)

        self._xa = torch.tensor([[cfg.a]], dtype=cfg.dtype, device=self.device)
        self._xb = torch.tensor([[cfg.b]], dtype=cfg.dtype, device=self.device)
        self._ua, self._ub = self._bc_values()
        self._x_col = self._sample_uniform(cfg.n_pts)

    def _bc_values(self):
        pde = self.cfg.pde
        if pde == "helmholtz":
            return (
                torch.zeros(1, dtype=self.dtype, device=self.device),
                torch.tensor(
                    [math.sin(self.cfg.k)], dtype=self.dtype, device=self.device
                ),
            )
        elif pde == "cd":
            return (
                torch.zeros(1, dtype=self.dtype, device=self.device),
                torch.ones(1, dtype=self.dtype, device=self.device),
            )
        else:
            return (
                torch.tensor([self.cfg.psi_left], dtype=self.dtype, device=self.device),
                torch.zeros(1, dtype=self.dtype, device=self.device),
            )

    def _sample_uniform(self, n: int) -> torch.Tensor:
        x = torch.rand(n, 1, dtype=self.dtype, device=self.device)
        x = x * (self.cfg.b - self.cfg.a) + self.cfg.a
        x.requires_grad_(True)
        return x

    def _pde_residual_vec(self, x: torch.Tensor) -> torch.Tensor:
        """Return element-wise |R_i| (no reduction)."""
        pde = self.cfg.pde
        if pde == "helmholtz":
            u, _, d2u = _u_and_derivatives(self.net, x)
            return (d2u + self.cfg.k**2 * u).abs()
        elif pde == "cd":
            u, du, d2u = _u_and_derivatives(self.net, x)
            return (self.cfg.eps * d2u + self.cfg.a_conv * du).abs()
        else:
            u, _, d2u = _u_and_derivatives(self.net, x)
            return (d2u - self.cfg.kappa_sq * u).abs()

    def _pde_loss(self, x: torch.Tensor) -> torch.Tensor:
        return self._pde_residual_vec(x).pow(2).mean()

    def _bc_loss(self) -> torch.Tensor:
        ua_pred = self.net(self._xa).squeeze()
        ub_pred = self.net(self._xb).squeeze()
        return (ua_pred - self._ua) ** 2 + (ub_pred - self._ub) ** 2

    def _refine_collocation(self):
        """Resample collocation points towards high-residual regions."""
        x_cand = self._sample_uniform(self.cfg.n_candidate)
        with torch.no_grad():
            # need grad for autodiff PDE
            x_cand_g = x_cand.detach().requires_grad_(True)
        r = self._pde_residual_vec(x_cand_g).detach().squeeze()
        prob = r / (r.sum() + 1e-30)
        n_replace = max(1, int(self.cfg.n_pts * self.cfg.refine_frac))
        idx = torch.multinomial(prob, n_replace, replacement=False)
        x_new = x_cand[idx].detach().clone()

        x_keep = self._x_col.detach()[: self.cfg.n_pts - n_replace]
        self._x_col = torch.cat([x_keep, x_new], dim=0).requires_grad_(True)

    def train(self, log_every: int = 500) -> list[float]:
        """Run Adam optimization with periodic residual-based refinement.

        Parameters
        ----------
        log_every : int
            Interval at which the loss is printed.

        Returns
        -------
        list of float
            Per-step total loss history.
        """
        history: list[float] = []
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.cfg.lr)

        for step in range(self.cfg.n_adam):
            if step > 0 and step % self.cfg.refine_every == 0:
                self._refine_collocation()

            optimizer.zero_grad()
            loss = self._pde_loss(self._x_col) + self.cfg.lambda_bc * self._bc_loss()
            loss.backward()
            optimizer.step()
            history.append(loss.item())
            if step % log_every == 0:
                print(f"  PIRBN step {step:5d}: loss={loss.item():.3e}")

        return history

    def evaluate(self, n_eval: int = 1000) -> tuple[float, float]:
        """Return ``(max_abs_error, mean_abs_error)`` against the exact solution."""
        x = torch.linspace(
            self.cfg.a, self.cfg.b, n_eval, dtype=self.dtype, device=self.device
        ).unsqueeze(1)
        with torch.no_grad():
            u_pred = self.net(x).squeeze()
        u_ex = self._exact(x.squeeze())
        err = (u_pred - u_ex).abs()
        return float(err.max()), float(err.mean())

    def _exact(self, x: torch.Tensor) -> torch.Tensor:
        pde = self.cfg.pde
        if pde == "helmholtz":
            return helmholtz_exact(x, self.cfg.k)
        elif pde == "cd":
            return cd_exact(x, self.cfg.eps, self.cfg.a_conv)
        else:
            return gc_linear_exact(x, math.sqrt(self.cfg.kappa_sq), self.cfg.psi_left)
