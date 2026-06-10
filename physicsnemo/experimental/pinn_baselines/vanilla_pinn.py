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

"""Vanilla PINN baseline for 1-D boundary-value problems.

A vanilla physics-informed neural network: a small fully-connected MLP
``1 -> hidden x layers -> 1`` with ``Tanh`` activations, trained with Adam to
minimize the mean-squared PDE residual at randomly sampled collocation points
plus a penalized boundary-condition loss. Derivatives are taken with
``torch.autograd.grad``. Collocation points may be drawn once or resampled each
step.

Supported PDEs
--------------
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
import torch.nn as nn

# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------


def _make_mlp(
    hidden_dim: int, n_layers: int, dtype: torch.dtype, device: torch.device
) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(1, hidden_dim), nn.Tanh()]
    for _ in range(n_layers - 2):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
    layers.append(nn.Linear(hidden_dim, 1))
    net = nn.Sequential(*layers).to(dtype=dtype, device=device)
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    return net


# ---------------------------------------------------------------------------
# Autodiff helpers
# ---------------------------------------------------------------------------


def _grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """First derivative dy/dx via autograd (create_graph=True)."""
    return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]


def _u_and_derivatives(net: nn.Sequential, x: torch.Tensor):
    """Return u, u', u'' at collocation points x (requires_grad=True)."""
    u = net(x).squeeze(-1)
    du = _grad(u, x).squeeze(-1)
    d2u = _grad(du, x).squeeze(-1)
    return u, du, d2u


# ---------------------------------------------------------------------------
# PDE residuals (return scalar loss)
# ---------------------------------------------------------------------------


def _helmholtz_residual(net: nn.Sequential, x: torch.Tensor, k: float) -> torch.Tensor:
    u, _, d2u = _u_and_derivatives(net, x)
    R = d2u + k * k * u
    return R.pow(2).mean()


def _cd_residual(
    net: nn.Sequential, x: torch.Tensor, eps: float, a: float
) -> torch.Tensor:
    u, du, d2u = _u_and_derivatives(net, x)
    R = eps * d2u + a * du
    return R.pow(2).mean()


def _gc_linear_residual(
    net: nn.Sequential, x: torch.Tensor, kappa_sq: float
) -> torch.Tensor:
    u, _, d2u = _u_and_derivatives(net, x)
    R = d2u - kappa_sq * u
    return R.pow(2).mean()


# ---------------------------------------------------------------------------
# Exact solutions
# ---------------------------------------------------------------------------


def helmholtz_exact(x: torch.Tensor, k: float) -> torch.Tensor:
    """Exact solution of the Helmholtz problem ``u(x) = sin(k x)``."""
    return torch.sin(k * x)


def cd_exact(x: torch.Tensor, eps: float, a: float) -> torch.Tensor:
    """Exact solution of the 1-D convection-diffusion boundary-value problem."""
    return torch.expm1(a * x / eps) / math.expm1(a / eps)


def gc_linear_exact(x: torch.Tensor, kappa: float, psi_left: float) -> torch.Tensor:
    """Exact solution of the linearized Gouy-Chapman problem."""
    return psi_left * torch.exp(-kappa * x)


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


@dataclass
class VanillaPINNConfig:
    """Configuration for :class:`VanillaPINN`.

    Parameters
    ----------
    a, b : float
        Left/right endpoints of the 1-D domain ``[a, b]``.
    n_pts : int
        Number of interior collocation points.
    resample : bool
        If ``True``, redraw collocation points each training step.
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

    # domain
    a: float = 0.0
    b: float = 1.0
    # collocation
    n_pts: int = 64
    resample: bool = False  # re-draw collocation points each step
    # network
    hidden_dim: int = 64
    n_layers: int = 3
    # training
    n_adam: int = 10_000
    lr: float = 1e-3
    lambda_bc: float = 100.0
    # PDE
    pde: Literal["helmholtz", "cd", "gc_linear"] = "helmholtz"
    # PDE parameters
    k: float = 1.0  # Helmholtz wavenumber
    eps: float = 0.01  # CD diffusion
    a_conv: float = 1.0  # CD convection speed
    kappa_sq: float = 9.0  # gc_linear kappa**2
    psi_left: float = 1.0  # gc_linear left BC
    # hardware
    dtype: torch.dtype = torch.float64
    device: str = "cpu"
    seed: int = 42


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class VanillaPINN:
    """Vanilla PINN: random collocation + autodiff, Adam only.

    Build with a :class:`VanillaPINNConfig`, then call :meth:`train` to fit the
    network. Use :meth:`predict` to evaluate the trained network and
    :meth:`max_abs_error` to compare against the known exact solution.
    """

    def __init__(self, cfg: VanillaPINNConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.dtype = cfg.dtype

        torch.manual_seed(cfg.seed)
        self.net = _make_mlp(cfg.hidden_dim, cfg.n_layers, cfg.dtype, self.device)

        # fixed boundary tensors
        self._xa = torch.tensor([[cfg.a]], dtype=cfg.dtype, device=self.device)
        self._xb = torch.tensor([[cfg.b]], dtype=cfg.dtype, device=self.device)

        # boundary values
        self._ua, self._ub = self._bc_values()

        # pre-draw collocation if not resampling
        if not cfg.resample:
            self._x_col = self._sample_collocation()

        self.history: list[float] = []

    def _bc_values(self):
        cfg = self.cfg
        if cfg.pde == "helmholtz":
            return (
                torch.zeros(1, dtype=cfg.dtype, device=self.device),
                torch.tensor(
                    [math.sin(cfg.k * cfg.b)], dtype=cfg.dtype, device=self.device
                ),
            )
        elif cfg.pde == "cd":
            return (
                torch.zeros(1, dtype=cfg.dtype, device=self.device),
                torch.ones(1, dtype=cfg.dtype, device=self.device),
            )
        elif cfg.pde == "gc_linear":
            return (
                torch.tensor([cfg.psi_left], dtype=cfg.dtype, device=self.device),
                torch.zeros(1, dtype=cfg.dtype, device=self.device),
            )
        else:
            raise ValueError(f"Unknown PDE: {cfg.pde}")

    def _sample_collocation(self) -> torch.Tensor:
        cfg = self.cfg
        x = torch.rand(cfg.n_pts, 1, dtype=cfg.dtype, device=self.device)
        x = cfg.a + (cfg.b - cfg.a) * x
        x.requires_grad_(True)
        return x

    def _pde_loss(self, x: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        if cfg.pde == "helmholtz":
            return _helmholtz_residual(self.net, x, cfg.k)
        elif cfg.pde == "cd":
            return _cd_residual(self.net, x, cfg.eps, cfg.a_conv)
        elif cfg.pde == "gc_linear":
            return _gc_linear_residual(self.net, x, cfg.kappa_sq)
        else:
            raise ValueError(f"Unknown PDE: {cfg.pde}")

    def _bc_loss(self) -> torch.Tensor:
        ua_pred = self.net(self._xa).squeeze()
        ub_pred = self.net(self._xb).squeeze()
        return (ua_pred - self._ua).pow(2).mean() + (ub_pred - self._ub).pow(2).mean()

    def train(self, verbose: bool = False, log_every: int = 500) -> "VanillaPINN":
        """Run Adam optimization for ``cfg.n_adam`` steps.

        Parameters
        ----------
        verbose : bool
            If ``True``, print the loss every ``log_every`` steps.
        log_every : int
            Logging interval (only used when ``verbose`` is ``True``).

        Returns
        -------
        VanillaPINN
            ``self``, with the per-step loss recorded in ``self.history``.
        """
        opt = torch.optim.Adam(self.net.parameters(), lr=self.cfg.lr)
        for step in range(self.cfg.n_adam):
            opt.zero_grad()
            x = self._sample_collocation() if self.cfg.resample else self._x_col
            loss = self._pde_loss(x) + self.cfg.lambda_bc * self._bc_loss()
            loss.backward()
            opt.step()
            val = loss.item()
            self.history.append(val)
            if verbose and step % log_every == 0:
                print(f"  step {step:6d}  loss={val:.3e}")
        return self

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate network (no grad) at arbitrary x (1-D tensor)."""
        with torch.no_grad():
            return self.net(x.unsqueeze(-1) if x.dim() == 1 else x).squeeze(-1)

    def max_abs_error(self) -> float:
        """Return the max absolute error against the exact solution on a grid."""
        cfg = self.cfg
        x = torch.linspace(cfg.a, cfg.b, 200, dtype=cfg.dtype, device=self.device)
        u_pred = self.predict(x)
        if cfg.pde == "helmholtz":
            u_ref = helmholtz_exact(x, cfg.k)
        elif cfg.pde == "cd":
            u_ref = cd_exact(x, cfg.eps, cfg.a_conv)
        elif cfg.pde == "gc_linear":
            u_ref = gc_linear_exact(x, math.sqrt(cfg.kappa_sq), cfg.psi_left)
        else:
            raise ValueError(f"Unknown PDE: {cfg.pde}")
        return (u_pred - u_ref).abs().max().item()
