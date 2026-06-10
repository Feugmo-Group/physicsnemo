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

"""Causal PINN baseline for time-dependent problems.

A causal (time-weighted) physics-informed neural network. The space-time
domain is split into ``n_t_slabs`` ordered time slabs; the PDE-residual loss of
slab ``i`` is weighted by

    ``w_i = exp(-eps_causal * sum_{j < i} L_j)``,

where ``L_j`` is the mean residual loss in slab ``j``. This forces the network
to first satisfy the residual at early times before later slabs are weighted
appreciably; ``eps_causal = 0`` recovers a vanilla PINN. Collocation points are
random space-time samples sorted by increasing time.

Supported PDE
-------------
convection
    ``u_t + a u_x = 0`` on ``[0, L] x [0, T]`` with initial condition
    ``u(x, 0) = sin(2 pi x)`` and inflow boundary ``u(0, t) = sin(-2 pi a t)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# MLP  (2-D input: [x, t])
# ---------------------------------------------------------------------------


def _make_mlp_2d(
    hidden_dim: int, n_layers: int, dtype: torch.dtype, device: torch.device
) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(2, hidden_dim), nn.Tanh()]
    for _ in range(n_layers - 2):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
    layers.append(nn.Linear(hidden_dim, 1))
    net = nn.Sequential(*layers).to(dtype=dtype, device=device)
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    return net


def _grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class CausalPINNConfig:
    """Configuration for :class:`CausalPINN`.

    Parameters
    ----------
    L, T : float
        Spatial domain length and final time; the domain is ``[0, L] x [0, T]``.
    a : float
        Convection speed of the transport PDE.
    n_pts : int
        Number of interior space-time collocation points.
    n_t_slabs : int
        Number of ordered time slabs used for causal weighting.
    eps_causal : float
        Causality strength; ``0`` recovers a vanilla PINN.
    hidden_dim : int
        Width of each hidden layer.
    n_layers : int
        Total number of linear layers (including input and output).
    n_adam : int
        Number of Adam optimization steps.
    lr : float
        Adam learning rate.
    lambda_ic : float
        Weight on the initial-condition loss term.
    lambda_bc : float
        Weight on the boundary-condition loss term.
    dtype : torch.dtype
        Floating-point precision used for all tensors.
    device : str
        Torch device string (e.g. ``"cpu"`` or ``"cuda"``).
    seed : int
        Random seed for reproducibility.
    """

    # domain
    L: float = 1.0
    T: float = 1.0
    # PDE
    a: float = 1.0  # convection speed
    # collocation
    n_pts: int = 2048
    n_t_slabs: int = 32
    # causality
    eps_causal: float = 1.0
    # network
    hidden_dim: int = 64
    n_layers: int = 3
    # training
    n_adam: int = 20_000
    lr: float = 1e-3
    lambda_ic: float = 100.0
    lambda_bc: float = 100.0
    # hardware
    dtype: torch.dtype = torch.float64
    device: str = "cpu"
    seed: int = 42


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class CausalPINN:
    """Causal-weighted PINN for the time-dependent convection equation.

    Build with a :class:`CausalPINNConfig`, then call :meth:`train` to fit the
    network. Use :meth:`evaluate` to compute the max absolute error against the
    exact travelling-wave solution.
    """

    def __init__(self, cfg: CausalPINNConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.dtype = cfg.dtype

        torch.manual_seed(cfg.seed)
        self.net = _make_mlp_2d(cfg.hidden_dim, cfg.n_layers, cfg.dtype, self.device)
        self._sample_points()

    def _sample_points(self):
        cfg = self.cfg
        x = torch.rand(cfg.n_pts, 1, dtype=self.dtype, device=self.device) * cfg.L
        t = torch.rand(cfg.n_pts, 1, dtype=self.dtype, device=self.device) * cfg.T
        # sort by time so causality windowing works
        idx = t.squeeze().argsort()
        self._x_col = x[idx].requires_grad_(True)
        self._t_col = t[idx].requires_grad_(True)

        # IC points: t=0, x in [0, L]
        n_ic = cfg.n_pts // 4
        self._x_ic = torch.rand(n_ic, 1, dtype=self.dtype, device=self.device) * cfg.L
        self._t_ic = torch.zeros(n_ic, 1, dtype=self.dtype, device=self.device)

        # BC points: x=0, t in [0, T]
        n_bc = cfg.n_pts // 4
        self._x_bc = torch.zeros(n_bc, 1, dtype=self.dtype, device=self.device)
        self._t_bc = torch.rand(n_bc, 1, dtype=self.dtype, device=self.device) * cfg.T

    def _net(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, t], dim=1)).squeeze(-1)

    def _pde_residual_per_slab(self) -> torch.Tensor:
        """Return mean residual loss per time slab (length = n_t_slabs)."""
        cfg = self.cfg
        t_edges = torch.linspace(
            0, cfg.T, cfg.n_t_slabs + 1, dtype=self.dtype, device=self.device
        )
        t_vals = self._t_col.detach().squeeze()
        slab_losses = torch.zeros(cfg.n_t_slabs, dtype=self.dtype, device=self.device)

        for k in range(cfg.n_t_slabs):
            mask = (t_vals >= t_edges[k]) & (t_vals < t_edges[k + 1])
            if not mask.any():
                continue
            x_k = self._x_col[mask]
            t_k = self._t_col[mask]
            u = self._net(x_k, t_k)
            u_t = _grad(u, t_k).squeeze(-1)
            u_x = _grad(u, x_k).squeeze(-1)
            R = u_t + cfg.a * u_x
            slab_losses[k] = R.pow(2).mean()

        return slab_losses

    def _causal_weights(self, slab_losses: torch.Tensor) -> torch.Tensor:
        """w_k = exp(-eps * sum_{j<k} L_j)."""
        cum = torch.cat(
            [
                torch.zeros(1, dtype=self.dtype, device=self.device),
                slab_losses.cumsum(0)[:-1],
            ]
        )
        return torch.exp(-self.cfg.eps_causal * cum)

    def _ic_loss(self) -> torch.Tensor:
        u_ic = self._net(self._x_ic, self._t_ic)
        u_ex = torch.sin(2 * math.pi * self._x_ic.squeeze())
        return (u_ic - u_ex).pow(2).mean()

    def _bc_loss(self) -> torch.Tensor:
        u_bc = self._net(self._x_bc, self._t_bc)
        u_ex = torch.sin(-2 * math.pi * self.cfg.a * self._t_bc.squeeze())
        return (u_bc - u_ex).pow(2).mean()

    def train(self, log_every: int = 1000) -> list[float]:
        """Run Adam optimization with causal residual weighting.

        Parameters
        ----------
        log_every : int
            Interval at which the loss and minimum causal weight are printed.

        Returns
        -------
        list of float
            Per-step total loss history.
        """
        history: list[float] = []
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.cfg.lr)

        for step in range(self.cfg.n_adam):
            optimizer.zero_grad()
            slab_losses = self._pde_residual_per_slab()
            w = self._causal_weights(slab_losses.detach())
            pde_loss = (w * slab_losses).mean()
            loss = (
                pde_loss
                + self.cfg.lambda_ic * self._ic_loss()
                + self.cfg.lambda_bc * self._bc_loss()
            )
            loss.backward()
            optimizer.step()
            history.append(loss.item())
            if step % log_every == 0:
                print(
                    f"  CausalPINN step {step:5d}: loss={loss.item():.3e}  "
                    f"w_min={w.min().item():.3e}"
                )

        return history

    def evaluate(self, nx: int = 200, nt: int = 200) -> float:
        """Return the max absolute error against the exact solution on a grid."""
        x = torch.linspace(0, self.cfg.L, nx, dtype=self.dtype, device=self.device)
        t = torch.linspace(0, self.cfg.T, nt, dtype=self.dtype, device=self.device)
        X, T = torch.meshgrid(x, t, indexing="ij")
        xf = X.reshape(-1, 1)
        tf = T.reshape(-1, 1)
        with torch.no_grad():
            u_pred = self.net(torch.cat([xf, tf], dim=1)).squeeze()
        u_ex = torch.sin(2 * math.pi * (xf.squeeze() - self.cfg.a * tf.squeeze()))
        err = (u_pred - u_ex).abs()
        return float(err.max())
