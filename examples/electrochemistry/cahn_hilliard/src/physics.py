# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Steady Cahn-Hilliard (conserved phase-field) residuals.

PDE on x ∈ [-1, 1]:
    ∇²μ = 0,   μ = f'(c) − ε²∇²c
where f(c) = c²(1−c)²/4   (standard double-well potential)

Equivalently:
    R = D2 @ f'(c) − ε² D4 @ c = 0

Boundary conditions (no-flux):
    c'(-1) = c'(1) = 0
    μ'(-1) = μ'(1) = 0

A steady solution with prescribed mean composition c̄ and two-phase
separation is obtained by starting from a random field and constraining
the mean via a Lagrange multiplier term.
"""

from __future__ import annotations

import torch


def _f_prime(c: torch.Tensor) -> torch.Tensor:
    """f'(c) = c(1-c)(1-2c) for f = c²(1-c)²/4."""
    return c * (1.0 - c) * (1.0 - 2.0 * c)


def cahn_hilliard_residual(
    c: torch.Tensor,
    D2: torch.Tensor,
    D4: torch.Tensor,
    w_norm: torch.Tensor,
    eps_sq: float,
) -> torch.Tensor:
    """Weighted residual loss for ∇²μ = 0, μ = f'(c) − ε²∇²c."""
    fp = _f_prime(c)
    R = D2 @ fp - eps_sq * (D4 @ c)
    return w_norm @ (R * R)


def cahn_hilliard_noflux_bc_loss(
    c: torch.Tensor,
    D1: torch.Tensor,
    D2: torch.Tensor,
    eps_sq: float,
) -> torch.Tensor:
    """No-flux BCs: c'(±1) = 0, μ'(±1) = 0."""
    mu = _f_prime(c) - eps_sq * (D2 @ c)
    cp = D1 @ c
    mup = D1 @ mu
    return cp[0]**2 + cp[-1]**2 + mup[0]**2 + mup[-1]**2


def cahn_hilliard_mass_loss(c: torch.Tensor, w_norm: torch.Tensor, c_mean: float) -> torch.Tensor:
    """Mass conservation: (w_norm @ c − c_mean)²."""
    return (w_norm @ c - c_mean) ** 2
