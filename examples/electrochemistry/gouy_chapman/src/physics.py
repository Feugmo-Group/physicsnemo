# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gouy-Chapman double-layer: Poisson-Boltzmann residuals.

Two variants are provided:

**Linearized (Debye-Hückel)**:
    ψ'' = κ²ψ   on [0, L]
    ψ(0) = ψ₀,  ψ(L) = 0

Exact solution:
    ψ*(x) = ψ₀ · sinh(κ(L−x)) / sinh(κL)

**Nonlinear (sinh-Poisson)**:
    ψ'' = κ²sinh(ψ)   on [0, L]
    ψ(0) = ψ₀,  ψ(L) = 0

The nonlinear case has no closed-form solution in general;
the linearized form is used as a reference.

KTE node clustering (alpha ≈ 0.9) at x = 0 resolves the Debye layer.
"""

from __future__ import annotations

import math

import torch


def pb_linear_residual(
    psi: torch.Tensor,
    D2: torch.Tensor,
    w_norm: torch.Tensor,
    kappa_sq: float,
) -> torch.Tensor:
    """Linearized Poisson-Boltzmann: R = ψ'' − κ²ψ."""
    R = D2 @ psi - kappa_sq * psi
    return w_norm @ (R * R)


def pb_nonlinear_residual(
    psi: torch.Tensor,
    D2: torch.Tensor,
    w_norm: torch.Tensor,
    kappa_sq: float,
) -> torch.Tensor:
    """Nonlinear Poisson-Boltzmann (sinh form): R = ψ'' − κ²sinh(ψ)."""
    R = D2 @ psi - kappa_sq * torch.sinh(psi)
    return w_norm @ (R * R)


def pb_bc_loss(psi: torch.Tensor, psi_wall: float) -> torch.Tensor:
    """Dirichlet BCs: ψ(0) = ψ_wall, ψ(L) = 0."""
    return (psi[0] - psi_wall) ** 2 + psi[-1] ** 2


def pb_linear_exact(x: torch.Tensor, psi_wall: float, kappa: float, L: float) -> torch.Tensor:
    """Exact solution for linearized PB: ψ₀·sinh(κ(L−x))/sinh(κL)."""
    return psi_wall * torch.sinh(torch.tensor(kappa * (L - x), dtype=x.dtype)) / math.sinh(kappa * L)
