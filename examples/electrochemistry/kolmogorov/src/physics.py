# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""2D Kolmogorov flow residuals.

The 2D Kolmogorov flow is a forced steady Navier-Stokes problem on
(x, y) ∈ [0, 2π]² with periodic boundary conditions:

    −ν∇²ψ + J(ψ, ∇²ψ) = F(x, y)

where ψ is the stream function (u = ∂ψ/∂y, v = −∂ψ/∂x), ∇²ψ is the
vorticity ω, and J(ψ, ω) = (∂ψ/∂x)(∂ω/∂y) − (∂ψ/∂y)(∂ω/∂x).

Forcing function: F(x, y) = n_force · sin(n_force · y)

For small Re = 1/ν, the exact steady-state stream function is:
    ψ*(x, y) = −sin(n_force · y) / (ν · n_force²)

This LegendreKAN backbone example demonstrates spectral-accuracy
derivatives via the precomputed DVRMapper2D Kronecker operators.
"""

from __future__ import annotations

import math

import torch


def kolmogorov_forcing(xy: torch.Tensor, n_force: int) -> torch.Tensor:
    """Forcing F(x,y) = n_force · sin(n_force · y)."""
    return float(n_force) * torch.sin(float(n_force) * xy[:, 1])


def kolmogorov_residual(
    psi: torch.Tensor,
    D1x: torch.Tensor,
    D1y: torch.Tensor,
    lap: torch.Tensor,
    w_norm: torch.Tensor,
    xy: torch.Tensor,
    nu: float,
    n_force: int,
) -> torch.Tensor:
    """Weighted vorticity equation residual.

    R = −ν∇²ω + J(ψ, ω) − F,   ω = ∇²ψ.
    """
    omega = lap @ psi
    d_omega_dx = D1x @ omega
    d_omega_dy = D1y @ omega
    d_psi_dx = D1x @ psi
    d_psi_dy = D1y @ psi
    F = kolmogorov_forcing(xy, n_force)

    J = d_psi_dx * d_omega_dy - d_psi_dy * d_omega_dx
    lap_omega = lap @ omega
    R = -nu * lap_omega + J - F
    return w_norm @ (R**2)


def kolmogorov_exact(xy: torch.Tensor, nu: float, n_force: int) -> torch.Tensor:
    """Exact stream function ψ*(x,y) = −sin(n·y) / (ν·n²)."""
    return -torch.sin(float(n_force) * xy[:, 1]) / (nu * n_force**2)


def kolmogorov_bc_loss(psi: torch.Tensor, w_norm: torch.Tensor) -> torch.Tensor:
    """Zero mean constraint: ∫ψ dA = 0 (removes translation mode)."""
    return (w_norm @ psi) ** 2
