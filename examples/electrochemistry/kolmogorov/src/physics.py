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

import sympy as sp
import torch

from physicsnemo.experimental.models.scen import AxisOperator, DVRPhysicsInformer
from physicsnemo.sym.eq.pde import PDE


def kolmogorov_forcing(xy: torch.Tensor, n_force: int) -> torch.Tensor:
    """Forcing F(x,y) = n_force · sin(n_force · y)."""
    return float(n_force) * torch.sin(float(n_force) * xy[:, 1])


class KolmogorovMixedPDE(PDE):
    r"""Mixed (stream-vorticity) Kolmogorov-flow system for :class:`DVRPhysicsInformer`.

    The 4th-order (biharmonic) steady vorticity equation is recast as a coupled
    **second-order** system in the stream function ``psi`` and the vorticity
    ``omega``:

    .. math::
        \omega - \nabla^2\psi = 0, \qquad
        -\nu\,\nabla^2\omega + J(\psi,\omega) - F = 0,

    with the Jacobian ``J(ψ,ω) = ψ_x ω_y − ψ_y ω_x`` and ``F`` the forcing leaf.
    This removes the biharmonic operator entirely.

    Parameters
    ----------
    nu : float
        Kinematic viscosity ν.
    """

    name = "KolmogorovMixed"

    def __init__(self, nu: float):
        self.dim = 2
        x, y = sp.symbols("x y")
        psi = sp.Function("psi")(x, y)
        omega = sp.Function("omega")(x, y)
        force = sp.Function("F")(x, y)
        lap_psi = psi.diff(x, 2) + psi.diff(y, 2)
        lap_omega = omega.diff(x, 2) + omega.diff(y, 2)
        jacobian = psi.diff(x, 1) * omega.diff(y, 1) - psi.diff(y, 1) * omega.diff(x, 1)
        self.equations = {
            "res_w": omega - lap_psi,
            "res_vort": -sp.Number(nu) * lap_omega + jacobian - force,
        }


def make_kolmogorov_informer(
    D1x: torch.Tensor,
    D1y: torch.Tensor,
    D2x: torch.Tensor,
    D2y: torch.Tensor,
    nu: float,
    device: str | None = None,
) -> DVRPhysicsInformer:
    """Build a DVR-collocation informer for the mixed Kolmogorov system.

    Parameters
    ----------
    D1x, D1y, D2x, D2y : torch.Tensor
        First/second-derivative DVR operators (flattened 2D), each ``(N, N)``.
    nu : float
        Kinematic viscosity ν.
    device : str or None, optional
        Device for the informer.

    Returns
    -------
    DVRPhysicsInformer
    """
    return DVRPhysicsInformer(
        required_outputs=["res_w", "res_vort"],
        equations=KolmogorovMixedPDE(nu),
        operators={
            "x": AxisOperator(axis=0, D1=D1x, D2=D2x),
            "y": AxisOperator(axis=0, D1=D1y, D2=D2y),
        },
        device=device,
    )


def kolmogorov_residuals_dvr(
    informer: DVRPhysicsInformer,
    psi: torch.Tensor,
    omega: torch.Tensor,
    w_norm: torch.Tensor,
    xy: torch.Tensor,
    n_force: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Weighted residual losses for the mixed Kolmogorov system.

    Returns
    -------
    (l_w, l_vort) : tuple of torch.Tensor
        Scalar weighted MSEs of the vorticity definition and the vorticity-
        transport equation, respectively.
    """
    force = kolmogorov_forcing(xy, n_force)
    res = informer.forward({"psi": psi, "omega": omega, "F": force})
    r_w = res["res_w"].reshape(-1)
    r_vort = res["res_vort"].reshape(-1)
    return w_norm @ (r_w * r_w), w_norm @ (r_vort * r_vort)


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
