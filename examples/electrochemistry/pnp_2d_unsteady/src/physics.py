# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""2D unsteady PNP residuals using DVRMapper2D.

Problem on (x, y) ∈ [0,1]², t ∈ [0, T]:

    ∂c_p/∂t = ∇²c_p + ∇·(c_p ∇φ) + f₁(x,y,t)
    ∂c_n/∂t = ∇²c_n − ∇·(c_n ∇φ) + f₂(x,y,t)
    ∇²φ = −c_p + c_n

Manufactured exact solution:
    c_p(x,y,t) = sin(πx) sin(πy) e^{−t}
    c_n(x,y,t) = cos(πx) cos(πy) e^{−t}
    φ(x,y,t)   = sin(πx) cos(πy) e^{−t} / (2π²)
"""

from __future__ import annotations

import math

import sympy as sp
import torch

from physicsnemo.experimental.models.scen import AxisOperator, DVRPhysicsInformer
from physicsnemo.sym.eq.pde import PDE

PI = math.pi


class Pnp2DUnsteadyPDE(PDE):
    """Symbolic 2D unsteady PNP system for :class:`DVRPhysicsInformer`.

    Spatial coordinates ``x``, ``y`` (both acting on the flattened 2D grid),
    time ``z``.  Three coupled fields ``cp``, ``cn``, ``phi``; manufactured
    forcings ``f1``, ``f2`` enter as leaf functions supplied at evaluation time.
    """

    name = "Pnp2DUnsteady"

    def __init__(self):
        self.dim = 3
        x, y, z = sp.symbols("x y z")
        cp = sp.Function("cp")(x, y, z)
        cn = sp.Function("cn")(x, y, z)
        phi = sp.Function("phi")(x, y, z)
        f1 = sp.Function("f1")(x, y, z)
        f2 = sp.Function("f2")(x, y, z)

        def lap(f):
            return f.diff(x, 2) + f.diff(y, 2)

        def grad_dot(a, b):
            return a.diff(x, 1) * b.diff(x, 1) + a.diff(y, 1) * b.diff(y, 1)

        self.equations = {
            "res_cp": cp.diff(z, 1) - lap(cp) - grad_dot(cp, phi) - cp * lap(phi) - f1,
            "res_cn": cn.diff(z, 1) - lap(cn) + grad_dot(cn, phi) + cn * lap(phi) - f2,
            "res_phi": lap(phi) + cp - cn,
        }


def make_pnp_2d_informer(
    D1x: torch.Tensor,
    D1y: torch.Tensor,
    D2x: torch.Tensor,
    D2y: torch.Tensor,
    D1t: torch.Tensor,
    device: str | None = None,
) -> DVRPhysicsInformer:
    """Build a DVR-collocation informer for the 2D unsteady PNP system.

    Parameters
    ----------
    D1x, D1y, D2x, D2y : torch.Tensor
        Spatial first/second-derivative DVR operators (flattened 2D), each
        ``(N_spatial, N_spatial)``; act on grid axis 1.
    D1t : torch.Tensor
        Time first-derivative DVR matrix ``(Nt, Nt)``; acts on grid axis 0.
    device : str or None, optional
        Device for the informer.

    Returns
    -------
    DVRPhysicsInformer
    """
    return DVRPhysicsInformer(
        required_outputs=["res_cp", "res_cn", "res_phi"],
        equations=Pnp2DUnsteadyPDE(),
        operators={
            "x": AxisOperator(axis=1, D1=D1x, D2=D2x),
            "y": AxisOperator(axis=1, D1=D1y, D2=D2y),
            "z": AxisOperator(axis=0, D1=D1t, D2=None),
        },
        device=device,
    )


def pnp_2d_sources_spacetime(
    xy: torch.Tensor, t_grid: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack the manufactured sources over a full ``(Nt, N_spatial)`` grid."""
    f1_rows, f2_rows = [], []
    for t_val in t_grid.tolist():
        f1_i, f2_i = pnp_2d_sources(xy, t_val)
        f1_rows.append(f1_i)
        f2_rows.append(f2_i)
    return torch.stack(f1_rows, dim=0), torch.stack(f2_rows, dim=0)


def pnp_2d_residuals_dvr(
    informer: DVRPhysicsInformer,
    cp: torch.Tensor,
    cn: torch.Tensor,
    phi: torch.Tensor,
    w_xyt: torch.Tensor,
    xy: torch.Tensor,
    t_grid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Weighted space-time residual losses on the full grid via the informer.

    Parameters
    ----------
    cp, cn, phi : shape ``(Nt, N_spatial)`` — fields on the space-time grid.
    w_xyt : shape ``(Nt, N_spatial)`` — outer product ``wt ⊗ w_norm``.
    xy : shape ``(N_spatial, 2)`` — 2D spatial nodes.
    t_grid : shape ``(Nt,)`` — time nodes.
    """
    f1, f2 = pnp_2d_sources_spacetime(xy, t_grid)
    res = informer.forward({"cp": cp, "cn": cn, "phi": phi, "f1": f1, "f2": f2})
    r_cp, r_cn, r_phi = res["res_cp"], res["res_cn"], res["res_phi"]
    return (
        (w_xyt * r_cp**2).sum(),
        (w_xyt * r_cn**2).sum(),
        (w_xyt * r_phi**2).sum(),
    )


def pnp_2d_exact(xy: torch.Tensor, t: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact solution at 2D nodes xy (N×2) at scalar time t."""
    x, y = xy[:, 0], xy[:, 1]
    et = math.exp(-t)
    cp = torch.sin(PI * x) * torch.sin(PI * y) * et
    cn = torch.cos(PI * x) * torch.cos(PI * y) * et
    phi = torch.sin(PI * x) * torch.cos(PI * y) * et / (2.0 * PI**2)
    return cp, cn, phi


def pnp_2d_sources(xy: torch.Tensor, t: float):
    """Manufactured source terms f₁, f₂ at 2D nodes and scalar time t."""
    x, y = xy[:, 0], xy[:, 1]
    et = math.exp(-t)
    e2t = math.exp(-2.0 * t)
    sx, sy = torch.sin(PI * x), torch.sin(PI * y)
    cx, cy = torch.cos(PI * x), torch.cos(PI * y)

    p = sx * sy
    dp_dx = PI * cx * sy
    dp_dy = PI * sx * cy
    lap_p = -2.0 * PI**2 * p

    n = cx * cy
    dn_dx = -PI * sx * cy
    dn_dy = -PI * cx * sy
    lap_n = -2.0 * PI**2 * n

    phi0 = sx * cy / (2.0 * PI**2)
    dphi_dx = cy * cx / (2.0 * PI)
    dphi_dy = -sx * sy / (2.0 * PI)
    lap_phi = -n  # since ∇²φ = -c_n at t=0 e^{-t} (manufactured)

    f1 = et * (-p - lap_p) + e2t * (-(dp_dx * dphi_dx + dp_dy * dphi_dy) - p * lap_phi)
    f2 = et * (-n - lap_n) + e2t * (dn_dx * dphi_dx + dn_dy * dphi_dy + n * lap_phi)
    return f1, f2


def pnp_2d_residuals(
    cp: torch.Tensor,
    cn: torch.Tensor,
    phi: torch.Tensor,
    dcp_dt: torch.Tensor,
    dcn_dt: torch.Tensor,
    lap: torch.Tensor,
    D1x: torch.Tensor,
    D1y: torch.Tensor,
    w_norm: torch.Tensor,
    xy: torch.Tensor,
    t: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Weighted quadrature losses for the 2D unsteady PNP system.

    Parameters
    ----------
    cp, cn, phi : shape ``(N,)`` — fields on flat 2D grid.
    dcp_dt, dcn_dt : shape ``(N,)`` — time derivatives.
    lap : shape ``(N, N)`` — 2D Laplacian operator from DVRMapper2D.
    D1x, D1y : shape ``(N, N)`` — 2D first-derivative operators.
    w_norm : shape ``(N,)`` — normalised 2D quadrature weights.
    xy : shape ``(N, 2)`` — 2D physical nodes.
    t : scalar time value.
    """
    f1, f2 = pnp_2d_sources(xy, t)

    D1x_phi = D1x @ phi
    D1y_phi = D1y @ phi
    lap_phi = lap @ phi
    lap_cp = lap @ cp
    lap_cn = lap @ cn
    D1x_cp = D1x @ cp
    D1y_cp = D1y @ cp
    D1x_cn = D1x @ cn
    D1y_cn = D1y @ cn

    R_cp = dcp_dt - lap_cp - (D1x_cp * D1x_phi + D1y_cp * D1y_phi) - cp * lap_phi - f1
    R_cn = dcn_dt - lap_cn + (D1x_cn * D1x_phi + D1y_cn * D1y_phi) + cn * lap_phi - f2
    R_phi = lap_phi + cp - cn

    l_cp = w_norm @ (R_cp**2)
    l_cn = w_norm @ (R_cn**2)
    l_phi = w_norm @ (R_phi**2)
    return l_cp, l_cn, l_phi
