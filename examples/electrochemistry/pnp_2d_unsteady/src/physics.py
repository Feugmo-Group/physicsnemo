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

import torch

PI = math.pi


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
