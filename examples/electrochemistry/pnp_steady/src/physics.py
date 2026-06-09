# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""1D steady Poisson-Nernst-Planck residuals and exact solution.

Physical problem on x ∈ [-3, 3]:

    c_p'' = −π²(c_n + φ)
    3000 c_n'' + 100(c_n' c_p' + c_n c_p'') + f_v(x) = 0
    1000 φ''   + 50 (φ'  c_p' + φ  c_p'') + f_w(x) = 0

Exact solution:
    c_p(x) = sin(πx) + cos(πx)
    c_n(x) = sin(πx)
    φ(x)   = cos(πx)

Boundary conditions at x = ±3:
    c_p = -1,  c_n = 0,  φ = -1
"""

from __future__ import annotations

import math

import torch

PI = math.pi


def pnp_steady_sources(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalised source terms at physical nodes x.

    Returns (f_v/3000, f_w/1000).
    """
    pi2 = PI**2
    px = PI * x
    p2x = 2.0 * PI * x
    f_v = pi2 * torch.sin(px) - (pi2 / 30.0) * (torch.cos(p2x) - torch.sin(p2x))
    f_w = pi2 * torch.cos(px) + (pi2 / 20.0) * (torch.cos(p2x) + torch.sin(p2x))
    return f_v, f_w


def pnp_steady_residuals(
    cp: torch.Tensor,
    cn: torch.Tensor,
    phi: torch.Tensor,
    D1: torch.Tensor,
    D2: torch.Tensor,
    w_norm: torch.Tensor,
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quadrature-weighted interior residual losses.

    Parameters
    ----------
    cp, cn, phi : shape ``(N,)`` — network outputs at physical LGL nodes.
    D1, D2 : shape ``(N, N)`` — physical derivative matrices.
    w_norm : shape ``(N,)`` — normalised quadrature weights.
    x : shape ``(N,)`` — physical node positions.

    Returns
    -------
    (l_cp, l_cn, l_phi) : three scalar losses = ``w_norm @ R²``.
    """
    f_v, f_w = pnp_steady_sources(x)

    D1_cp = D1 @ cp
    D2_cp = D2 @ cp
    D1_cn = D1 @ cn
    D2_cn = D2 @ cn
    D1_phi = D1 @ phi
    D2_phi = D2 @ phi

    R_cp = D2_cp + PI**2 * (cn + phi)
    R_cn = D2_cn + (1.0 / 30.0) * (D1_cn * D1_cp + cn * D2_cp) + f_v
    R_phi = D2_phi + (1.0 / 20.0) * (D1_phi * D1_cp + phi * D2_cp) + f_w

    l_cp = w_norm @ (R_cp * R_cp)
    l_cn = w_norm @ (R_cn * R_cn)
    l_phi = w_norm @ (R_phi * R_phi)
    return l_cp, l_cn, l_phi


def pnp_steady_exact(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact solution at physical nodes x.

    Returns (cp_exact, cn_exact, phi_exact), each shape ``(N,)``.
    """
    px = PI * x
    return torch.sin(px) + torch.cos(px), torch.sin(px), torch.cos(px)


def pnp_steady_bc_loss(
    cp: torch.Tensor,
    cn: torch.Tensor,
    phi: torch.Tensor,
) -> torch.Tensor:
    """Dirichlet boundary penalty: squared mismatch at both endpoints.

    Expects the first and last entries of each field to be the boundary nodes.
    Exact BCs: cp(±3)=-1, cn(±3)=0, phi(±3)=-1.
    """
    loss = (
        (cp[0] + 1.0) ** 2
        + (cp[-1] + 1.0) ** 2
        + cn[0] ** 2
        + cn[-1] ** 2
        + (phi[0] + 1.0) ** 2
        + (phi[-1] + 1.0) ** 2
    )
    return loss
