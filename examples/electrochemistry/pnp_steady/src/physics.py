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

import sympy as sp
import torch

from physicsnemo.experimental.models.scen import AxisOperator, DVRPhysicsInformer
from physicsnemo.sym.eq.pde import PDE

PI = math.pi


class PnpSteadyPDE(PDE):
    """Symbolic 1D steady Poisson-Nernst-Planck system for :class:`DVRPhysicsInformer`.

    Three coupled fields ``cp``, ``cn``, ``phi``; the manufactured forcing terms
    enter as leaf functions ``fv``, ``fw`` whose precomputed values are supplied
    at evaluation time (see :func:`pnp_steady_residuals_dvr`).
    """

    name = "PnpSteady"

    def __init__(self):
        self.dim = 1
        x = sp.Symbol("x")
        cp = sp.Function("cp")(x)
        cn = sp.Function("cn")(x)
        phi = sp.Function("phi")(x)
        fv = sp.Function("fv")(x)
        fw = sp.Function("fw")(x)
        c30 = sp.Number(1.0 / 30.0)
        c20 = sp.Number(1.0 / 20.0)
        pi2 = sp.Number(PI**2)
        self.equations = {
            "res_cp": cp.diff(x, 2) + pi2 * (cn + phi),
            "res_cn": cn.diff(x, 2)
            + c30 * (cn.diff(x, 1) * cp.diff(x, 1) + cn * cp.diff(x, 2))
            + fv,
            "res_phi": phi.diff(x, 2)
            + c20 * (phi.diff(x, 1) * cp.diff(x, 1) + phi * cp.diff(x, 2))
            + fw,
        }


def make_pnp_steady_informer(
    D1: torch.Tensor, D2: torch.Tensor, device: str | None = None
) -> DVRPhysicsInformer:
    """Build a DVR-collocation informer for the steady PNP system.

    Parameters
    ----------
    D1, D2 : torch.Tensor
        First/second-derivative DVR matrices (global), ``(N, N)``.
    device : str or None, optional
        Device for the informer.

    Returns
    -------
    DVRPhysicsInformer
    """
    return DVRPhysicsInformer(
        required_outputs=["res_cp", "res_cn", "res_phi"],
        equations=PnpSteadyPDE(),
        operators={"x": AxisOperator(axis=0, D1=D1, D2=D2)},
        device=device,
    )


def pnp_steady_residuals_dvr(
    informer: DVRPhysicsInformer,
    cp: torch.Tensor,
    cn: torch.Tensor,
    phi: torch.Tensor,
    w_norm: torch.Tensor,
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quadrature-weighted interior residual losses via the DVR informer."""
    f_v, f_w = pnp_steady_sources(x)
    res = informer.forward(
        {"cp": cp, "cn": cn, "phi": phi, "fv": f_v, "fw": f_w, "x": x}
    )
    r_cp = res["res_cp"].reshape(-1)
    r_cn = res["res_cn"].reshape(-1)
    r_phi = res["res_phi"].reshape(-1)
    return w_norm @ (r_cp * r_cp), w_norm @ (r_cn * r_cn), w_norm @ (r_phi * r_phi)


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
