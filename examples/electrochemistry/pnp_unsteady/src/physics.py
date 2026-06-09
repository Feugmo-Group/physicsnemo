# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""1D time-dependent Poisson-Nernst-Planck residuals and exact solution.

Problem on x ∈ [0, 1], t ∈ [0, 1], q₁ = +1, q₂ = −1:

    ∂c_p/∂t = ∂²c_p/∂x² + ∂c_p/∂x · ∂φ/∂x + c_p · ∂²φ/∂x²  + f₁
    ∂c_n/∂t = ∂²c_n/∂x² − ∂c_n/∂x · ∂φ/∂x − c_n · ∂²φ/∂x²  + f₂
    ∂²φ/∂x²  = −c_p + c_n

Manufactured exact solution:
    c_p(x,t) = x²(1−x)² e^{−t}
    c_n(x,t) = x²(1−x)³ e^{−t}
    φ(x,t)   = −(10x⁷ − 28x⁶ + 21x⁵) e^{−t} / 420

Space-time SCEN approach:
- Build a tensor-product LGL grid (x, t) using two DVRMappers.
- Represent c_p(x,t), c_n(x,t), φ(x,t) on the flat (Nx·Nt,) grid.
- Spatial derivatives via precomputed D1x, D2x (applied along x-axis).
- Time derivatives via precomputed D1t (applied along t-axis).
"""

from __future__ import annotations

import torch


def pnp_unsteady_exact(x: torch.Tensor, t: torch.Tensor):
    """Exact solution at (x, t). Both tensors must have the same shape."""
    et = torch.exp(-t)
    cp_ex = x**2 * (1 - x)**2 * et
    cn_ex = x**2 * (1 - x)**3 * et
    phi_ex = -(10 * x**7 - 28 * x**6 + 21 * x**5) * et / 420.0
    return cp_ex, cn_ex, phi_ex


def pnp_unsteady_sources(x: torch.Tensor, t: torch.Tensor):
    """Manufactured source terms f₁, f₂ at (x, t)."""
    et = torch.exp(-t)
    e2t = torch.exp(-2.0 * t)

    p = x**2 * (1 - x)**2
    pp = 2 * x * (1 - x) * (1 - 2 * x)
    ppp = 2 * (1 - 6 * x + 6 * x**2)

    n = x**2 * (1 - x)**3
    np_ = 2 * x - 9 * x**2 + 12 * x**3 - 5 * x**4
    npp = 2 - 18 * x + 36 * x**2 - 20 * x**3

    phi_p = -x**4 * (10 * x**2 - 24 * x + 15) / 60.0
    phi_pp = -x**3 * (1 - x)**2

    f1 = et * (-p - ppp) + e2t * (-pp * phi_p - p * phi_pp)
    f2 = et * (-n - npp) + e2t * (np_ * phi_p + n * phi_pp)
    return f1, f2


def pnp_unsteady_residuals(
    cp: torch.Tensor,
    cn: torch.Tensor,
    phi: torch.Tensor,
    D1x: torch.Tensor,
    D2x: torch.Tensor,
    D1t: torch.Tensor,
    w_xt: torch.Tensor,
    x_grid: torch.Tensor,
    t_grid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Weighted residual losses on the (Nt × Nx) space-time grid.

    Parameters
    ----------
    cp, cn, phi : shape ``(Nt, Nx)`` — fields on the tensor-product grid.
    D1x, D2x   : shape ``(Nx, Nx)`` — spatial derivative matrices.
    D1t        : shape ``(Nt, Nt)`` — time derivative matrix.
    w_xt       : shape ``(Nt, Nx)`` — outer product of normalised weights.
    x_grid     : shape ``(Nx,)`` — physical x-nodes.
    t_grid     : shape ``(Nt,)`` — physical t-nodes.
    """
    x2d = x_grid.unsqueeze(0).expand_as(cp)
    t2d = t_grid.unsqueeze(1).expand_as(cp)
    f1, f2 = pnp_unsteady_sources(x2d, t2d)

    # Time derivatives via spectral matrix (applied row-wise over x)
    dcp_dt = D1t @ cp      # (Nt, Nx)
    dcn_dt = D1t @ cn

    # Spatial derivatives (applied col-wise over t)
    D1cp = cp @ D1x.T
    D2cp = cp @ D2x.T
    D1cn = cn @ D1x.T
    D2cn = cn @ D2x.T
    D1phi = phi @ D1x.T
    D2phi = phi @ D2x.T

    R_cp = dcp_dt - D2cp - D1cp * D1phi - cp * D2phi - f1
    R_cn = dcn_dt - D2cn + D1cn * D1phi + cn * D2phi - f2
    R_phi = D2phi + cp - cn

    l_cp = (w_xt * R_cp**2).sum()
    l_cn = (w_xt * R_cn**2).sum()
    l_phi = (w_xt * R_phi**2).sum()
    return l_cp, l_cn, l_phi


def pnp_unsteady_ic_loss(
    cp: torch.Tensor,
    cn: torch.Tensor,
    phi: torch.Tensor,
    x_grid: torch.Tensor,
    t0_idx: int = 0,
) -> torch.Tensor:
    """Initial condition loss: fields at t[0] match exact solution at t=0."""
    t0 = torch.zeros_like(x_grid)
    cp0_ex, cn0_ex, phi0_ex = pnp_unsteady_exact(x_grid, t0)
    cp0 = cp[t0_idx]
    cn0 = cn[t0_idx]
    phi0 = phi[t0_idx]
    return ((cp0 - cp0_ex)**2 + (cn0 - cn0_ex)**2 + (phi0 - phi0_ex)**2).mean()
