# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""3D steady PNP for KCl electrolyte using DVRMapper3D.

Dimensionless steady-state Poisson-Nernst-Planck for a 1:1 electrolyte
(K⁺ / Cl⁻) on a unit cube [0,1]³:

    −∇²c_K  − ∇·(c_K ∇φ) = f_K(x,y,z)
    −∇²c_Cl + ∇·(c_Cl ∇φ) = f_Cl(x,y,z)
    −∇²φ = (c_K − c_Cl)

Manufactured exact solution:
    c_K(x,y,z)  = 1 + 0.1 sin(πx) sin(πy) sin(πz)
    c_Cl(x,y,z) = 1 − 0.1 sin(πx) sin(πy) sin(πz)
    φ(x,y,z)    = 0.1 cos(πx) cos(πy) cos(πz) / (3π²)

(The ±0.1 amplitude keeps concentrations positive and the solution
linear enough that the coupling terms f_K, f_Cl are small.)
"""

from __future__ import annotations

import math

import torch

PI = math.pi


def pnp_3d_exact(xyz: torch.Tensor):
    """Exact (manufactured) solution at 3D nodes xyz (N×3)."""
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    sxyz = torch.sin(PI * x) * torch.sin(PI * y) * torch.sin(PI * z)
    cxyz = torch.cos(PI * x) * torch.cos(PI * y) * torch.cos(PI * z)
    c_K = 1.0 + 0.1 * sxyz
    c_Cl = 1.0 - 0.1 * sxyz
    phi = 0.1 * cxyz / (3.0 * PI**2)
    return c_K, c_Cl, phi


def pnp_3d_sources(xyz: torch.Tensor):
    """Compute manufactured source terms f_K, f_Cl at xyz (N×3)."""
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    sx, sy, sz = torch.sin(PI * x), torch.sin(PI * y), torch.sin(PI * z)
    cx, cy, cz = torch.cos(PI * x), torch.cos(PI * y), torch.cos(PI * z)

    sxyz = sx * sy * sz
    cxyz = cx * cy * cz

    lap_s = -3.0 * PI**2 * sxyz     # ∇²sin(πx)sin(πy)sin(πz) = -3π²s
    lap_c = -3.0 * PI**2 * cxyz

    c_K = 1.0 + 0.1 * sxyz
    c_Cl = 1.0 - 0.1 * sxyz
    phi = 0.1 * cxyz / (3.0 * PI**2)

    # Gradients of phi
    dphi_dx = -0.1 * cx * cy * cz * PI / (3.0 * PI**2)  # = -0.1 cx·cy·cz / (3π)
    dphi_dy = -0.1 * sx * sy * cz * PI / (3.0 * PI**2)  # wait — need correct formula
    # phi = 0.1/(3π²) cos(πx)cos(πy)cos(πz)
    # ∂phi/∂x = -0.1π/(3π²) sin(πx)cos(πy)cos(πz) = -0.1/(3π) sin(πx)cos(πy)cos(πz)
    dphi_dx = -0.1 / (3.0 * PI) * sx * cy * cz
    dphi_dy = -0.1 / (3.0 * PI) * cx * sy * cz
    dphi_dz = -0.1 / (3.0 * PI) * cx * cy * sz

    # ∇c_K = 0.1π (cx·sy·sz, sx·cy·sz, sx·sy·cz)
    dcK_dx = 0.1 * PI * cx * sy * sz
    dcK_dy = 0.1 * PI * sx * cy * sz
    dcK_dz = 0.1 * PI * sx * sy * cz

    # ∇·(c_K ∇φ) = ∇c_K · ∇φ + c_K ∇²φ
    # ∇²φ = -0.1/(π²) · (lap_c) / (3π²) ... let's compute directly
    lap_phi = 0.1 * lap_c / (3.0 * PI**2)  # = -0.1/(π²) cxyz

    div_cK_grad_phi = (dcK_dx * dphi_dx + dcK_dy * dphi_dy + dcK_dz * dphi_dz + c_K * lap_phi)
    div_cCl_grad_phi = (-dcK_dx * dphi_dx - dcK_dy * dphi_dy - dcK_dz * dphi_dz + c_Cl * lap_phi)

    f_K = -0.1 * lap_s - div_cK_grad_phi
    f_Cl = 0.1 * lap_s + div_cCl_grad_phi
    return f_K, f_Cl


def pnp_3d_residuals(
    c_K: torch.Tensor,
    c_Cl: torch.Tensor,
    phi: torch.Tensor,
    lap: torch.Tensor,
    D1x: torch.Tensor,
    D1y: torch.Tensor,
    D1z: torch.Tensor,
    w_norm: torch.Tensor,
    xyz: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Weighted quadrature residuals for the 3D steady PNP system."""
    f_K, f_Cl = pnp_3d_sources(xyz)

    lap_K = lap @ c_K
    lap_Cl = lap @ c_Cl
    lap_phi = lap @ phi
    dK_dx, dK_dy, dK_dz = D1x @ c_K, D1y @ c_K, D1z @ c_K
    dCl_dx, dCl_dy, dCl_dz = D1x @ c_Cl, D1y @ c_Cl, D1z @ c_Cl
    dphi_dx, dphi_dy, dphi_dz = D1x @ phi, D1y @ phi, D1z @ phi

    R_K = -lap_K - (dK_dx * dphi_dx + dK_dy * dphi_dy + dK_dz * dphi_dz + c_K * lap_phi) - f_K
    R_Cl = -lap_Cl + (dCl_dx * dphi_dx + dCl_dy * dphi_dy + dCl_dz * dphi_dz + c_Cl * lap_phi) - f_Cl
    R_phi = -lap_phi - (c_K - c_Cl)

    l_K = w_norm @ (R_K**2)
    l_Cl = w_norm @ (R_Cl**2)
    l_phi = w_norm @ (R_phi**2)
    return l_K, l_Cl, l_phi
