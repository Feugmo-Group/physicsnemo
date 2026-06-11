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

import sympy as sp
import torch

from physicsnemo.experimental.models.scen import AxisOperator, DVRPhysicsInformer
from physicsnemo.sym.eq.pde import PDE

PI = math.pi


class Pnp3DPDE(PDE):
    """Symbolic 3D steady PNP system for :class:`DVRPhysicsInformer`.

    Three coupled fields ``cK``, ``cCl``, ``phi`` on a unit cube; the
    manufactured forcings enter as leaf functions ``fK``, ``fCl`` supplied at
    evaluation time.
    """

    name = "Pnp3D"

    def __init__(self):
        self.dim = 3
        x, y, z = sp.symbols("x y z")
        cK = sp.Function("cK")(x, y, z)
        cCl = sp.Function("cCl")(x, y, z)
        phi = sp.Function("phi")(x, y, z)
        fK = sp.Function("fK")(x, y, z)
        fCl = sp.Function("fCl")(x, y, z)

        def lap(f):
            return f.diff(x, 2) + f.diff(y, 2) + f.diff(z, 2)

        def grad_dot(a, b):
            return (
                a.diff(x, 1) * b.diff(x, 1)
                + a.diff(y, 1) * b.diff(y, 1)
                + a.diff(z, 1) * b.diff(z, 1)
            )

        self.equations = {
            "res_K": -lap(cK) - (grad_dot(cK, phi) + cK * lap(phi)) - fK,
            "res_Cl": -lap(cCl) + (grad_dot(cCl, phi) + cCl * lap(phi)) - fCl,
            "res_phi": -lap(phi) - (cK - cCl),
        }


def make_pnp_3d_informer(
    D1x: torch.Tensor,
    D1y: torch.Tensor,
    D1z: torch.Tensor,
    D2x: torch.Tensor,
    D2y: torch.Tensor,
    D2z: torch.Tensor,
    device: str | None = None,
) -> DVRPhysicsInformer:
    """Build a DVR-collocation informer for the 3D steady PNP system.

    Parameters
    ----------
    D1x, D1y, D1z : torch.Tensor
        First-derivative DVR operators (flattened 3D), each ``(N, N)``.
    D2x, D2y, D2z : torch.Tensor
        Second-derivative DVR operators (flattened 3D), each ``(N, N)``.
    device : str or None, optional
        Device for the informer.

    Returns
    -------
    DVRPhysicsInformer
    """
    return DVRPhysicsInformer(
        required_outputs=["res_K", "res_Cl", "res_phi"],
        equations=Pnp3DPDE(),
        operators={
            "x": AxisOperator(axis=0, D1=D1x, D2=D2x),
            "y": AxisOperator(axis=0, D1=D1y, D2=D2y),
            "z": AxisOperator(axis=0, D1=D1z, D2=D2z),
        },
        device=device,
    )


def pnp_3d_residuals_dvr(
    informer: DVRPhysicsInformer,
    c_K: torch.Tensor,
    c_Cl: torch.Tensor,
    phi: torch.Tensor,
    w_norm: torch.Tensor,
    xyz: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Weighted quadrature residuals for the 3D steady PNP system via the informer."""
    f_K, f_Cl = pnp_3d_sources(xyz)
    res = informer.forward(
        {"cK": c_K, "cCl": c_Cl, "phi": phi, "fK": f_K, "fCl": f_Cl}
    )
    r_k = res["res_K"].reshape(-1)
    r_cl = res["res_Cl"].reshape(-1)
    r_phi = res["res_phi"].reshape(-1)
    return w_norm @ (r_k * r_k), w_norm @ (r_cl * r_cl), w_norm @ (r_phi * r_phi)


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
