# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Steady Cahn-Hilliard (conserved phase-field) residuals.

PDE on x ∈ [-1, 1]:
    ∇²μ = 0,   μ = f'(c) − ε²∇²c
where f(c) = c²(1−c)²/4   (standard double-well potential)

Equivalently:
    R = D2 @ f'(c) − ε² D4 @ c = 0

Boundary conditions (no-flux):
    c'(-1) = c'(1) = 0
    μ'(-1) = μ'(1) = 0

A steady solution with prescribed mean composition c̄ and two-phase
separation is obtained by starting from a random field and constraining
the mean via a Lagrange multiplier term.
"""

from __future__ import annotations

import sympy as sp
import torch

from physicsnemo.experimental.models.scen import AxisOperator, DVRPhysicsInformer
from physicsnemo.sym.eq.pde import PDE


def _f_prime(c: torch.Tensor) -> torch.Tensor:
    """f'(c) = c(1-c)(1-2c) for f = c²(1-c)²/4."""
    return c * (1.0 - c) * (1.0 - 2.0 * c)


class CahnHilliardMixedPDE(PDE):
    r"""Mixed (auxiliary-field) Cahn-Hilliard system for :class:`DVRPhysicsInformer`.

    The 4th-order steady Cahn-Hilliard equation ``∇²μ = 0``, ``μ = f'(c) − ε²∇²c``
    is recast as a **coupled second-order** system in the two fields ``c`` and the
    chemical potential ``mu``:

    .. math::
        \mu - f'(c) + \varepsilon^2 c_{xx} = 0, \qquad \mu_{xx} = 0.

    This removes the biharmonic operator entirely (only a second-derivative
    operator is needed) and fits the unified informer API.  At a converged
    solution the two residuals together reproduce the original single-field
    statement ``D2 f'(c) − ε² D4 c = 0``.

    Parameters
    ----------
    eps_sq : float
        Squared interface-width parameter ε².
    """

    name = "CahnHilliardMixed"

    def __init__(self, eps_sq: float):
        self.dim = 1
        x = sp.Symbol("x")
        c = sp.Function("c")(x)
        mu = sp.Function("mu")(x)
        f_prime = c * (1 - c) * (1 - 2 * c)
        self.equations = {
            "res_mu": mu - f_prime + sp.Number(eps_sq) * c.diff(x, 2),
            "res_ch": mu.diff(x, 2),
        }


def make_ch_informer(
    eps_sq: float, D2: torch.Tensor, device: str | None = None
) -> DVRPhysicsInformer:
    """Build a DVR-collocation informer for the mixed Cahn-Hilliard system.

    Parameters
    ----------
    eps_sq : float
        Squared interface-width parameter ε².
    D2 : torch.Tensor
        Second-derivative DVR matrix, shape ``(N, N)``.
    device : str or None, optional
        Device for the informer.

    Returns
    -------
    DVRPhysicsInformer
    """
    return DVRPhysicsInformer(
        required_outputs=["res_mu", "res_ch"],
        equations=CahnHilliardMixedPDE(eps_sq),
        operators={"x": AxisOperator(axis=0, D1=None, D2=D2)},
        device=device,
    )


def cahn_hilliard_residuals_dvr(
    informer: DVRPhysicsInformer,
    c: torch.Tensor,
    mu: torch.Tensor,
    w_norm: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Weighted residual losses for the mixed Cahn-Hilliard system.

    Parameters
    ----------
    informer : DVRPhysicsInformer
        Built by :func:`make_ch_informer`.
    c, mu : torch.Tensor
        Composition and chemical-potential fields on the DVR nodes.
    w_norm : torch.Tensor
        Normalised quadrature weights, shape ``(N,)``.

    Returns
    -------
    (l_mu, l_ch) : tuple of torch.Tensor
        Scalar weighted MSEs of the chemical-potential definition and the
        ``∇²μ = 0`` residual, respectively.
    """
    res = informer.forward({"c": c, "mu": mu})
    r_mu = res["res_mu"].reshape(-1)
    r_ch = res["res_ch"].reshape(-1)
    return w_norm @ (r_mu * r_mu), w_norm @ (r_ch * r_ch)


def cahn_hilliard_residual(
    c: torch.Tensor,
    D2: torch.Tensor,
    D4: torch.Tensor,
    w_norm: torch.Tensor,
    eps_sq: float,
) -> torch.Tensor:
    """Weighted residual loss for ∇²μ = 0, μ = f'(c) − ε²∇²c."""
    fp = _f_prime(c)
    R = D2 @ fp - eps_sq * (D4 @ c)
    return w_norm @ (R * R)


def cahn_hilliard_noflux_bc_mixed(
    c: torch.Tensor, mu: torch.Tensor, D1: torch.Tensor
) -> torch.Tensor:
    """No-flux BCs for the mixed system: c'(±1) = 0, μ'(±1) = 0.

    Unlike :func:`cahn_hilliard_noflux_bc_loss` (which derives μ from c), here μ
    is an independently trained field, so its slope is taken directly.
    """
    cp = D1 @ c
    mup = D1 @ mu
    return cp[0] ** 2 + cp[-1] ** 2 + mup[0] ** 2 + mup[-1] ** 2


def cahn_hilliard_noflux_bc_loss(
    c: torch.Tensor,
    D1: torch.Tensor,
    D2: torch.Tensor,
    eps_sq: float,
) -> torch.Tensor:
    """No-flux BCs: c'(±1) = 0, μ'(±1) = 0."""
    mu = _f_prime(c) - eps_sq * (D2 @ c)
    cp = D1 @ c
    mup = D1 @ mu
    return cp[0]**2 + cp[-1]**2 + mup[0]**2 + mup[-1]**2


def cahn_hilliard_mass_loss(c: torch.Tensor, w_norm: torch.Tensor, c_mean: float) -> torch.Tensor:
    """Mass conservation: (w_norm @ c − c_mean)²."""
    return (w_norm @ c - c_mean) ** 2
