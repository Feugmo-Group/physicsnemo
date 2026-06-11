# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""1D Allen-Cahn PDE residuals and exact solution.

PDE on x ∈ [-1, 1]:
    ε²u'' − (u³ − u) = 0
    u(−1) = −1,  u(1) = 1

Exact solution:
    u*(x) = tanh(x / (ε√2))

The interface width scales as ~ε√2; KTE node clustering at x = 0
(where the interface sits) is required for small ε.
"""

from __future__ import annotations

import math

import sympy as sp
import torch

from physicsnemo.experimental.models.scen import AxisOperator, DVRPhysicsInformer
from physicsnemo.sym.eq.pde import PDE


class AllenCahnPDE(PDE):
    """Symbolic statement of ε²u'' − (u³ − u) = 0 for :class:`DVRPhysicsInformer`.

    Building the residual as a :class:`~physicsnemo.sym.eq.pde.PDE` lets the
    DVR-collocation informer assemble it from precomputed differentiation
    operators, identically to the hand-rolled :func:`allen_cahn_residual`.

    Parameters
    ----------
    eps_sq : float
        Squared interface-width parameter ε².
    """

    name = "AllenCahn"

    def __init__(self, eps_sq: float):
        self.dim = 1
        x = sp.Symbol("x")
        u = sp.Function("u")(x)
        self.equations = {"allen_cahn": sp.Number(eps_sq) * u.diff(x, 2) - (u**3 - u)}


def make_allen_cahn_informer(
    eps_sq: float, D2: torch.Tensor, device: str | None = None
) -> DVRPhysicsInformer:
    """Build a DVR-collocation informer for the Allen-Cahn residual.

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
        Informer whose ``forward({"u": u})`` returns the residual field.
    """
    return DVRPhysicsInformer(
        required_outputs=["allen_cahn"],
        equations=AllenCahnPDE(eps_sq),
        operators={"x": AxisOperator(axis=0, D1=None, D2=D2)},
        device=device,
    )


def allen_cahn_residual_dvr(
    informer: DVRPhysicsInformer, u: torch.Tensor, w_norm: torch.Tensor
) -> torch.Tensor:
    """Weighted quadrature loss for ε²u'' − (u³ − u) = 0 via the DVR informer.

    Parameters
    ----------
    informer : DVRPhysicsInformer
        Built by :func:`make_allen_cahn_informer`.
    u : torch.Tensor
        Field values on the DVR nodes.
    w_norm : torch.Tensor
        Normalised quadrature weights, shape ``(N,)``.

    Returns
    -------
    torch.Tensor
        Scalar weighted mean-squared residual.
    """
    R = informer.forward({"u": u})["allen_cahn"].reshape(-1)
    return w_norm @ (R * R)


def allen_cahn_residual(
    u: torch.Tensor,
    D2: torch.Tensor,
    w_norm: torch.Tensor,
    eps_sq: float,
) -> torch.Tensor:
    """Weighted quadrature loss for ε²u'' − (u³ − u) = 0 (hand-rolled reference)."""
    R = eps_sq * (D2 @ u) - (u**3 - u)
    return w_norm @ (R * R)


def allen_cahn_bc_loss(u: torch.Tensor) -> torch.Tensor:
    """Dirichlet penalty: u(-1) = -1, u(1) = 1."""
    return (u[0] + 1.0) ** 2 + (u[-1] - 1.0) ** 2


def allen_cahn_exact(x: torch.Tensor, eps_sq: float) -> torch.Tensor:
    """Exact solution u*(x) = tanh(x / (ε√2))."""
    eps = math.sqrt(eps_sq)
    return torch.tanh(x / (eps * math.sqrt(2.0)))
