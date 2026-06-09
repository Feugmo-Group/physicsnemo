# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""1D convection-diffusion residuals and exact solution.

PDE on x ∈ [0, 1]:
    ε u'' + a u' = 0
    u(0) = 0,  u(1) = 1

Exact solution:
    u(x) = expm1(a·x/ε) / expm1(a/ε)

A boundary layer of thickness ~ε/a forms at x = 1 for a > 0.
KTE clustering (alpha ≈ 0.85) at the right endpoint resolves it.
"""

from __future__ import annotations

import torch


def cd_residual(
    u: torch.Tensor,
    D1: torch.Tensor,
    D2: torch.Tensor,
    w_norm: torch.Tensor,
    eps: float,
    a: float,
) -> torch.Tensor:
    """Weighted residual for ε u'' + a u' = 0."""
    R = eps * (D2 @ u) + a * (D1 @ u)
    return w_norm @ (R * R)


def cd_bc_loss(u: torch.Tensor) -> torch.Tensor:
    """Dirichlet BCs: u(0) = 0, u(1) = 1."""
    return u[0] ** 2 + (u[-1] - 1.0) ** 2


def interface_loss(
    u_left: torch.Tensor,
    u_right: torch.Tensor,
    D1_left: torch.Tensor,
    D1_right: torch.Tensor,
    cond: str = "both",
) -> torch.Tensor:
    """C0 / C1 / both continuity penalty at a single element interface.

    Parameters
    ----------
    u_left  : solution values at the LEFT element's nodes
    u_right : solution values at the RIGHT element's nodes
    D1_left : physical D1 matrix for the left element (shape N_L × N_L)
    D1_right: physical D1 matrix for the right element (shape N_R × N_R)
    cond    : ``'c0'``, ``'c1'``, or ``'both'``

    Returns
    -------
    scalar loss
    """
    loss = torch.zeros(1, dtype=u_left.dtype, device=u_left.device).squeeze()
    if cond in ("c0", "both"):
        # value continuity: right end of left element == left end of right element
        loss = loss + (u_left[-1] - u_right[0]) ** 2
    if cond in ("c1", "both"):
        # derivative continuity: u'(interface⁻) == u'(interface⁺)
        du_left = (D1_left @ u_left)[-1]
        du_right = (D1_right @ u_right)[0]
        loss = loss + (du_left - du_right) ** 2
    return loss


def all_interface_losses(
    u_elements: list[torch.Tensor],
    D1_elements: list[torch.Tensor],
    cond: str = "both",
) -> torch.Tensor:
    """Sum interface losses over all K-1 internal interfaces of K elements."""
    total = torch.zeros(1, dtype=u_elements[0].dtype, device=u_elements[0].device).squeeze()
    for k in range(len(u_elements) - 1):
        total = total + interface_loss(
            u_elements[k], u_elements[k + 1],
            D1_elements[k], D1_elements[k + 1],
            cond=cond,
        )
    return total


def cd_exact(x: torch.Tensor, eps: float, a: float) -> torch.Tensor:
    """Exact solution: expm1(a·x/ε) / expm1(a/ε)."""
    numer = -torch.expm1(-a * x / eps)
    denom = float(-torch.expm1(torch.tensor(-a / eps, dtype=x.dtype, device=x.device)))
    return numer / denom
