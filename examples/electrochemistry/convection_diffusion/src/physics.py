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


def cd_exact(x: torch.Tensor, eps: float, a: float) -> torch.Tensor:
    """Exact solution: expm1(a·x/ε) / expm1(a/ε)."""
    numer = -torch.expm1(-a * x / eps)
    denom = float(-torch.expm1(torch.tensor(-a / eps, dtype=x.dtype, device=x.device)))
    return numer / denom
