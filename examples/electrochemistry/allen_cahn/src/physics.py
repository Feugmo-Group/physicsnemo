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

import torch


def allen_cahn_residual(
    u: torch.Tensor,
    D2: torch.Tensor,
    w_norm: torch.Tensor,
    eps_sq: float,
) -> torch.Tensor:
    """Weighted quadrature loss for ε²u'' − (u³ − u) = 0."""
    R = eps_sq * (D2 @ u) - (u**3 - u)
    return w_norm @ (R * R)


def allen_cahn_bc_loss(u: torch.Tensor) -> torch.Tensor:
    """Dirichlet penalty: u(-1) = -1, u(1) = 1."""
    return (u[0] + 1.0) ** 2 + (u[-1] - 1.0) ** 2


def allen_cahn_exact(x: torch.Tensor, eps_sq: float) -> torch.Tensor:
    """Exact solution u*(x) = tanh(x / (ε√2))."""
    eps = math.sqrt(eps_sq)
    return torch.tanh(x / (eps * math.sqrt(2.0)))
