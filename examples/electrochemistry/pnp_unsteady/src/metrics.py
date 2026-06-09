# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metrics for the 1D unsteady PNP example."""

from __future__ import annotations

import torch

from src.physics import pnp_unsteady_exact


def compute_errors(
    cp: torch.Tensor,
    cn: torch.Tensor,
    phi: torch.Tensor,
    x_grid: torch.Tensor,
    t_grid: torch.Tensor,
    w_xt: torch.Tensor,
) -> dict[str, float]:
    """L∞ and L² errors on the full space-time grid.

    Parameters
    ----------
    cp, cn, phi : shape ``(Nt, Nx)``.
    x_grid, t_grid : 1D grids.
    w_xt : shape ``(Nt, Nx)`` — outer product of normalised weights.
    """
    x2d = x_grid.unsqueeze(0).expand_as(cp)
    t2d = t_grid.unsqueeze(1).expand_as(cp)
    cp_ex, cn_ex, phi_ex = pnp_unsteady_exact(x2d, t2d)

    results = {}
    for name, u, u_ex in [("cp", cp, cp_ex), ("cn", cn, cn_ex), ("phi", phi, phi_ex)]:
        err = (u - u_ex).abs()
        results[f"Linf_{name}"] = err.max().item()
        results[f"L2_{name}"] = (w_xt * err**2).sum().sqrt().item()
    return results
