# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metrics for the 2D unsteady PNP example."""

from __future__ import annotations

import torch

from src.physics import pnp_2d_exact


def compute_errors(
    cp: torch.Tensor,
    cn: torch.Tensor,
    phi: torch.Tensor,
    xy: torch.Tensor,
    w: torch.Tensor,
    t: float,
) -> dict[str, float]:
    """L∞ and L² errors at time t."""
    cp_ex, cn_ex, phi_ex = pnp_2d_exact(xy, t)
    w_norm = w / w.sum()
    results = {}
    for name, u, u_ex in [("cp", cp, cp_ex), ("cn", cn, cn_ex), ("phi", phi, phi_ex)]:
        err = (u - u_ex).abs()
        results[f"Linf_{name}"] = err.max().item()
        results[f"L2_{name}"] = (w_norm @ err**2).sqrt().item()
    return results
