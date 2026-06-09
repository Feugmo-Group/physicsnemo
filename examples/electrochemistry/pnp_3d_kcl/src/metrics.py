# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metrics for the 3D steady PNP / KCl example."""

from __future__ import annotations

import torch

from src.physics import pnp_3d_exact


def compute_errors(
    c_K: torch.Tensor,
    c_Cl: torch.Tensor,
    phi: torch.Tensor,
    xyz: torch.Tensor,
    weights: torch.Tensor,
) -> dict[str, float]:
    cK_ex, cCl_ex, phi_ex = pnp_3d_exact(xyz)
    w = weights / weights.sum()
    results = {}
    for name, u, u_ex in [("cK", c_K, cK_ex), ("cCl", c_Cl, cCl_ex), ("phi", phi, phi_ex)]:
        err = (u - u_ex).abs()
        results[f"Linf_{name}"] = err.max().item()
        results[f"L2_{name}"] = (w @ err**2).sqrt().item()
    return results
