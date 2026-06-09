# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Error metrics for the Allen-Cahn example."""

from __future__ import annotations

import torch

from src.physics import allen_cahn_exact


def compute_errors(
    u: torch.Tensor,
    x: torch.Tensor,
    weights: torch.Tensor,
    eps_sq: float,
) -> dict[str, float]:
    """Return L∞ and L² errors vs. exact solution."""
    u_ex = allen_cahn_exact(x, eps_sq)
    w = weights / weights.sum()
    err = (u - u_ex).abs()
    return {
        "Linf": err.max().item(),
        "L2": (w @ err**2).sqrt().item(),
    }
