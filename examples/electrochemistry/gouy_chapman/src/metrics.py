# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metrics for the Gouy-Chapman example."""

from __future__ import annotations

import torch

from src.physics import pb_linear_exact


def compute_errors_linear(
    psi: torch.Tensor,
    x: torch.Tensor,
    weights: torch.Tensor,
    psi_wall: float,
    kappa: float,
    L: float,
) -> dict[str, float]:
    """L∞ and L² errors for the linearized variant."""
    psi_ex = pb_linear_exact(x, psi_wall, kappa, L)
    w = weights / weights.sum()
    err = (psi - psi_ex).abs()
    return {"Linf": err.max().item(), "L2": (w @ err**2).sqrt().item()}
