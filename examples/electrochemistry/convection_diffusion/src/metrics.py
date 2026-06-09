# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metrics for the convection-diffusion example."""

from __future__ import annotations

import torch

from src.physics import cd_exact


def compute_errors(
    u: torch.Tensor, x: torch.Tensor, weights: torch.Tensor, eps: float, a: float
) -> dict[str, float]:
    u_ex = cd_exact(x, eps, a)
    w = weights / weights.sum()
    err = (u - u_ex).abs()
    return {"Linf": err.max().item(), "L2": (w @ err**2).sqrt().item()}
