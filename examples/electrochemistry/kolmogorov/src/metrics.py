# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metrics for the Kolmogorov flow example."""

from __future__ import annotations

import torch

from src.physics import kolmogorov_exact


def compute_errors(
    psi: torch.Tensor,
    xy: torch.Tensor,
    weights: torch.Tensor,
    nu: float,
    n_force: int,
) -> dict[str, float]:
    psi_ex = kolmogorov_exact(xy, nu, n_force)
    w = weights / weights.sum()
    err = (psi - psi_ex).abs()
    return {"Linf": err.max().item(), "L2": (w @ err**2).sqrt().item()}
