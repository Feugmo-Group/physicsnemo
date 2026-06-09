# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metrics for the Cahn-Hilliard example."""

from __future__ import annotations

import torch


def compute_diagnostics(
    c: torch.Tensor,
    w: torch.Tensor,
    c_mean: float,
) -> dict[str, float]:
    """Return mass conservation error and bulk free energy."""
    w_norm = w / w.sum()
    c_mean_num = (w_norm @ c).item()
    mass_err = abs(c_mean_num - c_mean)
    fp = c * (1.0 - c) * (1.0 - 2.0 * c)
    free_energy = (w_norm @ (c**2 * (1.0 - c)**2 / 4.0)).item()
    return {
        "mass_error": mass_err,
        "c_mean": c_mean_num,
        "free_energy": free_energy,
        "c_min": c.min().item(),
        "c_max": c.max().item(),
    }
