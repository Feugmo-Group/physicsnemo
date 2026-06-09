# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Error metrics for the 1D steady PNP example."""

from __future__ import annotations

import torch

from src.physics import pnp_steady_exact


def compute_errors(
    cp: torch.Tensor,
    cn: torch.Tensor,
    phi: torch.Tensor,
    x: torch.Tensor,
    weights: torch.Tensor,
) -> dict[str, float]:
    """Compute L∞ and L² errors versus the exact solution.

    Parameters
    ----------
    cp, cn, phi : shape ``(N,)`` — computed solution.
    x : shape ``(N,)`` — physical LGL nodes.
    weights : shape ``(N,)`` — physical quadrature weights (not normalised).

    Returns
    -------
    dict with keys ``"Linf_cp"``, ``"L2_cp"``, etc.
    """
    cp_ex, cn_ex, phi_ex = pnp_steady_exact(x)
    w = weights / weights.sum()

    results = {}
    for name, u, u_ex in [("cp", cp, cp_ex), ("cn", cn, cn_ex), ("phi", phi, phi_ex)]:
        err = (u - u_ex).abs()
        results[f"Linf_{name}"] = err.max().item()
        results[f"L2_{name}"] = (w @ (err**2)).sqrt().item()
    return results


def print_errors(errors: dict[str, float]) -> None:
    """Pretty-print error table."""
    header = f"{'Field':<8}  {'L∞':>12}  {'L²':>12}"
    print(header)
    print("-" * len(header))
    for field in ("cp", "cn", "phi"):
        linf = errors[f"Linf_{field}"]
        l2 = errors[f"L2_{field}"]
        print(f"{field:<8}  {linf:12.3e}  {l2:12.3e}")
