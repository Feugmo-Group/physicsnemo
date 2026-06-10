# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Film-thickness metric for the RPDM NSEM example.

Compares the predicted dimensionless film thickness ``l(y)`` against the COMSOL
reference solution in ``data/const_0.1_V.csv`` (columns ``T`` [s], ``L`` [m]).
The reference is nondimensionalised by ``tc`` (time) and ``lc`` (thickness) and
linearly interpolated onto the predicted time nodes for the L2 error.
"""

from __future__ import annotations

import csv
import os

import torch


def load_comsol_reference(
    csv_path: str, tc: float, lc: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load and nondimensionalise the COMSOL film-thickness reference.

    Returns
    -------
    y_ref : torch.Tensor — dimensionless time (= T / tc).
    l_ref : torch.Tensor — dimensionless thickness (= L / lc).
    """
    ts, ls = [], []
    with open(csv_path, newline="") as fh:
        reader = csv.reader(fh)
        next(reader)  # header
        for row in reader:
            if not row:
                continue
            ts.append(float(row[0]))
            ls.append(float(row[1]))
    y_ref = torch.tensor(ts, dtype=torch.get_default_dtype()) / tc
    l_ref = torch.tensor(ls, dtype=torch.get_default_dtype()) / lc
    return y_ref, l_ref


def _interp(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """Simple 1D linear interpolation (xp assumed sorted ascending)."""
    idx = torch.searchsorted(xp, x).clamp(1, len(xp) - 1)
    x0, x1 = xp[idx - 1], xp[idx]
    y0, y1 = fp[idx - 1], fp[idx]
    t = (x - x0) / (x1 - x0).clamp_min(1e-30)
    return y0 + t * (y1 - y0)


def compute_film_metrics(
    lvec: torch.Tensor,
    t_grid: torch.Tensor,
    tc: float,
    lc: float,
    csv_path: str | None = None,
) -> dict[str, float]:
    """L2 / L∞ error of predicted ``l(y)`` against the COMSOL reference.

    Parameters
    ----------
    lvec : shape ``(Nt,)`` — predicted dimensionless film thickness.
    t_grid : shape ``(Nt,)`` — dimensionless time nodes.
    tc, lc : characteristic time / thickness for nondimensionalisation.
    csv_path : path to ``const_0.1_V.csv``.  Defaults to ``../data/...``.
    """
    if csv_path is None:
        csv_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "const_0.1_V.csv"
        )
    results: dict[str, float] = {
        "l_final_pred": float(lvec[-1]),
        "l_min_pred": float(lvec.min()),
        "l_max_pred": float(lvec.max()),
    }
    if not os.path.exists(csv_path):
        return results

    y_ref, l_ref = load_comsol_reference(csv_path, tc, lc)
    l_ref_on_grid = _interp(t_grid.clamp(y_ref.min(), y_ref.max()), y_ref, l_ref)
    err = (lvec - l_ref_on_grid).abs()
    results["Linf_l_vs_comsol"] = float(err.max())
    results["L2_l_vs_comsol"] = float(((err**2).mean()).sqrt())
    return results
