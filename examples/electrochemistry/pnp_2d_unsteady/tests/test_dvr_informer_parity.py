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

"""Parity: DVR full-grid informer vs hand-rolled per-time-slice 2D PNP residuals."""

from __future__ import annotations

import math
import os
import sys

import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from physicsnemo.experimental.models.scen import DVRMapper, DVRMapper2D  # noqa: E402

from src.physics import (  # noqa: E402
    make_pnp_2d_informer,
    pnp_2d_residuals,
    pnp_2d_residuals_dvr,
)


def test_full_grid_informer_matches_time_loop():
    """The full-grid informer reproduces the per-slice time-loop sum to round-off."""
    torch.set_default_dtype(torch.float64)
    m2 = DVRMapper2D(5, 0.0, 1.0, Ny=5, ay=0.0, by=1.0, dtype=torch.float64)
    mt = DVRMapper(6, 0.0, 1.0, 0.0, dtype=torch.float64)
    xy, lap = m2.xy_nodes, m2.laplacian
    d1x, d1y, d2x, d2y = m2.D1x, m2.D1y, m2.D2x, m2.D2y
    w_norm = m2.weights / m2.weights.sum()
    t_grid, d1t = mt.nodes, mt.D1
    wt = mt.weights / mt.weights.sum()
    x_, y_ = xy[:, 0], xy[:, 1]

    def field(a, b):
        rows = []
        for tv in t_grid.tolist():
            rows.append(
                torch.sin(math.pi * x_) * torch.sin(math.pi * y_) * math.exp(-tv)
                + a * torch.cos(b * x_) * tv
            )
        return torch.stack(rows, 0)

    cp, cn, phi = field(0.02, 2.0), field(0.013, 3.0) + 0.01, field(0.001, 1.0)

    # Legacy per-time-slice loop.
    dcp, dcn = d1t @ cp, d1t @ cn
    loop = [0.0, 0.0, 0.0]
    for i, tv in enumerate(t_grid.tolist()):
        lc = pnp_2d_residuals(cp[i], cn[i], phi[i], dcp[i], dcn[i], lap, d1x, d1y, w_norm, xy, tv)
        for k in range(3):
            loop[k] = loop[k] + wt[i] * lc[k]

    informer = make_pnp_2d_informer(d1x, d1y, d2x, d2y, d1t, device=str(xy.device))
    w_xyt = wt.unsqueeze(1) * w_norm.unsqueeze(0)
    new = pnp_2d_residuals_dvr(informer, cp, cn, phi, w_xyt, xy, t_grid)

    for l_loop, l_new in zip(loop, new):
        assert abs(float(l_loop) - float(l_new)) <= 1e-10 * abs(float(l_loop))
