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

"""Consistency: mixed Kolmogorov residuals reduce to the single-field form.

Supplying the consistent vorticity ``ω = ∇²ψ`` makes ``res_w`` vanish and
``res_vort`` equal the original single-field vorticity-equation residual.
"""

from __future__ import annotations

import math
import os
import sys

import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from physicsnemo.experimental.models.scen import DVRMapper2D  # noqa: E402

from src.physics import (  # noqa: E402
    kolmogorov_forcing,
    kolmogorov_residuals_dvr,
    make_kolmogorov_informer,
)


def test_mixed_form_is_consistent():
    """Consistent ω gives res_w ≈ 0 and res_vort ≈ the single-field residual."""
    torch.set_default_dtype(torch.float64)
    m = DVRMapper2D(7, 0.0, 2 * math.pi, Ny=7, ay=0.0, by=2 * math.pi, dtype=torch.float64)
    xy, lap = m.xy_nodes, m.laplacian
    d1x, d1y, d2x, d2y = m.D1x, m.D1y, m.D2x, m.D2y
    w_norm = m.weights / m.weights.sum()
    nu, n_force = 0.1, 4
    x_, y_ = xy[:, 0], xy[:, 1]
    psi = -torch.sin(n_force * y_) / (nu * n_force**2) + 0.05 * torch.cos(x_) * torch.sin(y_)
    omega = lap @ psi

    force = kolmogorov_forcing(xy, n_force)
    r_single = (
        -nu * (lap @ omega)
        + ((d1x @ psi) * (d1y @ omega) - (d1y @ psi) * (d1x @ omega))
        - force
    )

    informer = make_kolmogorov_informer(d1x, d1y, d2x, d2y, nu, device=str(xy.device))
    res = informer.forward({"psi": psi, "omega": omega, "F": force})
    assert res["res_w"].abs().max().item() < 1e-10
    scale = r_single.abs().max().item()
    assert (res["res_vort"] - r_single).abs().max().item() < 1e-6 * scale

    l_w, l_vort = kolmogorov_residuals_dvr(informer, psi, omega, w_norm, xy, n_force)
    assert torch.isfinite(l_w) and torch.isfinite(l_vort)
