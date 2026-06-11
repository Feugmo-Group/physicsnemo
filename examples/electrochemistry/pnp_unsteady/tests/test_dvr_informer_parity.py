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

"""Parity: DVR-collocation informer vs hand-rolled space-time PNP residuals."""

from __future__ import annotations

import os
import sys

import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from physicsnemo.experimental.models.scen import DVRMapper  # noqa: E402

from src.physics import (  # noqa: E402
    make_pnp_unsteady_informer,
    pnp_unsteady_residuals,
    pnp_unsteady_residuals_dvr,
)


def test_dvr_informer_matches_handrolled():
    """All three space-time residual losses match the hand-rolled ones to round-off."""
    torch.set_default_dtype(torch.float64)
    mx = DVRMapper(14, 0.0, 1.0, 0.0, dtype=torch.float64)
    mt = DVRMapper(11, 0.0, 1.0, 0.0, dtype=torch.float64)
    d1x, d2x, d1t = mx.D1, mx.D2, mt.D1
    xg, tg = mx.nodes, mt.nodes
    wx = mx.weights / mx.weights.sum()
    wt = mt.weights / mt.weights.sum()
    w_xt = torch.outer(wt, wx)
    nt, nx = 11, 14
    x2 = xg.unsqueeze(0).expand(nt, nx)
    t2 = tg.unsqueeze(1).expand(nt, nx)
    # Perturbed fields so no residual is trivially zero.
    cp = x2**2 * (1 - x2) ** 2 * torch.exp(-t2) + 0.01 * torch.sin(3 * x2)
    cn = x2**2 * (1 - x2) ** 3 * torch.exp(-t2) + 0.013 * torch.cos(2 * x2) * t2
    phi = -(10 * x2**7 - 28 * x2**6 + 21 * x2**5) * torch.exp(-t2) / 420 + 0.007 * x2 * t2

    hand = pnp_unsteady_residuals(cp, cn, phi, d1x, d2x, d1t, w_xt, xg, tg)
    informer = make_pnp_unsteady_informer(d1x, d2x, d1t, device=str(xg.device))
    dvr = pnp_unsteady_residuals_dvr(informer, cp, cn, phi, w_xt, xg, tg)

    for l_hand, l_dvr in zip(hand, dvr):
        assert abs(float(l_hand) - float(l_dvr)) <= 1e-10 * abs(float(l_hand))
