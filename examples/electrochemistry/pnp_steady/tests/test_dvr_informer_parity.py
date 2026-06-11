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

"""Parity: DVR-collocation informer vs hand-rolled steady-PNP residuals."""

from __future__ import annotations

import math
import os
import sys

import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from physicsnemo.experimental.models.scen import DVRMapper  # noqa: E402

from src.physics import (  # noqa: E402
    make_pnp_steady_informer,
    pnp_steady_residuals,
    pnp_steady_residuals_dvr,
)


def test_dvr_informer_matches_handrolled():
    """All three interior residual losses match the hand-rolled ones to round-off."""
    torch.set_default_dtype(torch.float64)
    m = DVRMapper(60, -3.0, 3.0, 0.0, dtype=torch.float64)
    x, d1, d2, w = m.nodes, m.D1, m.D2, m.weights
    w_norm = w / w.sum()
    px = math.pi * x
    cp = torch.sin(px) + torch.cos(px) + 0.02 * torch.cos(2 * x)
    cn = torch.sin(px) + 0.01 * x
    phi = torch.cos(px)

    hand = pnp_steady_residuals(cp, cn, phi, d1, d2, w_norm, x)
    informer = make_pnp_steady_informer(d1, d2, device=str(x.device))
    dvr = pnp_steady_residuals_dvr(informer, cp, cn, phi, w_norm, x)

    for l_hand, l_dvr in zip(hand, dvr):
        assert abs(float(l_hand) - float(l_dvr)) <= 1e-10 * abs(float(l_hand))
