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

"""Parity: DVR-collocation informer vs hand-rolled 3D steady-PNP residuals."""

from __future__ import annotations

import math
import os
import sys

import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from physicsnemo.experimental.models.scen import DVRMapper3D  # noqa: E402

from src.physics import (  # noqa: E402
    make_pnp_3d_informer,
    pnp_3d_residuals,
    pnp_3d_residuals_dvr,
)


def test_dvr_informer_matches_handrolled():
    """All three 3D residual losses match the hand-rolled ones to round-off."""
    torch.set_default_dtype(torch.float64)
    m = DVRMapper3D(6, 0.0, 1.0, dtype=torch.float64)
    xyz, lap, w = m.xyz_nodes, m.laplacian, m.weights
    d1x, d1y, d1z = m.D1x, m.D1y, m.D1z
    d2x, d2y, d2z = m.D2x, m.D2y, m.D2z
    w_norm = w / w.sum()
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    s = torch.sin(math.pi * x) * torch.sin(math.pi * y) * torch.sin(math.pi * z)
    c_K = 1 + 0.1 * s + 0.02 * torch.cos(2 * x)
    c_Cl = 1 - 0.1 * s + 0.01 * y
    phi = 0.1 * torch.cos(math.pi * x) * torch.cos(math.pi * y) * torch.cos(
        math.pi * z
    ) / (3 * math.pi**2) + 0.005 * z

    hand = pnp_3d_residuals(c_K, c_Cl, phi, lap, d1x, d1y, d1z, w_norm, xyz)
    informer = make_pnp_3d_informer(d1x, d1y, d1z, d2x, d2y, d2z, device=str(xyz.device))
    dvr = pnp_3d_residuals_dvr(informer, c_K, c_Cl, phi, w_norm, xyz)

    for l_hand, l_dvr in zip(hand, dvr):
        assert abs(float(l_hand) - float(l_dvr)) <= 1e-10 * abs(float(l_hand))
