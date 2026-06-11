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

"""Parity: DVR-collocation informer vs hand-rolled convection-diffusion residual."""

from __future__ import annotations

import os
import sys

import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from physicsnemo.experimental.models.scen import DVRMapper  # noqa: E402

from src.physics import cd_residual, cd_residual_dvr, make_cd_informer  # noqa: E402


def test_dvr_informer_matches_handrolled():
    """The DVR informer residual equals the hand-rolled one to round-off."""
    torch.set_default_dtype(torch.float64)
    eps, a = 1e-2, 1.0
    m = DVRMapper(40, 0.0, 1.0, 0.0, dtype=torch.float64)
    x, d1, d2, w = m.nodes, m.D1, m.D2, m.weights
    w_norm = w / w.sum()
    u = (-torch.expm1(-a * x / eps)) + 0.07 * torch.sin(4 * x)

    informer = make_cd_informer(eps, a, d1, d2, device=str(x.device))
    l_hand = float(cd_residual(u, d1, d2, w_norm, eps, a))
    l_dvr = float(cd_residual_dvr(informer, u, w_norm))

    assert abs(l_hand - l_dvr) <= 1e-12 * abs(l_hand)
