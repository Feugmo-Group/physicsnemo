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

"""Parity: DVR-collocation informer vs hand-rolled Poisson-Boltzmann residual."""

from __future__ import annotations

import os
import sys

import pytest
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from physicsnemo.experimental.models.scen import DVRMapper  # noqa: E402

from src.physics import (  # noqa: E402
    make_pb_informer,
    pb_linear_residual,
    pb_nonlinear_residual,
    pb_residual_dvr,
)


@pytest.mark.parametrize("nonlinear", [False, True])
def test_dvr_informer_matches_handrolled(nonlinear):
    """The DVR informer residual equals the hand-rolled one to round-off."""
    torch.set_default_dtype(torch.float64)
    m = DVRMapper(50, 0.0, 1.0, 0.0, dtype=torch.float64)
    x, d2, w = m.nodes, m.D2, m.weights
    w_norm = w / w.sum()
    kappa_sq = 25.0
    psi = torch.exp(-5 * x) + 0.03 * torch.cos(3 * x)

    ref = pb_nonlinear_residual if nonlinear else pb_linear_residual
    informer = make_pb_informer(kappa_sq, d2, nonlinear=nonlinear, device=str(x.device))
    l_hand = float(ref(psi, d2, w_norm, kappa_sq))
    l_dvr = float(pb_residual_dvr(informer, psi, w_norm))

    assert abs(l_hand - l_dvr) <= 1e-12 * abs(l_hand)
