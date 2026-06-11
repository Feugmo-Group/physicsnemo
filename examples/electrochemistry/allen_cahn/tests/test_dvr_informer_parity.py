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

"""Parity: DVR-collocation informer vs hand-rolled Allen-Cahn residual.

Confirms that routing the residual through :class:`DVRPhysicsInformer`
(``grad_method="dvr"``) reproduces the original direct-operator residual to
float64 round-off, so the migration is behaviour-preserving.
"""

from __future__ import annotations

import os
import sys

import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from physicsnemo.experimental.models.scen import DVRMapper  # noqa: E402

from src.physics import (  # noqa: E402
    allen_cahn_residual,
    allen_cahn_residual_dvr,
    make_allen_cahn_informer,
)


def test_dvr_informer_matches_handrolled():
    """The DVR informer residual equals the hand-rolled one to round-off."""
    torch.set_default_dtype(torch.float64)
    eps_sq = 1e-3
    m = DVRMapper(48, -1.0, 1.0, 0.0, dtype=torch.float64)
    x, d2, w = m.nodes, m.D2, m.weights
    w_norm = w / w.sum()
    u = torch.tanh(x / (eps_sq**0.5 * 2**0.5)) + 0.05 * torch.cos(2 * x)

    informer = make_allen_cahn_informer(eps_sq, d2, device=str(x.device))
    l_hand = float(allen_cahn_residual(u, d2, w_norm, eps_sq))
    l_dvr = float(allen_cahn_residual_dvr(informer, u, w_norm))

    assert abs(l_hand - l_dvr) <= 1e-12 * abs(l_hand)
