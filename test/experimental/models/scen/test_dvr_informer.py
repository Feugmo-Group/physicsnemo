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

"""Tests for the DVR-collocation PhysicsInformer (``grad_method="dvr"``)."""

from __future__ import annotations

import sympy as sp
import torch

from physicsnemo.experimental.models.scen import (
    AxisOperator,
    DVRMapper,
    DVRPhysicsInformer,
    GradientsDVR,
)
from physicsnemo.sym.eq.pde import PDE


class _AllenCahnPDE(PDE):
    """1D test PDE: ε²u'' − (u³ − u)."""

    name = "AllenCahnTest"

    def __init__(self, eps_sq):
        self.dim = 1
        x = sp.Symbol("x")
        u = sp.Function("u")(x)
        self.equations = {"ac": sp.Number(eps_sq) * u.diff(x, 2) - (u**3 - u)}


class _SpaceTimePDE(PDE):
    """2D test PDE: u_y − u_xx + u·u_x (time = y)."""

    name = "STTest"

    def __init__(self):
        self.dim = 2
        x, y = sp.symbols("x y")
        u = sp.Function("u")(x, y)
        self.equations = {"st": u.diff(y, 1) - u.diff(x, 2) + u * u.diff(x, 1)}


def test_gradients_dvr_first_and_second_derivative():
    """GradientsDVR reproduces D1 @ f and D2 @ f along a 1D axis."""
    torch.set_default_dtype(torch.float64)
    m = DVRMapper(20, -1.0, 1.0, 0.0, dtype=torch.float64)
    f = torch.sin(2 * m.nodes) + 0.3 * m.nodes**2
    g1 = GradientsDVR("u", 1, 1, {"x": AxisOperator(0, m.D1, m.D2)})
    g2 = GradientsDVR("u", 1, 2, {"x": AxisOperator(0, m.D1, m.D2)})
    assert torch.allclose(g1.forward({"u": f})["u__x"], m.D1 @ f)
    assert torch.allclose(g2.forward({"u": f})["u__x__x"], m.D2 @ f)


def test_informer_matches_handrolled_1d():
    """1D informer residual equals the direct-operator residual to round-off."""
    torch.set_default_dtype(torch.float64)
    eps_sq = 1e-3
    m = DVRMapper(24, -1.0, 1.0, 0.0, dtype=torch.float64)
    u = torch.tanh(m.nodes / (eps_sq**0.5 * 2**0.5)) + 0.1 * torch.sin(3 * m.nodes)
    informer = DVRPhysicsInformer(
        required_outputs=["ac"],
        equations=_AllenCahnPDE(eps_sq),
        operators={"x": AxisOperator(0, m.D1, m.D2)},
    )
    r_dvr = informer.forward({"u": u})["ac"]
    r_hand = eps_sq * (m.D2 @ u) - (u**3 - u)
    assert torch.allclose(r_dvr, r_hand, atol=1e-12, rtol=0)


def test_informer_matches_handrolled_2d_spacetime():
    """2D space-time informer reproduces mixed-axis operator algebra."""
    torch.set_default_dtype(torch.float64)
    nx, nt = 16, 12
    mx = DVRMapper(nx, 0.0, 1.0, 0.0, dtype=torch.float64)
    mt = DVRMapper(nt, 0.0, 1.0, 0.0, dtype=torch.float64)
    xg, tg = mx.nodes, mt.nodes
    big_x = xg.unsqueeze(0).expand(nt, nx)
    big_t = tg.unsqueeze(1).expand(nt, nx)
    u = torch.sin(3 * big_x) * torch.exp(-0.5 * big_t) + 0.2 * big_x**2 * big_t

    informer = DVRPhysicsInformer(
        required_outputs=["st"],
        equations=_SpaceTimePDE(),
        operators={
            "x": AxisOperator(axis=1, D1=mx.D1, D2=mx.D2),
            "y": AxisOperator(axis=0, D1=mt.D1, D2=None),
        },
    )
    r_dvr = informer.forward({"u": u})["st"]
    r_hand = (mt.D1 @ u) - (u @ mx.D2.T) + u * (u @ mx.D1.T)
    assert torch.allclose(r_dvr, r_hand, atol=1e-12, rtol=0)
