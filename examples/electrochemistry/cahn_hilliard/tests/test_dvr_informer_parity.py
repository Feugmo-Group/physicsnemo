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

"""Consistency: mixed Cahn-Hilliard residuals reduce to the single-field form.

The mixed system trains ``(c, μ)`` with ``μ − f'(c) + ε²c_xx = 0`` and
``μ_xx = 0``.  Supplying the *consistent* ``μ = f'(c) − ε²c_xx`` makes the first
residual vanish and the second equal the original single-field statement
``D2 f'(c) − ε² D4 c`` (up to the operator-associativity round-off between
applying ``D2`` twice and the precomputed ``D4 = D2 @ D2``).
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
    _f_prime,
    cahn_hilliard_residuals_dvr,
    make_ch_informer,
)


def test_mixed_form_is_consistent():
    """Consistent μ gives res_mu ≈ 0 and res_ch ≈ the single-field residual."""
    torch.set_default_dtype(torch.float64)
    m = DVRMapper(40, -1.0, 1.0, 0.0, dtype=torch.float64)
    x, d2, w = m.nodes, m.D2, m.weights
    d4 = d2 @ d2
    w_norm = w / w.sum()
    eps_sq = 0.01
    c = 0.5 + 0.4 * torch.tanh(3 * x) + 0.05 * torch.cos(2 * x)
    mu = _f_prime(c) - eps_sq * (d2 @ c)

    informer = make_ch_informer(eps_sq, d2, device=str(x.device))
    res = informer.forward({"c": c, "mu": mu})
    r_mu, r_ch = res["res_mu"], res["res_ch"]
    r_single = d2 @ _f_prime(c) - eps_sq * (d4 @ c)

    # μ-definition residual is exactly satisfied (to round-off).
    assert r_mu.abs().max().item() < 1e-12
    # ∇²μ residual matches the single-field form to operator round-off.
    scale = r_single.abs().max().item()
    assert (r_ch - r_single).abs().max().item() < 1e-6 * scale

    # The loss helper returns finite, non-negative scalars.
    l_mu, l_ch = cahn_hilliard_residuals_dvr(informer, c, mu, w_norm)
    assert float(l_mu) >= 0.0 and torch.isfinite(l_mu)
    assert float(l_ch) >= 0.0 and torch.isfinite(l_ch)
