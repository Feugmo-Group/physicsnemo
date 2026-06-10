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

"""Numerical-parity check: symbolic make_computations() vs. hand-rolled residuals.

The current :func:`rpdm_residuals` evaluates the symbolic :class:`PointDefectModel`
equations through ``make_computations()``.  This test re-implements the *original*
hand-rolled tensor algebra inline and asserts the two agree to float64 round-off
on a fixed random field input, term by term.

Run::

    pytest tests/test_pde_parity.py
    python tests/test_pde_parity.py        # prints the per-term diff table
"""

from __future__ import annotations

import os
import sys

import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.physics import (  # noqa: E402
    NondimGroups,
    Parameters,
    make_computations_by_name,
    rpdm_residuals,
)

from physicsnemo.experimental.models.scen import DVRMapper  # noqa: E402


def _hand_rolled_residuals(cCV, cAV, phif, lvec, D1x, D2x, D1y, w_xt, x_grid, g):
    """The original explicit-tensor-algebra residuals (pre-refactor)."""
    eta, zCV, zAV = g.eta, g.zCV, g.zAV
    ly_vec = D1y @ lvec
    lbc = lvec.unsqueeze(1)
    lyb = ly_vec.unsqueeze(1)
    x2d = x_grid.unsqueeze(0)

    cCV_x = cCV @ D1x.T
    cCV_xx = cCV @ D2x.T
    cAV_x = cAV @ D1x.T
    cAV_xx = cAV @ D2x.T
    phi_x = phif @ D1x.T
    phi_xx = phif @ D2x.T
    cCV_y = D1y @ cCV
    cAV_y = D1y @ cAV
    l2 = lbc**2

    R_poisson = g.eps * phi_xx / l2 + (zCV * cCV + zAV * cAV)
    R_tCV = (
        g.xiCV * cCV_y
        - g.xiCV * x2d * lyb * cCV_x / lbc
        - cCV_xx / l2
        - eta * zCV * cCV_x * phi_x / l2
        - eta * zCV * cCV * phi_xx / l2
    )
    R_tAV = (
        g.xiAV * cAV_y
        - g.xiAV * x2d * lyb * cAV_x / lbc
        - cAV_xx / l2
        - eta * zAV * cAV_x * phi_x / l2
        - eta * zAV * cAV * phi_xx / l2
    )

    sl = g.sqrt_lmd
    phimf = phif[:, 0]
    R_fg = sl * (
        ly_vec
        - g.k0R2_hat_fg * g.eR2 * torch.exp(g.cR2 * (g.phiext - phimf - g.phiext_ref))
        + g.kR5_hat
    )

    l0 = lvec
    R_flux_R1 = sl * (
        -cCV_x[:, 0] / l0
        - eta * zCV * cCV[:, 0] * phi_x[:, 0] / l0
        + g.k0R1_hat
        * cCV[:, 0]
        * g.eR1
        * torch.exp(g.cR1 * (g.phiext - phif[:, 0] - g.phiext_ref))
    )
    R_flux_R2 = sl * (
        -cAV_x[:, 0] / l0
        - eta * zAV * cAV[:, 0] * phi_x[:, 0] / l0
        - g.k0R2_hat * g.eR2 * torch.exp(g.cR2 * (g.phiext - phif[:, 0] - g.phiext_ref))
    )
    R_mf_phif = g.nu_mf * (phif[:, 0] - g.phiext) - phi_x[:, 0] / l0

    R_flux_R3 = sl * (
        -cCV_x[:, -1] / l0
        - eta * zCV * cCV[:, -1] * phi_x[:, -1] / l0
        + g.k0R3_hat * g.eR3 * torch.exp(g.cR3 * (phif[:, -1] - g.phiext_ref))
    )
    R_flux_R4 = sl * (
        -cAV_x[:, -1] / l0
        - eta * zAV * cAV[:, -1] * phi_x[:, -1] / l0
        - g.k0R4_hat
        * cAV[:, -1]
        * g.eR4
        * torch.exp(g.cR4 * (phif[:, -1] - g.phiext_ref))
    )
    R_fs_phif = g.nu_fs * phif[:, -1] + phi_x[:, -1] / l0

    wt = w_xt[:, 0]
    wt = wt / wt.sum()
    return {
        "poisson": (w_xt * R_poisson**2).sum(),
        "transport_CV": (w_xt * R_tCV**2).sum(),
        "transport_AV": (w_xt * R_tAV**2).sum(),
        "film_growth": (wt * R_fg**2).sum(),
        "flux_R1": (wt * R_flux_R1**2).sum(),
        "flux_R2": (wt * R_flux_R2**2).sum(),
        "mf_phif": (wt * R_mf_phif**2).sum(),
        "flux_R3": (wt * R_flux_R3**2).sum(),
        "flux_R4": (wt * R_flux_R4**2).sum(),
        "fs_phif": (wt * R_fs_phif**2).sum(),
    }


def _build_inputs():
    torch.set_default_dtype(torch.float64)
    dtype = torch.float64
    p = Parameters()
    g = NondimGroups.from_parameters(p, Eext=0.1)

    x_elements = [
        dict(N=16, a=0.0, b=0.1, alpha=1.2, mapping="log"),
        dict(N=14, a=0.1, b=0.9, alpha=0.0, mapping="kte"),
        dict(N=16, a=0.9, b=1.0, alpha=0.9, mapping="kte"),
    ]
    x_mappers = [
        DVRMapper(e["N"], e["a"], e["b"], e["alpha"], mapping=e["mapping"], dtype=dtype)
        for e in x_elements
    ]
    x_grid = torch.cat([m.nodes for m in x_mappers])
    D1x = torch.block_diag(*[m.D1 for m in x_mappers])
    D2x = torch.block_diag(*[m.D2 for m in x_mappers])
    wx = torch.cat([m.weights for m in x_mappers])
    wx = wx / wx.sum()
    Nx = x_grid.numel()

    mt = DVRMapper(12, 0.0, 1.0, 0.0, dtype=dtype)
    Nt = 12
    D1y = mt.D1
    wt = mt.weights / mt.weights.sum()
    w_xt = wt.unsqueeze(1) * wx.unsqueeze(0)

    torch.manual_seed(0)
    cCV = torch.randn(Nt, Nx, dtype=dtype) * 0.1
    cAV = torch.randn(Nt, Nx, dtype=dtype) * 0.1
    phif = torch.randn(Nt, Nx, dtype=dtype) * 0.1 + 1.0
    lvec = torch.rand(Nt, dtype=dtype) * 0.5 + 0.5
    return cCV, cAV, phif, lvec, D1x, D2x, D1y, w_xt, x_grid, g


def test_pde_parity():
    cCV, cAV, phif, lvec, D1x, D2x, D1y, w_xt, x_grid, g = _build_inputs()
    comps = make_computations_by_name(g)

    new = rpdm_residuals(cCV, cAV, phif, lvec, D1x, D2x, D1y, w_xt, x_grid, g, comps)
    old = _hand_rolled_residuals(cCV, cAV, phif, lvec, D1x, D2x, D1y, w_xt, x_grid, g)

    worst_rel = 0.0
    for k in old:
        o, n = float(old[k]), float(new[k])
        rel = abs(o - n) / (abs(o) + 1e-30)
        worst_rel = max(worst_rel, rel)
        # float64 round-off level; the raw residuals span O(1) .. O(1e11).
        assert rel < 1e-10, f"{k}: rel diff {rel:.3e} (old={o:.6e}, new={n:.6e})"
    assert worst_rel < 1e-10


if __name__ == "__main__":
    cCV, cAV, phif, lvec, D1x, D2x, D1y, w_xt, x_grid, g = _build_inputs()
    comps = make_computations_by_name(g)
    new = rpdm_residuals(cCV, cAV, phif, lvec, D1x, D2x, D1y, w_xt, x_grid, g, comps)
    old = _hand_rolled_residuals(cCV, cAV, phif, lvec, D1x, D2x, D1y, w_xt, x_grid, g)
    print(f"{'term':14s} {'old':>14s} {'new':>14s} {'absdiff':>12s} {'reldiff':>12s}")
    for k in old:
        o, n = float(old[k]), float(new[k])
        rel = abs(o - n) / (abs(o) + 1e-30)
        print(f"{k:14s} {o:14.6e} {n:14.6e} {abs(o - n):12.3e} {rel:12.3e}")
