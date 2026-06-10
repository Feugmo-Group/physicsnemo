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

"""Refined Point Defect Model (RPDM) — spectral-collocation (NSEM/SCEN) physics.

This is the spectral (precomputed-operator, no-autodiff) reimplementation of the
autodiff PINN RPDM example.  The dimensionless model lives on a *fixed* reference
domain ``x in [0, 1]`` (Landau-transformed spatial coordinate) and ``y in [0, yf]``
(dimensionless time).  The moving metal/film/solution boundary enters through the
film-thickness field ``l(y)`` and its time derivative ``l_y``.

PASSIVE mode only is implemented here (the transpassive electron/hole transport
``transport_e``/``transport_h`` and concentrations ``ce``/``ch`` are dropped).
The four primary fields are:

* ``cCV(x, y)`` — cation-vacancy concentration
* ``cAV(x, y)`` — anion-vacancy concentration
* ``phif(x, y)`` — film potential
* ``l(y)``      — film thickness (function of time only, broadcast over x)

Discretisation
--------------
Two 1D :class:`DVRMapper` grids give the precomputed differentiation operators:

* ``D1x``, ``D2x`` — spatial first/second derivative, shape ``(Nx, Nx)``.
* ``D1y``          — time derivative, shape ``(Nt, Nt)``.

Fields are stored as ``(Nt, Nx)`` tensors.  A spatial derivative is
``field @ Dx.T`` (acts along the x-axis / columns); a time derivative is
``Dy @ field`` (acts along the t-axis / rows).  ``l(y)`` is an ``(Nt,)`` vector;
``l_y = D1y @ l``.  Because ``l`` is x-independent, ``l.diff(x) = 0`` and the
moving-boundary convective term ``x * l_y * field_x / l`` uses the broadcast
``l`` and ``l_y`` over the (Nt, Nx) grid.

All physical derived/nondimensional groups are ported verbatim from the source
``pdm/pdm.py`` (:class:`Parameters` and :class:`PointDefectModel`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import sympy as sp
import torch

from physicsnemo.sym.eq.pde import PDE


@dataclass
class Parameters:
    """Physical simulation parameters (ported from the source RPDM model)."""

    # universal constants
    F: float = 9.6485332e4  # Faraday constant
    R: float = 8.3144626  # gas constant
    NA: float = 6.02214076e23  # Avogadro constant
    e: float = 1.602176634e-19  # elementary charge
    eps0: float = 8.8541878188e-12  # vacuum permittivity
    kB: float = 1.380649e-23  # Boltzmann constant

    # charge numbers
    zCV: float = -8 / 3  # cation vacancy (CV) charge number
    zAV: float = 2  # anion vacancy (AV) charge number

    # system parameters
    T: float = 293  # [K] temperature
    Lini: float = 1e-10  # [m] initial film thickness
    ddl: float = 2e-10  # [m] defect layer thickness
    dcdl: float = 5e-10  # [m] compact double layer thickness
    Omega: float = 1.4e-5  # [m^3/mol] molar volume of oxide
    lc: float = 1e-10  # [m] characteristic film thickness
    tc: float = 50000  # [s] characteristic time

    # diffusivities
    DAV: float = 1e-21  # [m^2/s] AV diffusivity
    DCV: float = 1e-21  # [m^2/s] CV diffusivity

    # reaction parameters
    alphaR1: float = 0.3
    alphaR2: float = 0.8
    alphaR3: float = 0.1
    alphaR4: float = 0.8
    k0: float = 4.5e-8  # [-] rate constant reference value
    k0R1: float = field(init=False)
    k0R2: float = field(init=False)
    k0R3: float = field(init=False)
    k0R4: float = field(init=False)
    kR5: float = field(init=False)

    # permittivities
    epsf: float = field(init=False)
    epsdl: float = field(init=False)
    epscdl: float = field(init=False)

    def __post_init__(self):
        self.k0R1 = self.k0
        self.k0R2 = 80 * self.k0
        self.k0R3 = 0.1 * self.k0
        self.k0R4 = 5 * self.k0
        self.kR5 = 0.17 * self.k0
        self.epsf = 14 * self.eps0
        self.epsdl = 2 * self.eps0
        self.epscdl = 78.5 * self.eps0


@dataclass
class NondimGroups:
    """Dimensionless groups for the passive RPDM (ported from source).

    Built from :class:`Parameters` and a (constant) external potential ``Eext``.
    All ``*_hat`` reaction prefactors and transport coefficients match the source
    ``PointDefectModel.__init__`` symbolic definitions exactly.
    """

    eta: float
    eps: float  # poisson factor
    xiCV: float
    xiAV: float
    nu_mf: float
    nu_fs: float
    k0R1_hat: float
    k0R2_hat: float
    k0R3_hat: float
    k0R4_hat: float
    k0R2_hat_fg: float
    kR5_hat: float
    lini: float
    m: float  # initial potential spatial slope
    phiext_ref: float
    phiext: float  # dimensionless applied potential (constant Eext)
    zCV: float
    zAV: float
    # exponential reaction factors evaluated at the reference potential
    eR1: float
    eR2: float
    eR3: float
    eR4: float
    cR1: float
    cR2: float
    cR3: float
    cR4: float
    sqrt_lmd: float  # source residual weight for stiff reaction/film terms

    @classmethod
    def from_parameters(
        cls, p: Parameters, Eext: float, sqrt_lmd: float = math.sqrt(1e-8)
    ) -> "NondimGroups":
        phith = p.R * p.T / p.F  # thermal voltage
        phic = Eext  # characteristic potential (constant external potential)
        rho = 1.0 / (1e9 * p.Omega)  # characteristic vacancy concentration

        m = 1e7 * p.lc / phic  # initial potential spatial slope
        phiext_ref = Eext / phic  # = 1.0 for constant Eext
        phiext = Eext / phic

        eta = phic / phith
        eps = p.F * p.lc**2 / (phic * p.epsf)  # poisson factor
        xiCV = p.lc**2 / (p.DCV * p.tc)
        xiAV = p.lc**2 / (p.DAV * p.tc)
        nu_mf = p.epsdl * p.lc / (p.epsf * p.ddl)
        nu_fs = p.lc / (p.epsf * (p.ddl / p.epsdl + p.dcdl / p.epscdl))

        k0R1_hat = p.lc * p.k0R1 / p.DCV
        k0R2_hat = (4.0 / 3.0) * p.lc * p.k0R2 / (p.DAV * rho)
        k0R2_hat_fg = p.tc * p.Omega * p.k0R2 / p.lc
        k0R3_hat = p.lc * p.k0R3 / (p.DCV * rho)
        k0R4_hat = p.lc * p.k0R4 / p.DAV
        kR5_hat = p.tc * p.Omega * p.kR5 / p.lc
        lini = p.Lini / p.lc

        cR1 = p.alphaR1 * 2 * eta
        cR2 = p.alphaR2 * (8.0 / 3.0) * eta
        cR3 = p.alphaR3 * 2 * eta
        cR4 = p.alphaR4 * (8.0 / 3.0) * eta
        eR1 = math.exp(cR1 * phiext_ref)
        eR2 = math.exp(cR2 * phiext_ref)
        eR3 = math.exp(cR3 * phiext_ref)
        eR4 = math.exp(cR4 * phiext_ref)

        return cls(
            eta=eta,
            eps=eps,
            xiCV=xiCV,
            xiAV=xiAV,
            nu_mf=nu_mf,
            nu_fs=nu_fs,
            k0R1_hat=k0R1_hat,
            k0R2_hat=k0R2_hat,
            k0R3_hat=k0R3_hat,
            k0R4_hat=k0R4_hat,
            k0R2_hat_fg=k0R2_hat_fg,
            kR5_hat=kR5_hat,
            lini=lini,
            m=m,
            phiext_ref=phiext_ref,
            phiext=phiext,
            zCV=p.zCV,
            zAV=p.zAV,
            eR1=eR1,
            eR2=eR2,
            eR3=eR3,
            eR4=eR4,
            cR1=cR1,
            cR2=cR2,
            cR3=cR3,
            cR4=cR4,
            sqrt_lmd=sqrt_lmd,
        )


class PointDefectModel(PDE):
    """Inline SymPy ``PDE`` for the passive RPDM (spectral-collocation port).

    This is the symbolic statement of the same interior + interface residuals
    that were previously hand-coded in :func:`rpdm_residuals`.  Building it as a
    :class:`~physicsnemo.sym.eq.pde.PDE` lets us obtain per-equation
    :class:`~physicsnemo.sym.computation.Computation` objects via
    :meth:`make_computations`; each is then evaluated on the precomputed-operator
    derivatives (see :func:`_build_var_dict` / :func:`rpdm_residuals`).

    The constants are injected as SymPy :class:`~sympy.Number` from a
    :class:`NondimGroups` instance, so the symbolic residuals are numerically
    identical to the original tensor algebra (verified to float64 round-off by
    ``tests/test_pde_parity.py``).  ``phiext`` is the *constant* external
    potential here (the spectral example targets the constant-voltage case), so
    it enters as a Number rather than a symbolic ``Eext(y)``.

    Equation/leaf conventions (``diff_str == "__"``):

    * Fields ``cCV``, ``cAV``, ``phif`` and the film thickness ``l`` are SymPy
      ``Function(...)(x, y)``; ``phimf`` is the (detached) interface potential
      that drives the film-growth ODE.
    * Spatial derivatives map to ``cCV__x`` (``D1x``), ``cCV__x__x`` (``D2x``);
      time derivatives to ``cCV__y`` / ``l__y`` (``D1y``).
    * The bare coordinate symbol ``x`` (Landau convection term) maps to ``"x"``.
    """

    name = "RPDM"

    def __init__(self, g: NondimGroups):
        x = sp.Symbol("x")
        y = sp.Symbol("y")
        input_variables = {"x": x, "y": y}

        cCV = sp.Function("cCV")(*input_variables.values())
        cAV = sp.Function("cAV")(*input_variables.values())
        phif = sp.Function("phif")(*input_variables.values())
        l = sp.Function("l")(*input_variables.values())  # noqa: E741
        # phimf = phif at the metal/film interface (x=0); supplied detached.
        phimf = sp.Function("phimf")(*input_variables.values())

        N = sp.Number
        eta, zCV, zAV = N(g.eta), N(g.zCV), N(g.zAV)
        eps = N(g.eps)
        xiCV, xiAV = N(g.xiCV), N(g.xiAV)
        nu_mf, nu_fs = N(g.nu_mf), N(g.nu_fs)
        k0R1_hat, k0R2_hat = N(g.k0R1_hat), N(g.k0R2_hat)
        k0R3_hat, k0R4_hat = N(g.k0R3_hat), N(g.k0R4_hat)
        k0R2_hat_fg, kR5_hat = N(g.k0R2_hat_fg), N(g.kR5_hat)
        eR1, eR2, eR3, eR4 = N(g.eR1), N(g.eR2), N(g.eR3), N(g.eR4)
        cR1, cR2, cR3, cR4 = N(g.cR1), N(g.cR2), N(g.cR3), N(g.cR4)
        phiext, phiext_ref = N(g.phiext), N(g.phiext_ref)
        sl = N(g.sqrt_lmd)

        self.equations = {}

        # ── Interior (full grid) ───────────────────────────────────────────────
        self.equations["poisson"] = eps * phif.diff(x, 2) / l**2 + (
            zCV * cCV + zAV * cAV
        )
        self.equations["transport_CV"] = (
            xiCV * cCV.diff(y, 1)
            - xiCV * x * l.diff(y, 1) * cCV.diff(x, 1) / l
            - cCV.diff(x, 2) / l**2
            - eta * zCV * cCV.diff(x, 1) * phif.diff(x, 1) / l**2
            - eta * zCV * cCV * phif.diff(x, 2) / l**2
        )
        self.equations["transport_AV"] = (
            xiAV * cAV.diff(y, 1)
            - xiAV * x * l.diff(y, 1) * cAV.diff(x, 1) / l
            - cAV.diff(x, 2) / l**2
            - eta * zAV * cAV.diff(x, 1) * phif.diff(x, 1) / l**2
            - eta * zAV * cAV * phif.diff(x, 2) / l**2
        )

        # ── Film-growth ODE in time (passive) ──────────────────────────────────
        self.equations["film_growth"] = sl * (
            l.diff(y, 1)
            - k0R2_hat_fg * eR2 * sp.exp(cR2 * (phiext - phimf - phiext_ref))
            + kR5_hat
        )

        # ── Metal/film interface (x = 0) ───────────────────────────────────────
        self.equations["flux_R1"] = sl * (
            -cCV.diff(x, 1) / l
            - eta * zCV * cCV * phif.diff(x, 1) / l
            + k0R1_hat * cCV * eR1 * sp.exp(cR1 * (phiext - phif - phiext_ref))
        )
        self.equations["flux_R2"] = sl * (
            -cAV.diff(x, 1) / l
            - eta * zAV * cAV * phif.diff(x, 1) / l
            - k0R2_hat * eR2 * sp.exp(cR2 * (phiext - phif - phiext_ref))
        )
        self.equations["mf_phif"] = nu_mf * (phif - phiext) - phif.diff(x, 1) / l

        # ── Film/solution interface (x = 1) ────────────────────────────────────
        self.equations["flux_R3"] = sl * (
            -cCV.diff(x, 1) / l
            - eta * zCV * cCV * phif.diff(x, 1) / l
            + k0R3_hat * eR3 * sp.exp(cR3 * (phif - phiext_ref))
        )
        self.equations["flux_R4"] = sl * (
            -cAV.diff(x, 1) / l
            - eta * zAV * cAV * phif.diff(x, 1) / l
            - k0R4_hat * cAV * eR4 * sp.exp(cR4 * (phif - phiext_ref))
        )
        self.equations["fs_phif"] = nu_fs * phif + phif.diff(x, 1) / l


def make_computations_by_name(
    g: NondimGroups, detach_names: list[str] | None = None
) -> dict[str, "object"]:
    """Build :class:`PointDefectModel` and return its computations keyed by name.

    Each value is a :class:`~physicsnemo.sym.computation.Computation` whose
    ``evaluate(var_dict)`` consumes the ``"__"``-named derivative leaves; the
    required keys are introspectable via ``.inputs`` / ``.derivatives``.
    ``phimf`` is detached by default so the film-growth ODE treats the interface
    potential as data (matching the autodiff RPDM example).
    """
    if detach_names is None:
        detach_names = ["phimf"]
    pde = PointDefectModel(g)
    return {c.outputs[0]: c for c in pde.make_computations(detach_names=detach_names)}


# Residual terms integrated over the full space-time grid vs. those evaluated
# only on the x=0 / x=1 interface columns (per time slice).
_INTERIOR_TERMS = ("poisson", "transport_CV", "transport_AV")
_EDGE0_TERMS = ("flux_R1", "flux_R2", "mf_phif")  # metal/film, column 0
_EDGE1_TERMS = ("flux_R3", "flux_R4", "fs_phif")  # film/solution, column -1


def _dx(field: torch.Tensor, Dx: torch.Tensor) -> torch.Tensor:
    """Spatial derivative of an ``(Nt, Nx)`` field: applied along the x-axis."""
    return field @ Dx.T


def _dy(field: torch.Tensor, D1y: torch.Tensor) -> torch.Tensor:
    """Time derivative of an ``(Nt, Nx)`` field: applied along the t-axis."""
    return D1y @ field


def _build_full_var_dict(
    cCV: torch.Tensor,
    cAV: torch.Tensor,
    phif: torch.Tensor,
    lvec: torch.Tensor,
    D1x: torch.Tensor,
    D2x: torch.Tensor,
    D1y: torch.Tensor,
    x_grid: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Map the precomputed-operator derivatives to the ``"__"`` leaf names.

    Every key a :class:`~physicsnemo.sym.computation.Computation` can request
    (introspectable via ``comp.inputs`` / ``comp.derivatives``) is assembled
    here on the full ``(Nt, Nx)`` grid, with ``l`` and the bare coordinate ``x``
    broadcast across the grid:

    * ``cCV__x``  ← ``D1x`` applied along x        (:func:`_dx`)
    * ``cCV__x__x`` ← ``D2x`` applied along x
    * ``cCV__y``  ← ``D1y`` applied along t        (:func:`_dy`)
    * ``l__y``    ← ``D1y @ l(y)``, broadcast over x
    * ``x``       ← ``x_grid``, broadcast over t   (Landau convection term)
    """
    Nt, Nx = cCV.shape
    lfull = lvec.unsqueeze(1).expand(Nt, Nx)
    ly_full = (D1y @ lvec).unsqueeze(1).expand(Nt, Nx)
    return {
        "cCV": cCV,
        "cAV": cAV,
        "phif": phif,
        "l": lfull,
        "x": x_grid.unsqueeze(0).expand(Nt, Nx),
        "cCV__x": _dx(cCV, D1x),
        "cCV__x__x": _dx(cCV, D2x),
        "cCV__y": _dy(cCV, D1y),
        "cAV__x": _dx(cAV, D1x),
        "cAV__x__x": _dx(cAV, D2x),
        "cAV__y": _dy(cAV, D1y),
        "phif__x": _dx(phif, D1x),
        "phif__x__x": _dx(phif, D2x),
        "l__y": ly_full,
    }


def rpdm_residuals(
    cCV: torch.Tensor,
    cAV: torch.Tensor,
    phif: torch.Tensor,
    lvec: torch.Tensor,
    D1x: torch.Tensor,
    D2x: torch.Tensor,
    D1y: torch.Tensor,
    w_xt: torch.Tensor,
    x_grid: torch.Tensor,
    g: NondimGroups,
    comps: dict[str, "object"],
) -> dict[str, torch.Tensor]:
    """Per-term weighted squared-residual losses for the passive RPDM.

    The residual *expressions* come from :class:`PointDefectModel`'s symbolic
    equations via :meth:`make_computations`; this function only supplies the
    precomputed-operator derivatives (the ``"__"`` leaves) and reduces each
    raw residual to a weighted MSE with the same quadrature as before.

    Parameters
    ----------
    cCV, cAV, phif : shape ``(Nt, Nx)`` — fields on the tensor-product grid.
    lvec : shape ``(Nt,)`` — film thickness as a function of time.
    D1x, D2x : shape ``(Nx, Nx)`` — spatial derivative matrices.
    D1y : shape ``(Nt, Nt)`` — time derivative matrix.
    w_xt : shape ``(Nt, Nx)`` — outer product of normalised quadrature weights.
    x_grid : shape ``(Nx,)`` — spatial reference nodes.
    g : :class:`NondimGroups`.
    comps : dict mapping equation name -> :class:`Computation`, as returned by
        :func:`make_computations_by_name` (i.e. ``PointDefectModel.make_computations``).

    Returns
    -------
    dict mapping residual-term name -> scalar weighted MSE.  Interior terms
    (``poisson``, ``transport_CV``, ``transport_AV``) are integrated over the
    full grid; ``film_growth`` over time; interface terms over time on the
    x=0 / x=1 edges.
    """
    full = _build_full_var_dict(cCV, cAV, phif, lvec, D1x, D2x, D1y, x_grid)

    # ── Interior residuals on the full (Nt, Nx) grid ──────────────────────────
    R = {t: comps[t].evaluate(full)[t] for t in _INTERIOR_TERMS}

    # ── Film-growth ODE in time (passive) ─────────────────────────────────────
    # phimf = phif at the metal/film interface (x = 0), supplied detached-by-name
    # in the PDE; l__y is the time derivative of the film thickness.
    R["film_growth"] = comps["film_growth"].evaluate(
        {"phimf": phif[:, 0], "l__y": D1y @ lvec}
    )["film_growth"]

    # ── Interface residuals on the edge columns (per time slice) ──────────────
    # Each interface comp needs only the bare field, l, and the first spatial
    # derivative at that column; slice the already-computed full-grid leaves.
    def _edge_dict(col: int) -> dict[str, torch.Tensor]:
        return {
            "cCV": full["cCV"][:, col],
            "cAV": full["cAV"][:, col],
            "phif": full["phif"][:, col],
            "l": lvec,
            "cCV__x": full["cCV__x"][:, col],
            "cAV__x": full["cAV__x"][:, col],
            "phif__x": full["phif__x"][:, col],
        }

    e0, e1 = _edge_dict(0), _edge_dict(-1)
    for t in _EDGE0_TERMS:
        R[t] = comps[t].evaluate(e0)[t]
    for t in _EDGE1_TERMS:
        R[t] = comps[t].evaluate(e1)[t]

    # ── Aggregate to per-term scalar MSEs (same weighting/quadrature) ─────────
    wt = w_xt[:, 0]  # time-only normalised weights, (Nt,)
    wt = wt / wt.sum()

    out = {t: (w_xt * R[t] ** 2).sum() for t in _INTERIOR_TERMS}
    for t in ("film_growth", *_EDGE0_TERMS, *_EDGE1_TERMS):
        out[t] = (wt * R[t] ** 2).sum()
    return out


def rpdm_interface_loss(
    fields: list[torch.Tensor],
    D1x: torch.Tensor,
    split_idx: list[int],
    cond: str = "both",
) -> torch.Tensor:
    """C0/C1 continuity across internal spatial element interfaces, per time slice.

    With a multi-element spatial mesh the global operators ``D1x``/``D2x`` are
    *block-diagonal* (each element differentiates only itself), so element
    continuity must be imposed explicitly.  At every internal interface column
    ``j`` (start of element ``k+1``; the end of element ``k`` is column ``j-1``,
    which sits at the same physical ``x``) this enforces, for each field and each
    time row:

    * **C0** (value):      ``f[:, j-1] == f[:, j]``
    * **C1** (flux/slope): ``f_x[:, j-1] == f_x[:, j]``

    Parameters
    ----------
    fields : list of shape ``(Nt, Nx)`` tensors
        Fields whose continuity is enforced (e.g. ``[cCV, cAV, phif]``).
    D1x : shape ``(Nx, Nx)``
        Block-diagonal global spatial first-derivative operator.
    split_idx : list of int
        Column indices that start each element after the first (the internal
        interface locations).
    cond : str
        One of ``'c0'``, ``'c1'``, ``'both'``.

    Returns
    -------
    torch.Tensor
        Scalar sum of the squared continuity mismatches, averaged over time.
    """
    total = fields[0].new_zeros(())
    for f in fields:
        fx = f @ D1x.T  # (Nt, Nx), per-element derivative (block-diagonal)
        for j in split_idx:
            if cond in ("c0", "both"):
                total = total + ((f[:, j - 1] - f[:, j]) ** 2).mean()
            if cond in ("c1", "both"):
                total = total + ((fx[:, j - 1] - fx[:, j]) ** 2).mean()
    return total


def rpdm_ic_loss(
    cCV: torch.Tensor,
    cAV: torch.Tensor,
    phif: torch.Tensor,
    lvec: torch.Tensor,
    x_grid: torch.Tensor,
    g: NondimGroups,
    t0_idx: int = 0,
) -> torch.Tensor:
    """Initial-condition penalty at ``y = 0`` (ported from source ICs).

    ``cCV = 0``, ``cAV = 0``, ``phif = phiext - m*x``, ``l = lini``.
    """
    cCV0 = cCV[t0_idx]
    cAV0 = cAV[t0_idx]
    phi0 = phif[t0_idx]
    phi0_ex = g.phiext - g.m * x_grid
    ic = (
        (cCV0**2).mean()
        + (cAV0**2).mean()
        + ((phi0 - phi0_ex) ** 2).mean()
        + (lvec[t0_idx] - g.lini) ** 2
    )
    return ic
