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

"""Refined Point Defect Model (RPDM) physics for PhysicsNeMo v2.0.

This module ports the dimensionless RPDM equations describing electrochemical
oxide-film growth on an iron electrode into the v2.0 idiom: an inline SymPy
``PDE`` subclass whose ``self.equations`` dictionary is consumed by
``PhysicsInformer``.

The problem is posed on a 2D space-time rectangle ``(x in [0, 1], y in
[0, yf])``.  A Landau / boundary-immobilization transformation maps the
moving physical film thickness ``[0, L(t)]`` onto the fixed computational
interval ``x in [0, 1]``; ``y`` is dimensionless time.

SIMPLIFICATIONS (passive mode, ``transpassive=False``):
    The full model has a transpassive regime that additionally solves for the
    electron (``ce``) and hole (``ch``) concentrations and adds the
    ``transport_e``, ``transport_h``, ``mf_ce``, ``mf_ch``, ``flux_ch`` and
    ``initial_ce``/``initial_ch`` equations.  For this initial v2.0 port we
    keep the *passive* sub-system only, which is the regime exercised by the
    constant-0.1 V COMSOL validation case.  The transpassive equations are
    retained in the source but intentionally omitted here; enabling them would
    require two extra network outputs and the ``chfs`` hard-BC field.

The four solved (starred) network fields are ``cCV_star``, ``cAV_star``,
``phif_star`` and ``l_star``.  The hard-BC layer (``hard_bc.py``) maps these
raw outputs to the enforced physical fields ``cCV``, ``cAV``, ``phif``, ``l``
plus the derived ``phimf`` (potential nearest the metal/film interface) and
``phifs`` (nearest the film/solution interface).  Those enforced fields are
what we feed to ``PhysicsInformer``.
"""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from sympy import (
    Expr,
    Function,
    Max,
    Min,
    Number,
    Symbol,
    exp,
    lambdify,
)
from sympy import (
    floor as sp_floor,
)

from physicsnemo.sym.eq.pde import PDE


@dataclass
class Parameters:
    """Dimensional simulation parameters and derived nondimensional groups."""

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
    ze: float = -1  # electron charge number
    zh: float = 1  # hole charge number

    # system parameters
    T: float = 293  # [K] temperature
    Lini: float = 1e-10  # [m] initial film thickness
    ddl: float = 2e-10  # [m] defect layer thickness
    dcdl: float = 5e-10  # [m] compact double layer thickness
    Omega: float = 1.4e-5  # [m^3/mol] molar volume of oxide
    lc: float = 1e-10  # [m] characteristic film thickness
    tc: float = 50000  # [s] characteristic time

    # electron physics
    mue0: float = 2.40326e-19  # [J] potential of electrons
    Nv: float = 1e29 / NA  # [mol/m^3] density of states in valence band
    Nc: float = 1e26 / NA  # [mol/m^3] density of states in conduction band
    Ec0: float = 5.127e-19  # reference conduction band energy
    Ev0: float = 1.6022e-19  # reference valence band energy
    moe: float = 1.3e-2  # [m^2/(V s)] electron mobility
    moh: float = 1.3e-2  # [m^2/(V s)] hole mobility
    tau: float = 3e11 / NA  # [mol s / m^3] recombination lifetime parameter
    Ch0: float = field(init=False)  # [mol/m^3] equilibrium hole concentration
    Ce0: float = field(init=False)  # [mol/m^3] equilibrium electron concentration

    # diffusivities
    DAV: float = 1e-21  # [m^2/s] AV diffusivity
    DCV: float = 1e-21  # [m^2/s] CV diffusivity
    De: float = field(init=False)  # [m^2/s] electron diffusivity
    Dh: float = field(init=False)  # [m^2/s] hole diffusivity

    # reaction parameters
    alphaR1: float = 0.3  # [-] reaction 1 transfer coefficient
    alphaR2: float = 0.8  # [-] reaction 2 transfer coefficient
    alphaR3: float = 0.1  # [-] reaction 3 transfer coefficient
    alphaR4: float = 0.8  # [-] reaction 4 transfer coefficient
    alphatp: float = 0.2  # [-] transpassive dissolution transfer coefficient
    alphaO2: float = 0.45  # [-] oxygen reduction transfer coefficient
    Etpeq: float = 0.3  # [V] transpassive dissolution equilibrium potential
    EO2eq: float = 1.35  # [V] oxygen reduction dissolution potential
    k0: float = 4.5e-8  # [-] rate constant reference value
    k0tp: float = 1.635e-4  # [m/s] transpassive dissolution rate prefactor
    k0O2: float = 5e-3  # [m/s] oxygen reduction rate prefactor
    k0R1: float = field(init=False)  # [m/s] reaction 1 rate prefactor
    k0R2: float = field(init=False)  # [mol/(m^2 s)] reaction 2 rate prefactor
    k0R3: float = field(init=False)  # [mol/(m^2 s)] reaction 3 rate prefactor
    k0R4: float = field(init=False)  # [m/s] reaction 4 rate prefactor
    kR5: float = field(init=False)  # [mol/(m^2 s)] reaction 5 rate

    # permittivities
    epsf: float = field(init=False)  # film permittivity
    epsdl: float = field(init=False)  # defect layer permittivity
    epscdl: float = field(init=False)  # compact double layer permittivity

    def __post_init__(self):
        self.Ch0 = self.Nv * np.exp((self.Ev0 - self.mue0) / (self.kB * self.T))
        self.Ce0 = self.Nc * np.exp((self.mue0 - self.Ec0) / (self.kB * self.T))
        self.De = self.moe * self.kB * self.T / self.e
        self.Dh = self.moh * self.kB * self.T / self.e
        self.k0R1 = self.k0
        self.k0R2 = 80 * self.k0
        self.k0R3 = 0.1 * self.k0
        self.k0R4 = 5 * self.k0
        self.kR5 = 0.17 * self.k0
        self.epsf = 14 * self.eps0
        self.epsdl = 2 * self.eps0
        self.epscdl = 78.5 * self.eps0


def make_Eext(
    Eb: float = 0.1,
    Es: float = 0.2,
    Ts: float = 10e5,
    Tr: float = 200,
    nr_steps: int = 4,
    tc: float = 50000,
) -> Callable[[Symbol], Expr]:
    """Build the symbolic applied-potential function ``Eext(y)``.

    A ramped step function.  With the default ``Ts >> yf`` the potential stays
    at the constant base value ``Eb = 0.1 V`` throughout training -- the
    constant-voltage situation that matches the COMSOL validation dataset.
    """
    Eb_n = Number(Eb)
    Es_n = Number(Es)
    ts = Number(Ts / tc)
    tr = Number(Tr / tc)

    def Eext(y: Symbol) -> Expr:
        return Min(
            Max(
                Eb_n,
                Eb_n
                + (sp_floor(y / ts) - 1) * Es_n
                + Max(0, Min(Es_n, (Es_n / tr) * (y % ts))),
            ),
            Eb_n + nr_steps * Es_n,
        )

    return Eext


class PointDefectModel(PDE):
    """Dimensionless refined point defect model (RPDM) -- passive mode.

    Solves for the cation-vacancy (``cCV``) and anion-vacancy (``cAV``)
    concentrations, the film potential (``phif``) and the dimensionless film
    thickness (``l``) on the space-time rectangle ``(x, y)``.
    """

    name = "PointDefectModel"

    def __init__(
        self,
        Eext: Callable[[Symbol], Expr],
        yf: float = 1.0,
        transpassive: bool = False,
        **kwargs,
    ):
        if transpassive:
            raise NotImplementedError(
                "Transpassive mode is intentionally not ported in this v2.0 "
                "example; use transpassive=False (passive mode)."
            )

        # PhysicsInformer reads PDE.dim to size the coordinate tensor.
        self.dim = 2

        # coordinates: x = nondimensional space, y = nondimensional time
        x = Symbol("x")
        y = Symbol("y")
        input_variables = {"x": x, "y": y}

        # predicted (enforced) fields
        cCV = Function("cCV")(*input_variables)
        cAV = Function("cAV")(*input_variables)
        phif = Function("phif")(*input_variables)
        l = Function("l")(*input_variables)  # noqa: E741

        # custom eval function supplied by the hard-BC layer.
        # NOTE: ``phimf`` (potential nearest the metal/film interface) appears
        # in the passive ``film_growth`` ODE.  ``phifs`` (nearest the
        # film/solution interface) is only needed by the transpassive
        # ``flux_ch`` equation, so it is not declared here in passive mode --
        # the hard-BC layer still returns it for completeness/inference.
        phimf = Function("phimf")(*input_variables)

        # default symbolic constants (overridden by **kwargs == asdict(Parameters))
        defaults = {
            "zCV": 1,
            "zAV": 1,
            "ze": 1,
            "zh": 1,
            "T": 1,
            "Lini": 1,
            "ddl": 1,
            "dcdl": 1,
            "Omega": 1,
            "tau": 1,
            "Ch0": 1,
            "Ce0": 1,
            "DAV": 1,
            "DCV": 1,
            "De": 1,
            "Dh": 1,
            "alphaR1": 1,
            "alphaR2": 1,
            "alphaR3": 1,
            "alphaR4": 1,
            "alphatp": 1,
            "alphaO2": 1,
            "Etpeq": 1,
            "EO2eq": 1,
            "k0tp": 1,
            "k0O2": 1,
            "k0R1": 1,
            "k0R2": 1,
            "k0R3": 1,
            "k0R4": 1,
            "kR5": 1,
            "epsf": 1,
            "epsdl": 1,
            "epscdl": 1,
            "tc": 1,
            "lc": 1,
        }
        for kwarg, default in defaults.items():
            setattr(self, f"_{kwarg}", Number(kwargs.get(kwarg, default)))

        zCV = self._zCV
        zAV = self._zAV

        # universal constants
        self._F = Number(9.6485332e4)
        self._R = Number(8.3144626)

        # compute average external potential (characteristic potential)
        Eext_fn = lambdify(y, Eext(y), "numpy")
        y_np = np.linspace(0, yf, 10000)
        Eext_np = Eext_fn(y_np)
        Eext_avg = np.trapezoid(Eext_np, x=y_np) / (yf - 0)

        # derived parameters
        self._phith = self._R * self._T / self._F  # thermal voltage
        self._phic = Number(Eext_avg)  # characteristic potential
        self._rho = 1 / (1e9 * self._Omega)  # characteristic vacancy conc.
        m = Number(1e7) * self._lc / self._phic  # initial potential slope
        phiext_ref = Number(Eext_avg) / self._phic  # ref dimensionless potential
        phiext = Eext(y) / self._phic  # dimensionless applied potential

        # dimensionless groups
        eta = self._phic / self._phith
        eps = self._F * self._lc**2 / (self._phic * self._epsf)  # poisson factor
        xiCV = self._lc**2 / (self._DCV * self._tc)
        xiAV = self._lc**2 / (self._DAV * self._tc)
        nu_mf = self._epsdl * self._lc / (self._epsf * self._ddl)
        nu_fs = self._lc / (
            self._epsf * (self._ddl / self._epsdl + self._dcdl / self._epscdl)
        )
        k0R1_hat = self._lc * self._k0R1 / self._DCV
        k0R2_hat = Number(4 / 3) * self._lc * self._k0R2 / (self._DAV * self._rho)
        k0R2_hat_fg = self._tc * self._Omega * self._k0R2 / self._lc
        k0R3_hat = self._lc * self._k0R3 / (self._DCV * self._rho)
        k0R4_hat = self._lc * self._k0R4 / self._DAV
        kR5_hat = self._tc * self._Omega * self._kR5 / self._lc
        lini = self._Lini / self._lc

        # dimensionless exponential reaction factors
        cR1 = self._alphaR1 * 2 * eta
        cR2 = self._alphaR2 * (8 / 3) * eta
        cR3 = self._alphaR3 * 2 * eta
        cR4 = self._alphaR4 * (8 / 3) * eta
        eR1 = exp(cR1 * phiext_ref)
        eR2 = exp(cR2 * phiext_ref)
        eR3 = exp(cR3 * phiext_ref)
        eR4 = exp(cR4 * phiext_ref)

        # constant weight applied to large boundary residuals (as in source)
        sqrt_lmd = Number(np.sqrt(1e-8))

        # ---- equations ----
        self.equations = {}

        # interior: Poisson (passive: only vacancy charges)
        self.equations["poisson"] = eps * phif.diff(x, 2) / l**2 + (
            zCV * cCV + zAV * cAV
        )
        # interior: cation-vacancy transport (Landau-transformed convection-diffusion-migration)
        self.equations["transport_CV"] = (
            xiCV * cCV.diff(y, 1)
            - xiCV * x * l.diff(y, 1) * cCV.diff(x, 1) / l
            - cCV.diff(x, 2) / l**2
            - eta * zCV * cCV.diff(x, 1) * phif.diff(x, 1) / l**2
            - eta * zCV * cCV * phif.diff(x, 2) / l**2
        )
        # interior: anion-vacancy transport
        self.equations["transport_AV"] = (
            xiAV * cAV.diff(y, 1)
            - xiAV * x * l.diff(y, 1) * cAV.diff(x, 1) / l
            - cAV.diff(x, 2) / l**2
            - eta * zAV * cAV.diff(x, 1) * phif.diff(x, 1) / l**2
            - eta * zAV * cAV * phif.diff(x, 2) / l**2
        )
        # interior: film-growth ODE in time (passive)
        self.equations["film_growth"] = sqrt_lmd * (
            l.diff(y, 1)
            - k0R2_hat_fg * eR2 * exp(cR2 * (phiext - phimf - phiext_ref))
            + kR5_hat
        )

        # metal-film interface (x = 0)
        self.equations["flux_R1"] = sqrt_lmd * (
            -cCV.diff(x, 1) / l
            - eta * zCV * cCV * phif.diff(x, 1) / l
            + k0R1_hat * cCV * eR1 * exp(cR1 * (phiext - phif - phiext_ref))
        )
        self.equations["flux_R2"] = sqrt_lmd * (
            -cAV.diff(x, 1) / l
            - eta * zAV * cAV * phif.diff(x, 1) / l
            - k0R2_hat * eR2 * exp(cR2 * (phiext - phif - phiext_ref))
        )
        self.equations["mf_phif"] = nu_mf * (phif - phiext) - phif.diff(x, 1) / l

        # film-solution interface (x = 1)
        self.equations["flux_R3"] = sqrt_lmd * (
            -cCV.diff(x, 1) / l
            - eta * zCV * cCV * phif.diff(x, 1) / l
            + k0R3_hat * eR3 * exp(cR3 * (phif - phiext_ref))
        )
        self.equations["flux_R4"] = sqrt_lmd * (
            -cAV.diff(x, 1) / l
            - eta * zAV * cAV * phif.diff(x, 1) / l
            - k0R4_hat * cAV * eR4 * exp(cR4 * (phif - phiext_ref))
        )
        self.equations["fs_phif"] = nu_fs * phif + phif.diff(x, 1) / l

        # initial conditions (residual form; enforced exactly by hard BC, kept
        # for completeness / optional soft supervision)
        self.equations["initial_phif"] = phif - phiext + m * x
        self.equations["initial_cCV"] = cCV
        self.equations["initial_cAV"] = cAV
        self.equations["initial_l"] = l - lini

        # store data for the hard-BC layer
        self.phif_initial_fn = lambdify([x, y], phiext - m * x, "numpy")
        self.lini = float(lini)
        self.yf = float(yf)
        self.phic = float(Eext_avg)
