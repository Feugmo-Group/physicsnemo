"""Frequency-domain Helmholtz PINN (Eqs. 15-16 of the proposal). Week 8.

Solves the coupled (C_r, C_i) system on a tessellated pore geometry at a given ω,
extracts Z(ω) via Eq. (17), assembles a Nyquist sweep over the configured grid.
"""

from __future__ import annotations
import numpy as np
from .geometry import PoreGeometry


def solve_one_frequency(
    geom: PoreGeometry,
    omega: float,
    D: float,
    epochs: int,
    lr: float,
    rff_encoder=None,
) -> complex:
    """Train a PINN at fixed ω, return Z(ω) = -1 / (D ⟨∂Ĉ/∂z⟩_A)."""
    raise NotImplementedError


def nyquist_sweep(
    geom: PoreGeometry,
    omegas: list[float],
    D: float,
    epochs: int,
    lr: float,
    rff_encoder=None,
) -> np.ndarray:
    """Return Z(ω) over the frequency list. Shape (len(omegas),) complex."""
    raise NotImplementedError
