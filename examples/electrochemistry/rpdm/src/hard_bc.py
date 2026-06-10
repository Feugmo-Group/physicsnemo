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

"""Exact (hard) Dirichlet boundary-condition enforcement via approximate
distance functions (ADFs), reimplemented as plain-torch helpers.

The old ``physicsnemo.sym.geometry.adf.ADF`` class is removed in v2.0.  This
module re-derives the two ADF primitives we need -- ``line_segment_adf`` and
``r_equivalence`` -- as pure-torch functions operating directly on coordinate
tensors, then exposes ``enforce_hard_bc`` which maps the raw (starred) network
outputs to the enforced physical fields.

ADF math (Rvachev R-functions):
    For a boundary patch, the ADF ``omega`` is a smooth function that is zero
    *on* the boundary and approximately equal to the Euclidean distance away
    from it (to first order).  A field of the form ``g + omega * net`` then
    equals the prescribed value ``g`` exactly on the boundary (since omega=0
    there) while leaving ``net`` free in the interior.  ``r_equivalence``
    combines several patch ADFs into a single ADF for their union.
"""

from typing import Dict, List, Tuple

import torch
from torch import Tensor


def _distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Euclidean distance between two scalar points."""
    return float(((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5)


def _center(p1: Tuple[float, float], p2: Tuple[float, float]) -> Tuple[float, float]:
    """Midpoint of two scalar points."""
    return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)


def infinite_line_adf(
    points: Tuple[Tensor, Tensor],
    p1: Tuple[float, float],
    p2: Tuple[float, float],
) -> Tensor:
    """Signed distance to the infinite line through ``p1`` and ``p2``."""
    L = _distance(p1, p2)
    return (
        (points[0] - p1[0]) * (p2[1] - p1[1]) - (points[1] - p1[1]) * (p2[0] - p1[0])
    ) / L


def circle_adf(
    points: Tuple[Tensor, Tensor], radius: float, center: Tuple[float, float]
) -> Tensor:
    """ADF for a disk of given radius/center (positive inside)."""
    return (
        radius**2 - ((points[0] - center[0]) ** 2 + (points[1] - center[1]) ** 2)
    ) / (2 * radius)


def line_segment_adf(
    points: Tuple[Tensor, Tensor],
    p1: Tuple[float, float],
    p2: Tuple[float, float],
) -> Tensor:
    """Approximate distance function for a finite line segment ``p1``--``p2``.

    Uses the Rvachev trimming construction: the infinite-line ADF ``f`` is
    trimmed by the bounding circle ADF ``t`` so the result decays correctly
    past the segment endpoints.
    """
    L = _distance(p1, p2)
    center = _center(p1, p2)
    f = infinite_line_adf(points, p1, p2)
    t = circle_adf(points, L / 2.0, center)
    phi = torch.sqrt(t**2 + f**4)
    return torch.sqrt(f**2 + ((phi - t) / 2.0) ** 2)


def r_equivalence(omegas: List[Tensor], m: float = 2.0) -> Tensor:
    """R-equivalence: combine patch ADFs into a single union ADF.

    ``omega_E = (sum_i omega_i**-m) ** (-1/m)`` -- a smooth approximation to
    ``min_i omega_i`` that stays zero on every constituent boundary.
    """
    omega_E = torch.zeros_like(omegas[0])
    for omega in omegas:
        omega_E = omega_E + 1.0 / omega**m
    return 1.0 / omega_E ** (1.0 / m)


def enforce_hard_bc(
    x: Tensor,
    y: Tensor,
    starred: Dict[str, Tensor],
    phif_initial_fn,
    lini: float,
) -> Dict[str, Tensor]:
    """Map raw starred network outputs to enforced physical fields (passive mode).

    Parameters
    ----------
    x, y : Tensor
        Column tensors ``[N, 1]`` of nondimensional space and time.
    starred : Dict[str, Tensor]
        Raw network outputs keyed ``cCV_star``, ``cAV_star``, ``phif_star``,
        ``l_star`` (each ``[N, 1]``).
    phif_initial_fn : callable
        ``lambdify``-d ``phiext(y) - m*x`` giving the initial film potential.
    lini : float
        Dimensionless initial film thickness.

    Returns
    -------
    Dict[str, Tensor]
        Enforced fields ``cCV``, ``cAV``, ``phif``, ``l`` plus the derived
        ``phimf`` (potential nearest x=0) and ``phifs`` (nearest x=1).
    """
    # ADF for the bottom edge y=0 (initial-condition boundary)
    omega_initial = line_segment_adf((x, y), (0.0, 0.0), (1.0, 0.0))

    # non-constant initial film potential, evaluated on the coordinate grid
    x_np = x.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    phif_initial = torch.as_tensor(
        phif_initial_fn(x_np, y_np), dtype=x.dtype, device=x.device
    )

    # phif: equals phif_initial on y=0, free elsewhere
    phif = phif_initial + omega_initial * starred["phif_star"]

    # potentials nearest the two interfaces (used by film_growth / fluxes).
    # argmin/argmax of x picks the sample closest to x=0 / x=1 in the batch.
    phimf = torch.full_like(phif, float(phif[torch.argmin(x)].detach()))
    phifs = torch.full_like(phif, float(phif[torch.argmax(x)].detach()))

    out = {
        "phif": phif,
        "phimf": phimf,
        "phifs": phifs,
        # cCV, cAV: zero initial condition -> g = 0
        "cCV": omega_initial * starred["cCV_star"],
        "cAV": omega_initial * starred["cAV_star"],
        # l: equals lini on y=0
        "l": torch.full_like(x, lini) + omega_initial * starred["l_star"],
    }
    return out
