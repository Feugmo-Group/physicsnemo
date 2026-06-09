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

"""Spectral Collocation Element Networks (SCEN) — experimental.

Provides spectral element infrastructure and the
:class:`~physicsnemo.experimental.models.scen.element_network.SCENElementNetwork`
model for physics-informed neural network training on multi-element grids.

Key components
--------------
:class:`DVRMapper`
    1D spectral element mapper with precomputed D1/D2/D4 matrices.
:class:`DVRMapper2D`
    2D tensor-product grid via Kronecker products.
:class:`DVRMapper3D`
    3D tensor-product grid via Kronecker products.
:class:`LegendreKAN`
    Kolmogorov-Arnold Network with Legendre polynomial basis.
:class:`SCENElementNetwork`
    Vmap-batched multi-element PINN model.
:func:`compute_mortar_projection`
    Lagrange projection for non-conforming element interfaces.
:func:`get_quadrature_data`
    LGL / Chebyshev / Clenshaw-Curtis quadrature rules.
"""

from physicsnemo.experimental.models.scen.dvr_mapper import DVRMapper
from physicsnemo.experimental.models.scen.dvr_mapper_2d import DVRMapper2D
from physicsnemo.experimental.models.scen.dvr_mapper_3d import DVRMapper3D
from physicsnemo.experimental.models.scen.element_network import SCENElementNetwork
from physicsnemo.experimental.models.scen.legendre_kan import (
    LegendreKAN,
    LegendreKANLayer,
    legendre_basis,
    make_legendre_dleg,
    make_nodal_deriv_mats,
    make_vandermonde,
)
from physicsnemo.experimental.models.scen.mortar import compute_mortar_projection
from physicsnemo.experimental.models.scen.quadrature import get_quadrature_data

__all__ = [
    "DVRMapper",
    "DVRMapper2D",
    "DVRMapper3D",
    "SCENElementNetwork",
    "LegendreKAN",
    "LegendreKANLayer",
    "legendre_basis",
    "make_legendre_dleg",
    "make_nodal_deriv_mats",
    "make_vandermonde",
    "compute_mortar_projection",
    "get_quadrature_data",
]
