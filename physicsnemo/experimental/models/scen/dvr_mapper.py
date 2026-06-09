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

"""1D spectral element mapper: reference [-1,1] → physical [a, b]."""

from __future__ import annotations

import math

import torch

from physicsnemo.experimental.models.scen.quadrature import get_quadrature_data

_VALID_MAPPINGS = ("kte", "log")


class DVRMapper:
    """Maps N reference quadrature nodes on [-1, 1] to physical nodes on [a, b].

    Pre-computes physical differentiation matrices D1, D2, D4 and quadrature
    weights with the physical-domain Jacobian included.  These matrices are
    used in place of automatic differentiation in SCEN physics losses.

    Parameters
    ----------
    N : int
        Number of nodes (including both endpoints).
    a : float
        Physical domain left endpoint.
    b : float
        Physical domain right endpoint.
    alpha : float
        Mapping strength (0 = linear/uniform).

        - ``mapping='kte'`` : KTE parameter in [0, 1).
          0 → uniform, 0.97 → strong clustering near both endpoints.
        - ``mapping='log'`` : log10 of compression ratio, in [0, ∞).
          0 → uniform, 2 → 100× more nodes near x=a.
    quadrature : str
        One of ``'lgl'``, ``'chebyshev'``, ``'clenshaw_curtis'``.
    mapping : str
        One of ``'kte'`` (Kosloff-Tal-Ezer) or ``'log'`` (logarithmic).
        KTE clusters near both endpoints; log clusters near x=a only
        (ideal for electrode boundary layers).
    dtype : torch.dtype
        Tensor dtype for all precomputed arrays.  Default ``torch.float32``.
    device : torch.device or None
        Target device.  Default ``cpu``.
    """

    def __init__(
        self,
        N: int,
        a: float,
        b: float,
        alpha: float = 0.0,
        quadrature: str = "lgl",
        mapping: str = "kte",
        dtype: torch.dtype = torch.float32,
        device=None,
    ):
        if device is None:
            device = torch.device("cpu")
        self.dtype = dtype
        self.device = torch.device(device) if not isinstance(device, torch.device) else device

        mapping = mapping.lower()
        if mapping not in _VALID_MAPPINGS:
            raise ValueError(f"mapping must be one of {_VALID_MAPPINGS}, got '{mapping}'")

        xi_ref, w_ref, D_ref = get_quadrature_data(
            N, rule=quadrature, dtype=dtype, device=self.device
        )
        self._xi_ref = xi_ref

        if mapping == "kte":
            if alpha == 0.0:
                x_phys = (a + b) / 2 + (b - a) / 2 * xi_ref
                J = torch.full((N,), (b - a) / 2, dtype=dtype, device=self.device)
            else:
                asin_alpha = math.asin(alpha)
                x_phys = (
                    (a + b) / 2
                    + (b - a) / 2 * torch.arcsin(alpha * xi_ref) / asin_alpha
                )
                J = (
                    (b - a)
                    / 2
                    * alpha
                    / (asin_alpha * torch.sqrt(1 - alpha**2 * xi_ref**2))
                )
        else:  # log
            if alpha == 0.0:
                x_phys = (a + b) / 2 + (b - a) / 2 * xi_ref
                J = torch.full((N,), (b - a) / 2, dtype=dtype, device=self.device)
            else:
                ln_beta = alpha * math.log(10.0)
                beta = math.exp(ln_beta)
                t = (xi_ref + 1.0) / 2.0
                bt = torch.exp(t * ln_beta)
                x_phys = a + (b - a) * (bt - 1.0) / (beta - 1.0)
                J = (b - a) / 2.0 * bt * ln_beta / (beta - 1.0)

        # Physical D1: exact chain rule.  D2 = D1 @ D1 (exact for nonlinear maps).
        invJ = 1.0 / J
        D1_phys = invJ.unsqueeze(1) * D_ref
        D2_phys = D1_phys @ D1_phys
        D4_phys = D2_phys @ D2_phys
        w_phys = J.abs() * w_ref

        self._nodes = x_phys
        self._weights = w_phys
        self._D1 = D1_phys
        self._D2 = D2_phys
        self._D4 = D4_phys

    @property
    def xi_ref(self) -> torch.Tensor:
        """Reference LGL coordinates ξ ∈ [-1, 1], shape ``(N,)``."""
        return self._xi_ref

    @property
    def nodes(self) -> torch.Tensor:
        """Physical node positions x ∈ [a, b], shape ``(N,)``."""
        return self._nodes

    @property
    def weights(self) -> torch.Tensor:
        """Physical quadrature weights (Jacobian-scaled), shape ``(N,)``."""
        return self._weights

    @property
    def D1(self) -> torch.Tensor:
        """First-derivative matrix ∂/∂x, shape ``(N, N)``."""
        return self._D1

    @property
    def D2(self) -> torch.Tensor:
        """Second-derivative matrix ∂²/∂x², shape ``(N, N)``."""
        return self._D2

    @property
    def D4(self) -> torch.Tensor:
        """Fourth-derivative matrix ∂⁴/∂x⁴, shape ``(N, N)``."""
        return self._D4
