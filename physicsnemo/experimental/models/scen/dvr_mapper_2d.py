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

"""2D tensor-product LGL grid via Kronecker products.

Derivative operators::

    D1x = kron(D1_x, I_Ny)        shape (Nx·Ny, Nx·Ny)
    D1y = kron(I_Nx, D1_y)        shape (Nx·Ny, Nx·Ny)
    laplacian = D2x + D2y

Node ordering: row-major (x outer, y inner).
Flat index ``i*Ny + j`` ↔ physical point ``(x_i, y_j)``.
"""

from __future__ import annotations

import warnings

import torch

from physicsnemo.experimental.models.scen.dvr_mapper import DVRMapper


class DVRMapper2D:
    """2D tensor-product spectral element grid.

    Parameters
    ----------
    Nx : int
        Number of nodes along x-axis.
    ax : float
        x-domain left endpoint.
    bx : float
        x-domain right endpoint.
    alpha_x : float
        Mapping stretching for x-axis.  0 = uniform.
    Ny : int, optional
        Number of nodes along y-axis.  Defaults to ``Nx`` (square grid).
    ay : float, optional
        y-domain left endpoint.  Defaults to ``ax``.
    by : float, optional
        y-domain right endpoint.  Defaults to ``bx``.
    alpha_y : float, optional
        Mapping stretching for y-axis.  Defaults to ``alpha_x``.
    quadrature : str
        One of ``'lgl'``, ``'chebyshev'``, ``'clenshaw_curtis'``.
    mapping : str
        One of ``'kte'`` or ``'log'``.
    dtype : torch.dtype
        Tensor dtype.  Default ``torch.float32``.
    device : torch.device or None
        Target device.  Default ``cpu``.
    """

    def __init__(
        self,
        Nx: int,
        ax: float,
        bx: float,
        alpha_x: float = 0.0,
        Ny: int = None,
        ay: float = None,
        by: float = None,
        alpha_y: float = None,
        quadrature: str = "lgl",
        mapping: str = "kte",
        dtype: torch.dtype = torch.float32,
        device=None,
    ):
        if device is None:
            device = torch.device("cpu")
        if Ny is None:
            Ny = Nx
        if ay is None:
            ay = ax
        if by is None:
            by = bx
        if alpha_y is None:
            alpha_y = alpha_x

        self.Nx, self.Ny = Nx, Ny
        self.dtype = dtype
        self.device = torch.device(device) if not isinstance(device, torch.device) else device

        total = Nx * Ny
        if total > 1024:
            warnings.warn(
                f"DVRMapper2D: Nx·Ny = {total} nodes — each derivative matrix is "
                f"({total}×{total}) = {total**2 * 8 / 1e6:.0f} MB in float64. "
                f"Consider reducing N or using sparse operators.",
                stacklevel=2,
            )

        self.mapper_x = DVRMapper(Nx, ax, bx, alpha_x, quadrature, mapping, dtype, device)
        self.mapper_y = DVRMapper(Ny, ay, by, alpha_y, quadrature, mapping, dtype, device)

        kw = dict(dtype=dtype, device=self.device)
        Ix = torch.eye(Nx, **kw)
        Iy = torch.eye(Ny, **kw)

        self._D1x = torch.kron(self.mapper_x.D1, Iy)
        self._D1y = torch.kron(Ix, self.mapper_y.D1)
        self._D2x = torch.kron(self.mapper_x.D2, Iy)
        self._D2y = torch.kron(Ix, self.mapper_y.D2)
        self._laplacian = self._D2x + self._D2y
        self._weights = torch.kron(self.mapper_x.weights, self.mapper_y.weights)

        X, Y = torch.meshgrid(self.mapper_x.nodes, self.mapper_y.nodes, indexing="ij")
        self._xy_nodes = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)

    @property
    def xy_nodes(self) -> torch.Tensor:
        """Physical grid points, shape ``(Nx·Ny, 2)``."""
        return self._xy_nodes

    @property
    def x_nodes(self) -> torch.Tensor:
        """1D x-axis nodes, shape ``(Nx,)``."""
        return self.mapper_x.nodes

    @property
    def y_nodes(self) -> torch.Tensor:
        """1D y-axis nodes, shape ``(Ny,)``."""
        return self.mapper_y.nodes

    @property
    def weights(self) -> torch.Tensor:
        """Quadrature weights, shape ``(Nx·Ny,)``.  Sum = (bx-ax)·(by-ay)."""
        return self._weights

    @property
    def w_norm(self) -> torch.Tensor:
        """Normalised weights: ``weights / sum(weights)``."""
        return self._weights / self._weights.sum()

    @property
    def D1x(self) -> torch.Tensor:
        """∂/∂x operator, shape ``(Nx·Ny, Nx·Ny)``."""
        return self._D1x

    @property
    def D1y(self) -> torch.Tensor:
        """∂/∂y operator, shape ``(Nx·Ny, Nx·Ny)``."""
        return self._D1y

    @property
    def D2x(self) -> torch.Tensor:
        """∂²/∂x² operator, shape ``(Nx·Ny, Nx·Ny)``."""
        return self._D2x

    @property
    def D2y(self) -> torch.Tensor:
        """∂²/∂y² operator, shape ``(Nx·Ny, Nx·Ny)``."""
        return self._D2y

    @property
    def laplacian(self) -> torch.Tensor:
        """∇² = ∂²/∂x² + ∂²/∂y², shape ``(Nx·Ny, Nx·Ny)``."""
        return self._laplacian

    @property
    def shape(self) -> tuple[int, int]:
        """Grid shape ``(Nx, Ny)``."""
        return (self.Nx, self.Ny)

    def __repr__(self) -> str:
        x0, x1 = self.mapper_x.nodes[0].item(), self.mapper_x.nodes[-1].item()
        y0, y1 = self.mapper_y.nodes[0].item(), self.mapper_y.nodes[-1].item()
        return (
            f"DVRMapper2D(Nx={self.Nx}, Ny={self.Ny}, "
            f"x=[{x0:.3g},{x1:.3g}], y=[{y0:.3g},{y1:.3g}], "
            f"dtype={self.dtype})"
        )
