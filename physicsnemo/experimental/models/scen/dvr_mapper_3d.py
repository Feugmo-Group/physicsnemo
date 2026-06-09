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

"""3D tensor-product LGL grid via Kronecker products.

Derivative operators::

    D1x = kron(D1_x, I_{Ny·Nz})
    D1y = kron(I_Nx, kron(D1_y, I_Nz))
    D1z = kron(I_{Nx·Ny}, D1_z)
    laplacian = D2x + D2y + D2z

Node ordering: row-major (x outermost, z innermost).
Flat index ``i*(Ny·Nz) + j*Nz + k`` ↔ physical point ``(x_i, y_j, z_k)``.
"""

from __future__ import annotations

import warnings

import torch

from physicsnemo.experimental.models.scen.dvr_mapper import DVRMapper


class DVRMapper3D:
    """3D tensor-product spectral element grid.

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
        Nodes along y.  Defaults to ``Nx``.
    ay : float, optional
        y-domain left endpoint.  Defaults to ``ax``.
    by : float, optional
        y-domain right endpoint.  Defaults to ``bx``.
    alpha_y : float, optional
        y stretching.  Defaults to ``alpha_x``.
    Nz : int, optional
        Nodes along z.  Defaults to ``Nx``.
    az : float, optional
        z-domain left endpoint.  Defaults to ``ax``.
    bz : float, optional
        z-domain right endpoint.  Defaults to ``bx``.
    alpha_z : float, optional
        z stretching.  Defaults to ``alpha_x``.
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
        Nz: int = None,
        az: float = None,
        bz: float = None,
        alpha_z: float = None,
        quadrature: str = "lgl",
        mapping: str = "kte",
        dtype: torch.dtype = torch.float32,
        device=None,
    ):
        if device is None:
            device = torch.device("cpu")
        if Ny is None:
            Ny = Nx
        if Nz is None:
            Nz = Nx
        if ay is None:
            ay = ax
        if by is None:
            by = bx
        if az is None:
            az = ax
        if bz is None:
            bz = bx
        if alpha_y is None:
            alpha_y = alpha_x
        if alpha_z is None:
            alpha_z = alpha_x

        self.Nx, self.Ny, self.Nz = Nx, Ny, Nz
        self.dtype = dtype
        self.device = torch.device(device) if not isinstance(device, torch.device) else device

        total = Nx * Ny * Nz
        if total > 2048:
            mem_mb = total**2 * 8 / 1e6
            warnings.warn(
                f"DVRMapper3D: Nx·Ny·Nz = {total} nodes — each matrix is "
                f"({total}×{total}) = {mem_mb:.0f} MB in float64. "
                f"Consider reducing N or switching to matrix-free operators.",
                stacklevel=2,
            )

        self.mapper_x = DVRMapper(Nx, ax, bx, alpha_x, quadrature, mapping, dtype, device)
        self.mapper_y = DVRMapper(Ny, ay, by, alpha_y, quadrature, mapping, dtype, device)
        self.mapper_z = DVRMapper(Nz, az, bz, alpha_z, quadrature, mapping, dtype, device)

        kw = dict(dtype=dtype, device=self.device)
        Ix = torch.eye(Nx, **kw)
        Iy = torch.eye(Ny, **kw)
        Iz = torch.eye(Nz, **kw)
        IyIz = torch.kron(Iy, Iz)
        IxIy = torch.kron(Ix, Iy)

        self._D1x = torch.kron(self.mapper_x.D1, IyIz)
        self._D1y = torch.kron(Ix, torch.kron(self.mapper_y.D1, Iz))
        self._D1z = torch.kron(IxIy, self.mapper_z.D1)

        self._D2x = torch.kron(self.mapper_x.D2, IyIz)
        self._D2y = torch.kron(Ix, torch.kron(self.mapper_y.D2, Iz))
        self._D2z = torch.kron(IxIy, self.mapper_z.D2)

        self._laplacian = self._D2x + self._D2y + self._D2z
        self._weights = torch.kron(
            torch.kron(self.mapper_x.weights, self.mapper_y.weights),
            self.mapper_z.weights,
        )

        X, Y, Z = torch.meshgrid(
            self.mapper_x.nodes,
            self.mapper_y.nodes,
            self.mapper_z.nodes,
            indexing="ij",
        )
        self._xyz_nodes = torch.stack(
            [X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], dim=1
        )

    @property
    def xyz_nodes(self) -> torch.Tensor:
        """Physical grid points, shape ``(Nx·Ny·Nz, 3)``."""
        return self._xyz_nodes

    @property
    def x_nodes(self) -> torch.Tensor:
        """1D x-axis nodes, shape ``(Nx,)``."""
        return self.mapper_x.nodes

    @property
    def y_nodes(self) -> torch.Tensor:
        """1D y-axis nodes, shape ``(Ny,)``."""
        return self.mapper_y.nodes

    @property
    def z_nodes(self) -> torch.Tensor:
        """1D z-axis nodes, shape ``(Nz,)``."""
        return self.mapper_z.nodes

    @property
    def weights(self) -> torch.Tensor:
        """Quadrature weights, shape ``(Nx·Ny·Nz,)``.  Sum = Lx·Ly·Lz."""
        return self._weights

    @property
    def w_norm(self) -> torch.Tensor:
        """Normalised weights: ``weights / sum(weights)``."""
        return self._weights / self._weights.sum()

    @property
    def D1x(self) -> torch.Tensor:
        return self._D1x

    @property
    def D1y(self) -> torch.Tensor:
        return self._D1y

    @property
    def D1z(self) -> torch.Tensor:
        return self._D1z

    @property
    def D2x(self) -> torch.Tensor:
        return self._D2x

    @property
    def D2y(self) -> torch.Tensor:
        return self._D2y

    @property
    def D2z(self) -> torch.Tensor:
        return self._D2z

    @property
    def laplacian(self) -> torch.Tensor:
        """∇² = ∂²/∂x² + ∂²/∂y² + ∂²/∂z², shape ``(Nx·Ny·Nz, Nx·Ny·Nz)``."""
        return self._laplacian

    @property
    def shape(self) -> tuple[int, int, int]:
        """Grid shape ``(Nx, Ny, Nz)``."""
        return (self.Nx, self.Ny, self.Nz)

    def __repr__(self) -> str:
        x0, x1 = self.mapper_x.nodes[0].item(), self.mapper_x.nodes[-1].item()
        y0, y1 = self.mapper_y.nodes[0].item(), self.mapper_y.nodes[-1].item()
        z0, z1 = self.mapper_z.nodes[0].item(), self.mapper_z.nodes[-1].item()
        return (
            f"DVRMapper3D(Nx={self.Nx}, Ny={self.Ny}, Nz={self.Nz}, "
            f"x=[{x0:.3g},{x1:.3g}], y=[{y0:.3g},{y1:.3g}], "
            f"z=[{z0:.3g},{z1:.3g}], dtype={self.dtype})"
        )
