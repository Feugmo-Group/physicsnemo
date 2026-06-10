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

r"""Kolmogorov-Arnold Network (KAN) with B-spline edge activations.

Ported from the legacy ``physicsnemo.sym`` ``Arch`` API to a plain
tensor-in / tensor-out :class:`physicsnemo.Module`.  The original model
operated on dictionaries of named keys; this implementation takes a single
coordinate tensor of shape :math:`(B, D_{in})` and returns a tensor of shape
:math:`(B, D_{out})`.

This module also exposes :class:`KANLayer` and :class:`KolmogorovArnoldNetCore`,
which are reused by the Fourier-feature variant in
:mod:`physicsnemo.experimental.models.pinn.fourier_kan`.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

import physicsnemo
from physicsnemo.core.meta import ModelMetaData
from physicsnemo.nn import get_activation


class KANLayer(nn.Module):
    r"""Single Kolmogorov-Arnold Network layer with B-spline edge activations.

    Each directed edge between an input feature and an output feature carries a
    learnable B-spline of order ``spline_order`` defined over a knot grid.  A
    residual linear branch (applied to a base activation of the input) is added
    to the spline output.

    The knot grid is either uniform (a non-persistent buffer) or learnable
    (``free_knot=True``), in which case strictly-positive knot gaps are produced
    via a softplus and accumulated.

    Parameters
    ----------
    in_features : int
        Number of input features :math:`D_{in}`.
    out_features : int
        Number of output features :math:`D_{out}`.
    grid_size : int, optional, default=100
        Number of B-spline grid intervals.
    spline_order : int, optional, default=3
        Order of the B-splines (``3`` = cubic).
    base_activation : torch.nn.Module, optional, default=None
        Base activation applied to the residual linear branch.  Defaults to
        :class:`torch.nn.Tanh` when ``None``.
    grid_range : Tuple[float, float], optional, default=(-1.0, 1.0)
        Range over which the knot grid is initialised.
    free_knot : bool, optional, default=False
        Use learnable (non-uniform) knot placement.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(\dots, D_{in})`.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(\dots, D_{out})`.

    Examples
    --------
    >>> import torch
    >>> layer = KANLayer(3, 5, grid_size=8, spline_order=3)
    >>> layer(torch.randn(16, 3)).shape
    torch.Size([16, 5])
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 100,
        spline_order: int = 3,
        base_activation: Optional[nn.Module] = None,
        grid_range: Tuple[float, float] = (-1.0, 1.0),
        free_knot: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.grid_range = grid_range
        self.free_knot = free_knot

        # base activation for the residual branch
        if base_activation is None:
            self.base_activation = nn.Tanh()
        else:
            self.base_activation = base_activation

        # learnable weights
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, grid_size + spline_order)
        )

        # knot grid: learnable gaps or a fixed uniform grid
        if free_knot:
            self.knot_gaps = nn.Parameter(
                torch.zeros(in_features, grid_size + 2 * spline_order)
            )
            self.register_buffer("_grid_range", torch.tensor(grid_range))
        else:
            h = (grid_range[1] - grid_range[0]) / grid_size
            grid_uniform = (
                torch.linspace(
                    grid_range[0] - h * spline_order,
                    grid_range[1] + h * spline_order,
                    grid_size + 2 * spline_order + 1,
                )
                .expand(in_features, -1)
                .contiguous()
            )
            self.register_buffer("_grid_uniform", grid_uniform, persistent=False)

        self.reset_parameters()

    @property
    def grid(self) -> Tensor:
        r"""Return the knot grid of shape :math:`(D_{in}, G + 2s + 1)`.

        For ``free_knot=True`` the grid is reconstructed from strictly-positive
        learnable gaps; otherwise the fixed uniform grid buffer is returned.

        Returns
        -------
        torch.Tensor
            Knot grid of shape :math:`(D_{in}, G + 2s + 1)`.
        """
        if self.free_knot:
            # strictly-positive gaps via softplus
            gaps = F.softplus(self.knot_gaps) + 1e-6

            # normalise gaps so the total padded width is preserved
            total_width = self._grid_range[1] - self._grid_range[0]
            h_avg = total_width / self.grid_size
            total_padded_width = total_width + (2 * self.spline_order * h_avg)
            normalized_gaps = gaps * (
                total_padded_width / gaps.sum(dim=-1, keepdim=True)
            )

            # accumulate gaps into knot positions, anchored at the padded start
            start_point = self._grid_range[0] - (self.spline_order * h_avg)
            grid = torch.cumsum(normalized_gaps, dim=-1)
            zeros = torch.zeros(
                self.in_features, 1, device=grid.device, dtype=grid.dtype
            )
            grid = torch.cat([zeros, grid], dim=-1) + start_point
            return grid
        return self._grid_uniform

    def reset_parameters(self) -> None:
        r"""Initialise the base and spline weights with Kaiming-uniform.

        Returns
        -------
        None
        """
        nn.init.kaiming_uniform_(self.base_weight, nonlinearity="linear")
        nn.init.kaiming_uniform_(self.spline_weight, nonlinearity="linear")

    def b_splines(self, x: Tensor) -> Tensor:
        r"""Evaluate the B-spline basis at ``x``.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape :math:`(\dots, D_{in})`.

        Returns
        -------
        torch.Tensor
            Basis tensor of shape :math:`(\dots, D_{in}, G + s)`.
        """
        grid = self.grid
        x = x.to(grid.device)
        x_unsqueezed = x.unsqueeze(-1)

        # order-0 indicator bases on each knot interval
        bases = ((x_unsqueezed >= grid[..., :-1]) & (x_unsqueezed < grid[..., 1:])).to(
            x.dtype
        )

        # Cox-de Boor recursion to the requested spline order
        for k in range(1, self.spline_order + 1):
            left_intervals = grid[..., : -(k + 1)]
            right_intervals = grid[..., k:-1]
            next_intervals = grid[..., k + 1 :]
            shifted_intervals = grid[..., 1:-k]

            delta_left = torch.where(
                right_intervals == left_intervals,
                torch.ones_like(right_intervals),
                right_intervals - left_intervals,
            )
            delta_right = next_intervals - shifted_intervals

            term1 = (x_unsqueezed - left_intervals) / delta_left * bases[..., :-1]
            term2 = (next_intervals - x_unsqueezed) / delta_right * bases[..., 1:]
            bases = term1 + term2

        return bases.contiguous()

    def forward(self, x: Tensor) -> Tensor:
        r"""Apply the KAN layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape :math:`(\dots, D_{in})`.

        Returns
        -------
        torch.Tensor
            Output tensor of shape :math:`(\dots, D_{out})`.
        """
        # spline branch: contract bases with per-edge spline weights
        bases = self.b_splines(x)  # (..., D_in, G + s)
        spline_output = torch.einsum("...ib,oib->...o", bases, self.spline_weight)

        # residual linear branch on the base activation
        base_output = F.linear(self.base_activation(x), self.base_weight)
        return spline_output + base_output


class KolmogorovArnoldNetCore(nn.Module):
    r"""Stack of :class:`KANLayer` modules forming a KAN feed-forward network.

    Parameters
    ----------
    in_features : int
        Input dimension :math:`D_{in}`.
    out_features : int
        Output dimension :math:`D_{out}`.
    layer_size : int, optional, default=5
        Hidden-layer width.
    nr_layers : int, optional, default=2
        Number of hidden layers.
    grid_size : int, optional, default=10
        Number of B-spline grid intervals per layer.
    spline_order : int, optional, default=3
        Order of the B-splines.
    base_activation : torch.nn.Module, optional, default=None
        Residual base activation passed to every :class:`KANLayer`.  Defaults to
        :class:`torch.nn.Tanh`.
    grid_range : Tuple[float, float], optional, default=(-1.0, 1.0)
        Knot grid range.
    free_knot : bool, optional, default=False
        Use learnable knot placement.
    outer_activation : torch.nn.Module, optional, default=None
        Non-linearity applied between hidden layers.  Defaults to
        :class:`torch.nn.Tanh`.
    outer_layer : bool, optional, default=True
        Whether to apply ``outer_activation`` between hidden layers.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, D_{in})`.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(B, D_{out})`.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        layer_size: int = 5,
        nr_layers: int = 2,
        grid_size: int = 10,
        spline_order: int = 3,
        base_activation: Optional[nn.Module] = None,
        grid_range: Tuple[float, float] = (-1.0, 1.0),
        free_knot: bool = False,
        outer_activation: Optional[nn.Module] = None,
        outer_layer: bool = True,
    ) -> None:
        super().__init__()
        self.outer_layer = outer_layer
        self.outer_activation = (
            nn.Tanh() if outer_activation is None else outer_activation
        )

        sizes: List[int] = [in_features] + [layer_size] * nr_layers + [out_features]
        self.layers = nn.ModuleList(
            KANLayer(
                in_features=sizes[i],
                out_features=sizes[i + 1],
                grid_size=grid_size,
                spline_order=spline_order,
                base_activation=base_activation,
                grid_range=grid_range,
                free_knot=free_knot,
            )
            for i in range(len(sizes) - 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        r"""Apply the stacked KAN layers.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape :math:`(B, D_{in})`.

        Returns
        -------
        torch.Tensor
            Output tensor of shape :math:`(B, D_{out})`.
        """
        for layer in self.layers[:-1]:
            x = layer(x)
            if self.outer_layer:
                x = self.outer_activation(x)
        # final layer is linear so the output range is unconstrained
        return self.layers[-1](x)


class KolmogorovArnoldNet(physicsnemo.Module):
    r"""Kolmogorov-Arnold Network for physics-informed learning.

    A KAN replaces fixed scalar activations with learnable B-spline functions on
    every edge.  Inputs are normalised from ``domain_bounds`` into
    :math:`[-1, 1]` before the KAN core, keeping B-spline evaluations
    well-conditioned.

    Based on `Kolmogorov-Arnold-Informed neural networks
    <https://doi.org/10.1016/j.cma.2025.117518>`_.  The optional learnable knot
    placement follows `Free-Knots Kolmogorov-Arnold Networks
    <https://arxiv.org/abs/2501.09283>`_.

    Parameters
    ----------
    in_features : int
        Number of input coordinate features :math:`D_{in}`.
    out_features : int
        Number of output features :math:`D_{out}`.
    layer_size : int, optional, default=5
        Hidden-layer width.
    nr_layers : int, optional, default=2
        Number of hidden layers.
    grid_size : int, optional, default=10
        Number of B-spline grid intervals per layer.
    spline_order : int, optional, default=3
        Order of the B-splines (``3`` = cubic).
    base_activation_fn : str, optional, default="tanh"
        Name of the residual base activation, resolved via
        :func:`~physicsnemo.nn.get_activation`.
    grid_range : Tuple[float, float], optional, default=(-1.0, 1.0)
        Knot grid range.
    domain_bounds : List[Tuple[float, float]], optional, default=None
        Per-dimension input bounds ``[(min, max), ...]`` used for normalisation.
        Defaults to ``(-1, 1)`` per dimension when ``None``.
    free_knot : bool, optional, default=False
        Use learnable knot placement.
    outer_layer_fn : str, optional, default="tanh"
        Name of the inter-layer non-linearity, resolved via
        :func:`~physicsnemo.nn.get_activation`.
    outer_layer : bool, optional, default=True
        Whether to apply ``outer_layer_fn`` between hidden layers.

    Forward
    -------
    x : torch.Tensor
        Coordinate tensor of shape :math:`(B, D_{in})`.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(B, D_{out})`.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.experimental.models.pinn.kan import KolmogorovArnoldNet
    >>> model = KolmogorovArnoldNet(in_features=2, out_features=1)
    >>> x = torch.randn(32, 2)
    >>> model(x).shape
    torch.Size([32, 1])
    """

    __model_checkpoint_version__ = "0.1.0"

    def __init__(
        self,
        in_features: int,
        out_features: int,
        layer_size: int = 5,
        nr_layers: int = 2,
        grid_size: int = 10,
        spline_order: int = 3,
        base_activation_fn: str = "tanh",
        grid_range: Tuple[float, float] = (-1.0, 1.0),
        domain_bounds: Optional[List[Tuple[float, float]]] = None,
        free_knot: bool = False,
        outer_layer_fn: str = "tanh",
        outer_layer: bool = True,
    ) -> None:
        super().__init__(meta=ModelMetaData(func_torch=True, auto_grad=True))

        self.in_features = in_features
        self.out_features = out_features
        self.layer_size = layer_size
        self.nr_layers = nr_layers
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.base_activation_fn = base_activation_fn
        self.outer_layer_fn = outer_layer_fn
        self.grid_range = grid_range
        self.free_knot = free_knot
        self.outer_layer = outer_layer

        # normalisation bounds, default to [-1, 1] per dimension
        if domain_bounds is None:
            bounds = torch.tensor([[-1.0, 1.0]] * in_features)
        else:
            if len(domain_bounds) != in_features:
                raise ValueError(
                    f"Expected {in_features} domain bounds, got {len(domain_bounds)}"
                )
            bounds = torch.tensor([[float(a), float(b)] for a, b in domain_bounds])
        self.register_buffer("input_min", bounds[:, 0])
        self.register_buffer("input_max", bounds[:, 1])

        self._impl = KolmogorovArnoldNetCore(
            in_features=in_features,
            out_features=out_features,
            layer_size=layer_size,
            nr_layers=nr_layers,
            grid_size=grid_size,
            spline_order=spline_order,
            base_activation=get_activation(base_activation_fn),
            grid_range=grid_range,
            free_knot=free_knot,
            outer_activation=get_activation(outer_layer_fn),
            outer_layer=outer_layer,
        )

    def forward(
        self, x: Float[Tensor, "batch in_features"]
    ) -> Float[Tensor, "batch out_features"]:
        r"""Forward pass through the KAN.

        Parameters
        ----------
        x : torch.Tensor
            Coordinate tensor of shape :math:`(B, D_{in})`.

        Returns
        -------
        torch.Tensor
            Output tensor of shape :math:`(B, D_{out})`.
        """
        if not torch.compiler.is_compiling():
            if x.ndim < 2 or x.shape[-1] != self.in_features:
                raise ValueError(
                    f"Expected input of shape (B, {self.in_features}), "
                    f"got tensor of shape {tuple(x.shape)}"
                )

        # normalise coordinates into [-1, 1]
        denom = self.input_max - self.input_min
        denom = torch.where(denom == 0, torch.ones_like(denom), denom)
        x_norm = ((x - self.input_min) / denom) * 2 - 1

        return self._impl(x_norm)
