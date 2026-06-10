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

r"""Finite Basis Physics-Informed Neural Network (FB-PINN).

A finite-basis network built from overlapping domain decomposition.  The domain
is partitioned into a grid of overlapping subdomains, each served by its own MLP.
The subnetwork outputs are blended with smooth window functions and a
partition-of-unity normalisation.

All subnetworks share an identical topology and are evaluated in one batched
forward pass using :func:`torch.baddbmm` (see :class:`BatchedFullyConnected`).

Based on `Multilevel domain decomposition-based architectures for
physics-informed neural networks <https://doi.org/10.1016/j.cma.2024.117116>`_.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

import physicsnemo
from physicsnemo.core.meta import ModelMetaData
from physicsnemo.nn import get_activation


class BatchedFullyConnected(nn.Module):
    r"""Stack of identical MLPs evaluated in one batched forward pass.

    Represents ``nr_subnetworks`` fully-connected MLPs of identical topology.
    The stacked weights and biases are applied with :func:`torch.baddbmm` so
    that all subnetworks run in a single batched matmul.

    Parameters
    ----------
    nr_subnetworks : int
        Number of independent MLPs :math:`S`.
    in_features : int
        Input dimensionality per subnet :math:`D_{in}`.
    out_features : int
        Output dimensionality per subnet :math:`D_{out}`.
    layer_size : int, optional, default=512
        Width of every hidden layer.
    nr_layers : int, optional, default=6
        Number of hidden layers.
    activation : torch.nn.Module, optional, default=None
        Hidden-layer activation module.  Defaults to :class:`torch.nn.SiLU`.

    Forward
    -------
    x : torch.Tensor
        Input of shape :math:`(\dots, S, D_{in})`.

    Outputs
    -------
    torch.Tensor
        Output of shape :math:`(\dots, S, D_{out})`.
    """

    def __init__(
        self,
        nr_subnetworks: int,
        in_features: int,
        out_features: int,
        layer_size: int = 512,
        nr_layers: int = 6,
        activation: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.nr_subnetworks = nr_subnetworks
        self.in_features = in_features
        self.out_features = out_features
        self.nr_layers = nr_layers
        self.activation = nn.SiLU() if activation is None else activation

        sizes: List[int] = [in_features] + [layer_size] * nr_layers + [out_features]
        self.layer_sizes = sizes

        # one stacked (S, out, in) weight tensor and (S, out) bias per layer
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        for l_in, l_out in zip(sizes[:-1], sizes[1:]):
            w = nn.Parameter(torch.empty(nr_subnetworks, l_out, l_in))
            b = nn.Parameter(torch.zeros(nr_subnetworks, l_out))
            nn.init.xavier_uniform_(w.view(nr_subnetworks * l_out, l_in))
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, x: Tensor) -> Tensor:
        r"""Evaluate all subnetworks.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape :math:`(\dots, S, D_{in})`.

        Returns
        -------
        torch.Tensor
            Output of shape :math:`(\dots, S, D_{out})`.
        """
        # flatten leading batch dims and move subnet axis to the front
        leading = x.shape[:-2]
        n = x[..., 0, 0].numel()  # number of collocation points
        s = self.nr_subnetworks
        h = x.reshape(n, s, -1).permute(1, 0, 2)  # (S, N, D_in)

        n_layers = len(self.weights)
        for idx, (w, b) in enumerate(zip(self.weights, self.biases)):
            # batched affine: bias + h @ wᵀ for every subnetwork at once
            bias_expanded = b.unsqueeze(1).expand(s, n, -1)
            h = torch.baddbmm(bias_expanded, h, w.transpose(-1, -2))
            if idx < n_layers - 1:
                h = self.activation(h)

        return h.permute(1, 0, 2).reshape(*leading, s, self.out_features)


class FiniteBasisNet(physicsnemo.Module):
    r"""Finite Basis PINN over overlapping subdomains.

    The input domain is decomposed into a regular grid of overlapping
    subdomains.  Each subdomain owns an MLP (all evaluated together via
    :class:`BatchedFullyConnected`).  Subnetwork outputs are weighted by a smooth
    window function and combined with a partition-of-unity normalisation:

    .. math::

        u(x) = \frac{\sum_i w_i(x) \, f_i(x)}{\sum_i w_i(x) + \varepsilon}

    where :math:`w_i` is the window of subdomain :math:`i` and :math:`f_i` its
    subnetwork.

    Parameters
    ----------
    in_features : int
        Number of input coordinate features :math:`D_{in}`.
    out_features : int
        Number of output features :math:`D_{out}`.
    domain_bounds : List[Tuple[float, float]], optional, default=None
        Per-dimension domain bounds ``[(min, max), ...]``.  Defaults to
        ``(0, 1)`` per dimension when ``None``.
    nr_domains : int or List[int], optional, default=8
        Number of subdomains per dimension.  A scalar broadcasts to all
        dimensions.
    overlap_ratio : float, optional, default=2.7
        Subdomain overlap ratio (typically between ``1.0`` and ``3.0``).
    window_fn : str, optional, default="cosine"
        Window function: one of ``"cosine"``, ``"sigmoid"``, ``"bump"``.
    layer_size : int, optional, default=512
        Hidden-layer width of each subnetwork.
    nr_layers : int, optional, default=6
        Number of hidden layers per subnetwork.
    activation_fn : str, optional, default="silu"
        Subnetwork hidden activation, resolved via
        :func:`~physicsnemo.nn.get_activation`.

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
    >>> from physicsnemo.experimental.models.pinn.fb_pinn import FiniteBasisNet
    >>> model = FiniteBasisNet(
    ...     in_features=2, out_features=1, nr_domains=3, layer_size=16, nr_layers=2
    ... )
    >>> x = torch.rand(32, 2)
    >>> model(x).shape
    torch.Size([32, 1])
    """

    _VALID_WINDOWS = ("cosine", "sigmoid", "bump")
    __model_checkpoint_version__ = "0.1.0"

    def __init__(
        self,
        in_features: int,
        out_features: int,
        domain_bounds: Optional[List[Tuple[float, float]]] = None,
        nr_domains: Union[int, List[int]] = 8,
        overlap_ratio: float = 2.7,
        window_fn: str = "cosine",
        layer_size: int = 512,
        nr_layers: int = 6,
        activation_fn: str = "silu",
    ) -> None:
        super().__init__(meta=ModelMetaData(func_torch=True, auto_grad=True))

        if window_fn not in self._VALID_WINDOWS:
            raise ValueError(
                f"window_fn must be one of {self._VALID_WINDOWS}, got '{window_fn}'"
            )

        self.in_features = in_features
        self.out_features = out_features
        self.nr_domains = nr_domains
        self.overlap_ratio = overlap_ratio
        self.window_fn = window_fn
        self.layer_size = layer_size
        self.nr_layers = nr_layers
        self.activation_fn = activation_fn

        # default to the unit hypercube
        if domain_bounds is None:
            self.domain_bounds = tuple((0.0, 1.0) for _ in range(in_features))
        else:
            if len(domain_bounds) != in_features:
                raise ValueError(
                    f"Expected {in_features} domain bounds, got {len(domain_bounds)}"
                )
            self.domain_bounds = tuple(
                tuple(float(v) for v in b) for b in domain_bounds
            )

        # subdomain grid shape per dimension
        if isinstance(nr_domains, int):
            self.subdomain_shape = torch.tensor([nr_domains] * in_features)
        else:
            if len(nr_domains) != in_features:
                raise ValueError("nr_domains length must match in_features")
            self.subdomain_shape = torch.tensor(nr_domains)
        self.total_subdomains = int(torch.prod(self.subdomain_shape).item())

        self.subnetworks = BatchedFullyConnected(
            nr_subnetworks=self.total_subdomains,
            in_features=in_features,
            out_features=out_features,
            layer_size=layer_size,
            nr_layers=nr_layers,
            activation=get_activation(activation_fn),
        )

        self._initialize_subdomain_params()

    def _initialize_subdomain_params(self) -> None:
        r"""Register subdomain center, half-width, and base half-width buffers.

        Returns
        -------
        None
        """
        subdomain_shape = self.subdomain_shape
        nr_dims = len(subdomain_shape)

        centers: List[Tensor] = []
        half_widths: List[Tensor] = []
        for dim_idx, nr_subs in enumerate(subdomain_shape):
            x_min, x_max = self.domain_bounds[dim_idx]
            domain_size = x_max - x_min
            nr_subs = int(nr_subs.item())
            if nr_subs == 1:
                c = torch.tensor([x_min + domain_size * 0.5])
                hw = torch.tensor([domain_size * self.overlap_ratio * 0.5])
            else:
                step = domain_size / nr_subs
                c = torch.linspace(x_min + 0.5 * step, x_max - 0.5 * step, nr_subs)
                hw = torch.full((nr_subs,), self.overlap_ratio * step * 0.5)
            centers.append(c)
            half_widths.append(hw)

        # tensor-product grid of subdomain centers / half-widths
        grid_c = torch.stack(torch.meshgrid(*centers, indexing="ij"), dim=0)
        grid_hw = torch.stack(torch.meshgrid(*half_widths, indexing="ij"), dim=0)
        centers_t = grid_c.reshape(nr_dims, -1).T.contiguous()
        half_widths_t = grid_hw.reshape(nr_dims, -1).T.contiguous()

        # base half-width excludes the overlap factor (subnet normalisation)
        base_hw_t = half_widths_t / self.overlap_ratio

        self.register_buffer("centers", centers_t)
        self.register_buffer("half_widths", half_widths_t)
        self.register_buffer("base_hw", base_hw_t)

    def _compute_windows(self, x_norm_window: Tensor) -> Tensor:
        r"""Compute the per-subdomain product window weights.

        Parameters
        ----------
        x_norm_window : torch.Tensor
            Window-normalised coordinates of shape
            :math:`(B, S, D_{in})`.

        Returns
        -------
        torch.Tensor
            Window weights of shape :math:`(B, S)`.
        """
        if self.window_fn == "cosine":
            normalized = x_norm_window.clamp(-1.0, 1.0)
            window_1d = ((1.0 + torch.cos(torch.pi * normalized)) * 0.5) ** 2
        elif self.window_fn == "sigmoid":
            sd = self.half_widths / 8.0
            c_lo = self.centers - self.half_widths
            c_hi = self.centers + self.half_widths
            x_exp = x_norm_window * self.half_widths + self.centers
            window_1d = torch.sigmoid((x_exp - c_lo) / sd) * torch.sigmoid(
                (c_hi - x_exp) / sd
            )
        else:  # bump
            r_sq = x_norm_window**2
            safe_r_sq = r_sq.clamp(max=1.0 - 1e-6)
            bump_vals = torch.exp(3.0 / (safe_r_sq - 1.0)) / 4.9787e-2
            window_1d = torch.where(r_sq < 1.0, bump_vals, torch.zeros_like(r_sq))

        # product of per-dimension windows
        return window_1d.prod(dim=-1)

    def forward(
        self, x: Float[Tensor, "batch in_features"]
    ) -> Float[Tensor, "batch out_features"]:
        r"""Forward pass through the FB-PINN.

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

        c = self.centers
        hw = self.half_widths
        bhw = self.base_hw

        # broadcast coordinates against all subdomains
        x_exp = x.unsqueeze(-2)  # (B, 1, D_in)
        x_norm_window = (x_exp - c) / hw  # (B, S, D_in)
        x_norm_subnet = (x_exp - c) / bhw  # (B, S, D_in)

        # window weights and subnetwork outputs
        windows = self._compute_windows(x_norm_window)  # (B, S)
        subnet_out = self.subnetworks(x_norm_subnet)  # (B, S, D_out)

        # partition-of-unity blend
        w_i = windows.unsqueeze(-1)  # (B, S, 1)
        output = (w_i * subnet_out).sum(dim=-2)  # (B, D_out)
        window_total = w_i.sum(dim=-2)  # (B, 1)
        return output / (window_total + 1e-8)
