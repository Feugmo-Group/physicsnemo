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

r"""Gauss-Lobatto-Legendre KAN with non-overlapping domain decomposition.

A Legendre-KAN solver discretised on a regular grid of non-overlapping
subdomains.  Each subdomain owns a multi-input Legendre-KAN core; spatial
derivatives are available through the Gauss-Lobatto-Legendre (GLL) pseudospectral
differentiation matrix, and inter-element continuity can be enforced with a
mortar projection.

Reuse from :mod:`physicsnemo.experimental.models.scen`
------------------------------------------------------
- :func:`~physicsnemo.experimental.models.scen.quadrature.get_quadrature_data`
  supplies the GLL (``'lgl'``) nodes, weights, and spectral differentiation
  matrix (replacing the legacy bespoke ``compute_gll_nodes_weights`` /
  ``compute_diff_matrix``).
- :func:`~physicsnemo.experimental.models.scen.legendre_kan.legendre_basis`
  evaluates the Legendre polynomial basis used by each KAN edge.
- :func:`~physicsnemo.experimental.models.scen.mortar.compute_mortar_projection`
  builds the Lagrange interpolation operator used by
  :meth:`GLLKolmogorovArnoldNet.mortar_projection`.

.. note::

    The legacy ``Arch`` produced an augmented output dictionary bundling primary
    fields, requested derivatives, and per-interface mortar residuals into a
    single stacked tensor for the constraint solver.  This keyless port returns a
    clean ``(B, out_features)`` solution tensor.  The spectral derivative and
    mortar machinery are exposed through helper methods
    (:meth:`differentiation_matrix`, :meth:`mortar_projection`) so downstream
    physics-loss code can assemble those constraints explicitly.
"""

from __future__ import annotations

from itertools import product
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

import physicsnemo
from physicsnemo.core.meta import ModelMetaData
from physicsnemo.experimental.models.scen.legendre_kan import legendre_basis
from physicsnemo.experimental.models.scen.mortar import compute_mortar_projection
from physicsnemo.experimental.models.scen.quadrature import get_quadrature_data


class _LegendreKANLayerND(nn.Module):
    r"""Multi-input Legendre-KAN layer.

    Each directed edge between an input feature and an output feature is a
    degree-``max_order`` Legendre series; the output is the sum over input edges
    plus a residual linear branch.

    Parameters
    ----------
    in_features : int
        Input dimension :math:`n_{in}`.
    out_features : int
        Output dimension :math:`n_{out}`.
    max_order : int, optional, default=4
        Maximum Legendre polynomial degree.

    Forward
    -------
    x : torch.Tensor
        Input of shape :math:`(\dots, n_{in})` with values in :math:`[-1, 1]`.

    Outputs
    -------
    torch.Tensor
        Output of shape :math:`(\dots, n_{out})`.
    """

    def __init__(self, in_features: int, out_features: int, max_order: int = 4) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_order = max_order

        self.w = nn.Parameter(torch.empty(out_features, in_features))
        self.c = nn.Parameter(torch.empty(out_features, in_features, max_order + 1))
        nn.init.kaiming_uniform_(self.w, nonlinearity="linear")
        nn.init.kaiming_uniform_(self.c.view(out_features, -1), nonlinearity="linear")

    def forward(self, x: Tensor) -> Tensor:
        r"""Apply the multi-input Legendre-KAN layer.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape :math:`(\dots, n_{in})`.

        Returns
        -------
        torch.Tensor
            Output of shape :math:`(\dots, n_{out})`.
        """
        # Legendre basis per input feature (reused from scen)
        basis = legendre_basis(x, self.max_order)  # (..., n_in, K+1)
        out = torch.einsum("...id,oid->...o", basis, self.c)
        return out + torch.nn.functional.linear(x, self.w)


class _LegendreKANCoreND(nn.Module):
    r"""Multi-layer, multi-input Legendre-KAN core with input normalisation.

    Parameters
    ----------
    in_features : int
        Input dimension :math:`D_{in}`.
    out_features : int
        Output dimension :math:`D_{out}`.
    layer_size : int, optional, default=16
        Hidden-layer width.
    nr_layers : int, optional, default=2
        Number of hidden layers.
    max_order : int, optional, default=4
        Maximum Legendre polynomial degree.
    domain_bounds : List[Tuple[float, float]], optional, default=None
        Per-dimension bounds used to normalise inputs into
        :math:`[-1, 1]^{D_{in}}`.  Defaults to ``(-1, 1)`` per dimension.

    Forward
    -------
    x : torch.Tensor
        Input of shape :math:`(B, D_{in})`.

    Outputs
    -------
    torch.Tensor
        Output of shape :math:`(B, D_{out})`.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        layer_size: int = 16,
        nr_layers: int = 2,
        max_order: int = 4,
        domain_bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> None:
        super().__init__()
        self.max_order = max_order
        self.nr_layers = nr_layers

        sizes = [in_features] + [layer_size] * nr_layers + [out_features]
        self.layers = nn.ModuleList(
            _LegendreKANLayerND(sizes[i], sizes[i + 1], max_order)
            for i in range(len(sizes) - 1)
        )

        if domain_bounds is None:
            domain_bounds = [(-1.0, 1.0)] * in_features
        if len(domain_bounds) != in_features:
            raise ValueError("domain_bounds length must equal in_features")

        # affine normalisation [a, b] -> [-1, 1] per dimension
        scales = torch.tensor([2.0 / (b - a) for a, b in domain_bounds])
        shifts = torch.tensor([-(a + b) / (b - a) for a, b in domain_bounds])
        self.register_buffer("_norm_scale", scales)
        self.register_buffer("_norm_shift", shifts)

    def forward(self, x: Tensor) -> Tensor:
        r"""Apply the Legendre-KAN core.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape :math:`(B, D_{in})`.

        Returns
        -------
        torch.Tensor
            Output of shape :math:`(B, D_{out})`.
        """
        x = x * self._norm_scale + self._norm_shift
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # keep intermediate values in [-1, 1] for well-conditioned basis
            if i < len(self.layers) - 1:
                x = torch.tanh(x)
        return x


class GLLKolmogorovArnoldNet(physicsnemo.Module):
    r"""GLL Legendre-KAN with non-overlapping domain decomposition.

    The domain is partitioned into a regular grid of non-overlapping subdomains,
    each served by an independent multi-input Legendre-KAN core normalised to its
    own bounds.  At evaluation time every input point is routed to the subdomain
    that contains it; the matching subnetwork produces the output.

    Gauss-Lobatto-Legendre nodes, weights, and the spectral differentiation
    matrix (reused from :mod:`physicsnemo.experimental.models.scen`) are stored as
    buffers and exposed via :meth:`differentiation_matrix`.  A mortar projection
    operator for non-conforming interfaces is available through
    :meth:`mortar_projection`.

    Parameters
    ----------
    in_features : int
        Number of input coordinate features :math:`D_{in}`.
    out_features : int
        Number of output features :math:`D_{out}`.
    nr_domains : int or List[int], optional, default=4
        Number of non-overlapping subdomains per dimension.  A scalar broadcasts
        to all dimensions.
    layer_size : int, optional, default=16
        Hidden-layer width of each subnetwork.
    nr_layers : int, optional, default=2
        Number of hidden layers per subnetwork.
    max_order : int, optional, default=4
        Maximum Legendre polynomial degree.
    domain_bounds : List[Tuple[float, float]], optional, default=None
        Global per-dimension domain bounds.  Defaults to ``(-1, 1)`` per
        dimension when ``None``.

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
    >>> from physicsnemo.experimental.models.pinn.gll_kan import (
    ...     GLLKolmogorovArnoldNet,
    ... )
    >>> model = GLLKolmogorovArnoldNet(
    ...     in_features=1, out_features=1, nr_domains=2, domain_bounds=[(0.0, 1.0)]
    ... )
    >>> x = torch.rand(16, 1)
    >>> model(x).shape
    torch.Size([16, 1])
    """

    __model_checkpoint_version__ = "0.1.0"

    def __init__(
        self,
        in_features: int,
        out_features: int,
        nr_domains: Union[int, List[int]] = 4,
        layer_size: int = 16,
        nr_layers: int = 2,
        max_order: int = 4,
        domain_bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> None:
        super().__init__(meta=ModelMetaData(func_torch=True, auto_grad=True))

        self.in_features = in_features
        self.out_features = out_features
        self.layer_size = layer_size
        self.nr_layers = nr_layers
        self.max_order = max_order

        # subdomain counts per dimension
        if isinstance(nr_domains, int):
            nr_domains_list = [nr_domains] * in_features
        else:
            if len(nr_domains) != in_features:
                raise ValueError("nr_domains length must equal in_features")
            nr_domains_list = list(nr_domains)
        self.nr_domains_list = nr_domains_list
        self.total_subdomains = int(np.prod(nr_domains_list))

        # global domain bounds
        if domain_bounds is None:
            domain_bounds = [(-1.0, 1.0)] * in_features
        if len(domain_bounds) != in_features:
            raise ValueError("domain_bounds length must equal in_features")
        self.global_bounds = [(float(a), float(b)) for a, b in domain_bounds]

        # GLL quadrature data (reused from scen); poly_degree sized as in legacy
        poly_degree = max_order ** (nr_layers + 1)
        gll_n = int(2 * np.ceil((poly_degree + 1) / 2)) + 1
        self.gll_n = gll_n
        nodes, weights, d_mat = get_quadrature_data(gll_n, rule="lgl")
        self.register_buffer("gll_nodes", nodes)
        self.register_buffer("gll_weights", weights)
        self.register_buffer("gll_D", d_mat)

        # subdomain multi-indices and physical bounds
        self._subdomain_indices: List[Tuple[int, ...]] = list(
            product(*[range(nd) for nd in nr_domains_list])
        )
        sub_bounds = [self._subdomain_bounds(mi) for mi in self._subdomain_indices]

        # one Legendre-KAN core per subdomain, normalised to its own bounds
        self.subnetworks = nn.ModuleList(
            _LegendreKANCoreND(
                in_features,
                out_features,
                layer_size,
                nr_layers,
                max_order,
                domain_bounds=b,
            )
            for b in sub_bounds
        )

        # store subdomain lower/upper corners for fast point routing
        lowers = torch.tensor([[b[0] for b in bb] for bb in sub_bounds])
        uppers = torch.tensor([[b[1] for b in bb] for bb in sub_bounds])
        self.register_buffer("_sub_lower", lowers)  # (S, D_in)
        self.register_buffer("_sub_upper", uppers)  # (S, D_in)

    def _subdomain_bounds(self, mi: Tuple[int, ...]) -> List[Tuple[float, float]]:
        r"""Compute physical bounds for a subdomain multi-index.

        Parameters
        ----------
        mi : Tuple[int, ...]
            Per-dimension subdomain indices.

        Returns
        -------
        List[Tuple[float, float]]
            Per-dimension ``(min, max)`` bounds of the subdomain.
        """
        bounds = []
        for n, idx in enumerate(mi):
            a_g, b_g = self.global_bounds[n]
            width = (b_g - a_g) / self.nr_domains_list[n]
            a_s = a_g + idx * width
            bounds.append((a_s, a_s + width))
        return bounds

    def differentiation_matrix(self) -> Tensor:
        r"""Return the GLL spectral differentiation matrix.

        Returns
        -------
        torch.Tensor
            Differentiation matrix of shape :math:`(N, N)` with
            :math:`N = ` ``gll_n``, satisfying :math:`D f \approx df/d\xi` on the
            reference element :math:`[-1, 1]`.
        """
        return self.gll_D

    def mortar_projection(self, n_low: int) -> Tensor:
        r"""Return the mortar projection from this model's GLL grid to ``n_low``.

        Parameters
        ----------
        n_low : int
            Number of nodes on the coarse (low-resolution) interface side.

        Returns
        -------
        torch.Tensor
            Projection matrix of shape :math:`(n_{low}, N)` with
            :math:`N = ` ``gll_n`` (reused from
            :func:`~physicsnemo.experimental.models.scen.mortar.compute_mortar_projection`).
        """
        return compute_mortar_projection(
            self.gll_n, n_low, rule="lgl", dtype=self.gll_nodes.dtype
        )

    def forward(
        self, x: Float[Tensor, "batch in_features"]
    ) -> Float[Tensor, "batch out_features"]:
        r"""Forward pass through the GLL Legendre-KAN.

        Each input point is routed to the subdomain that contains it (the last
        matching subdomain wins on shared faces).  Points outside the global
        domain fall back to the nearest-by-index subdomain.

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
            if x.ndim != 2 or x.shape[-1] != self.in_features:
                raise ValueError(
                    f"Expected input of shape (B, {self.in_features}), "
                    f"got tensor of shape {tuple(x.shape)}"
                )

        out = x.new_zeros(x.shape[0], self.out_features)
        # default routing to subdomain 0, then override per containing subdomain
        assigned = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
        for s in range(self.total_subdomains):
            lower = self._sub_lower[s]
            upper = self._sub_upper[s]
            # closed-lower / closed-upper containment test
            inside = ((x >= lower) & (x <= upper)).all(dim=-1)
            if inside.any():
                out[inside] = self.subnetworks[s](x[inside])
                assigned = assigned | inside

        # any unassigned (out-of-domain) points go through subdomain 0
        if not bool(assigned.all()):
            missing = ~assigned
            out[missing] = self.subnetworks[0](x[missing])
        return out
