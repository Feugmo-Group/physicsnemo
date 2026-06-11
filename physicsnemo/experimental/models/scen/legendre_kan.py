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

"""Kolmogorov-Arnold Network with Legendre polynomial basis (LegendreKAN).

Each edge function is a truncated Legendre series of degree K:

    φ_{rs}(x; c) = Σ_{k=0}^K  c_{rs,k} · P_k(x),   x ∈ [-1, 1].

Exact edge derivatives are computed via the modal recurrence:

    φ'_{rs}(x; c) = φ_{rs}(x; D^Leg · c)

where D^Leg ∈ R^{(K+1)×(K+1)} is the Legendre modal differentiation matrix.
For multi-layer networks, spatial derivatives at LGL nodes are obtained
using the precomputed physical matrices D1, D2 from DVRMapper — no autograd
traversal is required.

Public API
----------
legendre_basis(x, K)           — evaluate P_0 … P_K at arbitrary points
make_legendre_dleg(K, ...)     — build D^Leg
make_vandermonde(xi, K)        — V[i,k] = P_k(xi[i])
make_nodal_deriv_mats(xi, K)   — [D1_nod, D2_nod, ...] each shape (N, K+1)
LegendreKANLayer               — single layer R^{n_in} → R^{n_out}
LegendreKAN                    — full 1→1 multi-layer Kolmogorov-Arnold network
"""

from __future__ import annotations

import torch
import torch.nn as nn
from jaxtyping import Float

import physicsnemo
from physicsnemo.core.meta import ModelMetaData
from physicsnemo.nn.functional import legendre_polynomials

# ---------------------------------------------------------------------------
# Basis and matrix utilities
# ---------------------------------------------------------------------------


def legendre_basis(x: torch.Tensor, K: int) -> torch.Tensor:
    """Evaluate Legendre polynomials P_0, P_1, …, P_K at each point in x.

    Thin wrapper over the shared
    :func:`physicsnemo.nn.functional.legendre_polynomials` (3-term recurrence,
    no in-place ops, so the graph is preserved for vmap + functional_call).

    Parameters
    ----------
    x : torch.Tensor
        Shape ``(*batch)``.  Values ideally in [-1, 1].
    K : int
        Maximum polynomial degree.

    Returns
    -------
    P : torch.Tensor
        Shape ``(*batch, K+1)``.  ``P[..., k] = P_k(x)``.
    """
    return torch.stack(legendre_polynomials(x, K + 1), dim=-1)


def make_legendre_dleg(
    K: int,
    dtype: torch.dtype = torch.float32,
    device=None,
) -> torch.Tensor:
    """Build the Legendre modal differentiation matrix D^Leg ∈ R^{(K+1)×(K+1)}.

    ``D^Leg_{i,j} = 2i+1`` when ``j > i`` and ``j-i`` is odd, else 0.

    Applying D^Leg to a coefficient vector c gives the coefficients of φ':
    ``(D^Leg · c)_k = Σ_{j>k, j-k odd} (2k+1) c_j``.

    Parameters
    ----------
    K : int
        Polynomial degree.
    dtype : torch.dtype
        Output tensor dtype.
    device : optional
        Output tensor device.

    Returns
    -------
    D : torch.Tensor
        Shape ``(K+1, K+1)``.
    """
    if device is None:
        device = torch.device("cpu")
    D = torch.zeros(K + 1, K + 1, dtype=dtype, device=device)
    for i in range(K):
        for j in range(i + 1, K + 1):
            if (j - i) % 2 == 1:
                D[i, j] = 2 * i + 1
    return D


def make_vandermonde(xi: torch.Tensor, K: int) -> torch.Tensor:
    """Build Vandermonde–Legendre matrix V ∈ R^{N×(K+1)}.

    ``V[i, k] = P_k(xi[i])``.

    Parameters
    ----------
    xi : torch.Tensor
        Shape ``(N,)``.  Quadrature nodes (LGL reference coords in [-1, 1]).
    K : int
        Maximum polynomial degree.

    Returns
    -------
    V : torch.Tensor
        Shape ``(N, K+1)``.
    """
    return legendre_basis(xi, K)


def make_nodal_deriv_mats(
    xi: torch.Tensor,
    K: int,
    max_order: int = 4,
) -> list[torch.Tensor]:
    """Build nodal derivative matrices D^(n)_nod = V · (D^Leg)^n.

    Applying ``D^(n)_nod`` to coefficient vector c gives the n-th derivative
    of the Legendre expansion at all quadrature nodes — exact, no autograd.

    Parameters
    ----------
    xi : torch.Tensor
        Shape ``(N,)``.  LGL reference nodes.
    K : int
        Polynomial degree.
    max_order : int
        Highest derivative order to build.  Default 4.

    Returns
    -------
    list[torch.Tensor]
        ``[D1_nod, D2_nod, …, Dmax_nod]``, each shape ``(N, K+1)``.
    """
    dtype, device = xi.dtype, xi.device
    V = make_vandermonde(xi, K)
    DL = make_legendre_dleg(K, dtype=dtype, device=device)
    mats = []
    DLn = DL.clone()
    for _ in range(max_order):
        mats.append(V @ DLn)
        DLn = DLn @ DL
    return mats


# ---------------------------------------------------------------------------
# Network modules
# ---------------------------------------------------------------------------


class LegendreKANLayer(physicsnemo.Module):
    r"""Single Legendre-KAN layer: :math:`\mathbb{R}^{n_{in}} \to \mathbb{R}^{n_{out}}`.

    Each of the :math:`n_{in} \times n_{out}` directed edges is a degree-K
    Legendre polynomial series:

    .. math::

        \varphi_{rs}(x) = \sum_{k=0}^{K} c_{rs,k} \cdot P_k(x)

    The output at node :math:`s` is the sum over all input nodes :math:`r`:

    .. math::

        \text{out}[s] = \sum_r \varphi_{rs}(\text{input}[r])

    Parameters
    ----------
    n_in : int
        Input dimension.
    n_out : int
        Output dimension.
    poly_degree : int
        Legendre polynomial degree per edge :math:`K`.
    dtype : torch.dtype
        Parameter dtype.  Default ``torch.float32``.
    device : optional
        Parameter device.  Default ``cpu``.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(N, n_{in})`.  Values ideally in
        :math:`[-1, 1]` for well-conditioned basis evaluations.

    Outputs
    -------
    out : torch.Tensor
        Output tensor of shape :math:`(N, n_{out})`.

    Examples
    --------
    >>> layer = LegendreKANLayer(n_in=4, n_out=8, poly_degree=3)
    >>> x = torch.randn(16, 4).clamp(-1, 1)
    >>> layer(x).shape
    torch.Size([16, 8])
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        poly_degree: int,
        dtype: torch.dtype = torch.float32,
        device=None,
    ):
        super().__init__(meta=ModelMetaData(func_torch=True))
        if device is None:
            device = torch.device("cpu")
        self.n_in = n_in
        self.n_out = n_out
        self.poly_degree = poly_degree
        K = poly_degree
        self.coeff = nn.Parameter(
            torch.zeros(n_in, n_out, K + 1, dtype=dtype, device=device)
        )
        nn.init.normal_(self.coeff, std=0.1 / (n_in**0.5))

    def forward(self, x: Float[torch.Tensor, "N n_in"]) -> Float[torch.Tensor, "N n_out"]:
        r"""Evaluate the Legendre-KAN layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape :math:`(N, n_{in})`.

        Returns
        -------
        out : torch.Tensor
            Output tensor of shape :math:`(N, n_{out})`.
        """
        if not torch.compiler.is_compiling():
            if x.ndim != 2 or x.shape[1] != self.n_in:
                raise ValueError(
                    f"Expected input shape (N, {self.n_in}), got {tuple(x.shape)}"
                )
        basis = legendre_basis(x, self.poly_degree)  # (N, n_in, K+1)
        return torch.einsum("nrk,rsk->ns", basis, self.coeff)  # (N, n_out)


class LegendreKAN(physicsnemo.Module):
    r"""Multi-layer Kolmogorov-Arnold Network with Legendre polynomial basis.

    Architecture: :math:`1 \to h \to \cdots \to h \to 1` with
    ``n_layers`` KAN layers total and ``n_layers - 1`` hidden layers.

    Input: reference LGL coordinates :math:`\xi \in [-1, 1]`, shape
    :math:`(N, 1)`.  Output: :math:`u(\xi)`, shape :math:`(N, 1)`.

    ``tanh`` is applied between hidden layers to keep intermediate values in
    :math:`[-1, 1]` so that Legendre basis evaluations remain well-conditioned.
    The final layer has no activation.

    Parameters
    ----------
    hidden_dim : int
        Width of each hidden KAN layer.  Default ``16``.
    n_layers : int
        Total number of KAN layers (depth).  Default ``2``.
    poly_degree : int
        Legendre polynomial degree :math:`K` per edge.  Default ``4``.
    dtype : torch.dtype
        Parameter dtype.  Default ``torch.float32``.
    device : optional
        Parameter device.  Default ``cpu``.

    Forward
    -------
    x : torch.Tensor
        Reference coordinates :math:`\xi`, shape :math:`(N, 1)`,
        with values in :math:`[-1, 1]`.

    Outputs
    -------
    u : torch.Tensor
        Network output, shape :math:`(N, 1)`.

    Examples
    --------
    >>> net = LegendreKAN(hidden_dim=16, n_layers=3, poly_degree=4)
    >>> xi = torch.linspace(-1, 1, 32).unsqueeze(1)
    >>> net(xi).shape
    torch.Size([32, 1])
    """

    def __init__(
        self,
        hidden_dim: int = 16,
        n_layers: int = 2,
        poly_degree: int = 4,
        dtype: str = "float32",
        device: str = "cpu",
    ):
        super().__init__(meta=ModelMetaData(func_torch=True))
        torch_dtype = getattr(torch, dtype)
        torch_device = torch.device(device)
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.poly_degree = poly_degree
        self.dtype = dtype
        # Note: don't store self.device — nn.Module already owns that property

        dims = [1] + [hidden_dim] * (n_layers - 1) + [1]
        self.layers = nn.ModuleList(
            [
                LegendreKANLayer(dims[i], dims[i + 1], poly_degree, torch_dtype, torch_device)
                for i in range(len(dims) - 1)
            ]
        )

    def forward(self, x: Float[torch.Tensor, "N 1"]) -> Float[torch.Tensor, "N 1"]:
        r"""Evaluate the multi-layer Legendre-KAN.

        Parameters
        ----------
        x : torch.Tensor
            Reference coordinates :math:`\xi`, shape :math:`(N, 1)`.

        Returns
        -------
        u : torch.Tensor
            Network output, shape :math:`(N, 1)`.
        """
        if not torch.compiler.is_compiling():
            if x.ndim != 2 or x.shape[1] != 1:
                raise ValueError(
                    f"Expected input shape (N, 1), got {tuple(x.shape)}"
                )
        h = x
        for layer in self.layers[:-1]:
            h = torch.tanh(layer(h))
        return self.layers[-1](h)
