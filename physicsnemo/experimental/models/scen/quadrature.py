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

"""Quadrature rules for spectral collocation element networks (SCEN).

All three rules use N nodes on [-1, 1] including both endpoints, and return
(nodes, weights, D) in ascending order (nodes[0] ≈ -1, nodes[-1] ≈ +1).

Rules
-----
lgl              : Legendre-Gauss-Lobatto nodes, LGL weights (exact for degree ≤ 2N-3),
                   spectral D matrix via P_{N-1} ratio formula.
chebyshev        : Chebyshev-Lobatto nodes cos(kπ/(N-1)), Chebyshev-Lobatto weights
                   (for weight function 1/sqrt(1-x²), rescaled to sum to 2),
                   spectral D matrix via c_i/c_j ratio formula.
clenshaw_curtis  : Same nodes and D as 'chebyshev', Clenshaw-Curtis weights
                   (uniform weight function, exact for degree ≤ N-1, sum to 2).
"""

from __future__ import annotations

import numpy as np
import torch

_VALID_RULES = ("lgl", "chebyshev", "clenshaw_curtis")


def _get_lgl(N: int):
    """Return (nodes, weights, D) for N-point LGL quadrature (float64 numpy)."""
    xi = np.cos(np.pi * np.arange(N) / (N - 1)).astype(np.float64)
    P = np.zeros((N, N), dtype=np.float64)
    for _ in range(100):
        P[:, 0] = 1.0
        P[:, 1] = xi
        for k in range(2, N):
            P[:, k] = ((2 * k - 1) * xi * P[:, k - 1] - (k - 1) * P[:, k - 2]) / k
        xi_new = xi - (xi * P[:, N - 1] - P[:, N - 2]) / (N * P[:, N - 1])
        if np.max(np.abs(xi_new - xi)) < 1e-15:
            xi = xi_new
            break
        xi = xi_new
    P[:, 0] = 1.0
    P[:, 1] = xi
    for k in range(2, N):
        P[:, k] = ((2 * k - 1) * xi * P[:, k - 1] - (k - 1) * P[:, k - 2]) / k
    idx = np.argsort(xi)
    xi = xi[idx]
    P = P[idx]
    w = 2.0 / (N * (N - 1) * P[:, N - 1] ** 2)
    D = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            if i != j:
                D[i, j] = (P[i, N - 1] / P[j, N - 1]) / (xi[i] - xi[j])
    for i in range(N):
        D[i, i] = -np.sum(D[i, :])
    return xi, w, D


def _chebyshev_dmat(N: int):
    """Chebyshev spectral D matrix in ascending node order."""
    k = np.arange(N, dtype=np.float64)
    x = np.cos(np.pi * k / (N - 1))
    c = np.ones(N)
    c[0] = 2.0
    c[-1] = 2.0
    D = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            if i != j:
                D[i, j] = (c[i] / c[j]) * (-1.0) ** (i + j) / (x[i] - x[j])
    for i in range(N):
        D[i, i] = -np.sum(D[i, :])
    return D[::-1, ::-1].copy()


def _get_chebyshev(N: int):
    """Chebyshev-Lobatto nodes + weights rescaled to sum to 2 + spectral D."""
    k = np.arange(N, dtype=np.float64)
    xi = np.cos(np.pi * (N - 1 - k) / (N - 1))
    w = np.full(N, np.pi / (N - 1))
    w[0] = np.pi / (2.0 * (N - 1))
    w[-1] = np.pi / (2.0 * (N - 1))
    w *= 2.0 / w.sum()
    return xi, w, _chebyshev_dmat(N)


def _get_clenshaw_curtis(N: int):
    """Clenshaw-Curtis quadrature: same nodes/D as Chebyshev, CC weights."""
    k = np.arange(N, dtype=np.float64)
    xi = np.cos(np.pi * (N - 1 - k) / (N - 1))
    n = N - 1
    if N == 1:
        w = np.array([2.0])
    else:
        n_half = n // 2
        g = np.zeros(n_half + 1)
        g[0] = 1.0
        for j in range(1, n_half):
            g[j] = 2.0 / (1.0 - (2 * j) ** 2)
        if n % 2 == 0:
            g[n_half] = 1.0 / (1.0 - n ** 2)
        else:
            g[n_half] = 2.0 / (1.0 - (2 * n_half) ** 2)
        ki = np.arange(N, dtype=np.float64)
        ji = np.arange(n_half + 1, dtype=np.float64)
        angles = np.outer(ki, ji) * (2.0 * np.pi / n)
        w_desc = (2.0 / n) * (np.cos(angles) @ g)
        w_desc[0] /= 2.0
        w_desc[-1] /= 2.0
        w = w_desc[::-1].copy()
    return xi, w, _chebyshev_dmat(N)


def get_quadrature_data(
    N: int,
    rule: str = "lgl",
    dtype: torch.dtype = torch.float32,
    device: torch.device = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (nodes, weights, D) for the requested quadrature rule.

    Parameters
    ----------
    N : int
        Number of nodes (includes both endpoints).
    rule : str
        One of ``'lgl'``, ``'chebyshev'``, ``'clenshaw_curtis'``.
    dtype : torch.dtype
        Output tensor dtype.  Default ``torch.float32``.
    device : torch.device, optional
        Output tensor device.  Default ``cpu``.

    Returns
    -------
    nodes : torch.Tensor
        Shape ``(N,)``, ascending, nodes[0] ≈ -1, nodes[-1] ≈ +1.
    weights : torch.Tensor
        Shape ``(N,)``, sum ≈ 2.
    D : torch.Tensor
        Shape ``(N, N)``, spectral differentiation matrix, ``D @ f ≈ df/dξ``.
    """
    if device is None:
        device = torch.device("cpu")
    rule = rule.lower()
    if rule not in _VALID_RULES:
        raise ValueError(f"rule must be one of {_VALID_RULES}, got '{rule}'")
    if rule == "lgl":
        xi, w, D = _get_lgl(N)
    elif rule == "chebyshev":
        xi, w, D = _get_chebyshev(N)
    else:
        xi, w, D = _get_clenshaw_curtis(N)
    return (
        torch.tensor(xi, dtype=dtype, device=device),
        torch.tensor(w, dtype=dtype, device=device),
        torch.tensor(D, dtype=dtype, device=device),
    )
