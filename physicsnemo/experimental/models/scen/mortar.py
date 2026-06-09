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

"""Mortar projection for non-conforming spectral element interfaces.

Used when adjacent elements have different node counts (mixed-N grids).
"""

from __future__ import annotations

import numpy as np
import torch

from physicsnemo.experimental.models.scen.quadrature import get_quadrature_data


def compute_mortar_projection(
    N_high: int,
    N_low: int,
    rule: str = "lgl",
    dtype: torch.dtype = torch.float32,
    device=None,
) -> torch.Tensor:
    """Lagrange interpolation projection from N_high nodes to N_low nodes.

    Returns ``P`` of shape ``(N_low, N_high)``.  Multiplying a function
    sampled on the high-resolution side by ``P`` gives its values on the
    low-resolution side.

    Parameters
    ----------
    N_high : int
        Number of quadrature nodes on the high-resolution element side.
    N_low : int
        Number of quadrature nodes on the low-resolution element side.
    rule : str
        Quadrature rule for both sides.  Default ``'lgl'``.
    dtype : torch.dtype
        Output tensor dtype.
    device : torch.device or None
        Output tensor device.

    Returns
    -------
    P : torch.Tensor
        Shape ``(N_low, N_high)``.
    """
    if device is None:
        device = torch.device("cpu")

    xi_high, _, _ = get_quadrature_data(
        N_high, rule=rule, dtype=torch.float64, device=torch.device("cpu")
    )
    xi_low, _, _ = get_quadrature_data(
        N_low, rule=rule, dtype=torch.float64, device=torch.device("cpu")
    )

    xi_h = xi_high.numpy()
    xi_l = xi_low.numpy()

    P = np.zeros((N_low, N_high))
    for i in range(N_low):
        for j in range(N_high):
            num = 1.0
            for k in range(N_high):
                if k != j:
                    num *= (xi_l[i] - xi_h[k]) / (xi_h[j] - xi_h[k])
            P[i, j] = num

    return torch.tensor(P, dtype=dtype, device=device)
