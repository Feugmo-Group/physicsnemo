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

"""Film-thickness metric: predicted L(t) vs COMSOL reference.

The network predicts the dimensionless film thickness ``l`` on the space-time
grid.  ``l`` is uniform in ``x`` at fixed ``y`` (the film thickness is a
function of time only), so we evaluate at any ``x`` and de-nondimensionalize
``L = l * lc`` and ``t = y * tc`` before comparing with ``data/const_0.1_V.csv``
(columns ``T`` [s], ``L`` [m]).
"""

import os

import numpy as np
import torch


def film_thickness_error(
    net,
    pde,
    p,
    device,
    csv_path: str | None = None,
    n_t: int = 256,
):
    """Compute L-inf and relative L2 error of predicted L(t) vs COMSOL.

    Returns a dict with ``linf`` (metres), ``rel_l2`` and the sampled curves.
    """
    if csv_path is None:
        csv_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "const_0.1_V.csv"
        )
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    t_comsol = np.atleast_1d(data["T"])  # [s]
    L_comsol = np.atleast_1d(data["L"])  # [m]

    # dimensionless COMSOL time, clipped to the trained interval [0, yf]
    y_comsol = t_comsol / p.tc

    # evaluate predicted l at x=0.5 across the dimensionless times
    y = torch.as_tensor(y_comsol, dtype=torch.get_default_dtype(), device=device)
    y = y.reshape(-1, 1)
    x = torch.full_like(y, 0.5)
    coords = torch.cat([x, y], dim=1)

    from src.hard_bc import enforce_hard_bc

    with torch.no_grad():
        raw = net(coords)
        starred = {
            "cCV_star": raw[:, 0:1],
            "cAV_star": raw[:, 1:2],
            "phif_star": raw[:, 2:3],
            "l_star": raw[:, 3:4],
        }
        fields = enforce_hard_bc(x, y, starred, pde.phif_initial_fn, pde.lini)
        l_pred = fields["l"].cpu().numpy().reshape(-1)

    L_pred = l_pred * p.lc  # de-nondimensionalize to metres

    linf = float(np.max(np.abs(L_pred - L_comsol)))
    denom = float(np.linalg.norm(L_comsol)) or 1.0
    rel_l2 = float(np.linalg.norm(L_pred - L_comsol) / denom)

    return {
        "linf": linf,
        "rel_l2": rel_l2,
        "t_comsol": t_comsol,
        "L_comsol": L_comsol,
        "L_pred": L_pred,
    }
