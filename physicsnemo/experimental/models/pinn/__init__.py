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

r"""Custom physics-informed neural network architectures (experimental).

Tensor-in / tensor-out :class:`physicsnemo.Module` ports of seven custom PINN
architectures.  Each model takes a coordinate tensor of shape
:math:`(B, D_{in})` and returns a tensor of shape :math:`(B, D_{out})`.

Models
------
:class:`FiniteBasisNet`
    Finite-basis PINN over overlapping subdomains (FB-PINN).
:class:`KolmogorovArnoldNet`
    Kolmogorov-Arnold Network with B-spline edge activations.
:class:`FourierKolmogorovArnoldNet`
    Fourier-feature Kolmogorov-Arnold Network.
:class:`SplitTrunkNet`
    Output groups split by input dependency.
:class:`SeparableNet`
    Separable PINN using CP tensor decomposition (SPINN).
:class:`GLLKolmogorovArnoldNet`
    Gauss-Lobatto-Legendre Legendre-KAN with domain decomposition.
:class:`PirateNet`
    Physics-informed residual adaptive network with random Fourier features.
"""

from physicsnemo.experimental.models.pinn.fb_pinn import (
    BatchedFullyConnected,
    FiniteBasisNet,
)
from physicsnemo.experimental.models.pinn.fourier_kan import (
    FourierKolmogorovArnoldNet,
)
from physicsnemo.experimental.models.pinn.gll_kan import GLLKolmogorovArnoldNet
from physicsnemo.experimental.models.pinn.kan import (
    KANLayer,
    KolmogorovArnoldNet,
    KolmogorovArnoldNetCore,
)
from physicsnemo.experimental.models.pinn.piratenet import (
    PirateNet,
    PirateNetBlock,
    RWFLinear,
)
from physicsnemo.experimental.models.pinn.spinn import SeparableNet
from physicsnemo.experimental.models.pinn.split_trunk import SplitTrunkNet

__all__ = [
    "FiniteBasisNet",
    "BatchedFullyConnected",
    "KolmogorovArnoldNet",
    "KANLayer",
    "KolmogorovArnoldNetCore",
    "FourierKolmogorovArnoldNet",
    "SplitTrunkNet",
    "SeparableNet",
    "GLLKolmogorovArnoldNet",
    "PirateNet",
    "PirateNetBlock",
    "RWFLinear",
]
