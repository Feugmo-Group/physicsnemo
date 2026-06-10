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

r"""Fourier-feature Kolmogorov-Arnold Network.

Applies a :class:`~physicsnemo.nn.FourierLayer` feature embedding to the raw
coordinates, concatenates the raw inputs with the Fourier features, and feeds
the result into a KAN core (reused from
:mod:`physicsnemo.experimental.models.pinn.kan`).

The legacy ``Arch`` implementation split inputs into ``(x, y, z, t)`` and
"parameter" key groups via the named-key machinery, applying separate Fourier
layers to each.  Since this port operates on a single coordinate tensor with no
key names, a single Fourier embedding is applied to the full input.  This is the
main behavioural simplification relative to the original (see module-level note).
"""

from __future__ import annotations

from typing import Tuple

import torch
from jaxtyping import Float
from torch import Tensor

import physicsnemo
from physicsnemo.core.meta import ModelMetaData
from physicsnemo.experimental.models.pinn.kan import KolmogorovArnoldNetCore
from physicsnemo.nn import FourierLayer, get_activation


class FourierKolmogorovArnoldNet(physicsnemo.Module):
    r"""Fourier-feature Kolmogorov-Arnold Network.

    The input coordinates are mapped through a Fourier-feature embedding,
    concatenated with the raw coordinates, and passed to a KAN core.  The
    Fourier embedding helps the network represent high-frequency solutions.

    .. note::

        The legacy implementation used named keys to apply distinct Fourier
        embeddings to spatio-temporal versus parameter inputs.  This port applies
        a single Fourier embedding to the full coordinate tensor.

    .. code-block:: python

        frequencies = ("gaussian", std, nr_freq)   # random Gaussian features
        frequencies = ("axis", [0, 1, 2, ...])      # per-axis integer frequencies
        frequencies = ("diagonal", [...])           # diagonal frequencies
        frequencies = ("full", [...])               # full tensor-product grid

    where the first element selects the frequency-construction mode and the
    second (and optional third) specify the frequencies.

    Parameters
    ----------
    in_features : int
        Number of input coordinate features :math:`D_{in}`.
    out_features : int
        Number of output features :math:`D_{out}`.
    frequencies : Tuple, optional, default=("axis", (0, 1, ..., 9))
        Fourier frequency specification.  See the code block above.
    layer_size : int, optional, default=64
        Hidden width of the KAN core.
    nr_layers : int, optional, default=3
        Number of hidden KAN layers.
    grid_size : int, optional, default=10
        Number of B-spline grid intervals per KAN layer.
    spline_order : int, optional, default=3
        Order of the B-splines.
    base_activation_fn : str, optional, default="tanh"
        Residual base activation inside each KAN layer.
    grid_range : Tuple[float, float], optional, default=(-1.0, 1.0)
        Knot grid range.
    free_knot : bool, optional, default=False
        Use learnable knot placement.
    outer_layer_fn : str, optional, default="tanh"
        Inter-layer non-linearity in the KAN core.
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
    >>> from physicsnemo.experimental.models.pinn.fourier_kan import (
    ...     FourierKolmogorovArnoldNet,
    ... )
    >>> model = FourierKolmogorovArnoldNet(in_features=2, out_features=1)
    >>> x = torch.randn(32, 2)
    >>> model(x).shape
    torch.Size([32, 1])
    """

    __model_checkpoint_version__ = "0.1.0"

    def __init__(
        self,
        in_features: int,
        out_features: int,
        frequencies: Tuple = ("axis", tuple(range(10))),
        layer_size: int = 64,
        nr_layers: int = 3,
        grid_size: int = 10,
        spline_order: int = 3,
        base_activation_fn: str = "tanh",
        grid_range: Tuple[float, float] = (-1.0, 1.0),
        free_knot: bool = False,
        outer_layer_fn: str = "tanh",
        outer_layer: bool = True,
    ) -> None:
        super().__init__(meta=ModelMetaData(func_torch=True, auto_grad=True))

        self.in_features = in_features
        self.out_features = out_features
        self.frequencies = frequencies
        self.layer_size = layer_size
        self.nr_layers = nr_layers
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.base_activation_fn = base_activation_fn
        self.outer_layer_fn = outer_layer_fn

        # Fourier-feature embedding over the full coordinate tensor
        self.fourier_layer = FourierLayer(
            in_features=in_features, frequencies=frequencies
        )
        fourier_out = self.fourier_layer.out_features()

        # KAN core consumes raw coordinates concatenated with Fourier features
        kan_in_features = in_features + fourier_out
        self._impl = KolmogorovArnoldNetCore(
            in_features=kan_in_features,
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
        r"""Forward pass through the Fourier-feature KAN.

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

        # concatenate raw coordinates with their Fourier features
        feats = self.fourier_layer(x)  # (B, fourier_out)
        x_cat = torch.cat([x, feats], dim=-1)  # (B, D_in + fourier_out)
        return self._impl(x_cat)
