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

r"""Separable Physics-Informed Neural Network (SPINN).

A separable network based on CP (CANDECOMP/PARAFAC) tensor decomposition: one
subnetwork per input dimension, each producing ``rank`` features.  The outputs
are combined as an elementwise product across dimensions and summed (or linearly
mixed) over the rank axis.

Based on `Separable physics-informed neural networks
<https://arxiv.org/abs/2306.15969>`_.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

import physicsnemo
from physicsnemo.core.meta import ModelMetaData
from physicsnemo.models.mlp.fully_connected import FullyConnected


class SeparableNet(physicsnemo.Module):
    r"""Separable PINN using CP tensor decomposition.

    Each scalar input dimension is processed by its own subnetwork producing a
    ``rank``-dimensional feature vector.  For a scalar output the network returns
    the rank-sum of the elementwise product of all per-dimension features; for a
    vector output a learned linear head mixes the rank features.

    Each subnetwork is built from a user-supplied factory (``subnet_cls``),
    mapping one input feature to ``rank`` output features.

    .. code-block:: python

        def subnet_cls(in_features: int, out_features: int, **subnet_kwargs):
            '''Returns an nn.Module mapping (B, in_features) -> (B, out_features).'''
            ...

    where ``subnet_cls`` is called once per input dimension with
    ``in_features=1`` and ``out_features=rank``.

    Parameters
    ----------
    in_features : int
        Number of (scalar) input dimensions :math:`D_{in}`.
    out_features : int
        Number of output features :math:`D_{out}`.
    rank : int, optional, default=64
        Number of features produced by each per-dimension subnetwork.  For
        vector outputs, ``rank`` must be divisible by ``out_features``.
    subnet_cls : Callable, optional, default=None
        Factory returning a subnetwork module.  Must accept ``in_features`` and
        ``out_features`` keyword arguments.  Defaults to
        :class:`~physicsnemo.models.mlp.fully_connected.FullyConnected`.
    subnet_kwargs : Dict[str, Any], optional, default=None
        Extra keyword arguments forwarded to ``subnet_cls``.

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
    >>> from physicsnemo.experimental.models.pinn.spinn import SeparableNet
    >>> model = SeparableNet(in_features=2, out_features=1, rank=16)
    >>> x = torch.randn(32, 2)
    >>> model(x).shape
    torch.Size([32, 1])
    """

    __model_checkpoint_version__ = "0.1.0"

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 64,
        subnet_cls: Optional[Callable[..., nn.Module]] = None,
        subnet_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(meta=ModelMetaData(func_torch=True, auto_grad=True))

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        # MOD-009: subnetwork is injected as a class/factory, never a string name
        if subnet_cls is None:
            subnet_cls = FullyConnected
        self.subnet_cls = subnet_cls
        subnet_kwargs = dict(subnet_kwargs or {})
        self.subnet_kwargs = subnet_kwargs

        # rank must split evenly across vector outputs
        if out_features > 1 and rank % out_features != 0:
            raise ValueError(
                f"For vector outputs (dim={out_features}), rank ({rank}) "
                f"must be divisible by out_features"
            )

        # one subnetwork per input dimension: (B, 1) -> (B, rank)
        self.subnetworks = nn.ModuleList(
            subnet_cls(in_features=1, out_features=rank, **subnet_kwargs)
            for _ in range(in_features)
        )

        # learned mixing head only needed for vector outputs
        if out_features > 1:
            self.linear_head = nn.Linear(rank, out_features, bias=False)
        else:
            self.linear_head = None

    def forward(
        self, x: Float[Tensor, "batch in_features"]
    ) -> Float[Tensor, "batch out_features"]:
        r"""Forward pass through the separable network.

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

        # elementwise product of per-dimension rank features (CP decomposition)
        output = self.subnetworks[0](x[..., 0:1])  # (B, rank)
        for i in range(1, self.in_features):
            output = output * self.subnetworks[i](x[..., i : i + 1])

        # scalar: sum over rank; vector: learned linear mix
        if self.linear_head is None:
            return output.sum(dim=-1, keepdim=True)
        return self.linear_head(output)
