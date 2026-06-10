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

r"""Split-trunk network: groups outputs by their input dependency.

Each group of output features is served by an independent subnetwork that only
sees the input features that group depends on.  Outputs that depend on the same
set of inputs share a subnetwork.  This mirrors the structure of many coupled
PDE systems where different fields depend on different subsets of coordinates.

The legacy ``Arch`` implementation worked with named keys.  This keyless port
operates on input/output *indices*: ``key_mapping`` maps a group name to the
list of input indices it depends on, and ``output_groups`` maps the same group
name to the list of output indices it produces.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

import physicsnemo
from physicsnemo.core.meta import ModelMetaData
from physicsnemo.models.mlp.fully_connected import FullyConnected


class SplitTrunkNet(physicsnemo.Module):
    r"""Split-trunk network grouping outputs by input dependency.

    Outputs are partitioned into groups; each group is produced by a dedicated
    subnetwork that receives only the input features the group depends on.  When
    no grouping is supplied, a single subnetwork maps all inputs to all outputs
    (a standard fully-connected topology).

    Subnetworks are built from a user-supplied factory (``subnet_cls``).  This
    follows rule MOD-009: a class / factory is injected for dependency injection
    rather than a string class name.

    .. code-block:: python

        def subnet_cls(in_features: int, out_features: int, **subnet_kwargs):
            '''Returns an nn.Module mapping (B, in_features) -> (B, out_features).'''
            ...

    where ``subnet_cls`` is called once per group.

    Parameters
    ----------
    in_features : int
        Number of input coordinate features :math:`D_{in}`.
    out_features : int
        Total number of output features :math:`D_{out}`.
    key_mapping : Dict[str, List[int]], optional, default=None
        Maps each group name to the list of input indices it depends on.  When
        ``None``, a single group depending on all inputs is used.
    output_groups : Dict[str, List[int]], optional, default=None
        Maps each group name to the list of output indices it produces.  Must
        share keys with ``key_mapping`` and partition ``range(out_features)``.
        When ``None``, the single group produces all outputs.
    positive_keys : List[int], optional, default=None
        Output indices on which a softplus is applied to enforce positivity.
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
    >>> from physicsnemo.experimental.models.pinn.split_trunk import SplitTrunkNet
    >>> model = SplitTrunkNet(
    ...     in_features=3,
    ...     out_features=2,
    ...     key_mapping={"u": [0, 1], "v": [2]},
    ...     output_groups={"u": [0], "v": [1]},
    ...     positive_keys=[1],
    ... )
    >>> x = torch.randn(16, 3)
    >>> model(x).shape
    torch.Size([16, 2])
    """

    __model_checkpoint_version__ = "0.1.0"

    def __init__(
        self,
        in_features: int,
        out_features: int,
        key_mapping: Optional[Dict[str, List[int]]] = None,
        output_groups: Optional[Dict[str, List[int]]] = None,
        positive_keys: Optional[List[int]] = None,
        subnet_cls: Optional[Callable[..., nn.Module]] = None,
        subnet_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(meta=ModelMetaData(func_torch=True, auto_grad=True))

        self.in_features = in_features
        self.out_features = out_features

        # MOD-009: subnetwork is injected as a class/factory, never a string name
        if subnet_cls is None:
            subnet_cls = FullyConnected
        self.subnet_cls = subnet_cls
        subnet_kwargs = dict(subnet_kwargs or {})
        self.subnet_kwargs = subnet_kwargs

        # default: a single group over all inputs and outputs
        if key_mapping is None or output_groups is None:
            key_mapping = {"all": list(range(in_features))}
            output_groups = {"all": list(range(out_features))}

        if set(key_mapping) != set(output_groups):
            raise ValueError(
                "key_mapping and output_groups must share the same group names"
            )

        # validate index ranges and that outputs partition range(out_features)
        seen_outputs: set = set()
        for name, in_idx in key_mapping.items():
            for i in in_idx:
                if not 0 <= i < in_features:
                    raise ValueError(
                        f"Group '{name}' references input index {i} out of range "
                        f"[0, {in_features})"
                    )
            for o in output_groups[name]:
                if not 0 <= o < out_features:
                    raise ValueError(
                        f"Group '{name}' references output index {o} out of range "
                        f"[0, {out_features})"
                    )
                if o in seen_outputs:
                    raise ValueError(f"Output index {o} assigned to multiple groups")
                seen_outputs.add(o)
        if len(seen_outputs) != out_features:
            raise ValueError(
                f"output_groups must cover all {out_features} outputs, "
                f"got {sorted(seen_outputs)}"
            )

        # deterministic group ordering for reproducible parameter layout
        self.key_mapping = OrderedDict(
            (name, list(key_mapping[name])) for name in sorted(key_mapping)
        )
        self.output_groups = OrderedDict(
            (name, list(output_groups[name])) for name in sorted(output_groups)
        )

        # validate requested positive output indices
        if positive_keys is None:
            positive_keys = []
        for o in positive_keys:
            if not 0 <= o < out_features:
                raise ValueError(
                    f"positive key index {o} out of range [0, {out_features})"
                )
        self.positive_keys = list(positive_keys)

        # build one subnetwork per group
        self.subnets = nn.ModuleList(
            subnet_cls(
                in_features=len(self.key_mapping[name]),
                out_features=len(self.output_groups[name]),
                **subnet_kwargs,
            )
            for name in self.key_mapping
        )

    def forward(
        self, x: Float[Tensor, "batch in_features"]
    ) -> Float[Tensor, "batch out_features"]:
        r"""Forward pass through the split-trunk network.

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

        # scatter each subnetwork's outputs into the full output tensor
        out = x.new_zeros(*x.shape[:-1], self.out_features)
        positive_mask = set(self.positive_keys)
        for subnet, name in zip(self.subnets, self.key_mapping):
            in_idx = self.key_mapping[name]
            out_idx = self.output_groups[name]
            x_sub = x[..., in_idx]  # (B, len(in_idx))
            y_sub = subnet(x_sub)  # (B, len(out_idx))
            for col, o in enumerate(out_idx):
                val = y_sub[..., col]
                if o in positive_mask:
                    val = F.softplus(val)  # enforce positivity
                out[..., o] = val
        return out
