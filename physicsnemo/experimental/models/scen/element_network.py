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

"""SCENElementNetwork: vmap-batched multi-element PINN model.

Design
------
``self.sp`` (stacked ParameterDict) is the ONLY source of trainable parameters.
``self.networks`` is a frozen ModuleList used solely as architecture template
for ``functional_call``.

This separation is required because ``torch.func.stack_module_state``
internally detaches copied parameters from the autograd graph, which would
silently break gradient flow.  With stacked params, gradients from
``loss.backward()`` propagate: loss → vmap unwinding → functional_call →
``self.sp`` (leaf tensors that the optimizer updates).

Training orchestration (optimizer loops, loss closures) lives in the example
``trainer.py`` files, not in this class.
"""

from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor
from torch.func import functional_call, vmap

import physicsnemo
from physicsnemo.core.meta import ModelMetaData
from physicsnemo.experimental.models.scen.dvr_mapper import DVRMapper
from physicsnemo.experimental.models.scen.legendre_kan import LegendreKAN


def _safe_key(name: str) -> str:
    """``'0.weight'`` → ``'0__weight'`` (dots not allowed in ParameterDict keys)."""
    return name.replace(".", "__")


def _orig_key(key: str) -> str:
    """``'0__weight'`` → ``'0.weight'``."""
    return key.replace("__", ".")


class SCENElementNetwork(physicsnemo.Module):
    r"""Spectral Collocation Element Network (SCEN) for multi-element PINN training.

    Manages :math:`K` sub-networks (one per spectral element), a stacked
    ``ParameterDict`` for vmap-compatible batched evaluation, and global
    block-diagonal differentiation matrices registered as non-trainable buffers.

    ``forward()`` evaluates all sub-networks at their LGL nodes and returns the
    concatenated solution over the full domain.  Physics loss computation and
    optimisation are handled externally by the caller (see the example trainers
    under ``examples/electrochemistry/``).

    .. note::

        ``self.sp`` (the stacked ``ParameterDict``) is the **only** source of
        trainable parameters.  ``self.networks`` is a frozen ``ModuleList``
        used solely as architecture template for ``functional_call``.  This
        separation is required because ``torch.func.stack_module_state``
        internally detaches parameters from the autograd graph, which would
        silently break gradient flow.

    Parameters
    ----------
    element_configs : list[dict]
        One dict per subdomain.  Required keys: ``N`` *(int)*, ``a`` *(float)*,
        ``b`` *(float)*.  Optional keys: ``alpha`` *(float, default 0.0)*,
        ``quadrature`` *(str, default ``'lgl'``)*,
        ``mapping`` *(str, default ``'kte'``)*
    hidden_dim : int
        Width of each hidden layer.  Default ``64``.
    n_layers : int
        Number of layers (depth).  Default ``3``.
    backbone : str
        Network backbone: ``'mlp'`` or ``'kan'``.  Default ``'mlp'``.
    poly_degree : int
        Legendre polynomial degree per edge (KAN only).  Default ``4``.
    dtype : torch.dtype
        Parameter dtype.  Default ``torch.float32``.
    device : optional
        Parameter device.  Default ``cpu``.

    Forward
    -------
    No input arguments.

    Outputs
    -------
    u_global : torch.Tensor
        Concatenated solution values at all LGL nodes, shape
        :math:`(N_{\text{total}},)` where
        :math:`N_{\text{total}} = \sum_k N_k`.

    Examples
    --------
    >>> cfg = [{"N": 8, "a": 0.0, "b": 0.5}, {"N": 8, "a": 0.5, "b": 1.0}]
    >>> model = SCENElementNetwork(cfg, hidden_dim=16, n_layers=2)
    >>> u = model()
    >>> u.shape
    torch.Size([16])
    """

    __model_checkpoint_version__ = "0.1.0"

    def __init__(
        self,
        element_configs: list[dict],
        hidden_dim: int = 64,
        n_layers: int = 3,
        backbone: Literal["mlp", "kan"] = "mlp",
        poly_degree: int = 4,
        dtype: torch.dtype = torch.float32,
        device=None,
    ):
        super().__init__(meta=ModelMetaData(func_torch=True))
        if device is None:
            device = torch.device("cpu")
        self.dtype = dtype
        self._device_arg = str(device)
        _device = torch.device(device) if not isinstance(device, torch.device) else device

        self.element_sizes = [cfg["N"] for cfg in element_configs]
        self._uniform_N = len(set(self.element_sizes)) == 1
        self._element_configs = list(element_configs)
        self.backbone = backbone
        self.poly_degree = poly_degree
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        K = len(element_configs)

        # ── Mappers (non-trainable geometry) ─────────────────────────────────
        self.mappers = [
            DVRMapper(
                cfg["N"],
                cfg["a"],
                cfg["b"],
                cfg.get("alpha", 0.0),
                quadrature=cfg.get("quadrature", "lgl"),
                mapping=cfg.get("mapping", "kte"),
                dtype=dtype,
                device=_device,
            )
            for cfg in element_configs
        ]

        # ── Network factory ───────────────────────────────────────────────────
        if backbone == "mlp":
            def _make_network():
                layers = [nn.Linear(1, hidden_dim), nn.Tanh()]
                for _ in range(n_layers - 2):
                    layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
                layers.append(nn.Linear(hidden_dim, 1))
                net = nn.Sequential(*layers).to(dtype=dtype, device=_device)
                for m in net.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        nn.init.zeros_(m.bias)
                return net
        elif backbone == "kan":
            def _make_network():
                return LegendreKAN(
                    hidden_dim=hidden_dim,
                    n_layers=n_layers,
                    poly_degree=poly_degree,
                    dtype=dtype,
                    device=_device,
                )
        else:
            raise ValueError(f"backbone must be 'mlp' or 'kan', got '{backbone}'")

        # ── Frozen reference networks (architecture templates only) ───────────
        nets = [_make_network() for _ in range(K)]
        self.networks = nn.ModuleList(nets)
        for p in self.networks.parameters():
            p.requires_grad_(False)

        # ── Stacked trainable ParameterDict ───────────────────────────────────
        # self.sp[safe_key] has shape (K, *param_shape).
        # These are the ONLY parameters seen by the optimizer.
        self.sp = nn.ParameterDict()
        self._param_names: list[str] = [
            name for name, _ in nets[0].named_parameters()
        ]
        for pname in self._param_names:
            slices = [
                dict(net.named_parameters())[pname].data.clone() for net in nets
            ]
            stacked = torch.stack(slices, dim=0)  # (K, *shape)
            self.sp[_safe_key(pname)] = nn.Parameter(stacked)

        # ── Global block-diagonal matrices (non-trainable buffers) ────────────
        D1_global = torch.block_diag(*[m.D1 for m in self.mappers])
        D2_global = torch.block_diag(*[m.D2 for m in self.mappers])
        D4_global = D2_global @ D2_global
        w_global = torch.cat([m.weights for m in self.mappers])
        # Per-element normalised weights so every element contributes equally.
        w_norm_global = torch.cat(
            [m.weights / m.weights.sum() for m in self.mappers]
        )
        self.register_buffer("D1_global", D1_global)
        self.register_buffer("D2_global", D2_global)
        self.register_buffer("D4_global", D4_global)
        self.register_buffer("w_global", w_global)
        self.register_buffer("w_norm_global", w_norm_global, persistent=False)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _params_for_call(self) -> dict[str, torch.Tensor]:
        return {name: self.sp[_safe_key(name)] for name in self._param_names}

    def _network_input(self, mapper: DVRMapper) -> torch.Tensor:
        """Return ``(N, 1)`` input tensor for one element."""
        if self.backbone == "kan":
            return mapper.xi_ref.unsqueeze(1)
        # MLP: normalise physical nodes to [-1, 1]
        a, b = mapper.nodes[0], mapper.nodes[-1]
        return (2.0 * (mapper.nodes - a) / (b - a) - 1.0).unsqueeze(1)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self) -> Float[Tensor, "N_total"]:
        r"""Evaluate all sub-networks at their LGL nodes.

        Uniform-N elements use a single vmapped call; mixed-N falls back to a
        loop with one ``functional_call`` per element.

        Returns
        -------
        u_global : torch.Tensor
            Concatenated solution values, shape :math:`(N_{\text{total}},)`.
        """
        params_dict = self._params_for_call()
        base = self.networks[0]

        if self._uniform_N:
            x_stacked = torch.stack(
                [self._network_input(m) for m in self.mappers], dim=0
            )  # (K, N, 1)

            def _single(params_slice, x):
                return functional_call(base, params_slice, (x,))

            outputs = vmap(_single, in_dims=(0, 0))(params_dict, x_stacked)
            return outputs.squeeze(-1).reshape(-1)  # (K*N,)
        else:
            parts = []
            for i, mapper in enumerate(self.mappers):
                params_slice = {name: tensor[i] for name, tensor in params_dict.items()}
                out = functional_call(base, params_slice, (self._network_input(mapper),))
                parts.append(out.squeeze(1))
            return torch.cat(parts)

    # ── Utilities ─────────────────────────────────────────────────────────────

    def split_global(self, u: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Split global solution vector into per-element tensors."""
        return torch.split(u, self.element_sizes)

    def sync_to_networks(self) -> None:
        """Copy stacked params back into ``self.networks`` for export."""
        params_dict = self._params_for_call()
        with torch.no_grad():
            for i, net in enumerate(self.networks):
                for pname in self._param_names:
                    p = dict(net.named_parameters())[pname]
                    p.data.copy_(params_dict[pname][i])
