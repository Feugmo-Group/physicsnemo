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

"""DVR-collocation gradient method for :class:`PhysicsInformer`.

This module adds a ``grad_method="dvr"`` path to the symbolic
:class:`~physicsnemo.sym.eq.phy_informer.PhysicsInformer`: instead of computing
derivatives by automatic differentiation or FFT, derivatives are obtained by
applying **precomputed DVRMapper differentiation matrices** (Gauss-Lobatto-
Legendre collocation operators) to a field laid out on a structured tensor-
product grid.  This is the spectral-element ("NSEM"/"SCEN") differentiation used
by the electrochemistry DVR examples; it is *not* the same as PhysicsNeMo's
built-in ``grad_method="spectral"`` (FFT differentiation on a periodic, uniform
grid).

The contract a gradient module must satisfy inside ``PhysicsInformer`` is small:
``forward(input_dict) -> {f"{var}__x": tensor, f"{var}__x__x": tensor, ...}``.
:class:`GradientsDVR` implements exactly that by contracting a differentiation
matrix with the field along a named axis, so every downstream stage (the
symbolic residual evaluation built from ``equations.make_computations()``) is
reused unchanged.

Axis operators
--------------
The caller supplies, per spatial axis name (``"x"``, ``"y"``, ``"z"`` in the
PDE's coordinate order), the tensor-grid axis the operator acts on and the
first/second derivative matrices::

    operators = {
        "x": AxisOperator(axis=1, D1=D1x, D2=D2x),  # acts along grid axis 1
        "y": AxisOperator(axis=0, D1=D1y, D2=None),  # time, only first deriv
    }

For a 1D problem the field is a vector ``(N,)`` and the single axis is ``0``;
for a space-time problem the field is ``(Nt, Nx)`` with time on axis ``0`` and
space on axis ``1``; for a flattened multi-dimensional grid the field is ``(N,)``
and the operators are the full flattened (e.g. Kronecker) matrices on axis ``0``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from physicsnemo.sym.computation import Computation
from physicsnemo.sym.eq.pde import PDE
from physicsnemo.sym.eq.phy_informer import PhysicsInformer

_AXIS_NAMES = ("x", "y", "z")


@dataclass
class AxisOperator:
    """Differentiation operators for one spatial axis of a DVR grid.

    Parameters
    ----------
    axis : int
        Index of the tensor-grid axis this operator contracts over (e.g. ``0``
        for time and ``1`` for space in an ``(Nt, Nx)`` field).
    D1 : torch.Tensor or None
        First-derivative matrix of shape ``(n, n)`` where ``n`` is the size of
        the field along ``axis``.  May be ``None`` if no first derivative along
        this axis is required by the PDE.
    D2 : torch.Tensor or None
        Second-derivative matrix of shape ``(n, n)``; ``None`` if not required.
    """

    axis: int
    D1: Optional[torch.Tensor] = None
    D2: Optional[torch.Tensor] = None


def _apply_along(matrix: torch.Tensor, field: torch.Tensor, axis: int) -> torch.Tensor:
    """Contract a differentiation matrix with a field along a single axis.

    Computes ``out[..., i, ...] = sum_j matrix[i, j] * field[..., j, ...]`` with
    ``j`` running over ``axis``.  Generalises ``D @ f`` (1D), ``f @ Dx.T`` (space
    axis of an ``(Nt, Nx)`` field) and ``Dy @ f`` (time axis) to any rank.

    Parameters
    ----------
    matrix : torch.Tensor
        Differentiation matrix, shape ``(n, n)``.
    field : torch.Tensor
        Field whose ``axis`` dimension has size ``n``.
    axis : int
        Axis of ``field`` to differentiate along.

    Returns
    -------
    torch.Tensor
        Derivative, same shape as ``field``.
    """
    out = torch.tensordot(matrix, field, dims=([1], [axis]))
    return torch.movedim(out, 0, axis)


class GradientsDVR(torch.nn.Module):
    """Compute derivatives of one field via precomputed DVR matrices.

    Implements the ``PhysicsInformer`` gradient-module contract for a single
    variable: ``forward(input_dict)`` returns the requested first- or second-
    order derivatives keyed ``f"{invar}__x"``, ``f"{invar}__x__x"``, mixed
    ``f"{invar}__x__y"``, etc., obtained by applying the per-axis operators.

    Parameters
    ----------
    invar : str
        Name of the field to differentiate (the key read from ``input_dict``).
    dim : int
        Number of spatial axes of the PDE (1, 2 or 3).
    order : int
        Derivative order this module produces (``1`` or ``2``).
    operators : dict of str to AxisOperator
        Per-axis operators keyed by axis name (``"x"``, ``"y"``, ``"z"``); must
        contain an entry for each of the first ``dim`` axis names.
    return_mixed_derivs : bool, default False
        If ``True`` and ``order == 2``, also emit off-diagonal mixed second
        derivatives ``f"{invar}__a__b"`` for ``a < b``.
    """

    def __init__(
        self,
        invar: str,
        dim: int,
        order: int,
        operators: Dict[str, AxisOperator],
        return_mixed_derivs: bool = False,
    ):
        super().__init__()
        self.invar = invar
        self.dim = dim
        self.order = order
        self.return_mixed_derivs = return_mixed_derivs
        self.axis_names = list(_AXIS_NAMES[:dim])

        # Register operator matrices as buffers so ``.to(device)`` moves them and
        # they are not treated as trainable parameters.
        self._tensor_axis: Dict[str, int] = {}
        self._has_d1: set[str] = set()
        self._has_d2: set[str] = set()
        for name in self.axis_names:
            op = operators[name]
            self._tensor_axis[name] = op.axis
            if op.D1 is not None:
                self.register_buffer(f"_D1_{name}", op.D1)
                self._has_d1.add(name)
            if op.D2 is not None:
                self.register_buffer(f"_D2_{name}", op.D2)
                self._has_d2.add(name)

    def forward(self, input_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Return the requested derivatives of ``self.invar``.

        Parameters
        ----------
        input_dict : dict of str to torch.Tensor
            Must contain ``self.invar`` as a field laid out on the DVR grid.

        Returns
        -------
        dict of str to torch.Tensor
            Derivative leaves keyed by the ``"__"`` convention.
        """
        field = input_dict[self.invar]
        result: Dict[str, torch.Tensor] = {}

        if self.order == 1:
            for name in self.axis_names:
                ax = self._tensor_axis[name]
                d1 = getattr(self, f"_D1_{name}")
                result[f"{self.invar}__{name}"] = _apply_along(d1, field, ax)
            return result

        # order == 2: pure second derivatives along each axis.  If an axis has no
        # dedicated D2, fall back to applying D1 twice (exact for nodal
        # collocation matrices, ``D2 == D1 @ D1``).
        for name in self.axis_names:
            ax = self._tensor_axis[name]
            if name in self._has_d2:
                d2 = getattr(self, f"_D2_{name}")
                result[f"{self.invar}__{name}__{name}"] = _apply_along(d2, field, ax)
            else:
                d1 = getattr(self, f"_D1_{name}")
                result[f"{self.invar}__{name}__{name}"] = _apply_along(
                    d1, _apply_along(d1, field, ax), ax
                )

        # ... plus mixed second derivatives if requested (D1 along a, then b).
        if self.return_mixed_derivs:
            for i in range(self.dim):
                for j in range(i + 1, self.dim):
                    a, b = self.axis_names[i], self.axis_names[j]
                    da = getattr(self, f"_D1_{a}")
                    db = getattr(self, f"_D1_{b}")
                    inner = _apply_along(da, field, self._tensor_axis[a])
                    result[f"{self.invar}__{a}__{b}"] = _apply_along(
                        db, inner, self._tensor_axis[b]
                    )
        return result


class DVRPhysicsInformer(PhysicsInformer):
    """:class:`PhysicsInformer` whose derivatives come from DVR operators.

    Selects ``grad_method="dvr"``: symbolic residuals are assembled exactly as in
    the base class (via ``equations.make_computations()``), but the derivative
    diff-nodes apply precomputed :class:`GradientsDVR` operators instead of
    autodiff/FFT/finite differences.  ``forward(input_dict)`` therefore takes the
    fields (and any non-derivative leaves the equations reference) laid out on the
    DVR grid and returns each requested residual on that grid.

    Parameters
    ----------
    required_outputs : list of str
        Names of the equations (``PDE`` keys) to evaluate.
    equations : PDE
        The symbolic PDE; its ``dim`` sets the number of spatial axes.
    operators : dict of str to AxisOperator
        Per-axis DVR operators, keyed by axis name (``"x"``, ``"y"``, ``"z"``).
    detach_names : list of str or None, optional
        Leaf names whose gradient is detached when building the computations
        (passed through to ``equations.make_computations``).
    device : str or torch.device or None, optional
        Device for the diff-node modules.

    Notes
    -----
    This path is for full-grid residuals.  Boundary-only or reduced-dimension
    terms (interface fluxes, ODE-in-time terms) are evaluated by the caller on
    the appropriate slice; they are not part of the ``required_outputs`` graph.
    """

    def __init__(
        self,
        required_outputs: List[str],
        equations: PDE,
        operators: Dict[str, AxisOperator],
        detach_names: Optional[List[str]] = None,
        device: Optional[str] = None,
    ):
        self._dvr_operators = operators
        super().__init__(
            required_outputs=required_outputs,
            equations=equations,
            grad_method="dvr",
            detach_names=detach_names,
            device=device,
        )

    def _create_diff_node(self, var, dim, order):
        """Build a DVR derivative node (overrides the base dispatch).

        Parameters
        ----------
        var : str
            Field name to differentiate.
        dim : int
            Number of spatial axes.
        order : int
            Derivative order (1 or 2).

        Returns
        -------
        Computation
            A computation mapping ``[var]`` to its derivative leaves.
        """
        module = GradientsDVR(
            var,
            dim,
            order,
            self._dvr_operators,
            return_mixed_derivs=self.require_mixed_derivs and order == 2,
        )
        output_keys = self._derivative_keys(
            var, dim, order, return_mixed_derivs=self.require_mixed_derivs
        )
        return Computation([var], output_keys, module.to(self.device))
