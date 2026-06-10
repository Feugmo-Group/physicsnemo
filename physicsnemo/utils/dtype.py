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

"""Utilities for controlling PyTorch's global default floating-point dtype.

Physics-informed neural networks (PINNs) frequently require double precision
to keep automatic-differentiation residuals well conditioned. These helpers
provide a small, explicit interface for toggling the global default dtype.
"""

import torch

__all__ = ["set_default_dtype", "get_default_dtype"]

# Floating-point dtypes that PyTorch accepts as a global default.
_FLOATING_DTYPES = frozenset(
    {torch.float16, torch.float32, torch.float64, torch.bfloat16}
)


def set_default_dtype(dtype: torch.dtype = torch.float64) -> None:
    r"""Set PyTorch's global default floating-point dtype.

    This updates the dtype used when creating tensors without an explicit
    ``dtype`` argument (for example via :func:`torch.tensor` or
    :func:`torch.zeros`). Double precision (``torch.float64``) is recommended
    for PINNs because it stabilizes the higher-order derivatives that arise in
    automatic differentiation of PDE residuals.

    Parameters
    ----------
    dtype : torch.dtype, optional
        The floating-point dtype to use as the global default. Must be one of
        ``torch.float16``, ``torch.float32``, ``torch.float64``, or
        ``torch.bfloat16``. Defaults to ``torch.float64``.

    Returns
    -------
    None
        This function mutates global PyTorch state and returns nothing.

    Raises
    ------
    ValueError
        If ``dtype`` is not a floating-point :class:`torch.dtype`.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.utils.dtype import set_default_dtype
    >>> set_default_dtype(torch.float64)
    >>> torch.get_default_dtype()
    torch.float64
    """
    if not isinstance(dtype, torch.dtype) or dtype not in _FLOATING_DTYPES:
        raise ValueError(
            "dtype must be a floating-point torch.dtype "
            "(torch.float16, torch.float32, torch.float64, or torch.bfloat16); "
            f"got {dtype!r}."
        )
    torch.set_default_dtype(dtype)


def get_default_dtype() -> torch.dtype:
    r"""Return PyTorch's current global default floating-point dtype.

    Returns
    -------
    torch.dtype
        The dtype currently used when creating tensors without an explicit
        ``dtype`` argument.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.utils.dtype import set_default_dtype, get_default_dtype
    >>> set_default_dtype(torch.float32)
    >>> get_default_dtype()
    torch.float32
    """
    return torch.get_default_dtype()
