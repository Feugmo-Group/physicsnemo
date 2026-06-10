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

r"""Thin factory around the optional ``pytorch_optimizer`` package.

The `pytorch_optimizer <https://github.com/kozistr/pytorch_optimizer>`_ project
provides a large collection of modern optimizers (SOAP, Shampoo, Lion, ...).
That package is an *optional* PhysicsNeMo dependency: importing this module never
fails, but actually constructing one of these optimizers requires the package to
be installed.

Install it via either of::

    pip install "nvidia-physicsnemo[optim]"
    pip install pytorch_optimizer

All imports are performed lazily inside the factory functions so that this
module can always be imported (e.g. by an optimizer registry) regardless of
whether the optional dependency is present.
"""

import torch

_INSTALL_HINT = (
    "The 'pytorch_optimizer' package is required to use this optimizer but is "
    "not installed. Install it with one of:\n"
    '    pip install "nvidia-physicsnemo[optim]"\n'
    "    pip install pytorch_optimizer"
)


def _import_pytorch_optimizer():
    r"""Import and return the ``pytorch_optimizer`` module or raise a clear error.

    Returns
    -------
    module
        The imported ``pytorch_optimizer`` module.

    Raises
    ------
    ModuleNotFoundError
        If ``pytorch_optimizer`` is not installed, with an actionable install
        hint in the message.
    """
    try:
        import pytorch_optimizer  # noqa: PLC0415  (intentional lazy import)
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ModuleNotFoundError(_INSTALL_HINT) from exc
    return pytorch_optimizer


def make_pytorch_optimizer(name: str, params, **kwargs) -> torch.optim.Optimizer:
    r"""Construct any optimizer exported by the ``pytorch_optimizer`` package.

    Parameters
    ----------
    name : str
        Name of the optimizer class as exported by ``pytorch_optimizer`` (for
        example ``"SOAP"``, ``"Shampoo"``, ``"Lion"``). The lookup is
        case-insensitive against the package's attributes.
    params : iterable
        Iterable of parameters or parameter-group dicts to optimize.
    **kwargs
        Keyword arguments forwarded verbatim to the optimizer constructor.

    Returns
    -------
    torch.optim.Optimizer
        The constructed optimizer instance.

    Raises
    ------
    ModuleNotFoundError
        If ``pytorch_optimizer`` is not installed.
    AttributeError
        If ``name`` does not correspond to an optimizer in the package.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.optim.pytorch_optimizer_wrapper import (
    ...     make_pytorch_optimizer,
    ... )
    >>> model = torch.nn.Linear(4, 4)  # doctest: +SKIP
    >>> opt = make_pytorch_optimizer("SOAP", model.parameters())  # doctest: +SKIP
    """
    pkg = _import_pytorch_optimizer()

    cls = getattr(pkg, name, None)
    if cls is None:
        # Case-insensitive fallback.
        lowered = name.lower()
        for attr in dir(pkg):
            if attr.lower() == lowered:
                cls = getattr(pkg, attr)
                break
    if cls is None:
        raise AttributeError(f"'pytorch_optimizer' has no optimizer named {name!r}.")
    return cls(params, **kwargs)


def SOAP(
    params,
    lr: float = 3e-3,
    betas: tuple[float, float] = (0.95, 0.95),
    shampoo_beta: float | None = None,
    weight_decay: float = 1e-2,
    precondition_frequency: int = 10,
    max_precondition_dim: int = 10000,
    merge_dims: bool = False,
    precondition_1d: bool = False,
    correct_bias: bool = True,
    normalize_gradient: bool = False,
    data_format: str = "channels_first",
    eps: float = 1e-8,
    **kwargs,
) -> torch.optim.Optimizer:
    r"""Convenience constructor for ``pytorch_optimizer.SOAP``.

    SOAP (ShampoO with Adam in the Preconditioner's eigenbasis) is a
    second-order-flavored optimizer. This wrapper exposes the commonly used
    hyperparameters with PhysicsNeMo defaults and forwards everything to the
    underlying implementation.

    Parameters
    ----------
    params : iterable
        Iterable of parameters or parameter-group dicts to optimize.
    lr : float, optional
        Learning rate. Default is ``3e-3``.
    betas : tuple of float, optional
        Adam ``(beta1, beta2)`` coefficients used inside the preconditioner
        eigenbasis. Default is ``(0.95, 0.95)``.
    shampoo_beta : float or None, optional
        Decay for the Shampoo preconditioner; ``None`` reuses ``betas[1]``.
        Default is ``None``.
    weight_decay : float, optional
        Decoupled weight decay. Default is ``1e-2``.
    precondition_frequency : int, optional
        Number of steps between preconditioner eigen-updates. Default is ``10``.
    max_precondition_dim : int, optional
        Maximum dimension that is preconditioned; larger dims fall back to Adam.
        Default is ``10000``.
    merge_dims : bool, optional
        Whether to merge tensor dimensions before preconditioning. Default is
        ``False``.
    precondition_1d : bool, optional
        Whether to precondition 1-D parameters. Default is ``False``.
    correct_bias : bool, optional
        Apply Adam bias correction. Default is ``True``.
    normalize_gradient : bool, optional
        Normalize gradients before the update. Default is ``False``.
    data_format : str, optional
        Layout of convolution weights, ``"channels_first"`` or
        ``"channels_last"``. Default is ``"channels_first"``.
    eps : float, optional
        Numerical-stability term. Default is ``1e-8``.
    **kwargs
        Additional keyword arguments forwarded to ``pytorch_optimizer.SOAP``.

    Returns
    -------
    torch.optim.Optimizer
        A configured SOAP optimizer instance.

    Raises
    ------
    ModuleNotFoundError
        If ``pytorch_optimizer`` is not installed.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.optim.pytorch_optimizer_wrapper import SOAP
    >>> model = torch.nn.Linear(4, 4)  # doctest: +SKIP
    >>> opt = SOAP(model.parameters(), lr=3e-3)  # doctest: +SKIP
    """
    return make_pytorch_optimizer(
        "SOAP",
        params,
        lr=lr,
        betas=betas,
        shampoo_beta=shampoo_beta,
        weight_decay=weight_decay,
        precondition_frequency=precondition_frequency,
        max_precondition_dim=max_precondition_dim,
        merge_dims=merge_dims,
        precondition_1d=precondition_1d,
        correct_bias=correct_bias,
        normalize_gradient=normalize_gradient,
        data_format=data_format,
        eps=eps,
        **kwargs,
    )
