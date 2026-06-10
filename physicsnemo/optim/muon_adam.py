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

r"""Hybrid Muon + Adam optimizer.

This module provides :class:`MuonAdam`, an optimizer that routes matrix-shaped
parameters (``ndim == 2`` or ``ndim == 4``) through the Muon update rule and all
remaining parameters (biases, norms, 1-D vectors, scalars) through Adam. Muon
applies Newton--Schulz orthogonalization to the momentum-smoothed gradient,
which is well suited to the dense weight matrices of neural-network hidden
layers, while Adam handles the parameters for which orthogonalization is not
meaningful.

The implementation prefers PyTorch's built-in functional kernels
(``torch.optim._muon.muon`` and ``torch.optim.adam.adam``) when they are
available, and otherwise falls back to a self-contained Newton--Schulz
implementation so that the optimizer remains usable on older PyTorch builds.
"""

from collections.abc import MutableMapping

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer, ParamsT

# ---------------------------------------------------------------------------
# Optional fast paths from torch.optim internals.
# ---------------------------------------------------------------------------
# torch >= 2.8 ships a Muon implementation under torch.optim._muon. We use it
# when present and otherwise provide an inline Newton--Schulz fallback below.
try:  # pragma: no cover - exercised implicitly depending on torch version
    from torch.optim._muon import (
        DEFAULT_A,
        DEFAULT_B,
        DEFAULT_C,
        DEFAULT_NS_STEPS,
        EPS,
    )
    from torch.optim._muon import (
        muon as _torch_muon,
    )

    _HAS_TORCH_MUON = True
except Exception:  # pragma: no cover - older torch
    DEFAULT_A, DEFAULT_B, DEFAULT_C = 3.4445, -4.775, 2.0315
    DEFAULT_NS_STEPS = 5
    EPS = 1e-7
    _torch_muon = None
    _HAS_TORCH_MUON = False

try:  # pragma: no cover
    from torch.optim.adam import adam as _torch_adam

    _HAS_TORCH_ADAM = True
except Exception:  # pragma: no cover
    _torch_adam = None
    _HAS_TORCH_ADAM = False


def zeropower_via_newtonschulz5(
    G: Tensor,
    ns_coefficients: tuple[float, float, float] = (DEFAULT_A, DEFAULT_B, DEFAULT_C),
    ns_steps: int = DEFAULT_NS_STEPS,
    eps: float = EPS,
) -> Tensor:
    r"""Orthogonalize a matrix via a quintic Newton--Schulz iteration.

    Computes an approximation of ``U V^T`` from the (reduced) SVD ``G = U S V^T``,
    i.e. it replaces the singular values of ``G`` with ones. The iteration uses
    the quintic polynomial ``a X + b (X X^T) X + c (X X^T)^2 X`` with the tuned
    coefficients ``(a, b, c)`` that converge quickly for inputs whose singular
    values lie in ``[0, 1]`` (the input is normalized to that range first).

    Parameters
    ----------
    G : torch.Tensor
        A 2-D matrix to orthogonalize. If ``G`` has more rows than columns the
        iteration is run on its transpose for efficiency and transposed back.
    ns_coefficients : tuple of float, optional
        The ``(a, b, c)`` coefficients of the quintic polynomial. Default is
        ``(3.4445, -4.775, 2.0315)``.
    ns_steps : int, optional
        Number of Newton--Schulz iterations. Default is ``5``.
    eps : float, optional
        Small constant added to the Frobenius-norm denominator for numerical
        stability. Default is ``1e-7``.

    Returns
    -------
    torch.Tensor
        The orthogonalized matrix, same shape and dtype as ``G``.
    """
    if G.ndim != 2:
        raise ValueError("Newton--Schulz orthogonalization expects a 2-D matrix")
    a, b, c = ns_coefficients

    X = G.bfloat16() if G.dtype != torch.bfloat16 else G.clone()

    transposed = False
    if X.size(0) > X.size(1):
        X = X.mT
        transposed = True

    # Normalize so the spectral norm is <= 1 (sufficient: divide by Frobenius norm).
    X = X / (X.norm() + eps)

    for _ in range(ns_steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if transposed:
        X = X.mT

    return X.to(G.dtype)


class MuonAdam(Optimizer):
    r"""Hybrid optimizer using Muon for matrix parameters and Adam for the rest.

    Parameters with ``ndim`` equal to ``2`` or ``4`` (dense weight matrices and
    convolution kernels) are updated with the Muon rule: a momentum-smoothed
    gradient is orthogonalized via a Newton--Schulz iteration before the step.
    All other parameters (1-D biases / norm weights, scalars, embeddings stored
    as ``ndim != 2/4``) are updated with Adam.

    Internally two parameter groups are created with an ``is_muon`` flag so the
    routing is inspectable after construction.

    Parameters
    ----------
    params : ParamsT
        Iterable of parameters or parameter-group dicts to optimize.
    lr : float, optional
        Learning rate. Default is ``1e-3``.
    momentum : float, optional
        Momentum factor for the Muon branch. Default is ``0.95``.
    nesterov : bool, optional
        Use Nesterov momentum in the Muon branch. Default is ``True``.
    ns_coefficients : tuple of float, optional
        ``(a, b, c)`` coefficients of the Newton--Schulz quintic polynomial.
        Default is ``(3.4445, -4.775, 2.0315)``.
    ns_steps : int, optional
        Number of Newton--Schulz iterations. Default is ``5``.
    muon_eps : float, optional
        Numerical-stability term for the Muon branch. Default is ``1e-7``.
    muon_weight_decay : float, optional
        Decoupled weight decay applied in the Muon branch. Default is ``0.1``.
    adjust_lr_fn : str or None, optional
        Learning-rate adjustment strategy passed to the torch Muon kernel; one
        of ``"original"`` or ``"match_rms_adamw"``. Ignored by the inline
        fallback. Default is ``None``.
    betas : tuple of float, optional
        Adam ``(beta1, beta2)`` coefficients. Default is ``(0.9, 0.999)``.
    adam_eps : float, optional
        Adam numerical-stability term. Default is ``1e-8``.
    adam_weight_decay : float, optional
        Adam weight decay (L2 penalty). Default is ``0.0``.
    amsgrad : bool, optional
        Use the AMSGrad variant of Adam. Default is ``False``.
    decoupled_weight_decay : bool, optional
        If ``True``, the Adam branch behaves like AdamW. Default is ``False``.
    maximize : bool, optional
        Maximize instead of minimize the objective. Default is ``False``.
    foreach : bool or None, optional
        Use the foreach (multi-tensor) kernels when available. Default is
        ``None`` (auto).

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.optim.muon_adam import MuonAdam
    >>> model = torch.nn.Linear(4, 4)
    >>> opt = MuonAdam(model.parameters(), lr=0.05)
    >>> x = torch.randn(16, 4)
    >>> for _ in range(50):
    ...     opt.zero_grad()
    ...     loss = (model(x) - x).pow(2).mean()
    ...     loss.backward()
    ...     _ = opt.step()
    >>> # The 2-D weight is routed through Muon, the 1-D bias through Adam.
    >>> [g["is_muon"] for g in opt.param_groups]
    [True, False]
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_coefficients: tuple[float, float, float] = (
            DEFAULT_A,
            DEFAULT_B,
            DEFAULT_C,
        ),
        ns_steps: int = DEFAULT_NS_STEPS,
        muon_eps: float = EPS,
        muon_weight_decay: float = 0.1,
        adjust_lr_fn: str | None = None,
        betas: tuple[float, float] = (0.9, 0.999),
        adam_eps: float = 1e-8,
        adam_weight_decay: float = 0.0,
        amsgrad: bool = False,
        decoupled_weight_decay: bool = False,
        maximize: bool = False,
        foreach: bool | None = None,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if ns_steps < 1:
            raise ValueError(f"Invalid ns_steps: {ns_steps}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid betas: {betas}")

        params = list(params)
        # Allow either raw parameters or pre-built group dicts; flatten to params.
        flat = []
        for p in params:
            if isinstance(p, dict):
                flat.extend(p["params"])
            else:
                flat.append(p)

        muon_params = [p for p in flat if p.ndim in (2, 4)]
        adam_params = [p for p in flat if p.ndim not in (2, 4)]

        param_groups = [
            {"params": muon_params, "is_muon": True},
            {"params": adam_params, "is_muon": False},
        ]

        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_coefficients=ns_coefficients,
            ns_steps=ns_steps,
            muon_eps=muon_eps,
            muon_weight_decay=muon_weight_decay,
            adjust_lr_fn=adjust_lr_fn,
            betas=betas,
            adam_eps=adam_eps,
            adam_weight_decay=adam_weight_decay,
            amsgrad=amsgrad,
            decoupled_weight_decay=decoupled_weight_decay,
            maximize=maximize,
            foreach=foreach,
        )
        super().__init__(param_groups, defaults)

    # ------------------------------------------------------------------
    # State initialization helpers
    # ------------------------------------------------------------------
    def _init_group_muon(
        self,
        group: MutableMapping,
        params_with_grad: list[Tensor],
        grads: list[Tensor],
        momentum_bufs: list[Tensor],
    ) -> None:
        for p in group["params"]:
            if p.grad is None:
                continue
            if torch.is_complex(p):
                raise RuntimeError("MuonAdam does not support complex parameters")
            if p.grad.is_sparse:
                raise RuntimeError("MuonAdam does not support sparse gradients")

            params_with_grad.append(p)
            grads.append(p.grad)

            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(
                    p.grad, memory_format=torch.preserve_format
                )
            momentum_bufs.append(state["momentum_buffer"])

    def _init_group_adam(
        self,
        group: MutableMapping,
        params_with_grad: list[Tensor],
        grads: list[Tensor],
        exp_avgs: list[Tensor],
        exp_avg_sqs: list[Tensor],
        max_exp_avg_sqs: list[Tensor],
        state_steps: list[Tensor],
    ) -> bool:
        has_complex = False
        for p in group["params"]:
            if p.grad is None:
                continue
            has_complex |= torch.is_complex(p)
            if p.grad.is_sparse:
                raise RuntimeError(
                    "MuonAdam Adam branch does not support sparse gradients"
                )
            params_with_grad.append(p)
            grads.append(p.grad)

            state = self.state[p]
            if len(state) == 0:
                state["step"] = torch.tensor(0.0, dtype=torch.float32)
                state["exp_avg"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )
                state["exp_avg_sq"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )
                if group["amsgrad"]:
                    state["max_exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

            exp_avgs.append(state["exp_avg"])
            exp_avg_sqs.append(state["exp_avg_sq"])
            if group["amsgrad"]:
                max_exp_avg_sqs.append(state["max_exp_avg_sq"])
            state_steps.append(state["step"])
        return has_complex

    # ------------------------------------------------------------------
    # Muon branch
    # ------------------------------------------------------------------
    def _muon_step(self, group: MutableMapping) -> None:
        params_with_grad: list[Tensor] = []
        grads: list[Tensor] = []
        momentum_bufs: list[Tensor] = []
        self._init_group_muon(group, params_with_grad, grads, momentum_bufs)
        if not params_with_grad:
            return

        # Flatten 4-D conv kernels to 2-D (out_channels, -1) views.
        view_params = [
            p.view(p.size(0), -1) if p.ndim == 4 else p for p in params_with_grad
        ]
        view_grads = [g.view(g.size(0), -1) if g.ndim == 4 else g for g in grads]
        view_bufs = [m.view(m.size(0), -1) if m.ndim == 4 else m for m in momentum_bufs]

        if _HAS_TORCH_MUON:
            _torch_muon(
                view_params,
                view_grads,
                view_bufs,
                lr=group["lr"],
                weight_decay=group["muon_weight_decay"],
                momentum=group["momentum"],
                nesterov=group["nesterov"],
                ns_coefficients=group["ns_coefficients"],
                eps=group["muon_eps"],
                ns_steps=group["ns_steps"],
                adjust_lr_fn=group["adjust_lr_fn"],
                has_complex=False,
                foreach=group["foreach"],
            )
            return

        # ---- Inline fallback (no torch.optim._muon available) ----
        lr = group["lr"]
        momentum = group["momentum"]
        nesterov = group["nesterov"]
        weight_decay = group["muon_weight_decay"]
        for p, g, buf in zip(view_params, view_grads, view_bufs):
            buf.mul_(momentum).add_(g)
            update = g.add(buf, alpha=momentum) if nesterov else buf
            update = zeropower_via_newtonschulz5(
                update,
                ns_coefficients=group["ns_coefficients"],
                ns_steps=group["ns_steps"],
                eps=group["muon_eps"],
            )
            # Scale so the update RMS is comparable across shapes.
            scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
            if weight_decay != 0:
                p.mul_(1 - lr * weight_decay)
            p.add_(update, alpha=-lr * scale)

    # ------------------------------------------------------------------
    # Adam branch
    # ------------------------------------------------------------------
    def _adam_step(self, group: MutableMapping) -> None:
        params_with_grad: list[Tensor] = []
        grads: list[Tensor] = []
        exp_avgs: list[Tensor] = []
        exp_avg_sqs: list[Tensor] = []
        max_exp_avg_sqs: list[Tensor] = []
        state_steps: list[Tensor] = []
        beta1, beta2 = group["betas"]

        has_complex = self._init_group_adam(
            group,
            params_with_grad,
            grads,
            exp_avgs,
            exp_avg_sqs,
            max_exp_avg_sqs,
            state_steps,
        )
        if not params_with_grad:
            return

        if _HAS_TORCH_ADAM:
            _torch_adam(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                amsgrad=group["amsgrad"],
                has_complex=has_complex,
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["adam_weight_decay"],
                eps=group["adam_eps"],
                maximize=group["maximize"],
                foreach=group["foreach"],
                capturable=False,
                differentiable=False,
                fused=None,
                grad_scale=None,
                found_inf=None,
                decoupled_weight_decay=group["decoupled_weight_decay"],
            )
            return

        # ---- Inline fallback Adam ----
        lr = group["lr"]
        eps = group["adam_eps"]
        weight_decay = group["adam_weight_decay"]
        decoupled = group["decoupled_weight_decay"]
        amsgrad = group["amsgrad"]
        for i, p in enumerate(params_with_grad):
            g = grads[i]
            if group["maximize"]:
                g = -g
            step_t = state_steps[i]
            step_t += 1
            step = step_t.item()

            if weight_decay != 0:
                if decoupled:
                    p.mul_(1 - lr * weight_decay)
                else:
                    g = g.add(p, alpha=weight_decay)

            exp_avgs[i].mul_(beta1).add_(g, alpha=1 - beta1)
            exp_avg_sqs[i].mul_(beta2).addcmul_(g, g, value=1 - beta2)

            bias_c1 = 1 - beta1**step
            bias_c2 = 1 - beta2**step
            if amsgrad:
                torch.maximum(
                    max_exp_avg_sqs[i], exp_avg_sqs[i], out=max_exp_avg_sqs[i]
                )
                denom = (max_exp_avg_sqs[i].sqrt() / (bias_c2**0.5)).add_(eps)
            else:
                denom = (exp_avg_sqs[i].sqrt() / (bias_c2**0.5)).add_(eps)

            p.addcdiv_(exp_avgs[i], denom, value=-lr / bias_c1)

    @torch.no_grad()
    def step(self, closure=None):  # noqa: D102
        r"""Perform a single optimization step.

        Parameters
        ----------
        closure : callable, optional
            A closure that reevaluates the model and returns the loss.

        Returns
        -------
        torch.Tensor or None
            The loss returned by ``closure`` if provided, otherwise ``None``.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if not group["params"]:
                continue
            if group["is_muon"]:
                self._muon_step(group)
            else:
                self._adam_step(group)

        return loss
