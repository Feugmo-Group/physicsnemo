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

"""Natural-gradient (energy / Gauss-Newton) optimizers for PyTorch PINNs.

The natural-gradient (also called energy or Gauss-Newton) method preconditions
the parameter gradient with a Gram (information) matrix built from the
Jacobian of a user-chosen scalar function ``f(x; theta)`` of the model.

The Gram matrix is

    G_{ij} = integral_Omega  phi_i(x) phi_j(x)  dx ,   phi_i = d f(x; theta) / d theta_i

(approximated as a weighted sum over a fixed set of integration points), and
the natural-gradient direction ``delta`` solves the linear system

    G delta = grad_theta L .

The parameter update is then ``theta <- theta - lr * delta``.  Compared with
plain gradient descent, conditioning the step by ``G^{-1}`` accounts for the
geometry induced by ``f`` and typically converges much faster on least-squares
and PINN objectives.

Two classes are provided:

- :class:`NaturalGradient` assembles the full ``(P, P)`` Gram matrix and solves
  the dense linear system (exact, cost ``O(N * P^2)``).
- :class:`SketchedNaturalGradient` builds a randomized low-rank approximation of
  ``G`` via a two-pass range finder, reducing the per-update cost to
  ``O(N * P * m)`` with sketch size ``m << P``.

Interface
---------
The quantity ``f`` whose Jacobian defines ``G`` is supplied by the caller in one
of two ways:

``output_fn(model_out, xi)`` (loop API)
    Receives the model output at a single point ``xi`` of shape ``(1, d)`` (with
    ``requires_grad=True``) and returns a scalar.  Works without
    :mod:`torch.func`; the Jacobian is computed by ``N`` serial backward passes.

``gram_fn(params_dict, xi)`` (functional API, preferred)
    Receives a parameter dict and a single ``d``-dimensional point ``xi`` and
    returns a scalar.  Vectorization is handled by ``vmap`` + ``jacrev``, which
    is substantially faster on GPU.  Takes priority over ``output_fn`` when
    :mod:`torch.func` is available.
"""

from __future__ import annotations

import math
import warnings
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn

try:
    from torch.func import (
        functional_call as _fc,
    )
    from torch.func import (
        jacrev,
        vmap,
    )

    _HAS_FUNC = True
except ImportError:  # pragma: no cover - torch.func present on torch >= 2.0
    _HAS_FUNC = False


# --------------------------------------------------------------------------- #
# Jacobian computation
# --------------------------------------------------------------------------- #


def _jacobian_loop(
    func: Callable[[torch.Tensor], torch.Tensor],
    params: List[nn.Parameter],
    x: torch.Tensor,
) -> torch.Tensor:
    """(N, P) Jacobian of ``func(x_i)`` w.r.t. trainable params via backward.

    Parameters
    ----------
    func : callable
        ``func(x: (1, d)) -> scalar`` tensor carrying ``grad_fn``.
    params : list of nn.Parameter
        Trainable parameters with ``requires_grad=True``.
    x : torch.Tensor
        Integration points of shape ``(N, d)``.

    Returns
    -------
    torch.Tensor
        Jacobian of shape ``(N, P)`` where ``P = sum(p.numel())``.
    """
    P = sum(p.numel() for p in params)
    N = x.shape[0]
    J = torch.zeros(N, P, dtype=x.dtype, device=x.device)

    for i in range(N):
        for p in params:
            if p.grad is not None:
                p.grad.zero_()

        out = func(x[i : i + 1])
        out.sum().backward(retain_graph=True)

        offset = 0
        for p in params:
            n = p.numel()
            if p.grad is not None:
                J[i, offset : offset + n] = p.grad.reshape(-1).detach()
            offset += n

    for p in params:
        if p.grad is not None:
            p.grad.zero_()

    return J


def _jacobian_func(
    model: nn.Module,
    func: Callable,
    x: torch.Tensor,
) -> torch.Tensor:
    """(N, P) Jacobian via ``vmap`` + ``jacrev`` with a backward-pass fallback.

    Parameters
    ----------
    model : nn.Module
        Model whose parameters define the Jacobian columns.
    func : callable
        ``func(model_out, x_i: (1, d)) -> scalar`` tensor.
    x : torch.Tensor
        Integration points of shape ``(N, d)``.

    Returns
    -------
    torch.Tensor
        Jacobian of shape ``(N, P)``.
    """
    if _HAS_FUNC:
        params_d = {k: v for k, v in model.named_parameters() if v.requires_grad}
        buffers_d = dict(model.named_buffers())

        def f_params(p_dict, x_i):
            return func(
                _fc(model, {**p_dict, **buffers_d}, (x_i.unsqueeze(0),)),
                x_i.unsqueeze(0),
            )

        try:
            J_nested = vmap(jacrev(f_params, argnums=0), in_dims=(None, 0))(params_d, x)
            N = x.shape[0]
            return torch.cat([J_nested[k].reshape(N, -1) for k in params_d], dim=1)
        except Exception:  # noqa: S110 - intentional fallback to the loop path
            pass  # vmap/jacrev unsupported for this model; use backward loop

    params = [p for p in model.parameters() if p.requires_grad]
    return _jacobian_loop(lambda xi: func(model(xi), xi), params, x)


# --------------------------------------------------------------------------- #
# Gram matrix
# --------------------------------------------------------------------------- #


def _jtj_functional(
    model: nn.Module,
    gram_fn: Callable,
    x: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute ``J^T W J`` (no damping) and return ``(JtWJ, J)``.

    ``J[i, :]`` is the gradient of ``gram_fn(params, x[i])`` w.r.t. the model
    parameters, evaluated with ``vmap`` + ``jacrev``.  The raw Jacobian is
    returned alongside the product so callers can reuse it.

    Parameters
    ----------
    model : nn.Module
        Model whose parameters define the Jacobian columns.
    gram_fn : callable
        ``gram_fn(params_dict, xi: (d,)) -> scalar`` tensor.
    x : torch.Tensor
        Integration points of shape ``(N, d)``.
    weights : torch.Tensor, optional
        Integration weights of shape ``(N,)``; defaults to uniform ``1/N``.

    Returns
    -------
    tuple of torch.Tensor
        ``(JtWJ, J)`` of shapes ``(P, P)`` and ``(N, P)``.
    """
    params_d = {k: v for k, v in model.named_parameters() if v.requires_grad}
    param_keys = list(params_d.keys())
    J_dict = vmap(jacrev(gram_fn, argnums=0), in_dims=(None, 0))(params_d, x)
    N = x.shape[0]
    J = torch.cat([J_dict[k].reshape(N, -1) for k in param_keys], dim=1)
    if weights is None:
        w = torch.ones(N, dtype=J.dtype, device=J.device) / N
    else:
        w = weights.to(dtype=J.dtype, device=J.device)
    return (J * w.unsqueeze(1)).T @ J, J


def compute_gram_functional(
    model: nn.Module,
    gram_fn: Callable,
    x: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    damping: float = 1e-4,
    bc_gram_fn: Optional[Callable] = None,
    x_bc: Optional[torch.Tensor] = None,
    lambda_bc: float = 1.0,
) -> torch.Tensor:
    """Assemble the full-residual Gram matrix via ``vmap`` + ``jacrev``.

    The Gram matrix is

        G = J_pde^T W_pde J_pde + lambda_bc * J_bc^T W_bc J_bc + damping * I .

    Including both interior (PDE) and boundary (BC) residual Jacobians aligns
    the natural-gradient metric with the full training loss rather than the
    interior term alone, which keeps the preconditioner sensitive to boundary
    directions.

    Parameters
    ----------
    model : nn.Module
        Model whose parameters define the Jacobian columns.
    gram_fn : callable
        ``gram_fn(params_dict, xi: (d,)) -> scalar`` PDE residual at one point.
    x : torch.Tensor
        Interior integration points of shape ``(N_int, d)``.
    weights : torch.Tensor, optional
        Interior integration weights of shape ``(N_int,)``.
    damping : float, optional
        Tikhonov ridge added to the diagonal.
    bc_gram_fn : callable, optional
        ``bc_gram_fn(params_dict, xi: (d,)) -> scalar`` BC residual at one
        point.  When provided, its Jacobian is added with weight ``lambda_bc``.
    x_bc : torch.Tensor, optional
        Boundary integration points of shape ``(N_bc, d)``.
    lambda_bc : float, optional
        BC loss weight (matching the weight used in the training loss).

    Returns
    -------
    torch.Tensor
        Symmetric positive semi-definite Gram matrix of shape ``(P, P)``.
    """
    if not _HAS_FUNC:
        raise RuntimeError(
            "compute_gram_functional requires torch.func (torch >= 2.0)."
        )

    G, _ = _jtj_functional(model, gram_fn, x, weights)
    P = G.shape[0]

    if bc_gram_fn is not None and x_bc is not None:
        G_bc, _ = _jtj_functional(model, bc_gram_fn, x_bc)
        G = G + lambda_bc * G_bc

    G = G + damping * torch.eye(P, dtype=G.dtype, device=G.device)
    return G


def _jacobian_functional(
    model: nn.Module,
    gram_fn: Callable,
    x: torch.Tensor,
) -> torch.Tensor:
    """(N, P) Jacobian of ``gram_fn`` via ``vmap`` + ``jacrev``.

    Parameters
    ----------
    model : nn.Module
        Model whose parameters define the Jacobian columns.
    gram_fn : callable
        ``gram_fn(params_dict, xi: (d,)) -> scalar`` tensor.
    x : torch.Tensor
        Integration points of shape ``(N, d)``.

    Returns
    -------
    torch.Tensor
        Jacobian of shape ``(N, P)``.
    """
    params_d = {k: v for k, v in model.named_parameters() if v.requires_grad}
    param_keys = list(params_d.keys())
    J_dict = vmap(jacrev(gram_fn, argnums=0), in_dims=(None, 0))(params_d, x)
    N = x.shape[0]
    return torch.cat([J_dict[k].reshape(N, -1) for k in param_keys], dim=1)


def compute_gram(
    func: Callable[[torch.Tensor], torch.Tensor],
    params: List[nn.Parameter],
    x: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    damping: float = 1e-4,
) -> torch.Tensor:
    """Assemble the Gram matrix ``G = J^T W J + damping * I``.

    Parameters
    ----------
    func : callable
        ``func(x_i: (1, d)) -> scalar`` evaluating the quantity whose Jacobian
        defines the inner product (typically a model output or a residual).
    params : list of nn.Parameter
        Trainable parameters with ``requires_grad=True``.
    x : torch.Tensor
        Integration points of shape ``(N, d)``.
    weights : torch.Tensor, optional
        Integration weights of shape ``(N,)``.  Defaults to uniform ``1/N``, so
        ``G`` approximates the expectation ``E[df/dtheta (df/dtheta)^T]``.  Pass
        normalized quadrature weights to approximate ``integral phi phi^T dx``.
    damping : float, optional
        Tikhonov regularization added to the diagonal for invertibility.

    Returns
    -------
    torch.Tensor
        Symmetric positive semi-definite Gram matrix of shape ``(P, P)``.
    """
    J = _jacobian_loop(func, params, x)
    N, P = J.shape

    if weights is None:
        w = torch.ones(N, dtype=J.dtype, device=J.device) / N
    else:
        w = weights.to(dtype=J.dtype, device=J.device)

    G = (J * w.unsqueeze(1)).T @ J
    G = G + damping * torch.eye(P, dtype=J.dtype, device=J.device)
    return G


# --------------------------------------------------------------------------- #
# Flat-parameter utilities
# --------------------------------------------------------------------------- #


def _flat_grad(params: List[nn.Parameter]) -> torch.Tensor:
    """Concatenate the ``.grad`` of every parameter into one flat vector."""
    return torch.cat(
        [
            p.grad.reshape(-1)
            if p.grad is not None
            else torch.zeros(p.numel(), dtype=p.data.dtype, device=p.data.device)
            for p in params
        ]
    )


def _apply_flat(params: List[nn.Parameter], flat: torch.Tensor, lr: float) -> None:
    """Apply an in-place gradient-descent step from a flat update vector."""
    offset = 0
    with torch.no_grad():
        for p in params:
            n = p.numel()
            p.data -= lr * flat[offset : offset + n].reshape(p.shape)
            offset += n


def _solve_gram(G: torch.Tensor, flat_grad: torch.Tensor) -> torch.Tensor:
    """Solve ``G x = flat_grad`` robustly via least squares.

    Uses :func:`torch.linalg.lstsq` (driver ``"gelsd"``) so the solve tolerates
    rank-deficient or ill-conditioned Gram matrices.  If the solution contains
    NaNs the raw gradient is returned instead.
    """
    x = torch.linalg.lstsq(G, flat_grad.unsqueeze(1), driver="gelsd").solution.squeeze(
        1
    )
    if torch.isnan(x).any():
        warnings.warn("NaturalGradient: Gram solve produced NaN; using raw gradient.")
        return flat_grad
    return x


# --------------------------------------------------------------------------- #
# NaturalGradient
# --------------------------------------------------------------------------- #


class NaturalGradient:
    """Full Gram-matrix natural-gradient optimizer for PyTorch models.

    The Gram matrix ``G = integral (df/dtheta) (df/dtheta)^T dx + damping * I``
    is assembled from a user-supplied ``gram_fn`` (functional API) or
    ``output_fn`` (loop API) evaluated at a fixed set of integration points.
    Each step computes the standard gradient of the loss, solves
    ``G delta = grad_theta L``, and updates ``theta <- theta - lr * delta``.

    Parameters
    ----------
    model : nn.Module
        The model whose trainable parameters are optimized.
    x : torch.Tensor
        Integration points of shape ``(N, d)`` (interior, or combined
        interior + boundary).
    weights : torch.Tensor, optional
        Integration weights of shape ``(N,)``; defaults to uniform ``1/N``.
    gram_fn : callable, optional
        Preferred functional API ``gram_fn(params_dict, xi: (d,)) -> scalar``.
        Enables GPU-efficient ``vmap`` + ``jacrev`` Jacobians.  Takes priority
        over ``output_fn`` when :mod:`torch.func` is available.
    output_fn : callable, optional
        Loop API ``output_fn(model_out, x_i: (1, d)) -> scalar`` where ``x_i``
        carries ``requires_grad=True``.  Used when ``gram_fn`` is not given or
        :mod:`torch.func` is unavailable.
    lr : float, optional
        Learning rate for the natural-gradient step.
    damping : float, optional
        Tikhonov regularization added to the Gram diagonal.
    gram_update_freq : int, optional
        Recompute the Gram matrix every this many ``step`` calls; ``1`` gives
        the exact (most expensive) natural gradient.
    bc_gram_fn : callable, optional
        Boundary residual ``bc_gram_fn(params_dict, xi: (d,)) -> scalar`` whose
        Jacobian is added to ``G`` (functional path only).
    x_bc : torch.Tensor, optional
        Boundary integration points of shape ``(N_bc, d)``.
    lambda_bc : float, optional
        Weight applied to the boundary contribution of ``G``.
    """

    def __init__(
        self,
        model: nn.Module,
        x: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        gram_fn: Optional[Callable] = None,
        output_fn: Optional[Callable] = None,
        lr: float = 1e-3,
        damping: float = 1e-4,
        gram_update_freq: int = 1,
        bc_gram_fn: Optional[Callable] = None,
        x_bc: Optional[torch.Tensor] = None,
        lambda_bc: float = 1.0,
    ):
        self.model = model
        self.x = x
        self.weights = weights
        self.gram_fn = gram_fn
        self.output_fn = output_fn
        self.lr = lr
        self.damping = damping
        self.gram_update_freq = gram_update_freq
        self.bc_gram_fn = bc_gram_fn
        self.x_bc = x_bc
        self.lambda_bc = lambda_bc

        self._params: List[nn.Parameter] = [
            p for p in model.parameters() if p.requires_grad
        ]
        self._gram: Optional[torch.Tensor] = None
        self._step_count = 0

    def _build_func(self):
        """Return the scalar function whose Jacobian defines the Gram matrix.

        ``xi`` is detached and re-enabled for gradients so that ``output_fn``
        may differentiate through the input (e.g. to form a PDE residual).
        """
        model = self.model
        ofn = self.output_fn

        if ofn is None:

            def func(xi):
                xi = xi.detach().requires_grad_(True)
                return model(xi).sum()
        else:

            def func(xi):
                xi = xi.detach().requires_grad_(True)
                u = model(xi)
                return ofn(u, xi).sum()

        return func

    def _update_gram(self) -> None:
        """Recompute and cache the Gram matrix on the current parameters."""
        was_training = self.model.training
        self.model.eval()
        with torch.enable_grad():
            if self.gram_fn is not None and _HAS_FUNC:
                G, _ = _jtj_functional(self.model, self.gram_fn, self.x, self.weights)
                if self.bc_gram_fn is not None and self.x_bc is not None:
                    G_bc, _ = _jtj_functional(self.model, self.bc_gram_fn, self.x_bc)
                    G = G + self.lambda_bc * G_bc
                P = G.shape[0]
                self._gram = G + self.damping * torch.eye(
                    P, dtype=G.dtype, device=G.device
                )
            else:
                func = self._build_func()
                self._gram = compute_gram(
                    func, self._params, self.x, self.weights, self.damping
                )
        if was_training:
            self.model.train()

    @property
    def gram(self) -> Optional[torch.Tensor]:
        """The currently cached Gram matrix, or ``None`` if not yet computed."""
        return self._gram

    def zero_grad(self) -> None:
        """Zero the ``.grad`` of every trainable parameter."""
        for p in self._params:
            if p.grad is not None:
                p.grad.zero_()

    def step(self, loss: torch.Tensor, max_norm: Optional[float] = None) -> None:
        """Compute the natural gradient from ``loss`` and update parameters.

        Parameters
        ----------
        loss : torch.Tensor
            Scalar loss with ``requires_grad=True``; ``loss.backward()`` is
            called internally.
        max_norm : float, optional
            Clip the natural-gradient vector to this L2 norm before the update.
            ``None`` disables clipping.
        """
        for p in self._params:
            if p.grad is not None:
                p.grad.zero_()
        loss.backward()
        flat_g = _flat_grad(self._params)

        if self._gram is None or self._step_count % self.gram_update_freq == 0:
            self._update_gram()

        flat_nat = _solve_gram(self._gram, flat_g)
        if max_norm is not None:
            nat_norm = flat_nat.norm()
            if nat_norm > max_norm:
                flat_nat = flat_nat * (max_norm / nat_norm)

        _apply_flat(self._params, flat_nat, self.lr)
        self._step_count += 1

    @property
    def current_step(self) -> int:
        """Number of ``step`` calls performed so far."""
        return self._step_count


# --------------------------------------------------------------------------- #
# SketchedNaturalGradient
# --------------------------------------------------------------------------- #


class SketchedNaturalGradient:
    """Randomized (two-pass) sketched natural-gradient optimizer.

    Approximates ``G`` by a rank-``m`` factorization ``G ~ U Lambda U^T`` built
    with a randomized range finder, reducing the per-update cost from
    ``O(N * P^2)`` to ``O(N * P * m)`` with sketch size ``m << P``.

    The two-pass scheme is:

    1. Draw a random sketch ``Omega`` of shape ``(P, m)``.
    2. Form ``Y = G Omega`` (matrix-vector products with ``G``).
    3. QR-factor ``Y = Q R`` for an orthonormal basis ``Q`` of shape ``(P, m)``.
    4. Build the reduced matrix ``T = Q^T G Q`` of shape ``(m, m)``.
    5. Eigen-decompose ``T = S Lambda S^T``.
    6. Approximate ``G^{-1} ~ (Q S) diag(1 / max(lambda, damping)) (Q S)^T``.

    Parameters
    ----------
    model : nn.Module
        The model whose trainable parameters are optimized.
    x : torch.Tensor
        Integration points of shape ``(N, d)``.
    weights : torch.Tensor, optional
        Integration weights of shape ``(N,)``; defaults to uniform ``1/N``.
    gram_fn : callable, optional
        Functional API ``gram_fn(params_dict, xi: (d,)) -> scalar`` enabling
        GPU-efficient ``vmap`` + ``jacrev`` Jacobians.  The Jacobian is computed
        once per sketch update and reused for both range-finder passes.
    output_fn : callable, optional
        Loop API ``output_fn(model_out, x_i: (1, d)) -> scalar`` (same contract
        as :class:`NaturalGradient`).
    lr : float, optional
        Learning rate for the natural-gradient step.
    damping : float, optional
        Eigenvalue floor; eigenvalues at or below this value are dropped to
        prevent ``1 / lambda`` blow-up.
    sketch_size : int, optional
        Rank ``m`` of the low-rank approximation.
    gram_update_freq : int, optional
        Recompute the sketch every this many ``step`` calls.
    """

    def __init__(
        self,
        model: nn.Module,
        x: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        gram_fn: Optional[Callable] = None,
        output_fn: Optional[Callable] = None,
        lr: float = 1e-3,
        damping: float = 1e-4,
        sketch_size: int = 50,
        gram_update_freq: int = 10,
    ):
        self.model = model
        self.x = x
        self.weights = weights
        self.gram_fn = gram_fn
        self.output_fn = output_fn
        self.lr = lr
        self.damping = damping
        self.sketch_size = sketch_size
        self.gram_update_freq = gram_update_freq
        self.bc_gram_fn = None
        self.x_bc = None
        self.lambda_bc = 1.0

        self._params: List[nn.Parameter] = [
            p for p in model.parameters() if p.requires_grad
        ]
        self._P = sum(p.numel() for p in self._params)

        self._U: Optional[torch.Tensor] = None
        self._eigv: Optional[torch.Tensor] = None
        self._step_count = 0

    def _gram_apply(self, V: torch.Tensor) -> torch.Tensor:
        """Compute ``G V`` for ``V`` of shape ``(P, m)`` without forming ``G``.

        Uses ``G V = J^T (W (J V))`` with the loop-computed Jacobian ``J``.
        """
        func = self._build_func()
        J = _jacobian_loop(func, self._params, self.x)
        N = J.shape[0]

        w = (
            self.weights
            if self.weights is not None
            else torch.ones(N, dtype=J.dtype, device=J.device) / N
        )

        JV = J @ V
        WJV = JV * w.unsqueeze(1)
        return J.T @ WJV

    def _build_func(self):
        """Return the scalar function whose Jacobian defines the Gram matrix."""
        model = self.model
        ofn = self.output_fn

        if ofn is None:

            def func(xi):
                xi = xi.detach().requires_grad_(True)
                return model(xi).sum()
        else:

            def func(xi):
                xi = xi.detach().requires_grad_(True)
                return ofn(model(xi), xi).sum()

        return func

    def _update_sketch(self) -> None:
        """Recompute and cache the low-rank sketch factors ``U`` and ``Lambda``."""
        P, m = self._P, self.sketch_size
        dtype, device = self._params[0].dtype, self._params[0].device

        Omega = torch.randn(P, m, dtype=dtype, device=device) / math.sqrt(m)

        was_training = self.model.training
        self.model.eval()
        with torch.enable_grad():
            if self.gram_fn is not None and _HAS_FUNC:
                J = _jacobian_functional(self.model, self.gram_fn, self.x)
                N = J.shape[0]
                w = (
                    self.weights
                    if self.weights is not None
                    else torch.ones(N, dtype=J.dtype, device=J.device) / N
                )

                if self.bc_gram_fn is not None and self.x_bc is not None:
                    J_bc = _jacobian_functional(self.model, self.bc_gram_fn, self.x_bc)
                    N_bc = J_bc.shape[0]
                    w_bc = torch.ones(N_bc, dtype=J.dtype, device=J.device) / N_bc
                    lbc = self.lambda_bc
                else:
                    J_bc = w_bc = lbc = None

                def _gv(V: torch.Tensor) -> torch.Tensor:
                    JV = J @ V
                    r = J.T @ (JV * w.unsqueeze(1))
                    if J_bc is not None:
                        r = r + lbc * J_bc.T @ (J_bc @ V * w_bc.unsqueeze(1))
                    return r

                Y = _gv(Omega)
                Q, _ = torch.linalg.qr(Y)
                T = _gv(Q).T @ Q
            else:
                Y = self._gram_apply(Omega)
                Q, _ = torch.linalg.qr(Y)
                T = self._gram_apply(Q).T @ Q
        if was_training:
            self.model.train()

        eigenvalues, S = torch.linalg.eigh(T)
        self._U = Q @ S
        self._eigv = eigenvalues

    def _apply_nat_grad(self, flat_g: torch.Tensor) -> torch.Tensor:
        """Return ``G^{-1} flat_g`` using the cached low-rank approximation."""
        U, lam = self._U, self._eigv
        inv_lam = torch.where(
            lam > self.damping,
            1.0 / lam,
            torch.zeros_like(lam),
        )
        Utg = U.T @ flat_g
        return U @ (inv_lam * Utg)

    def zero_grad(self) -> None:
        """Zero the ``.grad`` of every trainable parameter."""
        for p in self._params:
            if p.grad is not None:
                p.grad.zero_()

    def step(self, loss: torch.Tensor, max_norm: Optional[float] = None) -> None:
        """Compute the sketched natural gradient and update parameters.

        Parameters
        ----------
        loss : torch.Tensor
            Scalar loss with ``requires_grad=True``; ``loss.backward()`` is
            called internally.
        max_norm : float, optional
            Clip the natural-gradient vector to this L2 norm before the update.
        """
        for p in self._params:
            if p.grad is not None:
                p.grad.zero_()
        loss.backward()
        flat_g = _flat_grad(self._params)

        if self._U is None or self._step_count % self.gram_update_freq == 0:
            self._update_sketch()

        flat_nat = self._apply_nat_grad(flat_g)
        if torch.isnan(flat_nat).any():
            warnings.warn(
                "SketchedNaturalGradient: NaN in natural gradient; using raw gradient."
            )
            flat_nat = flat_g

        if max_norm is not None:
            nat_norm = flat_nat.norm()
            if nat_norm > max_norm:
                flat_nat = flat_nat * (max_norm / nat_norm)

        _apply_flat(self._params, flat_nat, self.lr)
        self._step_count += 1

    @property
    def current_step(self) -> int:
        """Number of ``step`` calls performed so far."""
        return self._step_count

    @property
    def effective_rank(self) -> Optional[int]:
        """Number of cached eigenvalues above the damping threshold."""
        if self._eigv is None:
            return None
        return int((self._eigv > self.damping).sum().item())
