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

"""Adaptive loss weighting for multi-term physics-informed training."""

from __future__ import annotations

from typing import Dict, Union

import torch

__all__ = ["BalancedResidualDecayRate"]


class BalancedResidualDecayRate(torch.nn.Module):
    r"""Balanced Residual Decay Rate (BRDR) adaptive loss weighting.

    A self-adaptive scheme that rebalances the per-term weights of a
    multi-term physics-informed loss so that every residual decays at a
    comparable rate during training.  It is a cheap, gradient-free
    alternative to NTK-based balancing: instead of forming a Gram matrix,
    it tracks an exponential moving average (EMA) of the **fourth power**
    of each per-term residual and derives weights from it.

    The per-term inputs ``losses`` are interpreted as squared residuals
    :math:`r_i^2` (the usual mean-squared-error contribution of term
    :math:`i`).  Let :math:`n` be the 1-based update count and
    :math:`\beta_c, \beta_w \in (0, 1)` the EMA decay rates.  Each
    training update performs:

    1. **Fourth-power EMA** of the squared residuals::

           m_i  <-  beta_c * m_i + (1 - beta_c) * (r_i^2)^2

    2. **Bias correction** (de-biasing the EMA toward its steady state)::

           m_hat_i = m_i / (1 - beta_c ** n)

    3. **Inverse residual decay rate (IRDR)** per term::

           irdr_i = r_i^2 / (sqrt(m_hat_i) + eps)

    4. **Normalised raw weights** (mean across terms is one)::

           w_i = irdr_i / (mean_j(irdr_j) + eps)

    5. **Weight EMA + renormalisation** so the weights sum to
       ``num_losses`` (i.e. average to one)::

           v_i  <-  beta_w * v_i + (1 - beta_w) * w_i
           v_i  <-  clamp(v_i, min=eps)
           v_i  <-  v_i * num_losses / (sum_j(v_j) + eps)

    The aggregated loss is the weighted sum :math:`\sum_i v_i\, r_i^2`,
    with ``v`` detached so gradients flow only through the residuals.

    A term whose residual is large relative to its own recent history is
    up-weighted; a term that has already decayed below its historical
    scale is down-weighted.  This equalises the *relative* decay rate of
    all terms rather than their absolute magnitudes, which keeps stiff
    terms from being ignored without letting them dominate.

    The EMA state lives in registered buffers, so the module participates
    correctly in :meth:`state_dict`/:meth:`load_state_dict`, moves with
    :meth:`to`, and respects :meth:`train`/:meth:`eval`.  **The EMA is
    only updated while the module is in training mode.**  Calling
    :meth:`eval` freezes the weights, which is required for composition
    with :class:`physicsnemo.optim.lbfgs_phase.TwoPhaseOptimizer`: that
    optimizer flips the aggregator to ``eval()`` at the Adam->L-BFGS
    boundary so L-BFGS's Wolfe line search sees a stationary loss
    landscape across its repeated closure evaluations.

    Parameters
    ----------
    num_losses : int
        Number of loss terms to balance.  Must be positive.
    beta_c : float, optional
        EMA decay rate for the fourth-power residual statistic, by
        default ``0.999``.  Must lie in the open interval ``(0, 1)``.
    beta_w : float, optional
        EMA decay rate for the per-term weights, by default ``0.999``.
        Must lie in the open interval ``(0, 1)``.
    eps : float, optional
        Small constant guarding all divisions and the lower clamp on the
        weights, by default ``1e-14``.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.optim.loss_weighting import BalancedResidualDecayRate
    >>> brdr = BalancedResidualDecayRate(num_losses=2)
    >>> losses = torch.tensor([1.0, 1e-3])
    >>> total = brdr(losses)  # forward returns the aggregated scalar
    >>> total.ndim
    0

    A dict of per-term scalars is also accepted; the key order is
    preserved so the returned weights align with insertion order:

    >>> brdr = BalancedResidualDecayRate(num_losses=2)
    >>> total = brdr({"pde": torch.tensor(1.0), "bc": torch.tensor(1e-3)})
    >>> bool(total > 0)
    True

    Calling :meth:`eval` freezes the EMA so the weights stop moving:

    >>> brdr.eval()
    BalancedResidualDecayRate(num_losses=2, beta_c=0.999, beta_w=0.999)
    >>> frozen = brdr.residual_4th_ema.clone()
    >>> _ = brdr(torch.tensor([5.0, 5.0]))
    >>> bool(torch.equal(frozen, brdr.residual_4th_ema))
    True
    """

    def __init__(
        self,
        num_losses: int,
        beta_c: float = 0.999,
        beta_w: float = 0.999,
        eps: float = 1e-14,
    ) -> None:
        super().__init__()
        if num_losses <= 0:
            raise ValueError(f"num_losses must be positive, got {num_losses}")
        if not 0.0 < beta_c < 1.0:
            raise ValueError(f"beta_c must lie in (0, 1), got {beta_c}")
        if not 0.0 < beta_w < 1.0:
            raise ValueError(f"beta_w must lie in (0, 1), got {beta_w}")

        self.num_losses: int = int(num_losses)
        self.beta_c: float = float(beta_c)
        self.beta_w: float = float(beta_w)
        self.eps: float = float(eps)

        # EMA of the fourth power of each residual (r^2)^2.
        self.register_buffer(
            "residual_4th_ema",
            torch.zeros(self.num_losses, dtype=torch.get_default_dtype()),
        )
        # EMA of the per-term weights (averages to one).
        self.register_buffer(
            "weights_ema", torch.ones(self.num_losses, dtype=torch.get_default_dtype())
        )
        # 1-based count of EMA updates performed so far (for bias correction).
        self.register_buffer("num_updates", torch.zeros((), dtype=torch.long))

    def _stack(
        self, losses: Union[torch.Tensor, Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """Coerce per-term losses into a 1D tensor, preserving dict order."""
        if isinstance(losses, dict):
            values = list(losses.values())
            if not values:
                raise ValueError("losses dict is empty")
            stacked = torch.stack([v.reshape(()) for v in values])
        else:
            stacked = losses.reshape(-1)

        if stacked.shape[0] != self.num_losses:
            raise ValueError(
                f"expected {self.num_losses} loss terms, got {stacked.shape[0]}"
            )
        return stacked

    def weights(self, losses: torch.Tensor) -> torch.Tensor:
        r"""Return the current per-term weights for the given losses.

        When the module is in training mode (:attr:`training` is ``True``)
        the EMA state is updated in-place using ``losses`` before the
        weights are returned.  In evaluation mode the EMA is left
        untouched and the most recently computed weights are returned.

        Parameters
        ----------
        losses : torch.Tensor
            1D tensor of per-term scalar losses (squared residuals) of
            length ``num_losses``.

        Returns
        -------
        torch.Tensor
            Detached 1D tensor of per-term weights of length
            ``num_losses``.  The weights average to one across terms.
        """
        residuals_squared = self._stack(losses)
        residuals_squared = torch.clamp(residuals_squared, min=0.0)

        if not self.training:
            # Frozen: return the standing weights without touching the EMA.
            return self.weights_ema.detach().clone()

        with torch.no_grad():
            residuals_squared = residuals_squared.detach().to(self.residual_4th_ema)
            n = int(self.num_updates.item()) + 1
            residual_4th = residuals_squared**2

            if self.num_updates.item() == 0:
                # Seed the fourth-power EMA on the very first update and emit
                # uniform weights, matching the reference initialisation.
                self.residual_4th_ema.copy_(residual_4th)
                self.num_updates.add_(1)
                return self.weights_ema.detach().clone()

            self.residual_4th_ema.mul_(self.beta_c).add_(
                residual_4th, alpha=(1.0 - self.beta_c)
            )

            # Bias correction toward the steady-state EMA.
            residual_4th_ema = self.residual_4th_ema / (1.0 - self.beta_c**n)

            # Inverse residual decay rate and its mean-normalised weights.
            irdr = residuals_squared / (torch.sqrt(residual_4th_ema) + self.eps)
            raw_weights = irdr / (irdr.mean() + self.eps)

            self.weights_ema.mul_(self.beta_w).add_(
                raw_weights, alpha=(1.0 - self.beta_w)
            )
            self.weights_ema.clamp_(min=self.eps)
            self.weights_ema.mul_(self.num_losses / (self.weights_ema.sum() + self.eps))
            self.num_updates.add_(1)

        return self.weights_ema.detach().clone()

    def forward(
        self,
        losses: Union[torch.Tensor, Dict[str, torch.Tensor]],
        step: int | None = None,
    ) -> torch.Tensor:
        r"""Aggregate per-term losses into a single weighted scalar.

        Parameters
        ----------
        losses : torch.Tensor or dict of str to torch.Tensor
            Per-term scalar losses (squared residuals).  A 1D tensor of
            length ``num_losses`` or a dict mapping names to scalar
            tensors.  For a dict, the insertion order of the keys defines
            the term order and is preserved.
        step : int, optional
            Ignored.  Accepted for API compatibility with the original
            BRDR aggregator signature; the internal update counter is
            tracked automatically and only advances in training mode.

        Returns
        -------
        torch.Tensor
            Scalar (0-dim) tensor: the weighted sum
            :math:`\sum_i v_i\, r_i^2`.  Gradients flow through the
            residuals only; the weights ``v_i`` are detached.
        """
        del step  # tracked internally; kept only for signature compatibility
        residuals_squared = self._stack(losses)
        weights = self.weights(residuals_squared)
        weights = weights.to(residuals_squared)
        return (weights * residuals_squared).sum()

    def extra_repr(self) -> str:
        """Compact representation of the configuration."""
        return (
            f"num_losses={self.num_losses}, beta_c={self.beta_c}, beta_w={self.beta_w}"
        )
