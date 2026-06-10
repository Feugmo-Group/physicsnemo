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

"""Adaptive multi-task loss aggregators for physics-informed training.

A suite of strategies for combining several per-term losses (PDE residual,
boundary conditions, interface conditions, …) into a single scalar that can
be back-propagated.  Every aggregator implements the common interface::

    forward(losses: Dict[str, Tensor], step: int) -> Tensor

and exposes a ``current_weights`` property for logging.  Stateful schemes
update their internal balancing state only while in training mode, so they
compose safely with optimizers that flip the aggregator to ``eval()`` and
evaluate the loss closure multiple times per step (e.g. L-BFGS line search).
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn


class Aggregator(nn.Module):
    """Base class for multi-task loss aggregators.

    All subclasses implement::

        forward(losses: Dict[str, Tensor], step: int) -> Tensor

    The ``losses`` dict keys are the loss component names in a fixed order
    (e.g. ``{"pde": ..., "bc": ..., "c0": ..., "c1": ...}``).
    ``weights`` provides the initial per-term multipliers (e.g. lambda_bc=100).
    """

    def __init__(
        self,
        params: List[nn.Parameter],
        num_losses: int,
        weights: Optional[List[float]] = None,
        **kwargs,
    ):
        super().__init__()
        self.params = params
        self.num_losses = num_losses
        self.weights = weights if weights is not None else [1.0] * num_losses

    def forward(self, losses: Dict[str, torch.Tensor], step: int) -> torch.Tensor:
        raise NotImplementedError

    @property
    def current_weights(self) -> Optional[List[float]]:
        """Return current effective per-loss weights for logging (or None)."""
        return None


# ── Aggregators ───────────────────────────────────────────────────────────────


class Sum(Aggregator):
    """Fixed weighted sum: total = sum_i  weights[i] * losses[i].

    This is the default baseline — no dynamic rebalancing.
    """

    def forward(self, losses: Dict[str, torch.Tensor], step: int) -> torch.Tensor:
        vals = list(losses.values())
        return sum(self.weights[i] * vals[i] for i in range(self.num_losses))


class GradNorm(Aggregator):
    """Gradient normalisation (GradNorm).

    Maintains learnable per-task weights whose gradient norms are kept equal.
    ``alpha`` controls how aggressively imbalanced training rates are corrected
    (0 = equal gradient norms; higher = faster-learning tasks penalised more).

    For each task ``i`` the gradient norm of its weighted loss w.r.t. the model
    parameters is computed; a target norm is formed from the mean gradient norm
    scaled by the relative inverse training rate ``(r_i / mean(r))**alpha``,
    where ``r_i = L_i(t) / L_i(0)``.  The ``task_weights`` are updated to drive
    each gradient norm toward this target.

    The internal ``task_weights`` are updated via a separate backward pass
    inside ``forward()``, orthogonal to the main network optimiser.
    """

    def __init__(
        self,
        params: List[nn.Parameter],
        num_losses: int,
        weights: Optional[List[float]] = None,
        alpha: float = 0.1,
        **kwargs,
    ):
        super().__init__(params, num_losses, weights)
        self.alpha = alpha
        self.task_weights = nn.Parameter(
            torch.ones(num_losses, dtype=torch.float32), requires_grad=True
        )
        self._w_opt = torch.optim.Adam([self.task_weights], lr=1e-2)
        self._initial_losses: Optional[torch.Tensor] = None

    @property
    def current_weights(self) -> Optional[List[float]]:
        """Return the softmax-normalised learnable task weights."""
        w = torch.softmax(self.task_weights.detach(), dim=0) * self.num_losses
        return w.tolist()

    def forward(self, losses: Dict[str, torch.Tensor], step: int) -> torch.Tensor:
        vals = list(losses.values())
        L = torch.stack([v.detach().float() for v in vals])

        if self._initial_losses is None:
            self._initial_losses = L.clone() + 1e-30

        w = torch.softmax(self.task_weights, dim=0) * self.num_losses
        weighted = sum(self.weights[i] * w[i] * vals[i] for i in range(self.num_losses))

        if self.training:
            all_p = [p for p in self.params if p.requires_grad]
            G = []
            for i, v in enumerate(vals):
                g = torch.autograd.grad(
                    self.weights[i] * w[i] * v,
                    all_p,
                    retain_graph=True,
                    allow_unused=True,
                    create_graph=True,
                )
                g_flat = torch.cat(
                    [
                        gi.reshape(-1) if gi is not None else v.new_zeros(p.numel())
                        for gi, p in zip(g, all_p)
                    ]
                )
                G.append(g_flat.norm())
            G = torch.stack(G)
            G_bar = G.mean().detach()
            r = L.detach() / self._initial_losses
            target = (G_bar * (r / r.mean().clamp(min=1e-30)) ** self.alpha).detach()
            gn_loss = (G - target).abs().sum()
            self._w_opt.zero_grad()
            gn_loss.backward(inputs=[self.task_weights], retain_graph=True)
            self._w_opt.step()

        return weighted


class ResNorm(Aggregator):
    """Residual normalisation.

    Each loss is scaled so all terms contribute roughly the same order of
    magnitude as the first loss (the PDE term).

    Uses an exponential moving average of each loss value to estimate its
    current scale.  Cheaper than GradNorm (no gradient computations).
    """

    def __init__(
        self,
        params: List[nn.Parameter],
        num_losses: int,
        weights: Optional[List[float]] = None,
        ema_alpha: float = 0.99,
        clamp_lo: float = 1e-3,
        clamp_hi: float = 1e3,
        **kwargs,
    ):
        super().__init__(params, num_losses, weights)
        self.ema_alpha = ema_alpha
        self.clamp_lo = clamp_lo
        self.clamp_hi = clamp_hi
        self._ema: Optional[List[float]] = None
        self._w: List[float] = list(self.weights)

    @property
    def current_weights(self) -> Optional[List[float]]:
        """Return the current per-term scaling weights."""
        return list(self._w)

    def forward(self, losses: Dict[str, torch.Tensor], step: int) -> torch.Tensor:
        vals = list(losses.values())
        curr = [float(v.detach()) + 1e-30 for v in vals]

        # Update EMA and weights only during training — keeping the state frozen
        # in eval mode lets L-BFGS line search see a stationary loss landscape.
        if self.training:
            if self._ema is None:
                self._ema = list(curr)
            else:
                a = self.ema_alpha
                self._ema = [a * e + (1 - a) * c for e, c in zip(self._ema, curr)]

            ref = self._ema[0]
            self._w = [
                max(min(ref / (self._ema[i] + 1e-30), self.clamp_hi), self.clamp_lo)
                * self.weights[i]
                for i in range(self.num_losses)
            ]

        return sum(self._w[i] * vals[i] for i in range(self.num_losses))


class HomoscedasticUncertainty(Aggregator):
    """Homoscedastic uncertainty weighting.

    Learns a log-variance ``log_sigma_i`` for each task::

        total = sum_i  weights[i] * 0.5 * exp(-2*log_sigma_i) * L_i + log_sigma_i

    The ``log_sigma`` parameters must be included in the main optimiser —
    ``build_aggregator()`` does this automatically when ``include_in_opt=True``
    (default).
    """

    def __init__(
        self,
        params: List[nn.Parameter],
        num_losses: int,
        weights: Optional[List[float]] = None,
        **kwargs,
    ):
        super().__init__(params, num_losses, weights)
        self.log_sigma = nn.Parameter(torch.zeros(num_losses, dtype=torch.float32))

    @property
    def current_weights(self) -> Optional[List[float]]:
        """Return the precision weights ``0.5 * exp(-2*log_sigma_i)``."""
        return (0.5 * torch.exp(-2.0 * self.log_sigma.detach())).tolist()

    def forward(self, losses: Dict[str, torch.Tensor], step: int) -> torch.Tensor:
        vals = list(losses.values())
        total = sum(
            self.weights[i] * 0.5 * torch.exp(-2.0 * self.log_sigma[i]) * vals[i]
            + self.log_sigma[i]
            for i in range(self.num_losses)
        )
        return total


class LRAnnealing(Aggregator):
    """Learning-rate annealing.

    Scales lambda_i based on the ratio of the mean absolute gradient of the
    reference loss (the PDE term, index 0) to that of each individual loss::

        lambda_i ← EMA( mean|grad L_pde| / mean|grad L_i| * weights[i] )

    Uses gradient computations → expensive; use ``update_freq > 1`` to amortise.
    """

    def __init__(
        self,
        params: List[nn.Parameter],
        num_losses: int,
        weights: Optional[List[float]] = None,
        ema_alpha: float = 0.9,
        update_freq: int = 1,
        **kwargs,
    ):
        super().__init__(params, num_losses, weights)
        self.ema_alpha = ema_alpha
        self.update_freq = update_freq
        self._lambdas: List[float] = list(self.weights)

    @property
    def current_weights(self) -> Optional[List[float]]:
        """Return the current per-term lambda multipliers."""
        return list(self._lambdas)

    def _mean_abs_grad(self, loss: torch.Tensor) -> float:
        """Return the mean absolute gradient of ``loss`` w.r.t. the parameters."""
        # A loss with no grad_fn is a constant (e.g. zero interface loss on a
        # single-element grid).  Its gradient w.r.t. parameters is exactly 0.
        if loss.grad_fn is None:
            return 0.0
        all_p = [p for p in self.params if p.requires_grad]
        if not all_p:
            return 0.0
        grads = torch.autograd.grad(loss, all_p, retain_graph=True, allow_unused=True)
        total = sum(g.abs().mean().item() for g in grads if g is not None)
        count = sum(1 for g in grads if g is not None)
        return total / max(count, 1)

    def forward(self, losses: Dict[str, torch.Tensor], step: int) -> torch.Tensor:
        vals = list(losses.values())

        if self.training and step % self.update_freq == 0:
            ref_grad = self._mean_abs_grad(vals[0]) + 1e-30
            new_l = []
            for i, v in enumerate(vals):
                g = self._mean_abs_grad(v)
                if g == 0.0:
                    # Constant loss (no grad_fn) — keep its current weight rather
                    # than dividing by zero and producing an infinite lambda.
                    new_l.append(self._lambdas[i])
                else:
                    new_l.append(ref_grad / (g + 1e-30) * self.weights[i])
            a = self.ema_alpha
            self._lambdas = [
                a * old + (1 - a) * new for old, new in zip(self._lambdas, new_l)
            ]

        return sum(self._lambdas[i] * vals[i] for i in range(self.num_losses))


class SoftAdapt(Aggregator):
    """SoftAdapt loss balancing.

    Normalises weights by the softmax of the rate of change of each loss,
    so terms that are improving fastest receive less attention::

        w_i = softmax(beta * dL_i / L_i)

    ``beta`` controls sharpness (higher → more emphasis on poorly-improving terms).
    """

    def __init__(
        self,
        params: List[nn.Parameter],
        num_losses: int,
        weights: Optional[List[float]] = None,
        beta: float = 0.1,
        epsilon: float = 1e-7,
        update_freq: int = 1,
        **kwargs,
    ):
        super().__init__(params, num_losses, weights)
        self.beta = beta
        self.epsilon = epsilon
        self.update_freq = update_freq
        self._prev: Optional[List[float]] = None
        self._w: List[float] = [1.0 / num_losses] * num_losses

    @property
    def current_weights(self) -> Optional[List[float]]:
        """Return the current softmax weights scaled by the fixed multipliers."""
        return [self._w[i] * self.weights[i] for i in range(self.num_losses)]

    def forward(self, losses: Dict[str, torch.Tensor], step: int) -> torch.Tensor:
        vals = list(losses.values())
        curr = [float(v.detach()) for v in vals]

        # Only advance the rate-of-change state during training so the weights
        # stay frozen when the aggregator is in eval mode (L-BFGS line search).
        if self.training:
            if self._prev is not None and step % self.update_freq == 0:
                rates = [
                    (c - p) / (abs(p) + self.epsilon) for c, p in zip(curr, self._prev)
                ]
                r_t = torch.tensor(rates, dtype=torch.float32)
                self._w = torch.softmax(self.beta * r_t, dim=0).tolist()

            self._prev = curr

        total_w = sum(self._w) + 1e-30
        return sum(
            (self._w[i] / total_w) * self.weights[i] * vals[i]
            for i in range(self.num_losses)
        )


class Relobralo(Aggregator):
    """ReLoBRaLo: Relative Loss Balancing Residual Algorithm.

    Tracks the relative progress of each loss from its initial value and
    uses it to balance the contribution of each term::

        lambda_hat_i(t) = (L_i(0)/L_i(t)) / mean_j(L_j(0)/L_j(t)) * weights[i]
        lambda_i(t)     = rho * lambda_i(t-1) + (1-rho) * lambda_hat_i(t)

    ``rho`` is the EMA decay (high values = slow adaptation, stable weights).
    """

    def __init__(
        self,
        params: List[nn.Parameter],
        num_losses: int,
        weights: Optional[List[float]] = None,
        rho: float = 0.999,
        epsilon: float = 1e-7,
        **kwargs,
    ):
        super().__init__(params, num_losses, weights)
        self.rho = rho
        self.epsilon = epsilon
        self._initial: Optional[List[float]] = None
        self._lambdas: List[float] = list(self.weights)

    @property
    def current_weights(self) -> Optional[List[float]]:
        """Return the current per-term lambda multipliers."""
        return list(self._lambdas)

    def forward(self, losses: Dict[str, torch.Tensor], step: int) -> torch.Tensor:
        vals = list(losses.values())
        curr = [float(v.detach()) + self.epsilon for v in vals]

        if self._initial is None:
            self._initial = list(curr)

        if self.training:
            ratios = [L0 / Lc for L0, Lc in zip(self._initial, curr)]
            mean_r = sum(ratios) / len(ratios) + self.epsilon
            lhat = [
                self.weights[i] * ratios[i] / mean_r for i in range(self.num_losses)
            ]
            rho = self.rho
            self._lambdas = [
                rho * old + (1 - rho) * new for old, new in zip(self._lambdas, lhat)
            ]

        return sum(self._lambdas[i] * vals[i] for i in range(self.num_losses))


class EMA(Aggregator):
    """EMA reweighting.

    Each loss is scaled so it contributes roughly the same order of magnitude
    as the first loss (the PDE term).  Uses an exponential moving average
    (decay=``ema_alpha``) of each raw loss value to estimate its current scale::

        w_i = clamp( ema_pde / ema_i, clamp_lo, clamp_hi ) * weights[i]

    Lightweight: no gradient computations required.  Good starting choice before
    committing to gradient-based aggregators.
    """

    def __init__(
        self,
        params: List[nn.Parameter],
        num_losses: int,
        weights: Optional[List[float]] = None,
        ema_alpha: float = 0.99,
        clamp_lo: float = 1e-3,
        clamp_hi: float = 1e3,
        **kwargs,
    ):
        super().__init__(params, num_losses, weights)
        self.ema_alpha = ema_alpha
        self.clamp_lo = clamp_lo
        self.clamp_hi = clamp_hi
        self._ema: Optional[List[float]] = None
        self._w: List[float] = list(self.weights)

    @property
    def current_weights(self) -> Optional[List[float]]:
        """Return the current per-term scaling weights."""
        return list(self._w)

    def forward(self, losses: Dict[str, torch.Tensor], step: int) -> torch.Tensor:
        vals = list(losses.values())
        curr = [float(v.detach()) + 1e-30 for v in vals]

        # Freeze the EMA and weights outside training mode so L-BFGS line search
        # evaluates a stationary loss landscape.
        if self.training:
            if self._ema is None:
                self._ema = list(curr)
            else:
                a = self.ema_alpha
                self._ema = [a * e + (1 - a) * c for e, c in zip(self._ema, curr)]

            ref = self._ema[0]
            self._w = [
                max(min(ref / (self._ema[i] + 1e-30), self.clamp_hi), self.clamp_lo)
                * self.weights[i]
                for i in range(self.num_losses)
            ]

        return sum(self._w[i] * vals[i] for i in range(self.num_losses))


class BalancedResidualDecayRate(Aggregator):
    r"""Balanced Residual Decay Rate (BRDR) adaptive weighting.

    Rebalances the per-term weights so that every residual decays at a
    comparable *relative* rate during training.  A term whose loss is large
    relative to its own recent history is up-weighted; a term that has already
    decayed below its historical scale is down-weighted.

    The weight for loss_i at step n is::

        irdr_i  = L_i / sqrt( EMA(L_i²) )          # inverse relative decay rate
        w_i     = irdr_i / mean(irdr)               # normalised
        λ_i(n)  = EMA_w( w_i )  *  num_losses / sum(EMA_w)   # bias-corrected

    where L_i is the fixed-weight-scaled loss (``weights[i] * loss_i``).
    At step 0 the fixed weights are used unchanged.

    Parameters
    ----------
    beta_c : float
        EMA decay for the 4th-moment residual tracker (default 0.999).
    beta_w : float
        EMA decay for the weight smoother (default 0.999).
    eps : float
        Small constant for numerical stability (default 1e-14).
    """

    def __init__(
        self,
        params: List[nn.Parameter],
        num_losses: int,
        weights: Optional[List[float]] = None,
        beta_c: float = 0.999,
        beta_w: float = 0.999,
        eps: float = 1e-14,
        **kwargs,
    ):
        super().__init__(params, num_losses, weights)
        self.beta_c = beta_c
        self.beta_w = beta_w
        self.eps = eps
        # Persistent EMA buffers — placed on correct device via .to()
        self.register_buffer("residual_4th_ema", torch.zeros(num_losses))
        self.register_buffer("weights_ema", torch.ones(num_losses))

    @property
    def current_weights(self) -> Optional[List[float]]:
        """Return the smoothed adaptive per-term weights."""
        return self.weights_ema.tolist()

    def forward(self, losses: Dict[str, torch.Tensor], step: int) -> torch.Tensor:
        n = step + 1

        # Apply fixed per-term multipliers
        vals = list(losses.values())
        weighted_vals = torch.stack(
            [self.weights[i] * vals[i] for i in range(self.num_losses)]
        )

        # Step 0: initialise EMA and return unmodified weighted sum
        if step == 0:
            with torch.no_grad():
                self.residual_4th_ema.copy_(weighted_vals.detach() ** 2)
            return weighted_vals.sum()

        # Update EMA only during training — L-BFGS calls the closure multiple
        # times per step (line search), so updating unconditionally makes the
        # loss non-stationary and breaks L-BFGS convergence guarantees.
        if self.training:
            with torch.no_grad():
                residual_4th = weighted_vals.detach() ** 2
                self.residual_4th_ema.mul_(self.beta_c).add_(
                    (1.0 - self.beta_c) * residual_4th
                )
                # Bias correction
                r4_bc = self.residual_4th_ema / (1.0 - self.beta_c**n)

                # Inverse relative decay rate → normalised adaptive weights
                irdr = weighted_vals.detach() / (torch.sqrt(r4_bc) + self.eps)
                w_hat = irdr / (irdr.mean() + self.eps)

                # Smooth weights with EMA, then normalise to mean = 1
                self.weights_ema.mul_(self.beta_w).add_((1.0 - self.beta_w) * w_hat)
                self.weights_ema.clamp_(min=self.eps)
                self.weights_ema.mul_(
                    self.num_losses / (self.weights_ema.sum() + self.eps)
                )

        return (self.weights_ema.detach() * weighted_vals).sum()


# ── NTK (standalone, not an Aggregator subclass) ─────────────────────────────


class NTK(nn.Module):
    """Neural Tangent Kernel (NTK) loss balancing.

    Weights each loss by ``ntk_sum / ntk_i`` where::

        ntk_i = || ∂ sqrt(|L_i|) / ∂θ ||₂

    NTK traces are recomputed every ``run_per_step`` steps; between updates
    the cached traces are reused.  At step 0 all weights default to 1.

    Unlike the ``Aggregator`` subclasses, ``NTK`` is not called as
    ``aggregator(losses, step)``; it requires the model to compute gradients::

        ntk = NTK(run_per_step=1000)
        ntk_cache = {}
        for step in range(n_steps):
            losses = compute_losses(model)
            weighted_losses, ntk_cache = ntk(model, losses, ntk_cache, step)
            total = sum(weighted_losses.values())
            total.backward()
            ...

    Parameters
    ----------
    run_per_step : int
        How often (in steps) to recompute the NTK traces (default 1000).
    save_name : str or None
        If set, NTK traces are appended to ``<save_name>.csv`` at each
        recomputation.  Requires ``pandas``.  Incompatible with CUDA graphs.
    """

    def __init__(self, run_per_step: int = 1000, save_name: Optional[str] = None):
        super().__init__()
        self.run_per_step = run_per_step
        self.if_csv_head = True
        self.save_name = save_name
        if save_name:
            import warnings

            warnings.warn(
                "NTK: CUDA graphs are incompatible with save_name; "
                "set cuda_graphs=False.",
                stacklevel=2,
            )

    def _group_ntk(
        self, model: nn.Module, losses: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute per-loss NTK trace proxy: ||∂√|L_i|/∂θ||₂."""
        ntk_values: Dict[str, torch.Tensor] = {}
        for key, loss in losses.items():
            grads = torch.autograd.grad(
                torch.sqrt(torch.abs(loss)),
                model.parameters(),
                retain_graph=True,
                allow_unused=True,
            )
            ntk_values[key] = torch.sqrt(
                torch.sum(
                    torch.stack(
                        [g.detach().pow(2).sum() for g in grads if g is not None]
                    )
                )
            )
        return ntk_values

    def _save_ntk(self, ntk_dict: Dict[str, torch.Tensor], step: int) -> None:
        """Append the NTK traces for ``step`` to ``<save_name>.csv``."""
        try:
            import pandas as pd
        except ImportError:
            print("  NTK: pandas not available — skipping CSV export.")
            return
        df = pd.DataFrame(
            {k: v.cpu().item() for k, v in ntk_dict.items()}, index=[step]
        )
        df.to_csv(self.save_name + ".csv", mode="a", header=self.if_csv_head)
        self.if_csv_head = False

    def forward(
        self,
        model: nn.Module,
        losses: Dict[str, torch.Tensor],
        ntk_cache: Dict[str, torch.Tensor],
        step: int,
    ):
        """Weight ``losses`` by their cached NTK traces.

        Parameters
        ----------
        model : nn.Module
            The PINN/KAN model (needed for gradient computations).
        losses : Dict[str, Tensor]
            Per-component scalar losses (not yet summed).
        ntk_cache : dict
            Mutable cache of NTK traces from the previous update.
            Pass ``{}`` on the first call and feed the returned dict back
            on every subsequent call.
        step : int
            Current optimiser step.

        Returns
        -------
        weighted_losses : Dict[str, Tensor]
        ntk_cache : dict  (updated in place)
        """
        # Recompute NTK traces periodically
        if (step % self.run_per_step == 0) and (step > 0):
            ntk_cache.update(self._group_ntk(model, losses))
            if self.save_name:
                self._save_ntk(ntk_cache, step)

        # Total NTK sum (used as numerator for each weight)
        if step == 0 or not ntk_cache:
            ntk_sum: float = 1.0
        else:
            ntk_sum = float(sum(v for v in ntk_cache.values() if v is not None))

        weighted: Dict[str, torch.Tensor] = {}
        for key, value in losses.items():
            ntk_i = ntk_cache.get(key)
            w = ntk_sum / float(ntk_i) if ntk_i is not None else ntk_sum
            weighted[key] = w * value

        return weighted, ntk_cache


class NTKAggregator(Aggregator):
    """NTK trace weighting wrapped in the standard Aggregator interface.

    Adapts the standalone ``NTK`` class so it can be selected via
    ``build_aggregator('ntk', ...)`` and dropped into any training loop that
    calls ``aggregator(losses, step)``.

    Weight for each loss term::

        w_i = ntk_sum / ntk_i,  ntk_i = || ∂√|λ_i L_i| / ∂θ ||₂

    where ``λ_i`` are the fixed multipliers from ``weights``.  Traces are
    recomputed every ``run_per_step`` steps; cached between updates.  Weight
    updates are skipped when ``self.training`` is ``False`` (L-BFGS phase).

    Parameters
    ----------
    run_per_step : int
        NTK recompute frequency (default 1000).  Gradient computations are
        expensive — keep this ≥ 100.
    save_name : str or None
        If set, NTK traces are appended to ``<save_name>.csv``.
        Incompatible with CUDA graphs.
    """

    def __init__(
        self,
        params: List[nn.Parameter],
        num_losses: int,
        weights: Optional[List[float]] = None,
        run_per_step: int = 1000,
        save_name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(params, num_losses, weights)
        self.run_per_step = run_per_step
        self.save_name = save_name
        self.if_csv_head = True
        self._cache: Dict[str, torch.Tensor] = {}  # loss_key → NTK trace scalar
        if save_name:
            import warnings

            warnings.warn(
                "NTKAggregator: CUDA graphs are incompatible with save_name.",
                stacklevel=2,
            )

    @property
    def current_weights(self) -> Optional[List[float]]:
        """Return the cached NTK weights (or None before the first update)."""
        if not self._cache:
            return None
        ntk_sum = sum(float(v) for v in self._cache.values())
        return [ntk_sum / float(v) for v in self._cache.values()]

    def _compute_ntk_traces(
        self, weighted_losses: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute ||∂√|L_i|/∂θ||₂ for each (already fixed-weight-scaled) loss."""
        trainable = [p for p in self.params if p.requires_grad]
        traces: Dict[str, torch.Tensor] = {}
        for key, loss in weighted_losses.items():
            grads = torch.autograd.grad(
                torch.sqrt(torch.abs(loss)),
                trainable,
                retain_graph=True,
                allow_unused=True,
            )
            traces[key] = torch.sqrt(
                torch.sum(
                    torch.stack(
                        [g.detach().pow(2).sum() for g in grads if g is not None]
                    )
                )
            )
        return traces

    def _save_csv(self, step: int) -> None:
        """Append the cached NTK traces for ``step`` to ``<save_name>.csv``."""
        try:
            import pandas as pd
        except ImportError:
            print("  NTKAggregator: pandas not available — skipping CSV export.")
            return
        df = pd.DataFrame(
            {k: v.cpu().item() for k, v in self._cache.items()}, index=[step]
        )
        df.to_csv(self.save_name + ".csv", mode="a", header=self.if_csv_head)
        self.if_csv_head = False

    def forward(self, losses: Dict[str, torch.Tensor], step: int) -> torch.Tensor:
        keys = list(losses.keys())
        # Apply fixed per-term multipliers
        weighted = {k: self.weights[i] * losses[k] for i, k in enumerate(keys)}

        # Recompute NTK traces periodically (training mode only, so L-BFGS caches)
        if self.training and (step % self.run_per_step == 0) and (step > 0):
            self._cache.update(self._compute_ntk_traces(weighted))
            if self.save_name:
                self._save_csv(step)

        # Step 0 or cache not yet populated → plain weighted sum
        if step == 0 or not self._cache:
            return sum(weighted.values())

        ntk_sum = sum(float(v) for v in self._cache.values())
        return sum(
            (ntk_sum / float(self._cache.get(k, 1.0))) * weighted[k] for k in keys
        )


# ── Registry & factory ────────────────────────────────────────────────────────

_REGISTRY: Dict[str, type] = {
    "sum": Sum,
    "grad_norm": GradNorm,
    "res_norm": ResNorm,
    "homoscedastic": HomoscedasticUncertainty,
    "lr_annealing": LRAnnealing,
    "soft_adapt": SoftAdapt,
    "relobralo": Relobralo,
    "ema": EMA,
    "brdr": BalancedResidualDecayRate,
    "ntk": NTKAggregator,
}

AGGREGATOR_NAMES = sorted(_REGISTRY.keys())


def build_aggregator(
    name: str,
    params: Iterable[nn.Parameter],
    num_losses: int,
    weights: Optional[List[float]] = None,
    **kwargs,
) -> Aggregator:
    """Construct a loss aggregator by name.

    Parameters
    ----------
    name : str
        Aggregator strategy.  One of:
        ``'sum'``, ``'grad_norm'``, ``'res_norm'``, ``'homoscedastic'``,
        ``'lr_annealing'``, ``'soft_adapt'``, ``'relobralo'``, ``'ema'``,
        ``'brdr'``, ``'ntk'``.
    params : iterable of nn.Parameter
        All trainable model parameters.  Needed for gradient-based aggregators
        (``grad_norm``, ``lr_annealing``); ignored for stateless ones.
    num_losses : int
        Number of loss components (must match the ``losses`` dict passed to
        ``forward()``).
    weights : list[float], optional
        Initial per-loss multipliers in the same order as the ``losses`` dict,
        e.g. ``[1.0, lambda_bc, lambda_c0, lambda_c1]``.
        Defaults to 1.0 for every term.
    **kwargs :
        Aggregator-specific hyper-parameters (``alpha``, ``beta``, ``rho``,
        ``ema_alpha``, ``update_freq``, …).

    Returns
    -------
    Aggregator
        An ``nn.Module`` instance.  Call ``aggregator(losses_dict, step)``
        to get the scalar total loss.
    """
    key = name.lower().strip()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown aggregator '{name}'.  Choose from: {AGGREGATOR_NAMES}"
        )
    return _REGISTRY[key](list(params), num_losses, weights, **kwargs)
