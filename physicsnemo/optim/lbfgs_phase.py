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

"""Two-phase Adam → L-BFGS optimizer for physics-informed training."""

from __future__ import annotations

from typing import Callable, Optional

import torch


class TwoPhaseOptimizer:
    """Sequential Adam → L-BFGS optimizer for PINN training.

    Adam escapes saddle points in the early training phase; L-BFGS
    exploits the deterministic, smooth loss landscape to converge to
    near-machine precision in the second phase.

    The aggregator (if provided) is automatically switched to ``eval()``
    mode at the Adam→L-BFGS boundary.  This freezes any learned loss
    weights so the landscape seen by L-BFGS's Wolfe line search is
    stationary — a critical correctness requirement since L-BFGS calls
    the closure multiple times per step.

    Parameters
    ----------
    adam_optimizer : torch.optim.Adam
        Pre-constructed Adam optimizer over model parameters.
    lbfgs_optimizer : torch.optim.LBFGS
        Pre-constructed L-BFGS optimizer. Must be constructed with
        ``line_search_fn='strong_wolfe'`` for best results.
    aggregator : optional
        Loss aggregator with ``train()`` / ``eval()`` methods (e.g. any
        aggregator from ``physicsnemo.loss.aggregators``).  When provided,
        ``aggregator.eval()`` is called before the L-BFGS phase begins.

    Examples
    --------
    >>> import torch
    >>> model = torch.nn.Linear(1, 1)
    >>> adam = torch.optim.Adam(model.parameters(), lr=1e-3)
    >>> lbfgs = torch.optim.LBFGS(model.parameters(), line_search_fn='strong_wolfe')
    >>> opt = TwoPhaseOptimizer(adam, lbfgs)
    >>> x = torch.linspace(-1, 1, 32).unsqueeze(1)
    >>> y = torch.sin(x)
    >>> def closure():
    ...     return torch.nn.functional.mse_loss(model(x), y)
    >>> history = opt.run(closure, n_adam_steps=50, n_lbfgs_steps=10)
    >>> len(history) <= 60
    True
    """

    def __init__(
        self,
        adam_optimizer: torch.optim.Adam,
        lbfgs_optimizer: torch.optim.LBFGS,
        aggregator=None,
    ):
        self.adam = adam_optimizer
        self.lbfgs = lbfgs_optimizer
        self.aggregator = aggregator

    def run(
        self,
        closure_fn: Callable[[], torch.Tensor],
        n_adam_steps: int,
        n_lbfgs_steps: int,
        tol: float = 1e-10,
        verbose: bool = False,
        log_every: int = 100,
    ) -> list[dict]:
        """Run two-phase optimization.

        Parameters
        ----------
        closure_fn : Callable[[], Tensor]
            Zero-argument callable that computes and returns the loss.
            Must call ``zero_grad()`` internally or rely on the optimizer.
            The function is called from within both Adam and L-BFGS steps.
        n_adam_steps : int
            Number of Adam gradient steps.
        n_lbfgs_steps : int
            Maximum number of L-BFGS steps (may terminate early via tol).
        tol : float
            Early-stopping threshold for L-BFGS: stop when loss < tol.
        verbose : bool
            Print loss at each ``log_every`` step.
        log_every : int
            Print interval when verbose=True.

        Returns
        -------
        list[dict]
            History records: ``[{"step": int, "loss": float, "phase": str}, ...]``.
            Length ≤ ``n_adam_steps + n_lbfgs_steps``.
        """
        history: list[dict] = []

        # ── Phase 1: Adam ─────────────────────────────────────────────────────
        for step in range(n_adam_steps):
            self.adam.zero_grad()
            loss = closure_fn()
            loss.backward()
            self.adam.step()
            val = loss.item()
            history.append({"step": step, "loss": val, "phase": "adam"})
            if verbose and step % log_every == 0:
                print(f"  Adam {step:5d}: loss={val:.4e}")

        if n_lbfgs_steps == 0:
            return history

        # ── Phase transition: freeze aggregator weights ───────────────────────
        if self.aggregator is not None:
            self.aggregator.eval()

        # ── Phase 2: L-BFGS ──────────────────────────────────────────────────
        prev_loss = float("inf")
        lbfgs_step = [0]  # mutable int accessible from closure

        def _lbfgs_closure() -> torch.Tensor:
            self.lbfgs.zero_grad()
            loss = closure_fn()
            loss.backward()
            return loss

        for step in range(n_lbfgs_steps):
            lbfgs_step[0] = step
            loss = self.lbfgs.step(_lbfgs_closure)
            val = float(loss.detach())
            global_step = n_adam_steps + step
            history.append({"step": global_step, "loss": val, "phase": "lbfgs"})
            if verbose:
                print(f"  LBFGS {step:4d}: loss={val:.4e}")
            if val < tol:
                break
            if abs(prev_loss - val) < 1e-14 * abs(val):
                break
            prev_loss = val

        return history
