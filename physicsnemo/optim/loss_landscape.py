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

"""2-D filter-normalised loss landscape visualisation for PINN models.

Usage example::

    from physicsnemo.optim.loss_landscape import loss_landscape_scan, plot_landscape

    surfaces = loss_landscape_scan(
        models=[net],
        closure=lambda: {"pde": pde_loss(), "bc": bc_loss()},
        nr_steps=24,
    )
    fig = plot_landscape(surfaces, title="Convection-Diffusion Landscape")
    fig.savefig("landscape.png", bbox_inches="tight")
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn


_DEFAULT_CMAPS = ["viridis", "plasma", "inferno", "magma", "cividis", "turbo"]


def loss_landscape_scan(
    models: list[nn.Module],
    closure: Callable[[], dict[str, float]],
    nr_steps: int = 24,
    range_scale: float = 0.5,
    seed: int = 0,
    verbose: bool = True,
) -> dict[str, np.ndarray]:
    """Scan the loss landscape along two random filter-normalised directions.

    Perturbs all model parameters jointly along two orthogonal random directions
    and evaluates ``closure()`` at each of the ``nr_steps²`` grid points.
    Parameters are restored to their original values afterwards.

    The perturbation radius is ``range_scale * ‖θ‖``, where ``θ`` is the
    concatenated parameter vector.  Using a fraction of the parameter norm
    (the "filter normalisation" of Li et al. 2018) makes the scale independent
    of model size.

    Parameters
    ----------
    models : list of nn.Module
        All networks whose parameters are perturbed together (pass a single-item
        list for one network).
    closure : callable () → dict[str, float]
        No-argument function returning a ``{key: scalar}`` dict of loss
        components for the **current** parameter values.  Must be compatible with
        ``torch.no_grad()`` (i.e. no autograd Laplacian inside).
    nr_steps : int
        Number of grid points per axis; ``nr_steps²`` total evaluations.
        Default 24 (≈ 576 evaluations, ~1 s for a small MLP).
    range_scale : float
        Perturbation radius as fraction of ‖θ‖.  Default 0.5.
    seed : int
        RNG seed for reproducible random directions.  Default 0.
    verbose : bool
        Print progress every 10 %.

    Returns
    -------
    dict[str, np.ndarray]
        One ``(nr_steps, nr_steps)`` float64 array per loss component key, plus
        a ``"total"`` key with the element-wise sum.  The ``"_alphas"`` and
        ``"_betas"`` keys hold the 1-D axis values (each in ``[-1, 1]``).
    """
    all_params = [p for m in models for p in m.parameters()]
    theta = torch.cat([p.data.view(-1) for p in all_params]).clone()
    norm = theta.norm().item() + 1e-12

    rng = torch.Generator()
    rng.manual_seed(seed)
    d1 = torch.randn(theta.shape, generator=rng, dtype=theta.dtype, device=theta.device)
    d2 = torch.randn(theta.shape, generator=rng, dtype=theta.dtype, device=theta.device)
    d1 = (d1 / d1.norm()) * norm * range_scale
    d2 = d2 - (d2.dot(d1) / (d1.dot(d1) + 1e-12)) * d1
    d2 = (d2 / (d2.norm() + 1e-12)) * norm * range_scale

    alphas = np.linspace(-1.0, 1.0, nr_steps)
    betas = np.linspace(-1.0, 1.0, nr_steps)

    # Probe keys at the trained point (alpha=0, beta=0)
    with torch.no_grad():
        probe = closure()
    keys = list(probe.keys())

    surfs: dict[str, np.ndarray] = {k: np.zeros((nr_steps, nr_steps), dtype=np.float64) for k in keys}
    n_eval = nr_steps * nr_steps
    done = 0

    with torch.no_grad():
        for i, alpha in enumerate(alphas):
            a_d1 = alpha * d1
            for j, beta in enumerate(betas):
                nn.utils.vector_to_parameters(theta + a_d1 + beta * d2, all_params)
                comp = closure()
                for k in keys:
                    surfs[k][i, j] = float(comp[k])
                done += 1
                if verbose and done % max(1, n_eval // 10) == 0:
                    total_val = sum(surfs[k][i, j] for k in keys)
                    print(f"  landscape {100 * done // n_eval:3d}%  total={total_val:.3e}")
        nn.utils.vector_to_parameters(theta, all_params)

    surfs["total"] = sum(surfs[k] for k in keys)
    surfs["_alphas"] = alphas
    surfs["_betas"] = betas
    return surfs


def plot_landscape(
    surfaces: dict[str, np.ndarray],
    component_keys: list[str] | None = None,
    component_labels: dict[str, str] | None = None,
    component_cmaps: dict[str, str] | None = None,
    title: str = "Loss Landscape",
    save_path: str | None = None,
) -> "matplotlib.figure.Figure":
    """Render per-component loss landscape surfaces as a grid of 3-D plots.

    Parameters
    ----------
    surfaces : dict
        Output of :func:`loss_landscape_scan`.  Must contain ``"_alphas"`` and
        ``"_betas"`` keys plus one ``(nr_steps, nr_steps)`` array per component.
        The ``"total"`` panel is always appended as the last subplot.
    component_keys : list[str], optional
        Which keys to plot (in order).  Defaults to all non-underscore, non-total
        keys found in ``surfaces``.
    component_labels : dict[str, str], optional
        Human-readable panel title per key.  Falls back to the key.
    component_cmaps : dict[str, str], optional
        Matplotlib colormap per key.  Cycles through defaults if absent.
    title : str
        Figure suptitle.
    save_path : str or None
        If given, save the figure to this path (PNG).  Parent dirs are created.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    alphas = surfaces["_alphas"]
    betas = surfaces["_betas"]
    a_g, b_g = np.meshgrid(alphas, betas, indexing="ij")

    if component_keys is None:
        component_keys = [k for k in surfaces if not k.startswith("_") and k != "total"]

    labels = dict(component_labels or {})
    cmaps = dict(component_cmaps or {})
    for i, k in enumerate(component_keys):
        labels.setdefault(k, k)
        cmaps.setdefault(k, _DEFAULT_CMAPS[i % len(_DEFAULT_CMAPS)])

    plot_keys = component_keys + ["total"]
    n_panels = len(plot_keys)
    n_cols = min(n_panels, 3)
    n_rows = (n_panels + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(7 * n_cols, 5 * n_rows), dpi=120)
    fig.suptitle(title, fontsize=13, y=1.01)

    for idx, key in enumerate(plot_keys):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection="3d")
        surf_data = surfaces["total"] if key == "total" else surfaces[key]
        cmap = "turbo" if key == "total" else cmaps.get(key, "viridis")
        lbl = "Total loss" if key == "total" else labels.get(key, key)

        log_s = np.log10(np.maximum(surf_data, 1e-20))
        sp = ax.plot_surface(
            a_g, b_g, log_s,
            alpha=0.88, cmap=cmap, antialiased=True,
            edgecolor="none", linewidth=0, shade=True,
        )
        ax.set_title(lbl, fontsize=10)
        ax.set_xlabel("Dir 1", fontsize=8)
        ax.set_ylabel("Dir 2", fontsize=8)
        ax.set_zlabel("log₁₀(Loss)", fontsize=8)
        ax.tick_params(axis="both", labelsize=6)
        ax.tick_params(axis="z", labelsize=6)
        cb = fig.colorbar(sp, ax=ax, shrink=0.45, aspect=10, pad=0.13)
        cb.ax.tick_params(labelsize=6)

    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"  Saved landscape → {save_path}")
    return fig
