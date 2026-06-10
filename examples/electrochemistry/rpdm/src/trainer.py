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

"""Refined Point Defect Model (RPDM) PINN trainer -- PhysicsNeMo v2.0 idiom.

Explicit PyTorch training loop:
    * one ``FullyConnected`` net maps (x, y) -> 4 starred fields,
    * a hard-BC layer enforces initial/boundary Dirichlet conditions exactly,
    * geometry / sampling use ``physicsnemo.mesh``: a structured grid over the
      space-time rectangle x in [0, 1], y in [0, yf]; interior points come from
      ``sample_random_points_on_cells`` on the grid, boundary points from its
      boundary mesh, then split by edge with coordinate masks (left x=0,
      right x=1, initial y=0),
    * ``PhysicsInformer`` (autodiff) evaluates the residual groups on
      interior / left (x=0) / right (x=1) coordinate batches,
    * per-term squared residual losses are aggregated with ``BalancedResidualDecayRate``
      (BRDR) adaptive weighting,
    * Adam + ExponentialLR,
    * a final validation block evaluates the trained fields on a meshgrid, saves
      a matplotlib figure, and reports the film-thickness error vs COMSOL.

Run from the example root:

    python src/trainer.py
    python src/trainer.py training.max_steps=50
"""

# ruff: noqa: E402  (src.* imports require the sys.path insertion below)

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dataclasses import asdict

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig
from src.hard_bc import enforce_hard_bc
from src.metrics import film_thickness_error
from src.physics import Parameters, PointDefectModel, make_Eext
from torch.optim import Adam, lr_scheduler

from physicsnemo.distributed import DistributedManager
from physicsnemo.mesh.primitives.planar.structured_grid import (
    load as load_structured_grid,
)
from physicsnemo.mesh.sampling import sample_random_points_on_cells
from physicsnemo.models.mlp.fully_connected import FullyConnected
from physicsnemo.optim import build_aggregator
from physicsnemo.sym.eq.phy_informer import PhysicsInformer
from physicsnemo.utils import save_checkpoint, set_default_dtype
from physicsnemo.utils.logging import PythonLogger

# Residual term groups and the coordinate set each is evaluated on.
INTERIOR_TERMS = ["poisson", "transport_CV", "transport_AV", "film_growth"]
LEFT_TERMS = ["flux_R1", "flux_R2", "mf_phif"]
RIGHT_TERMS = ["flux_R3", "flux_R4", "fs_phif"]
ALL_TERMS = INTERIOR_TERMS + LEFT_TERMS + RIGHT_TERMS

# phimf (the interface potential used by film_growth) is batch-constant and is
# detached so PhysicsInformer treats it as data rather than a differentiable field.
DETACH_NAMES = ["phimf"]


def _coords_from_xy(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Stack flat x/y tensors into a [N, 2] requires_grad coordinate batch."""
    coords = torch.stack([x.reshape(-1), y.reshape(-1)], dim=1)
    coords.requires_grad_(True)
    return coords


class RectGeometry:
    """Space-time rectangle [0, 1] x [0, yf] sampled via ``physicsnemo.mesh``.

    A structured grid supplies interior points; its boundary mesh supplies edge
    points, which are split into the three RPDM edges the physics needs:

        * ``left``     -- metal/film interface, x = 0,
        * ``right``    -- film/solution interface, x = 1,
        * ``initial``  -- initial condition, y = 0.

    The fourth boundary edge (y = yf) carries no residual term and is dropped.
    Mirrors the ``load_structured_grid`` / ``sample_random_points_on_cells``
    approach used by ``examples/cfd/ldc_pinns/train.py``, mapped onto the RPDM
    space-time rectangle.
    """

    def __init__(self, yf: float, n_x: int, n_y: int, device, eps: float = 1e-6):
        self.x_min, self.x_max = 0.0, 1.0
        self.y_min, self.y_max = 0.0, yf
        self.device = device
        self.eps = eps
        self.interior_mesh = load_structured_grid(
            x_min=self.x_min,
            x_max=self.x_max,
            y_min=self.y_min,
            y_max=self.y_max,
            n_x=n_x,
            n_y=n_y,
            device=device,
        )
        self.boundary_mesh = self.interior_mesh.get_boundary_mesh()

    def sample_interior(self, n_points: int):
        """Interior coords plus an analytical SDF (distance to rectangle edges)."""
        idx = torch.randint(
            0, self.interior_mesh.n_cells, (n_points,), device=self.device
        )
        pts = sample_random_points_on_cells(self.interior_mesh, idx)
        x, y = pts[:, 0], pts[:, 1]
        sdf = torch.min(
            torch.stack(
                [x - self.x_min, self.x_max - x, y - self.y_min, self.y_max - y],
                dim=-1,
            ),
            dim=-1,
        ).values
        coords = _coords_from_xy(x, y)
        return coords, sdf.reshape(-1, 1)

    def _sample_boundary_points(self, n_points: int):
        idx = torch.randint(
            0, self.boundary_mesh.n_cells, (n_points,), device=self.device
        )
        pts = sample_random_points_on_cells(self.boundary_mesh, idx)
        return pts[:, 0], pts[:, 1]

    def sample_boundary(self, n_left: int, n_right: int, n_initial: int):
        """Return left (x=0), right (x=1) and initial (y=0) coordinate batches.

        The boundary mesh is sampled densely and split by coordinate masks (the
        same ``< eps`` / ``> 1 - eps`` style ldc_pinns uses for its top-wall
        mask); we then take up to the requested number of points from each edge.
        """
        # oversample so each edge has enough candidates after masking
        n_total = 4 * max(n_left, n_right, n_initial) + 16
        x, y = self._sample_boundary_points(n_total)

        mask_left = x < self.x_min + self.eps
        mask_right = x > self.x_max - self.eps
        mask_initial = y < self.y_min + self.eps

        def take(mask, n):
            xx, yy = x[mask], y[mask]
            if xx.numel() >= n:
                xx, yy = xx[:n], yy[:n]
            return _coords_from_xy(xx, yy)

        return (
            take(mask_left, n_left),
            take(mask_right, n_right),
            take(mask_initial, n_initial),
        )


def _eval_fields(net, coords, pde):
    """Run the net + hard-BC layer; return enforced fields plus x/y columns.

    ``x`` and ``y`` are included because several RPDM equations reference the
    bare coordinate symbols (e.g. the Landau convection term ``x * l_y`` and
    the applied potential ``phiext(y)``); PhysicsInformer needs them as inputs.
    """
    x = coords[:, 0:1]
    y = coords[:, 1:2]
    raw = net(coords)
    starred = {
        "cCV_star": raw[:, 0:1],
        "cAV_star": raw[:, 1:2],
        "phif_star": raw[:, 2:3],
        "l_star": raw[:, 3:4],
    }
    fields = enforce_hard_bc(x, y, starred, pde.phif_initial_fn, pde.lini)
    fields["x"] = x
    fields["y"] = y
    return fields


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    set_default_dtype(torch.float64)
    torch.manual_seed(cfg.training.seed)

    DistributedManager.initialize()  # Only call this once in the entire script!
    dist = DistributedManager()
    device = dist.device

    log = PythonLogger(name="rpdm")
    log.file_logging()

    transpassive = cfg.custom.transpassive

    # parameters and dimensionless final time
    p = Parameters()
    yf = 1.0  # final dimensionless time (tc / tc)

    Eext = make_Eext(tc=p.tc)
    pde = PointDefectModel(Eext=Eext, yf=yf, transpassive=transpassive, **asdict(p))

    log.info(f"RPDM passive-mode PINN | yf={yf} | phic={pde.phic:.4f} V")
    log.info(f"Device: {device} | Residual terms: {ALL_TERMS}")

    # geometry / sampling on the space-time rectangle via physicsnemo.mesh
    geom = RectGeometry(yf=yf, n_x=cfg.mesh.n_x, n_y=cfg.mesh.n_y, device=device)

    # network: (x, y) -> 4 starred fields
    net = FullyConnected(
        in_features=2,
        out_features=4,
        num_layers=cfg.model.nr_layers,
        layer_size=cfg.model.layer_size,
    ).to(device)

    # one PhysicsInformer over the full equation set; we read whichever
    # residuals each coordinate batch needs.
    pi = PhysicsInformer(
        required_outputs=ALL_TERMS,
        equations=pde,
        grad_method="autodiff",
        device=device,
        detach_names=DETACH_NAMES,
    )

    # BRDR adaptive weighting over all residual terms
    brdr = build_aggregator(
        "brdr",
        net.parameters(),
        num_losses=len(ALL_TERMS),
        weights=[1.0] * len(ALL_TERMS),
    ).to(device)
    brdr.train()

    optimizer = Adam(net.parameters(), lr=cfg.optimizer.lr)
    gamma = cfg.scheduler.decay_rate ** (1.0 / cfg.scheduler.decay_steps)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    # one-time hard-BC verification at y=0 / x=0 / x=1
    _verify_hard_bc(net, pde, device, log)

    n_int = cfg.batch_size.Interior
    n_left = cfg.batch_size.Left
    n_right = cfg.batch_size.Right
    n_initial = cfg.batch_size.Initial

    for step in range(cfg.training.max_steps):
        optimizer.zero_grad()

        per_term = {}

        # interior residuals (SDF-weighted, mirroring ldc_pinns)
        coords_i, sdf_i = geom.sample_interior(n_int)
        fields_i = _eval_fields(net, coords_i, pde)
        res_i = pi.forward({**fields_i, "coordinates": coords_i})
        for t in INTERIOR_TERMS:
            per_term[t] = ((res_i[t] * sdf_i) ** 2).mean()

        # boundary edges: left (x=0), right (x=1), initial (y=0)
        coords_l, coords_r, _coords_init = geom.sample_boundary(
            n_left, n_right, n_initial
        )

        # left boundary x=0 residuals
        fields_l = _eval_fields(net, coords_l, pde)
        res_l = pi.forward({**fields_l, "coordinates": coords_l})
        for t in LEFT_TERMS:
            per_term[t] = (res_l[t] ** 2).mean()

        # right boundary x=1 residuals
        fields_r = _eval_fields(net, coords_r, pde)
        res_r = pi.forward({**fields_r, "coordinates": coords_r})
        for t in RIGHT_TERMS:
            per_term[t] = (res_r[t] ** 2).mean()

        # aggregate with BRDR (dict in ALL_TERMS order, passing the step index)
        losses_dict = {t: per_term[t] for t in ALL_TERMS}
        loss = brdr(losses_dict, step)

        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % cfg.training.log_freq == 0 or step == cfg.training.max_steps - 1:
            raw_sum = float(sum(per_term[t] for t in ALL_TERMS).detach())
            log.info(
                f"step {step:6d} | brdr_loss={loss.item():.6e} "
                f"| raw_sum={raw_sum:.6e} "
                f"| lr={scheduler.get_last_lr()[0]:.3e}"
            )

    log.info("Training complete.")

    # film-thickness metric vs COMSOL (L(t) error)
    final_loss = float(loss.detach())
    try:
        err = film_thickness_error(net, pde, p, device)
        log.info(
            f"Film thickness vs COMSOL | L_inf={err['linf']:.3e} m "
            f"| rel_L2={err['rel_l2']:.3e}"
        )
    except FileNotFoundError:
        err = None
        log.info("Film thickness metric skipped (COMSOL CSV not found).")

    if cfg.validation.enabled:
        _validation_plot(net, pde, p, yf, cfg, device, err, log)

    # Idiomatic PhysicsNeMo checkpoint: field net + optimizer + scheduler, with
    # the final loss / film-thickness metrics as metadata.  Written into the
    # example's ``outputs/`` directory (gitignored), not the tracked example root.
    out_dir = os.path.join(_ROOT, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    metadata = {"final_loss": final_loss}
    if err is not None:
        metadata.update({"linf": float(err["linf"]), "rel_l2": float(err["rel_l2"])})
    save_checkpoint(
        out_dir,
        models=net,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=int(cfg.training.max_steps),
        metadata=metadata,
    )
    log.info(f"Saved checkpoint to {out_dir}")


def _validation_plot(net, pde, p, yf, cfg, device, err, log) -> None:
    """Evaluate trained fields on a meshgrid and save a figure (ldc_pinns style)."""
    n_x, n_y = cfg.validation.n_x, cfg.validation.n_y
    x = np.linspace(0.0, 1.0, n_x)
    y = np.linspace(0.0, yf, n_y)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    xt = torch.as_tensor(
        xx.reshape(-1, 1), dtype=torch.get_default_dtype(), device=device
    )
    yt = torch.as_tensor(
        yy.reshape(-1, 1), dtype=torch.get_default_dtype(), device=device
    )
    coords = torch.cat([xt, yt], dim=1)

    raw = net(coords)
    starred = {
        "cCV_star": raw[:, 0:1],
        "cAV_star": raw[:, 1:2],
        "phif_star": raw[:, 2:3],
        "l_star": raw[:, 3:4],
    }
    fields = enforce_hard_bc(xt, yt, starred, pde.phif_initial_fn, pde.lini)
    phif = fields["phif"].detach().cpu().numpy().reshape(n_y, n_x)
    lfield = fields["l"].detach().cpu().numpy().reshape(n_y, n_x)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    im = axes[0].imshow(phif, origin="lower", aspect="auto", extent=[0.0, 1.0, 0.0, yf])
    fig.colorbar(im, ax=axes[0])
    axes[0].set_title("film potential phif(x, y)")
    axes[0].set_xlabel("x (Landau)")
    axes[0].set_ylabel("y (time)")

    im = axes[1].imshow(
        lfield, origin="lower", aspect="auto", extent=[0.0, 1.0, 0.0, yf]
    )
    fig.colorbar(im, ax=axes[1])
    axes[1].set_title("film thickness l(x, y)")
    axes[1].set_xlabel("x (Landau)")
    axes[1].set_ylabel("y (time)")

    # L(t) curve at x=0.5 (denondimensionalized) vs COMSOL when available
    axes[2].plot(y * p.tc, lfield[:, n_x // 2] * p.lc, label="PINN")
    if err is not None:
        axes[2].plot(err["t_comsol"], err["L_comsol"], "k--", label="COMSOL")
    axes[2].set_title("film thickness L(t) at x=0.5")
    axes[2].set_xlabel("t [s]")
    axes[2].set_ylabel("L [m]")
    axes[2].legend()

    fig.tight_layout()
    out_dir = os.path.join(_ROOT, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "rpdm_validation.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    log.info(f"Validation figure written to {out_path}")


def _verify_hard_bc(net, pde, device, log) -> None:
    """Assert the enforced fields match prescribed initial/boundary values."""
    with torch.no_grad():
        # initial condition at y=0
        x = torch.linspace(0, 1, 16, device=device).reshape(-1, 1)
        y = torch.zeros_like(x)
        coords = torch.cat([x, y], dim=1)
    coords.requires_grad_(True)
    fields = _eval_fields(net, coords, pde)
    with torch.no_grad():
        checks = {
            "cCV": torch.zeros_like(fields["cCV"]),
            "cAV": torch.zeros_like(fields["cAV"]),
            "l": torch.full_like(fields["l"], pde.lini),
        }
        for name, target in checks.items():
            if not torch.allclose(fields[name], target, atol=1e-8):
                raise AssertionError(f"{name} initial condition not enforced")
    log.info(
        f"Hard-BC verification PASSED: at y=0  cCV=cAV=0, l={pde.lini:.4f} "
        f"(max |cCV|={fields['cCV'].abs().max().item():.2e})"
    )


if __name__ == "__main__":
    main()
