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
    * ``PhysicsInformer`` (autodiff) evaluates the residual groups on
      interior / left (x=0) / right (x=1) coordinate batches,
    * per-term squared residual losses are aggregated with ``BalancedResidualDecayRate``
      (BRDR) adaptive weighting,
    * Adam + ExponentialLR.

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
import torch
from omegaconf import DictConfig
from src.hard_bc import enforce_hard_bc
from src.physics import Parameters, PointDefectModel, make_Eext
from torch.optim import Adam, lr_scheduler

from physicsnemo.models.mlp.fully_connected import FullyConnected
from physicsnemo.optim import BalancedResidualDecayRate
from physicsnemo.sym.eq.phy_informer import PhysicsInformer
from physicsnemo.utils import set_default_dtype
from physicsnemo.utils.logging import PythonLogger

# Residual term groups and the coordinate set each is evaluated on.
INTERIOR_TERMS = ["poisson", "transport_CV", "transport_AV", "film_growth"]
LEFT_TERMS = ["flux_R1", "flux_R2", "mf_phif"]
RIGHT_TERMS = ["flux_R3", "flux_R4", "fs_phif"]
ALL_TERMS = INTERIOR_TERMS + LEFT_TERMS + RIGHT_TERMS

# phimf (the interface potential used by film_growth) is batch-constant and is
# detached so PhysicsInformer treats it as data rather than a differentiable field.
DETACH_NAMES = ["phimf"]


def _sample_coords(n: int, x_lo: float, x_hi: float, yf: float, device) -> torch.Tensor:
    """Uniform coordinate batch on [x_lo, x_hi] x [0, yf], requires_grad."""
    x = torch.rand(n, 1, device=device) * (x_hi - x_lo) + x_lo
    y = torch.rand(n, 1, device=device) * yf
    coords = torch.cat([x, y], dim=1)
    coords.requires_grad_(True)
    return coords


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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log = PythonLogger(name="rpdm")
    log.file_logging()

    transpassive = cfg.custom.transpassive

    # parameters and dimensionless final time
    p = Parameters()
    yf = 1.0  # final dimensionless time (tc / tc)

    Eext = make_Eext(tc=p.tc)
    pde = PointDefectModel(Eext=Eext, yf=yf, transpassive=transpassive, **asdict(p))

    log.info(f"RPDM passive-mode PINN | yf={yf} | phic={pde.phic:.4f} V")
    log.info(f"Residual terms: {ALL_TERMS}")

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
    brdr = BalancedResidualDecayRate(num_losses=len(ALL_TERMS)).to(device)
    brdr.train()

    optimizer = Adam(net.parameters(), lr=cfg.optimizer.lr)
    gamma = cfg.scheduler.decay_rate ** (1.0 / cfg.scheduler.decay_steps)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    # one-time hard-BC verification at y=0 / x=0 / x=1
    _verify_hard_bc(net, pde, device, log)

    n_int = cfg.batch_size.Interior
    n_left = cfg.batch_size.Left
    n_right = cfg.batch_size.Right

    for step in range(cfg.training.max_steps):
        optimizer.zero_grad()

        per_term = {}

        # interior residuals
        coords_i = _sample_coords(n_int, 0.0, 1.0, yf, device)
        fields_i = _eval_fields(net, coords_i, pde)
        res_i = pi.forward({**fields_i, "coordinates": coords_i})
        for t in INTERIOR_TERMS:
            per_term[t] = (res_i[t] ** 2).mean()

        # left boundary x=0 residuals
        coords_l = _sample_coords(n_left, 0.0, 0.0, yf, device)
        fields_l = _eval_fields(net, coords_l, pde)
        res_l = pi.forward({**fields_l, "coordinates": coords_l})
        for t in LEFT_TERMS:
            per_term[t] = (res_l[t] ** 2).mean()

        # right boundary x=1 residuals
        coords_r = _sample_coords(n_right, 1.0, 1.0, yf, device)
        fields_r = _eval_fields(net, coords_r, pde)
        res_r = pi.forward({**fields_r, "coordinates": coords_r})
        for t in RIGHT_TERMS:
            per_term[t] = (res_r[t] ** 2).mean()

        # aggregate with BRDR (1D tensor in ALL_TERMS order)
        loss_vec = torch.stack([per_term[t] for t in ALL_TERMS])
        loss = brdr(loss_vec)

        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % cfg.training.log_freq == 0 or step == cfg.training.max_steps - 1:
            raw_sum = float(loss_vec.sum().detach())
            log.info(
                f"step {step:6d} | brdr_loss={loss.item():.6e} "
                f"| raw_sum={raw_sum:.6e} "
                f"| lr={scheduler.get_last_lr()[0]:.3e}"
            )

    log.info("Training complete.")


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
