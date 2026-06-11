# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""2D Kolmogorov flow trainer — demonstrates LegendreKAN (KAN backbone).

This example uses backbone="kan" in SCENElementNetwork, so the sub-networks
are LegendreKAN instances.  Spatial derivatives are computed via DVRMapper2D
Kronecker operators (no autograd).

Run:
    python src/trainer.py
    python src/trainer.py physics.physics.nu=0.01   # Re=100 (turbulent regime)
    python src/trainer.py model.backbone=mlp        # compare with MLP backbone
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from physicsnemo.experimental.models.scen import DVRMapper2D, SCENElementNetwork
from physicsnemo.optim import TwoPhaseOptimizer

from src.metrics import compute_errors
from src.physics import (
    kolmogorov_bc_loss,
    kolmogorov_residuals_dvr,
    make_kolmogorov_informer,
)


def _init_wandb(cfg):
    if not cfg.wandb.enabled:
        return None
    try:
        import wandb
        return wandb.init(
            project=cfg.wandb.project, entity=cfg.wandb.entity or None,
            name=cfg.wandb.name, tags=list(cfg.wandb.tags or []),
            config=OmegaConf.to_container(cfg, resolve=True),
        )
    except ImportError:
        print("wandb not installed — skipping.")
        return None


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    torch.manual_seed(cfg.train.seed)
    dtype_str = cfg.train.dtype  # "float32" or "float64"
    dtype = torch.float64 if dtype_str == "float64" else torch.float32

    dom = cfg.physics.domain
    phys = cfg.physics.physics
    nu = float(phys.nu)
    n_force = int(phys.n_force)
    lambda_mean = float(phys.lambda_mean)

    mapper2d = DVRMapper2D(
        dom.Nx, dom.ax, dom.bx,
        Ny=dom.Ny, ay=dom.ay, by=dom.by,
        alpha_x=dom.alpha_x, alpha_y=dom.alpha_y,
        dtype=dtype,
    )
    xy = mapper2d.xy_nodes
    D1x, D1y = mapper2d.D1x, mapper2d.D1y
    D2x, D2y = mapper2d.D2x, mapper2d.D2y
    w = mapper2d.weights
    w_norm = w / w.sum()
    N_total = dom.Nx * dom.Ny

    # KAN backbone — key feature of this example
    element_configs = [{"N": N_total, "a": -1.0, "b": 1.0}]
    net = SCENElementNetwork(
        element_configs,
        hidden_dim=cfg.model.hidden_dim,
        n_layers=cfg.model.n_layers,
        backbone=cfg.model.backbone,
        poly_degree=cfg.model.poly_degree,
        dtype=dtype_str,
    )
    # Mixed (stream-vorticity) formulation: a second network for the vorticity
    # ω, recasting the biharmonic equation as two coupled second-order residuals.
    net_omega = SCENElementNetwork(
        element_configs,
        hidden_dim=cfg.model.hidden_dim,
        n_layers=cfg.model.n_layers,
        backbone=cfg.model.backbone,
        poly_degree=cfg.model.poly_degree,
        dtype=dtype_str,
    )

    print(f"\nBackbone: {cfg.model.backbone} (poly_degree={cfg.model.poly_degree})")
    params = list(net.parameters()) + list(net_omega.parameters())
    print(f"Parameters: {sum(p.numel() for p in params):,}")

    informer = make_kolmogorov_informer(D1x, D1y, D2x, D2y, nu, device=str(w.device))
    lambda_w = float(phys.get("lambda_w", 1.0))
    adam = torch.optim.Adam(params, lr=cfg.train.adam_lr)
    lbfgs = torch.optim.LBFGS(params, line_search_fn="strong_wolfe", max_iter=20)
    opt = TwoPhaseOptimizer(adam, lbfgs)

    def closure():
        psi = net()
        omega = net_omega()
        l_w, l_vort = kolmogorov_residuals_dvr(informer, psi, omega, w_norm, xy, n_force)
        return (
            l_vort + lambda_w * l_w
            + lambda_mean * kolmogorov_bc_loss(psi, w_norm)
        )

    os.makedirs(cfg.output.dir, exist_ok=True)
    wandb_run = _init_wandb(cfg)

    history = opt.run(
        closure, n_adam_steps=cfg.train.n_adam, n_lbfgs_steps=cfg.train.n_lbfgs,
        verbose=True, log_every=cfg.output.log_every,
    )
    if wandb_run is not None:
        import wandb
        for r in history:
            wandb.log({"loss": r["loss"], "phase": r["phase"]}, step=r["step"])

    print(f"\nFinal loss: {history[-1]['loss']:.4e}")

    with torch.no_grad():
        psi = net()
    errors = compute_errors(psi, xy, w, nu, n_force)
    print(f"  L∞ = {errors['Linf']:.3e}   L² = {errors['L2']:.3e}")

    net.save(os.path.join(cfg.output.dir, cfg.output.checkpoint))

    if wandb_run is not None:
        import wandb
        wandb.log({f"final/{k}": v for k, v in errors.items()})
        wandb_run.finish()

    return errors


if __name__ == "__main__":
    main()
