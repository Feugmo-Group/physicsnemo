# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convection-diffusion trainer (boundary layer stiffness test).

Run:
    python src/trainer.py
    python src/trainer.py physics.physics.eps=0.001  # sharper boundary layer
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

from physicsnemo.experimental.models.scen import SCENElementNetwork
from physicsnemo.optim import TwoPhaseOptimizer

from src.metrics import compute_errors
from src.physics import cd_bc_loss, cd_residual


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
    dtype = torch.float64 if cfg.train.dtype == "float64" else torch.float32

    dom = cfg.physics.domain
    phys = cfg.physics.physics
    eps = float(phys.eps)
    a = float(phys.a)
    lambda_bc = float(phys.lambda_bc)

    element_configs = [
        {"N": dom.N_per_element, "a": float(dom.a), "b": float(dom.b),
         "alpha": float(dom.alpha), "quadrature": dom.quadrature, "mapping": dom.mapping}
    ]

    net = SCENElementNetwork(
        element_configs,
        hidden_dim=cfg.model.hidden_dim, n_layers=cfg.model.n_layers,
        backbone=cfg.model.backbone, poly_degree=cfg.model.poly_degree, dtype=dtype,
    )
    x = net.mappers[0].nodes
    D1, D2 = net.D1_global, net.D2_global
    w = net.mappers[0].weights
    w_norm = w / w.sum()

    adam = torch.optim.Adam(net.parameters(), lr=cfg.train.adam_lr)
    lbfgs = torch.optim.LBFGS(net.parameters(), line_search_fn="strong_wolfe", max_iter=20)
    opt = TwoPhaseOptimizer(adam, lbfgs)

    def closure():
        u = net()
        return cd_residual(u, D1, D2, w_norm, eps, a) + lambda_bc * cd_bc_loss(u)

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
        u = net()
    errors = compute_errors(u, x, w, eps, a)
    print(f"  L∞ = {errors['Linf']:.3e}   L² = {errors['L2']:.3e}")

    net.save(os.path.join(cfg.output.dir, cfg.output.checkpoint))

    if wandb_run is not None:
        import wandb
        wandb.log({f"final/{k}": v for k, v in errors.items()})
        wandb_run.finish()

    return errors


if __name__ == "__main__":
    main()
