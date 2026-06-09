# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Steady Cahn-Hilliard trainer (4th-order conserved phase-field).

Run:
    python src/trainer.py
    python src/trainer.py physics.physics.eps_sq=0.001
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

from src.metrics import compute_diagnostics
from src.physics import (
    cahn_hilliard_mass_loss,
    cahn_hilliard_noflux_bc_loss,
    cahn_hilliard_residual,
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


def _build_element_configs(dom) -> list[dict]:
    a, b = dom.a, dom.b
    n = dom.n_elements
    step = (b - a) / n
    return [
        {"N": dom.N_per_element, "a": float(a + i * step), "b": float(a + (i + 1) * step),
         "alpha": float(dom.alpha), "quadrature": dom.quadrature, "mapping": dom.mapping}
        for i in range(n)
    ]


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    torch.manual_seed(cfg.train.seed)
    dtype = torch.float64 if cfg.train.dtype == "float64" else torch.float32

    dom = cfg.physics.domain
    phys = cfg.physics.physics
    element_configs = _build_element_configs(dom)
    eps_sq = float(phys.eps_sq)
    c_mean = float(phys.c_mean)
    lambda_bc = float(phys.lambda_bc)
    lambda_mass = float(phys.lambda_mass)

    net = SCENElementNetwork(
        element_configs,
        hidden_dim=cfg.model.hidden_dim, n_layers=cfg.model.n_layers,
        backbone=cfg.model.backbone, poly_degree=cfg.model.poly_degree, dtype=dtype,
    )
    D1, D2, D4 = net.D1_global, net.D2_global, net.D4_global
    w = torch.cat([m.weights for m in net.mappers])
    w_norm = w / w.sum()

    adam = torch.optim.Adam(net.parameters(), lr=cfg.train.adam_lr)
    lbfgs = torch.optim.LBFGS(net.parameters(), line_search_fn="strong_wolfe", max_iter=20)
    opt = TwoPhaseOptimizer(adam, lbfgs)

    def closure():
        c = net()
        return (
            cahn_hilliard_residual(c, D2, D4, w_norm, eps_sq)
            + lambda_bc * cahn_hilliard_noflux_bc_loss(c, D1, D2, eps_sq)
            + lambda_mass * cahn_hilliard_mass_loss(c, w_norm, c_mean)
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
        c = net()
    diag = compute_diagnostics(c, w, c_mean)
    for k, v in diag.items():
        print(f"  {k}: {v:.4e}")

    net.save(os.path.join(cfg.output.dir, cfg.output.checkpoint))

    if wandb_run is not None:
        import wandb
        wandb.log({f"final/{k}": v for k, v in diag.items()})
        wandb_run.finish()

    return diag


if __name__ == "__main__":
    main()
