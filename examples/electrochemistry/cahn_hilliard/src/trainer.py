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
    cahn_hilliard_noflux_bc_mixed,
    cahn_hilliard_residuals_dvr,
    make_ch_informer,
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
    N = int(dom.N_per_element)
    alpha = float(dom.alpha)
    quad = str(dom.quadrature)
    mapping = str(dom.mapping)
    if "boundaries" in dom:
        bounds = list(dom.boundaries)
        return [
            {"N": N, "a": float(bounds[i]), "b": float(bounds[i + 1]),
             "alpha": alpha, "quadrature": quad, "mapping": mapping}
            for i in range(len(bounds) - 1)
        ]
    return [{"N": N, "a": float(dom.a), "b": float(dom.b),
             "alpha": alpha, "quadrature": quad, "mapping": mapping}]


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    torch.manual_seed(cfg.train.seed)
    dtype_str = cfg.train.dtype  # "float32" or "float64"
    dtype = torch.float64 if dtype_str == "float64" else torch.float32

    dom = cfg.physics.domain
    phys = cfg.physics.physics
    element_configs = _build_element_configs(dom)
    eps_sq = float(phys.eps_sq)
    c_mean = float(phys.c_mean)
    lambda_bc = float(phys.lambda_bc)
    lambda_mass = float(phys.lambda_mass)
    lambda_mu = float(phys.get("lambda_mu", 1.0))

    net = SCENElementNetwork(
        element_configs,
        hidden_dim=cfg.model.hidden_dim, n_layers=cfg.model.n_layers,
        backbone=cfg.model.backbone, poly_degree=cfg.model.poly_degree, dtype=dtype_str,
    )
    # Mixed (auxiliary-field) formulation: a second network for the chemical
    # potential μ, so the 4th-order Cahn-Hilliard equation becomes two coupled
    # second-order residuals (no D4 operator).
    net_mu = SCENElementNetwork(
        element_configs,
        hidden_dim=cfg.model.hidden_dim, n_layers=cfg.model.n_layers,
        backbone=cfg.model.backbone, poly_degree=cfg.model.poly_degree, dtype=dtype_str,
    )
    D1, D2 = net.D1_global, net.D2_global
    w = torch.cat([m.weights for m in net.mappers])
    w_norm = w / w.sum()

    informer = make_ch_informer(eps_sq, D2, device=str(w.device))
    params = list(net.parameters()) + list(net_mu.parameters())
    adam = torch.optim.Adam(params, lr=cfg.train.adam_lr)
    lbfgs = torch.optim.LBFGS(params, line_search_fn="strong_wolfe", max_iter=20)
    opt = TwoPhaseOptimizer(adam, lbfgs)

    def closure():
        c = net()
        mu = net_mu()
        l_mu, l_ch = cahn_hilliard_residuals_dvr(informer, c, mu, w_norm)
        return (
            lambda_mu * l_mu + l_ch
            + lambda_bc * cahn_hilliard_noflux_bc_mixed(c, mu, D1)
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
