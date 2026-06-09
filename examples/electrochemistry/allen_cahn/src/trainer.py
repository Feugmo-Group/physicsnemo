# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Allen-Cahn trainer.

Run:
    python src/trainer.py
    python src/trainer.py physics.physics.eps_sq=0.001 model.hidden_dim=128
"""

from __future__ import annotations

import math
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
from physicsnemo.optim.loss_landscape import loss_landscape_scan, plot_landscape

from src.metrics import compute_errors
from src.physics import allen_cahn_bc_loss, allen_cahn_exact, allen_cahn_residual


def _init_wandb(cfg):
    if not cfg.wandb.enabled:
        return None
    try:
        import wandb
        return wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity or None,
            name=cfg.wandb.name,
            tags=list(cfg.wandb.tags) if cfg.wandb.tags else [],
            config=OmegaConf.to_container(cfg, resolve=True),
        )
    except ImportError:
        print("wandb not installed — skipping.")
        return None


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    torch.manual_seed(cfg.train.seed)
    dtype_str = cfg.train.dtype
    dtype = torch.float64 if dtype_str == "float64" else torch.float32

    dom = cfg.physics.domain
    eps_sq = float(cfg.physics.physics.eps_sq)
    lambda_bc = float(cfg.physics.physics.lambda_bc)

    # Single element spanning [a, b] — NSEM uses one large element with KTE
    element_configs = [
        {
            "N": int(dom.N_per_element),
            "a": float(dom.a),
            "b": float(dom.b),
            "alpha": float(dom.alpha),
            "quadrature": str(dom.quadrature),
            "mapping": str(dom.mapping),
        }
    ]

    net = SCENElementNetwork(
        element_configs,
        hidden_dim=cfg.model.hidden_dim,
        n_layers=cfg.model.n_layers,
        backbone=cfg.model.backbone,
        poly_degree=cfg.model.poly_degree,
        dtype=dtype_str,
    )

    x = net.mappers[0].nodes
    D2 = net.D2_global
    w = net.mappers[0].weights
    w_norm = w / w.sum()

    # ── Stage 1: pretrain to tanh exact shape ────────────────────────────────
    # Without this, the network learns a linear ramp that satisfies BCs but has
    # near-zero PDE residual — a flat basin Adam cannot escape.
    n_pretrain = int(cfg.train.get("n_pretrain", 0))
    if n_pretrain > 0:
        print(f"\n  Pre-training to tanh exact shape ({n_pretrain} steps) …")
        with torch.no_grad():
            u_target = allen_cahn_exact(x, eps_sq)
        pre_opt = torch.optim.Adam(net.parameters(), lr=float(cfg.train.pretrain_lr))
        pre_sched = torch.optim.lr_scheduler.CosineAnnealingLR(pre_opt, T_max=n_pretrain, eta_min=1e-5)
        for step in range(n_pretrain):
            pre_opt.zero_grad()
            loss_pre = ((net() - u_target) ** 2).mean()
            loss_pre.backward()
            pre_opt.step()
            pre_sched.step()
            if step % 500 == 0:
                print(f"    pretrain {step:5d}: MSE = {loss_pre.item():.3e}")
        print(f"    pretrain final: MSE = {loss_pre.item():.3e}")
        print()

    # ── Stage 2: physics training ─────────────────────────────────────────────
    adam = torch.optim.Adam(net.parameters(), lr=cfg.train.adam_lr)
    lbfgs = torch.optim.LBFGS(net.parameters(), line_search_fn="strong_wolfe", max_iter=100)
    opt = TwoPhaseOptimizer(adam, lbfgs)

    def closure():
        u = net()
        return allen_cahn_residual(u, D2, w_norm, eps_sq) + lambda_bc * allen_cahn_bc_loss(u)

    os.makedirs(cfg.output.dir, exist_ok=True)
    wandb_run = _init_wandb(cfg)

    history = opt.run(
        closure,
        n_adam_steps=cfg.train.n_adam,
        n_lbfgs_steps=cfg.train.n_lbfgs,
        verbose=True,
        log_every=cfg.output.log_every,
        grad_clip=float(cfg.train.get("grad_clip", 0)) or None,
    )

    if wandb_run is not None:
        import wandb
        for r in history:
            wandb.log({"loss": r["loss"], "phase": r["phase"]}, step=r["step"])

    print(f"\nFinal loss: {history[-1]['loss']:.4e}")

    with torch.no_grad():
        u = net()
    errors = compute_errors(u, x, w, eps_sq)
    print(f"  L∞ = {errors['Linf']:.3e}   L² = {errors['L2']:.3e}")

    net.save(os.path.join(cfg.output.dir, cfg.output.checkpoint))

    # ── Loss landscape ────────────────────────────────────────────────────────
    print("\n  Computing loss landscape …")

    def _landscape_closure():
        u = net()
        return {
            "pde": float(allen_cahn_residual(u, D2, w_norm, eps_sq)),
            "bc":  float(lambda_bc * allen_cahn_bc_loss(u)),
        }

    surfaces = loss_landscape_scan([net], _landscape_closure, nr_steps=24)
    plot_landscape(
        surfaces,
        component_labels={"pde": "PDE  ε²u″ − (u³−u) = 0", "bc": "BC penalty"},
        component_cmaps={"pde": "viridis", "bc": "inferno"},
        title="Allen-Cahn Loss Landscape",
        save_path=os.path.join(cfg.output.dir, "plots", "loss_landscape.png"),
    )

    if wandb_run is not None:
        import wandb
        wandb.log({f"final/{k}": v for k, v in errors.items()})
        wandb_run.finish()

    return errors


if __name__ == "__main__":
    main()
