# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convection-diffusion trainer (boundary layer stiffness test).

Run (single element):
    python src/trainer.py

Run (two elements, split at x=0.05 to resolve boundary layer separately):
    python src/trainer.py 'physics.domain.boundaries=[0.0,0.05,1.0]'

Run (three elements with C0-only interfaces):
    python src/trainer.py 'physics.domain.boundaries=[0.0,0.03,0.1,1.0]' physics.domain.interface_cond=c0

Run (sharper boundary layer):
    python src/trainer.py physics.physics.eps=0.001
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
from physicsnemo.optim.loss_landscape import loss_landscape_scan, plot_landscape

from src.metrics import compute_errors
from src.physics import all_interface_losses, cd_bc_loss, cd_exact, cd_residual


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
    eps = float(phys.eps)
    a_conv = float(phys.a)
    lambda_bc = float(phys.lambda_bc)
    lambda_int = float(dom.get("lambda_interface", 100.0))
    interface_cond = str(dom.get("interface_cond", "both"))

    # ── Build element configs from boundary list ───────────────────────────────
    boundaries = list(dom.boundaries)  # e.g. [0.0, 0.05, 1.0]
    K = len(boundaries) - 1
    element_configs = [
        {
            "N": int(dom.N_per_element),
            "a": float(boundaries[k]),
            "b": float(boundaries[k + 1]),
            "alpha": float(dom.alpha),
            "quadrature": str(dom.quadrature),
            "mapping": str(dom.mapping),
        }
        for k in range(K)
    ]
    print(f"  {K} element(s): {boundaries}")
    if K > 1:
        print(f"  Interface condition: {interface_cond}  λ_int={lambda_int}")

    net = SCENElementNetwork(
        element_configs,
        hidden_dim=cfg.model.hidden_dim, n_layers=cfg.model.n_layers,
        backbone=cfg.model.backbone, poly_degree=cfg.model.poly_degree,
        dtype=dtype_str,
    )

    # Global matrices and per-element helpers
    D1_global = net.D1_global
    D2_global = net.D2_global
    w_norm_global = net.w_norm_global
    x_global = torch.cat([m.nodes for m in net.mappers])
    w_global = torch.cat([m.weights for m in net.mappers])

    # Per-element D1 matrices (block diagonal rows)
    sizes = net.element_sizes
    D1_elems = [net.mappers[k].D1 for k in range(K)]

    # ── Stage 1: pretrain to exact solution shape ─────────────────────────────
    n_pretrain = int(cfg.train.get("n_pretrain", 0))
    if n_pretrain > 0:
        print(f"\n  Pre-training to exact solution shape ({n_pretrain} steps) …")
        with torch.no_grad():
            u_target = cd_exact(x_global, eps, a_conv)
        pre_opt = torch.optim.Adam(net.parameters(), lr=float(cfg.train.pretrain_lr))
        for step in range(n_pretrain):
            pre_opt.zero_grad()
            loss_pre = ((net() - u_target) ** 2).mean()
            loss_pre.backward()
            pre_opt.step()
            if step % 200 == 0:
                print(f"    pretrain {step:4d}: MSE = {loss_pre.item():.3e}")
        print()

    # ── Stage 2: physics training ──────────────────────────────────────────────
    adam = torch.optim.Adam(net.parameters(), lr=cfg.train.adam_lr)
    lbfgs = torch.optim.LBFGS(net.parameters(), line_search_fn="strong_wolfe", max_iter=20)
    opt = TwoPhaseOptimizer(adam, lbfgs)

    def _split(u):
        return list(torch.split(u, sizes))

    def closure():
        u = net()
        u_elems = _split(u)
        loss = (
            cd_residual(u, D1_global, D2_global, w_norm_global, eps, a_conv)
            + lambda_bc * cd_bc_loss(u)
        )
        if K > 1:
            loss = loss + lambda_int * all_interface_losses(u_elems, D1_elems, interface_cond)
        return loss

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
    errors = compute_errors(u, x_global, w_global, eps, a_conv)
    print(f"  L∞ = {errors['Linf']:.3e}   L² = {errors['L2']:.3e}")

    net.save(os.path.join(cfg.output.dir, cfg.output.checkpoint))

    # ── Loss landscape ─────────────────────────────────────────────────────────
    print("\n  Computing loss landscape …")

    def _landscape_closure():
        u = net()
        u_elems = _split(u)
        pde = float(cd_residual(u, D1_global, D2_global, w_norm_global, eps, a_conv))
        bc = float(lambda_bc * cd_bc_loss(u))
        result = {"pde": pde, "bc": bc}
        if K > 1:
            result["interface"] = float(
                lambda_int * all_interface_losses(u_elems, D1_elems, interface_cond)
            )
        return result

    surfaces = loss_landscape_scan([net], _landscape_closure, nr_steps=24)
    plot_landscape(
        surfaces,
        component_labels={"pde": "PDE residual  εu″+au′=0", "bc": "BC penalty",
                          "interface": f"Interface ({interface_cond.upper()})"},
        component_cmaps={"pde": "viridis", "bc": "inferno", "interface": "plasma"},
        title="Convection-Diffusion Loss Landscape",
        save_path=os.path.join(cfg.output.dir, "plots", "loss_landscape.png"),
    )

    if wandb_run is not None:
        import wandb
        wandb.log({f"final/{k}": v for k, v in errors.items()})
        wandb_run.finish()

    return errors


if __name__ == "__main__":
    main()
