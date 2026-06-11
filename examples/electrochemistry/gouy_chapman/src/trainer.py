# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gouy-Chapman double-layer trainer (linearized & nonlinear Poisson-Boltzmann).

Run:
    python src/trainer.py                            # linearized (default)
    python src/trainer.py train.variant=nonlinear    # nonlinear sinh form
    python src/trainer.py physics.physics.psi_wall=5.0 train.variant=nonlinear
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

from src.metrics import compute_errors_linear
from src.physics import (
    interface_loss_list,
    make_pb_informer,
    pb_bc_loss,
    pb_residual_dvr,
)


def _init_wandb(cfg):
    if not cfg.wandb.enabled:
        return None
    try:
        import wandb
        return wandb.init(
            project=cfg.wandb.project, entity=cfg.wandb.entity or None,
            name=f"{cfg.wandb.name}_{cfg.train.variant}",
            tags=list(cfg.wandb.tags or []) + [cfg.train.variant],
            config=OmegaConf.to_container(cfg, resolve=True),
        )
    except ImportError:
        print("wandb not installed — skipping.")
        return None


def _build_element_configs(dom) -> list[dict]:
    """Build element_configs from either ``dom.elements`` list or legacy single-element keys."""
    N = int(dom.N_per_element)
    quad = str(dom.get("quadrature", "lgl"))
    mapping = str(dom.get("mapping", "kte"))
    if "elements" in dom:
        return [
            {"N": N, "a": float(e.a), "b": float(e.b),
             "alpha": float(e.alpha), "quadrature": quad, "mapping": mapping}
            for e in dom.elements
        ]
    # legacy single-element config
    return [{"N": N, "a": float(dom.a), "b": float(dom.b),
              "alpha": float(dom.alpha), "quadrature": quad, "mapping": mapping}]


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    torch.manual_seed(cfg.train.seed)
    dtype_str = cfg.train.dtype
    dtype = torch.float64 if dtype_str == "float64" else torch.float32

    dom = cfg.physics.domain
    phys = cfg.physics.physics
    psi_wall = float(phys.psi_wall)
    kappa = float(phys.kappa)
    kappa_sq = kappa ** 2
    lambda_bc = float(phys.lambda_bc)
    lambda_int = float(dom.get("lambda_interface", 100.0))
    interface_cond = str(dom.get("interface_cond", "both"))
    variant = cfg.train.variant

    element_configs = _build_element_configs(dom)
    K = len(element_configs)
    L = element_configs[-1]["b"]   # right end of domain
    print(f"  {K} element(s)  variant={variant}  L={L}  κ={kappa}")
    if K > 1:
        print(f"  Interface condition: {interface_cond}  λ_int={lambda_int}")

    net = SCENElementNetwork(
        element_configs,
        hidden_dim=cfg.model.hidden_dim, n_layers=cfg.model.n_layers,
        backbone=cfg.model.backbone, poly_degree=cfg.model.poly_degree,
        dtype=dtype_str,
    )

    D2_global = net.D2_global
    D1_elems = [net.mappers[k].D1 for k in range(K)]
    w_norm_global = net.w_norm_global
    x_global = torch.cat([m.nodes for m in net.mappers])
    w_global = torch.cat([m.weights for m in net.mappers])
    sizes = net.element_sizes

    # DVR-collocation informer (linear ψ″−κ²ψ or nonlinear ψ″−κ²sinh ψ).
    informer = make_pb_informer(
        kappa_sq, D2_global, nonlinear=(variant != "linear"), device=str(x_global.device)
    )

    def residual_fn(psi, _D2, w_norm, _kappa_sq):
        return pb_residual_dvr(informer, psi, w_norm)

    adam = torch.optim.Adam(net.parameters(), lr=cfg.train.adam_lr)
    lbfgs = torch.optim.LBFGS(net.parameters(), line_search_fn="strong_wolfe", max_iter=20)
    opt = TwoPhaseOptimizer(adam, lbfgs)

    def _split(psi):
        return list(torch.split(psi, sizes))

    def closure():
        psi = net()
        loss = (
            residual_fn(psi, D2_global, w_norm_global, kappa_sq)
            + lambda_bc * pb_bc_loss(psi, psi_wall)
        )
        if K > 1:
            psi_elems = _split(psi)
            loss = loss + lambda_int * interface_loss_list(psi_elems, D1_elems, interface_cond)
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
        psi = net()

    if variant == "linear":
        errors = compute_errors_linear(psi, x_global, w_global, psi_wall, kappa, L)
        print(f"  L∞ = {errors['Linf']:.3e}   L² = {errors['L2']:.3e}")

    ckpt = os.path.join(cfg.output.dir,
                        cfg.output.checkpoint.replace(".mdlus", f"_{variant}.mdlus"))
    net.save(ckpt)

    # ── Loss landscape ─────────────────────────────────────────────────────────
    print("\n  Computing loss landscape …")

    def _landscape_closure():
        psi = net()
        psi_elems = _split(psi)
        result = {
            "pde": float(residual_fn(psi, D2_global, w_norm_global, kappa_sq)),
            "bc":  float(lambda_bc * pb_bc_loss(psi, psi_wall)),
        }
        if K > 1:
            result["interface"] = float(
                lambda_int * interface_loss_list(psi_elems, D1_elems, interface_cond)
            )
        return result

    surfaces = loss_landscape_scan([net], _landscape_closure, nr_steps=24)
    plot_landscape(
        surfaces,
        component_labels={"pde": f"PDE  ψ″ = κ²{'sinh(ψ)' if variant=='nonlinear' else 'ψ'}",
                          "bc": "BC penalty", "interface": f"Interface ({interface_cond.upper()})"},
        component_cmaps={"pde": "viridis", "bc": "inferno", "interface": "plasma"},
        title=f"Gouy-Chapman Loss Landscape ({variant})",
        save_path=os.path.join(cfg.output.dir, "plots", "loss_landscape.png"),
    )

    if wandb_run is not None:
        import wandb
        if variant == "linear":
            wandb.log({f"final/{k}": v for k, v in errors.items()})
        wandb_run.finish()


if __name__ == "__main__":
    main()
