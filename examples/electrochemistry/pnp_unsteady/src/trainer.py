# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""1D unsteady PNP trainer (space-time tensor product SCEN).

Three coupled field networks (cp, cn, phi) are evaluated on a (Nt × Nx)
tensor-product LGL grid.  Each network outputs a flat (Nt·Nx,) vector that
is reshaped to (Nt, Nx) for spectral differentiation.

Run:
    python src/trainer.py
    python src/trainer.py domain.Nx=24 domain.Nt=16 train.dtype=float64
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

from physicsnemo.experimental.models.scen import DVRMapper
from physicsnemo.experimental.models.scen import SCENElementNetwork
from physicsnemo.optim import TwoPhaseOptimizer

from src.metrics import compute_errors
from src.physics import pnp_unsteady_ic_loss, pnp_unsteady_residuals


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

    # ── Space-time grids ─────────────────────────────────────────────────────
    Nx, Nt = dom.Nx, dom.Nt
    mapper_x = DVRMapper(Nx, dom.ax, dom.bx, dom.alpha_x, dtype=dtype)
    mapper_t = DVRMapper(Nt, dom.at, dom.bt, dom.alpha_t, dtype=dtype)

    x_grid = mapper_x.nodes                     # (Nx,)
    t_grid = mapper_t.nodes                     # (Nt,)
    D1x, D2x = mapper_x.D1, mapper_x.D2        # (Nx, Nx)
    D1t = mapper_t.D1                           # (Nt, Nt)
    wx = mapper_x.weights / mapper_x.weights.sum()
    wt = mapper_t.weights / mapper_t.weights.sum()
    w_xt = wt.unsqueeze(1) * wx.unsqueeze(0)    # (Nt, Nx)

    # ── Field networks: one element covering the full (Nt·Nx,) flat domain ──
    # Each network takes a 1D normalised coordinate on [-1, 1] and outputs a
    # scalar. We tile the network over the flat grid and reshape for operators.
    flat_element = [{"N": Nx * Nt, "a": -1.0, "b": 1.0}]
    model_kwargs = dict(
        hidden_dim=cfg.model.hidden_dim, n_layers=cfg.model.n_layers,
        backbone=cfg.model.backbone, poly_degree=cfg.model.poly_degree, dtype=dtype_str,
    )
    net_cp = SCENElementNetwork(flat_element, **model_kwargs)
    net_cn = SCENElementNetwork(flat_element, **model_kwargs)
    net_phi = SCENElementNetwork(flat_element, **model_kwargs)

    lambda_ic = float(phys.lambda_ic)
    lambda_bc = float(phys.lambda_bc)

    all_params = list(net_cp.parameters()) + list(net_cn.parameters()) + list(net_phi.parameters())
    adam = torch.optim.Adam(all_params, lr=cfg.train.adam_lr)
    lbfgs = torch.optim.LBFGS(all_params, line_search_fn="strong_wolfe", max_iter=20)
    opt = TwoPhaseOptimizer(adam, lbfgs)

    def closure():
        cp = net_cp().view(Nt, Nx)
        cn = net_cn().view(Nt, Nx)
        phi = net_phi().view(Nt, Nx)
        l_cp, l_cn, l_phi = pnp_unsteady_residuals(cp, cn, phi, D1x, D2x, D1t, w_xt, x_grid, t_grid)
        ic = pnp_unsteady_ic_loss(cp, cn, phi, x_grid)
        return l_cp + l_cn + l_phi + lambda_ic * ic

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
        cp = net_cp().view(Nt, Nx)
        cn = net_cn().view(Nt, Nx)
        phi = net_phi().view(Nt, Nx)

    errors = compute_errors(cp, cn, phi, x_grid, t_grid, w_xt)
    for k, v in errors.items():
        print(f"  {k}: {v:.3e}")

    for name, net in [("cp", net_cp), ("cn", net_cn), ("phi", net_phi)]:
        net.save(os.path.join(cfg.output.dir, cfg.output.checkpoint.replace(".mdlus", f"_{name}.mdlus")))

    if wandb_run is not None:
        import wandb
        wandb.log({f"final/{k}": v for k, v in errors.items()})
        wandb_run.finish()

    return errors


if __name__ == "__main__":
    main()
