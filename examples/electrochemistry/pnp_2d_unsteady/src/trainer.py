# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""2D unsteady PNP trainer using DVRMapper2D.

Spatial derivatives are computed via precomputed 2D Kronecker operators
from DVRMapper2D.  Time derivatives use a separate DVRMapper for the time
axis (space-time tensor product approach).

Run:
    python src/trainer.py
    python src/trainer.py physics.domain.Nx=16 physics.domain.Ny=16
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

from physicsnemo.experimental.models.scen import DVRMapper, DVRMapper2D, SCENElementNetwork
from physicsnemo.optim import TwoPhaseOptimizer

from src.metrics import compute_errors
from src.physics import pnp_2d_exact, pnp_2d_residuals


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

    # ── Spatial grid (2D) ────────────────────────────────────────────────────
    mapper2d = DVRMapper2D(
        dom.Nx, dom.ax, dom.bx,
        Ny=dom.Ny, ay=dom.ay, by=dom.by,
        alpha_x=dom.alpha_x, alpha_y=dom.alpha_y,
        dtype=dtype_str,
    )
    xy = mapper2d.xy_nodes          # (Nx*Ny, 2)
    D1x, D1y = mapper2d.D1x, mapper2d.D1y
    lap = mapper2d.laplacian
    w2d = mapper2d.weights
    w_norm = w2d / w2d.sum()
    N_spatial = mapper2d.Nx * mapper2d.Ny

    # ── Time grid ────────────────────────────────────────────────────────────
    mapper_t = DVRMapper(dom.Nt, dom.at, dom.bt, dtype=dtype_str)
    t_grid = mapper_t.nodes         # (Nt,)
    D1t = mapper_t.D1               # (Nt, Nt)
    Nt = dom.Nt

    # ── Networks: flat space-time grids ─────────────────────────────────────
    flat_elem = [{"N": N_spatial * Nt, "a": -1.0, "b": 1.0}]
    model_kw = dict(
        hidden_dim=cfg.model.hidden_dim, n_layers=cfg.model.n_layers,
        backbone=cfg.model.backbone, poly_degree=cfg.model.poly_degree, dtype=dtype_str,
    )
    net_cp = SCENElementNetwork(flat_elem, **model_kw)
    net_cn = SCENElementNetwork(flat_elem, **model_kw)
    net_phi = SCENElementNetwork(flat_elem, **model_kw)

    lambda_ic = float(phys.lambda_ic)
    wt = mapper_t.weights / mapper_t.weights.sum()   # (Nt,)
    w_xyt = (wt.unsqueeze(1) * w_norm.unsqueeze(0)).reshape(-1)  # (Nt*N_spatial,)

    def closure():
        cp_flat = net_cp()   # (Nt*N_spatial,)
        cn_flat = net_cn()
        phi_flat = net_phi()

        # Reshape to (Nt, N_spatial), apply time deriv row-wise
        cp_2t = cp_flat.view(Nt, N_spatial)
        cn_2t = cn_flat.view(Nt, N_spatial)
        phi_2t = phi_flat.view(Nt, N_spatial)
        dcp_dt_2t = D1t @ cp_2t    # (Nt, N_spatial)
        dcn_dt_2t = D1t @ cn_2t

        # Aggregate residual over all time slices
        total_loss = torch.tensor(0.0, dtype=dtype_str)
        for i, t_val in enumerate(t_grid.tolist()):
            cp_i = cp_2t[i]
            cn_i = cn_2t[i]
            phi_i = phi_2t[i]
            dcp_i = dcp_dt_2t[i]
            dcn_i = dcn_dt_2t[i]
            l_cp, l_cn, l_phi = pnp_2d_residuals(
                cp_i, cn_i, phi_i, dcp_i, dcn_i, lap, D1x, D1y, w_norm, xy, t_val
            )
            total_loss = total_loss + wt[i] * (l_cp + l_cn + l_phi)

        # IC: fields at t[0] match exact solution
        t0 = t_grid[0].item()
        cp0_ex, cn0_ex, phi0_ex = pnp_2d_exact(xy, t0)
        ic = ((cp_2t[0] - cp0_ex)**2 + (cn_2t[0] - cn0_ex)**2 + (phi_2t[0] - phi0_ex)**2).mean()
        return total_loss + lambda_ic * ic

    os.makedirs(cfg.output.dir, exist_ok=True)
    wandb_run = _init_wandb(cfg)

    history = opt_runner = TwoPhaseOptimizer(
        torch.optim.Adam(
            list(net_cp.parameters()) + list(net_cn.parameters()) + list(net_phi.parameters()),
            lr=cfg.train.adam_lr,
        ),
        torch.optim.LBFGS(
            list(net_cp.parameters()) + list(net_cn.parameters()) + list(net_phi.parameters()),
            line_search_fn="strong_wolfe", max_iter=20,
        ),
    )
    history = opt_runner.run(
        closure, n_adam_steps=cfg.train.n_adam, n_lbfgs_steps=cfg.train.n_lbfgs,
        verbose=True, log_every=cfg.output.log_every,
    )

    if wandb_run is not None:
        import wandb
        for r in history:
            wandb.log({"loss": r["loss"], "phase": r["phase"]}, step=r["step"])

    print(f"\nFinal loss: {history[-1]['loss']:.4e}")

    t_eval = float(cfg.train.t_eval)
    with torch.no_grad():
        t_idx = (t_grid - t_eval).abs().argmin().item()
        cp = net_cp().view(Nt, N_spatial)[t_idx]
        cn = net_cn().view(Nt, N_spatial)[t_idx]
        phi = net_phi().view(Nt, N_spatial)[t_idx]

    errors = compute_errors(cp, cn, phi, xy, w2d, t_eval)
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
