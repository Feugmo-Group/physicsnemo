# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""3D steady PNP (K⁺/Cl⁻) trainer using DVRMapper3D.

Run:
    python src/trainer.py
    python src/trainer.py physics.domain.Nx=10 physics.domain.Ny=10 physics.domain.Nz=10
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

from physicsnemo.experimental.models.scen import DVRMapper3D, SCENElementNetwork
from physicsnemo.optim import TwoPhaseOptimizer

from src.metrics import compute_errors
from src.physics import pnp_3d_exact, pnp_3d_residuals


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

    mapper3d = DVRMapper3D(
        dom.Nx, dom.ax, dom.bx,
        Ny=dom.Ny, ay=dom.ay, by=dom.by,
        Nz=dom.Nz, az=dom.az, bz=dom.bz,
        alpha=dom.alpha, dtype=dtype_str,
    )
    xyz = mapper3d.xyz_nodes
    lap = mapper3d.laplacian
    D1x, D1y, D1z = mapper3d.D1x, mapper3d.D1y, mapper3d.D1z
    w = mapper3d.weights
    w_norm = w / w.sum()
    N_total = dom.Nx * dom.Ny * dom.Nz
    lambda_bc = float(phys.lambda_bc)

    flat_elem = [{"N": N_total, "a": -1.0, "b": 1.0}]
    model_kw = dict(
        hidden_dim=cfg.model.hidden_dim, n_layers=cfg.model.n_layers,
        backbone=cfg.model.backbone, poly_degree=cfg.model.poly_degree, dtype=dtype_str,
    )
    net_K = SCENElementNetwork(flat_elem, **model_kw)
    net_Cl = SCENElementNetwork(flat_elem, **model_kw)
    net_phi = SCENElementNetwork(flat_elem, **model_kw)

    all_params = (list(net_K.parameters()) + list(net_Cl.parameters()) + list(net_phi.parameters()))

    def bc_loss(c_K, c_Cl, phi):
        cK_bc, cCl_bc, phi_bc = pnp_3d_exact(xyz)
        # Boundary nodes: where any coordinate is 0 or 1
        tol = 1e-4
        is_bc = ((xyz[:, 0] < tol) | (xyz[:, 0] > 1.0 - tol) |
                 (xyz[:, 1] < tol) | (xyz[:, 1] > 1.0 - tol) |
                 (xyz[:, 2] < tol) | (xyz[:, 2] > 1.0 - tol))
        if is_bc.any():
            return (((c_K[is_bc] - cK_bc[is_bc])**2 +
                     (c_Cl[is_bc] - cCl_bc[is_bc])**2 +
                     (phi[is_bc] - phi_bc[is_bc])**2).mean())
        return torch.tensor(0.0, dtype=dtype_str)

    def closure():
        c_K = net_K()
        c_Cl = net_Cl()
        phi = net_phi()
        l_K, l_Cl, l_phi = pnp_3d_residuals(c_K, c_Cl, phi, lap, D1x, D1y, D1z, w_norm, xyz)
        return l_K + l_Cl + l_phi + lambda_bc * bc_loss(c_K, c_Cl, phi)

    adam = torch.optim.Adam(all_params, lr=cfg.train.adam_lr)
    lbfgs = torch.optim.LBFGS(all_params, line_search_fn="strong_wolfe", max_iter=20)
    opt = TwoPhaseOptimizer(adam, lbfgs)

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
        c_K, c_Cl, phi = net_K(), net_Cl(), net_phi()
    errors = compute_errors(c_K, c_Cl, phi, xyz, w)
    for k, v in errors.items():
        print(f"  {k}: {v:.3e}")

    for name, net in [("K", net_K), ("Cl", net_Cl), ("phi", net_phi)]:
        net.save(os.path.join(cfg.output.dir, cfg.output.checkpoint.replace(".mdlus", f"_{name}.mdlus")))

    if wandb_run is not None:
        import wandb
        wandb.log({f"final/{k}": v for k, v in errors.items()})
        wandb_run.finish()

    return errors


if __name__ == "__main__":
    main()
