# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""1D steady Poisson-Nernst-Planck trainer.

Run from the example root directory:

    python src/trainer.py
    python src/trainer.py model.hidden_dim=128 train.n_adam=5000

The trainer builds three coupled SCENElementNetworks (one per field:
cp, cn, phi), assembles their operators from the shared DVRMapper, and
minimises the total PINN loss using TwoPhaseOptimizer (Adam → L-BFGS).
"""

from __future__ import annotations

import os
import sys

# Ensure the example root is on the path so `src.*` imports resolve
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from physicsnemo.experimental.models.scen import SCENElementNetwork
from physicsnemo.optim import TwoPhaseOptimizer

from src.metrics import compute_errors, print_errors
from src.physics import (
    make_pnp_steady_informer,
    pnp_steady_bc_loss,
    pnp_steady_residuals_dvr,
)


def _init_wandb(cfg: DictConfig) -> "wandb.Run | None":
    """Initialise W&B run if enabled; return run object or None."""
    if not cfg.wandb.enabled:
        return None
    try:
        import wandb
    except ImportError:
        print("wandb not installed — skipping W&B logging. Install with: pip install wandb")
        return None
    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity or None,
        name=cfg.wandb.name,
        tags=list(cfg.wandb.tags) if cfg.wandb.tags else [],
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    return run


def _build_element_configs(dom: DictConfig) -> list[dict]:
    """Build element_configs from domain config.

    Supports either a ``boundaries`` list (e.g. [-3,-2,-1,0,1,2,3]) or
    legacy single-element ``a``/``b`` keys.
    """
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
    # ── Domain setup ─────────────────────────────────────────────────────────
    dom = cfg.physics.domain
    element_configs = _build_element_configs(dom)

    model_kwargs = dict(
        element_configs=element_configs,
        hidden_dim=cfg.model.hidden_dim,
        n_layers=cfg.model.n_layers,
        backbone=cfg.model.backbone,
        poly_degree=cfg.model.poly_degree,
        dtype=dtype_str,
    )

    # Three coupled field networks sharing the same geometry
    net_cp = SCENElementNetwork(**model_kwargs)
    net_cn = SCENElementNetwork(**model_kwargs)
    net_phi = SCENElementNetwork(**model_kwargs)

    # Shared geometry tensors (same for all fields)
    x = torch.cat([m.nodes for m in net_cp.mappers])
    D1 = net_cp.D1_global
    D2 = net_cp.D2_global
    w = torch.cat([m.weights for m in net_cp.mappers])
    w_norm = w / w.sum()

    # DVR-collocation informer for the 3 coupled interior residuals.
    informer = make_pnp_steady_informer(D1, D2, device=str(x.device))

    lambda_bc = cfg.physics.physics.lambda_bc
    all_params = (
        list(net_cp.parameters())
        + list(net_cn.parameters())
        + list(net_phi.parameters())
    )
    adam = torch.optim.Adam(all_params, lr=cfg.train.adam_lr)
    lbfgs = torch.optim.LBFGS(all_params, line_search_fn="strong_wolfe", max_iter=20)
    opt = TwoPhaseOptimizer(adam, lbfgs)

    # ── Loss closure ─────────────────────────────────────────────────────────
    def closure() -> torch.Tensor:
        cp = net_cp()
        cn = net_cn()
        phi = net_phi()
        l_cp, l_cn, l_phi = pnp_steady_residuals_dvr(informer, cp, cn, phi, w_norm, x)
        bc = pnp_steady_bc_loss(cp, cn, phi)
        return l_cp + l_cn + l_phi + lambda_bc * bc

    # ── Train ─────────────────────────────────────────────────────────────────
    os.makedirs(cfg.output.dir, exist_ok=True)
    wandb_run = _init_wandb(cfg)

    history = opt.run(
        closure,
        n_adam_steps=cfg.train.n_adam,
        n_lbfgs_steps=cfg.train.n_lbfgs,
        verbose=True,
        log_every=cfg.output.log_every,
    )

    # Log training curve to W&B
    if wandb_run is not None:
        import wandb
        for record in history:
            wandb.log(
                {"loss": record["loss"], "phase": record["phase"]},
                step=record["step"],
            )

    print(f"\nFinal loss: {history[-1]['loss']:.4e}")

    # ── Evaluate ─────────────────────────────────────────────────────────────
    with torch.no_grad():
        cp = net_cp()
        cn = net_cn()
        phi = net_phi()

    errors = compute_errors(cp, cn, phi, x, w)
    print("\nFinal errors vs exact solution:")
    print_errors(errors)

    # ── Checkpoint ───────────────────────────────────────────────────────────
    ckpt_path = os.path.join(cfg.output.dir, cfg.output.checkpoint)
    net_cp.save(ckpt_path.replace(".mdlus", "_cp.mdlus"))
    net_cn.save(ckpt_path.replace(".mdlus", "_cn.mdlus"))
    net_phi.save(ckpt_path.replace(".mdlus", "_phi.mdlus"))
    print(f"\nCheckpoints saved to {cfg.output.dir}/")

    # Log final errors and close W&B run
    if wandb_run is not None:
        import wandb
        wandb.log({f"final/{k}": v for k, v in errors.items()})
        wandb_run.finish()

    return errors


if __name__ == "__main__":
    main()
