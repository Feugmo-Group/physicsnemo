# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Refined Point Defect Model (RPDM) — DVR-collocation (NSEM/SCEN) trainer.

Reimplements the passive RPDM on a space-time tensor-product DVR grid using
*precomputed differentiation operators* (no autodiff for the PDE).  Four field
networks are trained:

* ``cCV``, ``cAV``, ``phif`` — flat ``(Nt·Nx,)`` networks reshaped to ``(Nt, Nx)``
* ``l``                       — a time-only ``(Nt,)`` network (moving boundary)

Loss terms are balanced with :class:`BalancedResidualDecayRate` and optimised
with :class:`TwoPhaseOptimizer` (Adam -> L-BFGS).  An optional pretrain stage
seeds the networks to the initial-condition shape before physics training, which
is essential for this stiff system.

Run:
    python src/trainer.py
    python src/trainer.py train.n_adam=10 train.n_lbfgs=5
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import hydra  # noqa: E402
import torch  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402
from src.metrics import compute_film_metrics  # noqa: E402
from src.physics import (  # noqa: E402
    NondimGroups,
    Parameters,
    make_computations_by_name,
    make_interior_informer,
    rpdm_ic_loss,
    rpdm_interface_loss,
    rpdm_residuals,
)

from physicsnemo.experimental.models.scen import (  # noqa: E402
    DVRMapper,
    SCENElementNetwork,
)
from physicsnemo.optim import (  # noqa: E402
    TwoPhaseOptimizer,
    build_aggregator,
)
from physicsnemo.utils import save_checkpoint, set_default_dtype  # noqa: E402
from physicsnemo.utils.logging import PythonLogger  # noqa: E402


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    logger = PythonLogger("rpdm_nsem")
    logger.info("\n" + OmegaConf.to_yaml(cfg))

    torch.manual_seed(cfg.train.seed)
    dtype = torch.float64 if cfg.train.dtype == "float64" else torch.float32
    set_default_dtype(dtype)

    dom = cfg.domain

    # ── Physical / nondimensional groups ──────────────────────────────────────
    p = Parameters()
    yf = float(dom.yf)
    g = NondimGroups.from_parameters(p, Eext=float(dom.Eext))
    logger.info(
        f"Nondim groups: eps={g.eps:.3e} eta={g.eta:.3e} xiCV={g.xiCV:.3e} "
        f"k0R2_hat_fg={g.k0R2_hat_fg:.3e} kR5_hat={g.kR5_hat:.3e} lini={g.lini:.3e}"
    )

    # Symbolic RPDM equations -> per-equation Computations evaluated on the
    # precomputed-operator derivatives (see physics.rpdm_residuals).
    comps = make_computations_by_name(g)

    # ── Space-time DVR grids ──────────────────────────────────────────────────
    # Multi-element spatial mesh on x in [0, 1].  The film potential is near
    # electroneutral in the bulk (the Poisson factor eps is large) and drops
    # sharply inside thin space-charge layers at BOTH interfaces — x=0 (metal/
    # film) and x=1 (film/solution).  A single DVR element cannot resolve
    # those layers, so we place a refined element against each interface and a
    # coarser bulk element between them.  Block-diagonal global operators
    # (one DVRMapper per element) are tied together by C0/C1 continuity terms.
    x_mappers = [
        DVRMapper(
            int(e["N"]),
            float(e["a"]),
            float(e["b"]),
            float(e.get("alpha", 0.0)),
            mapping=str(e.get("mapping", "kte")),
            dtype=dtype,
        )
        for e in dom.x_elements
    ]
    elem_sizes = [m.nodes.numel() for m in x_mappers]
    # Internal-interface column indices (start of each element after the first).
    split_idx, _acc = [], 0
    for sz in elem_sizes[:-1]:
        _acc += sz
        split_idx.append(_acc)

    x_grid = torch.cat([m.nodes for m in x_mappers])  # (Nx,)  ordered 0 -> 1
    D1x = torch.block_diag(*[m.D1 for m in x_mappers])  # (Nx, Nx) block-diagonal
    D2x = torch.block_diag(*[m.D2 for m in x_mappers])
    wx = torch.cat([m.weights for m in x_mappers])
    wx = wx / wx.sum()
    Nx = x_grid.numel()

    mapper_t = DVRMapper(int(dom.Nt), 0.0, yf, float(dom.alpha_t), dtype=dtype)
    Nt = int(dom.Nt)
    t_grid = mapper_t.nodes  # (Nt,)
    D1y = mapper_t.D1
    wt = mapper_t.weights / mapper_t.weights.sum()
    w_xt = wt.unsqueeze(1) * wx.unsqueeze(0)  # (Nt, Nx)

    logger.info(
        f"Spatial mesh: {len(x_mappers)} elements, sizes={elem_sizes}, "
        f"Nx_total={Nx}, interfaces at columns {split_idx}"
    )

    # DVR-collocation informer for the full-grid interior residuals (the same
    # unified PhysicsInformer API as the autodiff RPDM example).  The boundary
    # and film-growth terms stay on the sliced-Computation path in rpdm_residuals.
    informer = make_interior_informer(g, D1x, D2x, D1y, device=str(x_grid.device))

    # ── Field networks ────────────────────────────────────────────────────────
    flat_element = [{"N": Nx * Nt, "a": -1.0, "b": 1.0}]
    time_element = [{"N": Nt, "a": -1.0, "b": 1.0}]
    model_kwargs = dict(
        hidden_dim=cfg.model.hidden_dim,
        n_layers=cfg.model.n_layers,
        backbone=cfg.model.backbone,
        poly_degree=cfg.model.poly_degree,
        dtype=cfg.train.dtype,
    )
    net_cCV = SCENElementNetwork(flat_element, **model_kwargs)
    net_cAV = SCENElementNetwork(flat_element, **model_kwargs)
    net_phi = SCENElementNetwork(flat_element, **model_kwargs)
    net_l = SCENElementNetwork(time_element, **model_kwargs)

    all_params = (
        list(net_cCV.parameters())
        + list(net_cAV.parameters())
        + list(net_phi.parameters())
        + list(net_l.parameters())
    )

    def fields():
        cCV = net_cCV().view(Nt, Nx)
        cAV = net_cAV().view(Nt, Nx)
        phi = net_phi().view(Nt, Nx)
        # l(y): time-only network, broadcast over x. Keep positive via softplus
        # offset so the film thickness never collapses to <= 0.
        lvec = torch.nn.functional.softplus(net_l().view(Nt)) + 1e-3
        return cCV, cAV, phi, lvec

    # ── Stage 1: pretrain to the initial-condition shape ──────────────────────
    # Seeds cCV=cAV=0, phif=phiext-m*x (independent of y), l=lini everywhere.
    # Without this the stiff coupled system sits in a flat, hard-to-escape basin.
    n_pretrain = int(cfg.train.get("n_pretrain", 0))
    if n_pretrain > 0:
        logger.info(f"Pre-training to IC shape ({n_pretrain} steps) …")
        phi_target = (g.phiext - g.m * x_grid).unsqueeze(0).expand(Nt, Nx)
        l_target = torch.full((Nt,), g.lini, dtype=dtype)
        pre_opt = torch.optim.Adam(all_params, lr=float(cfg.train.pretrain_lr))
        for step in range(n_pretrain):
            pre_opt.zero_grad()
            cCV, cAV, phi, lvec = fields()
            loss_pre = (
                (cCV**2).mean()
                + (cAV**2).mean()
                + ((phi - phi_target) ** 2).mean()
                + ((lvec - l_target) ** 2).mean()
            )
            loss_pre.backward()
            pre_opt.step()
            if step % max(1, n_pretrain // 5) == 0:
                logger.info(f"  pretrain {step:5d}: MSE = {loss_pre.item():.3e}")
        logger.info(f"  pretrain final: MSE = {loss_pre.item():.3e}")

    # ── Stage 2: physics training (BRDR + Adam -> L-BFGS) ─────────────────────
    lambda_ic = float(cfg.train.lambda_ic)
    lambda_int = float(cfg.train.get("lambda_interface", 100.0))
    interface_cond = str(cfg.train.get("interface_cond", "both"))
    has_interface = len(split_idx) > 0
    term_names = [
        "poisson",
        "transport_CV",
        "transport_AV",
        "film_growth",
        "flux_R1",
        "flux_R2",
        "mf_phif",
        "flux_R3",
        "flux_R4",
        "fs_phif",
        "ic",
    ]
    if has_interface:
        term_names.append("interface")
    brdr = build_aggregator(
        "brdr",
        all_params,
        num_losses=len(term_names),
        weights=[1.0] * len(term_names),
    )

    adam = torch.optim.Adam(all_params, lr=float(cfg.train.adam_lr))
    lbfgs = torch.optim.LBFGS(
        all_params,
        line_search_fn="strong_wolfe",
        max_iter=int(cfg.train.lbfgs_max_iter),
    )
    opt = TwoPhaseOptimizer(adam, lbfgs, aggregator=brdr)

    # TwoPhaseOptimizer's closure takes no step argument, so we keep an
    # incrementing counter here.  It only advances while BRDR is in training
    # mode (the Adam phase); during the L-BFGS phase the aggregator is in
    # eval() and freezes its state regardless, so reusing the same step across
    # the many line-search closure evaluations is safe.
    brdr_step = [0]

    def closure() -> torch.Tensor:
        cCV, cAV, phi, lvec = fields()
        terms = rpdm_residuals(
            cCV, cAV, phi, lvec, D1x, D2x, D1y, w_xt, x_grid, g, comps, informer
        )
        terms["ic"] = lambda_ic * rpdm_ic_loss(cCV, cAV, phi, lvec, x_grid, g)
        if has_interface:
            terms["interface"] = lambda_int * rpdm_interface_loss(
                [cCV, cAV, phi], D1x, split_idx, interface_cond
            )
        losses_dict = {name: terms[name] for name in term_names}
        total = brdr(losses_dict, brdr_step[0])
        if brdr.training:
            brdr_step[0] += 1
        return total

    os.makedirs(cfg.output.dir, exist_ok=True)
    history = opt.run(
        closure,
        n_adam_steps=int(cfg.train.n_adam),
        n_lbfgs_steps=int(cfg.train.n_lbfgs),
        verbose=True,
        log_every=int(cfg.output.log_every),
        grad_clip=float(cfg.train.get("grad_clip", 0)) or None,
    )
    logger.info(f"Final loss: {history[-1]['loss']:.4e}")

    # ── Evaluation ────────────────────────────────────────────────────────────
    with torch.no_grad():
        cCV, cAV, phi, lvec = fields()
        terms = rpdm_residuals(
            cCV, cAV, phi, lvec, D1x, D2x, D1y, w_xt, x_grid, g, comps, informer
        )
        for name, val in terms.items():
            logger.info(f"  residual[{name}] = {float(val):.3e}")
        if has_interface:
            interface_val = rpdm_interface_loss(
                [cCV, cAV, phi], D1x, split_idx, interface_cond
            )
            logger.info(f"  residual[interface] = {float(interface_val):.3e}")

    metrics = compute_film_metrics(lvec, t_grid, p.tc, p.lc)
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.3e}")

    field_nets = [net_cCV, net_cAV, net_phi, net_l]
    for name, net in zip(["cCV", "cAV", "phif", "l"], field_nets):
        net.save(
            os.path.join(
                cfg.output.dir,
                cfg.output.checkpoint.replace(".mdlus", f"_{name}.mdlus"),
            )
        )

    # Idiomatic PhysicsNeMo checkpoint: all four field nets + optimizer + final
    # loss/metrics metadata, written into the run's output directory.
    final_loss = float(history[-1]["loss"])
    save_checkpoint(
        cfg.output.dir,
        models=field_nets,
        optimizer=adam,
        epoch=int(cfg.train.n_adam),
        metadata={
            "final_loss": final_loss,
            **{k: float(v) for k, v in metrics.items()},
        },
    )

    return {"final_loss": final_loss, **metrics}


if __name__ == "__main__":
    main()
