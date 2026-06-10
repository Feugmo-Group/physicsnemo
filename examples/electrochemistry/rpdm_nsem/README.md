# RPDM (Refined Point Defect Model) — Spectral NSEM/SCEN variant

Spectral-collocation reimplementation of the passive Refined Point Defect Model
for electrochemical oxide-film growth on iron. This is the **NSEM/SCEN** counterpart
of the autodiff PINN example in `examples/electrochemistry/rpdm/`: it solves the same
dimensionless PDE system but uses **precomputed spectral differentiation operators**
instead of automatic differentiation.

> **Status: EXPERIMENTAL.** The goal of this example is correctness of the spectral
> idiom and an end-to-end-runnable pipeline, *not* quantitative accuracy. The RPDM
> reaction/flux terms are extremely stiff (exponential prefactors spanning many
> decades); two of the boundary-flux residuals do not fully converge with the small
> default grid/network. See **Challenges & Notes** below.

## Physics

The dimensionless passive RPDM has four primary fields on a fixed reference
square `(x, y) ∈ [0, 1] × [0, yf]`:

- `cCV(x, y)` — cation-vacancy concentration
- `cAV(x, y)` — anion-vacancy concentration
- `phif(x, y)` — film potential
- `l(y)` — film thickness (a function of **time only**)

The physical film occupies `[0, l(y)]`; a **Landau transform** maps it onto the
fixed `x ∈ [0, 1]` reference interval. The moving boundary therefore enters the
equations only through `l(y)` and its time derivative `l_y(y)` — there is no
remeshing. Residuals (ported verbatim from `pdm/pdm.py`):

- **poisson**: `eps·φ_xx / l² + (z_CV·cCV + z_AV·cAV) = 0`
- **transport_CV / transport_AV**: drift-diffusion with the Landau convective term
  `−xi·x·l_y·c_x / l` plus electromigration `−eta·z·(c_x·φ_x + c·φ_xx)/l²` and the
  time term `xi·c_y`
- **film_growth** (ODE in `y`): `l_y = k₀R2̂_fg·e_R2·exp(...) − k_R5̂`, using
  `phimf = φ(x=0)`
- **interface fluxes** at `x=0` (R1, R2, mf_phif) and `x=1` (R3, R4, fs_phif)
- **initial conditions** at `y=0`: `cCV=cAV=0`, `φ = φ_ext − m·x`, `l = l_ini`

**Passive mode only.** The transpassive electron/hole transport (`transport_e`,
`transport_h`) and the `ce`/`ch` concentrations are dropped (see *What's simplified*).

## Spectral (NSEM) vs. autodiff (PINN) trade-off

| | autodiff PINN (`rpdm/`) | spectral NSEM (`rpdm_nsem/`) |
|---|---|---|
| Derivatives | `torch.autograd` on a random point cloud | precomputed `D1`, `D2` matrices via `DVRMapper` |
| Sampling | random collocation, resampled each step | fixed tensor-product LGL grid |
| Cost per step | graph build + backward through derivatives | dense matmuls (`field @ D.T`, `D @ field`) |
| Accuracy/node | algebraic | spectral (exponential for smooth fields) |
| Boundary layers | needs many points | KTE node clustering (`alpha_x`) near endpoints |
| Moving boundary | `l.diff(y)` via autograd | `l_y = D1y @ l` on the time grid |

The spectral approach trades flexible point sampling for a structured grid on which
derivatives are *exact linear operators*. For smooth fields this is far more accurate
per degree of freedom; the price is the tensor-product structure and the need to keep
the grid fixed.

## How the spectral discretization works

- The **spatial** direction uses a **multi-element mesh** on `x ∈ [0, 1]`: one
  `DVRMapper` per element (`conf/config.yaml: domain.x_elements`). With a large
  Poisson factor `eps` the film potential is near-electroneutral in the bulk and
  drops sharply in thin **space-charge (Debye) layers at both interfaces** — `x=0`
  (metal/film) and `x=1` (film/solution). A single spectral element cannot resolve
  those layers, so we place a refined element against each interface and a coarse
  bulk element between them:
  - `[0.0, 0.1]` metal/film layer — `mapping='log'` clusters nodes near `x=0`,
  - `[0.1, 0.9]` bulk film — uniform,
  - `[0.9, 1.0]` film/solution layer — `mapping='kte'` clusters near `x=1`.
- The global operators `D1x`, `D2x` are **block-diagonal** (`torch.block_diag` over
  the per-element matrices), so each element differentiates only itself. Element
  continuity is therefore imposed explicitly by a **C0/C1 interface penalty**
  (`rpdm_interface_loss`): at every internal interface and every time slice the
  field value and its slope are matched across neighbouring elements
  (`train.lambda_interface`, `train.interface_cond`).
- A **time** `DVRMapper(Nt, 0, yf, alpha_t)` gives `t_grid`, `D1y`.
- Fields are `(Nt, Nx)` tensors (`Nx` = total nodes over all elements). A spatial
  derivative is `field @ Dx.T`; a time derivative is `Dy @ field`. The film
  thickness `l(y)` is an `(Nt,)` vector and `l_y = D1y @ l`.
- Each field is a `SCENElementNetwork`: `cCV/cAV/phif` are flat `(Nt·Nx,)` networks
  reshaped to `(Nt, Nx)`; `l` is a separate time-only `(Nt,)` network.

**Tuning the mesh:** add elements or raise per-element `N`/`alpha` where a layer is
still under-resolved (the `flux_R2`/`flux_R3` reaction-source terms at the
interfaces are the most demanding). The C0/C1 penalty keeps the assembled solution
continuous as you refine.

## Run

```bash
cd examples/electrochemistry/rpdm_nsem
python src/trainer.py                              # full default run
python src/trainer.py train.n_adam=10 train.n_lbfgs=5   # smoke test
python src/trainer.py model.hidden_dim=64                # bigger per-field network
```

Key config (`conf/config.yaml`):

- `domain.x_elements` — list of spatial elements `{N, a, b, alpha, mapping}` (the
  multi-element mesh); `domain.{Nt, alpha_t, yf, Eext}` — time grid + potential
- `model.{hidden_dim, n_layers, backbone, poly_degree}` — per-field network
- `train.{n_pretrain, n_adam, n_lbfgs, lambda_ic, lambda_interface, interface_cond, dtype, seed}` — schedule + continuity weighting

## What's simplified / reduced

1. **Passive mode only.** Transpassive `transport_e`, `transport_h`, `ce`, `ch` and
   the `flux_ch` / `mf_ch` / `mf_ce` interface terms are removed. Only `cCV`, `cAV`,
   `phif`, `l` are solved.
2. **Constant external potential.** `Eext` is a single constant (the `const_0.1_V`
   case), not the stepped/ramped `Min/Max/floor` waveform of the source. This makes
   the reference-potential terms `phiext − phiext_ref = 0`, collapsing several
   exponential factors.
3. **`l = l(y)` only.** Following the source physics, film thickness is x-independent;
   it is modeled by a time-only network and broadcast over x. This makes `l_x = 0`.
4. **Soft initial/interface conditions.** ICs and interface fluxes are penalty terms
   (`lambda_ic`, BRDR-weighted residuals), not hard-constructed BCs like the
   autodiff example's ADF `HardDirichletBC`.
5. **`sqrt_lmd = √1e-8` residual weight** is kept on the stiff reaction/film terms
   (R1–R4, film_growth), exactly as in the source, to keep their magnitudes tractable.

## Expected behavior

With the full default run (2000-step IC pretrain → Adam → L-BFGS):

- The IC pretrain drives the seed MSE to ~1e-5.
- The interior residuals (`poisson`, `transport_CV/AV`), `film_growth`, `flux_R1`,
  `flux_R4`, and the `mf_phif`/`fs_phif` potential BCs converge to ~1e-3 or below.
- The film/solution source fluxes `flux_R2` and `flux_R3` remain at a stiffness
  floor (~1e5–1e6). These are constant reaction sources that must be balanced by a
  sharp vacancy boundary layer; resolving them needs more spatial nodes, stronger
  `alpha_x` clustering, and a longer L-BFGS phase.
- `l(y)` stays near `l_ini ≈ 1` (it grows very slowly on this nondimensional time
  scale), so the COMSOL film-thickness L2 error is large — accuracy is **not** the
  objective of this pass.

## Challenges & Notes (for future engineers)

- **Moving boundary on a spectral grid.** The Landau transform is the key enabler:
  the physical domain `[0, l(y)]` is mapped to a *fixed* `x ∈ [0, 1]`, so no
  remeshing is needed. The moving boundary appears purely algebraically as `1/l²`
  scalings and the convective term `x·l_y·c_x/l`. `l(y)` lives on its own time-only
  spectral network; `l_y = D1y @ l` reuses the same time differentiation matrix as
  the field time derivatives. `l` is kept strictly positive with
  `softplus(net) + 1e-3` so the `1/l` and `1/l²` terms never blow up during training.

- **SCEN APIs used.**
  - `DVRMapper(N, a, b, alpha)` → `.nodes`, `.D1`, `.D2`, `.weights` (used directly
    for the operator matrices and quadrature weights).
  - `SCENElementNetwork(element_configs, ...)` with a single flat element
    `[{"N": Nx*Nt, "a": -1, "b": 1}]` for the (x,y) fields and `[{"N": Nt, ...}]` for
    `l`. `forward()` returns the concatenated node values; we `.view(Nt, Nx)` (or
    `.view(Nt)`) to apply the operators. The network's own block-diagonal
    `D*_global` buffers are *not* used here — the residuals use the standalone
    `DVRMapper` operators, matching the pnp_unsteady pattern.
  - `set_default_dtype(float64)` — float64 is important; the exponential reaction
    terms overflow easily in float32.

- **Loss aggregation.** `BalancedResidualDecayRate` over all terms (10 residuals
  + IC + the C0/C1 `interface` continuity term when the mesh has >1 element). It is
  passed as the `aggregator` to `TwoPhaseOptimizer`, which flips it to
  `eval()` at the Adam→L-BFGS boundary so the Wolfe line search sees a stationary
  loss. BRDR balances *relative* decay rates; terms that barely move (the stiff
  fluxes) keep a high weight, which is the intended behavior but also why they
  dominate the reported scalar loss.

- **Stiffness / convergence observations.**
  - The reaction prefactors are enormous after nondimensionalisation
    (`k0R2_hat ~ 1e4`, `eR2 = exp(cR2·phiext_ref) ~ 1e3`, etc.). Dropping the
    source's `sqrt_lmd` weight makes the loss start at ~1e24; restoring it brings
    the start to ~1e16 and the post-pretrain Adam start to ~1e11.
  - The **IC pretrain is essential** (same lesson as `allen_cahn`): without it the
    coupled system sits in a flat basin Adam cannot escape. With it, the Adam phase
    decreases monotonically by several orders before L-BFGS.
  - `grad_clip` (default 1.0) is needed during Adam — early exponential residuals
    produce huge gradients.
  - **What to try next for accuracy:** the spatial mesh is already multi-element
    with refined elements at both interfaces (`domain.x_elements`) — add elements or
    raise per-element `N`/`alpha` near the film/solution interface (the `flux_R3`/
    `flux_R4` source fluxes are the most demanding), lengthen the L-BFGS phase, raise
    `lambda_interface`, or non-dimensionally rescale the R2/R3 source terms so BRDR
    can balance them against the diffusive flux.

## Files

- `src/physics.py` — `Parameters`, `NondimGroups`, `rpdm_residuals`, `rpdm_ic_loss`
- `src/trainer.py` — Hydra entry point, grids, networks, pretrain + two-phase optim
- `src/metrics.py` — film-thickness `l(y)` vs. COMSOL reference
- `conf/config.yaml` — domain / model / train config
- `data/const_0.1_V.csv` — COMSOL film-thickness reference (`T` [s], `L` [m])
