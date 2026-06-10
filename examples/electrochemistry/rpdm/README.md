# Refined Point Defect Model (RPDM) — PhysicsNeMo v2.0 PINN

A physics-informed neural network (PINN) for **electrochemical oxide-film
growth** on an iron electrode, based on the Refined Point Defect Model. Ported
to the PhysicsNeMo v2.0 idiom: an inline SymPy `PDE` subclass evaluated by
`PhysicsInformer`, plain `torch.nn` networks, and an explicit PyTorch training
loop.

## Physics summary

The model solves a coupled space-time system on the rectangle
`(x ∈ [0, 1], y ∈ [0, yf])`:

- `x` — nondimensional position across the oxide film. A **Landau /
  boundary-immobilization** transformation maps the moving physical domain
  `[0, L(t)]` onto the fixed `[0, 1]` interval, so the moving film/solution
  interface becomes a stationary boundary at `x = 1`.
- `y` — nondimensional time (`y = T / tc`).

**Solved (starred) network fields:** `cCV*`, `cAV*`, `phif*`, `l*`. After the
hard-BC layer these become the physical fields:

| Field   | Meaning                                            |
|---------|----------------------------------------------------|
| `cCV`   | cation-vacancy concentration                       |
| `cAV`   | anion-vacancy concentration                        |
| `phif`  | film electrostatic potential                       |
| `l`     | dimensionless film thickness `L(t)/lc`             |
| `phimf` | potential nearest the metal/film interface (x→0)   |
| `phifs` | potential nearest the film/solution interface (x→1)|

**Equations (passive mode):**

- Interior: `poisson`, `transport_CV`, `transport_AV` (Landau-transformed
  drift-diffusion), and the `film_growth` ODE in time.
- Metal/film interface (`x = 0`): `flux_R1`, `flux_R2`, `mf_phif`.
- Film/solution interface (`x = 1`): `flux_R3`, `flux_R4`, `fs_phif`.
- Initial conditions (`y = 0`): enforced **exactly** by the hard-BC layer.

## Nondimensionalization

| Dimensional variable | Characteristic value                | Dimensionless |
|----------------------|-------------------------------------|---------------|
| `Φf` [V]             | `Φc` (= RT/F or time-avg of `Eext`) | `φf`          |
| `C_CV` [mol/m³]      | `ρ` (= 1/Ω)                         | `c_CV`        |
| `C_AV` [mol/m³]      | `ρ`                                 | `c_AV`        |
| `L(t)` [m]           | `lc` (characteristic thickness)     | `ℓ`           |
| `E_ext` [V]          | `Φc`                                | `φ_ext`       |
| `X` [m]              | `L(t)` (Landau)                     | `x`           |
| `T` [s]              | `tc` (characteristic time)          | `y` (= `t`)   |

The applied potential `Eext(y)` is a symbolic ramped step function. With the
default parameters (`Ts ≫ yf`) it holds constant at `Eb = 0.1 V` throughout
training — the constant-voltage case matching the COMSOL reference.

## Hard boundary conditions (ADF)

Dirichlet conditions are enforced **exactly** (not as soft penalties) with
approximate distance functions (ADFs). `src/hard_bc.py` reimplements the two
ADF primitives `line_segment_adf` and `r_equivalence` as pure-torch functions,
then `enforce_hard_bc` builds each field as `g + ω·net`, where `ω` is the ADF
(zero on the boundary) and `g` the prescribed boundary value. At `y = 0` this
gives `cCV = cAV = 0`, `l = lini`, and `phif = φ_ext − m·x` exactly.

## Run

```bash
# from this directory
python src/trainer.py                       # full run (20k steps)
python src/trainer.py training.max_steps=50 # quick smoke test
python src/trainer.py optimizer=muon_adam   # swap optimizer
python src/trainer.py model.layer_size=128 model.nr_layers=4
```

Config lives in `conf/config.yaml` (model size, optimizer group, scheduler,
`training.max_steps`/`log_freq`, per-region `batch_size`, `custom.transpassive`).

## Expected behaviour

The film thickness `L(t)` (the `l` field de-nondimensionalized by `lc`) should
track the COMSOL reference in `data/const_0.1_V.csv` (columns `T` [s], `L` [m]).
`src/metrics.py:film_thickness_error` reports the L-∞ and relative-L2 error.

The total residual loss is **stiff**: the Poisson term carries a factor
`eps = F·lc²/(Φc·εf) ≈ 1e8`, so absolute loss values start very large
(`~1e11`). `BalancedResidualDecayRate` (BRDR) equalizes the *decay rates*
across terms rather than their magnitudes, which is the right tool here. Over
a short smoke run the loss is finite and trends downward; full convergence
needs the complete 20k-step schedule (and is sensitive to network size).

## Simplifications (documented)

- **Passive mode only** (`custom.transpassive=false`). The transpassive regime
  (electron/hole transport: `transport_e`, `transport_h`, `mf_ce`, `mf_ch`,
  `flux_ch`, `initial_ce/ch`, and the `chfs` hard-BC field) is intentionally
  **not** ported in this first pass. Passive mode is the regime exercised by
  the constant-0.1 V COMSOL validation case. Requesting `transpassive=true`
  raises `NotImplementedError`.
- `phimf` / `phifs` are taken as the in-batch samples nearest `x=0` / `x=1`
  (matching the source) and are detached, so `PhysicsInformer` treats them as
  data rather than differentiable fields.

## Files

```
rpdm/
├── conf/
│   ├── config.yaml
│   └── optimizer/{adam.yaml, muon_adam.yaml}
├── data/const_0.1_V.csv        # COMSOL reference L(t)
├── src/
│   ├── physics.py              # Parameters, make_Eext, PointDefectModel(PDE)
│   ├── hard_bc.py              # pure-torch ADF helpers + enforce_hard_bc
│   ├── trainer.py              # explicit PINN training loop
│   └── metrics.py              # film-thickness error vs COMSOL
└── README.md
```
