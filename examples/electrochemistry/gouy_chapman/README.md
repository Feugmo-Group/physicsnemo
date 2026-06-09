# Gouy-Chapman Double Layer

Solves the 1D Poisson-Boltzmann equation for the Gouy-Chapman electrical
double layer using SCEN with strong KTE node clustering at the electrode wall.

## Physics

**Linearized (Debye-Hückel):**

    ψ'' = κ²ψ   on [0, L]
    ψ(0) = ψ₀,  ψ(L) = 0

Exact solution: `ψ*(x) = ψ₀·sinh(κ(L−x))/sinh(κL)`

**Nonlinear (sinh-Poisson):**

    ψ'' = κ²sinh(ψ)   on [0, L]
    ψ(0) = ψ₀,  ψ(L) = 0

The nonlinear form is physically relevant for large wall potentials
(ψ₀ ≫ kT/e, i.e. ψ₀ ≫ 1 in dimensionless units).

KTE node clustering (alpha = 0.9) at x = 0 resolves the Debye layer
where potential changes exponentially over a length scale κ⁻¹.

## Usage

```bash
# Linearized variant (has exact solution for validation):
python src/trainer.py

# Nonlinear variant:
python src/trainer.py train.variant=nonlinear physics.physics.psi_wall=4.0

# Enable W&B:
python src/trainer.py wandb.enabled=true wandb.entity=your-username
```
