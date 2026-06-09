# Steady Cahn-Hilliard (Conserved Phase-Field)

Solves the steady 1D Cahn-Hilliard equation on x ∈ [-1, 1] using SCEN with
LGL quadrature and 4th-order differentiation matrix D4.

## Physics

    ∇²μ = 0,   μ = f'(c) − ε²∇²c

where `f(c) = c²(1-c)²/4` is the standard double-well free energy.

Equivalently (pseudospectral form):

    R = D2 @ f'(c) − ε² D4 @ c = 0

**Boundary conditions** (no-flux, conserved dynamics):

    c'(-1) = c'(1) = 0
    μ'(-1) = μ'(1) = 0

The D4 = D2 @ D2 matrix gives exact spectral differentiation
for the 4th-order biharmonic operator — a key advantage of LGL quadrature.

## Usage

```bash
python src/trainer.py

# Sharper interface:
python src/trainer.py physics.physics.eps_sq=0.001

# Enable W&B:
python src/trainer.py wandb.enabled=true wandb.entity=your-username
```
