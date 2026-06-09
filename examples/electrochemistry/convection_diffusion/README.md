# 1D Convection-Diffusion (Boundary Layer Test)

Solves the 1D convection-diffusion equation on x ∈ [0, 1] with a boundary
layer at x = 1. Uses KTE node clustering to resolve exponential layers with
very few degrees of freedom.

## Physics

    ε u'' + a u' = 0   on [0, 1]
    u(0) = 0,  u(1) = 1

**Exact solution:**

    u(x) = expm1(a·x/ε) / expm1(a/ε)

Boundary layer thickness scales as ε/a. For ε = 0.01 the layer spans ~1%
of the domain, requiring KTE clustering (alpha = 0.85) at the right wall.

## Usage

```bash
python src/trainer.py

# Sharper layer (Pe = a/ε = 1000):
python src/trainer.py physics.physics.eps=0.001

# Enable W&B:
python src/trainer.py wandb.enabled=true wandb.entity=your-username
```
