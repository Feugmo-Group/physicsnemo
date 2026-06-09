# 2D Kolmogorov Flow (LegendreKAN Backbone)

Solves the 2D steady Kolmogorov flow problem using a **LegendreKAN** backbone
(`backbone="kan"`) in SCENElementNetwork.  This example demonstrates spectral
polynomial edge functions for physics-informed learning.

## Physics

Steady 2D Navier-Stokes in stream-function/vorticity form on [0,2π]²:

    −ν∇²ω + J(ψ, ω) = F(x, y)

where ω = ∇²ψ is vorticity, J(ψ,ω) the Jacobian, and forcing:

    F(x, y) = n_force · sin(n_force · y)

**Exact steady-state stream function** (valid for small Re = 1/ν):

    ψ*(x, y) = −sin(n_force · y) / (ν · n_force²)

## Method

- **DVRMapper2D** provides 2D Kronecker D1x, D1y, Laplacian
- **LegendreKAN** backbone: each edge is a degree-K Legendre polynomial series
- Derivatives via precomputed matrices (no autograd traversal through the network)
- Two-phase Adam → L-BFGS for fast convergence on the deterministic landscape

## Usage

```bash
# Default: KAN backbone, Re=10
python src/trainer.py

# Higher Reynolds number:
python src/trainer.py physics.physics.nu=0.01

# Compare with MLP backbone:
python src/trainer.py model.backbone=mlp model.hidden_dim=64

# Enable W&B:
python src/trainer.py wandb.enabled=true wandb.entity=your-username
```
