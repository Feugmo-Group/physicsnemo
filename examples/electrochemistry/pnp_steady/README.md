# 1D Steady Poisson-Nernst-Planck

Solves the 1D steady coupled Poisson-Nernst-Planck (PNP) system on x ∈ [-3, 3]
using the **SCEN** (Spectral Collocation Element Network) approach with
Legendre-Gauss-Lobatto quadrature and a two-phase Adam → L-BFGS optimizer.

## Physics

Three coupled PDEs for cation concentration `c_p`, anion concentration `c_n`,
and electric potential `φ`:

```
  c_p''          = −π²(c_n + φ)
  3000 c_n'' + 100(c_n' c_p' + c_n c_p'') + f_v(x) = 0
  1000 φ''   +  50(φ'  c_p' + φ  c_p'') + f_w(x) = 0
```

**Exact solution** (manufactured):
```
  c_p(x) = sin(πx) + cos(πx)
  c_n(x) = sin(πx)
  φ(x)   = cos(πx)
```

**Boundary conditions** at x = ±3:
```
  c_p = -1,  c_n = 0,  φ = -1
```

## Method

- **Domain decomposition**: two elements [-3, 0] and [0, 3]
- **Quadrature**: LGL nodes with precomputed D1, D2 matrices (no autograd)
- **Optimization**: Adam (2000 steps) → L-BFGS (200 steps, strong Wolfe)
- **Architecture**: three independent MLP networks (one per field), shared geometry

## Usage

```bash
# From the example root directory:
python src/trainer.py

# Higher resolution (float64):
python src/trainer.py model.hidden_dim=128 train.n_adam=5000 train.dtype=float64

# KAN backbone:
python src/trainer.py model.backbone=kan model.poly_degree=6

# Enable W&B logging:
python src/trainer.py wandb.enabled=true wandb.entity=your-username
```

## Expected accuracy (N=16, float64)

| Field | L∞     | L²     |
|-------|--------|--------|
| c_p   | < 1e-5 | < 1e-6 |
| c_n   | < 1e-5 | < 1e-6 |
| φ     | < 1e-5 | < 1e-6 |

