# 2D Unsteady Poisson-Nernst-Planck

Solves the 2D time-dependent PNP system on (x, y) ∈ [0,1]² using
**DVRMapper2D** for 2D Kronecker product operators and a space-time
tensor-product approach for the time direction.

## Physics

    ∂c_p/∂t = ∇²c_p + ∇·(c_p ∇φ) + f₁(x,y,t)
    ∂c_n/∂t = ∇²c_n − ∇·(c_n ∇φ) + f₂(x,y,t)
    ∇²φ = −c_p + c_n

**Exact (manufactured) solution:**

    c_p(x,y,t) = sin(πx) sin(πy) e^{−t}
    c_n(x,y,t) = cos(πx) cos(πy) e^{−t}
    φ(x,y,t)   = sin(πx) cos(πy) e^{−t} / (2π²)

## Method

DVRMapper2D provides `D1x`, `D1y`, and `laplacian = D2x + D2y` as 2D
Kronecker product matrices over the (Nx×Ny,) flat spatial grid.

## Usage

```bash
python src/trainer.py

# Higher resolution:
python src/trainer.py physics.domain.Nx=16 physics.domain.Ny=16 train.dtype=float64

# Enable W&B:
python src/trainer.py wandb.enabled=true wandb.entity=your-username
```
