# 1D Unsteady Poisson-Nernst-Planck (Space-Time SCEN)

Solves the 1D time-dependent PNP system on (x, t) ∈ [0,1] × [0,1] using a
**space-time tensor product** SCEN formulation. Both spatial and temporal
derivatives are computed via precomputed LGL differentiation matrices.

## Physics

    ∂c_p/∂t = ∂²c_p/∂x² + ∇·(c_p ∇φ) + f₁(x,t)
    ∂c_n/∂t = ∂²c_n/∂x² − ∇·(c_n ∇φ) + f₂(x,t)
    ∂²φ/∂x²  = −c_p + c_n

**Exact (manufactured) solution:**

    c_p(x,t) = x²(1−x)² e^{−t}
    c_n(x,t) = x²(1−x)³ e^{−t}
    φ(x,t)   = −(10x⁷ − 28x⁶ + 21x⁵) e^{−t} / 420

## Method

Three SCENElementNetworks output flat (Nt·Nx,) vectors, reshaped to (Nt, Nx).
Spatial derivatives use D1x, D2x applied row-wise; time derivatives use D1t
applied along the time axis. No autograd differentiation is required.

## Usage

```bash
python src/trainer.py

# Higher resolution:
python src/trainer.py physics.domain.Nx=24 physics.domain.Nt=16 train.dtype=float64

# Enable W&B:
python src/trainer.py wandb.enabled=true wandb.entity=your-username
```
