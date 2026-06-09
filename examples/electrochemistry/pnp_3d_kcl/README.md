# 3D Steady PNP — KCl Electrolyte

Solves the 3D steady Poisson-Nernst-Planck system for a 1:1 electrolyte
(K⁺/Cl⁻) on the unit cube [0,1]³ using **DVRMapper3D** for 3D Kronecker
product differentiation matrices.

## Physics

Dimensionless steady-state PNP:

    −∇²c_K  − ∇·(c_K ∇φ) = f_K(x,y,z)
    −∇²c_Cl + ∇·(c_Cl ∇φ) = f_Cl(x,y,z)
    −∇²φ = c_K − c_Cl

**Manufactured exact solution:**

    c_K(x,y,z)  = 1 + 0.1 sin(πx) sin(πy) sin(πz)
    c_Cl(x,y,z) = 1 − 0.1 sin(πx) sin(πy) sin(πz)
    φ(x,y,z)    = 0.1 cos(πx) cos(πy) cos(πz) / (3π²)

## Method

DVRMapper3D provides D1x, D1y, D1z, and the 3D Laplacian via Kronecker
products over the (Nx×Ny×Nz,) flat grid.

## Usage

```bash
python src/trainer.py

# Higher resolution (warning: N³ scaling):
python src/trainer.py physics.domain.Nx=10 physics.domain.Ny=10 physics.domain.Nz=10

# Enable W&B:
python src/trainer.py wandb.enabled=true wandb.entity=your-username
```
