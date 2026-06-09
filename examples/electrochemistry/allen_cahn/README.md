# 1D Allen-Cahn Equation

Solves the steady 1D Allen-Cahn (non-conserved phase-field) equation on
x ∈ [-1, 1] using SCEN with LGL quadrature and two-phase Adam → L-BFGS training.

## Physics

    ε²u'' − (u³ − u) = 0   on [-1, 1]
    u(-1) = -1,  u(1) = 1

**Exact solution:**

    u*(x) = tanh(x / (ε√2))

The solution has a sharp interface of width ~ε√2 centred at x = 0.
KTE node clustering (alpha = 0.7) is applied to resolve this interface.

## Usage

```bash
python src/trainer.py

# Sharper interface (small ε):
python src/trainer.py physics.physics.eps_sq=0.001 model.hidden_dim=128 train.n_adam=5000

# Enable W&B:
python src/trainer.py wandb.enabled=true wandb.entity=your-username
```
