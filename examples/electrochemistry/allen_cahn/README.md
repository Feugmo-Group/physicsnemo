# 1D Allen-Cahn Equation

Solves the steady 1D Allen-Cahn (non-conserved phase-field) equation on
x ∈ [-1, 1] using SCEN with LGL quadrature and a pretrain → L-BFGS protocol.

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
python src/trainer.py physics.physics.eps_sq=0.001 model.hidden_dim=128

# Enable W&B:
python src/trainer.py wandb.enabled=true wandb.entity=your-username
```

## Validated results (ε² = 0.01, N = 48, alpha = 0.7)

| Metric | Value |
|--------|-------|
| L∞     | 1.30e-3 |
| L²     | 7.96e-4 |

Training protocol: 15 000 pretrain steps (lr = 5e-4, cosine decay) → 2 L-BFGS
steps (max_iter = 100) to converge.

## Training protocol note

For stiff problems like Allen-Cahn with small ε, **Adam should be skipped
entirely** (`n_adam: 0` in `conf/config.yaml`).

The spectral second-derivative matrix D2 has eigenvalues of order N⁴ (roughly
5×10⁷ for N = 48). The Allen-Cahn PDE multiplies D2 by ε² = 0.01, leaving
condition numbers of order 10⁵. Even with gradient clipping, a single Adam
step can introduce high-frequency oscillations large enough to push the network
out of the tanh basin — after which the PDE residual may be small (multiple
solutions exist) but the solution is physically wrong.

The reliable protocol:

1. **Pretrain** the network to match the exact tanh shape directly (MSE < 1e-6).
   Use a cosine-annealing schedule to squeeze MSE below 5e-7 without
   oscillating.
2. **Skip Adam.** Set `n_adam: 0`. Adam's stochastic noise is harmful here.
3. **L-BFGS with max_iter = 100.** Starting from the pretrained weights (which
   are already in the correct basin), L-BFGS converges in 2–5 outer steps to
   physics-consistent spectral accuracy.
