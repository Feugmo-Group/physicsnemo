"""Experiment E2 v5 — streaming multi-geometry PINN with physicsnemo DataLoader.

Scales to hundreds of real CT scan volumes by streaming one volume at a time
from disk using physicsnemo's CUDA-stream-prefetching DataLoader.

Pipeline:
    preprocess_volumes.py  →  volume_001.npz, volume_002.npz, ...
        ↓
    NumpyReader(preproc_dir/)           — reads one .npz per step from disk
        ↓
    Dataset(device="cuda", prefetch=2)  — CUDA streams: loads vol i+1 while
                                          GPU trains on vol i
        ↓
    DataLoader(batch_size=1, shuffle=True)
        ↓
    FiLM PINN  (same E2FiLMMLP as v4)

Memory model: only 1–2 volumes in GPU RAM at any time.
Scales to 500+ real CT scans without modification.

Usage:
    # Step 1: preprocess (once)
    python -m ct_to_physicsnemo.preprocess_volumes \\
        --library-csv runs/synthetic_library/labels.csv \\
        --output-dir  runs/preproc_library

    # Step 2: train
    python -m ct_to_physicsnemo.e2_v5_streaming
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from physicsnemo.datapipes import DataLoader, Dataset, NumpyReader

from ct_to_physicsnemo.voxel_graph_laplace import solve_voxel_graph_laplace_for_tau

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TOP_BC     = -0.5
BOT_BC     =  0.5
DELTA_C    = abs(BOT_BC - TOP_BC)
INLET_AXIS = 0


# ---------------------------------------------------------------------------
# RFF encoder
# ---------------------------------------------------------------------------

class RFFEncoder(nn.Module):
    def __init__(self, sigma: float = 6.0, n_freqs: int = 128, in_dim: int = 4, seed: int = 42):
        super().__init__()
        rng = torch.Generator()
        rng.manual_seed(seed)
        B = torch.randn(in_dim, n_freqs, generator=rng) * sigma
        self.register_buffer("B", B)

    @property
    def out_dim(self) -> int:
        return 2 * int(self.B.shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = x @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


# ---------------------------------------------------------------------------
# FiLM-conditioned network (same architecture as v4)
# ---------------------------------------------------------------------------

class FiLMLayer(nn.Module):
    def __init__(self, hidden_dim: int, descriptor_dim: int):
        super().__init__()
        self.gamma = nn.Linear(descriptor_dim, hidden_dim)
        self.beta  = nn.Linear(descriptor_dim, hidden_dim)
        nn.init.zeros_(self.gamma.weight); nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight);  nn.init.zeros_(self.beta.bias)

    def forward(self, h: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        return h * (1.0 + self.gamma(g)) + self.beta(g)


class E2FiLMMLP(nn.Module):
    def __init__(self, rff_out_dim: int, descriptor_dim: int = 127,
                 hidden: int = 256, layers: int = 6):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(rff_out_dim, hidden), nn.Tanh())
        self.linears = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers - 1)])
        self.films   = nn.ModuleList([FiLMLayer(hidden, descriptor_dim) for _ in range(layers - 1)])
        self.out     = nn.Linear(hidden, 1)

    def forward(self, rff: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(rff)
        for lin, film in zip(self.linears, self.films):
            h = film(torch.tanh(lin(h)), g)
        return self.out(h)


# ---------------------------------------------------------------------------
# Per-step helpers (operate on one volume's TensorDict batch)
# ---------------------------------------------------------------------------

def _sample_batch(sample: dict, batch_per_vol: int, device: str):
    """Extract a random mini-batch of pore voxels from one streamed volume."""
    pore_idx    = sample["pore_idx"].squeeze(0).to(device)      # (N_pore, 3) int16→long
    teacher     = sample["teacher"].squeeze(0).to(device)       # (N_pore,)
    sdf         = sample["sdf"].squeeze(0).to(device)           # (N_pore,)
    descriptor  = sample["descriptor"].squeeze(0).to(device)    # (127,)
    inlet_mask  = sample["inlet_mask"].squeeze(0).to(device)    # (N_pore,) bool
    outlet_mask = sample["outlet_mask"].squeeze(0).to(device)   # (N_pore,) bool
    shape       = tuple(sample["shape"].squeeze(0).tolist())
    epsilon     = float(sample["epsilon"].squeeze())
    tau_cg      = float(sample["tau_cg"].squeeze())

    pore_idx = pore_idx.long()
    N        = pore_idx.shape[0]
    n        = min(batch_per_vol, N)

    idx      = torch.randint(0, N, (n,), device=device)
    sel_idx  = pore_idx[idx]          # (n, 3)

    shape_t  = torch.tensor(shape, dtype=torch.float32, device=device)
    xyz      = (sel_idx.float() + 0.5) / shape_t               # (n, 3)
    sdf_sel  = sdf[idx].unsqueeze(1)                            # (n, 1)
    xyzs     = torch.cat([xyz, sdf_sel], dim=1)                 # (n, 4)
    teacher_sel = teacher[idx].unsqueeze(1)                     # (n, 1)
    g        = descriptor.unsqueeze(0).expand(n, -1)            # (n, 127)

    # BC samples
    inlet_ids  = torch.where(inlet_mask)[0]
    outlet_ids = torch.where(outlet_mask)[0]
    n_bc       = min(512, len(inlet_ids))

    in_sel  = inlet_ids [torch.randint(0, len(inlet_ids),  (n_bc,), device=device)]
    out_sel = outlet_ids[torch.randint(0, len(outlet_ids), (n_bc,), device=device)]

    def bc_xyzs(bc_idx):
        bc_coords = pore_idx[bc_idx]
        xyz_bc    = (bc_coords.float() + 0.5) / shape_t
        sdf_bc    = sdf[bc_idx].unsqueeze(1)
        return torch.cat([xyz_bc, sdf_bc], dim=1)

    return {
        "xyzs":        xyzs,
        "g":           g,
        "teacher_sel": teacher_sel,
        "in_xyzs":     bc_xyzs(in_sel),
        "out_xyzs":    bc_xyzs(out_sel),
        "n_bc":        n_bc,
        "descriptor":  descriptor,
        "epsilon":     epsilon,
        "tau_cg":      tau_cg,
        "shape":       shape,
        "pore_idx":    pore_idx,
        "sdf":         sdf,
        "inlet_mask":  inlet_mask,
        "outlet_mask": outlet_mask,
    }


def estimate_tau(model, rff_enc, pore_idx, sdf, descriptor, shape,
                 epsilon, device, batch_size=50_000):
    """TauFactor-style slice flux from trained model."""
    axis      = INLET_AXIS
    other     = [a for a in range(3) if a != axis]
    full_area = shape[other[0]] * shape[other[1]]
    shape_t   = torch.tensor(shape, dtype=torch.float32, device=device)

    mask_np   = torch.zeros(shape, dtype=torch.bool, device=device)
    mask_np[pore_idx[:, 0], pore_idx[:, 1], pore_idx[:, 2]] = True
    mask_np   = mask_np.cpu().numpy()

    flux_per_slice = []
    g_1 = descriptor.unsqueeze(0)

    with torch.no_grad():
        for s in range(shape[axis] - 1):
            sl = [slice(None)] * 3; sl[axis] = s
            sr = [slice(None)] * 3; sr[axis] = s + 1
            conducting = (
                torch.tensor(mask_np[tuple(sl)], dtype=torch.bool, device=device) &
                torch.tensor(mask_np[tuple(sr)], dtype=torch.bool, device=device)
            )
            coords_2d = torch.nonzero(conducting, as_tuple=False)
            if coords_2d.shape[0] == 0:
                flux_per_slice.append(0.0)
                continue

            idx_l = torch.zeros((coords_2d.shape[0], 3), dtype=torch.long, device=device)
            idx_r = torch.zeros_like(idx_l)
            idx_l[:, axis] = s;   idx_r[:, axis] = s + 1
            idx_l[:, other[0]] = coords_2d[:, 0]; idx_r[:, other[0]] = coords_2d[:, 0]
            idx_l[:, other[1]] = coords_2d[:, 1]; idx_r[:, other[1]] = coords_2d[:, 1]

            # SDF lookup needs full 3D grid  — rebuild from sparse pore_idx
            sdf_grid = torch.zeros(shape, dtype=torch.float32, device=device)
            sdf_grid[pore_idx[:, 0], pore_idx[:, 1], pore_idx[:, 2]] = sdf

            total_flux = 0.0
            for start in range(0, idx_l.shape[0], batch_size):
                def make_xyzs(idx):
                    xyz = (idx.float() + 0.5) / shape_t
                    s_  = sdf_grid[idx[:, 0], idx[:, 1], idx[:, 2]].unsqueeze(1)
                    return torch.cat([xyz, s_], dim=1)
                n  = idx_l[start:start+batch_size].shape[0]
                g  = g_1.expand(n, -1)
                c_l = model(rff_enc(make_xyzs(idx_l[start:start+batch_size])), g)
                c_r = model(rff_enc(make_xyzs(idx_r[start:start+batch_size])), g)
                total_flux += (c_r - c_l).sum().item()

            flux_per_slice.append(abs(total_flux) / full_area)

    fps    = np.array(flux_per_slice)
    mean_f = float(np.mean(fps))
    D_rel  = mean_f * shape[axis] / DELTA_C
    tau    = epsilon / D_rel if D_rel > 0 else float("inf")
    return tau, D_rel


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_e2_v5(
    preproc_dir:      str | Path = "runs/preproc_library",
    epochs:           int        = 5,                   # passes over all volumes
    batch_per_vol:    int        = 8_000,               # voxels per volume per step
    lr:               float      = 3e-4,
    rff_sigma:        float      = 6.0,
    rff_n_freqs:      int        = 128,
    teacher_weight:   float      = 500.0,
    bc_weight:        float      = 50.0,
    descriptor_dim:   int        = 127,
    hidden:           int        = 256,
    layers:           int        = 6,
    prefetch_factor:  int        = 2,
    num_streams:      int        = 4,
    results_csv:      str | Path = "runs/e2_results/e2_v5_results.csv",
    checkpoint_path:  str | Path = "runs/e2_results/e2_v5_model.pt",
    log_every_vol:    int        = 5,
) -> pd.DataFrame:

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*65}")
    print("EXPERIMENT E2 v5 — streaming FiLM PINN (physicsnemo DataLoader)")
    print(f"{'='*65}")
    print(f"device: {device}  epochs: {epochs}  batch_per_vol: {batch_per_vol}")

    preproc_dir = Path(preproc_dir)
    npz_files   = sorted(preproc_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files in {preproc_dir}. Run preprocess_volumes.py first.")
    print(f"Found {len(npz_files)} preprocessed volumes in {preproc_dir}")

    # ---- physicsnemo DataLoader ----
    # NumpyReader streams one .npz per sample; Dataset prefetches on CUDA streams
    reader  = NumpyReader(
        preproc_dir,
        fields=["pore_idx", "teacher", "sdf", "descriptor",
                "inlet_mask", "outlet_mask", "shape", "epsilon", "tau_cg"],
        file_pattern="*.npz",
        pin_memory=True,          # pinned memory → faster GPU transfer
    )
    dataset = Dataset(
        reader,
        device=device,
        num_workers=2,
    )
    loader  = DataLoader(
        dataset,
        batch_size=1,             # one volume per step
        shuffle=True,
        prefetch_factor=prefetch_factor,
        num_streams=num_streams,
    )
    print(f"DataLoader: {len(loader)} volumes/epoch  prefetch={prefetch_factor}  streams={num_streams}")

    # ---- build model ----
    rff_enc = RFFEncoder(sigma=rff_sigma, n_freqs=rff_n_freqs, in_dim=4, seed=42).to(device)
    model   = E2FiLMMLP(
        rff_out_dim=rff_enc.out_dim,
        descriptor_dim=descriptor_dim,
        hidden=hidden,
        layers=layers,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    total_steps = epochs * len(npz_files)
    opt       = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=lr / 20)

    # adaptive EMA per volume (warm-started on first encounter)
    ema_alpha   = 0.02
    ema_teacher : dict[str, float] = {}
    ema_bc      : dict[str, float] = {}

    print(f"\nTraining  {epochs} epochs × {len(npz_files)} volumes = {total_steps} steps")
    print("-" * 65)

    t_start  = time.perf_counter()
    step     = 0
    vol_seen : set[str] = set()

    for epoch in range(epochs):
        loader.set_epoch(epoch)
        epoch_loss = 0.0

        for vol_i, (batch, meta) in enumerate(loader):
            name = meta[0].get("source_filename", f"vol_{vol_i}").replace(".npz", "")
            b    = _sample_batch(batch, batch_per_vol, device)

            opt.zero_grad()

            # forward pass
            rff      = rff_enc(b["xyzs"])
            g        = b["g"]
            c_pred   = model(rff, g)

            loss_teacher = torch.mean((c_pred - b["teacher_sel"]) ** 2)

            g_bc   = b["descriptor"].unsqueeze(0).expand(b["n_bc"], -1)
            c_in   = model(rff_enc(b["in_xyzs"]),  g_bc)
            c_out  = model(rff_enc(b["out_xyzs"]), g_bc)
            loss_bc = torch.mean((c_in - TOP_BC)**2) + torch.mean((c_out - BOT_BC)**2)

            # warm-start EMA on first encounter
            lt = loss_teacher.item()
            lb = loss_bc.item()
            if name not in ema_teacher:
                ema_teacher[name] = max(lt, 1e-12)
                ema_bc[name]      = max(lb, 1e-12)
                vol_seen.add(name)
            else:
                ema_teacher[name] = (1-ema_alpha)*ema_teacher[name] + ema_alpha*max(lt, 1e-12)
                ema_bc[name]      = (1-ema_alpha)*ema_bc[name]      + ema_alpha*max(lb, 1e-12)

            w_t   = teacher_weight / ema_teacher[name]
            w_b   = bc_weight      / ema_bc[name]
            loss  = w_t * loss_teacher + w_b * loss_bc

            loss.backward()
            opt.step()
            scheduler.step()

            epoch_loss += loss.item()
            step += 1

            if vol_i % log_every_vol == 0:
                elapsed = time.perf_counter() - t_start
                print(
                    f"  epoch {epoch+1}/{epochs}  vol {vol_i+1:3d}/{len(npz_files)}"
                    f"  loss={loss.item():.3e}  τ_CG={b['tau_cg']:.2f}"
                    f"  lr={opt.param_groups[0]['lr']:.2e}  {elapsed:.0f}s"
                )

        print(f"Epoch {epoch+1}/{epochs} done  avg_loss={epoch_loss/len(npz_files):.3e}"
              f"  elapsed={time.perf_counter()-t_start:.0f}s")

    print("\nTraining finished. Evaluating …")

    # ---- evaluate all volumes ----
    print(f"\n{'Name':35s}  {'ε':>6}  {'τ_CG':>7}  {'τ_E2':>7}  {'err%':>7}  pass")
    print("-" * 72)

    rows = []
    model.eval()
    eval_loader = DataLoader(Dataset(reader, device=device), batch_size=1, shuffle=False)

    for batch, meta in eval_loader:
        name    = meta[0].get("source_filename", "?").replace(".npz", "")
        epsilon = float(batch["epsilon"].squeeze())
        tau_cg  = float(batch["tau_cg"].squeeze())

        pore_idx   = batch["pore_idx"].squeeze(0).long().to(device)
        sdf        = batch["sdf"].squeeze(0).to(device)
        descriptor = batch["descriptor"].squeeze(0).to(device)
        shape      = tuple(batch["shape"].squeeze(0).tolist())

        tau_e2, D_rel_e2 = estimate_tau(
            model, rff_enc, pore_idx, sdf, descriptor, shape, epsilon, device
        )
        err_pct = abs(tau_e2 - tau_cg) / tau_cg * 100
        passed  = err_pct < 10.0
        print(
            f"{name:35s}  {epsilon:6.3f}  {tau_cg:7.3f}  {tau_e2:7.3f}"
            f"  {err_pct:7.2f}%  {'OK' if passed else 'FAIL'}"
        )
        rows.append({
            "name":     name,
            "epsilon":  epsilon,
            "tau_cg":   tau_cg,
            "tau_e2":   tau_e2,
            "D_rel_e2": D_rel_e2,
            "err_pct":  err_pct,
            "passed":   passed,
        })

    df = pd.DataFrame(rows)
    n_pass = df["passed"].sum()
    print(f"\n{'='*65}")
    print(f"SUMMARY  {n_pass}/{len(df)} within 10%  mean={df['err_pct'].mean():.1f}%"
          f"  max={df['err_pct'].max():.1f}%")

    Path(results_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_csv, index=False)
    torch.save({"model": model.state_dict(), "rff": rff_enc.state_dict()}, checkpoint_path)
    print(f"  Results    : {results_csv}")
    print(f"  Checkpoint : {checkpoint_path}")
    return df


if __name__ == "__main__":
    train_e2_v5()
