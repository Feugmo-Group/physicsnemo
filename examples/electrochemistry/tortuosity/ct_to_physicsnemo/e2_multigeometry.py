"""Experiment E2 — multi-geometry conditioned PINN (v4: batched + vmap).

Key changes from v3:
- All 20 geometries concatenated into one big batch per epoch (40K voxels),
  one forward pass instead of 20 sequential calls — full GPU utilisation.
- EDT SDF loading parallelised with ThreadPoolExecutor (IO/CPU overlap).
- FiLM conditioning architecture retained.
- tau-weighted loss and warm-started adaptive EMA retained.

Usage:
    python -m ct_to_physicsnemo.e2_multigeometry
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import torch
import torch.nn as nn

from ct_to_physicsnemo.mesh_sdf import compute_mesh_sdf_grid
from ct_to_physicsnemo.metrics import compute_fft_descriptor, compute_porosity
from ct_to_physicsnemo.voxel_graph_laplace import solve_voxel_graph_laplace_for_tau

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
# Constants
# ---------------------------------------------------------------------------
TOP_BC     = -0.5
BOT_BC     =  0.5
DELTA_C    = abs(BOT_BC - TOP_BC)
INLET_AXIS = 0


# ---------------------------------------------------------------------------
# FiLM-conditioned network
# ---------------------------------------------------------------------------

class FiLMLayer(nn.Module):
    """h → h * (1 + γ(g)) + β(g)  — initialised as identity."""
    def __init__(self, hidden_dim: int, descriptor_dim: int):
        super().__init__()
        self.gamma = nn.Linear(descriptor_dim, hidden_dim)
        self.beta  = nn.Linear(descriptor_dim, hidden_dim)
        nn.init.zeros_(self.gamma.weight); nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight);  nn.init.zeros_(self.beta.bias)

    def forward(self, h: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        return h * (1.0 + self.gamma(g)) + self.beta(g)


class E2FiLMMLP(nn.Module):
    """FiLM-conditioned MLP.  Input: (rff, g) where rff and g are per-sample."""

    def __init__(
        self,
        rff_out_dim: int,
        descriptor_dim: int = 127,
        hidden: int = 256,
        layers: int = 6,
    ):
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
# Per-geometry data container
# ---------------------------------------------------------------------------

class GeometryData:
    def __init__(
        self,
        name: str,
        mask: np.ndarray,
        concentration: np.ndarray,
        descriptor: np.ndarray,       # [FFT(125) || epsilon || a_v]
        sdf_grid: np.ndarray,         # mesh SDF via Warp BVH, normalised [0,1]
        epsilon: float,
        tau_cg: float,
        specific_surface_area: float,
        surface_area: float,
        device: str,
    ):
        self.name                  = name
        self.epsilon               = epsilon
        self.tau_cg                = tau_cg
        self.specific_surface_area = specific_surface_area
        self.surface_area          = surface_area
        self.tau_weight            = 1.0
        shape                      = mask.shape
        self.shape                 = shape

        mask_t        = torch.tensor(mask > 0, dtype=torch.bool, device=device)
        self.mask_t   = mask_t
        self.pore_idx = torch.nonzero(mask_t, as_tuple=False)   # (N_pore, 3)

        shape_t       = torch.tensor(shape, dtype=torch.float32, device=device)
        self.shape_t  = shape_t

        self.teacher_t    = torch.tensor(concentration, dtype=torch.float32, device=device)
        self.sdf_t        = torch.tensor(sdf_grid,      dtype=torch.float32, device=device)
        # descriptor stored as (1, 127) for expand
        self.descriptor_t = torch.tensor(descriptor, dtype=torch.float32, device=device).unsqueeze(0)

        axis             = INLET_AXIS
        self.inlet_mask  = self.pore_idx[:, axis] == 0
        self.outlet_mask = self.pore_idx[:, axis] == shape[axis] - 1
        self.n_pore      = self.pore_idx.shape[0]
        self.n_inlet     = int(self.inlet_mask.sum())
        self.n_outlet    = int(self.outlet_mask.sum())
        self.device      = device

    def normalized_coords(self, idx: torch.Tensor) -> torch.Tensor:
        return (idx.float() + 0.5) / self.shape_t

    def sdf_at(self, idx: torch.Tensor) -> torch.Tensor:
        return self.sdf_t[idx[:, 0], idx[:, 1], idx[:, 2]].unsqueeze(1)

    def rff_input(self, idx: torch.Tensor) -> torch.Tensor:
        """(N,4): [x_norm, y_norm, z_norm, sdf]"""
        return torch.cat([self.normalized_coords(idx), self.sdf_at(idx)], dim=1)

    def sample_indices(self, n: int) -> torch.Tensor:
        idx = torch.randint(0, self.n_pore, (min(n, self.n_pore),), device=self.device)
        return self.pore_idx[idx]

    def teacher_values(self, idx: torch.Tensor) -> torch.Tensor:
        return self.teacher_t[idx[:, 0], idx[:, 1], idx[:, 2]].unsqueeze(1)

    def descriptor_expand(self, n: int) -> torch.Tensor:
        return self.descriptor_t.expand(n, -1)


# ---------------------------------------------------------------------------
# Batched forward pass across all geometries
# ---------------------------------------------------------------------------

def batched_forward(
    model: E2FiLMMLP,
    rff_enc: RFFEncoder,
    geom_list: list[GeometryData],
    batch_per_geom: int,
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], list[int]]:
    """Sample from all geometries, concatenate, run ONE forward pass.

    Returns:
        c_pred_all  — (N_total, 1) predictions
        idx_list    — sampled indices per geometry
        teacher_list— teacher values per geometry
        sizes       — number of samples per geometry (for torch.split)
    """
    rff_parts     = []
    g_parts       = []
    idx_list      = []
    teacher_list  = []
    sizes         = []

    for gd in geom_list:
        idx  = gd.sample_indices(batch_per_geom)
        rff  = rff_enc(gd.rff_input(idx))
        g    = gd.descriptor_expand(idx.shape[0])
        rff_parts.append(rff)
        g_parts.append(g)
        idx_list.append(idx)
        teacher_list.append(gd.teacher_values(idx))
        sizes.append(idx.shape[0])

    rff_all   = torch.cat(rff_parts, dim=0)   # (N_total, rff_dim)
    g_all     = torch.cat(g_parts,   dim=0)   # (N_total, 127)
    c_pred_all = model(rff_all, g_all)         # ONE forward pass

    return c_pred_all, idx_list, teacher_list, sizes


# ---------------------------------------------------------------------------
# τ estimation
# ---------------------------------------------------------------------------

def estimate_tau(
    model: E2FiLMMLP,
    rff_enc: RFFEncoder,
    gdata: GeometryData,
    batch_size: int = 50_000,
) -> tuple[float, float]:
    device    = gdata.device
    mask_np   = gdata.mask_t.cpu().numpy().astype(np.uint8)
    shape     = gdata.shape
    axis      = INLET_AXIS
    other     = [a for a in range(3) if a != axis]
    full_area = shape[other[0]] * shape[other[1]]

    flux_per_slice = []

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

            total_flux = 0.0
            for start in range(0, idx_l.shape[0], batch_size):
                sl_l = idx_l[start:start+batch_size]
                sl_r = idx_r[start:start+batch_size]
                n    = sl_l.shape[0]
                g    = gdata.descriptor_expand(n)
                c_l  = model(rff_enc(gdata.rff_input(sl_l)), g)
                c_r  = model(rff_enc(gdata.rff_input(sl_r)), g)
                total_flux += (c_r - c_l).sum().item()

            flux_per_slice.append(abs(total_flux) / full_area)

    fps    = np.array(flux_per_slice)
    mean_f = float(np.mean(fps))
    D_rel  = mean_f * shape[axis] / DELTA_C
    tau    = gdata.epsilon / D_rel if D_rel > 0 else float("inf")
    return tau, D_rel


# ---------------------------------------------------------------------------
# Geometry loading (parallelised)
# ---------------------------------------------------------------------------

def _load_one_cpu(row: pd.Series, library_dir: Path, fft_cols: list[str]) -> dict:
    """IO + CG phase — safe to run in ThreadPoolExecutor (no CUDA calls)."""
    tif_path = Path(row["tif_path"])
    if not tif_path.exists():
        tif_path = library_dir / tif_path.name
    raw  = tifffile.imread(tif_path)
    mask = (raw > 0).astype(np.uint8)

    cg = solve_voxel_graph_laplace_for_tau(
        mask, epsilon=float(row["epsilon"]), axis=INLET_AXIS, return_field=True,
    )

    has_sa   = "specific_surface_area" in row.index
    epsilon  = float(row["epsilon"])
    av       = float(row["specific_surface_area"]) if has_sa else 0.0
    sa       = float(row["surface_area"])           if has_sa else float("nan")
    fft_desc = np.array([row[c] for c in fft_cols], dtype=np.float32)
    descriptor = np.concatenate([fft_desc, [epsilon, av]]).astype(np.float32)

    return dict(
        name=row["name"],
        mask=mask,
        concentration=cg["concentration"],
        descriptor=descriptor,
        epsilon=epsilon,
        tau_cg=float(row["tau"]),
        specific_surface_area=av,
        surface_area=sa,
    )


def _finish_load(data: dict, device: str) -> GeometryData:
    """SDF phase — must run in the main thread (Warp CUDA stream context)."""
    sdf_grid = compute_mesh_sdf_grid(data["mask"], device=device)
    return GeometryData(
        name=data["name"],
        mask=data["mask"],
        concentration=data["concentration"],
        descriptor=data["descriptor"],
        sdf_grid=sdf_grid,
        epsilon=data["epsilon"],
        tau_cg=data["tau_cg"],
        specific_surface_area=data["specific_surface_area"],
        surface_area=data["surface_area"],
        device=device,
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_e2(
    library_csv: str | Path = "runs/synthetic_library/labels.csv",
    library_dir: str | Path = "runs/synthetic_library",
    epochs: int = 15_000,
    batch_per_geom: int = 2_000,
    lr: float = 3e-4,
    rff_sigma: float = 6.0,
    rff_n_freqs: int = 128,
    teacher_weight: float = 500.0,
    bc_weight: float = 50.0,
    descriptor_dim: int = 127,
    hidden: int = 256,
    layers: int = 6,
    n_load_workers: int = 4,
    results_csv: str | Path = "runs/e2_results/e2_training_results.csv",
    checkpoint_path: str | Path = "runs/e2_results/e2_model.pt",
    log_every: int = 250,
) -> pd.DataFrame:

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*65}")
    print("EXPERIMENT E2 v4 — batched FiLM PINN")
    print(f"  FiLM | mesh SDF (Warp BVH) | tau-weight | adaptive EMA | ONE fwd pass/epoch")
    print(f"{'='*65}")
    print(f"device: {device}  epochs: {epochs}  batch_per_geom: {batch_per_geom}")

    labels_df = pd.read_csv(library_csv)
    fft_cols  = [c for c in labels_df.columns if c.startswith("fft_")]
    print(f"Library: {len(labels_df)} volumes  |  FFT cols: {len(fft_cols)}")

    # ---- parallel load ----
    print(f"\nLoading {len(labels_df)} geometries with {n_load_workers} workers …")
    t_load = time.perf_counter()
    rows_list = list(labels_df.iterrows())
    geom_list: list[GeometryData] = [None] * len(rows_list)  # type: ignore

    # Phase 1 — parallel IO + CG (no CUDA calls, safe in threads)
    cpu_data: list[dict] = [None] * len(rows_list)  # type: ignore
    with ThreadPoolExecutor(max_workers=n_load_workers) as pool:
        futures = {
            pool.submit(_load_one_cpu, row, Path(library_dir), fft_cols): i
            for i, (_, row) in enumerate(rows_list)
        }
        for fut in as_completed(futures):
            i = futures[fut]
            d = fut.result()
            cpu_data[i] = d
            print(f"  [{i+1:02d}/{len(rows_list)}] {d['name']:40s}"
                  f" ε={d['epsilon']:.3f}  τ_CG={d['tau_cg']:.3f}"
                  f"  pore={int((d['mask']>0).sum()):,}")

    # Phase 2 — mesh SDF in main thread (Warp CUDA stream context)
    print("Computing mesh SDF (Warp BVH) …")
    for i, d in enumerate(cpu_data):
        geom_list[i] = _finish_load(d, device)
        print(f"  [{i+1:02d}/{len(cpu_data)}] {geom_list[i].name:35s}  SDF done  pore={geom_list[i].n_pore:,}")

    print(f"Load time: {time.perf_counter() - t_load:.1f}s")

    # ---- tau weights ----
    mean_tau = float(np.mean([gd.tau_cg for gd in geom_list]))
    for gd in geom_list:
        gd.tau_weight = gd.tau_cg / mean_tau
    print(f"Tau weights: {min(gd.tau_weight for gd in geom_list):.2f} – "
          f"{max(gd.tau_weight for gd in geom_list):.2f}  (mean_tau={mean_tau:.3f})")

    # ---- build model ----
    rff_enc = RFFEncoder(sigma=rff_sigma, n_freqs=rff_n_freqs, in_dim=4, seed=42).to(device)
    model   = E2FiLMMLP(
        rff_out_dim=rff_enc.out_dim,
        descriptor_dim=descriptor_dim,
        hidden=hidden,
        layers=layers,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    total_batch = batch_per_geom * len(geom_list)
    print(f"\nModel parameters : {n_params:,}")
    print(f"Total batch/epoch: {total_batch:,} voxels  ({len(geom_list)} geoms × {batch_per_geom})")

    opt       = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr / 20)

    # ---- warm-start adaptive EMA ----
    print("\nWarm-starting adaptive EMA …")
    ema_alpha   = 0.02
    ema_teacher = {}
    ema_bc      = {}
    model.eval()
    with torch.no_grad():
        for gd in geom_list:
            idx    = gd.sample_indices(batch_per_geom)
            rff    = rff_enc(gd.rff_input(idx))
            g      = gd.descriptor_expand(idx.shape[0])
            c_pred = model(rff, g)
            lt     = float(torch.mean((c_pred - gd.teacher_values(idx)) ** 2).item())

            n_bc    = min(512, gd.n_inlet)
            in_idx  = gd.pore_idx[gd.inlet_mask][torch.randint(0, gd.n_inlet,  (n_bc,), device=device)]
            out_idx = gd.pore_idx[gd.outlet_mask][torch.randint(0, gd.n_outlet, (n_bc,), device=device)]
            c_in    = model(rff_enc(gd.rff_input(in_idx)),  gd.descriptor_expand(n_bc))
            c_out   = model(rff_enc(gd.rff_input(out_idx)), gd.descriptor_expand(n_bc))
            lb      = float((torch.mean((c_in  - TOP_BC)**2) + torch.mean((c_out - BOT_BC)**2)).item())

            ema_teacher[gd.name] = max(lt, 1e-12)
            ema_bc[gd.name]      = max(lb, 1e-12)
    model.train()

    # ---- training loop ----
    print(f"\nTraining  epochs={epochs}  lr={lr}")
    print("-" * 65)

    t_start = time.perf_counter()

    for epoch in range(epochs):
        opt.zero_grad()

        # ---- ONE batched forward pass for teacher loss ----
        c_pred_all, idx_list, teacher_list, sizes = batched_forward(
            model, rff_enc, geom_list, batch_per_geom
        )
        c_splits = torch.split(c_pred_all, sizes)

        total_loss = torch.zeros((), device=device)

        for gd, c_pred, teacher, idx in zip(geom_list, c_splits, teacher_list, idx_list):
            loss_teacher = torch.mean((c_pred - teacher) ** 2)

            # BC loss (small, per-geometry — cheap separate pass)
            n_bc    = min(512, gd.n_inlet)
            in_idx  = gd.pore_idx[gd.inlet_mask][torch.randint(0, gd.n_inlet,  (n_bc,), device=device)]
            out_idx = gd.pore_idx[gd.outlet_mask][torch.randint(0, gd.n_outlet, (n_bc,), device=device)]
            c_in    = model(rff_enc(gd.rff_input(in_idx)),  gd.descriptor_expand(n_bc))
            c_out   = model(rff_enc(gd.rff_input(out_idx)), gd.descriptor_expand(n_bc))
            loss_bc = torch.mean((c_in - TOP_BC)**2) + torch.mean((c_out - BOT_BC)**2)

            lt = loss_teacher.item()
            lb = loss_bc.item()
            ema_teacher[gd.name] = (1 - ema_alpha) * ema_teacher[gd.name] + ema_alpha * max(lt, 1e-12)
            ema_bc[gd.name]      = (1 - ema_alpha) * ema_bc[gd.name]      + ema_alpha * max(lb, 1e-12)

            w_t = teacher_weight / ema_teacher[gd.name]
            w_b = bc_weight      / ema_bc[gd.name]

            total_loss = total_loss + gd.tau_weight * (w_t * loss_teacher + w_b * loss_bc)

        total_loss.backward()
        opt.step()
        scheduler.step()

        if epoch % log_every == 0 or epoch == epochs - 1:
            elapsed = time.perf_counter() - t_start
            print(
                f"epoch {epoch:6d}/{epochs}  "
                f"loss={total_loss.item():.4e}  "
                f"lr={opt.param_groups[0]['lr']:.2e}  "
                f"elapsed={elapsed:.0f}s"
            )

    print("\nTraining finished.")

    # ---- evaluate ----
    print("\nEvaluating τ on all geometries …")
    print(f"{'Name':35s}  {'ε':>6}  {'a_v':>8}  {'τ_CG':>7}  {'τ_E2':>7}  {'err%':>7}  pass")
    print("-" * 84)

    rows = []
    model.eval()

    for gd in geom_list:
        tau_e2, D_rel_e2 = estimate_tau(model, rff_enc, gd)
        err_pct = abs(tau_e2 - gd.tau_cg) / gd.tau_cg * 100
        passed  = err_pct < 10.0
        print(
            f"{gd.name:35s}  {gd.epsilon:6.3f}  {gd.specific_surface_area:8.5f}"
            f"  {gd.tau_cg:7.3f}  {tau_e2:7.3f}  {err_pct:7.2f}%  {'OK' if passed else 'FAIL'}"
        )
        rows.append({
            "name":                  gd.name,
            "epsilon":               gd.epsilon,
            "surface_area":          gd.surface_area,
            "specific_surface_area": gd.specific_surface_area,
            "tau_cg":                gd.tau_cg,
            "tau_e2":                tau_e2,
            "D_rel_e2":              D_rel_e2,
            "err_pct":               err_pct,
            "passed":                passed,
        })

    df = pd.DataFrame(rows)
    n_pass = df["passed"].sum()
    print(f"\n{'='*65}")
    print(f"SUMMARY  {n_pass}/{len(df)} geometries within 10% of CG")
    print(f"  mean error : {df['err_pct'].mean():.2f}%")
    print(f"  max  error : {df['err_pct'].max():.2f}%")

    Path(results_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_csv, index=False)
    print(f"  Results    : {results_csv}")

    torch.save({"model": model.state_dict(), "rff": rff_enc.state_dict()}, checkpoint_path)
    print(f"  Checkpoint : {checkpoint_path}")

    return df


if __name__ == "__main__":
    train_e2()
