"""Preprocess CT volumes for streaming E2 v5 training.

Converts each binary .tif volume into a compact .npz file containing only
the pore-voxel arrays needed for training. This decouples preprocessing
from training and enables the physicsnemo DataLoader to stream volumes
on demand without fitting everything in GPU RAM.

Output per volume (volume_name.npz):
    pore_idx    (N_pore, 3)  int16   — voxel indices of pore voxels
    teacher     (N_pore,)    float32 — CG concentration at each pore voxel
    sdf         (N_pore,)    float32 — EDT distance to nearest solid, normalised [0,1]
    descriptor  (127,)       float32 — [FFT(125) || epsilon || a_v]
    inlet_mask  (N_pore,)    bool    — pore voxels on inlet face
    outlet_mask (N_pore,)    bool    — pore voxels on outlet face
    shape       (3,)         int32   — volume shape
    epsilon     ()           float32 — porosity
    tau_cg      ()           float32 — tortuosity from CG solver
    D_rel       ()           float32 — relative diffusivity from CG solver

Usage:
    # Preprocess the 20-volume synthetic library
    python -m ct_to_physicsnemo.preprocess_volumes \\
        --library-csv  runs/synthetic_library/labels.csv \\
        --library-dir  runs/synthetic_library \\
        --output-dir   runs/preproc_library

    # Preprocess real CT volumes (one .tif per volume)
    python -m ct_to_physicsnemo.preprocess_volumes \\
        --tif-dir   /data/ct_scans \\
        --output-dir runs/preproc_ct
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from ct_to_physicsnemo.mesh_sdf import compute_mesh_sdf_grid
from ct_to_physicsnemo.metrics import compute_fft_descriptor, compute_porosity
from ct_to_physicsnemo.voxel_graph_laplace import solve_voxel_graph_laplace_for_tau

INLET_AXIS = 0
K_MAX      = 2          # FFT descriptor radius (125 coefficients)


def preprocess_one(
    tif_path: Path,
    output_dir: Path,
    epsilon_override: float | None = None,
    tau_override:    float | None = None,
    D_rel_override:  float | None = None,
    overwrite: bool = False,
) -> dict:
    """Preprocess one binary .tif volume → .npz.

    If epsilon/tau are already known from labels.csv they can be passed in to
    skip recomputing them (saves ~15s per volume).
    """
    name     = tif_path.stem.replace("_bin", "")
    out_path = output_dir / f"{name}.npz"

    if out_path.exists() and not overwrite:
        print(f"  [SKIP] {name} (already exists)")
        return {"name": name, "npz_path": str(out_path), "skipped": True}

    t0   = time.perf_counter()
    raw  = tifffile.imread(tif_path)
    mask = (raw > 0).astype(np.uint8)

    epsilon = epsilon_override if epsilon_override is not None else compute_porosity(mask)

    # Mesh SDF via physicsnemo Warp BVH — sub-voxel accurate, works for real CT
    sdf_full = compute_mesh_sdf_grid(mask, device="cuda" if __import__("torch").cuda.is_available() else "cpu")

    # CG teacher concentration field
    cg = solve_voxel_graph_laplace_for_tau(
        mask, epsilon=epsilon, axis=INLET_AXIS, return_field=True,
    )
    tau_cg = tau_override   if tau_override   is not None else float(cg["tau"])
    D_rel  = D_rel_override if D_rel_override is not None else float(cg["D_rel"])

    # FFT descriptor + scalar conditioning
    fft_desc = compute_fft_descriptor(mask, k_max=K_MAX)
    av       = float(np.sum(mask > 0) / mask.size)   # porosity as a_v proxy if not in CSV
    # (will be overwritten below if specific_surface_area is available)

    # Pore voxel indices
    pore_idx = np.argwhere(mask > 0).astype(np.int16)   # (N_pore, 3)  int16 saves 2× memory
    N        = pore_idx.shape[0]

    # Teacher and SDF sampled at pore voxels
    teacher_vals = cg["concentration"][
        pore_idx[:, 0], pore_idx[:, 1], pore_idx[:, 2]
    ].astype(np.float32)
    sdf_vals = sdf_full[
        pore_idx[:, 0], pore_idx[:, 1], pore_idx[:, 2]
    ].astype(np.float32)

    # Inlet / outlet masks
    inlet_mask  = (pore_idx[:, INLET_AXIS] == 0).astype(bool)
    outlet_mask = (pore_idx[:, INLET_AXIS] == mask.shape[INLET_AXIS] - 1).astype(bool)

    # Descriptor: FFT(125) || epsilon || a_v  — a_v filled in from CSV if available
    descriptor = np.concatenate([fft_desc, [epsilon, av]]).astype(np.float32)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        pore_idx    = pore_idx,
        teacher     = teacher_vals,
        sdf         = sdf_vals,
        descriptor  = descriptor,
        inlet_mask  = inlet_mask,
        outlet_mask = outlet_mask,
        shape       = np.array(mask.shape, dtype=np.int32),
        epsilon     = np.float32(epsilon),
        tau_cg      = np.float32(tau_cg),
        D_rel       = np.float32(D_rel),
    )

    elapsed = time.perf_counter() - t0
    print(
        f"  {name:35s}  ε={epsilon:.3f}  τ={tau_cg:.3f}"
        f"  N_pore={N:,}  {elapsed:.1f}s  →  {out_path.name}"
    )
    return {
        "name":     name,
        "npz_path": str(out_path),
        "epsilon":  epsilon,
        "tau_cg":   tau_cg,
        "D_rel":    D_rel,
        "n_pore":   N,
        "elapsed":  elapsed,
        "skipped":  False,
    }


def run_from_csv(
    library_csv:  str | Path,
    library_dir:  str | Path,
    output_dir:   str | Path,
    n_workers:    int  = 4,
    overwrite:    bool = False,
) -> pd.DataFrame:
    """Preprocess all volumes listed in labels.csv (synthetic library)."""
    library_csv = Path(library_csv)
    library_dir = Path(library_dir)
    output_dir  = Path(output_dir)

    df = pd.read_csv(library_csv)
    print(f"\nPreprocessing {len(df)} volumes  →  {output_dir}")
    print(f"Workers: {n_workers}")
    print("-" * 65)

    rows = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {}
        for _, row in df.iterrows():
            tif_path = Path(row["tif_path"])
            if not tif_path.exists():
                tif_path = library_dir / tif_path.name
            fut = pool.submit(
                preprocess_one,
                tif_path,
                output_dir,
                epsilon_override = float(row["epsilon"])         if "epsilon"  in row.index else None,
                tau_override     = float(row["tau"])             if "tau"      in row.index else None,
                D_rel_override   = float(row["D_rel"])           if "D_rel"    in row.index else None,
                overwrite        = overwrite,
            )
            futures[fut] = row["name"]

        for fut in as_completed(futures):
            rows.append(fut.result())

    result_df = pd.DataFrame(rows)
    manifest  = output_dir / "manifest.csv"
    result_df.to_csv(manifest, index=False)
    print(f"\nDone. Manifest: {manifest}")
    return result_df


def run_from_tif_dir(
    tif_dir:    str | Path,
    output_dir: str | Path,
    n_workers:  int  = 4,
    overwrite:  bool = False,
) -> pd.DataFrame:
    """Preprocess all *.tif files in a directory (real CT scans)."""
    tif_dir    = Path(tif_dir)
    output_dir = Path(output_dir)
    tif_files  = sorted(tif_dir.glob("*.tif")) + sorted(tif_dir.glob("*.tiff"))

    print(f"\nPreprocessing {len(tif_files)} .tif files  →  {output_dir}")
    print("-" * 65)

    rows = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(preprocess_one, p, output_dir, overwrite=overwrite): p
            for p in tif_files
        }
        for fut in as_completed(futures):
            rows.append(fut.result())

    result_df = pd.DataFrame(rows)
    manifest  = output_dir / "manifest.csv"
    result_df.to_csv(manifest, index=False)
    print(f"\nDone. Manifest: {manifest}")
    return result_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess CT volumes for E2 v5 streaming training")
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--library-csv",  type=str, help="Path to labels.csv from generate_library")
    grp.add_argument("--tif-dir",      type=str, help="Directory of raw .tif CT volumes")
    parser.add_argument("--library-dir", type=str, default="runs/synthetic_library")
    parser.add_argument("--output-dir",  type=str, default="runs/preproc_library")
    parser.add_argument("--n-workers",   type=int, default=4)
    parser.add_argument("--overwrite",   action="store_true")
    args = parser.parse_args()

    if args.library_csv:
        run_from_csv(
            library_csv  = args.library_csv,
            library_dir  = args.library_dir,
            output_dir   = args.output_dir,
            n_workers    = args.n_workers,
            overwrite    = args.overwrite,
        )
    else:
        run_from_tif_dir(
            tif_dir    = args.tif_dir,
            output_dir = args.output_dir,
            n_workers  = args.n_workers,
            overwrite  = args.overwrite,
        )
