"""Week 1 milestone: generate the 20-volume synthetic library with τ labels.

Runs in sequence:
  1. Generate 20 binary volumes (8 ellipsoid + 12 GRF) via synthetic.py
  2. For each volume, solve the Laplace equation with the voxel-graph CG solver
  3. Compute the 3D-FFT descriptor (125 coefficients) for E2 conditioning
  4. Save everything to runs/synthetic_library/labels.csv

Usage:
    python -m ct_to_physicsnemo.generate_library

Output:
    runs/synthetic_library/
        *_bin.tif          — 20 binary volumes
        metadata.csv       — generator config per volume
        labels.csv         — τ, D_rel, ε, flux_err, FFT descriptor per volume
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from scipy.ndimage import label as _label

from ct_to_physicsnemo.metrics import compute_fft_descriptor, compute_porosity, compute_surface_area


def _pore_only_filter(mask: np.ndarray, pore_value: int = 1) -> np.ndarray:
    return (np.asarray(mask) == pore_value).astype(np.uint8)


def _keep_percolating_pores(mask: np.ndarray, axis: int = 0) -> np.ndarray:
    pore = mask > 0
    labels, n_labels = _label(pore)
    if n_labels == 0:
        raise ValueError("No pore clusters found.")
    inlet_sl  = [slice(None)] * 3; inlet_sl[axis]  = 0
    outlet_sl = [slice(None)] * 3; outlet_sl[axis] = mask.shape[axis] - 1
    inlet_labels  = set(np.unique(labels[tuple(inlet_sl)])) - {0}
    outlet_labels = set(np.unique(labels[tuple(outlet_sl)])) - {0}
    percolating = inlet_labels & outlet_labels
    if not percolating:
        raise ValueError("No percolating pore cluster found.")
    filtered = np.zeros_like(mask)
    for lbl in percolating:
        filtered[labels == lbl] = 1
    return filtered
from ct_to_physicsnemo.synthetic import make_library
from ct_to_physicsnemo.voxel_graph_laplace import solve_voxel_graph_laplace_for_tau


def label_one(
    tif_path: Path,
    inlet_axis: int = 0,
    k_max: int = 2,
) -> dict:
    """Load one binary volume and compute all labels needed for E2 training."""

    name = tif_path.stem.replace("_bin", "")
    raw = tifffile.imread(tif_path)

    # apply pore-only and percolation filters (same as validation pipeline)
    mask = _pore_only_filter(raw, pore_value=1)

    try:
        mask = _keep_percolating_pores(mask, axis=inlet_axis)
        percolating = True
    except ValueError:
        percolating = False

    epsilon      = compute_porosity(mask)
    surface_area = compute_surface_area(mask, voxel_size=1.0)
    specific_surface_area = surface_area / mask.size  # a_v = SA / V_box

    t0 = time.perf_counter()
    cg = solve_voxel_graph_laplace_for_tau(
        mask,
        epsilon=epsilon,
        axis=inlet_axis,
        return_field=False,
    )
    cg_time = time.perf_counter() - t0

    descriptor = compute_fft_descriptor(mask, k_max=k_max)

    row: dict = {
        "name": name,
        "tif_path": str(tif_path),
        "shape": str(mask.shape),
        "percolating": percolating,
        "epsilon": epsilon,
        "surface_area": surface_area,
        "specific_surface_area": specific_surface_area,
        "tau": cg["tau"],
        "D_rel": cg["D_rel"],
        "flux_relative_error": cg["flux_relative_error"],
        "flux_min": cg["flux_min"],
        "flux_max": cg["flux_max"],
        "cg_converged": cg["cg_info"] == 0,
        "cg_time_s": round(cg_time, 3),
        "n_pore": cg["n_pore"],
    }

    # store each FFT coefficient as a separate column: fft_000 … fft_124
    for i, v in enumerate(descriptor):
        row[f"fft_{i:03d}"] = float(v)

    return row


def run(
    library_dir: str | Path = "runs/synthetic_library",
    labels_csv: str | Path = "runs/synthetic_library/labels.csv",
    inlet_axis: int = 0,
    k_max: int = 2,
    regenerate: bool = False,
) -> pd.DataFrame:
    """Generate library (if needed) and compute labels for all volumes."""

    library_dir = Path(library_dir)
    labels_csv  = Path(labels_csv)

    tif_files = sorted(library_dir.glob("*_bin.tif"))

    if not tif_files or regenerate:
        print("Generating 20-volume synthetic library …")
        make_library(out_dir=library_dir)
        tif_files = sorted(library_dir.glob("*_bin.tif"))

    print(f"\nLabelling {len(tif_files)} volumes  (axis={inlet_axis}, k_max={k_max})")
    print("-" * 65)

    rows = []

    for tif_path in tif_files:
        row = label_one(tif_path, inlet_axis=inlet_axis, k_max=k_max)
        status = "OK" if row["cg_converged"] else "NOT CONVERGED"
        print(
            f"  {row['name']:35s}  ε={row['epsilon']:.3f}"
            f"  a_v={row['specific_surface_area']:.4f}"
            f"  τ={row['tau']:.3f}  flux_err={row['flux_relative_error']:.4f}"
            f"  {row['cg_time_s']:.2f}s  {status}"
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    labels_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(labels_csv, index=False)

    print(f"\n{'='*65}")
    print(f"Library summary ({len(df)} volumes)")
    print(f"  ε range : {df['epsilon'].min():.3f} – {df['epsilon'].max():.3f}")
    print(f"  τ range : {df['tau'].min():.3f} – {df['tau'].max():.3f}")
    print(f"  All CG converged : {df['cg_converged'].all()}")
    print(f"  Labels saved to  : {labels_csv}")
    return df


if __name__ == "__main__":
    run()
