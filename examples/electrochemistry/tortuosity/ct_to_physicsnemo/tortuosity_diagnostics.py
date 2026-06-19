"""Diagnostic comparison of TauFactor vs PhysicsNeMo PINN tortuosity.

Compares intermediate values:
- tau
- D_eff / D_rel
- mean_flux
- estimated flux from D_rel / Nx

Based on TauFactor Section 8:
D_rel = mean_flux * Nx / ΔC
tau = epsilon / D_rel
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import torch

from ct_to_physicsnemo.geometry import PoreGeometry, keep_percolating_pores
from ct_to_physicsnemo.mesh import mask_to_stl
from ct_to_physicsnemo.metrics import compute_porosity, compute_taufactor
from ct_to_physicsnemo.pinn_laplace import solve_laplace_for_tau
from ct_to_physicsnemo.fourier_encoding import RFFConfig, RFFEncoder


def run_one(
    tif_path: Path,
    voxel_size_m: float = 1.0,
    epochs: int = 5000,
    lr: float = 5e-5,
    inlet_axis: int = 0,
) -> dict:
    name = tif_path.stem.replace("_bin", "")

    print(f"\n===== DIAGNOSTIC RUN: {name} =====")

    mask = tifffile.imread(tif_path)
    mask = (mask > 0).astype(np.uint8)

    mask = keep_percolating_pores(mask, axis=inlet_axis)

    epsilon = compute_porosity(mask)
    nx = mask.shape[inlet_axis]

    tau_data = compute_taufactor(mask)

    tau_taufactor = float(tau_data["tau"])
    D_eff_taufactor = float(tau_data["D_eff"])

    D_rel_taufactor = D_eff_taufactor
    mean_flux_taufactor_est = D_rel_taufactor / nx

    stl_path = tif_path.parent / f"{name}_diagnostic.stl"

    mask_to_stl(
        mask=mask,
        voxel_size_m=voxel_size_m,
        out_path=stl_path,
    )

    geom = PoreGeometry(
        stl_path=stl_path,
        mask=mask,
        voxel_size_m=voxel_size_m,
        filter_connected=False,
        inlet_axis=inlet_axis,
    )

    print("\n--- Baseline PINN ---")

    tau_baseline = solve_laplace_for_tau(
        geom=geom,
        epsilon=epsilon,
        inlet_axis=inlet_axis,
        epochs=epochs,
        lr=lr,
        rff_encoder=None,
        output_vtk=tif_path.parent / f"{name}_diagnostic_baseline.vtk",
    )

    D_rel_baseline = epsilon / tau_baseline
    mean_flux_baseline_est = D_rel_baseline / nx

    print("\n--- SDF-RFF PINN ---")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    rff_encoder = RFFEncoder(
        RFFConfig(
            num_features=64,
            scales=(0.25, 0.5, 1.0, 2.0),
            seed=0,
        )
    ).to(device)

    tau_sdf_rff = solve_laplace_for_tau(
        geom=geom,
        epsilon=epsilon,
        inlet_axis=inlet_axis,
        epochs=epochs,
        lr=lr,
        rff_encoder=rff_encoder,
        output_vtk=tif_path.parent / f"{name}_diagnostic_sdf_rff.vtk",
    )

    D_rel_sdf_rff = epsilon / tau_sdf_rff
    mean_flux_sdf_rff_est = D_rel_sdf_rff / nx

    return {
        "name": name,
        "porosity": epsilon,
        "Nx_transport_axis": nx,

        "tau_taufactor": tau_taufactor,
        "D_rel_taufactor": D_rel_taufactor,
        "mean_flux_taufactor_est": mean_flux_taufactor_est,

        "tau_baseline_pinn": tau_baseline,
        "D_rel_baseline_pinn": D_rel_baseline,
        "mean_flux_baseline_est": mean_flux_baseline_est,

        "tau_sdf_rff_pinn": tau_sdf_rff,
        "D_rel_sdf_rff_pinn": D_rel_sdf_rff,
        "mean_flux_sdf_rff_est": mean_flux_sdf_rff_est,

        "baseline_D_rel_ratio_vs_taufactor": D_rel_baseline / D_rel_taufactor,
        "sdf_rff_D_rel_ratio_vs_taufactor": D_rel_sdf_rff / D_rel_taufactor,

        "baseline_tau_rel_error_percent": abs(tau_baseline - tau_taufactor)
        / tau_taufactor
        * 100,
        "sdf_rff_tau_rel_error_percent": abs(tau_sdf_rff - tau_taufactor)
        / tau_taufactor
        * 100,

        "stl_path": str(stl_path),
    }


def main() -> None:
    data_dir = Path("runs/synthetic_examples")
    output_csv = data_dir / "tortuosity_diagnostics.csv"

    tif_files = sorted(data_dir.glob("*_bin.tif"))

    if not tif_files:
        raise FileNotFoundError(f"No *_bin.tif files found in {data_dir}")

    rows = []

    for tif_path in tif_files:
        row = run_one(
            tif_path=tif_path,
            voxel_size_m=1.0,
            epochs=5000,
            lr=5e-5,
            inlet_axis=0,
        )

        rows.append(row)

        print("\n===== SUMMARY =====")
        print("TauFactor tau:", row["tau_taufactor"])
        print("Baseline PINN tau:", row["tau_baseline_pinn"])
        print("SDF-RFF PINN tau:", row["tau_sdf_rff_pinn"])
        print("Baseline D_rel ratio:", row["baseline_D_rel_ratio_vs_taufactor"])
        print("SDF-RFF D_rel ratio:", row["sdf_rff_D_rel_ratio_vs_taufactor"])

    df = pd.DataFrame(rows)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"\nSaved diagnostics to: {output_csv}")
    print(df)


if __name__ == "__main__":
    main()