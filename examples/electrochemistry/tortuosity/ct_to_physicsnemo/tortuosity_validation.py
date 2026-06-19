"""Tortuosity validation: TauFactor baseline vs PhysicsNeMo PINN.

This version:
- applies pore-only filtering
- keeps only inlet-outlet connected pores
- compares TauFactor vs baseline PINN vs SDF-RFF PINN
- uses transport axis = 0 to match TauFactor convention
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import torch

from ct_to_physicsnemo.fourier_encoding import RFFConfig, RFFEncoder
from ct_to_physicsnemo.geometry import (
    PoreGeometry,
    keep_percolating_pores,
    pore_only_filter,
)
from ct_to_physicsnemo.mesh import mask_to_stl
from ct_to_physicsnemo.metrics import compute_porosity, compute_taufactor
from ct_to_physicsnemo.pinn_laplace import solve_laplace_for_tau


def run_single_pinn(
    model_name: str,
    geom: PoreGeometry,
    epsilon: float,
    output_dir: Path,
    epochs: int,
    lr: float,
    inlet_axis: int,
    rff_encoder,
) -> float:
    return solve_laplace_for_tau(
        geom=geom,
        epsilon=epsilon,
        inlet_axis=inlet_axis,
        epochs=epochs,
        lr=lr,
        rff_encoder=rff_encoder,
        output_vtk=output_dir / f"{model_name}_concentration.vtk",
    )


def validate_one(
    tif_path: Path,
    voxel_size_m: float = 1.0,
    epochs: int = 5000,
    lr: float = 5e-5,
    inlet_axis: int = 0,
    pore_value: int = 1,
) -> dict:
    name = tif_path.stem.replace("_bin", "")

    output_dir = tif_path.parent / f"{name}_pinn_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n===== VALIDATING {name} =====")

    raw = tifffile.imread(tif_path)

    # 1 = pore / conducting phase, 0 = solid
    mask = pore_only_filter(
        raw,
        pore_value=pore_value,
    )

    original_porosity = compute_porosity(mask)

    # Keep only pore network connected from inlet to outlet
    try:
        mask = keep_percolating_pores(
            mask,
            axis=inlet_axis,
        )
        connected_filter_success = True
    except ValueError as e:
        print("WARNING: connected-pore filtering failed.")
        print("Reason:", e)
        print("Using pore-only mask without connected filtering.")
        connected_filter_success = False

    epsilon = compute_porosity(mask)

    tau_result = compute_taufactor(mask)

    tau_taufactor = float(tau_result["tau"])
    d_eff_taufactor = float(tau_result["D_eff"])

    stl_path = tif_path.parent / f"{name}.stl"

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

    tau_baseline = run_single_pinn(
        model_name="baseline",
        geom=geom,
        epsilon=epsilon,
        output_dir=output_dir,
        epochs=epochs,
        lr=lr,
        inlet_axis=inlet_axis,
        rff_encoder=None,
    )

    print("\n--- SDF-aware RFF PINN ---")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    rff_encoder = RFFEncoder(
        RFFConfig(
            num_features=64,
            scales=(0.25, 0.5, 1.0, 2.0),
            seed=0,
        )
    ).to(device)

    tau_sdf_rff = run_single_pinn(
        model_name="sdf_rff",
        geom=geom,
        epsilon=epsilon,
        output_dir=output_dir,
        epochs=epochs,
        lr=lr,
        inlet_axis=inlet_axis,
        rff_encoder=rff_encoder,
    )

    d_eff_baseline = epsilon / tau_baseline
    d_eff_sdf_rff = epsilon / tau_sdf_rff

    return {
        "name": name,
        "pore_value_used": pore_value,
        "transport_axis": inlet_axis,
        "connected_filter_success": connected_filter_success,
        "original_pore_only_porosity": original_porosity,
        "filtered_porosity": epsilon,
        "tau_taufactor": tau_taufactor,
        "D_eff_taufactor": d_eff_taufactor,
        "tau_baseline_pinn": tau_baseline,
        "D_eff_baseline_pinn": d_eff_baseline,
        "baseline_tau_abs_error": abs(tau_baseline - tau_taufactor),
        "baseline_tau_rel_error_percent": abs(tau_baseline - tau_taufactor)
        / tau_taufactor
        * 100,
        "tau_sdf_rff_pinn": tau_sdf_rff,
        "D_eff_sdf_rff_pinn": d_eff_sdf_rff,
        "sdf_rff_tau_abs_error": abs(tau_sdf_rff - tau_taufactor),
        "sdf_rff_tau_rel_error_percent": abs(tau_sdf_rff - tau_taufactor)
        / tau_taufactor
        * 100,
        "stl_path": str(stl_path),
        "output_dir": str(output_dir),
    }


def main() -> None:
    data_dir = Path("runs/synthetic_examples")
    output_csv = data_dir / "tortuosity_validation.csv"

    tif_files = sorted(data_dir.glob("*_bin.tif"))

    if not tif_files:
        raise FileNotFoundError(f"No *_bin.tif files found in {data_dir}")

    rows = []

    for tif_path in tif_files:
        row = validate_one(
            tif_path=tif_path,
            voxel_size_m=1.0,
            epochs=5000,
            lr=5e-5,
            inlet_axis=0,
            pore_value=1,
        )

        rows.append(row)

        print("\nResult:")
        print(f"TauFactor τ:      {row['tau_taufactor']:.6f}")
        print(f"Baseline PINN τ:  {row['tau_baseline_pinn']:.6f}")
        print(f"SDF-RFF PINN τ:   {row['tau_sdf_rff_pinn']:.6f}")
        print(f"Baseline err %:   {row['baseline_tau_rel_error_percent']:.2f}")
        print(f"SDF-RFF err %:    {row['sdf_rff_tau_rel_error_percent']:.2f}")

    df = pd.DataFrame(rows)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"\nSaved tortuosity validation to: {output_csv}")
    print(df)


if __name__ == "__main__":
    main()