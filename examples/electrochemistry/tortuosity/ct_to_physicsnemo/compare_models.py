"""Compare baseline PINN vs SDF-aware RFF PINN against TauFactor."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from .fourier_encoding import RFFConfig, RFFEncoder
from .geometry import PoreGeometry
from .io import crop_rev, load_ct
from .metrics import compute_porosity
from .pinn_laplace import solve_laplace_for_tau


def run_model(
    model_name: str,
    geom: PoreGeometry,
    epsilon: float,
    tau_taufactor: float,
    rff_encoder,
    output_vtk: str,
    epochs: int = 5000,
    lr: float = 5e-5,
) -> dict:
    tau_pinn = solve_laplace_for_tau(
        geom=geom,
        epsilon=epsilon,
        inlet_axis=2,
        epochs=epochs,
        lr=lr,
        rff_encoder=rff_encoder,
        output_vtk=output_vtk,
    )

    D_eff_taufactor = epsilon / tau_taufactor
    D_eff_pinn = epsilon / tau_pinn

    return {
        "model": model_name,
        "epochs": epochs,
        "lr": lr,
        "tau_pinn": tau_pinn,
        "tau_taufactor": tau_taufactor,
        "tau_absolute_error": abs(tau_pinn - tau_taufactor),
        "tau_relative_error_percent": abs(tau_pinn - tau_taufactor)
        / tau_taufactor
        * 100,
        "D_eff_pinn": D_eff_pinn,
        "D_eff_taufactor": D_eff_taufactor,
        "D_eff_absolute_error": abs(D_eff_pinn - D_eff_taufactor),
        "D_eff_relative_error_percent": abs(D_eff_pinn - D_eff_taufactor)
        / D_eff_taufactor
        * 100,
        "output_vtk": output_vtk,
    }


def main() -> None:
    ct_path = "../diffusion_in_porous_media/Electrode I/I_1/I_1_bin.tif"
    stl_path = "runs/meshes/I_1_crop.stl"
    metrics_csv = "runs/metrics.csv"
    output_csv = Path("runs/pinn_laplace_comparison.csv")

    epochs = 5000
    lr = 5e-5

    volume = load_ct(ct_path)

    cropped = crop_rev(
        volume,
        voxel_size_nm=50,
        rev_microns=5,
    )

    epsilon = compute_porosity(cropped)

    geom = PoreGeometry(
        stl_path=stl_path,
        mask=cropped,
        voxel_size_m=50e-9,
    )

    metrics_df = pd.read_csv(metrics_csv)
    ref_row = metrics_df[metrics_df["name"] == "I_1_bin"].iloc[0]
    tau_taufactor = float(ref_row["tau"])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    results = []

    print("\n===== TAUFACTOR REFERENCE =====")
    print("epsilon:", epsilon)
    print("tau_taufactor:", tau_taufactor)
    print("D_eff_taufactor:", epsilon / tau_taufactor)

    print("\n===== RUNNING BASELINE PINN =====")

    results.append(
        run_model(
            model_name="baseline_mlp",
            geom=geom,
            epsilon=epsilon,
            tau_taufactor=tau_taufactor,
            rff_encoder=None,
            output_vtk="runs/pinn_outputs/I_1_concentration_baseline.vtk",
            epochs=epochs,
            lr=lr,
        )
    )

    print("\n===== RUNNING SDF-AWARE RFF PINN =====")

    rff_encoder = RFFEncoder(
        RFFConfig(
            num_features=64,
            scales=(0.25, 0.5, 1.0, 2.0),
            seed=0,
        )
    ).to(device)

    results.append(
        run_model(
            model_name="sdf_rff_mlp",
            geom=geom,
            epsilon=epsilon,
            tau_taufactor=tau_taufactor,
            rff_encoder=rff_encoder,
            output_vtk="runs/pinn_outputs/I_1_concentration_sdf_rff.vtk",
            epochs=epochs,
            lr=lr,
        )
    )

    df = pd.DataFrame(results)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print("\n===== COMPARISON RESULTS =====")
    print(df)
    print(f"\nSaved to: {output_csv}")


if __name__ == "__main__":
    main()