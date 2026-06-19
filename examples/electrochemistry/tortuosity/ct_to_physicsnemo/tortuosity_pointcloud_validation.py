"""Validate point-cloud PINN tortuosity against TauFactor.

This version:
- applies pore-only filtering first
- keeps only inlet-outlet connected pore space
- automatically chooses point-cloud sample counts
- runs the point-cloud PINN
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from ct_to_physicsnemo.geometry import (
    PoreGeometry,
    pore_only_filter,
    keep_percolating_pores,
)
from ct_to_physicsnemo.mesh import mask_to_stl
from ct_to_physicsnemo.metrics import compute_porosity, compute_taufactor
from ct_to_physicsnemo.pinn_laplace_pointcloud import (
    solve_pointcloud_laplace_for_tau,
)
from ct_to_physicsnemo.voxel_graph_laplace import (
    solve_voxel_graph_laplace_for_tau,
)


def count_face_pores(
    mask: np.ndarray,
    axis: int,
    side: str,
) -> int:
    if side not in {"min", "max"}:
        raise ValueError("side must be 'min' or 'max'.")

    index = 0 if side == "min" else -1

    face = np.take(
        mask,
        indices=index,
        axis=axis,
    )

    return int(np.sum(face > 0))


def choose_sample_counts(
    mask: np.ndarray,
    inlet_axis: int,
) -> dict:
    pore_count = int(np.sum(mask > 0))

    inlet_count = count_face_pores(
        mask,
        axis=inlet_axis,
        side="min",
    )

    outlet_count = count_face_pores(
        mask,
        axis=inlet_axis,
        side="max",
    )

    if pore_count == 0:
        raise ValueError("No pore voxels found after pore-only filtering.")

    if inlet_count == 0:
        raise ValueError("No inlet pore voxels found.")

    if outlet_count == 0:
        raise ValueError("No outlet pore voxels found.")

    n_interior = min(
        100_000,
        max(30_000, pore_count // 4),
    )

    n_boundary = min(
        75_000,
        max(10_000, pore_count // 8),
    )

    n_inlet = min(
        8_000,
        inlet_count,
    )

    n_outlet = min(
        8_000,
        outlet_count,
    )

    return {
        "pore_count": pore_count,
        "inlet_count": inlet_count,
        "outlet_count": outlet_count,
        "n_interior": n_interior,
        "n_boundary": n_boundary,
        "n_inlet": n_inlet,
        "n_outlet": n_outlet,
    }


def validate_one(
    tif_path: Path,
    pore_value: int = 1,
    inlet_axis: int = 0,
    epochs: int = 20000,
) -> dict:
    name = tif_path.stem.replace("_bin", "")

    print(f"\n===== POINT-CLOUD VALIDATION: {name} =====")

    raw = tifffile.imread(tif_path)

    # ---------------- PORE-ONLY FILTER ----------------
    # This forces:
    # 1 = pore / empty space
    # 0 = solid
    mask = pore_only_filter(
        raw,
        pore_value=pore_value,
    )

    pore_only_porosity = compute_porosity(mask)

    print("Pore-only porosity:", pore_only_porosity)

    # ---------------- CONNECTED PORE FILTER ----------------
    # Keep only pores connected from inlet to outlet.
    try:
        mask = keep_percolating_pores(
            mask,
            axis=inlet_axis,
        )
        connected_filter_success = True

    except ValueError as e:
        print("WARNING: connected filtering failed:", e)
        print("Using pore-only mask without connected filtering.")
        connected_filter_success = False

    epsilon = compute_porosity(mask)

    print("Filtered porosity:", epsilon)

    sample_counts = choose_sample_counts(
        mask,
        inlet_axis=inlet_axis,
    )

    print("Pore voxels:", sample_counts["pore_count"])
    print("Inlet pore voxels:", sample_counts["inlet_count"])
    print("Outlet pore voxels:", sample_counts["outlet_count"])
    print("n_interior:", sample_counts["n_interior"])
    print("n_boundary:", sample_counts["n_boundary"])
    print("n_inlet:", sample_counts["n_inlet"])
    print("n_outlet:", sample_counts["n_outlet"])

    # ---------------- TAUFACTOR BASELINE ----------------
    tau_result = compute_taufactor(mask)

    tau_taufactor = float(tau_result["tau"])
    D_rel_taufactor = float(tau_result["D_eff"])

    # ---------------- MESH FOR PHYSICSNEMO ----------------
    stl_path = tif_path.parent / f"{name}_pointcloud.stl"

    mask_to_stl(
        mask=mask,
        voxel_size_m=1.0,
        out_path=stl_path,
    )

    geom = PoreGeometry(
        stl_path=stl_path,
        mask=mask,
        voxel_size_m=1.0,
        filter_connected=False,
        inlet_axis=inlet_axis,
    )

    output_dir = tif_path.parent / f"{name}_pointcloud_outputs"
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------- DETERMINISTIC VOXEL-GRAPH SOLVE ----------------
    graph_result = solve_voxel_graph_laplace_for_tau(
        mask=mask,
        epsilon=epsilon,
        axis=inlet_axis,
        output_vtk=output_dir / "voxel_graph_concentration.vtk",
        return_field=True,
    )

    tau_graph = float(graph_result["tau"])
    D_rel_graph = float(graph_result["D_rel"])

    # ---------------- POINT-CLOUD PINN ----------------
    tau_pinn, D_rel_pinn = solve_pointcloud_laplace_for_tau(
        geom=geom,
        epsilon=epsilon,
        inlet_axis=inlet_axis,
        epochs=epochs,
        lr=5e-5,
        n_interior=sample_counts["n_interior"],
        n_boundary=sample_counts["n_boundary"],
        n_inlet=sample_counts["n_inlet"],
        n_outlet=sample_counts["n_outlet"],
        near_wall_fraction=0.25,
        pde_weight=1.0,
        bc_weight=50.0,
        wall_weight=0.0,
        energy_weight=25.0,
        teacher_weight=250.0,
        teacher_edge_weight=250.0,
        teacher_transport_edge_weight=500.0,
        teacher_concentration_grid=graph_result["concentration"],
        rff_sigma=6.0,
        rff_n_freqs=128,
        rff_seed=42,
        output_vtk=output_dir / "pointcloud_concentration.vtk",
    )

    D_rel_ratio = D_rel_pinn / D_rel_taufactor
    tau_rel_error_percent = abs(tau_pinn - tau_taufactor) / tau_taufactor * 100
    graph_D_rel_ratio = D_rel_graph / D_rel_taufactor
    graph_tau_rel_error_percent = (
        abs(tau_graph - tau_taufactor) / tau_taufactor * 100
    )

    if D_rel_ratio > 1.5:
        interpretation = "pointcloud_overconducts_geometry"
    elif D_rel_ratio < 0.67:
        interpretation = "pointcloud_underconducts_geometry"
    else:
        interpretation = "pointcloud_reasonable"

    recommended_method = "voxel_graph_laplace"
    recommended_tau = tau_graph
    recommended_D_rel = D_rel_graph

    return {
        "name": name,
        "pore_value": pore_value,
        "transport_axis": inlet_axis,
        "connected_filter_success": connected_filter_success,
        "pore_only_porosity": pore_only_porosity,
        "filtered_porosity": epsilon,
        "pore_count": sample_counts["pore_count"],
        "inlet_pore_count": sample_counts["inlet_count"],
        "outlet_pore_count": sample_counts["outlet_count"],
        "n_interior": sample_counts["n_interior"],
        "n_boundary": sample_counts["n_boundary"],
        "n_inlet": sample_counts["n_inlet"],
        "n_outlet": sample_counts["n_outlet"],
        "tau_taufactor": tau_taufactor,
        "D_rel_taufactor": D_rel_taufactor,
        "recommended_method": recommended_method,
        "recommended_tau": recommended_tau,
        "recommended_D_rel": recommended_D_rel,
        "recommended_tau_abs_error_vs_taufactor": abs(
            recommended_tau - tau_taufactor
        ),
        "recommended_tau_rel_error_percent_vs_taufactor": (
            abs(recommended_tau - tau_taufactor) / tau_taufactor * 100
        ),
        "tau_voxel_graph": tau_graph,
        "D_rel_voxel_graph": D_rel_graph,
        "voxel_graph_D_rel_ratio_vs_taufactor": graph_D_rel_ratio,
        "voxel_graph_tau_abs_error": abs(tau_graph - tau_taufactor),
        "voxel_graph_tau_rel_error_percent": graph_tau_rel_error_percent,
        "voxel_graph_cg_info": graph_result["cg_info"],
        "voxel_graph_flux_relative_error": graph_result["flux_relative_error"],
        "tau_pointcloud_pinn": tau_pinn,
        "D_rel_pointcloud_pinn": D_rel_pinn,
        "D_rel_ratio_vs_taufactor": D_rel_ratio,
        "tau_abs_error": abs(tau_pinn - tau_taufactor),
        "tau_rel_error_percent": tau_rel_error_percent,
        "interpretation": interpretation,
        "stl_path": str(stl_path),
        "output_dir": str(output_dir),
    }


def main() -> None:
    data_dir = Path("runs/synthetic_examples")
    output_csv = data_dir / "tortuosity_pointcloud_validation.csv"
    summary_csv = data_dir / "tortuosity_recommended_summary.csv"

    tif_files = sorted(data_dir.glob("*_bin.tif"))

    if not tif_files:
        raise FileNotFoundError(f"No *_bin.tif files found in {data_dir}")

    rows = []

    for tif_path in tif_files:
        row = validate_one(
            tif_path=tif_path,
            pore_value=1,
            inlet_axis=0,
            epochs=20000,
        )

        rows.append(row)

        print("\nResult:")
        print("TauFactor tau:", row["tau_taufactor"])
        print("Recommended method:", row["recommended_method"])
        print("Recommended tau:", row["recommended_tau"])
        print("Voxel-graph tau:", row["tau_voxel_graph"])
        print("Voxel-graph error %:", row["voxel_graph_tau_rel_error_percent"])
        print("Voxel-graph CG info:", row["voxel_graph_cg_info"])
        print("Point-cloud PINN tau:", row["tau_pointcloud_pinn"])
        print("D_rel ratio:", row["D_rel_ratio_vs_taufactor"])
        print("Error %:", row["tau_rel_error_percent"])
        print("Interpretation:", row["interpretation"])

    df = pd.DataFrame(rows)

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        output_csv,
        index=False,
    )

    summary_columns = [
        "name",
        "filtered_porosity",
        "tau_taufactor",
        "D_rel_taufactor",
        "recommended_method",
        "recommended_tau",
        "recommended_D_rel",
        "recommended_tau_rel_error_percent_vs_taufactor",
        "tau_voxel_graph",
        "D_rel_voxel_graph",
        "voxel_graph_cg_info",
        "tau_pointcloud_pinn",
        "D_rel_pointcloud_pinn",
        "tau_rel_error_percent",
        "interpretation",
    ]

    df[summary_columns].to_csv(
        summary_csv,
        index=False,
    )

    print(f"\nSaved point-cloud validation to: {output_csv}")
    print(f"Saved recommended summary to: {summary_csv}")
    print(df)


if __name__ == "__main__":
    main()
