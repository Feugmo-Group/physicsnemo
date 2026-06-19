"""Baseline validation of PhysicsNeMo surface area against exact mesh area from SAME STL.

"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import trimesh

from physicsnemo.sym.geometry.tessellation import Tessellation

from ct_to_physicsnemo.mesh import mask_to_stl


def load_mask(path: str | Path) -> np.ndarray:
    mask = tifffile.imread(path)
    return (mask > 0).astype(np.uint8)


def trimesh_surface_area(stl_path: str | Path) -> float:
    """Baseline mesh area from EXACT SAME STL."""
    mesh = trimesh.load_mesh(stl_path)

    return float(mesh.area)


def physicsnemo_surface_area(
    stl_path: str | Path,
    n_boundary: int = 1_000_000,
) -> float:
    """PhysicsNeMo boundary-integrated surface area."""

    geo = Tessellation.from_stl(
        str(stl_path),
        airtight=False,
    )

    boundary = geo.sample_boundary(n_boundary)

    return float(np.sum(boundary["area"]))


def validate_surface_area_folder(
    data_dir: str | Path = "runs/synthetic_examples",
    output_csv: str | Path = "runs/synthetic_examples/surface_area_validation.csv",
    voxel_size: float = 1.0,
    n_boundary: int = 1_000_000,
) -> pd.DataFrame:

    data_dir = Path(data_dir)
    output_csv = Path(output_csv)

    tif_files = sorted(data_dir.glob("*_bin.tif"))

    rows = []

    for tif_path in tif_files:

        name = tif_path.stem.replace("_bin", "")

        print(f"\n===== PROCESSING {name} =====")

        mask = load_mask(tif_path)

        stl_path = data_dir / f"{name}.stl"

        # Generate ONE shared STL mesh
        mask_to_stl(
            mask=mask,
            voxel_size_m=voxel_size,
            out_path=stl_path,
        )

        # SAME mesh -> deterministic baseline
        area_trimesh = trimesh_surface_area(stl_path)

        # SAME mesh -> PhysicsNeMo estimate
        area_physicsnemo = physicsnemo_surface_area(
            stl_path=stl_path,
            n_boundary=n_boundary,
        )

        abs_error = abs(area_physicsnemo - area_trimesh)

        rel_error_percent = (
            abs_error / area_trimesh * 100.0
        )

        total_volume = (
            np.prod(mask.shape) * voxel_size**3
        )

        av_trimesh = area_trimesh / total_volume
        av_physicsnemo = area_physicsnemo / total_volume

        row = {
            "name": name,
            "shape": tuple(mask.shape),
            "mesh_surface_area": area_trimesh,
            "physicsnemo_surface_area": area_physicsnemo,
            "absolute_error": abs_error,
            "relative_error_percent": rel_error_percent,
            "mesh_specific_surface_area": av_trimesh,
            "physicsnemo_specific_surface_area": av_physicsnemo,
            "n_boundary_samples": n_boundary,
            "stl_path": str(stl_path),
        }

        rows.append(row)

        print(f"Mesh surface area:        {area_trimesh:.6f}")
        print(f"PhysicsNeMo surface area: {area_physicsnemo:.6f}")
        print(f"Relative error:           {rel_error_percent:.4f}%")

    df = pd.DataFrame(rows)

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(output_csv, index=False)

    print(f"\nSaved validation CSV:")
    print(output_csv)

    return df


if __name__ == "__main__":

    df = validate_surface_area_folder(
        data_dir="runs/synthetic_examples",
        output_csv="runs/synthetic_examples/surface_area_validation.csv",
        voxel_size=1.0,
        n_boundary=1_000_000,
    )

    print("\n===== FINAL VALIDATION =====")
    print(df)