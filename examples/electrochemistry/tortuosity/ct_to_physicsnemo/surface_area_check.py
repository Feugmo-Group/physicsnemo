"""Estimate synthetic surface area using PhysicsNeMo boundary sampling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from ct_to_physicsnemo.mesh import mask_to_stl
from physicsnemo.sym.geometry.tessellation import Tessellation


def physicsnemo_surface_area(
    tif_path: str | Path,
    stl_path: str | Path,
    n_boundary: int = 200_000,
    voxel_size: float = 1.0,
) -> float:
    mask = tifffile.imread(tif_path)
    mask = (mask > 0).astype(np.uint8)

    mask_to_stl(
        mask=mask,
        voxel_size_m=voxel_size,
        out_path=stl_path,
    )

    geo = Tessellation.from_stl(
        str(stl_path),
        airtight=False,
    )

    boundary = geo.sample_boundary(n_boundary)

    area = float(np.sum(boundary["area"]))

    return area


def run_surface_area_check(
    data_dir: str | Path = "runs/synthetic_examples",
    output_csv: str | Path = "runs/synthetic_examples/surface_area_physicsnemo_check.csv",
) -> pd.DataFrame:
    data_dir = Path(data_dir)
    output_csv = Path(output_csv)

    rows = []

    for tif_path in sorted(data_dir.glob("*_bin.tif")):
        name = tif_path.stem.replace("_bin", "")
        stl_path = data_dir / f"{name}.stl"

        print(f"\nProcessing: {name}")

        area = physicsnemo_surface_area(
            tif_path=tif_path,
            stl_path=stl_path,
            n_boundary=200_000,
            voxel_size=1.0,
        )

        rows.append(
            {
                "name": name,
                "physicsnemo_surface_area": area,
                "stl_path": str(stl_path),
            }
        )

        print("PhysicsNeMo surface area:", area)

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)

    print(f"\nSaved: {output_csv}")

    return df


if __name__ == "__main__":
    run_surface_area_check()