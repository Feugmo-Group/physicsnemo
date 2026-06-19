"""Confirm synthetic porosity using PhysicsNeMo sampling.

Compares:
1. voxel-count porosity
2. PhysicsNeMo Box Monte Carlo porosity

Convention:
1 = pore
0 = solid graphite
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from physicsnemo.sym.geometry.primitives_3d import Box


def load_mask(path: str | Path) -> np.ndarray:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Could not find: {path}")

    mask = tifffile.imread(path)

    mask = (mask > 0).astype(np.uint8)

    return mask


def voxel_porosity(mask: np.ndarray) -> float:
    """Porosity from direct voxel count."""
    return float(np.mean(mask > 0))


def physicsnemo_porosity(
    mask: np.ndarray,
    n_samples: int = 200_000,
    voxel_size: float = 1.0,
) -> float:
    """Estimate porosity by sampling a PhysicsNeMo Box.

    PhysicsNeMo samples random points inside the full box.
    We then check whether each point lands in pore phase.
    """

    nx, ny, nz = mask.shape

    box = Box(
        (0.0, 0.0, 0.0),
        (
            nx * voxel_size,
            ny * voxel_size,
            nz * voxel_size,
        ),
    )

    samples = box.sample_interior(n_samples)

    xyz = np.hstack(
        [
            samples["x"],
            samples["y"],
            samples["z"],
        ]
    )

    # Convert physical coordinates to voxel indices
    ijk = np.floor(xyz / voxel_size).astype(int)

    # Prevent edge index overflow
    ijk[:, 0] = np.clip(ijk[:, 0], 0, nx - 1)
    ijk[:, 1] = np.clip(ijk[:, 1], 0, ny - 1)
    ijk[:, 2] = np.clip(ijk[:, 2], 0, nz - 1)

    pore_hits = mask[
        ijk[:, 0],
        ijk[:, 1],
        ijk[:, 2],
    ] > 0

    return float(np.mean(pore_hits))


def check_porosity_folder(
    data_dir: str | Path = "runs/synthetic_examples",
    output_csv: str | Path = "runs/synthetic_examples/porosity_physicsnemo_check.csv",
    n_samples: int = 200_000,
    voxel_size: float = 1.0,
) -> pd.DataFrame:
    data_dir = Path(data_dir)
    output_csv = Path(output_csv)

    tif_files = sorted(data_dir.glob("*_bin.tif"))

    if not tif_files:
        raise FileNotFoundError(f"No *_bin.tif files found in {data_dir}")

    rows = []

    for path in tif_files:
        print(f"\nChecking: {path}")

        mask = load_mask(path)

        eps_voxel = voxel_porosity(mask)

        eps_physicsnemo = physicsnemo_porosity(
            mask,
            n_samples=n_samples,
            voxel_size=voxel_size,
        )

        abs_error = abs(eps_physicsnemo - eps_voxel)

        rel_error_percent = abs_error / eps_voxel * 100

        row = {
            "name": path.stem,
            "shape": tuple(mask.shape),
            "voxel_porosity": eps_voxel,
            "physicsnemo_porosity": eps_physicsnemo,
            "absolute_error": abs_error,
            "relative_error_percent": rel_error_percent,
            "n_samples": n_samples,
        }

        rows.append(row)

        print(f"Voxel porosity:       {eps_voxel:.6f}")
        print(f"PhysicsNeMo porosity: {eps_physicsnemo:.6f}")
        print(f"Relative error:       {rel_error_percent:.3f}%")

    df = pd.DataFrame(rows)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"\nSaved porosity check to: {output_csv}")

    return df


if __name__ == "__main__":
    df = check_porosity_folder(
        data_dir="runs/synthetic_examples",
        output_csv="runs/synthetic_examples/porosity_physicsnemo_check.csv",
        n_samples=200_000,
        voxel_size=1.0,
    )

    print("\n===== FINAL POROSITY CHECK =====")
    print(df)