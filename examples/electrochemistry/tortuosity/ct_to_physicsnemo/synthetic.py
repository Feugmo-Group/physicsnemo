"""Generate 4 synthetic graphite-like CT examples.

Examples:
1. ellipsoid_high_porosity
2. ellipsoid_low_porosity
3. grf_high_porosity
4. grf_low_porosity

Convention:
1 = pore
0 = solid graphite
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv
import tifffile
from scipy.ndimage import gaussian_filter


def compute_porosity(mask: np.ndarray) -> float:
    """Porosity = fraction of pore voxels."""
    return float(np.mean(mask > 0))


def add_ellipsoid(
    solid: np.ndarray,
    center: tuple[float, float, float],
    radii: tuple[float, float, float],
) -> np.ndarray:
    """Add one flattened ellipsoid to the solid phase."""

    nx, ny, nz = solid.shape
    cx, cy, cz = center
    rx, ry, rz = radii

    x, y, z = np.ogrid[:nx, :ny, :nz]

    ellipsoid = (
        ((x - cx) / rx) ** 2
        + ((y - cy) / ry) ** 2
        + ((z - cz) / rz) ** 2
    ) <= 1.0

    solid[ellipsoid] = 1

    return solid


def generate_ellipsoid_volume(
    shape: tuple[int, int, int] = (128, 128, 128),
    n_particles: int = 300,
    radius_xy_range: tuple[float, float] = (6.0, 15.0),
    radius_z_range: tuple[float, float] = (2.0, 6.0),
    seed: int | None = 0,
) -> np.ndarray:
    """Generate flattened M&M-like packed ellipsoid graphite particles."""

    rng = np.random.default_rng(seed)

    # 1 = solid, 0 = pore/background
    solid = np.zeros(shape, dtype=np.uint8)

    for _ in range(n_particles):
        rx = rng.uniform(*radius_xy_range)
        ry = rng.uniform(*radius_xy_range)
        rz = rng.uniform(*radius_z_range)

        margin = int(max(rx, ry, rz)) + 2

        if (
            shape[0] <= 2 * margin
            or shape[1] <= 2 * margin
            or shape[2] <= 2 * margin
        ):
            continue

        cx = rng.integers(margin, shape[0] - margin)
        cy = rng.integers(margin, shape[1] - margin)
        cz = rng.integers(margin, shape[2] - margin)

        solid = add_ellipsoid(
            solid,
            center=(cx, cy, cz),
            radii=(rx, ry, rz),
        )

    # Convert to project convention:
    # 1 = pore, 0 = solid
    pore_mask = 1 - solid

    return pore_mask.astype(np.uint8)


def generate_grf_volume(
    shape: tuple[int, int, int] = (128, 128, 128),
    porosity_target: float = 0.35,
    smooth_sigma: float = 5.0,
    seed: int | None = 0,
) -> np.ndarray:
    """Generate sponge-like porous volume using Gaussian Random Field."""

    rng = np.random.default_rng(seed)

    noise = rng.normal(size=shape)

    field = gaussian_filter(
        noise,
        sigma=smooth_sigma,
    )

    threshold = np.quantile(
        field,
        1.0 - porosity_target,
    )

    pore_mask = field > threshold

    return pore_mask.astype(np.uint8)


def save_vtk(
    mask: np.ndarray,
    path: str | Path,
) -> None:
    """Save binary volume as VTK for ParaView."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    grid = pv.ImageData()
    grid.dimensions = np.array(mask.shape) + 1
    grid.cell_data["values"] = mask.flatten(order="F")

    grid.save(path)

    print(f"Saved VTK: {path}")


def make_examples(
    out_dir: str | Path = "runs/synthetic_examples",
) -> pd.DataFrame:
    """Create 4 controlled synthetic microstructures (quick smoke test set)."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        {
            "name": "ellipsoid_high_porosity",
            "method": "ellipsoid",
            "seed": 0,
            "n_particles": 220,
        },
        {
            "name": "ellipsoid_low_porosity",
            "method": "ellipsoid",
            "seed": 1,
            "n_particles": 520,
        },
        {
            "name": "grf_high_porosity",
            "method": "grf",
            "seed": 2,
            "porosity_target": 0.55,
            "smooth_sigma": 5.0,
        },
        {
            "name": "grf_low_porosity",
            "method": "grf",
            "seed": 3,
            "porosity_target": 0.30,
            "smooth_sigma": 5.0,
        },
    ]

    rows = []

    for cfg in configs:
        name = cfg["name"]

        if cfg["method"] == "ellipsoid":
            mask = generate_ellipsoid_volume(
                n_particles=cfg["n_particles"],
                seed=cfg["seed"],
            )

        elif cfg["method"] == "grf":
            mask = generate_grf_volume(
                porosity_target=cfg["porosity_target"],
                smooth_sigma=cfg["smooth_sigma"],
                seed=cfg["seed"],
            )

        else:
            raise ValueError(f"Unknown method: {cfg['method']}")

        porosity = compute_porosity(mask)

        tif_path = out_dir / f"{name}_bin.tif"
        vtk_path = out_dir / f"{name}.vtk"

        tifffile.imwrite(tif_path, mask)
        save_vtk(mask, vtk_path)

        row = {
            "name": name,
            "method": cfg["method"],
            "porosity": porosity,
            "tif_path": str(tif_path),
            "vtk_path": str(vtk_path),
            "shape": str(mask.shape),
            **cfg,
        }

        rows.append(row)

        print(f"Saved {name} | porosity = {porosity:.4f}")

    df = pd.DataFrame(rows)

    metadata_path = out_dir / "metadata.csv"
    df.to_csv(metadata_path, index=False)

    print(f"\nSaved metadata: {metadata_path}")

    return df


def make_library(
    out_dir: str | Path = "runs/synthetic_library",
    shape: tuple[int, int, int] = (128, 128, 128),
) -> pd.DataFrame:
    """Generate the 20-volume synthetic library for E2 training.

    Covers a grid of (porosity, correlation_length) values using two generators:
    - 8 ellipsoid volumes: varying packing density (porosity range ~0.25–0.75)
    - 12 GRF volumes:      varying porosity_target × smooth_sigma

    Each volume is saved as a *_bin.tif file and indexed in metadata.csv.
    This is the Week 1 deliverable from the summer project plan.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 8 ellipsoid configs: n_particles sweeps porosity from ~0.70 down to ~0.25
    ellipsoid_configs = [
        {"name": f"ellipsoid_{i+1:02d}", "n_particles": n, "seed": i}
        for i, n in enumerate([80, 150, 220, 300, 380, 450, 520, 600])
    ]

    # 12 GRF configs: 3 porosity targets × 4 correlation lengths
    porosity_targets = [0.55, 0.40, 0.28]
    smooth_sigmas    = [3.0, 5.0, 8.0, 12.0]
    grf_configs = [
        {
            "name": f"grf_p{int(p*100):02d}_s{int(s):02d}",
            "porosity_target": p,
            "smooth_sigma": s,
            "seed": 100 + i,
        }
        for i, (p, s) in enumerate(
            (p, s) for p in porosity_targets for s in smooth_sigmas
        )
    ]

    rows = []

    for cfg in ellipsoid_configs:
        mask = generate_ellipsoid_volume(
            shape=shape,
            n_particles=cfg["n_particles"],
            seed=cfg["seed"],
        )
        porosity = compute_porosity(mask)
        tif_path = out_dir / f"{cfg['name']}_bin.tif"
        tifffile.imwrite(tif_path, mask)
        row = {
            "name": cfg["name"],
            "method": "ellipsoid",
            "porosity": porosity,
            "n_particles": cfg["n_particles"],
            "smooth_sigma": None,
            "porosity_target": None,
            "seed": cfg["seed"],
            "shape": str(mask.shape),
            "tif_path": str(tif_path),
        }
        rows.append(row)
        print(f"  {cfg['name']:30s}  ε={porosity:.3f}")

    for cfg in grf_configs:
        mask = generate_grf_volume(
            shape=shape,
            porosity_target=cfg["porosity_target"],
            smooth_sigma=cfg["smooth_sigma"],
            seed=cfg["seed"],
        )
        porosity = compute_porosity(mask)
        tif_path = out_dir / f"{cfg['name']}_bin.tif"
        tifffile.imwrite(tif_path, mask)
        row = {
            "name": cfg["name"],
            "method": "grf",
            "porosity": porosity,
            "n_particles": None,
            "smooth_sigma": cfg["smooth_sigma"],
            "porosity_target": cfg["porosity_target"],
            "seed": cfg["seed"],
            "shape": str(mask.shape),
            "tif_path": str(tif_path),
        }
        rows.append(row)
        print(f"  {cfg['name']:30s}  ε={porosity:.3f}  σ={cfg['smooth_sigma']}")

    df = pd.DataFrame(rows)
    metadata_path = out_dir / "metadata.csv"
    df.to_csv(metadata_path, index=False)
    print(f"\n{len(df)} volumes saved to {out_dir}")
    print(f"Metadata: {metadata_path}")
    return df


def make_library_200(
    out_dir: str | Path = "runs/synthetic_library_200",
    shape: tuple[int, int, int] = (128, 128, 128),
) -> pd.DataFrame:
    """Generate a 200-volume synthetic library (100 ellipsoid + 100 GRF).

    Ellipsoids (100):
      n_particles sweep from 20 to 700 — covers ε ≈ 0.20 to 0.95

    GRFs (100):
      5 porosity targets × 4 sigmas × 5 seeds = 100
      porosity_targets : [0.65, 0.55, 0.40, 0.28, 0.20]
      smooth_sigmas    : [3.0, 5.0, 8.0, 12.0]
      seeds per class  : 5

    Gives 25 GRF examples per (porosity, sigma) class — enough for the
    network to learn the fine-pore high-τ regimes (σ=3, σ=5) that the
    20-volume library failed on.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 100 ellipsoids: n_particles from 20 to 700
    n_particles_list = [int(20 + i * (700 - 20) / 99) for i in range(100)]
    ellipsoid_configs = [
        {"name": f"ellipsoid_{i+1:03d}", "n_particles": n, "seed": i}
        for i, n in enumerate(n_particles_list)
    ]

    # 100 GRFs: 5 porosity targets × 4 sigmas × 5 seeds
    porosity_targets = [0.65, 0.55, 0.40, 0.28, 0.20]
    smooth_sigmas    = [3.0, 5.0, 8.0, 12.0]
    seeds_per_class  = [0, 1, 2, 3, 4]
    grf_configs = [
        {
            "name": f"grf_p{int(p*100):02d}_s{int(s):02d}_r{r}",
            "porosity_target": p,
            "smooth_sigma": s,
            "seed": 500 + idx * 5 + r,
        }
        for idx, (p, s) in enumerate((p, s) for p in porosity_targets for s in smooth_sigmas)
        for r in seeds_per_class
    ]

    rows = []

    print(f"Generating 200-volume library → {out_dir}")
    print(f"  100 ellipsoids  +  100 GRFs (5 porosity × 4 sigma × 5 seeds)")
    print("-" * 65)

    for cfg in ellipsoid_configs:
        mask = generate_ellipsoid_volume(
            shape=shape,
            n_particles=cfg["n_particles"],
            seed=cfg["seed"],
        )
        porosity = compute_porosity(mask)
        tif_path = out_dir / f"{cfg['name']}_bin.tif"
        tifffile.imwrite(tif_path, mask)
        rows.append({
            "name": cfg["name"], "method": "ellipsoid",
            "porosity": porosity, "n_particles": cfg["n_particles"],
            "smooth_sigma": None, "porosity_target": None,
            "seed": cfg["seed"], "shape": str(mask.shape), "tif_path": str(tif_path),
        })
        print(f"  {cfg['name']:38s}  ε={porosity:.3f}  n_particles={cfg['n_particles']}")

    for cfg in grf_configs:
        mask = generate_grf_volume(
            shape=shape,
            porosity_target=cfg["porosity_target"],
            smooth_sigma=cfg["smooth_sigma"],
            seed=cfg["seed"],
        )
        porosity = compute_porosity(mask)
        tif_path = out_dir / f"{cfg['name']}_bin.tif"
        tifffile.imwrite(tif_path, mask)
        rows.append({
            "name": cfg["name"], "method": "grf",
            "porosity": porosity, "n_particles": None,
            "smooth_sigma": cfg["smooth_sigma"], "porosity_target": cfg["porosity_target"],
            "seed": cfg["seed"], "shape": str(mask.shape), "tif_path": str(tif_path),
        })
        print(f"  {cfg['name']:38s}  ε={porosity:.3f}  σ={cfg['smooth_sigma']}")

    df = pd.DataFrame(rows)
    metadata_path = out_dir / "metadata.csv"
    df.to_csv(metadata_path, index=False)
    print(f"\n{len(df)} volumes saved to {out_dir}")
    return df


if __name__ == "__main__":
    import sys
    if "--library200" in sys.argv:
        make_library_200()
    elif "--library" in sys.argv:
        make_library()
    else:
        make_examples()