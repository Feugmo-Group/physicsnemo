"""Microstructural metrics: porosity, tau, surface area, PSD."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import taufactor as tau
from scipy import ndimage
from skimage.measure import marching_cubes

from ct_to_physicsnemo.io import load_ct, crop_rev


def compute_porosity(volume: np.ndarray) -> float:
    """Porosity = fraction of pore voxels.

    Assumes:
    1 = pore
    0 = solid graphite
    """
    return float(np.mean(volume > 0))


def compute_taufactor(volume: np.ndarray) -> dict:
    """Compute effective diffusivity and tortuosity using TauFactor."""

    binary = (volume > 0).astype(np.uint8)

    solver = tau.Solver(binary)
    solver.solve()

    return {
        "D_eff": float(np.asarray(solver.D_eff).ravel()[0]),
        "tau": float(np.asarray(solver.tau).ravel()[0]),
    }


def compute_surface_area(
    volume: np.ndarray,
    voxel_size: float = 1.0,
) -> float:
    """Estimate pore-solid interface surface area using marching cubes."""

    binary = (volume > 0).astype(np.uint8)

    verts, faces, _, _ = marching_cubes(
        binary,
        level=0.5,
        spacing=(voxel_size, voxel_size, voxel_size),
    )

    triangles = verts[faces]

    a = triangles[:, 1] - triangles[:, 0]
    b = triangles[:, 2] - triangles[:, 0]

    area = 0.5 * np.linalg.norm(
        np.cross(a, b),
        axis=1,
    ).sum()

    return float(area)


def compute_particle_diameters(
    volume: np.ndarray,
    voxel_size: float = 1.0,
    min_voxels: int = 10,
) -> np.ndarray:
    """Compute equivalent sphere diameters from connected solid particles."""

    # solid graphite particles are labeled as 0
    solid = volume == 0

    labels, n_labels = ndimage.label(solid)

    diameters = []

    for label_id in range(1, n_labels + 1):
        voxel_count = np.sum(labels == label_id)

        if voxel_count < min_voxels:
            continue

        particle_volume = voxel_count * voxel_size**3

        # equivalent sphere diameter
        diameter = (6.0 * particle_volume / np.pi) ** (1.0 / 3.0)

        diameters.append(diameter)

    return np.asarray(diameters, dtype=float)


def compute_fft_descriptor(
    mask: np.ndarray,
    k_max: int = 5,
    normalize: bool = True,
) -> np.ndarray:
    """Compute a global geometry descriptor from the 3D FFT of the pore indicator.

    Takes the binary pore mask, computes its 3D FFT, and returns the magnitudes
    of the (2*k_max+1)^3 lowest-frequency coefficients as a flat descriptor vector.
    This is the E2 geometry fingerprint used to condition the multi-geometry PINN.

    Properties:
    - Translation-invariant: FFT magnitudes are phase-agnostic.
    - Multi-scale: low-k modes capture bulk porosity and elongation; mid-k modes
      capture connectivity texture and pore-size distribution.
    - Cheap: computed in milliseconds on CPU; no meshing required.
    - Fixed size regardless of volume dimensions.

    Args:
        mask: Binary pore mask (1 = pore, 0 = solid), shape (Nx, Ny, Nz).
        k_max: Half-width of the low-frequency shell to keep. Default 5 gives a
               (2*5+1)^3 = 11^3 = 1331 shell, but only the (2*k_max+1)^3 = 11^3
               central coefficients are taken after fftshift. Typical choice:
               k_max=2 → 125 coefficients, k_max=5 → 1331 coefficients.
        normalize: If True, divide all magnitudes by the DC component (k=0,0,0)
                   so the descriptor is scale-invariant with respect to porosity.

    Returns:
        Flat float32 array of shape ((2*k_max+1)**3,) containing FFT magnitudes.
    """
    chi = (mask > 0).astype(np.float32)

    # Zero-pad to next power of two for FFT efficiency
    fft_full = np.fft.fftn(chi)
    fft_shifted = np.fft.fftshift(fft_full)  # DC at center
    magnitudes = np.abs(fft_shifted)

    # Extract the (2*k_max+1)^3 central block
    cx, cy, cz = [s // 2 for s in magnitudes.shape]
    block = magnitudes[
        cx - k_max : cx + k_max + 1,
        cy - k_max : cy + k_max + 1,
        cz - k_max : cz + k_max + 1,
    ]

    descriptor = block.ravel().astype(np.float32)

    if normalize:
        dc = descriptor[len(descriptor) // 2]  # center element = DC after reshape
        if dc > 0:
            descriptor = descriptor / dc

    return descriptor


def compute_psd_summary(diameters: np.ndarray) -> dict:
    """Summarise particle size distribution."""

    if len(diameters) == 0:
        return {
            "psd_n_particles": 0,
            "psd_d_mean": np.nan,
            "psd_d_std": np.nan,
            "psd_d_min": np.nan,
            "psd_d_max": np.nan,
            "psd_d10": np.nan,
            "psd_d50": np.nan,
            "psd_d90": np.nan,
        }

    return {
        "psd_n_particles": int(len(diameters)),
        "psd_d_mean": float(np.mean(diameters)),
        "psd_d_std": float(np.std(diameters)),
        "psd_d_min": float(np.min(diameters)),
        "psd_d_max": float(np.max(diameters)),
        "psd_d10": float(np.percentile(diameters, 10)),
        "psd_d50": float(np.percentile(diameters, 50)),
        "psd_d90": float(np.percentile(diameters, 90)),
    }


def summarise_metrics(
    volume: np.ndarray,
    voxel_size: float = 1.0,
) -> dict:
    """Compute all microstructural metrics for one binary volume."""

    porosity = compute_porosity(volume)
    tau_result = compute_taufactor(volume)

    surface_area = compute_surface_area(
        volume,
        voxel_size=voxel_size,
    )

    diameters = compute_particle_diameters(
        volume,
        voxel_size=voxel_size,
    )

    psd_stats = compute_psd_summary(diameters)

    return {
        "porosity": porosity,
        "D_eff": tau_result["D_eff"],
        "tau": tau_result["tau"],
        "surface_area": surface_area,
        **psd_stats,
    }


def run_all_metrics(
    data_dir: str | Path,
    output_csv: str | Path,
    voxel_size_nm: float = 50,
    rev_microns: float = 5,
    crop: bool = True,
) -> pd.DataFrame:
    """Run metrics on every *_bin.tif file in a folder."""

    data_dir = Path(data_dir)
    output_csv = Path(output_csv)

    bin_files = sorted(data_dir.rglob("*_bin.tif"))

    if not bin_files:
        raise FileNotFoundError(f"No *_bin.tif files found in {data_dir}")

    results = []

    voxel_size = voxel_size_nm

    for path in bin_files:
        print(f"\nProcessing: {path}")

        volume = load_ct(path)

        if crop:
            used_volume = crop_rev(
                volume,
                voxel_size_nm=voxel_size_nm,
                rev_microns=rev_microns,
            )
        else:
            used_volume = volume

        metrics = summarise_metrics(
            used_volume,
            voxel_size=voxel_size,
        )

        row = {
            "file": str(path),
            "name": path.stem,
            "shape_original": tuple(volume.shape),
            "shape_used": tuple(used_volume.shape),
            "voxel_size_nm": voxel_size_nm,
            "rev_microns": rev_microns if crop else None,
            "cropped": crop,
            **metrics,
        }

        results.append(row)

    df = pd.DataFrame(results)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"\nSaved metrics to: {output_csv}")

    # ---------------- MERGE WITH METADATA ----------------

    metadata_path = data_dir / "metadata.csv"

    if metadata_path.exists():

        metadata_df = pd.read_csv(metadata_path)

        merged_df = metadata_df.merge(
            df,
            on="name",
            how="left",
        )

        merged_csv = output_csv.parent / "synthetic_metadata_with_metrics.csv"

        merged_df.to_csv(merged_csv, index=False)

        print(f"\nSaved merged metadata to: {merged_csv}")

    return df


if __name__ == "__main__":

    MODE = "real"  # change to "real" when needed

    if MODE == "synthetic":
        df = run_all_metrics(
            data_dir="runs/synthetic_library",
            output_csv="runs/synthetic_metrics.csv",
            voxel_size_nm=1,
            crop=False,
        )

    elif MODE == "real":
        df = run_all_metrics(
            data_dir="../diffusion_in_porous_media",
            output_csv="runs/metrics.csv",
            voxel_size_nm=50,
            rev_microns=5,
            crop=True,
        )

    else:
        raise ValueError(f"Unknown MODE: {MODE}")

    print("\n===== FINAL RESULTS =====")
    print(df)
    print("\n===== SUMMARY STATS =====")
    print(df[
    [
        "porosity",
        "tau",
        "surface_area",
        "psd_d50",
        "psd_d90",
    ]
])

