"""CT volume I/O and REV cropping. Week 2."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile


def load_ct(path: str | Path) -> np.ndarray:
    """Load a 3D CT volume from a TIFF file.

    For *_bin.tif files:
    - 1 = pore
    - 0 = solid
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"CT file not found: {path}")

    volume = tifffile.imread(path)

    # Convert binary CT into clean 0/1 float array
    volume = (volume > 0).astype(np.float32)

    return volume


def crop_rev(
    volume: np.ndarray,
    voxel_size_nm: float,
    rev_microns: float,
) -> np.ndarray:
    """Center-crop volume to a cubic REV region.

    Parameters
    ----------
    volume:
        3D CT array.

    voxel_size_nm:
        Voxel size in nanometers.

    rev_microns:
        Desired crop side length in microns.
    """

    if volume.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {volume.shape}")

    # Convert REV size from microns to nanometers
    rev_nm = rev_microns * 1000.0

    # Convert physical length to number of voxels
    crop_size = int(round(rev_nm / voxel_size_nm))

    if crop_size <= 0:
        raise ValueError("crop_size must be positive")

    if crop_size > min(volume.shape):
        raise ValueError(
            f"Requested crop size {crop_size} is larger than volume shape {volume.shape}"
        )

    nx, ny, nz = volume.shape

    cx = nx // 2
    cy = ny // 2
    cz = nz // 2

    half = crop_size // 2

    cropped = volume[
        cx - half : cx - half + crop_size,
        cy - half : cy - half + crop_size,
        cz - half : cz - half + crop_size,
    ]

    return cropped


def summarise(volume: np.ndarray) -> dict:
    """Return basic summary statistics for a CT volume."""

    if volume.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {volume.shape}")

    summary = {
        "shape": tuple(volume.shape),
        "dtype": str(volume.dtype),
        "min": float(np.min(volume)),
        "max": float(np.max(volume)),
        "mean": float(np.mean(volume)),
        "porosity": float(np.mean(volume > 0)),
        "nan_count": int(np.isnan(volume).sum()),
    }

    return summary



#------------------ Testing -------------------
if __name__ == "__main__":

    print("\n===== TESTING io.py =====")

    path = (
        "../diffusion_in_porous_media/"
        "Electrode I/I_1/I_1_bin.tif"
    )

    # ---------------- LOAD ----------------

    volume = load_ct(path)

    print("\nLoaded successfully")
    print("Original shape:", volume.shape)

    # ---------------- CROP ----------------

    cropped = crop_rev(
        volume,
        voxel_size_nm=50,
        rev_microns=5,
    )

    print("\nCropped successfully")
    print("Cropped shape:", cropped.shape)

    # ---------------- SUMMARY ----------------

    summary = summarise(cropped)

    print("\n===== SUMMARY =====")

    for k, v in summary.items():
        print(f"{k}: {v}")