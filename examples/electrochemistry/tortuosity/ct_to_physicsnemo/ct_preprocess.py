"""Real CT scan preprocessing pipeline — no external mesh tools required.

Full pipeline from raw .tif grayscale stack to training-ready .npz:

    .tif (grayscale)
        → REV crop           (numpy)
        → Otsu threshold     (skimage.filters)
        → binary mask        (pore=1, solid=0)
        → marching cubes     (skimage.measure)
        → mesh SDF           (physicsnemo Warp BVH, GPU)
        → CG solver          (sparse conjugate gradient)
        → .npz               (pore-voxel arrays for streaming training)

No PyMeshLab or any external mesh-repair tool is needed.
The Warp BVH uses winding-number sign determination, which is robust to
non-watertight meshes that marching cubes may produce on noisy CT data.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import tifffile
import torch
from scipy.ndimage import label as scipy_label
from skimage.filters import threshold_otsu
from skimage.measure import marching_cubes

from ct_to_physicsnemo.mesh_sdf import compute_mesh_sdf_grid
from ct_to_physicsnemo.metrics import compute_fft_descriptor, compute_porosity
from ct_to_physicsnemo.voxel_graph_laplace import solve_voxel_graph_laplace_for_tau

INLET_AXIS = 0
K_MAX = 4  # FFT descriptor: (2*4+1)^3 = 729 coefficients


# ---------------------------------------------------------------------------
# Step 1 – REV crop
# ---------------------------------------------------------------------------

def crop_rev(
    vol: np.ndarray,
    rev_voxels: int | None = None,
    origin: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """Extract a cubic Representative Elementary Volume.

    Parameters
    ----------
    vol : np.ndarray
        Full 3D grayscale volume.
    rev_voxels : int, optional
        Side length of the REV cube in voxels. If None, uses the smallest
        dimension of vol (largest cube that fits).
    origin : tuple, optional
        (x0, y0, z0) corner of the REV. If None, centres the REV in vol.

    Returns
    -------
    np.ndarray
        Cropped sub-volume of shape (rev_voxels, rev_voxels, rev_voxels).
    """
    if rev_voxels is None:
        rev_voxels = min(vol.shape)

    if origin is None:
        cx, cy, cz = [s // 2 for s in vol.shape]
        r = rev_voxels // 2
        origin = (cx - r, cy - r, cz - r)

    x0, y0, z0 = origin
    return vol[x0:x0+rev_voxels, y0:y0+rev_voxels, z0:z0+rev_voxels].copy()


# ---------------------------------------------------------------------------
# Step 2 – Otsu segmentation
# ---------------------------------------------------------------------------

def otsu_segment(
    vol: np.ndarray,
    pore_is_dark: bool = True,
) -> np.ndarray:
    """Otsu threshold → binary mask (1 = pore, 0 = solid).

    Parameters
    ----------
    vol : np.ndarray
        Grayscale volume (any integer or float dtype).
    pore_is_dark : bool
        True  → pore voxels are darker than solid (typical for X-ray CT of
                 graphite electrodes: dense graphite is bright, pore is dark).
        False → pore voxels are brighter.

    Returns
    -------
    np.ndarray uint8 (1=pore, 0=solid)
    """
    thresh = threshold_otsu(vol)
    if pore_is_dark:
        mask = (vol <= thresh).astype(np.uint8)
    else:
        mask = (vol > thresh).astype(np.uint8)
    return mask


# ---------------------------------------------------------------------------
# Step 3 – Percolation filter (keep only through-connected pore)
# ---------------------------------------------------------------------------

def keep_percolating_pore(
    mask: np.ndarray,
    axis: int = 0,
) -> np.ndarray:
    """Remove isolated pore clusters that don't span inlet→outlet.

    Keeps only pore voxels that belong to a connected component touching
    both the inlet face and the outlet face along `axis`.

    Parameters
    ----------
    mask : np.ndarray  uint8  (1=pore)
    axis : int  Transport axis (default 0).

    Returns
    -------
    np.ndarray uint8  filtered mask (non-percolating pores set to 0)
    """
    labeled, n = scipy_label(mask)
    if n == 0:
        return mask

    inlet_slice  = [slice(None)] * 3
    outlet_slice = [slice(None)] * 3
    inlet_slice[axis]  = 0
    outlet_slice[axis] = mask.shape[axis] - 1

    inlet_labels  = set(np.unique(labeled[tuple(inlet_slice)])) - {0}
    outlet_labels = set(np.unique(labeled[tuple(outlet_slice)])) - {0}
    percolating   = inlet_labels & outlet_labels

    filtered = np.zeros_like(mask)
    for lbl in percolating:
        filtered[labeled == lbl] = 1
    return filtered


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def preprocess_ct_volume(
    tif_path: str | Path,
    output_dir: str | Path,
    rev_voxels: int | None = None,
    rev_origin: tuple[int, int, int] | None = None,
    pore_is_dark: bool = True,
    axis: int = INLET_AXIS,
    k_max: int = K_MAX,
    device: str | None = None,
    overwrite: bool = False,
) -> dict:
    """Preprocess one real CT .tif volume → training-ready .npz.

    Steps (all pure Python/numpy/skimage/torch — no external mesh tools):
        1. Load .tif
        2. REV crop
        3. Otsu threshold → binary mask
        4. Percolation filter (keep inlet-to-outlet connected pore)
        5. Marching cubes → triangular surface mesh
        6. physicsnemo Warp BVH SDF (GPU, sub-voxel accurate)
        7. Sparse CG solver → concentration field + τ
        8. FFT descriptor
        9. Save .npz

    Returns
    -------
    dict with keys: name, npz_path, epsilon, tau_cg, D_rel, n_pore, elapsed
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tif_path   = Path(tif_path)
    output_dir = Path(output_dir)
    name       = tif_path.stem
    out_path   = output_dir / f"{name}.npz"

    if out_path.exists() and not overwrite:
        print(f"  [SKIP] {name} (already exists)")
        return {"name": name, "npz_path": str(out_path), "skipped": True}

    t0 = time.perf_counter()
    print(f"\n{'='*60}")
    print(f"Processing: {name}")

    # 1. Load
    raw = tifffile.imread(tif_path)
    print(f"  Loaded: {raw.shape}  dtype={raw.dtype}")

    # 2. REV crop
    vol = crop_rev(raw, rev_voxels=rev_voxels, origin=rev_origin)
    print(f"  REV:    {vol.shape}")

    # 3. Otsu threshold
    mask = otsu_segment(vol, pore_is_dark=pore_is_dark)
    raw_epsilon = float(mask.mean())
    print(f"  Otsu:   ε={raw_epsilon:.3f}  (pore_is_dark={pore_is_dark})")

    # 4. Percolation filter
    mask = keep_percolating_pore(mask, axis=axis)
    epsilon = float(mask.mean())
    print(f"  Percolation filter: ε={epsilon:.3f}  (removed {raw_epsilon - epsilon:.4f} isolated pore)")

    if epsilon < 0.05:
        raise ValueError(f"{name}: porosity {epsilon:.3f} < 5% after percolation filter — check pore_is_dark")

    # 5+6. Mesh SDF via physicsnemo Warp BVH (no PyMeshLab needed)
    print(f"  Computing mesh SDF (Warp BVH on {device}) …")
    sdf_grid = compute_mesh_sdf_grid(mask, device=device)
    print(f"  SDF:    range [{sdf_grid.min():.3f}, {sdf_grid.max():.3f}]")

    # 7. CG solver
    print(f"  Running CG solver …")
    cg = solve_voxel_graph_laplace_for_tau(
        mask, epsilon=epsilon, axis=axis, return_field=True,
    )
    tau_cg = float(cg["tau"])
    D_rel  = float(cg["D_rel"])
    print(f"  CG:     τ={tau_cg:.3f}  D_rel={D_rel:.4f}")

    # 8. FFT descriptor
    fft_desc   = compute_fft_descriptor(mask, k_max=k_max)
    from ct_to_physicsnemo.metrics import compute_surface_area
    _, av      = compute_surface_area(mask, voxel_size=1.0)
    descriptor = np.concatenate([fft_desc, [epsilon, av]]).astype(np.float32)
    print(f"  Descriptor: {descriptor.shape[0]} dims  (FFT={len(fft_desc)} + ε + a_v)")

    # 9. Pack pore-voxel arrays
    pore_idx = np.argwhere(mask > 0).astype(np.int16)
    N        = pore_idx.shape[0]

    teacher_vals = cg["concentration"][
        pore_idx[:, 0], pore_idx[:, 1], pore_idx[:, 2]
    ].astype(np.float32)
    sdf_vals = sdf_grid[
        pore_idx[:, 0], pore_idx[:, 1], pore_idx[:, 2]
    ].astype(np.float32)
    inlet_mask  = (pore_idx[:, axis] == 0).astype(bool)
    outlet_mask = (pore_idx[:, axis] == mask.shape[axis] - 1).astype(bool)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        pore_idx    = pore_idx,
        teacher     = teacher_vals,
        sdf         = sdf_vals,
        descriptor  = descriptor,
        inlet_mask  = inlet_mask,
        outlet_mask = outlet_mask,
        shape       = np.array(mask.shape, dtype=np.int32),
        epsilon     = np.float32(epsilon),
        tau_cg      = np.float32(tau_cg),
        D_rel       = np.float32(D_rel),
    )

    elapsed = time.perf_counter() - t0
    print(f"  Saved:  {out_path.name}  N_pore={N:,}  ({elapsed:.1f}s total)")

    return {
        "name":    name,
        "npz_path": str(out_path),
        "epsilon": epsilon,
        "tau_cg":  tau_cg,
        "D_rel":   D_rel,
        "n_pore":  N,
        "elapsed": elapsed,
        "skipped": False,
    }


def preprocess_ct_directory(
    tif_dir:    str | Path,
    output_dir: str | Path,
    rev_voxels: int | None = None,
    pore_is_dark: bool = True,
    axis: int = INLET_AXIS,
    k_max: int = K_MAX,
    overwrite: bool = False,
) -> list[dict]:
    """Preprocess all .tif files in a directory."""
    tif_dir = Path(tif_dir)
    tifs    = sorted(tif_dir.glob("*.tif")) + sorted(tif_dir.glob("*.tiff"))
    print(f"Found {len(tifs)} .tif files in {tif_dir}")

    results = []
    for tif in tifs:
        try:
            r = preprocess_ct_volume(
                tif, output_dir,
                rev_voxels=rev_voxels,
                pore_is_dark=pore_is_dark,
                axis=axis,
                k_max=k_max,
                overwrite=overwrite,
            )
            results.append(r)
        except Exception as e:
            print(f"  ERROR {tif.name}: {e}")
            results.append({"name": tif.stem, "error": str(e)})

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess real CT volumes")
    parser.add_argument("--tif-dir",     required=True,      help="Directory of .tif CT volumes")
    parser.add_argument("--output-dir",  required=True,      help="Output directory for .npz files")
    parser.add_argument("--rev-voxels",  type=int, default=None, help="REV cube side length in voxels")
    parser.add_argument("--pore-bright", action="store_true",    help="Pore voxels are bright (inverts Otsu)")
    parser.add_argument("--k-max",       type=int, default=4,    help="FFT descriptor radius (default 4 → 729 coeffs)")
    parser.add_argument("--overwrite",   action="store_true")
    args = parser.parse_args()

    preprocess_ct_directory(
        tif_dir      = args.tif_dir,
        output_dir   = args.output_dir,
        rev_voxels   = args.rev_voxels,
        pore_is_dark = not args.pore_bright,
        k_max        = args.k_max,
        overwrite    = args.overwrite,
    )
