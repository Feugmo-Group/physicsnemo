"""Mesh-based signed distance field for pore geometries.

Replaces scipy.ndimage.distance_transform_edt with physicsnemo's
GPU-accelerated Warp BVH SDF.  Works identically for:
  - Synthetic volumes (marching cubes on binary voxel mask)
  - Real CT volumes (marching cubes on Otsu-segmented binary mask)

Pipeline:
    binary mask  →  skimage.marching_cubes  →  triangular mesh
                 →  physicsnemo.nn.functional.signed_distance_field (Warp BVH)
                 →  SDF values at pore voxel centres

Advantages over EDT:
  - Sub-voxel accurate (geometry interpolated at real boundaries)
  - GPU-accelerated (BVH tree on CUDA, ~10-100× faster for large volumes)
  - Produces true signed distances (negative inside solid, positive inside pore)
  - Consistent pipeline for synthetic and real CT data
"""

from __future__ import annotations

import numpy as np
import torch
from skimage.measure import marching_cubes

from physicsnemo.nn.functional import signed_distance_field


def mask_to_mesh(
    mask: np.ndarray,
    level: float = 0.5,
    allow_degenerate: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Binary voxel mask → triangular surface mesh via marching cubes.

    Parameters
    ----------
    mask : np.ndarray
        Binary array where 1 = pore, 0 = solid.
    level : float
        Iso-surface level (default 0.5 = boundary between pore and solid).
    allow_degenerate : bool
        Passed to skimage.marching_cubes. False produces watertight meshes.

    Returns
    -------
    vertices : np.ndarray  (n_verts, 3)  float32
    faces    : np.ndarray  (n_faces, 3)  int32
    """
    verts, faces, _, _ = marching_cubes(
        mask.astype(np.float32),
        level=level,
        allow_degenerate=allow_degenerate,
    )
    return verts.astype(np.float32), faces.astype(np.int32)


def compute_mesh_sdf(
    mask: np.ndarray,
    query_points: np.ndarray | None = None,
    device: str = "cuda",
    use_sign_winding_number: bool = True,
    normalise: bool = True,
) -> np.ndarray:
    """Compute SDF at pore voxel centres using physicsnemo Warp BVH.

    Parameters
    ----------
    mask : np.ndarray
        Binary voxel mask (1=pore, 0=solid), shape (X, Y, Z).
    query_points : np.ndarray, optional
        Points at which to evaluate SDF, shape (N, 3).
        If None, evaluates at the centre of every pore voxel.
    device : str
        'cuda' or 'cpu'.
    use_sign_winding_number : bool
        Use winding-number sign determination — robust for non-watertight
        meshes that may arise from marching cubes on noisy CT data.
    normalise : bool
        If True, divide by the max SDF value so output is in [0, 1].
        Keeps the same scale convention as the old EDT pipeline.

    Returns
    -------
    sdf_vals : np.ndarray  float32, shape (N,)
        SDF values at query_points (positive inside pore, near-zero at wall).
    """
    # build mesh from mask
    verts, faces = mask_to_mesh(mask)

    if query_points is None:
        pore_idx     = np.argwhere(mask > 0).astype(np.float32)  # (N, 3)
        query_points = pore_idx + 0.5   # voxel centres (sub-voxel accurate)

    verts_t  = torch.tensor(verts,        dtype=torch.float32, device=device).contiguous()
    faces_t  = torch.tensor(faces,        dtype=torch.int32,   device=device).contiguous()
    query_t  = torch.tensor(query_points, dtype=torch.float32, device=device).contiguous()

    sdf_t, _ = signed_distance_field(
        verts_t, faces_t, query_t,
        use_sign_winding_number=use_sign_winding_number,
    )

    # SDF from physicsnemo: negative inside solid, positive inside pore.
    # We want positive = inside pore (distance to wall), same as EDT.
    sdf_vals = sdf_t.cpu().numpy().astype(np.float32)

    if normalise:
        max_val = float(np.abs(sdf_vals).max())
        if max_val > 0:
            sdf_vals = sdf_vals / max_val

    return sdf_vals


def compute_mesh_sdf_grid(
    mask: np.ndarray,
    device: str = "cuda",
    use_sign_winding_number: bool = True,
    normalise: bool = True,
) -> np.ndarray:
    """Compute SDF on the full voxel grid (same shape as mask).

    Returns a float32 array of shape mask.shape where each pore voxel
    contains the SDF value and solid voxels contain 0.
    Used to build the sdf_t lookup tensor in GeometryData.
    """
    pore_idx     = np.argwhere(mask > 0).astype(np.float32)
    query_points = pore_idx + 0.5

    sdf_vals = compute_mesh_sdf(
        mask, query_points=query_points,
        device=device,
        use_sign_winding_number=use_sign_winding_number,
        normalise=normalise,
    )

    sdf_grid = np.zeros(mask.shape, dtype=np.float32)
    idx      = np.argwhere(mask > 0)
    sdf_grid[idx[:, 0], idx[:, 1], idx[:, 2]] = sdf_vals
    return sdf_grid
