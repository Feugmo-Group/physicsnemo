"""Marching cubes + PyMeshLab cleanup. Week 5."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from skimage.measure import marching_cubes


def mask_to_stl(
    mask: np.ndarray,
    voxel_size_m: float,
    out_path: str | Path,
    target_triangles: int = 500_000,
    smooth_iters: int = 5,
    close_holes: bool = True,
) -> Path:
    """Extract pore/solid surface via marching cubes and return STL path."""

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    binary = (mask > 0).astype(np.uint8)

    pad_width = 1

    binary = np.pad(
        binary,
        pad_width=pad_width,
        mode="constant",
        constant_values=0,
    )

    verts, faces, normals, _ = marching_cubes(
        binary,
        level=0.5,
        spacing=(voxel_size_m, voxel_size_m, voxel_size_m),
    )

    verts -= pad_width * voxel_size_m

    mesh = trimesh.Trimesh(
        vertices=verts,
        faces=faces,
        vertex_normals=normals,
        process=True,
    )

    if smooth_iters > 0:
        trimesh.smoothing.filter_laplacian(
            mesh,
            lamb=0.5,
            iterations=smooth_iters,
        )

    #if len(mesh.faces) > target_triangles:
        #mesh = mesh.simplify_quadric_decimation(target_triangles)

    mesh.export(out_path)

    print(f"Saved STL: {out_path}")
    print(f"Vertices: {len(mesh.vertices)}")
    print(f"Faces: {len(mesh.faces)}")
    print(f"Watertight: {mesh.is_watertight}")

    return out_path


def verify_watertight(stl_path: str | Path) -> bool:
    """Check whether STL mesh is watertight."""

    mesh = trimesh.load_mesh(stl_path)

    print(f"Watertight: {mesh.is_watertight}")
    print(f"Vertices: {len(mesh.vertices)}")
    print(f"Faces: {len(mesh.faces)}")

    return bool(mesh.is_watertight)


#--------------------- test ---------------------
if __name__ == "__main__":

    from ct_to_physicsnemo.io import load_ct, crop_rev

    # ---------------- LOAD CT ----------------

    path = (
        "../diffusion_in_porous_media/"
        "Electrode I/I_1/I_1_bin.tif"
    )

    volume = load_ct(path)

    # ---------------- CROP REV ----------------

    cropped = crop_rev(
        volume,
        voxel_size_nm=50,
        rev_microns=5,
    )

    print("\nCropped shape:")
    print(cropped.shape)

    # ---------------- CREATE STL ----------------

    stl_path = mask_to_stl(
        mask=cropped,
        voxel_size_m=50e-9,  # 50 nm
        out_path="runs/meshes/I_1_crop.stl",
    )

    # ---------------- VERIFY ----------------

    watertight = verify_watertight(stl_path)

    print("\n===== FINAL =====")
    print(f"STL Path: {stl_path}")
    print(f"Watertight: {watertight}")
