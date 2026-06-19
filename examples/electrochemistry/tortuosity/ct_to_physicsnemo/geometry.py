"""Geometry utilities for CT-to-PhysicsNeMo porous-media pipeline.

Conventions:
1 = pore / empty space / conducting phase
0 = solid / non-conducting phase
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyvista as pv
from scipy import ndimage
from scipy.ndimage import distance_transform_edt
from physicsnemo.sym.geometry.tessellation import Tessellation


@dataclass
class SampledPoints:
    interior: np.ndarray
    interior_sdf: np.ndarray

    boundary: np.ndarray
    boundary_normals: np.ndarray
    boundary_area: np.ndarray
    boundary_sdf: np.ndarray

    inlet: np.ndarray
    inlet_sdf: np.ndarray

    outlet: np.ndarray
    outlet_sdf: np.ndarray


def pore_only_filter(
    mask: np.ndarray,
    pore_value: int = 1,
) -> np.ndarray:
    """Keep only pore / empty-space voxels.

    Parameters
    ----------
    mask:
        Input binary or labeled volume.

    pore_value:
        The value in the input mask that represents pore space.

    Returns
    -------
    np.ndarray
        Binary mask with:
        1 = pore
        0 = solid
    """

    mask = np.asarray(mask)

    return (mask == pore_value).astype(np.uint8)


def keep_percolating_pores(
    mask: np.ndarray,
    axis: int = 0,
) -> np.ndarray:
    """Keep only pore clusters connected from inlet to outlet.

    Parameters
    ----------
    mask:
        Binary pore mask.

        1 = pore
        0 = solid

    axis:
        Transport direction.

        TauFactor convention:
        axis = 0 is the default transport direction.

    Returns
    -------
    np.ndarray
        Filtered binary pore mask.
    """

    pore = mask > 0

    labels, n_labels = ndimage.label(pore)

    if n_labels == 0:
        raise ValueError("No pore clusters found.")

    inlet_slice = [slice(None)] * 3
    outlet_slice = [slice(None)] * 3

    inlet_slice[axis] = 0
    outlet_slice[axis] = mask.shape[axis] - 1

    inlet_labels = np.unique(labels[tuple(inlet_slice)])
    outlet_labels = np.unique(labels[tuple(outlet_slice)])

    inlet_labels = set(inlet_labels[inlet_labels != 0])
    outlet_labels = set(outlet_labels[outlet_labels != 0])

    connected_labels = inlet_labels.intersection(outlet_labels)

    if not connected_labels:
        raise ValueError("No inlet-outlet connected pore network found.")

    filtered = np.isin(
        labels,
        list(connected_labels),
    )

    return filtered.astype(np.uint8)


def signed_distance_from_mask(
    mask: np.ndarray,
    voxel_size_m: float,
) -> np.ndarray:
    """Approximate signed distance field from binary pore mask.

    Positive = pore
    Negative = solid
    """

    pore = mask > 0

    dist_pore = distance_transform_edt(pore) * voxel_size_m
    dist_solid = distance_transform_edt(~pore) * voxel_size_m

    sdf = np.where(
        pore,
        dist_pore,
        -dist_solid,
    )

    return sdf.astype(np.float32)


def sample_indices_from_mask(
    mask: np.ndarray,
    n_points: int,
) -> np.ndarray:
    """Randomly sample pore voxel indices."""

    pore_indices = np.argwhere(mask > 0)

    if len(pore_indices) == 0:
        raise ValueError("No pore voxels found.")

    choice = np.random.choice(
        len(pore_indices),
        size=n_points,
        replace=len(pore_indices) < n_points,
    )

    return pore_indices[choice]


def indices_to_points(
    indices: np.ndarray,
    voxel_size_m: float,
) -> np.ndarray:
    """Convert voxel indices to physical voxel-center coordinates."""

    return (indices.astype(np.float64) + 0.5) * voxel_size_m


def sample_face_indices(
    mask: np.ndarray,
    n_points: int,
    axis: int,
    side: str,
) -> np.ndarray:
    """Sample pore voxel indices on inlet/outlet face."""

    if side not in {"min", "max"}:
        raise ValueError("side must be 'min' or 'max'.")

    face_index = 0 if side == "min" else mask.shape[axis] - 1

    slicer = [slice(None)] * 3
    slicer[axis] = face_index

    face = mask[tuple(slicer)]

    pore_2d = np.argwhere(face > 0)

    if len(pore_2d) == 0:
        raise ValueError(f"No pore voxels on {side} face along axis {axis}.")

    choice = np.random.choice(
        len(pore_2d),
        size=n_points,
        replace=len(pore_2d) < n_points,
    )

    selected_2d = pore_2d[choice]

    indices = np.zeros((n_points, 3), dtype=int)
    other_axes = [a for a in range(3) if a != axis]

    indices[:, axis] = face_index
    indices[:, other_axes[0]] = selected_2d[:, 0]
    indices[:, other_axes[1]] = selected_2d[:, 1]

    return indices


def sample_wall_faces_from_mask(
    mask: np.ndarray,
    n_points: int,
    voxel_size_m: float,
    inlet_axis: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample pore-solid wall face centers and outward normals from a voxel mask.

    Inlet/outlet exterior caps along ``inlet_axis`` are excluded because those
    faces carry Dirichlet boundary conditions, not no-flux wall conditions.
    """

    pore_indices = np.argwhere(mask > 0)

    if len(pore_indices) == 0:
        raise ValueError("No pore voxels found.")

    shape = np.asarray(mask.shape)
    face_point_groups = []
    face_normal_groups = []

    for axis in range(3):
        for direction in (-1, 1):
            neighbor = pore_indices.copy()
            neighbor[:, axis] += direction

            inside = (
                (neighbor[:, 0] >= 0)
                & (neighbor[:, 0] < shape[0])
                & (neighbor[:, 1] >= 0)
                & (neighbor[:, 1] < shape[1])
                & (neighbor[:, 2] >= 0)
                & (neighbor[:, 2] < shape[2])
            )

            is_wall = np.zeros(len(pore_indices), dtype=bool)

            if np.any(inside):
                neighbor_inside = neighbor[inside]
                neighbor_is_solid = (
                    mask[
                        neighbor_inside[:, 0],
                        neighbor_inside[:, 1],
                        neighbor_inside[:, 2],
                    ]
                    == 0
                )
                is_wall[inside] = neighbor_is_solid

            if axis != inlet_axis:
                is_wall[~inside] = True

            if not np.any(is_wall):
                continue

            centers = indices_to_points(
                pore_indices[is_wall],
                voxel_size_m,
            )
            centers[:, axis] += 0.5 * direction * voxel_size_m

            normals = np.zeros_like(centers)
            normals[:, axis] = direction

            face_point_groups.append(centers)
            face_normal_groups.append(normals)

    if not face_point_groups:
        raise ValueError("No no-flux wall faces found.")

    sampled_point_groups = []
    sampled_normal_groups = []

    base_per_group = max(n_points // len(face_point_groups), 1)

    for points, normals in zip(face_point_groups, face_normal_groups):
        choice = np.random.choice(
            len(points),
            size=base_per_group,
            replace=len(points) < base_per_group,
        )
        sampled_point_groups.append(points[choice])
        sampled_normal_groups.append(normals[choice])

    sampled_points = np.vstack(sampled_point_groups)
    sampled_normals = np.vstack(sampled_normal_groups)

    if len(sampled_points) < n_points:
        all_points = np.vstack(face_point_groups)
        all_normals = np.vstack(face_normal_groups)
        remaining = n_points - len(sampled_points)
        choice = np.random.choice(
            len(all_points),
            size=remaining,
            replace=len(all_points) < remaining,
        )
        sampled_points = np.vstack([sampled_points, all_points[choice]])
        sampled_normals = np.vstack([sampled_normals, all_normals[choice]])

    if len(sampled_points) > n_points:
        choice = np.random.choice(
            len(sampled_points),
            size=n_points,
            replace=False,
        )
        sampled_points = sampled_points[choice]
        sampled_normals = sampled_normals[choice]

    sampled_area = np.full(
        n_points,
        voxel_size_m**2,
        dtype=np.float64,
    )

    return sampled_points, sampled_normals, sampled_area


class PoreGeometry:
    """PhysicsNeMo Tessellation + voxel-mask sampling wrapper."""

    def __init__(
        self,
        stl_path: str | Path,
        mask: np.ndarray | None = None,
        voxel_size_m: float = 1.0,
        filter_connected: bool = False,
        inlet_axis: int = 0,
    ) -> None:
        self.stl_path = Path(stl_path)
        self.voxel_size_m = voxel_size_m
        self.inlet_axis = inlet_axis

        if not self.stl_path.exists():
            raise FileNotFoundError(f"STL file not found: {self.stl_path}")

        if mask is not None:
            mask = (mask > 0).astype(np.uint8)

            if filter_connected:
                mask = keep_percolating_pores(
                    mask,
                    axis=inlet_axis,
                )

        self.mask = mask

        self.geo = Tessellation.from_stl(
            str(self.stl_path),
            airtight=False,
        )

        self.sdf_grid = None

        if self.mask is not None:
            self.sdf_grid = signed_distance_from_mask(
                self.mask,
                voxel_size_m=self.voxel_size_m,
            )

    def sample(
        self,
        n_interior: int,
        n_boundary: int,
        n_inlet: int,
        n_outlet: int,
        inlet_axis: int | None = None,
    ) -> SampledPoints:
        """Sample pore interior, boundary, inlet, and outlet points."""

        if inlet_axis is None:
            inlet_axis = self.inlet_axis

        if self.mask is None or self.sdf_grid is None:
            raise ValueError("Voxel mask is required for SDF-aware sampling.")

        # Interior pore voxels
        interior_idx = sample_indices_from_mask(
            self.mask,
            n_points=n_interior,
        )

        interior = indices_to_points(
            interior_idx,
            self.voxel_size_m,
        )

        interior_sdf = self.sdf_grid[
            interior_idx[:, 0],
            interior_idx[:, 1],
            interior_idx[:, 2],
        ].reshape(-1, 1)

        # No-flux walls from exact pore-solid voxel faces. This keeps wall
        # samples in the same coordinate frame as voxel-center training points.
        boundary_xyz, boundary_normals, boundary_area = sample_wall_faces_from_mask(
            self.mask,
            n_points=n_boundary,
            voxel_size_m=self.voxel_size_m,
            inlet_axis=inlet_axis,
        )

        # Boundary has approximately SDF = 0
        boundary_sdf = np.zeros(
            (len(boundary_xyz), 1),
            dtype=np.float32,
        )

        # Inlet and outlet pore voxels
        inlet_idx = sample_face_indices(
            self.mask,
            n_points=n_inlet,
            axis=inlet_axis,
            side="min",
        )

        outlet_idx = sample_face_indices(
            self.mask,
            n_points=n_outlet,
            axis=inlet_axis,
            side="max",
        )

        inlet = indices_to_points(
            inlet_idx,
            self.voxel_size_m,
        )

        outlet = indices_to_points(
            outlet_idx,
            self.voxel_size_m,
        )

        inlet_sdf = self.sdf_grid[
            inlet_idx[:, 0],
            inlet_idx[:, 1],
            inlet_idx[:, 2],
        ].reshape(-1, 1)

        outlet_sdf = self.sdf_grid[
            outlet_idx[:, 0],
            outlet_idx[:, 1],
            outlet_idx[:, 2],
        ].reshape(-1, 1)

        return SampledPoints(
            interior=interior,
            interior_sdf=interior_sdf,
            boundary=boundary_xyz,
            boundary_normals=boundary_normals,
            boundary_area=boundary_area,
            boundary_sdf=boundary_sdf,
            inlet=inlet,
            inlet_sdf=inlet_sdf,
            outlet=outlet,
            outlet_sdf=outlet_sdf,
        )

    def all_face_points(
        self,
        axis: int,
        side: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return all pore voxel-center points and SDF values on one face."""

        if self.mask is None or self.sdf_grid is None:
            raise ValueError("Voxel mask is required.")

        if side not in {"min", "max"}:
            raise ValueError("side must be 'min' or 'max'.")

        face_index = 0 if side == "min" else self.mask.shape[axis] - 1

        slicer = [slice(None)] * 3
        slicer[axis] = face_index

        face = self.mask[tuple(slicer)]

        pore_2d = np.argwhere(face > 0)

        if len(pore_2d) == 0:
            raise ValueError(f"No pore voxels on {side} face along axis {axis}.")

        indices = np.zeros((len(pore_2d), 3), dtype=int)
        other_axes = [a for a in range(3) if a != axis]

        indices[:, axis] = face_index
        indices[:, other_axes[0]] = pore_2d[:, 0]
        indices[:, other_axes[1]] = pore_2d[:, 1]

        points = indices_to_points(
            indices,
            self.voxel_size_m,
        )

        sdf = self.sdf_grid[
            indices[:, 0],
            indices[:, 1],
            indices[:, 2],
        ].reshape(-1, 1)

        return points, sdf


def save_samples_vtk(
    samples: SampledPoints,
    out_path: str | Path,
) -> Path:
    """Save sampled point sets to a VTK file."""

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    clouds = []

    interior_cloud = pv.PolyData(samples.interior)
    interior_cloud["region"] = np.zeros(len(samples.interior), dtype=np.int32)
    interior_cloud["sdf"] = samples.interior_sdf.ravel()
    clouds.append(interior_cloud)

    boundary_cloud = pv.PolyData(samples.boundary)
    boundary_cloud["region"] = np.ones(len(samples.boundary), dtype=np.int32)
    boundary_cloud["sdf"] = samples.boundary_sdf.ravel()
    clouds.append(boundary_cloud)

    inlet_cloud = pv.PolyData(samples.inlet)
    inlet_cloud["region"] = np.full(len(samples.inlet), 2, dtype=np.int32)
    inlet_cloud["sdf"] = samples.inlet_sdf.ravel()
    clouds.append(inlet_cloud)

    outlet_cloud = pv.PolyData(samples.outlet)
    outlet_cloud["region"] = np.full(len(samples.outlet), 3, dtype=np.int32)
    outlet_cloud["sdf"] = samples.outlet_sdf.ravel()
    clouds.append(outlet_cloud)

    combined = clouds[0]

    for cloud in clouds[1:]:
        combined = combined.merge(cloud)

    combined.save(out_path)

    print(f"Saved geometry samples to: {out_path}")

    return out_path


if __name__ == "__main__":

    from ct_to_physicsnemo.io import load_ct, crop_rev

    ct_path = (
        "../diffusion_in_porous_media/"
        "Electrode I/I_1/I_1_bin.tif"
    )

    volume = load_ct(ct_path)

    cropped = crop_rev(
        volume,
        voxel_size_nm=50,
        rev_microns=5,
    )

    # If pore value is 1:
    cropped = pore_only_filter(
        cropped,
        pore_value=1,
    )

    geom = PoreGeometry(
        stl_path="runs/meshes/I_1_crop.stl",
        mask=cropped,
        voxel_size_m=50e-9,
        filter_connected=True,
        inlet_axis=0,
    )

    samples = geom.sample(
        n_interior=10_000,
        n_boundary=10_000,
        n_inlet=1_000,
        n_outlet=1_000,
        inlet_axis=0,
    )

    print("\n===== GEOMETRY SAMPLE TEST =====")
    print("Interior:", samples.interior.shape)
    print("Interior SDF:", samples.interior_sdf.shape)
    print("Boundary:", samples.boundary.shape)
    print("Boundary normals:", samples.boundary_normals.shape)
    print("Boundary area:", samples.boundary_area.shape)
    print("Inlet:", samples.inlet.shape)
    print("Outlet:", samples.outlet.shape)

    save_samples_vtk(
        samples,
        out_path="runs/geometry_samples/I_1_samples.vtk",
    )
