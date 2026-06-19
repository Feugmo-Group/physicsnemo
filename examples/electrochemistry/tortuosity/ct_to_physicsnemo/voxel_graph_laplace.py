"""Deterministic voxel-graph Laplace solve for tortuosity.

This treats pore voxels as graph nodes and pore-pore 6-neighbor contacts as
unit conductances. Inlet/outlet faces are fixed Dirichlet boundaries, and
solid/outside neighbors are omitted, which is the discrete no-flux condition.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv
from scipy import sparse
from scipy.sparse import linalg as spla


TOP_BC = -0.5
BOT_BC = 0.5
DELTA_C = abs(BOT_BC - TOP_BC)


def _cg_solve(
    A: sparse.csr_matrix,
    b: np.ndarray,
    x0: np.ndarray,
    rtol: float,
    maxiter: int,
) -> tuple[np.ndarray, int]:
    """Run scipy CG across old/new scipy tolerance signatures."""

    diag = A.diagonal()
    inv_diag = np.zeros_like(diag)
    nonzero = diag != 0
    inv_diag[nonzero] = 1.0 / diag[nonzero]

    def precondition(x: np.ndarray) -> np.ndarray:
        return inv_diag * x

    M = spla.LinearOperator(A.shape, matvec=precondition)

    try:
        return spla.cg(
            A,
            b,
            x0=x0,
            rtol=rtol,
            atol=0.0,
            maxiter=maxiter,
            M=M,
        )
    except TypeError:
        return spla.cg(
            A,
            b,
            x0=x0,
            tol=rtol,
            maxiter=maxiter,
            M=M,
        )


def _build_graph_system(
    mask: np.ndarray,
    axis: int,
    top_bc: float,
    bot_bc: float,
) -> tuple[
    sparse.csr_matrix,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Build unknown-node graph Laplacian system A x = b."""

    pore = mask > 0
    shape = pore.shape
    pore_idx = np.argwhere(pore)

    if len(pore_idx) == 0:
        raise ValueError("No pore voxels found.")

    id_grid = np.full(shape, -1, dtype=np.int64)
    id_grid[pore] = np.arange(len(pore_idx), dtype=np.int64)

    is_inlet = pore_idx[:, axis] == 0
    is_outlet = pore_idx[:, axis] == shape[axis] - 1
    is_dirichlet = is_inlet | is_outlet
    is_unknown = ~is_dirichlet

    if not np.any(is_unknown):
        raise ValueError("No interior pore voxels available for graph solve.")

    fixed_values = np.zeros(len(pore_idx), dtype=np.float64)
    fixed_values[is_inlet] = top_bc
    fixed_values[is_outlet] = bot_bc

    unknown_pore_ids = np.flatnonzero(is_unknown)
    pore_to_unknown = np.full(len(pore_idx), -1, dtype=np.int64)
    pore_to_unknown[unknown_pore_ids] = np.arange(len(unknown_pore_ids), dtype=np.int64)

    unknown_idx = pore_idx[unknown_pore_ids]
    n_unknown = len(unknown_idx)

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    data: list[np.ndarray] = []
    rhs = np.zeros(n_unknown, dtype=np.float64)
    diagonal = np.zeros(n_unknown, dtype=np.float64)

    offsets = np.array(
        [
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 1],
            [0, 0, -1],
        ],
        dtype=np.int64,
    )

    row_ids = np.arange(n_unknown, dtype=np.int64)

    for offset in offsets:
        neighbor = unknown_idx + offset
        valid = (
            (neighbor[:, 0] >= 0)
            & (neighbor[:, 0] < shape[0])
            & (neighbor[:, 1] >= 0)
            & (neighbor[:, 1] < shape[1])
            & (neighbor[:, 2] >= 0)
            & (neighbor[:, 2] < shape[2])
        )

        neighbor_pore_ids = np.full(n_unknown, -1, dtype=np.int64)
        valid_neighbor = neighbor[valid]
        neighbor_pore_ids[valid] = id_grid[
            valid_neighbor[:, 0],
            valid_neighbor[:, 1],
            valid_neighbor[:, 2],
        ]

        has_pore_neighbor = neighbor_pore_ids >= 0
        if not np.any(has_pore_neighbor):
            continue

        diagonal[has_pore_neighbor] += 1.0

        neighbor_unknown_ids = pore_to_unknown[
            neighbor_pore_ids[has_pore_neighbor]
        ]

        unknown_neighbor = neighbor_unknown_ids >= 0
        if np.any(unknown_neighbor):
            coupled_rows = row_ids[has_pore_neighbor][unknown_neighbor]
            coupled_cols = neighbor_unknown_ids[unknown_neighbor]
            rows.append(coupled_rows)
            cols.append(coupled_cols)
            data.append(np.full(len(coupled_rows), -1.0, dtype=np.float64))

        fixed_neighbor = ~unknown_neighbor
        if np.any(fixed_neighbor):
            fixed_pore_ids = neighbor_pore_ids[has_pore_neighbor][fixed_neighbor]
            fixed_rows = row_ids[has_pore_neighbor][fixed_neighbor]
            rhs[fixed_rows] += fixed_values[fixed_pore_ids]

    rows.append(row_ids)
    cols.append(row_ids)
    data.append(diagonal)

    A = sparse.coo_matrix(
        (
            np.concatenate(data),
            (np.concatenate(rows), np.concatenate(cols)),
        ),
        shape=(n_unknown, n_unknown),
    ).tocsr()

    return A, rhs, pore_idx, unknown_pore_ids, fixed_values, is_dirichlet


def _slice_flux(
    concentration: np.ndarray,
    mask: np.ndarray,
    axis: int,
) -> tuple[float, float, np.ndarray]:
    """Compute TauFactor-style slice flux from a voxel concentration field."""

    pore = mask > 0
    shape = mask.shape
    n_axis = shape[axis]
    other_axes = [a for a in range(3) if a != axis]
    full_cross_section_area = shape[other_axes[0]] * shape[other_axes[1]]

    flux_per_slice = []

    for s in range(n_axis - 1):
        left_slice = [slice(None)] * 3
        right_slice = [slice(None)] * 3
        left_slice[axis] = s
        right_slice[axis] = s + 1

        conducting_face = pore[tuple(left_slice)] & pore[tuple(right_slice)]

        if not np.any(conducting_face):
            flux_per_slice.append(0.0)
            continue

        c_left = concentration[tuple(left_slice)][conducting_face]
        c_right = concentration[tuple(right_slice)][conducting_face]
        total_flux = float(np.sum(c_right - c_left))
        flux_per_slice.append(abs(total_flux) / full_cross_section_area)

    flux_per_slice_np = np.asarray(flux_per_slice, dtype=np.float64)
    mean_flux = float(np.mean(flux_per_slice_np))
    # Dirichlet values are imposed on inlet/outlet voxel centers, so the
    # center-to-center transport length has n_axis - 1 intervals.
    transport_length = max(n_axis - 1, 1)
    D_rel = mean_flux * transport_length / DELTA_C

    return mean_flux, D_rel, flux_per_slice_np


def solve_voxel_graph_helmholtz_for_tau(
    mask: np.ndarray,
    epsilon: float,
    reaction_rate: float,
    axis: int = 0,
    top_bc: float = TOP_BC,
    bot_bc: float = BOT_BC,
    rtol: float = 1e-8,
    maxiter: int = 20_000,
    output_vtk: str | Path | None = None,
    return_field: bool = False,
) -> dict:
    """Solve reaction-diffusion equation ∇²c = k·c on the pore graph.

    This is the EIS teacher solver. It extends the pure Laplace solve by adding
    a diagonal reaction term to every pore voxel node:

        sum_neighbours(c_j - c_i) - k * c_i = 0   for interior pore voxels

    The reaction rate k represents lithium intercalation at the solid-electrolyte
    interface (ElectrodeSolver convention). The solution has a hyperbolic cosine
    profile along the transport axis rather than the linear Laplace profile, and
    its frequency-dependent form (k → jω/D) gives the EIS impedance.

    For a given angular frequency ω and diffusivity D, set:
        reaction_rate = omega / D   (real part; imaginary part handled separately)

    Args:
        mask: Binary pore mask (1 = pore, 0 = solid).
        epsilon: Porosity (used for tau calculation).
        reaction_rate: Reaction coefficient k ≥ 0. Set to 0 to recover pure Laplace.
        axis: Transport axis (0 = x, 1 = y, 2 = z).
        top_bc: Concentration at the inlet face.
        bot_bc: Concentration at the outlet face.
        rtol: CG solver relative tolerance.
        maxiter: Maximum CG iterations.
        output_vtk: Optional path to write a VTK point-cloud of the solution.
        return_field: If True, include concentration array in the returned dict.

    Returns:
        Dict with tau, D_rel, flux diagnostics, and optionally the concentration field.
    """
    binary = (mask > 0).astype(np.uint8)
    shape = binary.shape

    A, rhs, pore_idx, unknown_pore_ids, fixed_values, is_dirichlet = (
        _build_graph_system(
            binary,
            axis=axis,
            top_bc=top_bc,
            bot_bc=bot_bc,
        )
    )

    if reaction_rate > 0.0:
        # Add k to every diagonal entry of the unknown-node system.
        # A was built as: D_ii = number of pore neighbours; off-diag = -1.
        # Adding k gives: D_ii = n_pore_neighbours + k, which makes the
        # system positive definite for any k > 0.
        A = A + sparse.eye(A.shape[0], format="csr", dtype=np.float64) * reaction_rate

        # The rhs must also absorb the fixed-node contribution from the Dirichlet BCs.
        # That was already done in _build_graph_system (fixed neighbours subtracted
        # from rhs). No further change to rhs is needed for the reaction term because
        # it only couples each unknown to itself.

    x0_all = top_bc + (bot_bc - top_bc) * (
        (pore_idx[:, axis].astype(np.float64) + 0.5) / shape[axis]
    )
    x0 = x0_all[unknown_pore_ids]

    solution, cg_info = _cg_solve(
        A,
        rhs,
        x0=x0,
        rtol=rtol,
        maxiter=maxiter,
    )

    pore_values = fixed_values.copy()
    pore_values[unknown_pore_ids] = solution

    concentration = np.zeros(shape, dtype=np.float32)
    concentration[
        pore_idx[:, 0],
        pore_idx[:, 1],
        pore_idx[:, 2],
    ] = pore_values.astype(np.float32)

    mean_flux, D_rel, flux_per_slice = _slice_flux(
        concentration,
        binary,
        axis=axis,
    )

    tau = float("inf") if D_rel <= 0 else float(epsilon / D_rel)

    if output_vtk is not None:
        xyz = pore_idx.astype(np.float32) + 0.5
        cloud = pv.PolyData(xyz)
        cloud["C_helmholtz"] = pore_values.astype(np.float32)
        out_path = Path(output_vtk)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cloud.save(out_path)

    result = {
        "tau": tau,
        "D_rel": D_rel,
        "reaction_rate": reaction_rate,
        "mean_flux": mean_flux,
        "flux_min": float(np.min(flux_per_slice)),
        "flux_max": float(np.max(flux_per_slice)),
        "flux_relative_error": (
            float((np.max(flux_per_slice) - np.min(flux_per_slice)) / np.max(flux_per_slice))
            if np.max(flux_per_slice) > 0
            else float("inf")
        ),
        "cg_info": int(cg_info),
        "n_pore": int(len(pore_idx)),
        "n_unknown": int(len(unknown_pore_ids)),
        "n_dirichlet": int(np.sum(is_dirichlet)),
    }

    if return_field:
        result["concentration"] = concentration
        result["pore_idx"] = pore_idx
        result["pore_values"] = pore_values

    return result


def solve_voxel_graph_laplace_for_tau(
    mask: np.ndarray,
    epsilon: float,
    axis: int = 0,
    top_bc: float = TOP_BC,
    bot_bc: float = BOT_BC,
    rtol: float = 1e-8,
    maxiter: int = 10_000,
    output_vtk: str | Path | None = None,
    return_field: bool = False,
) -> dict:
    """Solve graph Laplace equation and return tau/D_rel diagnostics."""

    binary = (mask > 0).astype(np.uint8)
    shape = binary.shape

    A, rhs, pore_idx, unknown_pore_ids, fixed_values, is_dirichlet = (
        _build_graph_system(
            binary,
            axis=axis,
            top_bc=top_bc,
            bot_bc=bot_bc,
        )
    )

    x0_all = top_bc + (bot_bc - top_bc) * (
        (pore_idx[:, axis].astype(np.float64) + 0.5) / shape[axis]
    )
    x0 = x0_all[unknown_pore_ids]

    solution, cg_info = _cg_solve(
        A,
        rhs,
        x0=x0,
        rtol=rtol,
        maxiter=maxiter,
    )

    pore_values = fixed_values.copy()
    pore_values[unknown_pore_ids] = solution

    concentration = np.zeros(shape, dtype=np.float32)
    concentration[
        pore_idx[:, 0],
        pore_idx[:, 1],
        pore_idx[:, 2],
    ] = pore_values.astype(np.float32)

    mean_flux, D_rel, flux_per_slice = _slice_flux(
        concentration,
        binary,
        axis=axis,
    )

    tau = float("inf") if D_rel <= 0 else float(epsilon / D_rel)

    if output_vtk is not None:
        xyz = pore_idx.astype(np.float32) + 0.5
        cloud = pv.PolyData(xyz)
        cloud["C_graph"] = pore_values.astype(np.float32)
        out_path = Path(output_vtk)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cloud.save(out_path)

    result = {
        "tau": tau,
        "D_rel": D_rel,
        "mean_flux": mean_flux,
        "flux_min": float(np.min(flux_per_slice)),
        "flux_max": float(np.max(flux_per_slice)),
        "flux_relative_error": (
            float((np.max(flux_per_slice) - np.min(flux_per_slice)) / np.max(flux_per_slice))
            if np.max(flux_per_slice) > 0
            else float("inf")
        ),
        "cg_info": int(cg_info),
        "n_pore": int(len(pore_idx)),
        "n_unknown": int(len(unknown_pore_ids)),
        "n_dirichlet": int(np.sum(is_dirichlet)),
    }

    if return_field:
        result["concentration"] = concentration
        result["pore_idx"] = pore_idx
        result["pore_values"] = pore_values

    return result
