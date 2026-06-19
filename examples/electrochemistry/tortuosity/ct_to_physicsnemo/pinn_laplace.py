"""Voxel-physics-aware Laplace PINN for tortuosity.

This version is closer to TauFactor because:
- trains on pore voxel centers
- uses discrete 6-neighbor Laplace residual
- treats solid neighbors as no-flux walls
- uses inlet/outlet voxel faces
- computes tortuosity using TauFactor-style slice-wise flux

Convention:
1 = pore / conducting phase
0 = solid / non-conducting phase
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv
import torch
from torch import nn

from .fourier_encoding import RFFEncoder
from .geometry import PoreGeometry


class MLP(nn.Module):
    """Simple fully-connected MLP used as the PINN function approximator.

    Output is a single scalar concentration value.
    Activation: tanh between hidden layers (smooth for PDE learning).
    """

    def __init__(self, in_dim: int, hidden: int = 128, layers: int = 6):
        super().__init__()

        net = [nn.Linear(in_dim, hidden), nn.Tanh()]

        for _ in range(layers - 1):
            net += [nn.Linear(hidden, hidden), nn.Tanh()]

        net.append(nn.Linear(hidden, 1))

        self.net = nn.Sequential(*net)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_dim) -> returns (..., 1) concentration
        return self.net(x)


def normalize_idx_points(
    idx: torch.Tensor,
    shape: tuple[int, int, int],
) -> torch.Tensor:
    """Voxel indices -> normalized voxel-center coordinates in [0, 1].

    Adds 0.5 to convert from integer corner-based index to voxel center.
    """
    shape_t = torch.tensor(
        shape,
        dtype=torch.float32,
        device=idx.device,
    )

    return (idx.float() + 0.5) / shape_t


def get_sdf_values(
    sdf_grid: torch.Tensor,
    idx: torch.Tensor,
    sdf_scale: torch.Tensor,
) -> torch.Tensor:
    """Sample SDF values at given voxel indices and scale them.

    Returns shape (N, 1).
    """
    sdf = sdf_grid[
        idx[:, 0],
        idx[:, 1],
        idx[:, 2],
    ].reshape(-1, 1)

    return sdf / sdf_scale


def get_face_indices(
    mask_t: torch.Tensor,
    axis: int,
    side: str,
) -> torch.Tensor:
    """Return pore voxel indices on inlet/outlet face.

    mask_t: boolean mask tensor (z,y,x) indicating pore voxels.
    axis: axis index (0,1,2) for face selection.
    side: 'min' selects index 0, 'max' selects last index along axis.
    """
    if side not in {"min", "max"}:
        raise ValueError("side must be 'min' or 'max'.")

    face_index = 0 if side == "min" else mask_t.shape[axis] - 1

    slicer = [slice(None)] * 3
    slicer[axis] = face_index

    face = mask_t[tuple(slicer)]
    pore_2d = torch.nonzero(face, as_tuple=False)

    if pore_2d.shape[0] == 0:
        raise ValueError(f"No pore voxels found on {side} face along axis {axis}.")

    idx = torch.zeros(
        (pore_2d.shape[0], 3),
        dtype=torch.long,
        device=mask_t.device,
    )

    other_axes = [a for a in range(3) if a != axis]

    # fill axis coordinate and the two other coordinates from 2D positions
    idx[:, axis] = face_index
    idx[:, other_axes[0]] = pore_2d[:, 0]
    idx[:, other_axes[1]] = pore_2d[:, 1]

    return idx


def tau_from_taufactor_style_flux(
    model_forward,
    mask_t: torch.Tensor,
    epsilon: float,
    shape: tuple[int, int, int],
    axis: int = 0,
) -> tuple[float, float, np.ndarray]:
    """Fast TauFactor-style slice-wise flux calculation.

    Computes absolute differences across adjacent slices restricted to
    voxels that are conducting on both sides. Returns (tau, D_rel, flux_per_slice).
    """
    #note to self never do triple nested loops in python again

    device = mask_t.device
    n_axis = shape[axis]

    flux_per_slice = []

    for s in range(n_axis - 1):
        # build boolean masks for left/right slice at positions s and s+1
        slicer_left = [slice(None)] * 3
        slicer_right = [slice(None)] * 3

        slicer_left[axis] = s
        slicer_right[axis] = s + 1

        left_face = mask_t[tuple(slicer_left)]
        right_face = mask_t[tuple(slicer_right)]

        # require pore in both adjacent voxels to consider flux there
        conducting_face = left_face & right_face

        coords_2d = torch.nonzero(conducting_face, as_tuple=False)

        if coords_2d.shape[0] == 0:
            flux_per_slice.append(0.0)
            continue

        # reconstruct full 3D indices for left and right voxels
        idx_left = torch.zeros(
            (coords_2d.shape[0], 3),
            dtype=torch.long,
            device=device,
        )

        idx_right = torch.zeros_like(idx_left)

        other_axes = [a for a in range(3) if a != axis]

        idx_left[:, axis] = s
        idx_right[:, axis] = s + 1

        idx_left[:, other_axes[0]] = coords_2d[:, 0]
        idx_left[:, other_axes[1]] = coords_2d[:, 1]

        idx_right[:, other_axes[0]] = coords_2d[:, 0]
        idx_right[:, other_axes[1]] = coords_2d[:, 1]

        with torch.no_grad():
            c_left = model_forward(idx_left)
            c_right = model_forward(idx_right)

        # face flux approximated by absolute concentration difference
        face_flux = torch.abs(c_right - c_left)

        # normalize by full cross-section area (including solids) to match TauFactor
        full_cross_section_area = int(np.prod([shape[a] for a in other_axes]))

        mean_flux_slice = torch.sum(face_flux).item() / full_cross_section_area

        flux_per_slice.append(mean_flux_slice)

    flux_per_slice_np = np.array(flux_per_slice, dtype=np.float64)

    mean_flux = float(np.mean(flux_per_slice_np))

    D_rel = mean_flux * n_axis

    tau = float("inf") if D_rel <= 0 else float(epsilon / D_rel)

    return tau, D_rel, flux_per_slice_np


def solve_laplace_for_tau(
    geom: PoreGeometry,
    epsilon: float,
    inlet_axis: int = 0,
    epochs: int = 5000,
    lr: float = 5e-5,
    rff_encoder: RFFEncoder | None = None,
    output_vtk: str | Path = "runs/pinn_outputs/voxel_pinn_concentration.vtk",
    n_pde_batch: int = 20_000,
    n_bc_batch: int = 3_000,
) -> float:
    """Train a voxel-aware PINN to solve Laplace on the pore phase and estimate tau.

    geom must provide mask (boolean array) and sdf_grid (signed distance).
    """
    if geom.mask is None or geom.sdf_grid is None:
        raise ValueError("Voxel-aware PINN requires geom.mask and geom.sdf_grid.")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    mask_np = (geom.mask > 0).astype(np.uint8)
    sdf_np = geom.sdf_grid.astype(np.float32)

    shape = mask_np.shape

    mask_t = torch.tensor(
        mask_np,
        dtype=torch.bool,
        device=device,
    )

    sdf_grid = torch.tensor(
        sdf_np,
        dtype=torch.float32,
        device=device,
    )

    # list of pore voxel indices (N,3)
    pore_idx = torch.nonzero(mask_t, as_tuple=False)

    # inlet/outlet voxel indices on faces
    inlet_idx_all = get_face_indices(
        mask_t,
        inlet_axis,
        "min",
    )

    outlet_idx_all = get_face_indices(
        mask_t,
        inlet_axis,
        "max",
    )

    # scale SDF to roughly unit magnitude for encoding stability
    sdf_scale = torch.max(torch.abs(sdf_grid[mask_t]))

    if sdf_scale.item() == 0:
        sdf_scale = torch.tensor(
            1.0,
            dtype=torch.float32,
            device=device,
        )

    # 6-neighbor offsets for discrete Laplacian
    offsets = torch.tensor(
        [
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 1],
            [0, 0, -1],
        ],
        dtype=torch.long,
        device=device,
    )

    # optional random Fourier features encoder
    if rff_encoder is not None:
        rff_encoder = rff_encoder.to(device)

        with torch.no_grad():
            test_x = torch.zeros((1, 3), dtype=torch.float32, device=device)
            test_sdf = torch.zeros((1,), dtype=torch.float32, device=device)
            test_encoded = rff_encoder(test_x, test_sdf)
            in_dim = test_encoded.shape[1]

        print("RFF encoded dimension:", in_dim)

        model = MLP(in_dim=in_dim).to(device)

    else:
        model = MLP(in_dim=3).to(device)

    def model_forward(idx: torch.Tensor) -> torch.Tensor:
        """Wrapper: take voxel indices, normalize, optionally encode, then forward."""
        x = normalize_idx_points(
            idx,
            shape,
        )

        if rff_encoder is None:
            return model(x)

        sdf = get_sdf_values(
            sdf_grid,
            idx,
            sdf_scale,
        ).squeeze(-1)

        encoded = rff_encoder(
            x,
            sdf,
        )

        return model(encoded)

    opt = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )

    print("\n===== VOXEL-PHYSICS-AWARE PINN =====")
    print("epsilon:", epsilon)
    print("device:", device)
    print("transport axis:", inlet_axis)
    print("pore voxels:", pore_idx.shape[0])
    print("inlet voxels:", inlet_idx_all.shape[0])
    print("outlet voxels:", outlet_idx_all.shape[0])
    print("using RFF:", rff_encoder is not None)

    for epoch in range(epochs):
        opt.zero_grad()

        # ---------------- PDE LOSS: DISCRETE 6-NEIGHBOR LAPLACIAN ----------------
        # sample random pore centers for PDE residual computation
        choice = torch.randint(
            0,
            pore_idx.shape[0],
            (min(n_pde_batch, pore_idx.shape[0]),),
            device=device,
        )

        center_idx = pore_idx[choice]

        # neighbors shape: (batch, 6, 3)
        neighbor_idx = center_idx[:, None, :] + offsets[None, :, :]

        # check neighbor indices are inside volume bounds
        valid = (
            (neighbor_idx[:, :, 0] >= 0)
            & (neighbor_idx[:, :, 0] < shape[0])
            & (neighbor_idx[:, :, 1] >= 0)
            & (neighbor_idx[:, :, 1] < shape[1])
            & (neighbor_idx[:, :, 2] >= 0)
            & (neighbor_idx[:, :, 2] < shape[2])
        )

        neighbor_clipped = neighbor_idx.clone()

        neighbor_clipped[:, :, 0] = neighbor_clipped[:, :, 0].clamp(
            0,
            shape[0] - 1,
        )
        neighbor_clipped[:, :, 1] = neighbor_clipped[:, :, 1].clamp(
            0,
            shape[1] - 1,
        )
        neighbor_clipped[:, :, 2] = neighbor_clipped[:, :, 2].clamp(
            0,
            shape[2] - 1,
        )

        # determine which neighbors are pores
        neighbor_is_pore = mask_t[
            neighbor_clipped[:, :, 0],
            neighbor_clipped[:, :, 1],
            neighbor_clipped[:, :, 2],
        ]

        # only use neighbors that are valid and pore; otherwise mirror center (no-flux)
        use_neighbor = valid & neighbor_is_pore

        # No-flux wall:
        # solid/outside neighbors mirror center value to impose zero normal flux
        final_neighbor_idx = torch.where(
            use_neighbor[:, :, None],
            neighbor_clipped,
            center_idx[:, None, :].expand(-1, 6, -1),
        )

        # center concentrations and neighbor concentrations
        c_center = model_forward(center_idx)

        c_neighbors = model_forward(
            final_neighbor_idx.reshape(-1, 3)
        ).reshape(-1, 6, 1)

        # discrete Laplacian: sum over neighbor differences (six neighbors)
        discrete_lap = torch.sum(
            c_neighbors - c_center[:, None, :],
            dim=1,
        )

        # mean squared residual
        loss_pde = torch.mean(discrete_lap**2)

        # ---------------- DIRICHLET BOUNDARY CONDITIONS ----------------
        # sample random inlet/outlet voxels and impose Dirichlet values
        in_choice = torch.randint(
            0,
            inlet_idx_all.shape[0],
            (min(n_bc_batch, inlet_idx_all.shape[0]),),
            device=device,
        )

        out_choice = torch.randint(
            0,
            outlet_idx_all.shape[0],
            (min(n_bc_batch, outlet_idx_all.shape[0]),),
            device=device,
        )

        inlet_idx = inlet_idx_all[in_choice]
        outlet_idx = outlet_idx_all[out_choice]

        # inlet = 1.0, outlet = 0.0
        c_in = model_forward(inlet_idx)
        c_out = model_forward(outlet_idx)
        loss_in = torch.mean((c_in - 1.0) ** 2)
        loss_out = torch.mean((c_out - 0.0) ** 2)

        # total loss: weight BCs stronger to enforce boundary conditions
        loss = (
            loss_pde
            + 10.0 * loss_in
            + 10.0 * loss_out
        )

        loss.backward()
        opt.step()

        if epoch % 100 == 0:
            print(
                f"epoch {epoch:5d} | "
                f"loss {loss.item():.4e} | "
                f"pde {loss_pde.item():.4e} | "
                f"in {loss_in.item():.4e} | "
                f"out {loss_out.item():.4e}"
            )

    print("\nTraining finished.")

    # ---------------- TAUFACTOR-STYLE TORTUOSITY ----------------
    # compute slice-wise flux and convert to D_rel and tau
    tau_pinn, D_rel, flux_per_slice = tau_from_taufactor_style_flux(
        model_forward=model_forward,
        mask_t=mask_t,
        epsilon=epsilon,
        shape=shape,
        axis=inlet_axis,
    )

    print("\n===== TAUFACTOR-STYLE PINN TAU ESTIMATE =====")
    print("mean slice flux:", float(np.mean(flux_per_slice)))
    print("min slice flux:", float(np.min(flux_per_slice)))
    print("max slice flux:", float(np.max(flux_per_slice)))
    print("D_rel:", D_rel)
    print("tau_PINN:", tau_pinn)

    # ---------------- SAVE VTK ----------------
    # save predicted concentrations at a random subset of pore centers for visualization
    max_vtk_points = min(100_000, pore_idx.shape[0])

    vtk_choice = torch.randint(
        0,
        pore_idx.shape[0],
        (max_vtk_points,),
        device=device,
    )

    vtk_idx = pore_idx[vtk_choice]

    with torch.no_grad():
        c_pred = model_forward(vtk_idx).detach().cpu().numpy().ravel()

    xyz = (
        vtk_idx.detach().cpu().numpy().astype(np.float32)
        + 0.5
    ) * geom.voxel_size_m

    cloud = pv.PolyData(xyz)
    cloud["C_pred"] = c_pred

    out_path = Path(output_vtk)
    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cloud.save(out_path)

    print(f"Saved VTK: {out_path}")

    return tau_pinn