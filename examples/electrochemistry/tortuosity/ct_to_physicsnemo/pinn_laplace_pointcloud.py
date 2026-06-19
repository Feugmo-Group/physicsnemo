"""Point-cloud/voxel-graph Laplace PINN for tortuosity.

The network represents concentration as c(x, y, z). SDF is used only for
near-wall-biased sampling and visualization, not as a differentiable coordinate
in the PDE. The PDE residual is evaluated on the pore voxel connectivity graph
so solid-blocked shortcuts are not part of the physics loss.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv
import torch
from torch import nn

from ct_to_physicsnemo.geometry import PoreGeometry


# Dirichlet BC values used in TauFactor-style point-cloud PINN
TOP_BC = -0.5
BOT_BC = 0.5
DELTA_C = abs(BOT_BC - TOP_BC)


class MLP(nn.Module):
    """Fully-connected MLP mapping features -> scalar concentration.

    Defaults: input dim 3 (x,y,z), larger hidden width and deeper network for
    improved capacity on point-cloud representations.
    """

    def __init__(self, in_dim: int = 3, hidden: int = 256, layers: int = 8):
        super().__init__()

        net = [nn.Linear(in_dim, hidden), nn.Tanh()]

        for _ in range(layers - 1):
            net += [nn.Linear(hidden, hidden), nn.Tanh()]

        net.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*net)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # returns (..., 1) concentration predictions
        return self.net(x)


class RFFEncoder(nn.Module):
    """Random Fourier Features encoder for (x, y, z, SDF) inputs.

    Samples Gaussian random frequencies once at construction and freezes them.
    Encoding (x,y,z,SDF) jointly at multiple scales overcomes spectral bias
    near pore walls, where the SDF dimension tells the network its distance
    from the solid boundary.

    Args:
        sigma: Bandwidth of the Gaussian frequency distribution. Should span
            the pore-size distribution; start with sigma=6 on [0,1]-normalized
            coordinates and increase for GRF geometries.
        n_freqs: Number of random frequency samples. Output dim = 2 * n_freqs.
        in_dim: Input dimension. 4 for (x, y, z, SDF); 3 for (x, y, z) only.
        seed: Optional seed for reproducible frequency sampling.
    """

    def __init__(
        self,
        sigma: float = 6.0,
        n_freqs: int = 128,
        in_dim: int = 4,
        seed: int | None = 42,
    ):
        super().__init__()
        rng = torch.Generator()
        if seed is not None:
            rng.manual_seed(seed)
        B = torch.randn(in_dim, n_freqs, generator=rng) * sigma
        self.register_buffer("B", B)  # frozen, not trained

    @property
    def out_dim(self) -> int:
        return 2 * int(self.B.shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, in_dim)  →  (N, 2*n_freqs)
        proj = x @ self.B  # (N, n_freqs)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


def normalize_points(x: torch.Tensor, xmin: torch.Tensor, scale: torch.Tensor):
    """Normalize raw physical coordinates to [0,1]-like range per-dimension."""
    return (x - xmin) / scale


def grad_c(model: nn.Module, features: torch.Tensor):
    """Compute gradient of model output with respect to input features.

    Returns tuple (c, grad) where grad has same leading dims as features and
    contains partial derivatives of concentration w.r.t each input feature.
    """
    features.requires_grad_(True)

    c = model(features)

    g = torch.autograd.grad(
        c,
        features,
        grad_outputs=torch.ones_like(c),
        create_graph=True,
    )[0]

    return c, g


def laplacian_xyz(
    model: nn.Module,
    xyz: torch.Tensor,
    coord_scale: torch.Tensor,
) -> torch.Tensor:
    """Compute a nondimensional Laplacian over spatial dims x,y,z.

    The model takes normalized coordinates. Multiplying the physical Laplacian
    by a reference length squared keeps the residual magnitude trainable while
    preserving anisotropic-domain scaling.
    """
    xyz.requires_grad_(True)

    c, g = grad_c(model, xyz)

    lap = 0.0
    reference_scale = torch.mean(coord_scale)

    # accumulate second derivatives for x,y,z components
    for i in range(3):
        g2 = torch.autograd.grad(
            g[:, i],
            xyz,
            grad_outputs=torch.ones_like(g[:, i]),
            create_graph=True,
        )[0][:, i]

        lap = lap + g2 * (reference_scale / coord_scale[i]) ** 2

    return lap


def random_batch(
    x: torch.Tensor,
    n: int,
) -> torch.Tensor:
    """Return random subset of points."""
    idx = torch.randint(
        0,
        x.shape[0],
        (min(n, x.shape[0]),),
        device=x.device,
    )

    return x[idx]


def biased_near_wall_batch(
    x: torch.Tensor,
    sdf: torch.Tensor,
    n: int,
    near_wall_fraction: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a mixture of near-wall points (small |SDF|) and random points.

    This biases PDE residual samples toward boundary layers where gradients are
    larger and learning is typically harder.
    """
    n = min(n, x.shape[0])
    n_wall = int(n * near_wall_fraction)
    n_rand = n - n_wall

    abs_sdf = torch.abs(sdf.squeeze())

    choices = []

    if n_wall > 0:
        # pick a pool of near-wall candidates, then sample uniformly from that pool
        k = min(max(n_wall * 4, n_wall), x.shape[0])
        _, near_idx_pool = torch.topk(abs_sdf, k=k, largest=False)

        wall_choice = near_idx_pool[
            torch.randint(0, near_idx_pool.shape[0], (n_wall,), device=x.device)
        ]
        choices.append(wall_choice)

    if n_rand > 0:
        rand_choice = torch.randint(0, x.shape[0], (n_rand,), device=x.device)
        choices.append(rand_choice)

    idx = torch.cat(choices, dim=0)

    return x[idx], sdf[idx]


def idx_to_physical_points(
    idx: torch.Tensor,
    voxel_size: float,
) -> torch.Tensor:
    """Convert integer voxel indices to physical coordinates at voxel centers."""
    return (idx.float() + 0.5) * voxel_size


def idx_to_normalized_points(
    idx: torch.Tensor,
    shape_t: torch.Tensor,
) -> torch.Tensor:
    """Convert voxel indices directly to normalized voxel-center coordinates."""
    return (idx.float() + 0.5) / shape_t


def taufactor_style_flux_from_model(
    model_forward,
    mask: np.ndarray,
    voxel_size: float,
    xmin: torch.Tensor,
    scale: torch.Tensor,
    axis: int = 0,
    batch_size: int = 200_000,
) -> tuple[float, float, np.ndarray]:
    """Compute slice-wise flux in TauFactor style from a trained point-cloud model.

    For each adjacent slice pair along `axis`, it:
    - finds pore voxels that are conducting on both sides,
    - evaluates concentration on left/right voxel centers in batches,
    - sums through-slice flux (c_right - c_left) and normalizes by full cross-section.

    Returns (mean_flux, D_rel, flux_per_slice_array).
    """
    device = xmin.device

    # boolean mask as tensor on device
    mask_t = torch.tensor(mask > 0, dtype=torch.bool, device=device)

    shape = mask.shape
    n_axis = shape[axis]

    other_axes = [a for a in range(3) if a != axis]
    full_cross_section_area = shape[other_axes[0]] * shape[other_axes[1]]

    flux_per_slice = []

    # iterate slice pairs and compute mean absolute through-slice flux per slice
    for s in range(n_axis - 1):
        left_slice = [slice(None)] * 3
        right_slice = [slice(None)] * 3

        left_slice[axis] = s
        right_slice[axis] = s + 1

        # only positions where both adjacent voxels are pore (conducting)
        conducting_face = mask_t[tuple(left_slice)] & mask_t[tuple(right_slice)]

        coords_2d = torch.nonzero(conducting_face, as_tuple=False)

        if coords_2d.shape[0] == 0:
            flux_per_slice.append(0.0)
            continue

        # reconstruct left/right 3D voxel indices for the conducting face
        idx_left = torch.zeros((coords_2d.shape[0], 3), dtype=torch.long, device=device)
        idx_right = torch.zeros_like(idx_left)

        idx_left[:, axis] = s
        idx_right[:, axis] = s + 1

        idx_left[:, other_axes[0]] = coords_2d[:, 0]
        idx_left[:, other_axes[1]] = coords_2d[:, 1]

        idx_right[:, other_axes[0]] = coords_2d[:, 0]
        idx_right[:, other_axes[1]] = coords_2d[:, 1]

        total_flux = 0.0

        # evaluate model in manageable batches to avoid memory blowup
        for start in range(0, idx_left.shape[0], batch_size):
            end = start + batch_size

            left_batch = idx_left[start:end]
            right_batch = idx_right[start:end]

            # convert voxel indices -> physical coords -> normalized features
            x_left_raw = idx_to_physical_points(left_batch, voxel_size)
            x_right_raw = idx_to_physical_points(right_batch, voxel_size)

            x_left = normalize_points(x_left_raw, xmin, scale)
            x_right = normalize_points(x_right_raw, xmin, scale)

            with torch.no_grad():
                c_left = model_forward(x_left)
                c_right = model_forward(x_right)

            # directed flux summed across the face (can be signed)
            face_flux = c_right - c_left
            total_flux += torch.sum(face_flux).item()

        # normalize by full cross-section area and take absolute value to match TauFactor
        mean_flux_slice = abs(total_flux) / full_cross_section_area
        flux_per_slice.append(mean_flux_slice)

    flux_per_slice = np.array(flux_per_slice, dtype=np.float64)

    mean_flux = float(np.mean(flux_per_slice))
    D_rel = mean_flux * n_axis / DELTA_C

    return mean_flux, D_rel, flux_per_slice


def solve_pointcloud_laplace_for_tau(
    geom: PoreGeometry,
    epsilon: float,
    inlet_axis: int = 0,
    epochs: int = 10_000,
    lr: float = 1e-4,
    n_interior: int = 100_000,
    n_boundary: int = 75_000,
    n_inlet: int = 8_000,
    n_outlet: int = 8_000,
    batch_interior: int = 12_000,
    batch_boundary: int = 12_000,
    batch_bc: int = 4_000,
    near_wall_fraction: float = 0.25,
    pde_weight: float = 1.0,
    bc_weight: float = 50.0,
    wall_weight: float = 0.0,
    energy_weight: float = 25.0,
    teacher_weight: float = 0.0,
    teacher_edge_weight: float = 0.0,
    teacher_transport_edge_weight: float = 0.0,
    teacher_concentration_grid: np.ndarray | None = None,
    rff_sigma: float = 6.0,
    rff_n_freqs: int = 128,
    rff_seed: int = 42,
    output_vtk: str | Path = "runs/pinn_outputs/pointcloud_concentration.vtk",
) -> tuple[float, float]:
    """Train point-cloud PINN to solve Laplace and estimate tortuosity (tau).

    Produces a TauFactor-style D_rel and tau estimate computed from slice flux.
    """

    if geom.mask is None or geom.sdf_grid is None:
        raise ValueError("Point-cloud PINN requires geom.mask and geom.sdf_grid.")

    if not 0.0 <= near_wall_fraction <= 1.0:
        raise ValueError("near_wall_fraction must be between 0.0 and 1.0.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mask_np = (geom.mask > 0).astype(np.uint8)
    shape = mask_np.shape

    mask_t = torch.tensor(
        mask_np,
        dtype=torch.bool,
        device=device,
    )

    teacher_grid_t = None
    if teacher_concentration_grid is not None:
        if teacher_concentration_grid.shape != shape:
            raise ValueError(
                "teacher_concentration_grid must have the same shape as geom.mask."
            )
        teacher_grid_t = torch.tensor(
            teacher_concentration_grid,
            dtype=torch.float32,
            device=device,
        )

    pore_idx = torch.nonzero(mask_t, as_tuple=False)
    shape_t = torch.tensor(
        shape,
        dtype=torch.float32,
        device=device,
    )

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

    transport_edge_left = None
    transport_edge_right = None
    if teacher_grid_t is not None and teacher_transport_edge_weight > 0.0:
        left_slice = [slice(None)] * 3
        right_slice = [slice(None)] * 3
        left_slice[inlet_axis] = slice(0, shape[inlet_axis] - 1)
        right_slice[inlet_axis] = slice(1, shape[inlet_axis])

        conducting_transport_face = (
            mask_t[tuple(left_slice)] & mask_t[tuple(right_slice)]
        )
        coords = torch.nonzero(conducting_transport_face, as_tuple=False)

        if coords.shape[0] > 0:
            idx_left = coords
            idx_right = coords.clone()
            idx_right[:, inlet_axis] += 1

            transport_edge_left = idx_left
            transport_edge_right = idx_right

    # sample point-clouds from geometry (interior, boundary, inlet/outlet)
    samples = geom.sample(
        n_interior=n_interior,
        n_boundary=n_boundary,
        n_inlet=n_inlet,
        n_outlet=n_outlet,
        inlet_axis=inlet_axis,
    )

    # move sampled arrays to torch tensors on device
    interior_raw = torch.tensor(samples.interior, dtype=torch.float32, device=device)
    boundary_raw = torch.tensor(samples.boundary, dtype=torch.float32, device=device)
    inlet_raw = torch.tensor(samples.inlet, dtype=torch.float32, device=device)
    outlet_raw = torch.tensor(samples.outlet, dtype=torch.float32, device=device)

    interior_sdf = torch.tensor(samples.interior_sdf, dtype=torch.float32, device=device)
    normals = torch.tensor(samples.boundary_normals, dtype=torch.float32, device=device)

    # Use deterministic volume extents so interior, wall, inlet, outlet, and
    # flux-evaluation points all share the same coordinate transform.
    xmin = torch.zeros(3, dtype=torch.float32, device=device)
    scale = torch.tensor(
        geom.mask.shape,
        dtype=torch.float32,
        device=device,
    ) * geom.voxel_size_m
    scale[scale == 0] = 1.0

    # SDF scaling for numerical stability
    sdf_scale = torch.max(torch.abs(interior_sdf))
    if sdf_scale.item() == 0:
        sdf_scale = torch.tensor(1.0, dtype=torch.float32, device=device)

    # normalize positions and SDFs
    interior = normalize_points(interior_raw, xmin, scale)
    boundary = normalize_points(boundary_raw, xmin, scale)
    inlet = normalize_points(inlet_raw, xmin, scale)
    outlet = normalize_points(outlet_raw, xmin, scale)

    interior_sdf_norm = interior_sdf / sdf_scale

    # Build SDF grid at voxel resolution for graph-batch SDF lookup.
    # Pore voxels get their SDF value; solid voxels get 0 (they are never sampled).
    sdf_grid_np = geom.sdf_grid if geom.sdf_grid is not None else np.zeros(shape, dtype=np.float32)
    sdf_grid_t = torch.tensor(
        sdf_grid_np / (sdf_scale.item() + 1e-8),
        dtype=torch.float32,
        device=device,
    )

    # RFF encoder: (x, y, z, SDF_norm) → 2*n_freqs features
    encoder = RFFEncoder(
        sigma=rff_sigma,
        n_freqs=rff_n_freqs,
        in_dim=4,
        seed=rff_seed,
    ).to(device)
    model = MLP(in_dim=encoder.out_dim).to(device)

    def _append_sdf_voxel(idx: torch.Tensor) -> torch.Tensor:
        """Append normalized SDF value to voxel-center normalized coordinates."""
        xyz_norm = idx_to_normalized_points(idx, shape_t)
        sdf_vals = sdf_grid_t[idx[:, 0], idx[:, 1], idx[:, 2]].unsqueeze(1)
        return torch.cat([xyz_norm, sdf_vals], dim=1)

    def _append_sdf_pointcloud(xyz_norm: torch.Tensor, sdf_vals: torch.Tensor) -> torch.Tensor:
        """Append per-point SDF values to normalized point-cloud coordinates."""
        return torch.cat([xyz_norm, sdf_vals.unsqueeze(1) if sdf_vals.dim() == 1 else sdf_vals], dim=1)

    def model_forward(x: torch.Tensor) -> torch.Tensor:
        return model(encoder(x))

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        opt,
        milestones=[
            max(epochs // 2, 1),
            max((3 * epochs) // 4, 1),
        ],
        gamma=0.5,
    )

    print("\n===== IMPROVED POINT-CLOUD LAPLACE PINN =====")
    print("device:", device)
    print("epsilon:", epsilon)
    print("transport axis:", inlet_axis)
    print("BCs:", TOP_BC, BOT_BC)
    print("interior points:", interior.shape[0])
    print("boundary points:", boundary.shape[0])
    print("inlet points:", inlet.shape[0])
    print("outlet points:", outlet.shape[0])
    print("pore graph voxels:", pore_idx.shape[0])
    print("lr:", lr)
    print("lr scheduler milestones:", scheduler.milestones)
    print("near-wall fraction:", near_wall_fraction)
    print(f"RFF encoder: sigma={rff_sigma}, n_freqs={rff_n_freqs}, in_dim=4 (x,y,z,SDF), seed={rff_seed}")
    print(
        "loss weights:",
        f"pde={pde_weight}",
        f"bc={bc_weight}",
        f"wall={wall_weight}",
        f"energy={energy_weight}",
        f"teacher={teacher_weight}",
        f"teacher_edge={teacher_edge_weight}",
        f"teacher_transport_edge={teacher_transport_edge_weight}",
    )

    # training loop
    for epoch in range(epochs):
        opt.zero_grad()

        # Graph PDE residual: each pore voxel exchanges only with connected
        # pore neighbors; solid/outside neighbors mirror the center value.
        pde_choice = torch.randint(
            0,
            pore_idx.shape[0],
            (min(batch_interior, pore_idx.shape[0]),),
            device=device,
        )

        center_idx = pore_idx[pde_choice]
        neighbor_idx = center_idx[:, None, :] + offsets[None, :, :]

        valid = (
            (neighbor_idx[:, :, 0] >= 0)
            & (neighbor_idx[:, :, 0] < shape[0])
            & (neighbor_idx[:, :, 1] >= 0)
            & (neighbor_idx[:, :, 1] < shape[1])
            & (neighbor_idx[:, :, 2] >= 0)
            & (neighbor_idx[:, :, 2] < shape[2])
        )

        neighbor_clipped = neighbor_idx.clone()
        neighbor_clipped[:, :, 0] = neighbor_clipped[:, :, 0].clamp(0, shape[0] - 1)
        neighbor_clipped[:, :, 1] = neighbor_clipped[:, :, 1].clamp(0, shape[1] - 1)
        neighbor_clipped[:, :, 2] = neighbor_clipped[:, :, 2].clamp(0, shape[2] - 1)

        neighbor_is_pore = mask_t[
            neighbor_clipped[:, :, 0],
            neighbor_clipped[:, :, 1],
            neighbor_clipped[:, :, 2],
        ]

        use_neighbor = valid & neighbor_is_pore

        final_neighbor_idx = torch.where(
            use_neighbor[:, :, None],
            neighbor_clipped,
            center_idx[:, None, :].expand(-1, 6, -1),
        )

        c_center = model_forward(
            _append_sdf_voxel(center_idx)
        )

        c_neighbors = model_forward(
            _append_sdf_voxel(final_neighbor_idx.reshape(-1, 3))
        ).reshape(-1, 6, 1)

        graph_lap = torch.sum(
            c_neighbors - c_center[:, None, :],
            dim=1,
        )

        loss_pde = torch.mean(graph_lap**2)

        connected_edge_diff = c_neighbors - c_center[:, None, :]
        connected_edge_mask = use_neighbor[:, :, None]
        if torch.any(connected_edge_mask):
            loss_energy = torch.mean(
                connected_edge_diff[connected_edge_mask] ** 2
            ) * (shape[inlet_axis] ** 2)
        else:
            loss_energy = torch.zeros((), dtype=torch.float32, device=device)

        if teacher_grid_t is not None and teacher_weight > 0.0:
            teacher_center = teacher_grid_t[
                center_idx[:, 0],
                center_idx[:, 1],
                center_idx[:, 2],
            ].reshape(-1, 1)
            loss_teacher = torch.mean((c_center - teacher_center) ** 2)
        else:
            loss_teacher = torch.zeros((), dtype=torch.float32, device=device)

        if teacher_grid_t is not None and teacher_edge_weight > 0.0:
            teacher_neighbors = teacher_grid_t[
                final_neighbor_idx[:, :, 0].reshape(-1),
                final_neighbor_idx[:, :, 1].reshape(-1),
                final_neighbor_idx[:, :, 2].reshape(-1),
            ].reshape(-1, 6, 1)

            teacher_center = teacher_grid_t[
                center_idx[:, 0],
                center_idx[:, 1],
                center_idx[:, 2],
            ].reshape(-1, 1)

            teacher_edge_diff = teacher_neighbors - teacher_center[:, None, :]

            if torch.any(connected_edge_mask):
                loss_teacher_edge = torch.mean(
                    (
                        connected_edge_diff[connected_edge_mask]
                        - teacher_edge_diff[connected_edge_mask]
                    )
                    ** 2
                ) * (shape[inlet_axis] ** 2)
            else:
                loss_teacher_edge = torch.zeros((), dtype=torch.float32, device=device)
        else:
            loss_teacher_edge = torch.zeros((), dtype=torch.float32, device=device)

        if (
            teacher_grid_t is not None
            and teacher_transport_edge_weight > 0.0
            and transport_edge_left is not None
        ):
            n_edges = transport_edge_left.shape[0]
            edge_choice = torch.randint(
                0,
                n_edges,
                (min(batch_interior, n_edges),),
                device=device,
            )

            left_idx = transport_edge_left[edge_choice]
            right_idx = transport_edge_right[edge_choice]

            c_left = model_forward(_append_sdf_voxel(left_idx))
            c_right = model_forward(_append_sdf_voxel(right_idx))

            teacher_left = teacher_grid_t[
                left_idx[:, 0],
                left_idx[:, 1],
                left_idx[:, 2],
            ].reshape(-1, 1)
            teacher_right = teacher_grid_t[
                right_idx[:, 0],
                right_idx[:, 1],
                right_idx[:, 2],
            ].reshape(-1, 1)

            loss_teacher_transport_edge = torch.mean(
                ((c_right - c_left) - (teacher_right - teacher_left)) ** 2
            ) * (shape[inlet_axis] ** 2)
        else:
            loss_teacher_transport_edge = torch.zeros(
                (),
                dtype=torch.float32,
                device=device,
            )

        # uniform random sample from boundary points
        xb_idx = torch.randint(
            0,
            boundary.shape[0],
            (min(batch_boundary, boundary.shape[0]),),
            device=device,
        )

        xb = boundary[xb_idx]
        nb = normals[xb_idx]  # boundary normals for no-flux constraint

        xin_idx = torch.randint(0, inlet.shape[0], (min(batch_bc, inlet.shape[0]),), device=device)
        xout_idx = torch.randint(0, outlet.shape[0], (min(batch_bc, outlet.shape[0]),), device=device)

        # Inlet/outlet SDF ≈ 0 (face voxels are on/near the boundary)
        xin_sdf = torch.zeros(xin_idx.shape[0], 1, device=device)
        xout_sdf = torch.zeros(xout_idx.shape[0], 1, device=device)

        xin_feat = _append_sdf_pointcloud(inlet[xin_idx], xin_sdf.squeeze(1))
        xout_feat = _append_sdf_pointcloud(outlet[xout_idx], xout_sdf.squeeze(1))

        c_in = model_forward(xin_feat)
        c_out = model_forward(xout_feat)

        loss_in = torch.mean((c_in - TOP_BC) ** 2)
        loss_out = torch.mean((c_out - BOT_BC) ** 2)

        if wall_weight > 0.0:
            # Wall SDF ≈ 0 by definition (boundary points sit on the pore wall)
            xb_sdf = torch.zeros(xb.shape[0], 1, device=device)
            xb_feat = _append_sdf_pointcloud(xb, xb_sdf.squeeze(1))
            _, gb = grad_c(model_forward, xb_feat)
            wall_flux = torch.sum(gb[:, :3] * nb, dim=1)  # gradient w.r.t. xyz only
            loss_wall = torch.mean(wall_flux**2)
        else:
            loss_wall = torch.zeros((), dtype=torch.float32, device=device)

        # weighted total loss: BCs emphasized, wall flux moderately enforced
        loss = (
            pde_weight * loss_pde
            + bc_weight * loss_in
            + bc_weight * loss_out
            + wall_weight * loss_wall
            + energy_weight * loss_energy
            + teacher_weight * loss_teacher
            + teacher_edge_weight * loss_teacher_edge
            + teacher_transport_edge_weight * loss_teacher_transport_edge
        )

        loss.backward()
        opt.step()
        scheduler.step()

        if epoch % 100 == 0:
            current_lr = opt.param_groups[0]["lr"]
            print(
                f"epoch {epoch:5d} | "
                f"lr {current_lr:.2e} | "
                f"loss {loss.item():.4e} | "
                f"pde {loss_pde.item():.4e} | "
                f"in {loss_in.item():.4e} | "
                f"out {loss_out.item():.4e} | "
                f"wall {loss_wall.item():.4e} | "
                f"energy {loss_energy.item():.4e} | "
                f"teacher {loss_teacher.item():.4e} | "
                f"teacher_edge {loss_teacher_edge.item():.4e} | "
                f"teacher_transport_edge {loss_teacher_transport_edge.item():.4e}"
            )

    print("\nTraining finished.")

    # compute slice-wise flux and derived transport metrics from trained model
    mean_flux, D_rel, flux_per_slice = taufactor_style_flux_from_model(
        model_forward=model_forward,
        mask=geom.mask,
        voxel_size=geom.voxel_size_m,
        xmin=xmin,
        scale=scale,
        axis=inlet_axis,
    )

    tau = epsilon / D_rel if D_rel > 0 else float("inf")

    flux_min = float(np.min(flux_per_slice))
    flux_max = float(np.max(flux_per_slice))

    flux_relative_error = (
        (flux_max - flux_min) / flux_max
        if flux_max > 0
        else float("inf")
    )

    print("\n===== TAUFACTOR-STYLE POINT-CLOUD TAU =====")
    print("mean slice flux:", mean_flux)
    print("min slice flux:", flux_min)
    print("max slice flux:", flux_max)
    print("flux relative error:", flux_relative_error)
    print("D_rel:", D_rel)
    print("tau:", tau)

    # save a subset of interior predictions for visualization (VTK)
    with torch.no_grad():
        max_pts = min(100_000, interior.shape[0])
        vtk_idx = torch.randint(0, interior.shape[0], (max_pts,), device=device)

        vtk_raw = interior_raw[vtk_idx]
        vtk_sdf = interior_sdf_norm[vtk_idx]
        vtk_feat = _append_sdf_pointcloud(interior[vtk_idx], vtk_sdf)
        c_pred = model_forward(vtk_feat).detach().cpu().numpy().ravel()

    cloud = pv.PolyData(vtk_raw.detach().cpu().numpy())
    cloud["C_pred"] = c_pred
    cloud["sdf"] = interior_sdf_norm[vtk_idx].detach().cpu().numpy().ravel()

    output_vtk = Path(output_vtk)
    output_vtk.parent.mkdir(parents=True, exist_ok=True)
    cloud.save(output_vtk)

    print("Saved VTK:", output_vtk)

    return tau, D_rel
