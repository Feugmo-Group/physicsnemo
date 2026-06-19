"""Fourier geometry encodings. Weeks 9-10.

E1: per-point Random Fourier Features on (x, SDF). Frozen, non-trainable.
E2: global low-frequency 3D-FFT descriptor of the binary pore indicator.

Both are referenced by the proposal §7.3 Eq. 20 and seed §7.4.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn


# -------------------- E1: per-point RFF on (x, SDF) --------------------

@dataclass
class RFFConfig:
    num_features: int = 64
    scales: tuple[float, ...] = (1.0, 4.0, 16.0, 64.0)
    seed: int = 0


class RFFEncoder(nn.Module):
    """gamma(x, d) = [sin(2π B_x x), cos(2π B_x x), sin(2π b_d d), cos(2π b_d d)].

    Frequencies sampled once from N(0, scale^2) per scale, then frozen.
    """

    def __init__(self, cfg: RFFConfig):
        super().__init__()
        rng = np.random.default_rng(cfg.seed)
        per_scale = cfg.num_features // len(cfg.scales)
        Bx = np.concatenate([s * rng.standard_normal((per_scale, 3)) for s in cfg.scales])
        bd = np.concatenate([s * rng.standard_normal((per_scale,))   for s in cfg.scales])
        self.register_buffer("Bx", torch.tensor(Bx, dtype=torch.float32))
        self.register_buffer("bd", torch.tensor(bd, dtype=torch.float32))

    def forward(self, x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        # x: (N,3), d: (N,)
        phi_x = 2 * torch.pi * x @ self.Bx.T          # (N, F)
        phi_d = 2 * torch.pi * d.unsqueeze(-1) * self.bd  # (N, F)
        return torch.cat(
            [torch.sin(phi_x), torch.cos(phi_x), torch.sin(phi_d), torch.cos(phi_d)],
            dim=-1,
        )


# -------------------- E2: global 3D-FFT descriptor --------------------

def fft_descriptor(
    mask: np.ndarray,
    k_shell: int = 5,
    rotational_avg: bool = False,
) -> np.ndarray:
    """Low-frequency magnitudes of FFT(χ_pore). Returns a flat vector.

    With rotational_avg=True returns the radially averaged power spectrum up to k_shell.
    """
    chi = mask.astype(np.float32) - mask.mean()
    F = np.fft.fftn(chi)
    F = np.fft.fftshift(F)
    cz, cy, cx = (s // 2 for s in F.shape)
    sl = (
        slice(cz - k_shell, cz + k_shell + 1),
        slice(cy - k_shell, cy + k_shell + 1),
        slice(cx - k_shell, cx + k_shell + 1),
    )
    mag = np.abs(F[sl])
    if rotational_avg:
        kz, ky, kx = np.indices(mag.shape) - k_shell
        kr = np.sqrt(kz**2 + ky**2 + kx**2).astype(int)
        out = np.array([mag[kr == r].mean() for r in range(k_shell + 1)])
        return out
    return mag.ravel()


#-----testing------

if __name__ == "__main__":

    from ct_to_physicsnemo.io import load_ct, crop_rev
    from ct_to_physicsnemo.geometry import PoreGeometry

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

    geom = PoreGeometry(
        stl_path="runs/meshes/I_1_crop.stl",
        mask=cropped,
        voxel_size_m=50e-9,
    )

    samples = geom.sample(
        n_interior=1000,
        n_boundary=1000,
        n_inlet=100,
        n_outlet=100,
        inlet_axis=2,
    )

    x = torch.tensor(
        samples.interior,
        dtype=torch.float32,
    )

    # Placeholder SDF for now
    d = torch.zeros(
        x.shape[0],
        dtype=torch.float32,
    )

    cfg = RFFConfig(
        num_features=64,
        scales=(1.0, 4.0, 16.0, 64.0),
        seed=0,
    )

    encoder = RFFEncoder(cfg)

    encoded = encoder(x, d)

    print("\n===== FOURIER ENCODING ON REAL GEOMETRY =====")
    print("Interior points:", x.shape)
    print("SDF values:", d.shape)
    print("Encoded features:", encoded.shape)

    desc = fft_descriptor(
        cropped,
        k_shell=5,
        rotational_avg=False,
    )

    print("\n===== FFT GEOMETRY DESCRIPTOR =====")
    print("Mask shape:", cropped.shape)
    print("FFT descriptor shape:", desc.shape)