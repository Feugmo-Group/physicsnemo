# Summer Project — From Graphite CT Scans to a PhysicsNeMo Sym Forward Solver

**Student:** TBD
**Supervisors:** C. G. Tetsassi Feugmo, V. S. Guruprasad
**Duration:** 12 weeks
**Project repo:** `ct_to_physicsnemo/`
**Status:** Scoped, ready to start

---

## 1. Why this project exists

The research proposal *"PINN for EIS-Driven Microstructure Characterisation and Inverse Pore Design
of Graphite Battery Electrodes"* (Guruprasad & Feugmo, 2026) needs a reliable pipeline that converts
a 3D X-ray CT scan of a graphite anode into a geometry object that NVIDIA PhysicsNeMo Sym can
sample, train on, and differentiate through.

That pipeline is currently the bottleneck for **Phase 1 (T1.3, T1.4)** and **Phase 2 (T2.3, T2.5)**
of the proposal timeline. Until it works on real volumes, none of the 3D PINN, surrogate, or
inverse-design work can start on real data.

**This project builds that pipeline, validates it against the TauFactor baseline, and runs one method
experiment — a Fourier-based geometry encoding — that directly seeds Phase 3 (operator-style
surrogate).**

---

## 2. How this project integrates with the EIS proposal

| Project deliverable | Proposal task it unblocks | Section |
|---|---|---|
| Reproducible CT → segmented voxel volume pipeline | T1.1 | §9.2 |
| Microstructural metrics (ε, τ, a_v, PSD) for 3 volumes | T1.1, T1.2 | §5, §9.2 |
| Watertight STL of pore/solid interface per volume | T1.3 | §9.3 |
| PhysicsNeMo Sym `Tessellated` geometry import + point-cloud sampling | T1.3 | §8, §9.3 |
| Steady-state Laplace PINN; τ_PINN vs. TauFactor agreement (<5%) | T1.4, T2.5 | §7.2 (steady-state limit of Eq. 14) |
| Frequency-domain diffusion solve at 5 representative ω on 1 real volume | T2.3 | §7.2 (Eqs. 15–16) |
| **Fourier geometry-encoding ablation** (per-point RFF on SDF + global FFT descriptor) | Direct seed for T2.x and T3.2 | §7.3 (Eq. 20), §7.4 |

At the end of the summer, the proposal can quote a real number for "PINN τ vs. TauFactor on graphite
CT data," cite the geometry-encoding ablation as preliminary results, and reuse the codebase verbatim.

---

## 2b. Datasets (confirmed available)

**Real CT (in `../diffusion_in_porous_media/`):** 4 electrodes × 3 samples = 12 volumes, each with
`*_raw.tif` (grayscale) and `*_bin.tif` (pre-segmented binary; segmentation step is optional and only
needed for E1 ablation against U-Net soft segmentation).

```
Electrode I  : I_1, I_2, I_3            (~182–486 MB / volume)
Electrode II : II_1 (Part_a), II_2, II_3 (Part_b)   (~254 MB–1 GB)
Electrode III: III_1, III_2, III_3
Electrode IV : IV_1, IV_2, IV_3
```

TauFactor reference code is in `../diffusion_in_porous_media/taufactor-main/`; use it directly for the
τ baseline.

**Synthetic CT (generated in week 1, via `ct_to_physicsnemo.synthetic`):** ~20 volumes covering a
(ε, τ, a_v) grid, produced by two generators:

- **CSG packed-sphere / packed-ellipsoid** (graphite-flake-like anisotropic ellipsoids): controls ε
  via packing density and a_v via particle size.
- **Gaussian Random Field (GRF) thresholded** (`porespy.generators.blobs`-style): controls ε via
  threshold and correlation length, giving naturalistic isotropic porosity.

The synthetic-first strategy decouples pipeline bugs from CT artefacts: ground-truth τ on synthetic
data is either analytical (sphere packing) or computed by the voxel-graph CG solver, so any pipeline
error shows up immediately.

**Library for the operator pilot (E2):** 3 real + ~20 synthetic = ~23 geometries. Small enough to
train on a single GPU; large enough that operator generalisation is a meaningful claim.

---

## 3. Deliverables

1. The `ct_to_physicsnemo/` Python package, documented, tested, runnable on a single GPU.
2. Three reproducible end-to-end runs (one per CT volume), with metrics committed.
3. A `results/` notebook with figures: τ_PINN vs. τ_TauFactor, one Nyquist plot per volume, and the
   geometry-encoding ablation figure.
4. A 6-page report following the section structure of the proposal so paragraphs can be lifted
   directly.
5. A 10-slide hand-off deck.

---

## 4. Weekly plan

| Wk | Goal | Concrete output | Proposal hook |
|---|---|---|---|
| 1 | Env setup (PhysicsNeMo Sym 25.08, PoreSpy, scikit-image, PyMeshLab, TauFactor); generate **20 synthetic CT volumes** via `synthetic.py` (CSG + GRF); smoke test | Smoke test passes; `synthetic/` dir with 20 binary volumes + per-volume metadata | seeds T3.1 |
| 2 | **Synthetic-first:** run full pipeline on 3 synthetic volumes — load → metrics → mesh → PhysicsNeMo geometry → Laplace PINN (voxel-graph CG + PINN distillation) | τ_PINN vs. analytical/FD reference within 5% on synthetic | T1.4 sanity check |
| 3 | Extend to all 20 synthetic; sweep (ε, τ) to reproduce Cooper-2017 finding "same τ, ε → different spectra" in the diffusion limit | Figure: scalar surrogate baseline fails; geometry-aware encoding required | Justifies §7.4 operator |
| 4 | EIS frequency-domain solve on 3 synthetic at 5 ω each; assemble Nyquist | Per-volume Nyquist sweep on synthetic | T2.3 dry run |
| 5 | **Switch to real CT:** load Electrode I/III/IV `*_bin.tif`, REV-crop, compute ε, a_v, PSD (PoreSpy), τ (TauFactor + voxel-graph CG) | `metrics.csv` for the 3 chosen real volumes (one per electrode) | T1.1 |
| 6 | Marching cubes → STL → PyMeshLab cleanup; PhysicsNeMo `Tessellation` import; verify SDF + boundary normals; sample point clouds | 3 watertight STLs (<500k tris); `geometry.py` working on real data | T1.3, §8, §9.3 |
| 7 | **Steady-state Laplace PINN + E1 RFF ablation** on the 3 real volumes; compute τ_PINN. Run E1 now so the correct encoder is in place before EIS. | <5% match vs. TauFactor; E1 RFF vs. plain MLP loss curves | T1.4, T2.5, §7.3 |
| 8 | Frequency-domain EIS solve on real volumes; partial Nyquist | Per-volume Nyquist sweep on real CT | T2.3 |
| 9 | **Experiment E1 full ablation:** compare RFF on (x,y,z,SDF) vs. plain MLP vs. PhysicsNeMo `ModifiedFourierNetArch` on both synthetic and real; EIS case is the definitive test | Loss curves + final τ accuracy table; wall-distance residual decomposition | §7.3 (Eq. 20) |
| 10 | **Experiment E2:** global 3D-FFT descriptor of χ_pore as conditioning; train ONE network on the library of 23 geometries | Compare to 23 independent per-volume PINNs on τ | Seeds T3.2 / operator surrogate |
| 11 | Cross-volume comparison; aged-vs-fresh axis if time allows; write methods section | Draft report §§2–4 | §7, §9 |
| 12 | Final report + clean repo + slides + hand-off meeting | Final package | — |

---

## 5. The two Fourier geometry-encoding experiments

These two experiments are the student's method contribution. Both are small, well-scoped, and land
directly in Eq. (20) and §7.4 of the proposal.

### E1 — Per-point Random Fourier Features on (x, y, z, SDF)

**Why:** The Helmholtz solution oscillates on the scale of l_δ(ω) = √(D/ω), which at high frequencies
can be smaller than two pore diameters. Plain MLPs have spectral bias against this — they blur
concentration gradients near walls. Standard NeRF-style RFF removes spectral bias.

**Why the SDF must be encoded, not just used for sampling:** The SDF tells the network its distance
from the wall, which is where the gradient is steepest. Encoding it jointly with (x,y,z) through
random Gaussian frequencies forces the network to represent the solution at fine scales near the wall.
Using it only as a sampling bias (as in the current teacher-distillation setup) leaves this capacity
untapped.

**What:** Replace the raw input (x, y, z, SDF) with

```
γ(x, d) = [sin(2π B_x x), cos(2π B_x x), sin(2π b_d d), cos(2π b_d d)]
```

where B_x ∈ R^{m×3} and b_d ∈ R^m are sampled from Gaussians at multiple log-spaced scales spanning
the pore-size distribution measured in week 4. The encoding layer is frozen (non-trainable). This sits
on top of Tony's teacher-distillation framework — the two are complementary.

**Implementation note:** Tony's current `FourierPositionEncoder` uses deterministic frequencies
(1, 2, 4, 8) and does not include SDF. Replace it with a proper RFF layer:

```python
class RFFEncoder(nn.Module):
    def __init__(self, sigma: float = 6.0, n_freqs: int = 128, in_dim: int = 4):
        super().__init__()
        B = torch.randn(in_dim, n_freqs) * sigma
        self.register_buffer("B", B)  # frozen

    @property
    def out_dim(self) -> int:
        return 2 * self.B.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = x @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
```

Feed it normalized (x, y, z, SDF) — four-dimensional input. σ ∈ [4, 10] on [0,1]-normalized
coordinates is a good starting range; tune upward for GRF cases.

**Measure:**
- PDE-residual loss decomposed by wall-distance bin (does it drop near walls vs. plain MLP?)
- Final τ_PINN accuracy vs. TauFactor
- Wall-clock per epoch
- Comparison row vs. PhysicsNeMo `ModifiedFourierNetArch`
- Most important test: the EIS/Helmholtz case (week 8), where l_δ shrinks with ω

### E2 — Global 3D-FFT descriptor of χ_pore as conditioning input

**Why:** A single PINN per CT volume cannot be deployed, cannot be inverted across geometries, and
cannot scale. The network must take geometry as an *input*. The 3D FFT of the binary pore indicator
is the cheapest descriptor that:
- Is translation-invariant (magnitudes are phase-agnostic)
- Is multi-scale by construction (low k = bulk porosity; mid k = connectivity texture)
- Avoids meshing entirely
- Can be computed in milliseconds per volume

**What:**
1. Compute FFT_3(χ_pore) on the binary mask.
2. Keep the K lowest-frequency magnitudes (5×5×5 shell → K = 125) as descriptor g ∈ R^K.
   Optionally use rotationally averaged spectra for orientation invariance.
3. Train ONE network conditioned on (x, ω, g) across all 23 volumes simultaneously.
4. Compare against 23 independent per-volume PINNs.

**Practical advantage:** Tony's `solve_voxel_graph_laplace_for_tau` gives the exact concentration
field for each geometry in seconds. This provides both the τ label and the teacher signal for the
conditioned PINN across all 23 geometries simultaneously — E2 training is therefore fully supervised.

**Measure:** Does the conditioned network match per-volume PINNs within 10% on τ at lower total
training cost? If yes, this is a publishable preliminary result and directly justifies the
operator-learning direction in the proposal's revised §7.4.

---

## 6. Current pipeline state

The following is implemented and tested in `src/ct_to_physicsnemo/`:

| Component | File | Status |
|---|---|---|
| Synthetic geometry generator | `synthetic.py` | Done |
| Geometry + SDF pipeline | `geometry.py` | Done |
| Marching Cubes → STL | `mesh.py` | Done |
| TauFactor baseline wrapper | `metrics.py` | Done |
| **E2 FFT geometry descriptor** | `metrics.compute_fft_descriptor` | **Done — tested** |
| **Voxel-graph CG Laplace solver** | `voxel_graph_laplace.solve_voxel_graph_laplace_for_tau` | Done (primary τ method) |
| **Voxel-graph CG Helmholtz solver (EIS teacher)** | `voxel_graph_laplace.solve_voxel_graph_helmholtz_for_tau` | **Done — tested** |
| **E1 RFF encoder on (x,y,z,SDF)** | `pinn_laplace_pointcloud.RFFEncoder` | **Done — tested** |
| Point-cloud PINN + teacher distillation | `pinn_laplace_pointcloud.solve_pointcloud_laplace_for_tau` | Done (uses RFF) |
| Full validation runner | `tortuosity_pointcloud_validation.py` | Done |
| Porosity / surface area checks | `porosity_check.py`, `surface_area_check.py` | Done |

### What remains for the student

- Run the full validation pipeline on the 6 synthetic geometries to confirm
  voxel-graph CG matches TauFactor (week 2)
- Run on real CT volumes (weeks 5–8)
- Implement the EIS frequency-domain PINN using `solve_voxel_graph_helmholtz_for_tau`
  as the teacher signal (weeks 4, 8) — the teacher solver is ready; the PINN training
  loop needs a complex-valued extension for jω/D
- Implement E2 multi-geometry training: use `compute_fft_descriptor` per geometry,
  concatenate to PINN input, train one network on the library of 23 (week 10)

---

## 7. Tools and versions

| Layer | Tool | Notes |
|---|---|---|
| CT I/O | `tifffile`, `nibabel` | Match whatever CT data ships as |
| Segmentation | `scikit-image` (Otsu); optional `monai`/`torchio` U-Net | Stay simple unless Otsu fails |
| Metrics | `porespy` (ε, a_v, PSD); `taufactor` (PyTorch GPU) | Both well-maintained |
| Surfacing | `skimage.measure.marching_cubes`; `pymeshlab` | Cleanup must produce watertight |
| Geometry → PINN | `physicsnemo-sym` 25.08, `Tessellation` class | Per linked docs |
| PINN | PhysicsNeMo Sym + PyTorch | Reuse Tony's codebase |
| Tracking | `mlflow` or `wandb` | Pick one in week 1 |

---

## 8. Success criteria

| # | Criterion | Threshold |
|---|---|---|
| 1 | End-to-end run from CT file → τ_PINN in a single `make` command | Yes/No |
| 2 | \|τ_PINN − τ_TauFactor\| / τ_TauFactor on each axis, each real volume | < 5% |
| 3 | \|τ_PINN − τ_ref\| / τ_ref on synthetic volumes | < 3% |
| 4 | Frequency-domain Nyquist sweep on at least one real and one synthetic volume | 5 frequencies, monotonic, physically sensible |
| 5 | Cooper-2017 reproduction on synthetic: pairs with identical (ε, τ) but different Z(ω) identified | At least 2 such pairs |
| 6 | E1 (per-point RFF on SDF) shows lower PDE residual near walls than baseline MLP | Strict improvement at all sampled wall-distance bins |
| 7 | E2 (FFT-conditioned net on library of 23) within 10% of per-volume PINNs on τ | Per axis, per volume |
| 8 | Report has a methods section reusable verbatim in the proposal | Yes/No (supervisor judgment) |

---

## 9. Out of scope

- Inverse pore design and gradient-based microstructure optimisation (proposal Phase 4).
- 3D diffusion / VAE generative microstructure models (proposal §10.4).
- Butler–Volmer / double-layer / full multi-physics EIS — student stays on diffusion-only Helmholtz.
- Uncertainty quantification on the inverse map.
- Cross-chemistry transfer (NMC, LFP).

---

## 10. Risks and mitigations

| Risk | Mitigation |
|---|---|
| CT volume too large to fit on one GPU (180 MB–1 GB) | REV-crop to ~60 µm cube on load; never load full volume to GPU |
| Segmentation effort | Use the supplied `*_bin.tif` directly; only do Otsu/U-Net if explicitly testing differentiable segmentation |
| Synthetic generator doesn't match real CT statistics | Match (ε, τ, a_v) ranges to real-volume measurements from week 5 |
| STL not watertight after marching cubes | PyMeshLab `meshing_repair_non_manifold_edges` + `meshing_close_holes`; verify with `pymeshlab.MeshSet.print_status` |
| `Tessellation` import slow on large STL | Decimate target ≤ 500k triangles; consistent with PhysicsNeMo guidance |
| PINN doesn't converge on a geometry | Run the voxel-graph CG solver first and verify τ. If CG converges but PINN doesn't, increase teacher weights or σ in the RFF encoder. |
| Frequency-domain run unstable at high ω | Stay within ω such that l_δ(ω) > 2× mean pore size; that is the physically meaningful range |
| E1 RFF σ badly tuned | Set σ from the pore-size distribution measured in week 4; sweep σ ∈ {2, 4, 6, 10} on ellipsoid_one before committing |

---

## 11. Reading list (week 1)

1. Guruprasad & Feugmo (2026) — the 1D PINN paper this builds on (`paper.tex` in the parent dir).
2. The research proposal (`research_proposal_PINN_EIS_graphite.md`) — §§5, 7, 8, 9 are mandatory.
3. Cooper, Bertei, Finegan & Brandon (2017), *Electrochim. Acta* — baseline FD method this project
   benchmarks against.
4. PhysicsNeMo Sym docs: CSG and Tessellated module (link from supervisor).
5. Tancik et al. (2020), "Fourier features let networks learn high frequency functions" — basis for E1.
6. Li et al. (2023), "Geometry-Informed Neural Operator" — motivation for E2.
