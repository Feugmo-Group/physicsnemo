# ct_to_physicsnemo

Pipeline for computing tortuosity (τ) from 3D X-ray CT scans of graphite battery electrodes using Physics-Informed Neural Networks (PINNs) trained via teacher distillation from a deterministic voxel-graph CG solver.

---

## Overview

Tortuosity τ quantifies how tortuous the diffusion path through the pore network is relative to free diffusion:

```
τ = ε / D_rel
```

where ε is porosity and D_rel is the relative effective diffusivity measured by solving the steady-state Laplace equation ∇²c = 0 through the pore space with Dirichlet BCs at inlet/outlet.

The pipeline has two stages:
1. **CG solver** — sparse conjugate-gradient solution of the discrete voxel-graph Laplacian (ground truth τ, deterministic, matches TauFactor)
2. **PINN surrogate** — neural network trained on the CG concentration field, enabling fast τ estimation for new geometries via a conditioned forward pass

---

## Why not solve the PDE directly with a PINN?

Mini-batch PDE residual training fails for tortuosity because:
- Global flux conservation cannot be enforced from local residual samples
- Dirichlet BCs c = ±0.5 admit a degenerate solution c(x) ≈ -0.5 + x/L (linear ramp) with zero PDE residual but τ ≈ 1 regardless of geometry
- Ghost-cell BCs and other variants break the degeneracy at the boundary but not in the interior

**Tony's solution**: solve the Laplace equation globally with the CG solver first, then train the PINN to match that solution (teacher distillation). The PINN becomes a continuous surrogate, not a PDE solver.

---

## Real CT preprocessing — no external mesh tools

The full pipeline from raw CT scan to training-ready data uses only Python/numpy/skimage/physicsnemo. No PyMeshLab, VTK, or Open3D is required.

```
.tif (raw grayscale)
  → REV crop               numpy — extract cubic sub-volume from centre
  → Otsu threshold         skimage.filters — automatic pore/solid split
  → percolation filter     scipy.ndimage — remove isolated pore clusters
  → marching cubes         skimage.measure — binary mask → triangular mesh
  → Warp BVH SDF           physicsnemo.nn.functional.signed_distance_field (GPU)
  → CG solver              sparse conjugate gradient
  → .npz                   pore-voxel arrays for streaming training
```

### Why Warp BVH replaces PyMeshLab

The original plan needed PyMeshLab to repair marching-cubes meshes to be watertight before computing SDF (conventional SDF algorithms require consistent face normals). physicsnemo's `signed_distance_field` uses NVIDIA Warp's BVH tree with **winding-number sign determination** (`use_sign_winding_number=True`), which computes the sign by counting how many times a ray crosses the mesh surface. This is robust to non-watertight meshes, so no repair step is needed.

| Step | Old plan | Current implementation |
|------|----------|----------------------|
| REV crop | numpy | numpy (same) |
| Segmentation | Otsu | `skimage.filters.threshold_otsu` |
| Mesh | Marching Cubes | `skimage.measure.marching_cubes` |
| Mesh repair | **PyMeshLab** (external) | **not needed** |
| SDF | physicsnemo-sym ComputeSDF | `physicsnemo.nn.functional.signed_distance_field` (Warp BVH) |

### SDF quality: mesh SDF vs EDT

| Property | EDT (scipy) | Mesh SDF (physicsnemo Warp) |
|----------|-------------|----------------------------|
| Resolution | voxel-grid only | sub-voxel (geometry interpolated at boundary) |
| Compute | CPU, serial | GPU BVH, ~10–100× faster for large volumes |
| Real CT ready | no — must re-implement | yes — identical path for synthetic and real data |
| Sign | unsigned (distance only) | signed (+ inside pore, – inside solid) |
| Non-watertight mesh | N/A | handled by winding number |

The `mesh_sdf.py` module implements the full conversion: `mask → marching_cubes → Warp BVH → sdf_grid`.

### Running on real CT data

```bash
python -m ct_to_physicsnemo.ct_preprocess \
    --tif-dir   /data/your_ct_scans \
    --output-dir runs/preproc_ct \
    --rev-voxels 128
```

Key options:
- `--rev-voxels N` — crop a N×N×N REV from each volume (centred; use the smallest size that is statistically representative)
- `--pore-bright` — invert Otsu polarity if pore voxels appear bright in the scan
- `--k-max 4` — FFT descriptor radius (default 4 → 729 coefficients)

---

## Module map

### Ground-truth solvers

| File | Purpose |
|------|---------|
| `voxel_graph_laplace.py` | Sparse CG solver for ∇²c = 0 on the voxel graph (primary τ method). Also includes Helmholtz solver ∇²c = k·c for EIS. |

### Geometry and metrics

| File | Purpose |
|------|---------|
| `mesh_sdf.py` | `compute_mesh_sdf_grid` — binary mask → marching cubes → physicsnemo Warp BVH → SDF grid. GPU-accelerated, sub-voxel accurate. Replaces scipy EDT. |
| `ct_preprocess.py` | Full CT preprocessing pipeline: REV crop → Otsu → percolation filter → mesh SDF → CG → .npz. No external mesh tools. |
| `geometry.py` | `PoreGeometry`: legacy per-volume geometry container (used by single-geometry E1 experiments) |
| `metrics.py` | `compute_porosity`, `compute_surface_area` (marching-cubes), `compute_fft_descriptor` (3D FFT → low-frequency magnitudes) |
| `synthetic.py` | Generates synthetic library: 8 ellipsoid + 12 GRF volumes |

### Library generation

| File | Purpose |
|------|---------|
| `generate_library.py` | Generates 20 synthetic volumes, runs CG for each, computes FFT descriptor + ε + a_v, saves `labels.csv` |
| `preprocess_volumes.py` | Converts each .tif volume (synthetic or CT) to a compact `.npz` for streaming training. Uses mesh SDF. Parallelised with `ProcessPoolExecutor`. |

### PINN models and training

| File | Purpose |
|------|---------|
| `pinn_laplace_pointcloud.py` | Single-geometry PINN with `RFFEncoder` (Random Fourier Features on x,y,z,SDF). Teacher distillation from CG field. |
| `e2_multigeometry.py` | **E2 v4** — multi-geometry FiLM-conditioned PINN. All geometries batched into one forward pass per epoch. Uses mesh SDF. |
| `e2_v5_streaming.py` | **E2 v5** — streaming version using `physicsnemo.datapipes.DataLoader`. Scales to hundreds of real CT volumes without fitting all in GPU RAM. |

### Validation and diagnostics

| File | Purpose |
|------|---------|
| `tortuosity_validation.py` | Validate CG solver against TauFactor on synthetic volumes |
| `tortuosity_pointcloud_validation.py` | Validate single-geometry PINN against CG solver |
| `compare_models.py` | Compare τ predictions across model variants |

---

## E2 experiment: multi-geometry conditioned PINN

### Problem

A single network that, given a query point (x,y,z) inside **any** pore geometry, predicts the steady-state concentration c(x,y,z) — and from that, τ.

### Architecture (v4b+): FiLM-conditioned MLP

```
Input per query point:
  RFF(x_norm, y_norm, z_norm, SDF)   →   (2*n_freqs,)  = 256 dims
    Random Fourier Features overcome spectral bias near pore walls.
    SDF = physicsnemo Warp BVH distance to nearest solid, normalised [0,1].

Conditioning vector g:
  [FFT descriptor (729) || ε (1) || a_v (1)]  =  731 dims
    FFT(729): low-frequency magnitudes of 3D pore indicator FFT (k_max=4)
    ε: porosity — strong predictor of τ
    a_v: specific surface area — encodes pore wall density

FiLM layers (Feature-wise Linear Modulation):
  Each hidden layer h is modulated:  h ← h * (1 + γ(g)) + β(g)
  γ, β are learned linear projections from g → hidden_dim.
  Initialised as identity (γ=0, β=0) so training starts from plain MLP.
  Unlike input-concatenation, FiLM forces g to influence EVERY layer —
  the network cannot ignore the geometry descriptor.

Output: scalar c ∈ [-0.5, 0.5]
```

### Training

- **Teacher signal**: MSE between predicted c and CG concentration field at randomly sampled pore voxels
- **BC loss**: MSE at inlet (c = -0.5) and outlet (c = +0.5) face voxels
- **τ-weighted geometry loss**: each geometry's loss weighted by τ_CG / mean_τ so high-τ GRF geometries (τ up to 6.8) contribute equal gradient pressure to low-τ ellipsoids (τ ≈ 1.03)
- **Adaptive EMA loss balancing**: per-geometry EMA of each loss term normalises teacher/BC ratio automatically; EMA warm-started from first actual batch to avoid instability
- **Batching**: all 20 geometries concatenated into one 40K-voxel batch per epoch — one GPU forward pass instead of 20 sequential calls

### τ estimation from trained model

TauFactor-style slice-average flux:
```
For each pair of adjacent slices (s, s+1) along the transport axis:
    flux_s = Σ_{conducting pairs} (c(s+1) - c(s)) / full_cross_section_area
D_rel = mean(flux_s) * L / ΔC
τ = ε / D_rel
```

---

## E2 v5: streaming with physicsnemo DataLoader

For real CT datasets (hundreds of volumes that don't fit in GPU RAM):

### Step 1 — Preprocess (run once)

**Synthetic library:**
```bash
python -m ct_to_physicsnemo.preprocess_volumes \
    --library-csv runs/synthetic_library/labels.csv \
    --output-dir  runs/preproc_library \
    --n-workers   4
```

**Real CT scans:**
```bash
python -m ct_to_physicsnemo.ct_preprocess \
    --tif-dir   /data/ct_scans \
    --output-dir runs/preproc_ct \
    --rev-voxels 128
```

Each volume → `volume_name.npz`:
```
pore_idx    (N_pore, 3)  int16   — voxel indices of pore voxels only
teacher     (N_pore,)    float32 — CG concentration at pore voxels
sdf         (N_pore,)    float32 — mesh SDF (Warp BVH) at pore voxels
descriptor  (731,)       float32 — [FFT(729) || ε || a_v]
inlet_mask  (N_pore,)    bool
outlet_mask (N_pore,)    bool
epsilon, tau_cg, shape, D_rel
```

Storing only pore voxels reduces file size by (1 - ε) ≈ 50–75% vs full grid.

### Step 2 — Train (streaming)
```bash
python -m ct_to_physicsnemo.e2_v5_streaming
```

```
physicsnemo.datapipes pipeline:
  NumpyReader(preproc_dir/, pin_memory=True)
      ↓
  Dataset(device="cuda", num_workers=2)   ← thread-pool prefetch
      ↓
  DataLoader(batch_size=1, shuffle=True,
             prefetch_factor=2, num_streams=4)
      ↓  CUDA streams overlap:
         training on vol i  ←→  loading vol i+1 from disk
```

Memory: only 1–2 volumes in GPU RAM at any time. Scales to 500+ volumes.

---

## Quick-start

```bash
cd taufactor/ct_extracted/src

# 1. Generate synthetic library (20 volumes, ~5 min)
python -m ct_to_physicsnemo.generate_library

# 2. Train E2 (all-in-GPU, 20 volumes, ~60 min with 50K epochs)
PYTHONUNBUFFERED=1 nohup python -u -m ct_to_physicsnemo.e2_multigeometry \
    > runs/e2_results/e2_run.log 2>&1 &

# 3. Preprocess real CT scans (once, ~10 min per volume)
python -m ct_to_physicsnemo.ct_preprocess \
    --tif-dir   /data/ct_scans \
    --output-dir runs/preproc_ct \
    --rev-voxels 128

# 4. Train E2 v5 streaming (real CT scale)
PYTHONUNBUFFERED=1 nohup python -u -m ct_to_physicsnemo.e2_v5_streaming \
    > runs/e2_results/e2_v5_run.log 2>&1 &
```

---

## Full experiment history: failures, diagnoses, and fixes

### Why plain PINNs fail for tortuosity (fundamental)

The starting point was a standard PINN approach: minimise the PDE residual ∇²c = 0 sampled at random interior points, with Dirichlet BCs c = -0.5 at inlet and c = +0.5 at outlet.

**Failure mode 1 — Linear ramp degeneracy**
The loss function has a trivially low-loss solution: c(x,y,z) = -0.5 + x/L (a linear ramp in the transport direction). This satisfies both the Laplace equation AND the BCs with zero loss, but it implies D_rel = ε (τ = 1) regardless of geometry. The PINN always converges to this degenerate solution.

**Failure mode 2 — Ghost-cell BCs**
TauFactor-style ghost-cell BCs extend the concentration field one voxel beyond the domain, breaking the linear-ramp degeneracy at the inlet/outlet faces. These were implemented and tested. The PINN produced non-trivial fields near the boundaries, but τ estimates were still inaccurate due to the third fundamental problem below.

**Failure mode 3 — Global flux cannot be enforced with mini-batches**
Even if the field is not a linear ramp, a PINN trained on random mini-batch residuals cannot enforce global flux conservation:
```
∫∫ (-∇c · n̂) dA = constant across ALL cross-sections
```
This integral constraint requires knowing the field everywhere simultaneously. Mini-batch training samples O(2000) points from a domain of O(2M) pore voxels. The gradient at slice s is statistically independent of slice s+1 in any given batch, so the optimizer has no incentive to make them equal.

**Tony's solution (teacher distillation)**
Solve globally with the sparse CG solver first. The CG solution enforces exact flux conservation because it solves the full linear system simultaneously. Then train the PINN to match the CG concentration field at sampled pore voxels. The PINN becomes a continuous interpolant of the CG solution, not a PDE solver.

---

### E2 v1 — Input-concatenated MLP, fixed weights

**Architecture**: `[RFF(x,y,z,SDF_approx) || FFT_descriptor(125)] → MLP(6 layers, 256 hidden) → c`

SDF approximation: fraction of solid 6-neighbours (1-voxel depth only, values in {0, 1/6, ..., 1}).

**Results**: 5/20 pass (< 10% error), mean error 35.1%, max 78.7%.

**Failure analysis**:
- Low-τ ellipsoids (τ ≈ 1.0–1.16) passed because the concentration field is nearly linear — easy to fit
- All GRF geometries failed (25–79% error). The network predicted τ ≈ 1.1 for geometries with true τ up to 6.8
- Root cause: the 125-dim FFT descriptor encodes the power spectrum of the geometry but has no direct τ signal. The network learned a "safe mean" field that satisfies both low-τ and high-τ samples with moderate MSE loss
- The fixed weights `teacher_weight=500` and `bc_weight=50` gave equal gradient contribution to all 20 geometries regardless of τ magnitude, so the 12 low-τ geometries dominated training

---

### E2 v2 — Proper SDF + ε/a_v conditioning + τ-weighted loss + adaptive EMA

**Changes**:
1. EDT SDF (`scipy.ndimage.distance_transform_edt`): true Euclidean distance to nearest solid, not 1-neighbour proxy
2. Augmented descriptor: `[FFT(125) || ε(1) || a_v(1)]` = 127 dims — gives network direct scalar geometry handle
3. τ-weighted geometry loss: each geometry weighted by `τ_CG / mean_τ` so high-τ GRFs get proportional gradient
4. Adaptive EMA loss balancing: `w_t = teacher_weight / EMA(loss_teacher)` normalises per-geometry teacher/BC ratio
5. Cosine annealing LR, 15K epochs

**Critical bug — adaptive EMA blowup**:
The EMA was initialised to 1.0 for all geometries. At epoch 0, actual teacher losses were O(0.1–0.5). The adaptive weight became `500 / 1.0 = 500`, inflating the effective loss 100×. The training loss jumped from 1,695 at epoch 0 to 9,230 by epoch 500 and never recovered. The loss oscillated at ~10⁴ for all 15,000 epochs.

**Results**: 6/20 pass, mean error 30.9% — marginal improvement despite correct ideas. The instability neutralised most of the benefit. One outlier: `grf_p28_s12` (τ=6.79) hit 2.2% error, proving the conditioning CAN work when the descriptor is distinctive enough.

---

### E2 v3 — FiLM conditioning + warm-started EMA

**Changes**:
1. Architecture replaced: `E2ConditionedMLP` (input-concat) → `E2FiLMMLP` (FiLM per hidden layer)

   Each hidden layer is modulated: `h ← h * (1 + γ(g)) + β(g)` where γ, β are linear projections from g. Initialised as identity so training starts stable. With input-concat, the 127-dim g competes with 256-dim RFF features at the first layer only; with FiLM, g influences every layer — the network cannot ignore it.

2. EMA warm-started: a forward pass is run for each geometry before training begins; EMA initialised to the actual first-batch loss value instead of 1.0.

**stdout buffering issue**: First launch produced 0 bytes in the log file. Python buffers stdout in full-buffering mode when redirected to a file. Fixed by relaunching with `PYTHONUNBUFFERED=1 python -u`.

**Results**: v3 was killed before completing — not enough data to report.

---

### E2 v4 — Batched forward pass across all geometries

**The key GPU utilisation problem**: v1–v3 looped over 20 geometries and did a separate forward pass for each batch of 2K voxels. That is 20 kernel launches × 2K voxels = 40K voxels per epoch, but each individual forward pass is too small to saturate the GPU. GPU utilisation was low.

**Fix**: concatenate all 20 geometry batches into one tensor `(40K, rff_dim)` and run a single forward pass per epoch. The GPU sees one large matrix multiply instead of 20 small ones.

**Additional fix — parallel loading**: SDF computation for 20 × 128³ volumes took ~10 min serially. Replaced with `ThreadPoolExecutor(max_workers=4)` — reduced to ~2 min.

**Architecture**: same FiLM-conditioned MLP as v3. descriptor_dim=127 (k_max=2, 125 FFT coeffs).

**Results**: 10/20 pass, mean error 23.2%. Remaining failures all in fine-pore GRFs (σ=3,5) where k_max=2 FFT doesn't capture enough fine-scale structure.

---

### E2 v4b — Larger FFT descriptor (k_max=4, 729 coefficients)

**Problem with v4**: GRFs with σ=3 and σ=5 (fine pores, high a_v) all fail. The FFT block at k_max=2 is a 5×5×5=125 coefficient window around the DC component — too low-resolution to distinguish fine pore structures.

**Fix**: raise k_max from 2 to 4 → 9×9×9=729 FFT coefficients. descriptor_dim=731 (729 + ε + a_v).

**Architecture change**: FiLM layers now project 731→256 instead of 127→256. Parameter count grows from 722K to 2.27M.

**Training**: 15K epochs (in progress).

---

### E2 v4c — k_max=4 + physicsnemo mesh SDF + 50K epochs (planned)

**Mesh SDF motivation**: EDT SDF is computed on the voxel grid — a pore voxel touching a solid neighbour gets SDF=1 voxel regardless of where exactly the boundary is. The physicsnemo Warp BVH SDF runs marching cubes on the binary mask to extract the exact pore-solid interface as a triangular mesh, then queries each pore-voxel centre against a GPU BVH tree. This gives sub-voxel-accurate distance to the nearest wall, which makes RFF encoding near boundaries more precise.

This also ensures the preprocessing pipeline for synthetic data and real CT data is identical — the same `compute_mesh_sdf_grid` call handles both.

**Training**: 50K epochs with cosine annealing (much longer schedule to let the larger FiLM layers converge).

---

## Key results (synthetic library, 20 volumes)

| Run | Architecture | Pass (< 10% err) | Mean err |
|-----|-------------|-----------------|---------|
| E2 v1 | Input-concat MLP, fixed weights, EDT SDF, k_max=2, 5K epochs | 5/20 | 35.1% |
| E2 v2 | + EDT SDF + ε/a_v + τ-weight + EMA (buggy init) | 6/20 | 30.9% |
| E2 v4 | FiLM + warm EMA + batched fwd, k_max=2, 15K epochs | 10/20 | 23.2% |
| E2 v4b | + k_max=4 (729 FFT coeffs), 15K epochs | in progress | — |
| E2 v4c | + mesh SDF (Warp BVH), 50K epochs | planned | — |

---

## Dependencies

```
torch >= 2.0          (vmap, batched forward)
scipy                 (sparse CG solver)
scikit-image          (marching_cubes, threshold_otsu)
tifffile              (read/write .tif volumes)
physicsnemo           (signed_distance_field Warp BVH, DataLoader)
warp >= 0.6           (BVH kernel, bundled with physicsnemo)
tensordict            (physicsnemo datapipes backend)
```
