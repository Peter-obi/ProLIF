# ProLIF JAX Acceleration Project

## Status: ✅ COMPLETE

All 67 tests passing. Core implementation finished.

## Goal
Accelerate interaction fingerprint calculations using JAX for vectorized geometric operations with optional GPU support.

## Architecture

```
prolif/interactions/_jax/
├── __init__.py          # JAX availability check, public API
├── primitives.py        # Stage 1: Geometric primitives ✅
├── vdw.py              # Stage 2: VdW contacts ✅
├── hydrophobic.py      # Stage 2: Hydrophobic contacts ✅
├── ionic.py            # Stage 2: Ionic contacts ✅
├── hbond.py            # Stage 2: Hydrogen bonds ✅
├── xbond.py            # Stage 2: Halogen bonds ✅
├── cation_pi.py        # Stage 2: Cation-pi interactions ✅
├── pi_stacking.py      # Stage 2: Pi-stacking interactions ✅
├── dispatch.py         # Stage 3: Batch processing with vmap ✅
├── accelerator.py      # Stage 4: Integration layer ✅
└── *_guide.py          # Educational versions (gitignored)
```

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| JAX dependency | Optional (`extras_require["jax"]`) | Don't break existing users |
| API changes | None - use `backend="jax"` flag | Backwards compatible |
| Fallback | Auto-detect JAX availability | Graceful degradation |
| GPU memory | Batch by frame, not full trajectory | Memory bounded |
| SMARTS matching | Keep in RDKit | Complex chemistry, not worth reimplementing |

## CRITICAL: Match ProLIF Implementations

All JAX primitives and interactions MUST match ProLIF's existing logic, just optimized with JAX.

**Before implementing any function:**
1. Find the original in `prolif/utils.py` or `prolif/interactions/`
2. Understand its exact logic
3. Replicate it with JAX (same math, vectorized)
4. Test that outputs match within numerical tolerance

**Reference locations:**
- `prolif/utils.py` - geometric primitives (centroid, ring_normal, etc.)
- `prolif/interactions/base.py` - base interaction classes
- `prolif/interactions/interactions.py` - specific interaction implementations

## What Stays in RDKit
- `GetSubstructMatches()` - atom index identification
- Molecule parsing/conversion
- Aromaticity/ring perception

## What Moves to JAX
- All coordinate-based geometry (after atom indices known)
- Distance matrix calculations
- Angle calculations (single, double)
- Centroid/normal vector calculations
- Threshold comparisons and filtering

---

## Stage 1: Geometric Primitives ✅

**Status**: COMPLETE (24 tests passing)

**Location**: `prolif/interactions/_jax/primitives.py`

### Functions Implemented

| Function | Purpose | Key Pattern |
|----------|---------|-------------|
| `pairwise_distances` | (N,3) × (M,3) → (N,M) distances | Broadcasting: `[:, None, :] - [None, :, :]` |
| `batch_centroids` | Variable-size groups → centroids | `segment_sum` for grouping |
| `ring_normal` | Ring atoms → perpendicular vector | Cross product of centroid vectors |
| `batch_ring_normals` | Multiple rings → normals | Vectorized cross products |
| `angle_between_vectors` | Two vectors → angle (radians) | `arccos(dot / norms)` with clipping |
| `angle_at_vertex` | Three points → angle at middle | Reuses `angle_between_vectors` |
| `point_to_plane_distance` | Point to plane → signed distance | Dot product projection |

---

## Stage 2: Individual Interaction Backends ✅

**Status**: COMPLETE (23 tests passing)

**Location**: `prolif/interactions/_jax/<interaction>.py`

### Interactions Implemented

| Interaction | Checks | Function Signature |
|-------------|--------|-------------------|
| VdW | `distance <= sum_of_radii + tolerance` | `(coords, radii) → (mask, distances)` |
| Hydrophobic | `distance <= cutoff` | `(coords1, coords2) → (mask, distances)` |
| Ionic | `distance <= cutoff` | Same as hydrophobic |
| HBond | Distance + D-H-A angle | `(A, D, H) → (mask, distances, angles)` |
| XBond | Distance + A-X-D + X-A-R angles | `(A, R, X, D) → (mask, dist, axd, xar)` |
| Cation-Pi | Distance + normal-cation angle | `(cations, ring_coords, indices) → ...` |
| Pi-Stacking | Distance + plane angle + NCC angle | `(ring1, ring2) → (mask, dist, plane, ncc)` |

### Key Patterns Learned

- **Distance-only**: `pairwise_distances` + threshold comparison
- **Single angle**: Add `angle_at_vertex` with inline broadcasting
- **Ring symmetry**: `jnp.minimum(angle, 180 - angle)` for bidirectional normals
- **Either ring satisfies**: `jnp.minimum(angle1, angle2)` for pi-stacking

---

## Stage 3: Dispatch Layer ✅

**Status**: COMPLETE (8 tests passing)

**Location**: `prolif/interactions/_jax/dispatch.py`

### Functions Implemented

1. **`prepare_batch`**: Convert variable-size residues to padded arrays
   - Pad all residues to `max_atoms` with zeros
   - Create validity mask: True for real atoms, False for padding
   - Store `original_sizes` for unbatching

2. **`run_all_interactions`**: Process all residues in parallel
   - Define `compute_single(res_coords, mask)` for one residue
   - Use `jax.vmap(compute_single, in_axes=(0, 0))` to parallelize
   - Apply mask to zero out padding from results

3. **`unbatch_results`**: Restore original shapes
   - Loop through residues, slice `[:, :M]` to remove padding

### Key JAX Concepts

```python
# vmap transforms function for single item → function for batch
process_all = jax.vmap(process_one, in_axes=(0, 0))
results = process_all(batched_coords, batched_masks)  # (R, max_atoms, 3) → (R, ...)

# Padding + Masking for variable sizes
padded = jnp.concatenate([coords, jnp.zeros((max_atoms - M, 3))])
mask = jnp.concatenate([jnp.ones(M, bool), jnp.zeros(max_atoms - M, bool)])
```

---

## Stage 4: Integration Layer ✅

**Status**: COMPLETE (12 tests passing)

**Location**: `prolif/interactions/_jax/accelerator.py`

### JAXAccelerator Class

```python
from prolif.interactions._jax import JAXAccelerator

# Initialize with desired interactions
accel = JAXAccelerator(interactions=['hydrophobic', 'ionic'])

# Compute for multiple residues (batched)
results = accel.compute_interactions(ligand, [res1, res2, res3])

# Convenience methods
result = accel.compute_single(ligand, residue)
has_contact = accel.has_interaction(ligand, residue, 'hydrophobic')
```

### Convenience Functions

```python
from prolif.interactions._jax import compute_hydrophobic_fast, compute_ionic_fast
import numpy as np

# Direct numpy array input/output
mask, dists = compute_hydrophobic_fast(ligand_coords, residue_coords)
```

### Design Notes

- SMARTS matching stays in RDKit (well-optimized)
- Geometry computation moves to JAX (parallelizable)
- `extract_coords()` bridges RDKit molecules → JAX arrays
- Results convert back to numpy for compatibility

---

## Test Summary

```bash
pytest tests/jax/ -v
# 67 passed in 5.76s
```

| Test File | Tests | Status |
|-----------|-------|--------|
| `test_primitives.py` | 24 | ✅ |
| `test_interactions.py` | 23 | ✅ |
| `test_dispatch.py` | 8 | ✅ |
| `test_accelerator.py` | 12 | ✅ |

---

## Usage Examples

### Check JAX Availability
```python
from prolif.interactions._jax import JAX_AVAILABLE

if JAX_AVAILABLE:
    from prolif.interactions._jax import JAXAccelerator
```

### Direct Primitive Usage
```python
from prolif.interactions._jax import pairwise_distances, angle_between_vectors
import jax.numpy as jnp

coords1 = jnp.array([[0, 0, 0], [1, 0, 0]])
coords2 = jnp.array([[3, 0, 0], [4, 0, 0]])
distances = pairwise_distances(coords1, coords2)
```

### Full Accelerator
```python
from prolif.interactions._jax import JAXAccelerator

accel = JAXAccelerator(interactions=['hydrophobic', 'ionic'])
results = accel.compute_interactions(ligand_mol, [res1, res2, res3])
```

---

## Educational Guide Files

Each stage has a `*_guide.py` file (gitignored) with:
- Detailed docstrings explaining the problem
- Step-by-step implementation hints
- Complete working solutions

These are for learning JAX patterns, not production use.

---

## Future Work

1. **Full Fingerprint Integration**: Replace ProLIF's interaction detection loop
2. **SMARTS Pre-filtering**: Filter atoms by type before JAX computation
3. **More Interactions in Dispatch**: Add hbond, xbond, cation_pi, pi_stacking
4. **JIT Compilation**: `@jax.jit` on entire pipeline
5. **GPU Benchmarks**: Test CUDA/Metal backends

---

## Dependencies

```bash
pip install jax jaxlib  # CPU
pip install jax[cuda12]  # GPU (NVIDIA)
```

The module gracefully degrades when JAX is not installed.

---

## References

- JAX documentation: https://jax.readthedocs.io
- ProLIF paper: doi.org/10.1186/s13321-021-00548-6
- Broadcasting tutorial: https://numpy.org/doc/stable/user/basics.broadcasting.html
