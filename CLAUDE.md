# ProLIF JAX Acceleration Project

## Status: Direction Change — Simplify JAX Path

All tests currently pass, but we’re simplifying the JAX layer to remove low‑ROI residue‑batched complexity. The new focus is:
- Keep simple, correct integration that uses JAX for distances only.
- Remove residue‑batched dispatch + index threading.
- Target future MD use via frame‑batching (fixed indices; vmap over frames).

## Goal
Accelerate interaction fingerprint calculations using JAX for vectorized geometric operations with optional GPU support.

## Architecture (after simplification)

```
prolif/interactions/_jax/
├── __init__.py          # JAX availability check, public API
├── primitives.py        # Geometric primitives (keep)
├── integration.py       # Simple path using JAX for distances (keep)
├── accelerator.py       # Thin wrapper, routes to simple path (slim)
└── (remove gradually)
    ├── dispatch.py      # Residue-batched vmap + padding (remove)
    └── per-interaction angle kernels used only by dispatch (review/trim)
```

## Design Decisions (updated)

- Keep SMARTS and chemistry in RDKit; do not duplicate with “prefilter”.
- Use JAX where it clearly helps without extra plumbing: pairwise distances.
- Remove residue‑batched dispatch: padding + index threading added complexity for ~1x CPU.
- Target MD: indices fixed across frames ⇒ vmap over frames; no padding/index juggling.

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
- Distance matrix calculations (pairwise distances).
- Optionally basic vector math reused by integration.
  - Angle checks remain in the simple integration path (Python control), unless we do frame‑batched MD later.

---

## Geometric Primitives (keep)

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

## Removed: Residue-batched Dispatch Layer

- Complex and brittle (padding, per‑residue index trees, vmap argument ordering).
- No clear CPU benefit; overhead cancels geometry gains.
- Will be replaced by a future frame‑batched MD path (simple vmap over frames).

---

## Integration Layer (keep, simplified)

- Simple path: RDKit SMARTS + Python control; JAX for distances only.
- Remove JAXAccelerator and residue-batched dispatch; keep only integration helpers.

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

1. Frame‑batched MD path: SMARTS once (structure), vmap across frames (coords).
2. Optional GPU: only when F×R×N×M is large enough to amortize transfer/launch.
3. Keep code surface area small; no re‑adding residue‑batched dispatch.

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
