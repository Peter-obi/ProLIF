# ProLIF JAX Acceleration - Task Guide

A hands-on, test-driven guide to building JAX-accelerated interaction detection.

---

## Progress Tracker

| Stage | Files | Functions | Est. Lines | Status |
|-------|-------|-----------|------------|--------|
| 1 | 2 | 8 | ~150 | Not started |
| 2 | 7 | 14 | ~350 | Not started |
| 3 | 1 | 4 | ~100 | Not started |
| 4 | 1 | 2 | ~50 | Not started |
| **Total** | **11** | **28** | **~650** | |

---

# Stage 1: Geometric Primitives

**Goal**: Build the foundation - pure math functions that all interactions will use.

## File 1: `prolif/interactions/_jax/__init__.py`

**Purpose**: Package setup and JAX availability check.

```python
# What to write:
# - JAX_AVAILABLE constant
# - Conditional imports
# - Public API exports
```

### Task 1.1: JAX Availability Check
**Lines**: ~15

```python
# Function signature
JAX_AVAILABLE: bool

# Test case
>>> from prolif.interactions._jax import JAX_AVAILABLE
>>> isinstance(JAX_AVAILABLE, bool)
True
```

---

## File 2: `prolif/interactions/_jax/primitives.py`

**Purpose**: Core geometric calculations - the building blocks.

### Task 1.2: `pairwise_distances`
**Lines**: ~10

Compute all distances between two sets of points.

```python
# Function signature
def pairwise_distances(coords1: jnp.ndarray, coords2: jnp.ndarray) -> jnp.ndarray:
    """
    Args:
        coords1: (N, 3) array of 3D points
        coords2: (M, 3) array of 3D points
    Returns:
        (N, M) array where result[i,j] = distance between coords1[i] and coords2[j]
    """
```

```python
# Test case 1: Simple 2x2
>>> import jax.numpy as jnp
>>> coords1 = jnp.array([[0., 0., 0.], [1., 0., 0.]])
>>> coords2 = jnp.array([[0., 0., 0.], [0., 1., 0.]])
>>> result = pairwise_distances(coords1, coords2)
>>> result.shape
(2, 2)
>>> result[0, 0]  # distance from origin to origin
0.0
>>> result[0, 1]  # distance from origin to (0,1,0)
1.0
>>> result[1, 0]  # distance from (1,0,0) to origin
1.0
>>> jnp.isclose(result[1, 1], 1.414, atol=0.01)  # sqrt(2)
True

# Test case 2: Different shapes
>>> coords1 = jnp.array([[0., 0., 0.]])  # (1, 3)
>>> coords2 = jnp.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])  # (3, 3)
>>> result = pairwise_distances(coords1, coords2)
>>> result.shape
(1, 3)
>>> jnp.allclose(result, jnp.array([[1., 1., 1.]]))
True
```

**Hint**: Use broadcasting `coords1[:, None, :] - coords2[None, :, :]` then `jnp.linalg.norm(..., axis=-1)`

---

### Task 1.3: `batch_centroids`
**Lines**: ~12

Compute centroids for multiple groups of atoms.

```python
# Function signature
def batch_centroids(coords: jnp.ndarray, group_indices: list[jnp.ndarray]) -> jnp.ndarray:
    """
    Args:
        coords: (N, 3) array of all atom coordinates
        group_indices: list of K arrays, each containing indices for one group
    Returns:
        (K, 3) array of centroid positions
    """
```

```python
# Test case: Two triangles
>>> coords = jnp.array([
...     [0., 0., 0.],  # 0 - triangle 1
...     [2., 0., 0.],  # 1 - triangle 1
...     [1., 1., 0.],  # 2 - triangle 1
...     [10., 10., 10.],  # 3 - triangle 2
...     [12., 10., 10.],  # 4 - triangle 2
...     [11., 11., 10.],  # 5 - triangle 2
... ])
>>> group_indices = [jnp.array([0, 1, 2]), jnp.array([3, 4, 5])]
>>> result = batch_centroids(coords, group_indices)
>>> result.shape
(2, 3)
>>> jnp.allclose(result[0], jnp.array([1., 1/3, 0.]), atol=0.01)  # centroid of triangle 1
True
>>> jnp.allclose(result[1], jnp.array([11., 31/3, 10.]), atol=0.01)  # centroid of triangle 2
True
```

**Hint**: Use `vmap` over a function that indexes and averages, or use padding + masking.

---

### Task 1.4: `ring_normal`
**Lines**: ~15

Compute normal vector to a ring (plane defined by atoms).

```python
# Function signature
def ring_normal(coords: jnp.ndarray, ring_indices: jnp.ndarray) -> jnp.ndarray:
    """
    Args:
        coords: (N, 3) array of all atom coordinates
        ring_indices: (R,) array of indices forming the ring (R >= 3)
    Returns:
        (3,) unit normal vector to the ring plane
    """
```

```python
# Test case: Ring in XY plane (normal should be Z-axis)
>>> coords = jnp.array([
...     [1., 0., 0.],
...     [0., 1., 0.],
...     [-1., 0., 0.],
...     [0., -1., 0.],
... ])
>>> ring_indices = jnp.array([0, 1, 2, 3])
>>> result = ring_normal(coords, ring_indices)
>>> result.shape
(3,)
>>> jnp.allclose(jnp.abs(result), jnp.array([0., 0., 1.]), atol=0.01)  # +Z or -Z
True

# Test case: Ring in XZ plane (normal should be Y-axis)
>>> coords = jnp.array([
...     [1., 0., 0.],
...     [0., 0., 1.],
...     [-1., 0., 0.],
... ])
>>> ring_indices = jnp.array([0, 1, 2])
>>> result = ring_normal(coords, ring_indices)
>>> jnp.allclose(jnp.abs(result), jnp.array([0., 1., 0.]), atol=0.01)  # +Y or -Y
True
```

**Hint**: Use cross product of two vectors in the plane. For robustness, use SVD or average multiple cross products.

---

### Task 1.5: `batch_ring_normals`
**Lines**: ~8

Batch version of `ring_normal` using vmap.

```python
# Function signature
def batch_ring_normals(coords: jnp.ndarray, ring_indices_list: list[jnp.ndarray]) -> jnp.ndarray:
    """
    Args:
        coords: (N, 3) array of all atom coordinates
        ring_indices_list: list of K arrays, each containing ring atom indices
    Returns:
        (K, 3) array of unit normal vectors
    """
```

```python
# Test case: Two rings, one in XY plane, one in XZ plane
>>> # ... similar to above but with two rings
>>> result = batch_ring_normals(coords, [ring1_indices, ring2_indices])
>>> result.shape
(2, 3)
```

---

### Task 1.6: `angle_between_vectors`
**Lines**: ~8

Angle between two vectors (or batches of vectors).

```python
# Function signature
def angle_between_vectors(v1: jnp.ndarray, v2: jnp.ndarray) -> jnp.ndarray:
    """
    Args:
        v1: (..., 3) array of vectors
        v2: (..., 3) array of vectors (same batch shape as v1)
    Returns:
        (...,) array of angles in radians [0, pi]
    """
```

```python
# Test case 1: Perpendicular vectors
>>> v1 = jnp.array([1., 0., 0.])
>>> v2 = jnp.array([0., 1., 0.])
>>> result = angle_between_vectors(v1, v2)
>>> jnp.isclose(result, jnp.pi / 2, atol=0.01)  # 90 degrees
True

# Test case 2: Parallel vectors
>>> v1 = jnp.array([1., 0., 0.])
>>> v2 = jnp.array([2., 0., 0.])
>>> result = angle_between_vectors(v1, v2)
>>> jnp.isclose(result, 0., atol=0.01)  # 0 degrees
True

# Test case 3: Opposite vectors
>>> v1 = jnp.array([1., 0., 0.])
>>> v2 = jnp.array([-1., 0., 0.])
>>> result = angle_between_vectors(v1, v2)
>>> jnp.isclose(result, jnp.pi, atol=0.01)  # 180 degrees
True

# Test case 4: Batch of vectors
>>> v1 = jnp.array([[1., 0., 0.], [0., 1., 0.]])
>>> v2 = jnp.array([[0., 1., 0.], [0., 1., 0.]])
>>> result = angle_between_vectors(v1, v2)
>>> result.shape
(2,)
>>> jnp.isclose(result[0], jnp.pi / 2, atol=0.01)  # 90 degrees
True
>>> jnp.isclose(result[1], 0., atol=0.01)  # 0 degrees (same vector)
True
```

**Hint**: `arccos(dot(v1, v2) / (norm(v1) * norm(v2)))`. Use `jnp.clip` to avoid numerical issues with arccos.

---

### Task 1.7: `angle_at_vertex`
**Lines**: ~10

Angle formed at a vertex point (for H-bond angles, etc.).

```python
# Function signature
def angle_at_vertex(p1: jnp.ndarray, vertex: jnp.ndarray, p2: jnp.ndarray) -> jnp.ndarray:
    """
    Compute angle p1-vertex-p2.

    Args:
        p1: (..., 3) first point
        vertex: (..., 3) vertex point (where angle is measured)
        p2: (..., 3) second point
    Returns:
        (...,) angles in radians
    """
```

```python
# Test case: Right angle
>>> p1 = jnp.array([1., 0., 0.])
>>> vertex = jnp.array([0., 0., 0.])
>>> p2 = jnp.array([0., 1., 0.])
>>> result = angle_at_vertex(p1, vertex, p2)
>>> jnp.isclose(result, jnp.pi / 2, atol=0.01)
True

# Test case: 180 degree angle (straight line)
>>> p1 = jnp.array([-1., 0., 0.])
>>> vertex = jnp.array([0., 0., 0.])
>>> p2 = jnp.array([1., 0., 0.])
>>> result = angle_at_vertex(p1, vertex, p2)
>>> jnp.isclose(result, jnp.pi, atol=0.01)
True
```

**Hint**: Create vectors `v1 = p1 - vertex`, `v2 = p2 - vertex`, then use `angle_between_vectors`.

---

### Task 1.8: `point_to_plane_distance`
**Lines**: ~8

Distance from point(s) to a plane.

```python
# Function signature
def point_to_plane_distance(
    points: jnp.ndarray,
    plane_point: jnp.ndarray,
    plane_normal: jnp.ndarray
) -> jnp.ndarray:
    """
    Args:
        points: (N, 3) array of points
        plane_point: (3,) a point on the plane
        plane_normal: (3,) unit normal to the plane
    Returns:
        (N,) signed distances to plane
    """
```

```python
# Test case: Points above and below XY plane
>>> points = jnp.array([
...     [0., 0., 5.],   # 5 units above
...     [0., 0., -3.],  # 3 units below
...     [1., 1., 0.],   # on the plane
... ])
>>> plane_point = jnp.array([0., 0., 0.])
>>> plane_normal = jnp.array([0., 0., 1.])
>>> result = point_to_plane_distance(points, plane_point, plane_normal)
>>> jnp.allclose(result, jnp.array([5., -3., 0.]))
True
```

**Hint**: `dot(points - plane_point, plane_normal)`

---

## File 3: `tests/jax/test_primitives.py`

**Purpose**: Comprehensive tests for all primitives.

### Task 1.9: Write Test File
**Lines**: ~80

```python
# Structure
import pytest
import jax.numpy as jnp
from prolif.interactions._jax.primitives import (
    pairwise_distances,
    batch_centroids,
    ring_normal,
    batch_ring_normals,
    angle_between_vectors,
    angle_at_vertex,
    point_to_plane_distance,
)

# Include all test cases from above as pytest functions
# Add edge cases:
# - Empty arrays
# - Single point
# - Numerical stability (very small/large values)
```

---

## Stage 1 Checklist

- [ ] `prolif/interactions/_jax/__init__.py` created
- [ ] `prolif/interactions/_jax/primitives.py` created
- [ ] `pairwise_distances` implemented and tested
- [ ] `batch_centroids` implemented and tested
- [ ] `ring_normal` implemented and tested
- [ ] `batch_ring_normals` implemented and tested
- [ ] `angle_between_vectors` implemented and tested
- [ ] `angle_at_vertex` implemented and tested
- [ ] `point_to_plane_distance` implemented and tested
- [ ] `tests/jax/test_primitives.py` passes
- [ ] Benchmark vs NumPy documented

---

# Stage 2: Interaction Backends

**Goal**: Implement JAX versions of each interaction type.

## File 4: `prolif/interactions/_jax/vdw.py`

**Purpose**: Van der Waals contact detection.

### Task 2.1: `vdw_contacts`
**Lines**: ~25

```python
# Function signature
def vdw_contacts(
    ligand_coords: jnp.ndarray,
    ligand_vdw_radii: jnp.ndarray,
    residue_coords: jnp.ndarray,
    residue_vdw_radii: jnp.ndarray,
    tolerance: float = 0.0
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Args:
        ligand_coords: (N, 3) ligand atom positions
        ligand_vdw_radii: (N,) vdW radii for ligand atoms
        residue_coords: (M, 3) residue atom positions
        residue_vdw_radii: (M,) vdW radii for residue atoms
        tolerance: additional distance tolerance
    Returns:
        contact_mask: (N, M) boolean array, True where contact exists
        distances: (N, M) actual distances
    """
```

```python
# Test case: Two atoms in contact
>>> lig_coords = jnp.array([[0., 0., 0.]])
>>> lig_radii = jnp.array([1.5])  # Carbon-like
>>> res_coords = jnp.array([[2.5, 0., 0.]])
>>> res_radii = jnp.array([1.5])
>>> mask, dists = vdw_contacts(lig_coords, lig_radii, res_coords, res_radii)
>>> mask[0, 0]  # distance 2.5 < sum of radii 3.0
True
>>> dists[0, 0]
2.5

# Test case: Two atoms NOT in contact
>>> res_coords = jnp.array([[5., 0., 0.]])
>>> mask, dists = vdw_contacts(lig_coords, lig_radii, res_coords, res_radii)
>>> mask[0, 0]  # distance 5.0 > sum of radii 3.0
False
```

---

## File 5: `prolif/interactions/_jax/hydrophobic.py`

**Purpose**: Hydrophobic contact detection.

### Task 2.2: `hydrophobic_contacts`
**Lines**: ~20

```python
# Function signature
def hydrophobic_contacts(
    ligand_coords: jnp.ndarray,
    residue_coords: jnp.ndarray,
    distance_cutoff: float = 4.5
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Args:
        ligand_coords: (N, 3) hydrophobic atom positions in ligand
        residue_coords: (M, 3) hydrophobic atom positions in residue
        distance_cutoff: maximum distance for contact
    Returns:
        contact_mask: (N, M) boolean array
        distances: (N, M) distances
    """
```

```python
# Test case
>>> lig_coords = jnp.array([[0., 0., 0.]])
>>> res_coords = jnp.array([[4., 0., 0.], [5., 0., 0.]])
>>> mask, dists = hydrophobic_contacts(lig_coords, res_coords, distance_cutoff=4.5)
>>> mask[0, 0]  # 4.0 < 4.5
True
>>> mask[0, 1]  # 5.0 > 4.5
False
```

---

## File 6: `prolif/interactions/_jax/hbond.py`

**Purpose**: Hydrogen bond detection.

### Task 2.3: `hbond_check`
**Lines**: ~35

```python
# Function signature
def hbond_check(
    donor_coords: jnp.ndarray,        # (N, 3) donor heavy atom
    hydrogen_coords: jnp.ndarray,     # (N, 3) hydrogen
    acceptor_coords: jnp.ndarray,     # (M, 3) acceptor
    distance_cutoff: float = 3.5,
    angle_cutoff: float = 2.356       # 135 degrees in radians (pi * 135/180)
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Check D-H...A hydrogen bonds.
    Angle measured at H: D-H-A should be > angle_cutoff (close to 180°)

    Returns:
        hbond_mask: (N, M) boolean array
        distances: (N, M) H...A distances
        angles: (N, M) D-H-A angles
    """
```

```python
# Test case: Good hydrogen bond geometry
>>> donor = jnp.array([[0., 0., 0.]])      # D
>>> hydrogen = jnp.array([[1., 0., 0.]])   # H
>>> acceptor = jnp.array([[2.5, 0., 0.]])  # A (linear D-H...A)
>>> mask, dists, angles = hbond_check(donor, hydrogen, acceptor)
>>> mask[0, 0]  # Linear geometry, good distance
True
>>> jnp.isclose(angles[0, 0], jnp.pi, atol=0.01)  # ~180 degrees
True

# Test case: Bad angle (bent)
>>> acceptor = jnp.array([[1., 1.5, 0.]])  # A at 90 degrees
>>> mask, dists, angles = hbond_check(donor, hydrogen, acceptor)
>>> mask[0, 0]  # Angle too small
False
```

**Hint**:
1. Compute H...A distances using `pairwise_distances`
2. For each (H, A) pair, compute D-H-A angle using `angle_at_vertex`
3. Need to broadcast donor positions to match (N, M) shape

---

## File 7: `prolif/interactions/_jax/electrostatic.py`

**Purpose**: Cationic and anionic interactions.

### Task 2.4: `ionic_interactions`
**Lines**: ~20

```python
# Function signature
def ionic_interactions(
    positive_coords: jnp.ndarray,  # (N, 3) cation positions (centroids)
    negative_coords: jnp.ndarray,  # (M, 3) anion positions (centroids)
    distance_cutoff: float = 4.5
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Returns:
        interaction_mask: (N, M) boolean
        distances: (N, M) distances
    """
```

```python
# Test case
>>> pos = jnp.array([[0., 0., 0.]])  # e.g., Lys NH3+
>>> neg = jnp.array([[4., 0., 0.]])  # e.g., Asp COO-
>>> mask, dists = ionic_interactions(pos, neg, distance_cutoff=4.5)
>>> mask[0, 0]
True
```

---

## File 8: `prolif/interactions/_jax/xbond.py`

**Purpose**: Halogen bond detection.

### Task 2.5: `xbond_check`
**Lines**: ~45

```python
# Function signature
def xbond_check(
    carbon_coords: jnp.ndarray,    # (N, 3) C attached to halogen
    halogen_coords: jnp.ndarray,   # (N, 3) halogen (donor)
    acceptor_coords: jnp.ndarray,  # (M, 3) acceptor
    acceptor_neighbor_coords: jnp.ndarray,  # (M, 3) neighbor of acceptor
    distance_cutoff: float = 3.5,
    angle_c_x_a_min: float = 2.356,   # C-X...A > 135°
    angle_x_a_n_min: float = 1.571    # X...A-N > 90°
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Returns:
        xbond_mask: (N, M) boolean
        distances: (N, M) X...A distances
        angles_cxa: (N, M) C-X...A angles
        angles_xan: (N, M) X...A-N angles
    """
```

---

## File 9: `prolif/interactions/_jax/cation_pi.py`

**Purpose**: Cation-pi interactions.

### Task 2.6: `cation_pi_check`
**Lines**: ~40

```python
# Function signature
def cation_pi_check(
    cation_coords: jnp.ndarray,     # (N, 3) cation positions
    ring_centroids: jnp.ndarray,    # (M, 3) aromatic ring centroids
    ring_normals: jnp.ndarray,      # (M, 3) ring normal vectors
    distance_cutoff: float = 4.5,
    angle_cutoff: float = 0.524     # 30 degrees - cation should be above ring
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Returns:
        interaction_mask: (N, M) boolean
        distances: (N, M) cation to centroid distances
        angles: (N, M) angle between cation-centroid vector and ring normal
    """
```

```python
# Test case: Cation directly above ring
>>> cation = jnp.array([[0., 0., 4.]])  # 4Å above origin
>>> centroid = jnp.array([[0., 0., 0.]])
>>> normal = jnp.array([[0., 0., 1.]])  # ring in XY plane
>>> mask, dists, angles = cation_pi_check(cation, centroid, normal)
>>> mask[0, 0]  # directly above, good distance
True
>>> jnp.isclose(angles[0, 0], 0., atol=0.01)  # 0 degrees (aligned with normal)
True

# Test case: Cation to the side (bad geometry)
>>> cation = jnp.array([[4., 0., 0.]])  # to the side
>>> mask, dists, angles = cation_pi_check(cation, centroid, normal)
>>> mask[0, 0]  # angle is 90 degrees, too large
False
```

---

## File 10: `prolif/interactions/_jax/pi_stacking.py`

**Purpose**: Pi-stacking detection (most complex).

### Task 2.7: `pi_stacking_check`
**Lines**: ~60

```python
# Function signature
def pi_stacking_check(
    ring1_centroids: jnp.ndarray,   # (N, 3)
    ring1_normals: jnp.ndarray,     # (N, 3)
    ring2_centroids: jnp.ndarray,   # (M, 3)
    ring2_normals: jnp.ndarray,     # (M, 3)
    distance_cutoff: float = 5.5,
    face_to_face_angle_max: float = 0.698,  # 40 degrees
    edge_to_face_angle_min: float = 0.872,  # 50 degrees
    edge_to_face_angle_max: float = 1.571,  # 90 degrees
) -> dict:
    """
    Returns dict with:
        'face_to_face_mask': (N, M) boolean
        'edge_to_face_mask': (N, M) boolean
        'distances': (N, M) centroid-centroid distances
        'angles': (N, M) angles between ring normals
    """
```

```python
# Test case: Face-to-face (parallel rings)
>>> c1 = jnp.array([[0., 0., 0.]])
>>> n1 = jnp.array([[0., 0., 1.]])
>>> c2 = jnp.array([[0., 0., 3.5]])  # 3.5Å apart, stacked
>>> n2 = jnp.array([[0., 0., 1.]])   # parallel normals
>>> result = pi_stacking_check(c1, n1, c2, n2)
>>> result['face_to_face_mask'][0, 0]
True
>>> result['edge_to_face_mask'][0, 0]
False

# Test case: Edge-to-face (perpendicular rings)
>>> n2 = jnp.array([[1., 0., 0.]])  # perpendicular
>>> result = pi_stacking_check(c1, n1, c2, n2)
>>> result['face_to_face_mask'][0, 0]
False
>>> result['edge_to_face_mask'][0, 0]
True
```

---

## Stage 2 Checklist

- [ ] `prolif/interactions/_jax/vdw.py` - `vdw_contacts`
- [ ] `prolif/interactions/_jax/hydrophobic.py` - `hydrophobic_contacts`
- [ ] `prolif/interactions/_jax/hbond.py` - `hbond_check`
- [ ] `prolif/interactions/_jax/electrostatic.py` - `ionic_interactions`
- [ ] `prolif/interactions/_jax/xbond.py` - `xbond_check`
- [ ] `prolif/interactions/_jax/cation_pi.py` - `cation_pi_check`
- [ ] `prolif/interactions/_jax/pi_stacking.py` - `pi_stacking_check`
- [ ] `tests/jax/test_interactions.py` passes
- [ ] Regression tests vs existing RDKit backend

---

# Stage 3: Dispatch Layer

**Goal**: Single entry point that batches data and calls all interactions.

## File 11: `prolif/interactions/_jax/dispatch.py`

### Task 3.1: `prepare_coordinates`
**Lines**: ~25

```python
# Function signature
def prepare_coordinates(
    ligand_mol,      # RDKit Mol or ProLIF Molecule
    residue_mol,     # RDKit Mol or ProLIF Molecule
    atom_indices: dict  # {'hbond_donors': [...], 'aromatic_rings': [...], ...}
) -> dict:
    """
    Extract coordinates for all relevant atom groups.

    Returns dict with JAX arrays ready for interaction functions.
    """
```

### Task 3.2: `run_interactions_jax`
**Lines**: ~40

```python
# Function signature
@jax.jit
def run_interactions_jax(
    ligand_data: dict,
    residue_data: dict,
    interaction_types: list[str],
    parameters: dict
) -> dict:
    """
    Run all requested interactions in one JIT-compiled function.

    Returns:
        Dict mapping interaction names to results
    """
```

### Task 3.3: `convert_results`
**Lines**: ~20

```python
# Function signature
def convert_results(
    jax_results: dict,
    atom_indices: dict
) -> dict:
    """
    Convert JAX results back to format expected by ProLIF IFP.
    Maps boolean masks back to atom index tuples.
    """
```

---

## Stage 3 Checklist

- [ ] `prepare_coordinates` implemented
- [ ] `run_interactions_jax` implemented with JIT
- [ ] `convert_results` implemented
- [ ] Integration test with real molecule pair
- [ ] Benchmark full dispatch vs per-interaction calls

---

# Stage 4: Fingerprint Integration

**Goal**: Add `backend="jax"` option to existing Fingerprint class.

## Modifications to `prolif/fingerprint.py`

### Task 4.1: Add Backend Parameter
**Lines**: ~15

```python
# In Fingerprint.__init__
def __init__(
    self,
    interactions=...,
    ...,
    backend: str = "auto"  # "auto", "rdkit", "jax"
):
```

### Task 4.2: Backend Dispatch in `_run_iter`
**Lines**: ~30

```python
# Logic
if self.backend == "jax" or (self.backend == "auto" and JAX_AVAILABLE):
    # Use JAX dispatch
    results = run_interactions_jax(...)
else:
    # Use existing RDKit path
    ...
```

---

## Stage 4 Checklist

- [ ] `backend` parameter added to `__init__`
- [ ] Auto-detection logic implemented
- [ ] `_run_iter` dispatches correctly
- [ ] All existing tests pass with `backend="rdkit"`
- [ ] All existing tests pass with `backend="jax"`
- [ ] Performance comparison documented

---

# Benchmarking Template

After each stage, run and document:

```python
import time
from prolif.data import datapath

# Load test system
# ... setup code ...

# Benchmark
times_rdkit = []
times_jax = []

for _ in range(10):
    start = time.perf_counter()
    # RDKit version
    times_rdkit.append(time.perf_counter() - start)

    start = time.perf_counter()
    # JAX version
    times_jax.append(time.perf_counter() - start)

print(f"RDKit: {np.mean(times_rdkit)*1000:.2f} ± {np.std(times_rdkit)*1000:.2f} ms")
print(f"JAX:   {np.mean(times_jax)*1000:.2f} ± {np.std(times_jax)*1000:.2f} ms")
print(f"Speedup: {np.mean(times_rdkit)/np.mean(times_jax):.1f}x")
```

---

# Final Accomplishment Tracker

When complete, fill in:

| Stage | My Lines Written | Tests Passing | Speedup Achieved |
|-------|------------------|---------------|------------------|
| 1 | ___ / 150 | ___% | ___x |
| 2 | ___ / 350 | ___% | ___x |
| 3 | ___ / 100 | ___% | ___x |
| 4 | ___ / 50 | ___% | ___x |
| **Total** | ___ / 650 | | |

---

# Quick Reference: Running Tests

```bash
# Run just JAX tests
pytest tests/jax/ -v

# Run with coverage
pytest tests/jax/ --cov=prolif/interactions/_jax

# Run specific test
pytest tests/jax/test_primitives.py::test_pairwise_distances -v

# Benchmark
pytest tests/jax/ --benchmark-only
```
