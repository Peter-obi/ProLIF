"""
JAX-accelerated interaction detection for ProLIF.

This module provides GPU/CPU accelerated geometric calculations
for molecular interaction fingerprinting.

Usage:
    from prolif.interactions._jax import JAX_AVAILABLE

    if JAX_AVAILABLE:
        from prolif.interactions._jax import JAXAccelerator

        accel = JAXAccelerator(interactions=['hydrophobic', 'ionic'])
        results = accel.compute_interactions(ligand, protein_residues)
"""

# JAX Availability Check
try:
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

# Public API
if JAX_AVAILABLE:
    from .primitives import (
        pairwise_distances,
        batch_centroids,
        ring_normal,
        batch_ring_normals,
        angle_between_vectors,
        angle_at_vertex,
        point_to_plane_distance,
    )

    from .dispatch import (
        prepare_batch,
        run_all_interactions,
        unbatch_results,
    )

    from .accelerator import (
        JAXAccelerator,
        compute_hydrophobic_fast,
        compute_ionic_fast,
    )

    __all__ = [
        "JAX_AVAILABLE",
        # Primitives
        "pairwise_distances",
        "batch_centroids",
        "ring_normal",
        "batch_ring_normals",
        "angle_between_vectors",
        "angle_at_vertex",
        "point_to_plane_distance",
        # Dispatch
        "prepare_batch",
        "run_all_interactions",
        "unbatch_results",
        # Accelerator
        "JAXAccelerator",
        "compute_hydrophobic_fast",
        "compute_ionic_fast",
    ]
else:
    __all__ = ["JAX_AVAILABLE"]
