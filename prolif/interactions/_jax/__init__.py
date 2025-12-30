"""
JAX-accelerated interaction detection for ProLIF.

This module provides GPU/CPU accelerated geometric calculations
for molecular interaction fingerprinting.
"""

# Task 1.1: JAX Availability Check
try:
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

# Public API - uncomment as you implement
# from .primitives import (
#     pairwise_distances,
#     batch_centroids,
#     ring_normal,
#     batch_ring_normals,
#     angle_between_vectors,
#     angle_at_vertex,
#     point_to_plane_distance,
# )

__all__ = ["JAX_AVAILABLE"]
