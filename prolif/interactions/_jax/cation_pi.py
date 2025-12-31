"""
JAX-accelerated cation-pi interaction detection.
"""

import jax.numpy as jnp

from .primitives import (
    angle_between_vectors,
    batch_centroids,
    batch_ring_normals,
    pairwise_distances,
)


def cation_pi_contacts(
    cation_coords: jnp.ndarray,
    ring_coords: jnp.ndarray,
    ring_indices: list[jnp.ndarray],
    distance_cutoff: float = 4.5,
    angle_max: float = 30.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Detect cation-pi interactions between cations and aromatic rings.

    A cation-pi interaction exists when:
        1. Distance(cation, ring_centroid) <= distance_cutoff
        2. Angle(ring_normal, centroid→cation) <= angle_max
           OR angle >= (180 - angle_max) (cation can be above or below)

    Args:
        cation_coords: (N, 3) cation positions.
        ring_coords: (A, 3) all atom coordinates (rings index into this).
        ring_indices: List of K arrays, each containing atom indices for one ring.
        distance_cutoff: Max cation-centroid distance (default 4.5 Å).
        angle_max: Max angle from perpendicular in degrees (default 30°).

    Returns:
        contact_mask: (N, K) boolean array, True where cation-pi exists.
        distances: (N, K) cation-centroid distances.
        angles: (N, K) angles in degrees.
    """
    centroids = batch_centroids(ring_coords, ring_indices)
    normals = batch_ring_normals(ring_coords, ring_indices)

    distances = pairwise_distances(cation_coords, centroids)
    cation_vectors = cation_coords[:, None, :] - centroids[None, :, :]

    angles_rad = angle_between_vectors(normals[None, :, :], cation_vectors)
    angles_deg = jnp.degrees(angles_rad)
    angles_corrected = jnp.minimum(angles_deg, 180.0 - angles_deg)

    distance_ok = distances <= distance_cutoff
    angle_ok = angles_corrected <= angle_max
    contact_mask = distance_ok & angle_ok

    return contact_mask, distances, angles_corrected
