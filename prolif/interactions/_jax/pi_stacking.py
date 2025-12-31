"""
JAX-accelerated pi-stacking interaction detection.
"""

import jax.numpy as jnp

from .primitives import (
    angle_between_vectors,
    batch_centroids,
    batch_ring_normals,
    pairwise_distances,
)


def pi_stacking_contacts(
    lig_ring_coords: jnp.ndarray,
    lig_ring_indices: list[jnp.ndarray],
    res_ring_coords: jnp.ndarray,
    res_ring_indices: list[jnp.ndarray],
    distance_cutoff: float = 5.5,
    plane_angle_min: float = 0.0,
    plane_angle_max: float = 35.0,
    normal_centroid_angle_max: float = 33.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Detect pi-stacking interactions between aromatic rings.

    A pi-stacking interaction exists when:
        1. Distance(centroid1, centroid2) <= distance_cutoff
        2. plane_angle_min <= plane_angle <= plane_angle_max
        3. normal_to_centroid_angle <= normal_centroid_angle_max (for either ring)

    Args:
        lig_ring_coords: (A, 3) ligand atom coordinates.
        lig_ring_indices: List of N arrays for ligand rings.
        res_ring_coords: (B, 3) residue atom coordinates.
        res_ring_indices: List of M arrays for residue rings.
        distance_cutoff: Max centroid distance (default 5.5 Å for FTF).
        plane_angle_min/max: Plane angle range in degrees.
        normal_centroid_angle_max: Max normal-to-centroid angle.

    Returns:
        contact_mask: (N, M) boolean array, True where stacking exists.
        distances: (N, M) centroid distances.
        plane_angles: (N, M) angles between ring planes in degrees.
        normal_centroid_angles: (N, M) min normal-to-centroid angles in degrees.
    """
    lig_centroids = batch_centroids(lig_ring_coords, lig_ring_indices)
    lig_normals = batch_ring_normals(lig_ring_coords, lig_ring_indices)
    res_centroids = batch_centroids(res_ring_coords, res_ring_indices)
    res_normals = batch_ring_normals(res_ring_coords, res_ring_indices)

    distances = pairwise_distances(lig_centroids, res_centroids)
    plane_angle_rad = angle_between_vectors(lig_normals[:, None, :], res_normals[None, :, :])
    plane_angles = jnp.degrees(plane_angle_rad)
    plane_angles = jnp.minimum(plane_angles, 180.0 - plane_angles)

    c1_to_c2 = res_centroids[None, :, :] - lig_centroids[:, None, :]
    c2_to_c1 = -c1_to_c2

    n1_c1c2_rad = angle_between_vectors(lig_normals[:, None, :], c1_to_c2)
    n1_c1c2 = jnp.degrees(n1_c1c2_rad)
    n1_c1c2 = jnp.minimum(n1_c1c2, 180.0 - n1_c1c2)

    n2_c2c1_rad = angle_between_vectors(res_normals[None, :, :], c2_to_c1)
    n2_c2c1 = jnp.degrees(n2_c2c1_rad)
    n2_c2c1 = jnp.minimum(n2_c2c1, 180.0 - n2_c2c1)

    ncc_angles = jnp.minimum(n1_c1c2, n2_c2c1)

    distance_ok = distances <= distance_cutoff
    plane_ok = (plane_angles >= plane_angle_min) & (plane_angles <= plane_angle_max)
    ncc_ok = ncc_angles <= normal_centroid_angle_max
    contact_mask = distance_ok & plane_ok & ncc_ok

    return contact_mask, distances, plane_angles, ncc_angles
