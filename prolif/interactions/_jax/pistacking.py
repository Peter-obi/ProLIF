"""
JAX-accelerated pi-stacking interaction detection.

FaceToFace: parallel pi-stacking
EdgeToFace: perpendicular pi-stacking
PiStacking: FaceToFace OR EdgeToFace
"""

import jax.numpy as jnp

from .primitives import (
    angle_between_vectors,
    batch_centroids,
    batch_centroids_masked,
    batch_ring_normals,
    batch_ring_normals_masked,
    pairwise_distances,
)


def facetoface_contacts(
    lig_ring_coords: jnp.ndarray,
    lig_ring_indices: list,
    res_ring_coords: jnp.ndarray,
    res_ring_indices: list,
    distance_cutoff: float = 5.5,
    plane_angle_min: float = 0.0,
    plane_angle_max: float = 35.0,
    normal_centroid_angle_max: float = 33.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Detect FaceToFace pi-stacking (parallel rings).

    A face-to-face interaction exists when:
        1. Distance(centroid1, centroid2) <= distance_cutoff
        2. plane_angle_min <= plane_angle <= plane_angle_max
        3. normal_to_centroid_angle <= normal_centroid_angle_max

    Args:
        lig_ring_coords: (A, 3) ligand atom coordinates.
        lig_ring_indices: List of N arrays for ligand rings.
        res_ring_coords: (B, 3) residue atom coordinates.
        res_ring_indices: List of M arrays for residue rings.
        distance_cutoff: Max centroid distance (default 5.5 Å).
        plane_angle_min/max: Plane angle range in degrees (0-35° for parallel).
        normal_centroid_angle_max: Max normal-to-centroid angle (default 33°).

    Returns:
        contact_mask: (N, M) boolean array, True where stacking exists.
        distances: (N, M) centroid distances.
        plane_angles: (N, M) angles between ring planes in degrees.
        ncc_angles: (N, M) min normal-to-centroid angles in degrees.
    """
    lig_centroids = batch_centroids(lig_ring_coords, lig_ring_indices)
    lig_normals = batch_ring_normals(lig_ring_coords, lig_ring_indices)
    res_centroids = batch_centroids(res_ring_coords, res_ring_indices)
    res_normals = batch_ring_normals(res_ring_coords, res_ring_indices)
    distances = pairwise_distances(lig_centroids, res_centroids)
    plane_angle_rad = angle_between_vectors(
        lig_normals[:, None, :], res_normals[None, :, :]
    )
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


def facetoface_contacts_masked(
    lig_ring_coords: jnp.ndarray,
    lig_ring_index_padded: jnp.ndarray,
    lig_ring_mask: jnp.ndarray,
    res_ring_coords: jnp.ndarray,
    res_ring_index_padded: jnp.ndarray,
    res_ring_mask: jnp.ndarray,
    distance_cutoff: float = 5.5,
    plane_angle_min: float = 0.0,
    plane_angle_max: float = 35.0,
    normal_centroid_angle_max: float = 33.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Detect FaceToFace stacking using padded indices and masks."""
    lig_centroids = batch_centroids_masked(lig_ring_coords, lig_ring_index_padded, lig_ring_mask)
    lig_normals = batch_ring_normals_masked(lig_ring_coords, lig_ring_index_padded, lig_ring_mask)
    res_centroids = batch_centroids_masked(res_ring_coords, res_ring_index_padded, res_ring_mask)
    res_normals = batch_ring_normals_masked(res_ring_coords, res_ring_index_padded, res_ring_mask)
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


def edgetoface_contacts(
    lig_ring_coords: jnp.ndarray,
    lig_ring_indices: list,
    res_ring_coords: jnp.ndarray,
    res_ring_indices: list,
    distance_cutoff: float = 6.5,
    plane_angle_min: float = 50.0,
    plane_angle_max: float = 90.0,
    normal_centroid_angle_max: float = 30.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Detect EdgeToFace pi-stacking (perpendicular rings).

    A edge-to-face interaction exists when:
        1. Distance(centroid1, centroid2) <= distance_cutoff
        2. plane_angle_min <= plane_angle <= plane_angle_max
        3. normal_to_centroid_angle <= normal_centroid_angle_max

    Note: ProLIF also checks intersect point, not implemented here.

    Args:
        lig_ring_coords: (A, 3) ligand atom coordinates.
        lig_ring_indices: List of N arrays for ligand rings.
        res_ring_coords: (B, 3) residue atom coordinates.
        res_ring_indices: List of M arrays for residue rings.
        distance_cutoff: Max centroid distance (default 6.5 Å).
        plane_angle_min/max: Plane angle range in degrees (50-90° for perpendicular).
        normal_centroid_angle_max: Max normal-to-centroid angle (default 30°).

    Returns:
        contact_mask: (N, M) boolean array, True where stacking exists.
        distances: (N, M) centroid distances.
        plane_angles: (N, M) angles between ring planes in degrees.
        ncc_angles: (N, M) min normal-to-centroid angles in degrees.
    """
    lig_centroids = batch_centroids(lig_ring_coords, lig_ring_indices)
    lig_normals = batch_ring_normals(lig_ring_coords, lig_ring_indices)
    res_centroids = batch_centroids(res_ring_coords, res_ring_indices)
    res_normals = batch_ring_normals(res_ring_coords, res_ring_indices)
    distances = pairwise_distances(lig_centroids, res_centroids)
    plane_angle_rad = angle_between_vectors(
        lig_normals[:, None, :], res_normals[None, :, :]
    )
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


def edgetoface_contacts_masked(
    lig_ring_coords: jnp.ndarray,
    lig_ring_index_padded: jnp.ndarray,
    lig_ring_mask: jnp.ndarray,
    res_ring_coords: jnp.ndarray,
    res_ring_index_padded: jnp.ndarray,
    res_ring_mask: jnp.ndarray,
    distance_cutoff: float = 6.5,
    plane_angle_min: float = 50.0,
    plane_angle_max: float = 90.0,
    normal_centroid_angle_max: float = 30.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Detect EdgeToFace stacking using padded indices and masks."""
    lig_centroids = batch_centroids_masked(lig_ring_coords, lig_ring_index_padded, lig_ring_mask)
    lig_normals = batch_ring_normals_masked(lig_ring_coords, lig_ring_index_padded, lig_ring_mask)
    res_centroids = batch_centroids_masked(res_ring_coords, res_ring_index_padded, res_ring_mask)
    res_normals = batch_ring_normals_masked(res_ring_coords, res_ring_index_padded, res_ring_mask)
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


def pistacking_contacts(
    lig_ring_coords: jnp.ndarray,
    lig_ring_indices: list,
    res_ring_coords: jnp.ndarray,
    res_ring_indices: list,
    ftf_kwargs: dict = None,
    etf_kwargs: dict = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Detect PiStacking interactions (FaceToFace OR EdgeToFace).

    Args:
        lig_ring_coords: (A, 3) ligand atom coordinates.
        lig_ring_indices: List of N arrays for ligand rings.
        res_ring_coords: (B, 3) residue atom coordinates.
        res_ring_indices: List of M arrays for residue rings.
        ftf_kwargs: Optional kwargs for FaceToFace.
        etf_kwargs: Optional kwargs for EdgeToFace.

    Returns:
        contact_mask: (N, M) boolean array, True where any stacking exists.
        distances: (N, M) centroid distances.
        plane_angles: (N, M) angles between ring planes.
        ncc_angles: (N, M) normal-to-centroid angles.
        stacking_type: (N, M) int array, 1=FTF, 2=ETF, 0=none.
    """
    ftf_kwargs = ftf_kwargs or {}
    etf_kwargs = etf_kwargs or {}

    ftf_mask, ftf_dist, ftf_plane, ftf_ncc = facetoface_contacts(
        lig_ring_coords, lig_ring_indices,
        res_ring_coords, res_ring_indices,
        **ftf_kwargs
    )
    etf_mask, etf_dist, etf_plane, etf_ncc = edgetoface_contacts(
        lig_ring_coords, lig_ring_indices,
        res_ring_coords, res_ring_indices,
        **etf_kwargs
    )
    contact_mask = ftf_mask | etf_mask
    distances = jnp.where(ftf_mask, ftf_dist, etf_dist)
    plane_angles = jnp.where(ftf_mask, ftf_plane, etf_plane)
    ncc_angles = jnp.where(ftf_mask, ftf_ncc, etf_ncc)
    stacking_type = jnp.where(ftf_mask, 1, jnp.where(etf_mask, 2, 0))

    return contact_mask, distances, plane_angles, ncc_angles, stacking_type


def pistacking_contacts_masked(
    lig_ring_coords: jnp.ndarray,
    lig_ring_index_padded: jnp.ndarray,
    lig_ring_mask: jnp.ndarray,
    res_ring_coords: jnp.ndarray,
    res_ring_index_padded: jnp.ndarray,
    res_ring_mask: jnp.ndarray,
    ftf_kwargs: dict = None,
    etf_kwargs: dict = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Detect PiStacking interactions using padded indices and masks."""
    ftf_kwargs = ftf_kwargs or {}
    etf_kwargs = etf_kwargs or {}
    ftf_mask, ftf_dist, ftf_plane, ftf_ncc = facetoface_contacts_masked(
        lig_ring_coords, lig_ring_index_padded, lig_ring_mask,
        res_ring_coords, res_ring_index_padded, res_ring_mask,
        **ftf_kwargs,
    )
    etf_mask, etf_dist, etf_plane, etf_ncc = edgetoface_contacts_masked(
        lig_ring_coords, lig_ring_index_padded, lig_ring_mask,
        res_ring_coords, res_ring_index_padded, res_ring_mask,
        **etf_kwargs,
    )
    contact_mask = ftf_mask | etf_mask
    distances = jnp.where(ftf_mask, ftf_dist, etf_dist)
    plane_angles = jnp.where(ftf_mask, ftf_plane, etf_plane)
    ncc_angles = jnp.where(ftf_mask, ftf_ncc, etf_ncc)
    stacking_type = jnp.where(ftf_mask, 1, jnp.where(etf_mask, 2, 0))
    return contact_mask, distances, plane_angles, ncc_angles, stacking_type
