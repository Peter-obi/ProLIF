"""
JAX-accelerated halogen bond detection.
"""

import jax.numpy as jnp

from .primitives import angle_at_vertex, pairwise_distances


def xbond_contacts(
    acceptor_coords: jnp.ndarray,
    acceptor_neighbor_coords: jnp.ndarray,
    halogen_coords: jnp.ndarray,
    donor_coords: jnp.ndarray,
    distance_cutoff: float = 3.5,
    axd_angle_min: float = 130.0,
    axd_angle_max: float = 180.0,
    xar_angle_min: float = 80.0,
    xar_angle_max: float = 140.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Detect halogen bonds between acceptors and halogen-donor pairs.

    A halogen bond exists when:
        1. Distance(acceptor, halogen) <= distance_cutoff
        2. axd_angle_min <= A-X-D angle <= axd_angle_max
        3. xar_angle_min <= X-A-R angle <= xar_angle_max

    Args:
        acceptor_coords: (N, 3) acceptor positions (A).
        acceptor_neighbor_coords: (N, 3) acceptor neighbor positions (R).
        halogen_coords: (M, 3) halogen positions (X).
        donor_coords: (M, 3) donor positions (D).
        distance_cutoff: Max acceptor-halogen distance (default 3.5 Å).
        axd_angle_min/max: A-X-D angle range in degrees.
        xar_angle_min/max: X-A-R angle range in degrees.

    Returns:
        contact_mask: (N, M) boolean array, True where X-bond exists.
        distances: (N, M) acceptor-halogen distances.
        axd_angles: (N, M) A-X-D angles in degrees.
        xar_angles: (N, M) X-A-R angles in degrees.
    """
    distances = pairwise_distances(acceptor_coords, halogen_coords)
    axd_angles_rad = angle_at_vertex(
        acceptor_coords[:, None, :],
        halogen_coords[None, :, :],
        donor_coords[None, :, :]
    )
    xar_angles_rad = angle_at_vertex(
        halogen_coords[None, :, :],
        acceptor_coords[:, None, :],
        acceptor_neighbor_coords[:, None, :]
    )
    axd_angles = jnp.degrees(axd_angles_rad)
    xar_angles = jnp.degrees(xar_angles_rad)
    distance_ok = distances <= distance_cutoff
    axd_ok = (axd_angles >= axd_angle_min) & (axd_angles <= axd_angle_max)
    xar_ok = (xar_angles >= xar_angle_min) & (xar_angles <= xar_angle_max)
    contact_mask = distance_ok & axd_ok & xar_ok
    return contact_mask, distances, axd_angles, xar_angles
