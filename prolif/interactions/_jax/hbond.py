"""
JAX-accelerated hydrogen bond detection.
"""

import jax.numpy as jnp

from .primitives import angle_at_vertex, pairwise_distances


def hbond_contacts(
    acceptor_coords: jnp.ndarray,
    donor_coords: jnp.ndarray,
    hydrogen_coords: jnp.ndarray,
    distance_cutoff: float = 3.5,
    angle_min: float = 130.0,
    angle_max: float = 180.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Detect hydrogen bonds between acceptors and donor-hydrogen pairs.

    A hydrogen bond exists when:
        1. Distance(acceptor, donor) <= distance_cutoff
        2. angle_min <= D-H-A angle <= angle_max

    Args:
        acceptor_coords: (N, 3) acceptor atom positions.
        donor_coords: (M, 3) donor atom positions.
        hydrogen_coords: (M, 3) hydrogen positions (hydrogen[i] bonded to donor[i]).
        distance_cutoff: Max acceptor-donor distance (default 3.5 Å).
        angle_min: Minimum D-H-A angle in degrees (default 130°).
        angle_max: Maximum D-H-A angle in degrees (default 180°).

    Returns:
        contact_mask: (N, M) boolean array, True where H-bond exists.
        distances: (N, M) acceptor-donor distances.
        angles: (N, M) D-H-A angles in degrees.
    """
    distances = pairwise_distances(acceptor_coords, donor_coords)
    angles_rad = angle_at_vertex(
        donor_coords[None, :, :],
        hydrogen_coords[None, :, :],
        acceptor_coords[:, None, :]
    )
    angles_deg = jnp.degrees(angles_rad)
    distance_ok = distances <= distance_cutoff
    angle_ok = (angles_deg >= angle_min) & (angles_deg <= angle_max)
    contact_mask = distance_ok & angle_ok
    return contact_mask, distances, angles_deg
