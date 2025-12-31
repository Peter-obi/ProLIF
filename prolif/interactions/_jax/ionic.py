"""
JAX-accelerated ionic (electrostatic) interaction detection.
"""

import jax.numpy as jnp

from .primitives import pairwise_distances


def ionic_contacts(
    cation_coords: jnp.ndarray,
    anion_coords: jnp.ndarray,
    distance_cutoff: float = 4.5,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Detect ionic interactions between cations and anions.

    A contact exists when distance <= distance_cutoff.

    Args:
        cation_coords: (N, 3) cation positions.
        anion_coords: (M, 3) anion positions.
        distance_cutoff: Maximum distance for contact (default 4.5 Å).

    Returns:
        contact_mask: (N, M) boolean array, True where ionic contact exists.
        distances: (N, M) pairwise distances.
    """
    distances = pairwise_distances(cation_coords, anion_coords)
    contact_mask = distances <= distance_cutoff
    return contact_mask, distances
