"""
JAX-accelerated Cationic interaction detection.

Cationic: ligand cation + residue anion
Anionic: ligand anion + residue cation (inverted)
"""

import jax.numpy as jnp

from .primitives import pairwise_distances


def cationic_contacts(
    cation_coords: jnp.ndarray,
    anion_coords: jnp.ndarray,
    distance_cutoff: float = 4.5,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Detect cationic interactions (ligand cation + residue anion).

    Args:
        cation_coords: (N, 3) cation positions (ligand).
        anion_coords: (M, 3) anion positions (residue).
        distance_cutoff: Maximum distance for contact (default 4.5 Å).

    Returns:
        contact_mask: (N, M) boolean array, True where contact exists.
        distances: (N, M) pairwise distances.
    """
    distances = pairwise_distances(cation_coords, anion_coords)
    contact_mask = distances <= distance_cutoff
    return contact_mask, distances


def anionic_contacts(
    anion_coords: jnp.ndarray,
    cation_coords: jnp.ndarray,
    distance_cutoff: float = 4.5,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Detect anionic interactions (ligand anion + residue cation).

    This is the inverted form of cationic_contacts.

    Args:
        anion_coords: (N, 3) anion positions (ligand).
        cation_coords: (M, 3) cation positions (residue).
        distance_cutoff: Maximum distance for contact (default 4.5 Å).

    Returns:
        contact_mask: (N, M) boolean array, True where contact exists.
        distances: (N, M) pairwise distances.
    """
    distances = pairwise_distances(anion_coords, cation_coords)
    contact_mask = distances <= distance_cutoff
    return contact_mask, distances
