"""
JAX-accelerated hydrophobic contact detection.
"""

import jax.numpy as jnp

from .primitives import pairwise_distances


def hydrophobic_contacts(
    ligand_coords: jnp.ndarray,
    residue_coords: jnp.ndarray,
    distance_cutoff: float = 4.5,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Detect hydrophobic contacts between ligand and residue atoms.

    A contact exists when distance <= distance_cutoff.

    Args:
        ligand_coords: (N, 3) hydrophobic ligand atom positions.
        residue_coords: (M, 3) hydrophobic residue atom positions.
        distance_cutoff: Maximum distance for contact (default 4.5 Å).

    Returns:
        contact_mask: (N, M) boolean array, True where contact exists.
        distances: (N, M) pairwise distances.
    """
    distances = pairwise_distances(ligand_coords, residue_coords)
    contact_mask = distances <= distance_cutoff
    return contact_mask, distances
