"""
JAX-accelerated van der Waals contact detection.
"""

import jax.numpy as jnp

from .primitives import pairwise_distances


def vdw_contacts(
    ligand_coords: jnp.ndarray,
    ligand_vdw_radii: jnp.ndarray,
    residue_coords: jnp.ndarray,
    residue_vdw_radii: jnp.ndarray,
    tolerance: float = 0.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Detect van der Waals contacts between ligand and residue atoms.

    A contact exists when distance <= sum of vdW radii + tolerance.

    Args:
        ligand_coords: (N, 3) ligand atom positions.
        ligand_vdw_radii: (N,) vdW radii for ligand atoms.
        residue_coords: (M, 3) residue atom positions.
        residue_vdw_radii: (M,) vdW radii for residue atoms.
        tolerance: Additional distance tolerance.

    Returns:
        contact_mask: (N, M) boolean array, True where contact exists.
        distances: (N, M) pairwise distances.
    """
    distances = pairwise_distances(ligand_coords, residue_coords)
    vdw_sums = ligand_vdw_radii[:, None] + residue_vdw_radii[None, :]
    contact_mask = distances <= vdw_sums + tolerance
    return contact_mask, distances
