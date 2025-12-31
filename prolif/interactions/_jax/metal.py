"""
JAX-accelerated metal complexation interaction detection.

MetalDonor: ligand metal + residue chelated atom
MetalAcceptor: ligand chelated atom + residue metal (inverted)
"""

import jax.numpy as jnp

from .primitives import pairwise_distances


def metaldonor_contacts(
    metal_coords: jnp.ndarray,
    ligand_coords: jnp.ndarray,
    distance_cutoff: float = 2.8,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Detect MetalDonor interactions (ligand metal + residue chelated).

    Args:
        metal_coords: (N, 3) metal positions (ligand).
        ligand_coords: (M, 3) chelating atom positions (residue).
        distance_cutoff: Maximum distance for contact (default 2.8 Å).

    Returns:
        contact_mask: (N, M) boolean array, True where contact exists.
        distances: (N, M) pairwise distances.
    """
    distances = pairwise_distances(metal_coords, ligand_coords)
    contact_mask = distances <= distance_cutoff
    return contact_mask, distances


def metalacceptor_contacts(
    ligand_coords: jnp.ndarray,
    metal_coords: jnp.ndarray,
    distance_cutoff: float = 2.8,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Detect MetalAcceptor interactions (ligand chelated + residue metal).

    This is the inverted form of metaldonor_contacts.

    Args:
        ligand_coords: (N, 3) chelating atom positions (ligand).
        metal_coords: (M, 3) metal positions (residue).
        distance_cutoff: Maximum distance for contact (default 2.8 Å).

    Returns:
        contact_mask: (N, M) boolean array, True where contact exists.
        distances: (N, M) pairwise distances.
    """
    distances = pairwise_distances(ligand_coords, metal_coords)
    contact_mask = distances <= distance_cutoff
    return contact_mask, distances
