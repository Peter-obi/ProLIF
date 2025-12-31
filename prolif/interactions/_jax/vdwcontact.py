"""
JAX-accelerated van der Waals contact detection.

VdWContact: distance <= sum_of_vdw_radii + tolerance
"""

import jax.numpy as jnp

from .primitives import pairwise_distances


VDW_RADII_MDANALYSIS = {
    'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'F': 1.47,
    'P': 1.80, 'S': 1.80, 'Cl': 1.75, 'Br': 1.85, 'I': 1.98,
}

VDW_RADII_RDKIT = {
    'H': 1.20, 'He': 1.40, 'Li': 1.82, 'Be': 1.53, 'B': 1.92,
    'C': 1.70, 'N': 1.55, 'O': 1.52, 'F': 1.47, 'Ne': 1.54,
    'Na': 2.27, 'Mg': 1.73, 'Al': 1.84, 'Si': 2.10, 'P': 1.80,
    'S': 1.80, 'Cl': 1.75, 'Ar': 1.88, 'K': 2.75, 'Ca': 2.31,
    'Br': 1.85, 'I': 1.98, 'Zn': 1.39, 'Fe': 1.40, 'Cu': 1.40,
}


def vdwcontact_contacts(
    lig_coords: jnp.ndarray,
    lig_elements: list,
    res_coords: jnp.ndarray,
    res_elements: list,
    tolerance: float = 0.0,
    vdw_radii: dict = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Detect VdW contacts based on atomic radii.

    A contact exists when:
        distance <= vdw_radius_1 + vdw_radius_2 + tolerance

    Args:
        lig_coords: (N, 3) ligand atom coordinates.
        lig_elements: List of N element symbols.
        res_coords: (M, 3) residue atom coordinates.
        res_elements: List of M element symbols.
        tolerance: Added to sum of radii (default 0.0 Å).
        vdw_radii: Dict of element -> radius. Defaults to MDAnalysis values.

    Returns:
        contact_mask: (N, M) boolean array, True where contact exists.
        distances: (N, M) pairwise distances.
        radii_sums: (N, M) sum of VdW radii for each pair.
    """
    if vdw_radii is None:
        vdw_radii = VDW_RADII_MDANALYSIS

    lig_radii = jnp.array([vdw_radii.get(e, 1.70) for e in lig_elements])
    res_radii = jnp.array([vdw_radii.get(e, 1.70) for e in res_elements])

    radii_sums = lig_radii[:, None] + res_radii[None, :]

    distances = pairwise_distances(lig_coords, res_coords)

    contact_mask = distances <= (radii_sums + tolerance)

    return contact_mask, distances, radii_sums


def vdwcontact_contacts_precomputed(
    lig_coords: jnp.ndarray,
    lig_radii: jnp.ndarray,
    res_coords: jnp.ndarray,
    res_radii: jnp.ndarray,
    tolerance: float = 0.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Detect VdW contacts with precomputed radii arrays.

    Use this version when radii are already converted to arrays,
    avoiding Python list operations inside JIT.

    Args:
        lig_coords: (N, 3) ligand atom coordinates.
        lig_radii: (N,) ligand VdW radii.
        res_coords: (M, 3) residue atom coordinates.
        res_radii: (M,) residue VdW radii.
        tolerance: Added to sum of radii (default 0.0 Å).

    Returns:
        contact_mask: (N, M) boolean array, True where contact exists.
        distances: (N, M) pairwise distances.
        radii_sums: (N, M) sum of VdW radii for each pair.
    """
    radii_sums = lig_radii[:, None] + res_radii[None, :]
    distances = pairwise_distances(lig_coords, res_coords)
    contact_mask = distances <= (radii_sums + tolerance)

    return contact_mask, distances, radii_sums
