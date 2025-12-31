"""
JAX-accelerated interaction detection for ProLIF.

This module provides a JAX-accelerated wrapper that can be used alongside
or as a drop-in replacement for ProLIF's interaction detection.

Stage 4: Integration layer that bridges ProLIF's RDKit-based molecule
handling with JAX-accelerated geometric computations.

Usage:
    from prolif.interactions._jax import JAX_AVAILABLE
    if JAX_AVAILABLE:
        from prolif.interactions._jax.accelerator import JAXAccelerator

        accel = JAXAccelerator(interactions=['hydrophobic', 'ionic'])
        results = accel.compute_interactions(ligand, protein_residues)
"""

import jax.numpy as jnp
import numpy as np

from . import JAX_AVAILABLE
from .dispatch import prepare_batch, run_all_interactions, unbatch_results


class JAXAccelerator:
    """JAX-accelerated interaction computation.

    This class extracts coordinates from RDKit molecules and uses JAX
    for vectorized geometric computations. SMARTS matching remains in
    RDKit for pattern identification.

    Attributes:
        interaction_types: List of interaction types to compute.
        distance_cutoffs: Dict mapping interaction names to distance cutoffs.
    """

    # Default distance cutoffs matching ProLIF defaults
    DEFAULT_CUTOFFS = {
        'hydrophobic': 4.5,
        'ionic': 4.5,
        'vdw': 4.0,
        'hbond': 3.5,
        'xbond': 3.5,
        'cation_pi': 4.5,
        'pi_stacking': 5.5,
    }

    def __init__(
        self,
        interactions: list[str] | None = None,
        distance_cutoffs: dict[str, float] | None = None,
    ):
        """Initialize the JAX accelerator.

        Args:
            interactions: List of interaction types to compute.
                Defaults to ['hydrophobic', 'ionic'].
            distance_cutoffs: Optional dict overriding default cutoffs.
        """
        if not JAX_AVAILABLE:
            raise ImportError(
                "JAX is not available. Install with: pip install jax jaxlib"
            )

        self.interaction_types = interactions or ['hydrophobic', 'ionic']
        self.distance_cutoffs = {**self.DEFAULT_CUTOFFS}
        if distance_cutoffs:
            self.distance_cutoffs.update(distance_cutoffs)

    def extract_coords(self, mol) -> tuple[jnp.ndarray, list[str]]:
        """Extract coordinates and elements from an RDKit molecule.

        Args:
            mol: RDKit molecule or ProLIF Residue with .xyz property.

        Returns:
            Tuple of (coords array (N, 3), element symbols list).
        """
        # Handle ProLIF Residue/Molecule (has .xyz property)
        if hasattr(mol, 'xyz'):
            coords = jnp.array(mol.xyz)
        else:
            # Fall back to RDKit conformer
            conf = mol.GetConformer()
            coords = jnp.array([
                [conf.GetAtomPosition(i).x,
                 conf.GetAtomPosition(i).y,
                 conf.GetAtomPosition(i).z]
                for i in range(mol.GetNumAtoms())
            ])

        elements = [mol.GetAtomWithIdx(i).GetSymbol()
                   for i in range(mol.GetNumAtoms())]

        return coords, elements

    def compute_interactions(
        self,
        ligand,
        residues: list,
    ) -> list[dict]:
        """Compute interactions between ligand and multiple residues.

        This is the main entry point for JAX-accelerated computation.

        Args:
            ligand: RDKit molecule or ProLIF Residue for the ligand.
            residues: List of RDKit molecules or ProLIF Residues for
                protein residues to check.

        Returns:
            List of result dicts, one per residue. Each dict maps
            interaction names to their results (mask, distances).
        """
        # Extract ligand coordinates
        ligand_coords, ligand_elements = self.extract_coords(ligand)

        # Extract residue coordinates
        residue_coords_list = []
        residue_elements_list = []
        for res in residues:
            coords, elements = self.extract_coords(res)
            residue_coords_list.append(coords)
            residue_elements_list.append(elements)

        # Prepare batched arrays
        batch = prepare_batch(
            ligand_coords,
            ligand_elements,
            residue_coords_list,
            residue_elements_list,
            self.interaction_types,
        )

        # Run JAX-accelerated computation
        results = run_all_interactions(batch)

        # Unbatch results
        unbatched = unbatch_results(results, batch)

        return unbatched

    def compute_single(
        self,
        ligand,
        residue,
    ) -> dict:
        """Compute interactions between ligand and a single residue.

        Convenience method for single-residue computation.

        Args:
            ligand: RDKit molecule or ProLIF Residue.
            residue: Single protein residue.

        Returns:
            Dict mapping interaction names to results.
        """
        results = self.compute_interactions(ligand, [residue])
        return results[0] if results else {}

    def has_interaction(
        self,
        ligand,
        residue,
        interaction_type: str,
    ) -> bool:
        """Check if a specific interaction exists.

        Args:
            ligand: RDKit molecule or ProLIF Residue.
            residue: Single protein residue.
            interaction_type: Name of interaction to check.

        Returns:
            True if at least one atom pair satisfies the interaction.
        """
        if interaction_type not in self.interaction_types:
            raise ValueError(
                f"Interaction '{interaction_type}' not in configured types: "
                f"{self.interaction_types}"
            )

        result = self.compute_single(ligand, residue)
        if interaction_type in result:
            return bool(result[interaction_type]['mask'].any())
        return False


def compute_hydrophobic_fast(
    ligand_coords: np.ndarray,
    residue_coords: np.ndarray,
    distance_cutoff: float = 4.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Fast hydrophobic contact detection using JAX.

    Convenience function for direct numpy array input.

    Args:
        ligand_coords: (N, 3) numpy array of ligand coordinates.
        residue_coords: (M, 3) numpy array of residue coordinates.
        distance_cutoff: Maximum distance for contact.

    Returns:
        Tuple of (contact_mask (N, M), distances (N, M)) as numpy arrays.
    """
    from .hydrophobic import hydrophobic_contacts

    lig = jnp.array(ligand_coords)
    res = jnp.array(residue_coords)

    mask, dists = hydrophobic_contacts(lig, res, distance_cutoff)

    return np.array(mask), np.array(dists)


def compute_ionic_fast(
    cation_coords: np.ndarray,
    anion_coords: np.ndarray,
    distance_cutoff: float = 4.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Fast ionic contact detection using JAX.

    Convenience function for direct numpy array input.

    Args:
        cation_coords: (N, 3) numpy array of cation coordinates.
        anion_coords: (M, 3) numpy array of anion coordinates.
        distance_cutoff: Maximum distance for contact.

    Returns:
        Tuple of (contact_mask (N, M), distances (N, M)) as numpy arrays.
    """
    from .ionic import ionic_contacts

    cat = jnp.array(cation_coords)
    ani = jnp.array(anion_coords)

    mask, dists = ionic_contacts(cat, ani, distance_cutoff)

    return np.array(mask), np.array(dists)
