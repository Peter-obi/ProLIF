"""
JAX dispatch layer for batched interaction computation.

This module coordinates:
1. Batching coordinates across residue pairs
2. Running all interaction checks in a single JIT-compiled function
3. Unbatching results back to per-residue-pair format
"""

import jax
import jax.numpy as jnp


def prepare_batch(
    ligand_coords: jnp.ndarray,
    ligand_elements: list[str],
    residue_coords_list: list[jnp.ndarray],
    residue_elements_list: list[list[str]],
    interaction_types: list[str],
) -> dict:
    """Prepare batched arrays for JAX computation.

    Collects coordinates and metadata from multiple residue pairs into
    padded arrays suitable for vectorized computation.

    Args:
        ligand_coords: (N, 3) ligand atom coordinates.
        ligand_elements: List of N element symbols.
        residue_coords_list: List of R arrays, each (M_r, 3) for residue r.
        residue_elements_list: List of R lists of element symbols.
        interaction_types: Which interactions to compute.

    Returns:
        Dictionary with batched arrays and metadata for unbatching.
    """
    max_atoms = max(coords.shape[0] for coords in residue_coords_list)
    padded_coords = []
    masks = []
    for coords in residue_coords_list:
        M = coords.shape[0]
        if M < max_atoms:
            padding = jnp.zeros((max_atoms - M, 3))
            padded = jnp.concatenate([coords, padding], axis=0)
        else:
            padded = coords
        padded_coords.append(padded)

        real_mask = jnp.ones(M, dtype=bool)
        pad_mask = jnp.zeros(max_atoms - M, dtype=bool)
        mask = jnp.concatenate([real_mask, pad_mask])
        masks.append(mask)

    residue_coords = jnp.stack(padded_coords)
    valid_mask = jnp.stack(masks)

    original_sizes = [coords.shape[0] for coords in residue_coords_list]

    return {
        'ligand_coords': ligand_coords,
        'residue_coords': residue_coords,
        'valid_mask': valid_mask,
        'original_sizes': original_sizes,
        'interaction_types': interaction_types,
    }


def run_all_interactions(batch: dict) -> dict:
    """Run all requested interactions on batched data.

    Uses jax.vmap to vectorize computation across all residues in parallel.

    Args:
        batch: Output from prepare_batch.

    Returns:
        Dictionary mapping interaction names to result arrays.
        Each array has shape (R, ...) where R is number of residues.
    """
    ligand_coords = batch['ligand_coords']
    residue_coords = batch['residue_coords']
    valid_mask = batch['valid_mask']
    interaction_types = batch['interaction_types']

    def compute_single(res_coords, mask):
        """Compute interactions for a single residue.

        Args:
            res_coords: (max_atoms, 3) coordinates for one residue
            mask: (max_atoms,) boolean mask for valid atoms

        Returns:
            Dict of interaction results for this residue
        """
        results = {}

        if 'hydrophobic' in interaction_types:
            from .hydrophobic import hydrophobic_contacts
            contact_mask, distances = hydrophobic_contacts(ligand_coords, res_coords)
            # Mask out padded atoms
            contact_mask = contact_mask & mask[None, :]
            results['hydrophobic'] = {
                'mask': contact_mask,
                'distances': distances,
            }

        if 'vdw' in interaction_types:
            # VdW requires radii - for now use simple distance check
            from .hydrophobic import hydrophobic_contacts
            contact_mask, distances = hydrophobic_contacts(
                ligand_coords, res_coords, distance_cutoff=4.0
            )
            contact_mask = contact_mask & mask[None, :]
            results['vdw'] = {
                'mask': contact_mask,
                'distances': distances,
            }

        if 'ionic' in interaction_types:
            from .ionic import ionic_contacts
            contact_mask, distances = ionic_contacts(ligand_coords, res_coords)
            contact_mask = contact_mask & mask[None, :]
            results['ionic'] = {
                'mask': contact_mask,
                'distances': distances,
            }

        return results

    # Apply vmap to process all residues in parallel
    batched_fn = jax.vmap(compute_single, in_axes=(0, 0))
    results = batched_fn(residue_coords, valid_mask)

    return results


def unbatch_results(
    results: dict,
    batch_metadata: dict,
) -> list[dict]:
    """Convert batched results back to per-residue-pair format.

    Slices the padded batch arrays to recover original shapes for each residue.

    Args:
        results: Output from run_all_interactions.
        batch_metadata: Metadata from prepare_batch for unbatching.

    Returns:
        List of R dictionaries, one per residue pair.
        Each dict maps interaction names to their results with original shapes.
    """
    original_sizes = batch_metadata['original_sizes']
    R = len(original_sizes)

    unbatched = []
    for r in range(R):
        M = original_sizes[r]  # actual atom count for this residue

        residue_result = {}

        for interaction_name, interaction_data in results.items():
            residue_result[interaction_name] = {}

            for key, batched_array in interaction_data.items():
                # batched_array shape: (R, N_lig, max_atoms)
                # Slice to get (N_lig, M) for this residue
                residue_result[interaction_name][key] = batched_array[r][:, :M]

        unbatched.append(residue_result)

    return unbatched
