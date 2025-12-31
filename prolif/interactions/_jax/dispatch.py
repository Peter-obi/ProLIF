"""
JAX dispatch layer for batched interaction computation.

This module provides utilities to prepare batched inputs, execute a
vectorized interaction kernel, and optionally unbatch or summarize the
results. Geometry stays in JAX to minimize host transfers.
"""

import jax
import jax.numpy as jnp

from .primitives import pairwise_distances

DISTANCE_ONLY_TYPES = frozenset({
    'Hydrophobic', 'Cationic', 'Anionic',
    'VdWContact', 'MetalDonor', 'MetalAcceptor',
})


def _compute_single(
    ligand_coords: jnp.ndarray,
    res_coords: jnp.ndarray,
    mask: jnp.ndarray,
    interaction_types: tuple[str, ...],
    lig_ring_idx: jnp.ndarray | None,
    lig_ring_mask: jnp.ndarray | None,
    res_ring_idx: jnp.ndarray | None,
    res_ring_mask: jnp.ndarray | None,
):
    """Compute interactions for a single residue against the ligand.

    Returns a PyTree of arrays keyed by interaction name. The set of keys
    is stable for a given ``interaction_types`` value.
    """
    results = {}

    distances = pairwise_distances(ligand_coords, res_coords)
    results['_shared'] = {'distances': distances}

    types = set(interaction_types)

    masked_distances = jnp.where(mask[None, :], distances, jnp.inf)

    need45 = any(t in types for t in ('Hydrophobic', 'Cationic', 'Anionic'))
    need40 = 'VdWContact' in types
    need28 = any(t in types for t in ('MetalDonor', 'MetalAcceptor'))

    contact45 = masked_distances <= 4.5 if need45 else None
    contact40 = masked_distances <= 4.0 if need40 else None
    contact28 = masked_distances <= 2.8 if need28 else None

    if 'Hydrophobic' in types:
        contact_mask = contact45
        results['Hydrophobic'] = {
            'mask': contact_mask,
        }

    if 'Cationic' in types:
        contact_mask = contact45
        results['Cationic'] = {
            'mask': contact_mask,
        }

    if 'Anionic' in types:
        contact_mask = contact45
        results['Anionic'] = {
            'mask': contact_mask,
        }

    if 'MetalDonor' in types:
        contact_mask = contact28
        results['MetalDonor'] = {
            'mask': contact_mask,
        }

    if 'MetalAcceptor' in types:
        contact_mask = contact28
        results['MetalAcceptor'] = {
            'mask': contact_mask,
        }

    if 'VdWContact' in types:
        contact_mask = contact40
        results['VdWContact'] = {
            'mask': contact_mask,
        }

    if 'HBAcceptor' in types:
        from .hbacceptor import hbacceptor_contacts
        h_offset = jnp.array([0.97, 0.0, 0.0])
        hydrogen_coords = res_coords + h_offset
        contact_mask, dists, angles = hbacceptor_contacts(
            ligand_coords,
            res_coords,
            hydrogen_coords,
            distance_cutoff=3.5,
            dha_angle_min=130.0,
            dha_angle_max=180.0,
        )
        contact_mask = contact_mask & mask[None, :]
        results['HBAcceptor'] = {
            'mask': contact_mask,
            'distances': dists,
            'angles': angles,
        }

    if 'HBDonor' in types:
        from .hbacceptor import hbdonor_contacts
        h_offset = jnp.array([0.97, 0.0, 0.0])
        hydrogen_coords = ligand_coords + h_offset
        contact_mask, dists, angles = hbdonor_contacts(
            ligand_coords,
            hydrogen_coords,
            res_coords,
            distance_cutoff=3.5,
            dha_angle_min=130.0,
            dha_angle_max=180.0,
        )
        contact_mask = contact_mask & mask[None, :]
        results['HBDonor'] = {
            'mask': contact_mask,
            'distances': dists,
            'angles': angles,
        }

    if 'XBAcceptor' in types:
        from .xbacceptor import xbacceptor_contacts
        r_offset = jnp.array([-1.5, 0.0, 0.0])
        acceptor_neighbor = ligand_coords + r_offset
        d_offset = jnp.array([-1.8, 0.0, 0.0])
        donor_coords = res_coords + d_offset
        contact_mask, dists, axd_angles, xar_angles = xbacceptor_contacts(
            ligand_coords,
            acceptor_neighbor,
            res_coords,
            donor_coords,
        )
        contact_mask = contact_mask & mask[None, :]
        results['XBAcceptor'] = {
            'mask': contact_mask,
            'distances': dists,
            'axd_angles': axd_angles,
            'xar_angles': xar_angles,
        }

    if 'XBDonor' in types:
        from .xbacceptor import xbdonor_contacts
        d_offset = jnp.array([-1.8, 0.0, 0.0])
        donor_coords = ligand_coords + d_offset
        r_offset = jnp.array([-1.5, 0.0, 0.0])
        acceptor_neighbor = res_coords + r_offset
        contact_mask, dists, axd_angles, xar_angles = xbdonor_contacts(
            ligand_coords,
            donor_coords,
            res_coords,
            acceptor_neighbor,
        )
        contact_mask = contact_mask & mask[None, :]
        results['XBDonor'] = {
            'mask': contact_mask,
            'distances': dists,
            'axd_angles': axd_angles,
            'xar_angles': xar_angles,
        }

    if 'CationPi' in types and res_ring_idx is not None and res_ring_mask is not None:
        from .cationpi import cationpi_contacts_masked
        contact_mask, dists, angles = cationpi_contacts_masked(
            ligand_coords,
            res_coords,
            res_ring_idx,
            res_ring_mask,
        )
        results['CationPi'] = {
            'mask': contact_mask,
            'distances': dists,
            'angles': angles,
        }

    if 'PiCation' in types and lig_ring_idx is not None and lig_ring_mask is not None:
        from .cationpi import pication_contacts_masked
        contact_mask, dists, angles = pication_contacts_masked(
            ligand_coords,
            lig_ring_idx,
            lig_ring_mask,
            res_coords,
        )
        results['PiCation'] = {
            'mask': contact_mask,
            'distances': dists,
            'angles': angles,
        }

    if 'FaceToFace' in types and lig_ring_idx is not None and res_ring_idx is not None:
        from .pistacking import facetoface_contacts_masked
        contact_mask, dists, plane_angles, ncc_angles = facetoface_contacts_masked(
            ligand_coords,
            lig_ring_idx,
            lig_ring_mask,
            res_coords,
            res_ring_idx,
            res_ring_mask,
        )
        results['FaceToFace'] = {
            'mask': contact_mask,
            'distances': dists,
            'plane_angles': plane_angles,
            'ncc_angles': ncc_angles,
        }

    if 'EdgeToFace' in types and lig_ring_idx is not None and res_ring_idx is not None:
        from .pistacking import edgetoface_contacts_masked
        contact_mask, dists, plane_angles, ncc_angles = edgetoface_contacts_masked(
            ligand_coords,
            lig_ring_idx,
            lig_ring_mask,
            res_coords,
            res_ring_idx,
            res_ring_mask,
        )
        results['EdgeToFace'] = {
            'mask': contact_mask,
            'distances': dists,
            'plane_angles': plane_angles,
            'ncc_angles': ncc_angles,
        }

    if 'PiStacking' in types and lig_ring_idx is not None and res_ring_idx is not None:
        from .pistacking import pistacking_contacts_masked
        contact_mask, dists, plane_angles, ncc_angles, stacking_type = pistacking_contacts_masked(
            ligand_coords,
            lig_ring_idx,
            lig_ring_mask,
            res_coords,
            res_ring_idx,
            res_ring_mask,
        )
        results['PiStacking'] = {
            'mask': contact_mask,
            'distances': dists,
            'plane_angles': plane_angles,
            'ncc_angles': ncc_angles,
            'stacking_type': stacking_type,
        }

    return results


def _batched_interactions(
    ligand_coords: jnp.ndarray,
    residue_coords: jnp.ndarray,
    valid_mask: jnp.ndarray,
    interaction_types: tuple[str, ...],
    lig_ring_idx_all: jnp.ndarray | None,
    lig_ring_mask_all: jnp.ndarray | None,
    res_ring_idx_all: jnp.ndarray | None,
    res_ring_mask_all: jnp.ndarray | None,
):
    """Vectorized interaction computation across residues."""
    fn = jax.vmap(
        lambda rc, m, ri, rm: _compute_single(
            ligand_coords,
            rc,
            m,
            interaction_types,
            lig_ring_idx_all,
            lig_ring_mask_all,
            ri,
            rm,
        ),
        in_axes=(0, 0, 0, 0),
    )
    return fn(
        residue_coords,
        valid_mask,
        res_ring_idx_all if res_ring_idx_all is not None else jnp.array([[]]),
        res_ring_mask_all if res_ring_mask_all is not None else jnp.array([[]]),
    )


def _compute_single_distance_only(
    ligand_coords: jnp.ndarray,
    res_coords: jnp.ndarray,
    mask: jnp.ndarray,
    interaction_types: tuple[str, ...],
):
    """Distance-only interactions with stable shapes for JIT."""
    results = {}

    distances = pairwise_distances(ligand_coords, res_coords)
    results['_shared'] = {'distances': distances}

    masked = jnp.where(mask[None, :], distances, jnp.inf)

    if 'Hydrophobic' in interaction_types:
        results['Hydrophobic'] = {'mask': masked <= 4.5}
    if 'Cationic' in interaction_types:
        results['Cationic'] = {'mask': masked <= 4.5}
    if 'Anionic' in interaction_types:
        results['Anionic'] = {'mask': masked <= 4.5}
    if 'VdWContact' in interaction_types:
        results['VdWContact'] = {'mask': masked <= 4.0}
    if 'MetalDonor' in interaction_types:
        results['MetalDonor'] = {'mask': masked <= 2.8}
    if 'MetalAcceptor' in interaction_types:
        results['MetalAcceptor'] = {'mask': masked <= 2.8}

    return results


def _batched_distance_only(
    ligand_coords: jnp.ndarray,
    residue_coords: jnp.ndarray,
    valid_mask: jnp.ndarray,
    interaction_types: tuple[str, ...],
):
    """Vectorized distance-only variant suitable for JIT."""
    fn = jax.vmap(
        lambda rc, m: _compute_single_distance_only(
            ligand_coords, rc, m, interaction_types
        ),
        in_axes=(0, 0),
    )
    return fn(residue_coords, valid_mask)


_batched_distance_only_jitted = jax.jit(
    _batched_distance_only,
    static_argnames=('interaction_types',),
)


_batched_interactions_jitted = jax.jit(
    _batched_interactions,
    static_argnames=('interaction_types',),
)



def prepare_batch(
    ligand_coords: jnp.ndarray,
    ligand_elements: list[str],
    residue_coords_list: list[jnp.ndarray],
    residue_elements_list: list[list[str]],
    interaction_types: list[str],
    layout: dict | None = None,
    lig_ring_idx: jnp.ndarray | None = None,
    lig_ring_mask: jnp.ndarray | None = None,
    res_ring_idx: jnp.ndarray | None = None,
    res_ring_mask: jnp.ndarray | None = None,
) -> dict:
    """Prepare batched arrays for JAX computation.

    Args:
        ligand_coords: (N, 3) ligand atom coordinates.
        ligand_elements: List of N element symbols.
        residue_coords_list: List of R arrays, each (M_r, 3) for residue r.
        residue_elements_list: List of R lists of element symbols.
        interaction_types: Which interactions to compute.

    Returns:
        Dictionary with batched arrays and metadata for unbatching.
    """
    if layout is not None:
        max_atoms = layout['max_atoms']
        valid_mask = layout['valid_mask']
        original_sizes = layout['original_sizes']
    else:
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

    if layout is not None:
        padded_coords = []
        for coords in residue_coords_list:
            M = coords.shape[0]
            if M < max_atoms:
                padding = jnp.zeros((max_atoms - M, 3))
                padded = jnp.concatenate([coords, padding], axis=0)
            else:
                padded = coords
            padded_coords.append(padded)
        residue_coords = jnp.stack(padded_coords)

    return {
        'ligand_coords': ligand_coords,
        'ligand_elements': ligand_elements,
        'residue_coords': residue_coords,
        'residue_elements_list': residue_elements_list,
        'valid_mask': valid_mask,
        'original_sizes': original_sizes,
        'interaction_types': interaction_types,
        'lig_ring_idx': lig_ring_idx,
        'lig_ring_mask': lig_ring_mask,
        'res_ring_idx': res_ring_idx,
        'res_ring_mask': res_ring_mask,
    }


def run_all_interactions(batch: dict) -> dict:
    """Run all requested interactions on batched data.

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
    lig_ring_idx = batch.get('lig_ring_idx')
    lig_ring_mask = batch.get('lig_ring_mask')
    res_ring_idx = batch.get('res_ring_idx')
    res_ring_mask = batch.get('res_ring_mask')

    interaction_types_tuple = tuple(interaction_types)

    if set(interaction_types_tuple).issubset(DISTANCE_ONLY_TYPES):
        results = _batched_distance_only_jitted(
            ligand_coords,
            residue_coords,
            valid_mask,
            interaction_types_tuple,
        )
    else:
        results = _batched_interactions_jitted(
            ligand_coords,
            residue_coords,
            valid_mask,
            interaction_types_tuple,
            lig_ring_idx,
            lig_ring_mask,
            res_ring_idx,
            res_ring_mask,
        )
    return results


def summarize_batched_results(results: dict) -> dict:
    """Summarize batched results to per-residue booleans."""
    summary = {}
    for name, data in results.items():
        if 'mask' in data:
            mask = data['mask']
            has_any = jnp.any(mask, axis=(1, 2))
            summary[name] = has_any
    return summary


def unbatch_results(
    results: dict,
    batch_metadata: dict,
) -> list[dict]:
    """Convert batched results back to per-residue-pair format.

    Args:
        results: Output from run_all_interactions.
        batch_metadata: Metadata from prepare_batch for unbatching.

    Returns:
        List of R dictionaries, one per residue pair.
        Each dict maps interaction names to their results with original shapes.
    """
    results = jax.device_get(results)
    original_sizes = batch_metadata['original_sizes']
    R = len(original_sizes)

    unbatched = []
    for r in range(R):
        M = original_sizes[r]

        residue_result = {}

        shared = results.get('_shared')
        shared_dists = None
        if isinstance(shared, dict) and 'distances' in shared:
            shared_dists = shared['distances'][r][:, :M]

        for interaction_name, interaction_data in results.items():
            if interaction_name.startswith('_'):
                continue
            residue_result[interaction_name] = {}

            for key, batched_array in interaction_data.items():
                if batched_array.ndim == 3:
                    residue_result[interaction_name][key] = batched_array[r][:, :M]
                elif batched_array.ndim == 2:
                    residue_result[interaction_name][key] = batched_array[r]
                else:
                    residue_result[interaction_name][key] = batched_array[r]

            
            if 'distances' not in residue_result[interaction_name] and shared_dists is not None:
                residue_result[interaction_name]['distances'] = shared_dists

        unbatched.append(residue_result)

    return unbatched
