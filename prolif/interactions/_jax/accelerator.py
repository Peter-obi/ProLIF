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

        accel = JAXAccelerator(interactions=['Hydrophobic', 'Cationic'])
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

    DEFAULT_CUTOFFS = {
        'Hydrophobic': 4.5,
        'Cationic': 4.5,
        'Anionic': 4.5,
        'VdWContact': 4.0,
        'HBAcceptor': 3.5,
        'HBDonor': 3.5,
        'XBAcceptor': 3.5,
        'XBDonor': 3.5,
        'CationPi': 4.5,
        'PiCation': 4.5,
        'FaceToFace': 5.5,
        'EdgeToFace': 6.5,
        'PiStacking': 6.5,
        'MetalDonor': 2.8,
        'MetalAcceptor': 2.8,
    }

    def __init__(
        self,
        interactions: list[str] | None = None,
        distance_cutoffs: dict[str, float] | None = None,
        prefilter_smarts: bool = False,
        gate_angles_by_distance: bool = False,
    ):
        """Initialize the JAX accelerator.

        Args:
            interactions: List of interaction types to compute.
                Defaults to ['Hydrophobic', 'Cationic'].
            distance_cutoffs: Optional dict overriding default cutoffs.
        """
        if not JAX_AVAILABLE:
            raise ImportError(
                "JAX is not available. Install with: pip install jax jaxlib"
            )

        self.interaction_types = interactions or ['Hydrophobic', 'Cationic']
        self.distance_cutoffs = {**self.DEFAULT_CUTOFFS}
        if distance_cutoffs:
            self.distance_cutoffs.update(distance_cutoffs)
        self._layout_cache: dict[tuple, dict] = {}
        self._ring_cache: dict[int, list[list[int]]] = {}
        self.prefilter_smarts = prefilter_smarts
        self.gate_angles_by_distance = gate_angles_by_distance

    def extract_coords(self, mol) -> tuple[jnp.ndarray, list[str]]:
        """Extract coordinates and elements from an RDKit molecule.

        Args:
            mol: RDKit molecule or ProLIF Residue with .xyz property.

        Returns:
            Tuple of (coords array (N, 3), element symbols list).
        """
        if hasattr(mol, 'xyz'):
            coords = jnp.array(mol.xyz)
        else:
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

    def _layout_key(self, ligand_coords, residue_coords_list) -> tuple:
        """Compute a cache key based on atom counts for ligand and residues.

        The key encodes only sizes. Coordinates or elements changes will
        invalidate the cache when sizes differ.
        """
        lig_n = int(ligand_coords.shape[0])
        res_sizes = tuple(int(c.shape[0]) for c in residue_coords_list)
        return (lig_n, res_sizes)

    def _to_rdmol(self, mol):
        """Return an RDKit Mol from supported molecule wrappers."""
        if hasattr(mol, 'GetRingInfo') and hasattr(mol, 'GetAtomWithIdx'):
            return mol
        if hasattr(mol, 'rdmol'):
            return mol.rdmol
        if hasattr(mol, 'to_rdkit'):
            return mol.to_rdkit()
        if hasattr(mol, 'ToRDKit'):
            return mol.ToRDKit()
        return None

    def _smarts_prefilter_masks(self, ligand, residues, interaction_types, max_atoms):
        """Return per-interaction candidate masks for ligand and residues.

        Ligand mask shape: (T, N)
        Residue mask shape: (R, T, max_atoms)

        Only distance-only interactions use SMARTS prefilter here.
        Other interactions receive all-True masks (no filtering).
        """
        from prolif.interactions import Hydrophobic, Cationic, Anionic, MetalDonor, MetalAcceptor

        rdl = self._to_rdmol(ligand)
        rds = [self._to_rdmol(r) for r in residues]

        N = int(ligand.GetNumAtoms() if hasattr(ligand, 'GetNumAtoms') else rdl.GetNumAtoms())
        T = len(interaction_types)

        lig_masks = []
        res_masks_rows = []

        # Build a mapping from type string to RDKit SMARTS patterns (lig, prot)
        patterns = {}
        for t in interaction_types:
            if t == 'Hydrophobic':
                inter = Hydrophobic()
                patterns[t] = (inter.lig_pattern, inter.prot_pattern)
            elif t == 'Cationic':
                inter = Cationic()
                patterns[t] = (inter.lig_pattern, inter.prot_pattern)
            elif t == 'Anionic':
                inter = Anionic()
                patterns[t] = (inter.lig_pattern, inter.prot_pattern)
            elif t == 'MetalDonor':
                inter = MetalDonor()
                patterns[t] = (inter.lig_pattern, inter.prot_pattern)
            elif t == 'MetalAcceptor':
                inter = MetalAcceptor()
                patterns[t] = (inter.lig_pattern, inter.prot_pattern)
            else:
                patterns[t] = (None, None)

        for t in interaction_types:
            lig_pat, prot_pat = patterns[t]
            if lig_pat is None:
                lig_masks.append(jnp.ones((N,), dtype=bool))
                # placeholder residues filled below
                continue
            lig_mask = jnp.zeros((N,), dtype=bool)
            if rdl is not None:
                lmatches = rdl.GetSubstructMatches(lig_pat)
                if lmatches:
                    idxs = [m[0] for m in lmatches]
                    lig_mask = lig_mask.at[jnp.array(idxs)].set(True)
            lig_masks.append(lig_mask)

        # Build residue masks per residue and type
        for rdm in rds:
            M = int(rdm.GetNumAtoms()) if rdm is not None else 0
            row = []
            for t in interaction_types:
                lig_pat, prot_pat = patterns[t]
                if prot_pat is None or rdm is None:
                    m = jnp.zeros((max_atoms,), dtype=bool)
                    if M:
                        m = m.at[:M].set(True)
                    row.append(m)
                    continue
                m = jnp.zeros((max_atoms,), dtype=bool)
                pmatches = rdm.GetSubstructMatches(prot_pat)
                if pmatches:
                    idxs = [mm[0] for mm in pmatches]
                    mmask = jnp.zeros((M,), dtype=bool).at[jnp.array(idxs)].set(True)
                    m = m.at[:M].set(mmask)
                else:
                    if M:
                        # no matches → no candidates
                        pass
                row.append(m)
            res_masks_rows.append(jnp.stack(row))  # (T, max_atoms)

        lig_cand_mat = jnp.stack(lig_masks) if lig_masks else jnp.zeros((0, N), dtype=bool)
        res_cand_mat = jnp.stack(res_masks_rows) if res_masks_rows else jnp.zeros((0, T, max_atoms), dtype=bool)
        return lig_cand_mat, res_cand_mat

    def _extract_aromatic_rings(self, mol) -> list[list[int]]:
        """Return a cached list of aromatic ring atom indices for ``mol``.

        Raises a ValueError if conversion to RDKit Mol fails, when called
        from a ring-enabled computation path.
        """
        rdmol = self._to_rdmol(mol)
        if rdmol is None:
            raise ValueError("Aromatic ring interactions requested but RDKit molecule conversion failed.")
        key = id(rdmol)
        cached = self._ring_cache.get(key)
        if cached is not None:
            return cached
        ring_info = rdmol.GetRingInfo()
        rings = ring_info.AtomRings()
        aromatic = []
        for ring in rings:
            if len(ring) >= 3 and all(rdmol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
                aromatic.append(list(ring))
        self._ring_cache[key] = aromatic
        return aromatic

    def _pad_ring_indices(self, rings: list[list[int]], s_max: int) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Pad ring index lists to fixed size with a mask."""
        k = max(1, len(rings))
        idx = jnp.zeros((k, s_max), dtype=int)
        mask = jnp.zeros((k, s_max), dtype=bool)
        for r_i, ring in enumerate(rings):
            L = min(len(ring), s_max)
            if L:
                idx = idx.at[r_i, :L].set(jnp.array(ring[:L], dtype=int))
                mask = mask.at[r_i, :L].set(True)
        return idx, mask

    def _get_or_build_layout(self, ligand_coords, residue_coords_list) -> dict:
        """Return a cached layout dictionary or build a new one.

        The layout contains the padded mask, the original sizes, and the
        maximum number of atoms across residues. It is reused when atom
        counts do not change.
        """
        key = self._layout_key(ligand_coords, residue_coords_list)
        layout = self._layout_cache.get(key)
        if layout is not None:
            return layout

        max_atoms = max(coords.shape[0] for coords in residue_coords_list)
        masks = []
        for coords in residue_coords_list:
            M = coords.shape[0]
            real_mask = jnp.ones(M, dtype=bool)
            pad_mask = jnp.zeros(max_atoms - M, dtype=bool)
            mask = jnp.concatenate([real_mask, pad_mask])
            masks.append(mask)

        valid_mask = jnp.stack(masks)
        original_sizes = [coords.shape[0] for coords in residue_coords_list]
        layout = {
            'max_atoms': max_atoms,
            'valid_mask': valid_mask,
            'original_sizes': original_sizes,
        }
        self._layout_cache[key] = layout
        return layout

    def compute_interactions(
        self,
        ligand,
        residues: list,
        return_mode: str = 'full',
    ) -> list[dict] | list[dict[str, bool]]:
        """Compute interactions between ligand and multiple residues.

        This is the main entry point for JAX-accelerated computation.

        Args:
            ligand: RDKit molecule or ProLIF Residue for the ligand.
            residues: List of RDKit molecules or ProLIF Residues for
                protein residues to check.
            return_mode: Either ``'full'`` for detailed arrays per
                interaction, or ``'summary'`` for per-residue booleans
                indicating presence of each interaction.

        Returns:
            If ``return_mode='full'``, a list of result dicts, one per
            residue. Each dict maps interaction names to their detailed
            results (mask, distances, angles).

            If ``return_mode='summary'``, a list of dictionaries with
            per-residue booleans keyed by interaction name.
        """
        ligand_coords, ligand_elements = self.extract_coords(ligand)

        residue_coords_list = []
        residue_elements_list = []
        for res in residues:
            coords, elements = self.extract_coords(res)
            residue_coords_list.append(coords)
            residue_elements_list.append(elements)

        layout = self._get_or_build_layout(ligand_coords, residue_coords_list)

        ring_types = {'CationPi', 'PiCation', 'FaceToFace', 'EdgeToFace', 'PiStacking'}
        need_rings = any(t in ring_types for t in self.interaction_types)

        lig_ring_idx = None
        lig_ring_mask = None
        res_ring_idx = None
        res_ring_mask = None

        if need_rings:
            lig_rings = self._extract_aromatic_rings(ligand)
            res_rings_all = [self._extract_aromatic_rings(res) for res in residues]
            s_max = 0
            k_max = 1
            for rl in [lig_rings] + res_rings_all:
                if rl:
                    s_max = max(s_max, max(len(r) for r in rl))
                    k_max = max(k_max, len(rl))
            s_max = max(s_max, 3)
            lig_idx, lig_mask = self._pad_ring_indices(lig_rings, s_max)
            lig_ring_idx = lig_idx
            lig_ring_mask = lig_mask
            rr_idx_list = []
            rr_mask_list = []
            for rl in res_rings_all:
                idx, msk = self._pad_ring_indices(rl, s_max)
                if idx.shape[0] < k_max:
                    pad_k = k_max - idx.shape[0]
                    idx = jnp.concatenate([idx, jnp.zeros((pad_k, s_max), dtype=int)], axis=0)
                    msk = jnp.concatenate([msk, jnp.zeros((pad_k, s_max), dtype=bool)], axis=0)
                rr_idx_list.append(idx)
                rr_mask_list.append(msk)
            res_ring_idx = jnp.stack(rr_idx_list)
            res_ring_mask = jnp.stack(rr_mask_list)
        lig_cand_mat = None
        res_cand_mat = None
        if self.prefilter_smarts:
            lig_cand_mat, res_cand_mat = self._smarts_prefilter_masks(ligand, residues, self.interaction_types, layout['max_atoms'])
        batch = prepare_batch(
            ligand_coords,
            ligand_elements,
            residue_coords_list,
            residue_elements_list,
            self.interaction_types,
            layout=layout,
            lig_ring_idx=lig_ring_idx,
            lig_ring_mask=lig_ring_mask,
            res_ring_idx=res_ring_idx,
            res_ring_mask=res_ring_mask,
            gate_angles_by_distance=self.gate_angles_by_distance,
            lig_cand_mat=lig_cand_mat,
            res_cand_mat=res_cand_mat,
        )

        results = run_all_interactions(batch)

        if return_mode == 'summary':
            from .dispatch import summarize_batched_results
            summary = summarize_batched_results(results)

            original_sizes = batch['original_sizes']
            R = len(original_sizes)
            per_residue = []
            for r in range(R):
                item = {}
                for name, arr in summary.items():
                    item[name] = bool(arr[r])
                per_residue.append(item)
            return per_residue

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
    from .cationic import cationic_contacts

    cat = jnp.array(cation_coords)
    ani = jnp.array(anion_coords)

    mask, dists = cationic_contacts(cat, ani, distance_cutoff)

    return np.array(mask), np.array(dists)
