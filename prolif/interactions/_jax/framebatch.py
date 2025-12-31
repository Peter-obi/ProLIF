"""
Frame-batched JAX helpers for MD trajectories.

This module provides minimal, safe utilities to vectorize geometry across
many frames while keeping indices fixed (SMARTS done once per system).

Design goals:
- Avoid residue-batched complexity (no per-residue index threading/padding).
- Keep shapes stable across frames to enable JIT specialization.
- Only implement clear wins; leave angles to the integration layer for now.

API stability: experimental. Functions here are intentionally small and
focused to avoid future breakage.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .primitives import pairwise_distances, angle_at_vertex
from .primitives import angle_between_vectors


def pairwise_distances_frames(
    lig_coords_f: jnp.ndarray,
    res_coords_f: jnp.ndarray,
) -> jnp.ndarray:
    """Compute distances for each frame and residue in a trajectory.

    Args:
        lig_coords_f: (F, N, 3) ligand coordinates per frame.
        res_coords_f: (F, R, M, 3) residue coordinates per frame
            (padded to the same M across residues; masks handled upstream).

    Returns:
        (F, R, N, M) array of pairwise distances.

    Notes:
        - Shapes must be consistent across frames to avoid recompilation.
        - This function does not apply masks; callers should mask padded
          atoms as needed.
    """

    def _frame_distances(lig: jnp.ndarray, res_batch: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(lambda rc: pairwise_distances(lig, rc))(res_batch)

    return jax.vmap(_frame_distances)(lig_coords_f, res_coords_f)


def hbacceptor_frames(
    lig_coords_f: jnp.ndarray,
    res_coords_f: jnp.ndarray,
    acc_idx: jnp.ndarray,
    d_idx: jnp.ndarray,
    h_idx: jnp.ndarray,
    *,
    distance_cutoff: float = 3.5,
    dha_angle_min: float = 130.0,
    dha_angle_max: float = 180.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Frame-batched hydrogen bond (acceptor) geometry.

    Args:
        lig_coords_f: (F, N, 3) ligand coords per frame.
        res_coords_f: (F, M, 3) residue coords per frame.
        acc_idx: (Na,) ligand acceptor indices.
        d_idx: (K,) residue donor indices.
        h_idx: (K,) residue hydrogen indices (paired with donors).
        distance_cutoff: Max A–D distance.
        dha_angle_min/max: D–H–A angle limits (deg).

    Returns:
        mask: (F, Na, K) boolean
        distances: (F, Na, K) A–D distances
        angles: (F, Na, K) D–H–A angles (deg)
    """
    acc = lig_coords_f[:, acc_idx, :]
    donors = res_coords_f[:, d_idx, :]
    hydrogens = res_coords_f[:, h_idx, :]

    dvec = acc[:, :, None, :] - donors[:, None, :, :]
    dists = jnp.linalg.norm(dvec, axis=-1)
    dist_ok = dists <= distance_cutoff

    ang = angle_at_vertex(
        donors[:, None, :, :],
        hydrogens[:, None, :, :],
        acc[:, :, None, :],
    )
    ang_deg = jnp.degrees(ang)
    ang_ok = (ang_deg >= dha_angle_min) & (ang_deg <= dha_angle_max)

    mask = dist_ok & ang_ok
    return mask, dists, ang_deg


def hbdonor_frames(
    lig_coords_f: jnp.ndarray,
    res_coords_f: jnp.ndarray,
    d_idx: jnp.ndarray,
    h_idx: jnp.ndarray,
    acc_idx: jnp.ndarray,
    *,
    distance_cutoff: float = 3.5,
    dha_angle_min: float = 130.0,
    dha_angle_max: float = 180.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Frame-batched hydrogen bond (donor) geometry (inverted).

    Returns (mask, distances, angles) with shapes (F, Nd, Ka).
    """
    donors = lig_coords_f[:, d_idx, :]
    hydrogens = lig_coords_f[:, h_idx, :]
    acc = res_coords_f[:, acc_idx, :]

    dvec = donors[:, :, None, :] - acc[:, None, :, :]
    dists = jnp.linalg.norm(dvec, axis=-1)
    dist_ok = dists <= distance_cutoff

    ang = angle_at_vertex(
        donors[:, :, None, :],
        hydrogens[:, :, None, :],
        acc[:, None, :, :],
    )
    ang_deg = jnp.degrees(ang)
    ang_ok = (ang_deg >= dha_angle_min) & (ang_deg <= dha_angle_max)
    mask = dist_ok & ang_ok
    return mask, dists, ang_deg


def xbacceptor_frames(
    lig_coords_f: jnp.ndarray,
    res_coords_f: jnp.ndarray,
    a_idx: jnp.ndarray,
    r_idx: jnp.ndarray,
    x_idx: jnp.ndarray,
    d_idx: jnp.ndarray,
    *,
    distance_cutoff: float = 3.5,
    axd_angle_min: float = 130.0,
    axd_angle_max: float = 180.0,
    xar_angle_min: float = 80.0,
    xar_angle_max: float = 140.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Frame-batched halogen bond (acceptor) geometry.

    Returns (mask, distances[A–X], axd_angles, xar_angles) with shapes (F, Na, K).
    """
    acc = lig_coords_f[:, a_idx, :]
    neigh = lig_coords_f[:, r_idx, :]
    hal = res_coords_f[:, x_idx, :]
    don = res_coords_f[:, d_idx, :]

    dvec = acc[:, :, None, :] - hal[:, None, :, :]
    dists = jnp.linalg.norm(dvec, axis=-1)
    dist_ok = dists <= distance_cutoff

    axd = angle_at_vertex(
        acc[:, :, None, :],
        hal[:, None, :, :],
        don[:, None, :, :],
    )
    axd_deg = jnp.degrees(axd)
    axd_ok = (axd_deg >= axd_angle_min) & (axd_deg <= axd_angle_max)

    xar = angle_at_vertex(
        hal[:, None, :, :],
        acc[:, :, None, :],
        neigh[:, :, None, :],
    )
    xar_deg = jnp.degrees(xar)
    xar_ok = (xar_deg >= xar_angle_min) & (xar_deg <= xar_angle_max)

    mask = dist_ok & axd_ok & xar_ok
    return mask, dists, axd_deg, xar_deg


def xbdonor_frames(
    lig_coords_f: jnp.ndarray,
    res_coords_f: jnp.ndarray,
    x_idx: jnp.ndarray,
    d_idx: jnp.ndarray,
    a_idx: jnp.ndarray,
    r_idx: jnp.ndarray,
    *,
    distance_cutoff: float = 3.5,
    axd_angle_min: float = 130.0,
    axd_angle_max: float = 180.0,
    xar_angle_min: float = 80.0,
    xar_angle_max: float = 140.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Frame-batched halogen bond (donor) geometry (inverted).

    Returns (mask, distances[X–A], axd_angles, xar_angles) with shapes (F, Nx, Ka).
    """
    hal = lig_coords_f[:, x_idx, :]
    don = lig_coords_f[:, d_idx, :]
    acc = res_coords_f[:, a_idx, :]
    neigh = res_coords_f[:, r_idx, :]

    dvec = hal[:, :, None, :] - acc[:, None, :, :]
    dists = jnp.linalg.norm(dvec, axis=-1)
    dist_ok = dists <= distance_cutoff

    axd = angle_at_vertex(
        acc[:, None, :, :],
        hal[:, :, None, :],
        don[:, :, None, :],
    )
    axd_deg = jnp.degrees(axd)
    axd_ok = (axd_deg >= axd_angle_min) & (axd_deg <= axd_angle_max)

    xar = angle_at_vertex(
        hal[:, :, None, :],
        acc[:, None, :, :],
        neigh[:, None, :, :],
    )
    xar_deg = jnp.degrees(xar)
    xar_ok = (xar_deg >= xar_angle_min) & (xar_deg <= xar_angle_max)

    mask = dist_ok & axd_ok & xar_ok
    return mask, dists, axd_deg, xar_deg



def _ring_centroids_normals_frames(
    coords_f: jnp.ndarray,
    rings: list[jnp.ndarray],
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute ring centroids and normals across frames for a list of rings.

    Args:
        coords_f: (F, N, 3) coordinates per frame
        rings: list of index arrays (variable ring sizes allowed)

    Returns:
        centroids: (F, K, 3)
        normals: (F, K, 3) unit normals (cross of first two centroid vectors)
    """
    F = int(coords_f.shape[0])
    centroids_list = []
    normals_list = []
    for ring_idx in rings:
        rc = coords_f[:, ring_idx, :]
        centroid = rc.mean(axis=1)
        v1 = rc[:, 0, :] - centroid
        v2 = rc[:, 1, :] - centroid
        n = jnp.cross(v1, v2)
        n = n / jnp.clip(jnp.linalg.norm(n, axis=-1, keepdims=True), 1e-8)
        centroids_list.append(centroid)
        normals_list.append(n)
    if centroids_list:
        centroids = jnp.stack(centroids_list, axis=1)
        normals = jnp.stack(normals_list, axis=1)
    else:
        centroids = jnp.zeros((F, 0, 3))
        normals = jnp.zeros((F, 0, 3))
    return centroids, normals


def cationpi_frames(
    ring_coords_f: jnp.ndarray,
    ring_list: list[jnp.ndarray],
    cation_coords_f: jnp.ndarray,
    cation_idx: jnp.ndarray,
    *,
    distance_cutoff: float = 4.5,
    angle_min: float = 0.0,
    angle_max: float = 90.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Frame-batched Cation–Pi geometry for one orientation (ring vs cation).

    Returns:
        mask: (F, Kr, Kc)
        distances: (F, Kr, Kc) centroid–cation distances
        angles: (F, Kr, Kc) normal–centroid vector angles (deg)
    """
    F = int(ring_coords_f.shape[0])
    centroids, normals = _ring_centroids_normals_frames(ring_coords_f, ring_list)
    cations = cation_coords_f[:, cation_idx, :] if cation_idx.size else jnp.zeros((F, 0, 3))
    if centroids.shape[1] == 0 or cations.shape[1] == 0:
        z = jnp.zeros((F, centroids.shape[1], cations.shape[1]))
        return z.astype(bool), z, z

    vec = cations[:, None, :, :] - centroids[:, :, None, :]
    dists = jnp.linalg.norm(vec, axis=-1)
    dist_ok = dists <= distance_cutoff

    ang = angle_between_vectors(normals[:, :, None, :], vec)
    ang_deg = jnp.degrees(ang)
    ang_ok = (ang_deg >= angle_min) & (ang_deg <= angle_max)
    mask = dist_ok & ang_ok
    return mask, dists, ang_deg


def pistacking_frames(
    lig_coords_f: jnp.ndarray,
    lig_rings: list[jnp.ndarray],
    res_coords_f: jnp.ndarray,
    res_rings: list[jnp.ndarray],
    *,
    distance_cutoff: float = 6.5,
    plane_angle_min: float = 0.0,
    plane_angle_max: float = 30.0,
    ncc_angle_min: float = 0.0,
    ncc_angle_max: float = 60.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Frame-batched Pi-stacking geometry for ring–ring pairs.

    Returns:
        mask: (F, Kl, Kr)
        distances: (F, Kl, Kr) centroid–centroid distances
        plane_angles: (F, Kl, Kr) normals plane angle (deg)
        ncc_angles: (F, Kl, Kr) min of the two normal→centroid angles (deg)
    """
    F = int(lig_coords_f.shape[0])
    lc, ln = _ring_centroids_normals_frames(lig_coords_f, lig_rings)
    rc, rn = _ring_centroids_normals_frames(res_coords_f, res_rings)
    if lc.shape[1] == 0 or rc.shape[1] == 0:
        z = jnp.zeros((F, lc.shape[1], rc.shape[1]))
        return z.astype(bool), z, z, z

    cc_vec = rc[:, None, :, :] - lc[:, :, None, :]
    dists = jnp.linalg.norm(cc_vec, axis=-1)
    dist_ok = dists <= distance_cutoff

    pa = angle_between_vectors(ln[:, :, None, :], rn[:, None, :, :])
    pa_deg = jnp.degrees(pa)
    pa_ok = (pa_deg >= plane_angle_min) & (pa_deg <= plane_angle_max)

    n1 = angle_between_vectors(ln[:, :, None, :], cc_vec)
    n2 = angle_between_vectors(rn[:, None, :, :], -cc_vec)
    n1_deg = jnp.degrees(n1)
    n2_deg = jnp.degrees(n2)
    ncc_deg = jnp.minimum(n1_deg, n2_deg)
    ncc_ok = (ncc_deg >= ncc_angle_min) & (ncc_deg <= ncc_angle_max)

    mask = dist_ok & pa_ok & ncc_ok
    return mask, dists, pa_deg, ncc_deg


def build_actor_masks(lig_mol, residues):
    """Compute boolean masks for distance-only actor atoms.

    Returns ligand and residue masks for Hydrophobic, Cationic, Anionic,
    MetalDonor, and MetalAcceptor SMARTS patterns using ProLIF interactions.
    Residue masks are padded to a common length across residues.
    """
    import jax.numpy as jnp
    from prolif.interactions import Hydrophobic, MetalDonor, MetalAcceptor, Cationic, Anionic

    inters = {
        'Hydrophobic': Hydrophobic(),
        'Cationic': Cationic(),
        'Anionic': Anionic(),
        'MetalDonor': MetalDonor(),
        'MetalAcceptor': MetalAcceptor(),
    }

    N = lig_mol.GetNumAtoms()
    max_m = max((r.GetNumAtoms() for r in residues), default=0)

    lig_masks = {}
    res_masks = {}
    for name, inter in inters.items():
        lm = jnp.zeros((N,), dtype=bool)
        lmatches = lig_mol.GetSubstructMatches(inter.lig_pattern)
        if lmatches:
            idxs = [m[0] for m in lmatches]
            lm = lm.at[jnp.array(idxs)].set(True)
        lig_masks[name] = lm

        r_rows = []
        for r in residues:
            m = jnp.zeros((max_m,), dtype=bool)
            pmatches = r.GetSubstructMatches(inter.prot_pattern)
            if pmatches:
                idxs = [mm[0] for mm in pmatches]
                mm = jnp.zeros((r.GetNumAtoms(),), dtype=bool)
                mm = mm.at[jnp.array(idxs)].set(True)
                m = m.at[:r.GetNumAtoms()].set(mm)
            r_rows.append(m)
        res_masks[name] = jnp.stack(r_rows, axis=0) if r_rows else jnp.zeros((0, 0), dtype=bool)

    return lig_masks, res_masks


def build_angle_indices(lig_mol, residues):
    """Precompute indices for hydrogen and halogen bond geometry.

    Returns a mapping with indices for both acceptor and donor orientations
    of hydrogen and halogen bonds. For HB donor orientation, donors and
    hydrogens are on the ligand and acceptors on the residue; for acceptor
    orientation, acceptors are on the ligand and donor–hydrogen pairs on the
    residue. The same convention applies for XB with donors and halogens.
    """
    import jax.numpy as jnp
    from prolif.interactions import HBAcceptor, HBDonor, XBAcceptor, XBDonor

    hb_acc = HBAcceptor()
    hb_don = HBDonor()
    xb_acc = XBAcceptor()
    xb_don = XBDonor()

    hb_acc_idx = []
    lmatches = lig_mol.GetSubstructMatches(hb_acc.lig_pattern)
    if lmatches:
        hb_acc_idx = [m[0] for m in lmatches]
    hb_acc_idx = jnp.array(hb_acc_idx, dtype=int) if hb_acc_idx else jnp.zeros((0,), dtype=int)

    hb_d_rows, hb_h_rows = [], []
    for r in residues:
        pmatches = r.GetSubstructMatches(hb_acc.prot_pattern)
        pairs = []
        for m in (pmatches or []):
            if len(m) >= 2:
                pairs.append((m[0], m[1]))
        d = jnp.array([p[0] for p in pairs], dtype=int) if pairs else jnp.zeros((0,), dtype=int)
        h = jnp.array([p[1] for p in pairs], dtype=int) if pairs else jnp.zeros((0,), dtype=int)
        hb_d_rows.append(d)
        hb_h_rows.append(h)

    xb_a_rows, xb_r_rows = [], []
    lmatches = lig_mol.GetSubstructMatches(xb_acc.lig_pattern)
    a = [m[0] for m in lmatches] if lmatches else []
    r = [m[1] for m in lmatches] if lmatches else []
    xbacc_a_idx = jnp.array(a, dtype=int) if a else jnp.zeros((0,), dtype=int)
    xbacc_r_idx = jnp.array(r, dtype=int) if r else jnp.zeros((0,), dtype=int)

    xb_x_rows, xb_d_rows = [], []
    for res in residues:
        pmatches = res.GetSubstructMatches(xb_acc.prot_pattern)
        pairs = []
        for m in (pmatches or []):
            if len(m) >= 2:
                pairs.append((m[1], m[0]))
        x = jnp.array([p[0] for p in pairs], dtype=int) if pairs else jnp.zeros((0,), dtype=int)
        d = jnp.array([p[1] for p in pairs], dtype=int) if pairs else jnp.zeros((0,), dtype=int)
        xb_x_rows.append(x)
        xb_d_rows.append(d)

    lmatches = lig_mol.GetSubstructMatches(hb_don.lig_pattern)
    hb_lig_pairs = []
    for m in (lmatches or []):
        if len(m) >= 2:
            hb_lig_pairs.append((m[0], m[1]))
    hb_lig_d_idx = jnp.array([p[0] for p in hb_lig_pairs], dtype=int) if hb_lig_pairs else jnp.zeros((0,), dtype=int)
    hb_lig_h_idx = jnp.array([p[1] for p in hb_lig_pairs], dtype=int) if hb_lig_pairs else jnp.zeros((0,), dtype=int)

    hb_res_acc_rows = []
    for res in residues:
        pmatches = res.GetSubstructMatches(hb_don.prot_pattern)
        acc = jnp.array([m[0] for m in (pmatches or [])], dtype=int) if pmatches else jnp.zeros((0,), dtype=int)
        hb_res_acc_rows.append(acc)

    lmatches = lig_mol.GetSubstructMatches(xb_don.lig_pattern)
    xbdon_pairs = []
    for m in (lmatches or []):
        if len(m) >= 2:
            xbdon_pairs.append((m[1], m[0]))
    xbdon_lig_x_idx = jnp.array([p[0] for p in xbdon_pairs], dtype=int) if xbdon_pairs else jnp.zeros((0,), dtype=int)
    xbdon_lig_d_idx = jnp.array([p[1] for p in xbdon_pairs], dtype=int) if xbdon_pairs else jnp.zeros((0,), dtype=int)

    xbdon_res_a_rows, xbdon_res_r_rows = [], []
    for res in residues:
        pmatches = res.GetSubstructMatches(xb_don.prot_pattern)
        a = jnp.array([m[0] for m in (pmatches or [])], dtype=int) if pmatches else jnp.zeros((0,), dtype=int)
        r = jnp.array([m[1] for m in (pmatches or [])], dtype=int) if pmatches else jnp.zeros((0,), dtype=int)
        xbdon_res_a_rows.append(a)
        xbdon_res_r_rows.append(r)

    return {
        'hb': {
            'acc_idx': hb_acc_idx,
            'res_d_idx': hb_d_rows,
            'res_h_idx': hb_h_rows,
        },
        'hb_donor': {
            'lig_d_idx': hb_lig_d_idx,
            'lig_h_idx': hb_lig_h_idx,
            'res_a_idx': hb_res_acc_rows,
        },
        'xbacc': {
            'lig_a_idx': xbacc_a_idx,
            'lig_r_idx': xbacc_r_idx,
            'res_x_idx': xb_x_rows,
            'res_d_idx': xb_d_rows,
        },
        'xbdon': {
            'lig_x_idx': xbdon_lig_x_idx,
            'lig_d_idx': xbdon_lig_d_idx,
            'res_a_idx': xbdon_res_a_rows,
            'res_r_idx': xbdon_res_r_rows,
        },
    }


def build_ring_cation_indices(lig_mol, residues):
    """Precompute ring and cation indices for ring-based interactions.

    Uses ProLIF patterns to find aromatic rings and cations on the ligand and
    per residue; returns lists of ring index arrays and arrays of cation indices.
    """
    import jax.numpy as jnp
    from prolif.interactions import PiStacking, CationPi, FaceToFace, EdgeToFace

    pi = PiStacking()
    ftf = FaceToFace()
    etf = EdgeToFace()
    ring_patterns = getattr(ftf, "pi_ring", []) or getattr(etf, "pi_ring", [])
    lig_rings = []
    for pat in ring_patterns:
        for m in lig_mol.GetSubstructMatches(pat):
            lig_rings.append(jnp.array(list(m), dtype=int))
    res_rings = []
    for r in residues:
        rr = []
        for pat in ring_patterns:
            for m in r.GetSubstructMatches(pat):
                rr.append(jnp.array(list(m), dtype=int))
        res_rings.append(rr)

    cp = CationPi()
    lmatches = lig_mol.GetSubstructMatches(cp.cation)
    lig_cations = jnp.array([m[0] for m in lmatches], dtype=int) if lmatches else jnp.zeros((0,), dtype=int)
    res_cations = []
    for r in residues:
        pm = r.GetSubstructMatches(cp.cation)
        rc = jnp.array([m[0] for m in pm], dtype=int) if pm else jnp.zeros((0,), dtype=int)
        res_cations.append(rc)

    return {
        'lig_rings': lig_rings,
        'res_rings': res_rings,
        'lig_cations': lig_cations,
        'res_cations': res_cations,
        'pi': pi,
        'cp': cp,
    }


def build_vdw_radii(
    lig_mol,
    residues,
    *,
    lig_ag=None,
    residue_ags=None,
    use_real: bool = False,
):
    """Build per-atom van der Waals radii arrays aligned with coordinate order.

    When ``use_real`` is True, radii are derived from MDAnalysis AtomGroups to
    match the coordinate ordering in real-frame mode. Otherwise, radii are
    derived from RDKit molecules to match duplicate-frame mode.

    Returns ligand radii (N,), residue radii (R, M) padded to max M, and
    ProLIF's VdW tolerance.
    """
    import jax.numpy as jnp
    from prolif.interactions import VdWContact

    def _symbol_from_atom(atom):
        try:
            sym = atom.element
        except Exception:
            sym = None
        if not sym:
            try:
                from MDAnalysis.topology import guessers
                sym = guessers.guess_atom_element(getattr(atom, 'name', ''))
            except Exception:
                sym = 'C'
        return str(sym).capitalize()

    vdw = VdWContact()

    if use_real:
        assert lig_ag is not None and residue_ags is not None
        lig_elems = [_symbol_from_atom(a) for a in lig_ag.atoms]
        lig_radii = jnp.array([vdw.vdwradii.get(e, 1.7) for e in lig_elems], dtype=float)

        max_m = max((ag.n_atoms for ag in residue_ags), default=0)
        res_rows = []
        for ag in residue_ags:
            elems = [_symbol_from_atom(a) for a in ag.atoms]
            row = jnp.array([vdw.vdwradii.get(e, 1.7) for e in elems], dtype=float)
            if ag.n_atoms < max_m:
                pad = jnp.zeros((max_m - ag.n_atoms,), dtype=float)
                row = jnp.concatenate([row, pad], axis=0)
            res_rows.append(row)
        res_radii = jnp.stack(res_rows, axis=0) if res_rows else jnp.zeros((0, 0), dtype=float)
    else:
        N = lig_mol.GetNumAtoms()
        lig_elems = [lig_mol.GetAtomWithIdx(i).GetSymbol() for i in range(N)]
        lig_radii = jnp.array([vdw.vdwradii.get(e, 1.7) for e in lig_elems], dtype=float)

        max_m = max((r.GetNumAtoms() for r in residues), default=0)
        res_rows = []
        for r in residues:
            m = r.GetNumAtoms()
            elems = [r.GetAtomWithIdx(i).GetSymbol() for i in range(m)]
            row = jnp.array([vdw.vdwradii.get(e, 1.7) for e in elems], dtype=float)
            if m < max_m:
                pad = jnp.zeros((max_m - m,), dtype=float)
                row = jnp.concatenate([row, pad], axis=0)
            res_rows.append(row)
        res_radii = jnp.stack(res_rows, axis=0) if res_rows else jnp.zeros((0, 0), dtype=float)

    return lig_radii, res_radii, float(vdw.tolerance)


def has_interactions_frames(
    lig_f: jnp.ndarray,
    res_f: jnp.ndarray,
    res_valid_mask: jnp.ndarray,
    lig_masks: dict,
    res_actor_masks: dict,
    angle_idx: dict,
    ring_idx: dict,
    vdw_radii: tuple,
) -> dict[str, jnp.ndarray]:
    """Evaluate nine interactions across frames and residues, returning booleans.

    Returns a mapping name → (F, R) boolean arrays for:
    Hydrophobic, Cationic, Anionic, VdWContact, HBAcceptor, HBDonor,
    PiStacking, CationPi, PiCation.
    """
    import jax.numpy as jnp

    F = int(lig_f.shape[0])
    R = int(res_f.shape[1]) if res_f.ndim == 4 else 0
    results = {}

    d = pairwise_distances_frames(lig_f, res_f)

    m = (d <= 4.5) & lig_masks['Hydrophobic'][None, None, :, None] & (res_actor_masks['Hydrophobic'] & res_valid_mask)[None, :, None, :]
    results['Hydrophobic'] = jnp.any(m, axis=(2, 3)) if R else jnp.zeros((F, 0), dtype=bool)

    for k in ('Cationic', 'Anionic'):
        m = (d <= 4.5) & lig_masks[k][None, None, :, None] & (res_actor_masks[k] & res_valid_mask)[None, :, None, :]
        results[k] = jnp.any(m, axis=(2, 3)) if R else jnp.zeros((F, 0), dtype=bool)

    lig_radii, res_radii, vdw_tol = vdw_radii
    radii_sum = lig_radii[None, None, :, None] + res_radii[None, :, None, :]
    m = (d <= (radii_sum + vdw_tol)) & res_valid_mask[None, :, None, :]
    results['VdWContact'] = jnp.any(m, axis=(2, 3)) if R else jnp.zeros((F, 0), dtype=bool)

    acc_idx = angle_idx['hb']['acc_idx']
    hb_acc_out = []
    for r_i in range(R):
        d_idx = angle_idx['hb']['res_d_idx'][r_i]
        h_idx = angle_idx['hb']['res_h_idx'][r_i]
        if acc_idx.size and d_idx.size:
            _m, _, _ = hbacceptor_frames(lig_f, res_f[:, r_i, :, :], acc_idx, d_idx, h_idx)
            hb_acc_out.append(jnp.any(_m, axis=(1, 2)))
        else:
            hb_acc_out.append(jnp.zeros((F,), dtype=bool))
    results['HBAcceptor'] = jnp.stack(hb_acc_out, axis=1) if R else jnp.zeros((F, 0), dtype=bool)

    hb_don_out = []
    lig_d = angle_idx['hb_donor']['lig_d_idx']
    lig_h = angle_idx['hb_donor']['lig_h_idx']
    for r_i in range(R):
        acc = angle_idx['hb_donor']['res_a_idx'][r_i]
        if lig_d.size and acc.size:
            _m, _, _ = hbdonor_frames(lig_f, res_f[:, r_i, :, :], lig_d, lig_h, acc)
            hb_don_out.append(jnp.any(_m, axis=(1, 2)))
        else:
            hb_don_out.append(jnp.zeros((F,), dtype=bool))
    results['HBDonor'] = jnp.stack(hb_don_out, axis=1) if R else jnp.zeros((F, 0), dtype=bool)

    lig_rings = ring_idx['lig_rings']
    res_rings = ring_idx['res_rings']
    lig_cations = ring_idx['lig_cations']
    res_cations = ring_idx['res_cations']
    pi = ring_idx['pi']
    cp = ring_idx['cp']

    cationpi_out = []
    picat_out = []
    for r_i in range(R):
        has_cationpi = jnp.zeros((F,), dtype=bool)
        has_picat = jnp.zeros((F,), dtype=bool)
        if lig_cations.size and len(res_rings[r_i]):
            _m, _, _ = cationpi_frames(
                res_f[:, r_i, :, :], res_rings[r_i], lig_f, lig_cations,
                distance_cutoff=float(cp.distance),
                angle_min=float(cp.angle[0]), angle_max=float(cp.angle[1]),
            )
            has_cationpi = jnp.any(_m, axis=(1, 2))
        if len(lig_rings) and res_cations[r_i].size:
            _m, _, _ = cationpi_frames(
                lig_f, lig_rings, res_f[:, r_i, :, :], res_cations[r_i],
                distance_cutoff=float(cp.distance),
                angle_min=float(cp.angle[0]), angle_max=float(cp.angle[1]),
            )
            has_picat = jnp.any(_m, axis=(1, 2))
        cationpi_out.append(has_cationpi)
        picat_out.append(has_picat)
    results['CationPi'] = jnp.stack(cationpi_out, axis=1) if R else jnp.zeros((F, 0), dtype=bool)
    results['PiCation'] = jnp.stack(picat_out, axis=1) if R else jnp.zeros((F, 0), dtype=bool)

    ps_out = []
    for r_i in range(R):
        if len(lig_rings) and len(res_rings[r_i]):
            ftf = pi.ftf
            mF, _, _, _ = pistacking_frames(
                lig_f, lig_rings, res_f[:, r_i, :, :], res_rings[r_i],
                distance_cutoff=float(ftf.distance),
                plane_angle_min=float(ftf.plane_angle[0]), plane_angle_max=float(ftf.plane_angle[1]),
                ncc_angle_min=float(ftf.normal_to_centroid_angle[0]), ncc_angle_max=float(ftf.normal_to_centroid_angle[1]),
            )
            etf = pi.etf
            mE, _, _, _ = pistacking_frames(
                lig_f, lig_rings, res_f[:, r_i, :, :], res_rings[r_i],
                distance_cutoff=float(etf.distance),
                plane_angle_min=float(etf.plane_angle[0]), plane_angle_max=float(etf.plane_angle[1]),
                ncc_angle_min=float(etf.normal_to_centroid_angle[0]), ncc_angle_max=float(etf.normal_to_centroid_angle[1]),
            )
            ps_out.append(jnp.any(mF | mE, axis=(1, 2)))
        else:
            ps_out.append(jnp.zeros((F,), dtype=bool))
    results['PiStacking'] = jnp.stack(ps_out, axis=1) if R else jnp.zeros((F, 0), dtype=bool)

    return results


def prepare_for_device(
    lig_f: jnp.ndarray,
    res_f: jnp.ndarray,
    res_valid_mask: jnp.ndarray,
    lig_masks: dict,
    res_actor_masks: dict,
    angle_idx: dict,
    ring_idx: dict,
    vdw_radii: tuple,
    device: str = 'cpu',
) -> tuple:
    """Move frame-batched, small metadata arrays to a target device.

    Args:
        lig_f: (F, N, 3) ligand coordinates per frame.
        res_f: (F, R, M, 3) residue coordinates per frame.
        res_valid_mask: (R, M) boolean mask for valid atoms.
        lig_masks: Dict of ligand SMARTS masks per interaction.
        res_actor_masks: Dict of residue SMARTS masks per interaction.
        angle_idx: Dict of angle indices from build_angle_indices.
        ring_idx: Dict of ring indices from build_ring_cation_indices.
        vdw_radii: Tuple of (lig_radii, res_radii, tolerance).
        device: 'cpu' or 'gpu'. Coordinates remain on host; chunks are moved
            transiently during computation. Reused small structures (masks,
            indices, radii) are placed on the device.

    Returns:
        Tuple of all inputs moved to the specified device.
        Returns unchanged inputs if device='cpu' or no GPU available.
    """
    if device != 'gpu':
        return lig_f, res_f, res_valid_mask, lig_masks, res_actor_masks, angle_idx, ring_idx, vdw_radii

    gpus = jax.devices('gpu')
    if not gpus:
        return lig_f, res_f, res_valid_mask, lig_masks, res_actor_masks, angle_idx, ring_idx, vdw_radii

    dev = gpus[0]

    res_valid_mask = jax.device_put(res_valid_mask, device=dev)
    lig_masks = jax.tree_util.tree_map(lambda x: jax.device_put(x, device=dev), lig_masks)
    res_actor_masks = jax.tree_util.tree_map(lambda x: jax.device_put(x, device=dev), res_actor_masks)

    angle_idx = {
        'hb': {
            'acc_idx': jax.device_put(angle_idx['hb']['acc_idx'], device=dev),
            'res_d_idx': [jax.device_put(a, device=dev) for a in angle_idx['hb']['res_d_idx']],
            'res_h_idx': [jax.device_put(a, device=dev) for a in angle_idx['hb']['res_h_idx']],
        },
        'hb_donor': {
            'lig_d_idx': jax.device_put(angle_idx['hb_donor']['lig_d_idx'], device=dev),
            'lig_h_idx': jax.device_put(angle_idx['hb_donor']['lig_h_idx'], device=dev),
            'res_a_idx': [jax.device_put(a, device=dev) for a in angle_idx['hb_donor']['res_a_idx']],
        },
        'xbacc': {
            'lig_a_idx': jax.device_put(angle_idx['xbacc']['lig_a_idx'], device=dev),
            'lig_r_idx': jax.device_put(angle_idx['xbacc']['lig_r_idx'], device=dev),
            'res_x_idx': [jax.device_put(a, device=dev) for a in angle_idx['xbacc']['res_x_idx']],
            'res_d_idx': [jax.device_put(a, device=dev) for a in angle_idx['xbacc']['res_d_idx']],
        },
        'xbdon': {
            'lig_x_idx': jax.device_put(angle_idx['xbdon']['lig_x_idx'], device=dev),
            'lig_d_idx': jax.device_put(angle_idx['xbdon']['lig_d_idx'], device=dev),
            'res_a_idx': [jax.device_put(a, device=dev) for a in angle_idx['xbdon']['res_a_idx']],
            'res_r_idx': [jax.device_put(a, device=dev) for a in angle_idx['xbdon']['res_r_idx']],
        },
    }

    ring_idx = {
        'lig_rings': [jax.device_put(a, device=dev) for a in ring_idx['lig_rings']],
        'res_rings': [[jax.device_put(a, device=dev) for a in rr] for rr in ring_idx['res_rings']],
        'lig_cations': jax.device_put(ring_idx['lig_cations'], device=dev),
        'res_cations': [jax.device_put(a, device=dev) for a in ring_idx['res_cations']],
        'pi': ring_idx['pi'],
        'cp': ring_idx['cp'],
    }

    lr, rr, vdw_tol = vdw_radii
    vdw_radii = (
        jax.device_put(lr, device=dev),
        jax.device_put(rr, device=dev),
        vdw_tol,
    )

    return lig_f, res_f, res_valid_mask, lig_masks, res_actor_masks, angle_idx, ring_idx, vdw_radii


def get_gpu_device():
    """Return the first GPU device if available, else None."""
    gpus = jax.devices('gpu')
    return gpus[0] if gpus else None


def get_gpu_memory_info() -> tuple[float, float] | None:
    """Return (free_mb, total_mb) for GPU 0 if available, else None.

    Tries NVML first, then falls back to parsing `nvidia-smi` output.
    """
    dev = get_gpu_device()
    if dev is None:
        return None

    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_mb = info.free / (1024 * 1024)
        total_mb = info.total / (1024 * 1024)
        return float(free_mb), float(total_mb)
    except Exception:
        pass

    try:
        import subprocess
        out = subprocess.check_output([
            'nvidia-smi',
            '--query-gpu=memory.free,memory.total',
            '--format=csv,noheader,nounits'
        ], stderr=subprocess.STDOUT, text=True)
        line = out.strip().splitlines()[0]
        cols = [c.strip() for c in line.split(',')]
        free_mb = float(cols[0])
        total_mb = float(cols[1])
        return free_mb, total_mb
    except Exception:
        return None


def estimate_memory_per_frame(
    n_ligand_atoms: int,
    n_residues: int,
    max_residue_atoms: int,
) -> float:
    """Estimate GPU memory usage per frame in bytes.

    This is a rough estimate accounting for:
        - Distance matrices: R × N × M × 4 bytes
        - Intermediate arrays and masks
        - JAX overhead multiplier (~3x for JIT buffers)

    Args:
        n_ligand_atoms: Number of ligand atoms (N).
        n_residues: Number of residues (R).
        max_residue_atoms: Max atoms per residue (M).

    Returns:
        Estimated bytes per frame.
    """
    N, R, M = n_ligand_atoms, n_residues, max_residue_atoms

    coords_lig = N * 3 * 4
    coords_res = R * M * 3 * 4
    distance_matrix = R * N * M * 4
    masks_and_results = R * N * M * 4

    base = coords_lig + coords_res + distance_matrix + masks_and_results

    overhead_multiplier = 3.0

    return base * overhead_multiplier


def calculate_chunk_size(
    n_ligand_atoms: int,
    n_residues: int,
    max_residue_atoms: int,
    available_memory_mb: float | None = None,
    memory_fraction: float = 0.7,
) -> int:
    """Calculate safe chunk size (frames per batch) for GPU.

    Args:
        n_ligand_atoms: Number of ligand atoms.
        n_residues: Number of residues.
        max_residue_atoms: Max atoms per residue.
        available_memory_mb: GPU memory in MB. If None, auto-detect.
        memory_fraction: Fraction of memory to use (default 0.7 for safety).

    Returns:
        Recommended number of frames per chunk.
    """
    if available_memory_mb is None:
        mem = get_gpu_memory_info()
        if mem is None:
            return 1000
        available_memory_mb, total_mb = mem[0], mem[1]

    bytes_per_frame = estimate_memory_per_frame(
        n_ligand_atoms, n_residues, max_residue_atoms
    )

    usable_bytes = available_memory_mb * 1024 * 1024 * memory_fraction
    chunk_size = int(usable_bytes / bytes_per_frame)

    chunk_size = max(1, min(chunk_size, 10000))

    return chunk_size


def chunked_has_interactions_frames(
    lig_f: jnp.ndarray,
    res_f: jnp.ndarray,
    res_valid_mask: jnp.ndarray,
    lig_masks: dict,
    res_actor_masks: dict,
    angle_idx: dict,
    ring_idx: dict,
    vdw_radii: tuple,
    chunk_size: int | None = None,
) -> dict[str, jnp.ndarray]:
    """Evaluate interactions in memory-safe chunks.

    Automatically chunks large trajectories to avoid GPU OOM.
    Results are concatenated along the frame dimension.

    Args:
        lig_f: (F, N, 3) ligand coordinates.
        res_f: (F, R, M, 3) residue coordinates.
        res_valid_mask: (R, M) valid atom mask.
        lig_masks: SMARTS masks per interaction.
        res_actor_masks: Residue SMARTS masks.
        angle_idx: Angle indices dict.
        ring_idx: Ring indices dict.
        vdw_radii: VdW radii tuple.
        chunk_size: Frames per chunk. If None, auto-calculate.

    Returns:
        Dict of interaction name → (F, R) boolean arrays.
    """
    F = int(lig_f.shape[0])
    N = int(lig_f.shape[1])
    R = int(res_f.shape[1])
    M = int(res_f.shape[2])

    if chunk_size is None:
        chunk_size = calculate_chunk_size(N, R, M)

    dev = get_gpu_device()

    all_results = []
    for i in range(0, F, chunk_size):
        j = min(i + chunk_size, F)

        if dev is not None:
            chunk_lig = jax.device_put(lig_f[i:j], device=dev)
            chunk_res = jax.device_put(res_f[i:j], device=dev)
        else:
            chunk_lig = lig_f[i:j]
            chunk_res = res_f[i:j]

        chunk_result = has_interactions_frames(
            chunk_lig, chunk_res, res_valid_mask,
            lig_masks, res_actor_masks, angle_idx, ring_idx, vdw_radii
        )

        chunk_result = jax.device_get(chunk_result)
        all_results.append(chunk_result)

    if len(all_results) == 1:
        return all_results[0]

    combined = {}
    for key in all_results[0].keys():
        combined[key] = jnp.concatenate([r[key] for r in all_results], axis=0)

    return combined
