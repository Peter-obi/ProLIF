"""
Benchmark frame-batched JAX geometry vs ProLIF per-frame checks.

Usage:
    python benchmarks/benchmark_frames.py --frames 100 --runs 3 [--gpu]
    python benchmarks/benchmark_frames.py --top full.pdb --traj full.xtc --real --runs 3 [--gpu]

Notes:
    - Uses duplicated frame-0 coordinates to isolate geometry throughput.
    - Benchmarks 9 interactions:
      Hydrophobic, Cationic, Anionic, VdWContact, HBAcceptor, HBDonor,
      XBAcceptor, XBDonor, PiStacking (composite of FaceToFace/EdgeToFace),
      and CationPi.
    - JAX path:
      - Frame-batched for distance-only + HB/XB using new helpers
      - For ring interactions (PiStacking, CationPi), falls back to ProLIF
        per-frame loop (structure-only indices, geometry varies with frames)
    - ProLIF path: per-frame `.any(...)` loop for all interactions.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class BenchmarkResult:
    name: str
    frames: int
    runs: int
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float

    def __str__(self) -> str:
        return (
            f"{self.name:28s} | "
            f"{self.mean_ms:8.3f} ± {self.std_ms:6.3f} ms | "
            f"min: {self.min_ms:7.3f} | max: {self.max_ms:7.3f}"
        )


def load_system_and_residues(top: str | None = None, traj: list[str] | None = None,
                             ligsel: str = "resname LIG", protsel: str = "protein",
                             cutoff: float = 6.0):
    """Load system and select ligand/protein with nearby residues.

    Args:
        top: Topology path; if None, use ProLIF built-in dataset.
        traj: Optional trajectory files; accepts multiple segments.
        ligsel: MDAnalysis selection string for ligand.
        protsel: MDAnalysis selection string for protein.
        cutoff: Distance cutoff in Å for selecting nearby residues.

    Returns:
        Tuple containing the Universe, ligand AtomGroup, protein AtomGroup,
        ProLIF ligand molecule, list of ProLIF protein residues near the
        ligand, and MDAnalysis residue AtomGroups corresponding to those residues.
    """
    import MDAnalysis as mda
    import prolif
    from prolif.datafiles import TOP, TRAJ
    from prolif.utils import get_residues_near_ligand

    if top is None:
        u = mda.Universe(TOP, TRAJ)
    else:
        u = mda.Universe(top, *(traj or []))

    try:
        u.atoms.guess_bonds()
    except Exception:
        pass

    prot = u.select_atoms(protsel)
    lig = u.select_atoms(ligsel)
    u.trajectory[0]

    lig_mol = prolif.Molecule.from_mda(lig)
    prot_mol = prolif.Molecule.from_mda(prot)
    residue_ids = get_residues_near_ligand(lig_mol, prot_mol, cutoff=cutoff)
    residues = [prot_mol[rid] for rid in residue_ids]
    residue_ags = [prot.select_atoms(f"resid {rid}") for rid in residue_ids]
    return u, lig, prot, lig_mol, residues, residue_ags


def build_duplicate_frames(lig_mol, residues, frames: int):
    import jax.numpy as jnp

    lig0 = jnp.array(lig_mol.xyz)
    lig_f = jnp.tile(lig0[None, ...], (frames, 1, 1))

    max_m = max(r.GetNumAtoms() for r in residues) if residues else 0
    res_rows = []
    res_masks = []
    for r in residues:
        xyz = jnp.array(r.xyz)
        m = xyz.shape[0]
        pad = max_m - m
        padded = jnp.concatenate([xyz, jnp.zeros((pad, 3))], axis=0) if pad else xyz
        res_rows.append(jnp.tile(padded[None, ...], (frames, 1, 1)))
        rm = jnp.concatenate([jnp.ones((m,), dtype=bool), jnp.zeros((pad,), dtype=bool)])
        res_masks.append(rm)
    res_f = jnp.stack(res_rows, axis=1) if res_rows else jnp.zeros((frames, 0, 0, 3))
    res_mask = jnp.stack(res_masks, axis=0) if res_masks else jnp.zeros((0, 0), dtype=bool)
    return lig_f, res_f, res_mask


def build_trajectory_frames(u, lig_ag, residue_ags, max_frames: int | None = None):
    """Build per-frame coordinates from the trajectory.

    Args:
        u: MDAnalysis Universe (with trajectory loaded)
        lig_ag: ligand AtomGroup
        residue_ags: list of residue AtomGroups (subset used for evaluation)
        max_frames: optional cap on number of frames to use

    Returns:
        lig_f: (F, N, 3)
        res_f: (F, R, M, 3) padded per residue to max M
        res_valid_mask: (R, M) boolean mask of real atoms
    """
    import numpy as np
    import jax.numpy as jnp

    F_total = len(u.trajectory)
    F = F_total if not max_frames or max_frames <= 0 else min(F_total, int(max_frames))

    N = lig_ag.n_atoms
    R = len(residue_ags)
    max_m = max((r.n_atoms for r in residue_ags), default=0)

    lig_frames = []
    res_frames = []
    res_masks = []
    for r in residue_ags:
        m = r.n_atoms
        rm = np.zeros((max_m,), dtype=bool)
        rm[:m] = True
        res_masks.append(rm)
    res_valid_mask = jnp.array(np.stack(res_masks, axis=0) if res_masks else np.zeros((0, 0), dtype=bool))

    for _ in u.trajectory[:F]:
        lig_frames.append(np.array(lig_ag.positions, dtype=float).reshape(N, 3))
        row_list = []
        for r in residue_ags:
            coords = np.array(r.positions, dtype=float)
            m = coords.shape[0]
            if m < max_m:
                pad = np.zeros((max_m - m, 3), dtype=float)
                coords = np.concatenate([coords, pad], axis=0)
            row_list.append(coords)
        res_frames.append(np.stack(row_list, axis=0) if row_list else np.zeros((0, 0, 3), dtype=float))

    lig_f = jnp.array(np.stack(lig_frames, axis=0) if lig_frames else np.zeros((0, N, 3), dtype=float))
    res_f = jnp.array(np.stack(res_frames, axis=0) if res_frames else np.zeros((0, R, max_m, 3), dtype=float))
    return lig_f, res_f, res_valid_mask


def build_actor_masks(lig_mol, residues):
    """Compute boolean masks for distance-only actor atoms.

    Returns ligand and residue masks for Hydrophobic, Cationic, Anionic,
    MetalDonor, and MetalAcceptor patterns using SMARTS on ProLIF molecules.
    Masks for residues are padded to a common length.
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
    R = len(residues)
    max_m = max(r.GetNumAtoms() for r in residues) if residues else 0

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

    Returns a mapping with ligand acceptor indices and per-residue donor/H
    indices for hydrogen bonds, and ligand acceptor/neighbor plus per-residue
    halogen/donor indices for halogen bonds.
    """
    import jax.numpy as jnp
    from prolif.interactions import HBAcceptor, XBAcceptor

    hb = HBAcceptor()
    xb = XBAcceptor()

    hb_acc_idx = []
    lmatches = lig_mol.GetSubstructMatches(hb.lig_pattern)
    if lmatches:
        hb_acc_idx = [m[0] for m in lmatches]
    hb_acc_idx = jnp.array(hb_acc_idx, dtype=int) if hb_acc_idx else jnp.zeros((0,), dtype=int)

    hb_d_rows, hb_h_rows = [], []
    for r in residues:
        pmatches = r.GetSubstructMatches(hb.prot_pattern)
        pairs = [(m[0], m[1]) for m in pmatches] if pmatches else []
        d = jnp.array([p[0] for p in pairs], dtype=int) if pairs else jnp.zeros((0,), dtype=int)
        h = jnp.array([p[1] for p in pairs], dtype=int) if pairs else jnp.zeros((0,), dtype=int)
        hb_d_rows.append(d)
        hb_h_rows.append(h)

    xb_a_rows, xb_r_rows = [], []
    lmatches = lig_mol.GetSubstructMatches(xb.lig_pattern)
    a = [m[0] for m in lmatches] if lmatches else []
    r = [m[1] for m in lmatches] if lmatches else []
    xbacc_a_idx = jnp.array(a, dtype=int) if a else jnp.zeros((0,), dtype=int)
    xbacc_r_idx = jnp.array(r, dtype=int) if r else jnp.zeros((0,), dtype=int)

    xb_x_rows, xb_d_rows = [], []
    for res in residues:
        pmatches = res.GetSubstructMatches(xb.prot_pattern)
        pairs = [(m[1], m[0]) for m in pmatches] if pmatches else []
        x = jnp.array([p[0] for p in pairs], dtype=int) if pairs else jnp.zeros((0,), dtype=int)
        d = jnp.array([p[1] for p in pairs], dtype=int) if pairs else jnp.zeros((0,), dtype=int)
        xb_x_rows.append(x)
        xb_d_rows.append(d)

    return {
        'hb': {'acc_idx': hb_acc_idx, 'res_d_idx': hb_d_rows, 'res_h_idx': hb_h_rows},
        'xbacc': {'lig_a_idx': xbacc_a_idx, 'lig_r_idx': xbacc_r_idx, 'res_x_idx': xb_x_rows, 'res_d_idx': xb_d_rows},
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


def jax_benchmark(
    lig_f,
    res_f,
    res_valid_mask,
    lig_masks,
    res_actor_masks,
    angle_idx,
    ring_idx,
    vdw_radii,
    runs: int,
    use_gpu: bool = False,
) -> BenchmarkResult:
    """Benchmark JAX frame-batched geometry for nine interactions.

    Places arrays on GPU when requested and times the end-to-end evaluation,
    synchronizing with block_until_ready to measure device execution.
    """
    import jax
    import jax.numpy as jnp
    from prolif.interactions._jax.framebatch import (
        pairwise_distances_frames,
        hbacceptor_frames,
        hbdonor_frames,
        xbacceptor_frames,
        xbdonor_frames,
        cationpi_frames,
        pistacking_frames,
    )

    if use_gpu:
        gpus = jax.devices("gpu")
        if gpus:
            dev = gpus[0]
            lig_f = jax.device_put(lig_f, device=dev)
            res_f = jax.device_put(res_f, device=dev)
            res_valid_mask = jax.device_put(res_valid_mask, device=dev)
            lig_masks = jax.tree_util.tree_map(lambda x: jax.device_put(x, device=dev), lig_masks)
            res_actor_masks = jax.tree_util.tree_map(lambda x: jax.device_put(x, device=dev), res_actor_masks)
            angle_idx = {
                'hb': {
                    'acc_idx': jax.device_put(angle_idx['hb']['acc_idx'], device=dev),
                    'res_d_idx': [jax.device_put(a, device=dev) for a in angle_idx['hb']['res_d_idx']],
                    'res_h_idx': [jax.device_put(a, device=dev) for a in angle_idx['hb']['res_h_idx']],
                },
                'xbacc': {
                    'lig_a_idx': jax.device_put(angle_idx['xbacc']['lig_a_idx'], device=dev),
                    'lig_r_idx': jax.device_put(angle_idx['xbacc']['lig_r_idx'], device=dev),
                    'res_x_idx': [jax.device_put(a, device=dev) for a in angle_idx['xbacc']['res_x_idx']],
                    'res_d_idx': [jax.device_put(a, device=dev) for a in angle_idx['xbacc']['res_d_idx']],
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
            lig_radii, res_radii, vdw_tol = vdw_radii
            vdw_radii = (
                jax.device_put(lig_radii, device=dev),
                jax.device_put(res_radii, device=dev),
                vdw_tol,
            )
        else:
            use_gpu = False

    d = pairwise_distances_frames(lig_f, res_f)
    _ = d[0, 0, 0, 0].block_until_ready()

    times = []
    lig_radii, res_radii, vdw_tol = vdw_radii

    for _ in range(runs):
        t0 = time.perf_counter()
        d = pairwise_distances_frames(lig_f, res_f)
        lm = lig_masks['Hydrophobic']
        rm = res_actor_masks['Hydrophobic'] & res_valid_mask
        mask = (d <= 4.5) & lm[None, None, :, None] & rm[None, :, None, :]
        last = jnp.any(mask, axis=(2, 3))
        for k in ('Cationic', 'Anionic'):
            lm = lig_masks[k]
            rm = res_actor_masks[k] & res_valid_mask
            mask = (d <= 4.5) & lm[None, None, :, None] & rm[None, :, None, :]
            last = jnp.any(mask, axis=(2, 3))
        for k in ('MetalDonor', 'MetalAcceptor'):
            lm = lig_masks[k]
            rm = res_actor_masks[k] & res_valid_mask
            mask = (d <= 2.8) & lm[None, None, :, None] & rm[None, :, None, :]
            last = jnp.any(mask, axis=(2, 3))
        radii_sum = lig_radii[None, None, :, None] + res_radii[None, :, None, :]
        mask = (d <= (radii_sum + vdw_tol)) & res_valid_mask[None, :, None, :]
        last = jnp.any(mask, axis=(2, 3))

        acc_idx = angle_idx['hb']['acc_idx']
        for r_i in range(res_f.shape[1]):
            d_idx = angle_idx['hb']['res_d_idx'][r_i]
            h_idx = angle_idx['hb']['res_h_idx'][r_i]
            if acc_idx.size and d_idx.size:
                _m, _d, _a = hbacceptor_frames(lig_f, res_f[:, r_i, :, :], acc_idx, d_idx, h_idx)
                last = jnp.any(_m, axis=(1, 2))
            if d_idx.size:
                pass

        for r_i in range(res_f.shape[1]):
            a_idx = angle_idx['xbacc']['lig_a_idx']
            r_idx = angle_idx['xbacc']['lig_r_idx']
            x_idx = angle_idx['xbacc']['res_x_idx'][r_i]
            dd_idx = angle_idx['xbacc']['res_d_idx'][r_i]
            if a_idx.size and x_idx.size:
                _m, _d, _axd, _xar = xbacceptor_frames(lig_f, res_f[:, r_i, :, :], a_idx, r_idx, x_idx, dd_idx)
                last = jnp.any(_m, axis=(1, 2))
            if x_idx.size:
                _m, _d, _axd, _xar = xbdonor_frames(lig_f, res_f[:, r_i, :, :], x_idx, dd_idx, a_idx, r_idx)
                last = jnp.any(_m, axis=(1, 2))

        lig_rings = ring_idx['lig_rings']
        res_rings = ring_idx['res_rings']
        lig_cations = ring_idx['lig_cations']
        res_cations = ring_idx['res_cations']
        pi = ring_idx['pi']
        cp = ring_idx['cp']

        for r_i in range(res_f.shape[1]):
            if lig_cations.size and len(res_rings[r_i]):
                _m1, _d1, _a1 = cationpi_frames(
                    res_f[:, r_i, :, :], res_rings[r_i], lig_f, lig_cations,
                    distance_cutoff=float(cp.distance),
                    angle_min=float(cp.angle[0]), angle_max=float(cp.angle[1]),
                )
                last = jnp.any(_m1, axis=(1, 2))
            if len(lig_rings) and res_cations[r_i].size:
                _m2, _d2, _a2 = cationpi_frames(
                    lig_f, lig_rings, res_f[:, r_i, :, :], res_cations[r_i],
                    distance_cutoff=float(cp.distance),
                    angle_min=float(cp.angle[0]), angle_max=float(cp.angle[1]),
                )
                last = jnp.any(_m2, axis=(1, 2))

            if len(lig_rings) and len(res_rings[r_i]):
                ftf = pi.ftf
                _mF, _dF, _paF, _nccF = pistacking_frames(
                    lig_f, lig_rings, res_f[:, r_i, :, :], res_rings[r_i],
                    distance_cutoff=float(ftf.distance),
                    plane_angle_min=float(ftf.plane_angle[0]), plane_angle_max=float(ftf.plane_angle[1]),
                    ncc_angle_min=float(ftf.normal_to_centroid_angle[0]), ncc_angle_max=float(ftf.normal_to_centroid_angle[1]),
                )
                etf = pi.etf
                _mE, _dE, _paE, _nccE = pistacking_frames(
                    lig_f, lig_rings, res_f[:, r_i, :, :], res_rings[r_i],
                    distance_cutoff=float(etf.distance),
                    plane_angle_min=float(etf.plane_angle[0]), plane_angle_max=float(etf.plane_angle[1]),
                    ncc_angle_min=float(etf.normal_to_centroid_angle[0]), ncc_angle_max=float(etf.normal_to_centroid_angle[1]),
                )
                last = jnp.any(_mF | _mE, axis=(1, 2))
        _ = last.block_until_ready()
        times.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(times)
    return BenchmarkResult(
        name="JAX frame-batched (9 interactions)",
        frames=int(lig_f.shape[0]),
        runs=runs,
        mean_ms=float(arr.mean()),
        std_ms=float(arr.std()),
        min_ms=float(arr.min()),
        max_ms=float(arr.max()),
    )


def prolif_benchmark(lig_mol, residues, frames: int, runs: int) -> BenchmarkResult:
    """Benchmark ProLIF per-frame interaction checks across nine interactions."""
    from prolif.interactions import (
        Hydrophobic, Cationic, Anionic, VdWContact,
        HBAcceptor, HBDonor, XBAcceptor, XBDonor,
        PiStacking, CationPi,
    )

    inters = [
        Hydrophobic(), Cationic(), Anionic(), VdWContact(),
        HBAcceptor(), HBDonor(), XBAcceptor(), XBDonor(),
        PiStacking(), CationPi(),
    ]

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        for _f in range(frames):
            for res in residues:
                for inter in inters:
                    _ = inter.any(lig_mol, res)
        times.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(times)
    return BenchmarkResult(
        name="ProLIF per-frame (distance)",
        frames=frames,
        runs=runs,
        mean_ms=float(arr.mean()),
        std_ms=float(arr.std()),
        min_ms=float(arr.min()),
        max_ms=float(arr.max()),
    )


def build_vdw_radii(lig_mol, residues):
    """Build per-atom van der Waals radii arrays.

    Returns ligand radii (N,), residue radii (R, M) padded to max M, and
    ProLIF's VdW tolerance.
    """
    import jax.numpy as jnp
    from prolif.interactions import VdWContact

    vdw = VdWContact()
    N = lig_mol.GetNumAtoms()
    lig_elems = [lig_mol.GetAtomWithIdx(i).GetSymbol() for i in range(N)]
    lig_radii = jnp.array([vdw.vdwradii.get(e, 1.7) for e in lig_elems], dtype=float)

    R = len(residues)
    max_m = max(r.GetNumAtoms() for r in residues) if residues else 0
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


def main():
    parser = argparse.ArgumentParser(description="Frame-batched JAX vs ProLIF benchmark (9 interactions)")
    parser.add_argument('--frames', type=int, default=100, help='Number of frames. With --real, caps frames used.')
    parser.add_argument('--runs', type=int, default=3, help='Number of timed runs')
    parser.add_argument('--gpu', action='store_true', help='Run JAX path on GPU (device_put + sync timing)')
    parser.add_argument('--top', type=str, default=None, help='Topology file path (e.g., PDB/PRMTOP/GRO)')
    parser.add_argument('--traj', type=str, nargs='*', default=None, help='Trajectory file(s) (e.g., XTC/DCD)')
    parser.add_argument('--ligsel', type=str, default='resname LIG', help='MDAnalysis ligand selection string')
    parser.add_argument('--protsel', type=str, default='protein', help='MDAnalysis protein selection string')
    parser.add_argument('--cutoff', type=float, default=6.0, help='Residues within cutoff of ligand (A)')
    parser.add_argument('--real', action='store_true', help='Use real frames from trajectory instead of duplicating frame 0')
    args = parser.parse_args()

    u, lig_ag, prot_ag, lig_mol, residues, residue_ags = load_system_and_residues(
        args.top, args.traj, args.ligsel, args.protsel, args.cutoff
    )

    if args.real:
        lig_f, res_f, res_valid_mask = build_trajectory_frames(u, lig_ag, residue_ags, max_frames=args.frames)
    else:
        lig_f, res_f, res_valid_mask = build_duplicate_frames(lig_mol, residues, int(args.frames))
    F = int(lig_f.shape[0])
    lig_masks, res_actor_masks = build_actor_masks(lig_mol, residues)

    angle_idx = build_angle_indices(lig_mol, residues)
    ring_idx = build_ring_cation_indices(lig_mol, residues)
    vdw_radii = build_vdw_radii(lig_mol, residues)
    jax_res = jax_benchmark(
        lig_f, res_f, res_valid_mask, lig_masks, res_actor_masks, angle_idx, ring_idx, vdw_radii, args.runs, use_gpu=bool(args.gpu)
    )
    prolif_res = prolif_benchmark(lig_mol, residues, F, args.runs)

    print("\nFrame-batched benchmark (9 interactions; all frame-batched in JAX path):")
    print("-" * 88)
    print(jax_res)
    print(prolif_res)
    print("-" * 88)
    print(f"Speedup: {prolif_res.mean_ms / jax_res.mean_ms:.2f}x (JAX vs ProLIF)")


if __name__ == "__main__":
    main()
