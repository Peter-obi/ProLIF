"""
Benchmark frame-batched JAX geometry vs ProLIF per-frame checks (default 9 interactions).

Usage:
    python benchmarks/benchmark_frames.py --top full.pdb --traj full.xtc --runs 3 [--gpu]

Notes:
    - Consumes all frames from the provided trajectory.
    - Evaluated interactions (9):
      Hydrophobic, Cationic, Anionic, VdWContact, HBAcceptor, HBDonor,
      PiStacking (FaceToFace OR EdgeToFace), CationPi, PiCation.
    - JAX path: frame-batched for distances, HB angles, and ring geometry.
    - ProLIF path: per-frame `.any(...)` loop for all interactions.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from prolif.interactions._jax import (
    has_interactions_frames as jax_has_interactions_frames,
    build_actor_masks as jax_build_actor_masks,
    build_angle_indices as jax_build_angle_indices,
    build_ring_cation_indices as jax_build_ring_cation_indices,
    build_vdw_radii as jax_build_vdw_radii,
    prepare_for_device,
    get_gpu_device,
    calculate_chunk_size,
)


@dataclass
class BenchmarkResult:
    name: str
    frames: int
    runs: int
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    mem_peak_mb: float | None = None
    mem_total_mb: float | None = None

    def __str__(self) -> str:
        base = (
            f"{self.name:28s} | "
            f"{self.mean_ms:8.3f} ± {self.std_ms:6.3f} ms | "
            f"min: {self.min_ms:7.3f} | max: {self.max_ms:7.3f}"
        )
        if self.mem_peak_mb is not None and self.mem_total_mb is not None:
            base += f" | GPU mem: {self.mem_peak_mb:.0f}/{self.mem_total_mb:.0f} MiB"
        return base


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
    residue_ags = []
    for rid in residue_ids:
        name = rid.name or ""
        num = rid.number or 0
        chain = rid.chain or ""
        sel_core = []
        if name:
            sel_core.append(f"resname {name}")
        if num:
            sel_core.append(f"resid {num}")
        sel = " and ".join(sel_core) if sel_core else f"resid {num}"
        ag = prot.select_atoms(sel)
        if ag.n_atoms == 0 and chain:
            try_sel = f"{sel} and chainID {chain}"
            ag = prot.select_atoms(try_sel)
        if ag.n_atoms == 0 and chain:
            try_sel = f"{sel} and segid {chain}"
            ag = prot.select_atoms(try_sel)
        if ag.n_atoms == 0:
            ag = prot.select_atoms(sel)
        residue_ags.append(ag)
    return u, lig, prot, lig_mol, residues, residue_ags



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



def _complete_angle_indices(angle_idx: dict, residues: list) -> dict:
    """Ensure angle index mapping contains donor variants expected by JAX helpers.

    Some environments may provide an older build of the angle index helpers that
    only include acceptor-side mappings (e.g., 'hb', 'xbacc'). This function
    fills in missing donor-side mappings ('hb_donor', 'xbdon') with empty
    arrays/lists so downstream code can run without KeyError and simply find no
    donor-side matches.
    """
    import jax.numpy as jnp

    R = len(residues)
    if 'hb_donor' not in angle_idx:
        angle_idx['hb_donor'] = {
            'lig_d_idx': jnp.zeros((0,), dtype=int),
            'lig_h_idx': jnp.zeros((0,), dtype=int),
            'res_a_idx': [jnp.zeros((0,), dtype=int) for _ in range(R)],
        }
    if 'xbacc' not in angle_idx:
        angle_idx['xbacc'] = {
            'lig_a_idx': jnp.zeros((0,), dtype=int),
            'lig_r_idx': jnp.zeros((0,), dtype=int),
            'res_x_idx': [jnp.zeros((0,), dtype=int) for _ in range(R)],
            'res_d_idx': [jnp.zeros((0,), dtype=int) for _ in range(R)],
        }
    if 'xbdon' not in angle_idx:
        angle_idx['xbdon'] = {
            'lig_x_idx': jnp.zeros((0,), dtype=int),
            'lig_d_idx': jnp.zeros((0,), dtype=int),
            'res_a_idx': [jnp.zeros((0,), dtype=int) for _ in range(R)],
            'res_r_idx': [jnp.zeros((0,), dtype=int) for _ in range(R)],
        }
    return angle_idx

def jax_benchmark(lig_f, res_f, res_valid_mask, lig_masks, res_actor_masks, angle_idx, ring_idx, vdw_radii, runs: int, use_gpu: bool = False, chunk_size: int | None = None) -> BenchmarkResult:
    """Benchmark JAX frame-batched geometry for nine interactions.

    Places arrays on GPU when requested and times the end-to-end evaluation,
    synchronizing with block_until_ready to measure device execution.
    """
    import jax

    # Move small, structure-only arrays to device
    device = 'gpu' if use_gpu else 'cpu'
    lig_f, res_f, res_valid_mask, lig_masks, res_actor_masks, angle_idx, ring_idx, vdw_radii = prepare_for_device(
        lig_f, res_f, res_valid_mask, lig_masks, res_actor_masks, angle_idx, ring_idx, vdw_radii,
        device=device,
    )

    # Check if GPU is actually available
    dev = get_gpu_device()
    if use_gpu and dev is None:
        use_gpu = False

    # Timing loop
    times = []
    mem_peak = None
    mem_total = None

    # Determine chunk size for GPU path
    N = int(lig_f.shape[1])
    R = int(res_f.shape[1])
    M = int(res_f.shape[2])
    step = None
    if use_gpu:
        if chunk_size is not None and chunk_size > 0:
            step = int(chunk_size)
        else:
            step = calculate_chunk_size(N, R, M)

    for _ in range(runs):
        t0 = time.perf_counter()
        if use_gpu:
            F = int(lig_f.shape[0])
            for i in range(0, F, step):
                j = min(i + step, F)
                chunk_l = jax.device_put(lig_f[i:j], device=dev)
                chunk_r = jax.device_put(res_f[i:j], device=dev)
                res = jax_has_interactions_frames(
                    chunk_l, chunk_r, res_valid_mask, lig_masks, res_actor_masks, angle_idx, ring_idx, vdw_radii
                )
                last = next(iter(res.values()))
                _ = last.block_until_ready()
                try:
                    from prolif.interactions._jax import get_gpu_memory_info
                    mem = get_gpu_memory_info()
                    if mem is not None:
                        free_mb, total_mb = mem
                        used_mb = total_mb - free_mb
                        mem_peak = max(mem_peak or 0.0, used_mb)
                        mem_total = total_mb
                except Exception:
                    pass
        else:
            res = jax_has_interactions_frames(
                lig_f, res_f, res_valid_mask, lig_masks, res_actor_masks, angle_idx, ring_idx, vdw_radii
            )
            last = next(iter(res.values()))
            _ = last.block_until_ready()
        times.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(times)
    return BenchmarkResult(
        name='JAX frame-batched (9 interactions)',
        frames=int(lig_f.shape[0]),
        runs=runs,
        mean_ms=float(arr.mean()),
        std_ms=float(arr.std()),
        min_ms=float(arr.min()),
        max_ms=float(arr.max()),
        mem_peak_mb=float(mem_peak) if mem_peak is not None else None,
        mem_total_mb=float(mem_total) if mem_total is not None else None,
    )


def prolif_benchmark(lig_mol, residues, frames: int, runs: int) -> BenchmarkResult:
    """Benchmark ProLIF per-frame interaction checks across nine interactions."""
    from prolif.interactions import (
        Hydrophobic, Cationic, Anionic, VdWContact,
        HBAcceptor, HBDonor, PiStacking, CationPi, PiCation,
    )

    inters = [
        Hydrophobic(), Cationic(), Anionic(), VdWContact(),
        HBAcceptor(), HBDonor(),
        PiStacking(), CationPi(), PiCation(),
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
        name="ProLIF per-frame (9 interactions)",
        frames=frames,
        runs=runs,
        mean_ms=float(arr.mean()),
        std_ms=float(arr.std()),
        min_ms=float(arr.min()),
        max_ms=float(arr.max()),
    )




def jax_boolean_results(
    lig_f,
    res_f,
    res_valid_mask,
    lig_masks,
    res_actor_masks,
    angle_idx,
    ring_idx,
    vdw_radii,
):
    """Compute per-interaction boolean results using the JAX frame-batched path.

    Returns a mapping from interaction name to a boolean array of shape (F, R)
    indicating whether that interaction occurs for each frame and residue.
    """
    import jax.numpy as jnp
    from prolif.interactions._jax.framebatch import pairwise_distances_frames

    F = int(lig_f.shape[0])
    R = int(res_f.shape[1]) if res_f.ndim == 4 else 0
    results = {}

    d = pairwise_distances_frames(lig_f, res_f)

    hyd_mask = (d <= 4.5)
    m = hyd_mask & lig_masks['Hydrophobic'][None, None, :, None] & (res_actor_masks['Hydrophobic'] & res_valid_mask)[None, :, None, :]
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
            _m, _d, _a = hbacceptor_frames(lig_f, res_f[:, r_i, :, :], acc_idx, d_idx, h_idx)
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
            _m, _d, _a = hbdonor_frames(lig_f, res_f[:, r_i, :, :], lig_d, lig_h, acc)
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
            _m, _d, _a = cationpi_frames(
                res_f[:, r_i, :, :], res_rings[r_i], lig_f, lig_cations,
                distance_cutoff=float(cp.distance),
                angle_min=float(cp.angle[0]), angle_max=float(cp.angle[1]),
            )
            has_cationpi = jnp.any(_m, axis=(1, 2))
        if len(lig_rings) and res_cations[r_i].size:
            _m, _d, _a = cationpi_frames(
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


def prolif_boolean_results(u, lig_ag, residue_ags, max_frames: int | None = None):
    """Compute per-interaction boolean results using ProLIF per frame.

    Rebuilds ProLIF Molecule objects for ligand and residues at each frame and
    evaluates the nine interactions with the standard API.
    """
    import numpy as np
    import prolif
    from prolif.interactions import (
        Hydrophobic, Cationic, Anionic, VdWContact,
        HBAcceptor, HBDonor, PiStacking, CationPi, PiCation,
    )

    inters = [
        Hydrophobic(), Cationic(), Anionic(), VdWContact(),
        HBAcceptor(), HBDonor(),
        PiStacking(), CationPi(), PiCation(),
    ]
    names = [
        'Hydrophobic', 'Cationic', 'Anionic', 'VdWContact',
        'HBAcceptor', 'HBDonor', 'PiStacking', 'CationPi', 'PiCation',
    ]

    F_total = len(u.trajectory)
    F = F_total if not max_frames or max_frames <= 0 else min(F_total, int(max_frames))
    R = len(residue_ags)
    out = {n: np.zeros((F, R), dtype=bool) for n in names}

    for f_i, ts in enumerate(u.trajectory[:F]):
        _ = ts
        lig_mol = prolif.Molecule.from_mda(lig_ag)
        res_mols = [prolif.Molecule.from_mda(ag) for ag in residue_ags]
        for r_i, res_mol in enumerate(res_mols):
            for inter, name in zip(inters, names):
                out[name][f_i, r_i] = bool(inter.any(lig_mol, res_mol))
    return out


def prolif_boolean_results_fixed(u, lig_ag, residue_ags, max_frames: int | None = None):
    """Compute per-interaction booleans using fixed RDKit connectivity.

    Builds ligand and residue RDKit molecules once from the first frame and
    only updates coordinates for each subsequent frame before evaluating the
    nine interactions.
    """
    import numpy as np
    import prolif
    from rdkit.Geometry import Point3D
    from prolif.interactions import (
        Hydrophobic, Cationic, Anionic, VdWContact,
        HBAcceptor, HBDonor, PiStacking, CationPi, PiCation,
    )

    inters = [
        Hydrophobic(), Cationic(), Anionic(), VdWContact(),
        HBAcceptor(), HBDonor(),
        PiStacking(), CationPi(), PiCation(),
    ]
    names = [
        'Hydrophobic', 'Cationic', 'Anionic', 'VdWContact',
        'HBAcceptor', 'HBDonor', 'PiStacking', 'CationPi', 'PiCation',
    ]

    F_total = len(u.trajectory)
    F = F_total if not max_frames or max_frames <= 0 else min(F_total, int(max_frames))
    R = len(residue_ags)
    out = {n: np.zeros((F, R), dtype=bool) for n in names}

    lig_mol = prolif.Molecule.from_mda(lig_ag)
    res_mols = [prolif.Molecule.from_mda(ag) for ag in residue_ags]

    def _update_coords(mol, coords):
        conf = mol.GetConformer()
        for i in range(coords.shape[0]):
            x, y, z = float(coords[i, 0]), float(coords[i, 1]), float(coords[i, 2])
            conf.SetAtomPosition(i, Point3D(x, y, z))

    for f_i, ts in enumerate(u.trajectory[:F]):
        _ = ts
        _update_coords(lig_mol, lig_ag.positions)
        for r_i, ag in enumerate(residue_ags):
            _update_coords(res_mols[r_i], ag.positions)
            for inter, name in zip(inters, names):
                out[name][f_i, r_i] = bool(inter.any(lig_mol, res_mols[r_i]))
    return out



def summarize_accuracy(jax_out, pl_out):
    """Compute simple accuracy per interaction between JAX and ProLIF outputs."""
    import numpy as np

    names = [
        'Hydrophobic', 'Cationic', 'Anionic', 'VdWContact',
        'HBAcceptor', 'HBDonor', 'PiStacking', 'CationPi', 'PiCation',
    ]
    summary = {}
    for n in names:
        if n in jax_out and n in pl_out:
            a = np.asarray(jax_out[n], dtype=bool)
            b = np.asarray(pl_out[n], dtype=bool)
            if a.shape != b.shape:
                summary[n] = float('nan')
            else:
                summary[n] = float((a == b).mean())
    return summary

def main():
    parser = argparse.ArgumentParser(description="Frame-batched JAX vs ProLIF benchmark (9 interactions)")
    parser.add_argument('--runs', type=int, default=3, help='Number of timed runs')
    parser.add_argument('--max-frames', type=int, default=None, help='Cap frames processed from the trajectory')
    parser.add_argument('--gpu', action='store_true', help='Run JAX path on GPU (device_put + sync timing)')
    parser.add_argument('--chunk-size', '--chunk_size', type=int, default=None, help='Override GPU frames-per-chunk (auto by default)')
    parser.add_argument('--top', type=str, default=None, help='Topology file path (e.g., PDB/PRMTOP/GRO)')
    parser.add_argument('--traj', type=str, nargs='*', default=None, help='Trajectory file(s) (e.g., XTC/DCD)')
    parser.add_argument('--ligsel', type=str, default='resname LIG', help='MDAnalysis ligand selection string')
    parser.add_argument('--protsel', type=str, default='protein', help='MDAnalysis protein selection string')
    parser.add_argument('--cutoff', type=float, default=6.0, help='Residues within cutoff of ligand (A)')
    parser.add_argument('--debug-hb', action='store_true', help='Print details for HBDonor mismatches')
    parser.add_argument('--fixed-connectivity', action='store_true', help='Freeze RDKit connectivity; update only coordinates per frame')
    args = parser.parse_args()

    u, lig_ag, prot_ag, lig_mol, residues, residue_ags = load_system_and_residues(
        args.top, args.traj, args.ligsel, args.protsel, args.cutoff
    )

    lig_f, res_f, res_valid_mask = build_trajectory_frames(u, lig_ag, residue_ags, max_frames=args.max_frames)
    F = int(lig_f.shape[0])
    lig_masks, res_actor_masks = jax_build_actor_masks(lig_mol, residues)

    angle_idx = jax_build_angle_indices(lig_ag, residue_ags)
    angle_idx = _complete_angle_indices(angle_idx, residues)
    ring_idx = jax_build_ring_cation_indices(lig_mol, residues)
    vdw_radii = jax_build_vdw_radii(lig_mol, residues, lig_ag=lig_ag, residue_ags=residue_ags, use_real=True)
    chunk_override = args.chunk_size
    if args.gpu and (args.max_frames is not None) and (chunk_override is None or chunk_override <= 0):
        # If user capped frames and did not set chunk size, process in one chunk (skip mem query)
        chunk_override = F

    jax_res = jax_benchmark(
        lig_f, res_f, res_valid_mask, lig_masks, res_actor_masks, angle_idx, ring_idx, vdw_radii, args.runs, use_gpu=bool(args.gpu), chunk_size=chunk_override
    )
    prolif_res = prolif_benchmark(lig_mol, residues, F, args.runs)

    jax_out = jax_has_interactions_frames(lig_f, res_f, res_valid_mask, lig_masks, res_actor_masks, angle_idx, ring_idx, vdw_radii)
    if args.fixed_connectivity:
        pl_out = prolif_boolean_results_fixed(u, lig_ag, residue_ags, max_frames=F)
    else:
        pl_out = prolif_boolean_results(u, lig_ag, residue_ags, max_frames=F)
    acc = summarize_accuracy(jax_out, pl_out)

    print("\nFrame-batched benchmark (9 interactions; all frame-batched in JAX path):")
    print(f"Frames processed: {F}")
    print("-" * 88)
    print(jax_res)
    print(prolif_res)
    print("-" * 88)
    print(f"Speedup: {prolif_res.mean_ms / jax_res.mean_ms:.2f}x (JAX vs ProLIF)")
    if acc:
        acc_str = ", ".join([f"{k}: {v*100:.1f}%" for k, v in acc.items()])
        print(f"Accuracy (JAX vs ProLIF): {acc_str}")
    if args.debug_hb:
        import numpy as np
        hb_j = np.asarray(jax_out.get('HBDonor'), dtype=bool)
        hb_p = np.asarray(pl_out.get('HBDonor'), dtype=bool)
        mism = np.argwhere(hb_j != hb_p)
        print(f"HBDonor mismatches: {len(mism)}")
        if len(mism):
            from prolif.interactions._jax.framebatch import hbdonor_frames
            lig_d = angle_idx['hb_donor']['lig_d_idx']
            lig_h = angle_idx['hb_donor']['lig_h_idx']
            from prolif.interactions import HBDonor
            hb = HBDonor()
            shown = 0
            for f_i, r_i in mism:
                acc_idx = angle_idx['hb_donor']['res_a_idx'][r_i]
                m, d, a = hbdonor_frames(
                    lig_f, res_f[:, r_i, :, :], lig_d, lig_h, acc_idx,
                    distance_cutoff=float(hb.distance),
                    dha_angle_min=float(hb.angle[0]), dha_angle_max=float(hb.angle[1]),
                )
                dmin = float(np.min(np.asarray(d[f_i])) if d.shape[-1] else float('inf'))
                amax = float(np.max(np.asarray(a[f_i])) if a.shape[-1] else float('-inf'))
                # Also check ProLIF-detected pairs for this frame/residue
                import prolif as _pl
                u.trajectory[int(f_i)]
                lig_mol = _pl.Molecule.from_mda(lig_ag)
                res_mol = _pl.Molecule.from_mda(residue_ags[int(r_i)])
                pl_pairs = list(hb.detect(lig_mol, res_mol))
                npairs = len(pl_pairs)
                print(f"  Frame {int(f_i)}, Resid {int(r_i)} | JAX={hb_j[f_i,r_i]} ProLIF={hb_p[f_i,r_i]} | min d={dmin:.3f}A, max ang={amax:.2f}deg | ProLIF pairs={npairs}")
                if npairs and shown < 10:
                    for k, meta in enumerate(pl_pairs[:3]):
                        print(f"    ProLIF pair {k}: distance={meta.get('distance'):.3f}A, angle={meta.get('DHA_angle'):.2f}deg | idxs={meta['indices']}")
                shown += 1
                if shown >= 50:
                    break


if __name__ == "__main__":
    main()
