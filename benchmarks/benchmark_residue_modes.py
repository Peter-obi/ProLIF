"""
Benchmark comparing residue selection strategies.

Compares 3 scenarios:
1. ProLIF API: Fingerprint.run per-frame residue selection
2. JAX first: residue_mode="first"
3. JAX all: residue_mode="all" (pre-scan + evaluation)

Usage:
    python benchmarks/benchmark_residue_modes.py --frames 2000 --runs 3
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np


@dataclass
class BenchmarkResult:
    name: str
    frames: int
    residues: int
    runs: int
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float

    def __str__(self) -> str:
        return (
            f"{self.name:40s} | "
            f"{self.mean_ms:10.1f} +/- {self.std_ms:8.1f} ms | "
            f"min: {self.min_ms:10.1f} | "
            f"R={self.residues}"
        )


def prolif_api_benchmark(
    universe, lig_ag, prot_ag, cutoff: float, max_frames: int, runs: int
) -> BenchmarkResult:
    """ProLIF API using Fingerprint.run (per-frame selection)."""
    import prolif
    from prolif.utils import get_residues_near_ligand

    # Estimate residues from the first frame for reporting only (outside timing)
    universe.trajectory[0]
    lig0 = prolif.Molecule.from_mda(lig_ag)
    prot0 = prolif.Molecule.from_mda(prot_ag)
    r_first = len(get_residues_near_ligand(lig0, prot0, cutoff=cutoff))

    times = []
    for _ in range(runs):
        fp = prolif.Fingerprint(vicinity_cutoff=cutoff)
        t0 = time.perf_counter()
        fp.run(universe.trajectory[:max_frames], lig_ag, prot_ag, progress=False, n_jobs=1)
        times.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(times)
    return BenchmarkResult(
        name="ProLIF API (Fingerprint.run)",
        frames=max_frames,
        residues=r_first,
        runs=runs,
        mean_ms=float(arr.mean()),
        std_ms=float(arr.std()),
        min_ms=float(arr.min()),
        max_ms=float(arr.max()),
    )



def jax_first_benchmark(
    universe, cutoff: float, max_frames: int, runs: int, *, device: str = "cpu"
) -> BenchmarkResult:
    """JAX with residue_mode='first'."""
    from prolif.interactions._jax import analyze_trajectory

    times = []
    n_residues = 0

    for run in range(runs):
        universe.trajectory[0]
        t0 = time.perf_counter()

        result = analyze_trajectory(
            universe,
            ligand_selection="resname LIG",
            protein_selection="protein",
            cutoff=cutoff,
            max_frames=max_frames,
            device=device,
            residue_mode="first",
        )

        times.append((time.perf_counter() - t0) * 1000.0)
        n_residues = result.n_residues

    arr = np.array(times)
    return BenchmarkResult(
        name="JAX first (fixed residues)",
        frames=max_frames,
        residues=n_residues,
        runs=runs,
        mean_ms=float(arr.mean()),
        std_ms=float(arr.std()),
        min_ms=float(arr.min()),
        max_ms=float(arr.max()),
    )


def jax_all_benchmark(
    universe, cutoff: float, max_frames: int, runs: int, *, device: str = "cpu"
) -> BenchmarkResult:
    """JAX with residue_mode='all' (includes pre-scan time)."""
    from prolif.interactions._jax import analyze_trajectory

    times = []
    n_residues = 0

    for run in range(runs):
        universe.trajectory[0]
        t0 = time.perf_counter()

        result = analyze_trajectory(
            universe,
            ligand_selection="resname LIG",
            protein_selection="protein",
            cutoff=cutoff,
            max_frames=max_frames,
            device=device,
            residue_mode="all",
            scan_stride=1,
        )

        times.append((time.perf_counter() - t0) * 1000.0)
        n_residues = result.n_residues

    arr = np.array(times)
    return BenchmarkResult(
        name="JAX all (pre-scan + eval)",
        frames=max_frames,
        residues=n_residues,
        runs=runs,
        mean_ms=float(arr.mean()),
        std_ms=float(arr.std()),
        min_ms=float(arr.min()),
        max_ms=float(arr.max()),
    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark residue selection modes")
    parser.add_argument("--frames", type=int, default=2000, help="Number of frames")
    parser.add_argument("--runs", type=int, default=3, help="Number of timed runs")
    parser.add_argument("--cutoff", type=float, default=6.0, help="Residue cutoff (A)")
    parser.add_argument("--top", type=str, default=None, help="Topology file")
    parser.add_argument("--traj", type=str, default=None, help="Trajectory file")
    parser.add_argument("--gpu", action="store_true", help="Run JAX modes on GPU if available")
    args = parser.parse_args()

    import MDAnalysis as mda

    if args.top is None:
        top = "/Users/peterobi/Documents/ideas/ProLIF/benchmarks/MDR00020797/full.pdb"
        traj = "/Users/peterobi/Documents/ideas/ProLIF/benchmarks/MDR00020797/full.xtc"
    else:
        top = args.top
        traj = args.traj

    print(f"Loading {top} + {traj}...")
    u = mda.Universe(top, traj)
    print(f"Total frames: {len(u.trajectory)}, using first {args.frames}")

    lig_ag = u.select_atoms("resname LIG")
    prot_ag = u.select_atoms("protein")
    print(f"Ligand: {lig_ag.n_atoms} atoms, Protein: {prot_ag.n_atoms} atoms")
    print(f"Cutoff: {args.cutoff} A, Runs: {args.runs}")
    print()
    print("=" * 90)

    print("\n[1/3] Running ProLIF API (Fingerprint.run)...")
    r1 = prolif_api_benchmark(u, lig_ag, prot_ag, args.cutoff, args.frames, args.runs)
    print(r1)

    print("\n[2/3] Running JAX first (residue_mode='first')...")
    device = "gpu" if args.gpu else "cpu"
    r2 = jax_first_benchmark(u, args.cutoff, args.frames, args.runs, device=device)
    print(r2)

    print("\n[3/3] Running JAX all (residue_mode='all')...")
    r3 = jax_all_benchmark(u, args.cutoff, args.frames, args.runs, device=device)
    print(r3)

    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"{'Method':<42} | {'Time (ms)':>18} | {'Speedup':>10}")
    print("-" * 90)

    baseline = r1.mean_ms
    for r in [r1, r2, r3]:
        speedup = baseline / r.mean_ms
        print(f"{r.name:<42} | {r.mean_ms:>10.1f} +/- {r.std_ms:>5.1f} | {speedup:>10.1f}x")

    print("-" * 90)
    print(f"\nJAX first vs ProLIF default: {r1.mean_ms / r2.mean_ms:.1f}x speedup")
    print(f"JAX all vs ProLIF default: {r1.mean_ms / r3.mean_ms:.1f}x speedup")


if __name__ == "__main__":
    main()
