"""
Benchmark script for JAX-accelerated ProLIF.

This script compares the performance of:
1. Original ProLIF (NumPy/RDKit)
2. JAX CPU
3. JAX GPU (if available)

Usage:
    # Run all benchmarks
    python benchmarks/benchmark_jax.py

    # Run only CPU benchmarks
    python benchmarks/benchmark_jax.py --cpu-only

    # Run on Google Colab
    # 1. Upload this file and the prolif package
    # 2. !pip install prolif jax jaxlib
    # 3. !python benchmark_jax.py
"""

import argparse
import sys
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    name: str
    n_residues: int
    n_runs: int
    mean_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float

    def __str__(self) -> str:
        return (
            f"{self.name:25s} | "
            f"{self.mean_time_ms:8.3f} ± {self.std_time_ms:6.3f} ms | "
            f"min: {self.min_time_ms:7.3f} ms | "
            f"max: {self.max_time_ms:7.3f} ms"
        )


def validate_jax_vs_prolif(lig_mol, residues):
    """Compare JAX results against ProLIF to verify correctness.

    Returns dict with validation results for each interaction type.
    """
    from prolif.interactions import (
        Hydrophobic, Cationic, Anionic, HBAcceptor, HBDonor,
        XBAcceptor, XBDonor, CationPi, PiCation, FaceToFace,
        EdgeToFace, PiStacking, MetalDonor, MetalAcceptor, VdWContact,
    )
    from prolif.interactions._jax.integration import has_interaction_batch

    print("\nValidating JAX (with SMARTS) vs ProLIF results...")

    # All ProLIF interaction types
    prolif_interactions = {
        'Hydrophobic': Hydrophobic(),
        'Cationic': Cationic(),
        'Anionic': Anionic(),
        'HBAcceptor': HBAcceptor(),
        'HBDonor': HBDonor(),
        'XBAcceptor': XBAcceptor(),
        'XBDonor': XBDonor(),
        'CationPi': CationPi(),
        'PiCation': PiCation(),
        'FaceToFace': FaceToFace(),
        'EdgeToFace': EdgeToFace(),
        'PiStacking': PiStacking(),
        'MetalDonor': MetalDonor(),
        'MetalAcceptor': MetalAcceptor(),
        'VdWContact': VdWContact(),
    }

    validation = {}

    for itype, prolif_fn in prolif_interactions.items():
        # ProLIF results (one at a time)
        prolif_results = []
        for res in residues:
            result = prolif_fn.any(lig_mol, res)
            prolif_results.append(result is not None)

        # JAX batch results (using proper SMARTS integration)
        jax_results = has_interaction_batch(prolif_fn, lig_mol, residues)

        prolif_count = sum(prolif_results)
        jax_count = sum(jax_results)
        matches = sum(p == j for p, j in zip(prolif_results, jax_results))
        mismatches = len(residues) - matches

        validation[itype] = {
            'prolif_count': prolif_count,
            'jax_count': jax_count,
            'matches': matches,
            'mismatches': mismatches,
            'match_rate': matches / len(residues) * 100,
        }

        status = "✓" if mismatches == 0 else "✗"
        print(f"  {itype:15s}: ProLIF={prolif_count:2d}, JAX={jax_count:2d}, "
              f"Match={matches}/{len(residues)} ({validation[itype]['match_rate']:.1f}%) {status}")

    return validation


def load_test_system():
    """Load ProLIF's test protein-ligand system."""
    import MDAnalysis as mda
    import prolif
    from prolif.datafiles import TOP, TRAJ

    u = mda.Universe(TOP, TRAJ)
    prot = u.select_atoms("protein")
    lig = u.select_atoms("resname LIG")

    # Get first frame
    u.trajectory[0]

    # Convert to ProLIF molecules
    lig_mol = prolif.Molecule.from_mda(lig)
    prot_mol = prolif.Molecule.from_mda(prot)

    # Get residues near ligand
    from prolif.utils import get_residues_near_ligand
    residue_ids = get_residues_near_ligand(lig_mol, prot_mol, cutoff=6.0)
    residues = [prot_mol[rid] for rid in residue_ids]

    print(f"Loaded system: {len(lig_mol.GetAtoms())} ligand atoms, "
          f"{len(residues)} nearby residues")

    return lig_mol, residues, prot_mol


def benchmark_prolif_original(lig_mol, residues, n_runs: int = 100) -> BenchmarkResult:
    """Benchmark original ProLIF implementation with 9 interactions."""
    from prolif.interactions import (
        Hydrophobic, Cationic, Anionic, VdWContact, HBDonor, HBAcceptor,
        XBDonor, CationPi, PiStacking,
    )

    # All interaction types matching JAX implementation
    interactions = [
        Hydrophobic(),
        Cationic(),      # ionic (positive)
        Anionic(),       # ionic (negative)
        VdWContact(),
        HBDonor(),
        HBAcceptor(),
        XBDonor(),
        CationPi(),
        PiStacking(),
    ]

    # Warmup
    for res in residues[:3]:
        for interaction in interactions:
            _ = interaction(lig_mol, res)

    # Timed runs
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        for res in residues:
            for interaction in interactions:
                _ = interaction(lig_mol, res)
        times.append((time.perf_counter() - start) * 1000)  # Convert to ms

    times = np.array(times)
    return BenchmarkResult(
        name="ProLIF Original (9 types)",
        n_residues=len(residues),
        n_runs=n_runs,
        mean_time_ms=times.mean(),
        std_time_ms=times.std(),
        min_time_ms=times.min(),
        max_time_ms=times.max(),
    )


def benchmark_jax_cpu(lig_mol, residues, n_runs: int = 100) -> BenchmarkResult:
    """Benchmark JAX implementation on CPU."""
    import jax
    jax.config.update('jax_platform_name', 'cpu')

    from prolif.interactions._jax import JAXAccelerator

    # Align to the same 9 interactions as ProLIF baseline
    accel = JAXAccelerator(interactions=[
        'Hydrophobic', 'Cationic', 'Anionic', 'VdWContact',
        'HBDonor', 'HBAcceptor', 'XBDonor', 'CationPi', 'PiStacking',
    ])

    # Warmup (includes JIT compilation)
    _ = accel.compute_interactions(lig_mol, residues)
    _ = accel.compute_interactions(lig_mol, residues)

    # Timed runs
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = accel.compute_interactions(lig_mol, residues)
        times.append((time.perf_counter() - start) * 1000)

    times = np.array(times)
    return BenchmarkResult(
        name="JAX CPU",
        n_residues=len(residues),
        n_runs=n_runs,
        mean_time_ms=times.mean(),
        std_time_ms=times.std(),
        min_time_ms=times.min(),
        max_time_ms=times.max(),
    )


def benchmark_jax_gpu(lig_mol, residues, n_runs: int = 100) -> Optional[BenchmarkResult]:
    """Benchmark JAX implementation on GPU."""
    import jax

    # Check GPU availability
    try:
        jax.config.update('jax_platform_name', 'gpu')
        devices = jax.devices('gpu')
        if not devices:
            print("No GPU devices found")
            return None
        print(f"GPU detected: {devices[0]}")
    except Exception as e:
        print(f"GPU not available: {e}")
        return None

    from prolif.interactions._jax import JAXAccelerator

    # Align to the same 9 interactions as ProLIF baseline
    accel = JAXAccelerator(interactions=[
        'Hydrophobic', 'Cationic', 'Anionic', 'VdWContact',
        'HBDonor', 'HBAcceptor', 'XBDonor', 'CationPi', 'PiStacking',
    ])

    # Warmup (includes JIT compilation and GPU transfer)
    _ = accel.compute_interactions(lig_mol, residues)
    _ = accel.compute_interactions(lig_mol, residues)
    _ = accel.compute_interactions(lig_mol, residues)

    # Timed runs
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = accel.compute_interactions(lig_mol, residues)
        # Block until computation is complete
        jax.block_until_ready(_)
        times.append((time.perf_counter() - start) * 1000)

    times = np.array(times)
    return BenchmarkResult(
        name="JAX GPU",
        n_residues=len(residues),
        n_runs=n_runs,
        mean_time_ms=times.mean(),
        std_time_ms=times.std(),
        min_time_ms=times.min(),
        max_time_ms=times.max(),
    )


def _tile_residues(residues, n: int):
    """Return a list of length n by repeating residues as needed."""
    if n <= len(residues):
        return residues[:n]
    reps = (n + len(residues) - 1) // len(residues)
    tiled = (residues * reps)[:n]
    return tiled


def benchmark_scaling(lig_mol, residues, backend: str = 'cpu', counts: list[int] | None = None) -> list[BenchmarkResult]:
    """Benchmark scaling with different numbers of residues."""
    import jax

    if backend == 'gpu':
        try:
            jax.config.update('jax_platform_name', 'gpu')
        except Exception:
            print("GPU not available, falling back to CPU")
            jax.config.update('jax_platform_name', 'cpu')
    else:
        jax.config.update('jax_platform_name', 'cpu')

    from prolif.interactions._jax import JAXAccelerator

    accel = JAXAccelerator(interactions=['Hydrophobic'])

    results = []
    n_residue_counts = counts or [1, 5, 10, 20, len(residues)]

    for n in n_residue_counts:
        subset = _tile_residues(residues, n)

        # Warmup
        _ = accel.compute_interactions(lig_mol, subset)
        _ = accel.compute_interactions(lig_mol, subset)

        # Timed runs
        times = []
        n_runs = 50
        for _ in range(n_runs):
            start = time.perf_counter()
            result = accel.compute_interactions(lig_mol, subset)
            if backend == 'gpu':
                jax.block_until_ready(result)
            times.append((time.perf_counter() - start) * 1000)

        times = np.array(times)
        results.append(BenchmarkResult(
            name=f"JAX {backend.upper()} ({n} residues)",
            n_residues=n,
            n_runs=n_runs,
            mean_time_ms=times.mean(),
            std_time_ms=times.std(),
            min_time_ms=times.min(),
            max_time_ms=times.max(),
        ))

    return results


def print_header():
    """Print benchmark header."""
    print("=" * 80)
    print("ProLIF JAX Acceleration Benchmark")
    print("=" * 80)
    print()


def print_results(results: list[BenchmarkResult]):
    """Print benchmark results in a formatted table."""
    print()
    print("-" * 80)
    print(f"{'Backend':25s} | {'Mean ± Std':20s} | {'Min':12s} | {'Max':12s}")
    print("-" * 80)
    for r in results:
        print(r)
    print("-" * 80)


def main():
    parser = argparse.ArgumentParser(description="Benchmark JAX-accelerated ProLIF")
    parser.add_argument('--cpu-only', action='store_true',
                        help="Run only CPU benchmarks")
    parser.add_argument('--gpu-only', action='store_true',
                        help="Run only GPU benchmarks")
    parser.add_argument('--scaling', action='store_true',
                        help="Run scaling benchmarks")
    parser.add_argument('--scaling-list', type=str, default=None,
                        help="Comma-separated residue counts for scaling (e.g., 1,5,10,20,50,100)")
    parser.add_argument('--n-runs', type=int, default=100,
                        help="Number of runs per benchmark (default: 100)")
    args = parser.parse_args()

    print_header()

    # Check JAX availability
    try:
        import jax
        print(f"JAX version: {jax.__version__}")
        print(f"JAX devices: {jax.devices()}")
    except ImportError:
        print("ERROR: JAX not installed. Install with: pip install jax jaxlib")
        sys.exit(1)

    print()

    # Load test system
    print("Loading test system...")
    lig_mol, residues, prot_mol = load_test_system()

    # Validate JAX vs ProLIF results
    validate_jax_vs_prolif(lig_mol, residues)
    print()

    results = []

    # Original ProLIF benchmark
    if not args.gpu_only:
        print("Running ProLIF original benchmark...")
        results.append(benchmark_prolif_original(lig_mol, residues, args.n_runs))

    # JAX CPU benchmark
    if not args.gpu_only:
        print("Running JAX CPU benchmark...")
        results.append(benchmark_jax_cpu(lig_mol, residues, args.n_runs))

    # JAX GPU benchmark
    if not args.cpu_only:
        print("Running JAX GPU benchmark...")
        gpu_result = benchmark_jax_gpu(lig_mol, residues, args.n_runs)
        if gpu_result:
            results.append(gpu_result)

    print_results(results)

    # Calculate speedups
    if len(results) >= 2:
        print()
        print("Speedups:")
        baseline = results[0].mean_time_ms
        for r in results[1:]:
            speedup = baseline / r.mean_time_ms
            print(f"  {r.name}: {speedup:.2f}x vs {results[0].name}")

    # Scaling benchmark
    if args.scaling:
        print()
        print("=" * 80)
        print("Scaling Benchmark")
        print("=" * 80)

        # Parse custom scaling counts if provided
        counts = None
        if args.scaling_list:
            try:
                counts = [int(x.strip()) for x in args.scaling_list.split(',') if x.strip()]
            except Exception:
                print("Invalid --scaling-list; falling back to defaults")
                counts = None

        if not args.gpu_only:
            print("\nCPU Scaling:")
            scaling_results = benchmark_scaling(lig_mol, residues, 'cpu', counts)
            print_results(scaling_results)

        if not args.cpu_only:
            print("\nGPU Scaling:")
            scaling_results = benchmark_scaling(lig_mol, residues, 'gpu', counts)
            print_results(scaling_results)


if __name__ == "__main__":
    main()
