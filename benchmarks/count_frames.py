"""
Count frames in XTC trajectories under the benchmarks folder.

Usage:
    python benchmarks/count_frames.py
    python benchmarks/count_frames.py --top benchmarks/full.pdb --traj benchmarks/full.xtc
"""

from __future__ import annotations

import argparse
from pathlib import Path


def find_pairs(root: Path) -> list[tuple[Path, Path]]:
    """Find pairs of full.pdb/full.xtc under the given root directory.

    Returns a list of (topology, trajectory) path pairs in the same directory.
    """
    pairs: list[tuple[Path, Path]] = []
    by_dir: dict[Path, dict[str, Path]] = {}
    for p in root.rglob("full.pdb"):
        by_dir.setdefault(p.parent, {})["pdb"] = p
    for x in root.rglob("full.xtc"):
        by_dir.setdefault(x.parent, {})["xtc"] = x
    for d, m in by_dir.items():
        if "pdb" in m and "xtc" in m:
            pairs.append((m["pdb"], m["xtc"]))
    return pairs


def main() -> None:
    """Print frame counts for provided or discovered topology/trajectory pairs."""
    parser = argparse.ArgumentParser(description="Count frames in benchmarks trajectories")
    parser.add_argument("--top", type=str, default=None, help="Topology file path")
    parser.add_argument("--traj", type=str, default=None, help="Trajectory file path")
    args = parser.parse_args()

    try:
        import MDAnalysis as mda
    except Exception as e:  # pragma: no cover
        print(f"Failed to import MDAnalysis: {e}")
        return

    pairs: list[tuple[Path, Path]] = []
    if args.top and args.traj:
        pairs = [(Path(args.top), Path(args.traj))]
    else:
        root = Path("benchmarks")
        direct_top = root / "full.pdb"
        direct_traj = root / "full.xtc"
        if direct_top.exists() and direct_traj.exists():
            pairs = [(direct_top, direct_traj)]
        else:
            pairs = find_pairs(root)

    if not pairs:
        print("No full.pdb/full.xtc pairs found under benchmarks/.")
        return

    for top, traj in pairs:
        try:
            u = mda.Universe(str(top), str(traj))
            print(f"{top} | {traj} -> Frames: {len(u.trajectory)}")
        except Exception as e:
            print(f"Failed reading {top} + {traj}: {e}")


if __name__ == "__main__":
    main()
