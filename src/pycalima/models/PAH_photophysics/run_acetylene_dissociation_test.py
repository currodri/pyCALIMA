"""Run a focused PAH dissociation test and save outputs in one flat folder.

This script calls plot_acetylene_dissociation_rate and writes the generated
table/plot files to model_data/PAH_dissociation_data (no subfolders).
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path

from models.PAH_photophysics.PAH_photophysics import plot_acetylene_dissociation_rate


@contextmanager
def pushd(path: Path):
    """Temporarily change working directory."""
    import os

    old_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test PAH acetylene dissociation table/plot generation.",
    )
    parser.add_argument("--g0-min", type=float, default=1e-2, help="Minimum G0 value.")
    parser.add_argument("--g0-max", type=float, default=1e6, help="Maximum G0 value.")
    parser.add_argument("--nh-min", type=float, default=1e-2, help="Minimum nH value [cm^-3].")
    parser.add_argument("--nh-max", type=float, default=1e6, help="Maximum nH value [cm^-3].")
    parser.add_argument(
        "--pah-bin-id",
        type=str,
        default=None,
        help="PAH bin id from grain size config (optional).",
    )
    parser.add_argument(
        "--pah-bin-rank",
        type=int,
        default=0,
        help="PAH bin rank fallback when --pah-bin-id is not provided.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    output_dir = repo_root / "model_data" / "PAH_dissociation_data"
    output_dir.mkdir(parents=True, exist_ok=True)

    with pushd(output_dir):
        plot_acetylene_dissociation_rate(
            G0min=args.g0_min,
            G0max=args.g0_max,
            nHmin=args.nh_min,
            nHmax=args.nh_max,
            pah_bin_id=args.pah_bin_id,
            pah_bin_rank=args.pah_bin_rank,
        )

    produced = sorted(output_dir.glob("acetylene_dissociation_table_*.dat"))
    plot_file = output_dir / "C54_integrated_dissociation_rate.png"

    print(f"Output directory: {output_dir}")
    if produced:
        print("Dissociation table files:")
        for path in produced:
            print(f"  - {path.name}")
    else:
        print("No acetylene dissociation table file was detected.")

    print(f"Plot file exists: {plot_file.exists()} ({plot_file.name})")


if __name__ == "__main__":
    main()
