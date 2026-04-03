#!/usr/bin/env python3
"""Smoke-test runner for dust photoelectric heating efficiency plots.

This script calls `plot_efficiency` from `dust_photoelectric_heating` using
small defaults so it is practical for quick validation runs.
"""

import argparse
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run plot_efficiency smoke test and verify output artifacts."
    )
    parser.add_argument("--nsizes", type=int, default=8, help="Number of grain sizes")
    return parser.parse_args()


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    args = parse_args()

    # Import after sys.path setup.
    from models.dust_charge import dust_photoelectric_heating as dph

    # WNM reference conditions used for the Weingartner & Draine 2001 comparison.
    T = 6000.0
    ne = 0.03
    radiation_model = "Mathis"
    G0factor = 1.0

    import matplotlib as mpl
    import matplotlib.pyplot as plt

    # Many environments running tests do not have a LaTeX installation.
    plt.rcParams["text.usetex"] = False
    # Use bundled fonts to avoid repeated font fallback warnings.
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["font.serif"] = ["DejaVu Serif"]
    mpl.set_loglevel("error")

    external_data_dir = os.path.join(repo_root, "external_data")
    out_dir = os.path.join(repo_root, "model_data", "dust_photoelectric_heating_data")
    out_eff = os.path.join(out_dir, f"dust_photoelectric_heating_efficiency_{radiation_model}.pdf")
    out_pot = os.path.join(out_dir, f"dust_photoelectric_heating_potential_{radiation_model}.pdf")

    cwd_before = os.getcwd()
    try:
        # plot_efficiency currently reads comparison CSV files using relative names.
        os.chdir(external_data_dir)

        print("[test_plot_efficiency] Running with:")
        print(f"  T={T}")
        print(f"  ne={ne}")
        print(f"  radiation_model={radiation_model}")
        print(f"  G0factor={G0factor}")
        print(f"  nsizes={args.nsizes}")

        dph.plot_efficiency(
            T=T,
            ne=ne,
            radiation_model=radiation_model,
            G0factor=G0factor,
            nsizes=args.nsizes,
        )
    finally:
        os.chdir(cwd_before)

    missing = [p for p in (out_eff, out_pot) if not os.path.exists(p)]
    if missing:
        print("[test_plot_efficiency] Missing expected output files:")
        for path in missing:
            print(f"  - {path}")
        raise SystemExit(1)

    print("[test_plot_efficiency] Success. Generated:")
    print(f"  - {out_eff}")
    print(f"  - {out_pot}")


if __name__ == "__main__":
    main()
