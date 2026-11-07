#!/usr/bin/env python3
"""Finished non-compact isomer spectra of DIaL3's ``library_small`` against the compact molecule of the same size and charge in the
library (TD-LDA positions, the isomers' own): per-carbon cross section 0-25 eV and the oscillator strength per carbon below 30 eV.

    check_isomer_spectra.py [--root ~/Documents/RAMSES_dev/pahspec/dial3/library_small] [--out DIR]
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pycalima import _paths
from pycalima.pahchem import uvlib
from pycalima.plotting_style import use_calima_style

MB_EV_PER_F = 109.761                         # Mb eV per unit oscillator strength
A2_TO_CM2 = 1e-16


def f_per_c(e, s, e_max=30.0):
    k = e <= e_max
    return np.trapezoid(s[k], e[k]) / 1e-18 / MB_EV_PER_F


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=Path.home() / "Documents/RAMSES_dev/pahspec/dial3/library_small")
    ap.add_argument("--out", type=Path, default=_paths.get_plots_dir("pahchem"))
    a = ap.parse_args()
    runs = sorted(d for d in a.root.iterdir() if (d / "spectrum" / "cross_section_tensor").exists())
    if not runs:
        raise SystemExit(f"no finished spectra under {a.root}")
    use_calima_style()
    fig, axs = plt.subplots(1, len(runs), figsize=(4.6 * len(runs), 3.6), squeeze=False)
    for ax, d in zip(axs[0], runs):
        label, charge = d.name.rsplit("_", 1)
        nc, z = int(label.split("H")[0][1:]), int(charge)
        t = np.loadtxt(d / "spectrum" / "cross_section_tensor", comments="#")
        e, s = t[:, 0], t[:, 1] * A2_TO_CM2 / nc
        ref = uvlib.per_carbon(nc, z, e, "tdlda", "compact")
        ax.plot(e, s * 1e18, color="C0", label=f"{label}, isomer")
        ax.plot(e, ref * 1e18, color="0.4", ls="--", label=f"compact C$_{{{nc}}}$ of the library")
        ax.set_xlim(0, 25)
        ax.set_xlabel("photon energy [eV]")
        ax.set_ylabel(r"cross section per carbon [Mb]" if ax is axs[0, 0] else "")
        ax.set_title(f"{label} (charge {z:+d})")
        ax.legend(frameon=False)
        ax.minorticks_on()
        ax.tick_params(which="both", direction="in", top=True, right=True)
        print(f"{d.name:20s} f/C below 30 eV: isomer {f_per_c(e, s):.3f}  compact {f_per_c(e, ref):.3f}   "
              f"s.max isomer {s[(e > 1) & (e < 8)].max() * 1e18:.2f} Mb, compact {ref[(e > 1) & (e < 8)].max() * 1e18:.2f} Mb")
    a.out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(a.out / "isomer_spectra_check.png", dpi=150)
    print("wrote", a.out / "isomer_spectra_check.png")


if __name__ == "__main__":
    main()
