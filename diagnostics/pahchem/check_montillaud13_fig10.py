#!/usr/bin/env python3
"""Montillaud+13 Fig. 10 for C54H18: where 10/50/90 % is totally dehydrogenated.

Runs pahchem.montillaud13 on a (G0, nH) grid at their Fig. 10 conditions
(Mathis ISRF, T = 100 K, n(H) = 0.5 nH, n_e = 1.4e-4 nH) and prints, for each
digitised contour point, log G0(ours) - log G0(theirs) at the same nH.

Measured (2026-10-04), median offset (range):

| Contour | ``--modes parent`` | ``--modes stripped`` |
|---|---|---|
| 10 % | +1.5 | +0.60 (+0.39 to +0.89) |
| 50 % | +1.2 | +0.35 (-0.26 to +0.56) |
| 90 % | +1.0 | +0.00 (-0.64 to +0.12) |

With the parent's modes, every state dissociates like C54H18 (210 modes); the
even-N_H steps are then ~100x too slow. Stripping C-H modes per missing H
closes most of the gap. The rest awaits DFT modes of each state
(pahchem-data-requests, T1), Malloci's C54 sigma, and cation IR intensities.
Takes ~1 min with parent modes and ~10 min with stripped modes.
"""
import argparse
from pathlib import Path

import numpy as np

from pycalima import _paths
from pycalima.models.PAH_photophysics.PAH_photophysics import read_sections
from pycalima.pahchem.molecule import PAHSpecies
from pycalima.pahchem.montillaud13 import Montillaud13


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--modes", choices=("parent", "stripped"), default="stripped")
    ap.add_argument("--states-dir", type=Path, default=_paths.get_model_data_dir() / "PAH_states")
    ap.add_argument("--de-cm", type=float, default=100.0)
    args = ap.parse_args()

    sp = PAHSpecies.from_states_file(args.states_dir / "C54H18_0.dat")
    model = Montillaud13(sp, "C54", modes=args.modes, de_cm=args.de_cm)
    g0 = 10 ** np.arange(0.0, 4.01, 0.25)
    nh = 10 ** np.arange(1.0, 5.01, 0.25)
    F = model.hydrogenation_grid(g0, nh)
    contours = read_sections("montillaud13_hydrogenation_fraction_C54.csv")
    for level, (x, y) in zip((0.1, 0.5, 0.9), contours):
        offsets = []
        for G, n in zip(np.array(x, float), np.array(y, float)):
            if not nh[0] <= n <= nh[-1]:
                continue
            j = np.interp(np.log10(n), np.log10(nh), np.arange(nh.size))
            column = np.array([np.interp(j, np.arange(nh.size), F[i]) for i in range(g0.size)])
            if column.min() < level < column.max():
                offsets.append(np.interp(level, column, np.log10(g0)) - np.log10(G))
        offsets = np.array(offsets)
        print(f"{level:.0%} contour: log G0 offset median {np.median(offsets):+.2f} "
              f"(range {offsets.min():+.2f} to {offsets.max():+.2f}, {offsets.size} points)")


if __name__ == "__main__":
    main()
