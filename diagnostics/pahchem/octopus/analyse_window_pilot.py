#!/usr/bin/env python3
"""The Coulomb-window pilot's yields (``make_window_pilot.py`` runs): electrons lost through the mask per photon absorbed.

For each pulse run ``td_<axis>_<omega>``: N(t) is the electron number in the box (the per-spin charge columns of
``td.general/multipoles``), so N(0) - N(t) is what the mask has removed. The energy the field gives the electrons is
W(t) = -int E(t) . d<r>/dt dt (``td.general/laser`` and the electronic dipole <r> = int r n, e = 1, eV and A): the
photons absorbed are W / omega, and their ratio is the yield within the run. After the pulse (t > 2 tau) N keeps falling
while the electrons excited into the molecule's bound states reach the mask or do not: its slope is their escape in time.
The yield is the x and z runs combined over the orientations, (2 Y_x N_x + Y_z N_z)-style: lost and absorbed summed
over x, y (= x for D6h coronene) and z.

    analyse_window_pilot.py RUNDIR [--tau 10]
"""
import argparse
from pathlib import Path

import numpy as np


def read_table(path):
    """(header lines, data) of an Octopus td.general file."""
    lines = Path(path).read_text().splitlines()
    head = [ln for ln in lines if ln.startswith("#")]
    data = np.array([[float(x) for x in ln.split()] for ln in lines if ln.strip() and not ln.startswith("#")])
    return head, data


def run_yield(d, omega):
    """(t, lost(t), photons absorbed(t)) of one pulse run directory ``d``."""
    head, m = read_table(d / "td.general" / "multipoles")
    nspin = int(next(h for h in head if "nspin" in h).split()[-1])
    t = m[:, 1]
    cols = 4                                                 # per spin: charge, <x>, <y>, <z> (lmax 1)
    charge = sum(m[:, 2 + cols * s] for s in range(nspin))
    dip = sum(m[:, 3 + cols * s: 6 + cols * s] for s in range(nspin))
    _, las = read_table(d / "td.general" / "laser")
    E = np.interp(t, las[:, 1], las[:, 2]), np.interp(t, las[:, 1], las[:, 3]), np.interp(t, las[:, 1], las[:, 4])
    E = np.array(E).T
    ddip = np.gradient(dip, t, axis=0)
    power = -(E * ddip).sum(axis=1)
    work = np.concatenate([[0.0], np.cumsum(0.5 * (power[1:] + power[:-1]) * np.diff(t))])
    return t, charge[0] - charge, work / omega


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("rundir", type=Path)
    ap.add_argument("--tau", type=float, default=10.0, help="pulse half-width [hbar/eV]: the pulse ends at 2 tau")
    a = ap.parse_args()
    for box in sorted(a.rundir.glob("box_*")):
        print(f"{box.name}:")
        by_omega = {}
        for d in sorted(box.glob("td_*_*")):
            if not (d / "td.general" / "multipoles").exists():
                continue
            axis, omega = d.name.split("_")[1], float(d.name.split("_")[2])
            t, lost, absorbed = run_yield(d, omega)
            end = t >= 2 * a.tau
            slope = np.polyfit(t[end], lost[end], 1)[0] if end.sum() > 10 else np.nan
            print(f"  {d.name}: t to {t[-1]:.1f} hbar/eV, lost {lost[-1]:.3e}, absorbed {absorbed[-1]:.3e} photons, "
                  f"yield {lost[-1] / absorbed[-1]:.3f}; after the pulse lost/t {slope:.2e} per hbar/eV "
                  f"({lost[-1] - lost[end][0] if end.any() else np.nan:.2e} of it)")
            w = 2.0 if axis in ("x", "y") else 1.0           # x stands for y too (D6h)
            acc = by_omega.setdefault(omega, [0.0, 0.0])
            acc[0] += w * lost[-1]
            acc[1] += w * absorbed[-1]
        for omega, (lost, absorbed) in sorted(by_omega.items()):
            print(f"  {omega:g} eV, orientation average: yield {lost / absorbed:.3f}")


if __name__ == "__main__":
    main()
