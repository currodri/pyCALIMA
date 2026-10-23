#!/usr/bin/env python3
"""An Octopus spectrum (``make_run.py``) against the Malloci+07 spectrum of the same molecule and charge.

Reads ``<run>/spectrum/cross_section_tensor`` (energy [eV], (1/3) Tr sigma [A^2], ...) and compares:
- the oscillator strength to 30 eV and its share of the valence electrons (``malloci2007.oscillator_strength``);
- the integrals over 1-5, 5-13.6 and 13.6-30 eV;
- the strongest peak in 5-8 eV and in 14-20 eV;
- the Mathis-ISRF absorption rate in 1-5 and 5-13.6 eV.

    compare_malloci.py <run dir> --name coronene --charge 0 --nc 24 --nh 12
"""
import argparse
from pathlib import Path

import numpy as np

from pycalima.pahchem import fields
from pycalima.pahchem import malloci2007 as M

A2_CM2 = 1e-16


def read_octopus(run):
    d = np.loadtxt(Path(run) / "spectrum" / "cross_section_tensor", comments="#")
    return d[:, 0], d[:, 1] * A2_CM2


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run", type=Path)
    ap.add_argument("--name", required=True)
    ap.add_argument("--charge", type=int, default=0)
    ap.add_argument("--nc", type=int, required=True)
    ap.add_argument("--nh", type=int, required=True)
    a = ap.parse_args()
    ours = read_octopus(a.run)
    theirs = M.spectra()[a.name]["sigma"].get(a.charge)
    if theirs is None:
        raise SystemExit(f"no usable Malloci spectrum for {a.name} Z = {a.charge} (malloci2007.INCONSISTENT?)")
    ma = fields.mathis1983(1.0, e_min=0.5)
    n_val = M.valence_electrons(a.nc, a.nh, a.charge)

    def summary(E, s):
        k = E > 0
        E, s = E[k], s[k]
        out = {"f30": M.oscillator_strength(E, s, 0, 30)}
        out["share"] = out["f30"] / n_val
        for lo, hi in ((1, 5), (5, 13.6), (13.6, 30)):
            out[f"f {lo}-{hi}"] = M.oscillator_strength(E, s, lo, hi)
        for lo, hi in ((5, 8), (14, 20)):
            w = (E >= lo) & (E <= hi)
            out[f"peak {lo}-{hi} eV"] = E[w][np.argmax(s[w])]
        for lo, hi in ((1, 5), (5, 13.6)):
            w = (E >= lo) & (E <= hi)
            out[f"Mathis {lo}-{hi}"] = ma.photon_rate(s[w], E[w])
        return out

    o, t = summary(*ours), summary(*theirs)
    print(f"{a.name} Z = {a.charge:+d} ({n_val} valence electrons): Octopus run {a.run} against Malloci+07")
    for key in o:
        ratio = "" if key.startswith("peak") else f"   ratio {o[key] / t[key]:.3f}"
        print(f"  {key:16s} ours {o[key]:10.4g}   Malloci {t[key]:10.4g}{ratio}")


if __name__ == "__main__":
    main()
