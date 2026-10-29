"""How much does the band-position choice matter for pahchem's rates? (2026-10-28)

The final library's neutrals (``build_library.py``), each with the part below 7 eV moved rigidly up by a shift
(its f is kept; where it lands on the unmoved part above 7 eV the two add). Rates of the moved over the unmoved
spectrum in the Mathis+83 field (0.1-13.6 eV), Draine 1978, Kurucz 10 000 K and 6000 K blackbody fields, and the mean
absorbed Mathis photon energy. Shifts: +0.3, +0.5, +0.6 eV, and 0.5 +- 0.1 (the spread of the per-band shifts of
pyrene's sTDA test), i.e. what per-band against uniform can change.

    PYTHONPATH=diagnostics/pahchem python diagnostics/pahchem/octopus/shift_sensitivity.py
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "diagnostics/pahchem/octopus"))
from plot_dial3_library import isrf_stats  # noqa: E402

from pycalima.pahchem import fields  # noqa: E402

CUT = 7.0
LIB = REPO / "results/pahchem/octopus_library/pah_sigma_library.npz"


def moved(E, s, d):
    """The part of ``s`` below CUT moved up by d [eV]; the rest stays."""
    low = np.where(E < CUT, s, 0.0)
    shifted = np.interp(E - d, E, low, left=0.0, right=0.0)
    return shifted + np.where(E >= CUT, s, 0.0)


def rate(f, E, s):
    e = f.energy_ev
    return np.trapezoid(np.clip(np.interp(e, E, s, left=0.0, right=0.0), 0, None) * f.flux_at(e), e)


def main():
    z = np.load(LIB)
    E = z["energy_ev"]
    k = (E <= 20.0)
    fl = {"Mathis": fields.mathis1983(e_min=0.1, n=2000), "Draine78": fields.draine1978(),
          "Kurucz 10kK": fields.kurucz(10000, 1.0), "BB 6000K": fields.blackbody(6000, 1.0)}
    neutral = [i for i in range(len(z["z"])) if z["z"][i] == 0]
    shifts = (0.3, 0.4, 0.5, 0.6)
    print(f"{len(neutral)} neutrals; part below {CUT} eV moved up by d; rate(moved)/rate(unmoved), min..max over the neutrals")
    print(f"{'d [eV]':>7s} " + " ".join(f"{n:>16s}" for n in fl) + "   mean Mathis photon [eV]  (unmoved: "
          + f"{np.mean([isrf_stats(E[k], z['sigma_cm2'][i][k])[0] for i in neutral]):.2f})")
    for d in shifts:
        cols = []
        for n, f in fl.items():
            r = [rate(f, E[k], moved(E[k], z["sigma_cm2"][i][k], d)) / rate(f, E[k], z["sigma_cm2"][i][k])
                 for i in neutral]
            cols.append(f"{min(r):.3f}..{max(r):.3f}")
        mp = np.mean([isrf_stats(E[k], moved(E[k], z["sigma_cm2"][i][k], d))[0] for i in neutral])
        print(f"{d:7.2f} " + " ".join(f"{c:>16s}" for c in cols) + f"   {mp:.2f}")
    print("per-band against uniform: 0.5 vs 0.4 and 0.6 (rate ratio of the two, min..max)")
    for n, f in fl.items():
        for a, b in ((0.5, 0.4), (0.5, 0.6)):
            r = [rate(f, E[k], moved(E[k], z["sigma_cm2"][i][k], a)) / rate(f, E[k], moved(E[k], z["sigma_cm2"][i][k], b))
                 for i in neutral]
            print(f"  {n:12s} {a} / {b}: {min(r):.3f}..{max(r):.3f}")


if __name__ == "__main__":
    main()
