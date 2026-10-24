#!/usr/bin/env python3
"""The Malloci+07 TD-DFT cross sections (``pahchem.malloci2007``): sum rule, mislabelled files, and Li & Draine.

A. Thomas-Reiche-Kuhn census: the oscillator strength to 30 eV of every state against its C 2s2p + H 1s electrons,
   and the step between adjacent charges (one electron each). Flags what breaks the pattern.
B. Berné+22's anion cross section: the per-C mean over their 10 molecules, with and without the mislabelled
   coronene anion file their code reads (``anions/coronene_anion.txt``).
C. Against ``photo.sigma_li_draine`` (what pahchem uses; anions take the neutral curve): Mathis-ISRF absorption
   rates in 1-5 and 5-13.6 eV, per molecule with N_C >= 20.

    calima-fetch-data fetch malloci2007-spectra
    check_malloci_sum_rule.py
"""
import numpy as np

from pycalima.pahchem import fields, photo
from pycalima.pahchem import malloci2007 as M

BERNE22 = ("coronene", "ovalene", "tetrabenzocoronene", "circumbiphenyl", "circumanthracene", "circumpyrene",
           "hexabenzocoronene", "dicoronylene", "circumcoronene", "circumovalene")


def main():
    every = M.spectra(include_inconsistent=True)
    print("A. oscillator strength to 30 eV, share of the valence electrons, and steps between charges")
    shares, steps = [], []
    for name, m in sorted(every.items(), key=lambda kv: kv[1]["nc"]):
        f = {z: M.oscillator_strength(*m["sigma"][z], 0, 30) for z in m["sigma"]}
        sh = {z: f[z] / M.valence_electrons(m["nc"], m["nh"], z) for z in f}
        st = {z: f[z] - f[z + 1] for z in f if z + 1 in f}
        flag = [f"{M.STATES[z]}: {M.INCONSISTENT[(name, z)]}" for z in f if (name, z) in M.INCONSISTENT]
        shares += [v for z, v in sh.items() if (name, z) not in M.INCONSISTENT]
        steps += [v for z, v in st.items() if (name, z) not in M.INCONSISTENT and (name, z + 1) not in M.INCONSISTENT]
        print(f"  {name[:24]:24s} C{m['nc']:<3d} f " + " ".join(f"{f[z]:6.1f}" for z in sorted(f))
              + "  share " + " ".join(f"{sh[z]:.2f}" for z in sorted(sh))
              + "  steps " + " ".join(f"{st[z]:5.1f}" for z in sorted(st)) + (f"  <- {'; '.join(flag)}" if flag else ""))
    print(f"  the rest: share {min(shares):.2f}-{max(shares):.2f}, steps {min(steps):.1f}-{max(steps):.1f} "
          f"(median {np.median(steps):.1f})")

    ma = fields.mathis1983(1.0, e_min=0.5)

    def rate(E, s, lo, hi):
        k = (E >= lo) & (E <= hi)
        return ma.photon_rate(s[k], E[k])

    E = every["ovalene"]["sigma"][0][0]
    per_c = lambda names: np.mean([every[n]["sigma"][-1][1] / every[n]["nc"] for n in names], axis=0)
    print("\nB. Berne+22 anion cross section (per-C mean of 10 molecules), with / without the coronene anion file:")
    for lo, hi in ((1, 5), (5, 13.6)):
        print(f"  Mathis-weighted {lo}-{hi} eV: x{rate(E, per_c(BERNE22), lo, hi) / rate(E, per_c(BERNE22[1:]), lo, hi):.3f}")

    good = M.spectra()
    print("\nC. Mathis-ISRF absorption rate / Li & Draine neutral (pahchem's anion and neutral curve), N_C >= 20:")
    rows = []
    for name, m in sorted(good.items(), key=lambda kv: kv[1]["nc"]):
        if m["nc"] < 20:
            continue
        row = []
        for lo, hi in ((1, 5), (5, 13.6)):
            ld = rate(E[1:], photo.sigma_li_draine(m["nc"], 0, E[1:]), lo, hi)        # E[0] = 0
            row += [rate(*m["sigma"][z], lo, hi) / ld if z in m["sigma"] else np.nan for z in (-1, 0, 1)]
        rows.append(row)
    r = np.nanmedian(np.array(rows), axis=0)
    print(f"  median over {len(rows)} molecules: 1-5 eV anion x{r[0]:.2f}, neutral x{r[1]:.2f}, cation x{r[2]:.2f}; "
          f"5-13.6 eV anion x{r[3]:.2f}, neutral x{r[4]:.2f}, cation x{r[5]:.2f}")


if __name__ == "__main__":
    main()
