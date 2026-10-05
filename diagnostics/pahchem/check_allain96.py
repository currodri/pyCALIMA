#!/usr/bin/env python3
"""Allain+96 (I and II) through pahchem.allain96: every tabulated comparison.

Prints dex = log10(pahchem / Allain) for:
- I Tables 4 and 5 (Set 1, Draine);
- II Tables 2 and 3 (Set 2, digitised from I Fig. 5);
- II Table 5 destruction times in the six media.

It also prints our alpha_ion and alpha_H (the targets of II Figs. 7 and 10,
which are not digitised) and the critical sizes (II Sect. 4, Fig. 15).

Measured (2026-10-05; Malloci+07 sigma, PAHdb A, Set 2 from I Fig. 5):

- **I Table 5** (photoionisation): -0.11 to +0.03 dex for C14-C32 with
  Malloci sigma; PAH 50 C +0.20 with Li & Draine sigma.
- **I Table 4** (H, H2, C2H2 loss):
  - N_C <= 24: within 0.1 dex. The C16H10 Set 1 C2H2 entry, 1.29e-11, is
    compared with 0.1 x its H2 rate (a misprint).
  - C32H14: -0.22 dex with k_IR = sum A (the card's reading), +0.20 with
    the occupation factor kept.
  - PAH 50 C (C50H20): -1.35 dex, and -0.34 with the occupation kept.
    Allain's 2.27e-16 lies between our occupation-kept N_H = 18-20 (4.3e-16,
    1.0e-16) and occupation-dropped N_H = 16 (1.7e-16): N_H0 and the k_IR
    convention are both unstated.
- **II Tables 2-3** (Set 2):
  - N_C <= 24: within 0.23 dex in every column.
  - C32H14: normal states -0.17 to -0.19 dex, dehydrogenated -0.02 to +0.04.
  - PAH 50 C and 100 C: normal states -1.2 to -1.4 dex, alpha_H = 0.5 states
    -0.09 to -0.15, alpha_H <= 0.25 states -0.19 to +0.07.
  - The deficit appears only where many C-H modes remain under k_IR = sum A.
- **II Table 5:**
  - The single-state times (normal neutral, bare cation) agree within 0.21
    dex. The exception is pyrene in the Orion Bar, which breaks Allain's own
    1/chi scaling.
  - The network means are 1-2 dex shorter for pyrene (except NGC 7027 at
    -0.36) and for ovalene in NGC 7023 and the diffuse medium. Our steady
    states are more dehydrogenated there.
- **II Fig. 7** (alpha_ion vs N_C, Orion Bar and NGC 7027, digitised):
  - C10-C32 (Malloci sigma): Orion Bar -4.7 to -1.1 points, NGC 7027 -2.4 to
    -0.9 points.
  - As log odds, the Orion Bar residual equals the I Table 5 photoionisation
    residual to 0.06 dex, so recombination (II eq. 9) reproduces and the rest
    is sigma.
  - PAH 50 C and 100 C (Li & Draine sigma): +9 and +10 points in the Orion
    Bar; removing the 0.2 dex photoionisation excess gives 27.5 and 35.2 %
    against 28.5 and 36.1 %.
- **Critical sizes** [C]: Orion Bar 39.2, HI 37.9, M82 42.0, NGC 7023 44.4,
  NGC 7027 46.6, diffuse warm 64.2. Allain: > 40 everywhere, smallest in the
  Orion Bar and HI, 50 in NGC 7027, 55 in the warm medium.

Needs the PAHdb mode files named in pahchem.allain96.SPECIES under
``--states-dir``. Takes ~1 min.
"""
import argparse
from pathlib import Path

import numpy as np

from pycalima import _paths
from pycalima.pahchem import allain96 as A
from pycalima.pahchem.cards import load_card


def dex(a, b):
    return f"{np.log10(a / b):+6.2f}" if isinstance(b, float) and a > 0 and b > 0 else "    --"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--states-dir", type=Path, default=_paths.get_model_data_dir() / "PAH_states")
    args = ap.parse_args()
    T = load_card("allain96")["targets"]
    modes = {n: A.load_modes(n, args.states_dir) for n in A.SPECIES if A.SPECIES[n][3]}
    f1, f2 = A.set1_field(), A.set2_field()
    print(f"Set 2 field: {f2.name}\n")

    t4, t5 = T["paperI_table4"], T["paperI_table5"]
    print("I Tables 4-5, Set 1: dex(ours/theirs)    ion      H     H2   C2H2 | H with occupation kept")
    for i, name in enumerate(t4["species"]):
        if name not in modes:
            continue
        r = A.AllainState(name, modes[name]).rates(f1)
        st = A.AllainState(name, modes[name])
        st.k_ir = A.k_ir(modes[name], st.energy_ev, multi_quanta=True)
        kd = sum(st.k.values())
        y_h = (1 - st.y_ion) * st.k["H"] / (st.k_ir + kd)
        h_mq = np.trapezoid(st.sigma * f1.flux_at(st.energy_ev) * y_h, st.energy_ev)
        c2 = 0.1 * t4["H2_set1"][i] if name == "C16H10" else t4["C2H2_set1"][i]
        print(f"  {name:10s} {dex(r['ion'], t5['set1'][i])} {dex(r['H'], t4['H_set1'][i])} "
              f"{dex(r['H2'], t4['H2_set1'][i])} {dex(r['C2H2'], c2)} | {dex(h_mq, t4['H_set1'][i])}")

    t2, t3 = T["paperII_table2"], T["paperII_table3"]
    print("\nII Tables 2-3, Set 2:  cat H  cat H2  a50 H a50 H2  a25 H |  C neu  C cat  C a50  C a00")
    for i, name in enumerate(t2["species"]):
        if name not in modes:
            continue
        nh0 = A.SPECIES[name][1]
        r = lambda z, n: {k: v * A.YR for k, v in A.AllainState(name, modes[name], z, n).rates(f2).items()}
        cat, neu, h50, h25, h0 = r(1, nh0), r(0, nh0), r(0, int(0.5 * nh0)), r(0, int(0.25 * nh0)), r(0, 0)
        print(f"  {name:10s} {dex(cat['H'], t2['cation_H'][i])} {dex(cat['H2'], t2['cation_H2'][i])} "
              f"{dex(h50['H'], t2['aH050_H'][i])} {dex(h50['H2'], t2['aH050_H2'][i])} {dex(h25['H'], t2['aH025_H'][i])} | "
              f"{dex(neu['C2H2'], t3['neutral'][i])} {dex(cat['C2H2'], t3['cation'][i])} "
              f"{dex(h50['C2H2'], t3['aH050'][i])} {dex(h0['C2H2'], t3['aH000'][i])}")

    t5b = T["paperII_table5"]
    names = [n for n in ("C10H8", "C14H10", "C16H10", "C24H12", "C32H14", "PAH 50 C", "PAH 100 C") if n in modes]
    fams = {n: A.AllainFamily(n, modes[n], f2) for n in names}
    print("\nII Table 5, dex(tau ours/theirs):     neutral  bare+   mean")
    for name, key in (("C16H10", "pyrene"), ("C32H14", "ovalene")):
        for j, med in enumerate(t5b["medium"]):
            s = fams[name].steady(med)
            print(f"  {key:8s} {med:20s} {dex(s['tau_neutral_yr'], t5b[f'{key}_neutral'][j])} "
                  f"{dex(s['tau_bare_cation_yr'], t5b[f'{key}_dehydrogenated_cation'][j])} "
                  f"{dex(s['tau_uv_yr'], t5b[f'{key}_mean'][j])}")

    print("\nalpha_ion / alpha_H by medium (II Figs. 7, 10 targets; Fig. 7 Orion Bar and NGC 7027 digitised,"
          " see tests/test_pahchem_allain96.py):")
    print("  " + " " * 20 + " ".join(f"{n:>11s}" for n in names))
    for med in A.MEDIA:
        cells = []
        for n in names:
            s = fams[n].steady(med)
            cells.append(f"{s['alpha_ion']:.2f}/{s['alpha_H']:.2f}")
        print(f"  {med:20s}" + " ".join(f"{c:>11s}" for c in cells))

    nc = [A.SPECIES[n][0] for n in names]
    print("\nCritical size [C] (log k_C/n_C = -12; linear in N_C between species):")
    for med, (n_h, *_rest) in A.MEDIA.items():
        k_net = [fams[n].steady(med)["k_C_uv"] for n in names]
        chi = A.MEDIA[med][3]
        k_neu = [chi * fams[n].photo[(0, A.SPECIES[n][1])]["C2H2"] for n in names]
        print(f"  {med:20s} network {A.critical_size(nc, k_net, n_h):5.1f}   normal neutral only "
              f"{A.critical_size(nc, k_neu, n_h):5.1f}")


if __name__ == "__main__":
    main()
