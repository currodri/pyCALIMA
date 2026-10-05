#!/usr/bin/env python3
"""Le Page+01/03 in their configuration (pahchem.lepage01) against their numbers.

Prints:
- the 2nu-RRKM checks (coronene curve, quoted cation rates, Paper I Table 1);
- the charge rates against LP03 Table 2 and the Fig. 4 fractions;
- the LP03 Fig. 9 hydrogenation at the standard conditions, varied over
  the choices the card leaves open.

Measured (2026-10-05):

- **2nu-RRKM.**
  - Coronene H loss against Paper I Fig. 9: +0.23 dex at threshold
    (7.6 eV), -0.07 to +0.15 at 8.3-15 eV.
  - Benzene+ 3.05/6.07 against 3.19/6.1; anthracene+ 2.09/7.54 against
    2.2/7.6.
  - Naphthalene+ is 2.88/6.92 against 3.9/7.5, ~1 dex low with the Table 2
    E0 (4.5 eV; ~4.2 eV would fit).
  - Table 1 (2nu column) is the neutral PAHH H loss (1.2 eV, dS = 0), within
    0.25 dex. The RRK column needs E0 ~ 0.7 eV.
- **Charge.**
  - k_rec n_e, k_e and k_p equal Table 2.
  - k_ion with Li & Draine sigma is 1.2-1.7x Table 2 (Malloci per C:
    0.9-1.2x), so the cation fractions of Fig. 4 are 0.02-0.13 high. With
    the Table 2 k_ion, Fig. 4 is reproduced to 0.015.
- **Fig. 9** (nt = 100, chi = 1, naphthalene recombination). C16 and C24 sit
  at N_H <= 1; C32H14+/C32H14 are 0.55/0.30 (card 0.45/0.37); C50H20+ is
  0.64 (0.6); C80H24+/C80H25+ are 0.59/0.26 (0.6/0.27).
  - With the benzene recombination, C80H25+ falls to 0.08 (Fig. 9: 0.27),
    and C32 and C50 become mostly neutral. This supports the card's
    inference that Fig. 9 uses naphthalene.
  - Dropping the stepladder or adding (1 - Y_ion) to eq. 11 moves no
    fraction by more than 0.07. The Malloci sigma lowers f+ by 0.03-0.07.

Takes ~1 min.
"""
import argparse

import numpy as np
import pandas as pd

from pycalima._paths import get_external_data_path
from pycalima.pahchem import lepage01 as L

LP03_TABLE2_KION = {16: 4.0e-9, 24: 6.9e-9, 32: 1.0e-8, 50: 2.0e-8, 100: 5.0e-8}
FAMILIES = ((16, 10), (24, 12), (32, 14), (50, 20), (80, 24))


def top_states(model, f, n=4):
    order = np.argsort(f.ravel())[::-1][:n]
    return " ".join(f"{'+' if z else '0'}H{model.nh_values[j]}:{f[z, j]:.2f}"
                    for z, j in zip(*np.unravel_index(order, f.shape)))


def main():
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()

    d = pd.read_csv(get_external_data_path("dissoc_rate_coronene_Montillaud.csv"), skiprows=1)
    x, y = d["X.1"].dropna().to_numpy(), d["Y.1"].dropna().to_numpy()
    print("coronene k(E) vs Paper I Fig. 9 [dex]:",
          " ".join(f"{e:.1f}:{v:+.2f}" for e, v in zip(x, np.log10(L.k_of_E(24, 12, 4.8, 5.0, 12.0, x) / y))))
    for name, nc, nh, E0, Es, q in (("benzene+", 6, 6, 3.8, [4.5, 5.5], [3.19, 6.1]),
                                     ("naphthalene+", 10, 8, 4.5, [7, 10], [3.9, 7.5]),
                                     ("anthracene+", 14, 10, 4.4, [7.9, 13.6], [2.2, 7.6])):
        got = np.log10(L.k_of_E(nc, nh, E0, 5.0, float(nh), np.array(Es, float)))
        print(f"{name:13s} log k at {Es} eV: {np.round(got, 2)} (quoted {q})")

    print("\nk_ion [s^-1] at chi = 1:  ours (Li & Draine)  ours (Malloci/C)  LP03 Table 2")
    for nc, theirs in LP03_TABLE2_KION.items():
        print(f"  C{nc:<4d} {L.k_ion(nc):18.2e} {L.k_ion(nc, substitute='malloci'):17.2e} {theirs:13.2e}")

    print("\nLP03 Fig. 9, nt = 100, chi = 1, T = 100 K (top states, Z H_N: fraction)")
    variants = (("baseline (naphthalene rec.)", {}, L.STANDARD),
                ("benzene recombination", {}, L.Environment(recombination="benzene")),
                ("no stepladder", {"stepladder": False}, L.STANDARD),
                ("(1 - Y_ion) in eq. 11", {"one_minus_yion": True}, L.STANDARD),
                ("Malloci sigma per C", {"substitute": "malloci"}, L.STANDARD))
    for nc, nh0 in FAMILIES:
        for label, opts, env in variants:
            m = L.LePage01(nc, nh0, **opts)
            f = m.steady_state(env)
            print(f"  C{nc}H{nh0:<3d} {label:28s} f+ = {f[1].sum():.2f}  {top_states(m, f)}", flush=True)


if __name__ == "__main__":
    main()
