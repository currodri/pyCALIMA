#!/usr/bin/env python3
"""Murga+2020 per-molecule evolution (pahchem.murga2020) against Fig. 4 (and Fig. 5).

Prints, for the 20 parents at Orion (isochoric) and Horsehead A_V = 0.1 and
2.0, Z = 0 and 1, the C2H2 produced per molecule within 1e6 yr (eq. 3)
against the exact Fig. 4 values (external_data/pahchem/
murga2020_fig4_c2h2_per_molecule.csv), as log10(ours/theirs), for a chosen
set of options, plus the sensitivity of a few cases to every option.

Measured (2026-10-05/06; dex = log10(ours/Fig. 4); defaults: DL01 T(E),
time-averaged p(E), independent channels, eq. 6 over all absorptions,
Goumans held below 40 K):

- **Orion A_V = 0.1, Z = 0** (chi = 1.1e4, T = 4400 K, n(H) = 2e5): every
  parent fully destroyed, N_C2H2 = N_C/2 to 0.1 % (C16H10 in 8 yr, C150H30
  in 1.6e3 yr). Mechanism: Goumans addition (~2e4 yr^-1) super-hydrogenates
  the neutrals and the 2.0 eV super-H C2H2 channel destroys them. Tielens
  T(E) gives the same counts but kills C150H30 in 24 yr, against the end of
  Figshare C2H2 production at 1e4 yr; DL01 (-0.8 dex) is preferred.
- **p(E) reading:** "post" makes the normal-state rates ~1e9 x those of
  "time" at low chi (normal C24H12 at Horsehead A_V = 0.1: 3 yr^-1); Fig. 4
  (1e-3 C2H2 per molecule there) needs "time". rms over the 16 probe points:
  time 2.7 dex, post 4.6.
- **Partial destruction** (Orion Z = 1, Orion A_V = 2, Horsehead): 1-4 dex
  above Fig. 4 with independent channels. 16-point probe rms (2026-10-05):

  | option set | rms |
  |---|---|
  | DL01 / time (default) | 2.67 |
  | Tielens / time | 2.74 |
  | + super-H offset 1 H | 2.60 |
  | + ion H2 addition only when dehydrogenated | 2.67 |
  | DL01 / post | 4.64 |

  The real-valued N_H drifts above X_H0 under any H addition and opens the
  super-H channels (rates ~1e12 x the normal ones). The Orion cations
  (Fig. 4: 0.1 C2H2 for C16H10) need dehydrogenation to N_H <= 1 before C
  loss, i.e. normal-state H loss > ion H addition (~440 yr^-1), which only
  the "post" magnitudes give.
- **Channel competition** (2026-10-06; M19 eq. B2, k_diss = k_H + k_H2 +
  k_C2H2, the outcome probabilities "sum up to unity"; ``compete=True``).
  Over all 20 parents, 140 partial points (dex):

  | option set | rms | median | full-destruction max abs(dex) |
  |---|---|---|---|
  | independent, time (default) | 2.53 | +2.45 | 0.00 |
  | compete, time | 1.16 | +0.06 | 1.28 |
  | compete, time, (1 - Y_ion) | 1.20 | +0.01 | 1.44 |
  | compete, post | 4.40 | +2.69 | 0.00 |

  - With competition the cations with N_C >= 40 fit within 1 dex (mostly
    within 0.5). C16-C24
    stay +1 to +2 dex high, and the large neutrals 1-3 dex low (Horsehead
    A_V = 0.1, C150H30: -3.0). The size trend is too steep where the
    super-H H loss (1.4 eV, Z <= 0) outruns the 2.0 eV C2H2 channel.
  - It breaks the Orion A_V = 0.1 neutrals for N_C >= 110: C150H30 sits at
    the maximum N_H = 180 and yields 3.9 of 75 C2H2 within 1e6 yr, where
    the Figshare C2H2 production ends at 1e4 yr. That gate holds the
    default at independent channels; the conflict is a strict xfail.
  - Competition also cuts the M22 deep-MC photo-loss rate 3.2x; it then sits
    2.8-4.6x below Figshare (independent channels: 0.87-1.44x;
    check_murga2022.py). So two of the three data sets favour independent
    channels and one favours competition.
  - (1 - Y_ion) (M19 eq. B2; M20: the photon energy "either is spent to
    ionization or converts to the internal energy") against eq. 6 as
    printed: < 0.1 dex rms. The default is eq. 6.
- **Super-H path switched off** (``superh_offset = 1e9``; Orion A_V = 2 and
  Horsehead, Z = 0, 1): the normal-state rates alone are +1 to +2 dex for
  C16H10, -0.6 to +2 for C24H12, and -2 to -22 dex for C66H20 and C150H30
  (DL01/time; "post" +2 to +8 for the small ones). Fig. 4's nearly
  size-independent 1e-4-1e-3 for N_C >= 48 needs a super-H contribution,
  1e2-1e4 x smaller than ours.
- **Integrator:** the stiff solver equals M20's explicit Euler (dt = 1 yr)
  to 1e-8 where Euler is stable (Horsehead A_V = 2).

Usage: check_murga2020.py [--temperature dl01|tielens] [--p-reading time|post]
       [--parents C16H10,C24H12,...] [--sensitivity]
Takes ~1-10 min (rate tables are built once per (Z, chi)).
"""
import argparse
import time

import numpy as np
import pandas as pd

from pycalima._paths import get_external_data_path
from pycalima.pahchem import murga2020 as T

CASES = (("orion_isochoric", 0.1), ("orion_isochoric", 2.0), ("horsehead", 0.1), ("horsehead", 2.0))


def fig4():
    return pd.read_csv(get_external_data_path("pahchem/murga2020_fig4_c2h2_per_molecule.csv"), comment="#")


def compare(parents, opts, cases=CASES, charges=(0, 1), verbose=True):
    """{(site, av, z, parent): (ours, theirs, dex)}."""
    f4, out = fig4(), {}
    for site, av in cases:
        env = T.environment(site, av)
        for z in charges:
            cells = []
            for p in parents:
                tr = T.evolve(p, env, z, opts)
                theirs = float(f4[(f4.site == site) & (f4.av == av) & (f4.z == z) & (f4.molecule == p)].n_c2h2.iloc[0])
                d = float(np.log10(max(tr.total_c2h2, 1e-30) / theirs))
                out[(site, av, z, p)] = (tr.total_c2h2, theirs, d)
                cells.append(f"{p}:{d:+.2f}")
            if verbose:
                print(f"  {site:16s} A_V={av:<4} Z={z}: " + " ".join(cells), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--temperature", default=T.DEFAULT.temperature)
    ap.add_argument("--p-reading", default=T.DEFAULT.p_reading)
    ap.add_argument("--parents", default=",".join(T.PARENTS))
    ap.add_argument("--sensitivity", action="store_true")
    args = ap.parse_args()
    opts = T.Options(temperature=args.temperature, p_reading=args.p_reading)
    parents = args.parents.split(",")
    t0 = time.time()
    print(f"Fig. 4, dex(ours/theirs), {opts}")
    compare(parents, opts)
    print(f"({time.time() - t0:.0f} s)")
    if args.sensitivity:
        probe = ["C24H12", "C66H20", "C150H30"]
        variants = {"compete": dict(compete=True), "atoms x4": dict(atoms_per_molecule_event=4.0),
                    "low_nh none": dict(low_nh="none"), "no N_H cap": dict(nh_max="none"),
                    "ER 2 atoms": dict(er_atoms=2.0), "normal band 0.1": dict(normal_band=0.1),
                    "Goumans clamped": dict(goumans_t_range="clamp"), "p post": dict(p_reading="post"),
                    "T tielens" if args.temperature == "dl01" else "T dl01":
                        dict(temperature="tielens" if args.temperature == "dl01" else "dl01")}
        base = compare(probe, opts, verbose=False)
        for name, kw in variants.items():
            res = compare(probe, T.Options(**{**opts.__dict__, **kw}), verbose=False)
            shifts = [res[k][2] - base[k][2] for k in base]
            print(f"  {name:16s}: shift in dex(ours/theirs) median {np.median(shifts):+.2f}, "
                  f"range {min(shifts):+.2f} to {max(shifts):+.2f}", flush=True)


if __name__ == "__main__":
    main()
