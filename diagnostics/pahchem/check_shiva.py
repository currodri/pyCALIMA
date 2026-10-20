#!/usr/bin/env python3
"""SHIVA's single-molecule photo-rates (pahchem.shiva) against its published pieces.

Prints:
1. our DL01 T(E) against Draine & Li 2001 Fig. 7 (vector-extracted);
2. our WD01a yield against Murga+2019 Fig. 2 (a = 3.3 A, Z = 0-3);
3. C66H20 R_H and R_C(chi) by four rate forms against Murga & Wiebe 2022 Fig. 1
   (vector-extracted), with the DL01 T(E) (SHIVA as M20 states it) and with
   Tielens' T(E) = 2000 K (E/N_C)^0.4;
4. R_H, R_H2, R_C2H2 for C24H12, C66H20, C128H28 at the Murga+2020 A_V = 0.1
   points (Orion Bar chi = 1.1e4, Horsehead chi = 45), Z = 0 and 1.

Measured (2026-10-05):
- **DL01 Fig. 7**: median |dex| <= 0.0015, max 0.004 for N_C = 14, 24, 50, 100
  (E > 2000 cm^-1).
- **M19 Fig. 2 Y_ion**: max |diff| 0.001 for Z = 0-3: pyCALIMA's WD01 graphite
  yield is SHIVA's.
- **MW22 Fig. 1, log10(ours / MW22)** at chi = 1 / 1e2 / 1e4 (all absorptions
  heating; time-averaged p(E), their eq. 1):
  - RRK H: -1.40 / -0.15 / +0.33; RRK C: -2.15 / -0.97 / -0.48. RRK has no
    T(E) and no free convention: it tests p(E) and the absorption, which
    agree within 0.33 dex for chi >= 100 (H). At chi = 1, where RRK needs
    two-photon events, our tail is 1.4 dex lighter.
  - Arrh. 1 / 2 with the DL01 T(E): +1.6 to +3.4 dex at chi <= 100, +3 to
    +4.7 at 1e4. With Tielens' T(E): H +0.58 / +0.51-0.55, C -0.09 / -0.04
    at chi <= 100; +0.2 to +1.1 at 1e4. MW22's Arrhenius curves use Tielens'
    T(E), not the DL01 one that M20 states. DL01 runs 5-10 % hotter at
    10-14 eV; with E0/kT ~ 40-50 that is 2-3 dex.
  - Laplace H: -4.1 / -2.9 / -1.9 (PAHdb modes, as MW22 state). Not
    resolved: MW22's Laplace behaves as if rho had fewer modes.
- **Single-molecule rates** for M20's trajectories: printed in section 4
  (``m20`` = yield per absorption x absorption rate, eqs. 4-6; ``mw22`` =
  time-averaged k).

Needs ``--c66-modes`` (PAHdb C66H20_0.dat) for the Laplace form only. Takes
~30 s.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pycalima._paths import get_external_data_path
from pycalima.pahchem import dos, rates
from pycalima.pahchem import shiva as S

YR = 3.15576e7
EV = dos.CM_PER_EV


def _csv(name):
    return pd.read_csv(get_external_data_path(f"pahchem/{name}"), comment="#")


def mw22_forms(m, c66_modes=None):
    E = m.energy_ev
    forms = {"H": {"Arrh. 1": (3.3, 5.0), "Arrh. 2": (4.3, 11.8)},
             "C": {"Arrh. 1": (3.65, 5.0), "Arrh. 2": (4.6, 10.0)}}
    out = {p: {n: S.k_arrhenius_te(E, *v, m.t_m) for n, v in f.items()} for p, f in forms.items()}
    out["H"]["RRK"] = rates.rrk(E * EV, m.modes.size, 1e16, 2.8 * EV)
    out["C"]["RRK"] = rates.rrk(E * EV, m.modes.size, 1e15, 2.9 * EV)
    if c66_modes is not None:
        from pycalima.pahchem.molecule import PAHSpecies

        sp = PAHSpecies.from_states_file(c66_modes)
        Eg, counts = dos.beyer_swinehart(sp.freqs_cm, E[-1] * EV + 200.0)
        idx = np.rint(E * EV).astype(int)
        out["H"]["Laplace"] = rates.inverse_laplace_arrhenius(Eg, counts, 1.4e16, 4.3 * EV)[idx]
        out["C"]["Laplace"] = rates.inverse_laplace_arrhenius(Eg, counts, 1.4e16, 4.6 * EV)[idx]
    return out


def mw22_log(product, method, chi):
    g = _csv("murga_wiebe22_fig1.csv")
    g = g[(g["product"] == product) & (g.method == method)].sort_values("chi")
    if chi < g.chi.min():
        return np.nan
    return float(np.interp(np.log10(chi), np.log10(g.chi), np.log10(g.rate_per_yr)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--c66-modes", type=Path, default=None)
    args = ap.parse_args()

    print("1. DL01 Fig. 7 T(E), dex(ours/DL01) for E > 2000 cm^-1:")
    t = _csv("shiva_dl01_fig7_temperature.csv")
    for nc in sorted(t.n_c.unique()):
        g = t[(t.n_c == nc) & (t.energy_cm > 2e3)]
        r = np.log10(S.Temperature(S.dl01_modes(nc, S.dl01_nh(nc)))(g.energy_cm.values / EV) / g.temperature_k.values)
        print(f"   N_C = {nc:3d}: median {np.median(r):+.4f}, max |.| {np.abs(r).max():.4f}")

    print("\n2. Murga+2019 Fig. 2 Y_ion (a = 3.3 A), max |ours - theirs|:")
    y = _csv("murga2019_fig2_yion.csv")
    for z in range(4):
        g = y[y.z == z]
        print(f"   Z = {z}: {np.abs(S.y_ion_wd01(g.energy_eV.values, 3.3e-8, z) - g.y_ion.values).max():.4f}")

    print("\n3. MW22 Fig. 1, C66H20: log10(R / yr^-1) ours / MW22 at chi = 1, 1e2, 1e4")
    for temp in ("dl01", "tielens"):
        m = S.ShivaMolecule(66, 20, 0, temperature=temp)
        forms = mw22_forms(m, args.c66_modes)
        p = {chi: m.distribution(S.mmp83(chi), ionising_photons_heat=True)[0] for chi in (1.0, 1e2, 1e4)}
        for prod, ks in forms.items():
            for name, k in ks.items():
                cells = [f"{np.log10(p[c] @ k * YR + 1e-300):+6.2f}/{mw22_log(prod, name, c):+6.2f}" for c in p]
                print(f"   T(E) {temp:7s} {prod} {name:8s} " + "  ".join(cells))

    print("\n4. M20 single-molecule rates [s^-1] (m20 = <Y_sp>_post x int_5^13.6 C_abs F; mw22 = <k>_time)")
    sites = {"Orion A_V=0.1": 1.1e4, "Horsehead A_V=0.1": 45.0}
    for nc, nh in ((24, 12), (66, 20), (128, 28)):
        for z in (0, 1):
            m = S.ShivaMolecule(nc, nh, z)
            ks = {c: m.k_diss(c) for c in ("H", "H2", "C2H2")}
            for site, chi in sites.items():
                r = m.rates(S.mmp83(chi), ks)
                print(f"   C{nc}H{nh} Z={z} {site:18s} " + "  ".join(
                    f"{c}: m20 {r[c]['m20']:.2e} mw22 {r[c]['mw22']:.2e}" for c in ks), flush=True)


if __name__ == "__main__":
    main()
