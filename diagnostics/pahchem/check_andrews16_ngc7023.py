#!/usr/bin/env python3
"""Andrews+16 Figs. 7-8: C54H18 hydrogenation and charge in NGC 7023 NW.

Runs pahchem.andrews16.Andrews16Model for circumcoronene along the digitised
Meudon PDR profile (``external_data/NWPDR_NGC7023``: G0/n(H), n(H), n_e, T
against A_V). G0 = (G0/n(H)) n(H); this reproduces their gamma profile
(gamma.csv, ~2.5e4 at the edge), so the edge G0 is ~4100, not the incident
2600. It prints our N_H and charge fractions next to their Fig. 7
(N_H = 14-17) and Fig. 8 (f(Z) against gamma).

Measured (2026-10-04, Kurucz 15000 K, all absorptions heating, eq. 8 with
k_IR along the cooling path to T_i):

- **Charge:** right shape, but more neutral and anionic than theirs at every
  gamma. Ours resembles theirs at 2-3x lower gamma; for example at A_V = 0.05,
  f(+1) is 0.35 against their 0.48, and f(-1) is 0.02 against 0.00.
- **Hydrogenation:** C54H18 stays >= 97 % normally hydrogenated at every depth.
  Their N_H = 16 is 0.19 at the edge and 0.11 at A_V = 2. That needs an
  effective H2-loss rate of ~1e-9 s^-1 at A_V = 2, against 2e-11 from their
  own Table B there and less from eq. 8 as read here. Their network behaves
  as if k_IR in eq. 8 were ~1e3 smaller, which is what a G(T) weighted near
  T_i (3-20 K, where the thermal k_IR vanishes) gives. Eq. 8 is too loosely
  specified to settle this without their code.

Takes ~7 min (8 depths x ~12 stationary solves).
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from pycalima import _paths
from pycalima.pahchem.andrews16 import Andrews16Model
from pycalima.pahchem.molecule import PAHSpecies

AV = (0.05, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--states-dir", type=Path, default=_paths.get_model_data_dir() / "PAH_states")
    args = ap.parse_args()

    modes = {}
    for f in sorted(args.states_dir.glob("C54*.dat")):
        sp = PAHSpecies.from_states_file(f)
        modes[(sp.charge, sp.nh)] = sp
    model = Andrews16Model("C54", modes, de_cm=150.0)

    D = Path(_paths.get_external_data_path("NWPDR_NGC7023"))
    prof = {n: np.loadtxt(D / f"{n}.csv", delimiter=",") for n in ("G0nH", "nH", "ne", "temperature")}
    at = lambda n, a: np.interp(a, prof[n][:, 0], prof[n][:, 1])
    fig8 = {}
    for z, name in ((-1, "anion"), (0, "neutral"), (1, "cation"), (2, "dication")):
        d = np.loadtxt(_paths.get_external_data_path(f"C54HN_{name}_andrews16.csv"), delimiter=",")
        fig8[z] = d[np.argsort(d[:, 0])]
    rows = list(csv.reader(open(D / "Nh_circumcoronene.csv")))
    fig7 = {}
    for j in range(0, len(rows[0]), 2):
        pts = np.array([[float(r[j]), float(r[j + 1])] for r in rows[2:] if len(r) > j + 1 and r[j] and r[j + 1]])
        fig7[int(rows[0][j].split("=")[1])] = pts[np.argsort(pts[:, 0])]

    print(" A_V  gamma | N_H=14..17 ours / Andrews           | f(Z=-1,0,1,2) ours / Andrews")
    for a in AV:
        n_h, n_e, T = at("nH", a), at("ne", a), at("temperature", a)
        g0 = at("G0nH", a) * n_h
        gam = g0 * np.sqrt(T) / n_e
        net = model.network(model.photo_rates(g0), n_h, n_e, T)
        f = net.steady_state()
        fh, fz = net.hydrogenation_fractions(f), net.charge_fractions(f)
        theirs_h = {n: np.interp(a, p[:, 0], p[:, 1]) for n, p in fig7.items()}
        theirs_z = {z: np.interp(np.log10(gam), np.log10(c[:, 0]), c[:, 1]) for z, c in fig8.items()}
        print(f"{a:4.2f} {gam:6.0f} | "
              + " ".join(f"{fh[n]:.3f}/{theirs_h[n]:.3f}" for n in (14, 15, 16, 17)) + " | "
              + " ".join(f"{fz[z]:.2f}/{theirs_z[z]:.2f}" for z in (-1, 0, 1, 2)), flush=True)


if __name__ == "__main__":
    main()
