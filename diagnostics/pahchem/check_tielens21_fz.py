#!/usr/bin/env python3
"""Charge distributions of C54H18 and C96H24 against Tielens (2021) Fig. 12.34 (validation V14).

Tielens (2021, Molecular Astrophysics, §12.8.1, Fig. 12.34) solves photoionisation against recombination and
attachment (his eq. 12.18) for circumcoronene and circumcircumcoronene, x_e = 1.6e-4, in three environments:
a diffuse cloud (G0 = 2, n = 50, T = 80 K; gamma = 2.2e3), the NGC 7023 PDR (2600, 2e4, 500; 1.8e4) and its
cavity (2e4, 5e3, 500; 5.6e5). The caption puts the cavity in the middle column, but the panel labels and the
gamma order put it on the right. The C54H18 cavity panel repeats the C96H24 PDR panel bar for bar, so the book
gives no C54H18 cavity distribution.

The values below are read by eye from the book's figure (±0.01). pahchem uses v4.00 "C54H18 (37)" and
"C96H24 (108)", the calibrated IP shifts, the Draine field with G0 taken as Habing (the book does not state the
field), and no ions. It is shown with its associative detachment of anions by H (n_H = n) and without it
(n_H = 0, as the book's balance).

Result (2026-10-24, 1 eV grid with the solar-circle old stars below 5 eV, n_H = 0):
- diffuse cloud: every f(Z) within 0.05 for both molecules (C54H18 f(-1, 0, +1) 0.11, 0.83, 0.05 against 0.13,
  0.78, 0.085; 0.13, 0.81, 0.05 on the old 5 eV grid); associative detachment at n_H = n lowers f(-1) to 0.08-0.09;
- NGC 7023 PDR: unchanged by the old stars at this G0; pahchem is less ionised, <Z> +0.36 against +0.55 (C54H18)
  and +0.54 against +0.78 (C96H24); f(0) 0.61 against 0.47 and 0.49 against 0.35, the same direction as against
  WD01 at high psi (V4);
- cavity, C96H24: <Z> +2.62 against +2.57, f(+2, +3) 0.27, 0.68 against 0.35, 0.61.

    check_tielens21_fz.py [--workers 4]
"""
import argparse

import numpy as np

from pycalima import _paths
from pycalima.pahchem import population
from pycalima.pahchem.baseline import Environment

IP_SHIFTS = {-1: 0.37, 0: 0.6, 1: 0.64, 2: 0.64}
X_E = 1.6e-4
#: name -> (G0, n [cm^-3], T [K])
ENVIRONMENTS = {"diffuse": (2.0, 50.0, 80.0), "NGC 7023 PDR": (2600.0, 2e4, 500.0),
                "NGC 7023 cavity": (2e4, 5e3, 500.0)}
#: (molecule, environment) -> f(-1, 0, +1, +2, +3) read from Fig. 12.34
FIG_12_34 = {("C54H18 (37)", "diffuse"): (0.13, 0.78, 0.085, 0.003, 0.0),
             ("C54H18 (37)", "NGC 7023 PDR"): (0.02, 0.47, 0.43, 0.07, 0.0),
             ("C96H24 (108)", "diffuse"): (0.13, 0.745, 0.125, 0.005, 0.0),
             ("C96H24 (108)", "NGC 7023 PDR"): (0.015, 0.35, 0.47, 0.15, 0.01),
             ("C96H24 (108)", "NGC 7023 cavity"): (0.0, 0.0, 0.035, 0.35, 0.61)}
BASIS_CACHE = _paths.get_model_data_dir() / "pahchem_population_basis"


def molecules():
    lib = population.shift_thresholds(population.library_v400(), IP_SHIFTS)
    return [m for m in lib if m.name in ("C54H18 (37)", "C96H24 (108)")]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    mols = molecules()
    pop = population.Population(mols, {"all": (1.0, 1e4)}, cache_dir=BASIS_CACHE)
    zs = (-1, 0, 1, 2, 3)
    cols = [pop.zs.index(z) for z in zs]
    print(f"f(Z) for Z = {zs}; Draine field, x_e = {X_E:g}, no ions")
    for env_name, (g0, n, t) in ENVIRONMENTS.items():
        ne = X_E * n
        print(f"\n{env_name}: G0 = {g0:g}, n = {n:g}, T = {t:g} K, gamma = {g0 * np.sqrt(t) / ne:.2e}")
        fz = {}
        for n_h in (n, 0.0):
            env = Environment(env_name, g0, n_h, ne, t, None)
            pop.prepare(population.field_shape(env), workers=args.workers)
            fz[n_h] = pop.charge_matrix(env)[:, cols]
        for s, m in enumerate(mols):
            ref = FIG_12_34.get((m.name, env_name))
            shown = "not given (panel repeated)" if ref is None else np.round(ref, 3)
            print(f"  {m.name:13s} Tielens Fig. 12.34    {shown}")
            print(f"  {'':13s} pahchem, n_H = n      {np.round(fz[n][s], 3)}   <Z> {fz[n][s] @ zs:+.2f}")
            print(f"  {'':13s} pahchem, n_H = 0      {np.round(fz[0.0][s], 3)}   <Z> {fz[0.0][s] @ zs:+.2f}"
                  + ("" if ref is None else f"   Tielens <Z> {np.dot(ref, zs):+.2f}"))


if __name__ == "__main__":
    main()
