#!/usr/bin/env python3
"""The joint (Z, N_H) chain (Params.hydrogen_chain = "joint") over a G0 x n_H grid, and its brackets.

For the compact parents C24H12 (18), C54H18 (37) and C96H24 (108) of PAHdb v4.00, in the Draine field at 100 K with
n_e = 1.4e-4 n_H (the GSD session's T1 conditions). It prints <N_H>/N_H0, the super-H share and the share at the
ladder's top ("at_cap": where it is more than a few per cent, the cap sets the answer), the cation share and the H2
released, for:
- the joint default;
- the inputs the 2026-11-08 literature check flagged (docs/physics/pahchem-gas-coupling.md, section 2.2h): super-H
  dS 12 -> 55.6 J/K/mol, Eley-Rideal sigma x7 (Foley+18's gas-phase lower bound), super-H cation addition 1e-12
  (Montillaud+13, Murga+20);
- the anion options: no H addition to anions, associative detachment on every anion (Andrews+16);
- the charge-averaged chain, for reference.

Usage: check_joint_hydrogenation.py [--levels 4] [--out results/pahchem/joint_hydrogenation/brackets.csv]
"""
import argparse
import csv
from dataclasses import replace
from pathlib import Path

import numpy as np

from pycalima import _paths
from pycalima.pahchem import population as P
from pycalima.pahchem.baseline import Environment, Params

PARENTS = ("C24H12 (18)", "C54H18 (37)", "C96H24 (108)")
G0 = (1e-3, 1e-2, 1.0, 1e2, 1e4)
NH = (1e2, 1e4)


def variants(levels):
    base = replace(Params(), hydrogen_chain="joint", superh_levels=levels)
    return {"joint": base,
            "dS 55.6": replace(base, h_superh_ds=55.6),
            "ER x7": replace(base, eley_rideal_a2=7 * base.eley_rideal_a2),
            "cation super-H 1e-12": replace(base, addition_cation_superh=1e-12),
            "anions add no H": replace(base, anion_h_addition="none"),
            "AD on every anion": replace(base, associative_detachment="all"),
            "charge-averaged": Params()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--levels", type=int, default=4)
    ap.add_argument("--out", type=Path, default=_paths.get_results_dir() / "pahchem" / "joint_hydrogenation" / "brackets.csv")
    args = ap.parse_args()
    lib = P.library_v400(cache_dir=_paths.get_model_data_dir() / "pahchem_pahdb_index")
    lib = [m for m in lib if m.name in PARENTS]
    cache = _paths.get_model_data_dir() / "pahchem_population_basis"
    rows = []
    for name, params in variants(args.levels).items():
        pop = P.Population(lib, {"3.3": (3.2, 3.36)}, params=params, cache_dir=cache)
        for g0 in G0:
            for n in NH:
                env = Environment(f"{g0:g}/{n:g}", g0, n, 1.4e-4 * n, 100.0, None)
                sm, f = pop.hydrogenation_summary(env), pop.charge_matrix(env)
                cat = f[:, np.array(pop.zs) > 0].sum(axis=1)
                for s, mol in enumerate(pop.library):
                    rows.append(dict(variant=name, molecule=mol.name, g0=g0, n_h=n, x=sm["x"][s], superh=sm["superh"][s],
                                     at_cap=sm["at_cap"][s], intact=sm["intact"][s], bare=sm["bare"][s], cation=cat[s],
                                     h2=sm["h2"][s]))
        print(f"{name}: done", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{'variant':22s} {'molecule':13s} {'G0':>7s} {'n_H':>7s}  x      superh at_cap cation")
    for r in rows:
        if r["superh"] > 1e-3 or r["variant"] == "joint":
            print(f"{r['variant']:22s} {r['molecule']:13s} {r['g0']:7.0e} {r['n_h']:7.0e}  {r['x']:.3f}  {r['superh']:.3f}  "
                  f"{r['at_cap']:.3f}  {r['cation']:.3f}")
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
