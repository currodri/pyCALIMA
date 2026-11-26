#!/usr/bin/env python3
"""The odd-parent threshold law in the joint (Z, N_H) chain: odd-N_H0 parents with ``threshold_shifts.ODD_LAWS`` and
with the even parents' laws they took before (2026-11-25).

An odd parent's extra H closes the shell, so the law raises the first super-H state's IP0 and lowers its EA; the even
laws had made it a radical, with the IP0 lower and the EA held. This compares the two for five odd parents of PAHdb
v4.00, from N_C 27 to 145 (``PARENTS``), in the Draine field at 100 K with n_e = 1.4e-4 n_H (the GSD session's T1
conditions, as ``check_joint_hydrogenation.py``), over G0 1e-3 ... 1e4 and n_H 1e2, 1e4. Two chains: the joint
default (4 super-H levels) and the deep one (``superh_chemistry = "deep"``, 24 levels).

For each molecule and condition it writes, under both laws:
- <N_H>/N_H0;
- the super-H share and the share at the ladder's top;
- <Z>, and the anion and cation shares;
- the share of the first super-H state (N_H0 + 1), and its anion and cation shares;
- the H2 released.

Usage: check_odd_parent_thresholds.py [--chains joint deep] [--out results/pahchem/odd_parent_thresholds/compare.csv]
"""
import argparse
import csv
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import numpy as np

from pycalima import _paths
from pycalima.pahchem import population as P
from pycalima.pahchem import threshold_shifts as TS
from pycalima.pahchem.baseline import Environment, Params

PARENTS = ("C27H13 (66)", "C31H15 (3226)", "C55H19 (3244)", "C89H27 (6525)", "C145H37 (6448)")
G0 = (1e-3, 1e-2, 1.0, 1e2, 1e4)
NH = (1e2, 1e4)
CHAINS = {"joint": dict(hydrogen_chain="joint"),
          "deep": dict(hydrogen_chain="joint", superh_chemistry="deep", superh_levels=24)}


@contextmanager
def even_laws():
    """Odd parents' super-H thresholds from the even parents' laws, as before 2026-11-25 (``parent_odd`` dropped)."""
    law = TS.law_shift
    TS.law_shift = lambda nc, dnh, z, ip0, ea, parent_odd=False: law(nc, dnh, z, ip0, ea)
    try:
        yield
    finally:
        TS.law_shift = law


def observables(pop, env):
    sm = pop.hydrogenation_summary(env)
    out = []
    for s, ((zs, F), mol) in enumerate(zip(pop.joint_fractions(env), pop.library)):
        z = np.asarray(zs)
        fz = F.sum(axis=1)
        p1 = F[:, mol.nh + 1]
        out.append(dict(x=sm["x"][s], superh=sm["superh"][s], at_cap=sm["at_cap"][s], h2=sm["h2"][s],
                        z_mean=float(fz @ z), anion=fz[z < 0].sum(), cation=fz[z > 0].sum(),
                        p1=p1.sum(), p1_anion=p1[z < 0].sum() / max(p1.sum(), 1e-300),
                        p1_cation=p1[z > 0].sum() / max(p1.sum(), 1e-300)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--chains", nargs="+", default=list(CHAINS), choices=list(CHAINS))
    ap.add_argument("--parents", nargs="+", default=list(PARENTS))
    ap.add_argument("--out", type=Path,
                    default=_paths.get_results_dir() / "pahchem" / "odd_parent_thresholds" / "compare.csv")
    args = ap.parse_args()
    lib = P.library_v400(cache_dir=_paths.get_model_data_dir() / "pahchem_pahdb_index")
    lib = [m for m in lib if m.name in args.parents]
    assert all(m.nh % 2 == 1 for m in lib) and len(lib) == len(args.parents), [m.name for m in lib]
    cache = _paths.get_model_data_dir() / "pahchem_population_basis"
    rows = []
    for chain in args.chains:
        params = replace(Params(), **CHAINS[chain])
        for law in ("odd", "even"):
            pop = P.Population(lib, {"3.3": (3.2, 3.36)}, params=params, cache_dir=cache)
            for g0 in G0:
                for n in NH:
                    env = Environment(f"{g0:g}/{n:g}", g0, n, 1.4e-4 * n, 100.0, None)
                    if law == "odd":
                        obs = observables(pop, env)
                    else:
                        with even_laws():
                            obs = observables(pop, env)
                    rows += [dict(chain=chain, law=law, molecule=mol.name, nc=mol.nc, g0=g0, n_h=n, **o)
                             for mol, o in zip(pop.library, obs)]
            print(f"{chain} / {law} laws: done", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    key = {(r["chain"], r["law"], r["molecule"], r["g0"], r["n_h"]): r for r in rows}
    print(f"{'chain':5s} {'molecule':14s} {'G0':>6s} {'n_H':>6s} | {'superh odd/even':>17s} | {'<Z> odd/even':>13s} | "
          f"{'p1 anion odd/even':>17s} | {'p1 cation odd/even':>18s}")
    for (chain, law, name, g0, n), o in key.items():
        if law != "odd":
            continue
        e = key[(chain, "even", name, g0, n)]
        print(f"{chain:5s} {name:14s} {g0:6.0e} {n:6.0e} | {o['superh']:8.3f} {e['superh']:8.3f} | "
              f"{o['z_mean']:+6.2f} {e['z_mean']:+6.2f} | {o['p1_anion']:8.3f} {e['p1_anion']:8.3f} | "
              f"{o['p1_cation']:9.3f} {e['p1_cation']:8.3f}")
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
