#!/usr/bin/env python3
"""The per-bin PAH group tables (``pahchem.group_tables``) against ``Population`` at the same cells.

``Population`` solves the charge chain of every molecule on its 1-500 eV grid with the exact spectrum. The table
solves one chain per bin with group-averaged kernels. Three things separate them, each measured here:

A. The in-group shape (blackbody T_s against the true spectrum): tables of single molecules, solved one by one and
   averaged, against ``Population``. Groups 5-5.6-11.2-13.6 eV, so both see the same photons.
B. Averaging the rates over the bin before the balance: the bin table against the average of the single-molecule
   solves of A.
C. Photons below 5 eV (RTZ group 1-5.6 eV): they detach anions (the population's grid reaches them since 2026-10-24). The
   bin table on the RTZ groups against the same table with that group zeroed, for the Kurucz field and the Mathis
   ISRF, with the Jochims (the default before 2026-10-25) and the statistical detachment.

Field: Kurucz 40000 K cut at 13.6 eV (the Orion Bar shape, whose bases are cached by fit_pdrs4all_spectra.py).
Cells: G0, T and gamma = G0 sqrt(T) / n_e on a grid, n_H = 1e3 n_e, ions C+ with n(C+) = n_e. The bins are N_C
ranges for the test only, not a recommended scheme.

    check_pah_group_tables.py [--workers 8]
"""
import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import fit_34um_sidegroups as S
import fit_pdrs4all_population as FP

from pycalima.pahchem import fields, population
from pycalima.pahchem import group_tables as G
from pycalima.pahchem.baseline import Environment, Params

TEFF = 40000
BINS = ((22, 40), (40, 70), (70, 120), (120, 270))
G0S, TS, GAMMAS = (1.0, 1e2, 1e4), (100.0, 1e3), (1e2, 1e3, 1e4, 1e5)
VAL_L0, VAL_L1 = (5.0, 5.6, 11.2), (5.6, 11.2, 13.6)
PARAMS = Params(stretch_scale=population.LAB_STRETCH_SCALE)       # as the cached bases (fit_pdrs4all_spectra.py)


def cells():
    for g0 in G0S:
        for t in TS:
            for gam in GAMMAS:
                ne = g0 * np.sqrt(t) / gam
                yield Environment(f"G0={g0:g} T={t:g} gamma={gam:g}", g0, 1e3 * ne, ne, t, TEFF, ions=(("C+", ne),))


def table_rates(tab, env, L0, L1, keep_below_5ev=True, field=None):
    F, egy = G.group_fluxes(fields.kurucz(TEFF, env.g0) if field is None else field(env.g0), L0, L1)
    if not keep_below_5ev:
        F = np.where(np.asarray(L1) <= 5.6, 0.0, F)
    return G.PAHGroupTable(tab, egy=egy).solve(F, env.temperature_k, env.n_e, env.n_h, env.ions)


def summary(name, a, b):
    """Worst ratio a/b - 1 of the positive entries, and where."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = (b > 1e-6) if name.startswith("f") else (np.abs(b) > 0)
    r = np.where(ok, a / np.where(ok, b, 1.0) - 1.0, 0.0)
    return f"{name} {np.max(np.abs(r)) * 100:6.1f}%"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    lib = S.library()
    for lo, hi in BINS:
        w = G.nc_range_weights(lib, lo, hi)
        mols = [m for m, x in zip(lib, w) if x > 0]
        pop = population.Population(mols, FP.WINDOWS, cache_dir=FP.BASIS_CACHE, params=PARAMS)
        pop.prepare(("kurucz", TEFF), workers=args.workers)
        jobs = lambda L0, L1, p=PARAMS: [(m, p, 0.46, L0, L1, G.N_E_PER_GROUP, G.LT) for m in mols]
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(args.workers) as ex:
            kv = list(ex.map(G.molecule_kernels, jobs(VAL_L0, VAL_L1), chunksize=4))
            kr = {det: list(ex.map(G.molecule_kernels, jobs(G.rg.DEFAULT_GROUP_L0, G.rg.DEFAULT_GROUP_L1,
                                                             replace(PARAMS, detachment=det)), chunksize=4))
                  for det in ("jochims", "statistical")}
        single = [G.combine([k], [1.0], VAL_L0, VAL_L1) for k in kv]
        binv = G.combine(kv, np.ones(len(kv)), VAL_L0, VAL_L1)
        binr = {det: G.combine(k, np.ones(len(k)), G.rg.DEFAULT_GROUP_L0, G.rg.DEFAULT_GROUP_L1) for det, k in kr.items()}
        x = np.full(len(mols), 1.0 / len(mols))
        print(f"\nN_C {lo}-{hi}: {len(mols)} molecules, <N_C> {binv['nc']:.0f}")
        worst = {"A": {}, "B": {}}
        for env in cells():
            gr = pop.gas_rates(env, x)
            zi = [z - pop.zmin for z in G.CHARGES]
            f_pop = (x @ gr["f"])[zi]
            f_pop[-1] += (x @ gr["f"])[zi[-1] + 1:].sum()             # +3 where the molecule has it: the table's +2
            ref = dict(f=f_pop, gamma=gr["gamma_pe"], lam=gr["lambda_rec"],
                       cplus=gr["alpha_mn"]["C+"] + gr["alpha_ct"]["C+"])
            ss = [table_rates(t, env, VAL_L0, VAL_L1) for t in single]
            a = dict(f=np.mean([s["f"] for s in ss], axis=0), gamma=np.mean([s["gamma"] for s in ss]),
                     lam=np.mean([s["lam"] for s in ss]),
                     cplus=np.mean([s["alpha_mn"]["C+"] + s["alpha_ct"]["C+"] for s in ss]))
            s = table_rates(binv, env, VAL_L0, VAL_L1)
            b = dict(f=s["f"], gamma=s["gamma"], lam=s["lam"], cplus=s["alpha_mn"]["C+"] + s["alpha_ct"]["C+"])
            for key in ref:
                worst["A"].setdefault(key, []).append((a[key], ref[key]))
                worst["B"].setdefault(key, []).append((b[key], a[key]))
        for part, label in (("A", "A in-group shape (single-molecule tables / Population)"),
                            ("B", "B bin-averaged rates (bin table / mean of single-molecule tables)")):
            parts = []
            for key, pairs in worst[part].items():
                u, v = (np.array([p[i] for p in pairs]) for i in (0, 1))
                if key == "f":
                    parts += [summary(f"f({z:+d})", u[:, i], v[:, i]) for i, z in enumerate(G.CHARGES)]
                else:
                    parts.append(summary(key, u, v))
            print(f"  {label}:\n    " + ", ".join(parts))
        print("  C photons below 5 eV (G0 = 1, T = 100 K; with / without the 1-5.6 eV group):")
        rtz = (G.rg.DEFAULT_GROUP_L0, G.rg.DEFAULT_GROUP_L1)
        cp = lambda r: r["alpha_mn"]["C+"] + r["alpha_ct"]["C+"]
        for fname, field in ((f"Kurucz {TEFF} K", None), ("Mathis ISRF", lambda g0: fields.mathis1983(g0, e_min=0.5))):
            for det, tab in binr.items():
                for env in cells():
                    if env.g0 != 1.0 or env.temperature_k != 100.0:
                        continue
                    on = table_rates(tab, env, *rtz, field=field)
                    off = table_rates(tab, env, *rtz, keep_below_5ev=False, field=field)
                    print(f"    {fname:15s} {det:11s} gamma {env.g0 * np.sqrt(env.temperature_k) / env.n_e:7.0e}: "
                          f"f(-1) {off['f'][0]:.3f} -> {on['f'][0]:.3f}; alpha(C+) x{cp(on) / cp(off):.2f}; "
                          f"Gamma x{on['gamma'] / off['gamma']:.2f}")

if __name__ == "__main__":
    main()
