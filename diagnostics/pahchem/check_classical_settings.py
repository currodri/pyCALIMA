#!/usr/bin/env python3
"""Which pahchem settings bring the PAHs' ion neutralisation, photoelectric heating, recombination cooling and charge closest to the
classical grain models (WD01 / WDB06, the grain physics of Cloudy) for H II-region conditions?

The reference is ``check_wd01b_alpha.py``'s WD01 physics on the same PAH carriers (graphite grains of 3.5-8.3 A, case A, ISRF background),
read from its stored result (``results/pahchem_wd01b_alpha_valence.json``); the pahchem side is recomputed here for each variant, with the
same group tables, bins and conditions (psi = G sqrt(T) / n_e, G = 1 Habing, n_H = 1). Output: for each variant, the geometric-mean ratio
pahchem / WD01(PAH) of alpha(H+), alpha(C+), Gamma_PE and Lambda_rec in three psi ranges at each T, and the mean charge difference.

Result (2026-11-01, the library UV default; results/pahchem/classical_settings_2026-11-01/report.txt): at psi <= 1e3 the default
gives alpha(H+) 1.29 / 0.94, Gamma_PE 1.49 / 1.26, Lambda_rec 2.40 / 1.88 at 1000 / 6000 K, within ~3 % of the Malloci default (C);
eta = 0.46 (D) gives 1.00 / 0.83, 1.24 / 1.11, 1.81 / 1.57 and stays the best classical-comparison setting. At psi >= 1e5 Gamma_PE is
3.7-8x WD01's whatever eta or f.

    check_classical_settings.py [--workers 2] [--only A D ...] [--report-only]
"""
import argparse
import json
import sys
from itertools import pairwise
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import check_wd01b_alpha as W

from pycalima import _paths
from pycalima.pahchem import group_tables as G
from pycalima.pahchem import population
from pycalima.pahchem.baseline import Params

#: variant -> (Params keywords, IP shifts of the library [eV])
VARIANTS = {
    "A default (library UV, valence, eta = f = 1)": ({}, W.IP_SHIFTS),
    "B Li & Draine UV": ({"uv_cross_section": "li_draine"}, W.IP_SHIFTS),
    "C Malloci average UV (the default until 2026-10-31)": ({"uv_cross_section": "malloci_average"}, W.IP_SHIFTS),
    "D eta = 0.46 (Tielens' capture limit)": ({"capture_scale": 0.46}, W.IP_SHIFTS),
    "E eta = 0.2": ({"capture_scale": 0.2}, W.IP_SHIFTS),
    "F f = 0.2": ({"attachment_fraction": 0.2}, W.IP_SHIFTS),
    "G f = 0.05": ({"attachment_fraction": 0.05}, W.IP_SHIFTS),
    "H eta = 0.46, f = 0.2": ({"capture_scale": 0.46, "attachment_fraction": 0.2}, W.IP_SHIFTS),
    "I capture_model lab_trend": ({"capture_model": "lab_trend"}, W.IP_SHIFTS),
    "J statistical detachment": ({"detachment": "statistical"}, W.IP_SHIFTS),
    "K no IP shifts": ({}, {}),
    "L eta = 0.46, no IP shifts": ({"capture_scale": 0.46}, {}),
    "M eta = 0.46, f = 0.2, no IP shifts": ({"capture_scale": 0.46, "attachment_fraction": 0.2}, {}),
}
T_USE = (1000.0, 6000.0)
RANGES = (("psi <= 1e3", 0.0, 1e3), ("1e3 < psi <= 1e4", 1e3, 1e4), ("psi >= 1e5", 1e5, 1e9))
OUT = _paths.get_results_dir("pahchem") / "classical_settings_2026-11-01"      # 2026-10-29: the Malloci-average default
REF = _paths.get_results_dir() / "pahchem_wd01b_alpha_valence.json"


def compute(name, workers):
    kw, shifts = VARIANTS[name]
    params = Params(**kw)
    lib = population.library_v400()
    if shifts:
        lib = population.shift_thresholds(lib, shifts)
    nT, npsi = len(W.T_GRID), W.PSI_GRID.size
    a_fine = np.geomspace(W.gsd.A_MIN, W.radius_cm(W.NC_EDGES[-1]), 400)
    pah = np.zeros((nT, npsi, len(W.IONS) + 2))
    zbar = np.zeros((nT, npsi))
    n_tot = 0.0
    for lo, hi in pairwise(W.NC_EDGES):
        w = G.nc_range_weights(lib, lo, hi)
        a = a_fine[(a_fine >= W.radius_cm(lo)) & (a_fine <= W.radius_cm(hi))]
        n_bin = float(np.trapezoid([W.gsd.graphite_dn_da(x) for x in a], a))
        tab = G.PAHGroupTable(G.build_bin_table(lib, w, params=params, workers=workers, label=f"{lo}-{hi}"))
        n_tot += n_bin
        for iT, T in enumerate(W.T_GRID):
            for ip, psi in enumerate(W.PSI_GRID):
                ne = np.sqrt(T) / psi
                r = tab.solve(np.zeros(tab.t["L0"].size), T, ne, 0.0, (("H+", ne),), G0_bg=1.0)
                pah[iT, ip] += n_bin * np.array([r["alpha_mn"][i] + r["alpha_ct"][i] for i in W.IONS] + [r["gamma"], r["lam"]])
                zbar[iT, ip] += n_bin * float(r["f"] @ np.array(G.CHARGES, dtype=float))
    return {"pah": pah.tolist(), "zbar": (zbar / n_tot).tolist()}


def gmean(x):
    return float(np.exp(np.mean(np.log(x))))


def report(results):
    ref = json.loads(REF.read_text())
    psi = np.array(ref["psi"])
    wd = np.concatenate([np.array(ref["wd01_pah"]), np.array(ref["heat"]["wd01_pah"])], axis=2)       # (T, psi, ions + 2)
    cols = {"alpha(H+)": W.IONS.index("H+"), "alpha(C+)": W.IONS.index("C+"), "Gamma_PE": len(W.IONS), "Lambda_rec": len(W.IONS) + 1}
    print("pahchem / WD01 physics on the same PAH carriers; geometric mean over the psi grid points in each range")
    for T in T_USE:
        iT = list(ref["T"]).index(T)
        print(f"\nT = {T:.0f} K")
        head = f"{'variant':44s}" + "".join(f" | {r[0]:>16s}: " + " ".join(f"{c[:9]:>9s}" for c in cols) for r in RANGES)
        print(head)
        for name, res in results.items():
            p = np.array(res["pah"])
            row = f"{name:44s}"
            for _, lo, hi in RANGES:
                k = (psi > lo) & (psi <= hi)
                row += " | " + " " * 18 + " ".join(f"{gmean(p[iT, k, j] / wd[iT, k, j]):9.2f}" for j in cols.values())
            print(row)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--only", nargs="*", default=None, help="variant letters")
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    for name in VARIANTS:
        if a.only and name.split()[0] not in a.only:
            continue
        f = OUT / (name.split()[0] + ".json")
        if not f.exists() and not a.report_only:
            print(f"variant {name}", flush=True)
            f.write_text(json.dumps(compute(name, a.workers)))
        if f.exists():
            results[name] = json.loads(f.read_text())
    report(results)


if __name__ == "__main__":
    main()
