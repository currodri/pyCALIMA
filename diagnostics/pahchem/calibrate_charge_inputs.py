#!/usr/bin/env python3
"""Calibrate pahchem's charge inputs: PAHdb v4.00 IP/EA against measured values, BT94 capture against FALP.

1. Ionisation energies. PAHdb v4.00 thresholds are adiabatic B3LYP/6-31G*
   total-energy differences (``population.library_v400``). They are compared
   with the Cagliari database (Malloci+07, external_data/cagliari_pah_database.csv):
   computed adiabatic IP1, EA and double-ionisation energies, with measured
   values where known.
   - Matching: a Cagliari molecule is matched to the v4.00 isomer of its
     formula only when the library holds one isomer ("unique"). Otherwise it
     goes to the isomer closest to Malloci's adiabatic value minus the unique
     offset ("nearest"), reported separately (that choice assumes the offset).
   - Steps: IP1 = 0 -> +1; EA = -1 -> 0; IP2 = +1 -> +2, which in Malloci's
     table is the double-ionisation energy minus IP1.
2. Recombination. BT94 electron capture (chemistry.electron_capture, sticking
   1) against the measured PAH-cation recombination coefficients at 300 K
   (pah_electron_recombination.csv, Bierbaum+11 compilation).
"""
from collections import defaultdict

import numpy as np
import pandas as pd

from pycalima import _paths
from pycalima._paths import get_external_data_path
from pycalima.pahchem import chemistry, population


def main():
    lib = population.library_v400(min_nc=9, cache_dir=_paths.get_model_data_dir() / "pahchem_pahdb_index")
    by_formula = defaultdict(list)
    for m in lib:
        by_formula[m.name.split()[0]].append(m)
    cag = pd.read_csv(get_external_data_path("cagliari_pah_database.csv"))
    rows = []
    for r in cag.itertuples():
        c = by_formula.get(r.formula, [])
        if not c:
            continue
        rows.append(dict(name=r.name, formula=r.formula, nc=r.Nc, n_iso=len(c), cands=c, ip_m=r.IP1_adiab_eV,
                         ip_x=r.IP1_exp_eV, ea_m=r.EA_adiab_eV, ea_x=r.EA_exp_eV,
                         ip2_m=r.IP2_singlet_adiab_eV - r.IP1_adiab_eV,
                         dip_m=r.IP2_singlet_adiab_eV, dip_x=r.IP2_exp_eV))
    uniq = [r for r in rows if r["n_iso"] == 1]
    off_u = np.array([r["cands"][0].ionisation_ev[0] - r["ip_m"] for r in uniq])
    print(f"IP1, v4.00 - Malloci adiabatic, unique matches ({len(uniq)}): "
          + ", ".join(f"{r['name']} {o:+.2f}" for r, o in zip(uniq, off_u)))
    med = float(np.median(off_u))
    print(f"   median {med:+.2f} eV, scatter (MAD) {np.median(np.abs(off_u - med)):.2f}")
    for r in rows:
        r["m"] = r["cands"][0] if r["n_iso"] == 1 else min(r["cands"], key=lambda m: abs(m.ionisation_ev[0] - med - r["ip_m"]))
    for label, sel in (("unique", [r for r in rows if r["n_iso"] == 1]), ("nearest", [r for r in rows if r["n_iso"] > 1])):
        x = [(r["name"], r["m"].ionisation_ev[0] - r["ip_x"]) for r in sel if np.isfinite(r["ip_x"])]
        if x:
            v = np.array([o for _, o in x])
            print(f"IP1, v4.00 - measured, {label} ({len(v)}): median {np.median(v):+.2f}, range {v.min():+.2f} to {v.max():+.2f}"
                  f"  [{', '.join(f'{n} {o:+.2f}' for n, o in x)}]")
    mx = np.array([r["ip_m"] - r["ip_x"] for r in rows if np.isfinite(r["ip_x"])])
    print(f"IP1, Malloci adiabatic - measured (all with data, {mx.size}): median {np.median(mx):+.2f}, range {mx.min():+.2f} to {mx.max():+.2f}")
    ea = [(r["name"], r["m"].ionisation_ev.get(-1), r["ea_m"], r["ea_x"]) for r in rows if -1 in r["m"].ionisation_ev]
    d_m = np.array([v - m for _, v, m, _ in ea if np.isfinite(m)])
    d_x = np.array([v - x for _, v, _, x in ea if np.isfinite(x)])
    print(f"EA, v4.00 - Malloci adiabatic ({d_m.size}): median {np.median(d_m):+.2f}, range {d_m.min():+.2f} to {d_m.max():+.2f}")
    if d_x.size:
        print(f"EA, v4.00 - measured ({d_x.size}): median {np.median(d_x):+.2f}  [{', '.join(f'{n} {v - x:+.2f}' for n, v, _, x in ea if np.isfinite(x))}]")
    ip2 = np.array([r["m"].ionisation_ev[1] - r["ip2_m"] for r in rows if 1 in r["m"].ionisation_ev])
    print(f"IP2 (cation -> dication), v4.00 - Malloci adiabatic ({ip2.size}): median {np.median(ip2):+.2f}, range {ip2.min():+.2f} to {ip2.max():+.2f}")
    dd = np.array([(r["dip_m"] - r["dip_x"]) - (r["ip_m"] - r["ip_x"]) for r in rows
                   if np.isfinite(r["dip_x"]) and np.isfinite(r["ip_x"])])
    print(f"IP2, Malloci - measured (double-ionisation minus IP1 terms, {dd.size}): median {np.median(dd):+.2f}, "
          f"range {dd.min():+.2f} to {dd.max():+.2f}")
    # -- recombination -------------------------------------------------------------------------
    rec = pd.read_csv(get_external_data_path("pahchem/pah_electron_recombination.csv"), comment="#")
    rec["bt94"] = [chemistry.electron_capture(n, 1, 300.0) for n in rec.n_c]
    rec["ratio"] = rec.k_cm3_s / rec.bt94
    print("\nRecombination at 300 K, measured / BT94 (sticking 1):")
    for r in rec.itertuples():
        print(f"  {r.species:14s} N_C {r.n_c:3d}  measured {r.k_cm3_s:.2e}  BT94 {r.bt94:.2e}  ratio {r.ratio:.2f}")
    big = rec[rec.n_c >= 14]
    print(f"  median ratio, all {rec.ratio.median():.2f}; N_C >= 14 {big.ratio.median():.2f} "
          f"(log-slope vs N_C: {np.polyfit(np.log(rec.n_c), np.log(rec.ratio), 1)[0]:+.2f})")


if __name__ == "__main__":
    main()
