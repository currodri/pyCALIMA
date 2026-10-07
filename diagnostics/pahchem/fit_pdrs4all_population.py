#!/usr/bin/env python3
"""Which PAH population explains the PDRs4All Orion Bar bands, at the stated conditions (M7d).

Library: the 133 Maragkoudakis+20 PAHdb molecules (``population.library_m20``).
Charge and the first H loss follow from the conditions card
(environments/orion_bar.toml, both corners of each template); only the
abundances are fitted (``population.fit_abundances``, NNLS, 15 % per ratio).
Molecules less than ``--intact-min`` intact at the most-ionising corner are
left out of that template's fit.

Observed bands: the Van De Putte+25 templates, measured as in
compare_pdrs4all.py, plus the 12.0, 13.5 and 14.2 um out-of-plane bands, in
pahchem's windows (after Hony+01; not Khan+25's). The 10-15 um plateau sits
under all the out-of-plane bands, so they are measured twice:
- "spline": over the Chown+24 global spline, so the plateau is counted in;
- "local": over a line across each window's edges, so the plateau is taken off.
The model is measured the same way (``population._band_matrix``). The bands
below 10 um always use the spline. The two continua bracket the systematic.

    fit_pdrs4all_population.py                        # fit the bundled band table
    fit_pdrs4all_population.py --templates .../orionbar.dat --write-bands out.csv
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

import compare_pdrs4all as C
from pycalima import _paths
from pycalima._paths import get_external_data_path
from pycalima.pahchem import conditions, population

OOP_WINDOWS = {"12.0": (11.8, 12.25), "13.5": (13.3, 13.8), "14.2": (13.95, 14.45)}
WINDOWS = {**C.WINDOWS, **OOP_WINDOWS}
OOP = ("11.0", "11.2", "12.0", "12.7", "13.5", "14.2")
REFERENCE = "11.2"
CONTINUA = ("spline", "local")
BANDS_CSV = "pahchem/pdrs4all_orionbar_bands.csv"
#: state bases of population.Population (G0 = 1), kept between runs
BASIS_CACHE = _paths.get_model_data_dir() / "pahchem_population_basis"
#: Khan+25 (PDRs4All XIV) Table 2: number ratios of edge H from the 11-14 um bands
KHAN25 = {"solo/duo": (14.0, 17.0), "solo/trio": (1.5, 1.5), "solo/quartet": (22.0, 46.0)}


def measure(lam, fnu):
    """{continuum: {band: flux / flux(11.2)}} of one template."""
    out = {}
    base = C.observed_bands(lam, fnu)
    lm, fm = C._clean(lam[lam < 20], fnu[lam < 20], 0.025)
    cont = CubicSpline(C.ANCHORS, [np.median(fm[np.abs(lm - a) < 0.004]) for a in C.ANCHORS])(lm)
    spline = {**base, **{b: C._integ(lm, fm - cont, *w) for b, w in OOP_WINDOWS.items()}}
    local = dict(spline)
    edge = lambda x: np.median(fm[np.abs(lm - x) < 0.01])
    for b in OOP:
        lo, hi = WINDOWS[b]
        line = edge(lo) + (edge(hi) - edge(lo)) * (lm - lo) / (hi - lo)
        local[b] = C._integ(lm, fm - line, lo, hi)
    for name, v in (("spline", spline), ("local", local)):
        out[name] = {b: v[b] / v[REFERENCE] for b in WINDOWS}
    return out


def write_bands(templates, path):
    lam, tmpl = C.load_templates(templates)
    rows = []
    m = {t: measure(lam, tmpl[t]) for t in C.TEMPLATES}
    for c in CONTINUA:
        for b in WINDOWS:
            rows.append({"continuum": c, "band": b, **{t: round(m[t][c][b], 4) for t in C.TEMPLATES}})
    head = ("# PAH band fluxes relative to 11.2 of the PDRs4All Orion Bar templates, MEASURED HERE from the\n"
            "# Van De Putte+25 combined templates (CDS J/A+A/701/A111 orionbar.dat, corrected 05-Nov-2025) by\n"
            "# diagnostics/pahchem/fit_pdrs4all_population.py. Windows [um]: "
            + ", ".join(f"{b} {lo}-{hi}" for b, (lo, hi) in WINDOWS.items()) + "\n"
            "# (3.3-12.7 as compare_pdrs4all; 12.0/13.5/14.2 pahchem's, after Hony+01). 'spline': over the\n"
            "# Chown+24 global spline (10-15 um plateau included); 'local': out-of-plane bands (11.0-14.2)\n"
            "# over a line between the window edges (plateau removed), shorter bands as 'spline'.\n"
            "# Systematic ~10-20 % (continuum); no uncertainties propagated; no extinction correction.\n")
    with open(path, "w") as fh:
        fh.write(head)
        pd.DataFrame(rows).to_csv(fh, index=False)


def observed(continuum):
    t = pd.read_csv(get_external_data_path(BANDS_CSV), comment="#", dtype={"band": str})
    t = t[t.continuum == continuum].set_index("band")
    return {tpl: {b: float(t.loc[b, tpl]) for b in WINDOWS} for tpl in C.TEMPLATES}


def run(min_nc=0, intact_min=0.5, n_boot=200, cache=None, all_anions=True):
    """{(template, corner, continuum): result dict}; ``cache`` a pickle path to reuse."""
    if cache is not None and Path(cache).exists():
        with open(cache, "rb") as fh:
            return pickle.load(fh)
    lib = population.library_m20(min_nc=min_nc, all_anions=all_anions)
    pops = {c: population.Population(lib, WINDOWS, local=OOP if c == "local" else (), cache_dir=BASIS_CACHE)
            for c in CONTINUA}
    obs = {c: observed(c) for c in CONTINUA}
    out = {"library": [(m.name, m.nc, m.nh, m.edges) for m in lib]}
    for t in C.TEMPLATES:
        envs = conditions.bracketing_environments("orion_bar", t)
        intact = pops["spline"].intact_fractions(envs["most ionising"])
        keep = intact >= intact_min
        for corner, env in envs.items():
            for c in CONTINUA:
                F, fz = pops[c].band_fluxes(env)
                o = obs[c][t]
                res = population.fit_abundances(F[keep], o, REFERENCE)
                w = np.zeros(len(lib))
                w[keep] = res["weights"]
                boot = np.zeros((n_boot, len(lib)))
                boot[:, keep] = population.bootstrap(F[keep], o, REFERENCE, n=n_boot)
                single = []
                for s in np.flatnonzero(keep):
                    r = population.fit_abundances(F[s:s + 1], o, REFERENCE)
                    single.append((r["chi2"], lib[s].name))
                out[(t, corner, c)] = {
                    "env": env, "observed": o, "fit": res, "weights": w, "boot": boot, "F": F, "keep": keep,
                    "intact": intact, "f_cation": np.array([fz[m.name].get(1, 0.0) for m in lib]),
                    "stats": population.population_statistics(lib, w),
                    "boot_stats": [population.population_statistics(lib, b) for b in boot],
                    "best_single": min(single)}
    if cache is not None:
        with open(cache, "wb") as fh:
            pickle.dump(out, fh)
    return out



def _grid(lo, hi, n):
    return np.geomspace(lo, hi, n) if hi > lo else np.array([lo])


def scan(n_g0=4, n_nh=3, n_e=4, intact_min=0.5, cache=None, all_anions=True):
    """chi^2 and the implied population over a grid inside each template's stated ranges.

    The grid is G0 x n_H x (n_e, or x_e where the card gives x_e), log-spaced
    between the card's bounds. Molecules are kept by their intact fraction at
    the template's most-ionising corner, as in ``run``. Returns
    {template: list of cells}, each cell
    {g0, n_h, n_e, gamma, continuum: {chi2, mean_nc, f_anion, f_cation, solo/duo, ...}},
    where f_anion and f_cation are shares of the population's carbon.
    """
    if cache is not None and Path(cache).exists():
        with open(cache, "rb") as fh:
            return pickle.load(fh)
    from pycalima.pahchem.baseline import Environment

    lib = population.library_m20(all_anions=all_anions)
    pops = {c: population.Population(lib, WINDOWS, local=OOP if c == "local" else (), cache_dir=BASIS_CACHE)
            for c in CONTINUA}
    obs = {c: observed(c) for c in CONTINUA}
    nc = np.array([m.nc for m in lib], dtype=float)
    card = conditions.load("orion_bar")
    teff = int(card["region"]["field_teff_k"]["value"])
    out = {"library": [(m.name, m.nc, m.nh, m.edges) for m in lib]}
    for t in C.TEMPLATES:
        e = card[t]
        envs = conditions.bracketing_environments("orion_bar", t)
        keep = pops["spline"].intact_fractions(envs["most ionising"]) >= intact_min
        temp = conditions.bounds(e["t_gas"])[0]
        cells = []
        for g0 in _grid(*conditions.bounds(e["g0"]), n_g0):
            for n_h in _grid(*conditions.bounds(e["n_h"]), n_nh):
                electron = e["n_e"] if "n_e" in e else e["x_e"]
                for v in _grid(*conditions.bounds(electron), n_e):
                    ne = v if "n_e" in e else v * n_h
                    env = Environment(f"{t} scan", g0, n_h, ne, temp, teff)
                    cell = {"g0": g0, "n_h": n_h, "n_e": ne, "gamma": g0 * np.sqrt(temp) / ne}
                    for c in CONTINUA:
                        F, fz = pops[c].band_fluxes(env)
                        r = population.fit_abundances(F[keep], obs[c][t], REFERENCE)
                        w = np.zeros(len(lib))
                        w[keep] = r["weights"]
                        wc = w * nc / (w @ nc)
                        cell[c] = {"chi2": r["chi2"], "weights": w,
                                   "f_anion": float(sum(wc[i] * fz[m.name].get(-1, 0.0) for i, m in enumerate(lib))),
                                   "f_cation": float(sum(wc[i] * fz[m.name].get(1, 0.0) for i, m in enumerate(lib))),
                                   **population.population_statistics(lib, w)}
                    cells.append(cell)
                    print(f"{t} G0 {g0:.2g} n_H {n_h:.2g} n_e {ne:.2g} gamma {cell['gamma']:.2g}: "
                          + "  ".join(f"{c} chi2 {cell[c]['chi2']:.1f} <N_C> {cell[c]['mean_nc']:.0f} "
                                      f"f-/f+ {cell[c]['f_anion']:.2f}/{cell[c]['f_cation']:.2f}" for c in CONTINUA),
                          flush=True)
        out[t] = cells
    if cache is not None:
        with open(cache, "wb") as fh:
            pickle.dump(out, fh)
    return out

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--templates", type=Path, help="orionbar.dat; with --write-bands, re-measure the bands")
    ap.add_argument("--write-bands", type=Path)
    ap.add_argument("--min-nc", type=int, default=0)
    ap.add_argument("--intact-min", type=float, default=0.5)
    ap.add_argument("--cache", type=Path, default=_paths.get_results_dir() / "pahchem_population_fit.pkl")
    ap.add_argument("--scan", action="store_true", help="chi^2 over the stated (G0, n_H, n_e) ranges instead")
    args = ap.parse_args()
    if args.scan:
        scan(cache=_paths.get_results_dir() / "pahchem_population_scan.pkl")
        return
    if args.write_bands:
        write_bands(args.templates, args.write_bands)
        print("wrote", args.write_bands)
        return
    res = run(args.min_nc, args.intact_min, cache=args.cache)
    lib = res["library"]
    for key, r in res.items():
        if key == "library":
            continue
        t, corner, c = key
        s, bs = r["stats"], r["boot_stats"]
        q = lambda k: np.percentile([b[k] for b in bs if np.isfinite(b[k])], [16, 84]) if any(
            np.isfinite(b[k]) for b in bs) else (np.inf, np.inf)
        print(f"\n=== {t} | {corner} | {c} continuum: chi2 {r['fit']['chi2']:.1f} "
              f"(best single molecule {r['best_single'][0]:.1f}, {r['best_single'][1]}); "
              f"{int(r['keep'].sum())}/{len(lib)} intact")
        print("  bands  " + " ".join(f"{b:>6s}" for b in WINDOWS))
        print("  obs    " + " ".join(f"{r['observed'][b]:6.3f}" for b in WINDOWS))
        print("  model  " + " ".join(f"{r['fit']['ratios'][b]:6.3f}" for b in WINDOWS))
        print(f"  <N_C> {s['mean_nc']:.0f} (68% {q('mean_nc')[0]:.0f}-{q('mean_nc')[1]:.0f}); "
              + "; ".join(f"{k} {s[k]:.2g} ({q(k)[0]:.2g}-{q(k)[1]:.2g})" for k in KHAN25))
        top = np.argsort(r["weights"] * [m[1] for m in lib])[::-1][:5]
        print("  carbon-weighted top: " + ", ".join(
            f"{lib[i][0]} {r['weights'][i] * lib[i][1] / (r['weights'] @ [m[1] for m in lib]):.2f}"
            f" (f+ {r['f_cation'][i]:.2f})" for i in top if r["weights"][i] > 0))


if __name__ == "__main__":
    main()
