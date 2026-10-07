#!/usr/bin/env python3
"""PDRs4All Orion Bar templates fitted as spectra: a PAH population plus a smooth continuum (step 1 of the review).

This replaces the band-ratio fits of fit_pdrs4all_population.py, which compared
observed bands over the Chown+24 spline with raw model windows (review
finding C1). Here the observed spectrum and every model are treated alike:
``spectral_fit.fit_spectrum`` fits
I_nu = sum_s w_s P_s(conditions) + continuum (cubic B-splines in lambda, knots
``--knots`` um apart, per segment), with w, c >= 0.
- Observed: the Van De Putte+25 combined templates, gas lines removed with
  running medians (0.012 um at 3.3 um, 0.025 um in the MRS), binned at
  2.5 cm^-1 over 3.1-3.65 and 5-15 um, Chown+24 artefacts masked.
- Model: ``Population`` with the stated inputs of fit_pdrs4all_population
  (library, IP shifts, capture, stretch scale), charges at each cell of the
  conditions grid, molecules kept by the intact filter at the most-ionising
  corner. The filter is to be replaced (review finding 3).
- Per cell: sse, rms (residual / mean intensity), the population (N_C by number
  and by carbon; charge shares of the emitted power, dications counted with
  the cations), the derived band ratios of the PAH component, and the
  continuum share of the 5-15 um power.
- "Allowed" cells: Delta chi^2 < 9 with chi^2 = sse / s^2, s^2 = sse_min / (n - p),
  p the number of non-zero coefficients. This is an approximate scale; the
  statistics are step 3.

    fit_pdrs4all_spectra.py --templates .../orionbar.dat --library v400 --ip-shift=-1:0.37,0:0.6,1:0.64,2:0.64 --stretch-scale lab
"""
import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

import compare_pdrs4all as C
import fit_pdrs4all_population as FP
from pycalima import _paths
from pycalima.pahchem import conditions, population, spectral_fit
from pycalima.pahchem.baseline import Environment, Params

KNOTS = (1.0, 1.5, 2.5)


def observed_on_grid(lam, fnu, grid):
    s1 = (lam > 2.95) & (lam < 3.8)
    l1, f1 = C._clean(lam[s1], fnu[s1], 0.012)
    s2 = (lam > 4.9) & (lam < 15.5)
    l2, f2 = C._clean(lam[s2], fnu[s2], 0.025)
    return spectral_fit.bin_spectrum(np.concatenate([l1, l2]), np.concatenate([f1, f2]), grid)


def run(args):
    key = hashlib.sha1(json.dumps({k: str(v) for k, v in sorted(vars(args).items()) if k not in ("plot",)},
                                  sort_keys=True).encode()).hexdigest()[:12]
    cache = _paths.get_results_dir() / f"pahchem_spectral_fit_{key}.pkl"
    if cache.exists():
        with open(cache, "rb") as fh:
            return pickle.load(fh)
    lib = FP.library(args.library)
    shifts = {int(k): float(v) for k, v in (kv.split(":") for kv in args.ip_shift.split(",") if kv)}
    if shifts:
        lib = population.shift_thresholds(lib, shifts)
    stretch = population.LAB_STRETCH_SCALE if args.stretch_scale == "lab" else float(args.stretch_scale)
    params = Params(capture_scale=args.capture_scale, capture_model=args.capture_model, stretch_scale=stretch)
    pop = population.Population(lib, FP.WINDOWS, cache_dir=FP.BASIS_CACHE, params=params)
    grid = spectral_fit.Grid.build()
    lam, tmpl = C.load_templates(args.templates)
    nc = np.array([m.nc for m in lib], dtype=float)
    zs = np.array(pop.zs)
    shape = ("kurucz", 40000)
    pop.prepare(shape)
    absorbed = np.zeros((len(lib), len(zs)))
    for s, m in enumerate(lib):
        for z in m.charges:
            absorbed[s, z - pop.zmin] = pop.basis(m, z, shape).absorbed
    card = conditions.load("orion_bar")
    teff = int(card["region"]["field_teff_k"]["value"])
    bases = {k: spectral_fit.continuum_basis(grid, k) for k in KNOTS}
    i112 = list(FP.WINDOWS).index("11.2")
    out = {"library": [(m.name, m.nc, m.nh, m.edges) for m in lib], "grid": grid, "args": vars(args)}
    for t in C.TEMPLATES:
        y = observed_on_grid(lam, tmpl[t], grid)
        e = card[t]
        envs = conditions.bracketing_environments("orion_bar", t)
        keep = pop.intact_fractions(envs["most ionising"]) >= args.intact_min
        temp = conditions.bounds(e["t_gas"])[0]
        cells = []
        for g0 in FP._grid(*conditions.bounds(e["g0"]), 4):
            for n_h in FP._grid(*conditions.bounds(e["n_h"]), 3):
                el = e["n_e"] if "n_e" in e else e["x_e"]
                for v in FP._grid(*conditions.bounds(el), 4):
                    ne = v if "n_e" in e else v * n_h
                    env = Environment(f"{t}", g0, n_h, ne, temp, teff)
                    P = pop.spectra(env, grid.nu)[keep].T
                    F, _ = pop.band_fluxes(env)
                    f = pop.charge_matrix(env)
                    cell = {"g0": g0, "n_h": n_h, "n_e": ne, "gamma": g0 * np.sqrt(temp) / ne}
                    for k in KNOTS:
                        fit = spectral_fit.fit_spectrum(y, P, bases[k])
                        w = np.zeros(len(lib))
                        w[keep] = fit.weights
                        power = w[:, None] * f * absorbed                       # emitted power by state
                        band = w @ F
                        tot = power.sum()
                        ok = np.isfinite(y)
                        cell[k] = {"sse": fit.sse, "rms": fit.rms, "n": int(ok.sum()),
                                   "p": int((fit.weights > 0).sum() + (fit.continuum > 0).sum()),
                                   "weights": w,
                                   "mean_nc": float(w @ nc / w.sum()) if w.sum() > 0 else np.nan,
                                   "mean_nc_carbon": float((w * nc) @ nc / (w @ nc)) if w.sum() > 0 else np.nan,
                                   "e_anion": float(power[:, zs < 0].sum() / tot), "e_cation": float(power[:, zs > 0].sum() / tot),
                                   "e_dication": float(power[:, zs >= 2].sum() / tot),
                                   "ratios": {b: float(band[j] / band[i112]) for j, b in enumerate(FP.WINDOWS)},
                                   "cont_share": float(fit.cont[grid.segment == 1].sum()
                                                       / (fit.cont[grid.segment == 1].sum() + fit.pah[grid.segment == 1].sum())),
                                   "pah": fit.pah, "cont": fit.cont}
                    cells.append(cell)
        out[t] = {"observed": y, "cells": cells}
        print(f"{t} done", flush=True)
    with open(cache, "wb") as fh:
        pickle.dump(out, fh)
    return out


def summarise(res):
    for t in C.TEMPLATES:
        cells = res[t]["cells"]
        g = np.array([c["gamma"] for c in cells])
        for k in KNOTS:
            sse = np.array([c[k]["sse"] for c in cells])
            i = int(sse.argmin())
            b = cells[i][k]
            s2 = sse.min() / max(b["n"] - b["p"], 1)
            ok = (sse - sse.min()) / s2 < 9.0
            edge = "low edge" if g[i] <= g.min() * 1.0001 else ("high edge" if g[i] >= g.max() * 0.9999 else "interior")
            r = b["ratios"]
            print(f"{t:6s} knots {k:3.1f}: rms {b['rms']:.3f}; best gamma {g[i]:.2g} ({edge}), allowed {g[ok].min():.2g}-{g[ok].max():.2g} "
                  f"of {g.min():.2g}-{g.max():.2g}; <N_C> {b['mean_nc']:.0f} (C-wt {b['mean_nc_carbon']:.0f}); "
                  f"emission f-/f+ {b['e_anion']:.2f}/{b['e_cation']:.2f} (Z>=2 {b['e_dication']:.2f}); cont {b['cont_share']:.2f}; "
                  f"PAH 3.3/11.2 {r['3.3']:.2f} 6.2 {r['6.2']:.2f} 7.7 {r['7.7']:.2f} 12.7 {r['12.7']:.2f}")


def plot(res, out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = res["grid"]
    fig, axes = plt.subplots(len(C.TEMPLATES), 2, figsize=(16, 3.2 * len(C.TEMPLATES)),
                             gridspec_kw={"width_ratios": [1, 3]})
    for row, t in zip(axes, C.TEMPLATES):
        cells, y = res[t]["cells"], res[t]["observed"]
        b = min(cells, key=lambda c: c[1.5]["sse"])[1.5]
        for ax, seg in zip(row, (0, 1)):
            m = grid.segment == seg
            lam = grid.lam_um[m]
            o = np.argsort(lam)
            ax.plot(lam[o], y[m][o], "k", lw=0.8, label="observed")
            ax.plot(lam[o], (b["pah"] + b["cont"])[m][o], "C3", lw=0.8, label="PAH + continuum")
            ax.plot(lam[o], b["cont"][m][o], "C0", lw=0.8, label="continuum")
            ax.plot(lam[o], b["pah"][m][o], "C2", lw=0.6, alpha=0.8, label="PAH population")
            ax.plot(lam[o], (y - b["pah"] - b["cont"])[m][o], "0.5", lw=0.5, label="residual")
            ax.set_title(f"{t}: rms {b['rms']:.3f}, <N_C>_C {b['mean_nc_carbon']:.0f}", fontsize=9)
        row[1].legend(fontsize=7, ncol=5)
    axes[-1][0].set_xlabel("wavelength [um]")
    axes[-1][1].set_xlabel("wavelength [um]")
    fig.tight_layout()
    path = Path(out_dir) / "population_spectral_fit.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--templates", type=Path, required=True)
    ap.add_argument("--library", choices=("m20", "v400"), default="v400")
    ap.add_argument("--ip-shift", default="")
    ap.add_argument("--capture-scale", type=float, default=1.0)
    ap.add_argument("--capture-model", choices=("bt94", "lab_trend"), default="bt94")
    ap.add_argument("--stretch-scale", default="1")
    ap.add_argument("--intact-min", type=float, default=0.5)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    res = run(args)
    summarise(res)
    if args.plot:
        plot(res, _paths.get_plots_dir("pahchem"))


if __name__ == "__main__":
    main()
