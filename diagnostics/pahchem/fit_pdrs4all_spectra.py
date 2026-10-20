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
  2.5 cm^-1 over 3.1-3.65 and 5-15 um, Chown+24 artefacts and the strong gas
  lines masked (spectral_fit.MASKED_UM).
- Model: ``Population`` with the stated inputs of fit_pdrs4all_population
  (library, IP shifts, capture, stretch scale), charges at each cell of the
  conditions grid, molecules kept where the dehydrogenation chain
  (``Population.intact_fractions``) leaves them intact at that cell.
- Per cell: sse, rms (residual / mean intensity), the population (N_C by number
  and by carbon; charge shares of the emitted power, dications counted with
  the cations), the derived band ratios of the PAH component, and the
  continuum share of the 5-15 um power.
- "Allowed" cells: Delta chi^2 < 9 with chi^2 = sse / s^2, s^2 = sse_min / (n - p),
  p the number of non-zero coefficients. This is an approximate scale; the
  statistics are step 3.

With ``--fit-shape`` the band profile is fitted as well: shifts for the CH
stretch, CC and CH oop modes, and FWHMs for the CH stretch and the rest, shared
by all molecules (``spectral_fit.fit_band_shape``), Lorentzian and Gaussian
each. Per template the shape is fitted at the middle cell, the conditions are
scanned with that shape, and the shape is refitted at the best cell and the
scan repeated. Band ratios are then measured on the fitted PAH spectrum.

    fit_pdrs4all_spectra.py --templates .../orionbar.dat --library v400 --ip-shift=-1:0.37,0:0.6,1:0.64,2:0.64 --stretch-scale lab
"""
import argparse
import dataclasses
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
#: the 3.4 um complex (3.40, 3.46, 3.51, 3.56 um; Peeters+24), measured against the 3.3 um window
ALIPHATIC_UM = (3.36, 3.62)
#: second start of the band-shape fit: the broad-CH-stretch branch found in DF2 (the default start can end
#: in a narrow-band, cation-dominated branch with ~2x the residual; spec, step 1)
BROAD_START = spectral_fit.BandShape(shift_ch=-28.0, shift_cc=16.0, shift_oop=-3.0, fwhm_ch=40.0, fwhm_rest=8.0)


#: bump when the fit code changes what it stores; part of every result-cache key with the code versions below
RESULT_VERSION = 6


def _result_key(args):
    """Cache key of a run: every argument plus the versions of the code that produce the result."""
    stamp = {k: str(v) for k, v in sorted(vars(args).items()) if k != "plot"}
    stamp.update(result=RESULT_VERSION, basis=population.BASIS_VERSION, masks=repr(spectral_fit.MASKED_UM),
                 broad=repr(BROAD_START))
    return hashlib.sha1(json.dumps(stamp, sort_keys=True).encode()).hexdigest()[:12]

def observed_on_grid(lam, fnu, grid):
    s1 = (lam > 2.95) & (lam < 3.8)
    l1, f1 = C._clean(lam[s1], fnu[s1], 0.012)
    s2 = (lam > 4.9) & (lam < 15.5)
    l2, f2 = C._clean(lam[s2], fnu[s2], 0.025)
    return spectral_fit.bin_spectrum(np.concatenate([l1, l2]), np.concatenate([f1, f2]), grid)


def run(args):
    key = _result_key(args)
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
    emitted = np.zeros((len(lib), len(zs)))
    for s, m in enumerate(lib):
        for z in m.charges:
            emitted[s, z - pop.zmin] = pop.basis(m, z, shape).emitted.sum()
    card = conditions.load("orion_bar")
    teff = int(card["region"]["field_teff_k"]["value"])
    bases = {k: spectral_fit.continuum_basis(grid, k) for k in KNOTS}
    i112 = list(FP.WINDOWS).index("11.2")
    out = {"library": [(m.name, m.nc, m.nh, m.edges) for m in lib], "grid": grid, "args": vars(args)}
    for t in C.TEMPLATES:
        y = observed_on_grid(lam, tmpl[t], grid)
        e = card[t]
        temp = conditions.bounds(e["t_gas"])[0]
        cells = []
        for g0 in FP._grid(*(v / conditions.g0_conversion(e["g0"], teff) for v in conditions.bounds(e["g0"])), 4):
            for n_h in FP._grid(*conditions.bounds(e["n_h"]), 3):
                el = e["n_e"] if "n_e" in e else e["x_e"]
                for v in FP._grid(*conditions.bounds(el), 4):
                    ne = v if "n_e" in e else v * n_h
                    env = Environment(f"{t}", g0, n_h, ne, temp, teff)
                    keep = pop.intact_fractions(env) >= args.intact_min           # hydrogenation chain, per cell
                    P = pop.spectra(env, grid.nu)[keep].T
                    F, _ = pop.band_fluxes(env)
                    f = pop.charge_matrix(env)
                    cell = {"g0": g0, "n_h": n_h, "n_e": ne, "gamma": g0 * np.sqrt(temp) / ne}
                    for k in KNOTS:
                        fit = spectral_fit.fit_spectrum(y, P, bases[k])
                        w = np.zeros(len(lib))
                        w[keep] = fit.weights
                        power = w[:, None] * f * emitted        # emitted power by state (photon + recombination heating)
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


def _cells(card, t):
    e = card[t]
    temp = conditions.bounds(e["t_gas"])[0]
    out = []
    teff = int(card["region"]["field_teff_k"]["value"])
    for g0 in FP._grid(*(v / conditions.g0_conversion(e["g0"], teff) for v in conditions.bounds(e["g0"])), 4):
        for n_h in FP._grid(*conditions.bounds(e["n_h"]), 3):
            el = e["n_e"] if "n_e" in e else e["x_e"]
            for v in FP._grid(*conditions.bounds(el), 4):
                out.append((g0, n_h, v if "n_e" in e else v * n_h, temp))
    return out


def run_shape(args, knot=1.5, iterations=2):
    """Band shape fitted per template and profile, then the conditions scan with that shape (see --fit-shape)."""
    key = _result_key(args)
    cache = _paths.get_results_dir() / f"pahchem_spectral_shape_{key}.pkl"
    if cache.exists():
        with open(cache, "rb") as fh:
            return pickle.load(fh)
    if args.aliphatic == "none":
        lib = FP.library(args.library)
    elif args.library == "v400":
        lib = population.library_v400(include_superh=True, cache_dir=_paths.get_model_data_dir() / "pahchem_pahdb_index")
    else:
        raise SystemExit("--aliphatic needs --library v400")
    shifts = {int(k): float(v) for k, v in (kv.split(":") for kv in args.ip_shift.split(",") if kv)}
    if shifts:
        lib = population.shift_thresholds(lib, shifts)
    stretch = population.LAB_STRETCH_SCALE if args.stretch_scale == "lab" else float(args.stretch_scale)
    params = Params(capture_scale=args.capture_scale, capture_model=args.capture_model, stretch_scale=stretch)
    pop = population.Population(lib, FP.WINDOWS, cache_dir=FP.BASIS_CACHE, params=params)
    grid = spectral_fit.Grid.build()
    C_ = spectral_fit.continuum_basis(grid, knot)
    n_ali = np.array([m.edges.get("aliphatic", 0) for m in lib], dtype=float)       # H off the aromatic edges
    nh = np.array([m.nh for m in lib], dtype=float)
    aliphatic = n_ali > 0
    w34 = (grid.lam_um >= ALIPHATIC_UM[0]) & (grid.lam_um < ALIPHATIC_UM[1])
    w33 = (grid.lam_um >= C.WINDOWS["3.3"][0]) & (grid.lam_um < C.WINDOWS["3.3"][1])
    lam, tmpl = C.load_templates(args.templates)
    card = conditions.load("orion_bar")
    teff = int(card["region"]["field_teff_k"]["value"])
    fshape = ("kurucz", teff)
    pop.prepare(fshape)
    nc = np.array([m.nc for m in lib], dtype=float)
    zs = np.array(pop.zs)
    emitted = np.zeros((len(lib), len(zs)))
    for s, m in enumerate(lib):
        for z in m.charges:
            emitted[s, z - pop.zmin] = pop.basis(m, z, fshape).emitted.sum()
    out = {"library": [(m.name, m.nc, m.nh, m.edges) for m in lib], "grid": grid, "args": vars(args), "knot": knot}
    for t in C.TEMPLATES:
        y = observed_on_grid(lam, tmpl[t], grid)
        cells = _cells(card, t)
        out[t] = {"observed": y}
        keeps = [pop.intact_fractions(Environment(t, *c[:3], c[3], teff)) >= args.intact_min for c in cells]
        if args.aliphatic == "free":
            # the chain knows only aromatic H loss; the aliphatic species are let in at every cell, so their
            # weights are what the spectrum asks for, not what the conditions allow
            keeps = [k | aliphatic for k in keeps]
        ok = np.isfinite(y)

        def scan(shape):
            S = spectral_fit.shaped_state_spectra(pop, fshape, grid.nu, shape)
            rows = []
            for (g0, n_h, ne, temp), keep in zip(cells, keeps):
                env = Environment(t, g0, n_h, ne, temp, teff)
                f = pop.charge_matrix(env)
                P = (g0 * np.einsum("sz,szi->si", f, S))[keep].T
                fit = spectral_fit.fit_spectrum(y, P, C_)
                w = np.zeros(len(lib))
                w[keep] = fit.weights
                power = w[:, None] * f * emitted
                tot = power.sum()
                wf = spectral_fit.window_fluxes(grid, fit.pah, FP.WINDOWS)
                pah_ali = P[:, aliphatic[keep]] @ fit.weights[aliphatic[keep]]
                band = y - fit.cont
                r34 = lambda v: float(v[w34 & ok].sum() / v[w33 & ok].sum())
                rows.append({"ali_h": float(w @ n_ali / (w @ nh)) if w.sum() > 0 else np.nan,
                             "r34_obs": r34(band), "r34_pah": r34(fit.pah), "r34_ali": r34(pah_ali),
                             "rms34": float(np.sqrt(np.mean((band - fit.pah)[w34 & ok] ** 2)) / y[w34 & ok].mean()),
                             "pah_ali": pah_ali,
"g0": g0, "n_h": n_h, "n_e": ne, "gamma": g0 * np.sqrt(temp) / ne, "sse": fit.sse,
                             "rms": fit.rms, "n": int(np.isfinite(y).sum()),
                             "p": int((fit.weights > 0).sum() + (fit.continuum > 0).sum()), "weights": w,
                             "mean_nc_carbon": float((w * nc) @ nc / (w @ nc)) if w.sum() > 0 else np.nan,
                             "e_anion": float(power[:, zs < 0].sum() / tot), "e_cation": float(power[:, zs > 0].sum() / tot),
                             "e_dication": float(power[:, zs >= 2].sum() / tot),
                             "ratios": {b: v / wf["11.2"] for b, v in wf.items()},
                             "cont_share": float(fit.cont[grid.segment == 1].sum() / max(
                                 fit.cont[grid.segment == 1].sum() + fit.pah[grid.segment == 1].sum(), 1e-300)),
                             "pah": fit.pah, "cont": fit.cont})
            return rows

        for profile in ("lorentzian", "gaussian"):
            runs = []
            # each start goes through the whole alternation: shape and gamma are coupled, and a start
            # compared only at one cell can pick the wrong branch (two exist; spec, step 1)
            for start in (spectral_fit.BandShape(profile=profile), dataclasses.replace(BROAD_START, profile=profile)):
                # the first shape fit is made at the best cell of a scan with the start shape itself: fitted at a
                # fixed (high-gamma) cell, every start fell into the narrow branch
                ref, shape, history = int(np.argmin([r["sse"] for r in scan(start)])), start, []
                for it in range(iterations):
                    g0, n_h, ne, temp = cells[ref]
                    shape, _, _ = spectral_fit.fit_band_shape(y, grid, pop, Environment(t, g0, n_h, ne, temp, teff),
                                                              keeps[ref], C_, start=shape)
                    rows = scan(shape)
                    ref = int(np.argmin([r["sse"] for r in rows]))
                    history.append((shape, rows[ref]["gamma"], rows[ref]["rms"]))
                runs.append({"shape": shape, "history": history, "cells": rows, "start": start,
                             "best_sse": rows[ref]["sse"]})
            best = min(runs, key=lambda r: r["best_sse"])
            out[t][profile] = best | {"alternatives": [{k: r[k] for k in ("shape", "history", "best_sse", "start")}
                                                       for r in runs]}
            for r in runs:
                tag = "*" if r is best else " "
                print(f"{t} {profile} {tag} start {r['start'].shift_ch:+.0f}/{r['start'].fwhm_ch:.0f}: "
                      + " -> ".join(f"[rms {h[2]:.3f}, gamma {h[1]:.2g}]" for h in r["history"])
                      + f"; shape CH {r['shape'].shift_ch:+.0f}/{r['shape'].fwhm_ch:.0f}, CC {r['shape'].shift_cc:+.0f}, "
                      f"oop {r['shape'].shift_oop:+.0f}, rest FWHM {r['shape'].fwhm_rest:.0f}", flush=True)
    with open(cache, "wb") as fh:
        pickle.dump(out, fh)
    return out


def summarise_shape(res):
    for t in C.TEMPLATES:
        for profile in ("lorentzian", "gaussian"):
            r = res[t][profile]
            cells = r["cells"]
            g = np.array([c["gamma"] for c in cells])
            sse = np.array([c["sse"] for c in cells])
            i = int(sse.argmin())
            b = cells[i]
            s2 = sse.min() / max(b["n"] - b["p"], 1)
            ok = (sse - sse.min()) / s2 < 9.0
            edge = "low edge" if g[i] <= g.min() * 1.0001 else ("high edge" if g[i] >= g.max() * 0.9999 else "interior")
            sh = r["shape"]
            q = b["ratios"]
            print(f"{t:6s} {profile:10s} rms {b['rms']:.3f}; shifts CH/CC/oop {sh.shift_ch:+.0f}/{sh.shift_cc:+.0f}/{sh.shift_oop:+.0f}, "
                  f"FWHM CH/rest {sh.fwhm_ch:.0f}/{sh.fwhm_rest:.0f}; best gamma {g[i]:.2g} ({edge}), allowed "
                  f"{g[ok].min():.2g}-{g[ok].max():.2g}; <N_C>_C {b['mean_nc_carbon']:.0f}; emission f-/f+ "
                  f"{b['e_anion']:.2f}/{b['e_cation']:.2f} (Z>=2 {b['e_dication']:.2f}); cont {b['cont_share']:.2f}; "
                  f"PAH 3.3/11.2 {q['3.3']:.2f} 6.2 {q['6.2']:.2f} 7.7 {q['7.7']:.2f} 12.7 {q['12.7']:.2f}; "
                  f"3.4/3.3 obs {b['r34_obs']:.3f} PAH {b['r34_pah']:.3f} (aliphatic species {b['r34_ali']:.3f}), "
                  f"rms 3.36-3.62 {b['rms34']:.3f}, H off-edge {b['ali_h']:.3f}")


def plot_shape(res, out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = res["grid"]
    fig, axes = plt.subplots(len(C.TEMPLATES), 2, figsize=(16, 3.2 * len(C.TEMPLATES)),
                             gridspec_kw={"width_ratios": [1, 3]})
    for row, t in zip(axes, C.TEMPLATES):
        y = res[t]["observed"]
        prof = min(("lorentzian", "gaussian"), key=lambda p: min(c["sse"] for c in res[t][p]["cells"]))
        b = min(res[t][prof]["cells"], key=lambda c: c["sse"])
        sh = res[t][prof]["shape"]
        for ax, seg in zip(row, (0, 1)):
            m = grid.segment == seg
            lam = grid.lam_um[m]
            o = np.argsort(lam)
            ax.plot(lam[o], y[m][o], "k", lw=0.8, label="observed")
            ax.plot(lam[o], (b["pah"] + b["cont"])[m][o], "C3", lw=0.8, label="PAH + continuum")
            ax.plot(lam[o], b["cont"][m][o], "C0", lw=0.8, label="continuum")
            ax.plot(lam[o], b["pah"][m][o], "C2", lw=0.6, alpha=0.8, label="PAH population")
            if b["pah_ali"].any():
                ax.plot(lam[o], (b["pah_ali"] + b["cont"])[m][o], "C1", lw=0.6, label="aliphatic species + continuum")
            ax.plot(lam[o], (y - b["pah"] - b["cont"])[m][o], "0.5", lw=0.5, label="residual")
        row[1].set_title(f"{t} ({prof}): rms {b['rms']:.3f}, gamma {b['gamma']:.2g}, <N_C>_C {b['mean_nc_carbon']:.0f}; "
                         f"shifts {sh.shift_ch:+.0f}/{sh.shift_cc:+.0f}/{sh.shift_oop:+.0f}, FWHM {sh.fwhm_ch:.0f}/{sh.fwhm_rest:.0f}",
                         fontsize=9)
        row[1].legend(fontsize=7, ncol=5)
    fig.tight_layout()
    tag = "" if res["args"].get("aliphatic", "none") == "none" else f"_aliphatic_{res['args']['aliphatic']}"
    path = Path(out_dir) / f"population_spectral_fit_shape{tag}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


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
    ap.add_argument("--aliphatic", choices=("none", "free"), default="none",
                    help="with --fit-shape: add the v4.00 species with H off the aromatic edges; 'free' lets them in "
                         "at every cell (no hydrogenation constraint)")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--fit-shape", action="store_true",
                    help="fit the global band shape (shifts per mode class, two widths, Lorentzian or Gaussian) per template")
    args = ap.parse_args()
    if args.fit_shape:
        res = run_shape(args)
        summarise_shape(res)
        if args.plot:
            plot_shape(res, _paths.get_plots_dir("pahchem"))
        return
    res = run(args)
    summarise(res)
    if args.plot:
        plot(res, _paths.get_plots_dir("pahchem"))


if __name__ == "__main__":
    main()
