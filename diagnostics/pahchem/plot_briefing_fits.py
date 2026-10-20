#!/usr/bin/env python3
"""Briefing figures of the PDRs4All spectral fits: how a model spectrum is built, each fit, and what they imply.

Reads a cached ``fit_pdrs4all_spectra.py --fit-shape`` result (no refit) and
rebuilds, at each template's best cell, the per-state spectra with the fitted
band shape, so that the fitted PAH spectrum can be split by charge. The split
is checked to add up to the cached PAH spectrum.

Figures, written to ``--out`` (default ``<plots dir>/pahchem/briefing``):
- ``fit_00_how_a_model_spectrum_is_built.png``: one molecule from IR modes to the fitted population;
- ``fit_01_<template>.png``: observed spectrum, continuum, PAH emission by charge, residuals;
- ``fit_02_summary.png``: fit quality and derived properties of the five templates.

Run from the worktree root (writable paths resolve by cwd):
  CALIMA_PAHDB_DIR=<main checkout>/reference_data PYTHONPATH=src:diagnostics/pahchem \\
      python diagnostics/pahchem/plot_briefing_fits.py --cache results/pahchem_spectral_shape_3ddd190a5731.pkl
"""
import argparse
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import compare_pdrs4all as C  # noqa: E402
import fit_pdrs4all_population as FP  # noqa: E402
from pycalima import _paths  # noqa: E402
from pycalima.plotting_style import use_calima_style  # noqa: E402
from pycalima.pahchem import conditions, population, spectral_fit  # noqa: E402
from pycalima.pahchem.baseline import Environment, Params  # noqa: E402

#: charge classes of the split, in stacking order, with their colours (validated categorical palette, slots 1-4)
CLASSES = (("anions", lambda z: z < 0, "#2a78d6"), ("neutrals", lambda z: z == 0, "#eb6834"),
           ("cations", lambda z: z == 1, "#1baf7a"), ("dications and higher", lambda z: z >= 2, "#eda100"))
CONTINUUM = "#d6d5d0"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
TITLES = {"HII": r"H\,{\sc ii} region (background face-on PDR)", "Atomic": "Atomic PDR",
          "DF1": "Dissociation front 1", "DF2": "Dissociation front 2", "DF3": "Dissociation front 3"}
SEGMENTS = ((3.1, 3.65), (5.0, 15.0))
BANDS_UM = (3.29, 6.2, 7.7, 8.6, 11.2, 12.7)
LAM = r"$\lambda$ [$\mu$m]"
INU = r"$I_\nu$ [MJy sr$^{-1}$]"


def style():
    """pyCALIMA's figure style (seaborn white, serif, LaTeX when available)."""
    use_calima_style()
    plt.rcParams.update({"savefig.dpi": 300, "legend.frameon": False})


def ticks(ax, labelsize=12):
    """The house tick convention: minor ticks, all ticks pointing in."""
    ax.minorticks_on()
    ax.tick_params(which="both", axis="both", direction="in", labelsize=labelsize)


def sci(v):
    """A number in LaTeX: 2.6, 1.9\\times10^{4}."""
    m, e = f"{v:.2g}".split("e") if "e" in f"{v:.2g}" else (f"{v:.2g}", None)
    return m if e is None else (rf"{m}\times10^{{{int(e)}}}" if m != "1" else rf"10^{{{int(e)}}}")


def load(cache):
    with open(cache, "rb") as fh:
        res = pickle.load(fh)
    a = res["args"]
    lib = FP.library(a["library"])
    shifts = {int(k): float(v) for k, v in (kv.split(":") for kv in a["ip_shift"].split(",") if kv)}
    if shifts:
        lib = population.shift_thresholds(lib, shifts)
    if [m.name for m in lib] != [r[0] for r in res["library"]]:
        raise SystemExit("the library differs from the cached run's; refit with fit_pdrs4all_spectra.py")
    stretch = population.LAB_STRETCH_SCALE if a["stretch_scale"] == "lab" else float(a["stretch_scale"])
    params = Params(capture_scale=a["capture_scale"], capture_model=a["capture_model"], stretch_scale=stretch)
    pop = population.Population(lib, FP.WINDOWS, cache_dir=FP.BASIS_CACHE, params=params)
    card = conditions.load("orion_bar")
    teff = int(card["region"]["field_teff_k"]["value"])
    fshape = ("kurucz", teff)
    pop.prepare(fshape)
    out = {}
    for t in C.TEMPLATES:
        prof = min(("lorentzian", "gaussian"), key=lambda p: min(c["sse"] for c in res[t][p]["cells"]))
        run = res[t][prof]
        cells = run["cells"]
        sse = np.array([c["sse"] for c in cells])
        b = cells[int(sse.argmin())]
        s2 = sse.min() / max(b["n"] - b["p"], 1)
        allowed = [c for c, s in zip(cells, sse) if (s - sse.min()) / s2 < 9.0]
        env = Environment(t, b["g0"], b["n_h"], b["n_e"], conditions.bounds(card[t]["t_gas"])[0], teff)
        S = spectral_fit.shaped_state_spectra(pop, fshape, res["grid"].nu, run["shape"])
        f = pop.charge_matrix(env)
        by_z = b["g0"] * np.einsum("s,sz,szi->zi", b["weights"], f, S)
        if not np.allclose(by_z.sum(0), b["pah"], rtol=1e-8, atol=1e-12 * b["pah"].max()):
            raise SystemExit(f"{t}: the charge split does not add up to the cached PAH spectrum")
        split = {name: by_z[[i for i, z in enumerate(pop.zs) if sel(z)]].sum(0) for name, sel, _ in CLASSES}
        out[t] = {"best": b, "allowed": allowed, "cells": cells, "profile": prof, "shape": run["shape"], "env": env, "S": S,
                  "f": f, "split": split, "observed": res[t]["observed"]}
    return res, pop, out


def _segments(lam, *ys, max_gap_um=0.05):
    """Sort by wavelength and break lines at gaps wider than ``max_gap_um`` (artefacts, unfitted ranges).

    The +-0.01 um gas-line masks are narrower and are bridged."""
    o = np.argsort(lam)
    lam = lam[o]
    gaps = np.flatnonzero(np.diff(lam) > max_gap_um) + 1
    lam = np.insert(lam, gaps, np.nan)
    return (lam,) + tuple(np.insert(np.asarray(y, dtype=float)[o], gaps, np.nan) for y in ys)


def fig_template(t, d, grid, out_dir):
    b, y = d["best"], d["observed"]
    model = b["pah"] + b["cont"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 7.2), sharex="col", facecolor="w",
                             gridspec_kw={"width_ratios": [1, 3.2], "height_ratios": [3, 1], "hspace": 0,
                                          "wspace": 0.14})
    for col, (lo, hi) in enumerate(SEGMENTS):
        m = (grid.lam_um >= lo) & (grid.lam_um <= hi)
        lam, yo, cont, mod = _segments(grid.lam_um[m], y[m], b["cont"][m], model[m])
        parts = [_segments(grid.lam_um[m], d["split"][name][m])[1] for name, _, _ in CLASSES]
        ax, axr = axes[0, col], axes[1, col]
        base = cont.copy()
        ax.fill_between(lam, 0, base, color=CONTINUUM, lw=0, label="continuum (B-splines)")
        for (name, _, colour), p in zip(CLASSES, parts):
            ax.fill_between(lam, base, base + p, color=colour, lw=0, alpha=0.85, label=f"PAH {name}")
            base = base + p
        ax.plot(lam, mod, color=INK2, lw=1.0, label="model total")
        ax.plot(lam, yo, color=INK, lw=1.4, label="observed (gas lines removed)")
        ax.set_xlim(lo, hi)
        ax.set_ylim(0, 1.12 * np.nanmax(yo))
        axr.plot(lam, 100 * (yo - mod) / np.nanmean(yo), color=INK2, lw=1.0)
        axr.axhline(0, color=MUTED, lw=0.8)
        axr.set_ylim(-28, 28)
        axr.set_xlabel(LAM, fontsize=15)
        for a in (ax, axr):
            ticks(a)
        if col == 1:
            for band in BANDS_UM[1:]:
                ax.annotate(f"{band:g}", (band, 0.975), xycoords=("data", "axes fraction"), ha="center", va="top",
                            fontsize=11, color=MUTED)
            # direct labels on the PAH areas where each is thickest (the colours alone are below 3:1 contrast)
            below = np.nan_to_num(cont)
            inner = (lam > 5.6) & (lam < 13.0)
            for (name, _, colour), p in zip(CLASSES, parts):
                pp = np.where(inner, np.nan_to_num(p), 0.0)
                i = int(np.argmax(pp))
                if pp[i] > 0.06 * np.nanmax(mod):
                    ax.annotate(name, (lam[i] + 0.12, below[i] + 0.5 * pp[i]), fontsize=12, color=INK, va="center",
                                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
                below = below + np.nan_to_num(p)
        else:
            ax.annotate("3.29", (3.29, 0.975), xycoords=("data", "axes fraction"), ha="center", va="top",
                        fontsize=11, color=MUTED)
            ax.annotate("3.4 complex:\nnot in the model", (3.415, 0.62), xycoords=("data", "axes fraction"),
                        fontsize=11, color=INK2)
    axes[0, 0].set_ylabel(INU, fontsize=15)
    axes[1, 0].set_ylabel("residual\n" + r"[\% of mean $I_\nu$]", fontsize=13)
    axes[0, 1].legend(loc="upper left", fontsize=11.5, ncol=2, bbox_to_anchor=(0.0, 0.93))
    sh, env = d["shape"], d["env"]
    fig.suptitle(rf"{TITLES[t]}: PAH population and continuum fitted together "
                 rf"(rms residual {100 * b['rms']:.1f}\% of the mean intensity)", x=0.07, y=1.0, ha="left",
                 fontsize=16)
    fig.text(0.07, 0.935,
             rf"Best cell of the stated conditions: $G_0 = {sci(env.g0)}$ (Habing), $n_{{\rm H}} = {sci(env.n_h)}$ "
             rf"cm$^{{-3}}$, $n_e = {sci(env.n_e)}$ cm$^{{-3}}$, $T = {env.temperature_k:.0f}$ K, "
             rf"$\gamma = G_0 T^{{1/2}}/n_e = {sci(b['gamma'])}$. "
             rf"Band shape ({d['profile']}): CH stretch ${sh.shift_ch:+.0f}$ cm$^{{-1}}$, FWHM {sh.fwhm_ch:.0f}; "
             rf"CC ${sh.shift_cc:+.0f}$; CH oop ${sh.shift_oop:+.0f}$; other FWHM {sh.fwhm_rest:.0f} cm$^{{-1}}$.",
             fontsize=12, color=INK2)
    path = Path(out_dir) / f"fit_01_{t}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_pipeline(t, d, pop, grid, out_dir):
    """One molecule, from its IR modes to the fitted population, at the best cell of template t."""
    b, env, f, S = d["best"], d["env"], d["f"], d["S"]
    contrib = b["weights"] * (b["g0"] * np.einsum("sz,szi->si", f, S)).sum(1)
    s = int(np.argmax(contrib))
    mol = pop.library[s]
    name = mol.name.split(" ")[0]
    formula = "".join(rf"$_{{{c}}}$" if c.isdigit() else c for c in name)
    fs = ("kurucz", env.teff_k)
    zmain = [z for z in mol.charges if z in (0, 1)]
    lam_all = grid.lam_um
    fig, axes = plt.subplots(4, 1, figsize=(12, 13), sharex=True, facecolor="w", gridspec_kw={"hspace": 0.32})
    colours = {name_: c for name_, _, c in CLASSES}
    cls = lambda z: next(n for n, sel, _ in CLASSES if sel(z))  # noqa: E731
    title = dict(loc="left", fontsize=12.5)
    # 1: emitted power per mode after the absorbed photons, cascade folded with the heating spectrum
    ax = axes[0]
    for z in zmain:
        bb = pop.basis(mol, z, fs)
        lam = 1e4 / np.asarray(bb.freqs_cm)
        e = np.asarray(bb.emitted) / np.max(bb.emitted)
        ax.vlines(lam, 0, e, color=colours[cls(z)], lw=1.0, label=rf"{formula}, charge ${z:+d}$")
    ax.set_ylabel("emitted power\nper mode (norm.)", fontsize=13)
    ax.set_title("1. Each charge state: the IR power of every vibrational mode, from the exact cascade of the "
                 "absorbed UV photons (PAHdb modes)", **title)
    ax.legend(fontsize=11, loc="upper right")
    # 2: the fitted band shape
    ax = axes[1]
    for z in zmain:
        yz = S[s, z - pop.zmin]
        lam, yy = _segments(lam_all, yz / yz.max())
        ax.plot(lam, yy, color=colours[cls(z)], lw=1.5, label=rf"charge ${z:+d}$")
    sh = d["shape"]
    ax.set_ylabel("state spectrum\n(norm.)", fontsize=13)
    ax.set_title(rf"2. A band profile for every mode: anharmonic shift and width per mode class, fitted once per "
                 rf"template ({d['profile']}; CH ${sh.shift_ch:+.0f}$/{sh.fwhm_ch:.0f}, CC ${sh.shift_cc:+.0f}$, "
                 rf"oop ${sh.shift_oop:+.0f}$, FWHM {sh.fwhm_rest:.0f} cm$^{{-1}}$)", **title)
    ax.legend(fontsize=11, loc="upper right")
    # 3: charge fractions at the conditions
    ax = axes[2]
    tot = np.zeros(lam_all.size)
    for z in mol.charges:
        tot += env.g0 * f[s, z - pop.zmin] * S[s, z - pop.zmin]
    norm = tot.max()
    base = np.zeros(lam_all.size)
    for z in mol.charges:
        part = env.g0 * f[s, z - pop.zmin] * S[s, z - pop.zmin] / norm
        lam, lo_, hi_ = _segments(lam_all, base, base + part)
        ax.fill_between(lam, lo_, hi_, color=colours[cls(z)], lw=0, alpha=0.85,
                        label=rf"charge ${z:+d}$: $f = {f[s, z - pop.zmin]:.2f}$")
        base = base + part
    ax.set_ylabel("molecule spectrum\n(norm.)", fontsize=13)
    ax.set_title(rf"3. Charge balance from the local conditions (photoionisation against electron capture, "
                 rf"$\gamma = {sci(b['gamma'])}$): the molecule's spectrum is the $f$-weighted sum", **title)
    ax.legend(fontsize=11, loc="upper right")
    # 4: population + continuum against the observation
    ax = axes[3]
    yv = d["observed"]
    lam, yo, cont, mod, pah, one = _segments(lam_all, yv, b["cont"], b["pah"] + b["cont"], b["pah"],
                                             b["weights"][s] * tot)
    ax.fill_between(lam, 0, cont, color=CONTINUUM, lw=0, label="continuum")
    ax.fill_between(lam, cont, cont + pah, color="#c9c3f2", lw=0,
                    label=rf"all PAHs: {int((b['weights'] > 0).sum())} molecules with $w > 0$")
    ax.fill_between(lam, cont, cont + one, color="#6250d6", lw=0,
                    label=rf"of which {formula} (all charges), {100 * contrib[s] / contrib.sum():.0f}\% of the PAH power")
    ax.plot(lam, mod, color=INK2, lw=1.0, label="model total")
    ax.plot(lam, yo, color=INK, lw=1.3, label=f"observed, {TITLES[t]}")
    ax.set_ylabel(INU, fontsize=14)
    ax.set_title("4. Non-negative least squares over the PAHdb library (abundances only) plus a positive "
                 "B-spline continuum, on the spectrum itself", **title)
    ax.legend(fontsize=11, loc="upper left")
    ax.set_xlabel(LAM, fontsize=15)
    ax.set_xscale("log")
    ax.set_xlim(3.1, 15.0)
    for a in axes:
        ticks(a)
        a.axvspan(3.65, 5.0, color="#f4f3f0", lw=0)
    tick_at = [3.3, 4, 5, 6, 7, 8, 10, 12, 15]
    ax.set_xticks(tick_at, [f"{v:g}" for v in tick_at])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[0].annotate("not fitted", (4.3, 0.5), xycoords=("data", "axes fraction"), ha="center", fontsize=11,
                     color=MUTED)
    fig.suptitle(f"How a model spectrum is built ({TITLES[t]}, best cell)", x=0.07, y=0.93, ha="left",
                 fontsize=16)
    path = Path(out_dir) / "fit_00_how_a_model_spectrum_is_built.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path, mol.name


def fig_summary(data, lib, out_dir):
    ts = list(C.TEMPLATES)
    x = np.arange(len(ts))
    labels = [r"H\,{\sc ii}" if t == "HII" else t for t in ts]
    nc = np.array([r[1] for r in lib], dtype=float)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), facecolor="w", gridspec_kw={"hspace": 0.38, "wspace": 0.32})
    dot = dict(color="#2a78d6", ms=9, marker="o", ls="none", zorder=3)

    def panel(ax, vals, title, ylabel, lohi=None, logy=False, note=None):
        vals = np.asarray(vals, dtype=float)
        if lohi is not None:
            lo, hi = np.asarray(lohi, dtype=float).T
            ax.vlines(x, lo, hi, color="#86b6ef", lw=7, zorder=2, label=r"allowed cells ($\Delta\chi^2 < 9$)")
        ax.plot(x, vals, **dot, label="best cell")
        ax.set_xticks(x, labels)
        ax.set_xlim(-0.5, len(ts) - 0.5)
        ax.set_title(title, loc="left", fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        if logy:
            ax.set_yscale("log")
        ticks(ax)
        ax.tick_params(axis="x", which="minor", bottom=False, top=False)
        if note:
            ax.annotate(note, (0.03, 0.05), xycoords="axes fraction", fontsize=10.5, color=INK2)

    b = [data[t]["best"] for t in ts]
    al = [data[t]["allowed"] for t in ts]
    rng = lambda key: [(min(c[key] for c in a), max(c[key] for c in a)) for a in al]  # noqa: E731
    panel(axes[0, 0], [100 * c["rms"] for c in b], "Fit quality", r"rms residual [\% of mean $I_\nu$]")
    axes[0, 0].set_ylim(0, 12)
    ax = axes[0, 1]
    lo, hi = np.array([(min(c["gamma"] for c in data[t]["cells"]), max(c["gamma"] for c in data[t]["cells"]))
                       for t in ts]).T
    ax.vlines(x, lo, hi, color="#e4e3df", lw=12, zorder=1, label="scanned (stated conditions)")
    panel(ax, [c["gamma"] for c in b], "Ionisation parameter", r"$\gamma = G_0 T^{1/2}/n_e$ [K$^{1/2}$ cm$^3$]",
          lohi=rng("gamma"), logy=True)
    ax.set_ylim(2e3, 8e5)
    ax.legend(fontsize=10.5, loc="upper left")
    panel(axes[0, 2], [c["e_cation"] for c in b], "Cations in the PAH emission",
          r"share of emitted power ($Z \geq +1$)", lohi=rng("e_cation"))
    axes[0, 2].set_ylim(0, 1)
    panel(axes[1, 0], [c["mean_nc_carbon"] for c in b], "Size of the emitting population",
          r"$\langle N_{\rm C} \rangle$, carbon-weighted", lohi=rng("mean_nc_carbon"))
    axes[1, 0].set_ylim(0, 150)
    panel(axes[1, 1], [c["cont_share"] for c in b], r"Continuum under the 5--15 $\mu$m emission",
          "continuum / (continuum + PAH)", lohi=rng("cont_share"))
    axes[1, 1].set_ylim(0, 1)
    nmol = [int((c["weights"] > 0).sum()) for c in b]
    panel(axes[1, 2], nmol, "Molecules in the best fit", r"number with $w > 0$",
          note=rf"library: {len(lib)} molecules, $N_{{\rm C}}$ = {nc.min():.0f}--{nc.max():.0f}")
    axes[1, 2].set_ylim(0, None)
    fig.suptitle("What the five Orion Bar fits imply (templates ordered from the star into the cloud)", x=0.07,
                 ha="left", fontsize=16)
    path = Path(out_dir) / "fit_02_summary.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", type=Path, required=True, help="pahchem_spectral_shape_<key>.pkl")
    ap.add_argument("--pipeline-template", default="DF2")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or (_paths.get_plots_dir("pahchem") / "briefing")
    out.mkdir(parents=True, exist_ok=True)
    style()
    res, pop, data = load(args.cache)
    for t in C.TEMPLATES:
        print("wrote", fig_template(t, data[t], res["grid"], out))
    path, name = fig_pipeline(args.pipeline_template, data[args.pipeline_template], pop, res["grid"], out)
    print("wrote", path, f"(molecule {name})")
    print("wrote", fig_summary(data, res["library"], out))
    for t in C.TEMPLATES:
        b, sh = data[t]["best"], data[t]["shape"]
        g = [c["gamma"] for c in data[t]["allowed"]]
        print(f"{t}: {data[t]['profile']} rms {b['rms']:.3f}; gamma {b['gamma']:.2g} (allowed {min(g):.2g}-{max(g):.2g}); "
              f"cation share {b['e_cation']:.2f} (Z>=2 {b['e_dication']:.2f}, anion {b['e_anion']:.2f}); "
              f"<N_C>_C {b['mean_nc_carbon']:.0f}; continuum {b['cont_share']:.2f}; molecules {(b['weights'] > 0).sum()}; "
              f"shape CH {sh.shift_ch:+.0f}/{sh.fwhm_ch:.0f} CC {sh.shift_cc:+.0f} oop {sh.shift_oop:+.0f} FWHM {sh.fwhm_rest:.0f}")


if __name__ == "__main__":
    main()
