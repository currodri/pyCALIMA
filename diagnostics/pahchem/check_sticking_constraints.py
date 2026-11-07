#!/usr/bin/env python3
"""The two electron sticking factors against what constrains them: the figures of pahchem-gas-coupling.md section 2.2b.

- ``pahchem_sticking_lab.png``:
  (a) the electron-PAH cation recombination data at 300 K (``pah_electron_recombination.csv``) over BT94 with sticking 1,
      and the cation factor of each ``Params.capture_model`` (``population._capture_factor``);
  (b) the attachment fraction f of the attachment models (``population.attachment_fraction``) against the anchors:
      Carelli+13's temporary-anion rates over pahchem's classical capture at 10-1000 K (upper bounds,
      ``check_carelli13_attachment.py``), anthracene in the flow tubes (48-300 K, collisionally stabilised) and C60
      (Viggiano+10, 400 and 625 K, a factor 3). The EA logistic is shown per library molecule (its f depends on the EA).
- ``pahchem_pdrs4all_sticking.png``: rms of the PDRs4All Orion Bar fit (``fit_pdrs4all_spectra.py --fit-shape``, the
  standard settings) at each of the card's 48 condition cells, against gamma = G0 sqrt(T) / n_e, per template (the
  template's better profile). Top: the cation capture models (full fits). Middle and bottom: the attachment models with
  the band shape of the s = 1, f = 1 fit held (``--shape-from``), and the anion share of the PAH emission. Read from
  the cached fits in results/ whose key matches the current code, so those fits must have been run.
- ``pahchem_w08_attachment.png``: alpha(C+) on PAHs over Wolfire+08's (Phi_PAH = 0.4) at W08's CNM G0, T and
  abundances, n = 3-300 cm^-3 (``check_wolfire08_cplus.py``), with s = 1 on cations so that only f changes. The PAH carbon of W08's x_PAH = 6e-7 at N_C = 35
  (2.1e-5 C per H) is shared equally by carbon among the library's N_C 32-40 (W08's size) or N_C 50-100.

Nothing is fitted or tuned here.

Result (2026-11-02):
- Cation sticking: N_s 23.9, b 2.63 (cold, 5 points) and 23.1, 3.11 (every PAH, 8 points); s = 0.29, 0.63-0.68 and 1.00
  at N_C 16, 24 and 54. WD01's ratio is flat, 0.14-0.19 over N_C 6-300: it meets C10-C14 and misses pyrene's 0.37.
- Attachment anchors: f = 0.0014-0.0044 (Carelli+13, C14-C24), 0.0016-0.0049 (anthracene flow tubes), 0.72 and 0.47
  (C60, 400 and 625 K). The size logistic (N_half 50) gives 0.0030, 0.0036 and 0.86 at N_C 14, 24 and 60; the EA
  logistic (2.5 eV) a median 0.03 (16-84 %: 0.006-0.40) over the 514 library anions of N_C 40-120.
- PDRs4All: section 2.2b's table. In HII and Atomic WD01's rms still falls towards the card's lowest gamma; in the
  DFs every model is best at the lowest gamma, and f = 1 beats f <= 0.1 at every cell.
- Wolfire+08 at n = 30: alpha(C+)/W08 0.78, 0.20 and 0.23 (f = 1, f = 0.003, size logistic) for N_C 32-40, 0.88, 0.14
  and 0.83 for N_C 50-100; over n = 3-300 the size logistic gives 0.16-0.83 and 0.83-0.91.

    check_sticking_constraints.py [--out DIR] [--workers 8]
"""
import argparse
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import check_carelli13_attachment as CA
import check_wolfire08_cplus as W
import fit_pdrs4all_spectra as F

from pycalima import _paths
from pycalima.pahchem import chemistry
from pycalima.pahchem import population as P
from pycalima.pahchem.baseline import Params
from pycalima.plotting_style import use_calima_style

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"

#: anthracene attachment [cm^3 s^-1] at T [K]: Moustefaoui+98 (CRESU, normalised to FALP at 300 K), as in check_carelli13_attachment.py
ANTHRACENE_LAB = {48: 1.7e-9, 75: 8.4e-10, 170: 2.6e-9, 300: 1.0e-9}
#: C60 attachment [cm^3 s^-1] at T [K], uncertain by a factor 3 (Viggiano+10; pahchem-gas-coupling-literature.md)
C60_LAB = {400: 1.2e-6, 625: 8e-7}
C60_FACTOR = 3.0
#: <N_C> by carbon of the emitting PAHs in the PDRs4All fits (section 2.2b)
ORION_NC = (76, 87)

#: label -> (Params overrides, colour, line style)
CAPTURE_CURVES = {r"$s = 1$ (BT94)": ({"capture_model": "bt94"}, "k", "-"),
                  "lab fit, photo-ions (cold; default)": ({"capture_model": "lab_cold"}, BLUE, "-"),
                  r"lab fit, every PAH ($N_{\rm C} \geq 10$)": ({"capture_model": "lab_pahs"}, ORANGE, "--"),
                  "lab power law (with benzene)": ({"capture_model": "lab_trend"}, AQUA, "-."),
                  "WD01 (graphite sphere)": ({"capture_model": "wd01"}, YELLOW, ":")}
ATTACH_CURVES = {r"$f = 1$": ({"attachment_model": "constant"}, "k", "-"),
                 r"$f = 0.003$": ({"attachment_model": "constant", "attachment_fraction": 0.003}, "k", ":"),
                 r"size logistic, $N_{1/2} = 50$ (default)": ({"attachment_model": "size", "attachment_nc_half": 50.0}, BLUE, "-"),
                 r"size logistic, $N_{1/2} = 30$": ({"attachment_model": "size", "attachment_nc_half": 30.0}, BLUE, "--"),
                 r"size logistic, $N_{1/2} = 80$": ({"attachment_model": "size", "attachment_nc_half": 80.0}, BLUE, ":")}
EA_MODEL = ({"attachment_model": "ea", "attachment_ea_half": 2.5}, ORANGE)

#: the PDRs4All runs: label -> (fit arguments that differ from the defaults, colour, line style, marker). "shape_from": True
#: is the band shape of the all-default fit, held fixed.
STANDARD = {"templates": "results/pahchem/data/vdp25/orionbar.dat", "library": "v400", "ip_shift": "-1:0.37,0:0.6,1:0.64,2:0.64",
            "stretch_scale": "lab", "capture_scale": "1.0", "intact_min": "0.5", "hydrogenation": "emit", "aliphatic": "none",
            "fit_shape": "True"}
#: what a cached fit without the argument ran with (the fit's defaults before 2026-11-02)
DEFAULTS = {"capture_model": "bt94", "capture_t_power": "0.0", "attachment_fraction": "1.0", "attachment_model": "constant",
            "attachment_ea_half": "2.0", "attachment_nc_half": "50.0", "shape_from": "None"}
CAPTURE_RUNS = {r"$s = 1$": ({}, "k", "-", "o"),
                "lab fit, cold": ({"capture_model": "lab_cold"}, BLUE, "-", "s"),
                "lab fit, every PAH": ({"capture_model": "lab_pahs"}, ORANGE, "--", "^"),
                "lab power law": ({"capture_model": "lab_trend"}, AQUA, "-.", "v"),
                "WD01 capture": ({"capture_model": "wd01"}, YELLOW, ":", "D")}
ATTACH_RUNS = {r"$f = 1$": ({"shape_from": True}, "k", "-", "o"),
               r"$f = 0.1$": ({"shape_from": True, "attachment_fraction": "0.1"}, "k", "--", "^"),
               r"$f = 0.003$": ({"shape_from": True, "attachment_fraction": "0.003"}, "k", ":", "v"),
               r"size, $N_{1/2} = 50$": ({"shape_from": True, "attachment_model": "size"}, BLUE, "-", "s"),
               r"size, $N_{1/2} = 80$": ({"shape_from": True, "attachment_model": "size", "attachment_nc_half": "80.0"}, BLUE, ":", "D"),
               r"EA, 2.5 eV": ({"shape_from": True, "attachment_model": "ea", "attachment_ea_half": "2.5"}, ORANGE, "-", "P")}
#: the W08 comparison isolates f: s = 1 on cations throughout (lab_cold is 0.88-0.98 at N_C 32-40, 1 from N_C 50)
W08_MODELS = {r"$f = 1$": ({"capture_model": "bt94", "attachment_model": "constant"}, "k", "-"),
              r"$f = 0.003$": ({"capture_model": "bt94", "attachment_model": "constant", "attachment_fraction": 0.003}, "k", ":"),
              r"size logistic, $N_{1/2} = 50$": ({"capture_model": "bt94", "attachment_model": "size", "attachment_nc_half": 50.0}, BLUE, "-")}
W08_CARBON = 6e-7 * 35.0                     # C per H in PAHs: W08's x_PAH at N_C = 35


def _style(ax):
    ax.minorticks_on()
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


# ---------------------------------------------------------------------------------------------------------------- the lab
def lab_figure(lib, path):
    t = pd.read_csv(_paths.get_external_data_path("pahchem/pah_electron_recombination.csv"), comment="#")
    t["s"] = t.k_cm3_s / [chemistry.electron_capture(n, 1, 300.0) for n in t.n_c]
    print("cation recombination at 300 K over BT94 (sticking 1):")
    for r in t.itertuples():
        print(f"  {r.species:13s} N_C {r.n_c:3d} {r.method:8s} {r.s:.3f} (x/ {1 + r.err_frac:.2f})")
    for sub in chemistry.LAB_STICKING_SUBSETS:
        n_s, b, n = chemistry.lab_sticking_fit(sub)
        print(f"  lab fit {sub}: N_s {n_s:.1f}, b {b:.2f} ({n} points); s at N_C 16, 24, 54: "
              + ", ".join(f"{float(chemistry.lab_sticking(x, sub)):.2f}" for x in (16, 24, 54)))
    nc = np.geomspace(6.0, 300.0, 200)
    use_calima_style()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"wspace": 0.25})
    ax.axvspan(16.5, 300.0, color="0.96", zorder=0)
    for a in (ax, bx):
        a.axvspan(*ORION_NC, color="0.86", zorder=0)
    ax.text(0.97, 0.04, "no data", transform=ax.transAxes, ha="right", fontsize=10, color="0.4")
    for lab, (over, c, ls) in CAPTURE_CURVES.items():
        p = Params(**over)
        ax.loglog(nc, [P._capture_factor(SimpleNamespace(nc=x), 1, p, 300.0) for x in nc], color=c, ls=ls, lw=2, label=lab)
    marks = {"FlAPI": ("o", "k", 1.0), "VENDAMS": ("s", "k", 1.04), "FALP-MS": ("o", "none", 0.96)}
    for method, (m, fc, dx) in marks.items():
        q = t[(t.method == method) & (t.n_c >= 10)]
        ax.errorbar(q.n_c * dx, q.s, yerr=[q.s - q.s / (1 + q.err_frac), q.s * q.err_frac], fmt=m, mfc=fc, mec="k", ecolor="k",
                    ms=7, capsize=2, lw=1, label=f"{method} (photo-ions)" if method != "FALP-MS" else "FALP-MS (charge transfer)")
    q = t[t.n_c < 10]
    ax.errorbar(q.n_c, q.s, yerr=[q.s - q.s / (1 + q.err_frac), q.s * q.err_frac], fmt="x", color="0.5", ms=7, capsize=2, lw=1,
                label="benzene (dissociative; not fitted)")
    ax.set(xlim=(5.0, 300.0), ylim=(0.02, 2.0), xlabel=r"$N_{\rm C}$",
           ylabel=r"$k_{\rm rec}\,/\,k_{\rm BT94}(s = 1)$, 300 K")
    ax.set_title("(a) electron sticking on PAH cations", fontsize=12)
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)

    print("attachment fraction f at the anchors:")
    for name, (n, par) in CA.CARELLI.items():
        f = [CA.k_tni(par, x) / chemistry.electron_capture(n, 0, x) for x in CA.T_GRID]
        print(f"  {name:18s} Carelli+13 TNI bound, 10-1000 K: {min(f):.4f}-{max(f):.4f}")
        bx.errorbar([n], [max(f)], yerr=[[max(f) - min(f)], [0.0]], fmt="v", color="k", ms=8, capsize=3, lw=1.2,
                    label=r"Carelli+13 TNI rate / capture, 10--1000 K (upper bounds)" if n == 14 else None)
    fa = [k / chemistry.electron_capture(14, 0, x) for x, k in ANTHRACENE_LAB.items()]
    print(f"  anthracene lab (48-300 K, buffer-stabilised): {min(fa):.4f}-{max(fa):.4f}")
    bx.errorbar([14 * 1.1],[np.sqrt(min(fa) * max(fa))], yerr=[[np.sqrt(min(fa) * max(fa)) - min(fa)], [max(fa) - np.sqrt(min(fa) * max(fa))]],
                fmt="s", color="k", mfc="none", ms=7, capsize=3, lw=1.2, label="anthracene, flow tubes 48--300 K")
    for x, k in C60_LAB.items():
        f = k / chemistry.electron_capture(60, 0, x)
        print(f"  C60 Viggiano+10 at {x} K: {f:.2f} (x/ {C60_FACTOR:g})")
        bx.errorbar([60 * (0.97 if x == 400 else 1.03)], [f], yerr=[[f - f / C60_FACTOR], [f * (C60_FACTOR - 1)]], fmt="o" if x == 400 else "s",
                    color="k", ms=7, capsize=3, lw=1.2, label=rf"C$_{{60}}$, Viggiano+10 ({x} K, factor {C60_FACTOR:g})")
    for lab, (over, c, ls) in ATTACH_CURVES.items():
        p = Params(**over)
        bx.loglog(nc, [P.attachment_fraction(SimpleNamespace(nc=x), p) for x in nc], color=c, ls=ls, lw=2, label=lab)
        print(f"  {lab}: f at N_C 14, 24, 60, 100 = "
              + ", ".join(f"{P.attachment_fraction(SimpleNamespace(nc=x), p):.4f}" for x in (14, 24, 60, 100)))
    pe = Params(**EA_MODEL[0])
    sel = [m for m in lib if -1 in m.charges]
    fe = np.array([P.attachment_fraction(m, pe) for m in sel])
    ne = np.array([m.nc for m in sel])
    bx.scatter(ne, fe, s=6, color=EA_MODEL[1], alpha=0.6, lw=0, label=r"EA logistic, 2.5 eV (library molecules)")
    mid = (ne >= 40) & (ne <= 120)
    print(f"  EA logistic 2.5 eV: f of the {mid.sum()} library anions with N_C 40-120: median {np.median(fe[mid]):.3f}, "
          f"16-84 % {np.percentile(fe[mid], 16):.3f}-{np.percentile(fe[mid], 84):.3f}")
    bx.set(xlim=(5.0, 300.0), ylim=(5e-4, 4.0), xlabel=r"$N_{\rm C}$", ylabel=r"attachment fraction $f$")
    bx.set_title("(b) electron attachment on neutral PAHs", fontsize=12)
    bx.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
    for a in (ax, bx):
        a.text(np.sqrt(ORION_NC[0] * ORION_NC[1]), a.get_ylim()[0] * 1.15, r"Orion Bar $\langle N_{\rm C}\rangle$", ha="center",
               fontsize=8, color="0.35")
        _style(a)
    _save(fig, path)


# ------------------------------------------------------------------------------------------------------------- PDRs4All
def cached_fits(results_dir):
    """[(args as str, {template: (gamma, rms, anion share) over the cells of the better profile})] of the cached shape fits whose
    key is the current code's for their own arguments."""
    out = []
    for f in sorted(results_dir.glob("pahchem_spectral_shape_*.pkl")):
        try:
            with open(f, "rb") as fh:
                r = pickle.load(fh)
        except (ModuleNotFoundError, AttributeError):             # pickled by another Python: not the current code's
            continue
        if F._result_key(argparse.Namespace(**r["args"])) != f.stem.rsplit("_", 1)[1]:
            continue
        cur = {}
        for t in F.C.TEMPLATES:
            prof = min(("lorentzian", "gaussian"), key=lambda p: min(c["sse"] for c in r[t][p]["cells"]))
            c = sorted(r[t][prof]["cells"], key=lambda c: c["gamma"])
            cur[t] = tuple(np.array([x[k] for x in c]) for k in ("gamma", "rms", "e_anion"))
        out.append(({k: str(v) for k, v in r["args"].items()}, f.name, cur))
    return out


def _find(fits, over, base_name):
    want = {**STANDARD, **DEFAULTS, **{k: v for k, v in over.items() if k != "shape_from"}}
    hits = []
    for a, name, cur in fits:
        a = {**DEFAULTS, **a}
        shape = a["shape_from"] != "None"
        if shape != bool(over.get("shape_from")) or (shape and Path(a["shape_from"]).name != base_name):
            continue
        if all(a.get(k) == v for k, v in want.items() if k != "shape_from"):
            hits.append((name, cur))
    if len(hits) != 1:
        raise SystemExit(f"{len(hits)} cached fits for {over}; run fit_pdrs4all_spectra.py with these arguments first")
    return hits[0]


def pdrs4all_figure(path):
    fits = cached_fits(_paths.get_results_dir())
    base_name, _ = _find(fits, {}, None)
    print(f"PDRs4All: {len(fits)} cached fits of the current code; the band shape held is {base_name}'s")
    runs = [(CAPTURE_RUNS, {k: _find(fits, v[0], base_name) for k, v in CAPTURE_RUNS.items()}),
            (ATTACH_RUNS, {k: _find(fits, v[0], base_name) for k, v in ATTACH_RUNS.items()})]
    use_calima_style()
    temps = F.C.TEMPLATES
    fig, axes = plt.subplots(3, len(temps), figsize=(3.4 * len(temps), 9.5), sharex="col", gridspec_kw={"hspace": 0.08, "wspace": 0.3})
    for row, (spec, got) in enumerate(runs):
        for label, (name, cur) in got.items():
            _, c, ls, m = spec[label]
            line = f"  {label:22s} {name}:"
            for j, t in enumerate(temps):
                g, rms, an = cur[t]
                i = int(rms.argmin())
                edge = "low edge" if i == 0 else ("high edge" if i == len(g) - 1 else "")
                line += f" {t} {rms[i]:.3f} (gamma {g[i]:.2g} {edge}, f- {an[i]:.2f});"
                big = label in (r"$s = 1$", r"$f = 1$")
                axes[row, j].semilogx(g, rms, color=c, ls=ls, lw=1.2, marker=m, ms=7 if big else 4, mfc="none" if big else c, mec=c,
                                      label=label)
                if row == 1:
                    axes[2, j].semilogx(g, an, color=c, ls=ls, lw=1.2, marker=m, ms=7 if big else 4, mfc="none" if big else c, mec=c)
            print(line)
    for j, t in enumerate(temps):
        axes[0, j].set_title(t, fontsize=12)
        axes[2, j].set_xlabel(r"$\gamma = G_0\sqrt{T}/n_e$ (card cells)")
        axes[2, j].set_ylim(-0.03, 0.75)
        for a in axes[:, j]:
            _style(a)
    axes[0, 0].set_ylabel("rms, cation capture\n(full fits)")
    axes[1, 0].set_ylabel("rms, attachment\n(band shape of $s = f = 1$)")
    axes[2, 0].set_ylabel("anion share of the\nPAH emission")
    for row in (0, 1):
        axes[row, -1].legend(frameon=False, fontsize=10, loc="center left", bbox_to_anchor=(1.04, 0.5))
    fig.suptitle("PDRs4All Orion Bar templates: fit rms over the conditions card, per sticking model", fontsize=13, y=0.93)
    _save(fig, path)


# --------------------------------------------------------------------------------------------------------- Wolfire+08
def w08_figure(lib, path, workers):
    use_calima_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True, gridspec_kw={"wspace": 0.06})
    w = np.array([W.wolfire08(0.4, n=n)[0] for n in W.N_GRID])
    w1 = np.array([W.wolfire08(1.0, n=n)[0] for n in W.N_GRID])
    for ax, (lo, hi) in zip(axes, ((32, 40), (50, 100))):
        sel = [m for m in lib if lo <= m.nc <= hi and -1 in m.charges]
        nc = np.array([m.nc for m in sel], dtype=float)
        x = W08_CARBON / nc / len(sel)
        ax.loglog(W.N_GRID, w1 / w, color="0.5", ls="--", lw=1.5, label=r"Wolfire+08, $\Phi_{\rm PAH} = 1$")
        ax.axhline(1.0, color="0.5", lw=1)
        for label, (over, c, ls) in W08_MODELS.items():
            pop = P.Population(sel, {"all": (1.0, 1e4)}, params=Params(**over), cache_dir=F.FP.BASIS_CACHE)
            pop.prepare(P.field_shape(W.cell(W.N)), workers=workers)
            a = []
            for n in W.N_GRID:
                r = pop.gas_rates(W.cell(n), x)
                a.append(r["alpha_mn"]["C+"] + r["alpha_ct"]["C+"])
            ratio = np.array(a) / w
            k = int(np.argmin(abs(W.N_GRID - W.N)))
            print(f"  W08, N_C {lo}-{hi} ({len(sel)} molecules), {label}: alpha(C+)/W08(0.4) {ratio.min():.2f}-{ratio.max():.2f} over n "
                  f"{W.N_GRID[0]:g}-{W.N_GRID[-1]:g}; {ratio[k]:.2f} at n = {W.N_GRID[k]:.3g}")
            ax.loglog(W.N_GRID, ratio, color=c, ls=ls, lw=2.2, label=label)
        ax.axvline(W.N, color="0.8", lw=1, zorder=0)
        ax.set_title(rf"$N_{{\rm C}}$ {lo}--{hi}, {len(sel)} molecules, $2.1\times10^{{-5}}$ C per H", fontsize=12)
        ax.set_xlabel(r"$n$ [cm$^{-3}$] ($G_0 = 1.7$, $T = 100$ K, $x_e = 4.5\times10^{-4}$)")
        _style(ax)
    axes[0].set_ylabel(r"$\alpha({\rm C^+})\,/\,$Wolfire+08 ($\Phi_{\rm PAH} = 0.4$)")
    axes[0].legend(frameon=False, fontsize=9)
    _save(fig, path)


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=None, help="plot directory (default results/plots/pahchem)")
    ap.add_argument("--workers", type=int, default=8, help="processes for the population bases of the W08 figure")
    args = ap.parse_args()
    out = args.out or _paths.get_plots_dir("pahchem")
    lib = W.S.library()
    lab_figure(lib, out / "pahchem_sticking_lab.png")
    pdrs4all_figure(out / "pahchem_pdrs4all_sticking.png")
    w08_figure(lib, out / "pahchem_w08_attachment.png", args.workers)


if __name__ == "__main__":
    main()
