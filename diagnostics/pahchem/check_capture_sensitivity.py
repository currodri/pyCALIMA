#!/usr/bin/env python3
"""How much the electron-capture efficiencies move the gas coupling of the RAMSES group tables (step 0 of the
capture-efficiency study, 2026-10-26).

Two unknowns, each scanned over its literature range with everything else at the defaults:
- f, ``Params.attachment_fraction``: the share of classical captures on a neutral that form the anion (scales
  autodetachment too, by detailed balance). 1 (pahchem's default, the classical limit), 0.1, 0.01, 0.002
  (anthracene's flow-tube value; Carelli+13 give ~0.003 for coronene).
- eta, the efficiency of e + PAH^Z (Z >= 1) on pahchem's DS87 capture: 1 (default), 0.46 (Tielens 2021 eq. 8.106),
  0.2 (WD01's sticking for N_C ~ 50). A diagnostic scaling of the tabulated alpha and L of the cations, labelled as
  such; not a model option.

On the V4 grid of ``check_wd01b_alpha.py`` (the v4.00 library with the calibrated IP shifts in four N_C bins at
the WD01 carbon budget, the Mathis field at G0 = 1 as UV background, psi = G0 sqrt(T) / n_e from 1e2 to 1e6 by
n_e, T = 100, 1e3, 6e3 K, H+ the only ion at n_e), per H: the photoelectric heating Gamma, the recombination
cooling Lambda, alpha(C+) and <Z> (N_C-weighted over the bins). Printed as ratios to f = 1, eta = 1. Each bin
keeps every ``--stride``-th of its molecules in N_C order (3: ~240 of 712), enough for ratios; f rebuilds the photon
kernels (the anion's delayed detachment), ~8 s per molecule on one core.

    check_capture_sensitivity.py [--workers 4] [--stride 3] [--plot-only] [--out DIR]
"""
import argparse
import json
import sys
from itertools import pairwise
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "dust_charge"))
import check_grain_size_distribution as gsd
import check_wd01b_alpha as V4

from pycalima import _paths
from pycalima.pahchem import group_tables as G
from pycalima.pahchem import population
from pycalima.pahchem.baseline import Params
from pycalima.plotting_style import use_calima_style

F_GRID = (1.0, 0.1, 0.01, 0.002)
ETA_GRID = (1.0, 0.46, 0.2)
RESULTS = _paths.get_results_dir() / "pahchem_capture_sensitivity.json"
OUTPUTS = ("gamma", "lambda", "alpha_c", "z")
LABELS = {"gamma": r"$\Gamma_{\rm PE}$", "lambda": r"$\Lambda_{\rm rec}$", "alpha_c": r"$\alpha({\rm C^+})$",
          "z": r"$\langle Z\rangle$"}


def scaled(tab, eta):
    """A copy of the bin table with the cations' capture (alpha, L of Z = +1, +2) times eta."""
    t = dict(tab.t)
    for k in ("alpha", "L"):
        t[k] = t[k].copy()
        t[k][1:] *= eta                                      # rows: capture by Z = 0, +1, +2
    return G.PAHGroupTable(t)


def thinned(lib, w, stride):
    """w with every ``stride``-th molecule of the bin kept (in N_C order), the rest zero."""
    idx = sorted(np.flatnonzero(w), key=lambda i: lib[i].nc)
    out = np.zeros_like(w)
    out[idx[::stride]] = w[idx[::stride]]
    return out


def compute(workers, stride):
    lib = population.shift_thresholds(population.library_v400(), V4.IP_SHIFTS)
    a_fine = np.geomspace(gsd.A_MIN, V4.radius_cm(V4.NC_EDGES[-1]), 400)
    out = {"f": F_GRID, "eta": ETA_GRID, "stride": stride, "T": V4.T_GRID, "psi": V4.PSI_GRID.tolist(), "runs": {}}
    shape = (len(V4.T_GRID), V4.PSI_GRID.size)
    for f in F_GRID:
        acc = {(f, eta): {k: np.zeros(shape) for k in OUTPUTS} for eta in ETA_GRID}
        n_tot = 0.0
        for lo, hi in pairwise(V4.NC_EDGES):
            w = thinned(lib, G.nc_range_weights(lib, lo, hi), stride)
            a = a_fine[(a_fine >= V4.radius_cm(lo)) & (a_fine <= V4.radius_cm(hi))]
            n_bin = float(np.trapezoid([gsd.graphite_dn_da(x) for x in a], a))
            tab0 = G.PAHGroupTable(G.build_bin_table(lib, w, params=Params(attachment_fraction=f), workers=workers,
                                                     label=f"{lo}-{hi}"))
            print(f"f = {f:g}: bin N_C {lo}-{hi} built ({int((w > 0).sum())} molecules)", flush=True)
            n_tot += n_bin * tab0.t["nc"]
            for eta in ETA_GRID:
                tab = scaled(tab0, eta)
                r_acc = acc[(f, eta)]
                for iT, T in enumerate(V4.T_GRID):
                    for ip, psi in enumerate(V4.PSI_GRID):
                        ne = np.sqrt(T) / psi
                        r = tab.solve(np.zeros(tab.t["L0"].size), T, ne, 0.0, (("H+", ne),), G0_bg=1.0)
                        r_acc["gamma"][iT, ip] += n_bin * r["gamma"]
                        r_acc["lambda"][iT, ip] += n_bin * r["lam"]
                        r_acc["alpha_c"][iT, ip] += n_bin * (r["alpha_mn"]["C+"] + r["alpha_ct"]["C+"])
                        r_acc["z"][iT, ip] += n_bin * tab0.t["nc"] * (r["f"] @ np.array(G.CHARGES, dtype=float))
        for (f_, eta), r_acc in acc.items():
            r_acc["z"] /= n_tot
            out["runs"][f"{f_:g},{eta:g}"] = {k: v.tolist() for k, v in r_acc.items()}
    return out


def report(d):
    ref = {k: np.array(v) for k, v in d["runs"]["1,1"].items()}
    psi = np.array(d["psi"])
    cols = [0, 2, 4, 6, 8]                                   # psi = 1e2, 1e3, 1e4, 1e5, 1e6
    for iT, T in enumerate(d["T"]):
        print(f"\nT = {T:g} K: ratio to f = 1, eta = 1 (<Z>: difference) at psi = "
              + ", ".join(f"{psi[c]:.0e}" for c in cols))
        print(f"{'f':>6} {'eta':>5} | " + " | ".join(f"{o:^34}" for o in OUTPUTS))
        for key, run in d["runs"].items():
            f, eta = key.split(",")
            cells = []
            for o in OUTPUTS:
                v = np.array(run[o])[iT, cols]
                r = v - ref[o][iT, cols] if o == "z" else v / ref[o][iT, cols]
                cells.append(" ".join(f"{x:+6.2f}" if o == "z" else f"{x:6.2f}" for x in r))
            print(f"{f:>6} {eta:>5} | " + " | ".join(cells))


def _ticks(ax):
    ax.minorticks_on()
    ax.tick_params(which="both", direction="in", top=True, right=True)


def plot(d, path):
    use_calima_style()
    psi = np.array(d["psi"])
    ref = {k: np.array(v) for k, v in d["runs"]["1,1"].items()}
    colours = dict(zip(d["f"], ("#2a78d6", "#1baf7a", "#eb6834", "#7b4ea3"), strict=True))
    styles = dict(zip(d["eta"], ("-", "--", ":"), strict=True))
    fig, axes = plt.subplots(len(OUTPUTS), len(d["T"]), figsize=(13, 12), sharex=True)
    for key, run in d["runs"].items():
        f, eta = (float(x) for x in key.split(","))
        for io, o in enumerate(OUTPUTS):
            for iT in range(len(d["T"])):
                v = np.array(run[o])[iT]
                y = v - ref[o][iT] if o == "z" else v / ref[o][iT]
                axes[io, iT].plot(psi, y, color=colours[f], ls=styles[eta], lw=1.8)
    for io, o in enumerate(OUTPUTS):
        for iT, T in enumerate(d["T"]):
            ax = axes[io, iT]
            ax.set_xscale("log")
            if o != "z":
                ax.set_yscale("log")
            ax.axhline(0.0 if o == "z" else 1.0, color="0.6", lw=0.8)
            _ticks(ax)
            if io == 0:
                ax.set_title(rf"$T = {T:g}$ K", fontsize=14)
            if iT == 0:
                ax.set_ylabel(LABELS[o] + (r" $-$ default" if o == "z" else r" / default"), fontsize=13)
            if io == len(OUTPUTS) - 1:
                ax.set_xlabel(r"$\psi = G\sqrt{T}/n_e$", fontsize=13)
    handles = [plt.Line2D([], [], color=c, lw=2, label=rf"$f = {f:g}$") for f, c in colours.items()]
    handles += [plt.Line2D([], [], color="k", ls=s, lw=1.5, label=rf"$\eta = {e:g}$") for e, s in styles.items()]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=11, frameon=False)
    fig.suptitle(r"Electron-capture efficiencies against pahchem's default ($f = 1$, $\eta = 1$), group-table bins",
                 fontsize=14)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--out", type=Path, default=None, help="plot directory (default results/plots/pahchem)")
    args = ap.parse_args()
    if args.plot_only:
        d = json.loads(RESULTS.read_text())
    else:
        d = compute(args.workers, args.stride)
        RESULTS.write_text(json.dumps(d))
    report(d)
    out = args.out or _paths.get_plots_dir("pahchem")
    plot(d, Path(out) / "pahchem_capture_sensitivity.png")


if __name__ == "__main__":
    main()
