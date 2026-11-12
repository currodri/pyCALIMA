#!/usr/bin/env python3
"""PAH charge across ISM phases: pahchem against WD01 physics and the published balances.

Each phase is a cell some paper states (nothing here is predicted, ``conditions-are-inputs``):
- WD01 (ApJS 134, 263, Sect. 4) CNM and WNM: ISRF (G = 1.13), T = 100 and 6000 K, n_e = 0.03;
- Wolfire+08 CNM: G0 = 1.7, n = 30, T = 100 K, x_e = 4.5e-4, C+ and H+ (``check_wolfire08_cplus``);
- Tielens (2021) Fig. 12.34: diffuse cloud, NGC 7023 PDR and cavity, x_e = 1.6e-4 (``check_tielens21_fz``);
- the NGC 7023 NW edge (Kurucz 15 000 K) and the Orion Bar atomic PDR (Kurucz 40 000 K) of ``baseline``.

pahchem: the v4.00 molecules with every charge -1..+2 (712, N_C 22-268; calibrated IP shifts), the default
detachment, the Draine shape (with the solar-circle old stars) where the paper uses an ISRF, no associative
detachment except where the paper's balance has the gas (W08). WD01 physics: pyCALIMA's full-spectrum graphite
charging with WD01 yields (as V4), Mathis-shaped field at the same G (also for the Kurucz phases, where it is only
indicative), for the grain of the same carbon count, N_C = 468 (a / 10 A)^3.
References drawn: Tielens' <Z> for C54H18 and C96H24, and W08's three-state <Z> (N_C = 35, Phi = 0.4 and 1).

Figures: ``pahchem_charge_phases.png`` (<Z> against N_C per phase) and ``pahchem_charge_phases_fz.png`` (f(Z) by
N_C range and phase). Results in results/pahchem_charge_phases.json (``--plot-only`` redraws from it).

    check_charge_phases.py [--workers 4] [--plot-only]
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import pairwise
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_tielens21_fz as T21
import check_wd01b_alpha as V4
import check_wolfire08_cplus as W08

from pycalima import _paths
from pycalima.pahchem import group_tables as G
from pycalima.pahchem import population
from pycalima.pahchem.baseline import NGC7023_EDGE, ORION_BAR_ATOMIC, Environment, Params
from pycalima.plotting_style import use_calima_style

IP_SHIFTS = population.CALIBRATED_SHIFTS
#: label -> (Environment, source); n_h is the atomic H of the paper's balance (0 where it has none)
PHASES = {
    "WD01 CNM": (Environment("WD01 CNM", 1.13, 0.0, 0.03, 100.0, None), "WD01 Sect. 4"),
    "WD01 WNM": (Environment("WD01 WNM", 1.13, 0.0, 0.03, 6000.0, None), "WD01 Sect. 4"),
    "W08 CNM": (W08.cell(W08.N), "Wolfire+08"),
    "diffuse cloud": (Environment("diffuse", 2.0, 0.0, T21.X_E * 50.0, 80.0, None), "Tielens (2021)"),
    "NGC 7023 PDR": (Environment("NGC 7023 PDR", 2600.0, 0.0, T21.X_E * 2e4, 500.0, None), "Tielens (2021)"),
    "NGC 7023 cavity": (Environment("NGC 7023 cavity", 2e4, 0.0, T21.X_E * 5e3, 500.0, None), "Tielens (2021)"),
    "NGC 7023 NW edge": (NGC7023_EDGE, "baseline (Kurucz 15 kK)"),
    "Orion Bar atomic": (ORION_BAR_ATOMIC, "baseline (Kurucz 40 kK)"),
}
TIELENS_NAME = {"diffuse cloud": "diffuse", "NGC 7023 PDR": "NGC 7023 PDR", "NGC 7023 cavity": "NGC 7023 cavity"}
NC_EDGES = (20, 40, 70, 120, 270)
NC_WD01 = np.geomspace(22.0, 268.0, 12)
RESULTS = _paths.get_results_dir() / "pahchem_charge_phases.json"


def gamma(env):
    return env.g0 * np.sqrt(env.temperature_k) / env.n_e


def wd01_mean_charge(env, workers):
    """<Z> of WD01 graphite grains of N_C = NC_WD01 at the phase's G, n_e, T (Mathis shape, H+ at n_e)."""
    g = 1.0 / V4.mathis_G()
    jobs = [(env.g0 * g, env.n_e, env.temperature_k, "graphite", float(V4.radius_cm(nc)), "WD01") for nc in NC_WD01]
    with ProcessPoolExecutor(workers) as ex:
        return [float(r[-1]) for r in ex.map(V4._alpha_full, jobs)]


def compute(workers):
    lib = [m for m in population.shift_thresholds(population.library_v400(), IP_SHIFTS) if G.complete(m)
           and NC_EDGES[0] <= m.nc < NC_EDGES[-1]]
    pop = population.Population(lib, {"all": (1.0, 1e4)}, cache_dir=T21.BASIS_CACHE, params=Params())
    zs = np.array(pop.zs, dtype=float)
    out = {"nc": [m.nc for m in lib], "zs": pop.zs, "detachment": Params().detachment, "phases": {}}
    for label, (env, source) in PHASES.items():
        print(f"{label}: G0 {env.g0:g}, n_e {env.n_e:.3g}, T {env.temperature_k:g} K, gamma {gamma(env):.2e}",
              flush=True)
        pop.prepare(population.field_shape(env), workers=workers)
        f = pop.charge_matrix(env)
        out["phases"][label] = {"f": f.tolist(), "mean_z": (f @ zs).tolist(), "gamma": gamma(env), "source": source,
                                "wd01_z": wd01_mean_charge(env, workers)}
    return out


def references(label):
    """[(N_C, <Z>, marker label)] of the published balances for the phase."""
    refs = []
    if label in TIELENS_NAME:
        for (mol, env), f in T21.FIG_12_34.items():
            if env == TIELENS_NAME[label]:
                refs.append((int(mol[1:3]), float(np.dot(f, T21.ZS)), "Tielens (2021) Fig. 12.34"))
    if label == "W08 CNM":
        for phi in (0.4, 1.0):
            fm, _, fp = W08.wolfire08(phi)[1]
            refs.append((35, fp - fm, rf"Wolfire+08, $\Phi$ = {phi:g}"))
    return refs


def report(d):
    nc = np.array(d["nc"])
    for label, r in d["phases"].items():
        z = np.array(r["mean_z"])
        print(f"\n{label} ({r['source']}), gamma {r['gamma']:.2e}")
        for lo, hi in pairwise(NC_EDGES):
            m = (nc >= lo) & (nc < hi)
            fz = np.array(r["f"])[m].mean(axis=0)
            wd = np.interp(np.log(nc[m].mean()), np.log(NC_WD01), r["wd01_z"])
            print(f"  N_C {lo:3d}-{hi:3d} ({m.sum():3d}): <Z> {z[m].mean():+.2f} (16-84 % {np.percentile(z[m], 16):+.2f} "
                  f"to {np.percentile(z[m], 84):+.2f}); WD01 physics {wd:+.2f}; f(Z) {np.round(fz, 3)}")
        for n, zr, lab in references(label):
            print(f"  {lab}: N_C {n}, <Z> {zr:+.2f}")


def _ticks(ax):
    ax.minorticks_on()
    ax.tick_params(which="both", direction="in", top=True, right=True)


def plot_mean(d, path):
    use_calima_style()
    nc = np.array(d["nc"])
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.5), sharex=True, gridspec_kw={"hspace": 0.12, "wspace": 0.22})
    centres = np.geomspace(22, 260, 14)
    markers = {"Tielens (2021) Fig. 12.34": ("s", "k"), r"Wolfire+08, $\Phi$ = 0.4": ("D", "k"),
               r"Wolfire+08, $\Phi$ = 1": ("D", "0.55")}
    for ax, (label, r) in zip(axes.flat, d["phases"].items(), strict=True):
        z = np.array(r["mean_z"])
        ax.semilogx(nc, z, ".", color="#2a78d6", ms=3, alpha=0.25)
        edges = np.sqrt(centres[1:] * centres[:-1])
        idx = np.digitize(nc, edges)
        med = [np.median(z[idx == i]) if np.any(idx == i) else np.nan for i in range(centres.size)]
        lo = [np.percentile(z[idx == i], 16) if np.any(idx == i) else np.nan for i in range(centres.size)]
        hi = [np.percentile(z[idx == i], 84) if np.any(idx == i) else np.nan for i in range(centres.size)]
        ax.fill_between(centres, lo, hi, color="#2a78d6", alpha=0.18, lw=0)
        ax.semilogx(centres, med, color="#2a78d6", lw=2.4, label=f"pahchem ({d['detachment']}), median, 16-84 %")
        ax.semilogx(NC_WD01, r["wd01_z"], color="#eb6834", lw=2, ls="--", label="WD01 physics (graphite, Mathis shape)")
        for n, zr, lab in references(label):
            mk, col = markers[lab]
            ax.plot(n, zr, mk, color=col, ms=9, mec="white", mew=1.2, label=lab)
        ax.axhline(0.0, color="0.75", lw=1, zorder=0)
        ax.set_title(rf"{label}: $\gamma$ = {r['gamma']:.1e}", fontsize=12)
        _ticks(ax)
    for ax in axes[1]:
        ax.set_xlabel(r"$N_{\rm C}$", fontsize=14)
    for ax in axes.flat:                    # plain labels on the log N_C axis (the minor 2x10^1 ... labels overlapped)
        ax.set_xticks([30, 50, 100, 200])
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\langle Z \rangle$", fontsize=14)
    handles = {}
    for ax in axes.flat:
        for h, lab in zip(*ax.get_legend_handles_labels(), strict=True):
            handles.setdefault(lab, h)
    fig.legend(list(handles.values()), list(handles), frameon=False, fontsize=11, ncol=5, loc="lower center",
               bbox_to_anchor=(0.5, -0.03))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def plot_fz(d, path):
    use_calima_style()
    nc, zs = np.array(d["nc"]), d["zs"]
    classes = (("PAH$^-$", lambda z: z < 0, "#2a78d6"), ("PAH$^0$", lambda z: z == 0, "#eb6834"),
               ("PAH$^+$", lambda z: z == 1, "#1baf7a"), ("PAH$^{2+}$", lambda z: z >= 2, "#eda100"))
    phases = list(d["phases"])
    fig, ax = plt.subplots(figsize=(16, 5.2))
    w, nb = 0.2, len(NC_EDGES) - 1
    for p, label in enumerate(phases):
        f = np.array(d["phases"][label]["f"])
        for b, (lo, hi) in enumerate(pairwise(NC_EDGES)):
            fz = f[(nc >= lo) & (nc < hi)].mean(axis=0)
            x, bottom = p + (b - (nb - 1) / 2) * (w + 0.01), 0.0
            for name, sel, col in classes:
                v = sum(fz[k] for k, z in enumerate(zs) if sel(z))
                ax.bar(x, v, w, bottom=bottom, color=col, edgecolor="white", linewidth=1.5,
                       label=name if (p, b) == (0, 0) else None)
                bottom += v
            if p == 0:
                ax.annotate(f"{lo}-{hi}", (x, 1.01), ha="center", va="bottom", fontsize=8, rotation=90, color="0.35")
    ax.set_xticks(range(len(phases)), [f"{lab}\n" + rf"$\gamma$ = {d['phases'][lab]['gamma']:.0e}" for lab in phases],
                  fontsize=10)
    ax.set_ylabel("f(Z), mean over the N_C range", fontsize=13)
    ax.set_ylim(0, 1.12)
    ax.set_title(r"Charge fractions by phase; four bars per phase: $N_{\rm C}$ 20-40, 40-70, 70-120, 120-270"
                 f" (pahchem, {d['detachment']})", fontsize=12)
    ax.legend(frameon=False, fontsize=11, ncol=4, loc="upper right", bbox_to_anchor=(1.0, 1.0))
    ax.minorticks_on()
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)
    ax.tick_params(which="both", direction="in", right=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--out", type=Path, default=None, help="plot directory (default results/plots/pahchem)")
    args = ap.parse_args()
    if not args.plot_only:
        RESULTS.parent.mkdir(parents=True, exist_ok=True)
        RESULTS.write_text(json.dumps(compute(args.workers)))
    d = json.loads(RESULTS.read_text())
    report(d)
    out = args.out or _paths.get_plots_dir("pahchem")
    out.mkdir(parents=True, exist_ok=True)
    plot_mean(d, out / "pahchem_charge_phases.png")
    plot_fz(d, out / "pahchem_charge_phases_fz.png")


if __name__ == "__main__":
    main()
