#!/usr/bin/env python3
"""Errors of the RAMSES grain-charging methods against the full charge distribution.

Methods, on cells sampled from RAMSES dumps (dust_rtgroups_debug_*.dat, e.g. the
2D Stromgren verification restarts):

  rt   charging_model='WDB06rt': rt_group_charging.solve_cell (cold), i.e. the
       per-group solve with its approximations (Gaussian P(Z) averaged at the
       mean, trimmed discrete windows, Gaussian recombination estimate for the
       bins that carry little of it)
  tab  charging_model='WDB06tab': charging_psi_tables.psi_lookup (FUV with the
       Mathis shape scaled to G0, EUV with the reference blackbody shape, n(H+) = n_e)
  old  charging_model='RM2026': the original (log gamma, log T) charge tables
       (mean charge only)

The reference ("full") is the exact P(Z) on every integer charge within
Z* +- (15 sigma + 10) from the same RT-group rates (PCHIP kernels), with the
exact averages of Gamma and Lambda and the exact recombination sum. Errors are
given per bin and for the cell totals per H (Milky Way abundances of the RAMSES
test setups), for Gamma, Lambda (+ autoionisation), alpha of H+, C+, S+, Fe+
(resolved: > 1e-6 of the geometric rate) and <Z> (relative to max(|Z|, 1)).

    python check_charging_methods_errors.py RUN_DIR [RUN_DIR ...] --rtgroups DIR --psitab DIR
        [--original-tables DIR] [--cells 1500] [--workers 8]
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pycalima import _paths
from pycalima.models.dust_charge import charging_psi_tables as cpt
from pycalima.models.dust_charge import rt_group_charging as rg

import compare_ramses_rtgroups as cr

MW = {1: 5.106243e-4, 2: 5.106243e-4, 3: 1.835089e-3, 4: 1.835089e-3}     # dust mass per gas mass
RHO = {"graphite": 2.2, "silicate": 3.3}
M_P, X_H = 1.67262192e-24, 0.73738783
IONS = {"H+": 0, "C+": 2, "S+": 8, "Fe+": 9}
QTY = ("Zmean", "Gamma", "Lambda") + tuple(f"alpha {k}" for k in IONS)
_G = {}


def full_solve(gt, N, c_red, T, ne, ions, G0_bg):
    """Exact P(Z) on the integers around Z*, its moments, Gamma, Lambda + auto and alpha."""
    r = gt.solve(N, c_red, T, ne, ions, G0_bg=G0_bg, recomb=None)
    Zs, sig = r["Zstar"], max(r["Zsigma"], 0.5)
    Z = np.arange(max(gt.t["Zmin"], np.floor(Zs - 15 * sig - 10)), min(gt.t["Zmax"], np.ceil(Zs + 15 * sig + 10)) + 1.0)
    lnf = np.concatenate([[0.0], np.cumsum([gt._g(z) for z in Z[:-1]])])
    P = np.exp(lnf - lnf.max())
    P /= P.sum()
    k = P > 1e-15 * P.max()
    gam, lam, auto = gt._averages(Z[k], P[k] / P[k].sum(), True)
    mean = P @ Z
    return dict(Zmean=mean, Gamma=gam, Lambda=lam + auto, alpha=gt._recombination(Z, P),
                geo=gt._geometric_rates())


def one_cell(args):
    c, rtdir, psidir, origdir = args
    nb = 4
    key = tuple(c["egy"])
    if key not in _G:
        tabs = [rg.read_bin_table(Path(rtdir) / f"dust_charging_rtgroups_DustBin_{b:02d}.dat") for b in range(1, nb + 1)]
        _G[key] = dict(gts=[rg.GroupChargingTable(t, egy=c["egy"], kernels="pchip") for t in tabs],
                       pts=[cpt.read_psi_table(Path(psidir) / f"dust_charging_psitab_DustBin_{b:02d}.dat") for b in range(1, nb + 1)],
                       n=[MW[b] * M_P / X_H / (4 / 3 * np.pi * t["a"] ** 3 * RHO[t["material"]]) for b, t in zip(range(1, nb + 1), tabs)])
    G = _G[key]
    ions = [(c["n_Hp"], 1.0, c["m_H"]), (c["n_Hep"], 1.0, c["m_He"]), (c["n_Hepp"], 2.0, c["m_He"])]
    N, cr_ = (c["Np"], c["c_red"]) if c["path"] != 1 else (np.zeros_like(c["Np"]), 0.0)
    full = [full_solve(gt, N, cr_, c["T"], c["ne"], ions, c["G0_bg"]) for gt in G["gts"]]
    rt = rg.solve_cell(G["gts"], G["n"], N, cr_, c["T"], c["ne"], ions, G0_bg=c["G0_bg"])
    G_F, G_E = cpt.cell_G(c["Np"], c["c_red"], c["egy"], G["pts"][0]["L0"], c["G0_bg"])
    tab = [cpt.psi_lookup(pt, c["T"], c["ne"], G_F, G_E) for pt in G["pts"]]
    old = None
    if origdir:
        old = [cr.original_table_charge(Path(origdir) / f"dust_charge_Z_vs_T_DustBin_{b:02d}", G_F, c["T"], c["ne"])
               for b in range(1, nb + 1)]
    out = dict(T=c["T"], hii=c["n_Hp"] > 0.5 * c["ne"] and c["T"] > 3e3, n=G["n"], bins={})

    def vals(r, f):
        v = dict(Zmean=r["Zmean"], Gamma=r["Gamma"], Lambda=r["Lambda"] + r.get("Lambda_auto", 0.0))
        for k, i in IONS.items():
            v[f"alpha {k}"] = r["alpha_rec"][i] if f["alpha"][i] > rg.RECOMB_RESOLVED * f["geo"][i] else np.nan
        return v

    for b in range(nb):
        f = full[b]
        fv = {"Zmean": f["Zmean"], "Gamma": f["Gamma"], "Lambda": f["Lambda"],
              **{f"alpha {k}": f["alpha"][i] if f["alpha"][i] > rg.RECOMB_RESOLVED * f["geo"][i] else np.nan
                 for k, i in IONS.items()}}
        out["bins"][b + 1] = dict(full=fv, rt=vals(rt[b], f), tab=vals(tab[b], f),
                                  old=None if old is None else dict(Zmean=old[b]))
    return out


def err(a, f, q):
    if a is None or f is None or not np.isfinite(f) or not np.isfinite(a):
        return np.nan
    if q == "Zmean":
        return abs(a - f) / max(abs(f), 1.0)
    return abs(a - f) / abs(f) if f != 0 else np.nan


def summarize(res):
    """Per bin, method, quantity: errors and the bin's share of the cell total; totals per cell."""
    rows = {}
    for c in res:
        for b, d in c["bins"].items():
            for m in ("rt", "tab", "old"):
                if d[m] is None:
                    continue
                for q in QTY:
                    if q not in d[m]:
                        continue
                    tot = np.nansum([c["n"][bb - 1] * abs(dd["full"][q]) for bb, dd in c["bins"].items()]) if q != "Zmean" else np.nan
                    share = c["n"][b - 1] * abs(d["full"][q]) / tot if q != "Zmean" and tot > 0 and np.isfinite(d["full"][q]) else np.nan
                    rows.setdefault((m, b, q), []).append((err(d[m][q], d["full"][q], q), share, c["hii"]))
        for m in ("rt", "tab"):
            for q in QTY[1:]:
                f = np.array([c["n"][b - 1] * d["full"][q] for b, d in c["bins"].items()])
                a = np.array([c["n"][b - 1] * d[m][q] for b, d in c["bins"].items()])
                ok = np.isfinite(f) & np.isfinite(a)
                if ok.any() and np.sum(f[ok]) != 0:
                    rows.setdefault((m, "total", q), []).append((abs(np.sum(a[ok]) / np.sum(f[ok]) - 1), 1.0, c["hii"]))
    return rows


def report(rows):
    lines = []
    for (m, b, q), v in sorted(rows.items(), key=lambda t: (t[0][0], str(t[0][1]), QTY.index(t[0][2]))):
        e = np.array([x[0] for x in v])
        s = np.array([x[1] for x in v])
        ok = np.isfinite(e)
        if not ok.any():
            continue
        imp = ok & (np.nan_to_num(s, nan=1.0) > 0.1)
        line = (f"{m:4s} {str(b):5s} {q:9s} n={ok.sum():5d} | median {np.median(e[ok]):.1e} 90% {np.percentile(e[ok], 90):.1e} "
                f"99% {np.percentile(e[ok], 99):.1e} max {e[ok].max():.1e} | >30%: {np.mean(e[ok] > 0.3):5.1%}")
        if b != "total" and q != "Zmean":
            line += (f" | where the bin has >10% of the total: n={imp.sum():5d}, 99% "
                     f"{np.percentile(e[imp], 99) if imp.any() else np.nan:.1e}, >30% {np.mean(e[imp] > 0.3) if imp.any() else np.nan:5.1%}")
        lines.append(line)
    return lines


def plot(rows, path):
    qs = ("Zmean", "Gamma", "Lambda", "alpha C+")
    fig, axes = plt.subplots(2, len(qs), figsize=(4.2 * len(qs), 7.5), sharey=True)
    cols = {1: "#2a78d6", 2: "#7fb3e8", 3: "#d6542a", 4: "#eba27f", "total": "0.15"}
    for r, m in enumerate(("rt", "tab")):
        for c, q in enumerate(qs):
            ax = axes[r, c]
            for b in (1, 2, 3, 4, "total"):
                v = rows.get((m, b, q))
                if not v:
                    continue
                e = np.sort(np.array([x[0] for x in v if np.isfinite(x[0])]))
                if e.size == 0:
                    continue
                ax.loglog(np.maximum(e, 1e-8), 1 - np.arange(e.size) / e.size, color=cols[b], lw=2.2 if b == "total" else 1.4,
                          label=f"bin {b}" if b != "total" else "cell total")
            ax.axvline(0.3, color="0.6", lw=0.8, ls="--")
            ax.set_xlim(1e-6, 10)
            ax.set_ylim(1e-3, 1.2)
            ax.grid(color="0.92", lw=0.6)
            ax.set_title(f"{'WDB06rt (per group)' if m == 'rt' else 'WDB06tab (tables)'}: {q}", fontsize=10)
            if c == 0:
                ax.set_ylabel("fraction of cells with error > x")
            if r == 1:
                ax.set_xlabel("relative error vs full P(Z)")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Grain charging methods against the full charge distribution (Stromgren cells; line: 30%)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print("Saved", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--rtgroups", required=True)
    ap.add_argument("--psitab", required=True)
    ap.add_argument("--original-tables", default=None)
    ap.add_argument("--cells", type=int, default=1500, help="cells sampled per run")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    rng = np.random.default_rng(1)
    cells = []
    for run in args.runs:
        seen = {}
        for d in sorted(Path(run).glob("dust_rtgroups_debug_*.dat")):
            for r in cr.read_dump(d):
                if r["path"] in (0, 1):
                    seen.setdefault((r["T"], r["ne"], tuple(r["Np"])), r)
        cs = list(seen.values())
        cells += [cs[i] for i in rng.choice(len(cs), min(args.cells, len(cs)), replace=False)]
    print(f"{len(cells)} cells", flush=True)
    with ProcessPoolExecutor(args.workers) as ex:
        res = list(ex.map(one_cell, [(c, args.rtgroups, args.psitab, args.original_tables) for c in cells], chunksize=8))
    rows = summarize(res)
    lines = report(rows)
    print("\n".join(lines))
    out = _paths.get_results_dir() / "charging_methods_errors.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    plot(rows, _paths.get_plots_dir() / "charging_methods_errors.png")


if __name__ == "__main__":
    main()
