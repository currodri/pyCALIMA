"""Do the grains' thermal and chemical rates converge with the number of bins, like the optics do?

A first look at the test the plan needs once every species' chemistry is final. It reads the final (N, M) of the A0
runs (``check_bin_convergence.py``: graphite and silicate, five cases, 2-32 bins per composition, and the
high-resolution reference) for two gsd methods,

* EA, the default since 2026-11-19 (runs in results/gsd_overview/variants/a0_prod/gsd_bin_convergence/), and
* PL2M, kept for comparison (runs in results/binconv_v4/gsd_bin_convergence/),

and evaluates for each run the population's

* photoelectric heating and recombination cooling (``gsd.charging``, the RAMSES charge solve: WDB06 graphite and
  silicate), in the WNM, CNM, a PDR and dense gas (the cells of ``check_pah_grain_seam``, Mathis background only);
* total grain surface, which sets H2 formation, neutral accretion, gas-grain collisional cooling and sputtering.

Each rate is a sum over grains, computed from a run's (N, M) with that run's own closure (``gsd.closures``):

* exact: the integral over the reconstruction (``closures.bin_weights``), what front-aware s tables give;
* median / Gauss nodes: the collision nodes (number halves) or the 2-point Gauss rule in radius (the default rate
  nodes since 2026-11-18);
* centre: every grain of a bin at the bin centre (today's RAMSES tables).

Errors are against the reference run integrated exactly, at the end of each run (the A0 files keep no history), in
fixed cells rather than the cases' own gas.

    python diagnostics/gsd/check_rates_convergence.py --out results/gsd_overview/figs/rates_convergence.pdf
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from pycalima.gsd import charging as ch
from pycalima.gsd import closures as gc
from pycalima.gsd.grid import MassGrid
from pycalima.gsd.optics import node_u

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_pah_grain_seam import AMU, CELLS, ION_MASS   # noqa: E402

RUNS = {"ea": Path("results/gsd_overview/variants/a0_prod/gsd_bin_convergence"),
        "pl2m": Path("results/binconv_v4/gsd_bin_convergence")}
REF_DIR = Path("results/binconv_v4/gsd_bin_convergence")
CASES = ("sn_diffuse", "mrn_accretion", "mrn_coagulation", "mrn_shattering", "mrn_sputtering")
POPS = {"graphite": "graphite", "silicate": "silicate"}     # population -> charging material
N_NODES = 8                                                 # intervals per bin of the exact profile integral
PER_DECADE = 16                                             # radii per decade of the single-grain rates
# (method, way): the curves of the figure
WAYS = (("ea", "exact"), ("ea", "gauss"), ("pl2m", "exact"), ("pl2m", "median"), ("pl2m", "centre"))


def load(path):
    z = np.load(path, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    return meta, {p: (z[f"{p}:m_edges"], z[f"{p}:N"], z[f"{p}:M"]) for p in POPS}


def rho_of(m_edges, a_min):
    return float(m_edges[0] / (4.0 / 3.0 * np.pi * a_min ** 3))


def single_grain(material, a_lo, a_hi):
    """Radii and per-grain rates {cell: {Gamma, Lambda}, surface: {S}} [erg/s, cm^2]."""
    a = np.geomspace(a_lo, a_hi, int(np.ceil(np.log10(a_hi / a_lo) * PER_DECADE)) + 1)
    model = ch.ChargingModel(material, a)
    L0, L1 = model.L0, model.L1
    photons = ch.PhotonGroups(L0, L1, np.zeros(L0.size), np.sqrt(L0 * L1))
    out = {}
    for cell, (T, n_h, n_e, ion, G0) in CELLS.items():
        st = model.solve(photons, T, n_e, ions=[(n_e, 1.0, ION_MASS[ion] * AMU)], G0_bg=G0)
        out[cell] = dict(Gamma=np.maximum(st.Gamma, 1e-300),
                         Lambda=np.maximum([r["Lambda"] + r["Lambda_auto"] for r in st.results], 1e-300))
    out["surface"] = dict(S=4.0 * np.pi * a ** 2)
    return a, out


def totals(grid, N, M, a, g, method):
    """Sum of the per-grain quantity g(a) over the population under ``method``'s closure, every way."""
    at = lambda r: np.exp(np.interp(np.log(r), np.log(a), np.log(g)))   # noqa: E731
    filled = N > 0
    P = gc.pieces(grid, N, M, method)
    u = np.stack([node_u(e, N_NODES) for e in grid.etas])
    W = gc.bin_weights(grid, P, u, "mass")
    r = grid.a_edges[:-1, None] * u
    exact = float(np.sum(M[filled] * np.einsum("ij,ij->i", W, at(r) / grid.mass(r))[filled]))
    centre = float(np.sum(N[filled] * at(grid.a_centres[filled])))
    out = dict(exact=exact, centre=centre)
    for rule in ("median", "gauss"):
        m, w = gc.nodes(grid, N, M, method, rule=rule)[:2]
        out[rule] = float(np.sum(w[filled] * at(grid.radius(m[filled]))))
    return out


def run():
    rates = {}
    res = {}
    sources = [("ref", f) for f in sorted(REF_DIR.glob("*_pl2m_ref_amin3.50A.npz"))]
    for meth, d in RUNS.items():
        sources += [(meth, f) for f in sorted(d.glob(f"*_{meth}_N*_amin3.50A.npz"))]
    if not sources:
        raise SystemExit("no A0 runs found")
    for meth, f in sources:
        meta, pops = load(f)
        case = meta["case"]
        nb = "ref" if meth == "ref" else int(meta["n_bins"])
        for p, (me, N, M) in pops.items():
            grid = MassGrid(me, rho_of(me, meta["a_min_c"] if p == "graphite" else meta["a_min_si"]))
            key = (p, round(grid.rho, 9))
            if key not in rates:
                rates[key] = single_grain(POPS[p], grid.a_edges[0], 5e-4)
            a, qs = rates[key]
            for cell, qd in qs.items():
                for q, g in qd.items():
                    t = totals(grid, N, M, a, g, "pl2m" if meth == "ref" else meth)
                    acc = res.setdefault((case, cell, q), {}).setdefault((meth, nb), {k: 0.0 for k in t})
                    for k, v in t.items():
                        acc[k] += v
    return res


def errors(res):
    """{(case, cell, q): {(method, way): (bins, |run/ref - 1|)}} against the reference integrated exactly."""
    out = {}
    for key, by in res.items():
        if ("ref", "ref") not in by:
            continue
        ref = by[("ref", "ref")]["exact"]
        out[key] = {}
        for meth, way in WAYS:
            ns = sorted(n for (m, n) in by if m == meth)
            if ns:
                out[key][(meth, way)] = (np.array(ns), np.array([abs(by[(meth, n)][way] / ref - 1.0) for n in ns]))
    return out


def worst(err, q, meth, way, case=None):
    """Bins and the worst error over cells (and cases unless ``case``) for rate q."""
    keys = [k for k in err if k[2] == q and (case is None or k[0] == case) and (meth, way) in err[k]]
    if not keys:
        return None, None
    ns = err[keys[0]][(meth, way)][0]
    return ns, np.max([err[k][(meth, way)][1] for k in keys], axis=0)


def report(err, path):
    cols = (2, 3, 4, 6, 8, 12, 16, 24, 32)
    lines = ["Population rates against the high-resolution reference: |run/ref - 1| at the end of each A0 run, worst "
             "over the cells (WNM, CNM, PDR, dense). EA runs (default) and PL2M runs, each evaluated with its own "
             "closure: exact in-bin integral, median or Gauss nodes, bin centre.",
             f"{'case':16s} {'rate':7s} {'method':14s} " + " ".join(f"{n:>6d}" for n in cols)]
    for case in CASES + (None,):
        for q in ("Gamma", "Lambda", "S"):
            for meth, way in WAYS:
                ns, w = worst(err, q, meth, way, case)
                if ns is None:
                    continue
                row = {int(n): x for n, x in zip(ns, w)}
                lines.append(f"{case or 'worst of all':16s} {q:7s} {meth + ' ' + way:14s} "
                             + " ".join(f"{row[n]:6.3f}" if n in row else f"{'-':>6s}" for n in cols))
    Path(path).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def plot(err, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker  # noqa: F401
    from pycalima.plotting_style import use_calima_style

    use_calima_style()
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
    styles = {("ea", "exact"): ("C0", "-", "EA (default), exact in-bin integral"),
              ("ea", "gauss"): ("C0", "--", "EA, Gauss rate nodes (default)"),
              ("pl2m", "exact"): ("C1", "-", "PL2M, exact in-bin integral"),
              ("pl2m", "median"): ("C1", "--", "PL2M, median nodes"),
              ("pl2m", "centre"): ("C3", ":", "PL2M, bin centre (today's tables)")}
    titles = {"Gamma": r"photoelectric heating", "Lambda": r"recombination cooling",
              "S": r"grain surface (H$_2$, accretion, sputtering)"}
    for ax, q in zip(axs, ("Gamma", "Lambda", "S")):
        for key, (c, ls, lab) in styles.items():
            ns, w = worst(err, q, *key)
            if ns is not None:
                ax.plot(ns, np.maximum(w, 1e-4), c=c, ls=ls, marker="o", ms=3, lw=1.1, label=lab)
        ax.set_title(titles[q], fontsize=9)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([2, 3, 4, 6, 8, 12, 16, 24, 32])
        ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%d"))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.axhline(0.05, c="0.5", lw=0.6, ls=":")
        ax.axvline(8, c="0.5", lw=0.6, ls=":")
        ax.set_xlabel("bins per composition")
        ax.tick_params(which="both", axis="both", direction="in", top=True, right=True, bottom=True, left=True)
    axs[0].set_ylabel("worst over cases and cells\n" + r"$|$run/ref$-1|$")
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, fontsize=7.5, frameon=False)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(out)
    print("wrote", out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="results/gsd_overview/figs/rates_convergence.pdf")
    args = ap.parse_args(argv)
    err = errors(run())
    report(err, Path(args.out).with_suffix(".txt"))
    plot(err, args.out)


if __name__ == "__main__":
    main()
