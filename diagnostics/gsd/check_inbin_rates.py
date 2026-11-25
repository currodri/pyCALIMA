"""How much do the grain rates of a PL2M bin depend on where its mass sits inside the bin?

The optics tables carry the in-bin state s (``gsd.optics``): s = (r - 1)/(eta - 1), r = M/(N m_L), runs from 0 (all
grains at the lower edge) to 1 (all at the upper edge). Today's RAMSES-CALIMA charging, photoelectric-heating,
recombination-cooling, sputtering and surface tables are built at one radius per bin, the bin centre. This script
measures what that costs, per gram of a bin's dust, for

* photoelectric heating Gamma and recombination cooling Lambda (``gsd.charging``, the RAMSES charge solve), in the
  WNM, CNM, a PDR and dense gas (the cells of ``check_pah_grain_seam``, background Mathis field only);
* processes per unit grain surface (H2 formation, neutral accretion, gas-grain collisional cooling, sputtering mass
  loss), whose rate per gram goes as 3/(rho a).

Four ways to get the bin's rate per gram at state s:

* exact: the mass-weighted mean over the power-law profile, ``optics.profile_weights`` on n_nodes intervals per bin
  (the optics tables' method; it is what an s axis on the tables gives);
* centre: the single-grain value at the bin centre (today);
* mean mass: the single-grain value at the bin's current mean mass M/N (one charge solve; exact if the rate per gram
  is the same at every size);
* area node: one charge solve at a* = <a^3>/<a^2>, weighted to the bin's exact area (the 1-point Gauss rule of the
  measure a^2 dN; exact if the rate per grain over a^2 is linear in a). Per gram it is the single-grain value at a*;
  both moments come from the closure at no cost. Adopted 2026-11-18 for RAMSES's charge-dependent rates;
* median nodes: the two nodes of ``collisions.bin_nodes`` (number halves at their mean masses; exact N and M), the
  collision operator's quadrature;
* Gauss nodes: the 2-point Gauss quadrature of the bin in radius (``closures.nodes(..., rule='gauss')``; exact for
  int a^k dN, k <= 3), the default rate nodes since 2026-11-18.

The single bin's profile is the power law (PL2M); with the default EA closure (and EAF) an isolated bin would be a spike at its
mean mass, which is not what this test is about (how to evaluate a rate over a given in-bin profile).

    python diagnostics/gsd/check_inbin_rates.py --out results/gsd_overview/figs/inbin_rates.pdf

The charge tables are cached (``gsd.charging.bin_table``); a first run builds about 200 of them (~1 s each).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from pycalima.gsd import charging as ch
from pycalima.gsd.carbon import CarbonLayout
from pycalima.gsd import closures as gc
from pycalima.gsd.collisions import bin_nodes
from pycalima.gsd.grid import MassGrid
from pycalima.gsd.materials import get_material
from pycalima.gsd.optics import node_u, profile_weights

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_pah_grain_seam import AMU, CELLS, ION_MASS   # noqa: E402

S = np.linspace(0.0, 1.0, 101)
WAYS = ("centre", "mean", "area", "two", "gauss")
A_MAX_CM = 1e-4
SIL_A_MIN_CM = 4.5e-8                 # the silicate grid of check_optics_tables
N_SIL_BINS = 8


def populations():
    """(label, grid, charging material, bins to test) for carbon grains and silicate."""
    lay = CarbonLayout()
    cg = lay.grid()
    sil = get_material("silicate")
    m = sil.rho * 4.0 / 3.0 * np.pi * np.array([SIL_A_MIN_CM, A_MAX_CM]) ** 3
    sg = MassGrid(np.geomspace(m[0], m[1], N_SIL_BINS + 1), sil.rho)
    return [("carbon (a-C)", cg, lay.material, np.nonzero(~lay.pah_bins(cg))[0]),
            ("silicate", sg, "silicate", np.arange(sg.n_bins))]


def single_grain_rates(grid, material, bins, n_nodes):
    """Radii a [cm] (n_nodes intervals per bin) and per-gram Gamma, Lambda [erg s^-1 g^-1] per cell there."""
    a = np.unique(np.concatenate([grid.a_edges[i] * node_u(grid.etas[i], n_nodes) for i in bins]))
    model = ch.ChargingModel(material, a)
    L0, L1 = model.L0, model.L1
    photons = ch.PhotonGroups(L0, L1, np.zeros(L0.size), np.sqrt(L0 * L1))
    mass = grid.mass(a)
    out = {}
    for cell, (T, n_h, n_e, ion, G0) in CELLS.items():
        st = model.solve(photons, T, n_e, ions=[(n_e, 1.0, ION_MASS[ion] * AMU)], G0_bg=G0)
        lam = np.array([r["Lambda"] + r["Lambda_auto"] for r in st.results])
        out[cell] = dict(Gamma=st.Gamma / mass, Lambda=lam / mass)
    out["surface"] = dict(S=3.0 / (grid.rho * a))
    return a, out


def bin_values(grid, i, a, k, n_nodes):
    """{method: k per gram of bin i at the states S} for single-grain per-gram values k(a)."""
    eta = grid.etas[i]
    u = node_u(eta, n_nodes)
    ku = np.exp(np.interp(np.log(grid.a_edges[i] * u), np.log(a), np.log(k)))
    exact = profile_weights(eta, S, u) @ ku
    mL = grid.m_edges[i]
    N = np.ones(S.size)
    M = mL * (1.0 + S * (eta - 1.0))
    at = lambda m: np.exp(np.interp(np.log(grid.radius(m)), np.log(a), np.log(k)))   # noqa: E731
    centre = np.full(S.size, at(grid.m_centres[i]))
    mean = at(M / N)
    k_at_a = lambda r: np.exp(np.interp(np.log(r), np.log(a), np.log(k)))   # noqa: E731
    area = np.empty(S.size)
    two = np.empty(S.size)
    gauss = np.empty(S.size)
    for n in range(S.size):
        Nb = np.zeros(grid.n_bins)
        Mb = np.zeros(grid.n_bins)
        Nb[i], Mb[i] = N[n], M[n]
        mm, ww = bin_nodes(grid, Nb, Mb, method="pl2m")
        two[n] = np.sum(ww[i] * mm[i] * at(mm[i])) / M[n]
        mg, wg = gc.nodes(grid, Nb, Mb, "pl2m", rule="gauss")[:2]
        gauss[n] = np.sum(wg[i] * mg[i] * at(mg[i])) / M[n]
        P = gc.pieces(grid, Nb, Mb, "pl2m")
        area[n] = k_at_a(gc.radius_moments(P, 3)[i] / gc.radius_moments(P, 2)[i])
    return dict(exact=exact, centre=centre, mean=mean, area=area, two=two, gauss=gauss)


def s_of_slope(eta, q):
    """s of the in-bin profile dN/da ~ a^-q (q = 3.5: MRN)."""
    from pycalima.gsd.pl2m import mean_mass_ratio
    beta = (-q - 2.0) / 3.0
    return float((mean_mass_ratio(np.array([beta]), eta)[0] - 1.0) / (eta - 1.0))


def run(n_nodes):
    rows = []
    curves = {}
    for label, grid, material, bins in populations():
        a, rates = single_grain_rates(grid, material, bins, n_nodes)
        for i in bins:
            for cell, qs in rates.items():
                for q, k in qs.items():
                    v = bin_values(grid, i, a, k, n_nodes)
                    err = {m: np.abs(v[m] / v["exact"] - 1.0) for m in WAYS}
                    s_mrn = s_of_slope(grid.etas[i], 3.5)
                    j = int(np.argmin(np.abs(S - s_mrn)))
                    rows.append(dict(pop=label, bin=int(i), a_lo=grid.a_edges[i], a_hi=grid.a_edges[i + 1],
                                     cell=cell, q=q, s_mrn=s_mrn,
                                     **{f"max_{m}": float(e.max()) for m, e in err.items()},
                                     **{f"mrn_{m}": float(e[j]) for m, e in err.items()},
                                     span=float(v["exact"].max() / v["exact"].min())))
                    curves[(label, int(i), cell, q)] = v
    return rows, curves


def report(rows, path):
    lines = ["In-bin state and grain rates: |method/exact - 1|, worst over s in [0, 1] (and at the MRN slope, "
             "dN/da ~ a^-3.5); span = max/min of the exact per-gram rate over s.",
             f"{'population':14s} {'bin':>3s} {'a [nm]':>13s} {'cell':8s} {'q':7s} {'span':>6s} "
             f"{'centre':>13s} {'mean mass':>13s} {'area node':>13s} {'median nodes':>13s} {'Gauss nodes':>13s}"]
    for r in rows:
        lines.append(f"{r['pop']:14s} {r['bin']:3d} {r['a_lo'] * 1e7:6.2f}-{r['a_hi'] * 1e7:6.1f} {r['cell']:8s} "
                     f"{r['q']:7s} {r['span']:6.2f} "
                     + " ".join(f"{r['max_' + m]:6.3f}/{r['mrn_' + m]:6.3f}" for m in WAYS))
    Path(path).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def plot(rows, curves, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker  # noqa: F401
    from pycalima.plotting_style import use_calima_style

    use_calima_style()
    fig, axs = plt.subplots(2, 2, figsize=(10, 7.6))
    ax = axs[0, 0]
    show = [k for k in curves if k[0].startswith("carbon") and k[2] == "CNM" and k[3] == "Gamma"]
    show = [show[0], show[len(show) // 2], show[-1]]
    for k, c in zip(show, ("C0", "C1", "C2")):
        v = curves[k]
        r = [x for x in rows if (x["pop"], x["bin"], x["cell"], x["q"]) == k][0]
        lab = rf"{r['a_lo'] * 1e7:.3g}--{r['a_hi'] * 1e7:.4g} nm"
        ax.plot(S, v["centre"] / v["exact"], c=c, ls="-", label=lab + ", centre")
        ax.plot(S, v["mean"] / v["exact"], c=c, ls=":")
        ax.plot(S, v["area"] / v["exact"], c=c, ls=(0, (6, 1.5, 1, 1.5, 1, 1.5)), lw=1.8)
        ax.plot(S, v["two"] / v["exact"], c=c, ls="--")
        ax.plot(S, v["gauss"] / v["exact"], c=c, ls="-.")
    ax.axhline(1.0, c="k", lw=0.6)
    ax.set_xlabel(r"in-bin state $s$")
    ax.set_ylabel(r"$\Gamma_{\rm pe}$ per gram / exact")
    ax.set_title(r"(a) a-C, CNM: centre (solid), mean mass (dotted), area node (thick dash-dot-dot), "
                 r"median (dashed), Gauss (dash-dot)", fontsize=7)
    ax.legend(fontsize=7, loc="upper right")
    ax.set_yscale("log")
    ax.set_yticks([0.3, 0.5, 1.0, 2.0, 3.0])
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    marks = {"WNM": "o", "CNM": "s", "PDR": "^", "dense": "v", "surface": "D"}
    styles = {"centre": ("C3", "-"), "mean": ("C0", ":"), "area": ("C1", (0, (6, 1.5, 1, 1.5, 1, 1.5))),
              "two": ("C2", "--"), "gauss": ("C4", "-.")}
    names = {"centre": "bin centre (today)", "mean": "mean mass (1 solve)", "area": "area node (adopted, 1 solve)",
             "two": "median nodes (2 solves)", "gauss": "Gauss nodes (2 solves)"}
    for ax, q, title in ((axs[0, 1], "Gamma", r"(b) photoelectric heating per gram"),
                         (axs[1, 0], "Lambda", r"(c) recombination cooling per gram"),
                         (axs[1, 1], "S", r"(d) surface per gram (H$_2$ formation, accretion, sputtering)")):
        for pop, filled in (("carbon (a-C)", True), ("silicate", False)):
            for cell in marks:
                sel = [x for x in rows if x["pop"] == pop and x["cell"] == cell and x["q"] == q]
                if not sel:
                    continue
                ac = np.sqrt([x["a_lo"] * x["a_hi"] for x in sel]) * 1e7
                for m, (c, ls) in styles.items():
                    y = np.maximum([x["max_" + m] for x in sel], 1e-4)
                    ax.plot(ac, y, c=c, ls=ls, lw=1.6 if m == "area" else 0.8, marker=marks[cell], ms=4,
                            mfc=c if filled else "none", zorder=3 if m == "area" else 2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"bin centre radius [nm]")
        ax.set_ylabel(r"worst $|$method/exact$-1|$ over $s$")
        ax.set_title(title, fontsize=9)
        ax.axhline(0.05, c="0.5", lw=0.6, ls="--")
        ax.set_ylim(1e-3, 3.0)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], c=c, ls=ls, lw=1.6 if m == "area" else 0.8, label=names[m]) for m, (c, ls) in styles.items()]
    h += [Line2D([], [], c="k", ls="none", marker=mk, label=cl) for cl, mk in marks.items()]
    h += [Line2D([], [], c="k", ls="none", marker="o", mfc="none", label="silicate (open)")]
    fig.legend(handles=h, loc="lower center", ncol=6, fontsize=7.5, frameon=False)  # 5 ways + 5 cells + silicate
    for a in axs.flat:
        a.tick_params(which="both", axis="both", direction="in", top=True, right=True, bottom=True, left=True)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out)
    print("wrote", out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-nodes", type=int, default=12, help="intervals per bin of the exact profile average")
    ap.add_argument("--out", default="results/gsd_overview/figs/inbin_rates.pdf")
    args = ap.parse_args(argv)
    rows, curves = run(args.n_nodes)
    report(rows, Path(args.out).with_suffix(".txt"))
    plot(rows, curves, args.out)


if __name__ == "__main__":
    main()
