"""Figures for the initial-size-distribution study of low-metallicity collapse.

    python diagnostics/collapse/plot_size_distribution.py

Writes under ``collapse/`` in the plots directory:

* ``size_distributions.png``: the initial shapes of ``science_grid_size_distribution.py``
  (20 Msun, all species summed), dM / d log a.
* ``zcr_vs_S0.png``: [Z_cr] against the initial cross-section per dust mass S0,
  static and with accretion, both criteria; from that script's results.
* ``size_evolution_m20.png``: one collapse per a_S (20 Msun, [Z/H] = -5, energy
  equation, accretion): S, D, T and the final size distribution.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pycalima import _paths
from pycalima.collapse.integrator import run_collapse
from pycalima.collapse.tracks import Track
from pycalima.plotting_style import use_calima_style

sys.path.insert(0, str(Path(__file__).resolve().parent))
import science_grid_size_distribution as grid   # noqa: E402

OUT = _paths.get_plots_dir() / "collapse"
RESULTS = _paths.get_results_dir() / "collapse" / "science_grid_size_distribution.txt"
MASSES = (13, 20, 25, 30)
COLOR = dict(zip(MASSES, ("C0", "C1", "C2", "C3")))
A_S = [s for s in grid.SHAPES if s.startswith("aS=")]
SIG = [s for s in grid.SHAPES if s.startswith("sig=")]
LIT = ["grackle", "N03", "N07", "eroded", "MRN"]
MARK = {"aS": "o", "sig": "D", "grackle": "*", "N03": "s", "N07": "^", "eroded": "v", "MRN": "P"}
RUN = dict(n0=1e-1, n_end=1e16, dlnn=0.1, energy=True)


def label(s):
    if s.startswith("aS="):
        return rf"$a_S = {s[3:]}\,\mu$m"
    if s.startswith("sig="):
        return rf"$\sigma = {s[4:]}$"
    return {"grackle": "Grackle (lognormal)", "N03": "Nozawa+03 (unprocessed)", "N07": "Nozawa+07 Fig. 6b",
            "eroded": "N03, uniform erosion", "MRN": "MRN 5--250 nm"}[s]


def mark(s):
    return MARK[s.split("=")[0]] if "=" in s else MARK[s]


def _dm_dloga(pops):
    """(a [um], dM / d log10 a summed over species, normalised to its peak) on the common grid."""
    g = pops[0].grid
    ae = g.a_edges
    a = np.sqrt(ae[1:] * ae[:-1]) * 1e4
    y = sum(p.x for p in pops) / np.log10(ae[1:] / ae[:-1])
    return a, y / y.max()


def fig_shapes():
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, group, title in zip(axs, (A_S, SIG, LIT), (r"lognormal, $\sigma = 0.7$",
                                                       r"lognormal, $a_S = 0.1\,\mu$m", "literature")):
        cols = plt.cm.viridis(np.linspace(0, 0.9, len(group)))
        for c, s in zip(cols, group):
            a, y = _dm_dloga(grid.populations(20, 1e-4, s)[0])
            ax.plot(a, y, color=c, label=label(s))
        ax.set(xscale="log", yscale="log", xlim=(1e-4, 30), ylim=(1e-3, 2), xlabel=r"$a$ [$\mu$m]", title=title)
        ax.legend(fontsize=7, loc="lower center")
    axs[0].set_ylabel(r"$dM/d\log a$ (peak $= 1$)")
    fig.suptitle(r"initial size distributions, 20 M$_\odot$ (all species)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "size_distributions.png", dpi=150)
    plt.close(fig)


def _results():
    out = {}
    for ln in RESULTS.read_text().splitlines():
        if ln.startswith("#"):
            continue
        cr, M, s, S0, S0n, zs, za = ln.split()
        out[(cr, int(M), s)] = (float(S0), float(S0n), float(zs), float(za))
    return out


def fig_zcr(res):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, cr in zip(axs, ("C", "frag")):
        for M in MASSES:
            r = [res[(cr, M, s)] for s in A_S]
            ax.plot([x[0] for x in r], [x[2] for x in r], ":", color=COLOR[M], lw=1)
            ax.plot([x[0] for x in r], [x[3] for x in r], "-", color=COLOR[M], lw=1)
            for s in grid.SHAPES:
                S0, _, zs, za = res[(cr, M, s)]
                ax.plot(S0, min(zs, -2.4), mark(s), mfc="none", mec=COLOR[M], ms=6)
                ax.plot(S0, za, mark(s), color=COLOR[M], ms=6)
        S = np.geomspace(1e3, 1e6, 10)
        ax.plot(S, -4.5 - np.log10(S / 2e4), "k--", lw=0.8, label=r"$Z_{\rm cr} \propto S_0^{-1}$")
        ax.axhspan(-2.5, -2.3, color="0.85", lw=0)
        ax.set(xscale="log", xlim=(7e2, 1.5e6), ylim=(-6.6, -2.3), xlabel=r"initial $S_0$ [cm$^2$ g$^{-1}$]",
               title={"C": r"criterion C ($t_{\rm cool} < t_{\rm col}$, prescribed $T$)",
                      "frag": "criterion frag (energy equation)"}[cr])
    axs[0].set_ylabel(r"[$Z_{\rm cr}$/H] ($Z_\odot = 0.02$)")
    h = [plt.Line2D([], [], color=COLOR[M], lw=2, label=rf"{M} M$_\odot$") for M in MASSES]
    h += [plt.Line2D([], [], color="k", ls=":", marker="o", mfc="none", label="static"),
          plt.Line2D([], [], color="k", ls="-", marker="o", label="accretion")]
    axs[0].legend(handles=h + [axs[0].lines[-1]], fontsize=7, loc="lower left")
    h = [plt.Line2D([], [], ls="none", marker=MARK[k], color="0.3", label=t) for k, t in
         (("aS", r"lognormal $\sigma = 0.7$, $a_S$ scan"), ("sig", r"$\sigma$ scan at $a_S = 0.1\,\mu$m"),
          ("grackle", label("grackle")), ("N03", label("N03")), ("N07", label("N07")),
          ("eroded", label("eroded")), ("MRN", label("MRN")))]
    axs[1].legend(handles=h, fontsize=7, loc="lower left")
    axs[1].text(1.2e6, -2.4, "no frag.", fontsize=7, ha="right", va="center")
    fig.tight_layout()
    fig.savefig(OUT / "zcr_vs_S0.png", dpi=150)
    plt.close(fig)


def fig_evolution(M=20, lz=-5.0):
    tr = Track.grackle(M, 10.0 ** lz)
    fig, axs = plt.subplots(1, 4, figsize=(16, 3.8))
    cols = plt.cm.viridis(np.linspace(0, 0.9, len(A_S)))
    for c, s in zip(cols, A_S):
        pops, y = grid.populations(M, 10.0 ** lz, s)
        D0 = sum(p.x.sum() for p in pops)
        for acc, ls in ((False, ":"), (True, "-")):
            h, end = run_collapse(tr, pops, y_gas=y, accretion=acc, **RUN)
            a = h.arrays()
            axs[0].plot(a["nH"], a["S"], ls, color=c, label=label(s) if acc else None)
            axs[2].plot(a["nH"], a["T"], ls, color=c)
            if acc:
                axs[1].plot(a["nH"], a["D"] / a["D"][0], ls, color=c)
                x, yy = _dm_dloga(end)
                axs[3].plot(x, yy * sum(p.x.sum() for p in end) / D0, ls, color=c)
                x, yy = _dm_dloga(pops)
                axs[3].plot(x, yy, ":", color=c, lw=0.8)
    axs[2].plot(10.0 ** tr.log_n, 10.0 ** tr.log_T, "k", lw=0.8, alpha=0.5, label="dust-free track")
    axs[0].set(xscale="log", yscale="log", xlabel=r"$n_{\rm H}$ [cm$^{-3}$]", ylabel=r"$S$ [cm$^2$ g$^{-1}$]")
    axs[1].set(xscale="log", xlabel=r"$n_{\rm H}$ [cm$^{-3}$]", ylabel=r"$D / D_0$")
    axs[2].set(xscale="log", yscale="log", xlabel=r"$n_{\rm H}$ [cm$^{-3}$]", ylabel=r"$T$ [K]", ylim=(20, 5e3))
    axs[3].set(xscale="log", yscale="log", xlim=(1e-4, 30), ylim=(1e-3, 30), xlabel=r"$a$ [$\mu$m]",
               ylabel=r"$dM/d\log a$ (initial peak $= 1$)")
    axs[0].legend(fontsize=7)
    axs[2].legend(fontsize=7)
    axs[3].text(0.03, 0.95, r"dotted: initial / static; solid: accretion, at $10^{16}$ cm$^{-3}$",
                transform=axs[3].transAxes, fontsize=7, va="top")
    fig.suptitle(rf"{M} M$_\odot$, $n_{{\rm amb}} = 1$, [Z/H] $= {lz:g}$, Grackle species and gas", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "size_evolution_m20.png", dpi=150)
    plt.close(fig)


def main():
    use_calima_style()
    OUT.mkdir(parents=True, exist_ok=True)
    fig_shapes()
    fig_zcr(_results())
    fig_evolution()
    print("wrote", ", ".join(str(OUT / f) for f in ("size_distributions.png", "zcr_vs_S0.png",
                                                    "size_evolution_m20.png")))


if __name__ == "__main__":
    main()
