"""Figure for the cosmic-ray study of low-metallicity collapse.

    python diagnostics/collapse/plot_cosmic_rays.py

Writes ``collapse/cosmic_rays.png`` under the plots directory, from
``grackle_cosmic_rays.py``'s tracks and results: for 20 Msun at [Z/H] = -5, the
dust-free T and f_H2 with cosmic rays and the attenuated zeta; the Mg+ / Fe+
focusing bound against where accretion grows the dust; and [Z_cr] against zeta0
for Grackle and for our criteria.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pycalima import _paths
from pycalima.collapse.ics import nozawa_ccsn
from pycalima.collapse.integrator import run_collapse
from pycalima.collapse.tracks import Track
from pycalima.plotting_style import use_calima_style

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grackle_cosmic_rays as cr   # noqa: E402

OUT = _paths.get_plots_dir() / "collapse"
ZETAS = (0.0, 1e-19, 1e-18, 1e-17, 1e-16)
MASSES = (13, 20, 25, 30)
COLOR = dict(zip(MASSES, ("C0", "C1", "C2", "C3")))
X0 = 3e-20          # where zeta0 = 0 is drawn


def _track(M, Z, z0):
    return Track.grackle(M, Z, path=cr.track_path(M, z0) if z0 > 0 else None)


def _results():
    out = {}
    for ln in (cr.OUT / "grackle_cosmic_rays.txt").read_text().splitlines():
        if ln.startswith("#"):
            continue
        v = ln.split()
        out[(int(v[0]), float(v[1]))] = [float(x) for x in v[2:]]
    return out


def main(M=20, lz=-5.0):
    use_calima_style()
    OUT.mkdir(parents=True, exist_ok=True)
    Z = 10.0 ** lz
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))
    cols = plt.cm.plasma(np.linspace(0, 0.85, len(ZETAS)))
    pops, y, _ = nozawa_ccsn(M, Z, n_amb=1, grains="grackle")
    n = np.logspace(-1, 16, 400)
    for c, z0 in zip(cols, ZETAS):
        tr = _track(M, Z, z0)
        lab = r"no CR" if z0 == 0 else rf"$\zeta_0 = 10^{{{np.log10(z0):.0f}}}$ s$^{{-1}}$"
        ax[0, 0].loglog(n, tr.T(n), color=c, label=lab)
        ax[0, 1].loglog(10.0 ** tr.log_n, 10.0 ** tr.log_fH2, color=c, label=lab)
        if z0 > 0:
            zeta = np.array([cr.zeta_along(tr, x, z0) for x in n])
            ax[0, 1].plot(n, zeta / z0 * 1e-4, color=c, ls=":", lw=1)
            f = cr.focus_function(tr, pops, y, z0)
            ax[0, 2].semilogx(n, [f(x, 0.0)["Mg"] for x in n], color=c, label=lab)
        h = run_collapse(tr, pops, y_gas=y, n0=1e-1, n_end=1e16, dlnn=0.1, accretion=True, energy=True)[0].arrays()
        ax[0, 2].semilogx(h["nH"], 1.0 + 3.5 * (h["D"] - h["D"][0]) / (h["D"][-1] - h["D"][0]), color=c, ls="--",
                          lw=1)
    ax[0, 0].set(xlabel=r"$n_{\rm H}$ [cm$^{-3}$]", ylabel=r"$T$ [K] (dust-free, Grackle)", ylim=(20, 5e3),
                 title=rf"{M} M$_\odot$, [Z/H] $= {lz:g}$: HD cooling with cosmic rays")
    ax[0, 0].legend(fontsize=7)
    ax[0, 1].set(xlabel=r"$n_{\rm H}$ [cm$^{-3}$]", ylabel=r"$f_{\rm H_2}$; dotted: $10^{-4}\,\zeta/\zeta_0$",
                 ylim=(1e-5, 2), title=r"attenuation: $e^{-\Sigma/96\,{\rm g\,cm^{-2}}}$, $\Sigma = \rho\lambda_J/2$")
    ax[0, 2].set(xlabel=r"$n_{\rm H}$ [cm$^{-3}$]", ylabel=r"Mg$^+$/Fe$^+$ focus bound (solid)", ylim=(0.8, 5),
                 title="focusing ends before accretion starts")
    ax[0, 2].text(0.03, 0.05, "dashed: dust growth by accretion,\n" r"$1 + 3.5\,(D - D_0)/(D_{\rm end} - D_0)$",
                  transform=ax[0, 2].transAxes, fontsize=7)
    ax[0, 2].legend(fontsize=7, loc="upper left")

    res = _results()
    x = lambda z0: X0 if z0 == 0 else z0
    # columns: Grackle, C static, C acc, C acc+focus, frag static, frag acc, frag acc+focus
    for M_ in MASSES:
        zs = [z for z in ZETAS if (M_, z) in res]
        xs = [x(z) for z in zs]
        r = np.array([res[(M_, z)] for z in zs])
        r[~np.isfinite(r)] = -2.4
        ax[1, 0].plot(xs, r[:, 0], "o-", color=COLOR[M_], label=rf"{M_} M$_\odot$")
        for a_, off in ((ax[1, 1], 1), (ax[1, 2], 4)):
            a_.plot(xs, r[:, off], "o:", mfc="none", color=COLOR[M_])
            a_.plot(xs, r[:, off + 1], "s-", color=COLOR[M_])
            a_.plot(xs, r[:, off + 2], "^--", color=COLOR[M_], ms=4)
    for a_, t in zip(ax[1], ("Grackle (full chemistry, dust)", "ours, criterion C", "ours, criterion frag")):
        a_.set(xscale="log", xlabel=r"$\zeta_0$ [s$^{-1}$]", title=t, ylim=(-5.8, -2.3))
        a_.set_xticks([X0, 1e-19, 1e-18, 1e-17, 1e-16], ["0", r"$10^{-19}$", r"$10^{-18}$", r"$10^{-17}$",
                                                         r"$10^{-16}$"])
        a_.axhspan(-2.5, -2.3, color="0.85", lw=0)
    ax[1, 0].set_ylabel(r"[$Z_{\rm cr}$/H]")
    ax[1, 0].legend(fontsize=7)
    hd = [plt.Line2D([], [], color="k", ls=":", marker="o", mfc="none", label="static"),
          plt.Line2D([], [], color="k", ls="-", marker="s", label="accretion"),
          plt.Line2D([], [], color="k", ls="--", marker="^", ms=4, label=r"accretion + focus bound")]
    ax[1, 1].legend(handles=hd, fontsize=7)
    ax[1, 2].text(0.97, 0.97, "grey band: no fragmentation", transform=ax[1, 2].transAxes, fontsize=7, ha="right",
                  va="top")
    fig.tight_layout()
    fig.savefig(OUT / "cosmic_rays.png", dpi=150)
    plt.close(fig)
    print("wrote", OUT / "cosmic_rays.png")


if __name__ == "__main__":
    main()
