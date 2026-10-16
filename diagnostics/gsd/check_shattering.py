#!/usr/bin/env python3
"""Shattering checks for the grain-size-distribution solver (pycalima.gsd).

1. Collisional cascade. Mass fed into the top of a grid and removed below
   it, constant relative velocity v = 1 km/s and strength Q*_D = 1e9 erg/g,
   KT10 ejecta from the larger grain, projectile shattered, fragments
   dN/da ~ a^-alpha_f up to (f/2)^(1/3) a_t. The outcome depends only on the
   mass ratio, so the steady state is dN/da ~ a^-3.5 whatever alpha_f
   (Dohnanyi 1969; Tanaka, Inaba & Nakazawa 1996), with waves imprinted by
   the lower cut-off (Campo Bagatin et al. 1994).

2. Shattering in the warm ionised medium after Hirashita & Kobayashi (2013,
   HK13): silicate, n_H = 1 cm^-3, dust-to-H mass ratio 4e-3, grains above
   a_c = 0.1 um moving at 20 km/s and the smaller ones at rest (their Model
   H, smoothed over ~10% in a), HY09's four velocity directions, fragments
   down to 3 A and removed below the grid. The initial distribution is MRN
   (HK13 used Nozawa et al. 2007 supernova dust, so the curves are not theirs).
   Both ejecta laws (KT10, JTH96) are shown at 10 Myr, and the KT10 runs
   with (P1, t) = (1e11 dyn cm^-2, 3.3 Myr) and (9e11, 30 Myr): HK13 find that
   equal P1/t gives the same n(a) within a factor 3 (their Fig. 7).

Usage
-----
    python diagnostics/gsd/check_shattering.py
"""
import argparse
from dataclasses import replace

import numpy as np

from pycalima import _paths
from pycalima.gsd.collisions import Channel, collide
from pycalima.gsd.grid import MassGrid
from pycalima.gsd.shattering import SILICATE, GrainCollisions, hy09_directions

MYR = 3.156e13
M_H = 1.6735e-24


class SelfSimilarCascade:
    """Constant v and Q*_D; target = larger grain (see module docstring)."""

    collector = "remove"

    def __init__(self, alpha_f, v=1e5, Q=1e9):
        self.alpha_f, self.v, self.Q = alpha_f, v, Q

    def channels(self, a, m):
        A1, A2 = np.meshgrid(a, a, indexing="ij")
        M1, M2 = np.meshgrid(m, m, indexing="ij")
        big = M1 >= M2
        mt, mp, at = np.where(big, M1, M2), np.where(big, M2, M1), np.where(big, A1, A2)
        K = np.pi * (A1 + A2) ** 2 * self.v
        phi = 0.5 * mt * mp / (mt + mp) * self.v ** 2 / (mt * self.Q)
        mej = phi / (1 + phi) * mt
        ahi = (0.5 * np.minimum((mej + mp) / mt, 1.0)) ** (1 / 3) * at
        return [Channel(K, rem1=mt - mej, frag1=mej + mp, a_hi1=ahi, a_lo1=ahi / 100)]


def cascade(alpha_f, bpd=6, t_end=3e11):
    g = MassGrid.from_radii(1e-6, 1e-2, bpd, 3.0)
    top = g.n_bins - 2
    sN, sM = np.zeros(g.n_bins), np.zeros(g.n_bins)
    sM[top] = 1e-25
    sN[top] = sM[top] / g.m_centres[top]
    N, M = sN * 1e5, sM * 1e5
    model = SelfSimilarCascade(alpha_f)
    t, dt = 0.0, 1e6
    while t < t_end:
        r = collide(g, N, M, model, dt, errmax=1e-2, source=(sN, sM))
        N, M = r["N"], r["M"]
        t += dt
        dt *= 1.1
    a = g.a_centres
    dn = N / np.diff(g.a_edges)
    sel = (a > 1e-5) & (a < 1e-3)
    slope = np.polyfit(np.log(a[sel]), np.log(dn[sel]), 1)[0]
    return a, dn, slope


def mrn(g, D=4e-3, a_lo=5e-7, a_hi=2.5e-5):
    a_e = g.a_edges
    N, M = np.zeros(g.n_bins), np.zeros(g.n_bins)
    for i in range(g.n_bins):
        x1, x2 = max(a_lo, a_e[i]), min(a_hi, a_e[i + 1])
        if x2 > x1:
            N[i] = (x1 ** -2.5 - x2 ** -2.5) / 2.5
            M[i] = 4 / 3 * np.pi * SILICATE.rho * (x2 ** 0.5 - x1 ** 0.5) / 0.5
    s = D * 1.4 * M_H / M.sum()
    return N * s, M * s


def wim(law, P1, t_myr, bpd=10):
    g = MassGrid.from_radii(3e-8, 3e-4, bpd, SILICATE.rho)
    N, M = mrn(g)
    vel = hy09_directions(lambda a: 1e6 * (1 + np.tanh(np.log(np.asarray(a) / 1e-5) / 0.1)))
    model = GrainCollisions(replace(SILICATE, P1=P1), vel, law=law, a_frag_min=3e-8,
                            collector="remove")
    r = collide(g, N, M, model, t_myr * MYR, errmax=1e-3)
    dn0 = N / np.diff(g.a_edges)
    dn = r["N"] / np.diff(g.a_edges)
    return g.a_centres, dn0, dn, r["sink_M"] / M.sum()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    casc = {}
    for af in (2.5, 3.0, 3.5):
        a, dn, slope = cascade(af)
        casc[af] = (a, dn)
        print(f"cascade alpha_f = {af}: dN/da slope over 0.1-10 um = {slope:.3f} (Dohnanyi -3.5)")

    runs = {}
    for key, (law, P1, t) in {"KT10": ("KT10", 3e11, 10.0), "JTH96": ("JTH96", 3e11, 10.0),
                              "KT10 P1/3": ("KT10", 1e11, 10 / 3),
                              "KT10 3P1": ("KT10", 9e11, 30.0)}.items():
        a, dn0, dn, sink = wim(law, P1, t)
        runs[key] = dn
        print(f"WIM {key:10s} (P1 = {P1:.0e}, t = {t:.1f} Myr): mass removed below 3 A = {sink:.1e}")
    small = (a < 3e-6) & (runs["KT10"] > 0)
    for key in ("KT10 P1/3", "KT10 3P1"):
        r = runs[key][small] / runs["KT10"][small]
        print(f"  {key} / KT10 for a < 0.03 um: {r.min():.2f}-{r.max():.2f} (HK13: within 3)")
    r = runs["JTH96"][small] / runs["KT10"][small]
    print(f"  JTH96 / KT10 for a < 0.03 um: {r.min():.2f}-{r.max():.2f} (HK13: <~1.8 for silicate)")

    if args.no_plot:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 4.2))
    for af, color in zip((2.5, 3.0, 3.5), ("#2a78d6", "#1baf7a", "#eb6834")):
        a_c, dn_c = casc[af]
        ax1.plot(a_c * 1e4, dn_c * a_c ** 3.5 / np.median(dn_c * a_c ** 3.5), color=color,
                 lw=1.5, label=rf"$\alpha_f$ = {af}")
    ax1.axhline(1.0, color="#52514e", lw=1, ls="--", label=r"$a^{-3.5}$ (Dohnanyi)")
    ax1.set(xscale="log", yscale="log", xlabel=r"$a$ [$\mu$m]",
            ylabel=r"$a^{3.5}\,dN/da$ (normalised)", ylim=(0.1, 10),
            title="Steady collisional cascade")
    ax1.legend(frameon=False, fontsize=8)
    ax2.plot(a * 1e4, dn0 * a ** 4, color="#c3c2be", lw=1.5, label="MRN, t = 0")
    for key, color, ls in (("KT10", "#2a78d6", "-"), ("JTH96", "#eb6834", "-"),
                           ("KT10 P1/3", "#2a78d6", "--"), ("KT10 3P1", "#2a78d6", ":")):
        ax2.plot(a * 1e4, runs[key] * a ** 4, color=color, ls=ls, lw=1.5,
                 label={"KT10": "KT10, 10 Myr", "JTH96": "JTH96, 10 Myr",
                        "KT10 P1/3": r"KT10, $P_1/3$, 3.3 Myr",
                        "KT10 3P1": r"KT10, $3P_1$, 30 Myr"}[key])
    ax2.set(xscale="log", yscale="log", xlabel=r"$a$ [$\mu$m]",
            ylabel=r"$a^4\,dN/da$ [cm$^{3}$ per cm$^{3}$]", ylim=(1e-30, 1e-27),
            title="WIM shattering (HK13 setup, MRN start)")
    ax2.legend(frameon=False, fontsize=8, loc="lower center")
    for ax in (ax1, ax2):
        ax.tick_params(which="both", direction="in", top=True, right=True)
    fig.tight_layout()
    out = _paths.get_plots_dir()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "gsd_shattering.png", dpi=150)
    print("Saved", out / "gsd_shattering.png")


if __name__ == "__main__":
    main()
