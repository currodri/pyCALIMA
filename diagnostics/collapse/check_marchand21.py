#!/usr/bin/env python3
"""Benchmarks of the collapse dust module against Marchand+21 (A&A 649, A50).

Figures written to ``_paths.get_plots_dir("collapse")``:

* ``coag_exact_solutions.png`` -- Marchand+21 Fig. 5: constant (K = 2) and
  additive (K = m + m') kernels from n(m) = exp(-m), against the exact
  solutions, on 30 and 60 bins over m = 4e-6 to 4e8.
* ``coag_marchand21_fig1.png`` -- Marchand+21 Fig. 1: turbulent coagulation of
  an MRN distribution to chi = 8.12e17 cgs on their 46-bin grid, against their
  nH4 curve (read from the paper's vector graphics, tests/data/), with our
  nH0 / nH4 / nH8 runs (path independence) and the run with the physical
  T**+1/4 exponent of the turbulent velocity, which their figure does not use.
* ``charge_marchand21_fig2.png`` -- Marchand+21 Fig. 2: mean grain charge and
  ion / electron densities vs n_H (MRN, zeta = 5e-17, mu_i = 25, 10 K), their
  analytic method and Dustdap against our version of their method and our exact
  WDB06-dark P(Z) (grain abundance x 0.85, see tests/test_collapse_charge.py).
* ``frag_marchand21_fig4.png`` -- Marchand+21 Fig. 4: E_kin / 5 E_roll vs n_H
  and the larger grain, with the ratio = 1 band read off their figure.

    python diagnostics/collapse/check_marchand21.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.special import ive

from pycalima import _paths
from pycalima.collapse.charge import (MH_CGS, DarkChargeTable, charge_balance,
                                     marchand21_charge_balance)
from pycalima.collapse.coagulation import advance, intermediate_turbulence_kernel
from pycalima.collapse.velocities import rolling_energy
from pycalima.collapse.grid import SizeGrid
from pycalima.plotting_style import use_calima_style

YR = 3.15576e7
DATA = Path(__file__).resolve().parents[2] / "tests" / "data"
FIG1 = DATA / "marchand21_fig1_nH4.txt"
FIG2 = DATA / "marchand21_fig2.txt"


# ---------------------------------------------------------------------------
# exact solutions
# ---------------------------------------------------------------------------

def rho_constant(edges, t):
    x = edges / (1.0 + t)
    F = (1.0 + x) * np.exp(-x)
    return F[:-1] - F[1:]


def n_additive(m, t):
    tau = -np.expm1(-t)
    st = np.sqrt(tau)
    return (1.0 - tau) / (m * st) * ive(1, 2.0 * m * st) * np.exp(-m * (1.0 - st) ** 2)


def rho_additive(edges, t):
    return np.array([quad(lambda lm: np.exp(2 * lm) * n_additive(np.exp(lm), t),
                          np.log(lo), np.log(hi), limit=200)[0]
                     for lo, hi in zip(edges[:-1], edges[1:])])


def exact_solutions_figure(out):
    cases = [
        ("constant", "Constant kernel  $K=2$", lambda mm, aa: np.full((mm.size,) * 2, 2.0),
         (0.0, 10.0, 1e2, 1e3, 1e4),
         lambda m, t: m ** 2 * np.exp(-m / (1 + t)) / (1 + t) ** 2, rho_constant),
        ("additive", "Additive kernel  $K=m+m'$", lambda mm, aa: mm[:, None] + mm[None, :],
         (0.0, 2.0, 4.0, 6.0, 8.0),
         lambda m, t: m ** 2 * (np.exp(-m) if t == 0 else n_additive(m, t)), rho_additive),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex="col",
                             gridspec_kw={"height_ratios": [3, 1.3]})
    colors = plt.cm.viridis(np.linspace(0, 0.9, 5))
    for col, (w, title, K, times, dens, rho_ex) in enumerate(cases):
        ax, axr = axes[0, col], axes[1, col]
        mfine = np.geomspace(4e-6, 4e8, 2000)
        for nb, mk in ((30, "o"), (60, "s")):
            g = SizeGrid.from_masses(4e-6, 4e8, nb)
            rho = rho_constant(g.m_edges, 0.0)
            t_prev = 0.0
            for t, c in zip(times, colors):
                if t > t_prev:
                    rho, _, _ = advance(g, rho, t - t_prev, K, weights=w)
                t_prev = t
                mc = np.sqrt(g.m_edges[1:] * g.m_edges[:-1])
                ax.plot(mc, rho / g.log_eta, mk, color=c, ms=4 if nb == 60 else 6,
                        mfc="none" if nb == 30 else c, alpha=0.9)
                if t > 0:
                    ex = rho_ex(g.m_edges, t)
                    ok = ex > 1e-6 * ex.max()
                    axr.plot(mc[ok], rho[ok] / ex[ok] - 1.0, mk + "-", color=c, ms=3,
                             mfc="none" if nb == 30 else c, lw=0.6)
        for t, c in zip(times, colors):
            ax.plot(mfine, dens(mfine, t), "-", color=c, lw=1.2, label=f"$t={t:g}$")
        ax.plot([], [], "ko", mfc="none", label="30 bins")
        ax.plot([], [], "ks", ms=4, label="60 bins")
        ax.plot([], [], "k-", label="exact")
        ax.set(xscale="log", yscale="log", ylim=(1e-6, 2), title=title,
               ylabel=r"$m^2 n(m)$  (mass per $\ln m$)")
        ax.legend(fontsize=8, ncol=2)
        axr.axhline(0, color="k", lw=0.5)
        axr.set(xscale="log", ylim=(-0.5, 0.5), xlabel="$m$",
                ylabel=r"$\rho_{\rm bin}/\rho_{\rm exact}-1$")
        axr.text(0.02, 0.06, "bins with $>10^{-6}$ of the peak", transform=axr.transAxes, fontsize=8)
    fig.suptitle("Smoluchowski scheme vs exact solutions (Marchand+21 Fig. 5 tests)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Marchand+21 Fig. 1
# ---------------------------------------------------------------------------

def marchand_grid():
    zeta = 50.0 ** (1.0 / 19)
    g = SizeGrid.from_radii(5e-7 * zeta ** -5, 2.5e-5 * zeta ** 22, 46, 2.3)
    ae = g.a_edges
    w = np.zeros(46)
    w[5:24] = ae[6:25] ** 0.5 - ae[5:24] ** 0.5
    return g, w / w.sum()


def marchand_fig1_figure(out):
    _, log_a_ref, log_X_ref, log_mX_ref = np.loadtxt(FIG1, unpack=True)
    g, w = marchand_grid()
    dust = 3.592e-26          # g per H in their initial MRN (see the data header)
    chi = 8.12e17
    runs = {}
    for name, nH, T in (("nH0", 1.0, 1e5), ("nH4", 1e4, 10.0), ("nH8", 1e8, 10.0)):
        t = chi / (nH ** 0.75 * T ** -0.25)
        rho, _, _ = advance(g, dust * nH * w, t,
                            intermediate_turbulence_kernel(nH, T, 2.3, mu=2.3, T_power=-0.25))
        runs[name] = rho / nH
    rho_phys, _, _ = advance(g, dust * 1e4 * w, 4.58e7 * YR,
                             intermediate_turbulence_kernel(1e4, 10.0, 2.3, mu=2.3, T_power=0.25))
    runs["phys"] = rho_phys / 1e4

    def X_of(mX):
        beta = g.slopes(mX)
        return g.number_density(mX, beta), np.log10(g.mean_radius(beta) * 1e4)

    phys = slice(5, 38)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    ax_X, ax_m, ax_e = axes
    mX0 = dust * w
    X0, la0 = X_of(mX0)
    ok0 = mX0 > 0
    for ax, y0 in ((ax_X, X0), (ax_m, mX0)):
        ax.plot(10 ** la0[ok0], y0[ok0], "k+-", lw=0.8, label="initial MRN")
    ax_X.plot(10 ** log_a_ref[phys], 10 ** log_X_ref[phys], "o", color="tab:red", mfc="none",
              ms=7, label="Marchand+21 nH4 (Fig. 1)")
    ax_m.plot(10 ** log_a_ref[phys], 10 ** log_mX_ref[phys], "o", color="tab:red", mfc="none",
              ms=7, label="Marchand+21 nH4 (Fig. 1)")
    styles = {"nH0": ("tab:cyan", "x", "nH0: $10^0$ cm$^{-3}$, $10^5$ K"),
              "nH4": ("tab:green", "s", "nH4: $10^4$ cm$^{-3}$, 10 K"),
              "nH8": ("darkgreen", ".", "nH8: $10^8$ cm$^{-3}$, 10 K")}
    for name, (c, mk, lab) in styles.items():
        X, la = X_of(runs[name])
        ax_X.plot(10 ** la[phys], X[phys], mk + "-", color=c, ms=4, lw=0.8, label="pyCALIMA " + lab)
        ax_m.plot(10 ** la[phys], runs[name][phys], mk + "-", color=c, ms=4, lw=0.8,
                  label="pyCALIMA " + lab)
        ax_e.plot(10 ** log_a_ref[phys], runs[name][phys] / 10 ** log_mX_ref[phys] - 1.0,
                  mk + "-", color=c, ms=4, lw=0.8, label=name)
    X, la = X_of(runs["phys"])
    full = slice(5, 46)   # it grows further: show it up to the top of the grid
    ax_X.plot(10 ** la[full], X[full], "--", color="tab:purple", lw=1.2,
              label=r"pyCALIMA nH4, physical $T^{+1/4}$")
    ax_m.plot(10 ** la[full], runs["phys"][full], "--", color="tab:purple", lw=1.2,
              label=r"pyCALIMA nH4, physical $T^{+1/4}$")
    for ax, lab in ((ax_X, r"$X_{\rm bin}=n_{\rm bin}/n_{\rm H}$"),
                    (ax_m, r"$mX_{\rm bin}=\rho_{\rm bin}/n_{\rm H}$ [g]")):
        ax.set(xscale="log", yscale="log", xlabel=r"$a$ [$\mu$m]", ylabel=lab)
    ax_X.set_ylim(1e-24, 1e-9)
    ax_m.set_ylim(1e-31, 1e-25)
    for ax in (ax_X, ax_m):
        ax.set_xlim(3e-3, 25)
    ax_X.legend(fontsize=7, loc="lower left")
    ax_e.axhline(0, color="k", lw=0.5)
    ax_e.axhspan(-0.05, 0.05, color="0.9")
    l1 = np.abs(runs["nH4"][phys] - 10 ** log_mX_ref[phys]).sum() / (10 ** log_mX_ref[phys]).sum()
    ax_e.set(xscale="log", ylim=(-0.15, 0.15), xlabel=r"$a$ [$\mu$m]",
             ylabel=r"$mX$ / Marchand+21 $-1$",
             title=f"L1 = {100 * l1:.1f} % (mass-weighted)\nnH0, nH4, nH8 coincide")
    ax_e.legend(fontsize=8)
    fig.suptitle(r"Turbulent coagulation of MRN to $\chi = 8.12\times10^{17}$ cgs "
                 "(Marchand+21 Fig. 1; their $T^{-1/4}$ convention unless noted)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Marchand+21 Fig. 2 and Fig. 4
# ---------------------------------------------------------------------------

def marchand_fig2_figure(out):
    ref = np.loadtxt(FIG2)
    logn = ref[:, 0]
    g = SizeGrid.from_radii(5e-7, 2.5e-5, 50, 2.3)
    beta = np.full(50, -11 / 6)
    a_core = g.mean_radius(beta)
    a = a_core + 8.7e-7
    pick = [int(np.argmin(abs(np.log(a_core / s)))) for s in (1.5e-6, 3e-6, 1e-5, 2.5e-5)]
    per_nH = 0.85 * 0.006 * 2.3 * MH_CGS * g.power_law_rho(-11 / 6, 1.0) / g.mean_mass(beta)
    tab = DarkChargeTable(a, ["silicate"] * 50, log_T=np.array([0.9, 1.1]), log_xi=np.linspace(-5, 5, 201))
    ln = np.linspace(4, 12, 81)
    runs = {}
    for name, solve in (("analytic", lambda nk, nH: marchand21_charge_balance(a, nk, 10.0, nH, 5e-17, 25.0)),
                        ("exact", lambda nk, nH: charge_balance(tab, nk, 10.0, nH, 5e-17, 25.0))):
        rs = [solve(per_nH * 10 ** x, 10 ** x) for x in ln]
        runs[name] = (np.array([-r["Zmean"][pick] for r in rs]), np.array([r["n_i"] for r in rs]),
                      np.array([r["n_e"] for r in rs]))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    cols = ["tab:orange", "tab:cyan", "tab:green", "tab:purple"]
    labels = ["15 nm", "30 nm", "100 nm", "250 nm"]
    for ax, (pname, off) in zip(axes, (("their analytic method (bottom panel)", 1), ("Dustdap (top panel)", 7))):
        axr = ax.twinx()
        for k in range(4):
            ax.plot(10 ** logn, 10 ** ref[:, off + k], "o", color=cols[k], mfc="none", ms=5)
            ax.plot(10 ** ln, runs["analytic"][0][:, k], "-", color=cols[k], lw=1.4, label=labels[k])
            ax.plot(10 ** ln, runs["exact"][0][:, k], ":", color=cols[k], lw=1.6)
        axr.plot(10 ** logn, 10 ** ref[:, off + 4], "ks", mfc="none", ms=4)
        axr.plot(10 ** logn, 10 ** ref[:, off + 5], "k^", mfc="none", ms=4)
        for i, ls in ((1, "-"), (2, "--")):
            axr.plot(10 ** ln, runs["analytic"][i], "k" + ls, lw=1.0)
            axr.plot(10 ** ln, runs["exact"][i], ls, color="0.5", lw=1.0)
        ax.set(xscale="log", yscale="log", ylim=(1e-4, 10), xlabel=r"$n_{\rm H}$ [cm$^{-3}$]",
               title=f"Marchand+21 Fig. 2: {pname}")
        axr.set(yscale="log", ylim=(1e-4, 1e-1))
        axr.set_ylabel(r"$n_{\rm i}$ (solid, squares), $n_{\rm e}$ (dashed, triangles) [cm$^{-3}$]")
    axes[0].set_ylabel(r"$-\langle Z\rangle$")
    axes[0].plot([], [], "ko", mfc="none", label="Marchand+21 (digitised)")
    axes[0].plot([], [], "k-", label="pyCALIMA: their analytic method")
    axes[0].plot([], [], "k:", label="pyCALIMA: exact WDB06-dark $P(Z)$")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle(r"Grain charge in shielded gas: MRN 5-250 nm, $\zeta=5\times10^{-17}$ s$^{-1}$, "
                 r"$\mu_i=25$, 10 K (grain abundance $\times0.85$)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def marchand_fig4_figure(out):
    E5 = 5.0 * rolling_energy(370.0, 0.05e-4)
    ln = np.linspace(4, 12, 161)
    a = np.geomspace(1e-5, 1e-3, 201)
    m = 4.0 / 3.0 * np.pi * 2.3 * a ** 3
    R = np.empty((a.size, ln.size))
    for j, x in enumerate(ln):
        K = intermediate_turbulence_kernel(10 ** x, 10.0, 2.3, mu=2.3, T_power=-0.25)(None, a)
        dv = np.diag(K) / (np.sqrt(8 / (3 * np.pi)) * np.pi * (2 * a) ** 2)
        R[:, j] = 0.5 * (m / 2) * dv ** 2 / E5
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    pc = ax.pcolormesh(10 ** ln, a * 1e4, R, cmap="bwr", norm=matplotlib.colors.LogNorm(1e-2, 1e2),
                       shading="auto")
    fig.colorbar(pc, label=r"$E_{\rm kin}/5E_{\rm roll}$ (equal-mass partners)")
    ax.contour(10 ** ln, a * 1e4, R, levels=[1.0], colors="k")
    ax.plot([1e4, 1e6, 1e8, 1e10, 1e12], [0.45, 0.8, 1.4, 2.4, 4.3], "ko", mfc="yellow",
            label="Marchand+21 Fig. 4: ratio = 1 (read off)")
    ax.plot([], [], "k-", label="pyCALIMA: ratio = 1")
    ax.set(xscale="log", yscale="log", xlabel=r"$n_{\rm H}$ [cm$^{-3}$]", ylabel=r"larger grain $a$ [$\mu$m]",
           title="Collision restructuring threshold (Marchand+21 Fig. 4)")
    ax.legend(fontsize=8, loc="upper left")
    ax.text(0.98, 0.04, r"same slope $a\propto n^{1/8}$; theirs $1.25\times$ higher in $a$" "\n"
            "(their partner mass is not stated)", transform=ax.transAxes, ha="right", fontsize=8,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    use_calima_style()
    plt.rcParams["text.usetex"] = False
    outdir = _paths.get_plots_dir("collapse")
    for name, fn in (("coag_exact_solutions.png", exact_solutions_figure),
                     ("coag_marchand21_fig1.png", marchand_fig1_figure),
                     ("charge_marchand21_fig2.png", marchand_fig2_figure),
                     ("frag_marchand21_fig4.png", marchand_fig4_figure)):
        fn(outdir / name)
        print("wrote", outdir / name)


if __name__ == "__main__":
    main()
