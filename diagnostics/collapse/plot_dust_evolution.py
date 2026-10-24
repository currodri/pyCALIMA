"""Figures for the dust-evolution study of low-metallicity collapse.

    python diagnostics/collapse/plot_dust_evolution.py

Writes under ``collapse/`` in the plots directory:

* ``zcr_ladder.png``: [Z_cr] of the N07 CCSN models by process (static,
  accretion, accretion + turbulent coagulation), both criteria, against
  Chiaki+15 and Grackle; from ``science_grid_dust_evolution.py``'s results.
* ``evolution_m20n1.png``: one collapse (20 Msun, n_amb = 1, Grackle grains and
  gas, [Z/H] = -4.5) for static dust, accretion, and accretion + coagulation
  (Mach 10): T, S, gas-phase Mg / Fe / SiO, and the size distribution.
* ``timescales.png``: accretion, coagulation and H2O freeze-out times over the
  collapse time, at [Z/H] = -5 and -4.
* ``ice_and_charge.png``: dust and gas temperatures against the H2O
  sublimation temperature; the charge budget (x_e without and with cosmic
  rays, grains per H) and the Coulomb focusing it allows for Mg+ / Fe+ accretion.
* ``grackle_comparison.png``: T(n) on the Omukai+05 and Grackle tracks, ours
  and Grackle's.
"""

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pycalima import _paths
from pycalima.collapse.charge import marchand21_charge_balance
from pycalima.collapse.grid import bin_means
from pycalima.collapse.ics import nozawa_ccsn
from pycalima.collapse.integrator import run_collapse, turbulent_kernel
from pycalima.collapse.tracks import KB_CGS, MH_CGS, Track
from pycalima.plotting_style import use_calima_style

OUT = _paths.get_plots_dir() / "collapse"
GRACKLE_ZCR = Path(__file__).resolve().parents[2] / "tests" / "data" / "grackle_ccsn_n1_zcr.txt"
ROOT = _paths.get_results_dir() / "collapse"
CHIAKI = {"n0": {13: (-5.3, -5.9), 20: (-5.2, -5.2), 25: (-5.1, -5.1), 30: (-5.1, -5.1)},
          "n1": {13: (-4.9, -5.3), 20: (-4.3, -4.9), 25: (-4.7, -5.0), 30: (-4.1, -5.1)}}   # (static, growth)
LADDER = ["static", "acc", "acc+coag M1", "acc+coag M3", "acc+coag M10", "acc+coag M10 Kx10"]
STYLE = {"static": ("k", "o"), "acc": ("C0", "s"), "acc+coag M1": ("C1", "^"), "acc+coag M3": ("C2", "v"),
         "acc+coag M10": ("C3", "D"), "acc+coag M10 Kx10": ("C4", "x")}
E_H2O, NU_H2O, N_SITES = 5700.0, 1e13, 1e15     # K (Fraser+01), s^-1, sites cm^-2
RUN = dict(n0=1e-1, n_end=1e16, dlnn=0.1, energy=True)


def _profile(M, lz):
    """Grackle's dusty collapse (n_amb = 1) at [Z/H] = lz: dict of log n, log T, log fH2, log x_e, log xi, log x_H2O."""
    d = np.loadtxt(_paths.get_external_data_path("collapse_tracks", "grackle_ccsn_n1_dusty_profiles.txt"))
    d = d[(d[:, 0] == M) & np.isclose(d[:, 1], lz)]
    return dict(zip(("ln", "lT", "lf", "lxe", "lxi", "lxw"), d[:, 2:].T))


def _runs(M, lz, which=("static", "acc", "coag")):
    pops, y, _ = nozawa_ccsn(M, 10.0 ** lz, n_amb=1, grains="grackle")
    tr = Track.grackle(M, 10.0 ** lz)
    out = {}
    for w in which:
        kw = dict(accretion=w in ("acc", "coag"))
        if w == "coag":
            kw.update(coagulation=True, coag_kernel=turbulent_kernel(mach=10.0))
        if w == "coag only":
            kw = dict(coagulation=True, coag_kernel=turbulent_kernel(mach=10.0))
        out[w] = run_collapse(tr, pops, y_gas=y, **kw, **RUN)
    return out, pops, y


def _size_dist(pops):
    """(a [um], dM/dlog a summed over species, per H) on the common grid."""
    g = pops[0].grid
    a = np.sqrt(g.a_edges[1:] * g.a_edges[:-1]) * 1e4
    dlog = np.log10(g.a_edges[1:] / g.a_edges[:-1])
    return a, sum(p.x for p in pops) / dlog


def fig_ladder():
    path = ROOT / "science_grid_dust_evolution.txt"
    if not path.exists():
        print(f"skip zcr_ladder: run science_grid_dust_evolution.py first ({path})")
        return
    rows = [ln.split() for ln in path.read_text().splitlines() if not ln.startswith("#")]
    zcr = {(r[0], int(r[1]), r[2]): np.array(r[3:], float) for r in rows}
    grk = {round(r[0]): r[1] for r in np.loadtxt(GRACKLE_ZCR)}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    models = [(M, ic) for ic in ("n0", "n1") for M in (13, 20, 25, 30)]
    for ax, cr in zip(axes, ("C", "frag")):
        for x, (M, ic) in enumerate(models):
            z = zcr.get((cr, M, ic))
            if z is None:
                continue
            for k, p in enumerate(LADDER):
                c, m = STYLE[p]
                v = z[k] if np.isfinite(z[k]) else -2.6
                ax.plot(x + (k - 2.5) * 0.08, v, m, color=c, mfc="none" if not np.isfinite(z[k]) else c, ms=5,
                        label=p if x == 0 else None)
            ax.plot(x - 0.3, CHIAKI[ic][M][0], "_", color="k", ms=14, mew=2, label="Chiaki+15 static" if x == 0 else None)
            ax.plot(x + 0.3, CHIAKI[ic][M][1], "_", color="C0", ms=14, mew=2, label="Chiaki+15 growth" if x == 0 else None)
            if ic == "n1":
                ax.plot(x, grk[M], "*", color="0.5", ms=9, label="Grackle static" if x == 4 else None)
        ax.axvline(3.5, color="0.7", lw=0.8)
        ax.set_xticks(range(len(models)), [f"M{M}\n{ic}" for M, ic in models])
        ax.set_title({"C": "criterion C (cooling time vs collapse time)", "frag": "criterion frag (Schneider and Omukai 2010)"}[cr],
                     fontsize=10)
        ax.text(1.5, -3.5, "unprocessed,\nOmukai+05 Z = 0 track", ha="center", fontsize=8)
        ax.text(5.5, -3.2, "reverse shock n_amb = 1,\nGrackle grains + gas", ha="center", fontsize=8)
    axes[0].set_ylabel(r"[$Z_{\rm cr}$/H]  (open: no fragmentation found)")
    axes[0].set_ylim(-5.5, -2.5)
    axes[0].invert_yaxis()
    axes[1].legend(fontsize=7, ncol=2, loc="lower left", bbox_to_anchor=(0.0, 0.02))
    fig.suptitle("Critical metallicity by process: coagulation (up to Mach 10, kernel x10) moves nothing; accretion does",
                 fontsize=10)
    _save(fig, "zcr_ladder.png")


def fig_evolution():
    lz = -4.5
    runs, pops0, y0 = _runs(20, lz)
    prof = _profile(20, lz)
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    lab = {"static": "static", "acc": "accretion", "coag": "accretion + coagulation (Mach 10)"}
    col = {"static": "k", "acc": "C0", "coag": "C3"}
    for w, (h, pops) in runs.items():
        a = h.arrays()
        ax[0, 0].semilogx(a["nH"], a["T"], color=col[w], label=lab[w], ls="--" if w == "coag" else "-")
        ax[0, 1].loglog(a["nH"], a["S"], color=col[w], label=lab[w], ls="--" if w == "coag" else "-")
        if w != "static":
            for e, ls in (("Mg", "-"), ("Fe", ":"), ("Si", "-.")):
                yy = np.array([d[e] for d in h.y_gas]) / max(y0[e], 1e-300)
                if y0[e] > 0:
                    ax[1, 0].semilogx(a["nH"], yy, color=col[w], ls=ls, label=f"{e} ({lab[w]})" if w == "acc" or e == "Mg" else None)
        ad, dm = _size_dist(pops)
        ax[1, 1].loglog(ad, np.maximum(dm, 1e-60), color=col[w], ls="--" if w == "coag" else "-", label=f"{lab[w]}, n = 1e16")
    ad, dm = _size_dist(pops0)
    ax[1, 1].loglog(ad, np.maximum(dm, 1e-60), color="0.6", lw=3, alpha=0.5, label="initial (Grackle ccsn20 grains)")
    tr = Track.grackle(20, 10.0 ** lz)
    ax[0, 0].semilogx(10.0 ** tr.log_n, 10.0 ** tr.log_T, color="0.6", lw=3, alpha=0.5, label="Grackle dust-free track")
    ax[0, 0].semilogx(10.0 ** prof["ln"], 10.0 ** prof["lT"], color="C2", ls=":", label="Grackle with dust")
    ax[0, 0].set(xlabel=r"$n_{\rm H}$ [cm$^{-3}$]", ylabel="T [K]", ylim=(50, 3000), yscale="log")
    ax[0, 1].set(xlabel=r"$n_{\rm H}$ [cm$^{-3}$]", ylabel=r"S = grain area / dust mass [cm$^2$ g$^{-1}$]")
    ax[1, 0].set(xlabel=r"$n_{\rm H}$ [cm$^{-3}$]", ylabel="gas-phase fraction left", ylim=(0, 1.05))
    ax[1, 1].set(xlabel=r"a [$\mu$m]", ylabel=r"d(dust mass per H)/dlog a [g]", xlim=(1e-3, 3))
    top = max(np.max(_size_dist(p)[1]) for _, p in runs.values())
    ax[1, 1].set_ylim(top * 1e-4, top * 3)
    for a_ in ax.flat:
        a_.legend(fontsize=7)
    fig.suptitle(f"20 Msun CCSN dust (reverse shock, n_amb = 1), [Z/H] = {lz}: accretion depletes the gas-phase Mg, Fe "
                 f"above 1e10 cm$^{{-3}}$ and raises S; coagulation leaves it unchanged", fontsize=10)
    _save(fig, "evolution_m20n1.png")


def _t_change(h):
    """Time for S to change by e from one step to the next, over the collapse time, per step."""
    a = h.arrays()
    dt = np.diff(a["t"])
    dl = np.abs(np.diff(np.log(np.maximum(a["S"], 1e-300))))
    with np.errstate(divide="ignore"):
        return 0.5 * (a["nH"][1:] + a["nH"][:-1]), np.where(dl > 0, dt / dl, np.inf) / a["t_col"][1:]


def _t_freeze(h, pops_end=None):
    """H2O freeze-out time (sticking 1) over the collapse time: 1 / (sum n_gr pi a^2 v_th(H2O))."""
    a = h.arrays()
    v = np.sqrt(8.0 * KB_CGS * a["T"] / (math.pi * 18.0 * MH_CGS))
    area_per_H = np.asarray(h.S) * np.asarray(h.D) * (1.0 + 4.0 * 0.0833) * MH_CGS   # grain area per H nucleus
    return a["nH"], 1.0 / (area_per_H * a["nH"] * v) / a["t_col"]


def fig_timescales():
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for lz, ls in ((-5.0, "-"), (-4.0, "--")):
        runs, _, _ = _runs(20, lz, which=("acc", "coag only"))
        n, t = _t_change(runs["acc"][0])
        ax.loglog(n, t, color="C0", ls=ls, label=f"accretion, [Z/H] = {lz:g}")
        n, t = _t_change(runs["coag only"][0])
        ax.loglog(n, t, color="C3", ls=ls, label=f"coagulation (Mach 10), [Z/H] = {lz:g}")
        n, t = _t_freeze(runs["acc"][0])
        ax.loglog(n, t, color="C2", ls=ls, label=f"H$_2$O freeze-out, [Z/H] = {lz:g}")
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set(xlabel=r"$n_{\rm H}$ [cm$^{-3}$]", ylim=(1e-2, 1e12),
           ylabel=r"$t_{\rm process} / t_{\rm col}$  ($= |d\ln S/dt|^{-1}$; spikes: S turning over)",
           title="Only accretion acts within a collapse time (20 Msun, n_amb = 1)")
    ax.legend(fontsize=7)
    _save(fig, "timescales.png")


def fig_ice_and_charge():
    lz = -4.5
    runs, pops, y = _runs(20, lz, which=("acc",))
    h, _ = runs["acc"]
    a = h.arrays()
    prof = _profile(20, lz)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    Td = np.array(h.Td)
    for k, p in enumerate(pops):
        ax[0].loglog(a["nH"], Td[:, k], lw=0.8, label=f"$T_d$ {p.material}")
    ax[0].loglog(a["nH"], a["T"], "k", lw=2, label=r"$T_{\rm gas}$")
    # H2O sublimation: desorption nu N_s exp(-E/kT) = adsorption n_H2O v / 4, with Grackle's x(H2O)
    n = 10.0 ** prof["ln"]
    n_w = 10.0 ** prof["lxw"] * n
    T_sub = np.full_like(n, 100.0)
    for _ in range(30):
        v = np.sqrt(8 * KB_CGS * T_sub / (math.pi * 18 * MH_CGS))
        T_sub = E_H2O / np.log(NU_H2O * N_SITES / np.maximum(n_w * v / 4.0, 1e-300))
    ax[0].loglog(n, T_sub, color="C0", lw=3, alpha=0.6, label="H$_2$O sublimation (Grackle x(H$_2$O))")
    ax[0].set(xlabel=r"$n_{\rm H}$ [cm$^{-3}$]", ylabel="T [K]", ylim=(10, 3000),
              title="Ice: grains are cold enough only where they are too rare to collect it")
    ax[0].legend(fontsize=6, ncol=2)
    # charge budget
    xe_grackle = 10.0 ** np.interp(np.log10(a["nH"]), prof["ln"], prof["lxe"])
    ax[1].loglog(a["nH"], xe_grackle, "k", label="$x_e$, Grackle (no cosmic rays)")
    ax[1].loglog(a["nH"], a["n_grains"], color="0.5", lw=2, label="grains per H")
    x_metal = np.array([d["Mg"] + d["Fe"] for d in h.y_gas])
    ax[1].loglog(a["nH"], x_metal, color="C1", lw=2, label="gas-phase Mg + Fe per H")
    focus_ax = ax[1].twinx()
    focus_ax.semilogx(a["nH"], np.ones_like(a["nH"]) * 1.0 + (xe_grackle / np.maximum(x_metal, 1e-300)).clip(0, 1) * 2.5,
                      color="k", ls=":", label="Mg/Fe accretion boost, no CR")
    for zeta, c in ((1e-17, "C3"), (1e-16, "C4")):
        xe, boost = [], []
        for k in range(len(a["nH"])):
            nH, T = a["nH"][k], a["T"][k]
            st = [(p.grid, p.x * nH) for p in pops]
            rad = np.concatenate([g.a_edges[:-1] * bin_means(np.ones(g.n_bins), g.eta, g.slopes(r), 1 / 3) for g, r in st])
            ng = np.concatenate([g.number_density(r, g.slopes(r)) for g, r in st])
            ok = ng > 0
            # Mg+ / Fe+: radiative recombination, 2.8e-12 (T / 300)^-0.68 cm^3 s^-1 (UMIST, Mg+ + e)
            try:
                r = marchand21_charge_balance(rad[ok], ng[ok], T, nH, zeta, 24.3, k_rec=2.8e-12 * (T / 300.0) ** -0.68)
            except ValueError:          # no root within (psi0, 0): leave the point out
                xe.append(math.nan)
                boost.append(math.nan)
                continue
            xe.append(r["n_e"] / nH)
            f_ion = min(r["n_i"] / max(x_metal[k] * nH, 1e-300), 1.0)
            boost.append(1.0 + f_ion * (-r["psi"]))
        ax[1].loglog(a["nH"], xe, color=c, label=fr"$x_e$, $\zeta$ = {zeta:g} s$^{{-1}}$")
        focus_ax.semilogx(a["nH"], boost, color=c, ls=":", label=fr"boost, $\zeta$ = {zeta:g}")
    focus_ax.set(ylabel="Mg/Fe accretion boost (dotted)", ylim=(0.9, 4.5))
    ax[1].set(xlabel=r"$n_{\rm H}$ [cm$^{-3}$]", ylabel="abundance per H", ylim=(1e-20, 1e-6),
              title="Charge: Coulomb-focused Mg$^+$/Fe$^+$ accretion needs an ionisation source")
    h1, l1 = ax[1].get_legend_handles_labels()
    h2, l2 = focus_ax.get_legend_handles_labels()
    ax[1].legend(h1 + h2, l1 + l2, fontsize=6, loc="lower left")
    fig.suptitle(f"20 Msun CCSN dust (n_amb = 1, Grackle grains and gas), [Z/H] = {lz}, with accretion", fontsize=10)
    _save(fig, "ice_and_charge.png")


def fig_grackle():
    lz = -4.5
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    om = Track.omukai2005("Z0")
    ax.loglog(10.0 ** om.log_n, 10.0 ** om.log_T, color="0.6", lw=3, alpha=0.6, label="Omukai+05 Z = 0 (our default track)")
    g7 = Track.grackle(20, 1e-7)
    ax.loglog(10.0 ** g7.log_n, 10.0 ** g7.log_T, color="C2", lw=3, alpha=0.4, label="Grackle dust-free, [Z/H] = -7")
    g = Track.grackle(20, 10.0 ** lz)
    ax.loglog(10.0 ** g.log_n, 10.0 ** g.log_T, color="C2", ls="--", label=f"Grackle dust-free, [Z/H] = {lz}")
    prof = _profile(20, lz)
    ax.loglog(10.0 ** prof["ln"], 10.0 ** prof["lT"], color="C2", label=f"Grackle with dust, [Z/H] = {lz}")
    pops, y, _ = nozawa_ccsn(20, 10.0 ** lz, n_amb=1, grains="grackle")
    for tr, c, lab in ((om, "k", "ours on Omukai+05"), (g, "C0", "ours on Grackle's dust-free gas")):
        h, _ = run_collapse(tr, pops, y_gas=y, **RUN)
        ax.loglog(h.arrays()["nH"], h.arrays()["T"], color=c, label=f"{lab}, static dust")
    ax.set(xlabel=r"$n_{\rm H}$ [cm$^{-3}$]", ylabel="T [K]", ylim=(80, 4000), xlim=(0.1, 1e16),
           title=f"20 Msun CCSN dust (n_amb = 1), [Z/H] = {lz}, no growth")
    ax.legend(fontsize=7)
    _save(fig, "grackle_comparison.png")


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=140)
    plt.close(fig)
    print(f"wrote {OUT / name}")


def main():
    use_calima_style()
    fig_ladder()
    fig_evolution()
    fig_timescales()
    fig_ice_and_charge()
    fig_grackle()


if __name__ == "__main__":
    main()
