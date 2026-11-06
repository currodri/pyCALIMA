#!/usr/bin/env python3
"""PAH ion neutralisation alpha_PAH against Weingartner & Draine (2001b) alpha_g (validation V4).

WD01b's fit integrates the WD01 size distribution (R_V = 3.1, b_C = 6e-5), PAH-sized grains included, in the
Mathis ISRF; psi = G sqrt(T) / n_e with G in Habing. Three numbers per ion, T and psi:

- **pahchem:** the v4.00 molecules with every charge -1..+2 (N_C 22-268), in four N_C bins
  (``group_tables``, UV background only, G0 = 1 Habing Mathis); each bin carries the WD01 graphite grains of
  the same carbon count, N_C = 468 (a / 10 A)^3, i.e. a = 3.5-8.3 A. alpha = alpha_mn + alpha_ct per H.
- **WD01 physics, PAH sizes:** pyCALIMA's full-spectrum grain charging with WD01 yields and case A thresholds
  (``check_grain_recombination_wdb06._alpha_full``), integrated over the same 3.5-8.3 A graphite grains.
- **WD01 physics, all sizes,** and the WD01b fit (``check_grain_size_distribution``).

pahchem / (WD01 physics, PAH sizes) compares the two treatments of the same carriers; (PAH sizes) / (all sizes)
is the PAHs' share of WD01b's alpha_g. On both sides the charge balance has n(H+) = n_e; pahchem's has no
associative detachment (n_H = 0), which WD01b does not include either. Nothing is tuned.

Result (2026-10-23; pahchem / WD01(PAH) is the same for H+, He+, C+, Mg+, Si+, S+ and Fe+ to within 5 %, except
at T = 6000 K and psi >= 3e4, where it spreads 1.2-4x between ions):
- PAH sizes carry 0.8-0.93 of WD01's alpha at psi <= 1e4; at 1e6, 0.3-0.7 for H+, He+, C+ and S+, and all of it
  for Mg+, Si+ and Fe+;
- pahchem / WD01(PAH) is 2.0 at T = 100 K and 1.2-1.5 at 1000 K for psi <= 1e3, 0.9-1.1 at 6000 K, rising
  to 6-9 at psi >= 1e5 for T <= 1000 K (3.7-7.8 at 6000 K). The excess is in the charge distribution, not the
  collision rates. For N_C 40-70 at T = 100 K, pahchem has f(-) = 0.60 against WD01's 0.18 at psi = 1e2, and
  f(0) = 0.155 against 0.016 at 1e5, where charge transfer on neutrals dominates (pahchem's measured-and-shifted
  IPs, e.g. 6.5 eV for C54H18 against WD01's 6.24 (eq. 2 at a = 4.87 A), and the Jochims yields). Above psi ~ 1e5 pahchem's chain stops
  at +2 (WD01 reaches +3), so the ratio there is partly an upper bound;
- WD01 physics against its own WD01b fit: 0.55-2.5 over the grid.

Results are kept in results/pahchem_wd01b_alpha.json (``--plot-only`` redraws from it).

Rerun on 2026-11-02 with the current defaults: section 4 of the gas-coupling doc, where ``check_default_changes.py``
adds the changes since 2026-10-25 one at a time.

    check_wd01b_alpha.py [--workers 4] [--plot-only]
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
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "dust_charge"))
import check_grain_recombination_wdb06 as R
import check_grain_size_distribution as gsd

from pycalima import _paths
from pycalima.pahchem import group_tables as G
from pycalima.pahchem import population
from pycalima.pahchem.baseline import Params
from pycalima.plotting_style import use_calima_style

IONS = ("H+", "He+", "C+", "Mg+", "Si+", "S+", "Fe+")
T_GRID = (100.0, 1000.0, 6000.0)
PSI_GRID = np.logspace(2.0, 6.0, 9)
NC_EDGES = (20, 40, 70, 120, 270)                    # 20: WD01's a_min = 3.5 A
IP_SHIFTS = {-1: 0.37, 0: 0.6, 1: 0.64, 2: 0.64}


def radius_cm(nc):
    return 1e-7 * (np.asarray(nc, dtype=float) / 468.0) ** (1.0 / 3.0)


A_PAH = np.geomspace(gsd.A_MIN, radius_cm(NC_EDGES[-1]), 10)
A_GRAPHITE = np.geomspace(radius_cm(NC_EDGES[-1]), 1e-4, 32)
A_SILICATE = np.geomspace(gsd.A_MIN, 1e-4, 32)


def per_h(dn_da, sizes, alpha):
    """int alpha(a) dn/da da over ``sizes`` [per H]; alpha[size, ...]."""
    dn = np.array([dn_da(a) for a in sizes])
    return np.trapezoid(alpha * dn.reshape((-1,) + (1,) * (alpha.ndim - 1)), sizes, axis=0)


def mathis_G():
    """Habing G of the unscaled Mathis ISRF that ``equilibrium_charge_for_grain`` uses at G0 = 1."""
    from pycalima.models.dust_charge import rt_group_charging as rg
    from pycalima.models.tools.radiation_fields import Mathis83_radiation_field

    E = np.geomspace(6.0, 13.6, 400)
    uE = np.array([Mathis83_radiation_field(e) for e in E])     # erg cm^-3 eV^-1
    return float(rg.C_CGS * np.trapezoid(uE, E) / rg.HABING_FLUX)


def _alpha_full(args):
    """WD01 grain-assisted recombination alpha_i [cm^3 s^-1 per grain] of ``rt_group_charging.RECOMB_IONS`` over
    the full-spectrum P(Z): sum_Z P(Z) pi a^2 v_i J~(Z) theta(IP(X) - IP(a, Z)) (WD01 "case a"). The reference that
    ``models/dust_charge/dust_ion_recombination.py`` held until its removal (6a56802); kept here for V4 only.
    Appended: the same solve's photoelectric heating and recombination cooling [erg s^-1 per grain] and <Z>."""
    from pycalima.models.dust_charge import rt_group_charging as rg
    from pycalima.models.dust_charge.dust_charging import equilibrium_charge_for_grain
    from pycalima.models.dust_charge.shared_physics import (DS87_J_function_vec, KB_CGS,
                                                            electron_affinity_graphite_vec,
                                                            electron_affinity_silicate_vec,
                                                            ionisation_potential_valence_vec)

    G0_mathis, ne, T, material, a, yield_model = args
    Zs, P, rates, z_mean, _ = equilibrium_charge_for_grain(G0_mathis, ne, T, material, a,
                                                  yield_params={"yield_model": yield_model},
                                                  ion_species=[{"n": ne, "T": T, "m": 1.67262192e-24, "z": 1.0}])
    Zs, P = np.asarray(Zs, dtype=float), np.asarray(P, dtype=float)
    graphite = material.lower().startswith(("gra", "car"))
    ea = electron_affinity_graphite_vec if graphite else electron_affinity_silicate_vec
    ip_grain = np.where(Zs >= 0, ionisation_potential_valence_vec(4.4 if graphite else 8.0, Zs, a), ea(Zs + 1, a))
    J = DS87_J_function_vec(Zs, np.array([1.0]), a, T)
    out = []
    for _el, m_u, ip_x in rg.RECOMB_IONS:
        v = np.sqrt(8.0 * KB_CGS * T / (np.pi * m_u * rg.M_U))
        out.append(float(np.sum(P * np.pi * a * a * v * J * (ip_x - ip_grain >= 0.0))))
    return np.array([*out, rates["Gamma_total"], rates["Recomb_total"], z_mean])


def compute(workers, detachment=Params().detachment):
    G_mathis = mathis_G()
    jobs, keys = [], []
    for iT, T in enumerate(T_GRID):
        for ip, psi in enumerate(PSI_GRID):
            ne = np.sqrt(T) / psi
            for part, mat, sizes in (("pah", "graphite", A_PAH), ("graphite", "graphite", A_GRAPHITE),
                                     ("silicate", "silicate", A_SILICATE)):
                for ia, a in enumerate(sizes):
                    jobs.append((1.0 / G_mathis, ne, T, mat, a, "WD01"))
                    keys.append((part, iT, ip, ia))
    print(f"WD01 physics: {len(jobs)} full-spectrum grain solves on {workers} workers", flush=True)
    with ProcessPoolExecutor(workers) as ex:
        res = dict(zip(keys, ex.map(_alpha_full, jobs, chunksize=4), strict=True))
    n_ion = len(R.ION_INDEX)
    cols = [R.ION_INDEX[i] for i in IONS] + [n_ion, n_ion + 1]               # the ions, then heating and cooling
    wd = {}
    for part, dn_da, sizes in (("pah", gsd.graphite_dn_da, A_PAH), ("graphite", gsd.graphite_dn_da, A_GRAPHITE),
                               ("silicate", gsd.silicate_dn_da, A_SILICATE)):
        al = np.array([[[res[(part, iT, ip, ia)][cols] for ip in range(PSI_GRID.size)]
                        for iT in range(len(T_GRID))] for ia in range(sizes.size)])
        wd[part] = per_h(dn_da, sizes, al)                                   # (T, psi, ion + 2)
    wd_total = wd["pah"] + wd["graphite"] + wd["silicate"]
    z_wd = [[[res[("pah", iT, ip, ia)][n_ion + 2] for ia in range(A_PAH.size)] for ip in range(PSI_GRID.size)]
            for iT in range(len(T_GRID))]

    lib = population.shift_thresholds(population.library_v400(), IP_SHIFTS)
    a_fine = np.geomspace(gsd.A_MIN, radius_cm(NC_EDGES[-1]), 400)
    pah = np.zeros_like(wd_total)
    bins, z_bins = [], []
    for lo, hi in pairwise(NC_EDGES):
        w = G.nc_range_weights(lib, lo, hi)
        a = a_fine[(a_fine >= radius_cm(lo)) & (a_fine <= radius_cm(hi))]
        n_bin = float(np.trapezoid([gsd.graphite_dn_da(x) for x in a], a))
        print(f"pahchem bin N_C {lo}-{hi}: {int(w.sum())} molecules, {n_bin:.3e} per H", flush=True)
        tab = G.PAHGroupTable(G.build_bin_table(lib, w, params=Params(detachment=detachment), workers=workers, label=f"{lo}-{hi}"))
        bins.append({"nc": (lo, hi), "n_mol": int(w.sum()), "n_per_h": n_bin, "mean_nc": tab.t["nc"]})
        z_bin = np.zeros((len(T_GRID), PSI_GRID.size))
        for iT, T in enumerate(T_GRID):
            for ip, psi in enumerate(PSI_GRID):
                ne = np.sqrt(T) / psi
                r = tab.solve(np.zeros(tab.t["L0"].size), T, ne, 0.0, (("H+", ne),), G0_bg=1.0)
                pah[iT, ip] += n_bin * np.array([r["alpha_mn"][i] + r["alpha_ct"][i] for i in IONS]
                                                + [r["gamma"], r["lam"]])
                z_bin[iT, ip] = r["f"] @ np.array(G.CHARGES, dtype=float)
        z_bins.append(z_bin.tolist())
    fit = np.array([[[gsd.get_analytical_recomb_coefficient(i, 1.0, np.sqrt(T) / psi, T) for i in IONS]
                     for psi in PSI_GRID] for T in T_GRID])
    n = len(IONS)
    return {"ions": IONS, "T": T_GRID, "psi": PSI_GRID.tolist(), "bins": bins, "pahchem": pah[..., :n].tolist(),
            "wd01_pah": wd["pah"][..., :n].tolist(), "wd01_total": wd_total[..., :n].tolist(), "fit": fit.tolist(),
            "heat": {"pahchem": pah[..., n:].tolist(), "wd01_pah": wd["pah"][..., n:].tolist(),
                     "wd01_total": wd_total[..., n:].tolist()},
            "z": {"a_wd01": A_PAH.tolist(), "wd01": z_wd, "pahchem": z_bins}}


#: WD01 (ApJS 134, 263) eqs. 44 and 45 for the ISRF, R_V = 3.1, b_C = 6e-5 (Table 2 rows; literature page)
WD01_EQ44 = (9.30, 0.248, 0.00697, 0.01848, 0.365, 0.633, 0.509)
WD01_EQ45 = (0.3632, 2.937, -7.601, 1.742, 0.06228)


def heating_fits(T, psi):
    """{name: (Gamma / (G n_H), Lambda / (n_e n_H))} [erg s^-1 cm^3 or cm^3] of WD01 eqs. 44-45 and BT94 eqs. 43-44
    (gamma = psi, G0 in Habing)."""
    c = WD01_EQ44
    d = WD01_EQ45
    x = np.log(psi)
    wd_g = 1e-26 * (c[0] + c[1] * T ** c[4]) / (1.0 + c[2] * psi ** c[5] * (1.0 + c[3] * psi ** c[6]))
    wd_l = 1e-28 * T ** (d[0] + d[1] / x) * np.exp(d[2] + d[3] * x - d[4] * x * x)
    eps = 4.87e-2 / (1.0 + 4e-3 * psi ** 0.73) + 3.65e-2 * (T / 1e4) ** 0.7 / (1.0 + 2e-4 * psi)
    bt_l = 3.49e-30 * T ** 0.944 * psi ** (0.735 / T ** 0.068)
    return {"WD01 eqs. 44-45": (wd_g, wd_l), "BT94 eqs. 43-44": (1e-24 * eps, bt_l)}


def report(d):
    pah, wp, wt, fit = (np.array(d[k]) for k in ("pahchem", "wd01_pah", "wd01_total", "fit"))
    for k, ion in enumerate(d["ions"]):
        print(f"\n{ion}: alpha per H [cm^3 s^-1]")
        print("      T      psi   pahchem  WD01(PAH)  WD01(all)  WD01b fit | pahchem/WD01(PAH)  PAH share  "
              "pahchem/fit  WD01(all)/fit")
        for iT, T in enumerate(d["T"]):
            for ip, psi in enumerate(d["psi"]):
                p, a, t, f = pah[iT, ip, k], wp[iT, ip, k], wt[iT, ip, k], fit[iT, ip, k]
                print(f"  {T:6.0f} {psi:8.0e}  {p:.2e}  {a:.2e}   {t:.2e}   {f:.2e} | {p / a:17.2f}  {a / t:9.2f}  "
                      f"{p / f:11.2f}  {t / f:13.2f}")


def report_heat(d):
    h, wp, wt = (np.array(d["heat"][k]) for k in ("pahchem", "wd01_pah", "wd01_total"))
    psi = np.array(d["psi"])
    print("\nPhotoelectric heating Gamma and recombination cooling Lambda per H (G = 1 Habing, n_H = 1)")
    print("      T      psi | Gamma: pahchem  WD01(PAH)  WD01(all)  eq.44   BT94 | Lambda: pahchem  WD01(PAH)  "
          "WD01(all)  eq.45   BT94 | pahchem/WD01(PAH) G, L")
    for iT, T in enumerate(d["T"]):
        fits = heating_fits(T, psi)
        for ip, ps in enumerate(psi):
            ne = np.sqrt(T) / ps
            wg, wl = (fits["WD01 eqs. 44-45"][k][ip] for k in (0, 1))
            bg, bl = (fits["BT94 eqs. 43-44"][k][ip] for k in (0, 1))
            print(f"  {T:6.0f} {ps:8.0e} | {h[iT, ip, 0]:.2e} {wp[iT, ip, 0]:.2e} {wt[iT, ip, 0]:.2e} {wg:.2e} "
                  f"{bg:.2e} | {h[iT, ip, 1]:.2e} {wp[iT, ip, 1]:.2e} {wt[iT, ip, 1]:.2e} {wl * ne:.2e} {bl * ne:.2e} "
                  f"| {h[iT, ip, 0] / wp[iT, ip, 0]:5.2f} {h[iT, ip, 1] / wp[iT, ip, 1]:5.2f}")


def _ticks(ax):
    ax.minorticks_on()
    ax.tick_params(which="both", direction="in", top=True, right=True)


def plot_heat(d, path):
    """Gamma / (G n_H), Lambda / (n_e n_H) and Lambda / Gamma against psi, per T."""
    use_calima_style()
    h, wp, wt = (np.array(d["heat"][k]) for k in ("pahchem", "wd01_pah", "wd01_total"))
    psi, Ts = np.array(d["psi"]), d["T"]
    fig, axes = plt.subplots(3, len(Ts), figsize=(4.4 * len(Ts), 10), sharex=True, squeeze=False,
                             gridspec_kw={"hspace": 0.05, "wspace": 0.3})
    for c, T in enumerate(Ts):
        ne = np.sqrt(T) / psi
        fits = heating_fits(T, psi)
        series = [(r"pahchem, $N_{\rm C}$ 22-268", h[c], "#2a78d6", "-", 2.2),
                  (r"WD01 physics, $a \leq 8.3$ \AA", wp[c], "#eb6834", ":", 1.6),
                  ("WD01 physics, all sizes", wt[c], "#eb6834", "-", 1.8)]
        for lab, v, col, ls, lw in series:
            axes[0, c].loglog(psi, v[:, 0], color=col, ls=ls, lw=lw, label=lab)
            axes[1, c].loglog(psi, v[:, 1] / ne, color=col, ls=ls, lw=lw)
            axes[2, c].loglog(psi, v[:, 1] / v[:, 0], color=col, ls=ls, lw=lw)
        for (lab, (g, lam)), ls in zip(fits.items(), ("--", "-."), strict=True):
            axes[0, c].loglog(psi, g, color="k", ls=ls, lw=1.4, label=lab)
            if lab.startswith("WD01") and T < 1e3:                     # eq. 45 is fitted for 1e3 <= T <= 1e4 K
                continue
            axes[1, c].loglog(psi, lam, color="k", ls=ls, lw=1.4)
            axes[2, c].loglog(psi, lam * ne / g, color="k", ls=ls, lw=1.4)
        axes[0, c].set_title(rf"$T = {T:g}$ K", fontsize=14)
        axes[2, c].axhline(1.0, color="0.7", lw=1, zorder=0)
        axes[2, c].set_xlabel(r"$\psi = G\sqrt{T}/n_e$", fontsize=14)
        for r in range(3):
            _ticks(axes[r, c])
    axes[0, 0].set_ylabel(r"$\Gamma_{\rm PE}/(G n_{\rm H})$ [erg s$^{-1}$]", fontsize=14)
    axes[1, 0].set_ylabel(r"$\Lambda_{\rm rec}/(n_e n_{\rm H})$ [erg cm$^3$ s$^{-1}$]", fontsize=14)
    axes[2, 0].set_ylabel(r"$\Lambda_{\rm rec}/\Gamma_{\rm PE}$", fontsize=14)
    axes[0, 0].legend(frameon=False, fontsize=9, loc="lower left")
    axes[1, 0].annotate("WD01 eq. 45: fitted for $T \\geq 10^3$ K only", (0.03, 0.92), xycoords="axes fraction",
                        fontsize=9, color="0.35")
    fig.suptitle(r"Heating and recombination cooling, WD01 carbon budget ($b_{\rm C} = 6\times10^{-5}$), Mathis field",
                 fontsize=14, y=0.95)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def plot_charge(d, path):
    """<Z> of the pahchem bins against WD01's graphite grain of the bin's mean N_C, against psi, per T."""
    use_calima_style()
    psi, Ts = np.array(d["psi"]), d["T"]
    a_wd, z_wd = np.array(d["z"]["a_wd01"]), np.array(d["z"]["wd01"])          # (T, psi, size)
    colors = ("#2a78d6", "#1baf7a", "#eda100", "#8a5cc2")
    fig, axes = plt.subplots(1, len(Ts), figsize=(4.4 * len(Ts), 4.6), sharey=True, squeeze=False,
                             gridspec_kw={"wspace": 0.05})
    for c, T in enumerate(Ts):
        ax = axes[0, c]
        for b, zb, col in zip(d["bins"], d["z"]["pahchem"], colors[:len(d["bins"])], strict=True):
            lo, hi = b["nc"]
            a = float(radius_cm(b["mean_nc"]))
            zw = [np.interp(np.log(a), np.log(a_wd), z_wd[c, ip]) for ip in range(psi.size)]
            ax.semilogx(psi, np.array(zb)[c], color=col, lw=2.2,
                        label=rf"$N_{{\rm C}}$ {lo}-{hi} ($\langle N_{{\rm C}}\rangle$ {b['mean_nc']:.0f})")
            ax.semilogx(psi, zw, color=col, lw=1.4, ls="--")
        ax.plot([], [], color="0.4", ls="--", lw=1.4, label="WD01 physics, same carbon count")
        ax.axhline(0.0, color="0.7", lw=1, zorder=0)
        ax.set_title(rf"$T = {T:g}$ K", fontsize=14)
        ax.set_xlabel(r"$\psi = G\sqrt{T}/n_e$", fontsize=14)
        _ticks(ax)
    axes[0, 0].set_ylabel(r"$\langle Z \rangle$", fontsize=14)
    axes[0, 0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def plot(d, path):
    use_calima_style()
    pah, wp, wt, fit = (np.array(d[k]) for k in ("pahchem", "wd01_pah", "wd01_total", "fit"))
    psi, ions_plot = np.array(d["psi"]), ("H+", "C+", "Mg+", "S+", "Fe+")
    fig, axes = plt.subplots(len(ions_plot), len(d["T"]), figsize=(4.2 * len(d["T"]), 2.9 * len(ions_plot)),
                             sharex=True, squeeze=False)
    for r, ion in enumerate(ions_plot):
        k = d["ions"].index(ion)
        for c, T in enumerate(d["T"]):
            ax = axes[r, c]
            ax.loglog(psi, fit[c, :, k], "k--", lw=1.4, label="WD01b fit")
            ax.loglog(psi, wt[c, :, k], color="#eb6834", lw=1.8, label="WD01 physics, all sizes")
            ax.loglog(psi, wp[c, :, k], color="#eb6834", lw=1.4, ls=":", label=r"WD01 physics, $a \leq 8.3$ \AA")
            ax.loglog(psi, pah[c, :, k], color="#2a78d6", lw=1.8, label=r"pahchem, $N_{\rm C}$ 22-268")
            ax.minorticks_on()
            ax.tick_params(which="both", direction="in")
            if r == 0:
                ax.set_title(rf"$T = {T:g}$ K", fontsize=14)
            if c == 0:
                ax.set_ylabel(rf"$\alpha$({ion}) [cm$^3$ s$^{{-1}}$ H$^{{-1}}$]", fontsize=14)
            if r == len(ions_plot) - 1:
                ax.set_xlabel(r"$\psi = G\sqrt{T}/n_e$", fontsize=14)
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.subplots_adjust(hspace=0, wspace=0.3)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--detachment", default=Params().detachment, choices=("jochims", "statistical", "valence"))
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--out", type=Path, default=None, help="plot directory (default results/plots/pahchem)")
    args = ap.parse_args()
    js = _paths.get_results_dir() / ("pahchem_wd01b_alpha.json" if args.detachment == "jochims"
                                      else f"pahchem_wd01b_alpha_{args.detachment}.json")
    if not args.plot_only:
        d = compute(args.workers, args.detachment)
        js.parent.mkdir(parents=True, exist_ok=True)
        js.write_text(json.dumps(d))
    d = json.loads(js.read_text())
    report(d)
    out = args.out or _paths.get_plots_dir("pahchem")
    out.mkdir(parents=True, exist_ok=True)
    plot(d, out / "pahchem_wd01b_alpha.png")
    if "heat" in d:                                                     # runs from 2026-10-25 on
        report_heat(d)
        plot_heat(d, out / "pahchem_wd01_heating_cooling.png")
        plot_charge(d, out / "pahchem_wd01_mean_charge.png")


if __name__ == "__main__":
    main()
