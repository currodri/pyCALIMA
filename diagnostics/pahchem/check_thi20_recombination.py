#!/usr/bin/env python3
"""Electron recombination on C54H18^Z+: pahchem against Thi+20 Table B.3 (validation V10).

Thi+20 (A&A 634, A42, App. B) use one circumcoronene, a = 10 A (N_C/468)^(1/3) = 4.69 A, with
k(e + PAH+) = 4.1e-5 phi f(a) (N_C/50)^(1/2) (100/T)^(1/2), phi = 0.4 for C54 (Table B.3: 1.7e-5 (100/T)^(1/2)),
times (1 + W/kT) for multiply charged ions, W = IP(Z-1) - IP(0): (1 + 36438.8/T) for 2+, (1 + 72877.6/T) for 3+.
pahchem is ``population.capture_coefficient`` for v4.00 "C54H18 (37)" with the calibrated thresholds
(``shift_thresholds``, -1: 0.37, 0: 0.6, 1: 0.64, 2: 0.64 eV; capture does not depend on them): BT94/DS87,
0.82 pi a^2 v_e J~(tau, -Z), a = 0.9 sqrt(N_C) = 6.61 A, sticking 1 (``Params()``), and the capped lab trend
(``Params(capture_model="lab_trend")``). Nothing is tuned.

Their Fig. B.1 compares the formula at phi = 0.2 with the FALP data; its numbers are not in the literature page,
so the same comparison is redone at 300 K against pahchem's FALP compilation
(data/external_data/pahchem/pah_electron_recombination.csv), with f(a) = (1 - exp(-12 a / l_e)) /
(1 + exp(12 - N_C)), l_e = 10 A, as transcribed in pahchem-gas-coupling-literature.md §11.

Result (2026-10-24):
- 1+: pahchem / Thi = 2.08-2.10 at 10-300 K, 2.15 (1e3 K), 2.29 (3e3 K), 2.78 (1e4 K); lab_trend 1.28-1.71.
  That is an effective phi of 0.83 (0.51 with lab_trend; Thi quote 0.1-0.8): x1.41 from the radius (6.61 against
  4.69 A) and the rest from 0.82 x the DS87 image term. PASS within the x2-2.5 of the image-form uncertainty
  (R1) up to 3e3 K; the rise at 1e4 K is DS87's geometric term, which Thi's pure T^-1/2 omits.
- 2+, 3+: pahchem / Thi = 1e-3 (10 K), 0.01 (100 K), 0.03 (300 K), 0.9 (1e4 K) for 2+; 7e-4-0.67 for 3+.
  Thi's (1 + W/kT) multiplies a rate that is already Coulomb-focused: their 2+ value is 195x (100 K) to 2e3x
  (10 K) the Spitzer limit pi a^2 v_e (1 + Z e^2/(a kT)) at their own radius. pahchem's k(2+)/k(1+) = 1.71 and
  k(3+)/k(1+) = 2.37 at 10 K, the DS87 tau -> 0 limits (2 (1 + 2^-1/2)/2 = 1.71). FAIL as a match, attributed
  to Thi's multiply-charged scaling, not to pahchem.
- Fig. B.1 redone (FALP, 300 K): Thi with phi = 0.2 gives 0.22-2.3 of the data for N_C 10-16 (0.004 for
  benzene, cut by their f(a)); pahchem bt94 2.7-29, lab_trend 0.46-3.3.

    check_thi20_recombination.py [--out DIR]
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pycalima import _paths
from pycalima.pahchem import chemistry
from pycalima.pahchem import population as P
from pycalima.pahchem.baseline import Params
from pycalima.plotting_style import use_calima_style

SHIFTS = P.CALIBRATED_SHIFTS
T_TAB = (10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 1e4)
T_PLOT = np.geomspace(10.0, 1e4, 40)
W_K = {1: 0.0, 2: 36438.8, 3: 72877.6}                    # Table B.3: W / k [K]
E2 = 4.80320471e-10 ** 2
KB = 1.380649e-16


def thi20(z, temperature_k, phi=0.4, nc=54):
    """Thi+20 e + PAH^Z+ [cm^3 s^-1]; at N_C = 54, phi = 0.4 their Table B.3 (f(a) = 1)."""
    t = np.asarray(temperature_k, dtype=float)
    base = 1.7e-5 * (phi / 0.4) * (100.0 / t) ** 0.5 if nc == 54 else 4.1e-5 * phi * f_a(nc) * (nc / 50.0) ** 0.5 * (100.0 / t) ** 0.5
    return base * (1.0 + W_K[z] / t)


def f_a(nc, l_e=10.0):
    a = 10.0 * (nc / 468.0) ** (1.0 / 3.0)
    return (1.0 - np.exp(-12.0 * a / l_e)) / (1.0 + np.exp(12.0 - nc))


def focused_limit(z, temperature_k, a_cm):
    """pi a^2 v_e (1 + Z e^2 / (a kT)), the Spitzer form at Thi's radius."""
    t = np.asarray(temperature_k, dtype=float)
    return np.pi * a_cm**2 * chemistry.thermal_speed(t, chemistry.ME_G) * (1.0 + z * E2 / (a_cm * KB * t))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=None, help="plot directory (default results/plots/pahchem)")
    args = ap.parse_args()
    lib = P.library_v400(min_nc=54, max_nc=54)
    mol = P.shift_thresholds([m for m in lib if m.name == "C54H18 (37)"], SHIFTS)[0]
    print(f"{mol.name}: charges {mol.charges}, thresholds {({z: round(v, 2) for z, v in mol.ionisation_ev.items()})} eV")
    models = {"bt94": Params(capture_model="bt94"), "lab_trend": Params(capture_model="lab_trend")}
    a_thi = 1e-7 * (54 / 468.0) ** (1.0 / 3.0)
    print("\nV10: e + C54H18^Z+ [cm^3 s^-1]; Thi+20 Table B.3 (phi = 0.4)")
    print("  Z       T     Thi+20   pahchem bt94  lab_trend   bt94/Thi  lab_trend/Thi   Spitzer limit at a_Thi  "
          "Thi/limit")
    for z in (1, 2, 3):
        for T in T_TAB:
            k_t = float(thi20(z, T))
            kb, kl = (P.capture_coefficient(mol, z, T, p) for p in models.values())
            lim = float(focused_limit(z, T, a_thi))
            print(f"  {z}  {T:7.0f}   {k_t:.3e}   {kb:.3e}   {kl:.3e}   {kb / k_t:8.3g}   {kl / k_t:12.3g}   "
                  f"{lim:21.3e}   {k_t / lim:9.3g}")
    print("\n  pahchem k(Z)/k(1+) (bt94):  T      2+      3+")
    for T in T_TAB:
        k1 = P.capture_coefficient(mol, 1, T)
        print(f"                       {T:7.0f}  {P.capture_coefficient(mol, 2, T) / k1:6.2f}  "
              f"{P.capture_coefficient(mol, 3, T) / k1:6.2f}")

    lab = pd.read_csv(_paths.get_external_data_path("pahchem/pah_electron_recombination.csv"), comment="#")
    print("\nFig. B.1 redone at 300 K: FALP against Thi+20 (phi = 0.2, 0.4) and pahchem")
    print("  species        N_C   k_lab     Thi 0.2   Thi 0.4   bt94      lab_trend | Thi0.2/lab  bt94/lab  "
          "lab_trend/lab")
    falp = []
    for _, r in lab.iterrows():
        nc = int(r.n_c)
        t2, t4 = (4.1e-5 * phi * f_a(nc) * (nc / 50.0) ** 0.5 * (100.0 / 300.0) ** 0.5 for phi in (0.2, 0.4))
        kb = chemistry.electron_capture(nc, 1, 300.0)
        kl = kb * min(float(chemistry.lab_recombination_factor(nc, P.CAPTURE_MAX)), P.CAPTURE_MAX)
        falp.append((nc, r.k_cm3_s, r.err_frac, t2, t4, kb, kl))
        print(f"  {r.species:13s} {nc:4d}   {r.k_cm3_s:.2e}  {t2:.2e}  {t4:.2e}  {kb:.2e}  {kl:.2e} | "
              f"{t2 / r.k_cm3_s:10.3g}  {kb / r.k_cm3_s:8.3g}  {kl / r.k_cm3_s:13.3g}")
    out = args.out or _paths.get_plots_dir("pahchem")
    out.mkdir(parents=True, exist_ok=True)
    plot(mol, models, np.array(falp), out / "pahchem_thi20_recombination.png")


def plot(mol, models, falp, path):
    use_calima_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    ax = axes[0]
    colors = {1: "#2a78d6", 2: "#eb6834", 3: "#3a9e5f"}
    for z, c in colors.items():
        ax.loglog(T_PLOT, thi20(z, T_PLOT), color=c, ls="--", lw=1.4, label=rf"Thi+20, $Z = {z}$")
        ax.loglog(T_PLOT, [P.capture_coefficient(mol, z, t, models["bt94"]) for t in T_PLOT], color=c, lw=1.8,
                  label=rf"pahchem, $Z = {z}$")
        ax.loglog(T_PLOT, [P.capture_coefficient(mol, z, t, models["lab_trend"]) for t in T_PLOT], color=c, lw=1.2,
                  ls=":", label=rf"pahchem lab trend, $Z = {z}$")
    ax.set_xlabel(r"$T$ [K]", fontsize=14)
    ax.set_ylabel(r"$k(e + {\rm C_{54}H_{18}}^{Z+})$ [cm$^3$ s$^{-1}$]", fontsize=14)
    ax.legend(frameon=False, fontsize=7, ncol=1)
    ax = axes[1]
    nc = np.arange(6, 61)
    ax.errorbar(falp[:, 0], falp[:, 1], yerr=falp[:, 1] * falp[:, 2], fmt="ks", ms=5, label="FALP/FlAPI, 300 K")
    for phi, ls in ((0.2, "--"), (0.4, ":")):
        ax.loglog(nc, 4.1e-5 * phi * f_a(nc) * (nc / 50.0) ** 0.5 * (1 / 3.0) ** 0.5, "k", ls=ls, lw=1.3,
                  label=rf"Thi+20, $\phi = {phi}$")
    kb = np.array([chemistry.electron_capture(n, 1, 300.0) for n in nc])
    ax.loglog(nc, kb, color="#2a78d6", lw=1.8, label="pahchem bt94")
    ax.loglog(nc, kb * np.minimum(chemistry.lab_recombination_factor(nc, P.CAPTURE_MAX), P.CAPTURE_MAX),
              color="#eb6834", lw=1.8, label="pahchem lab trend")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-8, 1e-4)
    ax.set_xlabel(r"$N_{\rm C}$", fontsize=14)
    ax.set_ylabel(r"$k(e + {\rm PAH}^+)$ at 300 K [cm$^3$ s$^{-1}$]", fontsize=14)
    ax.legend(frameon=False, fontsize=7)
    for ax in axes:
        ax.minorticks_on()
        ax.tick_params(which="both", direction="in")
    fig.subplots_adjust(wspace=0.3)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    main()
