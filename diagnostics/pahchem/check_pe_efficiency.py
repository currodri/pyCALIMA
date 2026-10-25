#!/usr/bin/env python3
"""PAH photoelectric heating efficiency eps(gamma) against BT94, Tielens (2021) and Berné+22 (validation V9).

pahchem's ``Population.photoelectric_heating`` for v4.00 "C54H18 (37)" and "C96H24 (108)" (calibrated IP shifts,
Draine field at G0 = 1.7 Habing with the solar-circle old stars below 5 eV, gamma = G0 sqrt(T) / n_e set by n_e, no ions, no associative detachment, as in
the references' balances), with three anion detachments: Jochims (the default before 2026-10-25: the Jochims
yield above the EA, every electron taking E - EA), statistical (prompt plus delayed, thermionic electrons) and
valence (statistical plus valence photodetachment; the default since 2026-10-25). Its eps is the heating over all the power absorbed (1-13.6 eV since 2026-10-24). Against:

- BT94 eq. 43, eps = 4.87e-2 / (1 + 4e-3 gamma^0.73) + 3.65e-2 (T/1e4)^0.7 / (1 + 2e-4 gamma), for the BT94 size
  distribution (N_C ~ 50 to 1000 A grains) and relative to the 6-13.6 eV absorption;
- Tielens (2021) eq. 12.26, 0.06 / (1 + 7e-5 gamma), for N_C ~ 50; its 7e-5 rests on eq. 8.106, which is 0.46 x
  pahchem's capture, so with pahchem's capture it would be ~3.2e-5 (both are drawn). His examples: f(0) ~ 0.85,
  eps ~ 0.05 for T = 100 K, n_e = 0.008, G0 = 1; f(0) ~ 0.13, eps ~ 0.008 at gamma ~ 1e5;
- Berné+22 Table 4 for the diffuse ISM (N_C = 54, T = 80 K, n_e = 3e-2, Draine field, f_C = 0.1, C/H = 2.7e-4):
  Gamma = 2.7e-26 erg s^-1 H^-1 and eps = 0.037. Their eps is over all the absorbed power (down to 0.6 eV), so
  Gamma is the number to compare; and their charge fractions at gamma = 1e5, {f++, f+, f0, f-} = {0.5, 0.4, 0.1, 0}.

The ingredient-by-ingredient match to Berné+22's own code is ``check_berne22_pe_heating.py``.

Result (2026-10-24, 1 eV grid, Draine plus the solar-circle old stars below 5 eV, eps over all absorption):
- gamma >= 1e5 (anions gone): both detachments agree, eps 0.0100 for C54H18 at 1e5 against 0.0143 for eq. 12.26
  with pahchem's capture (0.0139 on the old 5-13.6 eV grid and normalisation) and 0.0075 for eq. 12.26; 3-4x BT94;
- gamma <= 1e3, Jochims detachment: eps = 0.15-0.40, 3-9x BT94 and eq. 12.26; Berné+22's diffuse Gamma
  is exceeded 5.8x (f(-) = 0.37 there);
- with the statistical detachment: C54H18 eps = 0.028-0.034; Berné+22's diffuse point gives Gamma 0.90x
  (2.4e-26) and eps 0.032 against their 0.037 (same normalisation now). C96H24 (detachment threshold 2.14 eV)
  is 81 % anionic in the diffuse cloud (23 % before the old stars), with eps = 0.006-0.012 at gamma <= 1e3;
- at gamma = 1e5 C54H18 is less ionised than Berné+22's {0.5, 0.4, 0.1}, which are for T_eff = 3e4 K:
  {f++, f+, f0} = {0.31, 0.51, 0.17} in the Draine field.

With the valence channel (``detachment = "valence"``, 2026-10-25: statistical plus the prompt removal of a valence
electron above ``population.anion_valence_threshold``, 3.75 eV for C54H18, 3.3-3.9 eV for C96H24-like):
- Tielens' diffuse example (gamma 1.25e3): eps 0.045 (C54H18) and 0.052 (C96H24) against his ~0.05, f(-1) 0.16 and
  0.27 (statistical: 0.030 and 0.012, C96H24 0.81 anionic);
- eps = 0.10-0.12 at gamma = 1e2, 2-3x BT94 and 1.7-2x eq. 12.26: where anions dominate, each detachment leaves from
  a threshold below the neutral IP and heats more (as WD01's negative grains);
- Berne+22's diffuse point: Gamma 1.86x, eps 0.067 (f(-1) 0.32; theirs has no anions: coronene's attachment law);
- gamma >= 1e5: the same as the other two.

    check_pe_efficiency.py [--workers 4] [--out DIR]
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pycalima import _paths
from pycalima.pahchem import population
from pycalima.pahchem.baseline import Environment, Params
from pycalima.plotting_style import use_calima_style

IP_SHIFTS = {-1: 0.37, 0: 0.6, 1: 0.64, 2: 0.64}
G0 = 1.7
GAMMAS = np.logspace(2.0, 6.0, 17)
DETACHMENT = ("jochims", "statistical", "valence")
TS = (100.0, 1000.0)
BASIS_CACHE = _paths.get_model_data_dir() / "pahchem_population_basis"


def bt94(gamma, t):
    return 4.87e-2 / (1.0 + 4e-3 * gamma**0.73) + 3.65e-2 * (t / 1e4) ** 0.7 / (1.0 + 2e-4 * gamma)


def tielens(gamma, k=7e-5):
    return 0.06 / (1.0 + k * gamma)


def at(pop, g0, t, ne):
    """(eps[s], Gamma[s] [erg s^-1 per molecule], f[s, slot]) at one point."""
    env = Environment("eps", g0, 0.0, ne, t, None)
    gamma, _, eps = pop.photoelectric_heating(env)
    return eps, gamma, pop.charge_matrix(env)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=None, help="plot directory (default results/plots/pahchem)")
    args = ap.parse_args()
    lib = population.shift_thresholds(population.library_v400(), IP_SHIFTS)
    mols = [m for m in lib if m.name in ("C54H18 (37)", "C96H24 (108)")]
    pops = {d: population.Population(mols, {"all": (1.0, 1e4)}, params=Params(detachment=d), cache_dir=BASIS_CACHE)
            for d in DETACHMENT}
    for pop in pops.values():
        for g0 in (G0, 1.0):                  # the Draine cells' shapes (old stars per unit G0, population.field_shape)
            pop.prepare(population.field_shape(Environment("eps", g0, 0.0, 1.0, 100.0, None)), workers=args.workers)
    zs = pops["jochims"].zs
    z0, zm = zs.index(0), zs.index(-1)

    eps = {(d, t): np.array([at(pop, G0, t, G0 * np.sqrt(t) / g)[0] for g in GAMMAS])
           for d, pop in pops.items() for t in TS}
    for t in TS:
        print(f"\nT = {t:g} K: eps(gamma); pahchem with Jochims (J), statistical (S) and valence (V) detachment")
        print("     gamma  C54H18 J  C54H18 S  C54H18 V  C96H24 J  C96H24 S  C96H24 V     BT94  Tielens  Tielens(3.2e-5)")
        for i, g in enumerate(GAMMAS):
            row = [eps[(d, t)][i][s] for s in (0, 1) for d in DETACHMENT]
            print(f"  {g:8.1e}  " + "  ".join(f"{v:8.4f}" for v in row) + f"   {bt94(g, t):.4f}   "
                  f"{tielens(g):.4f}   {tielens(g, 3.2e-5):.4f}")

    print("\nTielens (2021) examples")
    for label, g0, t, ne, ref in (("diffuse cloud", 1.0, 100.0, 0.008, "f(0) ~ 0.85, eps ~ 0.05"),
                                  ("dense PDR", G0, 100.0, G0 * 10.0 / 1e5, "f(0) ~ 0.13, eps ~ 0.008")):
        print(f"  {label}, gamma {g0 * np.sqrt(t) / ne:.2e}: Tielens {ref}")
        for d, pop in pops.items():
            e, _, f = at(pop, g0, t, ne)
            print(f"    {d:11s} C54H18 f(-1, 0) {f[0, zm]:.2f}, {f[0, z0]:.2f}, eps {e[0]:.4f}; "
                  f"C96H24 f(-1, 0) {f[1, zm]:.2f}, {f[1, z0]:.2f}, eps {e[1]:.4f}")

    per_h = 0.1 * 2.7e-4 / mols[0].nc
    print(f"\nBerne+22 diffuse ISM (gamma {G0 * np.sqrt(80.0) / 3e-2:.0f}): Gamma 2.7e-26 erg s^-1 H^-1, eps 0.037 "
          "(over all absorption)")
    for d, pop in pops.items():
        e, g, f = at(pop, G0, 80.0, 3e-2)
        print(f"    {d:11s} C54H18 Gamma {g[0] * per_h:.2e} ({g[0] * per_h / 2.7e-26:.2f}x), eps {e[0]:.4f} "
              f"(over all absorption), f(-1, 0) {f[0, zm]:.2f}, {f[0, z0]:.2f}")
    for t in TS:
        _, _, f = at(pops["jochims"], G0, t, G0 * np.sqrt(t) / 1e5)
        fz = {z: f[0, zs.index(z)] for z in (2, 1, 0, -1)}
        print(f"  gamma 1e5, T = {t:g} K: Berne {{f++, f+, f0, f-}} = {{0.5, 0.4, 0.1, 0}}; pahchem C54H18 "
              f"{{{', '.join(f'{v:.2f}' for v in fz.values())}}}, f(+3) {f[0, zs.index(3)]:.2f}")

    use_calima_style()
    fig, axes = plt.subplots(1, len(TS), figsize=(5.0 * len(TS), 4.2), sharey=True)
    for ax, t in zip(axes, TS, strict=True):
        for d, c in zip(DETACHMENT, ("#2a78d6", "#eb6834", "#1baf7a"), strict=True):
            ax.loglog(GAMMAS, eps[(d, t)][:, 0], color=c, lw=1.8, label=rf"pahchem C$_{{54}}$H$_{{18}}$, {d}")
            ax.loglog(GAMMAS, eps[(d, t)][:, 1], color=c, lw=1.4, ls="--", label=rf"pahchem C$_{{96}}$H$_{{24}}$, {d}")
        ax.loglog(GAMMAS, bt94(GAMMAS, t), "k-", lw=1.2, label="BT94 eq. 43")
        ax.loglog(GAMMAS, tielens(GAMMAS), "k:", lw=1.4, label="Tielens (2021) eq. 12.26")
        ax.loglog(GAMMAS, tielens(GAMMAS, 3.2e-5), color="0.5", ls="-.", lw=1.2,
                  label=r"eq. 12.26, $3.2\times10^{-5}\gamma$")
        ax.set_title(rf"$T = {t:g}$ K", fontsize=14)
        ax.set_xlabel(r"$\gamma = G_0\sqrt{T}/n_e$", fontsize=14)
        ax.minorticks_on()
        ax.tick_params(which="both", direction="in")
    axes[0].set_ylabel(r"$\epsilon$", fontsize=14)
    axes[0].legend(frameon=False, fontsize=9)
    fig.subplots_adjust(wspace=0)
    out = args.out or _paths.get_plots_dir("pahchem")
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "pahchem_pe_efficiency.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out / "pahchem_pe_efficiency.png")


if __name__ == "__main__":
    main()
