#!/usr/bin/env python3
"""Hot-gas destruction by thermal electrons and ions (gas-coupling section 2.2g) against Micelotta, Jones & Tielens (2010b,
A&A 510, A37, "PAH processing in a hot gas"; its companion on shocks, 2010a, A&A 510, A36, is MJT).

Models of the carbon loss from PAHs of N_C ~ 50, 100, 200 at T = 1e4.5-1e8 K:
- Micelotta+10b's published rate constants J(T), their Table 2 fits g(T) = 10^f(log T), f a fifth-order polynomial, for
  electrons and for H, He and C ions (nuclear plus electronic, at their abundances H : He : C = 1 : 0.1 : 1e-4, per n_H),
  of the carbon-atom loss (the 2 C per electronic event of their eq. 27 included: only so do they give their Table 3,
  tau_0 = N_C / (J n_H) for M82, M17, N132D at N_C = 200, and Orion from the ions). Their N_C = 50 electron fit is printed
  with a = +2136.83, which overflows; with -2136.83, as for N_C = 100 and 200, it gives their Fig. 7 curve;
- this repo's implementation of their model, ``models/PAH_gas_collisions/PAH_sputtering`` (electrons: x 2 for the carbon;
  nuclear: 0.5 N_C x the per-atom rate, as its exporter);
- ``pahchem.impact.thermal_ion_passages``: the ions' knock-outs with MJT's T_0 = 7.5 eV (C targets only, neutral PAH),
  without MJT's orientation factor 1/2, against the ion fits where the nuclear part dominates them (T <= 1e5 K);
- pahchem (``Params.hot_gas``): ``Population.destruction_rates`` for C54H18, C96H24 and C216H36 in gas of n_H = n_e =
  1 cm^-3 with Micelotta's ions (H+ 1, He+ 0.1, C+ 1e-4 cm^-3), the carbon lost per second (bare-skeleton C loss and
  folding after the gas strips the molecule, passages beyond 320 eV, Coulomb explosions, knock-outs of C at
  ``impact.KNOCKOUT_T0_EV``), with re-hydrogenation n(H) = 1e-4 n_e and field G0 = 1 (Draine, no old stars), which
  Micelotta's PAHs do not have; their charge is the chain's, Micelotta's neutral.
Lifetimes with Micelotta's definition, tau_0 = N_C / (carbon-loss rate); for pahchem also 1 / (destruction events): a
bare skeleton of N_C >= 60 folds into a cage, which destroys the PAH but loses no carbon. Nothing is fitted.

Result (2026-11-04):
- Table 2 gives Table 3 to 4-10 % for the electrons (M82 156 vs 162 yr, M17 7.9 vs 8.2, N132D 0.27 vs 0.3) and to 35 %
  for Orion (ions, 1.8e7 vs 1.3e7 yr).
- Knock-outs at MJT's T_0 are 1.7-2.8x their nuclear-dominated ion fits (3e4-1e5 K; H, He, C; N_C 50, 200): the factor
  1/2 of MJT eq. 22, which pahchem drops. The repo's nuclear rate puts T_0 at E where MJT eq. 15 has E_0n = T_0 / gamma:
  H x3000 at 3e4 K, x24 at 1e5 K.
- pahchem (step 3: a fast electron must lose its energy to be captured, ``population.capture_stop_factors``): above
  ~3e5 K the PAHs charge up to their ladder tops (C54H18 +10, C96H24 +19, C216H36 +32 from 1e6 K), and Coulomb
  explosions out of the top destroy them in 0.005-0.02 yr at n_e = 1 from 1e6 K (0.02-0.08 yr at 3e5 K); the carbon
  loss, a whole molecule per explosion, is 5-850x Micelotta's electrons + ions from 1e6 K. At 1e5 K (<Z> ~ 0) it is
  3x theirs for N_C 50, 0.02-0.2x for 100-200. Knock-outs (T_0 = 23.3 eV) are 0.3-0.8 of MJT's above 1e6 K and a small
  part of the carbon lost. Step 2's numbers, with every arriving electron captured, are in gas-coupling 2.2g.

    check_hot_gas_micelotta.py [--workers 4] [--out DIR]
"""
import argparse
import sys
import warnings
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import check_charge_ladder as L
import fit_pdrs4all_population as FP

from pycalima import _paths
from pycalima.models.PAH_gas_collisions import PAH_sputtering as S
from pycalima.pahchem import impact
from pycalima.pahchem import population as P
from pycalima.pahchem.baseline import Environment, Params
from pycalima.plotting_style import use_calima_style

#: Micelotta+10b Table 2: (a, b, c, d, e, f) of f(x) = a + b x + ... + f x^5, x = log10 T, J = 10^f [cm^3 s^-1] per n_H
TABLE2 = {50: {"e": (-2136.83, 1632.17, -499.822, 76.4347, -5.82964, 0.177174),   # a printed +2136.83 (see above)
               "H": (-1896.69, 1480.8, -462.733, 71.8957, -5.54719, 0.169996),
               "He": (-971.448, 770.259, -245.561, 38.8995, -3.05787, 0.0954303),
               "C": (-704.392, 551.313, -175.063, 27.6506, -2.16871, 0.0675643)},
          100: {"e": (-2255.38, 1681.45, -503.451, 75.4339, -5.64956, 0.168959),
                "H": (-1645.64, 1257.06, -384.309, 58.4103, -4.41007, 0.132356),
                "He": (-945.901, 747.921, -237.984, 37.6808, -2.96613, 0.0928852),
                "C": (-711.244, 558.48, -177.983, 28.2547, -2.23145, 0.0701374)},
          200: {"e": (-2234.37, 1597.71, -459.647, 66.332, -4.79841, 0.139019),
                "H": (-1473.64, 1109.01, -334.292, 50.1417, -3.74133, 0.111164),
                "He": (-963.188, 765.639, -245.054, 39.0745, -3.10123, 0.0980047),
                "C": (-738.791, 584.928, -187.884, 30.0819, -2.39748, 0.0760639)}}
#: Micelotta's ion abundances per H, and the ions of chemistry.IONS standing for them
CHI = {"H": 1.0, "He": 0.1, "C": 1e-4}
IONS = (("H+", 1.0), ("He+", 0.1), ("C+", 1e-4))
#: Micelotta+10b Table 3 (N_C = 200): object, n_H [cm^-3], T [K], tau_0(ref) [yr], the agent
TABLE3 = (("M82", 0.013, 5.8e6, 162.0, "e"), ("M17", 0.3, 7e6, 8.2, "e"), ("N132D", 10.0, 8e6, 0.3, "e"),
          ("Orion", 1e4, 7e3, 1.3e7, "ions"))
YR = 3.15576e7
TEMPS = np.array([10 ** 4.5, 1e5, 10 ** 5.5, 1e6, 10 ** 6.5, 1e7, 10 ** 7.5, 1e8])
PAIRS = ((50, "C54H18 (37)"), (100, "C96H24 (108)"), (200, "C216H36 (615)"))
MJT_T0 = {"C": 7.5}


def micelotta_j(nc, t, who="e"):
    """J(T) [cm^3 s^-1 per n_H] of Micelotta+10b's Table 2 fit for N_C = nc: "e", an ion, or "ions" (their sum)."""
    if who == "ions":
        return sum(micelotta_j(nc, t, x) for x in CHI)
    x = np.log10(np.asarray(t, dtype=float))
    return 10.0 ** sum(ci * x ** i for i, ci in enumerate(TABLE2[nc][who]))


def repo_nuclear(nc, el, t):
    """The repo's nuclear rate per ion of ``el`` (PAH_sputtering: 0.5 N_C x the per-atom rate, T_0 = 7.5 eV)."""
    z1, m1 = impact.ATOMS[el]
    return 0.5 * nc * S.nuclear_destruction_rate_T(float(t), m1, 12.0107, z1, 6, 7.5, m1 * 1.66053906660e-24, nbins_v=400)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    warnings.simplefilter("ignore")
    print("1. Micelotta+10b Table 3 from their Table 2 fits (N_C = 200, tau_0 = N_C / (J n_H)):")
    for name, nh, t, tau, who in TABLE3:
        print(f"   {name} ({who}): {200.0 / (micelotta_j(200, t, who) * nh) / YR:9.3g} yr against their {tau:g} yr")
    print("\n2. Knock-outs per ion [cm^3 s^-1] at MJT's T_0 = 7.5 eV, neutral PAH: pahchem.impact (no 1/2) / Micelotta "
          "fit / chi / repo PAH_sputtering (1/2, T_0 at E)")
    for nc in (50, 200):
        for el in CHI:
            row = []
            for t in (10 ** 4.5, 1e5, 1e6):
                ko = impact.thermal_ion_passages(nc, el, t, t0_ev=MJT_T0)[1]["C"]
                mic = micelotta_j(nc, t, el) / CHI[el]
                row.append(f"T {t:.0e}: {ko:.2e} / {mic:.2e} (x{ko / mic:.2f}) / repo {repo_nuclear(nc, el, t):.2e}")
            print(f"   N_C {nc} {el:2s}: " + ";  ".join(row))
    lib = [m for m in L.library() if m.name in dict((n, k) for k, n in PAIRS)]
    pop = P.Population(lib, {"all": (1.0, 1e4)}, params=replace(Params(), hot_gas=True), cache_dir=FP.BASIS_CACHE)
    keys = ("mic", "mic_all", "repo", "new", "events", "z", "ko", "ko_mjt")
    res = {nc: {k: [] for k in keys} for nc, _ in PAIRS}
    zs = np.array(pop.zs, float)
    for t in TEMPS:
        e = Environment("hot", 1.0, 1e-4, 1.0, float(t), None, optical_isrf=0.0, ions=IONS)
        shape = P.field_shape(e)
        pop.prepare(shape, workers=a.workers)
        pop.prepare_responses(shape, workers=a.workers)
        pop.prepare_bare(shape, workers=a.workers)
        pop.prepare_hot(t, workers=a.workers, ions=tuple(dict(IONS)))
        d = pop.destruction_rates(e)
        f = pop.charge_matrix(e)
        for nc, name in PAIRS:
            s = next(i for i, m in enumerate(pop.library) if m.name == name)
            r = res[nc]
            r["mic"].append(micelotta_j(nc, t))
            r["mic_all"].append(micelotta_j(nc, t) + micelotta_j(nc, t, "ions"))
            r["repo"].append(2.0 * S.electronic_destruction_rate_T(float(t), "e", nc))
            r["new"].append(d["carbon"][s])
            r["events"].append(d["events"][s])
            r["z"].append(float(f[s] @ zs))
            r["ko"].append(d["knockout"][s])
            r["ko_mjt"].append(0.5 * sum(CHI[el] * impact.thermal_ion_passages(nc, el, t, t0_ev=MJT_T0)[1]["C"]
                                         for el in CHI))
    print("\n3. Carbon atoms lost per molecule at n_H = n_e = 1 cm^-3 [s^-1]: Micelotta+10b electrons / electrons + ions / "
          "repo electrons / pahchem (electrons + ions); pahchem <Z>, its knock-outs; lifetimes N_C / rate [yr]")
    for nc, name in PAIRS:
        print(f"   N_C {nc} (pahchem {name}):")
        r = res[nc]
        for i, t in enumerate(TEMPS):
            life = [nc / x / YR if x > 0 else np.inf for x in (r["mic"][i], r["mic_all"][i], r["new"][i])]
            ev = 1.0 / r["events"][i] / YR if r["events"][i] > 0 else np.inf
            print(f"     T {t:.1e}: {r['mic'][i]:.2e} / {r['mic_all'][i]:.2e} / {r['repo'][i]:.2e} / {r['new'][i]:.2e}  "
                  f"(pahchem / Micelotta all {r['new'][i] / r['mic_all'][i]:.2e})  <Z> {r['z'][i]:5.2f}  "
                  f"knock-outs {r['ko'][i]:.1e} (MJT's {r['ko_mjt'][i]:.1e})  tau {life[0]:.1e} / {life[1]:.1e} / "
                  f"{life[2]:.1e}; pahchem 1/events {ev:.1e}")
    plot(res, (a.out or _paths.get_plots_dir("pahchem")) / "pahchem_hot_gas_micelotta.png")


def plot(res, path):
    use_calima_style()
    fig, axes = plt.subplots(1, len(res) + 1, figsize=(14.0, 3.8))
    styles = (("mic", "Micelotta+10b: electrons", "#1f1f1f", "-", None),
              ("mic_all", "Micelotta+10b: electrons + ions", "#1f1f1f", "--", None),
              ("repo", "this repo's implementation (electrons)", "#8a8a8a", ":", None),
              ("new", "pahchem: carbon lost", "#2a78d6", "--", "o"),
              ("events", "pahchem: destroyed (folded or C lost)", "#e07b39", "-.", "s"))
    for ax, (nc, r) in zip(axes, res.items()):
        for key, label, colour, ls, mk in styles:
            y = 1.0 / np.maximum(np.array(r[key]), 1e-300) / YR * (1.0 if key == "events" else nc)
            ax.plot(TEMPS, y, ls, color=colour, lw=2, marker=mk, ms=4, label=label)
        ax.set_title(rf"$N_\mathrm{{C}} \approx {nc}$", fontsize=10)
    axes[0].set_ylabel(r"lifetime at $n_\mathrm{H} = n_e = 1\,\mathrm{cm^{-3}}$ [yr]")
    axes[0].legend(fontsize=7, frameon=False, loc="upper right")
    for ax in axes[1:len(res)]:
        ax.sharey(axes[0])
    ax = axes[-1]
    r = res[50]
    ax.plot(TEMPS, micelotta_j(50, TEMPS, "ions"), "-", color="#1f1f1f", lw=2, label="Micelotta+10b: ions (fit)")
    ax.plot(TEMPS, r["ko_mjt"], ":", color="#8a8a8a", lw=2, label=r"knock-outs, MJT ($T_0 = 7.5$ eV, $\times 1/2$)")
    ax.plot(TEMPS, np.maximum(r["ko"], 1e-300), "--", color="#2a78d6", lw=2, marker="o", ms=4,
            label=rf"pahchem knock-outs ($T_0 = {impact.KNOCKOUT_T0_EV['C']}$ eV)")
    ax.set_ylim(1e-20, 1e-5)
    ax.yaxis.set_label_position("right")
    ax.set_ylabel(r"C atoms lost per $n_\mathrm{H}$ [cm$^3$ s$^{-1}$]")
    ax.set_title(r"ions, $N_\mathrm{C} \approx 50$", fontsize=10)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$T$ [K]")
        ax.minorticks_on()
        ax.tick_params(which="both", direction="in", top=True, right=True)
    axes[-1].tick_params(which="both", labelleft=False, labelright=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"\nfigure: {path}")


if __name__ == "__main__":
    main()
