#!/usr/bin/env python3
"""Hot-gas destruction by thermal electrons (gas-coupling section 2.2g) against Micelotta et al. (2010a, A&A 510, A36).

Three models of the electrons' carbon loss from PAHs of N_C ~ 50, 100, 200 at T = 1e4.5-1e8 K, per unit electron density:
- Micelotta+10a's published rate constants J_e(T): their Table 2 fits, g(T) = 10^f(log T), f a fifth-order polynomial, of
  the carbon-atom loss (the 2 C per event of their Eq. 27 included: only so do they give their Table 3, tau_0 = N_C /
  (J n_H) for M82, M17, N132D at N_C = 200). Their N_C = 50 electron fit is printed with a = +2136.83, which overflows;
  with -2136.83, as for N_C = 100 and 200, it gives their Fig. 7 curve (~1e-5 cm^3 s^-1 at 1e6 K);
- this repo's implementation of their model, ``models/PAH_gas_collisions/PAH_sputtering.electronic_destruction_rate_T``
  (the event rate: x 2 for the carbon);
- pahchem (``Params.hot_gas``): ``Population.destruction_rates`` for C54H18, C96H24 and C216H36 at n_e = 1 cm^-3, the carbon
  lost per second (bare-skeleton C loss and folding after the electrons strip the molecule, passages beyond 320 eV,
  Coulomb explosions), in gas whose re-hydrogenation is n(H) = 1e-4 n_e and whose field is G0 = 1 (Draine, no old stars),
  which Micelotta's PAHs do not have; their charge is the chain's, Micelotta's neutral.
Lifetimes with Micelotta's definition, tau_0 = N_C / (carbon-loss rate), at n_e = 1; for pahchem also 1 / (destruction
events): a bare skeleton of N_C >= 60 folds into a cage, which destroys the PAH but loses no carbon. Nothing is fitted.

Result (2026-11-04): Table 2 gives Table 3 to 4-10 % (M82 156 vs 162 yr, M17 7.9 vs 8.2, N132D 0.27 vs 0.3); the repo's
implementation runs 0.2-0.76x of the published rates. pahchem, N_C ~ 50: carbon loss 0.05-0.4x Micelotta above 3e5 K,
2.7x at 1e5 K; destroyed in 0.05-0.6 yr (1-4x sooner than Micelotta's carbon lifetime). N_C ~ 100-200: carbon loss
1e-3-1e-4x (the stripped skeletons fold into cages), destroyed 2-100x sooner (0.02-0.6 yr at n_e = 1). Far faster at 3e4 K.

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
from pycalima.pahchem import population as P
from pycalima.pahchem.baseline import Environment, Params
from pycalima.plotting_style import use_calima_style

#: Micelotta+10a Table 2, electrons: (a, b, c, d, e, f) of f(x) = a + b x + ... + f x^5, x = log10 T, J = 10^f [cm^3 s^-1]
TABLE2_ELECTRONS = {50: (-2136.83, 1632.17, -499.822, 76.4347, -5.82964, 0.177174),   # a printed +2136.83 (see above)
                    100: (-2255.38, 1681.45, -503.451, 75.4339, -5.64956, 0.168959),
                    200: (-2234.37, 1597.71, -459.647, 66.332, -4.79841, 0.139019)}
#: Micelotta+10a Table 3 (N_C = 200, electrons): object, n_H [cm^-3], T [K], tau_0(ref) [yr]
TABLE3 = (("M82", 0.013, 5.8e6, 162.0), ("M17", 0.3, 7e6, 8.2), ("N132D", 10.0, 8e6, 0.3))
YR = 3.15576e7
TEMPS = np.array([10 ** 4.5, 1e5, 10 ** 5.5, 1e6, 10 ** 6.5, 1e7, 10 ** 7.5, 1e8])
PAIRS = ((50, "C54H18 (37)"), (100, "C96H24 (108)"), (200, "C216H36 (615)"))


def micelotta_j(nc, t):
    """J_e(T) [cm^3 s^-1] of Micelotta+10a's Table 2 fit for N_C = nc."""
    x = np.log10(np.asarray(t, dtype=float))
    c = TABLE2_ELECTRONS[nc]
    return 10.0 ** sum(ci * x ** i for i, ci in enumerate(c))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    warnings.simplefilter("ignore")
    print("1. Micelotta+10a Table 3 from their Table 2 fit (N_C = 200, tau_0 = N_C / (J n_H)):")
    for name, nh, t, tau in TABLE3:
        print(f"   {name}: {200.0 / (micelotta_j(200, t) * nh) / YR:7.2f} yr against their {tau} yr")
    lib = [m for m in L.library() if m.name in dict((n, k) for k, n in PAIRS)]
    pop = P.Population(lib, {"all": (1.0, 1e4)}, params=replace(Params(), hot_gas=True), cache_dir=FP.BASIS_CACHE)
    res = {nc: {"mic": [], "repo": [], "new": [], "events": [], "z": []} for nc, _ in PAIRS}
    zs = np.array(pop.zs, float)
    for t in TEMPS:
        e = Environment("hot", 1.0, 1e-4, 1.0, float(t), None, optical_isrf=0.0, ions=(("H+", 1.0),))
        shape = P.field_shape(e)
        pop.prepare(shape, workers=a.workers)
        pop.prepare_responses(shape, workers=a.workers)
        pop.prepare_bare(shape, workers=a.workers)
        pop.prepare_hot(t, workers=a.workers)
        d = pop.destruction_rates(e)
        f = pop.charge_matrix(e)
        for nc, name in PAIRS:
            s = next(i for i, m in enumerate(pop.library) if m.name == name)
            res[nc]["mic"].append(micelotta_j(nc, t))
            res[nc]["repo"].append(2.0 * S.electronic_destruction_rate_T(float(t), "e", nc))
            res[nc]["new"].append(d["carbon"][s])
            res[nc]["events"].append(d["events"][s])
            res[nc]["z"].append(float(f[s] @ zs))
    print("\n2. Carbon atoms lost per molecule per unit n_e [cm^3 s^-1]: Micelotta+10a fit / repo PAH_sputtering / pahchem; "
          "pahchem <Z>; lifetimes N_C / rate at n_e = 1 [yr]")
    for nc, name in PAIRS:
        print(f"   N_C {nc} (pahchem {name}):")
        r = res[nc]
        for i, t in enumerate(TEMPS):
            life = [nc / x / YR if x > 0 else np.inf for x in (r["mic"][i], r["repo"][i], r["new"][i])]
            ev = 1.0 / r["events"][i] / YR if r["events"][i] > 0 else np.inf
            print(f"     T {t:.1e}: {r['mic'][i]:.2e} / {r['repo'][i]:.2e} / {r['new'][i]:.2e}  (new/Micelotta "
                  f"{r['new'][i] / r['mic'][i]:.2e}, repo/Micelotta {r['repo'][i] / r['mic'][i]:.2f})  <Z> {r['z'][i]:5.2f}  "
                  f"tau {life[0]:.1e} / {life[1]:.1e} / {life[2]:.1e}; pahchem 1/events {ev:.1e}")
    plot(res, (a.out or _paths.get_plots_dir("pahchem")) / "pahchem_hot_gas_micelotta.png")


def plot(res, path):
    use_calima_style()
    fig, axes = plt.subplots(1, len(res), figsize=(11.0, 3.8), sharey=True)
    styles = (("mic", "Micelotta+10a (Table 2 fit)", "#1f1f1f", "-", None),
              ("repo", "this repo's implementation", "#8a8a8a", ":", None),
              ("new", "pahchem: carbon lost", "#2a78d6", "--", "o"),
              ("events", "pahchem: destroyed (folded or C lost)", "#e07b39", "-.", "s"))
    for ax, (nc, r) in zip(axes, res.items()):
        for key, label, colour, ls, mk in styles:
            y = 1.0 / np.maximum(np.array(r[key]), 1e-300) / YR * (1.0 if key == "events" else nc)
            ax.plot(TEMPS, y, ls, color=colour, lw=2, marker=mk, ms=4, label=label)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$T$ [K]")
        ax.set_title(rf"$N_\mathrm{{C}} \approx {nc}$", fontsize=10)
        ax.minorticks_on()
        ax.tick_params(which="both", direction="in", top=True, right=True)
    axes[0].set_ylabel(r"lifetime at $n_e = 1\,\mathrm{cm^{-3}}$ [yr]")
    axes[0].legend(fontsize=7, frameon=False, loc="upper right")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"\nfigure: {path}")


if __name__ == "__main__":
    main()
