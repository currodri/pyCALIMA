#!/usr/bin/env python3
"""The charge of PAHs in hot gas (gas-coupling section 2.2g, step 3): fast electrons must lose their energy to be captured.

``Params.hot_gas``. The capture of an electron on a PAH is bounded by the chance that it loses its energy at infinity in
the molecule (Weingartner & Draine 2001, Sect. 3.1), ``population.capture_stopping`` from ``impact``'s collisions, above
the ~20-26 eV peak of that chance (``capture_stop_factors``; measured sticking below). Ions take electrons from cations
where that is exothermic (``chemistry.ion_cation_transfer``). Gas of n_H = n_e = 1 cm^-3 with Micelotta's ions (H+ 1,
He+ 0.1, C+ 1e-4), n(H) = 1e-4, G0 = 1 (Draine). Shown:
1. the factor on the capture coefficient of C24H12, C54H18, C96H24, C216H36 (Z = 1) against T;
2. <Z> of C54H18 and C216H36 with the bound and without it (the capture of step 2, every arriving electron at any
   energy), against the repo's WDB06 charging of graphite grains of the same radii (``dust_charge``: Draine & Salpeter
   1979 range transmission, secondary electrons), whose charge is capped by ion field emission.

Result (2026-11-04): the factor is 1 at 1e4 K, 0.85-0.9 at 1e5 K, 0.1-0.13 at 1e6 K, ~2e-3 at 1e7 K, ~2e-5 at 1e8 K.
With it the PAHs charge up to their ladder tops above 1e6 K (C54H18 +9-10, C216H36 +27-32), as the WDB06 grains reach
their field-emission caps (+4, +8); without it <Z> stays at 0-2 and goes negative for C216H36 above 1e7 K. At 1e5 and
3.2e5 K pahchem and the grains agree to ~1-3 charges.

    check_hot_gas_charge.py [--workers 4] [--out DIR]
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
from pycalima.models.dust_charge.dust_charging import equilibrium_charge_for_grain
from pycalima.pahchem import impact
from pycalima.pahchem import population as P
from pycalima.pahchem.baseline import Environment, Params
from pycalima.plotting_style import use_calima_style

IONS = (("H+", 1.0), ("He+", 0.1), ("C+", 1e-4))
TEMPS = np.array([1e4, 10 ** 4.5, 1e5, 10 ** 5.5, 1e6, 10 ** 6.5, 1e7, 1e8])
FACTOR_NAMES = ("C24H12 (18)", "C54H18 (37)", "C96H24 (108)", "C216H36 (615)")
CHARGE_NAMES = ("C54H18 (37)", "C216H36 (615)")


def charges(pop, workers):
    zs = np.array(pop.zs, float)
    out = {m.name: [] for m in pop.library}
    out["top"] = {m.name: max(m.charges) for m in pop.library}
    for t in TEMPS:
        e = Environment("hot", 1.0, 1e-4, 1.0, float(t), None, optical_isrf=0.0, ions=IONS)
        shape = P.field_shape(e)
        pop.prepare(shape, workers=workers)
        pop.prepare_responses(shape, workers=workers)
        pop.prepare_hot(t, workers=workers, ions=tuple(dict(IONS)))
        f = pop.charge_matrix(e)
        for s, m in enumerate(pop.library):
            out[m.name].append(float(f[s] @ zs))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    warnings.simplefilter("ignore")
    lib = {m.name: m for m in L.library() if m.name in FACTOR_NAMES}
    p = replace(Params(), hot_gas=True)
    print("1. Factor on the capture coefficient (Z = 1) in hot gas:")
    factors = {}
    for name in FACTOR_NAMES:
        ps = P.capture_stopping(lib[name], 1)
        factors[name] = [P.capture_stop_factors(lib[name], 1, t, p, ps)[0] for t in TEMPS]
        print(f"   {name}: P_stop peak {ps.max():.2f} at {P.CAPTURE_STOP_EV[np.argmax(ps)]:.0f} eV; "
              + "  ".join(f"{t:.1e}: {x:.3g}" for t, x in zip(TEMPS, factors[name])))
    mols = [lib[n] for n in CHARGE_NAMES]
    bound = charges(P.Population(mols, {"all": (1.0, 1e4)}, params=p, cache_dir=FP.BASIS_CACHE), a.workers)
    keep = P.capture_stop_factors
    P.capture_stop_factors = lambda *args: (1.0, 1.0)                     # step 2's capture, for comparison
    try:
        free = charges(P.Population([replace(m, _cache={}) for m in mols], {"all": (1.0, 1e4)}, params=p,
                                    cache_dir=FP.BASIS_CACHE), a.workers)
    finally:
        P.capture_stop_factors = keep
    grains = {}
    for name in CHARGE_NAMES:
        r = impact.disc_radius_cm(lib[name].nc)
        grains[name] = [equilibrium_charge_for_grain(
            1e-6, 1.0, float(t), "graphite", r, yield_params={"yield_model": "WDB06"},
            ion_species=[{"n": 1.0, "T": float(t), "m": 1.6726219e-27, "z": 1}])[3] for t in TEMPS]
    tops = bound.pop("top")
    free.pop("top")
    print("\n2. <Z>: pahchem with the bound / without it (step 2) / WDB06 graphite grain of the same radius")
    for name in CHARGE_NAMES:
        print(f"   {name} (ladder top +{tops[name]}):")
        for i, t in enumerate(TEMPS):
            print(f"     T {t:.1e}: {bound[name][i]:6.2f} / {free[name][i]:6.2f} / {grains[name][i]:6.2f}")
    plot(factors, bound, free, grains, tops, (a.out or _paths.get_plots_dir("pahchem")) / "pahchem_hot_gas_charge.png")


def plot(factors, bound, free, grains, tops, path):
    use_calima_style()
    fig, axes = plt.subplots(1, 1 + len(bound), figsize=(11.5, 3.6))
    ax = axes[0]
    for (name, f), colour in zip(factors.items(), ("#1f1f1f", "#2a78d6", "#e07b39", "#8a8a8a")):
        ax.plot(TEMPS, f, "-", color=colour, lw=2, marker="o", ms=3, label=name.split()[0])
    ax.set_yscale("log")
    ax.set_ylabel(r"capture factor ($Z = 1$)")
    ax.set_title("fast electrons must be stopped", fontsize=10)
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    for ax, name in zip(axes[1:], bound):
        ax.plot(TEMPS, bound[name], "-", color="#2a78d6", lw=2, marker="o", ms=4, label="pahchem, capture bounded")
        ax.plot(TEMPS, free[name], "--", color="#8a8a8a", lw=2, marker="s", ms=4, label="pahchem, every arrival captured")
        ax.plot(TEMPS, grains[name], ":", color="#1f1f1f", lw=2, label="WDB06 graphite grain, same radius")
        ax.axhline(tops[name], color="#e07b39", lw=1, ls="-.", label="ladder top")
        ax.set_title(name.split()[0], fontsize=10)
        ax.set_ylabel(r"$\langle Z \rangle$")
    axes[2].legend(fontsize=7, frameon=False, loc="center right")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel(r"$T$ [K]")
        ax.minorticks_on()
        ax.tick_params(which="both", direction="in", top=True, right=True)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"\nfigure: {path}")


if __name__ == "__main__":
    main()
