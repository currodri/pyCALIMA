#!/usr/bin/env python3
"""Charge distributions of C54H18 and C96H24 against Tielens (2021) Fig. 12.34 (validation V14).

Tielens (2021, Molecular Astrophysics, §12.8.1, Fig. 12.34) solves photoionisation against recombination and
attachment (his eq. 12.18) for circumcoronene and circumcircumcoronene, x_e = 1.6e-4, in three environments:
a diffuse cloud (G0 = 2, n = 50, T = 80 K; gamma = 2.2e3), the NGC 7023 PDR (2600, 2e4, 500; 1.8e4) and its
cavity (2e4, 5e3, 500; 5.6e5). The caption puts the cavity in the middle column, but the panel labels and the
gamma order put it on the right. The C54H18 cavity panel repeats the C96H24 PDR panel bar for bar, so the book
gives no C54H18 cavity distribution.

The values below are read by eye from the book's figure (±0.01). pahchem uses v4.00 "C54H18 (37)" and
"C96H24 (108)", the calibrated IP shifts, the Draine field with G0 taken as Habing (the book does not state the
field), and no ions. It is shown with atomic H (n_H = n) and without it (n_H = 0, as the book's balance): the
same since 2026-10-26, when associative detachment left the parent anions (``Params.associative_detachment``).

Result (2026-10-24, 1 eV grid with the solar-circle old stars below 5 eV, n_H = 0):
- diffuse cloud: every f(Z) within 0.05 for both molecules (C54H18 f(-1, 0, +1) 0.11, 0.83, 0.05 against 0.13,
  0.78, 0.085; 0.13, 0.81, 0.05 on the old 5 eV grid); associative detachment at n_H = n lowers f(-1) to 0.08-0.09;
- NGC 7023 PDR: unchanged by the old stars at this G0; pahchem is less ionised, <Z> +0.36 against +0.55 (C54H18)
  and +0.54 against +0.78 (C96H24); f(0) 0.61 against 0.47 and 0.49 against 0.35, the same direction as against
  WD01 at high psi (V4);
- cavity, C96H24: <Z> +2.62 against +2.57, f(+2, +3) 0.27, 0.68 against 0.35, 0.61.
By detachment (2026-10-25, ``--detachment``), the anion fraction f(-1) of C96H24, against Tielens' 0.13 (diffuse)
and 0.015 (PDR): Jochims 0.11 and 0.025; statistical 0.75 and 0.53 (a large anion keeps its electron below
~10 eV: fails); valence 0.155 and 0.034 (passes). C54H18 and the cavity are the same for all three.
With the valence default, n_H = 0: diffuse C54H18 f(-1, 0, +1) 0.093, 0.851, 0.055 (<Z> -0.04, Tielens -0.04),
C96H24 0.155, 0.765, 0.079 (<Z> -0.07 against 0.00): every f(Z) within 0.07; PDR <Z> +0.37 and +0.53.
With the Wenzel+20 yield on the cation only (2026-10-26; the dication takes Jochims'): the cavity C96H24 <Z> +2.46
against +2.57, f(+2, +3) 0.39, 0.54 against 0.35, 0.61; PDR C96H24 <Z> +0.52; nothing else moves by > 0.01.

The figure (``pahchem_v14_tielens21_fz.png``): f(Z) per molecule and environment, the book against pahchem with
each detachment (n_H = 0).

Rerun on 2026-11-02 with the current defaults: section 4 of the gas-coupling doc, where ``check_default_changes.py``
adds the changes since 2026-10-25 one at a time.

    check_tielens21_fz.py [--workers 4] [--detachment jochims|statistical|valence] [--out <plot dir>]
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
X_E = 1.6e-4
#: name -> (G0, n [cm^-3], T [K])
ENVIRONMENTS = {"diffuse": (2.0, 50.0, 80.0), "NGC 7023 PDR": (2600.0, 2e4, 500.0),
                "NGC 7023 cavity": (2e4, 5e3, 500.0)}
#: (molecule, environment) -> f(-1, 0, +1, +2, +3) read from Fig. 12.34
FIG_12_34 = {("C54H18 (37)", "diffuse"): (0.13, 0.78, 0.085, 0.003, 0.0),
             ("C54H18 (37)", "NGC 7023 PDR"): (0.02, 0.47, 0.43, 0.07, 0.0),
             ("C96H24 (108)", "diffuse"): (0.13, 0.745, 0.125, 0.005, 0.0),
             ("C96H24 (108)", "NGC 7023 PDR"): (0.015, 0.35, 0.47, 0.15, 0.01),
             ("C96H24 (108)", "NGC 7023 cavity"): (0.0, 0.0, 0.035, 0.35, 0.61)}
BASIS_CACHE = _paths.get_model_data_dir() / "pahchem_population_basis"
DETACHMENTS = {"valence": "#2a78d6", "statistical": "#1baf7a", "jochims": "#eb6834"}
ZS = (-1, 0, 1, 2, 3)


def molecules():
    lib = population.shift_thresholds(population.library_v400(), IP_SHIFTS)
    return [m for m in lib if m.name in ("C54H18 (37)", "C96H24 (108)")]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--detachment", default=Params().detachment, choices=("jochims", "statistical", "valence"))
    ap.add_argument("--out", type=Path, default=None, help="plot directory (default results/plots/pahchem)")
    args = ap.parse_args()
    mols = molecules()
    pop = population.Population(mols, {"all": (1.0, 1e4)}, cache_dir=BASIS_CACHE,
                                params=Params(detachment=args.detachment))
    zs = (-1, 0, 1, 2, 3)
    cols = [pop.zs.index(z) for z in zs]
    print(f"f(Z) for Z = {zs}; Draine field, x_e = {X_E:g}, no ions, detachment {args.detachment}")
    for env_name, (g0, n, t) in ENVIRONMENTS.items():
        ne = X_E * n
        print(f"\n{env_name}: G0 = {g0:g}, n = {n:g}, T = {t:g} K, gamma = {g0 * np.sqrt(t) / ne:.2e}")
        fz, zbar, above = {}, {}, {}
        for n_h in (n, 0.0):
            env = Environment(env_name, g0, n_h, ne, t, None)
            pop.prepare(population.field_shape(env), workers=args.workers)
            f = pop.charge_matrix(env)
            # <Z> over every charge of the ladder (to the cold limit), and the share above +3 that the columns leave out
            fz[n_h], zbar[n_h] = f[:, cols], f @ np.array(pop.zs, dtype=float)
            above[n_h] = f[:, [i for i, z in enumerate(pop.zs) if z > zs[-1]]].sum(axis=1)
        for s, m in enumerate(mols):
            ref = FIG_12_34.get((m.name, env_name))
            shown = "not given (panel repeated)" if ref is None else np.round(ref, 3)
            print(f"  {m.name:13s} Tielens Fig. 12.34    {shown}")
            for n_h, label in ((n, "n_H = n"), (0.0, "n_H = 0")):
                print(f"  {'':13s} pahchem, {label:9s}    {np.round(fz[n_h][s], 3)}   <Z> {zbar[n_h][s]:+.2f}"
                      + (f"   f(> +{zs[-1]}) {above[n_h][s]:.3f}" if above[n_h][s] >= 5e-4 else "")
                      + ("" if ref is None or n_h else f"   Tielens <Z> {np.dot(ref, zs):+.2f}"))
    plot(mols, by_detachment(mols, args.workers), (args.out or _paths.get_plots_dir("pahchem")) / "pahchem_v14_tielens21_fz.png")


def by_detachment(mols, workers):
    """{detachment: {(molecule, environment): f(Z) over ZS}} at n_H = 0."""
    out = {}
    for det in DETACHMENTS:
        pop = population.Population(mols, {"all": (1.0, 1e4)}, cache_dir=BASIS_CACHE, params=Params(detachment=det))
        cols = [pop.zs.index(z) for z in ZS]
        out[det] = {}
        for env_name, (g0, n, t) in ENVIRONMENTS.items():
            env = Environment(env_name, g0, 0.0, X_E * n, t, None)
            pop.prepare(population.field_shape(env), workers=workers)
            f = pop.charge_matrix(env)[:, cols]
            for s, m in enumerate(mols):
                out[det][m.name, env_name] = f[s]
    return out


def plot(mols, fz, path):
    use_calima_style()
    fig, axes = plt.subplots(len(mols), len(ENVIRONMENTS), figsize=(15, 7.5), sharex=True, sharey=True,
                             gridspec_kw={"hspace": 0.08, "wspace": 0.05})
    x, w = np.arange(len(ZS)), 0.2
    for r, m in enumerate(mols):
        for c, env_name in enumerate(ENVIRONMENTS):
            ax = axes[r, c]
            ref = FIG_12_34.get((m.name, env_name))
            if ref is not None:
                ax.bar(x - 1.5 * w, ref, w, color="0.15", label="Tielens (2021) Fig. 12.34")
            else:
                ax.annotate("book panel repeats\nC$_{96}$H$_{24}$ PDR: not shown", (0.03, 0.8), xycoords="axes fraction",
                            fontsize=10, color="0.4")
            for k, (det, col) in enumerate(DETACHMENTS.items()):
                lab = f"pahchem, {det}" + (" (default)" if det == Params().detachment else "")
                ax.bar(x + (k - 0.5) * w, fz[det][m.name, env_name], w, color=col, label=lab)
            g0, n, t = ENVIRONMENTS[env_name]
            if r == 0:
                ax.set_title(rf"{env_name}: $G_0$ {g0:g}, $n$ {n:g}, $T$ {t:g} K, $\gamma$ {g0 * np.sqrt(t) / (X_E * n):.1e}",
                             fontsize=11)
            ax.set_xticks(x, [f"{z:+d}" for z in ZS])
            ax.minorticks_on()
            ax.tick_params(axis="x", which="minor", bottom=False, top=False)
            ax.tick_params(which="both", direction="in", top=True, right=True)
            if c == 0:
                nc, nh = m.name.split()[0][1:].split("H")
                ax.set_ylabel(rf"C$_{{{nc}}}$H$_{{{nh}}}$: $f(Z)$", fontsize=14)
            if r == len(mols) - 1:
                ax.set_xlabel("Z", fontsize=14)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), frameon=False, fontsize=12, ncol=4, loc="lower center",
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(rf"Charge distributions, $x_e = {X_E * 1e4:g}\times10^{{-4}}$, Draine field, no ions, no associative"
                 " detachment", fontsize=14, y=0.97)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    main()
