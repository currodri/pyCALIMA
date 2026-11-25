"""THEMIS CM20 heat capacity (DustEM C_CM20.DAT) against the Dulong-Petit limit and
pyCALIMA's harmonic model (``carbon_heat_capacity``).

Prints, and plots, the heat capacity per atom over 3 k for CM20 grains
(``themis_optics.heat_capacity``), counting the atoms of a-C at the CM20 density of
1.6 g/cm^3 (X_H = 0.023), and the same for DustEM's graphite (C_Gra.DAT,
2.24 g/cm^3, Draine & Li 2001). A solid cannot exceed 3 k per atom; at 5000 K the
a-C values reach 1.35-1.49 x 3 k per a-C atom, while graphite's reach 0.99. The
harmonic model (CM20 structure, ``heat_capacity_cm20``) is printed beside them, and
its polyethylene terms against the measured amorphous polyethylene C_p (Gaur &
Wunderlich 1981, J. Phys. Chem. Ref. Data 10, 119, Table 9).

Needs $CALIMA_DUSTEM_DIR.

    python diagnostics/dust_radiation/check_cm20_heat_capacity.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pycalima import _paths
from pycalima._datasets import ensure_dataset
from pycalima.models.dust_radiation import carbon_heat_capacity as chc
from pycalima.models.dust_radiation import themis_optics as th

KB = 1.380649e-16
AMU = 1.66053906660e-24
X_H = 0.1 / 4.3
N_AC = 1.6 / (((1 - X_H) * 12.011 + X_H * 1.008) * AMU)       # atoms per cm^3 of a-C
N_GRA = 2.24 / (12.011 * AMU)
RADII_UM = (3e-4, 3e-3, 1.8e-2, 0.1, 1.0)
T_K = (1.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 5000.0)


def graphite_per_volume(T):
    rows = [r.split() for r in th._numbers(ensure_dataset(th.DATASET) / "hcap" / "C_Gra.DAT")]
    n_t = int(rows[2][0])
    d = np.array(rows[3:3 + n_t], float)
    return 10.0 ** np.interp(np.log10(T), d[:, 0], d[:, -1])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=None, help="figure path")
    args = ap.parse_args(argv)

    a = np.array(RADII_UM)
    V = 4.0 / 3.0 * np.pi * (a * 1e-4) ** 3
    T = np.array(T_K)
    ratio = th.heat_capacity(a, T) / (3 * KB * N_AC * V)
    gra = graphite_per_volume(T) / (3 * KB * N_GRA)
    print("C / (3 k N_atoms), N_atoms of a-C at 1.6 g/cm^3 (graphite: its own atoms at 2.24)")
    print(f"{'T [K]':>8s} " + " ".join(f"{x * 1e3:>9.3g}nm" for x in a) + f" {'graphite':>11s}")
    for i, t in enumerate(T):
        print(f"{t:8.0f} " + " ".join(f"{v:11.3f}" for v in ratio[i]) + f" {gra[i]:11.3f}")
    harm = chc.heat_capacity_cm20(a, T) / (3 * KB * N_AC * V)
    print("\nharmonic model, same normalisation (large grains: their own atoms are 1.74x more per volume)")
    for i, t in enumerate(T):
        print(f"{t:8.0f} " + " ".join(f"{v:11.3f}" for v in harm[i]))
    print("\nDustEM / harmonic:")
    for i, t in enumerate(T):
        print(f"{t:8.0f} " + " ".join(f"{v:11.3f}" for v in ratio[i] / harm[i]))
    Tp = np.array([10.0, 20.0, 50.0, 100.0, 150.0, 200.0])
    cp = np.array([0.3709, 1.402, 5.145, 9.747, 14.06, 18.92])      # J/mol/K per CH2, G&W Table 9
    pe = (chc.per_sp3_carbon(Tp) + 2 * chc.per_aliphatic_h(Tp)) * 8.314462618
    print("\npolyethylene C_v(harmonic) / C_p(amorphous, measured) at", Tp, "K:", np.round(pe / cp, 3))
    plot(a, args.out)


def plot(a, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pycalima.plotting_style import use_calima_style

    use_calima_style()
    T = np.geomspace(0.1, 5000.0, 200)
    V = 4.0 / 3.0 * np.pi * (a * 1e-4) ** 3
    fig, ax = plt.subplots(figsize=(6, 4.2))
    r = th.heat_capacity(a, T) / (3 * KB * N_AC * V)
    h = chc.heat_capacity_cm20(a, T) / (3 * KB * N_AC * V)
    for c, x in enumerate(a):
        ax.plot(T, r[:, c], color=f"C{c}", label=fr"DustEM CM20, {x * 1e3:.3g} nm")
        ax.plot(T, h[:, c], ":", color=f"C{c}")
    ax.plot(T, graphite_per_volume(T) / (3 * KB * N_GRA), "k--", label="graphite (DL01), per its atoms")
    ax.axhline(1.0, color="0.5", lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 3)
    ax.set_xlabel(r"$T$ [K]")
    ax.set_ylabel(r"$C / (3 k N_{\rm atoms})$; dotted: harmonic")
    ax.legend(frameon=False, fontsize=8)
    ax.minorticks_on()
    ax.tick_params(which="both", axis="both", direction="in")
    fig.tight_layout()
    out = Path(out) if out else _paths.get_plots_dir() / "dust_radiation" / "check_cm20_heat_capacity.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
