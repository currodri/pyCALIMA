#!/usr/bin/env python3
"""The compact dications of pyrene, coronene and ovalene (megatron, 2026-10-30) against the proxy the library uses for them (the cation's
spectrum), and the four charge states of the C24H14_538 isomer against coronene with its real dication.

    check_small_dications.py [--dications ~/Documents/RAMSES_dev/pahspec/dications_2026-10-30]
                             [--isomers ~/Documents/RAMSES_dev/pahspec/dial3/library_small] [--out DIR]
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pycalima import _paths
from pycalima.pahchem import uvlib
from pycalima.plotting_style import use_calima_style

MB_EV_PER_F = 109.761
A2_TO_CM2 = 1e-16
RAMSES = Path.home() / "Documents/RAMSES_dev/pahspec"
SMALL = (("pyrene_2", 16, "pyrene$^{2+}$"), ("coronene_2", 24, "coronene$^{2+}$"), ("ovalene_2", 32, "ovalene$^{2+}$"))


def tensor(run):
    t = np.loadtxt(Path(run) / "spectrum" / "cross_section_tensor", comments="#")
    return t[:, 0], t[:, 1] * A2_TO_CM2


def f_per_c(e, s, nc, e_max=30.0):
    k = e <= e_max
    return np.trapezoid(s[k], e[k]) / nc / 1e-18 / MB_EV_PER_F


def mean_energy(e, s, lo=1.0, hi=13.6):
    k = (e > lo) & (e < hi)
    return np.trapezoid(s[k] * e[k], e[k]) / np.trapezoid(s[k], e[k])


def style(ax, first):
    ax.set_xlim(0, 25)
    ax.set_xlabel("photon energy [eV]")
    if first:
        ax.set_ylabel("cross section per carbon [Mb]")
    ax.minorticks_on()
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.legend(frameon=False, fontsize=8)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dications", type=Path, default=RAMSES / "dications_2026-10-30")
    ap.add_argument("--isomers", type=Path, default=RAMSES / "dial3/library_small")
    ap.add_argument("--out", type=Path, default=_paths.get_plots_dir("pahchem"))
    a = ap.parse_args()
    use_calima_style()
    a.out.mkdir(parents=True, exist_ok=True)

    print("compact dications: the run vs the library's proxy (the cation anchor, TD-LDA positions)")
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 3.6))
    for i, (ax, (d, nc, name)) in enumerate(zip(axs, SMALL)):
        e, s = tensor(a.dications / d)
        s /= nc
        proxy = uvlib.per_carbon(nc, 2, e, "tdlda", "compact")
        ax.plot(e, s * 1e18, color="C3", label=f"{name}, TD-LDA run")
        ax.plot(e, proxy * 1e18, color="0.4", ls="--", label="library proxy (cation)")
        style(ax, i == 0)
        k = (e > 1.0) & (e < 8.0)
        print(f"  {d:11s} f/C<30 eV run {f_per_c(e, s * nc, nc):.3f} proxy {f_per_c(e, proxy * nc, nc):.3f}; "
              f"<E> 1-13.6 eV run {mean_energy(e, s):.2f} proxy {mean_energy(e, proxy):.2f} eV; "
              f"lowest band above 1 eV at {e[k][np.argmax(s[k] > 0.1 * s[k].max())]:.2f} (run) vs "
              f"{e[k][np.argmax(proxy[k] > 0.1 * proxy[k].max())]:.2f} eV (proxy)")
    fig.tight_layout()
    fig.savefig(a.out / "small_dications_vs_proxy.png", dpi=150)

    print("C24H14_538 vs coronene (the +2 reference is now the run, not the proxy)")
    fig, axs = plt.subplots(1, 4, figsize=(17, 3.6))
    for i, (ax, z) in enumerate(zip(axs, (-1, 0, 1, 2))):
        run = a.isomers / f"C24H14_538_{z:+d}"
        e, s = tensor(run)
        s /= 24
        if z == 2:
            ref = tensor(a.dications / "coronene_2")[1] / 24
            ax.plot(e, uvlib.per_carbon(24, 2, e, "tdlda", "compact") * 1e18, color="0.7", ls=":", label="old proxy (coronene$^+$)")
            lab = "coronene$^{2+}$ run"
        else:
            ref, lab = uvlib.per_carbon(24, z, e, "tdlda", "compact"), "coronene, library"
        ax.plot(e, s * 1e18, color="C0", label="C24H14_538 isomer")
        ax.plot(e, ref * 1e18, color="0.3", ls="--", label=lab)
        ax.set_title(f"charge {z:+d}")
        style(ax, i == 0)
        print(f"  {z:+d}: f/C<30 eV isomer {f_per_c(e, s * 24, 24):.3f} compact {f_per_c(e, ref * 24, 24):.3f}; "
              f"<E> 1-13.6 eV isomer {mean_energy(e, s):.2f} compact {mean_energy(e, ref):.2f} eV")
    fig.tight_layout()
    fig.savefig(a.out / "c24_isomer_vs_coronene_four_charges.png", dpi=150)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
