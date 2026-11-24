"""The hot-gas join of the molecular bins (A3c phase 6): their tabulated carbon loss against classical sputtering.

Above 1e4 K the molecular bins (N_C 20-400) erode by pahchem's hot-gas rates (``Params.hot_gas``: electron and ion
passages, C and H knock-outs; the ``--hot-gas`` hydrogen tables), and the grain bins above N_C,max by classical
thermal sputtering (``gsd.sputtering.erosion_rate``, the GSD's carbon: 'graphite' yields at rho 1.6, labelled). The
two meet at the seam, N_C 400 (1.06 nm). This script compares their carbon loss per C atom, k/N_C [s^-1]:
- each molecular bin at its mean N_C (the tables' ``nc_mean``), from the hot tables;
- classical sputtering at the top molecular bin's mean N_C, at N_C 400 and at N_C 1000.
The factor at the seam is the top bin's rate over the classical one at N_C 400. The plan's rule (2026-11-23): beyond
x2 the bins between take a labelled interpolation towards N_C ~ 1000 (Bocchio+12); this script reports, it decides
nothing.

The gas: n_H = 1 cm^-3 (both rates are linear in the density), n_e = 1.2 n_H (the tables' x_e axis ends at 1, so
they take x_e = 1, labelled), G0 at the tables' lowest value. A figure of k/N_C against T goes next to the text.

    python diagnostics/gsd/check_hot_erosion_join.py --hot results/gsd_a3c/tables/hot_k8 \
        [--out results/gsd_a3c/hot_erosion_join.txt] [--fig results/gsd_a3c/figs/hot_erosion_join.png]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pycalima.gsd.molecular import MolecularTables

M_C = 12.011 * 1.66053906660e-24
RHO = 1.6


def radius(nc):
    return (3.0 * np.asarray(nc, float) * M_C / (4.0 * np.pi * RHO)) ** (1.0 / 3.0)


def classical_per_c(T, nc, n_h=1.0):
    """Classical carbon loss per C atom [s^-1] at N_C: 3 |da/dt| / a."""
    from pycalima.gsd.sputtering import erosion_rate

    a = radius(nc)
    return 3.0 * np.abs(erosion_rate("graphite", T, n_h, rho=RHO)(a)) / a


def nc_means(d):
    fs = sorted(Path(d).glob("bin_nc*.npz"), key=lambda f: float(f.name.split("bin_nc")[1].split("-")[0]))
    return np.array([float(np.load(f, allow_pickle=True)["nc_mean"]) for f in fs]), [f.stem for f in fs]


def _style(ax):
    ax.minorticks_on()
    ax.tick_params(which="both", axis="both", direction="in", bottom=True, left=True, top=True, right=True)
    ax.tick_params(which="major", length=5)
    ax.tick_params(which="minor", length=2.6)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--hot", type=Path, required=True, help="hot-gas hydrogen tables (bin_nc*_carbon.dat and .npz)")
    ap.add_argument("--out", type=Path, default=Path("results/gsd_a3c/hot_erosion_join.txt"))
    ap.add_argument("--fig", type=Path, default=Path("results/gsd_a3c/figs/hot_erosion_join.png"))
    a = ap.parse_args(argv)
    mt = MolecularTables.from_directories(a.hot)
    nc, labels = nc_means(a.hot)
    lT = np.arange(4.0, 8.0 + 1e-9, 0.25)
    g0 = 1e-30                                               # clamped to the tables' lowest G0
    k_mol = np.array([mt.carbon_loss(g0, 1.0, 10.0 ** t, 1.2) for t in lT]) / nc     # [T, bin], per C
    cls = {n: np.array([float(classical_per_c(10.0 ** t, n)) for t in lT]) for n in (nc[-1], 400.0, 1000.0)}
    lines = ["Hot-gas join: carbon loss per C atom k/N_C [s^-1] at n_H = 1 cm^-3 (n_e = 1.2 n_H; tables at x_e = 1).",
             f"molecular bins (hot tables {a.hot}): " + ", ".join(f"{s} <N_C> {x:.0f}" for s, x in zip(labels, nc, strict=True)),
             "classical: gsd.sputtering 'graphite' yields at rho 1.6 (size-corrected), at the top bin's <N_C>, 400, 1000.",
             "factor = top molecular bin / classical at N_C 400 (the seam); plan rule: beyond x2, interpolate (labelled).",
             "log T  " + "  ".join(f"mol{i + 1:d}    " for i in range(nc.size))
             + f"  cl@{nc[-1]:.0f}    cl@400    cl@1000   factor"]
    for i, t in enumerate(lT):
        row = "  ".join(f"{x:9.2e}" for x in k_mol[i])
        row += "  " + "  ".join(f"{cls[n][i]:9.2e}" for n in (nc[-1], 400.0, 1000.0))
        f = k_mol[i, -1] / cls[400.0][i] if cls[400.0][i] > 0 else np.inf
        lines.append(f"{t:5.2f}  {row}  {f:8.2f}")
    hot = lT >= 5.0
    with np.errstate(divide="ignore"):
        fac = k_mol[hot, -1] / cls[400.0][hot]
    lines.append(f"factor over 1e5-1e8 K: {np.nanmin(fac):.2f} to {np.nanmax(fac):.2f}")
    txt = "\n".join(lines)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(txt + "\n")
    print(txt)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pycalima.plotting_style import use_calima_style

    use_calima_style()
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    cmap = plt.get_cmap("viridis")
    for j in range(nc.size):
        ax.plot(10.0 ** lT, k_mol[:, j], color=cmap(0.15 + 0.7 * j / max(nc.size - 1, 1)), lw=1.6,
                label=rf"molecular bin {j + 1} ($\langle N_\mathrm{{C}}\rangle$ {nc[j]:.0f})")
    for n, ls in ((nc[-1], ":"), (400.0, "--"), (1000.0, "-.")):
        ax.plot(10.0 ** lT, cls[n], color="0.25", ls=ls, lw=1.3, label=rf"classical, $N_\mathrm{{C}}$ {n:.0f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$T$ [K]", fontsize=14)
    ax.set_ylabel(r"carbon loss per C atom [s$^{-1}$], $n_\mathrm{H} = 1$ cm$^{-3}$", fontsize=12)
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    a.fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.fig, dpi=220, bbox_inches="tight")


if __name__ == "__main__":
    main()
