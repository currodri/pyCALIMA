"""Per-bin optics of the two carbon carriers on the bins that hold both (A3b).

On the default ``CarbonLayout`` (4 bins over N_C 20-400, then 7 to 1 um) the
aliphatic population (a-C:H, E_g = 2.5 eV, Jones 2012) shares the PAH-size bins
with the aromatic one (THEMIS CM20). For each of those bins this builds the
per-gram-of-carbon tables kappa(lambda, s) of both carriers with the A2 settings
(``gsd.optics``: 128 nodes per bin, s grids refined to tol), and prints

- the s-grid sizes at that tolerance;
- kappa_abs at s = 0.5 at a few wavelengths, and aliphatic / aromatic;
- the 3.4 um band of the aliphatic carrier over its continuum;
- the V-band (0.55 um) and 2175 A opacity per gram of carbon of each.

    python diagnostics/gsd/check_carbon_carriers.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pycalima import _paths
from pycalima.gsd.carbon import CARRIERS as ALL_CARRIERS
from pycalima.gsd.carbon import M_C, CarbonLayout
from pycalima.gsd.optics import adaptive_kappa_tables, lookup

LAM_UM = np.logspace(-3, 3, 241)          # the RAMSES optics grid
# the default carriers (on these sub-20 nm bins 'aromatic_dustem' is 'aromatic')
CARRIERS = {n: ALL_CARRIERS[n] for n in ("aromatic", "aliphatic")}
PROBE_UM = (0.1, 0.2175, 0.55, 3.0, 3.4, 6.2, 11.3, 100.0)
N_NODES = 128


def tables(layout, tol):
    grid = layout.grid()
    bins = np.nonzero(layout.aliphatic_bins(grid))[0]
    out = {}
    for name, carrier in CARRIERS.items():
        per_gram = carrier.per_gram_on(grid, LAM_UM)
        sub = _subgrid(grid, bins)
        out[name] = adaptive_kappa_tables(sub, N_NODES, per_gram, tol)
    return grid, bins, out


def _subgrid(grid, bins):
    from pycalima.gsd.grid import MassGrid

    return MassGrid(grid.m_edges[bins[0]:bins[-1] + 2], grid.rho)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tol", type=float, default=3e-3)
    ap.add_argument("--out", default=None, help="figure path")
    args = ap.parse_args(argv)

    layout = CarbonLayout()
    grid, bins, tab = tables(layout, args.tol)
    nc = grid.m_edges / M_C
    probe = [int(np.argmin(abs(LAM_UM - l))) for l in PROBE_UM]
    print(f"s grids at tol {args.tol:g} (points per bin)")
    for name in CARRIERS:
        print(f"  {name:>9s}: " + " ".join(str(t[0].size) for t in tab[name]))
    print("\nkappa_abs [cm^2 per g of C] at s = 0.5; aliphatic / aromatic")
    print(f"{'bin N_C':>14s} " + " ".join(f"{LAM_UM[j]:>9.3g}" for j in probe))
    for k, b in enumerate(bins):
        ka = {n: lookup(*tab[n][k], np.array([0.5]))[0, :, 0] for n in CARRIERS}
        print(f"{nc[b]:6.0f}-{nc[b + 1]:<6.0f}  arom " + " ".join(f"{ka['aromatic'][j]:9.3g}" for j in probe))
        print(f"{'':14s} alip " + " ".join(f"{ka['aliphatic'][j]:9.3g}" for j in probe))
        print(f"{'':14s} a/a  " + " ".join(f"{ka['aliphatic'][j] / ka['aromatic'][j]:9.3g}" for j in probe))
    plot(grid, bins, tab, nc, args.out)


def plot(grid, bins, tab, nc, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pycalima.plotting_style import use_calima_style

    use_calima_style()
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, name in zip(axs, CARRIERS):
        for k, b in enumerate(bins):
            for s, ls in ((0.0, ":"), (0.5, "-"), (1.0, "--")):
                ka = lookup(*tab[name][k], np.array([s]))[0, :, 0]
                ax.loglog(LAM_UM, ka, ls, color=f"C{k}", lw=1,
                          label=fr"$N_{{\rm C}}$ {nc[b]:.0f}-{nc[b + 1]:.0f}" if s == 0.5 else None)
        ax.set_title(f"{name} ({CARRIERS[name].optics}, $E_g$ = {CARRIERS[name].e_gap} eV)", fontsize=10)
        ax.set_xlabel(r"$\lambda$ [$\mu$m]")
        ax.set_xlim(1e-2, 1e3)
        ax.minorticks_on()
        ax.tick_params(which="both", axis="both", direction="in")
    axs[0].set_ylabel(r"$\kappa_{\rm abs}$ [cm$^2$ g$^{-1}$ of C]; $s$ = 0, 0.5, 1")
    axs[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out = Path(out) if out else _paths.get_plots_dir() / "gsd" / "check_carbon_carriers.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
