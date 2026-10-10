#!/usr/bin/env python3
"""Turbulence-only coagulation in the Guillet et al. (2020) collapse, against their Fig. 8.

Writes ``coag_guillet20_fig8.png`` to ``_paths.get_plots_dir("collapse")``: the size
distribution at n_H = 1e8, 1e10, 1e12 cm^-3 (each normalised to its initial peak)
from tests/test_collapse_guillet20.py's run, over the curves read from their figure:
solid with their prescriptions (conserved ice mantles, Ormel & Cuzzi limit
formulas), dotted with a fixed mantle and the full Ormel & Cuzzi form.

    python diagnostics/collapse/check_guillet20.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pycalima import _paths
from pycalima.plotting_style import use_calima_style

TESTS = Path(__file__).resolve().parents[2] / "tests"
sys.path.insert(0, str(TESTS))
import test_collapse_guillet20 as G  # noqa: E402


def main():
    use_calima_style()
    plt.rcParams["text.usetex"] = False
    g, snaps = G._run()
    g2, snaps2 = G._run(form="full", conserve_ice=False)
    ref = G._curves()
    norm_ref = ref[4][1].max()
    fig, ax = plt.subplots(figsize=(7.5, 5))
    xa, ya = ref[4]
    ax.plot(10 ** xa, 10 ** (ya - norm_ref), "o", color="0.5", mfc="none", ms=4, label="Guillet+20, n = 1e4")
    for ln, c in ((8, "tab:blue"), (10, "tab:orange"), (12, "tab:red")):
        xa, ya = ref[ln]
        ax.plot(10 ** xa, 10 ** (ya - norm_ref), "o", color=c, mfc="none", ms=4, label=f"Guillet+20, n = 1e{ln}")
        oa, oy = G._normalised(g, snaps, ln)
        ax.plot(10 ** oa, 10 ** oy, "-", color=c, lw=1.4, label=f"pyCALIMA (their prescriptions), 1e{ln}")
        oa, oy = G._normalised(g2, snaps2, ln)
        ax.plot(10 ** oa, 10 ** oy, ":", color=c, lw=1.0)
    ax.plot([], [], "k:", lw=1.0, label="fixed 8.8 nm mantle, full OC07")
    ax.set(xscale="log", yscale="log", xlim=(3e-3, 30), ylim=(3e-3, 3), xlabel=r"core radius [$\mu$m]",
           ylabel=r"$d\rho_d/d\log a$ / initial peak",
           title="Turbulence-only coagulation in collapse (Guillet+20 Fig. 8)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    out = _paths.get_plots_dir("collapse") / "coag_guillet20_fig8.png"
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    main()
