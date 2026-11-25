"""House-style plotting for the PAHrt study figures (pyCALIMA's style: ``use_calima_style``, inward major and minor
ticks on all four sides, frameless legends, LaTeX-safe labels)."""
from __future__ import annotations

import numpy as np

BIN_LABELS = (r"bin 1 ($N_{\rm C}$ 20--42)", r"bin 2 (42--89)", r"bin 3 (89--189)", r"bin 4 (189--400)")


def setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pycalima.plotting_style import use_calima_style

    use_calima_style()
    return plt


def style(ax, logx=False, logy=False):
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.minorticks_on()
    ax.tick_params(which="both", axis="both", direction="in", top=True, right=True, bottom=True, left=True)
    ax.tick_params(which="major", length=4)
    ax.tick_params(which="minor", length=2)


def bin_colors(plt, n=4):
    return plt.get_cmap("viridis")(np.linspace(0.1, 0.85, n))


def phase_colors(plt, names):
    cols = plt.get_cmap("tab10")(np.arange(len(names)))
    return dict(zip(names, cols, strict=True))
