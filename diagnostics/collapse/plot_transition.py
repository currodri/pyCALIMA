"""The Pop III / Pop II transition is a band, not a line: one small panel for a proposal.

    python diagnostics/collapse/plot_transition.py [--criterion frag]

From ``science_grid_transition.py``'s results, writes ``collapse/transition_<criterion>.png``
under the plots directory (and an editable ``.svg``, text kept as text; ``_dark`` versions for a dark
background): [Z_cr] with the classical dust assumptions of the published
thresholds; the range when each physical input alone departs from them (a white tick
per choice, the choices at the two ends named); the range of all combinations (bracketed by ``science_grid_transition.py``); and the published
dust-cooling and metal-line thresholds as bands.
"""

import argparse
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrow, Rectangle
from matplotlib.ticker import MultipleLocator

from pycalima import _paths
from pycalima.plotting_style import use_calima_style

sys.path.insert(0, str(Path(__file__).resolve().parent))
import science_grid_transition as grid   # noqa: E402

OUT = _paths.get_plots_dir() / "collapse"
X_LO, X_HI = -6.5, -2.5
# published critical metallicities, log Z/Z_sun as stated (normalisations differ by <~0.2 dex), checked against
# the papers; the bands span them.
LITERATURE = {
    "dust": [(-6.0, "Omukai+05, Schneider+06,12a"), (-5.5, "Chiaki+15"), (-5.0, "Clark+08, Dopcke+11,13, Chon+21"),
             # Ji+14 [Si/H] = -4.5 (Milky Way sizes), -5.3 (shocked sizes), as [Z/H] for the N07 ejecta,
             # whose [Si/H] - [Z/H] is 0.27 to 0.48 (Chiaki+15 Fig. 2, Asplund+09 Si): 0.38
             (-4.88, "Ji+14 (MW sizes)"), (-5.68, "Ji+14 (shocked)")],
    "lines": [(-4.0, "Schneider+02,12a"), (-3.75, "Smith+09"), (-3.5, "Bromm & Loeb 03 (C), Santoro & Shull 06"),
              (-3.3, "Bromm+01"), (-3.05, "Bromm & Loeb 03 (O)")],
}
ROWS = [   # (label, key, values, colour, names of the values): dust physics, then environment
    ("dust surviving\nthe reverse shock", "d", grid.DUST, "C1",
     {"MW": "Milky Way", "n0": "no shock", "n0.1": "0.1", "n1": "1", "n10": r"$n_{\rm amb} = 10\,$cm$^{-3}$"}),
    ("SN progenitor", "M", grid.MASSES, "C0",
     {13: r"13$\,$M$_\odot$ (C)", 20: "20", 25: r"25$\,$M$_\odot$", 30: r"30$\,$M$_\odot$ (Si)"}),
    ("initial sizes", "s", grid.SHAPES, "C4",
     {"MRN": "MRN", "grackle": "lognormal", "N03": "N03", "N07": "N07", "eroded": "eroded"}),
    ("accretion\n" r"($n_{\rm amb} = 1$ dust)", "p", grid.PROCESSES, "C2", {"static": "off", "acc": "on"}),
    ("accommodation", "a", grid.ALPHAS, "C5", {1.0: r"$\alpha_T = 1$", 0.5: "0.5", 0.3: "0.3"}),
    ("cosmic rays", "z", grid.ZETAS, "C3",
     {0.0: r"$\zeta_0 = 0$", 1e-18: r"$10^{-18}\,$s$^{-1}$", 1e-16: r"$10^{-16}\,$s$^{-1}$"}),
    ("Lyman--Werner", "lw", grid.J21S, "C8", {0.0: r"$J_{21} = 0$", 10.0: "10", 100.0: "100 (atomic)"}),
]
GROUP_AFTER = 5            # a solid line between the dust physics and the environment rows


# colours per theme: "light" for print, "dark" for slides on a dark background (row colours by row key)
THEMES = {
    "light": dict(bg="white", fg="k", sep="0.3", grid="0.75", group="0.5", combined="k",
                  bands={"dust": ("saddlebrown", 0.12, "saddlebrown"), "lines": ("indigo", 0.12, "indigo")}, rows={}),
    "dark": dict(bg="#15171c", fg="#ececec", sep="#9a9a9a", grid="#4a4d55", group="#7a7d85", combined="#f5f5f5",
                 bands={"dust": ("#c8925e", 0.22, "#e8b98c"), "lines": ("#9d7dff", 0.22, "#c0a8ff")},
                 rows={"d": "#ffa95c", "M": "#6fb6ff", "s": "#c6a8ff", "p": "#7fe08f", "a": "#dcb88f",
                       "z": "#ff7f7f", "lw": "#f2d46e"}),
}


def _bar(ax, lo, hi, y, color, width=0.24, open_end=False):
    """A filled bar from lo to hi at row y; with open_end it ends in an arrowhead at hi (no fragmentation)."""
    if open_end:
        ax.add_patch(FancyArrow(lo, y, hi - lo, 0.0, width=width, head_width=1.9 * width, head_length=0.17,
                                length_includes_head=True, color=color, lw=0, zorder=2))
    else:
        ax.add_patch(Rectangle((lo, y - width / 2), hi - lo, width, color=color, lw=0, zorder=2))


def _ends(vals, v, names):
    """Names of the choices at the low and the high end (joined when several share it; inf is the high end)."""
    lo = [names[x] for x, y in zip(vals, v) if y == np.nanmin(v)]
    top = np.inf if np.isinf(v).any() else np.nanmax(v)
    hi = [names[x] for x, y in zip(vals, v) if y == top]
    return ", ".join(lo), ", ".join(hi)


def figure(c="frag", ext="png", theme="light"):
    res, _ = grid.results()
    get = lambda **k: res[(c, *(k[x] for x in grid.KEYS))]
    allv = np.array([v for k, v in res.items() if k[0] == c])     # rows and the bracketing combinations
    z0 = get(**grid.CLASSICAL)
    n = len(ROWS)
    x_lo = min(X_LO, math.floor(2.0 * (allv.min() - 0.5)) / 2.0)
    th = THEMES[theme]
    rc = {"text.color": th["fg"], "axes.edgecolor": th["fg"], "axes.labelcolor": th["fg"], "xtick.color": th["fg"],
          "ytick.color": th["fg"], "figure.facecolor": th["bg"], "axes.facecolor": th["bg"],
          "savefig.facecolor": th["bg"]}
    with plt.rc_context(rc):
        _draw(c, ext, theme, th, get, allv, z0, x_lo)


def _draw(c, ext, theme, th, get, allv, z0, x_lo):
    n = len(ROWS)
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    for ch, lab in (("dust", "published (dust)"), ("lines", "published (metal lines)")):
        col, alpha, tcol = th["bands"][ch]
        v = [x for x, _ in LITERATURE[ch]]
        ax.axvspan(min(v), max(v), color=col, alpha=alpha, lw=0)
        ax.text(0.5 * (min(v) + max(v)), n + 1.7, lab, color=tcol, fontsize=9, ha="center", va="bottom")
    ax.axvline(z0, color=th["fg"], ls="--", lw=0.8, zorder=1)
    ax.plot(z0, n + 1, "D", color=th["fg"], ms=7, zorder=4)
    ax.text(z0 + 0.12, n + 1, "Milky Way dust/metal, MRN, static,\n" r"no CRs or LW",
            fontsize=7.5, va="center", linespacing=0.95)
    for row, (lab, key, vals, col, names) in zip(range(n, 0, -1), ROWS):
        col = th["rows"].get(key, col)
        v = np.array([get(**{**grid.row_reference(key), key: x}) for x in vals])
        fin = v[np.isfinite(v)]
        if fin.size == 0:                                 # every choice: no fragmentation
            ax.text(X_HI - 0.05, row, "no fragmentation at " + ", ".join(names[x] for x in vals), fontsize=7.5,
                    color=col, ha="right", va="center")
            continue
        lo, hi = fin.min(), (X_HI if np.isinf(v).any() else fin.max())
        _bar(ax, lo, hi - 0.02 if np.isinf(v).any() else hi, row, col, open_end=np.isinf(v).any())
        for x in fin:                                     # one tick per choice
            ax.plot([x, x], [row - 0.12, row + 0.12], color=th["bg"], lw=1.0, zorder=3)
        left, right = _ends(vals, v, names)
        if hi - lo < 0.02:                                # every choice gives the same [Z_cr]
            ax.text(hi + 0.07, row, ", ".join(names[x] for x in vals) + ": no change", fontsize=7.5, color=col,
                    ha="left", va="center")
            continue
        if lo - 0.07 - 0.07 * len(left.replace("$", "").replace("\\", "")) < x_lo:     # no room: above the bar
            ax.text(lo + 0.05, row + 0.22, left, fontsize=7.5, color=col, ha="left", va="bottom")
        else:
            ax.text(lo - 0.07, row, left, fontsize=7.5, color=col, ha="right", va="center")
        if np.isinf(v).any() or hi - lo > 1.5:
            ax.text(hi - 0.05, row + 0.22, right, fontsize=7.5, color=col, ha="right", va="bottom")
        else:
            ax.text(hi + 0.07, row, right, fontsize=7.5, color=col, ha="left", va="center")
    ax.axhline(n + 0.5, color=th["sep"], lw=0.6)
    ax.axhline(0.5, color=th["sep"], lw=0.6)
    for y in np.arange(1.5, n, 1.0):
        ax.axhline(y, color=th["grid"], lw=0.4, ls=":")
    ax.axhline(n - GROUP_AFTER + 0.5, color=th["group"], lw=0.5)
    lo = allv.min()
    _bar(ax, lo, X_HI - 0.02 if np.isinf(allv).any() else allv.max(), 0, th["combined"], width=0.3,
         open_end=np.isinf(allv).any())
    ax.text(lo - 0.07, 0, f"${lo:.1f}$", fontsize=7.5, ha="right", va="center")
    if np.isinf(allv).any():
        ax.text(X_HI - 0.05, 0.25, "no fragmentation", fontsize=7.5, ha="right", va="bottom")
    dash = (lambda t: t) if plt.rcParams["text.usetex"] else (lambda t: t.replace("--", "\u2013"))
    bold = r"\textbf{all combined}" if plt.rcParams["text.usetex"] else "all combined"
    ax.set_yticks(range(n + 1, -1, -1), ["classical"] + [dash(r[0]) for r in ROWS] + [bold],
                  fontsize=9.5)
    for t, r in zip(ax.get_yticklabels()[1:n + 1], ROWS):
        t.set_color(th["rows"].get(r[1], r[3]))
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.set(xlim=(x_lo, X_HI), ylim=(-0.6, n + 1.6))
    ax.set_xlabel(r"critical metallicity [$Z_{\rm cr}$/H]", fontsize=11)
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(MultipleLocator(0.25))
    ax.tick_params(axis="x", which="both", direction="in", top=True, bottom=True, labelsize=10)
    ax.tick_params(axis="x", which="major", length=5)
    ax.tick_params(axis="x", which="minor", length=2.5)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout(pad=0.3)
    name = f"transition_{c}{'' if theme == 'light' else '_' + theme}.{ext}"
    fig.savefig(OUT / name, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("wrote", OUT / name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--criterion", default="frag", choices=("frag", "C"))
    c = ap.parse_args().criterion
    use_calima_style()
    OUT.mkdir(parents=True, exist_ok=True)
    for theme in THEMES:
        figure(c, theme=theme)
    # an SVG to edit (e.g. in Illustrator): text kept as text, so no LaTeX (it would become outlines)
    use_calima_style(usetex=False, force=True)
    plt.rcParams.update({"svg.fonttype": "none", "mathtext.fontset": "dejavuserif"})
    for theme in THEMES:
        figure(c, "svg", theme)


if __name__ == "__main__":
    main()
