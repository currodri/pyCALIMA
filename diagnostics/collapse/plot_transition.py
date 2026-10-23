"""The Pop III / Pop II transition is a band, not a line: one small panel for a proposal.

    python diagnostics/collapse/plot_transition.py [--criterion frag]

From ``science_grid_transition.py``'s results, writes ``collapse/transition_<criterion>.pdf``
under the plots directory (and a ``.png`` preview and an editable ``.svg``, text kept as text; ``_dark`` versions for a dark
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
from matplotlib.transforms import Bbox
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
    ("dust surviving\nreverse shock", "d", grid.DUST, "C1",
     {"MW": "Milky Way", "n0": "no shock", "n0.1": "0.1", "n1": "1", "n10": r"$n_{\rm amb} = 10\,$cm$^{-3}$"}),
    ("SN progenitor", "M", grid.MASSES, "C0",
     {13: r"13$\,$M$_\odot$ (C)", 20: "20", 25: r"25$\,$M$_\odot$", 30: r"30$\,$M$_\odot$ (Si)"}),
    ("initial sizes", "s", grid.SHAPES, "C4",
     {"MRN": "MRN", "grackle": "lognormal", "N03": "N03", "N07": "N07", "eroded": "eroded"}),
    ("accretion", "p", grid.PROCESSES, "C2", {"static": "off", "acc": "on"}),     # with the n_amb = 1 dust
    ("accommodation", "a", grid.ALPHAS, "C5", {1.0: r"$\alpha_T = 1$", 0.5: "0.5", 0.3: "0.3"}),
    ("cosmic rays", "z", grid.ZETAS, "C3",
     {0.0: r"$\zeta_0 = 0$", 1e-18: r"$10^{-18}\,$s$^{-1}$", 1e-16: r"$10^{-16}\,$s$^{-1}$"}),
    ("Lyman--Werner", "lw", grid.J21S, "C8", {0.0: r"$J_{21} = 0$", 10.0: "10", 100.0: "100 (atomic)"}),
]
GROUP_AFTER = 5            # a solid line between the dust physics and the environment rows
SIZE_MM = (66.0, 54.0)     # the proposal's figure box; saved at exactly this size, to place at 100%
FS = 5.5                   # annotation font size (pt)
GAP = 0.07                 # dex between a label and its bar or the dashed line


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
        return ax.add_patch(FancyArrow(lo, y, hi - lo, 0.0, width=width, head_width=1.9 * width, head_length=0.17,
                                       length_includes_head=True, color=color, lw=0, zorder=2))
    return ax.add_patch(Rectangle((lo, y - width / 2), hi - lo, width, color=color, lw=0, zorder=2))


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
    x_lo = min(X_LO, math.floor(2.0 * (allv.min() - 0.5)) / 2.0)
    th = THEMES[theme]
    rc = {"text.color": th["fg"], "axes.edgecolor": th["fg"], "axes.labelcolor": th["fg"], "xtick.color": th["fg"],
          "ytick.color": th["fg"], "figure.facecolor": th["bg"], "axes.facecolor": th["bg"],
          "savefig.facecolor": th["bg"], "axes.linewidth": 0.6, "xtick.major.width": 0.5, "xtick.minor.width": 0.4,
          "lines.linewidth": 0.6}
    with plt.rc_context(rc):
        _draw(c, ext, theme, th, get, allv, z0, x_lo)


def _width(ax, t):
    """Width of text t in dex (the x axis is linear)."""
    bb = t.get_window_extent()
    x = ax.transData.inverted().transform([[bb.x0, 0.0], [bb.x1, 0.0]])[:, 0]
    return x[1] - x[0]


def _place(ax, labels, z0):
    """Puts each row label beside its bar, off the dashed line; returns the left edge they need."""
    need = np.inf
    for t, side, x, bar in labels:
        w = _width(ax, t)
        if side == "left":                               # right-aligned, ending GAP before the bar
            r = x - GAP
            if r - w - GAP / 2 < z0 < r + GAP / 2:        # across the dashed line: to its left
                r = z0 - GAP
            t.set(x=r, ha="right")
            need = min(need, r - w - GAP)
        else:                                            # left-aligned, starting GAP after the bar
            l = x + GAP
            if l - GAP / 2 < z0 < l + w + GAP / 2:
                l = z0 + GAP
            t.set(x=l, ha="left")
    return need


def _cut(ax, t, patch, lo, hi, open_end, width, color):
    """Writes t in a gap cut into the bar near its high end (before the arrowhead): no room right of it."""
    w = _width(ax, t)
    r = hi - (0.25 if open_end else GAP)
    t.set(x=r, ha="right")
    patch.remove()
    _bar(ax, lo, r - w - GAP / 2, t.get_position()[1], color, width)
    if open_end:
        _bar(ax, r + GAP / 2, hi, t.get_position()[1], color, width, open_end=True)
    else:
        _bar(ax, r + GAP / 2, hi, t.get_position()[1], color, width)


def _touching(fig, ax, skip):
    """Pairs (text, artist) that touch: every text in the axes against the lines, bars, frame and other texts."""
    fig.canvas.draw()
    pad = 0.4 * fig.dpi / 72.0                           # 0.4 pt clearance
    texts = [t for t in ax.texts if t.get_text()]
    others = [l for l in ax.lines] + [p for p in ax.patches if p not in skip] + list(ax.spines.values())
    frame = ax.get_window_extent()
    for minor, length in ((False, 2.5), (True, 1.3)):     # tick marks point in: their padded extents overshoot
        L = length * fig.dpi / 72.0
        for x in (ax.xaxis.get_minorticklocs() if minor else ax.xaxis.get_majorticklocs()):
            if ax.get_xlim()[0] < x < ax.get_xlim()[1]:
                X = ax.transData.transform([[x, 0.0]])[0, 0]
                others += [Bbox([[X, frame.y0], [X, frame.y0 + L]]), Bbox([[X, frame.y1 - L], [X, frame.y1]])]
    box = lambda a: a if isinstance(a, Bbox) else a.get_window_extent()
    hit = lambda a, b: a.x0 - pad < b.x1 and b.x0 < a.x1 + pad and a.y0 - pad < b.y1 and b.y0 < a.y1 + pad
    out = []
    for i, t in enumerate(texts):
        bt = box(t)
        out += [(t, o) for o in others + texts[i + 1:] if (isinstance(o, Bbox) or o.get_visible()) and hit(bt, box(o))]
    return out


def _draw(c, ext, theme, th, get, allv, z0, x_lo):
    n = len(ROWS)
    fig, ax = plt.subplots(figsize=tuple(x / 25.4 for x in SIZE_MM), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01)
    ax.set(xlim=(x_lo, X_HI), ylim=(-0.6, n + 1.6))
    note = lambda x, y, s, **k: ax.text(x, y, s, fontsize=FS, va="center", in_layout=False, **k)
    published = ax.text(x_lo, n + 1.7, "published:", color=th["fg"], fontsize=FS, ha="left", va="bottom")
    bands = []
    for ch, lab in (("dust", "dust"), ("lines", "metal lines")):
        col, alpha, tcol = th["bands"][ch]
        v = [x for x, _ in LITERATURE[ch]]
        bands.append(ax.axvspan(min(v), max(v), color=col, alpha=alpha, lw=0))
        ax.text(0.5 * (min(v) + max(v)), n + 1.7, lab, color=tcol, fontsize=FS, ha="center", va="bottom")
    ax.axvline(z0, color=th["fg"], ls="--", lw=0.6, zorder=1)
    ax.plot(z0, n + 1, "D", color=th["fg"], ms=3.5, zorder=4)
    tex = plt.rcParams["text.usetex"]                  # mathtext spaces lines tighter for the same linespacing
    note(z0 + 0.15, n + 1, "Milky Way dust/metal,\n" r"MRN, static, no CRs/LW", linespacing=0.95 if tex else 1.1)
    labels, cuts = [], []            # (text, side, bar end, bar); (text, bar, lo, hi, open end, width, colour)
    for row, (lab, key, vals, col, names) in zip(range(n, 0, -1), ROWS):
        col = th["rows"].get(key, col)
        v = np.array([get(**{**grid.row_reference(key), key: x}) for x in vals])
        fin = v[np.isfinite(v)]
        if fin.size == 0:                                 # every choice: no fragmentation
            note(X_HI - 0.05, row, "no fragmentation at " + ", ".join(names[x] for x in vals), color=col, ha="right")
            continue
        inf = np.isinf(v).any()
        lo, hi = fin.min(), (X_HI - 0.02 if inf else fin.max())
        bar = _bar(ax, lo, hi, row, col, open_end=inf)
        for x in fin:                                     # one tick per choice
            ax.plot([x, x], [row - 0.12, row + 0.12], color=th["bg"], lw=0.6, zorder=3)
        left, right = _ends(vals, v, names)
        if hi - lo < 0.02:                                # every choice gives the same [Z_cr]
            labels.append((note(hi, row, ", ".join(names[x] for x in vals) + ": no change", color=col),
                           "right", hi, bar))
            continue
        labels.append((note(lo, row, left, color=col), "left", lo, bar))
        if inf:
            cuts.append((note(hi, row, right, color=col), bar, lo, hi, True, 0.24, col))
        else:
            labels.append((note(hi, row, right, color=col), "right", hi, bar))
    ax.axhline(n + 0.5, color=th["sep"], lw=0.5)
    ax.axhline(0.5, color=th["sep"], lw=0.5)
    for y in np.arange(1.5, n, 1.0):
        ax.axhline(y, color=th["grid"], lw=0.3, ls=":")
    ax.axhline(n - GROUP_AFTER + 0.5, color=th["group"], lw=0.4)
    lo, inf = allv.min(), np.isinf(allv).any()
    hi = X_HI - 0.02 if inf else allv.max()
    bar = _bar(ax, lo, hi, 0, th["combined"], width=0.3, open_end=inf)
    labels.append((note(lo, 0, f"${lo:.1f}$"), "left", lo, bar))
    if inf:
        cuts.append((note(hi, 0, "no fragmentation"), bar, lo, hi, True, 0.3, th["combined"]))
    dash = (lambda t: t) if plt.rcParams["text.usetex"] else (lambda t: t.replace("--", "–"))
    bold = r"\textbf{all combined}" if plt.rcParams["text.usetex"] else "all combined"
    ax.set_yticks(range(n + 1, -1, -1), ["classical"] + [dash(r[0]) for r in ROWS] + [bold],
                  fontsize=6, linespacing=0.85 if tex else 1.1)
    for t, r in zip(ax.get_yticklabels()[1:n + 1], ROWS):
        t.set_color(th["rows"].get(r[1], r[3]))
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.set_xlabel(r"critical metallicity [$Z_{\rm cr}$/H]", fontsize=6.5, labelpad=1.5)
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(MultipleLocator(0.25))
    ax.tick_params(axis="x", which="both", direction="in", top=True, bottom=True, labelsize=6, pad=2)
    ax.tick_params(axis="x", which="major", length=2.5)
    ax.tick_params(axis="x", which="minor", length=1.3)
    ax.tick_params(axis="y", length=0, pad=2)
    for _ in range(4):                                   # widen the axis until the left labels fit
        fig.canvas.draw()
        need = _place(ax, labels, z0)
        if need >= ax.get_xlim()[0]:
            break
        ax.set_xlim(math.floor(2.0 * need) / 2.0, X_HI)
        published.set_x(ax.get_xlim()[0])
    for t, bar, lo, hi, open_end, width, col in cuts:
        _cut(ax, t, bar, lo, hi, open_end, width, col)
    for t, o in _touching(fig, ax, bands):
        print(f"  warning: {t.get_text()!r} touches {o.get_text() if hasattr(o, 'get_text') else o}")
    name = f"transition_{c}{'' if theme == 'light' else '_' + theme}.{ext}"
    fig.savefig(OUT / name, dpi=600)
    plt.close(fig)
    print("wrote", OUT / name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--criterion", default="frag", choices=("frag", "C"))
    c = ap.parse_args().criterion
    use_calima_style()
    OUT.mkdir(parents=True, exist_ok=True)
    for theme in THEMES:                  # a vector PDF (LaTeX, fonts embedded) and a PNG
        for ext in ("pdf", "png"):
            figure(c, ext, theme)
    # an SVG to edit (e.g. in Illustrator): text kept as text, so no LaTeX (it would become outlines)
    use_calima_style(usetex=False, force=True)
    plt.rcParams.update({"svg.fonttype": "none", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "mathtext.fontset": "stix"})     # Times, as in the PDF, so the layout matches
    for theme in THEMES:
        figure(c, "svg", theme)


if __name__ == "__main__":
    main()
