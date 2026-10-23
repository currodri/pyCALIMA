#!/usr/bin/env python3
"""Photoelectric yields of neutral grains against WDB06 Figs. 1-5.

Figs. 1-2: bulk yield y0 (WD01, eq. 9, adopted), carbonaceous and silicate.
Fig. 3: primary, Auger and secondary yields, carbonaceous a = 0.1 um.
Fig. 4: total yield, carbonaceous a = 0.001, 0.01, 0.1, 1 um.
Fig. 5: as Fig. 4 for silicate.

With ``--pdf`` pointing at the arXiv preprint (astro-ph/0601296), the figure
pages are rendered with pdftoppm and our curves (colour) are drawn over the
paper's (black). The pixel calibration below was measured on that preprint at
300 dpi from the frame lines and major ticks; it will not fit another edition.
"""
import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from pycalima import _paths
from pycalima.models.dust_charge import wdb06_yields as wy

COLORS = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100')  # categorical slots 1-4
SIZES_UM = (0.001, 0.01, 0.1, 1.0)

# page, frame (top, bottom, left, right) in px at 300 dpi, and the data values
# at the frame edges: (x_left, x_right, y_bottom, y_top). Edges are from the
# major ticks: Fig. 3 x ticks 10/100/1000 eV at 624/1132/1640 px, y tick 0.1 at
# 1565.5 px; Figs. 4-5 y ticks 1 and 0.1 at 1126/1754.5 (Fig. 5: 1159/1787.5).
FIGS = {
    1: dict(page=20, frame=(748, 2380, 514, 2145),
            lim=(10 ** (1 - 168 / 473), 10 ** (4 + 45 / 473),
                 10 ** (-5 - 285.5 / 287), 10 ** (-1 + 198.5 / 287))),
    2: dict(page=21, frame=(748, 2380, 514, 2145),
            lim=(10 ** (1 - 97 / 495.5), 10 ** (4 + 47.5 / 495.5),
                 10 ** (-4 - 379 / 380.5), 10 ** (-1 + 112 / 380.5))),
    3: dict(page=22, frame=(749, 2381, 514, 2145),
            lim=(10 ** (1 - 110 / 508), 10 ** (3 + 505 / 508), 1e-2, 1.0)),
    4: dict(page=23, frame=(749, 2381, 514, 2145),
            lim=(10 ** (1 - 110 / 508), 10 ** (3 + 505 / 508),
                 10 ** (-1 - 626.5 / 628.5), 10 ** (377 / 628.5))),
    5: dict(page=24, frame=(782, 2414, 514, 2145),
            lim=(10 ** (1 - 78 / 518.25), 10 ** (3 + 516.5 / 518.25),
                 10 ** (-1 - 626.5 / 628.5), 10 ** (377 / 628.5))),
}


def dielectric_la(E, material):
    """WD01 photon attenuation length [cm] on the E grid (used below 20 eV)."""
    return wy.dielectric_attenuation_length(E, material)


def components(E, a_um, material):
    comps = wy.yield_components(E, [0], a_um * 1e-4, material, la_wd01=dielectric_la(E, material))
    total = lambda pred: sum(Y[:, 0] for k, (Y, _) in comps.items() if pred(k))
    return {
        'primary': total(lambda k: k == 'band' or k.startswith('inner:')),
        'Auger': total(lambda k: k.startswith('auger:')),
        'secondary': total(lambda k: k.startswith('sec:')),
        'total': total(lambda k: True),
    }


def bulk_yields(E, material):
    """(WD01, eq. 9, adopted) bulk yields y0 at hnu = Theta + W (WDB06 Figs. 1-2)."""
    m = wy.MATERIALS[material]
    theta = E - m['W']
    le = wy.electron_escape_length(np.maximum(theta, 0.0), material)
    la = 1.0 / np.maximum(wy.absorption_coefficient(E, material), 1e-300)
    eq9 = np.where(theta > 0, wy._slab_escape(wy.absorption_coefficient(E, material, m['band']), le, la), 0.0)
    return m['y0_wd01'](theta, m['W']), eq9, wy.band_bulk_yield(theta, material)


def render_pages(pdf, pages, workdir):
    out = {}
    for p in pages:
        stem = Path(workdir) / f'p{p}'
        subprocess.run(['pdftoppm', '-f', str(p), '-l', str(p), '-r', '300', '-gray',
                        '-png', '-singlefile', str(pdf), str(stem)], check=True)
        out[p] = stem.with_suffix('.png')
    return out


def draw_panel(ax, fig_no, curves, page_png=None, title=''):
    import matplotlib.image as mpimg

    spec = FIGS[fig_no]
    x0, x1, y0, y1 = (np.log10(v) for v in spec['lim'])
    if page_png is not None:
        t, b, l, r = spec['frame']
        img = mpimg.imread(page_png)[t:b + 1, l:r + 1]
        ax.imshow(img, cmap='gray', extent=(x0, x1, y0, y1), aspect='auto',
                  vmin=0, vmax=1, alpha=0.75, zorder=0)
    for (label, E, Y), color in zip(curves, COLORS):
        ok = Y > 0
        ax.plot(np.log10(E[ok]), np.log10(Y[ok]), color=color, lw=2, label=label, zorder=2)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    xt = np.arange(np.ceil(x0), np.floor(x1) + 1)
    yt = np.arange(np.ceil(y0), np.floor(y1) + 1)
    ax.set_xticks(xt, [f'$10^{{{int(v)}}}$' for v in xt])
    ax.set_yticks(yt, [f'$10^{{{int(v)}}}$' for v in yt])
    ax.set_xlabel(r'$h\nu$ (eV)')
    ax.set_ylabel('Y')
    ax.set_title(title, fontsize=10)
    ax.legend(frameon=True, framealpha=0.9, fontsize=8, loc='lower right')


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--pdf', type=Path, help='WDB06 arXiv preprint, to overlay the figures')
    args = ap.parse_args()

    E = np.geomspace(6.0, 1e4, 3000)
    c01 = components(E, 0.1, 'graphite')
    fig3 = [(k, E, c01[k]) for k in ('primary', 'Auger', 'secondary')]
    fig4 = [(f'a = {a:g} um', E, components(E, a, 'graphite')['total']) for a in SIZES_UM]
    fig5 = [(f'a = {a:g} um', E, components(E, a, 'silicate')['total']) for a in SIZES_UM]

    bulk = {}
    for n, mat in ((1, 'graphite'), (2, 'silicate')):
        Eb = np.geomspace(wy.MATERIALS[mat]['W'] + 1e-3, 1e4, 3000)
        wd, eq9, adopted = bulk_yields(Eb, mat)
        bulk[n] = [('WD01', Eb, wd), ('eq. 9', Eb, eq9), ('adopted', Eb, adopted)]

    with tempfile.TemporaryDirectory() as tmp:
        pages = render_pages(args.pdf, [FIGS[n]['page'] for n in (1, 2, 3, 4, 5)], tmp) if args.pdf else {}
        fig, axes = plt.subplots(1, 2, figsize=(11, 5.6))
        for ax, n, title in zip(axes, (1, 2), ('Fig. 1: carbonaceous bulk yield',
                                               'Fig. 2: silicate bulk yield')):
            draw_panel(ax, n, bulk[n], pages.get(FIGS[n]['page']), title)
            ax.set_ylabel(r'$y_0$')
        fig.suptitle('pyCALIMA bulk yields (colour) over WDB06 Figs. 1-2 (black)', fontsize=11)
        fig.tight_layout()
        out1 = _paths.get_plots_dir() / 'wdb06_yields_figs1-2.png'
        out1.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out1, dpi=130)
        plt.close(fig)
        print('Saved', out1)

        fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
        for ax, n, curves, title in zip(
                axes, (3, 4, 5), (fig3, fig4, fig5),
                ('Fig. 3: carbonaceous, a = 0.1 um', 'Fig. 4: carbonaceous, total',
                 'Fig. 5: silicate, total')):
            draw_panel(ax, n, curves, pages.get(FIGS[n]['page']), title)
        fig.suptitle('pyCALIMA WDB06 yields (colour) over WDB06 Figs. 3-5 (black), neutral grains',
                     fontsize=11)
        fig.tight_layout()
        out = _paths.get_plots_dir() / 'wdb06_yields_figs3-5.png'
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=130)
        plt.close(fig)
    print('Saved', out)


if __name__ == '__main__':
    main()
