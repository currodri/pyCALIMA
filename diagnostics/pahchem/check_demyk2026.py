#!/usr/bin/env python3
"""pahchem's reproduction of the Demyk+26 single-photon emission model (pahchem.demyk2026), against their figures.

Figure ``demyk2026_reproduction`` (plots/pahchem), four panels:

(a) 6H-pyrene, 3.2-3.6 um after one 4 and 6 eV photon: their Fig. 5 inset (digitised) and ours, ours scaled
    once so that the 4 eV 3.40 um peaks agree (their units are arbitrary but common to all energies);
(b) Me-pyrene, 3.1-3.6 um, the same;
(c) the emission temperatures of the 6H-pyrene 3.4 um photons after 4 eV: their Fig. 11 (1e4 cascades) and
    ours (exact expectation, times 1e4), no scaling;
(d) T(U) of 6H-pyrene: their Fig. 10 (harmonic and anharmonic) and ours (microcanonical and canonical).

Printed: peak positions and FWHMs of the main CH bands, the photon count and mean temperature of the 3.4 um
band, and the 6.9/3.3 um ratios against their Sect. 4.3 values.

    check_demyk2026.py [--out DIR]       (needs the cosmicpah-irdb dataset: calima-fetch-data fetch cosmicpah-irdb)
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pycalima import _paths
from pycalima.pahchem import demyk2026 as D, dos

PEAK = {"6h-pyrene": 3.404, "me-pyrene": 3.288}
WINDOW = {"6h-pyrene": (3.2, 3.6), "me-pyrene": (3.1, 3.6)}
LABEL = {"6h-pyrene": "6H-pyrene", "me-pyrene": "Me-pyrene"}
RATIO_BANDS = {"6h-pyrene": (1448.8, 3028.0, 4.0, 1.08), "me-pyrene": (1448.6, 3049.9, 6.0, 0.19)}


def _house_style():
    from pycalima.plotting_style import use_calima_style

    use_calima_style(usetex=False)
    plt.rcParams.update({"font.serif": ["cmr10", "DejaVu Serif", "serif"], "mathtext.fontset": "cm",
                         "axes.formatter.use_mathtext": True, "legend.frameon": False})


def _csv(name):
    return pd.read_csv(_paths.get_external_data_path("pahchem", name), comment="#")


def f_lambda(em, lam_um):
    nu = 1e4 / lam_um
    return em.spectrum(nu) * nu**2 / 1e4


def fwhm_cm(lam, f, centre, half=0.03):
    """(FWHM [cm^-1], peak [um]) of the band nearest ``centre``, half-maximum crossings interpolated."""
    w = np.abs(lam - centre) < half
    l, y = lam[w], f[w]
    i = int(np.argmax(y))
    h = 0.5 * y[i]
    lo = next(k for k in range(i, 0, -1) if y[k - 1] < h)
    hi = next(k for k in range(i, y.size - 1) if y[k + 1] < h)
    l_lo = np.interp(h, [y[lo - 1], y[lo]], [l[lo - 1], l[lo]])
    l_hi = np.interp(h, [y[hi + 1], y[hi]], [l[hi + 1], l[hi]])
    return 1e4 / l_lo - 1e4 / l_hi, l[i]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or _paths.get_plots_dir("pahchem")
    out.mkdir(parents=True, exist_ok=True)
    _house_style()
    fig5 = _csv("demyk2026_fig5_insets.csv")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    lam = np.linspace(3.05, 3.65, 6001)
    for ax, key in zip(axes[0], ("6h-pyrene", "me-pyrene")):
        ems = {ev: D.run(key, ev) for ev in (4.0, 6.0)}
        theirs = {ev: fig5[(fig5.molecule == key) & (fig5.e_uv_ev == ev)] for ev in (4.0, 6.0)}
        p = PEAK[key]
        ref = theirs[4.0].f_lambda_au[np.abs(theirs[4.0].lam_um - p) < 0.01].max()
        s = ref / f_lambda(ems[4.0], lam)[np.abs(lam - p) < 0.01].max()
        for ev, c in ((4.0, "C0"), (6.0, "C3")):
            t = theirs[ev]
            ax.plot(t.lam_um, t.f_lambda_au, color=c, lw=2.5, alpha=0.45, label=f"D26 Fig. 5, {ev:.0f} eV")
            ax.plot(lam, s * f_lambda(ems[ev], lam), color=c, lw=1.0, ls="--", label=f"pahchem, {ev:.0f} eV")
            w_th, p_th = fwhm_cm(t.lam_um.to_numpy(), t.f_lambda_au.to_numpy(), p)
            w_us, p_us = fwhm_cm(lam, f_lambda(ems[ev], lam), p)
            print(f"{key} {ev:.0f} eV main CH band: D26 {p_th:.4f} um, FWHM {w_th:.1f} cm^-1; "
                  f"pahchem {p_us:.4f} um, FWHM {w_us:.1f} cm^-1")
        ax.set_xlim(*WINDOW[key])
        ax.set_xlabel(r"$\lambda$ [$\mu$m]")
        ax.set_ylabel(r"$F_\lambda$ per absorbed photon [D26 units]")
        ax.set_title(f"({'a' if key == '6h-pyrene' else 'b'}) {LABEL[key]}")
        ax.legend(fontsize=8)
    # (c) emission temperatures of the 3.4 um photons
    ax = axes[1, 0]
    h11 = _csv("demyk2026_fig11.csv")
    em = D.run("6h-pyrene", 4.0)
    band = em.band(2942.7)
    edges = np.arange(300, 1301, 100)
    ours, _ = np.histogram(em.temperature_k, bins=edges, weights=band.photons)
    centres = 0.5 * (edges[1:] + edges[:-1])
    ax.bar(0.5 * (h11.t_hi_k + h11.t_lo_k), h11.counts, width=100, color="0.75", edgecolor="0.4", label="D26 Fig. 11 (1e4 cascades)")
    ax.step(centres, 1e4 * ours, where="mid", color="C3", label="pahchem (expectation x 1e4)")
    ax.invert_xaxis()
    ax.set_xlabel("emission temperature [K]")
    ax.set_ylabel(r"3.4 $\mu$m photons per 100 K")
    ax.set_title(r"(c) 6H-pyrene 3.40 $\mu$m band, 4 eV")
    ax.legend(fontsize=8)
    mids = 0.5 * (h11.t_hi_k + h11.t_lo_k)
    w = band.photons / band.photons.sum()
    print(f"6H-pyrene 4 eV 3.4 um photons per UV photon: D26 {h11.counts.sum() / 1e4:.2f}, pahchem {band.photons.sum():.2f}; "
          f"mean T: D26 {np.sum(h11.counts * mids) / h11.counts.sum():.0f} K, pahchem {w @ em.temperature_k:.0f} K")
    # (d) T(U)
    ax = axes[1, 1]
    t10 = _csv("demyk2026_fig10.csv")
    u = np.linspace(0.05, 6.2, 300)
    E = u * dos.CM_PER_EV
    f6 = D.modes("6h-pyrene")
    ax.plot(t10.u_ev, t10.t_harmonic_k, color="C0", lw=2.5, alpha=0.45, label="D26 Fig. 10, harmonic")
    ax.plot(t10.u_ev, t10.t_anharmonic_k, color="C1", lw=2.5, alpha=0.45, label="D26 Fig. 10, anharmonic")
    ax.plot(u, D.temperature(f6, E, "microcanonical"), color="C0", lw=1.0, ls="--", label="pahchem, microcanonical")
    ax.plot(u, D.temperature(f6, E, "canonical"), color="C2", lw=1.0, ls=":", label="pahchem, canonical")
    ax.set_xlabel("U [eV]")
    ax.set_ylabel("T [K]")
    ax.set_title("(d) 6H-pyrene T(U)")
    ax.legend(fontsize=8)
    for key, (b69, b33, ev, target) in RATIO_BANDS.items():
        e = D.run(key, ev)
        print(f"{key} {ev:.0f} eV 6.9/3.3 um (band energies): pahchem {e.energy(b69) / e.energy(b33):.3f}, D26 Sect. 4.3 {target}")
    fig.tight_layout()
    path = out / "demyk2026_reproduction.png"
    for a in fig.axes:
        a.minorticks_on()
        a.tick_params(which="both", axis="both", direction="in")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
