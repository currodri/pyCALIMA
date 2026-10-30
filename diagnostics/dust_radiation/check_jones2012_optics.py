"""Jones (2012) optEC(s)(a) grain optics against DustEM's THEMIS Q_CM20.

Below 20 nm the CM20 grains are a-C with E_g = 0.1 eV, so the a-C column of the
Jones (2012) size-dependent tables (``jones2012_optics``) and DustEM's Q_CM20
(``themis_optics``) describe the same grains. THEMIS used slightly modified n, k
(Jones et al. 2013, App. C), so the two differ where those changes act. This script
prints Q_abs(Jones) / Q_abs(CM20) per radius and wavelength band, for both size
interpolations of ``jones2012_optics`` (the effective-gap scheme, default, and plain
ln-a interpolation of the 7 tabulated radii), the Henke splice at 0.02 um, and, for the
H-rich a-C:H (E_g = 2.5 eV) that has no THEMIS counterpart, the 3.4 um band against its
continuum, and plots them.

Needs $CALIMA_DUSTEM_DIR and the dataset optical-jones-2012-size
(``calima-fetch-data fetch optical-jones-2012-size``).

    python diagnostics/dust_radiation/check_jones2012_optics.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pycalima import _paths
from pycalima.models.dust_radiation import jones2012_optics as jo
from pycalima.models.dust_radiation import themis_optics as th

BANDS = (("FUV 0.09-0.2", 0.0912, 0.2), ("bump 0.2-0.25", 0.2, 0.25), ("optical 0.3-1", 0.3, 1.0),
         ("NIR 1-3", 1.0, 3.0), ("MIR 3-25", 3.0, 25.0), ("FIR 25-1000", 25.0, 1000.0))
RADII_UM = (3.5e-4, 5e-4, 1e-3, 1.5e-3, 2e-3, 3e-3, 5e-3, 7e-3, 1e-2, 2e-2)


def compare(lam, size_interp="effective_gap"):
    """Q_abs(Jones, E_g = 0.1) / Q_abs(CM20) on lam, (nlam, n radii)."""
    a = np.array(RADII_UM)
    return jo.efficiencies(a, lam, 0.1, size_interp)[0] / th.efficiencies(a, lam)[0]


def band_34(a_um):
    """Peak of Q_abs/a in 3.35-3.55 um over its linear 3.15-3.70 um baseline (a-C:H, E_g 2.5)."""
    lam = np.linspace(3.1, 3.75, 300)
    q = jo.efficiencies(np.atleast_1d(a_um), lam, 2.5)[0]
    lo, hi = q[np.argmin(abs(lam - 3.15))], q[np.argmin(abs(lam - 3.70))]
    base = lo + (hi - lo) * ((lam - 3.15) / 0.55)[:, None]
    win = (lam > 3.35) & (lam < 3.55)
    return (q[win] / base[win]).max(axis=0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=None, help="figure path")
    args = ap.parse_args(argv)

    t = th.load_cm20()
    lam = t.lam_um[(t.lam_um >= 0.0912) & (t.lam_um <= 1000.0)]
    for scheme in jo.SIZE_INTERP:
        rs = compare(lam, scheme)
        print(f"Q_abs(Jones 2012, E_g = 0.1 eV, size_interp={scheme!r}) / Q_abs(DustEM CM20): "
              "median [min, max] per band")
        print(f"{'a [nm]':>8s} " + " ".join(f"{b[0]:>22s}" for b in BANDS))
        for c, a in enumerate(RADII_UM):
            cells = []
            for _, l0, l1 in BANDS:
                x = rs[(lam >= l0) & (lam < l1), c]
                cells.append(f"{np.median(x):.3f} [{x.min():.2f}, {x.max():.2f}]")
            print(f"{a * 1e3:8.2f} " + " ".join(f"{s:>22s}" for s in cells))
        print()
    print("size-limited gap E_lim [eV]:", {round(a * 1e3, 2): round(float(jo.size_limited_gap(a)), 3)
                                          for a in RADII_UM})
    r = compare(lam)
    a = np.array(RADII_UM)
    print("\nHenke / table Q_abs at 0.02 um, E_g = 0.1:", np.round(jo.splice_ratio(a, 0.1), 3))
    print("Henke / table Q_abs at 0.02 um, E_g = 2.5:", np.round(jo.splice_ratio(a, 2.5), 3))
    b34 = band_34(a)
    print("a-C:H (E_g 2.5) 3.4 um band, peak / continuum:", np.round(b34, 2))
    plot(lam, r, a, args.out)


def plot(lam, r, a, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pycalima.plotting_style import use_calima_style

    use_calima_style()
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axs[0]
    for c, ai in enumerate(a):
        ax.plot(lam, r[:, c], lw=1, label=fr"{ai * 1e3:.2g} nm")
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\lambda$ [$\mu$m]")
    ax.set_ylabel(r"$Q_{\rm abs}$(Jones 2012) / $Q_{\rm abs}$(CM20), $E_g$ = 0.1 eV")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax = axs[1]
    lam2 = np.geomspace(0.03, 1000.0, 600)
    for e_gap, ls in ((0.1, "-"), (2.5, "--")):
        q = jo.efficiencies(a[[0, 2, 4, 7]], lam2, e_gap)[0]
        for c, ai in enumerate(a[[0, 2, 4, 7]]):
            ax.plot(lam2, q[:, c] / (ai * 1e-4), ls, color=f"C{c}", lw=1,
                    label=fr"$E_g$ = {e_gap} eV, {ai * 1e3:.2g} nm")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\lambda$ [$\mu$m]")
    ax.set_ylabel(r"$Q_{\rm abs}/a$ [cm$^{-1}$]")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    for ax in axs:
        ax.minorticks_on()
        ax.tick_params(which="both", axis="both", direction="in")
    fig.tight_layout()
    out = Path(out) if out else _paths.get_plots_dir() / "dust_radiation" / "check_jones2012_optics.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
