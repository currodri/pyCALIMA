"""Why the a-C(:H) optics use a fitted size-limited gap rather than DustEM's E_bg(a).

A grain's band gap cannot fall below a size limit E_lim(a): its aromatic clusters cannot
be larger than the grain (Jones 2012, paper III). DustEM uses this in two places, in two
different forms:

- in its charging (``CHRG_CM20``), the formula E_bg(a) = 0.2 (5 nm / a - 1) eV, kept
  within [0.1, 2.5] (the same as Jones's empirical limit 1/a - 0.2 eV);
- in its optics, implicitly: ``Q_CM20`` was computed from Jones's continuous-in-radius
  n, k, so the size limit is built into the tables at every one of its 98 radii.

pyCALIMA needs optics at intermediate H/C, which THEMIS does not tabulate, so it uses
Jones's published n, k, given at only 7 radii (0.33, 0.5, 1, 3, 10, 30, 100 nm).
Between 1 and 10 nm those tables change steeply with radius because of the size limit,
and ``jones2012_optics`` carries that steep part through E_lim(a). The figure shows:

(a) E_lim(a): DustEM's formula; the gaps that make the a-C far-IR match ``Q_CM20`` at
    each radius; the bulk gaps that best describe Jones's own tables at 1 and 3 nm; and
    the power law through them, 0.75 eV (1 nm / a)^1.06;
(b) the far-IR absorption Q_abs/a at 100 um of a-C (E_g = 0.1 eV) against radius:
    DustEM's ``Q_CM20``, and Jones's tables interpolated plainly in ln a, through
    DustEM's formula, and through the fitted law;
(c) the same as ratios to ``Q_CM20``;
(d) what the choice does to the charging: the electron affinity of a neutral 'ac'
    grain, W - E_lim(a) - e^2/2a, with each law.

    python diagnostics/dust_radiation/plot_size_limited_gap.py [--out FILE]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from pycalima.models.dust_radiation import jones2012_optics as jo
from pycalima.models.dust_radiation import themis_optics as th

FIR_UM = 100.0
LAM_FIT = np.geomspace(30.0, 1000.0, 40)
E2_EV_CM = 1.43996448e-7


def dustem_gap(a_um):
    """DustEM's E_bg(a) [eV] (DM_inout, nbg = -1)."""
    return np.clip(0.2 * (5e-3 / np.asarray(a_um, float) - 1.0), 0.1, 2.5)


def q_abs(a_um, lam_um, scheme, law=None):
    """Q_abs of a-C (E_g 0.1) with Jones's tables; ``law`` replaces E_lim(a)."""
    if law is None:
        return jo.efficiencies(a_um, lam_um, 0.1, scheme)[0]
    keep = jo.size_limited_gap
    jo.size_limited_gap = law
    try:
        return jo.efficiencies(a_um, lam_um, 0.1, scheme)[0]
    finally:
        jo.size_limited_gap = keep


def gap_matching_themis(a_um):
    """The E_lim at radius a (table radii keep the fitted law) for which the a-C far-IR
    median of Q_abs matches Q_CM20."""
    q_th = th.efficiencies(a_um, LAM_FIT)[0][:, 0]
    keep = jo.size_limited_gap

    def f(e):
        jo.size_limited_gap = lambda x: np.where(np.isclose(np.asarray(x, float), a_um), e, keep(x))
        try:
            q = jo.efficiencies(a_um, LAM_FIT, 0.1)[0][:, 0]
        finally:
            jo.size_limited_gap = keep
        return np.median(q / q_th) - 1.0

    return brentq(f, 0.1, 1.5, xtol=1e-3)


def table_gap(r):
    """Bulk gap whose 100 nm table best matches the E_g = 0.1 table at radius index r (far-IR ln k)."""
    lam, n, k = jo.load()
    fir = (lam > 30.0) & (lam < 1000.0)
    target = np.log(k[r, fir, jo.E_GAPS.index(0.1)])
    trial = np.linspace(0.1, 2.0, 381)
    err = [np.mean((jo._nk_at(len(jo.RADII_NM) - 1, e)[1][fir] - target) ** 2) for e in trial]
    return trial[int(np.argmin(err))]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="results/optics/size_limited_gap.png")
    args = ap.parse_args(argv)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pycalima.plotting_style import use_calima_style

    a_um = np.geomspace(1e-3, 1e-2, 46)
    a_nm = a_um * 1e3
    fit_nm = np.array([1.25, 1.5, 2.0, 2.5, 4.0, 5.0, 7.0])       # between the tables (at them the tables are exact)
    fit_gap = np.array([gap_matching_themis(x * 1e-3) for x in fit_nm])
    nodes = {R: table_gap(jo.RADII_NM.index(R)) for R in (1.0, 3.0)}
    print("gaps matching Q_CM20:", dict(zip(fit_nm, np.round(fit_gap, 3))))
    print("best bulk gaps of the tables:", {k: round(float(v), 3) for k, v in nodes.items()})

    lam = np.array([FIR_UM])
    q_th = th.efficiencies(a_um, lam)[0][0]
    curves = {
        "Jones tables, plain ln-$a$ interpolation": q_abs(a_um, lam, "lna")[0],
        "Jones tables through DustEM's $E_{\\rm bg}(a)$": q_abs(a_um, lam, "effective_gap", dustem_gap)[0],
        "Jones tables through the fitted $E_{\\rm lim}(a)$": q_abs(a_um, lam, "effective_gap")[0],
    }
    styles = {k: s for k, s in zip(curves, ("C0--", "C1-.", "C2-"))}
    q_nodes = jo.efficiencies(np.array([1e-3, 3e-3, 1e-2]), lam, 0.1)[0][0]

    use_calima_style()
    fig, axs = plt.subplots(2, 2, figsize=(11, 8.4))
    ax = axs[0, 0]
    aa = np.geomspace(0.7, 20.0, 200)
    ax.plot(aa, dustem_gap(aa * 1e-3), "C1-.", label=r"DustEM charging: $0.2\,(5\,{\rm nm}/a-1)$ eV")
    ax.plot(aa, jo.size_limited_gap(aa * 1e-3), "C2-", label=r"fitted: $0.75\,(1\,{\rm nm}/a)^{1.06}$ eV")
    ax.plot(fit_nm, fit_gap, "ko", ms=5, label=r"gap that makes Jones's FIR match $Q_{\rm CM20}$")
    ax.plot(list(nodes), list(nodes.values()), "rs", ms=7, mfc="none", mew=1.5,
            label=r"best bulk gap of Jones's own tables")
    ax.axhline(0.1, color="0.6", lw=0.8)
    ax.text(12, 0.11, r"bulk a-C, $E_g = 0.1$ eV", fontsize=10, color="0.4")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$a$ [nm]", fontsize=14)
    ax.set_ylabel(r"size-limited gap $E_{\rm lim}$ [eV]", fontsize=14)
    ax.set_title("(a) the size limit of the band gap", fontsize=12)
    ax.legend(frameon=False, fontsize=9)

    ax = axs[0, 1]
    ax.plot(a_nm, q_th / (a_um * 1e-4), "k-", lw=2.2, label=r"DustEM $Q_{\rm CM20}$ (THEMIS, 98 radii)")
    for k, q in curves.items():
        ax.plot(a_nm, q / (a_um * 1e-4), styles[k], label=k)
    ax.plot([1, 3, 10], q_nodes / np.array([1e-7, 3e-7, 1e-6]), "rs", ms=7, mfc="none", mew=1.5,
            label="Jones's tabulated radii")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$a$ [nm]", fontsize=14)
    ax.set_ylabel(r"$Q_{\rm abs}/a$ at 100 $\mu$m [cm$^{-1}$]", fontsize=14)
    ax.set_title(r"(b) far-IR absorption of a-C ($E_g$ = 0.1 eV)", fontsize=12)
    ax.legend(frameon=False, fontsize=8.5)

    ax = axs[1, 0]
    for k, q in curves.items():
        ax.plot(a_nm, q / q_th, styles[k], label=k)
    ax.axhline(1.0, color="k", lw=0.8)
    for R in (1, 3, 10):
        ax.axvline(R, color="r", lw=0.6, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel(r"$a$ [nm]", fontsize=14)
    ax.set_ylabel(r"$Q_{\rm abs}$ / $Q_{\rm CM20}$ at 100 $\mu$m", fontsize=14)
    ax.set_title(r"(c) the same, against DustEM's tables (dotted: Jones's radii)", fontsize=12)
    ax.legend(frameon=False, fontsize=9)

    ax = axs[1, 1]
    W = 4.75
    for law, st, lab in ((dustem_gap, "C1-.", r"with DustEM's $E_{\rm bg}(a)$ (current `ac')"),
                         (jo.size_limited_gap, "C2-", r"with the fitted $E_{\rm lim}(a)$")):
        ea = W - np.maximum(0.1, law(a_um)) - 0.5 * E2_EV_CM / (a_um * 1e-4)
        ax.plot(a_nm, ea, st, label=lab)
    ax.set_xscale("log")
    ax.set_xlabel(r"$a$ [nm]", fontsize=14)
    ax.set_ylabel(r"electron affinity of a neutral grain [eV]", fontsize=14)
    ax.set_title("(d) what the choice does to the charging", fontsize=12)
    ax.legend(frameon=False, fontsize=9)
    for ax in axs.flat:
        ax.minorticks_on()
        ax.tick_params(which="both", axis="both", direction="in")
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
