#!/usr/bin/env python3
"""Which time window Malloci+07's spectra carry, and whether it explains Octopus's 1-5 eV deficit against them.

A real-time spectrum is the sine transform of the kicked dipole d(t) times a window D(t), so the inverse transform
of Im alpha ~ sigma / E gives d(t) D(t) back. Malloci, Joblin & Mulas (2007, Chem. Phys. 332, 353) give T = 20
hbar/eV but no window. Their D(t) is recovered here against our undamped d(t), as rms(Malloci) / rms(ours) in
1 hbar/eV windows (phase-insensitive, as the two calculations' bands do not sit at exactly the same energies). The
method is checked by recovering our own polynomial window from our polynomial-damped spectrum.

Forward test: our multipoles transformed with exp(-gamma t), cut at our run's T, against Malloci's signal cut at
the same T (so both are cut at the same time and differ only in the window), for the Mathis+83 1-5 eV absorption
rate and oscillator strength in bands.

Result (2026-10-24, coronene 0 at T = 10, +1 at T = 14.6):
- the self-test recovers the polynomial window to 0.03 (neutral) and 0.02 (cation);
- Malloci/ours decays as exp(-gamma t), gamma = 0.21 eV (neutral) and 0.15 eV (cation), and their signal is cut
  at t = 20 (rms 6.9x lower at t = 21-23 than at 19-21 for the cation; 2.5x for the neutral, already near zero):
  an exponential window (Lorentzian bands of HWHM gamma), not the polynomial one assumed so far;
- forward test: with exp(-0.15 t) the cation matches Malloci within 4 % in every band from 0.5 to 30 eV
  (polynomial: 0.12x at 4.2-5 eV). The neutral with exp(-0.20 t): Mathis 1-5 eV rate 0.93x (polynomial 0.75x),
  f 0.5-2.5 eV 0.83x (0.009x), f 4.2-5 eV 0.91x (0.17x); exp(-0.25 t) gives 0.97 / 1.03 / 1.11x.
  The 1-5 eV deficit is Malloci's window, not our runs. Their floor below the first band is the Lorentzian wings of
  that window; it is not the molecule's absorption.

    check_damping.py [--root ~/Documents/RAMSES_dev/pahspec/results/runs] [--out results/plots/pahchem/octopus]
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_progress as P  # noqa: E402

from pycalima.pahchem import fields  # noqa: E402
from pycalima.pahchem import malloci2007 as M  # noqa: E402
from pycalima.plotting_style import use_calima_style  # noqa: E402

RUNS = [("coronene_0_h0.25", "coronene", 0), ("coronene_1_h0.25", "coronene", 1)]
GAMMAS = (0.05, 0.10, 0.15, 0.20, 0.25)     # exponential damping rates tried [eV]
T_SIG = np.linspace(0.0, 40.0, 4001)       # hbar/eV
BANDS = ((0.5, 2.5), (2.5, 4.2), (4.2, 5.0), (5.0, 13.6), (13.6, 30.0))


def signal(E, s, t=T_SIG):
    """d(t) D(t) up to a constant: int (sigma / E) sin(E t) dE."""
    g = np.where(E > 0, s / np.where(E > 0, E, 1.0), 0.0)
    return np.trapezoid(g * np.sin(np.outer(t, E)), E, axis=1)


def spectrum(sig, T, E=P.E_GRID[1:], t=T_SIG):
    """sigma(E) of a signal from ``signal``, cut at T: the inverse of ``signal``."""
    k = t <= T + 1e-9
    return E * (2 / np.pi) * np.trapezoid(sig[k] * np.sin(np.outer(E, t[k])), t[k], axis=1)


def rms_ratio(a, b, T, width=1.0):
    """(window centres, rms(a) / rms(b)) in windows of ``width`` up to T."""
    lo = np.arange(0.0, T - width + 1e-9, width)
    r = [np.sqrt(np.mean(a[(T_SIG >= x) & (T_SIG < x + width)] ** 2) /
                 np.mean(b[(T_SIG >= x) & (T_SIG < x + width)] ** 2)) for x in lo]
    return lo + width / 2, np.array(r)


MATHIS = fields.mathis1983(1.0, e_min=0.5)


def mathis_rate(E, s, lo=1.0, hi=5.0):
    w = (E >= lo) & (E <= hi)
    return MATHIS.photon_rate(s[w], E[w])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=P.ROOT, help="local mirror of megatron's ~/pahspec")
    ap.add_argument("--out", type=Path, default=P.OUT)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    use_calima_style()
    S, R = M.spectra(), P.runs(a.root)
    fig, axes = plt.subplots(2, len(RUNS), figsize=(6.5 * len(RUNS), 9), gridspec_kw={"hspace": 0.25})
    for col, (run, mol, z) in enumerate(RUNS):
        d, nc = R[run]["dir"], S[mol]["nc"]
        E, s_rect, T = P.octopus_sigma(d, damp=lambda t, T: np.ones_like(t))
        _, s_poly, _ = P.octopus_sigma(d)
        Em, sm = S[mol]["sigma"][z]
        ours, mal = signal(E, s_rect), signal(Em, sm)

        tc, r_self = rms_ratio(signal(E, s_poly), ours, T)
        _, r_mal = rms_ratio(mal, ours, T)
        gam = -np.sum(tc * np.log(r_mal)) / np.sum(tc ** 2)            # exp(-gamma t), through 1 at t = 0
        tail = [np.sqrt(np.mean(mal[(T_SIG >= x) & (T_SIG < x + 2)] ** 2)) for x in (19.0, 21.0)]
        print(f"{run} (T = {T:.1f} hbar/eV)")
        print(f"  window self-test: max |recovered - polynomial| = {np.max(np.abs(r_self - P.damping(tc, T))):.3f}")
        print(f"  Malloci / ours: exp(-gamma t) with gamma = {gam:.3f} eV; "
              f"their rms at t = 19-21 / 21-23: {tail[0] / tail[1]:.1f}")

        ax = axes[0, col]
        ax.plot(tc, r_self, "o", color=P.COLORS[0], ms=8, mfc="white", label="ours, polynomial (self-test)")
        ax.plot(tc, P.damping(tc, T), "-", color=P.COLORS[0], lw=2, label="polynomial window")
        ax.plot(tc, r_mal, "s", color=P.COLORS[1], ms=8, label="Malloci+07 / ours")
        ax.plot(tc, np.exp(-gam * tc), "--", color=P.COLORS[1], lw=2,
                label=rf"$\exp(-\gamma t)$, $\gamma = {gam:.2f}$ eV")
        ax.set_xlabel(r"$t$ [$\hbar$/eV]", fontsize=14)
        ax.set_ylabel(r"recovered window $D(t)$", fontsize=14)
        ax.set_title(f"{P.tex(mol)} $Z = {z:+d}$, $T = {T:.1f}$ " + r"$\hbar$/eV", fontsize=13)
        ax.set_ylim(0, 1.08)
        ax.legend(frameon=False, fontsize=9)
        P.style(ax)

        # forward test: both signals cut at our T, ours with exp(-gamma t)
        s_mal_T = spectrum(mal, T)
        Eg = P.E_GRID[1:]
        ref = {"rate": mathis_rate(Eg, s_mal_T), **{b: M.oscillator_strength(Eg, s_mal_T, *b) for b in BANDS}}
        print(f"  forward test against Malloci cut at T = {T:.1f} (ratios ours / Malloci):")
        print("    window         Mathis 1-5  " + "  ".join(f"f {lo:g}-{hi:g}" for lo, hi in BANDS))
        cand = [("polynomial", s_poly)] + [(f"exp {g:.2f} eV", P.octopus_sigma(d, damp=lambda t, T, g=g:
                                                                              np.exp(-g * t))[1]) for g in GAMMAS]
        for name, s in cand:
            s = s[1:]
            row = [mathis_rate(Eg, s) / ref["rate"]] + [M.oscillator_strength(Eg, s, *b) / ref[b] for b in BANDS]
            print(f"    {name:13s}  " + "  ".join(f"{v:10.3f}" for v in row))

        ax = axes[1, col]
        g_best = min(GAMMAS, key=lambda g: abs(g - gam))
        s_best = P.octopus_sigma(d, damp=lambda t, T, g=g_best: np.exp(-g * t))[1][1:]
        ax.plot(Em, sm / nc * 1e18, "-", color="0.55", lw=1.5, label="Malloci+07 as served ($T = 20$)")
        ax.plot(Eg, s_mal_T / nc * 1e18, "-", color="k", lw=2, label=f"Malloci+07 cut at $T = {T:.1f}$")
        ax.plot(Eg, s_poly[1:] / nc * 1e18, "-", color=P.COLORS[0], lw=2, label="Octopus, polynomial")
        ax.plot(Eg, s_best / nc * 1e18, "--", color=P.COLORS[1], lw=2,
                label=rf"Octopus, $\exp(-{g_best:.2f}\,t)$")
        ax.set_yscale("log")
        ax.set_xlim(0.5, 7.0)
        ax.set_ylim(1e-3, 50)
        ax.set_xlabel("$E$ [eV]", fontsize=14)
        ax.set_ylabel(r"$\sigma/N_{\rm C}$ [Mb]", fontsize=14)
        ax.legend(frameon=False, fontsize=9, loc="lower right")
        P.style(ax)
    fig.savefig(a.out / "damping.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"figure: {a.out / 'damping.png'}")


if __name__ == "__main__":
    main()
