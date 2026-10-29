#!/usr/bin/env python3
"""The charge-resolved PAH cross-section library, 0.02 eV to 10 keV, assembled from the finished DIaL3 states.

Per state whose three kicks have reached T = 20 (``plot_dial3_library.states``), in order
(docs/physics/pahchem-uv-absorption.md §4):

1. **TD-LDA** sigma(E) from the kicks (``plot_progress.octopus_sigma``) to 150 eV, with a Gaussian window of
   s0 = 0.15 eV. That is the narrowest width T = 20 gives cleanly (exp(-4.5) is left at the cut), and the warm-vapour
   width of the pyrene calibration. Cold molecules (s ~ 0.08 eV) would need T = 40; on coronene that changes
   pahchem's rates by <= 2.4 % (§4).
2. **Width** s(E):
   - s0 up to 7.5 eV;
   - rising to 0.8 eV at 14 eV, which removes the box artefacts of the discretised continuum. The width added to s0,
     sqrt(s^2 - s0^2), grows linearly from 7.5 eV;
   - flat to 30 eV, then rising to 4 eV at 40 eV, against the box ripples at 30-54 eV (compare_box_euv.py);
   - flat above.
   The width beyond s0 comes from diffusion in energy, d sigma / d tau = d/dE (D d sigma / dE), to tau = 1 with
   D = (s^2 - s0^2) / 2. At constant width that is the Gaussian convolution. It conserves the oscillator strength and
   keeps sigma >= 0. Unlike a convolution whose kernel width follows the source energy, it does not drift strength
   down the width gradient. That drift (w w' sigma) piled strength into the grid point where the width starts, and
   pushed it below 13.6 eV. The profile is empirical and labelled as such (the no-ad-hoc-factors rule).
3. **Positions** stay TD-LDA's: the sTD-DFT correction below ~6 eV is not designed yet (§5).
4. **EUV and X-rays:** the atoms' photoionisation cross sections, Verner phfit2 C 1s + 2s + 2p (``wdb06_atomic``)
   and hydrogenic H 1s, with the valence part normalised to TD-LDA.
   - TD-LDA is kept to 80 eV. Its excess over the atoms at 30-54.4 eV (x1.33) is real: measured benzene shows x1.30
     (Feng+ dipole (e,e), ``--benzene``).
   - Above ~80 eV both TD-LDA (x0.84-0.92) and benzene (x0.78-0.79, flat to 200 eV) lie below the atomic sum, by a
     factor that does not depend on energy. The atomic valence shape is therefore scaled by k, TD-LDA's integral over
     the atoms' at 80-130 eV (0.88-0.89 for every state), and blended in with a cos^2 weight over the same range. The
     result is continuous with Octopus by construction. The unscaled atoms would put 100-290 eV about 13 % above
     TD-LDA and 27 % above benzene.
   - C 1s is added unscaled from Verner's 291 eV step. The pi* resonance at 285 eV that molecules and graphite show
     is missing.

Checks per state go to stdout and ``checks.csv``:
- the oscillator strength to 13.6 and 30 eV, final over raw (the §4 guard asks < 0.5 % at 30 eV);
- the most negative sigma over the maximum (guard >= -1e-3);
- f to 54.4 eV, 130 eV and 10 keV, against the valence electrons and the C 1s electrons;
- the valence scale k;
- rates final over raw in three fields: Mathis+83 (0.1-13.6 eV), Draine 1978 (5-13.6 eV) and a Kurucz 40 000 K star
  (13.6-136 eV). Also the mean absorbed Mathis photon energy. Raw here is Octopus's own spectrum (polynomial window,
  T = 20).

Output:
- in ``--out``: ``pah_sigma_library.npz`` (energy_ev, label, nc, nh, z, sigma_cm2 [state, energy], the parameters),
  one ``sigma_<state>.dat`` per state, and ``checks.csv``;
- in ``--plots``: ``final_uv.png``, ``final_broadband.png``, ``final_stages.png`` (each stage for two states), and with
  ``--benzene``, ``final_vs_benzene.png``. That figure plots every state over the atomic sum against measured benzene:
  a two-column file of wavelength [nm] and sigma [cm^2], e.g. Feng+ 2002.

    build_library.py [--root ~/Documents/RAMSES_dev/pahspec/results/dial3_library] [--out <results>/pahchem/...]
                     [--benzene <bz_feng.dat>]
"""
import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.linalg import solve_banded  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_dial3_library import ROOT, finish_panels, isrf_stats, sigma_atomic, state_panels, states  # noqa: E402
from plot_progress import COLORS, octopus_sigma, style, tex  # noqa: E402

from pycalima import _paths  # noqa: E402
from pycalima.models.dust_charge import wdb06_atomic as W  # noqa: E402
from pycalima.pahchem import fields  # noqa: E402
from pycalima.pahchem import malloci2007 as M  # noqa: E402
from pycalima.plotting_style import use_calima_style  # noqa: E402

S0, S_MID, S_TOP = 0.15, 0.8, 4.0          # eV: the width below 7.5 eV, at 14-30 eV, and above 40 eV
E_JOIN = (80.0, 130.0)                      # eV: the cos^2 hand-over to the scaled atoms, and where k is set
E_TD = np.round(np.arange(0.02, 150.0 + 1e-9, 0.02), 10)
E_1S = W.shell_threshold_eV("C", "1s")
E_HI = np.unique(np.concatenate([np.geomspace(E_JOIN[1], 1e4, 401)[1:], [E_1S * (1 - 1e-6), E_1S]]))
E_OUT = np.concatenate([E_TD[E_TD <= E_JOIN[1] + 1e-9], E_HI])
CHARGE_COLOR = {-1: COLORS[0], 0: COLORS[1], 1: COLORS[2], 2: COLORS[3]}


def s_of_e(E):
    """The width s(E) [eV] of the final library (stage 2)."""
    added = np.sqrt(S_MID ** 2 - S0 ** 2) * np.clip((E - 7.5) / 6.5, 0.0, 1.0)
    return np.sqrt(S0 ** 2 + added ** 2) + (S_TOP - S_MID) * np.clip((E - 30.0) / 10.0, 0.0, 1.0)


def widen(E, s, steps=400):
    """``s`` (on the uniform grid ``E``, already at width S0) widened to s(E): conservative diffusion with
    D = (s(E)^2 - S0^2) / 2 to tau = 1, implicit Euler in ``steps`` steps, zero flux at the grid ends. The matrix is
    symmetric with unit row sums, so the integral is kept exactly; it is an M-matrix, so sigma stays >= 0."""
    dE = E[1] - E[0]
    D = 0.5 * np.maximum(s_of_e(E) ** 2 - S0 ** 2, 0.0)
    r = 0.5 * (D[1:] + D[:-1]) / dE ** 2 / steps
    ab = np.zeros((3, len(E)))
    ab[0, 1:], ab[2, :-1], ab[1] = -r, -r, 1.0
    ab[1, :-1] += r
    ab[1, 1:] += r
    out = np.asarray(s, dtype=float).copy()
    for _ in range(steps):
        out = solve_banded((1, 1), ab, out)
    return out


def sigma_atoms(nc, nh, E):
    """Valence (C 2s + 2p, H 1s; ``plot_dial3_library.sigma_atomic``) plus C 1s photoionisation [cm^2]."""
    return sigma_atomic(nc, nh, E) + sigma_1s(nc, E)


def sigma_1s(nc, E):
    return nc * W.photoionisation_cross_section("C", "1s", E)


def join_weight(E):
    """TD-LDA's weight in the hand-over: 1 below E_JOIN[0], cos^2 down to 0 at E_JOIN[1]."""
    x = np.clip((np.asarray(E) - E_JOIN[0]) / (E_JOIN[1] - E_JOIN[0]), 0.0, 1.0)
    return np.cos(0.5 * np.pi * x) ** 2


def build(run, nc, nh):
    """Stages 1-4 of one state: {stage: (E, sigma [cm^2])}, and the valence scale k."""
    _, raw, _ = octopus_sigma(run, energies=E_TD)
    _, gauss, _ = octopus_sigma(run, damp=lambda t, T: np.exp(-0.5 * (S0 * t) ** 2), energies=E_TD)
    wide = widen(E_TD, gauss)
    k = (E_TD >= E_JOIN[0]) & (E_TD <= E_JOIN[1])
    scale = np.trapezoid(wide[k], E_TD[k]) / np.trapezoid(sigma_atomic(nc, nh, E_TD[k]), E_TD[k])
    low = E_OUT <= E_JOIN[1] + 1e-9
    wt = np.where(low, join_weight(E_OUT), 0.0)
    td = np.interp(E_OUT, E_TD, wide, right=0.0)
    scaled = scale * sigma_atomic(nc, nh, E_OUT) + sigma_1s(nc, E_OUT)
    final = wt * td + (1 - wt) * scaled
    return {"raw": (E_TD, raw), "gauss": (E_TD, gauss), "wide": (E_TD, wide),
            "atoms": (E_OUT, sigma_atoms(nc, nh, E_OUT)), "scaled": (E_OUT, scaled), "final": (E_OUT, final)}, scale


def rate(field, E, s):
    """Absorbed photons [s^-1] of ``field`` over its own grid, negative ringing clipped."""
    e = field.energy_ev
    return np.trapezoid(np.clip(np.interp(e, E, s, left=0.0, right=0.0), 0.0, None) * field.flux_at(e), e)


def checks(label, nc, nh, z, st, scale):
    E, s = st["final"]
    Er, r = st["raw"]
    lo = E <= E_JOIN[1]
    mathis = fields.mathis1983(e_min=0.1, n=2000)
    d78 = fields.draine1978()
    star = fields.kurucz(40000, 1.0, e_max=136.0, e_min=13.6)
    n_val = M.valence_electrons(nc, nh, z)
    return {
        "state": f"{label}_{z:+d}", "nc": nc, "nh": nh, "z": z,
        "f13.6_final_over_raw": M.oscillator_strength(E, s, 0, 13.6) / M.oscillator_strength(Er, r, 0, 13.6),
        "f30_final_over_raw": M.oscillator_strength(E, s, 0, 30) / M.oscillator_strength(Er, r, 0, 30),
        "min_over_max": s[lo].min() / s[lo].max(),
        "f54_over_valence": M.oscillator_strength(E, s, 0, 54.4) / n_val,
        "f130_over_valence": M.oscillator_strength(E, s, 0, 130) / n_val,
        "f10keV_over_valence_plus_1s": M.oscillator_strength(E, s) / (n_val + 2 * nc),
        "valence_scale_k": scale,
        "mathis_final_over_raw": rate(mathis, E, s) / rate(mathis, Er, r),
        "draine78_final_over_raw": rate(d78, E, s) / rate(d78, Er, r),
        "kurucz40k_final_over_raw": rate(star, E, s) / rate(star, Er, r),
        "mathis_mean_photon_ev": isrf_stats(E, s)[0],
    }


def molecules(rows):
    """[(label, nc, nh, [(z, stages)])] in the library's order."""
    out = {}
    for label, nc, nh, z, st, _ in rows:
        out.setdefault(label, (label, nc, nh, []))[3].append((z, st))
    return list(out.values())


def fig_uv(mols, path):
    fig, axes = state_panels(len(mols))
    for ax, (label, nc, nh, zs) in zip(axes.flat, mols):
        for z, st in zs:
            E, s = st["final"]
            k = E <= 20.0
            ax.plot(E[k], s[k] / nc / M.MB_CM2, color=CHARGE_COLOR[z], lw=1.1, label=f"charge {z:+d}")
        ax.set_title(tex(label), fontsize=9)
        ax.set_xlim(0, 20)
        ax.set_ylim(bottom=0)
        style(ax)
    finish_panels(fig, axes, len(mols), r"$\sigma/N_{\rm C}$ [Mb]",
                  "Final library at 0--20 eV (TD-LDA positions; Gaussian $s = 0.15$ eV below 7.5 eV, "
                  "0.8 eV at 14--30 eV)", path)


def fig_broadband(mols, path):
    fig, axes = state_panels(len(mols))
    for ax, (label, nc, nh, zs) in zip(axes.flat, mols):
        ax.axvspan(*E_JOIN, color="0.9", lw=0, label="hand-over to the scaled atoms (cos$^2$, 80--130 eV)")
        for z, st in zs:
            E, s = st["final"]
            ax.plot(E, np.clip(s, 1e-30, None) / nc / M.MB_CM2, color=CHARGE_COLOR[z], lw=1.0,
                    label=f"charge {z:+d}")
        E, a = zs[0][1]["atoms"]
        ax.plot(E, np.clip(a, 1e-30, None) / nc / M.MB_CM2, color="k", lw=1.0, ls="--",
                label="sum of atoms, unscaled (Verner C 1s + 2s + 2p, hydrogenic H 1s)")
        ax.set_title(tex(label), fontsize=9)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1.0, 1e4)
        ax.set_ylim(1e-4, 60)
        style(ax)
    finish_panels(fig, axes, len(mols), r"$\sigma/N_{\rm C}$ [Mb]", "Final library, 1 eV -- 10 keV", path)


def fig_stages(picked, path):
    """Each stage for a few states, in three energy ranges."""
    ranges = ((0.0, 15.0, "linear"), (10.0, 150.0, "logy"), (30.0, 1e4, "loglog"))
    fig, axes = plt.subplots(len(picked), 3, figsize=(15, 3.4 * len(picked)), squeeze=False)
    curves = (("raw", "0.6", "-", "Octopus's own (polynomial window, $T = 20$)"),
              ("gauss", COLORS[0], "-", "1. Gaussian window, $s_0 = 0.15$ eV"),
              ("wide", COLORS[1], "-", "2. widened to $s(E)$"),
              ("atoms", COLORS[2], "--", "sum of atoms, unscaled"),
              ("scaled", COLORS[3], "--", "sum of atoms, valence part scaled by $k$"),
              ("final", "k", "-", "final (cos$^2$ hand-over at 80--130 eV)"))
    for row, (label, nc, nh, z, st, _) in zip(axes, picked):
        for ax, (lo, hi, scale) in zip(row, ranges):
            for key, col, ls, lab in curves:
                E, s = st[key]
                k = (E >= lo) & (E <= hi)
                y = s[k] / nc / M.MB_CM2
                ax.plot(E[k], y if scale == "linear" else np.clip(y, 1e-6, None), color=col, ls=ls,
                        lw=1.6 if key == "final" else 1.0, label=lab)
            ax.set_xlim(lo, hi)
            if scale != "linear":
                ax.set_yscale("log")
            if scale == "loglog":
                ax.set_xscale("log")
                ax.set_ylim(1e-4, 30)
            elif scale == "logy":
                ax.set_ylim(0.1, 40)
            ax.set_title(tex(f"{label} {z:+d}"), fontsize=9)
            ax.set_xlabel(r"$E$ [eV]")
            style(ax)
        row[0].set_ylabel(r"$\sigma/N_{\rm C}$ [Mb]")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=3, frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.06 / len(picked)))
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig_benzene(rows, path_bz, path):
    """Every state over the unscaled atomic sum, against measured benzene over its own."""
    d = np.loadtxt(path_bz, comments="#")
    E, s = 1239.84198 / d[:, 0], d[:, 1]
    o = np.argsort(E)
    E, s = E[o], s[o]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.axvspan(*E_JOIN, color="0.9", lw=0, label="hand-over to the scaled atoms")
    for i, (label, nc, nh, z, st, _) in enumerate(rows):
        e, f = st["final"]
        k = (e >= 12) & (e <= 400)
        ax.plot(e[k], f[k] / sigma_atoms(nc, nh, e[k]), color=COLORS[0], lw=0.6, alpha=0.5,
                label="library states (final)" if i == 0 else None)
    k = E >= 12
    ax.plot(E[k], s[k] / sigma_atoms(6, 6, E[k]), "o", ms=3.5, color=COLORS[1], label=tex(f"benzene ({path_bz.name})"))
    ax.axhline(1.0, color="k", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xlim(12, 400)
    ax.set_ylim(0, 2.2)
    ax.set_xlabel(r"$E$ [eV]")
    ax.set_ylabel(r"$\sigma\,/\,\sum$ atoms")
    ax.legend(frameon=False, fontsize=8.5)
    style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def write(out, rows, table):
    np.savez_compressed(
        out / "pah_sigma_library.npz", energy_ev=E_OUT, label=np.array([r[0] for r in rows]),
        nc=np.array([r[1] for r in rows]), nh=np.array([r[2] for r in rows]), z=np.array([r[3] for r in rows]),
        sigma_cm2=np.array([r[4]["final"][1] for r in rows]),
        valence_scale_k=np.array([r[5] for r in rows]),
        params=np.array(f"s0 {S0} eV to 7.5 eV, {S_MID} eV at 14-30 eV, {S_TOP} eV above 40 eV; hand-over cos^2 "
                        f"{E_JOIN[0]}-{E_JOIN[1]} eV to the atoms (valence x k, C 1s unscaled); TD-LDA positions "
                        f"(no sTD-DFT correction)"))
    for label, nc, nh, z, st, _ in rows:
        E, s = st["final"]
        np.savetxt(out / f"sigma_{label}_{z:+d}.dat", np.column_stack([E, s]), fmt="%.6e",
                   header=f"{label} charge {z:+d}: N_C {nc}, N_H {nh}. Octopus TD-LDA (DIaL3 library, T = 20) "
                          f"to {E_JOIN[0]} eV, cos^2 hand-over by {E_JOIN[1]} eV to the sum of atoms (valence part "
                          f"x {_:.3f}, C 1s unscaled); build_library.py\nE [eV]  sigma [cm^2]")
    with open(out / "checks.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(table[0]))
        wr.writeheader()
        wr.writerows(table)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=ROOT, help="local mirror of DIaL3's pahspec/library")
    ap.add_argument("--out", type=Path, default=None, help="default: <results>/pahchem/octopus_library")
    ap.add_argument("--plots", type=Path, default=None, help="default: <plots>/pahchem/octopus")
    ap.add_argument("--benzene", type=Path, default=None, help="measured benzene: wavelength [nm], sigma [cm^2]")
    a = ap.parse_args()
    out = a.out or _paths.get_results_dir("pahchem/octopus_library")
    plots = a.plots or _paths.get_plots_dir("pahchem/octopus")
    out.mkdir(parents=True, exist_ok=True)
    use_calima_style()
    rows = [(label, nc, nh, z, *build(a.root / f"{label}_{z:+d}", nc, nh))
            for label, nc, nh, z, *_, done in states(a.root, 20.0 - 1e-6) if done]
    table = [checks(*r) for r in rows]
    write(out, rows, table)
    mols = molecules(rows)
    fig_uv(mols, plots / "final_uv.png")
    fig_broadband(mols, plots / "final_broadband.png")
    fig_stages([r for r in rows if (r[0], r[3]) in {("C96H24", 0), ("C96H24", 1)}], plots / "final_stages.png")
    if a.benzene:
        fig_benzene(rows, a.benzene, plots / "final_vs_benzene.png")
    keys = list(table[0])[4:]
    print(f"{len(rows)} finished states; columns: " + ", ".join(keys))
    for t in table:
        print(f"  {t['state']:18s} " + " ".join(f"{t[k]:7.4f}" if abs(t[k]) < 100 else f"{t[k]:7.1f}" for k in keys))
    print(f"library in {out}, figures in {plots}")


if __name__ == "__main__":
    main()
