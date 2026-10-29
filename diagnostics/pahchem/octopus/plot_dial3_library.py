#!/usr/bin/env python3
"""The DIaL3 cross-section library so far (``make_dial3_library.py``), against Malloci+07 and pahchem's default.

Reads a local mirror of DIaL3's ``pahspec/library`` (``--sync dial3`` refreshes it with rsync: inputs, multipoles and
the .ok markers only) and builds each spectrum from the three kicks with ``plot_progress.octopus_sigma``, to 150 eV
(the dipoles are written every step). A state is shown when its three kicks have all reached T >= ``--min-t``; one
that has not reached T = 20 yet is marked partial and its references are broadened to the same T
(``plot_progress.broaden``).

References, per carbon:
- Malloci+07 for the same molecule and charge, where Malloci has it (C54H18 is their circumcoronene);
- pahchem's default (``photo.sigma_uv``): the Malloci average of nine compact PAHs at 5-30 eV, Li & Draine outside.
  Both are Malloci's T = 20 propagations with their own window, which lowers sigma at 1-5 eV
  (docs/physics/pahchem-spec.md);
- Maragkoudakis+20's interstellar-field cross section (``photo.sigma_amespahdb``, the AmesPAHdb suite's Draine & Li
  2007 Drude form times the Mattioda+05 cutoff), shown to 13.6 eV, the range it is used in. Above ~10 eV it exceeds
  the f-sum rule (5.2 per C to 30 eV against 4.3 valence electrons), so only its shape below 13.6 eV is meaningful.

Figures (``--out``):
- ``library_spectra.png``: one panel per state;
- ``library_bands.png``: band integrals of the oscillator strength, Octopus over each reference, one row per reference;
- ``library_isrf.png``: the mean energy of the Mathis+83 photons absorbed (0.1-13.6 eV, as Maragkoudakis+20) and the
  share absorbed below 2 eV, per state and cross section. Negative ringing of the polynomial window is clipped; a
  Gaussian window gives the same numbers to 0.01 eV;
- ``library_euv.png``: 13.6-150 eV, for ionising fields, against pahchem's default (Li & Draine above 30 eV, which
  is graphite there) and the sum of atomic cross sections (``sigma_atomic``). The raw spectrum keeps the box's
  10-15 % ripples above 30 eV (docs/physics/pahchem-uv-absorption.md §4); X-rays (the C 1s edge) are outside the
  pseudopotential calculation.
The tables go to stdout.

    plot_dial3_library.py [--sync dial3] [--root ~/Documents/RAMSES_dev/pahspec/results/dial3_library]
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_progress import COLORS, broaden, octopus_sigma, style, tex  # noqa: E402

from pycalima.models.dust_charge import wdb06_atomic  # noqa: E402
from pycalima.pahchem import fields, photo  # noqa: E402
from pycalima.pahchem import malloci2007 as M  # noqa: E402
from pycalima.plotting_style import use_calima_style  # noqa: E402

ROOT = Path.home() / "Documents/RAMSES_dev/pahspec/results/dial3_library"
OUT = Path(__file__).resolve().parents[3] / "results/plots/pahchem/octopus"
REMOTE = "/lustre/dirac3/scratch/dp016/dc-rodr4/pahspec/library/"
SSH = "ssh -o BatchMode=yes -o ControlPath=~/.ssh/cm-%r@%h:%p"
RSYNC = ["rsync", "-az", "--prune-empty-dirs", "-e", SSH, "--exclude=restart/", "--exclude=exec/",
         "--include=*/", "--include=inp", "--include=multipoles", "--include=*.ok", "--include=run.ok",
         "--include=cross_section_tensor", "--exclude=*"]
STATE_RE = re.compile(r"^C(\d+)H(\d+)(?:_(\d+))?_([+-]\d)$")
#: our label -> Malloci+07's name for the same molecule
MALLOCI_NAME = {"C54H18": "circumcoronene"}
BANDS = ((1.0, 5.0), (5.0, 13.6), (13.6, 30.0))
EUV_BANDS = ((13.6, 30.0), (30.0, 54.4), (54.4, 150.0))
E_LIB = np.round(np.arange(0.0, 150.0 + 1e-9, 0.05), 10)
RY_EV = 13.605693
MARKERS = "osD^"
ISRF_EV = (0.1, 13.6)          # Maragkoudakis+20's field range (maragkoudakis2020.ISRF_E_MIN_EV, the Lyman limit)
#: reference kind -> (legend label, colour); the same in every panel
REF_STYLE = {
    "malloci": ("Malloci+07, same molecule", "k"),
    "default": ("pahchem default (Malloci 9-PAH average at 5-30 eV; Li and Draine outside)", COLORS[1]),
    "suite": ("Maragkoudakis+20 field cross section (AmesPAHdb suite: Draine and Li 2007 x Mattioda+05), to 13.6 eV",
              COLORS[2]),
    "atomic": ("sum of atomic cross sections (Verner phfit2 C 2s + 2p, hydrogenic H 1s)", COLORS[3]),
}


def states(root, min_t):
    """[(label, nc, nh, z, E, sigma [cm^2], T, done)] of the states whose three kicks reached ``min_t``."""
    out = []
    for d in sorted(root.iterdir()):
        m = STATE_RE.match(d.name)
        if not m or not (d / "td_x" / "inp").exists():
            continue
        res = octopus_sigma(d, energies=E_LIB)
        if res is None or res[2] < min_t:
            continue
        nc, nh, z = int(m.group(1)), int(m.group(2)), int(m.group(4))
        label = d.name.rsplit("_", 1)[0]
        out.append((label, nc, nh, z, *res, res[2] > 20.0 - 1e-6))
    return sorted(out, key=lambda s: (s[1], s[0] != f"C{s[1]}H{s[2]}", s[0], s[3]))


def references(label, nc, z, T, S):
    """{kind: (E, sigma [cm^2])} at the run's T: Malloci's same molecule (if any), pahchem's default and
    Maragkoudakis+20's field cross section (the suite's per-C form times N_C)."""
    E = np.round(np.arange(0.05, 30.0 + 1e-9, 0.05), 10)
    at_T = (lambda e, s: s) if T > 20.0 - 1e-6 else (lambda e, s: broaden(e, s, T))
    refs = {}
    name = MALLOCI_NAME.get(label)
    if name is not None and z in S[name]["sigma"]:
        e, s = S[name]["sigma"][z]
        refs["malloci"] = (e, at_T(e, s))
    refs["default"] = (E, at_T(E, photo.sigma_uv(nc, z, E)))
    k = E <= ISRF_EV[1]
    refs["suite"] = (E[k], nc * photo.sigma_amespahdb(nc, z, E[k]))      # analytic, smooth: not broadened
    return refs


def sigma_atomic(nc, nh, energy_ev):
    """Sum of the neutral atoms' photoionisation cross sections [cm^2] over the shells TD-DFT treats: C 2s + 2p
    (Verner phfit2, ``wdb06_atomic``) and H 1s (hydrogenic, Osterbrock & Ferland eq. 2.4)."""
    E = np.asarray(energy_ev, dtype=float)
    s_c = sum(wdb06_atomic.photoionisation_cross_section("C", sh, E) for sh in ("2s", "2p"))
    x = np.maximum(E / RY_EV, 1.0 + 1e-12)
    eps = np.sqrt(x - 1.0)
    s_h = np.where(E > RY_EV, 6.304e-18 * x ** -4 * np.exp(4.0 - 4.0 * np.arctan(eps) / eps)
                   / -np.expm1(-2.0 * np.pi / eps), 0.0)
    return nc * s_c + nh * s_h


def band_ratios(E, s, ref, bands=BANDS):
    """Octopus over ``ref`` per band; None where the reference does not cover the band."""
    return [M.oscillator_strength(E, s, lo, hi) / M.oscillator_strength(*ref, lo, hi) if ref[0].max() >= hi - 1e-9
            else None for lo, hi in bands]


def state_panels(n):
    """(fig, axes) with one panel per state, four columns."""
    nrow = -(-n // 4)
    return plt.subplots(nrow, 4, figsize=(16, 2.9 * nrow + 0.5), sharex=True, squeeze=False)


def finish_panels(fig, axes, n, ylabel, title, path):
    """One legend for all panels, empty panels hidden, x labels under the last panel of each column."""
    nrow, ncol = axes.shape
    handles = {}
    for ax in axes.flat[:n]:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    fig.legend(handles.values(), handles.keys(), loc="upper center", ncol=2, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, 0.995 - 0.02 / nrow))
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    for ax in axes[-1]:
        ax.set_xlabel(r"$E$ [eV]")
    if nrow > 1:
        for j in range(n % ncol or ncol, ncol):      # the last filled panel of each empty column
            axes[-2, j].xaxis.set_tick_params(labelbottom=True)
            axes[-2, j].set_xlabel(r"$E$ [eV]")
    fig.suptitle(title, fontsize=12, y=1.0 + 0.03 / nrow)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.07 / nrow))
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig_spectra(rows, S, out, table):
    fig, axes = state_panels(len(rows))
    for ax, (label, nc, nh, z, E, s, T, done) in zip(axes.flat, rows):
        refs = references(label, nc, z, T, S)
        for kind, (e, r) in refs.items():
            lab, col = REF_STYLE[kind]
            ax.plot(e, r / nc / M.MB_CM2, color=col, lw=1.0 if kind == "malloci" else 1.2, label=lab)
        ax.plot(E, s / nc / M.MB_CM2, color=COLORS[0], lw=1.1, ls="-" if done else "--",
                label="Octopus (dashed: partial, references broadened to its T)")
        ref_name, ref = next(iter(refs.items()))
        q = band_ratios(E, s, ref)
        ax.text(0.22, 0.97, "Octopus / " + ("Malloci" if ref_name == "malloci" else "default") + ":\n"
                + "\n".join(f"{lo:g}--{hi:g} eV: {v:.2f}" for (lo, hi), v in zip(BANDS, q)),
                transform=ax.transAxes, ha="left", va="top", fontsize=7)
        ax.set_title(tex(f"{label} {z:+d}") + ("" if done else f", partial: $T = {T:.1f}$ of 20"), fontsize=9)
        ax.set_xlim(0, 30)
        ax.set_ylim(0, None)
        style(ax)
        table.append(f"  {label:14s} {z:+d}  T {T:5.2f}  vs {ref_name:28s} " +
                     "  ".join(f"{lo:g}-{hi:g} eV {v:.3f}" for (lo, hi), v in zip(BANDS, q)))
    finish_panels(fig, axes, len(rows), r"$\sigma/N_{\rm C}$ [Mb]",
                  "Octopus library (DIaL3) against Malloci+07, pahchem's default and Maragkoudakis+20",
                  out / "library_spectra.png")


def fig_euv(rows, out, table):
    """13.6-150 eV: Octopus against pahchem's default and the atomic sum; the share of the valence f-sum reached."""
    fig, axes = state_panels(len(rows))
    E = E_LIB[E_LIB >= 13.6]
    for ax, (label, nc, nh, z, Eo, so, T, done) in zip(axes.flat, rows):
        refs = {"default": (E, photo.sigma_uv(nc, z, E)), "atomic": (E, sigma_atomic(nc, nh, E))}
        for kind, (e, r) in refs.items():
            lab, col = REF_STYLE[kind]
            ax.plot(e, r / nc / M.MB_CM2, color=col, lw=1.2, ls="--" if kind == "atomic" else "-", label=lab)
        k = Eo >= 13.6
        ax.plot(Eo[k], np.clip(so[k], 1e-3 * so.max(), None) / nc / M.MB_CM2, color=COLORS[0], lw=1.0,
                ls="-" if done else "--", label="Octopus (raw, polynomial window; dashed: partial)")
        q = band_ratios(Eo, so, refs["atomic"], EUV_BANDS)
        f_share = M.oscillator_strength(Eo, so) / M.valence_electrons(nc, nh, z)
        ax.text(0.97, 0.97, "Octopus / atomic:\n" + "\n".join(f"{lo:g}--{hi:g} eV: {v:.2f}"
                                                           for (lo, hi), v in zip(EUV_BANDS, q))
                + f"\nf to 150 eV / valence electrons: {f_share:.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=7)
        ax.set_title(tex(f"{label} {z:+d}") + ("" if done else f", partial: $T = {T:.1f}$ of 20"), fontsize=9)
        ax.set_xlim(13.6, 150)
        ax.set_yscale("log")
        ax.set_ylim(0.1, 40)
        style(ax)
        table.append(f"  {label:14s} {z:+d}  " + "  ".join(f"{lo:g}-{hi:g} eV {v:.3f}" for (lo, hi), v in
                                                           zip(EUV_BANDS, q)) + f"  f(0-150 eV)/valence {f_share:.3f}")
    finish_panels(fig, axes, len(rows), r"$\sigma/N_{\rm C}$ [Mb]",
                  "Octopus library (DIaL3) at 13.6--150 eV, against pahchem's default and the atomic sum",
                  out / "library_euv.png")


def fig_bands(rows, S, out):
    """One row per reference: Octopus over it, per band (open markers: partial runs)."""
    kinds = ("malloci", "default", "suite")
    fig, axes = plt.subplots(len(kinds), 1, figsize=(max(7.0, 0.55 * len(rows) + 2.5), 2.6 * len(kinds) + 1.2),
                             sharex=True, gridspec_kw=dict(hspace=0.08))
    band_col = (COLORS[0], COLORS[3], COLORS[2])
    for ax, kind in zip(axes, kinds):
        for i, (label, nc, nh, z, E, s, T, done) in enumerate(rows):
            ref = references(label, nc, z, T, S).get(kind)
            if ref is None:
                continue
            for j, v in enumerate(band_ratios(E, s, ref)):
                if v is None:
                    continue
                lab = f"{BANDS[j][0]:g}--{BANDS[j][1]:g} eV"
                ax.plot(i + 0.13 * (j - 1), v, MARKERS[j], ms=6.5, color=band_col[j], mfc=None if done else "white",
                        mew=1.3, label=lab if lab not in ax.get_legend_handles_labels()[1] else None)
        ax.axhspan(0.95, 1.05, color="0.88", lw=0)
        ax.axhline(1.0, color="k", lw=0.7)
        ax.set_yscale("log")
        ax.set_ylim(0.3, 3.5)
        ax.set_yticks([0.3, 0.5, 1, 2, 3], ["0.3", "0.5", "1", "2", "3"])
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_ylabel("Octopus / " + {"malloci": "Malloci+07\n(same molecule)", "default": "pahchem default",
                                     "suite": "Maragkoudakis+20\n(to 13.6 eV)"}[kind], fontsize=9)
        style(ax)
        ax.tick_params(axis="x", which="minor", bottom=False, top=False)
    axes[0].legend(frameon=False, fontsize=7.5, ncol=3, loc="upper right")
    axes[-1].set_xticks(range(len(rows)), [tex(f"{r[0]} {r[3]:+d}") + ("" if r[7] else "\n(partial)") for r in rows],
                        rotation=60, ha="right", fontsize=7.5)
    axes[-1].set_xlim(-0.6, len(rows) - 0.4)
    axes[0].set_title("oscillator strength per band, Octopus over each reference (grey: 5 per cent; open: partial run)",
                      fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "library_bands.png", dpi=170)
    plt.close(fig)


def isrf_stats(E, s):
    """(mean energy [eV] of the absorbed Mathis+83 photons, share absorbed below 2 eV), over ``ISRF_EV``."""
    f = fields.mathis1983(e_min=ISRF_EV[0], n=2000)
    e = f.energy_ev
    w = np.clip(np.interp(e, E, s, left=0.0, right=0.0), 0.0, None) * f.flux_at(e)
    tot = np.trapezoid(w, e)
    return np.trapezoid(w * e, e) / tot, np.trapezoid(w[e < 2.0], e[e < 2.0]) / tot


def fig_isrf(rows, out, table):
    """What sets the heating in Maragkoudakis+20's field runs: which photons each cross section absorbs."""
    E = np.round(np.arange(0.05, 30.0 + 1e-9, 0.05), 10)
    series = (("Octopus", COLORS[0], "o"), ("pahchem default", COLORS[1], "s"),
              ("Maragkoudakis+20 (AmesPAHdb suite)", COLORS[2], "D"), ("Li and Draine 2001", "0.5", "^"))
    fig, axes = plt.subplots(2, 1, figsize=(max(7.0, 0.55 * len(rows) + 2.5), 6.4), sharex=True,
                             gridspec_kw=dict(hspace=0.08))
    for i, (label, nc, nh, z, Eo, so, T, done) in enumerate(rows):
        sig = {"Octopus": (Eo, so), "pahchem default": (E, photo.sigma_uv(nc, z, E)),
               "Maragkoudakis+20 (AmesPAHdb suite)": (E, photo.sigma_amespahdb(nc, z, E)),
               "Li and Draine 2001": (E, photo.sigma_li_draine(nc, z, E))}
        vals = {name: isrf_stats(*sig[name]) for name, _, _ in series}
        for k, (name, col, mk) in enumerate(series):
            for ax, v in zip(axes, vals[name]):
                ax.plot(i + 0.12 * (k - 1.5), v, mk, ms=6.5, color=col, mfc=None if done or k else "white", mew=1.3,
                        label=name if name not in ax.get_legend_handles_labels()[1] else None)
        table.append(f"  {label:14s} {z:+d}  " + "   ".join(f"{name.split(' (')[0]} {m:.2f} eV ({sh:.2f})"
                                                           for name, (m, sh) in vals.items()))
    axes[0].set_ylabel("mean absorbed\nphoton energy [eV]")
    axes[1].set_ylabel("share of absorbed\nphotons below 2 eV")
    axes[1].set_ylim(0, 1)
    for ax in axes:
        style(ax)
        ax.tick_params(axis="x", which="minor", bottom=False, top=False)
    axes[0].legend(frameon=False, fontsize=7.5, ncol=2, loc="lower left")
    axes[-1].set_xticks(range(len(rows)), [tex(f"{r[0]} {r[3]:+d}") + ("" if r[7] else "\n(partial)") for r in rows],
                        rotation=60, ha="right", fontsize=7.5)
    axes[-1].set_xlim(-0.6, len(rows) - 0.4)
    axes[0].set_title("Mathis+83 field photons absorbed, 0.1-13.6 eV as in Maragkoudakis+20 (open: partial run)",
                      fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "library_isrf.png", dpi=170)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=ROOT, help="local mirror of DIaL3's pahspec/library")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--sync", metavar="HOST", default=None, help="rsync HOST:<library> into --root first")
    ap.add_argument("--min-t", type=float, default=10.0, help="shortest propagation shown [hbar/eV]")
    a = ap.parse_args()
    if a.sync:
        a.root.mkdir(parents=True, exist_ok=True)
        subprocess.run(RSYNC + [f"{a.sync}:{REMOTE}", f"{a.root}/"], check=True)
    a.out.mkdir(parents=True, exist_ok=True)
    use_calima_style()
    S = M.spectra()
    rows = states(a.root, a.min_t)
    table = []
    fig_spectra(rows, S, a.out, table)
    fig_bands(rows, S, a.out)
    isrf = []
    fig_isrf(rows, a.out, isrf)
    euv = []
    fig_euv(rows, a.out, euv)
    print(f"{len(rows)} states (Octopus / reference, oscillator strength per band):")
    print("\n".join(table))
    print("\nMathis+83 field: mean absorbed photon energy (share absorbed below 2 eV):")
    print("\n".join(isrf))
    print("\n13.6-150 eV: Octopus / atomic sum, oscillator strength per band; f to 150 eV over the valence electrons:")
    print("\n".join(euv))
    print(f"figures in {a.out}")


if __name__ == "__main__":
    main()
