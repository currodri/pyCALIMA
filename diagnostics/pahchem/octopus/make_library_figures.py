#!/usr/bin/env python3
"""The figures of docs/physics/pahchem-uv-absorption.md: the PAH cross-section library, from the final ``pah_sigma_library.npz``
(``build_library.py``). Six figures are made here; a seventh, ``lin23_compare.png``, comes from ``compare_lin.py`` and is copied.

- ``library_overview.png``: sigma per carbon at 0-20 eV for four molecules, four charges, with TD-LDA's positions for the neutral;
- ``library_xray.png``: 1 eV to 10 keV for one molecule, and the C K edge against the atomic C 1s;
- ``library_euv_atoms.png``: the library over the sum of atomic cross sections, 12-250 eV, against measured benzene;
- ``band_positions.png``: calculated minus measured band positions, TD-LDA against hybrid-functional methods;
- ``shift_effect.png``: what shifting the neutrals' bands does to the mean absorbed photon energy and to the rates;
- ``processing_stages.png``: the steps from Octopus's spectrum to the library, for one state;
- ``size_rule.png``: the size and charge rule of ``pycalima.pahchem.uvlib`` (from the bundled anchor table).

    make_library_figures.py --lib <octopus_library>/pah_sigma_library.npz --out docs/_static/pahchem_uv
                            [--benzene <bz_feng.dat>] [--ck-edge <gordon03_pyrene.dat>] [--lin-fig <lin23_compare.png>]
"""
import argparse
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_library as B  # noqa: E402
from plot_progress import COLORS, style, tex  # noqa: E402

from pycalima.pahchem import fields, uvlib  # noqa: E402
from pycalima.pahchem import malloci2007 as M  # noqa: E402
from pycalima.plotting_style import use_calima_style  # noqa: E402

MB = M.MB_CM2
CHARGE_COLOR = {-1: COLORS[0], 0: COLORS[1], 1: COLORS[2], 2: COLORS[3]}
CHARGE_LABEL = {-1: "anion", 0: "neutral", 1: "cation", 2: "dication"}


def amp():
    """An ampersand that survives LaTeX labels (use_calima_style() switches usetex on in main)."""
    return r"\&" if plt.rcParams["text.usetex"] else "&"


def state(lib, label, z):
    i = [k for k in range(len(lib["z"])) if lib["label"][k] == label and lib["z"][k] == z][0]
    return i, int(lib["nc"][i]), int(lib["nh"][i])


def plain_log_ticks(ax, ticks):
    """Plain numbers (15, 20, 30, ...) on a logarithmic energy axis instead of 2 x 10^1."""
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def legend_below(ax, ncol=2):
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=ncol)


def panel_letters(axes):
    for ax, ch in zip(np.ravel(axes), "abcdefgh"):
        ax.text(0.02, 0.96, f"({ch})", transform=ax.transAxes, ha="left", va="top", fontsize=10, fontweight="bold")


def fig_overview(lib, path):
    E = lib["energy_ev"]
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.4), sharex=True)
    for ax, label in zip(axes.flat, ("C54H18", "C96H24", "C150H30", "C216H36")):
        for z in (-1, 0, 1, 2):
            try:
                i, nc, _ = state(lib, label, z)
            except IndexError:
                continue
            ax.plot(E, lib["sigma_cm2"][i] / nc / MB, color=CHARGE_COLOR[z], lw=1.1, label=CHARGE_LABEL[z])
            if z == 0:
                ax.plot(E, lib["sigma_cm2_tdlda"][i] / nc / MB, color=CHARGE_COLOR[0], lw=0.9, ls=":", label="neutral, TD-LDA positions")
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 27)
        ax.set_title(tex(label), fontsize=10)
        style(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\sigma/N_{\rm C}$ [Mb]")
    for ax in axes[1]:
        ax.set_xlabel(r"$E$ [eV]")
    panel_letters(axes)
    fig.tight_layout()
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=5, frameon=False, fontsize=8.5)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_xray(lib, path):
    E = lib["energy_ev"]
    i, nc, nh = state(lib, "C96H24", 0)
    s = lib["sigma_cm2"][i] / nc / MB
    atoms = B.sigma_atoms(nc, nh, E) / nc / MB
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.0, 4.2), gridspec_kw={"width_ratios": (1.5, 1)})
    a.axvspan(*B.E_JOIN, color="0.88", lw=0, label="hand-over to the atoms")
    a.plot(E, np.clip(atoms, 1e-6, None), color="k", ls="--", lw=0.9, label="sum of atoms (Verner C 1s 2s 2p, hydrogenic H 1s)")
    a.plot(E, np.clip(s, 1e-6, None), color=CHARGE_COLOR[0], lw=1.1, label=r"library, C$_{96}$H$_{24}$ neutral")
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlim(1, 1e4)
    a.set_ylim(1e-4, 60)
    a.set_xlabel(r"$E$ [eV]")
    a.set_ylabel(r"$\sigma/N_{\rm C}$ [Mb]")
    a.legend(frameon=False, fontsize=8, loc="upper right")
    style(a)
    k = (E >= 278) & (E <= 325)
    b.plot(E[k], atoms[k], color="k", ls="--", lw=0.9, label="atomic C 1s step (291 eV)")
    b.plot(E[k], s[k], color=CHARGE_COLOR[0], lw=1.2, label="library: measured C 1s (pyrene)")
    b.axvspan(295.9, 310.0, color="0.88", lw=0, label="interpolation to the atomic value")
    b.set_xlim(278, 325)
    b.set_ylim(0, 2.3)
    b.set_xlabel(r"$E$ [eV]")
    b.set_ylabel(r"$\sigma/N_{\rm C}$ [Mb]")
    b.legend(frameon=False, fontsize=8, loc="upper right")
    style(b)
    panel_letters([a, b])
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_euv_atoms(lib, path, benzene):
    E = lib["energy_ev"]
    k = (E >= 12) & (E <= 250)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.axvspan(*B.E_JOIN, color="0.88", lw=0, label="hand-over to the scaled atoms")
    for n in range(len(lib["z"])):
        nc, nh = int(lib["nc"][n]), int(lib["nh"][n])
        ax.plot(E[k], lib["sigma_cm2"][n][k] / B.sigma_atoms(nc, nh, E[k]), color=COLORS[0], lw=0.5, alpha=0.35)
    if benzene is not None:
        d = np.loadtxt(benzene, comments="#")
        Eb, sb = 1239.84198 / d[:, 0], d[:, 1]
        o = np.argsort(Eb)
        Eb, sb = Eb[o], sb[o]
        kb = (Eb >= 12) & (Eb <= 250)
        ax.plot(Eb[kb], sb[kb] / B.sigma_atoms(6, 6, Eb[kb]), "o", ms=3.0, color=COLORS[1], label="measured benzene (Feng+)")
    ax.plot([], [], color=COLORS[0], lw=1.2, label="48 library states")
    ax.axhline(1.0, color="k", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xlim(12, 250)
    plain_log_ticks(ax, [15, 20, 30, 50, 100, 200])
    ax.set_ylim(0, 2.0)
    ax.set_xlabel(r"$E$ [eV]")
    ax.set_ylabel(r"$\sigma\,/\,\sum$ atomic cross sections")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


#: signed errors, calculated - measured [eV]. Pyrene: band centroids of the 423 K vapour (Thony & Rossi 1997), 3.98, 4.81, 5.48 eV;
#: sTDA and sTD-DFT shifts matched by rank (docs section 5), PySCF TDA/def2-SVP for the full B3LYP. Ovalene: the strongest peak of the
#: 270-340 nm system in neon, 326.97 nm = 3.79 eV (Ruiterkamp+ 2002), and S1, whose origin is ~2.55 eV (S2 origin 2.61 eV in
#: para-H2, S1 ~0.06 eV lower; Wu+ 2024); full TD-B3LYP/def2-SVP. HBC: the S1 vibronic peak at 434 nm = 2.86 eV (Steglich+ 2011);
#: TD-LDA's S1 is not resolved in our spectrum and is omitted.
POSITIONS = (
    ("pyrene S$_2$ (3.98)", {"TD-LDA": 3.38 - 3.98, "sTDA": -0.17, "sTD-DFT": -0.34, "full TD-B3LYP": 3.89 - 3.98}),
    ("pyrene S$_3$ (4.81)", {"TD-LDA": 4.15 - 4.81, "sTDA": -0.09, "sTD-DFT": -0.28, "full TD-B3LYP": 4.85 - 4.81}),
    ("pyrene 5.3 eV band (5.48)", {"TD-LDA": 5.03 - 5.48, "sTDA": 0.15, "sTD-DFT": -0.07, "full TD-B3LYP": 5.72 - 5.48}),
    ("ovalene S$_1$ (2.55)", {"TD-LDA": 2.40 - 2.55, "sTDA": 2.68 - 2.55, "sTD-DFT": 2.56 - 2.55, "full TD-B3LYP": 2.70 - 2.55}),
    ("ovalene strongest band (3.79)", {"TD-LDA": 3.29 - 3.79, "sTDA": 3.74 - 3.79, "sTD-DFT": 3.55 - 3.79, "full TD-B3LYP": 3.70 - 3.79}),
    ("HBC S$_1$ (2.86)", {"sTDA": 2.74 - 2.86, "sTD-DFT": 2.73 - 2.86, "full TD-B3LYP": 2.88 - 2.86}),
)


def fig_positions(path):
    methods = ("TD-LDA", "sTDA", "sTD-DFT", "full TD-B3LYP")
    marks = dict(zip(methods, "ovsD"))
    cols = dict(zip(methods, (COLORS[0], COLORS[1], COLORS[2], "k")))
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.axvspan(-0.1, 0.1, color="0.93", lw=0)
    ax.axvline(0, color="k", lw=0.8)
    seen = set()
    for y, (name, vals) in enumerate(POSITIONS[::-1]):
        for m in methods:
            if m in vals:
                ax.plot(vals[m], y, marks[m], color=cols[m], ms=6.5, mfc=cols[m] if m != "full TD-B3LYP" else "none",
                        label=m if m not in seen else None)
                seen.add(m)
    ax.set_yticks(range(len(POSITIONS)))
    ax.set_yticklabels([n for n, _ in POSITIONS[::-1]], fontsize=9)
    ax.set_xlim(-0.8, 0.4)
    ax.set_ylim(-0.6, len(POSITIONS) - 0.4)
    ax.set_xlabel(r"calculated $-$ measured band position [eV]")
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4)
    style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def rate(f, E, s):
    e = f.energy_ev
    return np.trapezoid(np.clip(np.interp(e, E, s, left=0.0, right=0.0), 0.0, None) * f.flux_at(e), e)


def fig_shift(lib, path):
    E = lib["energy_ev"]
    neu = [n for n in range(len(lib["z"])) if lib["z"][n] == 0]
    shifts = np.arange(0.0, 0.801, 0.1)
    fl = {"Mathis ISRF": fields.mathis1983(1.0, e_min=0.1, n=2000), "Draine 1978": fields.draine1978(1.0),
          "Kurucz 10 000 K": fields.kurucz(10000, 1.0), "blackbody 6000 K": fields.blackbody(6000, 1.0, e_min=0.1)}
    k = E <= 20.0
    mean_e = np.zeros((len(shifts), len(neu)))
    rr = {n: np.zeros((len(shifts), len(neu))) for n in fl}
    for a, d in enumerate(shifts):
        for b, n in enumerate(neu):
            s = B.shift_bands(E[k], lib["sigma_cm2_tdlda"][n][k], d)
            mean_e[a, b] = B.isrf_stats(E[k], s)[0]
            for name, f in fl.items():
                rr[name][a, b] = rate(f, E[k], s) / rate(f, E[k], lib["sigma_cm2_tdlda"][n][k])
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 4.0))
    a.axhspan(4.6, 5.8, color="0.92", lw=0, label="earlier default, Li " + amp() + " Draine, AmesPAHdb suite")
    a.fill_between(shifts, mean_e.min(1), mean_e.max(1), color=COLORS[1], alpha=0.25, lw=0)
    a.plot(shifts, mean_e.mean(1), color=COLORS[1], lw=1.4, label="12 neutrals: mean and range")
    a.axvline(0.4, color="k", lw=0.8, ls=":")
    a.set_xlabel("upward shift of the bands below 6 eV [eV]")
    a.set_ylabel("mean absorbed Mathis photon energy [eV]")
    a.set_ylim(2, 7.0)
    legend_below(a, 1)
    style(a)
    for (name, v), c in zip(rr.items(), COLORS):
        b.fill_between(shifts, v.min(1), v.max(1), color=c, alpha=0.2, lw=0)
        b.plot(shifts, v.mean(1), color=c, lw=1.3, label=name)
    b.axhline(1.0, color="k", lw=0.7)
    b.axvline(0.4, color="k", lw=0.8, ls=":")
    b.set_xlabel("upward shift of the bands below 6 eV [eV]")
    b.set_ylabel("neutral absorption rate, over the unshifted")
    legend_below(b, 2)
    style(b)
    panel_letters([a, b])
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_stages(lib, path, root, kedge):
    st, _ = B.build(root / "C96H24_+0", 96, 24, kedge)
    E_td = B.E_TD
    wide_shifted = B.shift_bands(E_td, st["wide"][1], 0.4)
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.0, 4.2))
    a.plot(E_td, st["raw"][1] / 96 / MB, color="0.6", lw=0.9, label="Octopus, polynomial window")
    a.plot(E_td, st["gauss"][1] / 96 / MB, color=COLORS[0], lw=1.0, label=r"Gaussian window, $s_0 = 0.15$ eV")
    a.plot(E_td, st["wide"][1] / 96 / MB, color=COLORS[1], lw=1.0, label=r"widened to $s(E)$")
    a.plot(E_td, wide_shifted / 96 / MB, color="k", lw=1.2, label=r"+0.4 eV shift of the bands below 6 eV (as shipped)")
    a.set_xlim(0, 14)
    a.set_ylim(-1, 30)
    a.set_xlabel(r"$E$ [eV]")
    a.set_ylabel(r"$\sigma/N_{\rm C}$ [Mb]")
    legend_below(a, 2)
    style(a)
    k = (E_td >= 12) & (E_td <= 150)
    b.plot(E_td[k], np.clip(st["raw"][1][k], 1e-30, None) / 96 / MB, color="0.6", lw=0.8, label="Octopus, polynomial window")
    b.plot(E_td[k], np.clip(st["wide"][1][k], 1e-30, None) / 96 / MB, color=COLORS[1], lw=1.0, label=r"widened to $s(E)$")
    Eo = lib["energy_ev"]
    i, nc, nh = state(lib, "C96H24", 0)
    kk = (Eo >= 12) & (Eo <= 150)
    b.plot(Eo[kk], lib["sigma_cm2"][i][kk] / nc / MB, color="k", lw=1.2, label="library (blend into the scaled atoms)")
    b.plot(Eo[kk], B.sigma_atoms(nc, nh, Eo[kk]) / nc / MB, color="k", lw=0.9, ls="--", label="sum of atoms, unscaled")
    b.axvspan(*B.E_JOIN, color="0.88", lw=0)
    b.set_xscale("log")
    b.set_yscale("log")
    b.set_xlim(12, 150)
    b.set_ylim(0.1, 40)
    plain_log_ticks(b, [15, 20, 30, 50, 100])
    b.set_xlabel(r"$E$ [eV]")
    legend_below(b, 2)
    style(b)
    panel_letters([a, b])
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_rule(path):
    t = uvlib.library()
    E = np.geomspace(0.1, 20.0, 1500)
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.0, 4.2))
    cols = plt.cm.viridis(np.linspace(0.05, 0.9, len(t["nc"])))
    for n, c in zip(t["nc"], cols):
        a.plot(E, uvlib.per_carbon(n, 0, E) / MB, color=c, lw=1.0, label=f"C$_{{{int(n)}}}$")
    a.plot(E, uvlib.per_carbon(70, 0, E) / MB, color="k", lw=1.2, ls="--", label=r"$N_{\rm C} = 70$, interpolated")
    a.set_xlim(1.0, 8.0)
    a.set_ylim(0, 27)
    a.set_xlabel(r"$E$ [eV]")
    a.set_ylabel(r"$\sigma/N_{\rm C}$ [Mb]")
    legend_below(a, 4)
    style(a)
    f = fields.mathis1983(1.0, e_min=0.1, n=2000)
    Eg = t["energy_ev"]
    k = Eg <= 13.6

    def rate(s):
        return np.trapezoid(np.clip(np.interp(f.energy_ev, Eg[k], s[k], left=0, right=0), 0, None) * f.flux_at(f.energy_ev), f.energy_ev)
    n_grid = np.geomspace(16, 216, 60)
    for z in (-1, 0, 1, 2):
        for st, ls in (("mixed", "-"), ("compact", ":")):
            r = [rate(uvlib.per_carbon(n, z, Eg, structure=st)) for n in n_grid]
            b.plot(n_grid, np.array(r) / 1e-8, color=CHARGE_COLOR[z], ls=ls, lw=1.2, label=f"{CHARGE_LABEL[z]}, {st}")
    b.set_xscale("log")
    b.set_xlabel(r"$N_{\rm C}$")
    b.set_ylabel(r"Mathis photon absorption rate per carbon [$10^{-8}$ s$^{-1}$]")
    plain_log_ticks(b, [16, 24, 32, 54, 96, 150, 216])
    b.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.17))
    style(b)
    panel_letters([a, b])
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lib", type=Path, required=True, help="pah_sigma_library.npz from build_library.py")
    ap.add_argument("--out", type=Path, required=True, help="figure directory, e.g. docs/_static/pahchem_uv")
    ap.add_argument("--root", type=Path, default=B.ROOT, help="local mirror of DIaL3's pahspec/library (for the processing stages)")
    ap.add_argument("--benzene", type=Path, default=None, help="measured benzene: wavelength [nm], sigma [cm^2]")
    ap.add_argument("--ck-edge", type=Path, default=None, help="the pyrene C 1s curve used by build_library.py (for the stages)")
    ap.add_argument("--lin-fig", type=Path, default=None, help="compare_lin.py's lin23_compare.png, copied to --out")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    use_calima_style()
    lib = dict(np.load(a.lib))
    kedge = tuple(np.loadtxt(a.ck_edge, unpack=True)) if a.ck_edge else None
    fig_overview(lib, a.out / "library_overview.png")
    fig_xray(lib, a.out / "library_xray.png")
    fig_euv_atoms(lib, a.out / "library_euv_atoms.png", a.benzene)
    fig_positions(a.out / "band_positions.png")
    fig_shift(lib, a.out / "shift_effect.png")
    fig_stages(lib, a.out / "processing_stages.png", a.root, kedge)
    fig_rule(a.out / "size_rule.png")
    if a.lin_fig:
        shutil.copy(a.lin_fig, a.out / "lin23_compare.png")
    print("figures in", a.out)


if __name__ == "__main__":
    main()
