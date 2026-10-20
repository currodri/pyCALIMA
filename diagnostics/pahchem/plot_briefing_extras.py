#!/usr/bin/env python3
"""Briefing figures beyond the spectral fits: the benchmark scorecard, gate G2 and the CR secondary electrons.

- ``bench_00_scorecard.png``: every benchmark of pahchem-spec.md by milestone and verdict
  (``briefing_benchmarks.csv``, the spec's verdicts on 2026-10-19).
- ``cr_01_eii_vs_beb.png``: gate G2, the electron-impact ionisation cross section of ``impact``
  against BEB(HF) (``beb_hf_pah.py`` output) and the published BEB peaks.
- ``cr_02_secondary_electrons.png``: the secondary-electron spectrum per unit zeta from CRISpy
  (``extract_crispy_secondary_spectrum.py`` output).

Run from the worktree root:
  PYTHONPATH=src:diagnostics/pahchem python diagnostics/pahchem/plot_briefing_extras.py
"""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from plot_briefing_fits import INK2, MUTED, style, ticks  # noqa: E402
from pycalima import _paths  # noqa: E402
from pycalima.pahchem import impact, photo  # noqa: E402

HERE = Path(__file__).resolve().parent
#: status colours (reserved status palette: good, warning, critical) and grey for withdrawn
STATUS = (("PASS", "reproduced", "#0ca30c"), ("PARTIAL", "partly / looser bound", "#fab219"),
          ("OPEN", "fails or open", "#d03b3b"), ("WITHDRAWN", "withdrawn / superseded", "#b5b4ae"))
GROUPS = (("M1", "M1 statistical mechanics"), ("M2", "M2 unimolecular rates"),
          ("M3", "M3 photophysics, cooling, RF"), ("M4", "M4 charge and H network"),
          ("M5", "M5 published PAH models"), ("M6", "M6 tables, solver precision"),
          ("M7", "M7 emission and observations"), ("CR", "cosmic-ray collisions"))
#: ionisation energies [eV] used for the model curves (Appl. Sci. 15, 9562 Table 1, experimental; coronene and
#: circumcoronene as in compare.py of the G2 study)
MOLECULES = {"anthracene": (14, 10, 7.41, "C$_{14}$H$_{10}$"), "pyrene": (16, 10, 7.43, "C$_{16}$H$_{10}$"),
             "coronene": (24, 12, 7.29, "C$_{24}$H$_{12}$"), "circumcoronene": (54, 18, 6.2, "C$_{54}$H$_{18}$")}
#: published BEB peaks [A^2]: Krishnadas+24 Table 4 (HF), Appl. Sci. 15, 9562 Table 2 (OVGF)
PUBLISHED = {"Krishnadas+24 BEB(HF)": {"anthracene": 28.144, "pyrene": 30.991},
             "Appl. Sci. 2025 BEB(OVGF)": {"anthracene": 32.3}}


def fig_scorecard(rows, out):
    fig, ax = plt.subplots(figsize=(11, 5.2), facecolor="w")
    y = np.arange(len(GROUPS))[::-1]
    for yi, (key, label) in zip(y, GROUPS):
        left = 0
        for st, _, colour in STATUS:
            n = sum(r["milestone"] == key and r["status"] == st for r in rows)
            if n:
                ax.barh(yi, n, left=left, color=colour, height=0.62, edgecolor="white", linewidth=2)
                ax.annotate(str(n), (left + n / 2, yi), ha="center", va="center", fontsize=12,
                            color="white" if st in ("PASS", "OPEN") else "black")
                left += n
    ax.set_yticks(y, [label for _, label in GROUPS], fontsize=13)
    ax.set_xlabel("number of benchmarks", fontsize=14)
    ticks(ax)
    ax.tick_params(axis="y", which="minor", left=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in STATUS]
    tot = {st: sum(r["status"] == st for r in rows) for st, _, _ in STATUS}
    ax.legend(handles, [f"{lab} ({tot[st]})" for st, lab, _ in STATUS], loc="lower right", fontsize=12)
    fig.suptitle(f"pahchem benchmarks by milestone and verdict ({len(rows)} comparisons in pahchem-spec.md)",
                 x=0.02, ha="left", fontsize=16)
    path = out / "bench_00_scorecard.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def model_sigma(nc, nh, ip, t_ev):
    w = impact.w_grid()
    y = lambda e: photo.yield_jochims1996(e, ip)  # noqa: E731
    out = []
    for t in t_ev:
        c = impact.impact_collisions(impact.ELECTRON, t, nc, nh, 0, ip_ev=ip, w=w)
        d = impact.deposition(c, ip, ip + 8.0, y)
        out.append((d.total(1) + d.total(2)) / 1e-16)
    return np.array(out)


def fig_g2(beb_dir, out):
    t = np.geomspace(10.0, 5000.0, 60)
    colours = ("#2a78d6", "#eb6834", "#1baf7a", "#6250d6")
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(9, 8), sharex=True, facecolor="w",
                                  gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0})
    for (name, (nc, nh, ip, label)), colour in zip(MOLECULES.items(), colours):
        f = sorted(beb_dir.glob(f"{name}_*_sigma.csv"))
        if not f:
            continue
        tb, sb = np.loadtxt(f[0], delimiter=",", unpack=True)
        sm = model_sigma(nc, nh, ip, t)
        ax.plot(tb, sb, color=colour, lw=2.0, label=f"{label}: BEB(HF), this work")
        ax.plot(t, sm, color=colour, lw=1.4, ls="--", label=f"{label}: pahchem impact")
        axr.plot(t, sm / np.interp(t, tb, sb), color=colour, lw=1.6)
    for (src, vals), marker in zip(PUBLISHED.items(), ("s", "D")):
        for k, (name, v) in enumerate(vals.items()):
            ax.plot(69.0, v, marker=marker, ms=8, mfc="white", mec="black", ls="none",
                    label=src if k == 0 else None)
    ax.set_xscale("log")
    ax.set_ylabel(r"$\sigma_{\rm ion}$ [\AA$^2$]", fontsize=15)
    ax.set_ylim(0, None)
    ax.legend(fontsize=11, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    axr.axhline(1.0, color=MUTED, lw=0.9)
    axr.set_ylabel("pahchem / BEB", fontsize=14)
    axr.set_ylim(0.8, 2.95)
    axr.set_xlabel(r"electron energy $T$ [eV]", fontsize=15)
    axr.annotate(r"peak: 1.75--2$\times$; keV: 1.3--1.45$\times$ (dipole strength $M^2_{\rm ion}$ 1.28$\times$)",
                 (0.02, 0.85), xycoords="axes fraction", fontsize=11, color=INK2)
    for a in (ax, axr):
        ticks(a)
    fig.suptitle(r"Gate G2: electron-impact ionisation of neutral PAHs (fails)", x=0.1, y=0.95, ha="left", fontsize=16)
    path = out / "cr_01_eii_vs_beb.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_secondary(sec_csv, out):
    rows = list(csv.reader(line for line in open(sec_csv) if not line.startswith("#")))[1:]
    tab = {}
    for spec, xe, lg, lo, hi, v in rows:
        tab.setdefault((spec, float(xe), float(lg)), []).append((float(lo), float(hi), float(v)))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True, facecolor="w", gridspec_kw={"wspace": 0.06})
    blues = ("#b7d3f6", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b")
    oranges = ("#fbd0bd", "#f5a07c", "#eb6834", "#b8461c", "#7a2c0f")

    def draw(ax, key, colour, label):
        lo, hi, v = np.array(tab[key]).T
        e = np.sqrt(lo * hi)
        ax.plot(e, v, color=colour, lw=2.0, label=label)

    for lg, colour in zip((20.0, 21.0, 22.0, 23.0, 24.0), blues):
        draw(axes[0], ("L", 0.0, lg), colour, rf"$N({{\rm H_2}}) = 10^{{{lg:.0f}}}$ cm$^{{-2}}$")
    axes[0].set_title("Spectrum L, no Coulomb losses ($x_e = 0$):\n"
                      r"per $\zeta$, the shape hardly moves with column below 300 eV", loc="left", fontsize=12.5)
    for xe, colour in zip((0.0, 1e-6, 1e-5, 1e-4, 1e-3), oranges):
        lab = r"$x_e = 0$" if xe == 0 else rf"$x_e = 10^{{{np.log10(xe):.0f}}}$"
        draw(axes[1], ("L", xe, 22.0), colour, lab)
    axes[1].set_title("Spectrum L, $N({\\rm H_2}) = 10^{22}$ cm$^{-2}$:\n"
                      "Coulomb losses empty the sub-10 eV electrons", loc="left", fontsize=12.5)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"electron energy $E$ [eV]", fontsize=15)
        ax.legend(fontsize=11, loc="lower left")
        ax.axvspan(6.0, 30.0, color="#f4f3f0", lw=0, zorder=0)
        ax.annotate("PAH ionisation\nand excitation", (13.0, 0.93), xycoords=("data", "axes fraction"), ha="center",
                    va="top", fontsize=10, color=INK2)
        ticks(ax)
    axes[0].set_ylabel(r"$4\pi j_{\rm sec}(E) / \zeta$ [cm$^{-2}$ eV$^{-1}$]", fontsize=15)
    fig.suptitle(r"Secondary electrons per unit $\zeta$ in H$_2$ (CRISpy Spencer--Fano degradation; primaries excluded)",
                 x=0.07, y=1.06, ha="left", fontsize=16)
    path = out / "cr_02_secondary_electrons.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--beb-dir", type=Path, default=_paths.get_results_dir() / "pahchem" / "beb_hf")
    ap.add_argument("--secondary", type=Path,
                    default=_paths.get_results_dir() / "pahchem" / "cr_secondary_electrons" / "secondary_flux_per_zeta.csv")
    args = ap.parse_args()
    out = args.out or (_paths.get_plots_dir("pahchem") / "briefing")
    out.mkdir(parents=True, exist_ok=True)
    style()
    rows = list(csv.DictReader(line for line in open(HERE / "briefing_benchmarks.csv") if not line.startswith("#")))
    print("wrote", fig_scorecard(rows, out))
    print("wrote", fig_g2(args.beb_dir, out))
    print("wrote", fig_secondary(args.secondary, out))


if __name__ == "__main__":
    main()
