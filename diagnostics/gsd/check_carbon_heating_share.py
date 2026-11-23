#!/usr/bin/env python3
"""How much of the grain photoelectric heating and recombination cooling does carbon below N_C,max = 400 carry, and how
much does it matter whether that carbon is charged with the PAH (molecular) tables or the 'ac' grain tables?

Carbon-transfer study of 2026-11-23 (results/carbon_plan/carbon_transfer_2026-11-23/), check 1. For each test cell of
``check_pah_grain_seam`` (WNM, CNM, PDR, dense; Mathis background at the cell's G0, no RT photons) this computes the
heating Gamma and cooling Lambda per C atom of 'ac' grains on a radius grid from 0.39 nm to 1 um (rt_group_charging
tables and solve, as the seam check does), and takes the PAH-table values per C of the four molecular bins (N_C 20-42,
42-89, 89-189, 189-400) from the seam check's output. The carbon is distributed as THEMIS J13 (DustEM GRAIN_J13: CM20
power law a^-5 with an exponential edge, 1.70e-3 per H, rho 1.6, plus the lognormal large a-C, 6.3e-4 per H), and the
mass fraction below N_C = 400 is also scanned by rescaling that part.

Prints, per cell, the share of Gamma and Lambda carried by N_C < 400 carbon (ac identity) and the ratio of the totals
with the PAH identity to those with the ac identity.

    python diagnostics/gsd/check_carbon_heating_share.py \\
        --seam results/carbon_plan/check_pah_grain_seam.txt \\
        --out results/carbon_plan/carbon_transfer_2026-11-23/heating_share
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

CELLS = {"WNM": (8000.0, 0.4, 7.2e-3, "H+", 1.0),
         "CNM": (100.0, 30.0, 3e-2, "C+", 1.0),
         "PDR": (300.0, 1e4, 1.0, "C+", 1e3),
         "dense": (30.0, 1e3, 1e-3, "C+", 1e-2)}   # T, n_H, n_e, ion, G0 (as check_pah_grain_seam.CELLS)
ION_MASS = {"H+": 1.00794, "C+": 12.0107}
AMU = 1.66053906660e-24
M_C = 12.0107 * AMU
RHO = 1.6
NC_EDGES = (20.0, 42.0, 89.0, 189.0, 400.0)


def radius(nc):
    return (3.0 * nc * M_C / (4.0 * np.pi * RHO)) ** (1.0 / 3.0)


def pah_per_c(path):
    """{cell: [(Gamma/C, Lambda/C) for the 4 molecular bins]} from check_pah_grain_seam's text output."""
    out = {c: [] for c in CELLS}
    pat = re.compile(r"^\s+(WNM|CNM|PDR|dense)\s+PAH f = .*G/C\s+([0-9.eE+-]+)\s+L/C\s+([0-9.eE+-]+)")
    for line in Path(path).read_text().splitlines():
        m = pat.match(line)
        if m:
            out[m.group(1)].append((float(m.group(2)), float(m.group(3))))
    if any(len(v) != 4 for v in out.values()):
        raise ValueError(f"expected 4 molecular bins per cell in {path}")
    return out


def ac_per_c(a_grid):
    """Gamma/C, Lambda/C [erg/s per C atom] of 'ac' grains at radii a_grid, per cell."""
    from pycalima.models.dust_charge import rt_group_charging as rg

    out = {c: np.zeros((a_grid.size, 2)) for c in CELLS}
    for k, a in enumerate(a_grid):
        tab = rg.build_bin_table("ac", float(a))
        gt = rg.GroupChargingTable(tab)
        nG = tab["L0"].size
        nc = 4.0 / 3.0 * np.pi * RHO * a ** 3 / M_C
        for name, (T, n_h, n_e, ion, G0) in CELLS.items():
            r = gt.solve(np.zeros(nG), rg.C_CGS, T, n_e, [(n_e, 1.0, ION_MASS[ion] * AMU)], G0_bg=G0, recomb=None)
            out[name][k] = (r["Gamma"] / nc, (r["Lambda"] + r.get("Lambda_auto", 0.0)) / nc)
    return out


def j13_carbon_mass(a):
    """THEMIS J13 carbon mass per unit ln a, as a mass ratio to hydrogen (DustEM M_dust/M_H): only ratios are used."""
    def plaw_ed(x):
        f = x ** -5.0
        return np.where(x > 10e-7, f * np.exp(-((x - 10e-7) / 50e-7) ** 1.0), f)

    def logn(x):
        return np.exp(-0.5 * (np.log(x / 7e-7) / 1.0) ** 2) / x

    out = []
    for dn, mtot, rho, lo in ((plaw_ed, 1.70e-3, 1.6, 0.4e-7), (logn, 6.3e-4, 1.57, 0.5e-7)):
        x = np.geomspace(lo, 4900e-7, 4000)
        dmdlna = 4.0 / 3.0 * np.pi * rho * x ** 4 * dn(x)
        norm = mtot / np.trapezoid(dmdlna, np.log(x))
        out.append(np.where((a >= lo) & (a <= 4900e-7), norm * 4.0 / 3.0 * np.pi * rho * a ** 4 * dn(a), 0.0))
    return out[0] + out[1]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seam", default="results/carbon_plan/check_pah_grain_seam.txt")
    ap.add_argument("--n-radii", type=int, default=36)
    ap.add_argument("--out", default="results/carbon_plan/carbon_transfer_2026-11-23/heating_share")
    args = ap.parse_args(argv)

    a_lo, a_hi = radius(NC_EDGES[0]), 1e-4
    a_grid = np.geomspace(a_lo, a_hi, args.n_radii)
    pah = pah_per_c(args.seam)
    ac = ac_per_c(a_grid)
    lna = np.log(a_grid)
    dM = j13_carbon_mass(a_grid)                          # g/H per ln a
    dC = dM / M_C                                         # C atoms per H per ln a
    a_edges = [radius(n) for n in NC_EDGES]
    mol = a_grid < a_edges[-1]
    f400 = np.trapezoid(np.where(mol, dM, 0.0), lna) / np.trapezoid(dM, lna)

    lines = [f"THEMIS J13 carbon: M_C/M_H = {np.trapezoid(dM, lna):.3e}, of which {f400 * 100:.1f} % below N_C = 400 "
             f"(a < {a_edges[-1] * 1e7:.2f} nm); radii {a_grid[0] * 1e7:.2f} nm - {a_grid[-1] * 1e4:.1f} um, "
             f"{args.n_radii} points",
             "per cell: share of the total carried by N_C < 400 carbon with ac tables, and the total with the PAH "
             "tables below N_C = 400 over the total with ac tables (Gamma | Lambda)"]
    scan = np.array([0.02, 0.05, 0.10, 0.20, 0.30])
    rows = {}
    for name in CELLS:
        g_ac, l_ac = ac[name][:, 0], ac[name][:, 1]
        # PAH identity: below N_C = 400 the per-C value of the molecular bin the radius falls in
        idx = np.clip(np.searchsorted(a_edges, a_grid, side="right") - 1, 0, 3)
        g_p = np.where(mol, np.array([pah[name][i][0] for i in idx]), g_ac)
        l_p = np.where(mol, np.array([pah[name][i][1] for i in idx]), l_ac)
        res = {}
        for q, (yac, yp) in (("Gamma", (g_ac, g_p)), ("Lambda", (l_ac, l_p))):
            tot_ac = np.trapezoid(yac * dC, lna)
            tot_p = np.trapezoid(yp * dC, lna)
            share = np.trapezoid(np.where(mol, yac * dC, 0.0), lna) / tot_ac
            # rescale the N_C < 400 carbon to fraction f of the carbon mass, the rest unchanged in shape
            sc = []
            for f in scan:
                w = np.where(mol, f / f400, (1 - f) / (1 - f400))
                sc.append((np.trapezoid(yp * dC * w, lna) / np.trapezoid(yac * dC * w, lna),
                           np.trapezoid(np.where(mol, yac * dC * w, 0.0), lna) / np.trapezoid(yac * dC * w, lna)))
            res[q] = dict(share=share, ratio=tot_p / tot_ac, scan=sc, per_c_ac=yac, per_c_pah=yp)
        rows[name] = res
        lines.append(f"{name:6s} Gamma: N_C<400 share {res['Gamma']['share'] * 100:5.1f} %, PAH/ac total "
                     f"{res['Gamma']['ratio']:.2f}   |   Lambda: share {res['Lambda']['share'] * 100:5.1f} %, PAH/ac "
                     f"total {res['Lambda']['ratio']:.2f}")
    lines.append("scan of the carbon mass fraction below N_C = 400 (J13 shapes kept): PAH/ac total (share with ac)")
    for name in CELLS:
        for q in ("Gamma", "Lambda"):
            lines.append(f"  {name:6s} {q:6s} " + "  ".join(
                f"f={f:.2f}: {r:.2f} ({s * 100:.0f} %)" for f, (r, s) in zip(scan, rows[name][q]["scan"])))
    txt = "\n".join(lines)
    print(txt)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".txt").write_text(txt + "\n")
    plot(a_grid, dM, rows, a_edges, out.with_suffix(".pdf"))


def plot(a, dM, rows, a_edges, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pycalima.plotting_style import use_calima_style

    use_calima_style()
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.4))
    ax = axs[0]
    ax.loglog(a * 1e7, dM, "k-")
    ax.axvline(a_edges[-1] * 1e7, color="0.5", ls=":")
    ax.set_xlabel(r"$a$ [nm]")
    ax.set_ylabel(r"$dM/d\ln a$ [per $M_{\rm H}$]")
    ax.set_title("(a) THEMIS J13 carbon", fontsize=9)
    for ax, q, lab in ((axs[1], "Gamma", r"$\Gamma$ per C [erg s$^{-1}$]"),
                       (axs[2], "Lambda", r"$\Lambda$ per C [erg s$^{-1}$]")):
        for k, name in enumerate(rows):
            c = f"C{k}"
            ax.loglog(a * 1e7, rows[name][q]["per_c_ac"], "-", color=c, label=f"{name}, ac")
            ax.loglog(a * 1e7, rows[name][q]["per_c_pah"], "--", color=c, label=f"{name}, PAH below 400 C")
        ax.axvline(a_edges[-1] * 1e7, color="0.5", ls=":")
        ax.set_xlabel(r"$a$ [nm]")
        ax.set_ylabel(lab)
        ax.set_title(f"({'bc'[q == 'Lambda']}) {q} per C atom", fontsize=9)
    axs[1].legend(fontsize=6, ncol=2)
    for ax in axs:
        ax.tick_params(which="both", direction="in", top=True, right=True, bottom=True, left=True)
    fig.tight_layout()
    fig.savefig(path)
    print("wrote", path)


if __name__ == "__main__":
    main()
