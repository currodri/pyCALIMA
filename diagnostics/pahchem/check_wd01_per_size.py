#!/usr/bin/env python3
"""pahchem against WD01 one grain size at a time, and the photoelectric efficiency against the other models.

``check_wd01b_alpha.py`` compares bin totals, which carry WD01's size distribution and abundances. Here every quantity is **per grain**:
for each complete PAHdb v4.00 molecule of N_C 20-270 (712 of them; calibrated IPs, ``W.IP_SHIFTS``), pahchem's table of that one molecule
(``group_tables.molecule_kernels`` + ``combine`` with one weight) is solved at T = 100, 1000, 6000 K and psi = G sqrt(T)/n_e = 1e2..1e6
(G = 1 Habing, Mathis shape, n_H = 0, n(H+) = n_e), and the result is divided by WD01's physics (``check_wd01b_alpha._alpha_full``: pyCALIMA's
full-spectrum charging with WD01 yields and case A thresholds) for ONE graphite grain of the same carbon count, a = 10 A (N_C/468)^(1/3). WD01
is evaluated on 14 sizes and interpolated in ln N_C to each molecule's N_C. No size distribution, no abundance. Two pahchem variants: the
defaults (eta = 1) and eta = 0.46 (``capture_scale``, Tielens' capture limit; labelled, not a default).

Quantities: alpha(H+), Gamma_PE, Lambda_rec per grain, <Z>, and the net heating efficiency eps = (Gamma - Lambda_rec) / P_abs, with P_abs the power
absorbed in the Mathis field between 1 and 13.6 eV, charge-weighted (WD01's own ``efficiency`` is over the field it is given, 0.1-13.6 eV;
Bakes & Tielens 1994 / Wolfire+03 normalise to 6-13.6 eV, Berne+22 to everything below 13.6 eV: the conventions differ by a few tens of %
at the highest gamma and are not reconciled here).

The second figure puts pahchem's eps(gamma) against: Bakes & Tielens (1994) as Wolfire+03 write it with gamma = G0 sqrt(T)/n_e
(4.9e-2/(1+2.411e-3 g^0.73) + 3.7e-2 (T/1e4)^0.7/(1+1e-4 g), the form in ``PAH_photoelectric_heating.Wolfire2003_efficiency``, phi_PAH = 0.5
folded in), Tielens (2001) 0.06/(1+7e-5 g) (same module), WD01 per grain, and, with ``--berne-dir``, Berne+22's own model (their inputs, N_C = 54,
T = 100 K: ``check_berne22_pe_heating`` step 0, which reproduces their code to 0.3 %). Berne+22 use Carelli+13's coronene attachment rate.

Result (2026-11-01, library UV; median over the molecules of a bin, per grain, pahchem / WD01):
- psi <= 1e3, T = 1000 K: alpha(H+) 1.09-1.36, Gamma_PE 1.39-1.47, Lambda_rec 1.7-2.9 (eta = 1); 0.84-1.15, 1.15-1.28, 1.5-2.0
  (eta = 0.46). <Z> is 0.36-0.48 more negative than WD01's (0.17-0.31 with eta = 0.46): the anions.
- psi >= 1e5: alpha(H+) grows with size, 12x (N_C 20-40) to 171x (120-270) with eta = 1, 3x to 38x with eta = 0.46, while Gamma_PE
  is flat at 7-9x (3-4x): <Z> is up to 1.2 below WD01's at large N_C, partly the chain's +2 top state (WD01 reaches +3 and more).
- The isomer spread at one N_C (+-20-30 % in Gamma) is far below the offset from WD01.
- eps(gamma): WD01's grains lie 2-4x below pahchem at gamma >= 1e4; eta = 0.46 follows Bakes & Tielens / Wolfire+03 at
  gamma >= 1e4 (100, 1000 K), eta = 1 lies 1.5-2x above it; Berne+22's model is within ~30 %. At 6000 K pahchem and WD01 turn to net
  cooling below gamma ~ 300, which the BT94 fit does not.

    check_wd01_per_size.py [--workers 8] [--berne-dir ~/Documents/GitHub/photoelectric-heating] [--plot-only]
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "diagnostics" / "dust_charge"))
import check_wd01b_alpha as W

from pycalima import _paths
from pycalima.pahchem import chemistry, photo
from pycalima.pahchem import group_tables as G
from pycalima.pahchem import population
from pycalima.pahchem.baseline import Params
from pycalima.plotting_style import use_calima_style

T_GRID, PSI = W.T_GRID, W.PSI_GRID
NC_WD = np.geomspace(20.0, 270.0, 14)
QUANT = ("alpha(H+)", "Gamma_PE", "Lambda_rec", "<Z>", "eps")
VARIANTS = {"eta = 1 (default)": Params(), "eta = 0.46": Params(capture_scale=0.46)}
OUT = _paths.get_results_dir("pahchem") / "wd01_per_size_2026-11-01"
BINS = ((20, 40), (40, 70), (70, 120), (120, 270))
RANGES = (("psi <= 1e3", 0.0, 1e3), ("1e3 < psi <= 1e4", 1e3, 1e4), ("psi >= 1e5", 1e5, 1e9))


def wd01_job(args):
    """(alpha(H+), Gamma, Lambda, <Z>, eps) of one WD01 graphite grain."""
    from pycalima.models.dust_charge.dust_charging import equilibrium_charge_for_grain

    g0m, ne, t, a = args
    r = W._alpha_full((g0m, ne, t, "graphite", a, "WD01"))
    n_ion = len(W.R.ION_INDEX)
    eff = equilibrium_charge_for_grain(g0m, ne, t, "graphite", a, yield_params={"yield_model": "WD01"},
                                       ion_species=[{"n": ne, "T": t, "m": 1.67262192e-24, "z": 1.0}])[2]["efficiency"]
    return [r[W.R.ION_INDEX["H+"]], r[n_ion], r[n_ion + 1], r[n_ion + 2], eff]


def wd01_curves(workers):
    gm = W.mathis_G()
    jobs = [(1.0 / gm, np.sqrt(t) / psi, t, float(W.radius_cm(nc))) for t in T_GRID for psi in PSI for nc in NC_WD]
    print(f"WD01: {len(jobs)} grain solves", flush=True)
    with ProcessPoolExecutor(workers) as ex:
        out = np.array(list(ex.map(wd01_job, jobs, chunksize=8)))
    return out.reshape(len(T_GRID), PSI.size, NC_WD.size, len(QUANT)).tolist()


def molecule_job(job):
    """(T, psi, quantity) of one molecule from its own table."""
    mol, params = job
    k = G.molecule_kernels((mol, params, photo.PHOTOELECTRON_PARTITION, G.rg.DEFAULT_GROUP_L0, G.rg.DEFAULT_GROUP_L1, G.N_E_PER_GROUP, G.LT))
    tab = G.PAHGroupTable(G.combine([k], np.array([1.0]), G.rg.DEFAULT_GROUP_L0, G.rg.DEFAULT_GROUP_L1, G.N_E_PER_GROUP, G.rg.TS_GRID, G.LT,
                                    mol.name, ""))
    zs = np.array(G.CHARGES, dtype=float)
    fb = G._mathis_flux()
    absb = [photo.sigma_uv(mol.nc, z, G.BG_GRID, params.uv_cross_section) * params.sigma_scale * fb * G.BG_GRID for z in G.CHARGES]
    p_abs = [np.trapezoid(a, G.BG_GRID) * chemistry.EV_ERG for a in absb]
    out = np.zeros((len(T_GRID), PSI.size, len(QUANT)))
    for i, t in enumerate(T_GRID):
        for j, psi in enumerate(PSI):
            ne = np.sqrt(t) / psi
            r = tab.solve(np.zeros(tab.t["L0"].size), t, ne, 0.0, (("H+", ne),), G0_bg=1.0)
            f = r["f"]
            out[i, j] = (r["alpha_mn"]["H+"] + r["alpha_ct"]["H+"], r["gamma"], r["lam"], f @ zs,
                         (r["gamma"] - r["lam"]) / float(f @ np.array(p_abs)))
    return out


def pahchem_curves(workers):
    lib = population.shift_thresholds(population.library_v400(), W.IP_SHIFTS)
    mols = [m for m, x in zip(lib, G.nc_range_weights(lib, 20, 270)) if x > 0]
    res = {"nc": [m.nc for m in mols], "name": [m.name for m in mols]}
    for name, params in VARIANTS.items():
        print(f"pahchem {name}: {len(mols)} molecules", flush=True)
        with ProcessPoolExecutor(workers) as ex:
            res[name] = np.array(list(ex.map(molecule_job, [(m, params) for m in mols], chunksize=4))).tolist()
    return res


def berne_curve(berne_dir):
    """Berne+22's model (their inputs, N_C = 54, T = 100 K): [(gamma, Gamma per molecule, eps)] over n_e."""
    import check_berne22_pe_heating as B

    Eb, Fb, g0 = B.berne_field(berne_dir)
    keep = Eb <= 13.6
    Eb, Fb = Eb[keep], Fb[keep]
    a = (B.NC / 468.0) ** (1 / 3)
    e2 = 1.439964
    ip0 = 3.9 + 0.5 * e2 / a + 2 * e2 / a * 0.03 / a
    ip1 = 3.9 + 1.5 * e2 / a + 3 * e2 / a * 0.03 / a
    sigma = {z: B.berne_sigma(berne_dir, z, Eb) for z in (-1, 0, 1, 2)}
    yields = {-1: (Eb >= 6.0).astype(float), 0: photo.yield_jochims1996(Eb, ip0), 1: photo.yield_wenzel2020(Eb, ip1, B.NC)}
    th, part = {-1: 6.0, 0: ip0, 1: ip1}, {-1: 1.0, 0: B.PARTITION, 1: B.PARTITION}
    out = []
    for ne in np.geomspace(3e-4, 3e1, 13):
        phi = 1.85e5 / (B.T * np.sqrt(B.NC))
        down = {-1: ne * 2.74e-9 * (B.T / 300) ** 0.11 * np.exp(1.12 / B.T), 0: ne * 1.28e-10 * B.NC * np.sqrt(B.T) * (1 + phi),
                1: ne * 1.28e-10 * B.NC * np.sqrt(B.T) * (1 + 2 * phi)}
        _, gam, eps = B.chain(Eb, B.Flux(Eb, Fb), sigma, yields, th, part, down, 0.0)
        out.append((g0 * np.sqrt(B.T) / ne, gam, eps))
    return out


def ratios(d):
    """{variant: array (T, psi, molecule, quantity)} of pahchem over WD01 (<Z>: the difference), WD01 interpolated in ln N_C."""
    nc = np.log(np.array(d["pahchem"]["nc"], dtype=float))
    wd = np.array(d["wd01"])
    out = {}
    for name in VARIANTS:
        p = np.array(d["pahchem"][name])
        r = np.zeros_like(p)
        for i in range(len(T_GRID)):
            for j in range(PSI.size):
                for q in range(len(QUANT)):
                    w = np.interp(nc, np.log(NC_WD), wd[i, j, :, q])
                    r[:, i, j, q] = (p[:, i, j, q] - w) if QUANT[q] == "<Z>" else p[:, i, j, q] / w
        out[name] = np.transpose(r, (1, 2, 0, 3))
    return out


def report(d, rat):
    nc = np.array(d["pahchem"]["nc"])
    for name, r in rat.items():
        print(f"\n== {name}: median [16-84 %] of pahchem / WD01 per grain (<Z>: pahchem - WD01); molecules per N_C bin: "
              + ", ".join(f"{lo}-{hi}: {int(((nc >= lo) & (nc < hi)).sum())}" for lo, hi in BINS))
        for i, t in enumerate(T_GRID):
            print(f"\nT = {t:.0f} K")
            print(f"{'N_C bin':>9s} {'psi range':>17s} " + " ".join(f"{q:>26s}" for q in QUANT))
            for lo, hi in BINS:
                m = (nc >= lo) & (nc < hi)
                for lab, plo, phi in RANGES:
                    k = (PSI > plo) & (PSI <= phi)
                    cells = []
                    for q in range(len(QUANT)):
                        v = r[i][k][:, m, q].ravel()
                        if QUANT[q] != "<Z>":
                            v = np.log(v)
                        a, b, c = np.percentile(v, [50, 16, 84])
                        f = (lambda x: x) if QUANT[q] == "<Z>" else np.exp
                        cells.append(f"{f(a):7.2f} [{f(b):5.2f}-{f(c):5.2f}]".rjust(26))
                    print(f"{lo:4d}-{hi:<4d} {lab:>17s} " + " ".join(cells))


def binned(nc, y, edges):
    lo = np.geomspace(20, 270, 9)
    xs, ys = [], []
    for a, b in zip(lo[:-1], lo[1:]):
        m = (nc >= a) & (nc < b)
        if m.sum() >= 3:
            xs.append(np.sqrt(a * b))
            ys.append(np.median(y[m]))
    return np.array(xs), np.array(ys)


def plot_sizes(d, rat, path):
    use_calima_style()
    nc = np.array(d["pahchem"]["nc"], dtype=float)
    rows = (("Gamma_PE", r"$\Gamma_{\rm PE}$ per grain, pahchem / WD01"), ("alpha(H+)", r"$\alpha({\rm H^+})$ per grain, pahchem / WD01"),
            ("Lambda_rec", r"$\Lambda_{\rm rec}$ per grain, pahchem / WD01"), ("<Z>", r"$\langle Z\rangle$, pahchem $-$ WD01"))
    psi_show = ((1e3, "#2a78d6"), (1e4, "#1baf7a"), (1e5, "#eb6834"))
    fig, axes = plt.subplots(len(rows), len(T_GRID), figsize=(14, 12), sharex=True)
    for r_i, (q, lab) in enumerate(rows):
        qi = QUANT.index(q)
        for c, t in enumerate(T_GRID):
            ax = axes[r_i, c]
            for psi0, col in psi_show:
                j = int(np.argmin(abs(np.log(PSI / psi0))))
                y1 = rat["eta = 1 (default)"][c, j, :, qi]
                ax.scatter(nc, y1, s=4, color=col, alpha=0.25, lw=0, rasterized=True)
                for name, ls in (("eta = 1 (default)", "-"), ("eta = 0.46", "--")):
                    x, y = binned(nc, rat[name][c, j, :, qi], None)
                    ax.plot(x, y, color=col, ls=ls, lw=2.0, label=(rf"$\psi=10^{{{int(np.log10(PSI[j]))}}}$" if (name == "eta = 1 (default)" and r_i == 0 and c == 0) else None))
            ax.axhline(0.0 if q == "<Z>" else 1.0, color="k", lw=0.8)
            if q != "<Z>":
                ax.set_yscale("log")
            ax.set_xscale("log")
            ax.minorticks_on()
            ax.tick_params(which="both", direction="in", top=True, right=True)
            if c == 0:
                ax.set_ylabel(lab)
            if r_i == 0:
                ax.set_title(rf"$T={t:.0f}$ K")
            if r_i == len(rows) - 1:
                ax.set_xlabel(r"$N_{\rm C}$")
    h, lb = axes[0, 0].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    h += [Line2D([], [], color="0.3", ls="-", lw=2), Line2D([], [], color="0.3", ls="--", lw=2)]
    lb += [r"median, $\eta=1$ (default)", r"median, $\eta=0.46$"]
    axes[0, 0].legend(h, lb, fontsize=8, loc="best")
    fig.suptitle("pahchem (each v4.00 molecule, library UV) against WD01's physics for one graphite grain of the same N_C: no size distribution", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_eff(d, berne, path):
    use_calima_style()
    nc = np.array(d["pahchem"]["nc"], dtype=float)
    qi = QUANT.index("eps")
    wd = np.array(d["wd01"])
    fig, axes = plt.subplots(1, len(T_GRID), figsize=(15, 4.8), sharey=True)
    g = np.logspace(0, 8, 200)
    classes = ((20, 40, "#2a78d6"), (40, 100, "#1baf7a"), (100, 270, "#eb6834"))
    for c, t in enumerate(T_GRID):
        ax = axes[c]
        for lo, hi, col in classes:
            m = (nc >= lo) & (nc < hi)
            for name, ls in (("eta = 1 (default)", "-"), ("eta = 0.46", "--")):
                p = np.array(d["pahchem"][name])[m, c, :, qi]
                ax.plot(PSI, np.median(p, axis=0), color=col, ls=ls, lw=2.0,
                        label=(rf"pahchem $N_{{\rm C}}$ {lo}-{hi}" if name == "eta = 1 (default)" else None))
                if name == "eta = 1 (default)":
                    ax.fill_between(PSI, np.percentile(p, 16, axis=0), np.percentile(p, 84, axis=0), color=col, alpha=0.15, lw=0)
        for n0, ls in ((24, ":"), (54, "-."), (150, (0, (5, 1)))):
            w = [np.interp(np.log(n0), np.log(NC_WD), wd[c, j, :, qi]) for j in range(PSI.size)]
            ax.plot(PSI, w, color="k", ls=ls, lw=1.5, label=rf"WD01 grain, $N_{{\rm C}}={n0}$")
        ax.plot(g, 4.9e-2 / (1 + 2.411e-3 * g**0.73) + 3.7e-2 * (t / 1e4) ** 0.7 / (1 + 1e-4 * g), color="#8a5cc2", lw=2.5, label=r"Bakes \& Tielens 94 / Wolfire+03")
        ax.plot(g, 0.06 / (1 + 7e-5 * g), color="0.5", lw=2.0, ls=(0, (1, 1)), label="Tielens 2001")
        if berne and abs(t - 100.0) < 1:
            b = np.array(berne)
            ax.plot(b[:, 0], b[:, 2], "o-", color="#e87ba4", lw=2.0, ms=4, label=r"Berné+22 (their model, C54, 100 K)")
        ax.set(xscale="log", yscale="log", xlabel=r"$\gamma = G_0\sqrt{T}/n_e$", title=rf"$T={t:.0f}$ K", xlim=(1e2, 1e6), ylim=(1e-4, 0.3))
        ax.minorticks_on()
        ax.tick_params(which="both", direction="in", top=True, right=True)
    axes[0].set_ylabel(r"net heating efficiency $\epsilon$")
    axes[0].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--berne-dir", type=Path, default=None)
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--out", type=Path, default=None, help="plot directory (default results/plots/pahchem)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / "per_size.json"
    if a.plot_only and f.exists():
        d = json.loads(f.read_text())
    else:
        d = {"T": list(T_GRID), "psi": PSI.tolist(), "nc_wd": NC_WD.tolist(), "quantities": QUANT,
             "wd01": wd01_curves(a.workers), "pahchem": pahchem_curves(a.workers)}
        d["berne"] = berne_curve(a.berne_dir.expanduser()) if a.berne_dir else None
        f.write_text(json.dumps(d))
    rat = ratios(d)
    txt = OUT / "report.txt"
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report(d, rat)
    txt.write_text(buf.getvalue())
    print(buf.getvalue())
    out = a.out or _paths.get_plots_dir("pahchem")
    out.mkdir(parents=True, exist_ok=True)
    plot_sizes(d, rat, out / "pahchem_wd01_per_size.png")
    plot_eff(d, d.get("berne"), out / "pahchem_pe_efficiency_vs_models.png")
    print(f"figures in {out}")


if __name__ == "__main__":
    main()
