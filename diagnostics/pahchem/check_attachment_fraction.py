#!/usr/bin/env python3
"""Sensitivity of the PAH charge balance and gas coupling to the neutral-PAH electron attachment fraction f.

``Params.attachment_fraction`` (default 1) multiplies the neutral-PAH electron attachment rate. Carelli+13's ab initio scattering suggests
f ~ 0.003, or f ~ 0.08/N_C if the temporary-anion rate stays ~3e-9 cm^3/s at every size. Same setup as ``check_wd01_per_size.py`` (712
complete PAHdb v4.00 molecules of N_C 20-270, one-molecule group tables, T = 100, 1000, 6000 K, psi = 1e2..1e6, per grain against WD01),
with eta = 1 (default ``capture_scale``) and three variants: f = 1 (default), f = 0.003, f = min(1, 0.08/N_C) per molecule.

The default is recomputed here with a worker that also returns the charge distribution f(Z), Z = -1, 0, +1, +2, and checked against the
saved ``wd01_per_size_2026-11-01/per_size.json`` (the saved values are the ones used in the ratios). WD01 curves come from that file.
Outputs: ``attachment_fraction.json``, ``report.txt`` (ratio tables in ``check_wd01_per_size.report`` format, plus the median f(Z) per
N_C bin) and two figures (<Z> against N_C; per-grain ratios against WD01).

Result (2026-11-01): f acts only in the anion regime, psi <~ 3e3; at psi >= 1e4 the three variants agree to the printed precision. At
psi = 1e2 the anion fraction falls from 0.47-0.96 (f = 1) to 0.003-0.06 (f = 0.003), and at psi <= 1e3 pahchem per grain goes from above
WD01 to below it: alpha(H+) 1.1-2.3 -> 0.17-0.24, Gamma_PE 1.4-1.5 -> 0.46-0.94, Lambda_rec 1.7-2.9 -> 0.3-0.7. f = 0.003 and
f = 0.08/N_C agree to a few % in every gas-coupling quantity: the scale of f matters, not its size dependence. (The PDRs4All fits,
``fit_pdrs4all_spectra.py --attachment-fraction``, prefer f ~ 1 in the dissociation fronts.)

    check_attachment_fraction.py [--workers 8] [--limit N] [--plot-only]
"""
import argparse
import contextlib
import io
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import check_wd01_per_size as S

from pycalima import _paths
from pycalima.pahchem import chemistry, photo
from pycalima.pahchem import group_tables as G
from pycalima.pahchem import population
from pycalima.pahchem.baseline import Params
from pycalima.plotting_style import use_calima_style

T_GRID, PSI, QUANT, BINS = S.T_GRID, S.PSI, S.QUANT, S.BINS
NAMES = ("f = 1 (default)", "f = 0.003", "f = 0.08/N_C")
COLORS = {"f = 1 (default)": "#2a78d6", "f = 0.003": "#eb6834", "f = 0.08/N_C": "#1baf7a"}
SAVED = _paths.get_results_dir("pahchem") / "wd01_per_size_2026-11-01" / "per_size.json"
OUT = _paths.get_results_dir("pahchem") / "attachment_fraction_2026-11-01"
PSI_F = (1e2, 1e3, 1e4, 1e5)


def params_for(name, mol):
    if name == "f = 0.003":
        return Params(attachment_fraction=0.003)
    if name == "f = 0.08/N_C":
        return Params(attachment_fraction=min(1.0, 0.08 / mol.nc))
    return Params()


def molecule_job(job):
    """(quantities (T, psi, 5), f(Z) (T, psi, 4)) of one molecule from its own table; as check_wd01_per_size.molecule_job plus f(Z)."""
    mol, params = job
    k = G.molecule_kernels((mol, params, photo.PHOTOELECTRON_PARTITION, G.rg.DEFAULT_GROUP_L0, G.rg.DEFAULT_GROUP_L1, G.N_E_PER_GROUP, G.LT))
    tab = G.PAHGroupTable(G.combine([k], np.array([1.0]), G.rg.DEFAULT_GROUP_L0, G.rg.DEFAULT_GROUP_L1, G.N_E_PER_GROUP, G.rg.TS_GRID, G.LT,
                                    mol.name, ""))
    zs = np.array(G.CHARGES, dtype=float)
    fb = G._mathis_flux()
    absb = [photo.sigma_uv(mol.nc, z, G.BG_GRID, params.uv_cross_section) * params.sigma_scale * fb * G.BG_GRID for z in G.CHARGES]
    p_abs = np.array([np.trapezoid(a, G.BG_GRID) * chemistry.EV_ERG for a in absb])
    q = np.zeros((len(T_GRID), PSI.size, len(QUANT)))
    fz = np.zeros((len(T_GRID), PSI.size, len(zs)))
    for i, t in enumerate(T_GRID):
        for j, psi in enumerate(PSI):
            ne = np.sqrt(t) / psi
            r = tab.solve(np.zeros(tab.t["L0"].size), t, ne, 0.0, (("H+", ne),), G0_bg=1.0)
            f = r["f"]
            q[i, j] = (r["alpha_mn"]["H+"] + r["alpha_ct"]["H+"], r["gamma"], r["lam"], f @ zs, (r["gamma"] - r["lam"]) / float(f @ p_abs))
            fz[i, j] = f
    return q, fz


def molecules():
    lib = population.shift_thresholds(population.library_v400(), S.W.IP_SHIFTS)
    return [m for m, x in zip(lib, G.nc_range_weights(lib, 20, 270)) if x > 0]


def compute(workers, limit):
    mols = molecules()[:limit] if limit else molecules()
    res = {"nc": [m.nc for m in mols], "name": [m.name for m in mols]}
    fz = {}
    for name in NAMES:
        print(f"{name}: {len(mols)} molecules", flush=True)
        with ProcessPoolExecutor(workers) as ex:
            out = list(ex.map(molecule_job, [(m, params_for(name, m)) for m in mols], chunksize=4))
        res[name] = np.array([o[0] for o in out]).tolist()
        fz[name] = np.array([o[1] for o in out]).tolist()
    return res, fz


def ratios_all(d):
    """Per-grain ratios against WD01 for the three variants (check_wd01_per_size.ratios with this script's variant names)."""
    S.VARIANTS = {n: None for n in NAMES}
    return S.ratios(d)


def report_fz(d):
    nc = np.array(d["pahchem"]["nc"])
    for name in NAMES:
        fz = np.array(d["fz"][name])
        print(f"\n== {name}: median f(Z) per N_C bin (Z = -1, 0, +1, +2)")
        for i, t in enumerate(T_GRID):
            print(f"\nT = {t:.0f} K")
            print(f"{'N_C bin':>9s} {'psi':>8s} " + " ".join(f"{'f(%+d)' % z:>11s}" for z in G.CHARGES))
            for lo, hi in BINS:
                m = (nc >= lo) & (nc < hi)
                for p0 in PSI_F:
                    j = int(np.argmin(abs(np.log(PSI / p0))))
                    med = np.median(fz[m, i, j, :], axis=0)
                    print(f"{lo:4d}-{hi:<4d} {PSI[j]:8.0e} " + " ".join(f"{v:11.4e}" for v in med))


def plot_zbar(d, path):
    use_calima_style()
    nc = np.array(d["pahchem"]["nc"], dtype=float)
    wd = np.array(d["wd01"])
    zi = QUANT.index("<Z>")
    fig, axes = plt.subplots(len(T_GRID), len(PSI_F), figsize=(14, 9), sharex=True)
    for i, t in enumerate(T_GRID):
        for c, p0 in enumerate(PSI_F):
            ax = axes[i, c]
            j = int(np.argmin(abs(np.log(PSI / p0))))
            for name in NAMES:
                x, y = S.binned(nc, np.array(d["pahchem"][name])[:, i, j, zi], None)
                ax.plot(x, y, color=COLORS[name], lw=2.0, label=name.replace("_", r"\_"))
            ax.plot(d["nc_wd"], wd[i, j, :, zi], color="k", ls="--", lw=1.5, label="WD01")
            ax.set_xscale("log")
            ax.minorticks_on()
            ax.tick_params(which="both", direction="in", top=True, right=True)
            if i == 0:
                ax.set_title(rf"$\psi=10^{{{int(round(np.log10(PSI[j])))}}}$")
            if c == 0:
                ax.set_ylabel(rf"$T={t:.0f}$ K: median $\langle Z\rangle$")
            if i == len(T_GRID) - 1:
                ax.set_xlabel(r"$N_{\rm C}$")
    axes[0, 0].legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_ratios(d, rat, path):
    use_calima_style()
    nc = np.array(d["pahchem"]["nc"], dtype=float)
    rows = (("Gamma_PE", r"$\Gamma_{\rm PE}$ per grain, pahchem / WD01"), ("alpha(H+)", r"$\alpha({\rm H^+})$ per grain, pahchem / WD01"),
            ("Lambda_rec", r"$\Lambda_{\rm rec}$ per grain, pahchem / WD01"), ("<Z>", r"$\langle Z\rangle$, pahchem $-$ WD01"))
    psi_show = ((1e3, "-"), (1e5, "--"))
    fig, axes = plt.subplots(len(rows), len(T_GRID), figsize=(14, 12), sharex=True)
    for r_i, (q, lab) in enumerate(rows):
        qi = QUANT.index(q)
        for c, t in enumerate(T_GRID):
            ax = axes[r_i, c]
            for p0, ls in psi_show:
                j = int(np.argmin(abs(np.log(PSI / p0))))
                for name in NAMES:
                    x, y = S.binned(nc, rat[name][c, j, :, qi], None)
                    ax.plot(x, y, color=COLORS[name], ls=ls, lw=2.0)
            ax.axhline(0.0 if q == "<Z>" else 1.0, color="k", lw=0.8)
            ax.set_xscale("log")
            if q != "<Z>":
                ax.set_yscale("log")
            ax.minorticks_on()
            ax.tick_params(which="both", direction="in", top=True, right=True)
            if c == 0:
                ax.set_ylabel(lab)
            if r_i == 0:
                ax.set_title(rf"$T={t:.0f}$ K")
            if r_i == len(rows) - 1:
                ax.set_xlabel(r"$N_{\rm C}$")
    h = [Line2D([], [], color=COLORS[n], lw=2) for n in NAMES] + [Line2D([], [], color="0.3", ls=ls, lw=2) for _, ls in psi_show]
    axes[0, 0].legend(h, [n.replace("_", r"\_") for n in NAMES] + [r"$\psi=10^3$", r"$\psi=10^5$"], fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def anomalies(d):
    for name in NAMES:
        q = np.array(d["pahchem"][name])
        fz = np.array(d["fz"][name])
        nc = np.array(d["pahchem"]["nc"])
        eps = q[..., QUANT.index("eps")]
        print(f"[{name}] non-finite quantities: {int((~np.isfinite(q)).sum())}; non-finite f(Z): {int((~np.isfinite(fz)).sum())}; "
              f"eps < 0: {int((eps < 0).sum())} of {eps.size}; f(Z) < 0: {int((fz < 0).sum())}; max |sum f - 1|: {np.abs(fz.sum(-1) - 1).max():.2e}")
        for qi, qn in enumerate(QUANT[:3]):
            print(f"    {qn} <= 0: {int((q[..., qi] <= 0).sum())}")
        neg = np.argwhere(eps < 0)
        if len(neg):
            ts = sorted({(T_GRID[i], PSI[j]) for _, i, j in neg})
            print(f"    eps < 0 at (T, psi): {[(float(a), float(b)) for a, b in ts]}; N_C range {nc[neg[:, 0]].min()}-{nc[neg[:, 0]].max()}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="only the first N molecules; prints timing, writes nothing")
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / "attachment_fraction.json"
    saved = json.loads(SAVED.read_text())
    if a.limit:
        t0 = time.time()
        res, fz = compute(a.workers, a.limit)
        print(f"{a.limit} molecule(s) x {len(NAMES)} variants: {time.time() - t0:.1f} s")
        return
    if a.plot_only and f.exists():
        d = json.loads(f.read_text())
    else:
        t0 = time.time()
        res, fz = compute(a.workers, 0)
        assert res["nc"] == saved["pahchem"]["nc"] and res["name"] == saved["pahchem"]["name"]
        old = np.array(saved["pahchem"]["eta = 1 (default)"])
        new = np.array(res["f = 1 (default)"])
        diff = float(np.max(np.abs(new - old) / np.maximum(np.abs(old), 1e-300)))
        d = {"T": saved["T"], "psi": saved["psi"], "nc_wd": saved["nc_wd"], "quantities": QUANT, "wd01": saved["wd01"],
             "pahchem": res, "fz": fz, "default_recompute_max_rel_diff": diff, "wall_s": time.time() - t0}
        d["pahchem_recomputed_default"] = res["f = 1 (default)"]
        d["pahchem"]["f = 1 (default)"] = saved["pahchem"]["eta = 1 (default)"]
        f.write_text(json.dumps(d))
    rat = ratios_all(d)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print(f"recomputed default vs per_size.json: max relative difference {d['default_recompute_max_rel_diff']:.3e}; wall {d['wall_s']:.0f} s")
        S.report(d, rat)
        report_fz(d)
        print("\n== anomalies")
        anomalies(d)
    (OUT / "report.txt").write_text(buf.getvalue())
    print(buf.getvalue())
    out = _paths.get_plots_dir("pahchem")
    out.mkdir(parents=True, exist_ok=True)
    plot_zbar(d, out / "pahchem_attachment_fraction_zbar.png")
    plot_ratios(d, rat, out / "pahchem_attachment_fraction_ratios.png")
    print(f"figures in {out}")


if __name__ == "__main__":
    main()
