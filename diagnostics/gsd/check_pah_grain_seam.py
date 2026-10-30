"""The PAH / grain seam of the carbon GSD: pahchem's PAH tables against the 'ac' grain tables.

In the PAH-size bins of ``CarbonLayout`` (N_C 20-400) the aromatic population is
charged with pahchem's tables (``pahchem.group_tables``: PAHdb molecules, their DFT
thresholds, Jochims/Wenzel yields, statistical sticking on neutrals, ion-molecule
channels); above N_C,max the same carbon is charged as a grain ('ac', ``wdb06_yields``).
For each bin this builds both, at the bin's mean N_C, and compares:

1. thresholds z -> z+1 for z = -1, 0, +1: the molecules' (median and quartiles)
   against 'ac' (and DustEM's 'cm20');
2. electron capture coefficients alpha(z, T) for z = 0, +1, +2;
3. per-particle rates in the Mathis ISRF at G0 = 1 Habing: absorption, photoemission
   (photodetachment) out of each z and the photoelectron power;
4. the charge balance and the gas heating and cooling per C atom in a few test
   cells (fixed inputs, not predictions; ``CELLS``).

The molecules of a bin are all the complete ones (charges -1..+2) of PAHdb v4.00
(``population.library_v400``), or a random ``--max-mol`` of them. Needs the PAHdb
v4.00 library ($CALIMA_PAHDB_DIR), the DustEM data ($CALIMA_DUSTEM_DIR) and the Jones
(2012) tables. Prints the comparison; writes a figure.

    python diagnostics/gsd/check_pah_grain_seam.py --max-mol 60 --workers 2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pycalima import _paths
from pycalima.gsd.carbon import CARRIERS, M_C, CarbonLayout
from pycalima.models.dust_charge import rt_group_charging as rg
from pycalima.models.dust_charge.shared_physics import DS87_J_function_vec, KB_CGS, ME_CGS

AMU = 1.66053906660e-24
EV = 1.602176634e-12
LT_SEAM = np.log10([30.0, 100.0, 300.0, 1e3, 3e3, 8e3, 1e4])
#: name: (T [K], n_H [cm^-3], n_e [cm^-3], ion carrying the electrons, G0 [Habing])
CELLS = {"WNM": (8000.0, 0.4, 7.2e-3, "H+", 1.0),
         "CNM": (100.0, 30.0, 3e-2, "C+", 1.0),
         "PDR": (300.0, 1e4, 1.0, "C+", 1e3),
         "dense": (30.0, 1e3, 1e-3, "C+", 1e-2)}
ION_MASS = {"H+": 1.00794, "C+": 12.0107}


def bins(layout):
    grid = layout.grid()
    nc = grid.m_edges / M_C
    return [(nc[i], nc[i + 1]) for i in np.nonzero(layout.pah_bins(grid))[0]]


def pah_tables(lib, bin_edges, max_mol, workers, seed=1):
    from pycalima.pahchem import group_tables as gtab

    rng = np.random.default_rng(seed)
    comp = np.array([gtab.complete(m) for m in lib])
    out = []
    for lo, hi in bin_edges:
        idx = np.nonzero(comp & np.array([lo <= m.nc < hi for m in lib]))[0]
        if idx.size == 0:
            out.append(None)
            continue
        if max_mol and idx.size > max_mol:
            idx = np.sort(rng.choice(idx, max_mol, replace=False))
        w = np.zeros(len(lib))
        w[idx] = 1.0
        tab = gtab.build_bin_table(lib, w, lT=LT_SEAM, workers=workers, label=f"{lo:.0f}-{hi:.0f}")
        thr = np.array([[lib[i].ionisation_ev[z] for z in (-1, 0, 1)] for i in idx])
        out.append(dict(tab=tab, thresholds=thr, nc=np.array([lib[i].nc for i in idx])))
    return out


def grain(material, nc):
    a = float(CARRIERS["aromatic"].radius_cm(nc * M_C))
    tab = rg.build_bin_table(material, a)
    return a, tab, rg.GroupChargingTable(tab)


def grain_capture(gt, a, z, T):
    v_e = np.sqrt(8 * KB_CGS * T / (np.pi * ME_CGS))
    return np.pi * a * a * v_e * float(DS87_J_function_vec(float(z), -1.0, a, T)) * gt._sticking(float(z))


def grain_bg(gt, a, z):
    i, w = gt._u_index(z * rg.E2_EV_CM / a)
    return float(gt._kern(gt.t["K_bg"], None, i, w)), float(gt._kern(gt.t["H_bg"], None, i, w))


def grain_absorption(material, a):
    from pycalima.models.dust_charge import wdb06_yields as wy
    from pycalima.pahchem.group_tables import BG_GRID, _mathis_flux

    q = wy.absorption_efficiency(BG_GRID, a, material)
    return float(np.trapezoid(np.pi * a * a * q * _mathis_flux(), BG_GRID))


def solve_cells(pah, gt, a):
    from pycalima.pahchem.group_tables import PAHGroupTable

    pt = PAHGroupTable(pah["tab"])
    nG = pah["tab"]["L0"].size
    out = {}
    for name, (T, n_h, n_e, ion, G0) in CELLS.items():
        rp = pt.solve(np.zeros(nG), T, n_e, n_h=n_h, ions=((ion, n_e),), G0_bg=G0)
        rgr = gt.solve(np.zeros(nG), rg.C_CGS, T, n_e, [(n_e, 1.0, ION_MASS[ion] * AMU)], G0_bg=G0, recomb=None)
        Z, P = gt.exact_distribution(rgr["Zstar"], max(rgr["Zsigma"], 0.5))
        f_gr = {z: float(P[Z == z].sum()) for z in (-1, 0, 1, 2)}
        f_gr["<-1"], f_gr[">2"] = float(P[Z < -1].sum()), float(P[Z > 2].sum())
        out[name] = dict(pah_f=rp["f"], pah_Z=float(np.dot((-1, 0, 1, 2), rp["f"])), pah_gamma=rp["gamma"],
                         pah_lam=rp["lam"], gr_f=f_gr, gr_Z=rgr["Zmean"], gr_gamma=rgr["Gamma"],
                         gr_lam=rgr["Lambda"])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--max-mol", type=int, default=60, help="molecules per bin (0: all)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default=None, help="figure path")
    args = ap.parse_args(argv)

    from pycalima.pahchem import population as pop

    layout = CarbonLayout()
    edges = bins(layout)
    lib = pop.library_v400()
    pahs = pah_tables(lib, edges, args.max_mol, args.workers)
    rows = []
    for (lo, hi), pah in zip(edges, pahs):
        if pah is None:
            print(f"\nbin N_C {lo:.0f}-{hi:.0f}: no complete molecule in PAHdb v4.00")
            continue
        nc = pah["tab"]["nc"]
        a, tab, gt = grain("ac", nc)
        _, tab20, gt20 = grain("cm20", nc)
        print(f"\n=== bin N_C {lo:.0f}-{hi:.0f}: {pah['tab']['n_mol']} molecules, mean N_C {nc:.0f}; "
              f"grain a = {a * 1e7:.2f} nm (aromatic carrier, 1.6 g/cm^3)")
        q = np.percentile(pah["thresholds"], [25, 50, 75], axis=0)
        print("threshold z->z+1 [eV]   PAHs: median [IQR]          ac      cm20")
        for k, z in enumerate((-1, 0, 1)):
            g = float(rg.grain_ionisation_potential("ac", a, tab["W"], float(z)))
            g20 = float(rg.grain_ionisation_potential("cm20", a, tab20["W"], float(z)))
            print(f"   z = {z:+d}          {q[1, k]:6.2f} [{q[0, k]:.2f}, {q[2, k]:.2f}]      {g:6.2f}  {g20:6.2f}")
        print("electron capture alpha(z, T) [cm^3/s]: PAHs / ac")
        T = 10.0 ** LT_SEAM
        for k, z in enumerate((0, 1, 2)):
            ap_ = pah["tab"]["alpha"][k]
            ag = np.array([grain_capture(gt, a, z, t) for t in T])
            print(f"   z = {z:+d}  " + " ".join(f"{t:5.0f}K:{x:.2e}/{y:.2e}" for t, x, y in zip(T, ap_, ag)))
        print("Mathis ISRF, G0 = 1, per particle: PAHs / ac")
        A_g = grain_absorption("ac", a)
        print(f"   absorptions [s^-1]: {pah['tab']['A_bg'][1]:.3e} (neutral) / {A_g:.3e}")
        for k, z in enumerate((-1, 0, 1, 2)):
            Kg, Hg = grain_bg(gt, a, z)
            print(f"   z = {z:+d}: photoemission {pah['tab']['K_bg'][k]:.3e} / {Kg:.3e} s^-1; "
                  f"electron power {pah['tab']['H_bg'][k]:.3e} / {Hg:.3e} eV s^-1")
        cells = solve_cells(pah, gt, a)
        print("cells: f(-1, 0, +1, +2) and heating, cooling per C [erg/s]: PAHs | ac")
        for name, r in cells.items():
            fg = r["gr_f"]
            print(f"   {name:5s} PAH f = {np.round(r['pah_f'], 3)}  <Z> {r['pah_Z']:+.2f}  "
                  f"G/C {r['pah_gamma'] / nc:.2e}  L/C {r['pah_lam'] / nc:.2e}")
            print(f"         ac  f = [{fg[-1]:.3f} {fg[0]:.3f} {fg[1]:.3f} {fg[2]:.3f}] (<-1 {fg['<-1']:.3f}, >2 {fg['>2']:.3f}) "
                  f"<Z> {r['gr_Z']:+.2f}  G/C {r['gr_gamma'] / nc:.2e}  L/C {r['gr_lam'] / nc:.2e}")
        rows.append(dict(lo=lo, hi=hi, nc=nc, a=a, pah=pah, cells=cells, gt=gt, tab=tab))
    plot(rows, lib, args.out)


def plot(rows, lib, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pycalima.pahchem import group_tables as gtab
    from pycalima.plotting_style import use_calima_style

    use_calima_style()
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
    comp = [m for m in lib if gtab.complete(m)]
    nc_all = np.array([m.nc for m in comp])
    ncg = np.geomspace(20, 400, 40)
    ax = axs[0]
    for k, (z, c) in enumerate(((-1, "C0"), (0, "C1"), (1, "C2"))):
        ax.plot(nc_all, [m.ionisation_ev[z] for m in comp], ".", ms=2, color=c, alpha=0.4)
        g = [float(rg.grain_ionisation_potential("ac", float(CARRIERS["aromatic"].radius_cm(n * M_C)),
                                                 4.75, float(z))) for n in ncg]
        ax.plot(ncg, g, "-", color=c, label=fr"$z = {z:+d}\to{z + 1:+d}$")
        g20 = [float(rg.grain_ionisation_potential("cm20", float(CARRIERS["aromatic"].radius_cm(n * M_C)),
                                                   4.75, float(z))) for n in ncg]
        ax.plot(ncg, g20, ":", color=c)
    ax.set_xscale("log")
    ax.set_xlabel(r"$N_{\rm C}$")
    ax.set_ylabel("threshold [eV] (dots: PAHdb; lines: ac; dotted: DustEM)")
    ax.legend(frameon=False, fontsize=8)
    for j, (cell, ax) in enumerate(zip(("WNM", "PDR"), axs[1:])):
        nc = [r["nc"] for r in rows]
        ax.plot(nc, [r["cells"][cell]["pah_gamma"] / r["nc"] for r in rows], "o-", label=r"PAHs: $\Gamma$")
        ax.plot(nc, [r["cells"][cell]["gr_gamma"] / r["nc"] for r in rows], "s--", label=r"ac: $\Gamma$")
        ax.plot(nc, [r["cells"][cell]["pah_lam"] / r["nc"] for r in rows], "o-", label=r"PAHs: $\Lambda$")
        ax.plot(nc, [r["cells"][cell]["gr_lam"] / r["nc"] for r in rows], "s--", label=r"ac: $\Lambda$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        T, n_h, n_e, ion, G0 = CELLS[cell]
        ax.set_title(fr"{cell}: $T$ = {T:g} K, $n_e$ = {n_e:g}, $G_0$ = {G0:g}", fontsize=10)
        ax.set_xlabel(r"bin mean $N_{\rm C}$")
        ax.set_ylabel("erg s$^{-1}$ per C atom")
        ax.legend(frameon=False, fontsize=8)
    from matplotlib.ticker import NullFormatter

    for ax in axs:
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.minorticks_on()
        ax.tick_params(which="both", axis="both", direction="in")
    fig.tight_layout()
    out = Path(out) if out else _paths.get_plots_dir() / "gsd" / "check_pah_grain_seam.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
