#!/usr/bin/env python3
"""Track O1: does the GSD's binning keep what the benchmark dust models predict? (A3c; user's request, 2026-11-23)

The models that fit the Milky Way's extinction and emission (THEMIS J13, Draine & Li 2007) are run in DustEM twice,
both times with tabulated size distributions at the same 100 sizes per grain type:
- with their own distributions;
- with the distributions projected onto the GSD's grids and rebuilt from the bins' two moments. The projection gives
  N and M per bin, and the default closure (EA) rebuilds the profile.
Where DustEM's analytic route differs from the tabulated one (DL07's PAH types, whose charge and ionisation
options it handles differently), that difference is listed apart ("DustEM route"); it is not a binning error.
Everything else (optics, heat capacities, the Mathis ISRF, the mass per H of each type) is DustEM's own, so the
difference is the binning error alone.

The grids, both in grain mass (a DustEM grain is placed by its mass, i.e. at the grid radius with the same mass at
the grid's density):
- carbon: ``CarbonLayout`` (4 molecular bins over N_C 20-400, 7 grain bins to 1 um) plus the 1-10 um overflow bin;
- silicate: 8 bins over 0.45 nm - 1 um plus the overflow bin.
Mass outside a grid (DL07's PAHs below N_C 20) is lost from the rebuilt run, as it would be in the GSD.

Variants: every DustEM type on its own grid; for J13, its two CM20 carbon populations summed on the one carbon
grid, as the GSD carries carbon; for DL07, a carbon grid down to N_C 10 (0.31 nm) with 5 molecular bins (the same
width), which holds its whole PAH distribution and so separates the binning from the grid's lower cut at N_C 20.

Reported: the relative change of the emission per H in photometric bands and of the extinction per H at UV to
near-IR wavelengths, the worst over the extinction curve 0.1-3 um, and the mass outside the grids.

    python diagnostics/gsd/check_o1_binning.py [--out results/gsd_a3c/o1_binning.txt]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import dustem_tools as D
import numpy as np

N_SIZES = 100
CARBON_TYPES = ("CM20", "Gra", "PAH0_DL07", "PAH1_DL07")
BANDS_UM = {"IRAC 3.6": 3.6, "IRAC 8": 7.9, "IRAS 12": 12.0, "MIPS 24": 24.0, "IRAS 60": 60.0, "IRAS 100": 100.0,
            "MIPS 160": 160.0, "Planck 350": 350.0, "Planck 850": 850.0}
EXT_UM = {"0.1": 0.1, "0.15": 0.15, "bump 0.2175": 0.2175, "0.3": 0.3, "B 0.44": 0.44, "V 0.55": 0.55,
          "J 1.25": 1.25, "K 2.2": 2.2}


def grids(nc_min=20.0, n_pah_bins=4):
    from pycalima.gsd.carbon import CarbonLayout
    from pycalima.gsd.grid import MassGrid
    from pycalima.gsd.materials import get_material

    c = CarbonLayout(nc_min=nc_min, n_pah_bins=n_pah_bins, overflow_a_max_cm=1e-3).grid()
    rho = get_material("silicate").rho
    m = 4.0 / 3.0 * np.pi * rho * np.array([4.5e-8, 1e-4]) ** 3
    s = MassGrid(np.geomspace(m[0], m[1], 9), rho).with_overflow(1e-3)
    return c, s


def with_nsize(line, n):
    f = line.split()
    f[1] = str(n)
    return "  ".join(f)


def project(grid, a, a4n):
    """N, M per bin of the distribution a^4 dn/da sampled at radii a (log-log interpolated, trapezoid in ln a on a
    fine grid), with the grid's own mass-radius relation; and the mass fraction outside the grid."""
    la = np.log(a)
    fine = np.exp(np.linspace(la[0], la[-1], 20001))
    f = np.exp(np.interp(np.log(fine), la, np.log(np.maximum(a4n, 1e-300))))
    dn_dlna = f / fine ** 3
    m = grid.mass(fine)
    lnf = np.log(fine)
    N, M = np.zeros(grid.n_bins), np.zeros(grid.n_bins)
    edges = grid.a_edges
    for i in range(grid.n_bins):
        sel = (fine >= edges[i]) & (fine <= edges[i + 1])
        if sel.sum() > 1:
            N[i] = np.trapezoid(dn_dlna[sel], lnf[sel])
            M[i] = np.trapezoid((dn_dlna * m)[sel], lnf[sel])
    total = np.trapezoid(dn_dlna * m, lnf)
    return N, M, 1.0 - M.sum() / total


def rebuild(grid, N, M, a):
    """a^4 dn/da at the radii a of the EA reconstruction of (N, M) (zero outside the grid; spikes dropped)."""
    from pycalima.gsd import closures as C

    P = C.pieces(grid, N, M, "ea")
    out = np.zeros(a.size)
    for i in range(grid.n_bins):
        for j in range(P.kp):
            if P.n[i, j] == 0.0 or P.spike[i, j]:
                continue
            u = a / P.aL[i]
            sel = (u >= P.u1[i, j]) & (u <= P.u2[i, j])
            out[sel] += P.A[i, j] * u[sel] ** P.lam[i, j] / P.aL[i]          # dn/da
    return out * a ** 4


def at(lam, y, x):
    return float(np.exp(np.interp(np.log(x), np.log(lam), np.log(np.maximum(y, 1e-300)))))


def compare(ref, test):
    row = {k: at(ref["lam"], test["sed"], v) / at(ref["lam"], ref["sed"], v) - 1.0 for k, v in BANDS_UM.items()}
    row.update({k: at(ref["ext_lam"], test["ext"], v) / at(ref["ext_lam"], ref["ext"], v) - 1.0
                for k, v in EXT_UM.items()})
    sel = (ref["ext_lam"] >= 0.1) & (ref["ext_lam"] <= 3.0)
    row["EXT worst 0.1-3"] = float(np.max(np.abs(test["ext"][sel] / ref["ext"][sel] - 1.0)))
    return row


def run_model(work, model, cgrid, sgrid, summed):
    """(binning error, route difference, mass outside the grids). The binning error compares the rebuilt
    distributions with the original ones, both given to DustEM as tabulated distributions (the same input route);
    the route difference compares that identity run with DustEM's analytic one (0 for J13; not 0 for DL07's PAH
    types, whose charge and ionisation options DustEM treats differently with tabulated sizes)."""
    kw, lines = D.grain_lines(model)
    kw = "sdist" if "sdist" not in kw.split() else kw
    lines = [with_nsize(s, N_SIZES) for s in lines]
    analytic = D.run(work, kw, lines)
    sd = analytic["sdist"]
    new_lines, ident_lines, sizes, ident, lost = [], [], {}, {}, {}
    groups = []
    if summed:
        carbon = [k for k, (t, *_) in enumerate(sd) if t in CARBON_TYPES]
        groups.append(carbon)
        groups += [[k] for k in range(len(sd)) if k not in carbon]
    else:
        groups = [[k] for k in range(len(sd))]
    for g in groups:
        name = sd[g[0]][0]
        a = sd[g[0]][1]
        a4n = sum(sd[k][2] for k in g)
        rho = sum(sd[k][2] * sd[k][3] for k in g) / np.maximum(a4n, 1e-300)
        grid = cgrid if name in CARBON_TYPES else sgrid
        rho_t = np.where(a4n > 0, rho, sd[g[0]][3])
        # the grids are mass grids: a DustEM grain of radius a and density rho_t sits at the grid radius
        # a_g = a (rho_t / rho_grid)^(1/3), the radius of the same mass at the grid's density (dn/dln a is unchanged)
        k = (rho_t / grid.rho) ** (1.0 / 3.0)
        a_g = a * k
        N, M, out_frac = project(grid, a_g, a4n * k ** 3)
        tag = f"{name}{'S' if len(g) > 1 else ''}{g[0]}"
        sizes[tag] = (a, np.maximum(rebuild(grid, N, M, a_g), 0.0) / k ** 3, rho_t)
        ident[tag] = (a, a4n, rho_t)
        f = lines[g[0]].split()
        mprop = sum(float(lines[k].split()[3]) for k in g)
        opts = f[2].split("-")
        opts = "-".join(["size"] + [o for o in opts if o.lower() not in ("logn", "plaw", "ed", "cv")])
        new_lines.append(f"{tag}  {N_SIZES}  {opts}  {mprop * (1.0 - out_frac):.4e}  {f[4]}  {a[0]:.4e}  {a[-1]:.4e}")
        ident_lines.append(f"{tag}  {N_SIZES}  {opts}  {mprop:.4e}  {f[4]}  {a[0]:.4e}  {a[-1]:.4e}")
        lost[tag] = out_frac
    _link_optics(work, {tag: sd[g[0]][0] for tag, g in zip(sizes, groups, strict=True)})
    ref = D.run(work, kw, ident_lines, ident)
    test = D.run(work, kw, new_lines, sizes)
    return compare(ref, test), compare(analytic, ref), lost


def _link_optics(work, alias):
    """DustEM reads Q_<type>, G_<type>, C_<type> (and charge/mix data) by type name: link the renamed types."""
    run = D.build(work)
    root = D.dustem_root()
    for tag, base in alias.items():
        for f in (root / "data").glob(f"*_{base}.DAT"):
            dst = run / "data" / f.name.replace(f"_{base}.DAT", f"_{tag}.DAT")
            if not dst.exists() and not f.name.startswith("SIZE_"):
                dst.symlink_to(f)
    for sub in ("oprop", "hcap"):
        d = run / sub
        if d.is_symlink():                     # replace the folder link by a folder of links, to add aliases
            d.unlink()
            d.mkdir()
            for f in (root / sub).iterdir():
                (d / f.name).symlink_to(f)
        for tag, base in alias.items():
            for f in (root / sub).glob(f"*_{base}.DAT"):
                dst = d / f.name.replace(f"_{base}.DAT", f"_{tag}.DAT")
                if not dst.exists():
                    dst.symlink_to(f)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--work", default="results/gsd_a3c/dustem")
    ap.add_argument("--out", default="results/gsd_a3c/o1_binning.txt")
    args = ap.parse_args(argv)
    cgrid, sgrid = grids()
    cgrid10, _ = grids(nc_min=10.0, n_pah_bins=5)
    cols = list(BANDS_UM) + list(EXT_UM) + ["EXT worst 0.1-3"]
    lines = ["Track O1: DustEM with the benchmark size distributions rebuilt from the GSD's bins (EA) against the",
             "continuous ones; relative change [%] of the emission per H in bands and of the extinction per H.",
             "Grids: carbon 11 bins (4 molecular) + overflow 1-10 um; silicate 8 bins + overflow.", ""]
    lines.append(f"{'model':18s} " + " ".join(f"{c:>11s}" for c in cols))
    for model, summed, cg, label in (("J13", False, cgrid, "J13"), ("J13", True, cgrid, "J13 (C summed)"),
                                     ("DL07", False, cgrid, "DL07"), ("DL07", False, cgrid10, "DL07 (N_C>=10, 5)")):
        row, route, lost = run_model(Path(args.work), model, cg, sgrid, summed)
        lines.append(f"{label:18s} " + " ".join(f"{100 * row[c]:11.2f}" for c in cols))
        if max(abs(v) for v in route.values()) > 1e-6:
            lines.append(f"{'  (DustEM route)':18s} " + " ".join(f"{100 * route[c]:11.2f}" for c in cols))
        lines.append("   mass outside the grids: " + ", ".join(f"{k} {100 * v:.2f} %" for k, v in lost.items()))
        print("\n".join(lines[-3:]), flush=True)
    txt = "\n".join(lines)
    print(txt)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(txt + "\n")


if __name__ == "__main__":
    main()
