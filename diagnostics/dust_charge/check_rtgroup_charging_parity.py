#!/usr/bin/env python3
"""GroupChargingTable.solve against the RAMSES grain-charge bench, state by state (the full parity check).

RAMSES's bench driver lgx_bench (RAMSES_dev studies/charge_lg5_20261118/scripts/lgx_bench.f90) solves every bin of the
22-bin charging bench cold, as RAMSES's dust_interface would: the cell states of a RAMSES dump (wdb_dump_<rank>.bin,
68 doubles per record, the records with columns 1, 2 = 0, 1, sub-sampled evenly to --ncell), the guess the previous
bin's (same material) Z* scaled by the size ratio, dlnT 1e-5, alpha wanted (the Gaussian estimate) and the Coulomb
factors, in mode 0 (WDB06rt) or 1 (LG5, rtg_lattice), rtg_wide_moments on or off. Its output per cell and bin (32
doubles): Zmean, Zsigma, discrete, Gamma, Lambda + Lambda_auto, D(-1, 1, 2, 3), alpha (10 ions), alpha after
rtgroups_refine_alpha (wide P(Z)), Zstar, flags (1: D the solve's own, 2: alpha from the tail; LG5 flags D only),
time [ns].

This script makes the same solves with pyCALIMA and reports per column the largest |rel| difference and the states
whose branch flags differ. The criterion is round-off (1e-10); a charge near zero is compared to 1e-10 of
max(|Z|, 1e-3), and an alpha below 1e-100 by its ln (the far tail of the Gaussian estimate).

    python check_rtgroup_charging_parity.py (--dumps DUMP_DIR | --cells CELLS.npz) --tables TABLES_DIR \\
        --fortran OUT_DIR [--lattice] [--no-wide-moments] [--cases b1e2,pdr,t3ds:00009] [--ncell 3000] [--workers 6]

DUMP_DIR/<case>/wdb_dump_<rank>.bin (rank 00001 unless given as case:rank), or CELLS.npz with the selected records
already extracted (<case>[_<rank>]_C22, as in RAMSES_dev studies/reduced_charge_model_20261117/data); TABLES_DIR holds bins22.txt and the bins'
tables (dust_charging_rtgroups_<label>.dat); OUT_DIR/<case>[_<rank>].out are lgx_bench's outputs. The tests' fixture
(tests/data/rtgroup_parity_lg5.npz) is a few hundred of these states.
"""
from __future__ import annotations

import argparse
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

CASES = ("b1e2", "b1e3", "c3e3", "b1e2ns", "b1e4", "c3e4", "b1e5", "c1e5", "c1e6", "c3e4ns", "pdr", "t3ds:00001",
         "t3ds:00009")
NR, NOUT = 68, 32
M_HION, M_HEION = 1.67262192e-24, 6.6446573e-24
DLNT = 1e-5
COLS = (["Zmean", "Zsigma", "discrete", "Gamma", "Lambda", "D-1", "D+1", "D+2", "D+3"]
        + [f"alpha{j}" for j in range(10)] + [f"alpha_fit{j}" for j in range(10)] + ["Zstar", "flags"])


def case_files(dumps, fortran, case):
    c, rk = (case.split(":") + ["00001"])[:2]
    tag = c if ":" not in case else f"{c}_{rk}"
    return Path(dumps) / c / f"wdb_dump_{rk}.bin", Path(fortran) / f"{tag}.out", tag


def cells(path, ncell):
    """The cell records lgx_bench takes from a dump: columns 1, 2 = 0, 1, sub-sampled evenly to ncell."""
    R = np.fromfile(path, "<f8")
    R = R[: R.size // NR * NR].reshape(-1, NR)
    R = R[(R[:, 0] == 0.0) & (np.rint(R[:, 1]) == 1)]
    m = R.shape[0]
    n = min(ncell, m)
    take, iprev = [], 0
    for rep in range(1, m + 1):
        if ((rep - 1) * n) // m >= iprev and iprev < n:
            take.append(rep - 1)
            iprev += 1
    return R[take]


def bins(tables):
    out = []
    for line in (Path(tables) / "bins22.txt").read_text().split("\n"):
        if line.strip():
            lab, mat, a, *_ = line.split()
            out.append((lab, int(mat), float(a)))
    return out


def solve_case(args):
    from pycalima.models.dust_charge import rt_group_charging as rg

    src, tables, ncell, opts = args
    C = np.load(src[0])[src[1]][:ncell] if isinstance(src, tuple) else cells(src, ncell)
    bl = bins(tables)
    tabs = {lab: rg.read_bin_table(Path(tables) / f"dust_charging_rtgroups_{lab}.dat") for lab, _, _ in bl}
    gts = {}
    O = np.zeros((C.shape[0], len(bl), NOUT))
    for k, c in enumerate(C):
        Np, egy, c_red, T, ne = c[12:20], c[20:28], c[9], c[4], c[5]
        ions = [(c[6], 1.0, M_HION), (c[7], 1.0, M_HEION), (c[8], 2.0, M_HEION)]
        for ii, (lab, mat, a) in enumerate(bl):
            key = (lab, tuple(egy))
            if key not in gts:
                gts[key] = rg.GroupChargingTable(tabs[lab], egy=egy)
            gt = gts[key]
            zg = O[k, ii - 1, 29] * a / bl[ii - 1][2] if ii > 0 and bl[ii - 1][1] == mat else 0.0
            t0 = time.perf_counter_ns()
            r = gt.solve(Np, c_red, T, ne, ions, G0_bg=c[10], Z_guess=zg, recomb="gauss", dlnT=DLNT, coulomb=True,
                         **opts)
            t1 = time.perf_counter_ns()
            fl = 1.0 if r["D_abs"] else 0.0
            if not opts.get("lattice", False) and r["alpha_tail"]:
                fl += 2.0
            al = np.asarray(r["alpha_rec"], float)
            af = al.copy()
            if not r["discrete"]:
                rf = dict(r)
                gt.refine_recombination(rf, None)
                af = np.asarray(rf["alpha_rec"], float)
            O[k, ii] = np.concatenate([[r["Zmean"], r["Zsigma"], float(r["discrete"]), r["Gamma"],
                                        r["Lambda"] + r["Lambda_auto"]], r["D"], al, af, [r["Zstar"], fl, t1 - t0]])
    return O


def errors(P, F):
    """{column: (max error, states above 1e-10)} in the criterion's forms."""
    out = {}
    for j, name in enumerate(COLS):
        p, f = P[..., j], F[..., j]
        d = np.abs(p - f)
        if name in ("discrete", "flags"):
            e = (d > 0.5).astype(float)
        elif name in ("Zmean", "Zstar"):
            e = d / np.maximum(np.abs(f), 1e-3)
        elif name.startswith("alpha"):
            lnf = np.log(np.maximum(f, 1e-320))
            e = np.where(np.abs(f) < 1e-100, np.abs(np.log(np.maximum(p, 1e-320)) - lnf) / np.abs(lnf),
                         d / np.maximum(np.abs(f), 1e-300))
        else:
            e = d / np.maximum(np.abs(f), 1e-300)
        e = np.where(d == 0, 0.0, e)
        out[name] = (float(e.max()), int(np.sum(e > 1e-10)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--dumps", help="directory of <case>/wdb_dump_<rank>.bin")
    src.add_argument("--cells", help="npz of the selected records, <case>[_<rank>]_C22")
    ap.add_argument("--tables", required=True, help="directory of bins22.txt and the bins' tables")
    ap.add_argument("--fortran", required=True, help="directory of lgx_bench's <case>.out")
    ap.add_argument("--lattice", action="store_true", help="LG5 (lgx_bench mode 1)")
    ap.add_argument("--no-wide-moments", action="store_true")
    ap.add_argument("--cases", default=",".join(CASES))
    ap.add_argument("--ncell", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    args = ap.parse_args(argv)
    opts = dict(lattice=args.lattice, wide_moments=not args.no_wide_moments)
    cases = args.cases.split(",")
    files = [case_files(args.dumps or ".", args.fortran, c) for c in cases]
    srcs = [d if args.dumps else (args.cells, f"{tag}_C22") for d, _, tag in files]
    t0 = time.time()
    with Pool(min(args.workers, len(cases))) as pool:
        out = pool.map(solve_case, [(s_, args.tables, args.ncell, opts) for s_ in srcs])
    print(f"pyCALIMA against {args.fortran} ({opts}); {time.time() - t0:.0f} s")
    print(f"{'column':12s} {'max error':>10s} {'n > 1e-10':>10s}")
    tot = {}
    tim = []
    for (_, fout, tag), P in zip(files, out):
        F = np.fromfile(fout, "<f8").reshape(-1, P.shape[1], NOUT)[: P.shape[0]]
        for name, (e, n) in errors(P, F).items():
            m, c = tot.get(name, (0.0, 0))
            tot[name] = (max(m, e), c + n)
        tim.append(np.stack([P[..., 31].sum(1), F[..., 31].sum(1)], axis=1) * 1e-3)
    for name in COLS:
        print(f"{name:12s} {tot[name][0]:10.1e} {tot[name][1]:10d}")
    print("median cost per cell [us]: pyCALIMA %.0f, Fortran %.1f" % tuple(np.median(np.concatenate(tim), axis=0)))


if __name__ == "__main__":
    main()
