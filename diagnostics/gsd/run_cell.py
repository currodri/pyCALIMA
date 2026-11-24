#!/usr/bin/env python3
"""One-zone driver of the per-cell GSD operator (``pycalima.gsd.cell``, A3c).

``oracle``: the five A0 cases of ``check_bin_convergence.py`` (EA, 8 bins per composition), run step by step
with A0's own split step and with ``cell.update`` given the same models (A0's materials, pair tables, accretion
sticking and constant-rate sputtering; charging, sinks and the overflow bin off). The two must agree to
round-off: the operator adds nothing of its own. Reported: the largest difference of any bin's N and M relative to
the composition's total (empty bins make relative differences per bin meaningless), the relative difference of the
gas at the end, and the operator's element conservation.

    python diagnostics/gsd/run_cell.py oracle [--steps 25] [--out results/gsd_a3c/run_cell_oracle.txt]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import check_bin_convergence as A0
import numpy as np

YR, PC = A0.YR, A0.PC          # A0's own constants, so that both runs see the same L and dt


def oracle_case(case, n_bins=8, scheme="ea", steps=25):
    from pycalima.gsd import cell

    cs = A0.CASES[case]
    ph, processes = A0.PHASES[cs["phase"]], cs["processes"]
    nH = ph["n_H"]
    a0, ours, gas = [], [], {}
    for mat in A0.materials(3.5e-8, 4.5e-8):
        g = A0.make_grid(mat, n_bins)
        N, M = A0.initial_bins(g, mat, cs["ic"])
        a0.append(A0.Composition(mat, g, scheme, M.copy(), N.copy()))
        gas[mat.key] = A0.SOLAR[mat.key] * (1.0 - A0.F_DEP0)
        erosion = (lambda T, n, m=mat, r=g.rho: A0.erosion(m, cs["phase"], r)) if "sputtering" in processes else None
        ours.append(cell.Composition(g, N * nH, M * nH, mat.key, mat.per_gram, mat.shatter, method=A0.method_of(scheme),
                                     sputtering=erosion, sublimation=mat.sublimation if "accretion" in processes
                                     else None, a_frag_min=mat.a_min))
    opt = cell.Options(processes=frozenset(processes) & cell.PROCESSES, errmax=A0.ERRMAX, n_speed=A0.N_SPEED,
                       table_ppd=A0.TABLE_PPD, threads=1)
    gas_ours = {k: v * nH for k, v in gas.items()}
    dt = cs["myr"] * 1e6 * YR / steps
    stats = {"coll_substeps": 0, "t_coll": 0.0, "t_acc": 0.0, "t_sput": 0.0}
    cons = 0.0
    for _ in range(steps):
        A0.step(a0, gas, cs["phase"], dt, processes, 1, stats)
        inp = cell.CellInputs(T=ph["T"], n_H=nH, n_e=0.0, gas=gas_ours, T_d=A0.T_DUST, v_turb=ph["V"] * 1e5,
                              L=ph["L"] * PC)
        res = cell.update(ours, inp, None, dt, opt)
        gas_ours = res.gas
        cons = max(cons, max(abs(v) for v in res.conservation.values()))
    dev = 0.0
    for c0, c1 in zip(a0, ours, strict=True):
        for x0, x1 in ((c0.N * nH, c1.N), (c0.M * nH, c1.M)):
            dev = max(dev, float(np.max(np.abs(x1 - x0)) / np.sum(np.abs(x0))))
    dgas = max(abs(gas_ours[k] / (gas[k] * nH) - 1.0) for k in gas)
    return dev, dgas, cons


def cmd_oracle(args):
    lines = ["A0 oracle: cell.update against check_bin_convergence.step (EA, 8 bins, same models), "
             f"{args.steps} steps per case.", "case            max |dN|/N_tot, |dM|/M_tot   gas rel    conservation"]
    for case in A0.CASES:
        dev, dgas, cons = oracle_case(case, steps=args.steps)
        lines.append(f"{case:18s} {dev:16.2e} {dgas:10.2e} {cons:12.2e}")
        print(lines[-1], flush=True)
    txt = "\n".join(lines)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(txt + "\n")
    print(txt)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("oracle")
    o.add_argument("--steps", type=int, default=25)
    o.add_argument("--out", default="results/gsd_a3c/run_cell_oracle.txt")
    args = ap.parse_args(argv)
    {"oracle": cmd_oracle}[args.cmd](args)


if __name__ == "__main__":
    main()
