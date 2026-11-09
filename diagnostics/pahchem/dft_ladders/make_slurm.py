#!/usr/bin/env python3
"""Slurm jobs for the Delta-SCF ladders on DIaL3: ``jobs.txt`` (``make_ladder.py``) split into one-node jobs.

Each node job runs ``run_queue.sh`` on its share of the ladder, NPAR states at a time with THREADS threads each (NPAR x
THREADS = the node's cores). The parents are kept apart, their sizes differing ~10x in cost: by default C24 and C54 run
8 x 16 threads per node, C96 4 x 32 (with ~1/4 of the node's memory each, its density-fitting tensors fit in core).

Usage: make_slurm.py ROOT [--account dp016] [--partition P] [--time HH:MM:SS] [--per-node C24=44 C54=22 C96=4]
       [--layout C24=8x16 C54=8x16 C96=4x32] [--memory-mb C24=12000 C54=40000 C96=110000]
       [--maxsteps C24=150 C54=100 C96=60] [--no-diffuse C96] [--python PATH] [--setup CMD]
Writes ROOT/slurm/<parent>_<k>.jobs and .sbatch, and ROOT/slurm/submit.sh (sbatch of each).
"""
import argparse
from pathlib import Path


def kv(items, cast):
    return {k: cast(v) for k, v in (s.split("=") for s in items)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root", type=Path)
    ap.add_argument("--account", default="dp016")
    ap.add_argument("--partition", default=None)
    ap.add_argument("--time", default="36:00:00")
    ap.add_argument("--per-node", nargs="+", default=["C24=44", "C54=22", "C96=4"])
    ap.add_argument("--layout", nargs="+", default=["C24=8x16", "C54=8x16", "C96=4x32"])
    ap.add_argument("--memory-mb", nargs="+", default=["C24=12000", "C54=40000", "C96=110000"])
    ap.add_argument("--python", default="$HOME/opt/qc-venv/bin/python")
    ap.add_argument("--setup", default="", help="shell lines run first in each job (module loads)")
    ap.add_argument("--parents", nargs="+", default=None, help="only these parents")
    ap.add_argument("--maxsteps", nargs="+", default=["C24=150", "C54=100", "C96=60"])
    ap.add_argument("--no-diffuse", nargs="*", default=["C96"],
                    help="parents without the 6-31+G* single points (C96: ~1870 functions, the costliest SCF, and the least "
                         "basis-sensitive EA)")
    args = ap.parse_args()
    per, lay, mem = kv(args.per_node, int), kv(args.layout, str), kv(args.memory_mb, int)
    steps = kv(args.maxsteps, int)
    jobs = [l.split() for l in (args.root / "jobs.txt").read_text().splitlines() if l.strip()]
    out = args.root / "slurm"
    out.mkdir(exist_ok=True)
    submit = ["#!/usr/bin/env bash", "set -e", f"cd {out}"]
    for parent in per:
        if args.parents and parent not in args.parents:
            continue
        mine = [j for j in jobs if j[0].split("/")[0] == parent]
        if parent in args.no_diffuse:
            mine = [j[:3] + ["0"] for j in mine]
        npar, threads = (int(x) for x in lay[parent].split("x"))
        for k in range(0, len(mine), per[parent]):
            name = f"{parent}_{k // per[parent]:02d}"
            (out / f"{name}.jobs").write_text("".join(" ".join(j) + "\n" for j in mine[k:k + per[parent]]))
            lines = ["#!/bin/bash", f"#SBATCH --job-name=ladder_{name}", f"#SBATCH --account={args.account}",
                     "#SBATCH --nodes=1", "#SBATCH --ntasks=1", f"#SBATCH --cpus-per-task={npar * threads}",
                     f"#SBATCH --time={args.time}", f"#SBATCH --output={out}/{name}_%j.out"]
            if args.partition:
                lines.append(f"#SBATCH --partition={args.partition}")
            lines += [args.setup, f"export PY={args.python}", f"export LADDER_MAX_MEMORY={mem[parent]}",
                      f"export LADDER_MAXSTEPS={steps[parent]}",
                      f"bash {args.root}/scripts/run_queue.sh {args.root} {out}/{name}.jobs {npar} {threads}"]
            (out / f"{name}.sbatch").write_text("\n".join(l for l in lines if l) + "\n")
            submit.append(f"sbatch {name}.sbatch")
    (out / "submit.sh").write_text("\n".join(submit) + "\n")
    print(f"{len(submit) - 3} node jobs in {out}")


if __name__ == "__main__":
    main()
