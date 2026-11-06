#!/usr/bin/env python3
"""Run a list of Octopus jobs (``make_run.py`` run directories) within a budget of MPI ranks, niced.

Each line of the job file is ``<NP> <command...>``. A job takes 3 x NP ranks (three kicks at once). Jobs start
in file order while the budget allows. A job whose directory's ``run.log`` already ends in RUN_OK is skipped, so the
queue can be stopped and started again. Stdlib only (megatron's python3 has no pip).

    nohup queue.py jobs.txt --max-ranks 96 > queue.log 2>&1 &
"""
import argparse
import shlex
import subprocess
import time
from pathlib import Path


def done(run_dir):
    log = Path(run_dir) / "run.log"
    return log.exists() and log.read_text().rstrip().endswith("RUN_OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("jobs", type=Path)
    ap.add_argument("--max-ranks", type=int, default=96, help="MPI ranks in use at once (megatron: 128 cores)")
    ap.add_argument("--poll", type=float, default=60.0)
    a = ap.parse_args()
    pending = []
    for line in a.jobs.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            np_, *cmd = shlex.split(line)
            # "<dir>/run.sh" runs in its own directory; "resume_run.sh <dir>" names it
            run_dir = Path(cmd[-1] if len(cmd) > 1 else cmd[0]).expanduser()
            run_dir = run_dir if len(cmd) > 1 else run_dir.parent
            pending.append((int(np_), cmd, run_dir))
    running = []
    while pending or running:
        running = [(p, n, d) for p, n, d in running if p.poll() is None]
        used = sum(n for _, n, _ in running)
        while pending and used + 3 * pending[0][0] <= a.max_ranks:
            np_, cmd, run_dir = pending.pop(0)
            if done(run_dir):
                print(f"{time.strftime('%F %T')} skip {run_dir} (RUN_OK)", flush=True)
                continue
            env_cmd = ["env", f"NP={np_}"] + [str(Path(c).expanduser()) for c in cmd]
            log = open(run_dir / "run.log", "a")
            running.append((subprocess.Popen(env_cmd, stdout=log, stderr=subprocess.STDOUT), 3 * np_, run_dir))
            used += 3 * np_
            print(f"{time.strftime('%F %T')} start {run_dir} ({3 * np_} ranks; {used} in use)", flush=True)
        time.sleep(a.poll)
    print(f"{time.strftime('%F %T')} queue empty", flush=True)


if __name__ == "__main__":
    main()
