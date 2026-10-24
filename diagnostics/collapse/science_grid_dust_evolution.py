"""How much does dust evolution beyond accretion move the critical metallicity?

[Z_cr] of the N07 Pop III CCSN models for a ladder of processes, by two criteria
(``critical_dust_to_gas``): ``C`` (gas-grain cooling time vs collapse time,
n_H in [1e8, H2 90 %], on the prescribed track) and ``frag`` (Schneider & Omukai
2010 fragmentation of the energy-equation solution).

Initial conditions and tracks:
* ``n0``: unprocessed dust (Nozawa+03 C20 sizes) on the Omukai+05 Z = 0 track;
* ``n1``: reverse shock into n_amb = 1 cm^-3 with Grackle's grains on Grackle's
  dust-free track at each trial Z (``Track.grackle``).

Process ladder: static; accretion; accretion + turbulent coagulation (full
Ormel & Cuzzi + Brownian, perfect sticking, no Coulomb barrier: an upper bound
on the coagulation itself) at Mach 1, 3, 10; and Mach 10 with the kernel x10, an
upper bound on the cross-species collisions the model leaves out (each species
coagulates only with itself).

    python diagnostics/collapse/science_grid_dust_evolution.py [--workers 8] [--masses 13 20 25 30]

Writes ``collapse/science_grid_dust_evolution.txt`` under the results directory;
each finished search is appended to ``science_grid_dust_evolution.partial`` there
and skipped when rerun. About 1 h on 8 cores.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from pycalima import _paths
from pycalima.collapse.ics import nozawa_ccsn
from pycalima.collapse.integrator import critical_dust_to_gas, turbulent_kernel
from pycalima.collapse.tracks import Track

LADDER = {
    "static": dict(),
    "acc": dict(accretion=True),
    "acc+coag M1": dict(accretion=True, mach=1.0),
    "acc+coag M3": dict(accretion=True, mach=3.0),
    "acc+coag M10": dict(accretion=True, mach=10.0),
    "acc+coag M10 Kx10": dict(accretion=True, mach=10.0, boost=10.0),
}


def _boosted(factory, boost):
    def f(*args):
        k = factory(*args)
        return lambda m, a: boost * k(m, a)
    return f


def one(job):
    M, ic, process, criterion = job
    p = LADDER[process]
    if ic == "n0":
        make = lambda Z: nozawa_ccsn(M, Z)[:2]
        track = Track.omukai2005("Z0")
    else:
        make = lambda Z: nozawa_ccsn(M, Z, n_amb=1, grains="grackle")[:2]
        track = lambda Z: Track.grackle(M, Z)
    kw = dict(accretion=p.get("accretion", False), dlnn=0.1, n0=1e-1)
    if "mach" in p:
        k = turbulent_kernel(mach=p["mach"])
        kw.update(coagulation=True, coag_kernel=_boosted(k, p["boost"]) if "boost" in p else k)
    t0 = time.time()
    if criterion == "frag":
        Z = critical_dust_to_gas(make, track, criterion="frag", lo=1e-6, hi=10 ** -2.5, scan_dex=0.25,
                                 rtol=0.05, **kw)
    else:
        Z = critical_dust_to_gas(make, track, criterion="C", lo=1e-8, hi=1e-2, rtol=0.05, **kw)
    return job, math.log10(Z), time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--masses", type=int, nargs="+", default=[13, 20, 25, 30])
    a = ap.parse_args()
    jobs = [(M, ic, pr, cr) for cr in ("C", "frag") for M in a.masses for ic in ("n0", "n1") for pr in LADDER]
    # the slow ones (frag with coagulation) first
    jobs.sort(key=lambda j: (j[3] != "frag", "coag" not in j[2]))
    out = _paths.get_results_dir() / "collapse"
    out.mkdir(parents=True, exist_ok=True)
    partial = out / "science_grid_dust_evolution.partial"
    res = {}
    if partial.exists():
        for ln in partial.read_text().splitlines():
            j, z, dt = json.loads(ln)
            res[tuple(j)] = (z, dt)
    todo = [j for j in jobs if j not in res]
    with ProcessPoolExecutor(a.workers) as ex, open(partial, "a") as fh:
        for fut in as_completed([ex.submit(one, j) for j in todo]):
            job, z, dt = fut.result()
            res[job] = (z, dt)
            fh.write(json.dumps([list(job), z, dt]) + "\n")
            fh.flush()
    lines = ["# [Z_cr] (Z_sun = 0.02) by process, diagnostics/collapse/science_grid_dust_evolution.py",
             "# columns: criterion, progenitor mass, ICs, then [Z_cr] for " + ", ".join(LADDER)]
    for cr in ("C", "frag"):
        print(f"\ncriterion {cr}: [Z_cr], and in brackets the shift from accretion alone")
        print("   M  ICs | " + " | ".join(f"{p:>17s}" for p in LADDER))
        for M in a.masses:
            for ic in ("n0", "n1"):
                z = [res[(M, ic, p, cr)][0] for p in LADDER]
                acc = z[1]
                print(f"  {M:2d}  {ic}  | " + " | ".join(
                    f"{v:6.2f} ({v - acc:+5.2f})   " if i > 1 else f"{v:6.2f}           " for i, v in enumerate(z)))
                lines.append(f"{cr} {M} {ic} " + " ".join(f"{v:.3f}" for v in z))
    (out / "science_grid_dust_evolution.txt").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {out / 'science_grid_dust_evolution.txt'}; slowest search {max(d for _, d in res.values()):.0f} s")


if __name__ == "__main__":
    main()
