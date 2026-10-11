"""How much does the initial grain size distribution move the critical metallicity?

[Z_cr] of the N07 Pop III CCSN models (reverse shock into n_amb = 1 cm^-3) when
only the shape of each species' size distribution changes: the species, their
masses (Grackle's dust-to-metal fractions), the gas-phase metals (Chiaki+15) and
the gas track (Grackle's dust-free track at each trial Z, ``Track.grackle``) are
those of the ``n1`` ICs of ``science_grid_dust_evolution.py``. Static dust and
accretion, by criteria ``C`` and ``frag`` (``critical_dust_to_gas``).

Shapes, the same for every species:
* ``grackle``: Grackle's lognormals (the baseline);
* ``aS=<x>``: lognormal of width sigma_ln = 0.7 (about Grackle's Si) with
  a_S = <r^3> / <r^2> = x um, so that S = 3 / (4 rho_s a_S);
* ``sig=<s>``: lognormal of width s at a_S = 0.1 um (fixed S per species);
* ``MRN``: dn/da ∝ a^-3.5 from 5 to 250 nm (a_S = 35 nm);
* ``N03``: unprocessed (Nozawa+03 C20, their Fig. 8a);
* ``N07``: after the reverse shock (Nozawa+07 Fig. 6b, the 20 Msun model, used
  for every progenitor);
* ``eroded``: Nozawa+03 thinned uniformly to the Nozawa+07 surviving mass
  (``ics.erode``, as ``nozawa_ccsn(n_amb=1)``).
A species the table has no curve for keeps Grackle's lognormal.

    python diagnostics/collapse/science_grid_size_distribution.py [--workers 8] [--masses 13 20 25 30]

Writes ``collapse/science_grid_size_distribution.txt`` under the results
directory, checkpointing to ``science_grid_size_distribution.partial``. About
10 min on 8 cores.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from pycalima import _paths
from pycalima.collapse.ics import (distribution_on_grid, lognormal_from_moments, nozawa2003_sizes,
                                   nozawa2007_m20_n1_sizes, nozawa_ccsn)
from pycalima.collapse.integrator import Population, _state, critical_dust_to_gas
from pycalima.collapse.tracks import Track

UM = 1e-4
SHAPES = ["grackle"] + [f"aS={x:g}" for x in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)] \
    + [f"sig={s:g}" for s in (0.1, 0.4, 1.0, 1.4)] + ["MRN", "N03", "N07", "eroded"]
PROCESSES = ("static", "acc")


def _lognormal(a_S, sigma):
    mu = math.log(a_S) - 2.5 * sigma ** 2
    return lambda p: lognormal_from_moments(p.grid, math.exp(2 * mu + 2 * sigma ** 2),
                                            math.exp(3 * mu + 4.5 * sigma ** 2))


def _mrn(a_lo=5e-7, a_hi=2.5e-5):
    def w(p):
        ae = np.clip(p.grid.a_edges, a_lo, a_hi)
        return np.diff(np.sqrt(ae)) / (math.sqrt(a_hi) - math.sqrt(a_lo))     # ∫ a^3 a^-3.5 da
    return w


def _table(sizes):
    return lambda p: distribution_on_grid(p.grid, p.material, sizes)


def _eroded(M):
    pops = {p.material: p for p in nozawa_ccsn(M, 1e-4, n_amb=1)[0]}
    return lambda p: pops[p.material].x / pops[p.material].x.sum() if p.material in pops else np.zeros_like(p.x)


def shape(name, M):
    """Mass fraction per bin, as a function of a Population, for shape ``name``."""
    if name.startswith("aS="):
        return _lognormal(float(name[3:]) * UM, 0.7)
    if name.startswith("sig="):
        return _lognormal(0.1 * UM, float(name[4:]))
    return {"MRN": _mrn, "N03": lambda: _table(nozawa2003_sizes), "N07": lambda: _table(nozawa2007_m20_n1_sizes),
            "eroded": lambda: _eroded(M)}[name]()


def populations(M, Z, name):
    """(populations, y_gas) of the n1 ICs with every species reshaped to ``name``."""
    pops, y, _ = nozawa_ccsn(M, Z, n_amb=1, grains="grackle")
    if name == "grackle":
        return pops, y
    w = shape(name, M)
    out = []
    for p in pops:
        f = w(p)
        out.append(Population(p.material, p.grid, p.x.sum() * f / f.sum() if f.sum() > 0 else p.x, p.table))
    return out, y


def cross_section(pops, exclude=()):
    """S [cm^2/g]: geometric cross-section per gram of the dust (less the ``exclude`` species)."""
    keep = [(p, s) for p, s in zip(pops, _state(pops, 1.0)) if p.material not in exclude]
    return sum(float(np.sum(n * A)) for _, (n, _, A) in keep) / sum(float(p.x.sum()) for p, _ in keep)


def one(job):
    M, name, process, criterion = job
    make = lambda Z: populations(M, Z, name)
    track = lambda Z: Track.grackle(M, Z)
    kw = dict(accretion=process == "acc", dlnn=0.1, n0=1e-1)
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
    jobs = [(M, s, pr, cr) for cr in ("frag", "C") for M in a.masses for s in SHAPES for pr in PROCESSES]
    out = _paths.get_results_dir() / "collapse"
    out.mkdir(parents=True, exist_ok=True)
    partial = out / "science_grid_size_distribution.partial"
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
    lines = ["# [Z_cr] (Z_sun = 0.02) by initial size distribution, diagnostics/collapse/"
             "science_grid_size_distribution.py",
             "# columns: criterion, progenitor mass, shape, S0 [cm^2/g], S0 without Si [cm^2/g], "
             "[Z_cr] static, [Z_cr] accretion"]
    for cr in ("C", "frag"):
        print(f"\ncriterion {cr}: [Z_cr] static / accretion (S0 in 1e4 cm^2/g: all dust, without Si)")
        for M in a.masses:
            print(f"  M = {M}")
            for s in SHAPES:
                pops, _ = populations(M, 1e-4, s)
                S0, S0n = cross_section(pops), cross_section(pops, exclude=("Si",))
                zs, za = (res[(M, s, pr, cr)][0] for pr in PROCESSES)
                print(f"    {s:>9s}  S0 = {S0 / 1e4:7.3f} {S0n / 1e4:7.3f}   {zs:6.2f}  {za:6.2f}  ({za - zs:+5.2f})")
                lines.append(f"{cr} {M} {s} {S0:.4e} {S0n:.4e} {zs:.3f} {za:.3f}")
    (out / "science_grid_size_distribution.txt").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {out / 'science_grid_size_distribution.txt'}; slowest search {max(d for _, d in res.values()):.0f} s")


if __name__ == "__main__":
    main()
