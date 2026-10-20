"""Is the Pop III / Pop II transition one metallicity? How far seven inputs move it.

[Z_cr] of the N07 Pop III CCSN models by criteria ``frag`` and ``C``
(``critical_dust_to_gas``), starting from the classical assumptions of the published
dust-cooling thresholds (Milky Way dust per metal, MRN sizes, static dust, no cosmic
rays, z = 0 CMB, no Lyman-Werner background, Omukai+05 collapse, perfect thermal
accommodation; ``check_literature_zcr.py``), for uncertain inputs (``FACTORS``; the
redshift and collapse-time options are kept but fixed at their classical values):

* ``d``, dust surviving the reverse shock, as dust per metal f_dep: ``MW`` (0.5,
  the classical value), or Chiaki+15 Tables 3-4 for this progenitor with no reverse
  shock (``n0``) and ambient densities 0.1, 1, 10 cm^-3 (``n0.1``, ``n1``, ``n10``);
* ``M``, SN progenitor, i.e. dust composition: 13, 20, 25, 30 Msun (Grackle's species);
* ``s``, initial size distribution: MRN, Grackle's lognormals, Nozawa+03, Nozawa+07,
  eroded Nozawa+03 (``science_grid_size_distribution.py``);
* ``p``, accretion: off or on (coagulation and shattering move [Z_cr] by <= 0.02 dex,
  ``science_grid_dust_evolution.py``, so they are left out);
* ``z``, cosmic rays: zeta0 = 0, 1e-18, 1e-16 s^-1;
* ``zr``, redshift of the CMB: 0, 10, 20 (the gas track's CMB floor and Compton
  term, and the dust's radiation field);
* ``lw``, Lyman-Werner background: J_21 = 0, 10, 100 (H2 self-shielded over the
  Jeans length; at 100 the gas stays atomic);
* ``f``, collapse time x 1, 2, 3 (support by turbulence, rotation or magnetic
  fields), in the gas track and in the energy equation alike;
* ``a``, thermal accommodation coefficient of gas on grains: 1, 0.5, 0.3 (Burke &
  Hollenbach 1983 find ~0.3-0.5 for H on grains at T ~ 1000 K).

Cosmic rays, redshift, LW and the collapse time enter Grackle's dust-free gas track
(``grackle_environments.py``, built on demand). Each species keeps its share of the
dust; the gas-phase metals are what the dust leaves of Chiaki+15's abundances
(carbon in excess of oxygen only, as ``nozawa_ccsn``).

Not the full grid (10^4 combinations per criterion) but what bounds it: ``row``,
each input alone departing from the classical case (accretion from the dust after a
reverse shock into 1 cm^-3, as the Milky Way dust leaves no metals to accrete);
then ``min`` and ``max``, a steepest descent (ascent) from the combination of each
input's lowest (highest) choice in the rows, over single changes of one input to its
second-ranked choice in the rows, until no such change improves it (each move that
needs a new gas track costs 21 Grackle runs, so the other choices, which the rows
rank as moving [Z_cr] the wrong way, are not tried).

    python diagnostics/collapse/science_grid_transition.py --grackle-src G --grackle-python P [--workers 8]

Writes ``collapse/science_grid_transition.txt`` under the results directory,
checkpointing to ``science_grid_transition.partial``. About 1-2 h on 8 cores,
mostly Grackle tracks.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

import numpy as np

from pycalima import _paths
from pycalima.collapse.accretion import GAS_ELEMENTS, formula_atoms, formula_mass
from pycalima.collapse.ics import nozawa_ccsn
from pycalima.collapse.integrator import Population, critical_dust_to_gas
from pycalima.collapse.materials import get_material

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grackle_environments as genv                 # noqa: E402
import science_grid_size_distribution as sizes      # noqa: E402

MASSES = (13, 20, 25, 30)
DUST = ("MW", "n0", "n0.1", "n1", "n10")
SHAPES = ("MRN", "grackle", "N03", "N07", "eroded")
PROCESSES = ("static", "acc")
ZETAS = (0.0, 1e-18, 1e-16)
REDSHIFTS = (0.0, 10.0, 20.0)
J21S = (0.0, 10.0, 100.0)
DELAYS = (1.0, 2.0, 3.0)
ALPHAS = (1.0, 0.5, 0.3)
# the inputs varied; redshift and collapse time stay at their classical values (z = 0, free fall) unless
# ("zr", REDSHIFTS) and ("f", DELAYS) are added: with them the lowest [Z_cr] by frag goes from -6.30 to -7.13
# (a 2x slower collapse), and z = 10-20 does not restore fragmentation under zeta0 = 1e-16
FACTORS = (("d", DUST), ("M", MASSES), ("s", SHAPES), ("p", PROCESSES), ("a", ALPHAS), ("z", ZETAS), ("lw", J21S))
KEYS = ("M", "d", "s", "p", "z", "zr", "lw", "f", "a")          # order in a job, then the criterion
CLASSICAL = dict(M=20, d="MW", s="MRN", p="static", z=0.0, zr=0.0, lw=0.0, f=1.0, a=1.0)
ROW_REFERENCE = {"p": dict(d="n1"), "zr": dict(z=1e-16)}
F_DEP_MW = 0.5
OUT = _paths.get_results_dir() / "collapse"


@lru_cache(maxsize=None)
def f_dep(M, dust):
    if dust == "MW":
        return F_DEP_MW
    d = np.loadtxt(_paths.get_external_data_path("collapse_ics", "chiaki2015_n07_reverse_shock.txt"))
    na = float(dust[1:])
    return float(next(r[6] for r in d if r[0] == M and r[1] == na))


def populations(M, Z, dust, shape):
    """(populations, y_gas): the n1 ICs reshaped to ``shape`` and rescaled to f_dep(M, dust)."""
    info = nozawa_ccsn(M, Z, n_amb=1, grains="grackle")[2]
    pops, _ = sizes.populations(M, Z, shape)
    k = f_dep(M, dust) / info["f_dep"]
    pops = [Population(p.material, p.grid, k * p.x, p.table) for p in pops]
    locked = dict.fromkeys(GAS_ELEMENTS, 0.0)
    for p in pops:
        f = get_material(p.material).formula
        for e, nu in formula_atoms(f).items():
            if e in locked:
                locked[e] += nu * p.x.sum() / formula_mass(f)
    y_tot = info["y_tot"]
    y = {e: max(y_tot[e] - locked[e], 0.0) for e in GAS_ELEMENTS}
    y["C"] = max(y["C"] - y_tot["O"], 0.0)
    return pops, y


def _job(c, **k):
    return tuple(k[x] for x in KEYS) + (c,)


def _kw(job):
    return dict(zip(KEYS, job[:-1]))


def _env(job):
    k = _kw(job)
    return (k["z"], k["lw"], k["zr"], k["f"])


def one(job):
    k, criterion = _kw(job), job[-1]
    make = lambda Z: populations(k["M"], Z, k["d"], k["s"])
    track = lambda Z: genv.track(k["M"], Z, _env(job))
    kw = dict(accretion=k["p"] == "acc", dlnn=0.1, n0=1e-1, T_rad=2.725 * (1.0 + k["zr"]), alpha_T=k["a"])
    if criterion == "frag":
        Z = critical_dust_to_gas(make, track, criterion="frag", lo=1e-8, hi=10 ** -2.5, scan_dex=0.25, rtol=0.05,
                                 **kw)
    else:
        Z = critical_dust_to_gas(make, track, criterion="C", lo=1e-9, hi=1e-2, rtol=0.05, **kw)
    return job, math.log10(Z)


def row_reference(key):
    return {**CLASSICAL, **ROW_REFERENCE.get(key, {})}


def row_jobs(c):
    """The classical case and each input alone departing from it (``ROW_REFERENCE``)."""
    return list(dict.fromkeys([_job(c, **CLASSICAL)] + [_job(c, **{**row_reference(key), key: v})
                                                         for key, vals in FACTORS for v in vals]))


def results():
    """({(criterion, *KEYS): [Z_cr]}, {same key: set of tags}) from the results file; tags: ``row``,
    ``min``, ``max`` (evaluated in that search)."""
    out, tags = {}, {}
    for ln in (OUT / "science_grid_transition.txt").read_text().splitlines():
        if not ln.startswith("#"):
            v = ln.split()
            c, M, d, s, p = v[0], int(v[1]), v[2], v[3], v[4]
            z, zr, lw, f, a = (float(x) for x in v[5:10])
            key = (c, M, d, s, p, z, zr, lw, f, a)
            out[key], tags[key] = float(v[10]), set(v[11].split(","))
    return out, tags


class Runner:
    def __init__(self, a):
        self.a = a
        self.partial = OUT / "science_grid_transition.partial"
        self.res = {}
        if self.partial.exists():
            for ln in self.partial.read_text().splitlines():
                j, z = json.loads(ln)
                if len(j) == 6:          # before redshift, LW, collapse time and accommodation
                    j = j[:5] + [0.0, 0.0, 1.0, 1.0] + [j[5]]
                self.res[tuple(j)] = z

    def __call__(self, jobs):
        todo = [j for j in dict.fromkeys(jobs) if j not in self.res]
        genv.ensure([(_kw(j)["M"], _env(j)) for j in todo], self.a.grackle_python, self.a.grackle_src,
                    self.a.workers)
        with ProcessPoolExecutor(self.a.workers) as ex, open(self.partial, "a") as fh:
            for fut in as_completed([ex.submit(one, j) for j in todo]):
                job, z = fut.result()
                self.res[job] = z
                fh.write(json.dumps([list(job), z]) + "\n")
                fh.flush()
        return [self.res[j] for j in jobs]


def extreme(run, c, side, max_iter=6):
    """Steepest descent (``min``) or ascent (``max``) from each input's extreme choice in the rows,
    over single changes of one input between its two top-ranked choices; returns every job
    evaluated."""
    sign = 1.0 if side == "min" else -1.0
    val = lambda z: sign * (z if math.isfinite(z) else math.inf)
    top2 = {}
    for key, vals in FACTORS:
        z = run([_job(c, **{**row_reference(key), key: v}) for v in vals])
        top2[key] = [vals[i] for i in np.argsort([val(x) for x in z], kind="stable")[:2]]
    cur = _job(c, **{**CLASSICAL, **{k: v[0] for k, v in top2.items()}})
    seen = [cur]
    best = val(run([cur])[0])
    for _ in range(max_iter):
        if best == -math.inf:            # the maximum is "no fragmentation": nothing beats it
            break
        nbrs = [_job(c, **{**_kw(cur), key: v}) for key, _ in FACTORS for v in top2[key] if v != _kw(cur)[key]]
        z = run(nbrs)
        seen += nbrs
        i = int(np.argmin([val(x) for x in z]))
        if val(z[i]) >= best - 1e-3:
            break
        cur, best = nbrs[i], val(z[i])
    return seen


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--grackle-src", required=True)
    ap.add_argument("--grackle-python", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    run = Runner(a)
    tags = {}
    run([j for c in ("frag", "C") for j in row_jobs(c)])
    for c in ("frag", "C"):
        for j in row_jobs(c):
            tags.setdefault(j, set()).add("row")
        for side in ("min", "max"):
            for j in extreme(run, c, side):
                tags.setdefault(j, set()).add(side)
    res = run.res
    lines = ["# [Z_cr] (Z_sun = 0.02), diagnostics/collapse/science_grid_transition.py",
             "# columns: criterion, progenitor mass, dust surviving (f_dep), size distribution, accretion, zeta0 "
             "[s^-1], redshift, J_21, collapse-time factor, alpha_T, [Z_cr] (inf: none), tags (row: one input "
             "from classical; min / max: evaluated in the search for the extremes)"]
    for j, t in tags.items():
        k = _kw(j)
        lines.append(f"{j[-1]} {k['M']} {k['d']} {k['s']} {k['p']} {k['z']:.0e} {k['zr']:g} {k['lw']:g} {k['f']:g} "
                     f"{k['a']:g} {res[j]:.3f} {','.join(sorted(t))}")
    (OUT / "science_grid_transition.txt").write_text("\n".join(lines) + "\n")
    for c in ("frag", "C"):
        lo = min(res[j] for j, t in tags.items() if j[-1] == c and "min" in t)
        hi = max(res[j] for j, t in tags.items() if j[-1] == c and "max" in t)
        arg = min((res[j], j) for j, t in tags.items() if j[-1] == c and "min" in t)[1]
        print(f"{c}: classical {res[_job(c, **CLASSICAL)]:.2f}; all combinations from {lo:.2f} ({_kw(arg)}) "
              f"to {hi:.2f}")


if __name__ == "__main__":
    main()
