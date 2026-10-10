"""Regenerate the Grackle reference tables bundled for ``pycalima.collapse``.

Grackle's ``newchem-cpp`` branch carries the Chiaki & Wise (2019) metal and
dust chemistry with the N07 Pop III supernova injection pathways (Nozawa+07
dust after a reverse shock into n_amb = 1 cm^-3, as in Chiaki et al. 2015). This
script writes, from a Grackle checkout at commit ``PINNED_COMMIT``:

* ``collapse_ics/grackle_ccsn_n1_grains.txt``: per pathway and grain species,
  the dust-to-metal mass fraction and the size moments <r>, <r^2>, <r^3> of the
  initial distribution (``src/clib/inject_model/raw_data.cpp``);
* ``collapse_ics/grackle_ccsn_n1_planck_opacity.txt``: their Planck-mean opacity
  per grain volume (the delta-r^0 coefficient over <r^3>), on log T_d = 0..3.4;
* ``collapse_tracks/grackle_ccsn{M}_n1_dustfree.txt``: dust-free collapse tracks
  (``grackle_freefall.py --dust-free``) on a [Z] grid, for ``Track.grackle``;
* ``tests/data/grackle_ccsn_n1_zcr.txt``: Grackle's own [Z_cr] with its full
  chemistry, without and with grain growth, by the same fragmentation criterion.

Build (once; outside the project, e.g. in a scratch directory):

    git clone -b newchem-cpp https://github.com/grackle-project/grackle.git
    git -C grackle checkout PINNED_COMMIT && git -C grackle submodule update --init
    brew install hdf5                       # or any HDF5 >= 1.6
    python3 -m venv gvenv && gvenv/bin/pip install ./grackle

then

    python diagnostics/collapse/make_grackle_tables.py --grackle-src grackle \
        --grackle-python gvenv/bin/python

About 15 min on 8 cores (dust-free tracks 2 min, the [Z_cr] searches the rest).
"""

from __future__ import annotations

import argparse
import math
import re
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from pycalima.collapse.integrator import fragmentation
from pycalima.collapse.tracks import KB_CGS, MH_CGS

PINNED_COMMIT = "9ac7cd5"
PATHWAYS = {1: 13, 2: 20, 3: 25, 4: 30}           # Grackle metal_abundances -> progenitor mass
LOG_Z = np.round(np.arange(-7.0, -1.999, 0.25), 2)
LOG_N = np.round(np.arange(-1.0, 16.0001, 0.05), 2)
LOG_TD = np.round(np.arange(35) * 0.1, 1)          # Grackle's dust-temperature grid
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXT = ROOT / "src" / "pycalima" / "data" / "external_data"
NUM = r"[-+]?\d[\d.]*e[-+]?\d+"


def parse_raw_data(src):
    """{(pathway name, species): dict(frac, mom (3,), coef (35, 4))} from raw_data.cpp."""
    out, ns, cur = {}, None, None
    for ln in (Path(src) / "src" / "clib" / "inject_model" / "raw_data.cpp").read_text().splitlines():
        m = re.match(r"namespace (\w+) \{", ln)
        if m:
            ns, cur = m.group(1), None
            continue
        m = re.search(r'"(\w+)_dust"', ln)
        if m:
            cur = out.setdefault((ns, m.group(1)), dict(frac=None, mom=None, coef=[]))
            continue
        if cur is None:
            continue
        v = [float(x) for x in re.findall(NUM, ln)]
        if "nonprimoridal_yield_frac" in ln:
            cur["frac"] = v[0]
        elif "size_moments" in ln:
            cur["mom"] = np.array(v)
        elif len(v) == 4 and ln.strip().startswith("{"):
            cur["coef"].append(v)
    for c in out.values():
        c["coef"] = np.array(c["coef"])
    return out


def _grackle(args, py, src, tmp):
    """Run grackle_freefall.py; returns (n_H, T, mu, f_H2)."""
    log_z, pathway, growth, dust_free = args
    out = Path(tmp) / f"p{pathway}_z{log_z:+.4f}_g{int(growth)}_d{int(dust_free)}.npz"
    if not out.exists():
        cmd = [py, str(HERE / "grackle_freefall.py"), str(out), "--log-z", str(log_z), "--pathway", str(pathway),
               "--data-dir", str(Path(src) / "grackle_data_files" / "input")]
        cmd += ["--grain-growth"] * growth + ["--dust-free"] * dust_free
        subprocess.run(cmd, check=True, capture_output=True)
    d = np.load(out)
    return d["nH"], d["T"], d["mu"], d["fH2"]


def _fragments(run):
    nH, T, mu, _ = run
    p = nH / 0.76 / (mu * MH_CGS) * MH_CGS * KB_CGS * T      # rho / (mu m_H) k T, rho = n_H m_H / X_H
    return bool(fragmentation(SimpleNamespace(arrays=lambda: dict(nH=nH, T=T, mu=mu, p=p))))


def _zcr(case, py, src, tmp, step=0.25, tol=0.025):
    """Lowest [Z] that fragments: scanned up from -7 in ``step`` (above ~[Z] = -3.5
    the dust cools the gas to the end of the run, gamma never returns above 0.97
    and the criterion finds nothing, so it is not monotonic in Z), then bisected."""
    pathway, growth = case
    frag = lambda z: _fragments(_grackle((round(z, 4), pathway, growth, False), py, src, tmp))
    first = next((z for z in np.arange(-7.0, -1.999, step) if frag(z)), None)
    if first is None:
        return case, math.inf
    lo, hi = first - step, first
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if frag(mid) else (mid, hi)
    return case, hi


def write_grains(raw, commit):
    hdr = (f"# Initial grains of Grackle's N07 Pop III CCSN injection pathways (Chiaki & Wise 2019,\n"
           f"# MNRAS 482, 3933; Nozawa+07 dust after a reverse shock into n_amb = 1 cm^-3, the N07M*n1\n"
           f"# models of Chiaki+15), from src/clib/inject_model/raw_data.cpp of Grackle (newchem-cpp,\n"
           f"# commit {commit}; NCSA/BSD licence), {date.today().isoformat()}, by\n"
           f"# diagnostics/collapse/make_grackle_tables.py. Species names as in pycalima (Grackle:\n"
           f"# SiM = Si, FeM = Fe, AC = C).\n"
           f"# columns: progenitor mass [Msun], species, dust-to-metal mass fraction, <r> [cm], <r^2> [cm^2],\n"
           f"# <r^3> [cm^3] (number-weighted moments of the size distribution)\n")
    rows = [f"{PATHWAYS[k]:3d} {_name(sp):8s} {c['frac']:.5e} {c['mom'][0]:.5e} {c['mom'][1]:.5e} {c['mom'][2]:.5e}"
            for k in PATHWAYS for (ns, sp), c in raw.items() if ns == f"ccsn{PATHWAYS[k]}"]
    (EXT / "collapse_ics" / "grackle_ccsn_n1_grains.txt").write_text(hdr + "\n".join(rows) + "\n")


def write_opacity(raw, commit):
    cols = [(k, sp, c) for k in PATHWAYS for (ns, sp), c in raw.items() if ns == f"ccsn{PATHWAYS[k]}"]
    hdr = (f"# Planck-mean opacity of the grains of Grackle's N07 CCSN pathways per unit grain volume,\n"
           f"# (delta-r^0 coefficient of the opacity polynomial) / <r^3>, in Grackle's units: for carbon\n"
           f"# it is 7.85e15 times the Planck-mean Q_abs / a of pycalima's carbon at every T, so Q_abs / a\n"
           f"# [cm^-1] ~ value / 7.85e15 in the small-grain limit. Optical constants: Nozawa+08 Table 1\n"
           f"# (Si: Piller 1985). From raw_data.cpp of Grackle (newchem-cpp, commit {commit}),\n"
           f"# {date.today().isoformat()}, by diagnostics/collapse/make_grackle_tables.py.\n"
           f"# columns: log10 T_d [K], then " + " ".join(f"M{PATHWAYS[k]}:{_name(sp)}" for k, sp, _ in cols) + "\n")
    rows = [f"{t:4.1f} " + " ".join(f"{c['coef'][i, 0] / c['mom'][2]:.5e}" for _, _, c in cols)
            for i, t in enumerate(LOG_TD)]
    (EXT / "collapse_ics" / "grackle_ccsn_n1_planck_opacity.txt").write_text(hdr + "\n".join(rows) + "\n")


def write_tracks(runs, commit):
    for k, M in PATHWAYS.items():
        lT, lf = [], []
        for z in LOG_Z:
            nH, T, _, fH2 = runs[(z, k)]
            ln = np.log10(nH)
            lT.append(np.interp(LOG_N, ln, np.log10(T)))
            lf.append(np.interp(LOG_N, ln, np.log10(np.maximum(fH2, 1e-30))))
        hdr = (f"# Dust-free one-zone collapse tracks with Grackle's chemistry (newchem-cpp, commit {commit}):\n"
               f"# primordial (with HD) and Chiaki & Wise (2019) metal chemistry, the gas-phase metals of\n"
               f"# injection pathway ccsn{M} (N07 {M} Msun CCSN, reverse shock into n_amb = 1 cm^-3), dust\n"
               f"# module off; n_H = 0.1 cm^-3, 300 K, x_e = 1e-4 to 1e16 cm^-3, z = 0 CMB, Omukai+05 collapse\n"
               f"# with consistent compressional heating (diagnostics/collapse/grackle_freefall.py --dust-free).\n"
               f"# [Z/H] with Z_sun = 0.02. {date.today().isoformat()}, diagnostics/collapse/make_grackle_tables.py.\n"
               f"# columns: log10 n_H [cm^-3], log10 T at [Z/H] = {', '.join(f'{z:g}' for z in LOG_Z)},\n"
               f"# then log10 f_H2 (= 2 n(H2) / n_H) at the same [Z/H]\n")
        d = np.column_stack([LOG_N, *lT, *lf])
        np.savetxt(EXT / "collapse_tracks" / f"grackle_ccsn{M}_n1_dustfree.txt", d, fmt="%.4f", header=hdr.rstrip("\n"),
                   comments="")


def write_zcr(res, commit):
    hdr = (f"# Grackle's own critical metallicity for the N07 CCSN pathways (n_amb = 1 cm^-3), full\n"
           f"# chemistry and dust (Chiaki & Wise 2019; newchem-cpp, commit {commit}), by the Schneider &\n"
           f"# Omukai (2010) fragmentation criterion of pycalima.collapse.integrator.fragmentation, to\n"
           f"# 0.025 dex; {date.today().isoformat()}, diagnostics/collapse/make_grackle_tables.py.\n"
           f"# columns: progenitor mass [Msun], [Z_cr] without grain growth, [Z_cr] with grain growth\n")
    rows = [f"{M:3d} {res[(k, False)]:6.3f} {res[(k, True)]:6.3f}" for k, M in PATHWAYS.items()]
    (ROOT / "tests" / "data" / "grackle_ccsn_n1_zcr.txt").write_text(hdr + "\n".join(rows) + "\n")


def _name(sp):
    return {"SiM": "Si", "FeM": "Fe", "AC": "C"}.get(sp, sp)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--grackle-src", required=True, help="Grackle checkout (newchem-cpp)")
    ap.add_argument("--grackle-python", required=True, help="python of the gracklepy install")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    commit = subprocess.check_output(["git", "-C", a.grackle_src, "rev-parse", "--short", "HEAD"], text=True).strip()
    if not commit.startswith(PINNED_COMMIT) and not PINNED_COMMIT.startswith(commit):
        print(f"warning: Grackle checkout at {commit}, tables were made at {PINNED_COMMIT}")
    raw = parse_raw_data(a.grackle_src)
    write_grains(raw, commit)
    write_opacity(raw, commit)
    with tempfile.TemporaryDirectory() as tmp, ProcessPoolExecutor(a.workers) as ex:
        run = partial(_grackle, py=a.grackle_python, src=a.grackle_src, tmp=tmp)
        jobs = [(z, k, False, True) for z in LOG_Z for k in PATHWAYS]
        runs = {(z, k): r for (z, k, _, _), r in zip(jobs, ex.map(run, jobs))}
        write_tracks(runs, commit)
        zcr = partial(_zcr, py=a.grackle_python, src=a.grackle_src, tmp=tmp)
        res = dict(ex.map(zcr, [(k, g) for k in PATHWAYS for g in (False, True)]))
        write_zcr(res, commit)
    for (k, g), z in sorted(res.items()):
        print(f"ccsn{PATHWAYS[k]} grain_growth={g}: [Z_cr] = {z:.2f}")


if __name__ == "__main__":
    main()
