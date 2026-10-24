"""What do cosmic rays do to the critical metallicity?

Cosmic rays at zeta0 (rate per H outside the cloud) attenuated as
exp(-Sigma / 96 g cm^-2), Sigma = rho lambda_J / 2 (``grackle_freefall.py``),
for the N07 CCSN models with reverse shock into n_amb = 1 cm^-3:

1. Grackle dust-free tracks with cosmic rays on the usual [Z] grid
   (``collapse/grackle_cr_tracks/`` under the results directory);
2. Grackle's own [Z_cr] with its full chemistry and dust, by the fragmentation
   criterion of ``make_grackle_tables.py``;
3. ours on those tracks, with Grackle's grains: criteria C and frag, static
   dust and accretion; and accretion with Coulomb focusing of Mg+ and Fe+,
   an upper bound: every ion is a metal ion (Marchand+21 charge balance with
   radiative recombination, as in ``plot_dust_evolution.py``) at the attenuated
   zeta, focus 1 - psi on the ionised fraction, from the grains of each trial.

    python diagnostics/collapse/grackle_cosmic_rays.py --grackle-src G --grackle-python P \
        [--zetas 1e-19 1e-18 1e-17 1e-16] [--workers 8]

Grackle runs are kept in ``collapse/grackle_cr_runs/``; results go to
``collapse/grackle_cosmic_rays.txt``. About 30 min on 8 cores.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from pycalima import _paths
from pycalima.collapse.charge import marchand21_charge_balance
from pycalima.collapse.ics import nozawa_ccsn
from pycalima.collapse.integrator import _state, critical_dust_to_gas
from pycalima.collapse.tracks import G_CGS, KB_CGS, MH_CGS, Track

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_grackle_tables as mgt       # noqa: E402

OUT = _paths.get_results_dir() / "collapse"
RUNS = OUT / "grackle_cr_runs"
TRACKS = OUT / "grackle_cr_tracks"
SIGMA_CR = 96.0
MU_METAL, K_RAD = 24.3, 2.8e-12          # Mg+ (UMIST radiative recombination at 300 K, x (T/300)^-0.68)
PROCESSES = ("static", "acc", "acc+focus")
# floating potential e U / k T of a grain in a gas of electrons (s_e = 0.5) and Mg+:
# 1 - psi = s_e (m_i / m_e)^1/2 e^psi (Marchand+21, n_e = n_i)
PSI_FLOAT = brentq(lambda p: (1.0 - p) - 0.5 * math.sqrt(MU_METAL * MH_CGS / 9.1093837e-28) * math.exp(p), -50, 1)


def track_path(M, zeta0):
    return TRACKS / f"grackle_ccsn{M}_n1_dustfree_cr{zeta0:g}.txt"


def zeta_along(track, nH, zeta0):
    """Attenuated cosmic-ray ionisation rate at n_H on a track (as grackle_freefall.py)."""
    T, mu, rho = float(track.T(nH)), float(track.mu(nH)), float(track.rho(nH))
    lam_J = math.sqrt(KB_CGS * T / (mu * MH_CGS)) * math.sqrt(math.pi / (G_CGS * rho))
    return zeta0 * math.exp(-0.5 * rho * lam_J / SIGMA_CR)


def focus_function(track, pops, y_gas, zeta0):
    """(n_H, T) -> {"Mg": f, "Fe": f}: upper-bound Coulomb focusing (module docstring),
    tabulated on the track for the initial grains."""
    ln = np.arange(-1.0, 16.01, 0.25)
    f = np.ones_like(ln)
    x_metal = y_gas["Mg"] + y_gas["Fe"]
    for i, l in enumerate(ln):
        nH = 10.0 ** l
        zeta = zeta_along(track, nH, zeta0)
        if zeta < 1e-30 or x_metal <= 0.0:
            continue
        T = float(track.T(nH))
        st = _state(pops, nH)
        a = np.concatenate([s[1] for s in st])
        ng = np.concatenate([s[0] for s in st])
        ok = ng > 0
        k_rec = K_RAD * (T / 300.0) ** -0.68
        try:
            r = marchand21_charge_balance(a[ok], ng[ok], T, nH, zeta, MU_METAL, k_rec=k_rec)
            n_i, psi = r["n_i"], r["psi"]
        except ValueError:      # grains too few to matter: gas-phase balance, floating potential
            n_i, psi = math.sqrt(zeta * nH / k_rec), PSI_FLOAT
        f[i] = 1.0 + min(n_i / (x_metal * nH), 1.0) * (-psi)
    return lambda nH, T: dict.fromkeys(("Mg", "Fe"), float(np.interp(math.log10(nH), ln, f)))


def ours(job):
    """[Z_cr] of our model on Grackle's cosmic-ray tracks (zeta0 = 0: the bundled tracks)."""
    M, zeta0, process, criterion = job
    path = track_path(M, zeta0) if zeta0 > 0 else None
    track = lambda Z: Track.grackle(M, Z, path=path)
    kw = dict(accretion=process != "static", dlnn=0.1, n0=1e-1)
    focus = {}          # of the current trial: critical_dust_to_gas runs make(Z), then the collapse
    if process == "acc+focus":
        kw["accretion_focus"] = lambda nH, T: focus["f"](nH, T)

    def make(Z):
        pops, y, _ = nozawa_ccsn(M, Z, n_amb=1, grains="grackle")
        if process == "acc+focus":
            focus["f"] = focus_function(track(Z), pops, y, zeta0)
        return pops, y

    if criterion == "frag":
        Z = critical_dust_to_gas(make, track, criterion="frag", lo=1e-6, hi=10 ** -2.5, scan_dex=0.25, rtol=0.05,
                                 **kw)
    else:
        Z = critical_dust_to_gas(make, track, criterion="C", lo=1e-8, hi=1e-2, rtol=0.05, **kw)
    return job, math.log10(Z)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--grackle-src", required=True)
    ap.add_argument("--grackle-python", required=True)
    ap.add_argument("--zetas", type=float, nargs="+", default=[1e-19, 1e-18, 1e-17, 1e-16])
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    for d in (RUNS, TRACKS):
        d.mkdir(parents=True, exist_ok=True)
    commit = mgt.subprocess.check_output(["git", "-C", a.grackle_src, "rev-parse", "--short", "HEAD"],
                                         text=True).strip()
    partial_file = OUT / "grackle_cosmic_rays.partial"
    res = {}
    if partial_file.exists():
        for ln in partial_file.read_text().splitlines():
            k, v = json.loads(ln)
            res[tuple(k)] = v
    with ProcessPoolExecutor(a.workers) as ex, open(partial_file, "a") as fh:
        # 1. dust-free tracks
        for z0 in a.zetas:
            if all(track_path(M, z0).exists() for M in mgt.PATHWAYS.values()):
                continue
            run = partial(mgt._grackle, py=a.grackle_python, src=a.grackle_src, tmp=RUNS, zeta0=z0)
            jobs = [(z, k, False, True) for z in mgt.LOG_Z for k in mgt.PATHWAYS]
            runs = {(z, k): r for (z, k, _, _), r in zip(jobs, ex.map(run, jobs))}
            note = (f"# Cosmic rays: zeta0 = {z0:g} s^-1 per H, attenuated as exp(-Sigma / 96 g cm^-2), Sigma =\n"
                    f"# rho lambda_J / 2, 10 eV of heat per ionisation (grackle_freefall.py --zeta0);\n"
                    f"# diagnostics/collapse/grackle_cosmic_rays.py.\n")
            mgt.write_tracks(runs, commit, path=lambda M: track_path(M, z0), note=note)
            print(f"tracks zeta0 = {z0:g} written", flush=True)
        # 2 and 3: Grackle's [Z_cr] and ours
        futs = {}
        for z0 in [0.0] + a.zetas:
            for k, M in mgt.PATHWAYS.items():
                key = ("grackle", M, z0)
                if key not in res:
                    futs[ex.submit(mgt._zcr, (k, False), a.grackle_python, a.grackle_src, RUNS, zeta0=z0)] = key
                for pr in PROCESSES:
                    for cr in ("C", "frag"):
                        key = ("ours", M, z0, pr, cr)
                        if key not in res and not (pr == "acc+focus" and z0 == 0.0):
                            futs[ex.submit(ours, (M, z0, pr, cr))] = key
        for fut in as_completed(futs):
            key = futs[fut]
            r = fut.result()
            res[key] = r[1]
            fh.write(json.dumps([list(key), r[1]]) + "\n")
            fh.flush()
    lines = ["# [Z_cr] (Z_sun = 0.02) with cosmic rays, diagnostics/collapse/grackle_cosmic_rays.py",
             "# columns: progenitor mass, zeta0 [s^-1], Grackle (full chemistry), then ours: "
             + ", ".join(f"{cr} {pr}" for cr in ("C", "frag") for pr in PROCESSES)]
    print("\n[Z_cr]: Grackle | ours C: static acc acc+focus | ours frag: static acc acc+focus")
    for M in mgt.PATHWAYS.values():
        for z0 in [0.0] + a.zetas:
            v = [res[("grackle", M, z0)]] + [res.get(("ours", M, z0, pr, cr), res.get(("ours", M, z0, "acc", cr)))
                                              for cr in ("C", "frag") for pr in PROCESSES]
            print(f"  {M:2d}  zeta0 = {z0:7.0e} | {v[0]:6.2f} | " + " ".join(f"{x:6.2f}" for x in v[1:4]) + " | "
                  + " ".join(f"{x:6.2f}" for x in v[4:]))
            lines.append(f"{M} {z0:.0e} " + " ".join(f"{x:.3f}" for x in v))
    (OUT / "grackle_cosmic_rays.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
