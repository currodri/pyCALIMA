"""Grackle dust-free tracks of the N07 CCSN pathways in other environments.

An environment is (zeta0, j21, redshift, delay): cosmic rays at zeta0 [s^-1] outside
the cloud (attenuated), a Lyman-Werner background J_21 (H2 self-shielded over the
Jeans length), the CMB at that redshift, and the collapse time multiplied by delay
(``grackle_freefall.py``). ``track`` loads one at a trial metallicity (the bundled
z = 0 track when the environment is the default one); ``ensure`` runs Grackle on
the [Z/H] grid of ``make_grackle_tables.py`` for the ones not yet on disk.
Tracks go to ``collapse/grackle_cr_tracks/`` and the runs are kept in
``collapse/grackle_cr_runs/`` under the results directory.
"""

from __future__ import annotations

import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from pycalima import _paths
from pycalima.collapse.tracks import Track

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_grackle_tables as mgt       # noqa: E402

OUT = _paths.get_results_dir() / "collapse"
RUNS = OUT / "grackle_cr_runs"
TRACKS = OUT / "grackle_cr_tracks"
DEFAULT = (0.0, 0.0, 0.0, 1.0)          # zeta0, j21, redshift, delay


def track_path(M, zeta0=0.0, j21=0.0, redshift=0.0, delay=1.0):
    env = (f"_lw{j21:g}" if j21 else "") + (f"_zr{redshift:g}" if redshift else "") + \
        (f"_tc{delay:g}" if delay != 1.0 else "")
    return TRACKS / f"grackle_ccsn{M}_n1_dustfree_cr{zeta0:g}{env}.txt"


def track(M, Z, env=DEFAULT):
    """Dust-free track of progenitor M at Z / Z_sun in environment ``env``."""
    path = None if tuple(env) == DEFAULT else track_path(M, *env)
    return Track.grackle(M, Z, path=path, t_col_factor=env[3])


def _note(zeta0, j21, redshift, delay):
    parts = []
    if zeta0:
        parts.append(f"cosmic rays zeta0 = {zeta0:g} s^-1 per H, attenuated as exp(-Sigma / 96 g cm^-2), Sigma = "
                     f"rho lambda_J / 2, 10 eV per ionisation")
    if j21:
        parts.append(f"Lyman-Werner background J_21 = {j21:g} (H2 dissociation 1.38e-12 J_21 s^-1, self-shielded "
                     f"over the Jeans length)")
    if redshift:
        parts.append(f"CMB at z = {redshift:g}")
    if delay != 1.0:
        parts.append(f"collapse time x {delay:g}")
    return "".join(f"# except: {p};\n" for p in parts) + "# diagnostics/collapse/grackle_environments.py.\n"


def ensure(configs, py, src, workers=8):
    """Write the tracks of every (M, env) in ``configs`` that is not on disk yet."""
    for d in (RUNS, TRACKS):
        d.mkdir(parents=True, exist_ok=True)
    todo = sorted({(M, tuple(env)) for M, env in configs
                   if tuple(env) != DEFAULT and not track_path(M, *env).exists()})
    if not todo:
        return
    commit = subprocess.check_output(["git", "-C", src, "rev-parse", "--short", "HEAD"], text=True).strip()
    pathway = {M: k for k, M in mgt.PATHWAYS.items()}
    jobs = [(M, env, z) for M, env in todo for z in mgt.LOG_Z]
    with ProcessPoolExecutor(workers) as ex:
        out = list(ex.map(_one, [(pathway[M], env, z, py, src) for M, env, z in jobs]))
    for M, env in todo:
        runs = {(z, pathway[M]): r for (m, e, z), r in zip(jobs, out) if m == M and e == env}
        mgt.write_tracks(runs, commit, path=lambda _M, env=env: track_path(_M, *env), note=_note(*env),
                         masses=[M])
        print(f"track M = {M}, env = {env} written", flush=True)


def _one(job):
    k, (zeta0, j21, redshift, delay), z, py, src = job
    return mgt._grackle((z, k, False, True), py, src, RUNS, zeta0=zeta0, j21=j21, redshift=redshift, delay=delay)
