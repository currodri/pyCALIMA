#!/usr/bin/env python3
"""Octopus inputs for the Coulomb-window pilot (gas-coupling section 2.2c): a PAH ion's photoionisation yield from real-time
TD-DFT with a weak monochromatic pulse and a mask boundary.

Per box edge (atoms to the mask): the ion's ground state in a minimum box of radius edge + absorb (``make_run.system_block``,
the library's LDA, pseudopotentials and open-shell occupations), then one run per (photon energy, axis): a cosinoidal pulse
of half-width ``tau`` along the axis, weak enough for linear response, propagated to ``time``. The yield is the electron
number the mask removes (the l = 0 multipole) over the photons absorbed (the field's work on the dipole over omega):
``analyse_window_pilot.py``. A Coulomb-bound (window) electron whose orbit reaches past the mask is counted as ionised, so the
pilot runs two edges: their difference is the population of window states between them, and N_lost(t) after the pulse is
their autoionisation in time. ``--smoke`` writes a short propagation (T = 1) on the devel partition, to time the layout
before the pilot proper.

    make_window_pilot.py coronene+.xyz --charge 1 --out DIR [--edges 6 12] [--omegas 15 18] [--axes x z] [--smoke]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import make_dial3_library as D  # noqa: E402
import make_run as R  # noqa: E402


def td_input(sysb, occ, axis, omega, amplitude, tau, time, dt, absorb):
    """A pulse run along ``axis`` (Octopus has no volt: the field is eV/angstrom, V/A on a unit charge): amplitude cos(omega t) times tdf_cosinoidal (oct-help: a cosine bump of half-width tau0
    centred at t0, here both tau), so the pulse runs from 0 to 2 tau."""
    vec = {"x": "1 | 0 | 0", "y": "0 | 1 | 0", "z": "0 | 0 | 1"}[axis]
    return "\n".join([
        "CalculationMode = td", sysb,
        "TDPropagator = aetrs",
        f"TDTimeStep = {dt}/eV",
        f"TDPropagationTime = {time}/eV",
        "AbsorbingBoundaries = mask",
        f"ABWidth = {absorb}*angstrom",
        f"%TDExternalFields\n  electric_field | {vec} | {omega}*eV | \"envelope\"\n%",
        f"%TDFunctions\n  \"envelope\" | tdf_cosinoidal | {amplitude}*eV/angstrom | {tau}/eV | {tau}/eV\n%",
        "%TDOutput\n  multipoles\n  laser\n  energy\n%",          # a block in Octopus 16 (oct-help -p TDOutput)
        "ParStates = 4\nParDomains = 32",
        occ,
    ])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("xyz", type=Path)
    ap.add_argument("--charge", type=int, required=True)
    ap.add_argument("--multiplicity", type=int, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--edges", type=float, nargs="+", default=[6.0, 12.0], help="atoms to the mask [A]")
    ap.add_argument("--absorb", type=float, default=2.0, help="mask width [A]")
    ap.add_argument("--omegas", type=float, nargs="+", default=[15.0, 18.0], help="photon energies [eV]")
    ap.add_argument("--axes", nargs="+", default=["x", "z"], choices=["x", "y", "z"])
    ap.add_argument("--amplitude", type=float, default=0.02, help="peak field [V/A]; 0.02 is ~5e9 W/cm2")
    ap.add_argument("--tau", type=float, default=10.0, help="pulse half-width [hbar/eV]; the pulse lasts 2 tau")
    ap.add_argument("--time", type=float, default=60.0, help="propagation time [hbar/eV]")
    ap.add_argument("--dt", type=float, default=0.002, help="time step [hbar/eV]")
    ap.add_argument("--spacing", type=float, default=0.25)
    ap.add_argument("--hours", type=float, default=24.0, help="Slurm time per pulse run [h]")
    ap.add_argument("--smoke", action="store_true", help="T = 1 on devel, to time the layout")
    a = ap.parse_args()

    n_el = R.electrons(a.xyz, a.charge)
    mult = a.multiplicity if a.multiplicity is not None else 1 + n_el % 2
    occ = R.occupation_block(n_el, mult)
    time, hours = (1.0, 1.0) if a.smoke else (a.time, a.hours)
    for edge in a.edges:
        run = a.out / f"box_{edge:g}"
        sysb = R.system_block(a.charge, "lda", a.spacing, edge + a.absorb, mult > 1)
        (run / "gs").mkdir(parents=True, exist_ok=True)
        (run / "geom.xyz").write_text(a.xyz.read_text())
        (run / "gs" / "inp").write_text(f"CalculationMode = gs\n{sysb}\nConvRelDens = 1e-7\nMaximumIter = 300\n"
                                        f"LCAOStart = lcao_none\n{occ}")    # ParStates in a gs is experimental in Octopus 16
        dirs = []
        for omega in a.omegas:
            for axis in a.axes:
                d = f"td_{axis}_{omega:g}"
                (run / d).mkdir(exist_ok=True)
                (run / d / "inp").write_text(td_input(sysb, occ, axis, omega, a.amplitude, a.tau, time, a.dt, a.absorb))
                dirs.append(d)
        (run / "td_dirs.txt").write_text("\n".join(dirs) + "\n")
        tag = f"win_{edge:g}"
        part = "devel" if a.smoke else "slurm"
        gs = D.slurm(f"gs_{tag}", 2.0 if a.smoke else 6.0, """(cd gs && mpirun -np 128 octopus > out.log 2>&1)
grep -q "SCF converged" gs/out.log || { echo "SCF not converged" >&2; exit 1; }
echo GS_OK > gs.ok""").replace("--partition=slurm", f"--partition={part}")
        td = D.slurm(f"td_{tag}", hours, """d=$(sed -n "${SLURM_ARRAY_TASK_ID}p" td_dirs.txt)
mkdir -p $d/restart && rm -rf $d/restart/gs && cp -r gs/restart/gs $d/restart/gs
(cd $d && mpirun -np 128 octopus > out.log 2>&1)
echo TD_OK > $d/td.ok""", extra=f"#SBATCH --array=1-{len(dirs)}\n").replace("--partition=slurm", f"--partition={part}")
        (run / "job_gs.sh").write_text(gs)
        (run / "job_td.sh").write_text(td)
        (run / "submit.sh").write_text("""#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
g=$(sbatch --parsable job_gs.sh)
t=$(sbatch --parsable --dependency=afterok:$g job_td.sh)
echo "$(basename "$PWD"): gs $g, td $t"
""")
        (run / "submit.sh").chmod(0o755)
        print(f"{run}: {n_el} electrons, multiplicity {mult}, radius {edge + a.absorb:g} A, "
              f"{len(dirs)} pulse runs, T = {time:g}/eV, {part}")


if __name__ == "__main__":
    main()
