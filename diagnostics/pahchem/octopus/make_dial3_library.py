#!/usr/bin/env python3
"""Octopus production runs on DIaL3 for the charge-resolved PAH cross-section library (2026-10-25).

The molecules: the coronene family (C54H18, C96H24, C150H30, C216H36) and, to separate compactness from size, the
non-compact isomers of PAHdb v4.00 at fixed N_C (C54H20, H24, H28, H30; C96H22, H28, H30; C150H36), each as -1, 0,
+1 and +2 on that state's own PAHdb geometry (B3LYP). Coronene, pyrene and ovalene ran on megatron.

Per state, ``make_run.py`` writes the inputs of the converged set-up (standard pseudopotentials, 0.25 A, atoms 4 A
from the 1 A absorbing boundary, delta kick 0.01/A, AETRS dt = 0.002, T = 20 hbar/eV); here the kicks get the
cheapest converged layout of the C96H24 scaling tests (``ParStates = 4``, ``ParDomains = 32`` on one node: 235
core-h per kick, against 540 for states only), and Slurm jobs: the ground state, the three kicks as an array after
it, the spectrum after them (``submit.sh``). Time limits follow 235 core-h (N_C / 96)^2.5 per kick (which also gives
the measured coronene cost), x2 for the spin-polarised odd-electron states, x2.5 margin, at least 2 h.

PAHdb stays on this machine: only the .xyz files of these states go to DIaL3.

A second box (the user's choice, 2026-10-27): ``--edge 7 --time 10 --suffix _b8`` writes the same states in an 8 A
box at T = 10. Averaged with the production box above ~9 eV, it smooths the continuum that a finite box makes
discrete (results/pahchem/octopus_box_2026-10-26). The time limits scale with the grid volume (``box_factor``) and T.

    make_dial3_library.py --out ~/Documents/RAMSES_dev/pahspec/dial3/library [--xml <pahdb v4.00 xml>]
                          [--edge 4 --time 20 --suffix ""]
"""
import argparse
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NS = "{http://www.astrochemistry.org/pahdb/theoretical}"
SYMBOL = {1: "H", 6: "C"}
#: label -> PAHdb v4.00 UIDs of charges -1, 0, +1, +2 (normally hydrogenated, pure C/H)
MOLECULES = {
    "C54H18": (46, 37, 38, 44), "C96H24": (114, 108, 111, 112), "C150H30": (614, 612, 613, 4444),
    "C216H36": (622, 615, 621, 4445),
    "C54H20_3176": (3178, 3176, 3177, 6045), "C54H24_3870": (3898, 3870, 3907, 6285),
    "C54H28_3826": (3823, 3826, 3857, 6265), "C54H30_3835": (3832, 3835, 3829, 6268),
    "C96H22_699": (701, 699, 700, 4476), "C96H28_3373": (3372, 3373, 3371, 6119),
    "C96H30_3317": (3321, 3317, 3342, 6104), "C150H36_4010": (4026, 4010, 4036, 6325),
}
CHARGES = (-1, 0, 1, 2)
#: doublets whose hole sits in a quasi-degenerate pair and is shared 0.5/0.5 (make_run.py --shared-hole; the user's
#: choice, 2026-10-26): C216H36+ never converged with integer occupations
SHARED_HOLE = {("C216H36", 1)}
#: doublets whose extra electron sits in a quasi-degenerate pair and is shared 0.5/0.5 (make_run.py --shared-electron;
#: the user's choice, 2026-10-27): C54H30_3835- never converged with integer occupations
SHARED_ELECTRON = {("C54H30_3835", -1)}
CORE_H_C96 = 235.0           # per kick, T = 20, 1 node, ParStates 4 x ParDomains 32
MODULES = "intel-oneapi-compilers/2021.2.0/3q6eev openmpi/4.0.5/ra5ctk intel-oneapi-mkl/2021.4.0/mtuwg6 gsl/2.7/svkqkb"
ENV = ("export PATH=$HOME/opt/octopus-16.4/bin:$PATH LD_LIBRARY_PATH=$HOME/opt/deps/lib64:$HOME/opt/deps/lib:"
       "${LD_LIBRARY_PATH:-}\nexport OMP_NUM_THREADS=1 MKL_NUM_THREADS=1")


def geometries(xml_path, uids):
    """{uid: (formula, charge, method, multiplicity, [(symbol, x, y, z)])} of the PAHdb XML species ``uids``; the
    multiplicity is the leading number of PAHdb's state label ("3-A1G": a triplet)."""
    want, out = set(uids), {}
    for _, el in ET.iterparse(xml_path, events=("end",)):
        if el.tag != NS + "specie":
            continue
        uid = int(el.get("uid"))
        if uid in want:
            atoms = [(SYMBOL[int(a.findtext(NS + "type"))], *(float(a.findtext(NS + c)) for c in "xyz"))
                     for a in el.find(NS + "geometry")]
            label = (el.findtext(NS + "symmetry") or "").strip()
            mult = int(label.split("-")[0]) if label[:1].isdigit() else None
            out[uid] = (el.findtext(NS + "formula"), int(el.findtext(NS + "charge")), el.findtext(NS + "method"), mult,
                        atoms)
        el.clear()
        if len(out) == len(want):
            break
    missing = want - set(out)
    if missing:
        raise SystemExit(f"UIDs not in {xml_path}: {sorted(missing)}")
    return out


def kick_hours(nc, polarised, scale=1.0):
    """Slurm time per kick [h]. The cost per kick is CORE_H_C96 x (N_C / 96)^p: p = 1 below C96 (the first library runs,
    2026-10-26: closed-shell C54 kicks took 0.75-1.25 h, against 0.44 h from p = 2.5) and 2.5 above (unmeasured). Spin
    polarisation costs x3 (C54: 2.0-3.3 h). A x2 margin: a kick that still runs out resumes from restart/td
    (job_td_resume.sh in the library). ``scale``: the grid volume and propagation time relative to production."""
    p = 1.0 if nc <= 96 else 2.5
    return max(2.0, 2.0 * CORE_H_C96 * (nc / 96.0) ** p / 128.0 * (3.0 if polarised else 1.0) * scale)


def box_factor(atoms, edge, absorb=1.0, production=5.0):
    """Grid points of the minimum box (spheres of radius edge + absorb around every atom) relative to the production
    4 + 1 A box, for a planar PAH taken as a disk of radius r + R and height 2R (r: the in-plane radius)."""
    xyz = np.array([[x, y, z] for _, x, y, z in atoms])
    c = xyz - xyz.mean(axis=0)
    w, v = np.linalg.eigh(c.T @ c)
    r = np.sqrt(((c @ v[:, 1:]) ** 2).sum(axis=1)).max()
    R = edge + absorb
    return (r + R) ** 2 * R / ((r + production) ** 2 * production)


def gs_hours(nc):
    """Slurm time of a ground state [h]: the charged C216H36 states needed > 4 h (70-110 SCF iterations)."""
    return 12.0 if nc >= 150 else 4.0


def slurm(name, hours, body, extra=""):
    h, m = int(hours), int(round((hours - int(hours)) * 60))
    return f"""#!/bin/bash
#SBATCH --job-name={name}
#SBATCH -o slurm.%x.%j.out
#SBATCH -e slurm.%x.%j.err
#SBATCH -A dp016
#SBATCH --partition=slurm
#SBATCH --time={min(h, 96):02d}:{m:02d}:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --exclusive
{extra}set -euo pipefail
module purge
module load {MODULES}
{ENV}
cd "$SLURM_SUBMIT_DIR"
{body}
"""


def write_jobs(run, label, z, nc, polarised, scale=1.0, gs_scale=1.0, suffix=""):
    tag = f"{label}_{'m' if z < 0 else 'p'}{abs(z)}{suffix}"
    # Octopus exits 0 after "SCF *not* converged": gs.ok (and so the kicks, afterok) only for a converged ground state
    (run / "job_gs.sh").write_text(slurm(f"gs_{tag}", gs_hours(nc) * gs_scale, """(cd gs && mpirun -np 128 octopus > out.log 2>&1)
grep -q "SCF converged" gs/out.log || { echo "SCF not converged" >&2; exit 1; }
echo GS_OK > gs.ok"""))
    (run / "job_td.sh").write_text(slurm(f"td_{tag}", kick_hours(nc, polarised, scale), """axis=$(echo x y z | cut -d' ' -f$SLURM_ARRAY_TASK_ID)
d=td_$axis
mkdir -p $d/restart && rm -rf $d/restart/gs && cp -r gs/restart/gs $d/restart/gs
(cd $d && mpirun -np 128 octopus > out.log 2>&1)
echo TD_OK > $d/td.ok""", extra="#SBATCH --array=1-3\n"))
    # a kick that ran out of time resumes from its restart/td (FromScratch = no is Octopus's default); an axis that
    # already finished is skipped. Submit as: sbatch --dependency=afterany:<td job> job_td_resume.sh
    (run / "job_td_resume.sh").write_text(slurm(f"td_{tag}_r", kick_hours(nc, polarised, scale), """axis=$(echo x y z | cut -d' ' -f$SLURM_ARRAY_TASK_ID)
d=td_$axis
[ -f $d/td.ok ] && { echo "$d already done"; exit 0; }
mkdir -p $d/restart && rm -rf $d/restart/gs && cp -r gs/restart/gs $d/restart/gs
cp $d/out.log $d/out.log.part1; cp $d/td.general/multipoles $d/multipoles.part1
(cd $d && mpirun -np 128 octopus > out.log 2>&1)
echo TD_OK > $d/td.ok""", extra="#SBATCH --array=1-3\n"))
    (run / "job_spec.sh").write_text(slurm(f"sp_{tag}", 0.5, """i=1; for d in td_x td_y td_z; do cp $d/td.general/multipoles spectrum/multipoles.$i; i=$((i + 1)); done
(cd spectrum && oct-propagation_spectrum > spectrum.log 2>&1)
echo RUN_OK > run.ok""").replace("--nodes=1\n#SBATCH --ntasks-per-node=128\n#SBATCH --exclusive", "--nodes=1\n#SBATCH --ntasks=1"))
    (run / "submit.sh").write_text("""#!/bin/bash
# the ground state, then the three kicks after it, then the spectrum after them
set -euo pipefail
cd "$(dirname "$0")"
g=$(sbatch --parsable job_gs.sh)
t=$(sbatch --parsable --dependency=afterok:$g job_td.sh)
s=$(sbatch --parsable --dependency=afterok:$t job_spec.sh)
echo "$(basename "$PWD"): gs $g, td $t, spectrum $s"
""")
    (run / "submit.sh").chmod(0o755)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--xml", type=Path, default=None, help="PAHdb v4.00 XML (default: the registered dataset)")
    ap.add_argument("--edge", type=float, default=4.0, help="atom to absorbing region [A] (make_run.py)")
    ap.add_argument("--time", type=float, default=20.0, help="propagation time [hbar/eV] (make_run.py)")
    ap.add_argument("--suffix", default="", help="appended to the Slurm job names, e.g. _b8")
    a = ap.parse_args()
    if a.xml is None:
        from pycalima.pahchem.ames_suite import xml_path
        a.xml = xml_path()
    uids = [u for v in MOLECULES.values() for u in v if u is not None]
    geo = geometries(a.xml, uids)
    (a.out / "geometries").mkdir(parents=True, exist_ok=True)
    total = 0.0
    for label, ids in MOLECULES.items():
        for z, uid in zip(CHARGES, ids, strict=True):
            if uid is None:
                print(f"{label} {z:+d}: not in PAHdb v4.00, skipped")
                continue
            formula, charge, method, mult, atoms = geo[uid]
            if charge != z:
                raise SystemExit(f"UID {uid} has charge {charge}, not {z}")
            xyz = a.out / "geometries" / f"{label}_{z:+d}.xyz"
            xyz.write_text(f"{len(atoms)}\nPAHdb v4.00 UID {uid}, {formula}, charge {charge:+d}, {method}\n"
                           + "".join(f"{s} {x:.8f} {y:.8f} {zz:.8f}\n" for s, x, y, zz in atoms))
            run = a.out / f"{label}_{z:+d}"
            # PAHdb's spin state: 9 of the 12 dications are triplets (results/pahchem/octopus_dications_2026-10-26)
            cmd = [sys.executable, str(HERE / "make_run.py"), str(xyz), "--charge", str(z), "--out", str(run),
                   "--edge", str(a.edge), "--time", str(a.time)]
            subprocess.run(cmd + (["--multiplicity", str(mult)] if mult else [])
                           + (["--shared-hole"] if (label, z) in SHARED_HOLE else [])
                           + (["--shared-electron"] if (label, z) in SHARED_ELECTRON else []), check=True)
            for axis in "xyz":
                with (run / f"td_{axis}" / "inp").open("a") as fh:
                    fh.write("ParStates = 4\nParDomains = 32\n")
            nc = sum(1 for s, *_ in atoms if s == "C")
            n_el = 4 * nc + sum(1 for s, *_ in atoms if s == "H") - z
            polarised = n_el % 2 == 1 or (mult or 1) > 1
            box = box_factor(atoms, a.edge)
            scale = box * a.time / 20.0
            write_jobs(run, label, z, nc, polarised, scale, box, a.suffix)
            total += 3 * 128 * kick_hours(nc, polarised, scale) / 2.0          # the time limit holds a x2 margin
    print(f"estimated {total:,.0f} core-h for the kicks")


if __name__ == "__main__":
    main()
