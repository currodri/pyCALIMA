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

    make_dial3_library.py --out ~/Documents/RAMSES_dev/pahspec/dial3/library [--xml <pahdb v4.00 xml>]
"""
import argparse
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

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
CORE_H_C96 = 235.0           # per kick, T = 20, 1 node, ParStates 4 x ParDomains 32
MODULES = "intel-oneapi-compilers/2021.2.0/3q6eev openmpi/4.0.5/ra5ctk intel-oneapi-mkl/2021.4.0/mtuwg6 gsl/2.7/svkqkb"
ENV = ("export PATH=$HOME/opt/octopus-16.4/bin:$PATH LD_LIBRARY_PATH=$HOME/opt/deps/lib64:$HOME/opt/deps/lib:"
       "${LD_LIBRARY_PATH:-}\nexport OMP_NUM_THREADS=1 MKL_NUM_THREADS=1")


def geometries(xml_path, uids):
    """{uid: (formula, charge, method, [(symbol, x, y, z)])} of the PAHdb XML species ``uids``."""
    want, out = set(uids), {}
    for _, el in ET.iterparse(xml_path, events=("end",)):
        if el.tag != NS + "specie":
            continue
        uid = int(el.get("uid"))
        if uid in want:
            atoms = [(SYMBOL[int(a.findtext(NS + "type"))], *(float(a.findtext(NS + c)) for c in "xyz"))
                     for a in el.find(NS + "geometry")]
            out[uid] = (el.findtext(NS + "formula"), int(el.findtext(NS + "charge")), el.findtext(NS + "method"), atoms)
        el.clear()
        if len(out) == len(want):
            break
    missing = want - set(out)
    if missing:
        raise SystemExit(f"UIDs not in {xml_path}: {sorted(missing)}")
    return out


def kick_hours(nc, polarised):
    return max(2.0, 2.5 * CORE_H_C96 * (nc / 96.0) ** 2.5 / 128.0 * (2.0 if polarised else 1.0))


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


def write_jobs(run, label, z, nc, polarised):
    tag = f"{label}_{'m' if z < 0 else 'p'}{abs(z)}"
    (run / "job_gs.sh").write_text(slurm(f"gs_{tag}", 4.0, "(cd gs && mpirun -np 128 octopus > out.log 2>&1)\necho GS_OK > gs.ok"))
    (run / "job_td.sh").write_text(slurm(f"td_{tag}", kick_hours(nc, polarised), """axis=$(echo x y z | cut -d' ' -f$SLURM_ARRAY_TASK_ID)
d=td_$axis
mkdir -p $d/restart && rm -rf $d/restart/gs && cp -r gs/restart/gs $d/restart/gs
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
            formula, charge, method, atoms = geo[uid]
            if charge != z:
                raise SystemExit(f"UID {uid} has charge {charge}, not {z}")
            xyz = a.out / "geometries" / f"{label}_{z:+d}.xyz"
            xyz.write_text(f"{len(atoms)}\nPAHdb v4.00 UID {uid}, {formula}, charge {charge:+d}, {method}\n"
                           + "".join(f"{s} {x:.8f} {y:.8f} {zz:.8f}\n" for s, x, y, zz in atoms))
            run = a.out / f"{label}_{z:+d}"
            subprocess.run([sys.executable, str(HERE / "make_run.py"), str(xyz), "--charge", str(z), "--out", str(run)],
                           check=True)
            for axis in "xyz":
                with (run / f"td_{axis}" / "inp").open("a") as fh:
                    fh.write("ParStates = 4\nParDomains = 32\n")
            nc = sum(1 for s, *_ in atoms if s == "C")
            n_el = 4 * nc + sum(1 for s, *_ in atoms if s == "H") - z
            write_jobs(run, label, z, nc, n_el % 2 == 1)
            total += 3 * CORE_H_C96 * (nc / 96.0) ** 2.5 * (2.0 if n_el % 2 else 1.0)
    print(f"estimated {total:,.0f} core-h for the kicks (spin-polarised states x2)")


if __name__ == "__main__":
    main()
