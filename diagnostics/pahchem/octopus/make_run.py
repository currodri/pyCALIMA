#!/usr/bin/env python3
"""Octopus real-time TD-DFT inputs for a PAH photo-absorption spectrum, with Malloci+07's settings by default.

Malloci, Joblin & Mulas (2007, Chem. Phys. 332, 353; method in arXiv:0707.3045, Sect. 2):
- LDA (Perdew-Zunger) with norm-conserving pseudopotentials;
- 0.3 A grid spacing;
- every atom at least 4 A from the box edge, plus a 1 A absorbing boundary;
- a delta kick, then T = 20 hbar/eV with a 0.002 hbar/eV time step (0.05 eV resolution);
- B3LYP/6-31+G* geometries.

Writes ``<out>/gs/inp`` and one ``<out>/td_<axis>/inp`` per kick direction, plus ``run.sh``. The run is the ground
state first, then the kicks, which can run at the same time, then ``oct-propagation_spectrum`` on all three in
``spectrum/`` (``cross_section_tensor``: the full tensor and its orientation average).
Open shells (odd electron counts) are run spin-polarised.

    make_run.py coronene.xyz --charge 0 --out runs/coronene_0 [--axes x y z] [--xc lda]
"""
import argparse
from pathlib import Path

VALENCE = {"C": 4, "H": 1}
XC = {"lda": "lda_x + lda_c_pz"}


def electrons(xyz, charge):
    atoms = [ln.split()[0] for ln in Path(xyz).read_text().splitlines()[2:] if len(ln.split()) == 4]
    return sum(VALENCE[a] for a in atoms) - charge


def system_block(charge, xc, spacing, radius, polarised):
    return "\n".join([
        "UnitsOutput = eV_angstrom",                 # inputs carry their units; the xyz file is in A
        'XYZCoordinates = "../geom.xyz"',
        f"ExcessCharge = {charge}",
        f"SpinComponents = {'spin_polarized' if polarised else 'unpolarized'}",
        f"XCFunctional = {XC[xc]}",
        "PseudopotentialSet = standard",
        f"Spacing = {spacing}*angstrom",
        "BoxShape = minimum",
        f"Radius = {radius}*angstrom",
    ])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("xyz", type=Path)
    ap.add_argument("--charge", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--axes", nargs="+", default=["x", "y", "z"], choices=["x", "y", "z"])
    ap.add_argument("--xc", default="lda", choices=sorted(XC))
    # 0.25 A, not Malloci's 0.3: with Octopus 16.4's standard pseudopotentials coronene's spectrum at 0.3 A has
    # 10 % too much oscillator strength to 30 eV and bands 0.2 eV blue; 0.25 and 0.2 A agree to < 1 %
    ap.add_argument("--spacing", type=float, default=0.25)
    ap.add_argument("--edge", type=float, default=4.0, help="atom to absorbing region [A]")
    ap.add_argument("--absorb", type=float, default=1.0, help="absorbing boundary width [A]")
    ap.add_argument("--time", type=float, default=20.0, help="propagation time [hbar/eV]")
    ap.add_argument("--dt", type=float, default=0.002, help="time step [hbar/eV]")
    ap.add_argument("--kick", type=float, default=0.01, help="delta kick [1/A]")
    a = ap.parse_args()

    n_el = electrons(a.xyz, a.charge)
    polarised = n_el % 2 == 1
    radius = a.edge + a.absorb
    sysb = system_block(a.charge, a.xc, a.spacing, radius, polarised)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "geom.xyz").write_text(a.xyz.read_text())
    (a.out / "gs").mkdir(exist_ok=True)
    # LCAOStart = lcao_none: with OHPC OpenBLAS 0.3.32 the LCAO guess fails in DSYGVD ("parameter 11", the
    # workspace size) on megatron; random starting orbitals converge (coronene: 26 SCF steps, 16 s on 8 ranks)
    (a.out / "gs" / "inp").write_text(f"CalculationMode = gs\n{sysb}\nConvRelDens = 1e-7\nMaximumIter = 300\n"
                                      "LCAOStart = lcao_none\n")
    for axis in a.axes:
        d = a.out / f"td_{axis}"
        d.mkdir(exist_ok=True)
        (d / "inp").write_text("\n".join([
            "CalculationMode = td", sysb,
            "TDPropagator = aetrs",
            f"TDTimeStep = {a.dt}/eV",                # hbar = 1: hbar/eV is 1/eV
            f"TDPropagationTime = {a.time}/eV",
            f"TDDeltaStrength = {a.kick}/angstrom",
            f"TDPolarizationDirection = {'xyz'.index(axis) + 1}",
            "AbsorbingBoundaries = mask",
            f"ABWidth = {a.absorb}*angstrom",
            "",
        ]))
    axes = " ".join(f"td_{x}" for x in a.axes)
    (a.out / "spectrum").mkdir(exist_ok=True)
    (a.out / "spectrum" / "inp").write_text("UnitsOutput = eV_angstrom\nPropagationSpectrumMaxEnergy = 30*eV\n"
                                            "PropagationSpectrumEnergyStep = 0.05*eV\n")
    (a.out / "run.sh").write_text(f"""#!/bin/bash
# {a.xyz.name}, charge {a.charge:+d}, {n_el} valence electrons ({'spin-polarised' if polarised else 'closed shell'})
# usage: NP=<MPI ranks per run> ./run.sh   (niced; the kicks run in parallel after the ground state)
set -euo pipefail
module load gnu15 openmpi5 openblas fftw scalapack gsl hdf5
export PATH=$HOME/opt/octopus-16.4/bin:$PATH LD_LIBRARY_PATH=$HOME/opt/deps/lib64:$HOME/opt/deps/lib:${{LD_LIBRARY_PATH:-}}
export OMP_NUM_THREADS=1
NP=${{NP:-8}}
cd "$(dirname "$0")"
(cd gs && nice -n 19 mpirun -np $NP octopus > out.log 2>&1)
for d in {axes}; do
  rm -rf $d/restart && mkdir -p $d/restart && cp -r gs/restart/gs $d/restart/gs
  (cd $d && nice -n 19 mpirun -np $NP octopus > out.log 2>&1) &
done
wait
# the kicks as multipoles.1-3, so oct-propagation_spectrum builds the tensor and its orientation average
i=1; for d in {axes}; do cp $d/td.general/multipoles spectrum/multipoles.$i; i=$((i + 1)); done
(cd spectrum && oct-propagation_spectrum > spectrum.log 2>&1)
echo RUN_OK
""")
    (a.out / "run.sh").chmod(0o755)
    print(f"{a.out}: {n_el} electrons, {'spin-polarised' if polarised else 'closed shell'}, axes {' '.join(a.axes)}")


if __name__ == "__main__":
    main()
