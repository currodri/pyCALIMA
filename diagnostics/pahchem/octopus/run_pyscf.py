"""PySCF hybrid ground state -> Molden file for std2, plus PySCF's own TDA as a reference (2026-10-27).

Run with the separate environment (~/opt/qc-venv), not the pyCALIMA one:
    ~/opt/qc-venv/bin/python run_pyscf.py pyrene_0.xyz --charge 0 --xc b3lyp --basis def2-svp --nstates 12
                                          [--no-df] [--max-memory 40000]
std2 needs a Cartesian GTO basis, so the molecule is built with cart = True.
"""
import argparse
import time
from pathlib import Path

import numpy as np
from pyscf import dft, gto, tddft
from pyscf.tools import molden

HA_EV = 27.211386245988


def angs_atoms(path):
    """std2 reads [Atoms] coordinates as Angstrom; PySCF writes them in bohr ('[Atoms] (AU)'). With bohr read as
    Angstrom the overlap is wrong and std2's Mulliken check fails (78.7 of 106 electrons for pyrene)."""
    bohr = 0.529177210903
    out, in_atoms = [], False
    for ln in Path(path).read_text().splitlines():
        if ln.startswith("[Atoms]"):
            out.append("[Atoms] Angs")
            in_atoms = True
            continue
        if in_atoms and ln.startswith("["):
            in_atoms = False
        if in_atoms and ln.split():
            q = ln.split()
            ln = " ".join(q[:3] + [f"{float(v) * bohr:.10f}" for v in q[3:6]])
        out.append(ln)
    Path(path).write_text("\n".join(out) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xyz", type=Path)
    ap.add_argument("--charge", type=int, default=0)
    ap.add_argument("--spin", type=int, default=None, help="2S (default: 0 for even electron counts, 1 for odd)")
    ap.add_argument("--xc", default="b3lyp")
    ap.add_argument("--basis", default="def2-svp")
    ap.add_argument("--nstates", type=int, default=0, help="PySCF TDA reference states (0: none)")
    ap.add_argument("--no-df", action="store_true",
                    help="integral-direct SCF: the density-fitting tensor of C96 and up does not fit in memory")
    ap.add_argument("--max-memory", type=float, default=4000, help="PySCF max_memory [MB]")
    a = ap.parse_args()
    lines = a.xyz.read_text().splitlines()
    atoms = "\n".join(ln for ln in lines[2:] if len(ln.split()) == 4)
    n_el = sum({"H": 1, "C": 6}[ln.split()[0]] for ln in atoms.splitlines()) - a.charge
    mol = gto.M(atom=atoms, basis=a.basis, charge=a.charge, cart=True, verbose=0,
                spin=a.spin if a.spin is not None else n_el % 2)
    mol.max_memory = a.max_memory
    mf = dft.UKS(mol) if mol.spin else dft.RKS(mol)
    mf = mf if a.no_df else mf.density_fit()
    # wB97X-D3 by its libxc name: PySCF reads a "-d3"/"_d3" suffix as the D3 dispersion correction, which does not
    # change the orbitals anyway
    mf.xc = {"wb97x_d3": "hyb_gga_xc_wb97x_d3"}.get(a.xc, a.xc)
    t0 = time.time()
    e = mf.kernel()
    tag = f"{a.xyz.stem}_{a.xc}_{a.basis}"
    print(f"{tag}: {mol.nao} basis functions, {mol.nelectron} electrons, 2S = {mol.spin}, E = {e:.6f} Ha, "
          f"converged {mf.converged}, {time.time() - t0:.0f} s")
    molden.from_scf(mf, f"{tag}.molden")
    angs_atoms(f"{tag}.molden")
    if a.nstates:
        td = tddft.TDA(mf)
        td.nstates = a.nstates
        t0 = time.time()
        td.kernel()
        f = td.oscillator_strength(gauge="length")
        with open(f"{tag}_pyscf_tda.dat", "w") as fh:
            fh.write("# state  E [eV]  f (length)\n")
            for i, (w, fi) in enumerate(zip(td.e * HA_EV, f), 1):
                fh.write(f"{i:4d} {w:10.4f} {fi:10.5f}\n")
        print(f"PySCF TDA, {a.nstates} states in {time.time() - t0:.0f} s:",
              ", ".join(f"{w:.2f} ({fi:.3f})" for w, fi in zip(td.e * HA_EV, f) if fi > 1e-3))


if __name__ == "__main__":
    main()
