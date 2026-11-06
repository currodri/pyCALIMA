"""Full linear-response TD-B3LYP (RPA) with PySCF: the faithful hybrid reference for the weak low-lying states that std2 overshoots
(2026-10-28). B3LYP/def2-SVP, Cartesian, density fitting, the same ground state as run_pyscf.py.

    ~/opt/qc-venv/bin/python run_tddft.py geometries/HBC_0.xyz --nstates 60 [--tda]
Writes <stem>_tddft[_tda].dat: state, E [eV], f (length gauge), and the dominant orbital pair.
"""
import argparse
import time
from pathlib import Path

import numpy as np
from pyscf import dft, gto, tddft

HA_EV = 27.211386245988


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xyz", type=Path)
    ap.add_argument("--nstates", type=int, default=40)
    ap.add_argument("--tda", action="store_true")
    ap.add_argument("--basis", default="def2-svp")
    ap.add_argument("--max-memory", type=float, default=100000)
    a = ap.parse_args()
    lines = a.xyz.read_text().splitlines()
    atoms = "\n".join(ln for ln in lines[2:] if len(ln.split()) == 4)
    mol = gto.M(atom=atoms, basis=a.basis, cart=True, verbose=0, max_memory=a.max_memory)
    mf = dft.RKS(mol).density_fit()
    mf.xc = "b3lyp"
    t0 = time.time()
    mf.kernel()
    print(f"{a.xyz.stem}: {mol.nao} basis functions, E = {mf.e_tot:.6f}, converged {mf.converged}, {time.time() - t0:.0f} s", flush=True)
    td = (tddft.TDA if a.tda else tddft.TDDFT)(mf)
    td.nstates = a.nstates
    td.conv_tol = 1e-5
    t0 = time.time()
    td.kernel()
    f = td.oscillator_strength(gauge="length")
    tag = a.xyz.stem + ("_tddft_tda" if a.tda else "_tddft")
    with open(f"{tag}.dat", "w") as fh:
        fh.write("# state  E [eV]  f(length)\n")
        for i, (w, fi) in enumerate(zip(td.e * HA_EV, f), 1):
            fh.write(f"{i:4d} {w:10.4f} {fi:10.5f}\n")
    print(f"{tag}: {a.nstates} states in {time.time() - t0:.0f} s, converged {all(td.converged)}; highest {td.e[-1] * HA_EV:.2f} eV", flush=True)
    print("  lowest:", ", ".join(f"{w:.2f} ({fi:.3f})" for w, fi in list(zip(td.e * HA_EV, f))[:8]), flush=True)


if __name__ == "__main__":
    main()
