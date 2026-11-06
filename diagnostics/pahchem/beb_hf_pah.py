#!/usr/bin/env python3
"""BEB electron-impact ionisation cross sections of PAHs from HF orbitals (Kim & Rudd 1994).

The reference for gate G2 (pahchem.impact electron-impact ionisation). For each
molecule, the geometry is ideal and planar (C-C 1.40 A, C-H 1.085 A), built from
hexagonal ring centres. Each occupied RHF orbital gives B_i = -eps_i (Koopmans),
U_i = <phi_i|T|phi_i> and N_i = 2:

    sigma_i = S/(t+u+1) [ln t/2 (1 - 1/t^2) + 1 - 1/t - ln t/(t+1)],
    S = 4 pi a0^2 N (R/B)^2, t = T/B, u = U/B.

Checks against the literature (cc-pVDZ, peaks): anthracene 28.5 A^2 and pyrene
31.6 A^2, against Krishnadas+24 BEB(HF) 28.1 and 31.0 (Table 4). The OVGF-based
values of Appl. Sci. 15, 9562 (Table 2) are 12-15 % higher. Basis sensitivity:
6-31G and cc-pVDZ agree to 0.1 % (anthracene, coronene).

Needs PySCF, which is not a pycalima dependency; run it in its own environment:
  python diagnostics/pahchem/beb_hf_pah.py results/pahchem/beb_hf cc-pvdz anthracene coronene
"""
import sys
import time
from pathlib import Path

import numpy as np
from pyscf import gto, scf

HARTREE_EV = 27.211386
R_EV = 13.605693
A0_SQ_A2 = 0.529177**2
D_CC, D_CH = 1.40, 1.085

RINGS = {
    "naphthalene": [(i, 0) for i in range(2)],
    "anthracene": [(i, 0) for i in range(3)],
    "tetracene": [(i, 0) for i in range(4)],
    "pentacene": [(i, 0) for i in range(5)],
    "hexacene": [(i, 0) for i in range(6)],
    "pyrene": [(0, 0), (1, 0), (0, 1), (1, -1)],
    "coronene": [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)],
    "circumcoronene": [(i, j) for i in range(-2, 3) for j in range(-2, 3) if max(abs(i), abs(j), abs(i + j)) <= 2],
}
FORMULA = {"naphthalene": (10, 8), "anthracene": (14, 10), "tetracene": (18, 12), "pentacene": (22, 14),
           "hexacene": (26, 16), "pyrene": (16, 10), "coronene": (24, 12), "circumcoronene": (54, 18)}


def geometry(rings):
    a1 = np.array([np.sqrt(3) * D_CC, 0.0])
    a2 = np.array([np.sqrt(3) / 2 * D_CC, 1.5 * D_CC])
    ang = np.radians(30 + 60 * np.arange(6))
    verts = []
    for i, j in rings:
        c = i * a1 + j * a2
        for a in ang:
            p = c + D_CC * np.array([np.cos(a), np.sin(a)])
            if not any(np.linalg.norm(p - q) < 0.1 for q in verts):
                verts.append(p)
    C = np.array(verts)
    H = []
    for k, p in enumerate(C):
        nb = [q for m, q in enumerate(C) if m != k and np.linalg.norm(p - q) < 1.5]
        if len(nb) == 2:
            v = p - np.mean(nb, axis=0)
            H.append(p + D_CH * v / np.linalg.norm(v))
    return C, np.array(H)


def beb(T, B, U, N):
    T = np.asarray(T, float)[:, None]
    t, u = T / B, U / B
    S = 4 * np.pi * A0_SQ_A2 * N * (R_EV / B) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        lt = np.log(t)
        s = S / (t + u + 1) * (0.5 * lt * (1 - 1 / t**2) + 1 - 1 / t - lt / (t + 1))
    return np.where(t > 1, s, 0.0).sum(axis=1)


def main():
    out, basis, names = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
    out.mkdir(parents=True, exist_ok=True)
    T = np.geomspace(5.0, 5000.0, 400)
    for name in names:
        C, H = geometry(RINGS[name])
        assert (len(C), len(H)) == FORMULA[name], (name, len(C), len(H))
        mol = gto.M(atom=[("C", tuple(p) + (0.0,)) for p in C] + [("H", tuple(p) + (0.0,)) for p in H],
                    basis=basis, verbose=0, max_memory=6000)
        t0 = time.time()
        mf = scf.RHF(mol)
        mf.direct_scf = True
        mf.kernel()
        assert mf.converged, name
        occ = mf.mo_occ > 0
        Cmo = mf.mo_coeff[:, occ]
        B = -mf.mo_energy[occ] * HARTREE_EV
        U = np.einsum("pi,pq,qi->i", Cmo, mol.intor("int1e_kin"), Cmo) * HARTREE_EV
        s = beb(T, B, U, 2.0)
        head = f"BEB(HF/{basis}) {name} C{len(C)}H{len(H)}, ideal planar geometry; pycalima diagnostics/pahchem/beb_hf_pah.py"
        np.savetxt(out / f"{name}_{basis}_sigma.csv", np.column_stack([T, s]), delimiter=",", fmt="%.6g",
                   header=f"{head}\nT_eV,sigma_ion_A2")
        np.savetxt(out / f"{name}_{basis}_orbitals.csv", np.column_stack([B, U, np.full(B.size, 2.0)]),
                   delimiter=",", fmt="%.6g", header=f"{head}\nB_eV,U_eV,N")
        print(f"{name:15s} {basis}: nbf {mol.nao} HOMO B {B.min():.2f} eV, sigma_max {s.max():.2f} A2 "
              f"at {T[s.argmax()]:.1f} eV; sigma(1 keV) {np.interp(1e3, T, s):.2f}, "
              f"(5 keV) {np.interp(5e3, T, s):.2f}  [{time.time() - t0:.0f} s]", flush=True)


if __name__ == "__main__":
    main()
