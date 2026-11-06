#!/usr/bin/env python3
"""Harmonic frequencies behind data/external_data/pahchem/demyk2026_modes.csv (pahchem.demyk2026).

Demyk+26 do not publish their B3LYP-D3/6-311G(d,p) mode lists and PAHdb has no 1-methylpyrene, so the modes
are computed at RB3LYP/<basis> (default 4-31G; no dispersion: the pyscf-dispersion wheel did not load with
numpy 2.5), geometry optimised with geomeTRIC, then scaled per class so that pyrene matches PAHdb v4.00 UID 334
(the scale factors and the check against UID 336 are in the CSV header).

Start geometries from PAHdb v4.00 ($CALIMA_PAHDB_DIR): UID 334 pyrene, UID 336 6H-pyrene; 1-methylpyrene is
pyrene with the H at position 1 (the end of a trio, next to the ring fusion) replaced by CH3 (C-C 1.50 A,
C-H 1.09 A, tetrahedral).

Needs PySCF and geomeTRIC, which are not pyCALIMA dependencies (pip install pyscf geometric). The macOS PySCF
wheel runs on one thread: pyrene ~20 min, 6H-pyrene ~30 min, 1-methylpyrene ~45 min.

    demyk2026_harmonic_modes.py <me-pyrene|pyrene|6h-pyrene> <out.json> [basis]
"""
import json
import sys
import time

import numpy as np
from pyscf import dft, gto
from pyscf.geomopt.geometric_solver import optimize
from pyscf.hessian import thermo

SYM = {1: "H", 6: "C"}


def methylate(atoms):
    """Replace the position-1 H of pyrene by CH3."""
    xyz = np.array([a[1:] for a in atoms])
    z = np.array([a[0] for a in atoms])
    C, H = np.where(z == 6)[0], np.where(z == 1)[0]
    bonded = lambda i, j, r: np.linalg.norm(xyz[i] - xyz[j]) < r      # noqa: E731
    host = {h: next(c for c in C if bonded(h, c, 1.2)) for h in H}
    ch = set(host.values())
    nb = {c: [d for d in C if d != c and bonded(c, d, 1.6)] for c in C}
    for h, c in host.items():
        ch_nb = [d for d in nb[c] if d in ch]
        if len(ch_nb) == 1 and len([d for d in nb[ch_nb[0]] if d in ch]) == 2:    # trio end
            break
    u = (xyz[h] - xyz[c]) / np.linalg.norm(xyz[h] - xyz[c])
    cm = xyz[c] + 1.50 * u
    # an in-plane perpendicular and the plane normal
    ring = np.array([xyz[d] for d in nb[c]])
    n = np.cross(ring[0] - xyz[c], ring[1] - xyz[c])
    n /= np.linalg.norm(n)
    p = np.cross(n, u)
    hs = []
    for phi in (0.0, 2 * np.pi / 3, 4 * np.pi / 3):
        d = np.cos(np.radians(70.5)) * u + np.sin(np.radians(70.5)) * (np.cos(phi) * n + np.sin(phi) * p)
        hs.append(cm + 1.09 * d)
    keep = [i for i in range(len(atoms)) if i != h]
    out = [atoms[i] for i in keep] + [(6, *cm)] + [(1, *v) for v in hs]
    return out


def pahdb_geometries(uids=(334, 336)):
    """{uid: [(Z, x, y, z), ...]} [Angstrom] from the PAHdb v4.00 XML."""
    import xml.etree.ElementTree as ET

    from pycalima.pahchem import ames_suite

    out = {}
    for _, el in ET.iterparse(ames_suite.xml_path("pahdb-theoretical-v4-00"), events=("end",)):
        if el.tag.endswith("specie"):
            if int(el.get("uid")) in uids:
                atoms = []
                for a in el.iter():
                    if a.tag.endswith("atom"):
                        g = {c.tag.split("}")[1]: c.text for c in a}
                        atoms.append((int(g["type"]), float(g["x"]), float(g["y"]), float(g["z"])))
                out[str(el.get("uid"))] = atoms
                if len(out) == len(uids):
                    break
            el.clear()
    return out


def main():
    name, out = sys.argv[1], sys.argv[2]
    geoms = pahdb_geometries()
    basis = sys.argv[3] if len(sys.argv) > 3 else "4-31g"
    atoms = {"me-pyrene": lambda: methylate(geoms["334"]), "pyrene": lambda: [tuple(a) for a in geoms["334"]],
             "6h-pyrene": lambda: [tuple(a) for a in geoms["336"]]}[name]()
    mol = gto.M(atom=[(SYM[int(a[0])], a[1:]) for a in atoms], basis=basis, unit="Angstrom", verbose=4)
    mol.max_memory = 12000
    print(f"{name}: {mol.natm} atoms, {mol.nao} basis functions", flush=True)
    mf = dft.RKS(mol)
    mf.xc = "b3lyp"
    mf.grids.level = 4
    t = time.time()
    mol_eq = optimize(mf, maxsteps=100)
    print(f"optimised in {time.time() - t:.0f} s", flush=True)
    mf = dft.RKS(mol_eq)
    mf.xc = "b3lyp"
    mf.grids.level = 4
    mf.max_memory = 12000
    mf.kernel()
    t = time.time()
    h = mf.Hessian().kernel()
    print(f"hessian in {time.time() - t:.0f} s", flush=True)
    res = thermo.harmonic_analysis(mol_eq, h)
    freqs = np.asarray(res["freq_wavenumber"])
    json.dump({"name": name, "method": f"RB3LYP/{basis}, no dispersion; PySCF " + __import__("pyscf").__version__,
               "symbols": [mol_eq.atom_symbol(i) for i in range(mol_eq.natm)],
               "xyz_angstrom": (mol_eq.atom_coords(unit="Angstrom")).tolist(),
               "energy_hartree": float(mf.e_tot), "freqs_cm_unscaled": [float(np.real(f)) for f in freqs],
               "imaginary": int(np.sum(np.iscomplex(freqs) & (np.abs(np.imag(freqs)) > 1e-6)))},
              open(out, "w"), indent=1)
    print("done", out, flush=True)


if __name__ == "__main__":
    main()
