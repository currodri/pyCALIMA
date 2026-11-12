#!/usr/bin/env python3
"""One job of the Delta-SCF IP/EA ladders (``make_ladder.py``): optimise a (state, charge, spin), then single points.

PySCF B3LYP (Gaussian's VWN3 flavour, "b3lypg", as PAHdb's Gaussian runs) / 6-31G*, PAHdb v4.00's level, with density
fitting; geometry by geomeTRIC from the state's start geometry. With ``--diffuse`` (anions and neutrals: the EA pairs),
a 6-31+G* single point at the optimised geometry. Writes ``<state>/q<z>_s<2S>.json`` (energies [hartree], <S^2>,
convergence, timings) and the optimised (or last reached) geometry ``.xyz``; skips a job whose json says the optimisation
and the final SCF converged, and restarts any other from its ``.xyz`` if there is one. PYSCF_TMPDIR sets where density
fitting spills out of core (node-local /tmp filled up with four C96 jobs on a node).

Usage: run_state.py ROOT REL Z TWO_S DIFFUSE [--threads N] [--maxsteps N] [--max-memory MB] [--sp-only]
"""
import argparse
import json
import time
from pathlib import Path


def build_mf(mol, two_s, basis, guess_dm=None):
    from pyscf import dft

    mol.basis, mol.spin = basis, two_s
    mol.build()
    mf = (dft.RKS(mol) if two_s == 0 else dft.UKS(mol)).density_fit()
    mf.xc = "b3lypg"
    mf.conv_tol, mf.max_cycle = 1e-9, 200
    mf.grids.level = 3
    return mf


def scf(mf, dm0=None):
    mf.kernel(dm0)
    if not mf.converged:                       # second try: level shift, then damping, from the last density
        mf.level_shift, mf.damp = 0.3, 0.3
        mf.max_cycle = 400
        mf.kernel(mf.make_rdm1())
        mf.level_shift, mf.damp = 0.0, 0.0
        if mf.converged:                       # polish without the shift
            mf.kernel(mf.make_rdm1())
    if not mf.converged:                       # third: second-order (Newton) SCF, from the last density
        mf = mf.newton()
        mf.kernel(mf.make_rdm1())
    return mf


def s2(mf):
    try:
        return float(mf.spin_square()[0])
    except Exception:                          # RKS
        return 0.0


def read_xyz(path):
    """PySCF atom string from an xyz file (its comment line may hold anything)."""
    lines = Path(path).read_text().splitlines()
    return "; ".join(ln.strip() for ln in lines[2:2 + int(lines[0])])


def optimise(mf, maxsteps, conv):
    """geomeTRIC from mf's geometry: (converged, last molecule, error or None, steps). PySCF's wrapper logs, not raises,
    a run out of steps (its kernel returns the flag); an SCF failure raises, and the callback keeps the last geometry."""
    from pyscf.geomopt.geometric_solver import kernel as geomopt

    last = {"mol": mf.mol, "steps": 0}

    def keep(env):
        last["mol"], last["steps"] = env["mol"].copy(), last["steps"] + 1

    try:
        converged, mol_eq = geomopt(mf, maxsteps=maxsteps, callback=keep, **conv)
        return bool(converged), mol_eq, None, last["steps"]
    except Exception as err:
        return False, last["mol"], repr(err)[:400], last["steps"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root", type=Path)
    ap.add_argument("rel")
    ap.add_argument("z", type=int)
    ap.add_argument("two_s", type=int)
    ap.add_argument("diffuse", type=int)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--maxsteps", type=int, default=150)
    ap.add_argument("--max-memory", type=int, default=12000, help="PySCF max_memory [MB]; density fitting goes out of core above")
    ap.add_argument("--sp-only", action="store_true",
                    help="only the 6-31+G* single point, at the optimised .xyz of a converged job, added to its json")
    args = ap.parse_args()

    from pyscf import gto, lib

    lib.num_threads(args.threads)
    d = args.root / args.rel
    tag = f"q{args.z:+d}_s{args.two_s}"
    out = d / f"{tag}.json"
    if args.sp_only:
        rec = json.loads(out.read_text())
        if not (rec.get("opt_converged") and rec.get("scf_converged")):
            raise SystemExit(f"{args.rel} {tag}: not converged, no single point")
        if rec.get("scf_converged_631pgs"):
            print(f"{args.rel} {tag}: single point done")
            return
        mol = gto.M(atom=read_xyz(d / f"{tag}.xyz"), charge=args.z, spin=args.two_s, basis="6-31+g*", unit="Angstrom",
                    verbose=3, max_memory=args.max_memory)
        mol.output = str(d / f"{tag}_sp631pgs.log")
        mol.build()
        t1 = time.time()
        mfd = scf(build_mf(mol, args.two_s, "6-31+g*"))
        rec.update(e_tot_631pgs=float(mfd.e_tot), scf_converged_631pgs=bool(mfd.converged), s2_631pgs=s2(mfd),
                   t_sp_s=time.time() - t1)
        out.write_text(json.dumps(rec, indent=1))
        print(f"{args.rel} {tag}: E(6-31+G*) = {rec['e_tot_631pgs']:.8f}, {rec['t_sp_s'] / 3600:.2f} h")
        return
    if out.exists():
        old = json.loads(out.read_text())
        if old.get("opt_converged") and old.get("scf_converged"):
            print(f"{args.rel} {tag}: done")
            return
    # a rerun starts where the last run stopped (its .xyz: the last geometry reached), else from the state's start
    start = d / f"{tag}.xyz" if (d / f"{tag}.xyz").exists() else d / "start.xyz"
    rec = {"state": args.rel, "charge": args.z, "two_s": args.two_s, "xc": "b3lypg", "basis": "6-31g*", "df": True,
           "start": start.name}
    t0 = time.time()
    mol = gto.M(atom=read_xyz(start), charge=args.z, spin=args.two_s, basis="6-31g*", unit="Angstrom", verbose=3,
                max_memory=args.max_memory)
    mol.output = str(d / f"{tag}.log")
    mol.build()
    mf = scf(build_mf(mol, args.two_s, "6-31g*"))
    rec["start_scf_converged"] = bool(mf.converged)
    conv = {"convergence_energy": 1e-6, "convergence_grms": 3e-4, "convergence_gmax": 4.5e-4,
            "convergence_drms": 1.2e-3, "convergence_dmax": 1.8e-3}
    converged, mol_eq, err, steps = optimise(mf, args.maxsteps, conv)
    if err is not None and steps < args.maxsteps:
        # an SCF failure along the path (the scanner raises on one unconverged step; these stall near convergence,
        # near-degenerate sigma/pi occupations, which a level shift does not cure): once more from the last geometry
        # reached, with a second-order (Newton) SCF in the scanner
        rec["opt_error_first"], rec["opt_steps_first"] = err, steps
        mf = scf(build_mf(mol_eq, args.two_s, "6-31g*"))
        if not hasattr(mf, "undo_soscf"):
            mf = mf.newton()
        converged, mol_eq, err, more = optimise(mf, args.maxsteps - steps, conv)
        steps += more
    rec["opt_converged"], rec["opt_steps"] = converged, steps
    if err is not None:
        rec["opt_error"] = err
    mf = scf(build_mf(mol_eq, args.two_s, "6-31g*"))
    rec.update(e_tot=float(mf.e_tot), scf_converged=bool(mf.converged), s2=s2(mf), t_opt_s=time.time() - t0)
    with open(d / f"{tag}.xyz", "w") as f:
        xyz = mol_eq.atom_coords(unit="Angstrom")
        f.write(f"{mol_eq.natm}\n{args.rel} {tag} B3LYP/6-31G* optimised, E = {mf.e_tot:.10f}\n")
        for i in range(mol_eq.natm):
            f.write(f"{mol_eq.atom_symbol(i)} {xyz[i, 0]:.6f} {xyz[i, 1]:.6f} {xyz[i, 2]:.6f}\n")
    if args.diffuse:
        t1 = time.time()
        mfd = scf(build_mf(mol_eq.copy(), args.two_s, "6-31+g*"))
        rec.update(e_tot_631pgs=float(mfd.e_tot), scf_converged_631pgs=bool(mfd.converged), s2_631pgs=s2(mfd),
                   t_sp_s=time.time() - t1)
    out.write_text(json.dumps(rec, indent=1))
    print(f"{args.rel} {tag}: E = {rec['e_tot']:.8f}, opt {rec['opt_converged']}, {rec['t_opt_s'] / 3600:.2f} h")


if __name__ == "__main__":
    main()
