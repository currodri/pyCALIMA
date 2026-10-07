#!/usr/bin/env python3
"""Why pahchem's population differs from Maragkoudakis+26: their choices swapped for ours, one at a time.

Every variant runs on the same observed templates (MIRI v2 additive), on the
cube-like grid of check_maragkoudakis2026, through pyPAHdb's unweighted NNLS.
Fractions are shares of the emitted energy, as pyPAHdb's are
(``maragkoudakis2026.emission_shares``). Variants:

- R0: their matrix (PAHdb v4.00-alpha, 7 eV thermal cascade, Gaussian).
- R1: R0 without dications and trications.
- R2: pahchem-built 7 eV thermal Gaussian matrix of final PAHdb v4.00 (all
  6-31G*), same selection rule (pure C/H, 20 < N_C, H-bearing, no CH2/CH3).
- R3: as R2 but for pahchem's library (Maragkoudakis+20 v3.20, 133 molecules,
  4-31G; 81 anions, no dications).
- R4: R3 with pahchem's exact cascade (7 eV).
- R5: R3 with pahchem's heating (exact cascade under the Kurucz 40 000 K field).
- R6: R5 with pahchem's band profile (Lorentzian, -15 cm^-1): pahchem's
  emission, with the charge fitted freely.
- R7: R0 with the Chown+24 spline continuum instead of pybaselines.
- D1: R6's library and emission fitted on band ratios (5-15 um windows of
  fit_pdrs4all_population), charge still free (every state a column).
- D2: D1 with the charge set by the conditions (least-ionising corner) and the
  H-loss filter: pahchem's population fit, on the pybaselines continuum.
- D3: D2 on the Chown+24 spline continuum (pahchem's earlier set-up, without 3.3).

    explore_maragkoudakis2026.py --templates .../PDRs4All_templates_MIRI_MRS_v2_additive_stitching.dat --pylibs DIR
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

import check_maragkoudakis2026 as CK
import compare_pdrs4all as C
from pycalima import _paths
from pycalima.pahchem import ames_pahdb, maragkoudakis2026 as M, population

KEYS = ("f_neut", "f_cat", "f_an", "f_dication_plus", "f_sm", "f_med", "f_lar", "n_c")
CONTINUUM = (0.8, 3, "mor_then_imodpoly")        # best setting of check_maragkoudakis2026


def spline_continuum(lam, f):
    """The Chown+24 anchor spline (anchors inside the cut spectrum; the fit stops at 15 um)."""
    anchors = [a for a in C.ANCHORS if a < lam.max() - 0.01]
    return CubicSpline(anchors, [np.median(f[np.abs(lam - a) < 0.004]) for a in anchors])(lam)


def run(name, pre, obs_by_template, out):
    integ = np.trapezoid(pre["matrix"], pre["abscissa"], axis=0)
    rows = {}
    for t, (lam, y) in obs_by_template.items():
        w, fit = M.decompose(lam, y, pre)
        fr = M.emission_shares(w, integ, pre["charge"], pre["size"])
        fr["error"] = M.fit_error(lam, y, fit)
        rows[t] = fr
    out[name] = rows
    print(f"\n{name}")
    print(f"  {'':8s}" + "".join(f"{k:>8s}" for k in KEYS) + "   error")
    for t, fr in rows.items():
        print(f"  {t:8s}" + "".join(f"{fr[k]:8.2f}" for k in KEYS) + f"   {fr['error']:.2f}")
    sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--templates", type=Path, required=True)
    ap.add_argument("--pylibs", type=Path, default=None)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--library", choices=("m20", "v400"), default="m20",
                    help="pahchem library for R3-R6 and D1-D3: Maragkoudakis+20 (v3.20) or PAHdb v4.00")
    args = ap.parse_args()
    if args.pylibs:
        sys.path.append(str(args.pylibs))
    lam, tmpl = CK.load_v2(args.templates)
    s = (lam > 4.95) & (lam < 15.5)
    lam = lam[s]
    step = np.median(np.diff(lam))
    grid = CK.cube_grid()
    obs, obs_spline = {}, {}
    for t, f in tmpl.items():
        f = CK.remove_lines(lam, f[s])
        base = M.baseline_pybaselines(lam, f, int(CONTINUUM[0] / step), CONTINUUM[1], CONTINUUM[2])
        obs[t] = (grid, np.interp(grid, lam, f - base))
        obs_spline[t] = (grid, np.interp(grid, lam, f - spline_continuum(lam, f)))
    nu = np.sort(1e4 / grid)
    want = lambda k: args.only is None or k in args.only
    out = {}
    t1 = M.table1().set_index("zone")
    print("Table 1:")
    for t in CK.TEMPLATES:
        z = t1.loc[CK.TABLE1_ZONE[t]]
        print(f"  {t:8s}" + "".join(f"{z[k]:8.2f}" if k in z else f"{'':8s}" for k in KEYS))
    pre = M.load_precomputed()
    if want("R0"):
        run("R0 their matrix", pre, obs, out)
    if want("R1"):
        k = pre["charge"] <= 1
        run("R1 their matrix, no dications", {**pre, "matrix": pre["matrix"][:, k], "charge": pre["charge"][k],
                                               "size": pre["size"][k]}, obs, out)
    if want("R7"):
        run("R7 their matrix, Chown+24 spline continuum", pre, obs_spline, out)
    del pre
    if want("R2"):
        t0 = time.time()
        recs = ames_pahdb.xml_index(ames_pahdb.xml_path(), _paths.get_model_data_dir() / "pahchem_pahdb_index")
        sel = [r for r in recs if r["pure_ch"] and r["nc"] > 20 and r["nh"] > 0 and r["n_ch2"] == 0 and r["n_ch3"] == 0]
        sp = [ames_pahdb.species_from_record(r, f"v4.00 {r['uid']}") for r in sel]
        pre2 = M.matrix_from_species(sp, nu, window_cm=100.0)
        print(f"(R2 matrix: {len(sp)} species in {time.time() - t0:.0f}s)")
        run("R2 final v4.00 (6-31G*), thermal 7 eV, Gaussian", pre2, obs, out)
        del pre2, recs, sp
    lib = (population.library_m20(all_anions=True) if args.library == "m20" else
           population.library_v400(cache_dir=_paths.get_model_data_dir() / "pahchem_pahdb_index"))
    print(f"\n(pahchem library: {args.library}, {len(lib)} molecules, {sum(len(m.charges) for m in lib)} states)")
    states = [(m, z) for m in lib for z in m.charges]
    sp = [m.species[z] for m, z in states]
    if want("R3"):
        run(f"R3 pahchem library ({args.library}), thermal 7 eV, Gaussian",
            M.matrix_from_species(sp, nu, window_cm=100.0), obs, out)
    if want("R4"):
        run("R4 ... exact cascade 7 eV", M.matrix_from_species(sp, nu, model="exact_7ev", window_cm=100.0), obs, out)
    if want("R5") or want("R6"):
        pop = population.Population(lib, {"x": (5.0, 15.0)}, cache_dir=_paths.get_model_data_dir() / "pahchem_population_basis")
        shape = ("kurucz", 40000)
        pop.prepare(shape)
        bmap = {id(m.species[z]): pop.basis(m, z, shape) for m, z in states}
        basis = lambda s: bmap[id(s)]
        if want("R5"):
            run("R5 ... exact cascade, Kurucz-field heating", M.matrix_from_species(sp, nu, model="field", basis=basis,
                                                                                   window_cm=100.0), obs, out)
        if want("R6"):
            run("R6 ... and Lorentzian, -15 cm^-1 (pahchem's emission, free charge)",
                M.matrix_from_species(sp, nu, model="field", basis=basis, profile="lorentzian", shift_cm=-15.0,
                                      window_cm=300.0), obs, out)

    if any(want(k) for k in ("D1", "D2", "D3")):
        import fit_pdrs4all_population as FP
        from pycalima.pahchem import conditions

        bands = {b: w for b, w in FP.WINDOWS.items() if b != "3.3"}
        shape = ("kurucz", 40000)
        pop = population.Population(lib, bands, cache_dir=_paths.get_model_data_dir() / "pahchem_population_basis")
        _, _, B = pop._stack(shape)
        absorbed = np.zeros(B.shape[:2])
        for i, mol in enumerate(lib):
            for z in mol.charges:
                absorbed[i, z - pop.zmin] = pop.basis(mol, z, shape).absorbed
        zgrid = np.array(pop.zs)[None, :] * np.ones((len(lib), 1), int)
        ncgrid = pop.nc[:, None] * np.ones((1, len(pop.zs)))
        ratios = {t: M.band_ratios(*obs[t], bands) for t in obs}
        ratios_spline = {t: M.band_ratios(*obs_spline[t], bands) for t in obs_spline}

        def report(name, rows):
            out[name] = rows
            print(f"\n{name}")
            print(f"  {'':8s}" + "".join(f"{k:>8s}" for k in KEYS) + "    chi2")
            for t, fr in rows.items():
                print(f"  {t:8s}" + "".join(f"{fr[k]:8.2f}" for k in KEYS) + f"   {fr['chi2']:5.1f}")
            sys.stdout.flush()

        def shares(e):
            e = np.where(pop.present, e, 0.0)
            return M.emission_shares(e[pop.present], np.ones(int(pop.present.sum())), zgrid[pop.present],
                                     ncgrid[pop.present])

        if want("D1"):
            rows = {}
            for t, r in ratios.items():
                Fs = B[pop.present]                          # every charge state a free column
                res = population.fit_abundances(Fs, r, "11.2")
                e = np.zeros(B.shape[:2])
                e[pop.present] = res["weights"] * absorbed[pop.present]
                rows[t] = {**shares(e), "chi2": res["chi2"]}
            report("D1 pahchem library and emission, band ratios (5-15 um), free charge", rows)
        for key, rr, label in (("D2", ratios, "pybaselines"), ("D3", ratios_spline, "Chown+24 spline")):
            if not want(key):
                continue
            rows = {}
            for t, r in rr.items():
                envs = conditions.bracketing_environments("orion_bar", t)
                env = envs["least ionising"]
                keep = pop.intact_fractions(envs["most ionising"]) >= 0.5
                F, _ = pop.band_fluxes(env)
                f = pop.charge_matrix(env)
                res = population.fit_abundances(F[keep], r, "11.2")
                w = np.zeros(len(lib))
                w[keep] = res["weights"]
                rows[t] = {**shares(w[:, None] * f * absorbed), "chi2": res["chi2"]}
            report(f"{key} ... charge from the conditions (least-ionising corner), {label} continuum", rows)

if __name__ == "__main__":
    main()
