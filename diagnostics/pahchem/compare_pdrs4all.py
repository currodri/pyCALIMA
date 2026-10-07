#!/usr/bin/env python3
"""pahchem's PAH band ratios against the PDRs4All Orion Bar templates (M7c, first pass).

Observed: the Van De Putte+25 combined NIRSpec+MRS templates (CDS J/A+A/701/A111,
orionbar.dat; columns lambda, HII, Atomic, DF3, DF2, DF1, value/error pairs),
measured as follows:
- narrow lines removed with a running median;
- MIRI continuum: a cubic spline through the Chown+24 anchors;
- 3.3 um continuum: linear, through the Chown+24 windows;
- band fluxes: integrals over fixed windows (6.2/7.7/8.6/11.0/11.2 from Khan+26).
No published per-template mid-IR fluxes exist. The 3.4 window holds the
plateau, so 3.4/3.29 is compared with Peeters+24 Table 3 (Gaussian
decomposition) instead.

Model: the baseline families (FamilyEmission with super-H modes and RF),
measured with the same windows on the model spectrum, which has no continuum
or plateau. The local conditions are inputs, not predictions. They come from
the conditions card environments/orion_bar.toml, with sources; unstated
values are marked there. Each template is run at the two corners of its input
ranges (conditions.bracketing_environments), least and most ionising, and the
model is reported as that range.

    compare_pdrs4all.py --templates .../cds_vdp25_J_A+A_701_A111/orionbar.dat
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import median_filter

from pycalima import _paths
from pycalima.pahchem import conditions
from pycalima.pahchem.baseline import BaselineModel
from pycalima.pahchem.family_emission import FamilyEmission, load_superh_species
from pycalima.pahchem.molecule import PAHSpecies

C_UM = 2.99792458e14
ANCHORS = [5.011, 5.042, 5.073, 5.487, 5.948, 6.576, 6.785, 7.013, 9.131, 9.440, 10.191,
           10.344, 14.642, 14.913, 15.178, 15.461, 19.349, 19.632]
WINDOWS = {"3.3": (3.20, 3.36), "6.2": (6.0, 6.6), "7.7": (7.05, 8.15), "8.6": (8.15, 9.1),
           "11.0": (10.9, 11.1), "11.2": (11.1, 11.8), "12.7": (12.3, 13.1)}
RATIOS = [("3.3", "11.2"), ("6.2", "11.2"), ("7.7", "11.2"), ("8.6", "11.2"), ("11.0", "11.2"),
          ("12.7", "11.2"), ("6.2", "7.7")]
#: Peeters+24 Table 3: (3.39 + 3.40 + 3.42)/3.29, Gaussian decomposition
OBS_34_329 = {"HII": 0.11, "Atomic": 0.09, "DF1": 0.10, "DF2": 0.15, "DF3": 0.18}
TEMPLATES = tuple(conditions.load("orion_bar")["region"]["templates"])
FAMILIES = {"C54": ("C54H18", (-1, 0, 1, 2)), "C96": ("C96H24", (-1, 0, 1, 2, 3))}


def load_templates(path):
    cols = [(0, 11), (14, 29), (50, 65), (86, 101), (122, 137), (158, 173)]  # lam, HII, Atm, DF3, DF2, DF1
    rows = [[float(l[a:b]) if l[a:b].strip() else np.nan for a, b in cols] for l in open(path)]
    v = np.array(rows)
    return v[:, 0], {"HII": v[:, 1], "Atomic": v[:, 2], "DF3": v[:, 3], "DF2": v[:, 4], "DF1": v[:, 5]}


def _clean(lam, f, width_um):
    m = np.isfinite(f)
    lam, f = lam[m], f[m]
    n = max(3, int(round(width_um / np.median(np.diff(lam)))) | 1)
    return lam, median_filter(f, size=n, mode="nearest")


def _integ(lam, fnu, lo, hi):
    s = (lam >= lo) & (lam <= hi)
    return np.trapezoid(fnu[s] * C_UM / lam[s] ** 2, lam[s])


def observed_bands(lam, fnu):
    out = {}
    s = (lam > 2.95) & (lam < 3.8)
    l3, f3 = _clean(lam[s], fnu[s], 0.012)
    m = ((l3 > 3.05) & (l3 < 3.07)) | ((l3 > 3.667) & (l3 < 3.689)) | ((l3 > 3.7068) & (l3 < 3.720))
    out["3.3"] = _integ(l3, f3 - np.polyval(np.polyfit(l3[m], f3[m], 1), l3), *WINDOWS["3.3"])
    lm, fm = _clean(lam[lam < 20], fnu[lam < 20], 0.025)
    cont = CubicSpline(ANCHORS, [np.median(fm[np.abs(lm - a) < 0.004]) for a in ANCHORS])(lm)
    for b, w in WINDOWS.items():
        if b != "3.3":
            out[b] = _integ(lm, fm - cont, *w)
    return out


def model_bands(grid_cm, spectrum_per_cm):
    d = grid_cm[1] - grid_cm[0]
    return {b: float(spectrum_per_cm[(grid_cm >= 1e4 / hi) & (grid_cm <= 1e4 / lo)].sum() * d)
            for b, (lo, hi) in {**WINDOWS, "3.29": (3.20, 3.36), "3.4": (3.36, 3.44)}.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--templates", type=Path, required=True)
    ap.add_argument("--states-dir", type=Path, default=_paths.get_model_data_dir() / "PAH_states")
    args = ap.parse_args()
    lam, tmpl = load_templates(args.templates)
    obs = {t: observed_bands(lam, tmpl[t]) for t in TEMPLATES}
    grid = np.arange(500.0, 3500.0, 0.5)
    fams = {}
    for parent, (formula, charges) in FAMILIES.items():
        sp = {z: PAHSpecies.from_states_file(args.states_dir / f"{formula}_{z}.dat") for z in charges}
        sh = load_superh_species(args.states_dir / "superh", formula)
        fams[parent] = FamilyEmission(BaselineModel(parent, sp[0], anion=sp[-1]), sp, sh)
    for t in TEMPLATES:
        print(f"\n=== {t} ===")
        res = {}
        for parent, fam in fams.items():
            for corner, env in conditions.bracketing_environments("orion_bar", t).items():
                fr = fam.fractions(env)
                b = model_bands(grid, fam.spectrum(env, grid, fr)[0])
                fp = sum(v for (z, n), v in fr.items() if z > 0)
                res[(parent, corner)] = (b, fp)
                print(f"  {parent} {corner}: G0 {env.g0:.2g} n_H {env.n_h:.2g} n_e {env.n_e:.2g} "
                      f"gamma {env.g0 * np.sqrt(env.temperature_k) / env.n_e:.2g} -> cation fraction {fp:.2f}")
        print(f"  {'ratio':10s}{'observed':>10s}   " + "   ".join(f"{p} model range" for p in fams))
        for num, den in RATIOS + [("3.4", "3.29")]:
            o = OBS_34_329[t] if num == "3.4" else obs[t][num] / obs[t][den]
            cells = []
            for parent in fams:
                v = sorted(res[(parent, c)][0][num] / res[(parent, c)][0][den] for c in ("least ionising", "most ionising"))
                flag = "ok" if v[0] <= o * 1.5 and v[1] >= o / 1.5 else f"x{(v[0] / o if v[0] > o else o / v[1]):.1f}"
                cells.append(f"{v[0]:7.3f}-{v[1]:<8.3f}{flag:>6s}")
            print(f"  {num + '/' + den:10s}{o:10.3f}   " + "   ".join(cells))


if __name__ == "__main__":
    main()
