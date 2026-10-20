#!/usr/bin/env python3
"""Can recurrent fluorescence of PAH cations make the Extended Red Emission?

For each parent family and environment, the baseline network gives f(Z, N_H)
and each state's multiphoton steady state its RF photon rates by level
(baseline.RF_SPECTRA). This prints the ERE observables:
- the ERE photon efficiency: RF photons at 540-900 nm per absorbed heating photon,
  against Gordon+98 (>= 10 % in the diffuse ISM) and Smith & Witt 2002
  (IC 63 4.7 %, NGC 7023 0.2-0.6 %, Orion 0.4-1.3 %; falling ~100x with U);
- ERE/IR power, against Berne+08 (NGC 7023 N 1.6 %, S 0.35 %; <~ 6 %);
- blue (350-500 nm) to ERE photons (none observed; Witt & Vijh 2004).

``--hydrogenated-only`` gives RF only to states with N_H >= N_H0. Dehydrogenated
cations otherwise take the parent cation's levels, which is the model's weakest
assumption (pahchem-spec.md, M7b ERE test).
"""
import argparse
from pathlib import Path

import numpy as np

from pycalima import _paths
from pycalima.pahchem import dos, emission, photo
from pycalima.pahchem.baseline import E_PH, NGC7023_EDGE, ORION_BAR_ATOMIC, BaselineModel, Environment
from pycalima.pahchem.family_emission import FamilyEmission
from pycalima.pahchem.molecule import PAHSpecies

ERE_EV = (1239.84 / 900.0, 1239.84 / 540.0)
BLUE_EV = (1239.84 / 500.0, 1239.84 / 350.0)
ENVIRONMENTS = [Environment("diffuse ISM", 1.0, 30.0, 0.03, 100.0, teff_k=None),
                Environment("IC 63-like (G0=650)", 650.0, 1e4, 1.0, 200.0, teff_k=30000),
                NGC7023_EDGE, ORION_BAR_ATOMIC]
FAMILIES = {"C24": ("C24H12", (-1, 0, 1)), "C54": ("C54H18", (-1, 0, 1, 2))}


def ere_observables(fam, env, hydrogenated_only=False):
    model, field = fam.model, env.field()
    n_abs = n_ere = n_blue = p_ir = p_ere = 0.0
    for (z, nh), w in fam.fractions(env).items():
        if w < 1e-8:
            continue
        _, cm = fam._cascade(z, nh)
        _, h = model._absorption(field, photo.sigma_li_draine(model.nc, z, E_PH), z, nh)
        r = emission.steady_state_emission(cm, cm.steady_state(E_PH, h, {}))
        n_abs += w * np.trapezoid(h, E_PH)
        p_ir += w * cm.freqs_cm[:cm.n_ir] @ r[:cm.n_ir]
        e = cm.freqs_cm[cm.n_ir:] / dos.CM_PER_EV
        rl = r[cm.n_ir:] * (nh >= model.nh0 or not hydrogenated_only)
        m = (e >= ERE_EV[0]) & (e <= ERE_EV[1])
        n_ere += w * rl[m].sum()
        p_ere += w * e[m] @ rl[m] * dos.CM_PER_EV
        n_blue += w * rl[(e >= BLUE_EV[0]) & (e <= BLUE_EV[1])].sum()
    return {"eta": n_ere / n_abs, "ere_over_ir": p_ere / p_ir, "blue_over_ere": n_blue / n_ere if n_ere else np.nan}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--states-dir", type=Path, default=_paths.get_model_data_dir() / "PAH_states")
    ap.add_argument("--hydrogenated-only", action="store_true")
    args = ap.parse_args()
    for parent, (formula, charges) in FAMILIES.items():
        sp = {z: PAHSpecies.from_states_file(args.states_dir / f"{formula}_{z}.dat") for z in charges}
        fam = FamilyEmission(BaselineModel(parent, sp[0], anion=sp[-1]), sp)
        for env in ENVIRONMENTS:
            o = ere_observables(fam, env, args.hydrogenated_only)
            print(f"{parent:4s} {env.name:28s} eta {o['eta']:8.4f}  ERE/IR {o['ere_over_ir']:7.4f}  "
                  f"blue/ERE {o['blue_over_ere']:.1e}")


if __name__ == "__main__":
    main()
