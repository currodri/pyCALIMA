#!/usr/bin/env python3
"""Which change moved the section-4 validations between 2026-10-26 and 2026-11-02: their key numbers, one change at a time.

The validation scripts were last run on 2026-10-25/26. Since then these changes landed, here added cumulatively
(``Params``):
1. "2026-10-25": Li & Draine's UV (``uv_cross_section="li_draine"``, the default until 2026-10-26), s = 1 on cations
   (``capture_model="bt94"``), f = 1 on neutrals (``attachment_model="constant"``), the library's charges
   (``charge_ladder="library"``);
2. "+ Malloci UV": Berne+22's Malloci average, the default from 2026-10-26 to 2026-10-31;
3. "+ Octopus UV": our TD-DFT library, the default since 2026-11-01;
4. "+ sticking": ``lab_cold`` and the size-logistic f, the defaults since 2026-11-02 (section 2.2b);
5. "+ ladder": the cold-limit ladder (section 2.2c), i.e. the current defaults. (The anion's valence threshold of
   2026-11-02 cannot be undone through Params; it moves 2 of V2's 72 molecules, C34H16 6415 and 6459.) The other
   changes since 2026-10-25 (the Wenzel yield on the cation only, associative detachment of dehydrogenated anions
   only, BASIS_VERSION 9) are in every column.

Quantities, with the scripts' own conditions and the valence detachment:
- V2 (``check_wolfire08_cplus``): alpha(C+) on the 72 N_C 32-40 molecules at W08's CNM (n = 30), over Wolfire+08 at
  Phi = 0.4; it also stands for V3 and V13, which take the same molecules and the same neutralisation;
- V14 (``check_tielens21_fz``): f(Z) and <Z> of C54H18 and C96H24 in the diffuse cloud, the NGC 7023 PDR and its
  cavity (n_H = 0);
- V9 (``check_pe_efficiency``): eps at Tielens' diffuse example (G0 1, 100 K, n_e 0.008) and Gamma at Berne+22's
  diffuse point (C54H18, G0 1.7, 80 K, n_e 0.03) over their 2.7e-26 erg s^-1 H^-1.

Result: see section 4 of the gas-coupling doc.

    check_default_changes.py [--workers 6]
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import check_pe_efficiency as E
import check_tielens21_fz as T
import check_wolfire08_cplus as W

from pycalima.pahchem import population as P
from pycalima.pahchem.baseline import Environment, Params

CONFIGS = {"2026-10-25": dict(uv_cross_section="li_draine", capture_model="bt94", attachment_model="constant",
                              charge_ladder="library"),
           "+ Malloci UV": dict(uv_cross_section="malloci_average", capture_model="bt94", attachment_model="constant",
                                charge_ladder="library"),
           "+ Octopus UV": dict(capture_model="bt94", attachment_model="constant", charge_ladder="library"),
           "+ sticking": dict(charge_ladder="library"),
           "+ ladder (now)": {}}
WINDOWS = {"all": (1.0, 1e4)}


def v2(params, workers):
    lib = [m for m in W.S.library() if 32 <= m.nc <= 40 and -1 in m.charges]
    x = np.full(len(lib), 6e-7 / len(lib))
    pop = P.Population(lib, WINDOWS, params=params, cache_dir=T.BASIS_CACHE)
    pop.prepare(P.field_shape(W.cell(W.N)), workers=workers)
    r = pop.gas_rates(W.cell(W.N), x)
    fz = x @ r["f"] / x.sum()
    return (r["alpha_mn"]["C+"] + r["alpha_ct"]["C+"]) / W.wolfire08(0.4)[0], fz[pop.zs.index(-1)]


def v14_v9(params, workers):
    mols = T.molecules()
    pop = P.Population(mols, WINDOWS, params=params, cache_dir=T.BASIS_CACHE)
    out = {}
    for name, (g0, n, t) in T.ENVIRONMENTS.items():
        env = Environment(name, g0, 0.0, T.X_E * n, t, None)
        pop.prepare(P.field_shape(env), workers=workers)
        f = pop.charge_matrix(env)
        out[name] = [(f[s, [pop.zs.index(z) for z in (-1, 0, 1)]], f[s] @ np.array(pop.zs)) for s in range(len(mols))]
    for g0 in (1.0, E.G0):
        pop.prepare(P.field_shape(Environment("eps", g0, 0.0, 1.0, 100.0, None)), workers=workers)
    eps, _, _ = E.at(pop, 1.0, 100.0, 0.008)
    _, gamma, _ = E.at(pop, E.G0, 80.0, 3e-2)
    out["eps Tielens diffuse"] = eps
    out["Berne Gamma ratio"] = gamma[0] * 0.1 * 2.7e-4 / mols[0].nc / 2.7e-26
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    warnings.simplefilter("ignore")
    res = {}
    for label, over in CONFIGS.items():
        p = Params(**over)
        res[label] = (v2(p, args.workers), v14_v9(p, args.workers))
        print(f"{label} done", flush=True)
    names = [m.name for m in T.molecules()]
    w = 16
    print("\n" + " " * 44 + "".join(f"{k:>{w}s}" for k in CONFIGS))
    print(f"{'V2 alpha(C+)/W08, N_C 32-40':44s}" + "".join(f"{res[k][0][0]:{w}.3f}" for k in CONFIGS))
    print(f"{'V2 f(-1), N_C 32-40':44s}" + "".join(f"{res[k][0][1]:{w}.3f}" for k in CONFIGS))
    for env_name in T.ENVIRONMENTS:
        for s, name in enumerate(names):
            print(f"{'V14 ' + env_name + ' ' + name + ' <Z>':44s}" + "".join(f"{res[k][1][env_name][s][1]:{w}.2f}" for k in CONFIGS))
            print(f"{'    f(-1, 0, +1)':44s}" + "".join(f"{np.array2string(res[k][1][env_name][s][0], precision=3):>{w}s}"
                                                       for k in CONFIGS))
    for s, name in enumerate(names):
        print(f"{'V9 eps, Tielens diffuse, ' + name:44s}" + "".join(f"{res[k][1]['eps Tielens diffuse'][s]:{w}.4f}" for k in CONFIGS))
    print(f"{'V9 Gamma/Berne+22, C54H18':44s}" + "".join(f"{res[k][1]['Berne Gamma ratio']:{w}.2f}" for k in CONFIGS))


if __name__ == "__main__":
    main()
