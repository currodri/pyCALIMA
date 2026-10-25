#!/usr/bin/env python3
"""C+ neutralisation on PAHs: pahchem's untuned rates against Wolfire+08's calibrated model (validation V2-V3).

Wolfire+08 (ApJ 680, 384) fit Phi_PAH = 0.4 to C I/C II along diffuse sight lines, with one N_C = 35 PAH at
x_PAH = 6e-7 and their Table 2 rates (DS87 x Phi). pahchem has no Phi: its rates are DS87 capture with the
statistical sticking, mutual neutralisation at 0.58 x DS87 (Shuman+14) and 0.9 x Langevin (Canosa+95). Here the
v4.00 molecules with N_C 32-40 share x_PAH = 6e-7 at W08's CNM conditions (G0 = 1.7 Habing, n = 30 cm^-3,
T = 100 K, x(C+) = 1.4e-4, x(H+) = 3e-4, x_e = 4.5e-4), and alpha(C+) per H is compared with W08's three-state
balance at Phi = 0.4 (and 1).

alpha(C+) / W08 (Phi = 0.4), 2026-10-25: Jochims 0.91, statistical 0.73, valence 0.72 (f(-1) 0.15, 0.11, 0.11).

    check_wolfire08_cplus.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import fit_34um_sidegroups as S  # noqa: E402
from pycalima.pahchem import population as P  # noqa: E402
from pycalima.pahchem.baseline import Environment, Params  # noqa: E402

G0, N, T = 1.7, 30.0, 100.0
XC, XH = 1.4e-4, 3e-4


#: W08 Table 2 at T = 100 K, Phi = 1 [cm^3 s^-1]: (X+ + PAH-, X+ + PAH0)
W08_ION = {"C+": (2.4e-7, 8.8e-9), "H+": (8.3e-7, 3.1e-8)}


def wolfire08(phi, x=6e-7, ion="C+"):
    """(alpha(ion) per H [cm^3 s^-1], (f-, f0, f+)) of W08's Table 2 rates in a three-state balance."""
    ne, t2 = (XC + XH) * N, T / 100.0
    att, det = 1.3e-6 * phi * ne, 1.2e-8 * G0
    mn = phi * t2**-0.5 * (W08_ION["C+"][0] * XC * N + W08_ION["H+"][0] * XH * N)
    up = 4.6e-9 * G0 + phi * (W08_ION["C+"][1] * XC * N + W08_ION["H+"][1] * XH * N)
    rm, rp = att / (det + mn), up / (3.5e-5 * phi * t2**-0.5 * ne)
    f0 = 1.0 / (1.0 + rm + rp)
    k_mn, k_ct = W08_ION[ion]
    return x * phi * (k_mn * t2**-0.5 * rm * f0 + k_ct * f0), (rm * f0, f0, rp * f0)


def main():
    for phi in (0.4, 1.0):
        a, f = wolfire08(phi)
        print(f"Wolfire+08 Phi = {phi}: alpha(C+) {a:.3e} cm^3 s^-1 per H; f(-, 0, +) {np.round(f, 3)}")
    lib = [m for m in S.library() if 32 <= m.nc <= 40 and -1 in m.charges]
    ne = (XC + XH) * N
    env = Environment("cnm", G0, N, ne, T, None, ions=(("C+", XC * N), ("H+", XH * N)))
    for det in ("jochims", "statistical", "valence"):
        pop = P.Population(lib, {"all": (1.0, 1e4)}, params=Params(detachment=det))
        pop.prepare(P.field_shape(env), workers=8)
        x = np.full(len(lib), 6e-7 / len(lib))
        r = pop.gas_rates(env, x)
        a = r["alpha_mn"]["C+"] + r["alpha_ct"]["C+"]
        fz = x @ r["f"] / x.sum()
        print(f"pahchem, {len(lib)} molecules (N_C 32-40), detachment {det}: alpha(C+) {a:.3e} (mutual "
              f"neutralisation {r['alpha_mn']['C+']:.2e}, ion-neutral {r['alpha_ct']['C+']:.2e}); "
              f"f(-, 0, +, ...) {np.round(fz, 3)}; / W08(Phi = 0.4) = {a / wolfire08(0.4)[0]:.2f}")


if __name__ == "__main__":
    main()
