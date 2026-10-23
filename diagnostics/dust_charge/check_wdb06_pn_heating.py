#!/usr/bin/env python3
"""Photoelectric heating in a model planetary nebula: compare with WDB06 Table 2.

WDB06 sec. 9: blackbody with T_c = 2e5 K and G = 2e3 (G = u(6-13.6 eV) / u_Hab,
u_Hab = 5.33e-14 erg cm^-3), fully ionised H with n_H = 1e3 cm^-3, T = 1e4 K.
Carbonaceous or silicate grains alone, with the Weingartner & Draine (2001a)
size distributions (their Table 1). Table 2 gives Gamma_pe/(G n_H) and
Lambda/(G n_H) in 1e-25 erg s^-1.

Per-grain rates are solved once per material on a size grid (cached next to
the output) and integrated over each distribution. ``--models`` also runs the
WD01 yields: WDB06 find WD01 overestimates the heating by 1.1-2 (carbonaceous)
and 1.9-2.5 (silicate), the cooling by up to 28% and 1.3-1.5.
"""
import argparse
import warnings

import numpy as np
from scipy.special import erf

from pycalima import _paths
from pycalima.models.dust_charge.dust_charging import equilibrium_charge_for_grain

C_CGS = 2.99792458e10
EV2ERG = 1.602176634e-12
KB_EV = 8.617333262e-5
U_HAB = 5.33e-14
G_PN, NH_PN, T_PN, TC_PN = 2e3, 1e3, 1e4, 2e5
A_GRID = np.geomspace(3.5e-8, 1e-3, 60)  # cm

# WD01a Table 1 rows used in WDB06 Table 2: (R_V, 1e5 b_C, case) ->
# (alpha_g, beta_g, a_tg [um], a_cg [um], C_g, alpha_s, beta_s, a_ts [um], C_s); a_cs = 0.1 um.
WD01A = {
    (3.1, 0.0, 'A'): (-2.25, -0.0648, 0.00745, 0.606, 9.94e-11, -1.48, -9.34, 0.172, 1.02e-12),
    (3.1, 6.0, 'A'): (-1.54, -0.165, 0.0107, 0.428, 9.99e-12, -2.21, 0.300, 0.164, 1.00e-13),
    (4.0, 0.0, 'A'): (-2.26, -0.199, 0.0241, 0.861, 5.47e-12, -2.03, 0.668, 0.189, 5.20e-14),
    (4.0, 4.0, 'A'): (-1.64, -0.247, 0.0152, 0.536, 5.83e-12, -2.09, 1.58, 0.183, 3.94e-14),
    (5.5, 0.0, 'A'): (-2.35, -0.668, 0.148, 1.96, 4.82e-14, -1.57, 1.10, 0.198, 4.24e-14),
    (5.5, 3.0, 'A'): (-1.61, -0.722, 0.0418, 0.720, 7.58e-13, -1.59, 2.12, 0.193, 3.20e-14),
    (4.0, 0.0, 'B'): (-2.62, -0.0144, 0.0187, 5.74, 6.46e-12, -2.01, 0.894, 0.198, 4.95e-14),
    (4.0, 4.0, 'B'): (-1.96, -0.813, 0.0693, 3.48, 2.95e-13, -2.11, 2.10, 0.198, 3.13e-14),
    (5.5, 0.0, 'B'): (-2.80, 0.0356, 0.0203, 3.43, 2.74e-12, -1.09, -0.370, 0.218, 1.17e-13),
    (5.5, 3.0, 'B'): (-1.90, -0.0517, 0.0120, 7.28, 2.86e-12, -1.13, -0.109, 0.211, 1.04e-13),
}

# WDB06 Table 2: (Gamma_pe, Lambda) / (G n_H) in 1e-25 erg s^-1
TABLE2 = {
    ('graphite', 3.1, 0.0, 'A'): (16.2, 1.97), ('graphite', 3.1, 6.0, 'A'): (45.9, 5.56),
    ('graphite', 4.0, 0.0, 'A'): (12.8, 1.55), ('graphite', 4.0, 4.0, 'A'): (31.6, 3.82),
    ('graphite', 5.5, 0.0, 'A'): (9.75, 1.19), ('graphite', 5.5, 3.0, 'A'): (23.4, 2.84),
    ('graphite', 4.0, 0.0, 'B'): (16.7, 2.05), ('graphite', 4.0, 4.0, 'B'): (32.0, 3.89),
    ('graphite', 5.5, 0.0, 'B'): (13.3, 1.64), ('graphite', 5.5, 3.0, 'B'): (22.6, 2.75),
    ('silicate', 3.1, 0.0, 'A'): (15.6, 1.29), ('silicate', 3.1, 6.0, 'A'): (22.5, 1.89),
    ('silicate', 4.0, 0.0, 'A'): (9.84, 0.818), ('silicate', 4.0, 4.0, 'A'): (9.31, 0.775),
    ('silicate', 5.5, 0.0, 'A'): (3.83, 0.314), ('silicate', 5.5, 3.0, 'A'): (3.61, 0.297),
    ('silicate', 4.0, 0.0, 'B'): (10.1, 0.840), ('silicate', 4.0, 4.0, 'B'): (9.73, 0.810),
    ('silicate', 5.5, 0.0, 'B'): (3.87, 0.316), ('silicate', 5.5, 3.0, 'B'): (3.84, 0.314),
}


def _F(a, beta, at):
    return 1.0 + beta * a / at if beta >= 0 else 1.0 / (1.0 - beta * a / at)


def _cutoff(a, at, ac):
    return np.where(a < at, 1.0, np.exp(-((a - at) / ac) ** 3))


def vsg_lognormal(a, bC):
    """WD01a eqs. 2-3: very small carbonaceous grains, per H nucleus per cm."""
    if bC == 0:
        return np.zeros_like(a)
    mC, rho, sigma = 12.011 * 1.66053907e-24, 2.24, 0.4
    out = np.zeros_like(a)
    for frac, a0 in ((0.75, 3.5e-8), (0.25, 30e-8)):
        B = (3.0 / (2 * np.pi) ** 1.5 * np.exp(-4.5 * sigma ** 2) / (rho * a0 ** 3 * sigma)
             * frac * bC * mC / (1 + erf(3 * sigma / np.sqrt(2) + np.log(a0 / 3.5e-8) / (sigma * np.sqrt(2)))))
        out += B / a * np.exp(-0.5 * (np.log(a / a0) / sigma) ** 2)
    return out


def dn_da(a, material, key):
    """(1/n_H) dn/da [cm^-1] of WD01a eqs. 4-6, a in cm."""
    ag, bg, atg, acg, Cg, as_, bs, ats, Cs = WD01A[key]
    if material == 'graphite':
        at, ac = atg * 1e-4, acg * 1e-4
        return vsg_lognormal(a, key[1] * 1e-5) + Cg / a * (a / at) ** ag * _F(a, bg, at) * _cutoff(a, at, ac)
    at, ac = ats * 1e-4, 0.1e-4
    return Cs / a * (a / at) ** as_ * _F(a, bs, at) * _cutoff(a, at, ac)


def pn_field(E_min=1.0, E_max=3e3, n=1500):
    """[E_eV, lambda_nm, I_E = c u_E] for the WDB06 sec. 9 blackbody at G = 2e3."""
    E = np.geomspace(E_min, E_max, n)
    kT = KB_EV * TC_PN
    shape = E ** 3 / np.expm1(E / kT)  # u_E up to a constant
    band = (E >= 6.0) & (E <= 13.6)
    norm = G_PN * U_HAB / np.trapezoid(shape[band], E[band])
    return np.column_stack([E, 1239.84193 / E, C_CGS * norm * shape])


def per_grain_rates(material, yield_model):
    field = pn_field()
    ions = [{'n': NH_PN, 'T': T_PN, 'm': 1.6726219e-27, 'z': 1}]
    G, L = np.zeros(A_GRID.size), np.zeros(A_GRID.size)
    for i, a in enumerate(A_GRID):
        _, _, rates, Zm, _ = equilibrium_charge_for_grain(
            1.0, NH_PN, T_PN, material, a, rad_field=field, ion_species=ions,
            yield_params={'yield_model': yield_model})
        G[i] = rates['Gamma_total']
        L[i] = rates['Recomb_total']
        print(f'  {yield_model} {material} a={a:.2e} cm: Zmean={Zm:.1f} '
              f'Gamma={G[i]:.3e} Lambda={L[i]:.3e} erg/s', flush=True)
    return G, L


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--models', nargs='+', default=['WDB06'], choices=['WDB06', 'WD01'])
    ap.add_argument('--recompute', action='store_true')
    args = ap.parse_args()
    warnings.filterwarnings('ignore')

    out_dir = _paths.get_plots_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    res = {}
    for model in args.models:
        for mat in ('graphite', 'silicate'):
            cache = out_dir / f'wdb06_pn_rates_{model}_{mat}.npz'
            if cache.exists() and not args.recompute:
                d = np.load(cache)
                res[model, mat] = (d['Gamma'], d['Lambda'])
            else:
                res[model, mat] = per_grain_rates(mat, model)
                np.savez(cache, a=A_GRID, Gamma=res[model, mat][0], Lambda=res[model, mat][1])

    unit = G_PN * 1e-25
    lnA = np.log(A_GRID)
    print('\nGamma_pe/(G n_H), Lambda/(G n_H) [1e-25 erg/s]: ours (ratio to WDB06 Table 2)')
    hdr = ' '.join(f'{m:>26s}' for m in args.models)
    print(f'{"composition":9s} {"R_V":>4s} {"1e5bC":>5s} case | {"WDB06 Table 2":>15s} | {hdr}')
    for (mat, Rv, bC, case), (Gp, Lp) in TABLE2.items():
        w = dn_da(A_GRID, mat, (Rv, bC, case)) * A_GRID  # per ln a
        cols = []
        for model in args.models:
            G, L = res[model, mat]
            g = np.trapezoid(w * G, lnA) / unit
            l = np.trapezoid(w * L, lnA) / unit
            cols.append(f'{g:6.2f} ({g / Gp:4.2f}) {l:5.2f} ({l / Lp:4.2f})')
        print(f'{mat:9s} {Rv:4.1f} {bC:5.1f}  {case}   | {Gp:6.2f} {Lp:6.3f} | ' + ' '.join(cols))


if __name__ == '__main__':
    main()
