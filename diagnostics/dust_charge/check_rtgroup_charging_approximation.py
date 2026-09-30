#!/usr/bin/env python3
"""How good is grain charging from RT photon groups? Benchmark against the full spectrum.

RAMSES-RT knows, per cell, only the photon density N_g of each group and, per
run, one mean energy egy_g per group. This script takes known spectra, bins
them into groups (default: the 8 CALIMA RTZ groups) and compares, for every
dust bin:

  full    pyCALIMA equilibrium_charge_for_grain on the true spectrum (WDB06);
  groups  the RT-group solver (rt_group_charging) with the run's global egy_g
          (a blackbody at --t-source, as RAMSES sets them), i.e. what RAMSES
          will compute;
  ideal   the same solver given the true per-cell group mean energies (what a
          scheme that also tracked per-cell group energies could reach);
  G0only  the current table path: Mathis ISRF scaled by G0, cut at 13.6 eV.

Reported: grain potential error [V], mean-charge error, and relative errors
in photoelectric heating Gamma and recombination cooling Lambda.
"""
import argparse
import json
import warnings

import numpy as np

from pycalima import _paths
from pycalima.models.dust_charge import rt_group_charging as rg
from pycalima.models.dust_charge.dust_charging import equilibrium_charge_for_grain
from pycalima.models.grain_size_config import get_bins, get_lognormal_parameters, get_model_data_dir

C = 2.99792458e10
EV = 1.602176634e-12
KB_EV = 8.617333262e-5
U_HAB = 5.33e-14
M_P = 1.6726219e-24
E = np.geomspace(0.1, 1000.0, 8000)


def sig_H(E):
    return np.where(E >= 13.6, 6.3e-18 * (E / 13.6) ** -3, 0.0)


def bb(T):
    return E ** 3 / np.expm1(np.minimum(E / (KB_EV * T), 700.0))


def sazonov(E):
    return np.where(E < 10, 0.0798 * E ** -0.6,
                    np.where(E < 2e3, E ** -1.7 * np.exp(E / 2e3), 2.94e-3 * E ** -0.8 * np.exp(-E / 2e5)))


SPECTRA = {
    'BB 3e4 K': lambda: bb(3e4),
    'BB 5e4 K': lambda: bb(5e4),
    'BB 1e5 K': lambda: bb(1e5),
    'BB 2e5 K': lambda: bb(2e5),
    'BB 5e4 K, N_HI=1e18': lambda: bb(5e4) * np.exp(-1e18 * sig_H(E)),
    'BB 1e5 K, N_HI=1e18': lambda: bb(1e5) * np.exp(-1e18 * sig_H(E)),
    'BB 1e5 K, N_HI=1e19': lambda: bb(1e5) * np.exp(-1e19 * sig_H(E)),
    'quasar (Sazonov+04)': lambda: sazonov(E),
}
GAS = {'HII': (100.0, 8e3), 'WNM': (0.03, 6e3), 'hot': (1.0, 1e6)}


def groups_of(uE, L0, L1):
    N, egy = [], []
    for l0, l1 in zip(L0, L1):
        m = (E >= l0) & (E < l1)
        n = np.trapezoid(uE[m] / (E[m] * EV), E[m])
        N.append(n)
        egy.append(np.trapezoid(uE[m], E[m]) / EV / n if n > 0 else 0.5 * (l0 + l1))
    return np.array(N), np.array(egy)


def load_or_build(bin_id, material, a, L0, L1):
    path = get_model_data_dir() / 'dust_charging_rtgroups_data' / f'dust_charging_rtgroups_{bin_id}.dat'
    if path.exists():
        tab = rg.read_bin_table(path)
        if np.allclose(tab['L0'], L0) and np.allclose(tab['L1'], L1) and np.isclose(tab['a'], a):
            return tab
    print(f'  building table for {bin_id} (no matching export at {path})', flush=True)
    return rg.build_bin_table(material, a, L0, L1, label=bin_id)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--t-source', type=float, default=1e5,
                    help='blackbody that sets the run-wide group energies (rtz_source_temperature)')
    ap.add_argument('--g0', type=float, nargs='+', default=[1e2, 1e4], help='Habing G (6-13.6 eV)')
    ap.add_argument('--gas', nargs='+', default=list(GAS), choices=list(GAS))
    ap.add_argument('--group-L0', type=float, nargs='+', default=list(rg.DEFAULT_GROUP_L0))
    ap.add_argument('--group-L1', type=float, nargs='+', default=list(rg.DEFAULT_GROUP_L1))
    args = ap.parse_args()
    warnings.filterwarnings('ignore')

    L0, L1 = np.array(args.group_L0), np.array(args.group_L1)
    _, egy_run = groups_of(bb(args.t_source), L0, L1)
    bins = []
    for b in sorted(get_bins(is_pah=False), key=lambda b: b['index']):
        a = float(get_lognormal_parameters(b['id'], model_name='basic')['a0']) * 1e-4
        mat = 'silicate' if b['composition'].lower() == 'silicate' else 'graphite'
        bins.append((b['id'], mat, a, load_or_build(b['id'], mat, a, L0, L1)))

    rows = []
    for sname, sfun in SPECTRA.items():
        shape = sfun()
        for G0 in args.g0:
            b = (E >= 6) & (E <= 13.6)
            uE = shape * G0 * U_HAB / np.trapezoid(shape[b], E[b])
            N, egy_true = groups_of(uE, L0, L1)
            field = np.column_stack([E, 1239.84193 / E, C * uE])
            for gname in args.gas:
                ne, T = GAS[gname]
                ions = [(ne, 1.0, M_P)]
                for bin_id, mat, a, tab in bins:
                    full = equilibrium_charge_for_grain(
                        1.0, ne, T, mat, a, rad_field=field,
                        ion_species=[{'n': ne, 'T': T, 'm': M_P * 1e-3, 'z': 1}],
                        yield_params={'yield_model': 'WDB06', 'include_ion_transmission': False})
                    g0only = equilibrium_charge_for_grain(
                        G0, ne, T, mat, a, radiation_model='Mathis',
                        ion_species=[{'n': ne, 'T': T, 'm': M_P * 1e-3, 'z': 1}],
                        yield_params={'yield_model': 'WDB06', 'include_ion_transmission': False})
                    res = {'full': (full[3], full[2]['Gamma_total'], full[2]['Recomb_total']),
                           'G0only': (g0only[3], g0only[2]['Gamma_total'], g0only[2]['Recomb_total'])}
                    for key, egy in (('groups', egy_run), ('ideal', egy_true)):
                        r = rg.GroupChargingTable(tab, egy=egy).solve(N, C, T, ne, ions)
                        res[key] = (r['Zmean'], r['Gamma'], r['Lambda'])
                    row = dict(spectrum=sname, G0=G0, gas=gname, bin=bin_id, material=mat, a=a, **res)
                    rows.append(row)
                    Zf, Gf, Lf = res['full']
                    phi = lambda Z: Z * rg.E2_EV_CM / a
                    print(f"{sname:22s} G0={G0:6.0e} {gname:3s} {bin_id}: phi_full={phi(Zf):8.2f} V | dphi "
                          + " ".join(f"{k}={phi(res[k][0]) - phi(Zf):+7.2f}" for k in ('groups', 'ideal', 'G0only'))
                          + " | dGamma " + " ".join(f"{k}={res[k][1] / Gf - 1 if Gf else 0:+6.2f}"
                                                    for k in ('groups', 'ideal', 'G0only')), flush=True)

    out = _paths.get_results_dir() / 'rtgroup_charging_benchmark.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(out, 'w'), indent=1, default=float)
    print(f'\nSaved {out}\n')
    print(f"{'method':8s} {'|dphi| [V] median / 95% / max':>32s} {'|dGamma/Gamma| median / 95% / max':>36s}"
          f" {'|dLambda/Lambda| median / max':>30s}")
    for k in ('groups', 'ideal', 'G0only'):
        dphi = np.array([abs(r[k][0] - r['full'][0]) * rg.E2_EV_CM / r['a'] for r in rows])
        dg = np.array([abs(r[k][1] / r['full'][1] - 1) for r in rows if r['full'][1] > 0])
        dl = np.array([abs(r[k][2] / r['full'][2] - 1) for r in rows if r['full'][2] != 0])
        print(f"{k:8s} {np.median(dphi):10.3f} {np.percentile(dphi, 95):9.3f} {dphi.max():9.3f}"
              f" {np.median(dg):12.3f} {np.percentile(dg, 95):9.3f} {dg.max():9.3f}"
              f" {np.median(dl):12.3f} {dl.max():9.3f}")


if __name__ == '__main__':
    main()
