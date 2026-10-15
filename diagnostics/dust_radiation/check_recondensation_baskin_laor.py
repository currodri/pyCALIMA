#!/usr/bin/env python3
"""Recondensation: does the solver reproduce Baskin & Laor (2018) T_sub(n)?

With ``dust_sublimation`` and ``dust_condensation`` both on, a grain at T_d
loses atoms at the GD89 rate in the sublimation tables and regains them from
the gas at the kinetic flux of the key species X (C for graphite, Si here for
silicate). The two cancel at a saturated density n_sat(T_d); a grain survives
only if the gas holds less than that, so with all of X in the gas
(n_X = f_X n) the sublimation temperature becomes a function of density.
Baskin & Laor (2018, MNRAS 474, 1970; BL18) write this as eq. 27,

    n_X k T = 6e11 T_sub^0.5 exp(-T0 / T_sub),   T = 1e4 K,

fitted by eqs. 32-33, T_sub = 81200 / (66.003 - ln n) for graphite and
68100 / (67.957 - ln n) for silicate, with f_C = 10^-3.44, f_Si = 10^-4.45.
Eq. 27 balances pressures; the solver's default balances fluxes,
n_X v(T_gas) = n_sat v(T_d), and ``condensation_balance = 'pressure'``
switches to BL18's condition. Both are shown.

T_sub(n) from the solver is the T_d at which compute_rhs gives zero net dust
growth with all of X in the gas. Closed-box RK54 runs to equilibrium then
show the dust surviving below that temperature and gone above it.

The timescale is set against BL18 eq. 36, t_sub = N/(dN/dt) =
2.8 a_0.1 / n_10 days, which takes the condensation flux as n v 4 pi a^2
(sticking 1, no factor 1/4 for the one-sided flux).

The sublimation tables are written to a temporary directory straight from
``dust_sublimation.radius_loss_rate``, in the format the exporter uses.

Usage
-----
    python diagnostics/dust_radiation/check_recondensation_baskin_laor.py
"""
import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from pycalima import _paths
from pycalima.models.dust_radiation import dust_sublimation as ds
from pycalima.solvers.dust_init import load_initial_conditions
from pycalima.solvers.ode_driver import integrate_dust_ode
from pycalima.solvers.rhs import build_process_list, compute_rhs
from pycalima.solvers.rk54 import RK54Solver

KB = 1.380649e-16
AMU = 1.66053906660e-24
MH = 1.6726219e-24
DAY = 86400.0
A_UM = 0.1
T_GAS = 1.0e4
COLORS = {'graphite': '#2a78d6', 'silicate': '#eb6834'}

# BL18 abundances (number per H) of the key species. Silicate is Mg2SiO4 (the
# GD89 monomer); Mg and O are set in excess so that Si is the key species,
# as BL18 assume.
F_KEY = {'graphite': 10 ** -3.44, 'silicate': 10 ** -4.45}
BINS = {
    'graphite': {'elements': {'C': 1.0}, 'rho': 2.24,
                 'abund': {'C': F_KEY['graphite']}},
    'silicate': {'elements': {'Mg': 2 * 24.305 / 140.69, 'Si': 28.086 / 140.69,
                              'O': 4 * 15.999 / 140.69}, 'rho': 3.5,
                 'abund': {'Mg': 3 * F_KEY['silicate'], 'Si': F_KEY['silicate'],
                           'O': 10 * F_KEY['silicate']}},
}
MASS_AMU = {'C': 12.011, 'Mg': 24.305, 'Si': 28.086, 'O': 15.999}
KEY = {'graphite': ('C', 2), 'silicate': ('Si', 6)}   # name, index in y_gas


# ---------------------------------------------------------------------------
# Baskin & Laor 2018
# ---------------------------------------------------------------------------

def bl18_tsub_fit(material, n):
    """BL18 eqs. 32-33."""
    T0, c = {'graphite': (81200.0, 66.003), 'silicate': (68100.0, 67.957)}[material]
    return T0 / (c - np.log(n))


def bl18_tsub_eq27(material, n):
    """Direct solution of BL18 eq. 27 at T = 1e4 K."""
    T0 = {'graphite': 81200.0, 'silicate': 68100.0}[material]
    nx = F_KEY[material] * n
    return brentq(lambda T: np.log(nx * KB * T_GAS) - np.log(6e11 * T ** 0.5) + T0 / T,
                  300.0, 6000.0)


def bl18_tsub_days(n):
    """BL18 eq. 36 for a = 0.1 um graphite."""
    return 2.8 * (A_UM / 0.1) / (n / 1e10)


# ---------------------------------------------------------------------------
# solver set-up
# ---------------------------------------------------------------------------

def write_tables(model_data):
    out = model_data / 'dust_sublimation'
    out.mkdir(parents=True, exist_ok=True)
    a_cm = A_UM * 1e-4
    T = np.linspace(800.0, 3500.0, 2701)
    for i, material in enumerate(BINS, start=1):
        eps = np.array([ds.radius_loss_rate(material, a_cm, t) / a_cm for t in T])
        keep = eps > 1e-60
        rows = [f'# Dust bin: DustBin_0{i}, Material: {material}, Grain radius '
                f'(representative): {a_cm:.6e} cm',
                '# Columns: T_d [K] (dust temperature) | log10(epsilon/[s^-1])']
        rows += [f'{t:14.6E} {np.log10(e):14.6E}' for t, e in zip(T[keep], eps[keep])]
        (out / f'sublimation_rate_DustBin_0{i}.dat').write_text('\n'.join(rows) + '\n')


def load_state(model_data, material, n, T_d, balance='flux', T_gas=T_GAS,
               dust_frac=1e-6):
    """One bin of *material*, a fraction *dust_frac* of the key species in
    dust and the rest in the gas, with sublimation and condensation on."""
    b = BINS[material]
    bin_id = f'DustBin_0{list(BINS).index(material) + 1}'
    m_per_H = 1.4 * MH
    abund = {el: x * MASS_AMU[el] * AMU / m_per_H for el, x in b['abund'].items()}
    key, _ = KEY[material]
    # grain material carrying the whole key-species budget
    rho_dust_max = abund[key] * 1.4 * n * MH / b['elements'][key]
    cfg = {
        'model_data_dir': str(model_data),
        'environment': {'gas_temperature_K': T_gas,
                        'hydrogen_number_density_cm3': n,
                        'dust_temperature_K': T_d},
        'elemental_abundances': {el: {'mass_fraction': x} for el, x in abund.items()},
        'dust_bins': [{'id': bin_id, 'composition': material,
                       'grain_size_micron': A_UM, 'grain_density_gcm3': b['rho'],
                       'elements': [{'name': el, 'mass_fraction': f}
                                    for el, f in b['elements'].items()],
                       'initial_mass_density_gcm3': dust_frac * rho_dust_max}],
        'physics': {'dust_sublimation': True, 'dust_condensation': True},
        'models': {'condensation_balance': balance},
    }
    path = model_data.parent / 'recondensation_ic.json'
    path.write_text(json.dumps(cfg))
    return load_initial_conditions(path)


def solver_tsub(model_data, material, n, balance='flux', T_gas=T_GAS):
    """T_d of zero net growth with (almost) all of the key species in the gas."""
    state, y_gas, y_dust = load_state(model_data, material, n, 1500.0, balance, T_gas)
    procs = build_process_list(state)
    db = state.dust_bins[0]

    def net(T_d):
        db.dust_temperature_K = T_d
        _, ddust = compute_rhs(state, y_gas, y_dust, procs)
        return ddust[0] / y_dust[0]

    return brentq(lambda T: np.arcsinh(net(T) * 1e12), 800.0, 3400.0, xtol=0.01)


def mass_efolding_days(model_data, material, n, T_d):
    """1 / (3 epsilon): the time a grain at T_d takes to lose 1/e of its mass
    in vacuum (and to regain it at n_sat)."""
    state, _, _ = load_state(model_data, material, n, T_d)
    return 1.0 / (3.0 * float(state.dust_bins[0].erosion_rate_interp(T_d))) / DAY


def closed_box(model_data, material, n, T_d, t_end_s, balance='flux'):
    """Equilibrium fraction of the key species in dust, starting from 99%."""
    state, y_gas, y_dust = load_state(model_data, material, n, T_d, balance,
                                      dust_frac=0.99)
    _, iX = KEY[material]
    f_X = state.dust_bins[0].el_mfractions[state.dust_bins[0].el_indices.index(iX)]
    total = y_gas[iX] + f_X * y_dust[0]
    _, y_d, _ = integrate_dust_ode(state, t_end_s, y_gas, y_dust,
                                   build_process_list(state), RK54Solver(errmax=0.01),
                                   h_init=1.0, h_min=1e-6)
    return f_X * y_d[0] / total


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------

def print_comparison(model_data):
    print(f'Sublimation temperature vs density, a = {A_UM} um, T_gas = {T_GAS:.0e} K')
    print(f'{"":10s}{"n [cm^-3]":>10s}{"BL18 eq27":>11s}{"BL18 fit":>10s}'
          f'{"pressure":>10s}{"flux":>8s}{"flux, T_gas=T_d":>17s}')
    for material in BINS:
        for n in (1e2, 1e5, 1e8, 1e11, 1e14):
            p = solver_tsub(model_data, material, n, 'pressure')
            f = solver_tsub(model_data, material, n, 'flux')
            # flux balance with the gas at the grain temperature
            iso = brentq(lambda T: solver_tsub(model_data, material, n, 'flux', T) - T,
                         800.0, 3400.0, xtol=0.1)
            print(f'{material:10s}{n:10.0e}{bl18_tsub_eq27(material, n):11.0f}'
                  f'{bl18_tsub_fit(material, n):10.0f}{p:10.0f}{f:8.0f}{iso:17.0f}')

    print(f'\nGraphite mass e-folding time at T_sub(n), a = {A_UM} um:')
    print(f'{"n [cm^-3]":>10s}{"BL18 eq36":>12s}{"flux":>10s}{"pressure":>10s}'
          f'{"flux/BL18":>11s}')
    for n in (1e8, 1e10, 1e12):
        tf = mass_efolding_days(model_data, 'graphite', n,
                                solver_tsub(model_data, 'graphite', n, 'flux'))
        tp = mass_efolding_days(model_data, 'graphite', n,
                                solver_tsub(model_data, 'graphite', n, 'pressure'))
        tb = bl18_tsub_days(n)
        print(f'{n:10.0e}{tb:10.3g} d{tf:8.3g} d{tp:8.3g} d{tf / tb:11.1f}')
    print('  (BL18 take the flux as n v 4 pi a^2 with sticking 1; the kinetic flux is '
          'n v pi a^2 and GD89 use alpha = 0.1, so 4/alpha = 40x longer.)')


def plot(model_data, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, (ax_t, ax_f) = plt.subplots(1, 2, figsize=(11.0, 4.6))

    n_grid = np.logspace(0, 14, 29)
    for material, color in COLORS.items():
        ax_t.plot(n_grid, [bl18_tsub_eq27(material, n) for n in n_grid],
                  color='#52514e', lw=1, ls=':')
        ax_t.plot(n_grid, [solver_tsub(model_data, material, n, 'pressure')
                           for n in n_grid], color=color, lw=1.5, ls='--')
        ax_t.plot(n_grid, [solver_tsub(model_data, material, n, 'flux')
                           for n in n_grid], color=color, lw=2, label=material)
    ax_t.plot([], [], color='#52514e', lw=2, label='pyCALIMA, flux balance')
    ax_t.plot([], [], color='#52514e', lw=1.5, ls='--',
              label='pyCALIMA, pressure balance')
    ax_t.plot([], [], color='#52514e', lw=1, ls=':', label='Baskin & Laor 18 eq. 27')
    ax_t.set(xscale='log', xlabel=r'$n_{\rm H}$ [cm$^{-3}$]',
             ylabel=r'$T_{\rm sub}$ [K]  ($T_{\rm gas}=10^4$ K, $a=0.1\,\mu$m)')
    ax_t.legend(frameon=False, fontsize=8)
    ax_t.text(3e7, 1830, 'graphite: eq. 27 lies under\nthe pressure-balance curve',
              fontsize=8, color='#52514e', ha='right')

    for n, ls in ((1e8, '-'), (1e11, '--')):
        T_sub = solver_tsub(model_data, 'graphite', n, 'flux')
        T_grid = np.linspace(T_sub - 250.0, T_sub + 40.0, 30)
        # the gas relaxes to n_sat on about the e-folding time at T_sub
        t_end = 30.0 * mass_efolding_days(model_data, 'graphite', n, T_sub) * DAY
        frac = [closed_box(model_data, 'graphite', n, T, t_end) for T in T_grid]
        ax_f.plot(T_grid, frac, color=COLORS['graphite'], lw=2, ls=ls,
                  label=rf'$n_{{\rm H}}=10^{{{np.log10(n):.0f}}}$ cm$^{{-3}}$')
        # analytic: n_sat(T_d) / n_sat(T_sub) = epsilon(T_d) / epsilon(T_sub)
        a_cm = A_UM * 1e-4
        eps = np.array([ds.radius_loss_rate('graphite', a_cm, T) for T in T_grid])
        eps_sub = ds.radius_loss_rate('graphite', a_cm, T_sub)
        ax_f.plot(T_grid, np.clip(1.0 - eps / eps_sub, 0.0, None), color='#52514e',
                  lw=0.8, ls=':', label='analytic' if n == 1e8 else None)
        ax_f.axvline(T_sub, color='#52514e', lw=1, ls=ls)
    ax_f.set(xlabel=r'$T_d$ [K]', ylabel='fraction of C in graphite at equilibrium',
             ylim=(-0.03, 1.03))
    ax_f.legend(frameon=False, fontsize=8, loc='center', bbox_to_anchor=(0.6, 0.45),
                title='closed box, RK54\nstart: 99% of C in dust\n'
                      'vertical: solver $T_{\\rm sub}(n)$',
                title_fontsize=8)
    for ax in (ax_t, ax_f):
        ax.grid(alpha=0.25, which='major')
        ax.tick_params(which='both', direction='in')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print('Saved', out_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        model_data = Path(tmp) / 'model_data'
        write_tables(model_data)
        print_comparison(model_data)
        if not args.no_plot:
            out_dir = _paths.get_plots_dir()
            out_dir.mkdir(parents=True, exist_ok=True)
            plot(model_data, out_dir / 'recondensation_baskin_laor.png')


if __name__ == '__main__':
    main()
