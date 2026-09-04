"""
PAH_photochemistry.py — thin facade re-exporting the PAH photophysics subpackage.

Physics implementations live in:
  pah_mol_data.py     — vibrational modes, RRKM rates
  pah_charge_utils.py — ionisation potentials, recombination rates
  pah_temperature.py  — GD89 temperature distribution
  pah_dissociation.py — photodissociation rate integrators
  pah_radiation.py    — Kurucz radiation field loaders
"""

import os
from pathlib import Path

# Re-export public API
from pycalima.models.PAH_photophysics.pah_mol_data import (
    load_pah_modes,
    compute_thermal_energy_from_file,
    compute_thermal_ir_rate,
    compute_rrkm_dissociation_rate,
    compute_dissociation_rate_from_table,
    extract_transitions,
)
from pycalima.models.PAH_photophysics.pah_charge_utils import (
    afromNc,
    ionisation_potential_energy,
    electron_affinity_energy,
    ionisation_yield_Jochims1996,
    ionisation_yield_LePage2001,
    photoionisation_rate,
    recombination_rate_Spitzer,
    recombination_rate_Tielens21,
    attachment_rate_Carelli13,
    attachment_rate_Tielens05,
    J_function_DS87,
    recombination_rate_Bakes1994,
    se_neutral_Weingartner2001,
    se_anion_Weingartner2001,
    IONISATION_POTENTIAL,
    ELECTRON_AFFINITY,
)
from pycalima.models.PAH_photophysics.pah_temperature import (
    get_absorption_cross_section,
    compute_base_g0,
    compute_gd89_temperature_distribution,
    compute_adaptive_temperature_distribution,
    compute_bakes_temperature_distribution,
    compute_total_time_averaged_ir_rate,
)
from pycalima.models.PAH_photophysics.pah_dissociation import (
    compute_total_photon_absorption_rate,
    compute_total_photoionisation_rate,
    compute_total_dissociation_rate,
    compute_branching_integrated_rates,
    compute_andrews_direct_branching,
    compute_bakes_direct_branching,
    compute_bakes_dwek_branching,
    compare_dissociation_methods,
    print_method_comparison,
)
from pycalima.models.PAH_photophysics.pah_radiation import load_kurucz_u_E, load_kurucz_I_nu
from pycalima.models.PAH_photophysics.pah_h_state import (
    DissociationChannel,
    compute_solo_duo_counts,
    get_dissociation_channels,
    channel_params,
    HLOSS_EVEN_EACT_EV, HLOSS_EVEN_DS_JKM,
    HLOSS_ODD_EACT_EV,  HLOSS_ODD_DS_JKM,
    H2LOSS_EACT_EV,     H2LOSS_DS_JKM,
    SUPERH_HLOSS_NEUTRAL_ANION_EACT_EV, SUPERH_HLOSS_NEUTRAL_ANION_DS_JKM,
    SUPERH_HLOSS_CATION_EACT_EV,        SUPERH_HLOSS_CATION_DS_JKM,
    C54H18_NH0, C54H18_SOLO, C54H18_DUO,
    C96H24_NH0, C96H24_SOLO, C96H24_DUO,
)
from pycalima.models.PAH_photophysics.pah_hydrogen_chemistry import (
    h_addition_rate,
    collisional_rate,
    reaction_efficiency_neutral,
    h2_abstraction_rate,
    h2_abstraction_rate_coefficient,
    K_CATION_CM3S,
    K_ANION_CM3S,
    EFF_DEHYDROGENATED,
    K_ER_COEFF_CM3S,
)


_THIS_DIR          = os.path.dirname(os.path.abspath(__file__))
_CALIMA_ROOT       = os.path.abspath(os.path.join(_THIS_DIR, '..', '..'))
_EXTERNAL_DATA_DIR = os.path.join(_CALIMA_ROOT, 'external_data')


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    PAH_FILE = os.path.join(_CALIMA_ROOT, 'model_data', 'PAH_states', 'C54H18_0.dat')

    # Radiation field: Kurucz 15000 K only (per Andrews 2016)
    kurucz_I_nu  = load_kurucz_I_nu(15000)
    kurucz_u_E   = load_kurucz_u_E(15000)
    KURUCZ_BASE_G0 = compute_base_g0(kurucz_u_E)
    print(f"Kurucz 15000 K base G0 = {KURUCZ_BASE_G0:.4f}")

    # Cross sections for neutral C54H18
    a0       = afromNc(54)
    hc_ev    = 1.23984193e-4
    w, C_abs = get_absorption_cross_section(0, a0)
    E_cs     = hc_ev / w
    cross_section_table = np.column_stack([E_cs, C_abs])

    def make_field(g0):
        return lambda nu, _g=g0: (_g / KURUCZ_BASE_G0) * kurucz_I_nu(nu)

    g0_grid = np.logspace(0, 5, num=20)

    # Andrews 2016 reference data (G0-dependent digitised rates)
    H_G0_csv  = os.path.join(_EXTERNAL_DATA_DIR, 'H-loss_G0_C54H18_Andrews16.csv')
    H2_G0_csv = os.path.join(_EXTERNAL_DATA_DIR, 'H2-loss-G0_C54H18_Andrews16.csv')

    # Method B only — entirely from scratch, no Andrews lookup tables
    results = compare_dissociation_methods(
        pah_file=PAH_FILE,
        cross_section_table=cross_section_table,
        field_factory=make_field,
        g0_grid=g0_grid,
        E_act_H=4.6,  dS_H=44.8,
        E_act_H2=3.52, dS_H2=-53.1,
        num_bins=150,
        t_min=1.0,
    )

    print_method_comparison(results,
                            andrews_H_csv=H_G0_csv,
                            andrews_H2_csv=H2_G0_csv)

    # Andrews polynomial fit for reference
    fit_H  = [-14.148,  1.962, -0.031, -0.009,  0.003]
    fit_H2 = [-13.527,  1.051, -0.121,  0.060, -0.004]
    g0_smooth = np.logspace(0, 5, 200)
    lg_s      = np.log10(g0_smooth)
    H_fit  = 10**(sum(fit_H[k]  * lg_s**k for k in range(5)))
    H2_fit = 10**(sum(fit_H2[k] * lg_s**k for k in range(5)))

    H_G0_ref  = np.loadtxt(H_G0_csv,  delimiter=',')
    H2_G0_ref = np.loadtxt(H2_G0_csv, delimiter=',')

    plt.figure(figsize=(8, 5))
    plt.loglog(g0_grid, results['B_H'],  'm-.',  label='H-loss  (B: RRKM)')
    plt.loglog(g0_grid, results['B_H2'], 'c:',   label='H2-loss (B: RRKM)')
    plt.loglog(g0_smooth, H_fit,  'k-',  label='Andrews16 fit H-loss')
    plt.loglog(g0_smooth, H2_fit, 'k--', label='Andrews16 fit H2-loss')
    plt.loglog(H_G0_ref[:,0],  H_G0_ref[:,1],  'ks', markersize=5, label='Andrews16 data H-loss')
    plt.loglog(H2_G0_ref[:,0], H2_G0_ref[:,1], 'k^', markersize=5, label='Andrews16 data H2-loss')
    plt.xlabel('G0', fontsize=12)
    plt.ylabel('k [s$^{-1}$]', fontsize=12)
    plt.legend(fontsize=9)
    plt.title('C54H18 photodissociation — Method B (GD89 f(T) + RRKM k(E))')
    plt.tight_layout()
    plt.show()
