#!/usr/bin/env python3
"""
TEST SCRIPT FOR GRAIN SIZE DISTRIBUTIONS AND ION RECOMBINATION FITTING

This script implements the dust grain size distribution from Weingartner & Draine (2001)
for graphite and silicate populations, scales them to the total grain volumes (Vg, Vs),
plots the size distributions, and evaluates the analytical ion-grain recombination
rate coefficient (alpha) fitting formula using Table 2 parameters.

By: Curro Rodriguez Montero (currodri@gmail.com)
"""

import os
import sys

# Ensure parent directories are on path so 'models' package can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.special import erf

# Set up matplotlib style for professional publications
import seaborn as sns
sns.set_theme(style="white")
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "Computer Modern Roman",
})

# 1. Physics and Material Parameters
M_C = 12 * 1.66053892e-24   # mass of C atom in g
RHO_GRA = 2.24              # graphite density in g/cm3
BC_TOTAL = 6.0e-5           # total carbon abundance in very small grains
BC1 = 0.75 * BC_TOTAL
BC2 = 0.25 * BC_TOTAL
SIGMA = 0.4
A_MIN = 3.5e-8              # cm (3.5 Angstroms)
A01 = 3.5e-8                # cm (3.5 Angstroms)
A02 = 30.0e-8               # cm (30 Angstroms)

# Graphite Size Distribution Parameters (WD01 Rv=3.1, bc = 6e-5 model)
ALPHA_G = -1.54
BETA_G = -0.165
AT_G = 0.0107 * 1e-4        # cm (0.0107 microns)
AC_G = 0.428 * 1e-4         # cm (0.428 microns)
C_G_RAW = 9.99e-12

# Silicate Size Distribution Parameters
ALPHA_S = -2.21
BETA_S = 0.300
AT_S = 0.164 * 1e-4         # cm (0.164 microns)
AC_S = 0.1 * 1e-4         # cm (0.001 microns)
C_S_RAW = 1.0e-13

# Volume Targets
VG_TARGET = 1.092 * 2.07e-27  # cm3/H
VS_TARGET = 1.322 * 2.98e-27  # cm3/H

# Table 2: Analytical Recombination Fitting Coefficients (C0 to C6)
# alpha = 1e-14 * C0 / (1 + C1 * phi**C2 * (1 + C3 * T**C4 * phi**(-C5 - C6 * lnT)))
TABLE_2_COEFFS = {
    'H+':   [12.25, 8.074e-6, 1.378, 5.087e2, 1.586e-2, 0.4723, 1.102e-5],
    'He+':  [5.572, 3.185e-7, 1.512, 5.115e3, 3.903e-7, 0.4956, 5.494e-7],
    'C+':   [45.58, 6.089e-3, 1.128, 4.331e2, 4.845e-2, 0.8120, 1.333e-4],
    'Na+':  [2.178, 1.732e-7, 2.133, 1.029e4, 1.859e-6, 1.0341, 3.223e-5],
    'Mg+':  [2.510, 8.116e-8, 1.864, 6.170e4, 2.169e-6, 0.9605, 7.232e-5],
    'Si+':  [2.166, 5.678e-8, 1.874, 4.375e4, 1.635e-6, 0.8964, 7.538e-5],
    'S+':   [3.064, 7.769e-5, 1.319, 1.087e2, 3.475e-1, 0.4790, 4.689e-2],
    'K+':   [1.596, 1.907e-7, 2.123, 8.138e3, 1.530e-5, 1.0380, 4.550e-5],
    'Ca+':  [1.636, 8.208e-9, 2.289, 1.254e5, 1.349e-9, 1.1506, 7.204e-4],
    'Mn+':  [2.029, 1.433e-6, 1.673, 1.403e4, 1.865e-6, 0.9358, 4.339e-9],
    'Fe+':  [1.701, 9.554e-8, 1.851, 5.763e4, 4.116e-8, 0.9456, 2.198e-5],
    'Ca++': [8.270, 2.051e-4, 1.252, 1.590e2, 6.072e-2, 0.5980, 4.497e-7]
}

# 2. Grain Size Distribution Functions
def compute_Bi(bci, a0i):
    """
    Computes normalisation constant Bi for the lognormal part D(a)
    corresponding to Carbon abundance bci in small grains (PAHs).
    Formula from Weingartner & Draine (2001) Appendix A.
    """
    erf_arg = (3.0 * SIGMA / np.sqrt(2.0)) + np.log(a0i / A_MIN) / SIGMA * np.sqrt(2.0)
    erf_val = erf(erf_arg)
    num = 3.0 * bci * M_C * np.exp(-4.5 * SIGMA**2)
    den = (2.0 * np.pi)**1.5 * (a0i**3) * RHO_GRA * SIGMA * (1.0 + erf_val)
    return num / den

B1 = compute_Bi(BC1, A01)
B2 = compute_Bi(BC2, A02)

def F_envelope(a, beta, at):
    """Shape factor function F(a, beta, at) for size distributions."""
    if beta >= 0:
        return 1.0 + beta * (a / at)
    else:
        return 1.0 / (1.0 - beta * (a / at))

def graphite_dn_da_unscaled(a):
    """Unscaled graphite grain size distribution 1/nH * dn_gra/da in cm^-1."""
    if a < A_MIN:
        return 0.0
    
    # Lognormal part D(a)
    D_a = 0.0
    for Bi, a0i in [(B1, A01), (B2, A02)]:
        D_a += (Bi / a) * np.exp(-0.5 * (np.log(a / a0i) / SIGMA)**2)
    
    # Power-law part with cutoff
    if a < AT_G:
        M = 1.0
    else:
        M = np.exp(-((a - AT_G) / AC_G)**3)
        
    F = F_envelope(a, BETA_G, AT_G)
    PL_a = (C_G_RAW / a) * (a / AT_G)**ALPHA_G * F * M
    
    return D_a + PL_a

def silicate_dn_da_unscaled(a):
    """Unscaled silicate grain size distribution 1/nH * dn_sil/da in cm^-1."""
    if a < A_MIN:
        return 0.0
        
    if a < AT_S:
        M = 1.0
    else:
        M = np.exp(-((a - AT_S) / AC_S)**3)
        
    F = F_envelope(a, BETA_S, AT_S)
    PL_a = (C_S_RAW / a) * (a / AT_S)**ALPHA_S * F * M
    
    return PL_a

# 3. Volume Calculations and Scaling
def graphite_vol_integrand(a):
    return (4.0 / 3.0) * np.pi * (a**3) * graphite_dn_da_unscaled(a)

def silicate_vol_integrand(a):
    return (4.0 / 3.0) * np.pi * (a**3) * silicate_dn_da_unscaled(a)

# Integrate from a_min to 10 microns (1e-3 cm)
A_MAX = 10.0 * 1e-4

Vg_raw, _ = quad(graphite_vol_integrand, A_MIN, A_MAX)
Vs_raw, _ = quad(silicate_vol_integrand, A_MIN, A_MAX)

SCALE_G = VG_TARGET / Vg_raw
SCALE_S = VS_TARGET / Vs_raw

def graphite_dn_da(a):
    """Scaled graphite grain size distribution 1/nH * dn/da in cm^-1."""
    return graphite_dn_da_unscaled(a) * SCALE_G

def silicate_dn_da(a):
    """Scaled silicate grain size distribution 1/nH * dn/da in cm^-1."""
    return silicate_dn_da_unscaled(a) * SCALE_S


# 4. Plot Size Distributions
def plot_distributions(save_path):
    print("Generating grain size distribution plots...")
    sizes_micron = np.logspace(-4.0, 1.0, 500)  # size range 0.1 nm to 10 micron
    sizes_cm = sizes_micron * 1e-4
    
    dn_da_gra_vals = np.array([0.8*graphite_dn_da(a) for a in sizes_cm])
    dn_da_sil_vals = np.array([silicate_dn_da(a) for a in sizes_cm])
    
    # Extract components of Graphite for diagnostic plotting
    lognormal_vals = []
    powerlaw_vals = []
    for a in sizes_cm:
        if a < A_MIN:
            lognormal_vals.append(0.0)
            powerlaw_vals.append(0.0)
            continue
        D_a = 0.0
        for Bi, a0i in [(B1, A01), (B2, A02)]:
            D_a += (Bi / a) * np.exp(-0.5 * (np.log(a / a0i) / SIGMA)**2)
        lognormal_vals.append(D_a * SCALE_G)
        
        if a < AT_G:
            M = 1.0
        else:
            M = np.exp(-((a - AT_G) / AC_G)**3)
        F = F_envelope(a, BETA_G, AT_G)
        PL_a = (C_G_RAW / a) * (a / AT_G)**ALPHA_G * F * M
        powerlaw_vals.append(PL_a * SCALE_G)
        
    lognormal_vals = np.array(lognormal_vals)
    powerlaw_vals = np.array(powerlaw_vals)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    
    # Load WD01 digitized GSD comparison data
    wd01_loaded = False
    try:
        wd01_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "external_data", "gsd_WD01_RV3.1_bc6e5_graphite.csv")
        if os.path.exists(wd01_path):
            wd01_data = np.loadtxt(wd01_path, delimiter=',')
            wd01_a_micron = wd01_data[:, 0]
            wd01_y_right = wd01_data[:, 1]
            wd01_a_cm = wd01_a_micron * 1e-4
            wd01_y_left = wd01_y_right / (1e29 * wd01_a_cm**4)
            wd01_loaded = True
    except Exception as e:
        print(f"Warning: Could not load WD01 comparison data ({e})")

    # Left Panel: standard 1/nH * dn/da size distribution
    axes[0].loglog(sizes_micron, dn_da_gra_vals, color='k', label='Total Carbonaceous', lw=2)
    axes[0].loglog(sizes_micron, lognormal_vals, color='C0', linestyle='--', label='PAHs / Lognormal D(a)')
    axes[0].loglog(sizes_micron, powerlaw_vals, color='C1', linestyle=':', label='Power Law component')
    axes[0].loglog(sizes_micron, dn_da_sil_vals, color='C2', linestyle='-.', label='Silicates', lw=2)
    if wd01_loaded:
        axes[0].loglog(wd01_a_micron, wd01_y_left, color='gray', linestyle='none', marker='x', label='WD01 Graphite (Digitized)', alpha=0.7)
    axes[0].set_xlabel(r'Grain radius $a$ [$\mu$m]')
    axes[0].set_ylabel(r'$1/n_{\rm H}\,\,dn/da$ [cm / H]')
    axes[0].set_title('Size Distributions')
    axes[0].set_xlim(1e-4, 5.0)
    axes[0].set_ylim(1e-20, 1e-6)
    axes[0].grid(True, which='both', linestyle=':', alpha=0.5)
    axes[0].legend(loc='lower left')
    
    # Right Panel: 10^29 * a_cm^4 * (1/nH * dn/da) to match W&D 2001 paper units
    fac = 1e29 * (sizes_cm**4)
    axes[1].loglog(sizes_micron, fac * dn_da_gra_vals, color='k', label='Total Carbonaceous', lw=2)
    axes[1].loglog(sizes_micron, fac * lognormal_vals, color='C0', linestyle='--', label='PAHs / Lognormal D(a)')
    axes[1].loglog(sizes_micron, fac * powerlaw_vals, color='C1', linestyle=':', label='Power Law component')
    axes[1].loglog(sizes_micron, fac * dn_da_sil_vals, color='C2', linestyle='-.', label='Silicates', lw=2)
    if wd01_loaded:
        axes[1].loglog(wd01_a_micron, wd01_y_right, color='gray', linestyle='none', marker='x', label='WD01 Graphite (Digitized)', alpha=0.7)
    axes[1].set_xlabel(r'Grain radius $a$ [$\mu$m]')
    axes[1].set_ylabel(r'$10^{29} a^4 n_{\rm H}^{-1} dn/da$ [cm$^3$]')
    axes[1].set_title(r'Volume Distribution ($10^{29} a^4 n_{\rm H}^{-1} dn/da$)')
    axes[1].set_xlim(1e-4, 5.0)
    axes[1].set_ylim(1e-5, 2e2)
    axes[1].grid(True, which='both', linestyle=':', alpha=0.5)
    axes[1].legend(loc='upper right')
    
    plt.suptitle(r'Weingartner and Draine (2001) Grain Size Distributions ($R_V=3.1$, $b_C=6.0\times 10^{-5}$)', y=0.98, fontsize=13)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved grain size distribution plot to {save_path}")


# 5. Analytical Recombination Rates Formula
def get_analytical_recomb_coefficient(ion, G0, ne, T):
    """
    Computes the grain-assisted ion recombination rate coefficient alpha
    in units of cm^3 s^-1 per H atom using the WD01 fitting formula.
    """
    if ion not in TABLE_2_COEFFS:
        raise ValueError(f"Ion '{ion}' not in Table 2 coefficients list.")
    
    C0, C1, C2, C3, C4, C5, C6 = TABLE_2_COEFFS[ion]
    
    # phi = G0 * T^0.5 / ne
    # Add a small offset to prevent division by zero in phi
    phi = G0 * np.sqrt(T) / max(ne, 1e-30)
    
    lnT = np.log(T)
    
    # Formula: alpha = 1e-14 * C0 / (1 + C1 * phi**C2 * (1 + C3 * T**C4 * phi**(-C5 - C6*lnT)))
    inner_phi_term = phi**(-C5 - C6 * lnT)
    denominator = 1.0 + C1 * (phi**C2) * (1.0 + C3 * (T**C4) * inner_phi_term)
    
    alpha = 1e-14 * C0 / denominator
    return alpha


# 6. Evaluation in standard environments
def run_environment_checks():
    environments = {
        'Cold Neutral Medium (CNM)':  {'T': 100.0,  'ne': 0.03,  'G0': 1.0},
        'Warm Neutral Medium (WNM)':  {'T': 6000.0, 'ne': 0.03,  'G0': 1.0},
        'Warm Ionized Medium (WIM)':  {'T': 8000.0, 'ne': 0.099, 'G0': 1.0}
    }
    
    print("\n" + "="*80)
    print("EVALUATION OF RECOMBINATION RATES IN STANDARD ISM PHASES")
    print("="*80)
    
    for name, env in environments.items():
        T, ne, G0 = env['T'], env['ne'], env['G0']
        phi = G0 * np.sqrt(T) / ne
        print(f"\nEnvironment: {name}")
        print(f"  T = {T} K | ne = {ne} cm^-3 | G0 = {G0} | phi = {phi:.2f} K^0.5 cm^3")
        print(f"  {'Species':<10} | {'Rate Coefficient alpha (cm3 s^-1)':<40}")
        print(f"  {'-'*10} | {'-'*40}")
        for species in ['H+', 'He+', 'C+', 'Fe+']:
            alpha = get_analytical_recomb_coefficient(species, G0, ne, T)
            print(f"  {species:<10} | {alpha:.5e}")
            
    print("\n" + "="*80)


# 7. Optional comparison with full numerical charging solver in CALIMA
def run_numerical_comparison_if_available():
    try:
        from models.dust_charge.dust_ion_recombination import compute_grain_assisted_ion_recombination
        plt.rcParams["text.usetex"] = False
        print("\nCALIMA charging module loaded successfully! Performing numerical comparison...")
        
        # Test conditions: CNM
        T = 100.0
        ne = 0.03
        G0 = 1.0
        
        ion_species = [
            {'name': 'H+', 'n': ne, 'T': T, 'm': 1.6726219e-27, 'z': 1.0, 's_i': 1.0}
        ]
        
        # Integrate numerically over 60 size bins
        sizes_cm = np.logspace(np.log10(A_MIN), np.log10(A_MAX), 60)
        
        alpha_gra = []
        alpha_sil = []
        for a in sizes_cm:
            # Case B threshold mode is the standard physical mode corresponding to WD01
            res_gra = compute_grain_assisted_ion_recombination(G0, ne, T, 'graphite', a, ion_species=ion_species, recomb_mode='case_b')
            res_sil = compute_grain_assisted_ion_recombination(G0, ne, T, 'silicate', a, ion_species=ion_species, recomb_mode='case_b')
            alpha_gra.append(res_gra['ion_recomb_rate_coefficients'][0])
            alpha_sil.append(res_sil['ion_recomb_rate_coefficients'][0])
            
        alpha_gra = np.array(alpha_gra)
        alpha_sil = np.array(alpha_sil)
        
        # Size distribution values
        dn_da_gra = np.array([graphite_dn_da(a) for a in sizes_cm])
        dn_da_sil = np.array([silicate_dn_da(a) for a in sizes_cm])
        
        # Integrate
        alpha_tot_gra = np.trapezoid(alpha_gra * dn_da_gra, sizes_cm)
        alpha_tot_sil = np.trapezoid(alpha_sil * dn_da_sil, sizes_cm)
        alpha_numerical = alpha_tot_gra + alpha_tot_sil
        
        alpha_fit = get_analytical_recomb_coefficient('H+', G0, ne, T)
        
        print(f"\nH+ Recombination in CNM (T=100K, ne=0.03, G0=1.0):")
        print(f"  Numerical (graphite):   {alpha_tot_gra:.5e} cm3 s^-1")
        print(f"  Numerical (silicate):   {alpha_tot_sil:.5e} cm3 s^-1")
        print(f"  Numerical Total:        {alpha_numerical:.5e} cm3 s^-1")
        print(f"  WD01 Analytical Fit:    {alpha_fit:.5e} cm3 s^-1")
        print(f"  Ratio (Numerical/Fit):  {alpha_numerical / alpha_fit:.3f}")
        print("Note: The analytical fit is accurate within ~20-40% compared to the detailed numerical calculations.")
        
    except Exception as e:
        print(f"\nNote: Detailed numerical comparison skipped ({str(e)}). Run in CALIMA environment to compare.")


def compute_single_phi(args):
    """
    Worker function to calculate numerical recombination rates for all species
    at a single phi value. Defined at top-level for pickling in multiprocessing.
    """
    phi_idx, phi, T, species_list, masses, sizes_cm, dn_da_gra, dn_da_sil, A_MIN, A_MAX, use_li_draine = args
    import models.dust_radiation.dust_emission as de
    de.USE_LI_DRAINE_2001_CARBONACEOUS = use_li_draine
    
    from models.dust_charge.dust_ion_recombination import compute_grain_assisted_ion_recombination
    
    G0 = 1.0
    ne = G0 * np.sqrt(T) / phi
    
    # Construct list of ion species
    n_ion_charging = ne / len(species_list)
    ion_species = [
        {'name': sp, 'n': n_ion_charging, 'T': T, 'm': masses[sp], 'z': 1.0, 's_i': 1.0}
        for sp in species_list
    ]
    
    alpha_gra_sizes = np.zeros((len(sizes_cm), len(species_list)))
    alpha_sil_sizes = np.zeros((len(sizes_cm), len(species_list)))
    
    alpha_gra_sizes_case_b = np.zeros((len(sizes_cm), len(species_list)))
    alpha_sil_sizes_case_b = np.zeros((len(sizes_cm), len(species_list)))
    
    for j, a in enumerate(sizes_cm):
        res_gra = compute_grain_assisted_ion_recombination(G0, ne, T, 'graphite', a, ion_species=ion_species, recomb_mode='case_a')
        res_sil = compute_grain_assisted_ion_recombination(G0, ne, T, 'silicate', a, ion_species=ion_species, recomb_mode='case_a')
        
        alpha_gra_sizes[j, :] = res_gra['ion_recomb_rate_coefficients']
        alpha_sil_sizes[j, :] = res_sil['ion_recomb_rate_coefficients']
        
        res_gra_b = compute_grain_assisted_ion_recombination(G0, ne, T, 'graphite', a, ion_species=ion_species, recomb_mode='case_b')
        res_sil_b = compute_grain_assisted_ion_recombination(G0, ne, T, 'silicate', a, ion_species=ion_species, recomb_mode='case_b')
        
        alpha_gra_sizes_case_b[j, :] = res_gra_b['ion_recomb_rate_coefficients']
        alpha_sil_sizes_case_b[j, :] = res_sil_b['ion_recomb_rate_coefficients']
        
    # Integrate each species over the GSD
    numerical_rates_case_a = {}
    numerical_rates_case_b = {}
    for idx_sp, sp in enumerate(species_list):
        alpha_tot_gra_a = np.trapezoid(alpha_gra_sizes[:, idx_sp] * dn_da_gra, sizes_cm)
        alpha_tot_sil_a = np.trapezoid(alpha_sil_sizes[:, idx_sp] * dn_da_sil, sizes_cm)
        numerical_rates_case_a[sp] = alpha_tot_gra_a + alpha_tot_sil_a
        
        alpha_tot_gra_b = np.trapezoid(alpha_gra_sizes_case_b[:, idx_sp] * dn_da_gra, sizes_cm)
        alpha_tot_sil_b = np.trapezoid(alpha_sil_sizes_case_b[:, idx_sp] * dn_da_sil, sizes_cm)
        numerical_rates_case_b[sp] = alpha_tot_gra_b + alpha_tot_sil_b
        
    return phi_idx, numerical_rates_case_a, numerical_rates_case_b


def compare_recomb_rates_over_phi_range(T=100.0, save_plot_path=None, use_li_draine=False):
    """
    Computes recombination coefficients over a range of phi = G0 * T^0.5 / ne
    for fixed temperature T, comparing numerical and analytical fits,
    and plotting the result. Parallelized using ProcessPoolExecutor.
    """
    from concurrent.futures import ProcessPoolExecutor
    plt.rcParams["text.usetex"] = False
    print(f"\nComputing recombination coefficients over phi range (10^2 to 10^6) at fixed T = {T} K...")
    
    phi_range = np.logspace(2.0, 6.0, 15)  # 15 points
    species_list = ['H+', 'He+', 'C+', 'Mg+', 'Si+', 'Fe+']
    
    # Load digitized WD01 H+ case b / case a ratio comparison
    wd01_ratio_loaded = False
    try:
        wd01_ratio_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "external_data", "recombination_correction_WD01_H+.csv")
        if os.path.exists(wd01_ratio_path):
            wd01_ratio_data = np.loadtxt(wd01_ratio_path, delimiter=',')
            wd01_ratio_gamma = wd01_ratio_data[:, 0]
            wd01_ratio_val = wd01_ratio_data[:, 1]
            wd01_ratio_loaded = True
    except Exception as e:
        print(f"Warning: Could not load WD01 H+ ratio data ({e})")

    # Dic to store results
    results = {
        sp: {
            'phi': phi_range,
            'analytical': [],
            'numerical_case_a': np.zeros(len(phi_range)),
            'numerical_case_b': np.zeros(len(phi_range))
        } for sp in species_list
    }
    
    # Compute analytical fits first
    G0 = 1.0
    for sp in species_list:
        for phi in phi_range:
            ne = G0 * np.sqrt(T) / phi
            alpha_fit = get_analytical_recomb_coefficient(sp, G0, ne, T)
            results[sp]['analytical'].append(alpha_fit)
            
    # Grid of grain sizes for integration (40 bins is a good speed/accuracy balance)
    sizes_cm = np.logspace(np.log10(A_MIN), np.log10(A_MAX), 50)
    dn_da_gra = np.array([0.8*graphite_dn_da(a) for a in sizes_cm])
    dn_da_sil = np.array([silicate_dn_da(a) for a in sizes_cm])
    
    masses = {
        'H+': 1.6726219e-27,
        'He+': 6.64648e-27,
        'C+': 1.9926467e-26,
        'Mg+': 4.03594e-26,
        'Si+': 4.6637e-26,
        'Fe+': 9.27329e-26
    }
    
    # Prepare task payloads for process pool
    tasks = [
        (i, phi, T, species_list, masses, sizes_cm, dn_da_gra, dn_da_sil, A_MIN, A_MAX, use_li_draine)
        for i, phi in enumerate(phi_range)
    ]
    
    print(f"  Distributing {len(phi_range)} environments across parallel processes...")
    
    try:
        from tqdm import tqdm
        HAS_TQDM = True
    except ImportError:
        HAS_TQDM = False
        
    from concurrent.futures import as_completed
    
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(compute_single_phi, t) for t in tasks]
        
        if HAS_TQDM:
            for future in tqdm(as_completed(futures), total=len(futures), desc="Scanning phi environments"):
                phi_idx, numerical_rates_case_a, numerical_rates_case_b = future.result()
                for sp in species_list:
                    results[sp]['numerical_case_a'][phi_idx] = numerical_rates_case_a[sp]
                    results[sp]['numerical_case_b'][phi_idx] = numerical_rates_case_b[sp]
        else:
            for future in as_completed(futures):
                phi_idx, numerical_rates_case_a, numerical_rates_case_b = future.result()
                print(f"    ✓ Completed environment index {phi_idx+1}/{len(phi_range)} (phi = {phi_range[phi_idx]:.2e})")
                for sp in species_list:
                    results[sp]['numerical_case_a'][phi_idx] = numerical_rates_case_a[sp]
                    results[sp]['numerical_case_b'][phi_idx] = numerical_rates_case_b[sp]
            
    # Plotting comparison
    if save_plot_path:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
        colors = {
            'H+': 'C0',
            'He+': 'C1',
            'C+': 'C2',
            'Mg+': 'C3',
            'Si+': 'C4',
            'Fe+': 'C5'
        }
        # Panel 1: Rate Coefficients
        for sp in species_list:
            ax1.loglog(phi_range, results[sp]['numerical_case_a'], color=colors[sp], linestyle='-', marker='o', label=f'{sp} Numerical (CALIMA)')
            ax1.loglog(phi_range, results[sp]['analytical'], color=colors[sp], linestyle='--', label=f'{sp} Analytical Fit (WD01)')
            
        ax1.set_xlabel(r'$\phi = G_0 T^{1/2} / n_e$ [K$^{1/2}$ cm$^3$]')
        ax1.set_ylabel(r'Rate Coefficient $\alpha$ [cm$^3$ s$-1$ per H atom]')
        ax1.set_title(rf'Ion Recombination Rate Coefficient vs. $\phi$ (T = {T} K)')
        ax1.grid(True, which='both', linestyle=':', alpha=0.5)
        ax1.legend(loc='lower left', ncol=2, fontsize=8)
        
        # Panel 2: Case B / Case A Ratio for H+
        ratio_numerical_H = results['H+']['numerical_case_b'] / results['H+']['numerical_case_a']
        ax2.loglog(phi_range, ratio_numerical_H, color='C0', linestyle='-', marker='o', lw=2, label='CALIMA H+ (Numerical Ratio)')
        
        if wd01_ratio_loaded:
            ax2.loglog(wd01_ratio_gamma, wd01_ratio_val, color='gray', linestyle='--', marker='x', lw=2, label='WD01 H+ (Digitized Ratio)')
            
        ax2.set_xlabel(r'$\gamma = G_0 T^{1/2} / n_e$ [K$^{1/2}$ cm$^3$]')
        ax2.set_ylabel(r'Correction Factor $\alpha_{\rm case\_b} / \alpha_{\rm case\_a}$')
        ax2.set_title('H+ Recombination Correction Factor (Case B / Case A)')
        ax2.grid(True, which='both', linestyle=':', alpha=0.5)
        ax2.legend(loc='lower left', fontsize=10)
        # case_b is MORE restrictive than case_a (dU > 0 for positive ions), so ratio <= 1
        ax2.set_ylim([0.0, 1.05])
        
        plt.suptitle(f'Recombination Coefficient & Correction Factor Comparison (T = {T} K)', y=0.98, fontsize=14)
        fig.tight_layout()
        
        fig.savefig(save_plot_path, bbox_inches='tight')
        plt.close(fig)
        print(f"✓ Saved comparison plot to {save_plot_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Weingartner & Draine (2001) Size Distribution & Recombination Fitting Test")
    parser.add_argument("--use-li-draine", action="store_true", help="Use Li & Draine (2001) carbonaceous cross section blend.")
    args = parser.parse_args()
    
    use_ld = args.use_li_draine
    if use_ld:
        import models.dust_radiation.dust_emission as de
        de.USE_LI_DRAINE_2001_CARBONACEOUS = True
        print("Using Li & Draine (2001) carbonaceous cross sections for graphite...")
        
    print("=" * 80)
    print("WEINGARTNER & DRAINE (2001) SIZE DISTRIBUTION & RECOMBINATION FITTING TEST")
    print("=" * 80)
    
    print("Calculating unscaled grain size volumes...")
    print(f"  Target Vg: {VG_TARGET:.5e} cm3/H | Calculated unscaled Vg: {Vg_raw:.5e} cm3/H")
    print(f"  Target Vs: {VS_TARGET:.5e} cm3/H | Calculated unscaled Vs: {Vs_raw:.5e} cm3/H")
    print(f"  --> Graphite scaling factor: {SCALE_G:.5f}")
    print(f"  --> Silicate scaling factor: {SCALE_S:.5f}")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Save the size distribution plot to the same directory
    plot_path = os.path.join(script_dir, "grain_size_distribution_test.png")
    plot_distributions(plot_path)
    
    # Run environment test calculations
    run_environment_checks()
    
    # Numerical validation (single-point check)
    run_numerical_comparison_if_available()
    
    # Numerical comparison scan over phi range
    comparison_plot_path = os.path.join(script_dir, "recombination_rate_comparison.png")
    compare_recomb_rates_over_phi_range(T=100.0, save_plot_path=comparison_plot_path, use_li_draine=use_ld)


if __name__ == "__main__":
    main()
