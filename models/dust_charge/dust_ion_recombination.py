#!/usr/bin/env python3
"""
DUST ION RECOMBINATION IN THE ISM

This script computes the ion removal/recombination rate due to grain-assisted
ion recombination, based on the grain charge distributions from Weingartner & Draine (2001)
and Draine & Sutin (1987) collisional charging theory.

It includes the two cases from Weingartner & Draine (2001) that modify the
recombination rate by requiring that the electron transfer reaction be energetically allowed:
a) IP(X^(i-1)) - IP(a,Z) >= 0
b) IP(X^(i-1)) - IP(a,Z) + dU(Z,i) >= 0

By: Curro Rodriguez Montero (currodri@gmail.com)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure parent directories are on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from models.dust_charge.shared_physics import (
    KB_CGS, 
    DS87_J_function_vec, 
    ionisation_potential_valence_vec,
    electron_affinity_graphite_vec,
    electron_affinity_silicate_vec,
    E_STATC,
    EV2ERG
)
from models.dust_charge.dust_charging import equilibrium_charge_for_grain

# Atomic radii in Angstroms from Weingartner & Draine (2001) Table 1
ATOMIC_RADII = {
    'H': 0.37,
    'He': 0.50,
    'C': 0.77,
    'Na': 1.86,
    'Mg': 1.60,
    'Si': 1.18,
    'S': 1.03,
    'K': 2.27,
    'Ca': 1.97,
    'Mn': 1.37,
    'Fe': 1.24
}

# First ionization potentials in eV for neutral elements (equal to electron affinity of singly-charged ions)
IONIZATION_POTENTIALS = {
    'H': 13.60,
    'He': 24.60,
    'C': 11.26,
    'Na': 5.14,
    'Mg': 7.65,
    'Si': 8.15,
    'S': 10.36,
    'K': 4.34,
    'Ca': 6.11,
    'Mn': 7.43,
    'Fe': 7.90
}

def _get_element_key(name):
    if not name or not isinstance(name, str):
        return None
    # Look for two-letter elements first
    for el in ['He', 'Na', 'Mg', 'Si', 'Ca', 'Mn', 'Fe']:
        if name.startswith(el):
            return el
    # Look for one-letter elements
    for el in ['H', 'C', 'S', 'K']:
        if name.startswith(el):
            return el
    return None

def compute_ion_recombination_coefficients(Zs, P, a_cm, ion_species, grain_type='graphite', recomb_mode=None):
    """
    Compute the grain-assisted ion recombination rate coefficients for a list of ion species
    given the grain charge distribution P(Zs) for a grain of radius a_cm.

    Parameters
    ----------
    Zs : array-like
        Array of integer charge states.
    P : array-like
        Probability distribution of grain charge states.
    a_cm : float
        Grain radius in cm.
    ion_species : list of dict
        List of ion species dictionaries. Each dictionary must contain:
        - 'z' : charge state of the ion (in units of e, e.g. 1.0)
        - 'm' : ion mass in grams (if < 1e-25, assumed to be in kg and auto-converted to grams)
        - 'T' : ion temperature in K
        - 'n' : ion number density in cm^-3 (optional, defaults to 0.0)
        - 's_i' : ion sticking coefficient (optional, defaults to 1.0 for WD01)
        - 'r0' : atomic radius in Angstroms (optional, auto-inferred if name is provided)
        - 'IP_X_im1' : ionization potential of species X^(i-1) in eV (optional, auto-inferred if name is provided)
    grain_type : {'graphite','silicate'}, optional
        Grain material, needed to determine the work function W for IP(a,Z).
    recomb_mode : {None, 'case_a', 'case_b'}, optional
        - None: standard recombination rate (no threshold modification).
        - 'case_a': rate multiplied by theta(IP(X^(i-1)) - IP(a,Z)).
        - 'case_b': rate multiplied by theta(IP(X^(i-1)) - IP(a,Z) + dU(Z,i)).

    Returns
    -------
    recomb_rates : list of float
        Recombination rates (s^-1 per grain) for each ion species (n_i * alpha_i)
    recomb_coeffs : list of float
        Recombination rate coefficients (cm^3 s^-1 per grain) for each ion species
    """
    Zs = np.asarray(Zs, dtype=float)
    P = np.asarray(P, dtype=float)
    cross = np.pi * a_cm * a_cm
    
    # Work function W in eV
    if grain_type.lower().startswith('gra') or grain_type.lower().startswith('car'):
        W = 4.4
    else:
        W = 8.0
        
    recomb_rates = []
    recomb_coeffs = []
    
    for ion in ion_species:
        n_i = float(ion.get("n", 0.0))
        T_i = float(ion.get("T", 1.0))
        m_g = float(ion.get("m", 1.0))
        z_i = float(ion.get("z", 1.0))
        s_i = float(ion.get("s_i", 1.0)) # Sticking coefficient defaults to 1.0 (WD01)
        
        # Unit correction: mass of proton is ~1.67e-24 g (~1.67e-27 kg)
        # If mass < 1e-25, it is likely in kg, so convert to grams for CGS consistency
        m_g_cgs = m_g * 1e3 if m_g < 1e-25 else m_g

        vth_i = np.sqrt(8.0 * KB_CGS * T_i / (np.pi * m_g_cgs))
        Jtilde_i = DS87_J_function_vec(Zs, np.array([z_i]), a_cm, T_i)
        
        # Apply the threshold conditions if requested
        if recomb_mode in ['case_a', 'case_b']:
            el_key = _get_element_key(ion.get('name', ''))
            
            # Retrieve atomic radius r0 (in Angstroms) and ionization potential IP_X_im1 (in eV)
            r0 = float(ion.get('r0', ATOMIC_RADII.get(el_key, 0.77) if el_key else 0.77))
            IP_X_im1 = float(ion.get('IP_X_im1', ion.get('IP_ion', ion.get('ionization_potential', IONIZATION_POTENTIALS.get(el_key, 13.6) if el_key else 13.6))))
            
            # Calculate grain ionization potential IP(a,Z) in eV (piecewise: valence IP for Z >= 0, EA of Z+1 for Z < 0)
            is_graphite = grain_type.lower().startswith('gra') or grain_type.lower().startswith('car')
            IP_a_Z = np.zeros_like(Zs)
            for idx, Z in enumerate(Zs):
                if Z >= 0:
                    IP_a_Z[idx] = ionisation_potential_valence_vec(W, Z, a_cm)
                else:
                    if is_graphite:
                        IP_a_Z[idx] = electron_affinity_graphite_vec(Z + 1, a_cm)
                    else:
                        IP_a_Z[idx] = electron_affinity_silicate_vec(Z + 1, a_cm)
            
            if recomb_mode == 'case_a':
                y = IP_X_im1 - IP_a_Z
                theta = np.where(y >= 0.0, 1.0, 0.0)
                # print('case A: ',theta, IP_X_im1, IP_a_Z)
            elif recomb_mode == 'case_b':
                r0_cm = r0 * 1e-8
                term1 = (z_i - Zs - 1.0) * (E_STATC**2) / (a_cm + r0_cm)
                term2 = (1.0 - 2.0 * z_i) * (E_STATC**2) * (a_cm**3) / (2.0 * r0_cm * ((a_cm + r0_cm)**2) * (2.0 * a_cm + r0_cm))
                dU = (term1 + term2) / EV2ERG
                y = IP_X_im1 - IP_a_Z + dU   # WD01 Eq.6: IP(a,Z) - IP(X^i-1) + dU < 0 (This actually has a typo in the paper)
                theta = np.where(y >= 0.0, 1.0, 0.0)
                # if (any(theta>0)): print('case B: ',theta, dU, IP_a_Z,P)
            
            Jtilde_i = Jtilde_i * theta
        
        # Rate coefficient (cm^3 s^-1 per grain): alpha = sum_Z P(Z) * cross * vth * s_i * Jtilde
        alpha_i = float(np.sum(P * cross * vth_i * s_i * Jtilde_i))
        rate_i = n_i * alpha_i
        
        recomb_rates.append(rate_i)
        recomb_coeffs.append(alpha_i)
        
    return recomb_rates, recomb_coeffs

def compute_grain_assisted_ion_recombination(G0, ne, T, grain_type, a_cm, radiation_model='Mathis', ion_species=None, yield_params=None, recomb_mode=None):
    """
    Wrapper to compute both the grain charge distribution and the ion recombination rates.

    Parameters
    ----------
    G0 : float
        Radiation field scaling factor (dimensionless).
    ne : float
        Electron density in cm^-3.
    T : float
        Temperature in K.
    grain_type : {'graphite','silicate'}
        Grain material/type.
    a_cm : float
        Grain radius in cm.
    radiation_model : str
        Radiation model (default 'Mathis').
    ion_species : list or None
        Ion species definitions.
    yield_params : dict or None
        Yield params.
    recomb_mode : {None, 'case_a', 'case_b'}, optional
        Recombination threshold mode following Weingartner & Draine (2001):
        - None / 'case_a': rate weighted by theta(IP(X^i-1) - IP(a,Z) >= 0)
        - 'case_b': rate weighted by theta(IP(X^i-1) - IP(a,Z) + dU(Z,i) >= 0)

        IMPORTANT: the equilibrium charge distribution P(Z) is always solved
        without any threshold modification.  The case_a/case_b flag is applied
        only as a post-hoc mask on Jtilde when integrating the rate coefficient.
        Passing recomb_mode into equilibrium_charge_for_grain would couple the
        threshold into P(Z) itself and destroy the gamma-dependence of the ratio.

    Returns
    -------
    dict
        Dictionary containing the full charging outputs and computed ion recombination rates.
    """
    if ion_species is None:
        ion_species = []

    # 1) Compute equilibrium charge distribution — always with recomb_mode=None
    #    so that P(Z) is the physical grain charge distribution independent of
    #    which recombination threshold convention is being evaluated.
    Zs, P, rates, Zmean, Zsigma = equilibrium_charge_for_grain(
        G0, ne, T, grain_type, a_cm,
        radiation_model=radiation_model,
        ion_species=ion_species,
        yield_params=yield_params,
        recomb_mode=None
    )

    # 2) Apply the case_a / case_b threshold only when computing the rate
    #    coefficients from the already-determined P(Z).
    recomb_rates, recomb_coeffs = compute_ion_recombination_coefficients(
        Zs, P, a_cm, ion_species, grain_type=grain_type, recomb_mode=recomb_mode
    )

    return {
        'Zs': Zs,
        'P': P,
        'Zmean': Zmean,
        'Zsigma': Zsigma,
        'ion_recomb_rates': recomb_rates,
        'ion_recomb_rate_coefficients': recomb_coeffs,
        'rates_dict': rates
    }

def run_and_plot(G0, ne, T, suffix, env_name):
    # Ion species (using typical H+ and C+ abundances)
    ion_species = [
        {'name': 'H+', 'n': 0.005, 'T': T, 'm': 1.6726219e-27, 'z': 1.0, 's_i': 1.0},
        {'name': 'C+', 'n': 0.0001, 'T': T, 'm': 1.9926467e-26, 'z': 1.0, 's_i': 1.0}
    ]
    
    # Range of grain sizes from 5 Angstroms to 0.1 microns (in cm)
    sizes_micron = np.logspace(-3, -1, 30)
    sizes_cm = sizes_micron * 1e-4
    
    # Lists to store results for plotting
    results = {
        'graphite': {
            'H+': {None: [], 'case_a': [], 'case_b': []},
            'C+': {None: [], 'case_a': [], 'case_b': []}
        },
        'silicate': {
            'H+': {None: [], 'case_a': [], 'case_b': []},
            'C+': {None: [], 'case_a': [], 'case_b': []}
        }
    }
    
    print(f"Computing recombination rate coefficients for {env_name} over {len(sizes_micron)} grain sizes...")
    for a in sizes_cm:
        for material in ['graphite', 'silicate']:
            for mode in [None, 'case_a', 'case_b']:
                res = compute_grain_assisted_ion_recombination(G0, ne, T, material, a, ion_species=ion_species, recomb_mode=mode)
                results[material]['H+'][mode].append(res['ion_recomb_rate_coefficients'][0])
                results[material]['C+'][mode].append(res['ion_recomb_rate_coefficients'][1])

    # Plotting
    sns.set_theme(style="white")
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": "Computer Modern Roman",
    })
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=150, sharex=True)
    
    # Panel 0,0: H+ on Graphite
    axes[0, 0].loglog(sizes_micron, results['graphite']['H+'][None], label='Standard (No limit)', color='black', linestyle='-')
    axes[0, 0].loglog(sizes_micron, results['graphite']['H+']['case_a'], label='Case A: IP threshold', color='C0', linestyle='--')
    axes[0, 0].loglog(sizes_micron, results['graphite']['H+']['case_b'], label='Case B: IP + interaction energy', color='C1', linestyle='-.')
    axes[0, 0].set_ylabel(r'Rate Coefficient $\alpha_{\rm gr}$ [cm$^3$ s$^{-1}$ per grain]', fontsize=11)
    axes[0, 0].set_title(r'H$^+$ on Graphite', fontsize=12)
    axes[0, 0].grid(True, which='both', linestyle=':', alpha=0.5)
    axes[0, 0].legend(fontsize=9, loc='lower right')

    # Panel 0,1: H+ on Silicate
    axes[0, 1].loglog(sizes_micron, results['silicate']['H+'][None], label='Standard (No limit)', color='black', linestyle='-')
    axes[0, 1].loglog(sizes_micron, results['silicate']['H+']['case_a'], label='Case A: IP threshold', color='C0', linestyle='--')
    axes[0, 1].loglog(sizes_micron, results['silicate']['H+']['case_b'], label='Case B: IP + interaction energy', color='C1', linestyle='-.')
    axes[0, 1].set_title(r'H$^+$ on Silicate', fontsize=12)
    axes[0, 1].grid(True, which='both', linestyle=':', alpha=0.5)
    axes[0, 1].legend(fontsize=9, loc='lower right')

    # Panel 1,0: C+ on Graphite
    axes[1, 0].loglog(sizes_micron, results['graphite']['C+'][None], label='Standard (No limit)', color='black', linestyle='-')
    axes[1, 0].loglog(sizes_micron, results['graphite']['C+']['case_a'], label='Case A: IP threshold', color='C0', linestyle='--')
    axes[1, 0].loglog(sizes_micron, results['graphite']['C+']['case_b'], label='Case B: IP + interaction energy', color='C1', linestyle='-.')
    axes[1, 0].set_xlabel(r'Grain radius $a$ [$\mu$m]', fontsize=11)
    axes[1, 0].set_ylabel(r'Rate Coefficient $\alpha_{\rm gr}$ [cm$^3$ s$^{-1}$ per grain]', fontsize=11)
    axes[1, 0].set_title(r'C$^+$ on Graphite', fontsize=12)
    axes[1, 0].grid(True, which='both', linestyle=':', alpha=0.5)
    axes[1, 0].legend(fontsize=9, loc='lower right')

    # Panel 1,1: C+ on Silicate
    axes[1, 1].loglog(sizes_micron, results['silicate']['C+'][None], label='Standard (No limit)', color='black', linestyle='-')
    axes[1, 1].loglog(sizes_micron, results['silicate']['C+']['case_a'], label='Case A: IP threshold', color='C0', linestyle='--')
    axes[1, 1].loglog(sizes_micron, results['silicate']['C+']['case_b'], label='Case B: IP + interaction energy', color='C1', linestyle='-.')
    axes[1, 1].set_xlabel(r'Grain radius $a$ [$\mu$m]', fontsize=11)
    axes[1, 1].set_title(r'C$^+$ on Silicate', fontsize=12)
    axes[1, 1].grid(True, which='both', linestyle=':', alpha=0.5)
    axes[1, 1].legend(fontsize=9, loc='lower right')

    plt.suptitle(f'Comparison of Recombination Models in {env_name} ($T={T}$ K, $G_0={G0}$, $n_e={ne}$ cm$^{{-3}}$)', fontsize=14, y=0.98)
    
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_png = os.path.join(out_dir, f'recombination_rate_vs_size_{suffix}.png')
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"✓ Saved demo plot to {out_png}")

def main():
    print("=" * 80)
    print("DEMONSTRATION: GRAIN-ASSISTED ION RECOMBINATION WITH THRESHOLD EFFECTS")
    print("=" * 80)
    
    # Environment 1: Cold Neutral Medium (CNM)
    run_and_plot(G0=1.0, ne=0.03, T=100.0, suffix='cnm', env_name='Cold Neutral Medium')
    
    # Environment 2: Highly Radiated Environment
    run_and_plot(G0=1e4, ne=0.03, T=100.0, suffix='irradiated', env_name='Highly Radiated')

if __name__ == '__main__':
    main()
