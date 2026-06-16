"""
PAH SPUTTERING MODEL

The functions, data and models presented within this script
intend to encompass the large theoretical framework developed by
Elisabetta Rita Micelotta during her PhD at Leiden supervised by 
F. P. Israel and A. G. G. M. Tielens to explain the processing 
of PAHS in the ISM. It should be noted that some of the calculations
are explictly computed in the same way as it was presented in the
original papers, while others required fitting to some of their
results for reproducibility.

By: F. Rodriguez Montero (currodri@gmail.com)
"""

# Import libraries
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import concurrent.futures
import time
from types import SimpleNamespace

try:
    from . import PAHs_model
except ImportError:
    try:
        import PAHs_model
    except ImportError:
        from models.tools.utils import Nc_from_size, size_from_Nc
        PAHs_model = SimpleNamespace(Nc_from_size=Nc_from_size, size_from_Nc=size_from_Nc)

# Set OMP_NUM_THREADS to limit the number of threads used by OpenBLAS
os.environ["OMP_NUM_THREADS"] = "1"  # Set it to the desired number of threads
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# Constants
k_IR          = 100. # [photons / s] - IR emission rate
E_0           = 4.6 # [eV] - unimolecular dissociation threshold energy (acetylene loss)
T_0           = 7.5 # [eV] - threshold energy for C-atom dissociation
#k_0           = 1.4e16 # [1/s] - pre-exponential factor for dissociation rate
Delta_epsilon = 0.16 # [eV] - change in internal energy of PAH due to IR photon emission of a typical C-C mode
thickness     = 4.31e-8 # [cm] - PAH electron cloud thickness
e_sp_min = 10 # [eV] - minimum electron energy for the stopping power fitting function
a_0           = 5.291e-9 # [cm] - atomic length unit
eV2erg = 1.6021773300241e-12 # [erg] conversion between eV to erg
au2cgs_v = 2.18769126364e8    # [cm/s] conversion between a.u. velocity and cgs velocity
kb = 1.3806488e-16 # [erg/K]
elem_charge = 4.8032047e-10 # [statC]
KB_OVER_H = 2.083661912332757e10 # [K^-1 s^-1]
_ELECTRON_STOPPING_LOOKUP_N = 4096
_ELECTRON_STOPPING_LOOKUP_PAD = 1.2
_ELECTRON_STOPPING_LOOKUP_E_MAX = 1e6
_ELECTRON_STOPPING_E_GRID = None
_ELECTRON_STOPPING_F_GRID = None
_ION_COLLISION_S_SAMPLES = 128

# Fitting parameters for the friction coefficient scaling law from the
# results of Puska & Nieminen (1983). Gamma_0 is in terms of (a.u.)^2.
# (https://journals.aps.org/prb/pdf/10.1103/PhysRevB.27.6121)
friction_params = {
    'H' : {'Gamma_0': 0.33,'R_2': 2.28},
    'He': {'Gamma_0': 0.75,'R_2': 0.88},
    'C' : {'Gamma_0' : 1.68,'R_2': 0.90},
    'O' : {'Gamma_0' : 1.62,'R_2': 0.57}
}

Asplund09_massfractions = {
    'H': 0.7381,
    'He': 0.2485,
    'C': 2.38e-3,
    'O': 5.73e-3,
}


# Basic physical functions
def Maxwell_Boltzmann_function(v,m,T):
    """Maxwell-Boltzmann probability distribution function.

    Args:
        v (float): velocity in cm/s
        m (float): particle mass in g
        T (float): temperature in K

    Returns:
        float: probability of finding the particle of mass m at velocity v in a pool of temperature T
    """    
    
    f = (m/(2.*np.pi*kb*T))**(3./2.) * (4.*np.pi*v**2.) * np.exp(-m*v**2./(2*kb*T))
    
    return f


def _pah_radius_cm_from_Nc(Nc, radius_method='Draine21'):
    """Return PAH radius in cm from Nc and selected size relation."""

    Nc_eff = max(1.0, float(Nc))

    if radius_method == 'Draine21':
        return PAHs_model.size_from_Nc(Nc_eff) * 1e-8
    if radius_method == 'Omont86':
        return 0.9 * np.sqrt(Nc_eff) * 1e-8
    raise NameError(f'This radius_method is not included in this model: {radius_method}')


def coulomb_energy_shift_eV(ion_charge, pah_charge, pah_radius_cm):
    """Electrostatic impact-energy shift in eV.

    Positive values correspond to opposite-sign charges (attractive interaction).
    """

    r_cm = float(pah_radius_cm)
    if (not np.isfinite(r_cm)) or abs(r_cm) <= 0.0:
        return 0.0
    return -float(ion_charge) * float(pah_charge) * elem_charge**2.0 / (r_cm * eV2erg)


def _build_phi_grid_from_charge_sets(ion_charge_values, pah_charge_values, pah_radius_cm):
    """Build sorted unique phi[eV] values from ion and PAH charge sets."""

    phi_vals = []
    for zion in ion_charge_values:
        for zpah in pah_charge_values:
            phi_vals.append(coulomb_energy_shift_eV(zion, zpah, pah_radius_cm))

    # Ensure phi=0 is always included.
    phi_vals.append(0.0)

    # Round before unique to avoid tiny floating-point duplicates.
    phi_grid = np.unique(np.round(np.asarray(phi_vals, dtype=float), 14))
    return np.sort(phi_grid)


def cross_section_geometrical(R,d,theta):
    """Geometrical cross section seen by an incident particle with direction defined by theta.

    Args:
        R (float): radius of PAH in a.u.
        d (float): thickness of PAH in a.u.
        theta (float): angle of incidence in radians

    Returns:
        float: cross section in a.u.
    """    
    
    sigma = np.pi * R**2. * np.cos(theta) + 2. * R * d * np.sin(theta)
    
    return sigma


# Electronic or electron interactions

def path_l(R,d,theta):
    """Pathlength through the PAH, described by a thick-disk geometry
    of radius R and thickness d.

    Args:
        R (float): PAH radius in a.u.
        d (float): thickness of PAH in a.u.
        theta (float): angle of incidence in radians

    Returns:
        float: pathlength in a.u.
    """    
    
    alpha = np.pi/2. - np.arctan(d/R)
    
    if abs(np.tan(theta)) < np.tan(alpha):
        l = d / abs(np.cos(theta))
    else:
        l = 2. * R / abs(np.sin(theta))
        
    return l

def inv_electron_stopping_power(E):
    """Inverse of electron stopping power fitting function to the experimental
    data by Joy (1995).

    Args:
        E (float): electron energy in eV

    Returns:
        float: inverse stopping power in Angstrom / eV
    """    
    E_in_keV = E * 1e-3
    S1 = 1.41476 * np.log(1.-0.000423375*E_in_keV)
    S2 = -0.000232675 * E_in_keV **(1.53851) - 3.57429e-11 * E_in_keV **(-3.18688) - 3.37861e-7 * E_in_keV**(-0.587928)

    return S2 / S1


def _build_electron_stopping_lookup(E_max_eV):
    """Build lookup table for F(E)=int_{e_sp_min}^E inv_stopping(x) dx."""

    E_hi = max(float(E_max_eV), e_sp_min * 1.001)
    npts = int(max(256, _ELECTRON_STOPPING_LOOKUP_N))

    E_grid = np.logspace(np.log10(e_sp_min), np.log10(E_hi), npts)
    inv_s = inv_electron_stopping_power(E_grid)

    dE = np.diff(E_grid)
    f_mid = 0.5 * (inv_s[1:] + inv_s[:-1])

    F_grid = np.zeros_like(E_grid)
    F_grid[1:] = np.cumsum(dE * f_mid)
    return E_grid, F_grid


def _get_electron_stopping_lookup(required_E_eV):
    """Return cached lookup arrays; rebuild only if larger E coverage is needed."""

    global _ELECTRON_STOPPING_E_GRID, _ELECTRON_STOPPING_F_GRID

    required = max(float(required_E_eV), e_sp_min * 1.001)
    if _ELECTRON_STOPPING_E_GRID is None:
        E_max = max(_ELECTRON_STOPPING_LOOKUP_E_MAX, required * _ELECTRON_STOPPING_LOOKUP_PAD)
        _ELECTRON_STOPPING_E_GRID, _ELECTRON_STOPPING_F_GRID = _build_electron_stopping_lookup(E_max)
    elif required > _ELECTRON_STOPPING_E_GRID[-1]:
        E_max = required * _ELECTRON_STOPPING_LOOKUP_PAD
        _ELECTRON_STOPPING_E_GRID, _ELECTRON_STOPPING_F_GRID = _build_electron_stopping_lookup(E_max)

    return _ELECTRON_STOPPING_E_GRID, _ELECTRON_STOPPING_F_GRID

def effective_temperature(Nc,Te,binding_energy):
    """Effective temperature of the PAH in the microcanonical description
    of a PAH (see Tielens 2005, 2021).

    Args:
        Nc (float): number of carbon atoms in PAH
        Te (float): internal energy in eV
        binding_energy (float): binding energy of the fragment in eV

    Returns:
        float: effective temperature in K
    """    
    
    Teff = 2000. * (Te/Nc)**0.4 *(1.-0.2*binding_energy/Te)
    # Teff = 3750. * (Te/Nc)**0.45 * (1.-0.23*binding_energy/Te)
    return Teff

def electronic_electron_collision(R,d,theta,init_energy):
    """Excitation energy of the PAH molecule caused by the collision with a fast electron
    from the gas phase. Consider that the approximations here used can only be extended to ~1e8 K.

    Args:
        R (float): radius of PAH in a.u.
        d (float): thickness of PAH in a.u.
        theta (float): incidence angle in radians
        init_energy (float): initial energy of impacting electron in eV

    Returns:
        float: final excitation energy in eV
    """    
    
    # 1. Compute length through the PAH (convert from a.u. to Angstrom)
    l = path_l(R,d,theta) * a_0 / 1e-8

    Ei = float(init_energy)
    if Ei <= e_sp_min:
        return 0.0

    E_grid, F_grid = _get_electron_stopping_lookup(Ei)

    # 2. Obtain the value of F for E0
    F_0 = np.interp(Ei, E_grid, F_grid, left=0.0, right=F_grid[-1])
    
    # 3. Compute F(E_1)
    F_1 = F_0 - l

    # 4. Compute E1 by inverse interpolation of precomputed F(E)
    if F_1 <= 0.0:
        E_1 = e_sp_min
    else:
        E_1 = float(np.interp(F_1, F_grid, E_grid, left=e_sp_min, right=E_grid[-1]))
    
    # 5. The final excitation energy is the difference between E1 and the initial electron energy
    T = Ei - E_1

    return max(0.0, T)

def electron_density(s,theta):
    """Valence electron density in the jellium model for the thick-disk 
    geometry of PAHs.

    Args:
        s (float): length through the PAH in a.u.
        theta (float): incidence angle in radians

    Returns:
        float: electron number density in a.u.^-3
    """    
    
    n0 = 0.15 * np.exp(-(s*np.cos(theta))**2./2.7)
    return n0

def density_parameter(s,theta):
    """Density parameter for the friction coefficient of Puska & Nieminen (1983).

    Args:
        s (float): length through the PAH in a.u.
        theta (float): incidence angle in radians

    Returns:
        float: density parameter in a.u.
    """    
    
    r_s = (4./3. * np.pi * electron_density(s,theta))**(-1./3.)
    return r_s

def friction_coefficient(particle_type,s,theta):
    """Friction coefficient of Puska & Nieminen (1983).

    Args:
        particle_type (str): particle type, either 'H', 'He' and 'C'
        s (float): length through the PAH in a.u.
        theta (float): incidence angle in radians

    Returns:
        float: friction coefficient in a.u.^2
    """    
    
    params = friction_params[particle_type]
    gamma = params['Gamma_0'] * np.exp(-(density_parameter(s,theta)-1.5)/params['R_2'])
    return gamma

def electronic_ion_collision(v,R,d,theta,particle_type):
    """Inelatic energy loss of ions as they collide with PAHs.

    Args:
        v (float): velocity of ion in a.u.
        R (float): radius of PAH in a.u.
        theta (float): angle of incidence in radians
        particle_type (str):  particle type, either 'H', 'He' and 'C'

    Returns:
        float: final excitation energy in eV
    """    
    
    from scipy.integrate import quad
    
    def v_gamma(s):
        f = v * friction_coefficient(particle_type,s,theta)
        return f
    l = path_l(R,d,theta)
    F, error = quad(v_gamma,-l/2,l/2)
    T = 27.2116 * F
    
    return T


def _ion_collision_theta_loss_integral(R, d, theta, particle_type, nsamples=_ION_COLLISION_S_SAMPLES):
    """Compute integral of friction coefficient over path length for fixed theta.

    This is the expensive, velocity-independent part of ion electronic collision.
    """

    l = path_l(R, d, theta)
    n = int(max(16, nsamples))
    s_grid = np.linspace(-0.5 * l, 0.5 * l, n)
    gamma = np.zeros(n, dtype=float)
    for i in range(n):
        gamma[i] = friction_coefficient(particle_type, s_grid[i], theta)
    return float(np.trapezoid(gamma, s_grid))

def dissociation_probability(binding_energy,Nc,T_av):
    """Normalised PAH dissociation probability.

    Args:
        binding_energy (float): fragment binding energy to PAH molecule in eV
        Nc (int): number of carbon atoms in PAH molecule
        T_av (float): average temperature of the excitation in K

    Returns:
        float: normalised probability for dissociation
    """
    # 1. Compute the maximum number of IR photon emissions
    # as suggested by the results of Micelotta et al. (2010b)
    n_max = float(int(Nc / 5))
    
    # 2. Compute the probability based on the value of T_av
    DeltaS = 10 # [cal/K/mol]
    R = 1.98720425864083 # [cal/K/mol]
    k_0 = KB_OVER_H * float(T_av) * np.exp(1.0 + DeltaS / R) # [s^-1]
    boltz = np.exp(-binding_energy / (8.617e-5 * float(T_av)))
    
    P = (k_0 * boltz) / ((k_IR / (n_max + 1.0)) + (k_0 * boltz))
    
    return P
    
def electronic_destruction_rate_T(Tgas,particle_type,Nc,
                                  binding_energy=E_0,
                                  nbins_v=100,
                                  nbins_theta=30,
                                  radius_method='Draine21',
                                  ion_charge=1,
                                  pah_charge=0,
                                  phi_eV=None):
    
    from scipy.integrate import trapezoid

    # 1. Determine the interaction details that depend on the particle type
    if particle_type == 'e':
        m_particle = 9.10938e-28 # [g]
        int_type = 'electron'
    elif particle_type == 'H':
        m_particle = 1.673557e-24 # [g]
        int_type = 'ion'
    elif particle_type == 'He':
        m_particle = 6.6464731e-24 # [g]
        int_type ='ion'
    elif particle_type == 'C':
        m_particle = 1.9944733e-23 # [g]
        int_type ='ion'
    elif particle_type == 'O':
        m_particle = 2.6566962e-23 # [g]
        int_type ='ion'
    else:
        raise NameError(f'This particle_type is not included in this model: {particle_type}')
    
    # 2. Compute the PAH geometric parameters [a.u.]
    R_cm = _pah_radius_cm_from_Nc(Nc, radius_method=radius_method)
    R = R_cm / a_0
    d = thickness / a_0
    n_max = float(int(Nc / 5))

    # Electrostatic impact-energy shift [eV] for ions.
    if phi_eV is None:
        E_charge = coulomb_energy_shift_eV(ion_charge, pah_charge, R_cm)
    else:
        E_charge = float(phi_eV)
    
    # 2. Compute the particle minimum velocity needed for a given binding energy
    v_0 = np.sqrt(2. * binding_energy * eV2erg / m_particle) # [cm/s]
    v_0 = v_0 / au2cgs_v # [a.u.]
    
    # 3. We set the maximum velocity to the thermal energy of gas at ~1e9 K
    v_max = np.sqrt(2. * 1e5 * eV2erg / m_particle) # [cm/s]
    v_max = v_max / au2cgs_v # [a.u.]
    
    # 4. Perform loop over velocity range
    v = np.logspace(np.log10(v_0),np.log10(v_max),nbins_v)
    J_v = np.zeros(nbins_v)
    
    for i in range(0, nbins_v):
        vi = v[i]
        Ei = (0.5 * m_particle * (v[i] * au2cgs_v)**2.) / eV2erg
        mb_factor = Maxwell_Boltzmann_function(vi*au2cgs_v,m_particle,Tgas)
        
        theta = np.linspace(0.,np.pi/2.,nbins_theta)
        J_theta = np.zeros(nbins_theta)
        
        for j in range(0, nbins_theta):
            # Get the cross section for the given theta
            sigma = cross_section_geometrical(R,d,theta[j])
            
            # Compute the transfer energy depending on the type of interaction
            if int_type == 'electron':
                # For electrons, compute Coulomb shift with electron charge = -1
                E_charge_electron = coulomb_energy_shift_eV(-1, pah_charge, R_cm)
                Ei_eff = Ei + E_charge_electron
                if Ei_eff > e_sp_min:
                    vi_eff = np.sqrt(2.0 * Ei_eff * eV2erg / m_particle) / au2cgs_v
                    T_0 = electronic_electron_collision(R,d,theta[j],Ei_eff)
                else:
                    # If the initial electron energy is below the minimum for
                    # the stopping power function, the dissociation probability
                    # is assumed to be equal to zero
                    J_theta[j] = 0.0
                    continue
            else:
                Ei_eff = Ei + E_charge
                if Ei_eff <= 0.0:
                    J_theta[j] = 0.0
                    continue
                vi_eff = np.sqrt(2.0 * Ei_eff * eV2erg / m_particle) / au2cgs_v
                T_0 = electronic_ion_collision(vi_eff,R,d,theta[j],particle_type)

            # Internal energy after the emission of nmax photons
            T_nmax = T_0 - n_max * Delta_epsilon
            
            if T_nmax <= 0.0 or T_nmax<=0.2*binding_energy:
                # If the energy induced is sufficiently low to be quickly
                # radiated by IR photons, we just set the probability
                # (and hence rate) of dissociation to zero
                J_theta[j] = 0.0
            else:
                # Convert energies to effective temperatures
                T_0 = effective_temperature(Nc,T_0,binding_energy)
                T_nmax = effective_temperature(Nc,T_nmax,binding_energy)
                # Compute T_av as Eq. 18 in Micelotta et al. (2010b)
                T_av = np.sqrt(T_0 * T_nmax)
                
                # Now obtain the dissociation probability
                P = dissociation_probability(binding_energy,Nc,T_av)
                
                # Multiple all to return to the J_theta array
                J_theta[j] = sigma * P * np.sin(theta[j])

        # Integrate over theta with the trapezoid method
        J_theta = trapezoid(J_theta, theta)
        
        # Add all to the J_v array
        J_v[i] = J_theta * mb_factor * v[i]
    # 5. Integrate J_v with the trapezoid method
    J = trapezoid(J_v,v) * au2cgs_v**2. * (a_0**2.)
    
    return J
            

def wrapper_electronic_rate(args):
    Ti, particle_type, Nc, binding_energy, nbins_v, nbins_theta, radius_method, ion_charge, pah_charge, phi_eV = args
    return Ti, electronic_destruction_rate_T(
        Ti, particle_type, Nc, binding_energy, nbins_v, nbins_theta, radius_method,
        ion_charge=ion_charge, pah_charge=pah_charge, phi_eV=phi_eV
    )


def _adaptive_nbins_by_temperature(T, Tmin, Tmax, nbins_v,
                                   adaptive_nbins_v=True,
                                   nbins_v_min=None,
                                   nbins_v_power=1.0):
    """Return temperature-dependent velocity-bin counts."""

    T = np.asarray(T, dtype=float)
    if nbins_v_min is None:
        nbins_v_min = max(32, int(0.35 * nbins_v))
    nbins_v_min = int(max(8, nbins_v_min))
    nbins_v_max = int(max(nbins_v_min, nbins_v))

    if not adaptive_nbins_v:
        return np.full(len(T), nbins_v_max, dtype=int)

    logT = np.log10(T)
    logT_min = np.log10(Tmin)
    logT_max = np.log10(Tmax)
    if np.isclose(logT_min, logT_max):
        frac = np.zeros_like(logT)
    else:
        frac = (logT - logT_min) / (logT_max - logT_min)
    frac = np.clip(frac, 0.0, 1.0)
    frac = frac**float(max(1e-8, nbins_v_power))

    return np.rint(nbins_v_min + frac * (nbins_v_max - nbins_v_min)).astype(int)


def _build_electronic_kernel_lookup(particle_type, Nc,
                                    binding_energy=E_0,
                                    nbins_v_lookup=500,
                                    nbins_theta=30,
                                    radius_method='Draine21',
                                    ion_charge=1,
                                    pah_charge=0,
                                    phi_eV=None):
    """Precompute the expensive electronic sputtering kernel as a function of velocity."""

    # 1. Determine interaction details that depend on particle type
    if particle_type == 'e':
        m_particle = 9.10938e-28  # [g]
        int_type = 'electron'
    elif particle_type == 'H':
        m_particle = 1.673557e-24  # [g]
        int_type = 'ion'
    elif particle_type == 'He':
        m_particle = 6.6464731e-24  # [g]
        int_type = 'ion'
    elif particle_type == 'C':
        m_particle = 1.9944733e-23  # [g]
        int_type = 'ion'
    elif particle_type == 'O':
        m_particle = 2.6566962e-23  # [g]
        int_type = 'ion'
    else:
        raise NameError(f'This particle_type is not included in this model: {particle_type}')

    # 2. PAH geometric parameters [a.u.]
    R_cm = _pah_radius_cm_from_Nc(Nc, radius_method=radius_method)
    R = R_cm / a_0
    d = thickness / a_0
    n_max = float(int(Nc / 5))

    if phi_eV is None:
        E_charge = coulomb_energy_shift_eV(ion_charge, pah_charge, R_cm)
    else:
        E_charge = float(phi_eV)

    # 3. Velocity range [a.u.]
    v_0 = np.sqrt(2.0 * binding_energy * eV2erg / m_particle) / au2cgs_v
    v_max = np.sqrt(2.0 * 1e5 * eV2erg / m_particle) / au2cgs_v

    v = np.logspace(np.log10(v_0), np.log10(v_max), int(max(64, nbins_v_lookup)))
    K_v = np.zeros_like(v)

    theta = np.linspace(0.0, np.pi / 2.0, nbins_theta)
    sigma_theta = np.array([cross_section_geometrical(R, d, th) for th in theta], dtype=float)
    sin_theta = np.sin(theta)

    if int_type == 'electron':
        E_charge_electron = coulomb_energy_shift_eV(-1, pah_charge, R_cm)
        ion_loss_theta = None
    else:
        E_charge_electron = 0.0
        ion_loss_theta = np.array([
            _ion_collision_theta_loss_integral(R, d, th, particle_type)
            for th in theta
        ], dtype=float)

    for i in range(len(v)):
        vi = v[i]
        Ei = (0.5 * m_particle * (vi * au2cgs_v)**2.0) / eV2erg
        J_theta = np.zeros(nbins_theta)

        for j in range(nbins_theta):
            sigma = sigma_theta[j]

            if int_type == 'electron':
                Ei_eff = Ei + E_charge_electron
                if Ei_eff > e_sp_min:
                    T_0_local = electronic_electron_collision(R, d, theta[j], Ei_eff)
                else:
                    J_theta[j] = 0.0
                    continue
            else:
                Ei_eff = Ei + E_charge
                if Ei_eff <= 0.0:
                    J_theta[j] = 0.0
                    continue
                vi_eff = np.sqrt(2.0 * Ei_eff * eV2erg / m_particle) / au2cgs_v
                T_0_local = 27.2116 * vi_eff * ion_loss_theta[j]

            T_nmax = T_0_local - n_max * Delta_epsilon
            if T_nmax <= 0.0 or T_nmax <= 0.2 * binding_energy:
                J_theta[j] = 0.0
            else:
                T0_eff = effective_temperature(Nc, T_0_local, binding_energy)
                Tn_eff = effective_temperature(Nc, T_nmax, binding_energy)
                T_av = np.sqrt(T0_eff * Tn_eff)
                P = dissociation_probability(binding_energy, Nc, T_av)
                J_theta[j] = sigma * P * sin_theta[j]

        # Keep only T-independent kernel piece used in the Maxwellian integral.
        K_v[i] = np.trapezoid(J_theta, theta) * vi

    return v, K_v, m_particle


def _integrate_electronic_kernel_vs_T(T, v_lookup, K_lookup, m_particle, nbins_v_by_T):
    """Integrate precomputed electronic kernel against Maxwell-Boltzmann for each T."""

    T = np.asarray(T, dtype=float)
    v_lookup = np.asarray(v_lookup, dtype=float)
    K_lookup = np.asarray(K_lookup, dtype=float)
    logv_lookup = np.log10(v_lookup)

    J = np.zeros(len(T), dtype=float)
    for i, Ti in enumerate(T):
        nvi = int(max(8, nbins_v_by_T[i]))
        v = np.logspace(logv_lookup[0], logv_lookup[-1], nvi)
        K_v = np.interp(np.log10(v), logv_lookup, K_lookup, left=0.0, right=K_lookup[-1])
        mb_factor = Maxwell_Boltzmann_function(v * au2cgs_v, m_particle, Ti)
        J_v = K_v * mb_factor
        J[i] = np.trapezoid(J_v, v) * au2cgs_v**2.0 * (a_0**2.0)

    return J


def _build_nuclear_kernel_lookup(M1, M2, Z1, Z2, threshold_E, m_particle,
                                 nbins_v_lookup=500,
                                 ion_charge=1,
                                 pah_charge=0,
                                 pah_radius_cm=None,
                                 phi_eV=None):
    """Precompute nuclear sputtering kernel as function of velocity [cm/s]."""

    v_0 = np.sqrt(2.0 * threshold_E * eV2erg / m_particle)
    v_max = np.sqrt(2.0 * 1e5 * eV2erg / m_particle)
    v = np.logspace(np.log10(v_0), np.log10(v_max), int(max(64, nbins_v_lookup)))

    K_v = np.zeros_like(v)
    if phi_eV is None:
        if pah_radius_cm is None:
            E_charge = 0.0
        else:
            E_charge = coulomb_energy_shift_eV(ion_charge, pah_charge, pah_radius_cm)
    else:
        E_charge = float(phi_eV)

    for i in range(len(v)):
        Ei = (0.5 * m_particle * v[i]**2.0) / eV2erg
        Ei_eff = Ei + E_charge
        if Ei_eff <= 0.0:
            cross_section = 0.0
        else:
            cross_section = energy_transfer_cross_section(M1, M2, Z1, Z2, Ei_eff, threshold_E) * 1e-16
        K_v[i] = cross_section * v[i]

    return v, K_v


def _integrate_nuclear_kernel_vs_T(T, v_lookup, K_lookup, m_particle, nbins_v_by_T):
    """Integrate precomputed nuclear kernel against Maxwell-Boltzmann for each T."""

    T = np.asarray(T, dtype=float)
    v_lookup = np.asarray(v_lookup, dtype=float)
    K_lookup = np.asarray(K_lookup, dtype=float)
    logv_lookup = np.log10(v_lookup)

    J = np.zeros(len(T), dtype=float)
    for i, Ti in enumerate(T):
        nvi = int(max(8, nbins_v_by_T[i]))
        v = np.logspace(logv_lookup[0], logv_lookup[-1], nvi)
        K_v = np.interp(np.log10(v), logv_lookup, K_lookup, left=0.0, right=K_lookup[-1])
        mb_factor = Maxwell_Boltzmann_function(v, m_particle, Ti)
        J[i] = np.trapezoid(mb_factor * K_v, v)

    return J
            
def electronic_destruction_rate(Tmin,Tmax,particle_type,Nc,
                                binding_energy=E_0,nT=100,
                                nbins_v=100,nbins_theta=30,
                                radius_method='Draine21',
                                ion_charge=1,
                                pah_charge=0,
                                phi_eV=None,
                                adaptive_nbins_v=True,
                                nbins_v_min=None,
                                nbins_v_power=1.0,
                                use_kernel_lookup=True,
                                nbins_v_lookup=500):
    
    T = np.logspace(np.log10(Tmin),np.log10(Tmax),nT)
    
    nbins_v_by_T = _adaptive_nbins_by_temperature(
        T, Tmin, Tmax, nbins_v,
        adaptive_nbins_v=adaptive_nbins_v,
        nbins_v_min=nbins_v_min,
        nbins_v_power=nbins_v_power,
    )

    if use_kernel_lookup:
        v_lookup, K_lookup, m_particle = _build_electronic_kernel_lookup(
            particle_type, Nc,
            binding_energy=binding_energy,
            nbins_v_lookup=nbins_v_lookup,
            nbins_theta=nbins_theta,
            radius_method=radius_method,
            ion_charge=ion_charge,
            pah_charge=pah_charge,
            phi_eV=phi_eV,
        )
        J = _integrate_electronic_kernel_vs_T(T, v_lookup, K_lookup, m_particle, nbins_v_by_T)
        return np.array(T), np.array(J)

    J = np.zeros(nT)
    num_cores = os.cpu_count()
    print(f"    Number of cores available: {num_cores}")
    args_list = [(Ti, particle_type, Nc, binding_energy, int(nbins_v_by_T[i]), nbins_theta, radius_method,
                  ion_charge, pah_charge, phi_eV)
                 for i, Ti in enumerate(T)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(tqdm(executor.map(wrapper_electronic_rate, args_list), total=nT,
                            desc=f'    Calculating electronic {particle_type} rates', unit=' steps'))

    T, J = zip(*results)
    return np.array(T), np.array(J)

# Nuclear collisions

def nuclear_destruction_rate_T(Tgas,M1,M2,Z1,Z2,threshold_E,m_particle,nbins_v=100,
                               ion_charge=1,
                               pah_charge=0,
                               pah_radius_cm=None,
                               phi_eV=None):
    """Destruction rate per C atom of a PAH for nuclear collisions at temperature T.

    Args:
        Tgas (float): gas temperature in K
        M1 (float): mass of incident particle in amu
        M2 (float): mass of target particle in amu
        Z1 (int): atomic number of incident particle
        Z2 (int): atomic number of target particle
        threshold_E (float): threshold energy for the PAH dissociation in eV
        m_particle (float): particle mass in g
        nbins_v (int, optional): number of velocity bins for the Maxwellian integral. Defaults to 100.

    Returns:
        float: destruction rate in units of cm^3/s
    """    
    
    from scipy.integrate import trapezoid
    
    # 1. Compute minimum velocity based on threshold energy
    v_0 = np.sqrt(2. * threshold_E * eV2erg / m_particle) # [cm/s]
    
    # 2. We set the maximum velocity to the thermal energy of gas at ~1e9 K
    v_max = np.sqrt(2. * 1e5 * eV2erg / m_particle) # [cm/s]
    
    # 3. Perform loop over velocity range
    v = np.logspace(np.log10(v_0),np.log10(v_max),nbins_v)
    J_v = np.zeros(nbins_v)

    if phi_eV is None:
        if pah_radius_cm is None:
            E_charge = 0.0
        else:
            E_charge = coulomb_energy_shift_eV(ion_charge, pah_charge, pah_radius_cm)
    else:
        E_charge = float(phi_eV)
    
    for i in range(0, nbins_v):
        vi = v[i]
        Ei = (0.5 * m_particle * v[i]**2.) / eV2erg
        mb_factor = Maxwell_Boltzmann_function(vi,m_particle,Tgas)
        Ei_eff = Ei + E_charge
        if Ei_eff <= 0.0:
            cross_section = 0.0
        else:
            cross_section = energy_transfer_cross_section(M1,M2,Z1,Z2,Ei_eff,threshold_E) * 1e-16 # [cm^2]
        J_v[i] = mb_factor * cross_section * vi

    # 4. Integrate J_v with the trapezoid method
    J = trapezoid(J_v,v)
    
    return J

def wrapper_nuclear_rate(args):
    Tgas,M1,M2,Z1,Z2,threshold_E,m_particle,nbins_v,ion_charge,pah_charge,pah_radius_cm,phi_eV = args
    return Tgas,nuclear_destruction_rate_T(
         Tgas, M1, M2, Z1, Z2, threshold_E, m_particle, nbins_v,
         ion_charge=ion_charge, pah_charge=pah_charge,
         pah_radius_cm=pah_radius_cm, phi_eV=phi_eV
    ) 

def nuclear_destruction_rate(Tmin,Tmax,particle_type,Nc,
                             threshold_energy=7.5,
                             nT=100,nbins_v=1000,
                             radius_method='Draine21',
                             ion_charge=1,
                             pah_charge=0,
                             phi_eV=None,
                             adaptive_nbins_v=True,
                             nbins_v_min=None,
                             nbins_v_power=1.0,
                             use_kernel_lookup=True,
                             nbins_v_lookup=1000):
    """Nuclear destruction rate for a range of temperatures, particle type, threshold energy
    and PAH number of carbon atoms.

    Args:
        Tmin (float): minimum temperature in K
        Tmax (float): maximum temperature in K
        particle_type (str): particle type, either 'H', 'He' or 'C'
        Nc (int): number of carbon atoms in PAH
        threshold_energy (float): threshold energy for the PAH dissociation in eV. Defaults to 7.5
        nT (int, optional): number of temperature bins to consider in the range. Defaults to 100.
        nbins_v (int, optional): number of velocity bins to use for the Maxwellian integral. Defaults to 1000.

    Raises:
        NameError: in case the particle_type given is not specified, the function fails

    Returns:
        (float,float): temperature array [K], destruction rate [cm^3/s]
    """    
    
    # 1. Get the specific particle details
    if particle_type == 'H':
        particle_am = 1.00784 # [u]
        Z1 = 1
    elif particle_type == 'He':
        particle_am = 4.002602 # [u]
        Z1 = 2
    elif particle_type == 'C':
        particle_am = 12.0107 # [u]
        Z1 = 6
    elif particle_type == 'O':
        particle_am = 15.999 # [u]
        Z1 = 8
    else:
        raise NameError(f'This particle_type is not included in this model: {particle_type}')
    
    M1 = particle_am # [amu]
    M2 = 12.0107 # [amu] - carbon target
    Z2 = 6 # atomic number of carbon
    m_particle = particle_am * 1.66053906660e-24 # [g]
    pah_radius_cm = _pah_radius_cm_from_Nc(Nc, radius_method=radius_method)
    
    T = np.logspace(np.log10(Tmin),np.log10(Tmax),nT)
    
    nbins_v_by_T = _adaptive_nbins_by_temperature(
        T, Tmin, Tmax, nbins_v,
        adaptive_nbins_v=adaptive_nbins_v,
        nbins_v_min=nbins_v_min,
        nbins_v_power=nbins_v_power,
    )

    if use_kernel_lookup:
        v_lookup, K_lookup = _build_nuclear_kernel_lookup(
            M1, M2, Z1, Z2, threshold_energy, m_particle,
            nbins_v_lookup=nbins_v_lookup,
            ion_charge=ion_charge,
            pah_charge=pah_charge,
            pah_radius_cm=pah_radius_cm,
            phi_eV=phi_eV,
        )
        J = _integrate_nuclear_kernel_vs_T(T, v_lookup, K_lookup, m_particle, nbins_v_by_T)
        return np.array(T), np.array(J)

    J = np.zeros(nT)
    num_cores = os.cpu_count()
    print(f"    Number of cores available: {num_cores}")
    args_list = [(Ti, M1, M2, Z1, Z2, threshold_energy, m_particle, int(nbins_v_by_T[i]),
                  ion_charge, pah_charge, pah_radius_cm, phi_eV)
                 for i, Ti in enumerate(T)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(tqdm(executor.map(wrapper_nuclear_rate, args_list), total=nT,
                            desc=f'    Calculating nuclear {particle_type} rates', unit=' steps'))

    T, J = zip(*results)
    return np.array(T), np.array(J)


def nuclear_destruction_rate_mass_charge(Tmin, Tmax,
                                         ion_mass_amu, ion_atomic_number,
                                         Nc,
                                         threshold_energy=7.5,
                                         nT=100, nbins_v=1000,
                                         radius_method='Draine21',
                                         ion_charge=1,
                                         pah_charge=0,
                                         phi_eV=None,
                                         adaptive_nbins_v=True,
                                         nbins_v_min=None,
                                         nbins_v_power=1.0,
                                         use_kernel_lookup=True,
                                         nbins_v_lookup=1000):
    """Nuclear destruction rate for a generic ion defined by mass and atomic number."""

    M1 = float(ion_mass_amu)
    M2 = 12.0107  # [amu] carbon target
    Z1 = int(ion_atomic_number)
    Z2 = 6
    m_particle = M1 * 1.66053906660e-24
    pah_radius_cm = _pah_radius_cm_from_Nc(Nc, radius_method=radius_method)

    T = np.logspace(np.log10(Tmin), np.log10(Tmax), nT)
    nbins_v_by_T = _adaptive_nbins_by_temperature(
        T, Tmin, Tmax, nbins_v,
        adaptive_nbins_v=adaptive_nbins_v,
        nbins_v_min=nbins_v_min,
        nbins_v_power=nbins_v_power,
    )

    if use_kernel_lookup:
        v_lookup, K_lookup = _build_nuclear_kernel_lookup(
            M1, M2, Z1, Z2, threshold_energy, m_particle,
            nbins_v_lookup=nbins_v_lookup,
            ion_charge=ion_charge,
            pah_charge=pah_charge,
            pah_radius_cm=pah_radius_cm,
            phi_eV=phi_eV,
        )
        J = _integrate_nuclear_kernel_vs_T(T, v_lookup, K_lookup, m_particle, nbins_v_by_T)
        return np.array(T), np.array(J)

    args_list = [(Ti, M1, M2, Z1, Z2, threshold_energy, m_particle, int(nbins_v_by_T[i]),
                  ion_charge, pah_charge, pah_radius_cm, phi_eV)
                 for i, Ti in enumerate(T)]
    num_cores = os.cpu_count()
    print(f"    Number of cores available: {num_cores}")
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(tqdm(executor.map(wrapper_nuclear_rate, args_list), total=nT,
                            desc='    Calculating generic-ion nuclear rates', unit=' steps'))

    T, J = zip(*results)
    return np.array(T), np.array(J)


def plot_nuclear_phi_influence(Tmin, Tmax,
                               ion_mass_amu, ion_atomic_number, ion_charge,
                               RPAH_micron,
                               pah_charge_min=-1, pah_charge_max=2,
                               nT=100, nphi=None,
                               threshold_energy=7.5,
                               radius_method='Draine21',
                               nbins_v=1000,
                               adaptive_nbins_v=True,
                               nbins_v_min=None,
                               nbins_v_power=1.0,
                               use_kernel_lookup=True,
                               nbins_v_lookup=1000):
    """Explore how nuclear destruction changes over a phi (eV) grid for a fixed ion."""

    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    RPAH_cm = float(RPAH_micron) * 1e-4

    # Discrete phi values mapped from integer PAH charges in [Zmin, Zmax].
    pah_charge_grid = np.arange(int(pah_charge_min), int(pah_charge_max) + 1, dtype=int)
    if 0 not in pah_charge_grid:
        pah_charge_grid = np.sort(np.append(pah_charge_grid, 0))
    phi_grid = np.array([
        coulomb_energy_shift_eV(ion_charge, zpah, RPAH_cm) for zpah in pah_charge_grid
    ], dtype=float)

    Nc = PAHs_model.Nc_from_size(RPAH_micron * 1e4)
    T = np.logspace(np.log10(Tmin), np.log10(Tmax), nT)
    rates = np.zeros((nT, len(phi_grid)), dtype=float)

    for j, phi_j in enumerate(phi_grid):
        _, J = nuclear_destruction_rate_mass_charge(
            Tmin, Tmax,
            ion_mass_amu=ion_mass_amu,
            ion_atomic_number=ion_atomic_number,
            Nc=Nc,
            threshold_energy=threshold_energy,
            nT=nT,
            nbins_v=nbins_v,
            radius_method=radius_method,
            ion_charge=ion_charge,
            pah_charge=0,
            phi_eV=float(phi_j),
            adaptive_nbins_v=adaptive_nbins_v,
            nbins_v_min=nbins_v_min,
            nbins_v_power=nbins_v_power,
            use_kernel_lookup=use_kernel_lookup,
            nbins_v_lookup=nbins_v_lookup,
        )
        rates[:, j] = J

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=250)

    positive = rates[rates > 0.0]
    if positive.size > 0 and np.max(positive) > np.min(positive):
        norm = mcolors.LogNorm(vmin=float(np.min(positive)), vmax=float(np.max(positive)))
        img = ax1.imshow(rates, origin='lower', aspect='auto',
                         extent=[phi_grid[0], phi_grid[-1], np.log10(T[0]), np.log10(T[-1])],
                         cmap='magma', norm=norm)
    else:
        img = ax1.imshow(rates, origin='lower', aspect='auto',
                         extent=[phi_grid[0], phi_grid[-1], np.log10(T[0]), np.log10(T[-1])],
                         cmap='magma')
    ax1.set_xlabel('phi [eV]')
    ax1.set_ylabel(r'log$_{10}(T)$ [K]')
    ax1.set_title('Nuclear destruction rate')
    cb = fig.colorbar(img, ax=ax1)
    cb.set_label(r'$J_{\rm nuc}$ [cm$^3$ s$^{-1}$]')

    for j, zpah in enumerate(pah_charge_grid):
        ax2.plot(T, rates[:, j], label=f'Z_PAH={zpah}, phi={phi_grid[j]:.2e} eV')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('T [K]')
    ax2.set_ylim([1e-17,1e-4])
    ax2.set_ylabel(r'$J_{\rm nuc}$ [cm$^3$ s$^{-1}$]')
    ax2.set_title('Slices vs temperature')
    ax2.legend(frameon=False, fontsize=10)

    fig.tight_layout()
    return fig, T, phi_grid, rates

# Export functions
def export_rates(RPAH,Tmin,Tmax,threshold_energy=7.5,
                 binding_energy=E_0,nT=100,
                 nbins_v=100,nbins_theta=50,
                 radius_method='Draine21',
                 pah_charge=0,
                 pah_charge_states=(-1, 0, 1, 2),
                 ion_charge_states=None,
                 ion_charge_ranges=None,
                 adaptive_nbins_v=True,
                 nbins_v_min=None,
                 nbins_v_power=1.0,
                 use_kernel_lookup=True,
                 nbins_v_lookup=1000,
                 plot_rates=True,nH_plot=1.0,
                 Z_plot=1,plot_phi_curves=False):
    start_time = time.time()
    print(40*"-")
    print('PAH SPUTTERING IN A HOT GAS')
    print('By: F. Rodriguez Montero (2024)')
    print('1. Obtaining rates...')
    if radius_method == 'Draine21':
        # Nc_from_size expects Angstrom, while RPAH input here is in micron.
        Nc = PAHs_model.Nc_from_size(RPAH * 1e4)
        RPAH = RPAH * 1e-4  # [cm]
        print(f'The selected PAH radius corresponds to {Nc} carbon atoms')
    elif radius_method == 'Omont86':
        Nc = (RPAH/0.9)**2.
        RPAH = RPAH * 1e-4  # [cm]
        print(f'The selected PAH radius corresponds to {Nc} carbon atoms')
    else:
        raise NameError(f'This radius_method is not included in this model: {radius_method}')

    RPAH_micron = RPAH / 1e-4

    if ion_charge_states is None:
        ion_charge_states = {'H': 1, 'He': 1, 'C': 1, 'O': 1}

    if ion_charge_ranges is None:
        ion_charge_ranges = {k: [int(v)] for k, v in ion_charge_states.items()}

    # Build a global discrete phi grid from all requested ion-charge ranges and PAH-charge states.
    all_ion_charges = []
    for ptype in friction_params.keys():
        values = ion_charge_ranges.get(ptype, [int(ion_charge_states.get(ptype, 1))])
        if isinstance(values, int):
            values = list(range(1, int(values) + 1))
        for z in values:
            all_ion_charges.append(int(z))
    if len(all_ion_charges) == 0:
        all_ion_charges = [1]

    pah_charge_states = [int(z) for z in pah_charge_states]
    phi_grid = _build_phi_grid_from_charge_sets(all_ion_charges, pah_charge_states, RPAH)
    izero = int(np.argmin(np.abs(phi_grid)))
    phi_grid[izero] = 0.0
    print(f'    Using {len(phi_grid)} phi values from charge combinations.')

    
    # 1. Obtain the thermal electron destruction rate for all phi values
    print('    Electron sputtering...')
    J_electron_phi = np.zeros((len(phi_grid), nT), dtype=float)
    for jphi, phi_val in enumerate(phi_grid):
        T, J = electronic_destruction_rate(Tmin,Tmax,'e',Nc,
                                            binding_energy=binding_energy,
                                            nT=nT,nbins_v=nbins_v,
                                            nbins_theta=nbins_theta,
                                            radius_method=radius_method,
                                            ion_charge=-1,
                                            pah_charge=pah_charge,
                                            phi_eV=float(phi_val),
                                            adaptive_nbins_v=adaptive_nbins_v,
                                            nbins_v_min=nbins_v_min,
                                            nbins_v_power=nbins_v_power,
                                            use_kernel_lookup=use_kernel_lookup,
                                            nbins_v_lookup=nbins_v_lookup)
        J_electron_phi[jphi, :] = J
    J_electron = J_electron_phi[izero, :]

    # 2. Obtain the electronic excitation caused by ions
    J_electronic = {}
    J_electronic_phi = {}
    for ptype in friction_params.keys():
        print(f'    {ptype} electronic sputtering...')
        J_phi = np.zeros((len(phi_grid), nT), dtype=float)
        for jphi, phi_val in enumerate(phi_grid):
            T, J = electronic_destruction_rate(Tmin,Tmax,ptype,Nc,
                                                binding_energy=binding_energy,
                                                nT=nT,nbins_v=nbins_v,
                                                nbins_theta=nbins_theta,
                                                radius_method=radius_method,
                                                ion_charge=int(ion_charge_states.get(ptype, 1)),
                                                pah_charge=pah_charge,
                                                phi_eV=float(phi_val),
                                                adaptive_nbins_v=adaptive_nbins_v,
                                                nbins_v_min=nbins_v_min,
                                                nbins_v_power=nbins_v_power,
                                                use_kernel_lookup=use_kernel_lookup,
                                                nbins_v_lookup=nbins_v_lookup)
            J_phi[jphi, :] = J
        J_electronic_phi[ptype] = J_phi
        J_electronic[ptype] = J_phi[izero, :]
    
    # 3. Obtain the nuclear collision rate for ions
    J_ion = {}
    J_ion_phi = {}
    for ptype in friction_params.keys():
        print(f'    {ptype} nuclear sputtering...')
        J_phi = np.zeros((len(phi_grid), nT), dtype=float)
        for jphi, phi_val in enumerate(phi_grid):
            T, J = nuclear_destruction_rate(Tmin,Tmax,ptype,Nc,
                                            threshold_energy=threshold_energy,
                                            nT=nT,nbins_v=nbins_v,
                                            radius_method=radius_method,
                                            ion_charge=int(ion_charge_states.get(ptype, 1)),
                                            pah_charge=pah_charge,
                                            phi_eV=float(phi_val),
                                            adaptive_nbins_v=adaptive_nbins_v,
                                            nbins_v_min=nbins_v_min,
                                            nbins_v_power=nbins_v_power,
                                            use_kernel_lookup=use_kernel_lookup,
                                            nbins_v_lookup=nbins_v_lookup)
            J_phi[jphi, :] = 0.5 * Nc * J
        J_ion_phi[ptype] = J_phi
        J_ion[ptype] = J_phi[izero, :]
    
    print(' => Rates done!')
    
    print('2. Preparing data to save in files...')
    
    # 4. Save electronic rate into single file
    PAH_dir = './PAH_sputtering_data'
    if not os.path.exists(PAH_dir):
        os.mkdir(PAH_dir)
    electron_filename = PAH_dir+'/electron_sputtering_%.4f_micron_PAH'%(RPAH_micron)
    f = open(electron_filename, 'w', encoding="utf-8")
    f.write("{:8d}".format(nT)+'\n')
    for i in range(0,nT):
        f.write("{:14.6e}".format(np.log10(T[i]))+'\n')
    for i in range(0,nT):
        if J_electron[i] == 0.0:
            J_electron[i] = np.min(J_electron[J_electron>0.0])
        # We already multiply the rate constant by 2 as electronic
        # collisions results in the loss of two Carbon atoms
        f.write("{:14.6e}".format(np.log10(2.*J_electron[i]))+'\n')
    f.close()
    print(f'    Electron sputtering file saved in: {electron_filename}')
    
    # 5. Save rates caused by ion collisions (electronic + nuclear)
    for ptype in friction_params.keys():
        ion_filename =  PAH_dir+'/%s_sputtering_%.4f_micron_PAH'%(ptype,RPAH_micron)
        f = open(ion_filename, 'w', encoding="utf-8")
        f.write("{:8d}".format(nT)+'\n')
        for i in range(0,nT):
            f.write("{:14.6e}".format(np.log10(T[i]))+'\n')
        for i in range(0,nT):
            if 2.*J_electronic[ptype][i]+J_ion[ptype][i] == 0.0:
                J_electronic[ptype][i] = np.min(J_electronic[ptype][J_electronic[ptype]>0.0])
                J_ion[ptype][i] = np.min(J_ion[ptype][J_ion[ptype]>0.0])
            # We already multiply the rate constant by 2 as electronic
            # collisions results in the loss of two Carbon atoms
            f.write("{:14.6e}".format(np.log10(2.*J_electronic[ptype][i]+J_ion[ptype][i]))+'\n')
        f.close()
        print(f'    {ptype} sputtering file saved in: {ion_filename}')

        # Save full T-phi table for this ion species.
        ion_tphi_filename = PAH_dir+'/%s_sputtering_Tphi_%.4f_micron_PAH'%(ptype,RPAH_micron)
        total_phi = 2.0 * J_electronic_phi[ptype] + J_ion_phi[ptype]
        total_phi = np.maximum(total_phi, 1e-60)
        with open(ion_tphi_filename, 'w', encoding='utf-8') as ft:
            ft.write(f"{nT:8d} {len(phi_grid):8d}\n")
            ft.write(' '.join([f'{phi:.8e}' for phi in phi_grid]) + '\n')
            for iT, Ti in enumerate(T):
                row = [f'{np.log10(Ti):.8e}'] + [f'{np.log10(total_phi[j, iT]):.8e}' for j in range(len(phi_grid))]
                ft.write(' '.join(row) + '\n')
        print(f'    {ptype} T-phi sputtering table saved in: {ion_tphi_filename}')
        
    # 6. If it has been said, plot the resulting rates in a single graph
    # so they can be visually inspected
    if plot_rates:
        print('3. Plotting results in a single plot...')
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_theme(style="white")
        sns.color_palette("Paired")
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": "Computer Modern Roman",
        })
        fig, ax = plt.subplots(1,1, figsize=(6,4),dpi=300,facecolor='w',edgecolor='k',sharey=True)
        
        # Compute number densities for fully ionized gas
        # Assuming solar abundances: He/H ~ 0.1, C/H ~ 3e-4, O/H ~ 6e-4
        n_H = nH_plot  # [cm^-3]
        n_He = 0.1 * Z_plot * n_H  # [cm^-3]
        n_C = 3e-4 * Z_plot * n_H  # [cm^-3]
        n_O = 6e-4 * Z_plot * n_H  # [cm^-3]
        n_e = n_H  # [cm^-3]
        
        # Multiply rates by densities to get sputtering rates in 1/s
        total_rate = 2.*J_electron * n_e
        ax.plot(T, 2.*J_electron * n_e, label='Electrons', linestyle='-', linewidth=2.5)
        for i, ptype in enumerate(friction_params.keys()):
            if ptype == 'H':
                n_species = n_H
            elif ptype == 'He':
                n_species = n_He
            elif ptype == 'C':
                n_species = n_C
            elif ptype == 'O':
                n_species = n_O
            ax.plot(T, 2.*J_electronic[ptype] * n_species, label=f'{ptype} electronic', linestyle=':', linewidth=2.5)
            ax.plot(T, J_ion[ptype] * n_species, label=f'{ptype} nuclear', linestyle='-.', linewidth=2.5)
            total_rate += 2.*J_electronic[ptype] * n_species + J_ion[ptype] * n_species
        ax.plot(T, total_rate, linestyle='--', color='k', linewidth=3)

        # Add additional total-rate curves for nonzero phi with low opacity.
        total_rate_phi = np.zeros((len(phi_grid), nT), dtype=float)
        for jphi in range(len(phi_grid)):
            total_j = 2.0 * J_electron * n_e
            for ptype in friction_params.keys():
                if ptype == 'H':
                    n_species = n_H
                elif ptype == 'He':
                    n_species = n_He
                elif ptype == 'C':
                    n_species = n_C
                else:
                    n_species = n_O
                total_j += 2.0 * J_electronic_phi[ptype][jphi, :] * n_species + J_ion_phi[ptype][jphi, :] * n_species
            total_rate_phi[jphi, :] = total_j

        # If plot_phi_curves is enabled, add all individual phi curves for each species
        if plot_phi_curves:
            # Add electron curves for all phi values
            for jphi in range(len(phi_grid)):
                if jphi != izero:
                    ax.plot(T, 2.0 * J_electron_phi[jphi, :] * n_e, linestyle='-', linewidth=1.5, alpha=0.25, color='C0')
            # Add electronic and nuclear curves for each ion species for all phi values
            color_idx = 1
            for i, ptype in enumerate(friction_params.keys()):
                if ptype == 'H':
                    n_species = n_H
                elif ptype == 'He':
                    n_species = n_He
                elif ptype == 'C':
                    n_species = n_C
                elif ptype == 'O':
                    n_species = n_O
                # Add electronic curves for this species across all phi
                for jphi in range(len(phi_grid)):
                    if jphi != izero:
                        ax.plot(T, 2.0 * J_electronic_phi[ptype][jphi, :] * n_species, linestyle=':', linewidth=1.5, alpha=0.25, color=f'C{color_idx}')
                # Add nuclear curves for this species across all phi
                for jphi in range(len(phi_grid)):
                    if jphi != izero:
                        ax.plot(T, J_ion_phi[ptype][jphi, :] * n_species, linestyle='-.', linewidth=1.5, alpha=0.25, color=f'C{color_idx+1}')
                color_idx += 2
        else:
            # Original behavior: only add total-rate curves for nonzero phi
            for jphi in range(len(phi_grid)):
                if jphi == izero:
                    continue
                ax.plot(T, total_rate_phi[jphi, :], linestyle='--', color='k', linewidth=2.0, alpha=0.25)
        ax.tick_params(labelsize=14)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.minorticks_on()
        ax.tick_params(which='both',axis="both",direction="in")
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.legend(loc='upper left',frameon=False,ncol=2,fontsize=12)
        ax.set_ylabel(r'Sputtering Rate [C-atom/s]',fontsize=16)
        ax.set_xlabel(r'Gas Temperature [K]',fontsize=16)
        ax.set_ylim([1e-17,1e-2])
        
        # Add text showing nH and metallicity
        ax.text(0.95, 0.05, r'$n_{\rm H} = %.2f$ cm$^{-3}$' % nH_plot + '\n' + r'$Z = %.2f$ Z$_{\odot}$' % Z_plot,
                            verticalalignment='bottom', horizontalalignment='right',
                            transform=ax.transAxes, fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        fig.subplots_adjust(top=0.96,bottom=0.14,left=0.14,right=0.99,hspace=0,wspace=0)
        plot_name =  PAH_dir+f'/PAH_{Nc}_thermal_sputtering_rates.pdf'
        fig.savefig(plot_name, format='pdf', dpi=300)
        print(f'    Plot saved in: {plot_name}')

    end_time = time.time()
    print(f'This all took {end_time - start_time:.2f} seconds')
    print('All done!!')
    print(40*"-")


def export_rates_simple(RPAH, Tmin, Tmax, threshold_energy=7.5,
                        binding_energy=E_0, nT=100,
                        nbins_v=100, nbins_theta=50,
                        radius_method='Draine21',
                        pah_charge=0,
                        pah_label='default',
                        output_dir='./PAH_sputtering_data',
                        adaptive_nbins_v=True,
                        nbins_v_min=None,
                        nbins_v_power=1.0,
                        use_kernel_lookup=True,
                        nbins_v_lookup=1000):
    """Export sputtering rates without phi dependence to Fortran90-compatible tables.
    
    This function computes and saves destruction rates for phi=0 only, with one file per ion species
    (H, He, C, O). Output format is Fortran90-compatible and easy to read.
    
    Args:
        RPAH (float): PAH radius in microns
        Tmin (float): minimum temperature in K
        Tmax (float): maximum temperature in K
        threshold_energy (float, optional): threshold for nuclear collisions in eV. Defaults to 7.5.
        binding_energy (float, optional): binding energy in eV. Defaults to E_0.
        nT (int, optional): number of temperature bins. Defaults to 100.
        nbins_v (int, optional): velocity bins. Defaults to 100.
        nbins_theta (int, optional): angular bins. Defaults to 50.
        radius_method (str, optional): 'Draine21' or 'Omont86'. Defaults to 'Draine21'.
        pah_charge (int, optional): PAH charge state. Defaults to 0.
        pah_label (str, optional): label to identify this PAH model (used in filenames). Defaults to 'default'.
        output_dir (str, optional): directory where export files are written.
            Defaults to './PAH_sputtering_data'.
        adaptive_nbins_v (bool, optional): adaptive velocity binning. Defaults to True.
        nbins_v_min (int, optional): minimum velocity bins.
        nbins_v_power (float, optional): power for adaptive binning. Defaults to 1.0.
        use_kernel_lookup (bool, optional): use lookup tables. Defaults to True.
        nbins_v_lookup (int, optional): lookup table velocity bins. Defaults to 1000.
    
    Returns:
        tuple: (T, J_electron, J_electronic, J_ion) where J_* are dicts for each species.
    """
    
    start_time = time.time()
    print(40*"-")
    print('PAH SPUTTERING RATES (Simple, no phi dependence)')
    print('By: F. Rodriguez Montero (2024)')
    print('1. Obtaining rates...')
    
    if radius_method == 'Draine21':
        Nc = PAHs_model.Nc_from_size(RPAH * 1e4)
        RPAH_cm = RPAH * 1e-4  # [cm]
        print(f'The selected PAH radius corresponds to {Nc} carbon atoms')
    elif radius_method == 'Omont86':
        Nc = (RPAH/0.9)**2.
        RPAH_cm = RPAH * 1e-4  # [cm]
        print(f'The selected PAH radius corresponds to {Nc} carbon atoms')
    else:
        raise NameError(f'This radius_method is not included in this model: {radius_method}')

    RPAH_micron = RPAH_cm / 1e-4

    # 1. Electron sputtering rate (phi=0)
    print('    Electron sputtering...')
    T, J_electron = electronic_destruction_rate(
        Tmin, Tmax, 'e', Nc,
        binding_energy=binding_energy,
        nT=nT, nbins_v=nbins_v,
        nbins_theta=nbins_theta,
        radius_method=radius_method,
        ion_charge=-1,
        pah_charge=pah_charge,
        phi_eV=0.0,
        adaptive_nbins_v=adaptive_nbins_v,
        nbins_v_min=nbins_v_min,
        nbins_v_power=nbins_v_power,
        use_kernel_lookup=use_kernel_lookup,
        nbins_v_lookup=nbins_v_lookup)

    # 2. Ion electronic and nuclear rates (phi=0)
    J_electronic = {}
    J_ion = {}
    for ptype in friction_params.keys():
        print(f'    {ptype} electronic sputtering...')
        _, J_elec = electronic_destruction_rate(
            Tmin, Tmax, ptype, Nc,
            binding_energy=binding_energy,
            nT=nT, nbins_v=nbins_v,
            nbins_theta=nbins_theta,
            radius_method=radius_method,
            ion_charge=1,
            pah_charge=pah_charge,
            phi_eV=0.0,
            adaptive_nbins_v=adaptive_nbins_v,
            nbins_v_min=nbins_v_min,
            nbins_v_power=nbins_v_power,
            use_kernel_lookup=use_kernel_lookup,
            nbins_v_lookup=nbins_v_lookup)
        J_electronic[ptype] = J_elec

        print(f'    {ptype} nuclear sputtering...')
        _, J_nuc = nuclear_destruction_rate(
            Tmin, Tmax, ptype, Nc,
            threshold_energy=threshold_energy,
            nT=nT, nbins_v=nbins_v,
            radius_method=radius_method,
            ion_charge=1,
            pah_charge=pah_charge,
            phi_eV=0.0,
            adaptive_nbins_v=adaptive_nbins_v,
            nbins_v_min=nbins_v_min,
            nbins_v_power=nbins_v_power,
            use_kernel_lookup=use_kernel_lookup,
            nbins_v_lookup=nbins_v_lookup)
        J_ion[ptype] = 0.5 * Nc * J_nuc
    
    print(' => Rates done!')
    
    print('2. Saving Fortran90-compatible tables...')
    
    # Create output directory
    PAH_dir = str(output_dir)
    if not os.path.exists(PAH_dir):
        os.mkdir(PAH_dir)
    
    # Mapping from particle type to atomic number
    atomic_number_map = {
        'e': 0,  # electron (special case for Z=0)
        'H': 1,
        'He': 2,
        'C': 6,
        'O': 8
    }
    
    from models.grain_size_config import get_header_lines

    # Save electron rates (Z=0)
    Z_electron = atomic_number_map['e']
    electron_filename = f'{PAH_dir}/pah_sputtering_{pah_label}_Z_{Z_electron}'
    headers = get_header_lines(
        title="PAH Sputtering rates (Simple, no phi dependence)",
        script_name="models/PAH_gas_collisions/PAH_sputtering.py",
        bin_info=f"PAH Bin: {pah_label}, Size: {RPAH_micron:.4e} micron, Nc: {Nc}, Species: electrons (Z=0)",
        val_desc="Values: log10(rate [cm^3 s^-1])",
        num_lines=6
    )
    with open(electron_filename, 'w', encoding='utf-8') as f:
        for line in headers:
            f.write(f"{line}\n")
        # Header: nT
        f.write(f'{nT:8d}\n')
        # Temperature values (log10)
        for Ti in T:
            f.write(f'{np.log10(Ti):14.6e}\n')
        # Rate values (log10)
        for i in range(nT):
            rate = 2.0 * J_electron[i] if J_electron[i] > 0.0 else 1e-100
            f.write(f'{np.log10(rate):14.6e}\n')
    print(f'    Electron (Z=0) rates saved: {electron_filename}')
    
    # Save ion rates (electronic + nuclear)
    for ptype in friction_params.keys():
        Z = atomic_number_map[ptype]
        ion_filename = f'{PAH_dir}/pah_sputtering_{pah_label}_Z_{Z}'
        headers = get_header_lines(
            title="PAH Sputtering rates (Simple, no phi dependence)",
            script_name="models/PAH_gas_collisions/PAH_sputtering.py",
            bin_info=f"PAH Bin: {pah_label}, Size: {RPAH_micron:.4e} micron, Nc: {Nc}, Species: {ptype} (Z={Z})",
            val_desc="Values: log10(rate [cm^3 s^-1])",
            num_lines=6
        )
        with open(ion_filename, 'w', encoding='utf-8') as f:
            for line in headers:
                f.write(f"{line}\n")
            # Header: nT
            f.write(f'{nT:8d}\n')
            # Temperature values (log10)
            for Ti in T:
                f.write(f'{np.log10(Ti):14.6e}\n')
            # Rate values (log10)
            for i in range(nT):
                total_rate = 2.0 * J_electronic[ptype][i] + J_ion[ptype][i]
                if total_rate > 0.0:
                    f.write(f'{np.log10(total_rate):14.6e}\n')
                else:
                    f.write(f'{np.log10(1e-100):14.6e}\n')
        print(f'    {ptype} (Z={Z}) rates saved: {ion_filename}')
    
    end_time = time.time()
    print(f'Completed in {end_time - start_time:.2f} seconds')
    print('All done!!')
    print(40*"-")
    
    return T, J_electron, J_electronic, J_ion

    
def Lindhard_reduced_energy(M1,M2,Z1,Z2,a,E):
    """Lindhard reduced energy for nuclear collisions.

    Args:
        M1 (float): mass of incident particle in amu
        M2 (float): mass of target particle in amu
        Z1 (int): atomic number of incident particle
        Z2 (int): atomic number of target particle
        a (float): screening length in Angstrom
        E (float): kinetic energy of incident particle in eV
    """    
    
    epsilon = (M2/(M1+M2)) * (a/(Z1*Z2*14.39)) * E
    return epsilon

def ZBL_screening_length(Z1,Z2):
    """ZBL screening length.

    Args:
        Z1 (int): atomic number of incident particle
        Z2 (int): atomic number of target particle

    Returns:
        float: screening length in Angstrom
    """    
    
    a = 0.8854 * 0.529 / (Z1**(0.23) + Z2**(0.23))
    return a

def ZBL_reduced_nuclear_stopping_crosssection(epsilon):
    """ZBL reduced nuclear stopping cross section.

    Args:
        epsilon (float): Lindhard reduced energy

    Returns:
        float: reduced nuclear stopping cross section in Angstrom^2
    """    
    if epsilon <= 30:
        S_n = 0.5 * np.log(1.+1.1383*epsilon) / (epsilon + 0.01321 * epsilon**0.21226 + 0.19593 * epsilon**0.5)
    else:
        S_n = np.log(epsilon) / (2. * epsilon)
    return S_n

def Ziegler1985_m(epsilon):
    """Ziegler et al. (1985) expression for quantity m.

    Args:
        epsilon (float or array): Lindhard reduced energy
    Returns:
        float or array: quantity m [dimensionless]
    """
    # Base 10^-9 reduced energy constant
    eps1 = 1e-9

    # Compute u = 0.1 * ln(epsilon/eps1)
    u = 0.1 * np.log(epsilon / eps1)

    # Coefficients from Ziegler et al. (1985), via Micelotta et al. (2010)
    a = np.array([-2.432, -0.1509, 2.648, -2.742, 1.215, -0.1665])

    # Build the polynomial
    # Note: use u**i, not (0.1*x**i)
    X = sum(a[i] * u**i for i in range(len(a)))

    # Final expression
    m = 1.0 - np.exp(-np.exp(X))

    return m

def energy_transfer_cross_section(M1,M2,Z1,Z2,E,threshold_E):
    """Energy transfer cross section for nuclear collisions based on
    the ZBL theory.

    Args:
        M1 (float): mass of incident particle in amu
        M2 (float): mass of target particle in amu
        Z1 (int): atomic number of incident particle
        Z2 (int): atomic number of target particle
        E (float): kinetic energy of incident particle in eV
        threshold_E (float): threshold energy for the PAH dissociation in eV
    Returns:
        float: energy transfer cross section in Angstrom^2
    """
    if E < threshold_E:
        return 0.0
    # 1. Compute the screening length
    a = ZBL_screening_length(Z1,Z2)

    # 2. Compute the Lindhard reduced energy
    epsilon = Lindhard_reduced_energy(M1,M2,Z1,Z2,a,E)

    # 3. Compute the reduced nuclear stopping cross section
    S_n = ZBL_reduced_nuclear_stopping_crosssection(epsilon)

    # 4. Compute the quantity m
    m = Ziegler1985_m(epsilon)

    # 5. Compute the energy transfer cross section
    mu = M1 / (M1 + M2)
    gamma = 4. * M1 * M2 / (M1 + M2)**2.
    S_m = S_n * (1. - m)/m / (gamma * E)
    E_thres = (threshold_E / E)**(-m) - 1.
    sigma_E = 4. * np.pi * a * Z1 * Z2 * 14.39 * mu * S_m * E_thres

    return sigma_E