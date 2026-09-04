"""
DUST SPUTTERING MODEL

This script holds the details to recompute the thermal and non-thermal
sputtering of regular dust grains (i.e. no PAHs), following a model built
together by Kirchschlager and collaborators, based on the original modelling
by Tielens and Nozawa. A version of this model is summarised by the fitting
functions to the sputtering yields by Chia-Yu Hu to the results of Nozawa et
al. (2006), which is the model already implemented in Dusty-PRISM and the 
RAMSES-YOMP version of Yohan Dubois with dust. 

By: F. Rodriguez Montero (currodri@gmail.com)

Modification history:

- (06/03/2024): original implementation of the thermal sputtering (F. Rodriguez Montero)

"""

# Import libraries
import numpy as np
import pycalima.models.dust_model as dust_model
import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
import time

try:
    from numba import njit
    _NUMBA_AVAILABLE = True
except Exception:
    _NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def _identity(func):
            return func
        return _identity

# Set OMP_NUM_THREADS to limit the number of threads used by OpenBLAS
os.environ["OMP_NUM_THREADS"] = "1"  # Set it to the desired number of threads
os.environ['OPENBLAS_NUM_THREADS'] = '1'

def _sputtering_output_dir():
    """Directory for generated thermal-sputtering tables.

    Resolved on every call rather than frozen at import, because it depends on
    $CALIMA_DATA and on the active configuration's model_name.
    """
    from pycalima.models.grain_size_config import get_model_data_dir
    return get_model_data_dir() / 'thermal_sputtering_data'

# Constants
a_0              = 5.291e-9 # [cm] - atomic length unit
eV2erg           = 1.6021773300241e-12 # [erg] conversion between eV to erg
au2cgs_v         = 2.18769126364e8    # [cm/s] conversion between a.u. velocity and cgs velocity
au2cgs_m         = 1.66053906660e-24 # [g] conversion between a.u. mass and cgs mass
sec2yr           = 3.1536e7 # [s] conversion between yr to sec
kb               = 1.3806488e-16 # [erg/K] - Boltzmann constant
elem_charge      = 4.8032047e-10 # [statC] - elementary charge
c                = 2.99792458e10 # [cm/s] - speed of light
U0               = {'C': 4.0, 'Sil': 5.7} # [eV] - surface binding energy
Ksput            = {'C': 0.65, 'Sil': 0.1} # [] - free parameter fitting to the observed yields
size_correction_fitparams  = { # [] - fitting to the results of Bocchio et al. (2006) for the size correction
                    #       to the sputtering yields for the semi-infinite target approximation
    'C'          : {
        'p1'     : 4.9,
        'p2'     : 0.55,
        'p3'     : 0.77,
        'p4'     : 4.7,
        'p5'     : 3.0,
        'p6'     : 1.2,
        'alpha_P': -4.73,
        'a'      : 4.51,
        'b'      : 0.92,
        'c'      : 0.4,
        'E_exec' : 13.5
    },
    'MgSiO4'        : {
        'p1'     : 1.5,
        'p2'     : 1.2,
        'p3'     : 0.57,
        'p4'     : 1.1,
        'p5'     : 0.52,
        'p6'     : 0.37,
        'alpha_P': -3.34,
        'a'      : 1.48,
        'b'      : 1.31,
        'c'      : 0.59,
        'E_exec' : 13.0
        
    },
    'MgSiO3'        : {
        'p1'     : 1.0,
        'p2'     : 0.5,
        'p3'     : 1.0,
        'p4'     : 1.8,
        'p5'     : 2.1,
        'p6'     : 0.76,
        'alpha_P': -3.34,
        'a'      : 1.6,
        'b'      : 1.22,
        'c'      : 0.5,
        'E_exec' : 13.0
    }
}

# Functions

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

def penetration_depth(zion,alphaP,E):
    """
    Fitting function to the penetration depth by Kirchschlager et al. (2019).

    Args:
        zion (int): ion charge
        alphap (float): composition-specific fitting value
        E (float): initial energy of the ion [eV]

    Returns:
        float: penetration depth [cm]
    """    
    
    return 10.**(2.8 * float(zion)**(-0.21) + alphaP) * E * 1e-7

def bocchio_correction(x,fit_params):
    """
    Fitting function to the size correction to the semi-infinite yields,
    using the results of Bocchio et al. (2016).

    Args:
        x (float): ratio of grain radius to (0.7 * penetration depth)
        fit_params (dict): dictionary holding the fitting parameters

    Returns:
        float: correction factor to semi-infinite yield
    """    
    if x >= 1:
        f1 = fit_params['p1'] * np.exp(-(np.log(x/fit_params['p2']))**2. / (2. * fit_params['p3']**2.))
        f2 = -fit_params['p4'] * np.exp(-(fit_params['p5']*x - fit_params['p6'])**2.)
        return 1. + f1 + f2
    else:
        return fit_params['a'] * np.exp(-(np.log(x/fit_params['b']))**2. / (2. * fit_params['c']**2.))

def alpha_energy(mu):
    """
    Approximation factor to the energy redistribution upon impact.

    Args:
        mu (float): ratio of the mass numbers of target atom and incident ion

    Returns:
        float: energy redistribution factor
    """    
    
    
    if mu <= 0.5:
        alpha = 0.2
    elif 0.5 < mu <= 1.0:
        alpha = 0.1 / mu + 0.25 * (mu - 0.5)**2.
    else:
        alpha = 0.3 * (mu - 0.6)**(2./3.)
    
    return alpha

@njit(cache=True)
def screening_length(Zi,Zd):
    """
    Ion screening length within the dust material.

    Args:
        Zi (int): atomic number of the incident ion
        Zd (int): average atomic number of the dust material

    Returns:
        float: screening length in cm
    """    
    return 0.885 * a_0 / np.sqrt(float(Zi)**(2./3.) + float(Zd)**(2./3.))

@njit(cache=True)
def reduced_energy(dust_atomic_mass,ion_atomic_mass,
                   screen_length,Zi,Zd,E):
    """Reduced energy equation.

    Args:
        dust_atomic_mass (float): dust material average atomic mass [a.u.]
        ion_atomic_mass (float): incident ion atomic mass [a.u.]
        screen_length (float): ion screening length [cm]
        Zi (int): atomic number of the incident ion
        Zd (int): average atomic number of the dust material
        E (float): initial ion energy [erg]

    Returns:
        float: reduced energy [dimensionless]
    """    
    return dust_atomic_mass / (dust_atomic_mass + ion_atomic_mass) *\
            screen_length / (float(Zi * Zd) * elem_charge**2) * E
            
@njit(cache=True)
def screened_Coulomb_function(epsilon):
    """
    Screened Coulomb interaction approximation (Matsunami et al. 1980).

    Args:
        epsilon (float): reduced energy

    Returns:
        float: screened Coulomb factor
    """    
    # s1 = 3.441 * np.sqrt(epsilon) * np.log(epsilon + 2.718)
    # s2 = 1. + 6.35 * np.sqrt(epsilon) + epsilon * (-1.708 + 6.882 * np.sqrt(epsilon))
    
    if epsilon <= 30.:
        s1 = 0.5 * np.log(1.+1.1383*epsilon)
        s2 = epsilon + 0.01321*epsilon**0.21226 + 0.19593*np.sqrt(epsilon)
    
        return s1 / s2  
    else:
        return np.log(epsilon)/(2.*epsilon)      

def threshold_energy(surface_energy,dust_atomic_mass,ion_atomic_mass):
    """
    Minimum energy required for an ion to be able to sputter a dust atom.

    Args:
        surface_energy (float): dust surface energy [eV]
        dust_atomic_mass (float): average dust atomic mass [a.u.]
        ion_atomic_mass (float): ion atomic mass [a.u.]

    Returns:
        float: threshold energy
    """    
    
    g = 4. * ion_atomic_mass * dust_atomic_mass / (dust_atomic_mass + ion_atomic_mass)**2.
    inv_mu = ion_atomic_mass / dust_atomic_mass
    
    if inv_mu <= 0.3:
        Esp = surface_energy / (g * (1. - g))
    else:
        Esp = 8. * surface_energy * inv_mu**(1./3.)
        
    return Esp

def sputtering_yield(dust_radius,surface_energy,Kparam,rho_dust,
                        dust_atomic_mass,ion_atomic_mass,
                        dust_atomic_number,ion_atomic_number,ion_charge,
                        bocchio_fit_params,E_exec,ion_energy,
                        do_size_correction=False,
                        dust_charge=0):
    """Sputtering yields for ion collisions of a particular energy and charge.

    Args:
        dust_radius (float): grain radius [cm]
        surface_energy (float): surface binding energy [eV]
        Kparam (float): free parameter to the experimental yields
        dust_atomic_mass (float): average atomic mass for the dust material [a.u.]
        ion_atomic_mass (float): atomic mass of the ion [a.u.]
        dust_atomic_number (int): average atomic number for the dust material
        ion_atomic_number (int): ion atomic number
        ion_charge (int): ionisation charge of the ion
        bocchio_fit_params (dict): chosen parameters for the Bocchio fitting curve to the size correction
        alphaP (float): fitting parameter to the penetration depth
        ion_energy (float): initial energy of incident ion [eV]
        do_size_correction (bool, optional): whether or not to use the size correction by Bocchio. Defaults to False.

    Returns:
        float: sputtering yield
    """ 
    
    # 1. Determine the threshold energy for sputtering
    E_sp = threshold_energy(surface_energy,dust_atomic_mass,ion_atomic_mass)
    
    if dust_charge != 0:
        E_charge = (ion_charge * dust_charge * elem_charge**2. / dust_radius) / eV2erg
        # if (E_charge/ion_energy)>0.99:
        #     if ion_energy + E_charge > E_sp:
        #         print('Particle moved to high E')
        ion_energy = ion_energy - E_charge
    if ion_energy < E_sp:
        return 0.0
        
    if do_size_correction:
        # 2. Compute the ion penetration depth
        rp = compute_penetration_depth(ion_energy*eV2erg,ion_atomic_number*au2cgs_m,rho_dust,ion_atomic_number,dust_atomic_number,
                                        dust_atomic_mass*au2cgs_m)
        # 3. Compute the size-dependent correction
        x = dust_radius / (0.7 * rp)
        # print(bocchio_correction(x,size_correction_fitparams['MgSiO4']),bocchio_correction(x,size_correction_fitparams['MgSiO3']))
        size_correction = max(0.0, bocchio_correction(x,bocchio_fit_params))
    
    # 4. Compute the alpha factor for enery redistribution
    mu = dust_atomic_mass/ion_atomic_mass
    alpha = alpha_energy(dust_atomic_mass/ion_atomic_mass)
    
    # 5. Compute the nuclear stopping cross section
    a_sc = screening_length(ion_atomic_number,dust_atomic_number)
    eps  = reduced_energy(dust_atomic_mass,ion_atomic_mass,
                          a_sc,ion_atomic_number,
                          dust_atomic_number,ion_energy*eV2erg)
    s_factor = screened_Coulomb_function(eps)
    
    # 6. Obtain the semi-infinite yield
    Y = 3.56 / surface_energy * ion_atomic_mass / (ion_atomic_mass + dust_atomic_mass) * \
        ion_atomic_number * dust_atomic_number / np.sqrt(ion_atomic_number**(2./3.)+dust_atomic_number**(2./3.)) *\
            alpha /  (Kparam * mu + 1.) * s_factor * (1. - (E_sp/ion_energy)**(2./3.)) * (1. - E_sp/ion_energy)**2.
    
    # 7. Multiply by the size correction (if requested)
    if do_size_correction:
        Y = size_correction * Y

    return Y

def average_yields_T(args):
    """Computation of the average yield for a particular grain
    and ion at a given temperature T, assuming a Maxwell-Boltzmann
    distribution averaging.

    Args:
        args (tuple): arguments required by the sputtering_yield function

    Returns:
        float: final average(Y*v) yield [atom/ion]
    """    
    
    dust_radius,surface_energy,Kparam,rho_dust,\
        dust_atomic_mass,ion_atomic_mass,\
        dust_atomic_number,ion_atomic_number,\
        ion_charge,bocchio_fit_params,E_exec,Tgas,\
        nbins_v,do_size_correction,do_dust_charge = args
    
    # 1. Determine the minumum velocity for reaching the threshold energy
    E_sp = threshold_energy(surface_energy,dust_atomic_mass,ion_atomic_mass)
    v_0  = np.sqrt(2. * E_sp * eV2erg) # [cm/s]
    while Maxwell_Boltzmann_function(v_0,ion_atomic_mass*au2cgs_m,Tgas) < 1e-20:
        v_0 = 2 * v_0
    
    # 2. We set the maximum velocity to the thermal energy of gas at ~1e11 K
    v_max = np.sqrt(2. * 1e7 * eV2erg / (ion_atomic_mass*au2cgs_m)) # [cm/s]
    while Maxwell_Boltzmann_function(v_max,ion_atomic_mass*au2cgs_m,Tgas) < 1e-25:
        v_max = v_max/ 2.0
    # print(0.5 * ion_atomic_mass * au2cgs_m *v_0**2./ eV2erg,0.5 * ion_atomic_mass * au2cgs_m *v_max**2./ eV2erg)
    # 3. Perform loop over velocity range
    v = np.logspace(np.log10(v_0),np.log10(v_max),nbins_v)
    Y_v = np.zeros(nbins_v)
    
    # 4. Compute the mean dust charge
    if do_dust_charge:
        if dust_atomic_number == 6:
            Zmean = dust_model.grain_mean_charge(1.0,Tgas,0.1,'graphite',str(int(dust_radius*1e7))+'A')
        else:
            Zmean = dust_model.grain_mean_charge(1.0,Tgas,0.1,'silicate',str(int(dust_radius*1e7))+'A')
    else:
        Zmean = 0.0
    n_size_corr = 0
    for i in range(0, nbins_v):
        vi = v[i] # [cm/s]
        Ei = 0.5 * ion_atomic_mass * au2cgs_m * vi**2. # [erg]
        Ei = Ei / eV2erg
        mb_factor = Maxwell_Boltzmann_function(vi,ion_atomic_mass*au2cgs_m,Tgas)
        Y = sputtering_yield(dust_radius,surface_energy,Kparam,rho_dust,
                                dust_atomic_mass,ion_atomic_mass,
                                dust_atomic_number,ion_atomic_number,ion_charge,
                                bocchio_fit_params,E_exec,Ei,
                                do_size_correction,Zmean)
        Y_v[i] = mb_factor * vi * Y
        if Y == 0.0:
            n_size_corr += 1

    # 5. Integrate Y_v with the trapezoid method
    Y0 = np.trapezoid(Y_v, v)
    # print(Tgas,n_size_corr/nbins_v)
    return Y0
   
   
def total_erosion_rate(Tmin,Tmax,dust_type,
                         ion_atomic_masses,
                         ion_atomic_numbers,
                         ion_charges,
                         ion_abundances,
                         nT=100,nbins_v=100,
                         do_size_correction=False,
                         do_dust_charge=False):
    """_summary_

    Args:
        Tmin (float): minimum gas temperature
        Tmax (float): maximum gas temperature
        dust_type (str): dust type string
        ion_atomic_masses (np.ndarray): array with the ion atomic masses
        ion_atomic_numbers (np.ndarray): array with the ion atomic numbers
        ion_charges (np.ndarray): array with the ion charges
        ion_abundances (np.ndarray): array with the ion abundances over hydrogen
        nT (int, optional): number of temperature bins. Defaults to 100.
        nbins_v (int, optional): number of velocity bins. Defaults to 100.
        do_size_correction (bool, optional): whether or not to use the size correction. Defaults to False.

    Raises:
        NameError: if dust_type specified is not included in the model

    Returns:
        (np.ndarray,np.ndarray): gas temperature [K] and erosion rate [micron / yr * cm^3] 
    """    
    
    # 1. Prepare the dust grain properties
    if dust_type == 'smallC':
        a_dust = dust_model.basic_a0[2]*1e-4 # [cm]
        m_dust = 4./3. * np.pi * (a_dust)**2. * dust_model.basic_s[2] # [g]
        rho_dust = dust_model.basic_s[2] # [g/cm^3]
        am_dust = 12.011 # [a.u.]
        an_dust = 6
        E_exec = size_correction_fitparams['C']['E_exec']
        bocchio_fitparams = size_correction_fitparams['C']
        Kparam = Ksput['C']
        surface_energy = U0['C']
    elif dust_type == 'largeC':
        a_dust = dust_model.basic_a0[3]*1e-4 # [cm]
        m_dust = 4./3. * np.pi * (a_dust)**2. * dust_model.basic_s[3] # [g]
        rho_dust = dust_model.basic_s[3] # [g/cm^3]
        am_dust = 12.011 # [a.u.]
        an_dust = 6
        E_exec = size_correction_fitparams['C']['E_exec']
        bocchio_fitparams = size_correction_fitparams['C']
        Kparam = Ksput['C']
        surface_energy = U0['C']
    elif dust_type == 'smallSil':
        # Assuming MgFeSiO4 composition (olivine with Iron inclusions as the regular model)
        a_dust = dust_model.basic_a0[5]*1e-4 # [cm]
        m_dust = 4./3. * np.pi * (a_dust)**2. * dust_model.basic_s[5] # [g]
        rho_dust = dust_model.basic_s[5] # [g/cm^3]
        am_dust = (24.305 + 55.845 + 28.0855 + 4*15.999) / 7. # [a.u.]
        an_dust = int((4*8 + 14 + 26 + 12) / 7)
        E_exec = size_correction_fitparams['MgSiO4']['E_exec']
        bocchio_fitparams = size_correction_fitparams['MgSiO4']
        Kparam = Ksput['Sil']
        surface_energy = U0['Sil']
    elif dust_type == 'largeSil':
        # Assuming MgFeSiO4 composition (olivine with Iron inclusions as the regular model)
        a_dust = dust_model.basic_a0[6]*1e-4 # [cm]
        m_dust = 4./3. * np.pi * (a_dust)**2. * dust_model.basic_s[6] # [g]
        rho_dust = dust_model.basic_s[6] # [g/cm^3]
        am_dust = (24.305 + 55.845 + 28.0855 + 4*15.999) / 7. # [a.u.]
        an_dust = int((4*8 + 14 + 26 + 12) / 7)
        E_exec = size_correction_fitparams['MgSiO4']['E_exec']
        bocchio_fitparams = size_correction_fitparams['MgSiO4']
        Kparam = Ksput['Sil']
        surface_energy = U0['Sil']
    else:
        raise NameError(f'This dust type is not included in the present model: {dust_type}')
    
    # 2. Set the temperature and erosion rate arrays
    Tgas = np.logspace(np.log10(Tmin),np.log10(Tmax),nT)
    Y_tot = np.zeros(nT)
    
    # 3. Loop over temperatures with parallel processing
    print(f"Computing the erosion rates for the dust type: {dust_type}")
    num_cores = 5 #os.cpu_count()
    print(f"    Number of cores available: {num_cores}")
    for i in range(0, len(ion_abundances)):
        # Create argument list for parallel processing
        args_list = [(a_dust,surface_energy,Kparam,rho_dust,\
                        am_dust,ion_atomic_masses[i],\
                        an_dust,ion_atomic_numbers[i],\
                        ion_charges[i],bocchio_fitparams,E_exec,Ti,\
                        nbins_v,do_size_correction,\
                        do_dust_charge) for Ti in Tgas]
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
            results = list(tqdm(executor.map(average_yields_T, args_list), 
                                total=nT, 
                                desc=f'    Calculating erosion rates for ion mass {ion_atomic_masses[i]}',
                                unit=' steps'))

        Y_temp = np.array(results)
        
        Y_tot = Y_tot + ion_abundances[i] * Y_temp

    # 4. Obtain the final erosion rate in [microns / yr * cm^3]
    Y_tot = (am_dust * au2cgs_m) / (2. * rho_dust) * Y_tot * (1e4 * sec2yr)
    
    return a_dust, Tgas, Y_tot

def export_rates(Tmin,Tmax,ion_atomic_masses,
                 ion_atomic_numbers,ion_charges,
                 ion_abundances,nT=100,nbins_v=100,
                 label=''):
    
    # 1. Crete the directory for the table data
    table_dir = str(_sputtering_output_dir())
    os.makedirs(table_dir, exist_ok=True)
    
    # 2. Compute the rate for small carbonaceous grains
    a_dust, Tgas, Y_smallC = total_erosion_rate(Tmin,Tmax,'smallC',
                                            ion_atomic_masses,
                                            ion_atomic_numbers,
                                            ion_charges,
                                            ion_abundances,
                                            nT,nbins_v,
                                            True,False)
    smallC_filename = table_dir+'/thermal_sputtering_%.4f_micron_Gra'%(a_dust/1e-4)
    f = open(smallC_filename, 'w', encoding="utf-8")
    f.write("{:8d}".format(nT)+'\n')
    for i in range(0,nT):
        f.write("{:14.6e}".format(np.log10(Tgas[i]))+'\n')
    for i in range(0,nT):
        if Y_smallC[i] == 0.0:
            Y_smallC[i] = np.min(Y_smallC[Y_smallC>0.0])
        f.write("{:14.6e}".format(np.log10(Y_smallC[i]))+'\n')
    f.close()
    
    # 3. Compute the rate for large carbonaceous grains
    a_dust, Tgas, Y_largeC = total_erosion_rate(Tmin,Tmax,'largeC',
                                            ion_atomic_masses,
                                            ion_atomic_numbers,
                                            ion_charges,
                                            ion_abundances,
                                            nT,nbins_v,
                                            True,False)
    largeC_filename = table_dir+'/thermal_sputtering_%.4f_micron_Gra'%(a_dust/1e-4)
    f = open(largeC_filename, 'w', encoding="utf-8")
    f.write("{:8d}".format(nT)+'\n')
    for i in range(0,nT):
        f.write("{:14.6e}".format(np.log10(Tgas[i]))+'\n')
    for i in range(0,nT):
        if Y_largeC[i] == 0.0:
            Y_largeC[i] = np.min(Y_largeC[Y_largeC>0.0])
        f.write("{:14.6e}".format(np.log10(Y_largeC[i]))+'\n')
    f.close()
    
    # 4. Compute the rate for small silicate grains
    a_dust, Tgas, Y_smallSil = total_erosion_rate(Tmin,Tmax,'smallSil',
                                            ion_atomic_masses,
                                            ion_atomic_numbers,
                                            ion_charges,
                                            ion_abundances,
                                            nT,nbins_v,
                                            True,False)
    smallSil_filename = table_dir+'/thermal_sputtering_%.4f_micron_Sil'%(a_dust/1e-4)
    f = open(smallSil_filename, 'w', encoding="utf-8")
    f.write("{:8d}".format(nT)+'\n')
    for i in range(0,nT):
        f.write("{:14.6e}".format(np.log10(Tgas[i]))+'\n')
    for i in range(0,nT):
        if Y_smallSil[i] == 0.0:
            Y_smallSil[i] = np.min(Y_smallSil[Y_smallSil>0.0])
        f.write("{:14.6e}".format(np.log10(Y_smallSil[i]))+'\n')
    f.close()
    
    # 5. Compute the rate for large silicate grains
    a_dust, Tgas, Y_largeSil = total_erosion_rate(Tmin,Tmax,'largeSil',
                                            ion_atomic_masses,
                                            ion_atomic_numbers,
                                            ion_charges,
                                            ion_abundances,
                                            nT,nbins_v,
                                            True,False)
    largeSil_filename = table_dir+'/thermal_sputtering_%.4f_micron_Sil'%(a_dust/1e-4)
    f = open(largeSil_filename, 'w', encoding="utf-8")
    f.write("{:8d}".format(nT)+'\n')
    for i in range(0,nT):
        f.write("{:14.6e}".format(np.log10(Tgas[i]))+'\n')
    for i in range(0,nT):
        if Y_largeSil[i] == 0.0:
            Y_largeSil[i] = np.min(Y_largeSil[Y_largeSil>0.0])
        f.write("{:14.6e}".format(np.log10(Y_largeSil[i]))+'\n')
    f.close()


def _nearest_grain_radius_label(a_dust_cm):
    """Map grain size in cm to the closest Ibanez-Mejias radius label."""

    radius_a = a_dust_cm * 1e8
    allowed_radii = np.array([3.5, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0])
    allowed_labels = np.array(['3.5A', '5A', '10A', '50A', '100A', '500A', '1000A'])
    idx = int(np.argmin(np.abs(allowed_radii - radius_a)))
    return str(allowed_labels[idx])


def _get_dust_sputtering_setup(dust_type):
    """Return a consistent set of sputtering parameters for a given dust type."""

    if dust_type == 'smallC':
        a_dust = dust_model.basic_a0[2] * 1e-4
        rho_dust = dust_model.basic_s[2]
        am_dust = 12.011
        an_dust = 6
        bocchio_fitparams = size_correction_fitparams['C']
        Kparam = Ksput['C']
        surface_energy = U0['C']
        grain_type = 'graphite'
    elif dust_type == 'largeC':
        a_dust = dust_model.basic_a0[3] * 1e-4
        rho_dust = dust_model.basic_s[3]
        am_dust = 12.011
        an_dust = 6
        bocchio_fitparams = size_correction_fitparams['C']
        Kparam = Ksput['C']
        surface_energy = U0['C']
        grain_type = 'graphite'
    elif dust_type == 'smallSil':
        a_dust = dust_model.basic_a0[5] * 1e-4
        rho_dust = dust_model.basic_s[5]
        am_dust = (24.305 + 55.845 + 28.0855 + 4.0 * 15.999) / 7.0
        an_dust = int((4 * 8 + 14 + 26 + 12) / 7)
        bocchio_fitparams = size_correction_fitparams['MgSiO4']
        Kparam = Ksput['Sil']
        surface_energy = U0['Sil']
        grain_type = 'silicate'
    elif dust_type == 'largeSil':
        a_dust = dust_model.basic_a0[6] * 1e-4
        rho_dust = dust_model.basic_s[6]
        am_dust = (24.305 + 55.845 + 28.0855 + 4.0 * 15.999) / 7.0
        an_dust = int((4 * 8 + 14 + 26 + 12) / 7)
        bocchio_fitparams = size_correction_fitparams['MgSiO4']
        Kparam = Ksput['Sil']
        surface_energy = U0['Sil']
        grain_type = 'silicate'
    else:
        raise NameError(f'This dust type is not included in the present model: {dust_type}')

    return {
        'a_dust': a_dust,
        'rho_dust': rho_dust,
        'am_dust': am_dust,
        'an_dust': an_dust,
        'bocchio_fitparams': bocchio_fitparams,
        'Kparam': Kparam,
        'surface_energy': surface_energy,
        'grain_type': grain_type,
        'grain_radius_label': _nearest_grain_radius_label(a_dust),
    }


def _get_composition_sputtering_setup(composition, grain_size_micron):
    """Return sputtering setup from composition and grain size only."""

    comp = str(composition).strip().lower()
    if comp not in ('graphite', 'silicate'):
        raise ValueError("composition must be 'graphite' or 'silicate'.")
    if grain_size_micron is None:
        raise ValueError("grain_size_micron must be provided when using composition-based setup.")

    a_dust = float(grain_size_micron) * 1e-4
    if a_dust <= 0.0:
        raise ValueError('grain_size_micron must be > 0.')

    if comp == 'graphite':
        rho_dust = 2.24
        am_dust = 12.011
        an_dust = 6
        bocchio_fitparams = size_correction_fitparams['C']
        Kparam = Ksput['C']
        surface_energy = U0['C']
        grain_type = 'graphite'
    else:
        rho_dust = 3.3
        am_dust = (24.305 + 55.845 + 28.0855 + 4.0 * 15.999) / 7.0
        an_dust = int((4 * 8 + 14 + 26 + 12) / 7)
        bocchio_fitparams = size_correction_fitparams['MgSiO4']
        Kparam = Ksput['Sil']
        surface_energy = U0['Sil']
        grain_type = 'silicate'

    return {
        'a_dust': a_dust,
        'rho_dust': rho_dust,
        'am_dust': am_dust,
        'an_dust': an_dust,
        'bocchio_fitparams': bocchio_fitparams,
        'Kparam': Kparam,
        'surface_energy': surface_energy,
        'grain_type': grain_type,
        'grain_radius_label': _nearest_grain_radius_label(a_dust),
    }


def _resolve_sputtering_setup(dust_type, composition=None, grain_size_micron=None):
    """Resolve sputtering setup from either legacy dust_type or composition+size."""

    if composition is not None:
        if dust_type is not None:
            raise ValueError('Use either dust_type or composition, not both.')
        return _get_composition_sputtering_setup(composition, grain_size_micron)

    if dust_type is None:
        raise ValueError('Either dust_type or composition must be provided.')

    setup = _get_dust_sputtering_setup(dust_type)
    if grain_size_micron is not None:
        setup['a_dust'] = float(grain_size_micron) * 1e-4
        setup['grain_radius_label'] = _nearest_grain_radius_label(setup['a_dust'])
    return setup


def average_yields_T_fixed_phi(args):
    """Average sputtering yield over Maxwell-Boltzmann velocities at fixed phi (in eV)."""

    dust_radius, surface_energy, Kparam, rho_dust, \
        dust_atomic_mass, ion_atomic_mass, \
        dust_atomic_number, ion_atomic_number, \
        bocchio_fit_params, E_exec, Tgas, \
        nbins_v, do_size_correction, phi = args

    E_sp = threshold_energy(surface_energy, dust_atomic_mass, ion_atomic_mass)
    v_0 = np.sqrt(2.0 * E_sp * eV2erg)
    while Maxwell_Boltzmann_function(v_0, ion_atomic_mass * au2cgs_m, Tgas) < 1e-20:
        v_0 = 2.0 * v_0

    v_max = np.sqrt(2.0 * 1e7 * eV2erg / (ion_atomic_mass * au2cgs_m))
    while Maxwell_Boltzmann_function(v_max, ion_atomic_mass * au2cgs_m, Tgas) < 1e-25:
        v_max = v_max / 2.0

    v = np.logspace(np.log10(v_0), np.log10(v_max), nbins_v)
    Y_v = np.zeros(nbins_v)

    # Electrostatic energy shift in eV.
    # With the adopted convention, phi > 0 means opposite grain/ion signs
    # (attractive interaction), so the ion gains energy near the grain.
    E_charge = phi

    for i in range(0, nbins_v):
        vi = v[i]
        Ei = 0.5 * ion_atomic_mass * au2cgs_m * vi**2.0 / eV2erg
        Ei_eff = Ei + E_charge

        mb_factor = Maxwell_Boltzmann_function(vi, ion_atomic_mass * au2cgs_m, Tgas)
        if Ei_eff <= 0.0:
            Y = 0.0
        else:
            Y = sputtering_yield(dust_radius, surface_energy, Kparam, rho_dust,
                                 dust_atomic_mass, ion_atomic_mass,
                                 dust_atomic_number, ion_atomic_number, 0,
                                 bocchio_fit_params, E_exec, Ei_eff,
                                 do_size_correction, 0.0)
        Y_v[i] = mb_factor * vi * Y

    return np.trapezoid(Y_v, v)


def average_yields_T_phi_batch(args):
    """Compute average yields for all phi values at fixed temperature.

    This avoids rebuilding the velocity grid and Maxwell-Boltzmann factors for
    every (T, phi) cell, which significantly reduces overhead.
    """

    dust_radius, surface_energy, Kparam, rho_dust, \
        dust_atomic_mass, ion_atomic_mass, \
        dust_atomic_number, ion_atomic_number, \
        bocchio_fit_params, E_exec, Tgas, \
        nbins_v, do_size_correction, phi_grid, \
        E_lookup, Y_lookup = args

    E_sp = threshold_energy(surface_energy, dust_atomic_mass, ion_atomic_mass)
    v_0 = np.sqrt(2.0 * E_sp * eV2erg)
    while Maxwell_Boltzmann_function(v_0, ion_atomic_mass * au2cgs_m, Tgas) < 1e-20:
        v_0 = 2.0 * v_0

    v_max = np.sqrt(2.0 * 1e7 * eV2erg / (ion_atomic_mass * au2cgs_m))
    while Maxwell_Boltzmann_function(v_max, ion_atomic_mass * au2cgs_m, Tgas) < 1e-25:
        v_max = v_max / 2.0

    v = np.logspace(np.log10(v_0), np.log10(v_max), nbins_v)
    Ei = 0.5 * ion_atomic_mass * au2cgs_m * v**2.0 / eV2erg
    mb_v = Maxwell_Boltzmann_function(v, ion_atomic_mass * au2cgs_m, Tgas) * v

    phi_grid = np.asarray(phi_grid, dtype=float)
    Y_phi = np.zeros(len(phi_grid), dtype=float)

    use_lookup = E_lookup is not None and Y_lookup is not None
    if use_lookup:
        E_lookup = np.asarray(E_lookup, dtype=float)
        Y_lookup = np.asarray(Y_lookup, dtype=float)

    for jphi, phi in enumerate(phi_grid):
        Ei_eff = Ei + phi
        valid = Ei_eff > 0.0

        if not np.any(valid):
            Y_phi[jphi] = 0.0
            continue

        Y_v = np.zeros(nbins_v, dtype=float)
        if use_lookup:
            Y_v[valid] = np.interp(Ei_eff[valid], E_lookup, Y_lookup, left=0.0, right=Y_lookup[-1])
        else:
            valid_idx = np.where(valid)[0]
            for idx in valid_idx:
                Y_v[idx] = sputtering_yield(
                    dust_radius, surface_energy, Kparam, rho_dust,
                    dust_atomic_mass, ion_atomic_mass,
                    dust_atomic_number, ion_atomic_number, 0,
                    bocchio_fit_params, E_exec, Ei_eff[idx],
                    do_size_correction, 0.0,
                )

        Y_phi[jphi] = np.trapezoid(mb_v * Y_v, v)

    return Y_phi


def _build_sputtering_yield_lookup(dust_radius, surface_energy, Kparam, rho_dust,
                                   dust_atomic_mass, ion_atomic_mass,
                                   dust_atomic_number, ion_atomic_number,
                                   bocchio_fit_params, E_exec,
                                   do_size_correction,
                                   E_min_eV, E_max_eV,
                                   nE_lookup=2048):
    """Precompute sputtering yield as a function of ion energy for fast interpolation."""

    E_min = max(float(E_min_eV), 1e-8)
    E_max = max(float(E_max_eV), E_min * 1.01)
    nE = max(64, int(nE_lookup))

    E_lookup = np.logspace(np.log10(E_min), np.log10(E_max), nE)
    Y_lookup = np.zeros(nE, dtype=float)

    for iE, Ei in enumerate(E_lookup):
        Y_lookup[iE] = sputtering_yield(
            dust_radius, surface_energy, Kparam, rho_dust,
            dust_atomic_mass, ion_atomic_mass,
            dust_atomic_number, ion_atomic_number, 0,
            bocchio_fit_params, E_exec, Ei,
            do_size_correction, 0.0,
        )

    return E_lookup, Y_lookup


def _grain_allowed_charge_bounds(grain_type, a_dust_cm, hnu_max_ev=13.6):
    """Get physically allowed grain-charge bounds (Zmin, Zmax) from dust_charging."""

    from pycalima.models.dust_charge.dust_charging import (
        graphite_work_function,
        silicate_work_function,
        most_negative_allowed_charge_graphite,
        most_negative_allowed_charge_silicate,
        most_positive_allowed_charge,
    )

    if grain_type == 'graphite':
        zmin = int(np.floor(most_negative_allowed_charge_graphite(a_dust_cm)))
        work_function = graphite_work_function
    else:
        zmin = int(np.floor(most_negative_allowed_charge_silicate(a_dust_cm)))
        work_function = silicate_work_function

    zmax = int(np.floor(most_positive_allowed_charge(a_dust_cm, work_function, hnu_max_ev)))
    if zmin > zmax:
        zmin, zmax = zmax, zmin

    return zmin, zmax


def _compute_phi_bounds(Tmin, Tmax, a_dust, Zk_min, Zk_max, grain_type, hnu_max_ev=13.6):
    """Compute phi bounds in eV using allowed grain-charge and ion-charge limits."""

    Zg_min, Zg_max = _grain_allowed_charge_bounds(grain_type, a_dust, hnu_max_ev=hnu_max_ev)

    phi_candidates = [0.0]
    for Zg in [Zg_min, Zg_max]:
        for Zk in [Zk_min, Zk_max]:
            # Convention: phi > 0 for opposite grain/ion charge signs.
            # phi is defined as the electrostatic energy shift in eV.
            phi_candidates.append(-Zk * Zg * elem_charge**2.0 / (a_dust * eV2erg))

    phi_min = float(np.min(phi_candidates))
    phi_max = float(np.max(phi_candidates))

    if np.isclose(phi_min, phi_max):
        dphi = max(1e-6, abs(phi_min) * 1e-3)
        phi_min -= dphi
        phi_max += dphi

    return phi_min, phi_max, Zg_min, Zg_max


def export_rates_T_phi(Tmin, Tmax, dust_type,
                       ion_atomic_masses,
                       ion_atomic_numbers,
                       Zk_min, Zk_max,
                       dustlabel,
                       grain_radius_micron=None,
                       composition=None,
                       hnu_max_ev=13.6,
                       nT=60, nphi=60,
                       nbins_v=100,
                       adaptive_nbins_v=True,
                       nbins_v_min=None,
                       nbins_v_power=1.0,
                       do_size_correction=True,
                       use_yield_lookup=True,
                       nE_lookup=2048,
                       label='',
                       executor=None,
                       phi_min=None,
                       phi_max=None):
    """Export sputtering tables on a (T, phi) grid and create a validation figure.

     The phi range is estimated from Zk_min/Zk_max and physically allowed grain
    charges (Zmin/Zmax) from dust_charging. phi is in eV.
    One table is saved per ion species inside thermal_sputtering_data.
    If use_yield_lookup is True, a precomputed Y(E) table is used to speed up integration.
    If adaptive_nbins_v is True, the velocity-bin count increases with temperature.

     Setup modes:
     1. Legacy: provide ``dust_type`` (smallC/largeC/smallSil/largeSil).
     2. General: set ``dust_type=None`` and provide ``composition``
         ('graphite' or 'silicate') and ``grain_radius_micron``.
    """

    ion_atomic_masses = np.asarray(ion_atomic_masses, dtype=float)
    ion_atomic_numbers = np.asarray(ion_atomic_numbers, dtype=int)

    dustlabel = str(dustlabel).strip()
    if dustlabel == '':
        raise ValueError('dustlabel must be a non-empty string.')

    if len(ion_atomic_masses) != len(ion_atomic_numbers):
        raise ValueError('ion_atomic_masses and ion_atomic_numbers must have the same length.')
    if len(ion_atomic_masses) == 0:
        raise ValueError('At least one ion species must be provided.')

    setup = _resolve_sputtering_setup(
        dust_type=dust_type,
        composition=composition,
        grain_size_micron=grain_radius_micron,
    )
    a_dust = setup['a_dust']
    rho_dust = setup['rho_dust']
    am_dust = setup['am_dust']
    an_dust = setup['an_dust']
    bocchio_fitparams = setup['bocchio_fitparams']
    Kparam = setup['Kparam']
    surface_energy = setup['surface_energy']
    grain_type = setup['grain_type']
    if grain_radius_micron is not None:
        a_dust = float(grain_radius_micron) * 1e-4
    phi_min_calc, phi_max_calc, Zg_min, Zg_max = _compute_phi_bounds(
        Tmin, Tmax, a_dust, Zk_min, Zk_max, grain_type, hnu_max_ev=hnu_max_ev
    )
    if phi_min is None or phi_max is None:
        phi_min = phi_min_calc
        phi_max = phi_max_calc
    else:
        phi_min = float(phi_min)
        phi_max = float(phi_max)

    Tgas = np.logspace(np.log10(Tmin), np.log10(Tmax), nT)

    if nbins_v_min is None:
        nbins_v_min = max(32, int(0.35 * nbins_v))
    nbins_v_min = int(max(8, nbins_v_min))
    nbins_v_max = int(max(nbins_v_min, nbins_v))

    if adaptive_nbins_v:
        logT = np.log10(Tgas)
        logT_min = np.log10(Tmin)
        logT_max = np.log10(Tmax)
        if np.isclose(logT_max, logT_min):
            frac = np.zeros_like(logT)
        else:
            frac = (logT - logT_min) / (logT_max - logT_min)
        frac = np.clip(frac, 0.0, 1.0)
        frac = frac**float(max(1e-8, nbins_v_power))
        nbins_v_by_T = np.rint(nbins_v_min + frac * (nbins_v_max - nbins_v_min)).astype(int)
    else:
        nbins_v_by_T = np.full(nT, nbins_v_max, dtype=int)

    # Build a strictly uniform phi grid containing exactly 0.0
    if phi_min < 0.0 and phi_max > 0.0:
        ratio = abs(phi_min) / (phi_max - phi_min)
        n_neg = int(np.round(ratio * (nphi - 1)))
        n_neg = max(1, min(nphi - 2, n_neg))
        n_pos = (nphi - 1) - n_neg
        dphi = max(abs(phi_min) / n_neg, phi_max / n_pos)
        phi_grid = np.arange(-n_neg, n_pos + 1) * dphi
    elif phi_min >= 0.0:
        dphi = phi_max / (nphi - 1)
        phi_grid = np.arange(0, nphi) * dphi
    else:
        dphi = abs(phi_min) / (nphi - 1)
        phi_grid = np.arange(-(nphi - 1), 1) * dphi
    phi_min = float(phi_grid[0])
    phi_max = float(phi_grid[-1])

    num_cores = 5
    table_dir = str(_sputtering_output_dir())
    os.makedirs(table_dir, exist_ok=True)

    all_rate_tables = []
    output_files = []

    for ispec in range(len(ion_atomic_masses)):
        mi = ion_atomic_masses[ispec]
        Zi = int(ion_atomic_numbers[ispec])

        E_sp = threshold_energy(surface_energy, am_dust, mi)
        E_kin_max = 1e7
        E_lookup = None
        Y_lookup = None
        if use_yield_lookup:
            E_lookup_min = max(E_sp, 1e-8)
            E_lookup_max = max(E_kin_max + max(0.0, phi_max), E_lookup_min * 10.0)
            E_lookup, Y_lookup = _build_sputtering_yield_lookup(
                a_dust, surface_energy, Kparam, rho_dust,
                am_dust, mi,
                an_dust, Zi,
                bocchio_fitparams, bocchio_fitparams['E_exec'],
                do_size_correction,
                E_lookup_min, E_lookup_max,
                nE_lookup=nE_lookup,
            )

        args_list = [
            (a_dust, surface_energy, Kparam, rho_dust,
             am_dust, mi,
             an_dust, Zi,
             bocchio_fitparams, bocchio_fitparams['E_exec'], Ti,
             int(nbins_v_by_T[iT]), do_size_correction, phi_grid,
             E_lookup, Y_lookup)
            for iT, Ti in enumerate(Tgas)
        ]

        print(f'Computing T-phi table for ion Z={Zi}, m={mi:.3f} a.u....')
        if executor is None:
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as local_executor:
                results = list(tqdm(local_executor.map(average_yields_T_phi_batch, args_list),
                                    total=len(args_list),
                                    desc=f'    T-phi integration (Z={Zi})',
                                    unit=' temperatures'))
        else:
            results = list(tqdm(executor.map(average_yields_T_phi_batch, args_list),
                                total=len(args_list),
                                desc=f'    T-phi integration (Z={Zi})',
                                unit=' temperatures'))

        Y_species = np.asarray(results, dtype=float)
        if Y_species.shape != (nT, nphi):
            Y_species = Y_species.reshape(nT, nphi)
        rate_species = (am_dust * au2cgs_m) / (2.0 * rho_dust) * Y_species * (1e4 * sec2yr)
        rate_species = np.maximum(rate_species, 1e-30)
        all_rate_tables.append(rate_species)

        table_file = os.path.join(
            table_dir,
            f'sputtering_{dustlabel}_Z_{Zi}'
        )
        ion_name_dict = {1: 'H', 2: 'He', 6: 'C', 7: 'N', 8: 'O', 10: 'Ne', 12: 'Mg', 14: 'Si', 16: 'S', 26: 'Fe'}
        ion_name_resolved = ion_name_dict.get(int(Zi), f'Z{int(Zi)}')
        from pycalima.models.grain_size_config import get_header_lines
        headers = get_header_lines(
            title="Thermal sputtering rate table",
            script_name="models/dust_gas_collisions/dust_sputtering.py",
            bin_info=f"Dust Bin: {dustlabel}, Composition: {grain_type}, Size: {a_dust * 1e4:.4e} micron, Species: {ion_name_resolved} (Z={Zi})",
            val_desc="Values: log10(sputtering rate [micron yr^-1 cm^3])",
            num_lines=6
        )
        with open(table_file, 'w', encoding='utf-8') as f:
            for line in headers:
                f.write(f"{line}\n")
            f.write(f"{nT:8d} {nphi:8d}\n")
            f.write(' '.join([f'{phi:.8e}' for phi in phi_grid]) + '\n')
            for iT, Ti in enumerate(Tgas):
                row = [f'{np.log10(Ti):.8e}'] + [f'{np.log10(rate_species[iT, j]):.8e}' for j in range(nphi)]
                f.write(' '.join(row) + '\n')

        output_files.append(table_file)

    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    nspecies = len(ion_atomic_numbers)
    ncols = min(3, nspecies)
    nrows = int(np.ceil(nspecies / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.2 * nrows), dpi=250)
    axes = np.atleast_2d(axes)

    for ispec in range(nspecies):
        irow = ispec // ncols
        icol = ispec % ncols
        ax = axes[irow, icol]
        rate_species = all_rate_tables[ispec]
        Zi = int(ion_atomic_numbers[ispec])

        positive = rate_species[rate_species > 0.0]
        if positive.size > 0:
            vmin = float(np.min(positive))
            vmax = float(np.max(positive))
            if vmax > vmin:
                norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
                img = ax.imshow(rate_species, origin='lower', aspect='auto',
                                extent=[phi_grid[0], phi_grid[-1], np.log10(Tgas[0]), np.log10(Tgas[-1])],
                                norm=norm, cmap='magma')
            else:
                img = ax.imshow(rate_species, origin='lower', aspect='auto',
                                extent=[phi_grid[0], phi_grid[-1], np.log10(Tgas[0]), np.log10(Tgas[-1])],
                                cmap='magma')
        else:
            img = ax.imshow(rate_species, origin='lower', aspect='auto',
                            extent=[phi_grid[0], phi_grid[-1], np.log10(Tgas[0]), np.log10(Tgas[-1])],
                            cmap='magma')

        ax.set_xlabel('phi [eV]')
        ax.set_ylabel(r'log$_{10}(T)$ [K]')
        ax.set_title(f'Ion Z={Zi}, m={ion_atomic_masses[ispec]:.3f} a.u.')
        cbar = fig.colorbar(img, ax=ax)
        cbar.set_label(r'$(1/n_{\rm H})da/dt$ [$\mu$m yr$^{-1}$ cm$^3$]')

    for idx in range(nspecies, nrows * ncols):
        irow = idx // ncols
        icol = idx % ncols
        axes[irow, icol].axis('off')

    if composition is not None and dust_type is None:
        setup_label = f"{composition}, a={a_dust*1e4:.4g} micron"
    else:
        setup_label = str(dust_type)

    fig.suptitle(
        f'{setup_label}: sputtering rate on (T, phi) grid | '
        f'phi in [{phi_min:.3e}, {phi_max:.3e}] eV',
        fontsize=14
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    fig_file = os.path.join(table_dir, f'sputtering_Tphi_overview_{dustlabel}{label}.png')
    fig.savefig(fig_file, format='png')

    print('Saved T-phi tables:')
    for file_name in output_files:
        print(f'  - {file_name}')
    print(f'Saved validation figure: {fig_file}')

    return {
        'Tgas': Tgas,
        'phi_grid': phi_grid,
        'phi_min': phi_min,
        'phi_max': phi_max,
        'grain_charge_min_allowed': Zg_min,
        'grain_charge_max_allowed': Zg_max,
        'output_files': output_files,
        'figure_file': fig_file,
        'rates': all_rate_tables,
    }


def read_Tphi_table(table_file):
    """Read a thermal_sputtering_Tphi CSV file and return grid arrays.

    Returns
    -------
    Tgas : ndarray
        Temperature grid in K (axis 0 of rates).
    phi_grid : ndarray
        Phi grid in eV (axis 1 of rates).
    rates : ndarray
        Rate grid with shape (nT, nphi).
    metadata : dict
        Header key/value metadata parsed from comment lines.
    """

    metadata = {}

    with open(table_file, 'r', encoding='utf-8') as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    if len(lines) == 0:
        raise ValueError(f'Empty table file: {table_file}')

    # Check if this is the legacy CSV format (contains comma in the first few non-comment lines or starts with T_K/phi,)
    is_legacy_csv = False
    for ln in lines:
        if ln.startswith('#'):
            continue
        if ',' in ln or ln.startswith('T_K/phi,'):
            is_legacy_csv = True
        break

    if is_legacy_csv:
        phi_grid = None
        data_rows = []
        for text in lines:
            if text.startswith('#'):
                item = text[1:].strip()
                if '=' in item:
                    key, value = item.split('=', 1)
                    metadata[key.strip()] = value.strip()
                continue
            if text.startswith('T_K/phi,'):
                phi_grid = np.asarray([float(x) for x in text.split(',')[1:]], dtype=float)
                continue
            data_rows.append([float(x) for x in text.split(',')])

        if phi_grid is None:
            raise ValueError(f'Could not find phi header line in {table_file}')
        if len(data_rows) == 0:
            raise ValueError(f'No data rows found in {table_file}')

        data = np.asarray(data_rows, dtype=float)
        Tgas = data[:, 0]
        rates = data[:, 1:]
        if rates.shape[1] != len(phi_grid):
            raise ValueError(
                f'Inconsistent shape in {table_file}: rates has {rates.shape[1]} phi columns '
                f'but phi_grid has {len(phi_grid)} values.'
            )
        return Tgas, phi_grid, rates, metadata

    # New Fortran-friendly format (with potential comments starting with #)
    # Parse comments for metadata
    for ln in lines:
        if ln.startswith('#'):
            item = ln[1:].strip()
            if '=' in item:
                key, value = item.split('=', 1)
                metadata[key.strip()] = value.strip()

    # Filter out comments
    data_lines = [ln for ln in lines if not ln.startswith('#')]

    nT, nphi = [int(x) for x in data_lines[0].split()[:2]]
    phi_grid = np.asarray([float(x) for x in data_lines[1].split()], dtype=float)
    if len(phi_grid) != nphi:
        raise ValueError(
            f'Inconsistent phi count in {table_file}: header says {nphi} but line 2 has {len(phi_grid)} values.'
        )

    if len(data_lines) < 2 + nT:
        raise ValueError(
            f'Not enough data rows in {table_file}: expected {nT}, found {max(0, len(data_lines)-2)}.'
        )

    data_rows = []
    for ln in data_lines[2:2+nT]:
        row = [float(x) for x in ln.split()]
        if len(row) != 1 + nphi:
            raise ValueError(
                f'Inconsistent row length in {table_file}: expected {1+nphi}, got {len(row)} for row "{ln}"'
            )
        data_rows.append(row)

    data = np.asarray(data_rows, dtype=float)
    logT = data[:, 0]
    logR = data[:, 1:]
    Tgas = 10.0**logT
    rates = 10.0**logR
    metadata['format'] = 'fortran-logT-logR-linearPhi'
    metadata['nT'] = str(nT)
    metadata['nphi'] = str(nphi)

    return Tgas, phi_grid, rates, metadata


def build_Tphi_interpolator(table_file, use_log_rates=True, bounds_error=False, fill_value=None):
    """Build a 2D interpolator for a T-phi sputtering table.

    Interpolation axes are (log10(T), phi[eV]). By default, interpolation is also
    done in log10(rate), which is usually better for rates spanning many decades.

    Parameters
    ----------
    table_file : str
        Path to one thermal_sputtering_Tphi_*.csv file.
    use_log_rates : bool, optional
        If True, interpolate log10(rate) and return rate in linear units.
    bounds_error : bool, optional
        Passed to scipy.interpolate.RegularGridInterpolator.
    fill_value : float or None, optional
        Passed to scipy.interpolate.RegularGridInterpolator.

    Returns
    -------
    evaluate : callable
        Function evaluate(T_query, phi_query) -> interpolated rates.
    info : dict
        Dictionary with Tgas, phi_grid, rates, metadata.
    """

    from scipy.interpolate import RegularGridInterpolator

    Tgas, phi_grid, rates, metadata = read_Tphi_table(table_file)
    logT = np.log10(Tgas)

    if use_log_rates:
        positive = rates[rates > 0.0]
        if positive.size == 0:
            raise ValueError('All rates are non-positive; cannot build log-rate interpolator.')
        floor = np.min(positive)
        values = np.log10(np.clip(rates, floor, None))
    else:
        values = rates

    interp = RegularGridInterpolator(
        (logT, phi_grid),
        values,
        method='linear',
        bounds_error=bounds_error,
        fill_value=fill_value,
    )

    def evaluate(T_query, phi_query):
        T_query = np.asarray(T_query, dtype=float)
        phi_query = np.asarray(phi_query, dtype=float)
        T_b, phi_b = np.broadcast_arrays(T_query, phi_query)

        points = np.column_stack((np.log10(T_b.ravel()), phi_b.ravel()))
        y = interp(points)
        if use_log_rates:
            y = 10.0**y
        return y.reshape(T_b.shape)

    info = {
        'Tgas': Tgas,
        'phi_grid': phi_grid,
        'rates': rates,
        'metadata': metadata,
    }
    return evaluate, info

def compare_sputtering_rates(Tmin,Tmax,ion_atomic_masses,
                             ion_atomic_numbers,ion_charges,ion_abundances,
                             nT=100,nbins_v=100,label=''):
    """Plotting routine that allows the comparison of our current
    dust sputtering model with different properties and the original
    Nozawa et al. (2006) results.

    Args:
        Tmin (float): minimum temperature
        Tmax (float): maximum temperature
        ion_atomic_masses (np.ndarray): array with the ion atomic masses
        ion_atomic_numbers (np.ndarray): array with the ion atomic numbers
        ion_charges (np.ndarray): array with the ionisation states
        ion_abundances (np.ndarray): array with the ion abundances over hydrogen
        nT (int, optional): number of temperature bins. Defaults to 100.
        nbins_v (int, optional): number of velocity bins in the Maxwell-Boltzmann integration. Defaults to 100.

    Returns:
        matplotlib.pyplot.figure: final figure with the included plots
    """    
    
    # 1. Setup figure
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="white")
    sns.color_palette("Paired")
    sns.set_theme(style="white")
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": "Computer Modern Roman",
    })
    fig, ax = plt.subplots(1,1, figsize=(7,5),dpi=300,facecolor='w',edgecolor='k',sharey=True)
    
    ax.set_ylabel(r'$(1/n_{\rm H})da/dt$ [cm$^3 \mu$m yr$^{-1}$]', fontsize=16)
    ax.set_xlabel(r'$T$ [K]',fontsize=16)
    ax.set_ylim([1e-9,1e-4])
    ax.set_xlim([6e3,1e9])
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.tick_params(labelsize=14)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.minorticks_on()
    ax.tick_params(which='both',axis="both",direction="in")
    
    # 2. Add the curves from Nozawa et al. (2006)
    linestyles = ['-','--','-.',':']
    colours = ['b','r','m','g']
    Tgas = np.logspace(np.log10(Tmin),np.log10(Tmax),nT)
    Y_Sil = dust_model.Tielens_rate(dust_model.thermal_spu_nozawa06['Sil'],np.log10(Tgas))
    Y_C = dust_model.Tielens_rate(dust_model.thermal_spu_nozawa06['Car'],np.log10(Tgas))
    ax.plot(Tgas,Y_Sil,linestyle='--',color='sandybrown',linewidth=3,label='Sil: Nozawa et al. (2006)')
    ax.plot(Tgas,Y_C,linestyle='--',color='cornflowerblue',linewidth=3,label='C: Nozawa et al. (2006)')
    
    table_dir = str(_sputtering_output_dir())
    os.makedirs(table_dir, exist_ok=True)
    with open(os.path.join(table_dir,f"thermal_sputtering_polynomial_fits{label}.txt"), "w") as file:
        file.write("Thermal Dust sputtering Fit Results (with size and charge corrections)\n")
        
        # 3. Compute the rates for each grain type
        a_dust, Tgas, Y_smallC = total_erosion_rate(Tmin,Tmax,'smallC',
                                            ion_atomic_masses,
                                            ion_atomic_numbers,
                                            ion_charges,
                                            ion_abundances,
                                            nT,nbins_v,
                                            False,False)
        ax.plot(Tgas,Y_smallC,linestyle='-',color='steelblue',linewidth=3,label='smallC: No size correction')
        
        a_dust, Tgas, Y_smallC = total_erosion_rate(Tmin,Tmax,'smallC',
                                            ion_atomic_masses,
                                            ion_atomic_numbers,
                                            ion_charges,
                                            ion_abundances,
                                            nT,nbins_v,
                                            True,False)
        ax.plot(Tgas,Y_smallC,linestyle=(0, (3, 1, 1, 1)),color='steelblue',linewidth=3,label='smallC: With size correction')

        coefficients = np.polyfit(np.log10(Tgas[Y_smallC>0]), np.log10(Y_smallC[Y_smallC>0]), 6)
        poly_fit = np.poly1d(coefficients)
        # ax.plot(Tgas,10**poly_fit(np.log10(Tgas)),linestyle=':',color='k',alpha=0.6)
        file.write("======================\n")
        file.write("Polynomial Coefficients smallC grains (%.4f microns):\n"%(a_dust/1e-4))
        file.write("f(x) = ")
        for i, coeff in enumerate(coefficients):
            file.write(f"{coeff:.8e}x^{6-i} ")
            if i < len(coefficients) - 1:
                file.write("+ ")
        file.write("\n")
        file.write("\n")
        file.write("Original Data (Tgas, Yield):\n")
        for i in range(len(Tgas)):
            file.write(f"{Tgas[i]:.6e}, {Y_smallC[i]:.6e}\n")
        
        
        a_dust, Tgas, Y_largeC = total_erosion_rate(Tmin,Tmax,'largeC',
                                            ion_atomic_masses,
                                            ion_atomic_numbers,
                                            ion_charges,
                                            ion_abundances,
                                            nT,nbins_v,
                                            True,False)
        ax.plot(Tgas,Y_largeC,linestyle=':',color='cornflowerblue',linewidth=3,label='largeC: With size correction')

        coefficients = np.polyfit(np.log10(Tgas[Y_largeC>0]),  np.log10(Y_largeC[Y_largeC>0]), 6)
        poly_fit = np.poly1d(coefficients)
        # ax.plot(Tgas,10**poly_fit(np.log10(Tgas)),linestyle=':',color='k',alpha=0.6)
        file.write("======================\n")
        file.write("Polynomial Coefficients largeC grains (%.4f microns):\n"%(a_dust/1e-4))
        file.write("f(x) = ")
        for i, coeff in enumerate(coefficients):
            file.write(f"{coeff:.8e}x^{6-i} ")
            if i < len(coefficients) - 1:
                file.write("+ ")
        file.write("\n")
        file.write("\n")
        file.write("Original Data (Tgas, Yield):\n")
        for i in range(len(Tgas)):
            file.write(f"{Tgas[i]:.6e}, {Y_largeC[i]:.6e}\n")
        
        # a_dust, Tgas, Y_smallSil = total_erosion_rate(Tmin,Tmax,'smallSil',
        #                                     ion_atomic_masses,
        #                                     ion_atomic_numbers,
        #                                     ion_charges,
        #                                     ion_abundances,
        #                                     nT,nbins_v,
        #                                     False,False)
        # ax.plot(Tgas,Y_smallSil,linestyle='-',color='saddlebrown',linewidth=3,label='smallSil: No size correction')
        
        
        # a_dust, Tgas, Y_smallSil = total_erosion_rate(Tmin,Tmax,'smallSil',
        #                                     ion_atomic_masses,
        #                                     ion_atomic_numbers,
        #                                     ion_charges,
        #                                     ion_abundances,
        #                                     nT,nbins_v,
        #                                     True,False)
        # ax.plot(Tgas,Y_smallSil,linestyle=(0, (3, 1, 1, 1)),color='saddlebrown',linewidth=3,label='smallSil: With size correction')
        # coefficients = np.polyfit(np.log10(Tgas[Y_smallSil>0]),  np.log10(Y_smallSil[Y_smallSil>0]), 6)
        # poly_fit = np.poly1d(coefficients)
        # # ax.plot(Tgas,10**poly_fit(np.log10(Tgas)),linestyle=':',color='k',alpha=0.6)
        # file.write("======================\n")
        # file.write("Polynomial Coefficients smallSil grains (%.4f microns):\n"%(a_dust/1e-4))
        # file.write("f(x) = ")
        # for i, coeff in enumerate(coefficients):
        #     file.write(f"{coeff:.8e}x^{6-i} ")
        #     if i < len(coefficients) - 1:
        #         file.write("+ ")
        # file.write("\n")
        # file.write("\n")
        # file.write("Original Data (Tgas, Yield):\n")
        # for i in range(len(Tgas)):
        #     file.write(f"{Tgas[i]:.6e}, {Y_smallSil[i]:.6e}\n")
        
        
        # a_dust, Tgas, Y_largeSil = total_erosion_rate(Tmin,Tmax,'largeSil',
        #                                     ion_atomic_masses,
        #                                     ion_atomic_numbers,
        #                                     ion_charges,
        #                                     ion_abundances,
        #                                     nT,nbins_v,
        #                                     True,False)
        # ax.plot(Tgas,Y_largeSil,linestyle=':',linewidth=3,color='sandybrown',label='largeSil: With size correction')
        
        # coefficients = np.polyfit(np.log10(Tgas[Y_largeSil>0]),  np.log10(Y_largeSil[Y_largeSil>0]), 6)
        # poly_fit = np.poly1d(coefficients)
        # # ax.plot(Tgas,10**poly_fit(np.log10(Tgas)),linestyle=':',color='k',alpha=0.6)
        # file.write("======================\n")
        # file.write("Polynomial Coefficients largeSil grains (%.4f microns):\n"%(a_dust/1e-4))
        # file.write("f(x) = ")
        # for i, coeff in enumerate(coefficients):
        #     file.write(f"{coeff:.8e}x^{6-i} ")
        #     if i < len(coefficients) - 1:
        #         file.write("+ ")
        # file.write("\n")
        # file.write("\n")
        # file.write("Original Data (Tgas, Yield):\n")
        # for i in range(len(Tgas)):
        #     file.write(f"{Tgas[i]:.6e}, {Y_largeSil[i]:.6e}\n")
            
    print(f"Results written to thermal_sputtering_polynomial_fits{label}.txt")
    
    ax.legend(loc='best', frameon=False, fontsize=12, ncol=2)
    fig.subplots_adjust(top=0.98,bottom=0.1,left=0.11,right=0.96,hspace=0,wspace=0)
    return fig
    
def effective_charge_number(z,v_ion):
    
    z_eff = z * (1. - np.exp(-125.*v_ion / c * z**(-2./3.)))
    
    return z_eff

@njit(cache=True)
def electronic_stopping_cs(n_dust,ne_val,Zi,Zd,Md,Mi,E):
    
    hbar = 6.626176e-27/(2.*np.pi) # [erg s]
    me = 9.10938e-28 # [g]
    a_sc = screening_length(Zi,Zd) # [cm]
    
    Z = (float(Zi)**(2./3.) + float(Zd)**(2./3.))**(3./2.)
    E0 = hbar**2. / (2.*me) * (3*np.pi*2.*ne_val*n_dust)**(2./3.)
    v0 = np.sqrt(2.*E0/me)
    v = np.sqrt(2. * E / Mi)
    
    S = 8. * np.pi * Zi**(1./6.) * Zi * Zd / Z * a_sc * elem_charge**2. * (v/v0)
    return S

@njit(cache=True)
def compute_penetration_depth(E_init,m_ion,s_dust,Zi,Z_dust,M_dust,delta_max=0.01,nmax=1000):
    """Obtain the penetration depth using the Bethe-Bloch formula. It describes the mean energy loss
    per distance travelled of a charge particle traversing matter.

    Args:
        E_init (float): initial energy of the ion [eV]
        z (float): ion charge
        m_ion (float): ion atomic mass [g]
        s_dust (float): dust material density [g/cm^3]
        Z_dust (float): average dust atomic number
        M_dust (float): average dust material atomic mass [g]
        E_exc (float): average excitation energy [eV]
        delta_max (float, optional): maximum change of the ion energy during the integration. Defaults to 0.01.
        nmax (int, optional): maximum number of iterations. Defaults to 1000.

    Returns:
        float: penetration depth [cm]
    """    
    
    E_now = E_init
    
    dr = 1e-7 # [cm]
    r_pd = 0 # [cm]
    n = 0
    
    a_sc = screening_length(Zi,Z_dust)
    n_dust = s_dust / M_dust * Z_dust
    if Z_dust == 6:
        n_eval = 4
    else:
        n_eval = (2+8+4+4*6) / 7

    while E_now > 1e-3 * E_init and n < nmax:
        # 1. Compute the reduced nuclear stopping cross-section
        epsilon = reduced_energy(M_dust,m_ion,a_sc,Zi,Z_dust,E_now)
        s = screened_Coulomb_function(epsilon)
        
        # 2. Compute the dE/dr
        Se = electronic_stopping_cs(n_dust,n_eval,Zi,Z_dust,M_dust,m_ion,E_now)
        Sn = 4. * np.pi * a_sc * Zi * Z_dust * elem_charge**2. * m_ion / (m_ion+M_dust) * s
        dE = n_dust * (Sn + Se)   
        # 3. Figure out if the step needs to be changed
        if abs(dE*dr)/E_now >= delta_max:
            dr = dr / 2.
        else:
            r_pd = r_pd + dr
            # print(r_pd)
            E_now = E_now - dE * dr
        n += 1
    
    return r_pd # [cm]

def plot_penetration_depth(E_min,E_max,z_ion,m_ion,nE=100,delta_max=0.01,nmax=1000):
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="white")
    sns.color_palette("Paired")
    sns.set_theme(style="white")
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": "Computer Modern Roman",
    })
    
    # 1. Setup the energy array
    E = np.logspace(np.log10(E_min),np.log10(E_max),nE)
    
    # 2. Set the material properties for the different grains
    rho_dust_c = dust_model.basic_s[2] # [g/cm^3]
    am_dust_c = 12.011 * au2cgs_m # [g]
    an_dust_c = 6
    E_exec_c = 13.5 # [eV]
    
    # Assuming MgFeSiO4 composition (olivine with Iron inclusions as the regular model)
    rho_dust_sil = dust_model.basic_s[5] # [g/cm^3]
    am_dust_sil = (24.305 + 55.845 + 28.0855 + 4*15.999) / 7. * au2cgs_m  # [g]
    an_dust_sil = int((4*8 + 14 + 26 + 12) / 7)
    E_exec_sil = 13.0 # [eV]
    
    # 3. Compute the penetration depth for each energy
    r_dp_sil = np.zeros((2,nE))
    r_dp_c = np.zeros((2,nE))
    for i in range(0, nE):
        r_dp_sil[0,i] = compute_penetration_depth(E[i],z_ion,m_ion,
                                             rho_dust_sil,an_dust_sil,
                                             am_dust_sil,E_exec_sil,
                                             delta_max=delta_max,nmax=nmax)
        r_dp_sil[1,i] = 1e7*penetration_depth(z_ion,size_correction_fitparams['MgSiO4']['alpha_P'],E[i])
        r_dp_c[0,i] = compute_penetration_depth(E[i],z_ion,m_ion,
                                             rho_dust_c,an_dust_c,
                                             am_dust_c,E_exec_c,
                                             delta_max=delta_max,nmax=nmax)
        r_dp_c[1,i] = 1e7*penetration_depth(z_ion,size_correction_fitparams['C']['alpha_P'],E[i])
    
    # 4. Setup figure and plot
    fig, ax = plt.subplots(1,1, figsize=(5,4),dpi=300,facecolor='w',edgecolor='k',sharey=True)
    
    ax.set_ylabel(r'$r_{\rm pd}$ [nm]', fontsize=16)
    ax.set_xlabel(r'$E_{\rm init}$ [eV]',fontsize=16)
    ax.set_yscale('log')
    ax.set_xscale('log')
    #ax.set_ylim([1e-5,1e1])
    ax.tick_params(labelsize=14)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.minorticks_on()
    ax.tick_params(which='both',axis="both",direction="in")
    
    ax.plot(E,r_dp_sil[0,:],linestyle='-',color='saddlebrown',label='Silicates')
    ax.plot(E,r_dp_sil[1,:],linestyle='--',color='saddlebrown')
    ax.plot(E,r_dp_c[0,:],linestyle='-',color='royalblue',label='Carbonaceous')
    ax.plot(E,r_dp_c[1,:],linestyle='--',color='royalblue')
    ax.text(0.6, 0.1, r'$z_{\rm ion}=%i$'%int(z_ion)+'\n'+r'$m_{\rm ion}=%.1f$ [a.u.]'%(float(m_ion)/au2cgs_m),
                                    verticalalignment='bottom', horizontalalignment='left',
                                    transform=ax.transAxes,fontsize=15)

    ax.legend(loc='best',frameon=False,fontsize=14)
    fig.subplots_adjust(top=0.99,bottom=0.13,left=0.14,right=0.98,hspace=0,wspace=0)
    fig.savefig('test_penetration_depth.png',format='png')


def average_yields_T_fixed_charge(args):
    """Computation of the average yield for a particular grain
    and ion at a given temperature T, assuming a Maxwell-Boltzmann
    distribution averaging, with a fixed dust charge.

    Args:
        args (tuple): arguments required by the sputtering_yield function,
                      including a fixed dust_charge value

    Returns:
        float: final average(Y*v) yield [atom/ion]
    """    
    
    from scipy.integrate import trapezoid
    
    dust_radius,surface_energy,Kparam,rho_dust,\
        dust_atomic_mass,ion_atomic_mass,\
        dust_atomic_number,ion_atomic_number,\
        ion_charge,bocchio_fit_params,E_exec,Tgas,\
        nbins_v,do_size_correction,dust_charge = args
    
    # 1. Determine the minumum velocity for reaching the threshold energy
    E_sp = threshold_energy(surface_energy,dust_atomic_mass,ion_atomic_mass)
    v_0  = np.sqrt(2. * E_sp * eV2erg) # [cm/s]
    while Maxwell_Boltzmann_function(v_0,ion_atomic_mass*au2cgs_m,Tgas) < 1e-20:
        v_0 = 2 * v_0
    
    # 2. We set the maximum velocity to the thermal energy of gas at ~1e11 K
    v_max = np.sqrt(2. * 1e7 * eV2erg / (ion_atomic_mass*au2cgs_m)) # [cm/s]
    while Maxwell_Boltzmann_function(v_max,ion_atomic_mass*au2cgs_m,Tgas) < 1e-25:
        v_max = v_max/ 2.0
    
    # 3. Perform loop over velocity range
    v = np.logspace(np.log10(v_0),np.log10(v_max),nbins_v)
    Y_v = np.zeros(nbins_v)
    
    # 4. Use the provided fixed dust_charge
    Zmean = dust_charge
    
    for i in range(0, nbins_v):
        vi = v[i] # [cm/s]
        Ei = 0.5 * ion_atomic_mass * au2cgs_m * vi**2. # [erg]
        Ei = Ei / eV2erg
        mb_factor = Maxwell_Boltzmann_function(vi,ion_atomic_mass*au2cgs_m,Tgas)
        Y = sputtering_yield(dust_radius,surface_energy,Kparam,rho_dust,
                                dust_atomic_mass,ion_atomic_mass,
                                dust_atomic_number,ion_atomic_number,ion_charge,
                                bocchio_fit_params,E_exec,Ei,
                                do_size_correction,Zmean)
        Y_v[i] = mb_factor * vi * Y

    # 5. Integrate Y_v with the trapezoid method
    Y0 = trapezoid(Y_v,v)
    
    return Y0


def erosion_rate_with_charge_grid(Tmin, Tmax, G0, ne, nT=50, nZ_ion=11,
                                   ion_atomic_masses=None,
                                   ion_atomic_numbers=None,
                                   ion_charges=None,
                                   ion_abundances=None,
                                   nbins_v=100):
    """Compute erosion rates for smallC grains over a grid of ion charges 
    (0-10) and temperatures, exploring how grain charge influences erosion.

    For each temperature, computes:
    - Baseline rate (grain charge = 0)
    - Rates for each ion charge from 0 to 10 (using computed grain charges)
    - Ratio of rate with charge to baseline rate

    Parameters
    ----------
    Tmin : float
        Minimum temperature [K]
    Tmax : float
        Maximum temperature [K]
    G0 : float
        Radiation field strength (for grain charge computation)
    ne : float
        Electron density [cm^-3] (for grain charge computation)
    nT : int, optional
        Number of temperature points. Default is 50.
    nZ_ion : int, optional
        Number of ion charge states to compute (0 to nZ_ion-1). Default is 11 (0-10).
    ion_atomic_masses : ndarray, optional
        Array of ion atomic masses [a.u.]
    ion_atomic_numbers : ndarray, optional
        Array of ion atomic numbers
    ion_charges : ndarray, optional
        Kept for API compatibility; ignored in this function because ion charges
        are explicitly scanned from 0 to nZ_ion-1.
    ion_abundances : ndarray, optional
        Array of ion abundances relative to hydrogen
    nbins_v : int, optional
        Number of velocity bins for Maxwell-Boltzmann integration. Default is 100.

    Returns
    -------
    Tgas : ndarray
        Temperature array [K] (nT elements)
    Z_ion : ndarray
        Ion charge array (nZ_ion elements)
    rates_baseline : ndarray
        Baseline erosion rates (no grain charge), shape (nT,)
    rates_with_charge : ndarray
        Erosion rates with grain charge, shape (nT, nZ_ion)
    ratio_matrix : ndarray
        Ratio of rates_with_charge to baseline, shape (nT, nZ_ion)
    grain_charges : ndarray
        Computed grain charges at each temperature, shape (nT, nZ_ion)
    """
    
    if ion_atomic_masses is None:
        # Default to H+
        ion_atomic_masses = np.array([1.008])
        ion_atomic_numbers = np.array([1])
        ion_charges = np.array([1])
        ion_abundances = np.array([1.0])
    
    # 1. Prepare the dust grain properties for smallC
    a_dust = dust_model.basic_a0[2]*1e-4  # [cm]
    rho_dust = dust_model.basic_s[2]  # [g/cm^3]
    am_dust = 12.011  # [a.u.]
    an_dust = 6
    E_exec = size_correction_fitparams['C']['E_exec']
    bocchio_fitparams = size_correction_fitparams['C']
    Kparam = Ksput['C']
    surface_energy = U0['C']
    
    # dust_charging now expects cgs grain radius (cm).
    a_dust_cm = a_dust
    
    # 2. Set up temperature and ion charge arrays
    Tgas = np.logspace(np.log10(Tmin), np.log10(Tmax), nT)
    Z_ion = np.arange(0, nZ_ion, dtype=int)
    
    # 3. Initialize output arrays
    rates_baseline = np.zeros(nT)
    rates_with_charge = np.zeros((nT, nZ_ion))
    grain_charges = np.zeros((nT, nZ_ion))
    
    num_cores = 5
    
    # 4. Compute baseline rates (no ion charge and no grain charge effects)
    print("Computing baseline erosion rates (no ion/grain charge)...")
    for i in range(0, len(ion_abundances)):
        args_list = [(a_dust, surface_energy, Kparam, rho_dust,
                      am_dust, ion_atomic_masses[i],
                      an_dust, ion_atomic_numbers[i],
                      0, bocchio_fitparams, E_exec, Ti,
                      nbins_v, True, 0.0) for Ti in Tgas]
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
            results = list(tqdm(executor.map(average_yields_T_fixed_charge, args_list),
                                total=nT,
                                desc=f'    Baseline rates for ion mass {ion_atomic_masses[i]}',
                                unit=' steps'))
        
        Y_temp = np.array(results)
        rates_baseline = rates_baseline + ion_abundances[i] * Y_temp
    
    # Convert to physical units [microns / yr * cm^3]
    rates_baseline = (am_dust * au2cgs_m) / (2. * rho_dust) * rates_baseline * (1e4 * sec2yr)

    # 5. Compute mean grain charge from dust_charging (WD01 solver), once per temperature.
    from pycalima.models.dust_charge.dust_charging import equilibrium_charge_for_grain
    Z_dust_vs_T = np.zeros(nT)
    print("\nComputing mean grain charges with dust_charging...")
    for ti, Ti in enumerate(Tgas):
        try:
            _, _, _, Zmean, _ = equilibrium_charge_for_grain(
                G0=G0,
                ne=ne,
                T=Ti,
                grain_type='graphite',
                a_cm=a_dust_cm,
                radiation_model='Mathis',
                ion_species=None,
                debug=False,
            )
            Z_dust_vs_T[ti] = Zmean
        except Exception as e:
            print(f"    Warning: Could not compute grain charge at T={Ti}K: {e}")
            Z_dust_vs_T[ti] = 0.0

    # Store the same temperature-dependent grain-charge profile for all Z_ion columns.
    grain_charges[:, :] = Z_dust_vs_T[:, None]
    
    # 6. Compute rates for each ion charge state
    print(f"\nComputing erosion rates for ion charge grid (0-{nZ_ion-1})...")
    for zi in range(nZ_ion):
        print(f"  Processing ion charge Z_ion = {zi}...")
        
        # Compute erosion rates for each ion type at each temperature
        Y_with_charge = np.zeros(nT)
        for i in range(0, len(ion_abundances)):
            args_list = [(a_dust, surface_energy, Kparam, rho_dust,
                          am_dust, ion_atomic_masses[i],
                          an_dust, ion_atomic_numbers[i],
                          zi, bocchio_fitparams, E_exec, Ti,
                          nbins_v, True, Z_dust_vs_T[ti]) 
                         for ti, Ti in enumerate(Tgas)]
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
                results = list(tqdm(executor.map(average_yields_T_fixed_charge, args_list),
                                    total=nT,
                                    desc=f'    Rates (Z_ion={zi}) for ion mass {ion_atomic_masses[i]}',
                                    unit=' steps'))
            
            Y_temp = np.array(results)
            Y_with_charge = Y_with_charge + ion_abundances[i] * Y_temp
        
        # Convert to physical units [microns / yr * cm^3]
        Y_with_charge = (am_dust * au2cgs_m) / (2. * rho_dust) * Y_with_charge * (1e4 * sec2yr)
        rates_with_charge[:, zi] = Y_with_charge
    
    # 7. Compute ratio matrix
    ratio_matrix = np.zeros((nT, nZ_ion))
    for ti in range(nT):
        if rates_baseline[ti] > 0:
            ratio_matrix[ti, :] = rates_with_charge[ti, :] / rates_baseline[ti]
        else:
            ratio_matrix[ti, :] = 1.0
    
    return Tgas, Z_ion, rates_baseline, rates_with_charge, ratio_matrix, grain_charges


def plot_erosion_rate_charge_influence(Tmin, Tmax, G0, ne, 
                                        ion_atomic_masses=None,
                                        ion_atomic_numbers=None,
                                        ion_charges=None,
                                        ion_abundances=None,
                                        nT=50, nZ_ion=11, nbins_v=100):
    """Plot the influence of ion and grain charges on erosion rate for smallC grains.

    Creates a heatmap showing the ratio of erosion rate (with grain charge) to baseline
    (without charge) as a function of temperature and ion charge.

    Parameters
    ----------
    Tmin : float
        Minimum temperature [K]
    Tmax : float
        Maximum temperature [K]
    G0 : float
        Radiation field strength (for grain charge computation)
    ne : float
        Electron density [cm^-3] (for grain charge computation)
    ion_atomic_masses : ndarray, optional
        Array of ion atomic masses [a.u.]. Default: [1.008] (H+)
    ion_atomic_numbers : ndarray, optional
        Array of ion atomic numbers. Default: [1]
    ion_charges : ndarray, optional
        Kept for API compatibility; ignored because the plot scans ion charges
        from 0 to nZ_ion-1.
    ion_abundances : ndarray, optional
        Array of ion abundances. Default: [1.0]
    nT : int, optional
        Number of temperature points. Default is 50.
    nZ_ion : int, optional
        Number of ion charge states. Default is 11 (0-10).
    nbins_v : int, optional
        Number of velocity bins. Default is 100.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure
    """
    
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import seaborn as sns
    
    sns.set_theme(style="white")
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": "Computer Modern Roman",
    })
    
    # Compute the rate grid
    Tgas, Z_ion, rates_baseline, rates_with_charge, ratio_matrix, grain_charges = \
        erosion_rate_with_charge_grid(Tmin, Tmax, G0, ne, nT=nT, nZ_ion=nZ_ion,
                                       ion_atomic_masses=ion_atomic_masses,
                                       ion_atomic_numbers=ion_atomic_numbers,
                                       ion_charges=ion_charges,
                                       ion_abundances=ion_abundances,
                                       nbins_v=nbins_v)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(14, 10), dpi=300, facecolor='w', edgecolor='k')
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)
    
    # 1. Main heatmap: Ratio of rates
    ax1 = fig.add_subplot(gs[0:2, 0])
    im1 = ax1.imshow(ratio_matrix.T, aspect='auto', 
                     extent=[np.log10(Tgas[0]), np.log10(Tgas[-1]), -0.5, nZ_ion-0.5],
                     origin='lower', cmap='RdYlBu_r', vmin=ratio_matrix.min(), vmax=ratio_matrix.max())
    ax1.set_xlabel(r'$\log_{10}(T)$ [K]', fontsize=13)
    ax1.set_ylabel(r'Ion charge $Z_{\rm ion}$', fontsize=13)
    ax1.set_title(r'Ratio: Erosion rate (with charge) / Baseline', fontsize=13)
    ax1.set_yticks(Z_ion)
    cbar1 = plt.colorbar(im1, ax=ax1, label='Rate ratio')
    
    # 2. Heatmap: Grain charges
    ax2 = fig.add_subplot(gs[0:2, 1])
    im2 = ax2.imshow(grain_charges.T, aspect='auto',
                     extent=[np.log10(Tgas[0]), np.log10(Tgas[-1]), -0.5, nZ_ion-0.5],
                     origin='lower', cmap='viridis')
    ax2.set_xlabel(r'$\log_{10}(T)$ [K]', fontsize=13)
    ax2.set_ylabel(r'Ion charge $Z_{\rm ion}$', fontsize=13)
    ax2.set_title(r'Computed grain charge $Z_{\rm grain}$', fontsize=13)
    ax2.set_yticks(Z_ion)
    cbar2 = plt.colorbar(im2, ax=ax2, label=r'$Z_{\rm grain}$')
    
    # 3. Baseline rates vs temperature
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.loglog(Tgas, rates_baseline, 'o-', color='steelblue', linewidth=2, markersize=4)
    ax3.set_xlabel(r'$T$ [K]', fontsize=13)
    ax3.set_ylabel(r'$(1/n_{\rm H})da/dt$ [$\mu$m yr$^{-1}$ cm$^3$]', fontsize=13)
    ax3.set_title('Baseline erosion rates (no charge)', fontsize=13)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(labelsize=11)
    
    # 4. Rate enhancement factor vs temperature (at different ion charges)
    ax4 = fig.add_subplot(gs[2, 1])
    # Generate representative ion charge indices dynamically
    max_zi = nZ_ion - 1
    if nZ_ion <= 5:
        zi_indices = list(range(nZ_ion))
    else:
        zi_indices = list(np.linspace(0, max_zi, 5, dtype=int))
    
    for zi in zi_indices:
        enhancement = ratio_matrix[:, zi]
        ax4.plot(Tgas, enhancement, 'o-', label=f'$Z_{{\rm ion}}={zi}$', markersize=4)
    ax4.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
    ax4.set_xlabel(r'$T$ [K]', fontsize=13)
    ax4.set_ylabel('Rate ratio', fontsize=13)
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.set_title('Enhancement factor vs Temperature', fontsize=13)
    ax4.legend(loc='best', fontsize=11, frameon=False)
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(labelsize=11)
    
    # Add info text
    info_text = f'smallC grains: $G_0 = {G0}$, $n_e = {ne:.2e}$ cm$^{{-3}}$'
    fig.text(0.5, 0.02, info_text, ha='center', fontsize=12)
    
    return fig
        
    
