"""
DUST EMISSION

In this script there are tools to test the emission of dust from
the modelling used in Dusty-PRISM. This considers boths the emission
from quasi-steady temperature large grains as well as the stochastic
emission from small grains and PAHs.

By: Curro Rodriguez (currodri@gmail.com)
"""

# Import some libraries
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="white")
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "Computer Modern Roman",
})
import re
from scipy.integrate import quad
from scipy.optimize import root_scalar
from dust_model import basic_a0,basic_amin,basic_amax,basic_sigma,basic_s,LogNormal_Distribution
from dust_oppacity import dust_efficiencies,pah_efficiencies
from PAHs_model import Draine_1978_isrf
from joblib import Parallel, delayed

# Constants
kb               = 1.3806488e-16 # [erg/K] - Boltzmann constant
c                = 2.99792458e10 # [cm/s] - Speed of light
h                = 6.6260755e-27 # [erg s] - Planck constant
sigma_sb         = 5.6703744e-05 # [g s-3 K-4] - Stefan-Boltzmann constant

# PRIMA telescope band specifications
# (https://prima.ipac.caltech.edu/page/instruments)
PRIMA_bands = {
    'PHI1': {
        'band_name': 'PHI1',
        'band_center': 34.5,  # [micron]
        'band_width': 21,  # [micron]
        'spectral_resolving_power': 10,
        'band_min': 34.5 - 21 / 2,  # [micron]
        'band_max': 34.5 + 21 / 2,  # [micron]
        'band_FWHM': 4.1, # [arcsec]
        'pixel_size': 4.1, # [arcsec]
        'pixel_count': [63,23], # [x,y]
        'polarimetry':False
    },
    'PHI2': {
        'band_name': 'PHI2',
        'band_center': 64.5,  # [micron]
        'band_width': 39,  # [micron]
        'spectral_resolving_power': 10,
        'band_min': 64.5 - 39 / 2,  # [micron]
        'band_max': 64.5 + 39 / 2,  # [micron]
        'band_FWHM': 7.4, # [arcsec]
        'pixel_size': 7.4, # [arcsec]
        'pixel_count': [33,14], # [x,y],
        'polarimetry':False
    },
    'PPI1': {
        'band_name': 'PPI1',
        'band_center': 92,  # [micron]
        'spectral_resolving_power': 4,
        'band_FWHM': 10.8, # [arcsec]
        'pixel_size': 10.8, # [arcsec]
        'pixel_count': [36,31], # [x,y]
        'polarimetry':True
    },
    'PPI2': {
        'band_name': 'PPI2',
        'band_center': 126,  # [micron]
        'spectral_resolving_power': 4,
        'band_FWHM': 14.8, # [arcsec]
        'pixel_size': 14.8, # [arcsec]
        'pixel_count': [24,21], # [x,y]
        'polarimetry':True
    },
    'PPI3': {
        'band_name': 'PPI3',
        'band_center': 172,  # [micron]
        'spectral_resolving_power': 4,
        'band_FWHM': 20.2, # [arcsec]
        'pixel_size': 20.2, # [arcsec]
        'pixel_count': [18,16], # [x,y]
        'polarimetry':True
    },
    'PPI4': {
        'band_name': 'PPI4',
        'band_center': 235,  # [micron]
        'spectral_resolving_power': 4,
        'band_FWHM': 27.6, # [arcsec]
        'pixel_size': 27.6, # [arcsec]
        'pixel_count': [12,11], # [x,y]
        'polarimetry':True
    }
}

# Functions
def compute_cross_sections(dust_type, do_average=True):
    """This function generates the cross sections for a given dust type
    based on the public tables from B. Draine and co. The cross sections
    are averaged over the size distribution of the dust grains.

    Args:
        dust_type (str): The type of dust to be used. This can be:
        - SilSmall: Small silicate grains
        - SilLarge: Large silicate grains
        - CSmall: Small carbonaceous grains
        - CLarge: Large carbonaceous grains
        - iPAHSmall: Small ionised PAHs
        - iPAHLarge: Large ionised PAHs
        - nPAHSmall: Small neutral PAHs
        - nPAHLarge: Large neutral PAHs
        do_average (bool, optional): Whether or not to average of the assumed distribution. Defaults to True.

    Returns:
        np.array,np.array,np.array,np.array: The scattering, absorption and radiation pressure cross sections
    """    
    
    
    # 1. Read the efficiencies
    if dust_type == 'SilSmall' or dust_type == 'SilLarge':
        filename = './draine_lee_1984/suvSil_81'
        nwav,data, columns, name = dust_efficiencies(filename)
    elif dust_type == 'CSmall' or dust_type == 'CLarge':
        filename = './draine_lee_1984/Gra_81'
        nwav,data, columns, name = dust_efficiencies(filename)
    elif dust_type == 'iPAHSmall' or dust_type == 'iPAHLarge':
        filename = './li_draine_2001/PAHion_30'
        nwav,data,columns,dust_type = pah_efficiencies(filename)
    elif dust_type == 'nPAHSmall' or dust_type == 'nPAHLarge':
        filename = './li_draine_2001/PAHneu_30'
        nwav,data,columns,name = pah_efficiencies(filename)
    else:
        raise ValueError('Dust type not recognised: ',dust_type)

    # 2. Setup the underlying distribution
    if 'PAHSmall' in dust_type:
        dist = LogNormal_Distribution(basic_a0[0],basic_amin[0],basic_amax[0],basic_sigma[0],basic_s[0])
    elif 'PAHLarge' in dust_type:
        dist = LogNormal_Distribution(basic_a0[1],basic_amin[1],basic_amax[1],basic_sigma[1],basic_s[1])
    elif 'CSmall' in dust_type:
        dist = LogNormal_Distribution(basic_a0[2],basic_amin[2],basic_amax[2],basic_sigma[2],basic_s[2])
    elif 'CLarge' in dust_type:
        dist = LogNormal_Distribution(basic_a0[3],basic_amin[3],basic_amax[3],basic_sigma[3],basic_s[3])
    elif 'SilSmall' in dust_type:
        dist = LogNormal_Distribution(basic_a0[5],basic_amin[5],basic_amax[5],basic_sigma[5],basic_s[5])
    elif 'SilLarge' in dust_type:
        dist = LogNormal_Distribution(basic_a0[6],basic_amin[6],basic_amax[6],basic_sigma[6],basic_s[6])
    
    # 3. Return the cross sections
    if do_average:
        # Obtain the number of wavelengths using the length of the
        # of the w(micron) column of the first of the data dictionary
        nwav = data[list(data.keys())[0]].shape[0]
        wavelengths = data[list(data.keys())[0]][:,columns.index('w(micron)')]
        # Get the number of grain sizes by the length of the data dictionary
        nrad = len(data)
        C_sca_eff = np.zeros(nwav)
        C_abs_eff = np.zeros(nwav)
        C_rp_eff = np.zeros(nwav)
        
        # Loop over the wavelengths
        for i in range(0,nwav):
            # Construct arrays for all grain sizes
            Q_sca = np.zeros(nrad)
            Q_abs = np.zeros(nrad)
            Q_rp = np.zeros(nrad)
            sizes = np.zeros(nrad)
            # Loop over the grain sizes
            for j,a in enumerate(data):
                sizes[j] = float(a)
                # Get the efficiencies
                Q_sca[j] = data[a][i,columns.index('Q_sca')]
                Q_abs[j] = data[a][i,columns.index('Q_abs')]
                g = data[a][i,columns.index('g=<cos>')]
                # Compute the radiation pressure efficiency
                Q_rp[j] = Q_abs[j] + (1-g)*Q_sca[j]
            # Compute the average efficiencies
            C_sca_eff[i] = dist.averaged_over_number(Q_sca*np.pi*sizes**2,sizes)
            C_abs_eff[i] = dist.averaged_over_number(Q_abs*np.pi*sizes**2,sizes)
            C_rp_eff[i] = dist.averaged_over_number(Q_rp*np.pi*sizes**2,sizes)
        return dist.a0*1e-4,wavelengths*1e-4,C_sca_eff* 1e-8,C_abs_eff* 1e-8,C_rp_eff* 1e-8  # Convert cross section from micron^2 to cm^2
    else:
        # Compute the cross section by looking for the nearest grain size
        # in the data dictionary. If not found, interpolate
        C_sca = np.zeros(len(data[list(data.keys())[0]][:,columns.index('Q_sca')]))
        C_abs = np.zeros(len(data[list(data.keys())[0]][:,columns.index('Q_abs')]))
        C_rp = np.zeros(len(data[list(data.keys())[0]][:,columns.index('Q_abs')]))
        wavelengths = data[list(data.keys())[0]][:,columns.index('w(micron)')]

        # Check if the size dist.a0 is in the data dictionary
        if str(dist.a0) in data:
            C_sca = data[str(dist.a0)][:,columns.index('Q_sca')]
            C_abs = data[str(dist.a0)][:,columns.index('Q_abs')]
            g = data[str(dist.a0)][:,columns.index('g=<cos>')]
            C_rp = C_abs + (1-g)*C_sca
        else:
            # Interpolate
            for i in range(0,len(C_sca)):
                a = np.array([float(r) for r in data.keys()])
                Q_sca = np.array([d[i,columns.index('Q_sca')] for d in data.values()])
                Q_abs = np.array([d[i,columns.index('Q_abs')] for d in data.values()])
                g = np.array([d[i,columns.index('g=<cos>')] for d in data.values()])
                C_sca[i] = 10.**np.interp(np.log10(dist.a0),np.log10(a),np.log10(Q_sca)) * np.pi * dist.a0**2
                C_abs[i] = 10.**np.interp(np.log10(dist.a0),np.log10(a),np.log10(Q_abs)) * np.pi * dist.a0**2
                g = np.interp(dist.a0,a,g)
                C_rp[i] = C_abs[i] + (1-g)*C_sca[i]
        return dist.a0*1e-4,wavelengths*1e-4,C_sca* 1e-8,C_abs* 1e-8,C_rp* 1e-8
        
def planck_function(wavelength, T):
    """This function computes the Planck function for a given wavelength

    Args:
        wavelength (np.array): The wavelength in cm
        T (np.float): The temperature in K

    Returns:
        np.float: Emittance in erg/s/cm^2/cm/steradian
    """    
    return (2. * h * c**2. / wavelength**5.) / (np.exp((h * c / wavelength) / (kb * T)) - 1.)

def planck_function_derivative(wavelength, T):
    """This function computes the derivative of the Planck function for a given wavelength

    Args:
        wavelength (np.array): The wavelength in cm
        T (np.float): The temperature in K

    Returns:
        np.float: Derivative of emittance in erg/s/cm^2/cm/steradian/K
    """    
    return (2. * h * c**2. / wavelength**5.) / (np.exp((h * c / wavelength) / (kb * T)) - 1.)**2. * (h * c / wavelength) / (kb * T**2.)

def absorbed_power(wavelengths,radiation_field,C_abs):
    """This function computes the absorbed power by a dust grain given a radiation field
    and the absorption cross section.

    Args:
        wavelengths (np.array): The wavelength in cm
        radiation_field (np.array): The radiation field in erg/s/cm^2/cm
        C_abs (np.array): The absorption cross section in cm^2

    Returns:
        np.float: The absorbed power in erg/s
    """    
    
    # 1. Compute the absorbed power
    absorbed_power = np.trapz(radiation_field * C_abs, x=wavelengths)
    
    return absorbed_power

def absorbed_power_reemited(wavelengths,Tdust,C_abs):

    # 1. Compute the emitted power
    emp = np.zeros(len(wavelengths))
    for i in range(0,len(wavelengths)):
        emp[i] = planck_function(wavelengths[i],Tdust)
    

def emitted_power(Tdust,wavelengths,C_abs):
    """This function computes the emitted power by a dust grain given a temperature
    and the absorption cross section.

    Args:
        Tdust (np.float): The dust temperature in K
        wavelengths (np.array): The wavelength in cm
        C_abs (np.array): The absorption cross section in cm^2

    Returns:
        np.float: The emitted power in erg/s
    """    
    
    # 1. Compute the emitted power
    emp = np.zeros(len(wavelengths))
    for i in range(0,len(wavelengths)):
        emp[i] = planck_function(wavelengths[i],Tdust)
    emitted_power = 4. * np.pi * np.trapz(C_abs * emp, x=wavelengths)
    return emitted_power

def compute_equilibrium_temperature(wavelengths,wavelengths_em,radiation_field,C_abs,C_abs_em):
    """This function computes the equilibrium temperature of a dust grain given a radiation field
    and the absorption cross section.

    Args:
        wavelengths (np.array): The wavelength in cm
        radiation_field (np.array): The radiation field in erg/s/cm^2/cm
        C_abs (np.array): The absorption cross section in cm^2

    Raises:
        RuntimeError: If the solution did not converge

    Returns:
        np.float: The equilibrium temperature in K
    """    
    
    # 1. Define the function to be solved
    func = lambda T: absorbed_power(wavelengths,radiation_field,C_abs) - emitted_power(T,wavelengths_em,C_abs_em)
    result = root_scalar(func, bracket=[2.7, 800])  # Reasonable temperature range in K
    
    # 2. Check if the solution converged
    if result.converged:
        return result.root
    else:
        raise RuntimeError("Failed to find equilibrium temperature")
    
def compute_eqT_withcollisions(wavelengths,wavelengths_em,radiation_field,C_abs,C_abs_em,
                               ne,nH,nHe,nC,Tgas,T_dust_collisional,
                               electron_rate_table,H_rate_table,
                               He_rate_table,C_rate_table):
    """This function computes the equilibrium temperature of a dust grain given a radiation field
    and the absorption cross section.

    Args:
        wavelengths (np.array): The wavelength in cm
        radiation_field (np.array): The radiation field in erg/s/cm^2/cm
        C_abs (np.array): The absorption cross section in cm^2
        wavelengths_em (np.array): The wavelength in cm for the emission
        C_abs_em (np.array): The absorption cross section in cm^2 for the emission

    Raises:
        RuntimeError: If the solution did not converge

    Returns:
        np.float: The equilibrium temperature in K
    """    
    from dust_collisional_cooling import compute_dust_coll_heating
    
    # 1. Define the function to be solved
    func = lambda T: absorbed_power(wavelengths,radiation_field,C_abs) \
                    + compute_dust_coll_heating(ne,nH,nHe,nC,Tgas,T,T_dust_collisional,
                                                electron_rate_table,H_rate_table,
                                                He_rate_table,C_rate_table) \
                        - emitted_power(T,wavelengths_em,C_abs_em)
    result = root_scalar(func, bracket=[2.7, 800])  # Reasonable temperature range in K
    
    # 2. Check if the solution converged
    if result.converged:
        return result.root
    else:
        raise RuntimeError("Failed to find equilibrium temperature")
    
def compute_eqT_withcollisions_newton(dust_type,a,wavelengths,wavelengths_em,
                                      radiation_field,C_abs,C_abs_em,
                                      ne,nH,nHe,nC,Tgas,T_dust_collisional,
                                      electron_rate_table,H_rate_table,
                                      He_rate_table,C_rate_table):
    """This function computes the equilibrium temperature of a dust grain given a radiation field
    and the absorption cross section.

    Args:
        wavelengths (np.array): The wavelength in cm
        radiation_field (np.array): The radiation field in erg/s/cm^2/cm
        C_abs (np.array): The absorption cross section in cm^2
        wavelengths_em (np.array): The wavelength in cm for the emission
        C_abs_em (np.array): The absorption cross section in cm^2 for the emission

    Raises:
        RuntimeError: If the solution did not converge

    Returns:
        np.float: The equilibrium temperature in K
    """    
    from dust_collisional_cooling import compute_dust_coll_heating

    def f(wavelengths,wavelengths_em,radiation_field,C_abs,C_abs_em,
                               ne,nH,nHe,nC,Tgas,T,T_dust_collisional,
                               electron_rate_table,H_rate_table,
                               He_rate_table,C_rate_table):
        emission = emitted_power(T,wavelengths_em,C_abs_em)
        absorbed = absorbed_power(wavelengths,radiation_field,C_abs)
        collheat = compute_dust_coll_heating(ne,nH,nHe,nC,Tgas,T,T_dust_collisional,
                                                electron_rate_table,H_rate_table,
                                                He_rate_table,C_rate_table)
        return emission - (absorbed + collheat)
    
    def df_dT(wavelengths,wavelengths_em,radiation_field,C_abs,C_abs_em,
                               ne,nH,nHe,nC,Tgas,T,T_dust_collisional,
                               electron_rate_table,H_rate_table,
                               He_rate_table,C_rate_table):
        dT = 1e-4 * T
        f1 = f(wavelengths,wavelengths_em,radiation_field,C_abs,C_abs_em,
                                 ne,nH,nHe,nC,Tgas,T + dT,T_dust_collisional,
                                 electron_rate_table,H_rate_table,
                                 He_rate_table,C_rate_table)
        f2 = f(wavelengths,wavelengths_em,radiation_field,C_abs,C_abs_em,
                                    ne,nH,nHe,nC,Tgas,T - dT,T_dust_collisional,
                                    electron_rate_table,H_rate_table,
                                    He_rate_table,C_rate_table)
        return (f1 - f2) / (2 * dT)
    
    def newton_method(wavelengths,wavelengths_em,radiation_field,C_abs,C_abs_em,
                               ne,nH,nHe,nC,Tgas,T_dust_collisional,
                               electron_rate_table,H_rate_table,
                               He_rate_table,C_rate_table,
                               T0=20.,tol=1e-3,max_iter=100):
        T = T0
        for i in range(max_iter):
            f_val = f(wavelengths,wavelengths_em,radiation_field,C_abs,C_abs_em,
                                 ne,nH,nHe,nC,Tgas,T,T_dust_collisional,
                                 electron_rate_table,H_rate_table,
                                 He_rate_table,C_rate_table)
            df_val = df_dT(wavelengths,wavelengths_em,radiation_field,C_abs,C_abs_em,
                                 ne,nH,nHe,nC,Tgas,T,T_dust_collisional,
                                 electron_rate_table,H_rate_table,
                                 He_rate_table,C_rate_table)
            if df_val == 0:
                raise RuntimeError("Derivative is zero, cannot continue Newton's method")
            T_new = T - f_val / df_val
            if abs(T_new - T) < tol:
                print('Converged in', i, 'iterations at Tgas=', Tgas)
                return T_new
            T = T_new
        raise RuntimeError("Newton's method did not converge within the maximum number of iterations")
        
    # 1. Compute a guess of the equilibrium temperature using the cheap method
    T_guess = compute_equilibrium_temperature_cheap(dust_type,a,wavelengths,radiation_field,C_abs)

    # 2. Use the Newton's method to find the equilibrium temperature
    T_eq = newton_method(wavelengths,wavelengths_em,radiation_field,C_abs,C_abs_em,
                         ne,nH,nHe,nC,Tgas,T_dust_collisional,
                         electron_rate_table,H_rate_table,
                         He_rate_table,C_rate_table,
                         T0=T_guess)
    print('Initial guess for equilibrium temperature:', T_guess, 'final equilibrium temperature:', T_eq)
    return T_eq

def compute_equilibrium_temperature_cheap(dust_type,a,wavelengths,radiation_field,C_abs):
    
    # 1. Compute the absorbed power
    abs_power = absorbed_power(wavelengths,radiation_field,C_abs)

    # 2. Compute the emission cross-section based on the approximations by Draine 2008 (eqs. 24.15 and 24.16)
    if 'Sil' in dust_type:
        C_em = 4. * np.pi * (a)**2. * 1.3e-6 * (a*1e4/0.1)
    elif 'C' in dust_type:
        C_em = 4. * np.pi * (a)**2. * 8e-7 * (a*1e4/0.1)

    # 3. Solve for Td based on the scaling in the Draine 2008 approximations
    Td = (abs_power / (C_em*sigma_sb)) **(1./6.)

    return Td
    
def mathis_radiation_field(l):
    """This function computes the Mathis radiation field as a function of the wavelength

    Args:
        l (float or np.array): The wavelength in Angstrom

    Returns:
        float or np.array: erg cm-2 s-1 Å-1 sr-1
    """    
    
    return (np.tanh(4.07e-3*l-4.5991) + 1.) * 107.192 * l**(-2.89)

import numpy as np

def modified_mmp83_radiation_field(wavelength):
    """
    Calculate the modified MMP83 radiation field (Draine 2011) in units of erg/cm^3.

    Parameters:
    wavelength : float or numpy array
        Wavelength in cm.

    Returns:
    u_lambda : float or numpy array
        Radiation field energy density in erg/cm^3.
    """
    # Convert wavelength to microns and angstroms for convenience
    wavelength_micron = wavelength * 1e4  # cm to micron conversion
    wavelength_angstrom = wavelength * 1e8  # cm to Å conversion

    # Initialize u_lambda
    u_lambda_uv = np.zeros_like(wavelength)

    # UV component (equation 10 in the screenshot)
    mask1 = (1340 < wavelength_angstrom) & (wavelength_angstrom <= 2460)
    u_lambda_uv[mask1] = 2.373e-14 * (wavelength_micron[mask1])**-0.6678

    mask2 = (1100 < wavelength_angstrom) & (wavelength_angstrom <= 1340)
    u_lambda_uv[mask2] = 6.825e-13 * wavelength_micron[mask2]

    mask3 = (912 < wavelength_angstrom) & (wavelength_angstrom <= 1100)
    u_lambda_uv[mask3] = 1.287e-9 * (wavelength_micron[mask3])**4.4172

    # Optical component: sum of three blackbody radiation terms
    T_values = [3000, 4000, 7500]  # Temperatures in K
    W_values = [7e-13, 1.65e-13, 1e-14]  # Dilution factors

    u_lambda_optical = np.zeros_like(wavelength)
    for T, W in zip(T_values, W_values):
        B_lambda = (2 * h * c**2 / wavelength**5) / (np.exp(h * c / (wavelength * kb * T)) - 1)
        u_lambda_optical += (4 * np.pi / c) * W * B_lambda

    # CMB component
    T_CMB = 2.725  # CMB temperature in K
    B_lambda_CMB = (2 * h * c**2 / wavelength**5) / (np.exp(h * c / (wavelength * kb * T_CMB)) - 1)
    u_lambda_CMB = (4 * np.pi / c) * B_lambda_CMB

    # Total radiation field energy density u_lambda
    u_lambda = (u_lambda_uv + wavelength * u_lambda_optical) + wavelength * u_lambda_CMB

    return u_lambda

def plot_compare_radiation_fields():
    # This function compares the radiation fields from Mathis 1983, the modified MMP83 radiation field and the Draine 2011 radiation field
    wav = np.logspace(np.log10(0.0912*1e-4),np.log10(1000*1e-4),100) # in cm
    
    # 1. Draine ISRF is given in photons per cm^2/s/nm
    I_draine = Draine_1978_isrf(wav*1e7) # in photons/cm^2/s/nm
    I_draine = I_draine * h * 1e7 # in erg/cm^3
    
    # 2. Mathis ISRF is given in erg cm-2 s-1 Å-1
    I_mathis = 4. * np.pi * mathis_radiation_field(wav*1e8) # in erg cm-2 s-1 Å-1
    I_mathis = I_mathis / c * wav*1e8 # in erg/cm^3
    
    # 3. Modified MMP83 radiation field is given in erg/cm^3
    I_mmp83 = modified_mmp83_radiation_field(wav) # in erg/cm^3
    
    # 4. Setup the figure
    fig, ax = plt.subplots(1,1,figsize=(6,4),dpi=300,facecolor='w',edgecolor='k')
    ax.set_xlabel(r'$\lambda$ [$\mu$m]',fontsize=20)
    ax.set_ylabel(r'$\lambda u_{\lambda}$ [erg cm$^{-3}$]',fontsize=20)
    ax.tick_params
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.minorticks_on()
    ax.tick_params(which='both',axis="both",direction="in")
    
    # 5. Plot the results
    ax.plot(wav*1e4,I_mmp83,label='Modified Mathis et al. (1983)',color='k',linestyle='-',linewidth=2.5)
    ax.plot(wav*1e4,I_mathis,label='Mathis et al. (1983)',color='r',linestyle='-',linewidth=2.5)
    ax.plot(wav*1e4,I_draine,label='Draine (2011)',color='b',linestyle='-',linewidth=2.5)
    
    # 6. Finalise the figure and save
    ax.legend(loc='best',fontsize=14,frameon=False)
    fig.subplots_adjust(top=0.99,bottom=0.13,left=0.13,right=0.99,hspace=0,wspace=0)
    fig.savefig('./radiation_fields.png', format='png', dpi=300)
    

def plot_equilibrium_temperature(dust_types,nG0=100,G0min=1e-1,G0max=1e7):
    
    # 1. Define the radiation field
    G0 = np.logspace(np.log10(G0min),np.log10(G0max),nG0)
    # wav = np.linspace(91.2,240,1000) #in nm
    # radiation_field = np.zeros((len(wav),2))
    # radiation_field[:,0] = wav * 1e-7 # Convert to cm
    # radiation_field[:,1] = (h * c / (wav * 1e-7)) * Draine_1978_isrf(wav) * 1e7 # Convert to erg/s/cm^2/cm
    # wav = np.linspace(912,2460,10000) #in Angstrom
    # radiation_field = np.zeros((len(wav),2))
    # radiation_field[:,0] = wav * 1e-8 # Convert to cm
    # radiation_field[:,1] = 4. * np.pi * 1e8 * mathis_radiation_field(wav) # in erg/cm^2/s/cm
    wav = np.logspace(np.log10(0.0912*1e-4),np.log10(1000*1e-4),100) # in cm
    radiation_field = np.zeros((len(wav),2))
    radiation_field[:,0] = wav
    radiation_field[:,1] = modified_mmp83_radiation_field(wav) / wav * c # erg/cm^2/cm/s
    
    # radiation_field[(radiation_field[:,0]<2000*1e-8),1] = 0.0
    
    wavelengths_em = np.logspace(np.log10(0.1),np.log10(1000),1000) * 1e-4 # Convert to cm
    
    # 2. Setup the figures
    fig, ax = plt.subplots(1,1,figsize=(6,4),dpi=300,facecolor='w',edgecolor='k')
    ax.set_xlabel(r'$G_0$',fontsize=20)
    ax.set_ylabel(r'$T_{\rm eq}$ [K]',fontsize=20)
    ax.tick_params
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.minorticks_on()
    ax.tick_params(which='both',axis="both",direction="in")
    
    fig2, ax2 = plt.subplots(1,1,figsize=(6,4),dpi=300,facecolor='w',edgecolor='k')
    ax2.set_xlabel(r'$1/\lambda$ [$\mu$m$^{-1}$]',fontsize=20)
    ax2.set_ylabel(r'$Q_{\rm abs}/a$ [$\mu$m$^{-1}$]',fontsize=20)
    ax2.tick_params
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.xaxis.set_ticks_position('both')
    ax2.yaxis.set_ticks_position('both')
    ax2.minorticks_on()
    ax2.tick_params(which='both',axis="both",direction="in")

    # List of line colors and styles for the number of dust types
    colors = ['k','r','b','g','m','c']
    linestyles = ['-','--','-.',':']
    
    for dust_type in dust_types:
        # 3A. Obtain the absorption cross section and interpolate over the wavelengths
        a0, wavelengths,C_sca,C_abs,C_rp = compute_cross_sections(dust_type,do_average=False)
        C_abs_interp = np.interp(radiation_field[:,0],wavelengths[::-1],C_abs[::-1])
        C_abs_em_interp = np.interp(wavelengths_em,wavelengths[::-1],C_abs[::-1])
        print('Absorption cross section for',dust_type,'computed')
        # 3B. Compute the radiation field averaged cross section
        int_radfield = np.trapz(radiation_field[:,1],x=radiation_field[:,0])
        C_abs_avg = np.trapz(C_abs_interp * radiation_field[:,1],x=radiation_field[:,0]) / int_radfield /(np.pi*a0**2.)
        print('Average absorption cross section for',dust_type,'computed')
        print('Given by',C_abs_avg)
        linestyle = linestyles.pop()
        color = colors.pop()
        # Plot the cross section for the dust type in a second figure
        ax2.plot(1./(radiation_field[:,0]*1e4),C_abs_interp/(np.pi*a0**2.)/(a0*1e4),label=dust_type,color=color,
                 linestyle=linestyle,linewidth=2.5,alpha=0.5)
        ax2.plot(1./(wavelengths*1e4),C_abs/(np.pi*a0**2.)/(a0*1e4),label=dust_type,color=color,
                 linestyle=linestyle,linewidth=2.5)

        
        # 3C. Compute the equilibrium temperature
        Teq = np.zeros(nG0)
        def compute_temp(i):
            return compute_equilibrium_temperature(radiation_field[:,0],
                               wavelengths_em,
                               G0[i]*radiation_field[:,1],
                               C_abs_interp,C_abs_em_interp)
        def compute_temp_cheap(i):
            return compute_equilibrium_temperature_cheap(dust_type,
                                a0,radiation_field[:,0],
                                G0[i]*radiation_field[:,1],
                                C_abs_interp)

        Teq = Parallel(n_jobs=-1)(delayed(compute_temp)(i) for i in range(nG0))

        Teq_cheap = Parallel(n_jobs=-1)(delayed(compute_temp_cheap)(i) for i in range(nG0))
    
        # 3C. Plot the results
        ax.plot(G0,Teq,label=dust_type,color=color,linestyle=linestyle,linewidth=2.5)
        ax.plot(G0,Teq_cheap,color=color,linestyle=linestyle,linewidth=2.5,alpha=0.6)

    # 4. Finalise the figure and save
    ax.legend(loc='best',fontsize=14,frameon=False)
    fig.subplots_adjust(top=0.99,bottom=0.12,left=0.1,right=0.99,hspace=0,wspace=0)
    fig.savefig('./equilibrium_temperature.pdf', format='pdf', dpi=300)
    
    ax2.legend(loc='best',fontsize=14,frameon=False)
    fig2.subplots_adjust(top=0.99,bottom=0.125,left=0.12,right=0.99,hspace=0,wspace=0)
    fig2.savefig('./absorption_cross_sections.pdf', format='pdf', dpi=300)
    
def plot_emission_spectra(dust_types,G0=[1.]):
    
    # 1. Define the radiation field
    wav = np.logspace(np.log10(0.0912*1e-4),np.log10(1000*1e-4),100) # in cm
    radiation_field = np.zeros((len(wav),2))
    radiation_field[:,0] = wav
    radiation_field[:,1] = modified_mmp83_radiation_field(wav) / wav * c # erg/cm^2/cm/s
    
    wavelengths_em = np.logspace(np.log10(0.1),np.log10(1000),1000) * 1e-4 # Convert to cm
    
    # 2. Setup the figures
    fig, ax = plt.subplots(1,1,figsize=(6,4),dpi=300,facecolor='w',edgecolor='k')
    ax.set_xlabel(r'$\lambda$ [$\mu$m]',fontsize=20)
    ax.set_ylabel(r'$\lambda L_{\lambda}$ [erg/s]',fontsize=20)
    ax.tick_params
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.minorticks_on()
    ax.tick_params(which='both',axis="both",direction="in")
    ax.set_ylim([1e-20,1e-5])

    # List of line colors and styles for the number of dust types
    colors = ['k','r','b','g','m','c']
    linestyles = ['-','--','-.',':']
    
    import matplotlib.cm as cm

    for idx, dust_type in enumerate(dust_types):
        # 3A. Obtain the absorption cross section and interpolate over the wavelengths
        a0, wavelengths, C_sca, C_abs, C_rp = compute_cross_sections(dust_type, do_average=False)
        C_abs_interp = np.interp(radiation_field[:, 0], wavelengths[::-1], C_abs[::-1])
        C_abs_em_interp = np.interp(wavelengths_em, wavelengths[::-1], C_abs[::-1])
        print('Absorption cross section for', dust_type, 'computed')
        # 3B. Compute the radiation field averaged cross section
        int_radfield = np.trapz(radiation_field[:, 1], x=radiation_field[:, 0])
        C_abs_avg = np.trapz(C_abs_interp * radiation_field[:, 1], x=radiation_field[:, 0]) / int_radfield / (np.pi * a0**2.)
        print('Average absorption cross section for', dust_type, 'computed')
        print('Given by', C_abs_avg)
        linestyle = linestyles.pop()
        
        # 3C. Compute the equilibrium temperature
        for g0_idx, g0 in enumerate(G0):
            Teq = compute_equilibrium_temperature(radiation_field[:, 0],
                                                  wavelengths_em,
                                                  g0 * radiation_field[:, 1],
                                                  C_abs_interp, C_abs_em_interp)
            # 3D. Compute the emitted power
            L_lambda = np.zeros(len(wavelengths_em))
            for i in range(0, len(wavelengths_em)):
                L_lambda[i] = wavelengths_em[i] * planck_function(wavelengths_em[i], Teq) * C_abs_em_interp[i]
            color = cm.viridis(g0_idx / len(G0))
            ax.plot(wavelengths_em * 1e4, 4. * np.pi * L_lambda, label=None,
                    color=color, linestyle=linestyle, linewidth=2.5)
        # Add legend entry for the dust type with black color
        ax.plot([], [], label=dust_type, color='k', linestyle=linestyle, linewidth=2.5)
    # 4. Finalise the figure and save
    ax.legend(loc='best', fontsize=14, frameon=False)
    fig.subplots_adjust(top=0.99,bottom=0.13,left=0.13,right=0.99,hspace=0,wspace=0)
    fig.savefig('./dust_eq_emission_spectra.png', format='png', dpi=300)

def plot_emission_PRIMA_bands(dust_types,G0=[1.]):

    from itertools import cycle
    
    # 1. Define the radiation field
    wav = np.logspace(np.log10(0.0912*1e-4),np.log10(1000*1e-4),100) # in cm
    radiation_field = np.zeros((len(wav),2))
    radiation_field[:,0] = wav
    radiation_field[:,1] = modified_mmp83_radiation_field(wav) / wav * c # erg/cm^2/cm/s
    
    wavelengths_em = np.logspace(np.log10(0.1),np.log10(1000),1000) * 1e-4 # Convert to cm
    
    # 2. Setup the figures
    fig, ax = plt.subplots(1,1,figsize=(6,4),dpi=300,facecolor='w',edgecolor='k')
    ax.set_xlabel(r'$\lambda$ [$\mu$m]',fontsize=20)
    ax.set_ylabel(r'$L_{\lambda}$ [erg/s]',fontsize=20)
    ax.tick_params
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.minorticks_on()
    ax.tick_params(which='both',axis="both",direction="in")
    ax.set_ylim([1e-20,1e-5])

    # List of line colors and styles for the number of dust types
    colors = ['k','r','b','g','m','c']
    linestyles = ['-','--','-.',':']
    
    import matplotlib.cm as cm

    for idx, dust_type in enumerate(dust_types):
        # 3A. Obtain the absorption cross section and interpolate over the wavelengths
        a0, wavelengths, C_sca, C_abs, C_rp = compute_cross_sections(dust_type, do_average=False)
        C_abs_interp = np.interp(radiation_field[:, 0], wavelengths[::-1], C_abs[::-1])
        C_abs_em_interp = np.interp(wavelengths_em, wavelengths[::-1], C_abs[::-1])
        
        print('Absorption cross section for', dust_type, 'computed')
        # Create a dictionary where each key is the luminosity array for a given PRIMA band
        luminosity_prima = {}
        for band in PRIMA_bands.values():
            luminosity_prima[band['band_name']] = np.zeros(len(G0))

        # 3B. Compute the radiation field averaged cross section
        int_radfield = np.trapz(radiation_field[:, 1], x=radiation_field[:, 0])
        # 3C. Compute the equilibrium temperature
        for g0_idx, g0 in enumerate(G0):
            Teq = compute_equilibrium_temperature(radiation_field[:, 0],
                                                  wavelengths_em,
                                                  g0 * radiation_field[:, 1],
                                                  C_abs_interp, C_abs_em_interp)
            # 3D. Loop over the PRIMA bands, integrating the luminosity if it is not a polarimeter
            # and only showing the emission at a single wavelength if it is a polarimeter
            for band in PRIMA_bands.values():
                if band['polarimetry']:
                    wavelength = band['band_center'] * 1e-4
                    C_abs_em_interp = np.interp(wavelength, wavelengths[::-1], C_abs[::-1])
                    L_lambda = planck_function(wavelength, Teq) * C_abs_em_interp
                    luminosity_prima[band['band_name']][g0_idx] = 4. * np.pi * L_lambda
                else:
                    L_lambda = np.zeros(50)
                    wavelength = np.linspace(band['band_min'],band['band_max'],50) * 1e-4
                    C_abs_em_interp = np.interp(wavelength, wavelengths[::-1], C_abs[::-1])
                    for i in range(0, len(wavelength)):
                        L_lambda[i] = planck_function(wavelength[i], Teq) * C_abs_em_interp[i]
                    luminosity_prima[band['band_name']][g0_idx] = 4. * np.pi * np.trapz(L_lambda, x=wavelength)
        # 3E. Plot the results
        for band,linestyle in zip(PRIMA_bands.values(),cycle(linestyles)):
            color = cm.viridis(band['band_center'] / 1000)
            ax.plot(G0, luminosity_prima[band['band_name']], label=band['band_name'],
                    color=color, linestyle=linestyle, linewidth=2.5)
        # Add legend entry for the dust type with black color
        ax.plot([], [], label=dust_type, color='k', linestyle=linestyle, linewidth=2.5)
    # 4. Finalise the figure and save
    ax.legend(loc='best', fontsize=14, frameon=False)
    fig.subplots_adjust(top=0.99,bottom=0.13,left=0.13,right=0.99,hspace=0,wspace=0)
    fig.savefig('./dust_eq_emission_PRIMA_bands.png', format='png', dpi=300)
            

def compute_Rosseland_oppacity(wavelengths,kappa_abs,Td):
    """This function computes the Rosseland mean opacity given the absorption cross section
    and the dust temperature.
    Args:
        wavelengths (np.array): The wavelength in cm
        kappa_abs (np.array): The absorption mass cross section in cm^2/g
        Td (np.float): The dust temperature in K
    Returns:
        np.float: The Rosseland mean opacity in cm^2/g
    """

    # 1. Compute Planck function derivatives for all wavelengths
    dB_dT = planck_function_derivative(wavelengths,Td)

    # 2. Compute the Rosseland mean opacity using the harmonic mean
    integrand_denominator = dB_dT / kappa_abs
    integrand_numerator = dB_dT

    denominator = np.trapz(integrand_denominator,x=wavelengths)
    numerator = np.trapz(integrand_numerator,x=wavelengths)

    return numerator / denominator

def read_HensleyDraine2023_mean_oppacity(file_path):
    """This function reads the mean oppacity from the Hensley & Draine 2023 paper
    and returns a DataFrame with the data.
    Args:
        file_path (str): The path to the file with the data
    Returns:
        pd.DataFrame: The DataFrame with the data
    """
    # Read the file, skipping comment lines (starting with '#')
    df = pd.read_csv(
        file_path,
        delim_whitespace=True,  # Handles whitespace-delimited data
        comment='#',            # Ignores comment lines
        header=None             # No header in the data lines
    )
    
    # Set column names based on the header in the file
    df.columns = ["Temp", "kappa_abs_P", "kappa_ext_P", "kappa_abs_R", "kappa_ext_R"]
    
    # Return the DataFrame
    return df

def read_Semenov2003_mean_oppacity(file_path):
    """This function reads the mean oppacity from the Semenov et al. 2003 paper
    and returns a DataFrame with the data.
    Args:
        file_path (str): The path to the file with the data
    
    Returns:
        pd.DataFrame: The DataFrame with the data
    """

    df = pd.read_csv(
        file_path,
        delim_whitespace=True,  # Handle space-separated values
        header=None,            # No header in the file
        names=["Temperature", "Rosseland_Opacity"]  # Assign column names
    )
    
    return df

def plot_Rosseland_oppacity(dust_types):

    # 1. Setup the figure
    fig, ax = plt.subplots(1,1,figsize=(6,4),dpi=300,facecolor='w',edgecolor='k')
    ax.set_xlabel(r'$T_{\rm D}$ [K]',fontsize=20)
    ax.set_ylabel(r'$\kappa_{\rm R}$ [cm$^2$/g]',fontsize=20)
    ax.tick_params
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.minorticks_on()
    ax.tick_params(which='both',axis="both",direction="in")
    ax.set_ylim([1e-1,2e5])

    # 2. Compute the range of dust temperatures
    Td = np.logspace(np.log10(5),np.log10(1e5),100)

    # 3. List of line colors and styles for the number of dust types
    colors = ['k','r','b','g','m','c']
    linestyles = ['-','--','-.',':']

    for idx, dust_type in enumerate(dust_types):
        # 4.a Obtain the absorption cross section
        a0, wavelengths, C_sca, C_abs, C_rp = compute_cross_sections(dust_type, do_average=True)

        # 4.b Compute the mass absorption cross section based on the grain mass
        if dust_type == 'SilSmall' or dust_type == 'SilLarge':
            dust_s = basic_s[5]
        elif dust_type == 'CSmall' or dust_type == 'CLarge':
            dust_s = basic_s[2]
        dust_mass = dust_s * 3.92e-27 # [g/H] - Dust mass per hydrogen atom
        kappa_abs = C_abs * 3.31e-10 / dust_mass

        # 4.c Compute the Rosseland mean opacity
        kappa_R = np.zeros(len(Td))
        for i in range(0,len(Td)):
            kappa_R[i] = compute_Rosseland_oppacity(wavelengths[::-1],kappa_abs[::-1],Td[i])
        
        # 4.d Plot the results
        linestyle = linestyles.pop()
        color = colors.pop()
        ax.plot(Td,kappa_R,label=dust_type,color=color,linestyle=linestyle,linewidth=2.5)

    # 5. Add the Hensley & Draine 2023 data
    file_path = './hensley_draine_2023/astrodust+PAH_mean_opacities.dat'
    df = read_HensleyDraine2023_mean_oppacity(file_path)
    ax.plot(df["Temp"], df["kappa_abs_R"], label='Hensley \& Draine (2023)', color='b', linestyle='-', linewidth=2.5)

    # 6. Add the power-law scaling in Krumholz & Thompson 2013 paper
    kappa_PLT = 10.**(-1.5) * (Td / 10.)**2.
    ax.plot(Td, kappa_PLT, label='Krumholz \& Thompson (2013)', color='r', linestyle='-', linewidth=2.5)   

    # 7. Add the Semenov et al. 2003 data
    file_path = './semenov_2003/kR.out'
    df = read_Semenov2003_mean_oppacity(file_path)
    ax.plot(df["Temperature"], df["Rosseland_Opacity"], label='Semenov et al. (2003)', color='g', linestyle='-', linewidth=2.5)

    # 8. Add vertical shaded regions for the dust sublimation temperatures with the name of the dust type
    ax.axvspan(1100, 1300, color='k', alpha=0.4)
    ax.text(1200, 1e4, 'Silicate', fontsize=14)
    ax.axvspan(1900, 2100, color='r', alpha=0.4)
    ax.text(2000, 1e2, 'Carbonaceous', fontsize=14)

    # 8. Finalise the figure and save
    ax.legend(loc='best',fontsize=14,frameon=False)
    fig.subplots_adjust(top=0.99,bottom=0.13,left=0.13,right=0.99,hspace=0,wspace=0)
    fig.savefig('./Rosseland_opacity.png', format='png', dpi=300)
        

def plot_eqtemp_withcollision(dust_type,ne,nH,nHe,nC,Tmin,Tmax,nG0=100,nT=10,G0min=1e-1,G0max=1e7):
    """This function computes the equilibrium temperature of a dust grain given a radiation field
    and the absorption cross section, including the effect of collisions with gas particles.

    Args:
        dust_type (str): The type of dust grain
        ne (float): The electron density in cm^-3
        nH (float): The hydrogen density in cm^-3
        nHe (float): The helium density in cm^-3
        nC (float): The carbon density in cm^-3
        Tmin (float): The minimum temperature in K
        Tmax (float): The maximum temperature in K
        nG0 (int): The number of G0 values to compute
        nT (int): The number of temperatures to compute
        G0min (float): The minimum G0 value
        G0max (float): The maximum G0 value
    """
    from dust_collisional_cooling import load_cooling_tables
    
    # 1. Define the radiation field
    G0 = np.logspace(np.log10(G0min),np.log10(G0max),nG0)
    wav = np.logspace(np.log10(0.0912*1e-4),np.log10(1000*1e-4),100) # in cm
    radiation_field = np.zeros((len(wav),2))
    radiation_field[:,0] = wav
    radiation_field[:,1] = modified_mmp83_radiation_field(wav) / wav * c # erg/cm^2/cm/s
    
    wavelengths_em = np.logspace(np.log10(0.1),np.log10(1000),1000) * 1e-4 # Convert to cm
    
    # 2. Obtain the absorption cross section and interpolate over the wavelengths
    a0, wavelengths,C_sca,C_abs,C_rp = compute_cross_sections(dust_type,do_average=False)
    C_abs_interp = np.interp(radiation_field[:,0],wavelengths[::-1],C_abs[::-1])
    C_abs_em_interp = np.interp(wavelengths_em,wavelengths[::-1],C_abs[::-1])
    
    # 3. Compute the radiation field averaged cross section
    int_radfield = np.trapz(radiation_field[:,1],x=radiation_field[:,0])
    C_abs_avg = np.trapz(C_abs_interp * radiation_field[:,1],x=radiation_field[:,0]) / int_radfield /(np.pi*a0**2.)
    print('Average absorption cross section for',dust_type,'computed')
    print('Given by',C_abs_avg)

    # 4. Create the figure
    fig, ax = plt.subplots(1,1,figsize=(6,4),dpi=300,facecolor='w',edgecolor='k')
    ax.set_xlabel(r'$G_0$',fontsize=20)
    ax.set_ylabel(r'$T_{\rm eq}$ [K]',fontsize=20)
    ax.tick_params
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.minorticks_on()
    ax.tick_params(which='both',axis="both",direction="in")

    # 5. Load the rate tables for the collisions
    coll_tables = load_cooling_tables()
    a0 = a0 * 1e4  # Convert a0 to micron
    electron_rate_table = coll_tables[f'electron_cooling_{a0:.4f}_micron_{dust_type[:3]}']['logH']
    H_rate_table = coll_tables[f'H_cooling_{a0:.4f}_micron_{dust_type[:3]}']['logH']
    He_rate_table = coll_tables[f'He_cooling_{a0:.4f}_micron_{dust_type[:3]}']['logH']
    C_rate_table = coll_tables[f'C_cooling_{a0:.4f}_micron_{dust_type[:3]}']['logH']
    T_dust_collisional = coll_tables[f'C_cooling_{a0:.4f}_micron_{dust_type[:3]}']['logT']
    
    # 6. Set the range of temperatures and loop, with increasing temperature
    # having a different color with a colormap
    import matplotlib as mpl
    cmap = plt.get_cmap('viridis')
    T = np.logspace(np.log10(Tmin),np.log10(Tmax),nT)
    for i in range(0,nT):
        
        # 7. Compute the equilibrium temperature
        Teq = np.zeros(nG0)
        def compute_temp_coll(j):
            return compute_eqT_withcollisions(radiation_field[:,0],
                               wavelengths_em,
                               G0[j]*radiation_field[:,1],
                               C_abs_interp,C_abs_em_interp,
                               ne,nH,nHe,nC,T[i],T_dust_collisional,
                               electron_rate_table,H_rate_table,
                               He_rate_table,C_rate_table)
        
        def compute_temp_coll_newton(j):
            return compute_eqT_withcollisions_newton(
                               dust_type,a0*1e-4,
                               radiation_field[:,0],
                               wavelengths_em,
                               G0[j]*radiation_field[:,1],
                               C_abs_interp,C_abs_em_interp,
                               ne,nH,nHe,nC,T[i],T_dust_collisional,
                               electron_rate_table,H_rate_table,
                               He_rate_table,C_rate_table)

        def compute_temp(j):
            return compute_equilibrium_temperature(radiation_field[:,0],
                               wavelengths_em,
                               G0[j]*radiation_field[:,1],
                               C_abs_interp,C_abs_em_interp)
        
        Teq_coll = Parallel(n_jobs=-1)(delayed(compute_temp_coll)(j) for j in range(nG0))

        Teq_coll_newton = Parallel(n_jobs=-1)(delayed(compute_temp_coll_newton)(j) for j in range(nG0))

        Teq = Parallel(n_jobs=-1)(delayed(compute_temp)(j) for j in range(nG0))
        
        # 8. Plot the results with a colormap
        color = cmap(i / nT)
        ax.plot(G0,Teq_coll,color=color,linewidth=2.5)
        ax.plot(G0,Teq_coll_newton,color=color,linewidth=2.5,linestyle=':',alpha=0.6)
        ax.plot(G0,Teq,color='k',linewidth=2.5,alpha=0.3)
    
    # 9. Add a log colorbar
    norm = mpl.colors.LogNorm(vmin=Tmin, vmax=Tmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.02)
    cbar.set_label(r'$T_{\rm gas}$ [K]', fontsize=20)

    # 10. Save figure
    fig.subplots_adjust(top=0.99, bottom=0.13, left=0.13, right=0.99, hspace=0, wspace=0)
    fig.savefig(f'./eqtemp_withcollisions_{dust_type}.pdf', format='pdf', dpi=300)
