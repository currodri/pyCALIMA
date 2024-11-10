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
from dust_model import basic_a0,basic_amin,basic_amax,basic_sigma,basic_s,LogNormal_Distribution
from dust_oppacity import dust_efficiencies,pah_efficiencies
from PAHs_model import Draine_1978_isrf

# Constants
kb               = 1.3806488e-16 # [erg/K] - Boltzmann constant
c                = 2.99792458e10 # [cm/s] - Speed of light
h                = 6.6260755e-27 # [erg s] - Planck constant

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
        data, columns, name = dust_efficiencies(filename)
    elif dust_type == 'CSmall' or dust_type == 'CLarge':
        filename = './draine_lee_1984/Gra_81'
        data, columns, name = dust_efficiencies(filename)
    elif dust_type == 'iPAHSmall' or dust_type == 'iPAHLarge':
        filename = './li_draine_2001/PAHion_30'
        data, columns, name = pah_efficiencies(filename)
    elif dust_type == 'nPAHSmall' or dust_type == 'nPAHLarge':
        filename = './li_draine_2001/PAHneu_30'
        data, columns, name = pah_efficiencies(filename)

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
        return C_sca_eff* 1e-8,C_abs_eff* 1e-8,C_rp_eff* 1e-8  # Convert cross section from micron^2 to cm^2
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
        return wavelengths*1e-4,C_sca* 1e-8,C_abs* 1e-8,C_rp* 1e-8
        
def planck_function(wavelength, T):
    """This function computes the Planck function for a given wavelength

    Args:
        wavelength (np.array): The wavelength in cm
        T (np.float): The temperature in K

    Returns:
        np.float: Emittance in erg/s/cm^2/cm/steradian
    """    
    return (2. * h * c**2. / wavelength**5.) / (np.exp(h * c / (wavelength * kb * T)) - 1.)

def absorbed_power(radiation_field,dust_type):
    
    # 1. Compute the cross sections
    wav,C_sca,C_abs,C_rp = compute_cross_sections(dust_type)
    
    # 2. Interpolate the cross section for the wavelengths in the radiation field
    C_abs = np.interp(radiation_field[:,0],wav,C_abs)
    
    # 3. Compute the absorbed power
    absorbed_power = np.trapz(radiation_field[:,1] * C_abs, x=radiation_field[:,0])
    
    return absorbed_power

def emitted_power(Tdust,dust_type,min_wavelength=1e-1,
                  max_wavelength=1e3,nwavelengths=1000):
    
    # 1. Compute the cross sections
    wav,C_sca,C_abs,C_rp = compute_cross_sections(dust_type)
    
    # 2. Interpolate the cross section for the wavelengths in the radiation field
    wanted_wavelengths = np.logspace(np.log10(min_wavelength),np.log10(max_wavelength),nwavelengths)
    C_abs = np.interp(wanted_wavelengths,wav,C_abs)
    
    # 2. Compute the emitted power
    emitted_power = 4. * np.pi * np.trapz(planck_function(C_abs,Tdust), x=wanted_wavelengths)
    
    return emitted_power