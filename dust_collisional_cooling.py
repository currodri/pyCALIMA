"""
DUST COLLISIONAL COOLING


"""

# Import libraries
import numpy as np
import dust_model
import pandas as pd
import os
from tqdm import tqdm
import concurrent.futures
import time
from scipy.integrate import trapezoid

# Set OMP_NUM_THREADS to limit the number of threads used by OpenBLAS
os.environ["OMP_NUM_THREADS"] = "1"  # Set it to the desired number of threads
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# Constants
a_0              = 5.291e-9 # [cm] - atomic length unit
eV2erg           = 1.6021773300241e-12 # [erg] conversion between eV to erg
au2cgs_v         = 2.18769126364e8    # [cm/s] conversion between a.u. velocity and cgs velocity
au2cgs_m         = 1.66053906660e-24 # [g] conversion between a.u. mass and cgs mass
sec2yr           = 3.1536e7 # [s] conversion between yr to sec
kb               = 1.3806488e-16 # [erg/K] - Boltzmann constant
elem_charge      = 4.8032047e-10 # [statC] - elementary charge
c                = 2.99792458e10 # [cm/s] - speed of light
NA               = 6.02214076e23 # [mol-1] - Avogadro's number
U0               = {'C': 4.0, 'Sil': 5.7} # [eV] - surface binding energy

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

def effective_charge_number(z,v_ion):
    """Compute the effective charge number of the ion as it travels
    through the grain material, following the Barkas (1963) equation.

    Args:
        z (float): original ion charge number
        v_ion (float): current ion velocity [cm/s]

    Returns:
        float: effective ion charge charge number
    """    
    
    z_eff = z * (1. - np.exp(-125.*v_ion / c * z**(-2./3.)))
    
    return z_eff

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

def G(E,I,Z):
    a = np.sqrt(np.e/2.) * np.log(1. + (E/I)**2.) * I/E
    b = (1./3.) * np.log(Z/2.) * np.exp(-3./np.sqrt(Z)*(1.-2./np.sqrt(Z)+np.log(E/I))**2.) * E/I

    return 1. - a + b

def mean_excitation(Z):
    return 9.76 * Z + 58.5/Z**0.19

def compute_deposited_energy(E_init,z,m_ion,s_dust,Z_dust,M_dust,a_dust,E_exc,delta_max=0.01,nmax=1000):
    """Obtain the penetration depth using the Bethe-Bloch formula. It describes the mean energy loss
    per distance travelled of a charge particle traversing matter.

    Args:
        E_init (float): initial energy of the ion [eV]
        z (float): ion charge
        m_ion (float): ion atomic mass [g]
        s_dust (float): dust material density [g/cm^3]
        Z_dust (float): average dust atomic number
        M_dust (float): average dust material atomic mass [g]
        a_dust (float): dust grain radius [cm]
        E_exc (float): average excitation energy [eV]
        delta_max (float, optional): maximum change of the ion energy during the integration. Defaults to 0.01.
        nmax (int, optional): maximum number of iterations. Defaults to 1000.

    Returns:
        float,float: deposited energy [eV], penetration depth [nm]
    """  
    
    me = 510.9989461e6 # [eV/c^2]
    
    CBB = 458.2841953 # [eV cm^4 / s^2]
    
    v_ion = np.sqrt(2 * E_init *eV2erg / m_ion) # [cm/s]
    E_now = E_init
    
    dr = a_dust/10. # [cm]
    r_pd = 0 # [cm]
    n = 0
    
    I = E_exc * Z_dust
    mu_dust = Z_dust / M_dust
    # prefactor = CBB * s_dust * mu_dust
    prefactor = 2. * np.pi * elem_charge**4. * NA * s_dust * Z_dust / M_dust
    while E_now > 1e-3*E_init and r_pd < 4./3.*a_dust and dr > 0:
        # 1. Compute the effective charge number
        z_eff = effective_charge_number(z,v_ion)
        
        # 2. Compute the dE/dr
        beta = v_ion / c
        # dE =  prefactor * z_eff**2. / v_ion**2. * (np.log(2.*me*beta**2./(I*(1.-beta**2.))) - beta**2.) # [eV/cm]
        dE = beta
        if dE<0:
            E_now = 0.0
            break
        # 3. Figure out if the step needs to be changed
        if abs(dE*dr)/E_now >= delta_max:
            dr = dr / 2.
        else:
            E_now = E_now - dE * dr
            v_ion = np.sqrt(2 * E_now *eV2erg / m_ion) # [cm/s]
            r_pd = r_pd + dr
            if r_pd + dr >= 4./3.*a_dust:
                dr = 4./3.*a_dust - r_pd
            elif r_pd + dr == 4./3.*a_dust:
                break
        n += 1
    print(E_init,n,E_now)
    
    E_imp = E_init - E_now

    return E_imp,r_pd

def Dwek_1986_electrons(E,a):
    Eth = 3.7e-8 * a**(2./3.)
    
    if E < Eth:
        c = 1.
    else:
        c = 1. - (1.- (Eth/E)**(3./2.))**(2./3.)
    return c

def plot_deposited_energy(E_min,E_max,z_ion,m_ion,nE=100,delta_max=0.01,nmax=1000):
    
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
    a_dust_c = dust_model.basic_a0[2:4]*1e-4 # [cm]
    am_dust_c = 12.011 * au2cgs_m # [g]
    an_dust_c = 6
    E_exec_c = 13.5 # [eV]
    
    # Assuming MgFeSiO4 composition (olivine with Iron inclusions as the regular model)
    rho_dust_sil = dust_model.basic_s[5] # [g/cm^3]
    a_dust_sil = dust_model.basic_a0[5:7]*1.0e-4 # [cm]
    am_dust_sil = (24.305 + 55.845 + 28.0855 + 4*15.999) / 7. * au2cgs_m  # [g]
    an_dust_sil = int((4*8 + 14 + 26 + 12) / 7)
    E_exec_sil = 13.0 # [eV]

    E_imp_c = np.zeros((2,nE))
    E_imp_sil = np.zeros((2,nE))
    
    for i in range(0, nE):
        E_imp_c[0,i], _ = compute_deposited_energy(E[i],z_ion,m_ion,rho_dust_c,
                                                   an_dust_c,am_dust_c,a_dust_c[0],E_exec_c,
                                                   delta_max=delta_max,nmax=nmax)
        E_imp_c[1,i], _ = compute_deposited_energy(E[i],z_ion,m_ion,rho_dust_c,
                                                   an_dust_c,am_dust_c,a_dust_c[1],E_exec_c,
                                                   delta_max=delta_max,nmax=nmax)
        E_imp_sil[0,i], _ = compute_deposited_energy(E[i],z_ion,m_ion,rho_dust_sil,
                                                   an_dust_sil,am_dust_sil,a_dust_sil[0],E_exec_sil,
                                                   delta_max=delta_max,nmax=nmax)
        E_imp_sil[1,i], _ = compute_deposited_energy(E[i],z_ion,m_ion,rho_dust_sil,
                                                   an_dust_sil,am_dust_sil,a_dust_sil[1],E_exec_sil,
                                                   delta_max=delta_max,nmax=nmax)
    
    
    # 4. Setup figure and plot
    fig, ax = plt.subplots(1,1, figsize=(5,4),dpi=300,facecolor='w',edgecolor='k',sharey=True)
    ax.set_ylabel(r'$\zeta_{\rm imp}$ [eV]', fontsize=16)
    ax.set_xlabel(r'$E_{\rm init}$ [keV]',fontsize=16)
    ax.set_xlim([E_min/1e3,E_max/1e3])
    ax.tick_params(labelsize=14)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    # ax.set_yscale('log')
    ax.set_xscale('log')
    ax.minorticks_on()
    ax.tick_params(which='both',axis="both",direction="in")
    
    ax.plot(E/1e3,E_imp_sil[0,:]/E,linestyle='-',color='saddlebrown',label='smallSil')
    ax.plot(E/1e3,E_imp_sil[1,:]/E,linestyle='--',color='saddlebrown',label='largeSil')
    ax.plot(E/1e3,E_imp_c[0,:]/E,linestyle='-',color='royalblue',label='smallC')
    ax.plot(E/1e3,E_imp_c[1,:]/E,linestyle='--',color='royalblue',label='largeC')
    ax.hlines(1,E_min/1e3,E_max/1e3,linestyle=':',color='grey')
    
    # 5. Plot the estimate by Dwek (1986)
    print(a_dust_c,a_dust_sil)
    # c_dwek1986 = np.zeros((3,nE))
    # for i in range(0,nE):
    #     c_dwek1986[0,i] = Dwek_1986(E[i]*eV2erg,a_dust_c[0]*1e4)
    #     c_dwek1986[1,i] = Dwek_1986(E[i]*eV2erg,a_dust_c[1]*1e4)
    #     c_dwek1986[2,i] = Dwek_1986(E[i]*eV2erg,a_dust_sil[0]*1e4)
    
    # ax.plot(E/1e3,c_dwek1986[2,:],linestyle=':',color='k',label=r'$a = 0.005$ $\mu$m Dwek (1986)')
    # ax.plot(E/1e3,c_dwek1986[0,:],linestyle='--',color='k',label=r'$a = 0.01$ $\mu$m Dwek (1986)')
    # ax.plot(E/1e3,c_dwek1986[1,:],linestyle='-',color='k',label=r'$a = 0.1$ $\mu$m Dwek (1986)')
    
    ax.text(0.6, 0.8, r'$z_{\rm ion}=%i$'%int(z_ion)+'\n'+r'$m_{\rm ion}=%.2f$ [a.u.]'%(float(m_ion)/au2cgs_m),
                                    verticalalignment='bottom', horizontalalignment='left',
                                    transform=ax.transAxes,fontsize=13)
    
    ax.legend(loc='best',frameon=False,fontsize=14)
    fig.subplots_adjust(top=0.99,bottom=0.13,left=0.14,right=0.98,hspace=0,wspace=0)
    fig.savefig('test_ion_deposited_energy.png',format='png')
    
    
def HM89_cooling(T,a_min,Td):
    """Hollenbach and McKee (1989) dust collisional cooling
    equation.

    Args:
        T (float): gas temperature [K]
        a_min (float): grain size [AA]
        Td (float): dust temperature [K]

    Returns:
        float: collisional cooling rate [erg cm^3 / s]
    """    
    
    l1 = 1.2e-31 * np.sqrt(T/1000.) * np.sqrt(100./a_min)
    l2 = (1. - 0.8*np.exp(-75./T)) * (T-Td)
    
    return l1*l2

def low_temp_cooling(T,a,Td,M_dust,projectile):
    
    prefactor = np.pi * a**2. * 2. * kb * (T - Td)
    if projectile == 'H':
        # 1. Compute collisions with hydrogen
        alpha_0 = 2. * 1.00784 * M_dust/ (1.00784 + M_dust)**2.
        alpha = (1.-alpha_0)*np.exp(-np.sqrt(2.*(T+Td)/500.)) + alpha_0
        mx = 1.00784 * au2cgs_m
        l = alpha* np.sqrt(8.*kb*T/(np.pi*mx)) * prefactor
    elif projectile == 'He':
        # 2. Compute collisions with helium
        alpha_0 = 2. * 4.002602 * M_dust/ (4.002602 + M_dust)**2.
        mx = 4.002602 * au2cgs_m
        l = alpha_0 * np.sqrt(8.*kb*T/(np.pi*mx)) * prefactor
    else:
        alpha_0 = 1.
        mx = 4.002602 * au2cgs_m
        l = alpha_0 * np.sqrt(8.*kb*T/(np.pi*mx)) * prefactor
    
    return l


def plot_cooling(Tmin,Tmax,nT=100):
    
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
    
    # 1. Setup the figure
    fig, ax = plt.subplots(1,1, figsize=(5,4),dpi=300,facecolor='w',edgecolor='k',sharey=True)
    ax.set_xlabel(r'$T$ [K]', fontsize=16)
    ax.set_ylabel(r'$\Lambda$ [erg cm$^3$ s$^{-1}$]',fontsize=16)
    ax.set_xlim([Tmin,Tmax])
    ax.tick_params(labelsize=14)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.minorticks_on()
    ax.tick_params(which='both',axis="both",direction="in")
    
    T = np.logspace(np.log10(Tmin),np.log10(Tmax),nT)
    
    # 2. Plot the cooling from Hollenbach and McKee (1989)
    ax.plot(T, HM89_cooling(T,100.,2.73),linestyle=':',color='k',label=r'HM89 - $T_{\rm d}=2.73$ K')
    ax.plot(T, HM89_cooling(T,100.,10.),linestyle='-',color='k',label=r'HM89 - $T_{\rm d}=10$ K')
    ax.plot(T, HM89_cooling(T,100.,100.),linestyle='--',color='k',label=r'HM89 - $T_{\rm d}=100$ K')
    
    # 3. Low temperature cooling model
    ax.plot(T, low_temp_cooling(T,1e-6,2.73,12.011),linestyle=':',color='b',label=r'Low-T cooling - $T_{\rm d}=2.73$ K')
    ax.plot(T, low_temp_cooling(T,1e-6,10.,12.011),linestyle='-',color='b',label=r'Low-T cooling - $T_{\rm d}=10$ K')
    ax.plot(T, low_temp_cooling(T,1e-6,100.,12.011),linestyle='--',color='b',label=r'Low-T cooling - $T_{\rm d}=100$ K')
    
    # 4. High temperature cooling model
    
    ax.legend(loc='best',frameon=False,fontsize=12)
    fig.subplots_adjust(top=0.99,bottom=0.13,left=0.16,right=0.97,hspace=0,wspace=0)
    fig.savefig('collisional_cooling_rate.png',format='png')
    
    
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
            

def nuclear_stopping_cs(Zi,Zd,Md,Mi,E):
    """Nuclear stopping cross section.

    Args:
        Zi (float): atomic number of ion
        Zd (float): average atomic number of dust material
        Md (float): average atomic mass of dust material
        Mi (float): atomic mass of ion
        E (float): energy [erg]

    Returns:
        float: nuclear stopping cross section [erg cm^2 atom^-1]
    """    
    
    # 1. Compute the screening length
    a_sc = screening_length(Zi,Zd)
    
    # 2. Compute the reduced nuclear stopping cross-section
    epsilon = reduced_energy(Md,Mi,a_sc,Zi,Zd,E)
    s = screened_Coulomb_function(epsilon)
    
    # 3. Combine all
    S = 4. * np.pi * a_sc * Zi * Zd * elem_charge**2. * Mi / (Mi+Md) * s

    return S

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

def compute_energy_deposited(args):
    
    n_dust,ne_val,a_dust,Zi,Zd,Md,Mi,E,delta_max = args
    
    # 1. Prepare parameters
    if a_dust == None:
        a_dust = 1e10
        dr = 1e-7
    else:
        dr = a_dust / 10.
    r_pd = 0.
    n = 0
    E_now = E
    
    # 2. Compute the screening length
    a_sc = screening_length(Zi,Zd)
    Earray = [E_now]
    distance = [0.0]
    
    while E_now > 1e-3*E and r_pd < 4./3.*a_dust:
        # 1. Compute the reduced nuclear stopping cross-section
        epsilon = reduced_energy(Md,Mi,a_sc,Zi,Zd,E_now)
        s = screened_Coulomb_function(epsilon)
        
        Sn = 4. * np.pi * a_sc * Zi * Zd * elem_charge**2. * Mi / (Mi+Md) * s
        Se = electronic_stopping_cs(n_dust,ne_val,Zi,Zd,Md,Mi,E_now)

        S = n_dust * (Sn + Se)
        
        # 2. Figure out if the step needs to be changed
        if abs(S*dr)/E_now >= delta_max:
            dr = dr / 2.
        else:
            E_now = E_now - S * dr
            Earray.append(E_now)
            r_pd = r_pd + dr
            distance.append(r_pd)
            if r_pd + dr >= 4./3.*a_dust:
                dr = 4./3.*a_dust - r_pd
            elif r_pd + dr == 4./3.*a_dust:
                break
        n += 1
    # 3. Compute the deposited energy
    E_imp = E - E_now
    Earray = np.array(Earray)
    distance = np.array(distance)
    
    return E_imp,r_pd,Earray,distance

def test_deposited_energy(a_dust,s_dust,Zi,Zd,Md,Mi,E,delta_max=0.1):
    
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
    
    # 1. Setup the figure
    fig, ax = plt.subplots(1,1, figsize=(5,4),dpi=300,facecolor='w',edgecolor='k',sharey=True)
    ax.set_xlabel(r'$r$ [nm]', fontsize=16)
    ax.set_ylabel(r'$E$ [eV]',fontsize=16)
    ax.tick_params(labelsize=14)
    # ax.set_yscale('log')
    # ax.set_xscale('log')
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.minorticks_on()
    ax.tick_params(which='both',axis="both",direction="in")
    
    # 2. Compute stopping
    Md = Md * au2cgs_m
    Mi = Mi * au2cgs_m
    E = E * eV2erg
    n_dust = s_dust / Md
    E_imp,Earr,distance = compute_energy_deposited(n_dust,a_dust,Zi,Zd,Md,Mi,E,delta_max=delta_max)
    print(4./3.*a_dust*1e7,distance*1e7,Earr/eV2erg)
    ax.plot(distance*1e7,Earr/eV2erg)
    ax.vlines(4./3.*a_dust*1e7,0.0,Earr.max()/eV2erg,color='k',linestyle='--')
    ax.hlines(0.0,distance.min()*1e7,4./3.*a_dust*1e7,color='k',linestyle='--')
    
    fig.subplots_adjust(top=0.99,bottom=0.13,left=0.16,right=0.97,hspace=0,wspace=0)
    fig.savefig('testing_stopping.png',format='png')
    
def test_deposited_energy(a_dust,s_dust,Zi,Zd,Md,Mi,Emin,Emax,nE=100,delta_max=0.1):
    
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
    
    # 1. Setup the figure
    fig, axes = plt.subplots(2,1, figsize=(5,6),dpi=300,facecolor='w',edgecolor='k',sharex=True)
    axes[1].set_xlabel(r'$E_{\rm init}$ [eV]', fontsize=16)
    axes[0].set_ylabel(r'$\zeta$',fontsize=16)
    axes[1].set_ylabel(r'$r_{\rm pd}$ [nm]',fontsize=16)
    axes[0].tick_params(labelsize=14)
    axes[0].set_xscale('log')
    # ax.set_xscale('log')
    axes[0].xaxis.set_ticks_position('both')
    axes[0].yaxis.set_ticks_position('both')
    axes[0].minorticks_on()
    axes[0].tick_params(which='both',axis="both",direction="in")
    
    axes[1].tick_params(labelsize=14)
    # ax.set_yscale('log')
    axes[1].set_xscale('log')
    axes[1].xaxis.set_ticks_position('both')
    axes[1].yaxis.set_ticks_position('both')
    axes[1].minorticks_on()
    axes[1].tick_params(which='both',axis="both",direction="in")

    
    # 2. Compute stopping
    Md = Md * au2cgs_m
    Mi = Mi * au2cgs_m
    E = np.logspace(np.log10(Emin),np.log10(Emax),nE)
    E = E * eV2erg
    n_dust = s_dust / Md
    zeta = np.zeros(nE)
    r_pd = np.zeros(nE)
    for i in range(0, nE):
        E_imp,r,Earr,distance = compute_energy_deposited(n_dust,a_dust,Zi,Zd,Md,Mi,E[i],delta_max=delta_max)
        zeta[i] = E_imp
        r_pd[i] = r
        
    zeta = zeta / E
    axes[0].plot(E/eV2erg,zeta)
    axes[1].plot(E/eV2erg,r_pd*1e7) 
    
    E_sp = threshold_energy(U0['C'],Md,Mi)
    axes[0].vlines(E_sp,0,1,linestyle='--',color='r')
    axes[1].vlines(E_sp,r_pd.min()*1e7,r_pd.max()*1e7,linestyle='--',color='r')

    
    fig.subplots_adjust(top=0.99,bottom=0.13,left=0.16,right=0.97,hspace=0,wspace=0)
    fig.savefig('testing_deposited_energy.png',format='png')
    
def compute_ion_range(Zi,Mi,Emin,Emax,nE=100,delta_max=0.1):
    
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
    
    # 1. Setup the figure
    fig, ax = plt.subplots(1,1, figsize=(5,3),dpi=300,facecolor='w',edgecolor='k',sharey=True)
    ax.set_xlabel(r'$E$ [eV]', fontsize=16)
    ax.set_ylabel(r'$r_{\rm pd}$ [nm]',fontsize=16)
    ax.tick_params(labelsize=14)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.minorticks_on()
    ax.tick_params(which='both',axis="both",direction="in")
    
    E = np.logspace(np.log10(Emin),np.log10(Emax),nE) * eV2erg
    Mi = Mi * au2cgs_m
    # 1. First compute it for carbonaceous material
    s_dust = dust_model.basic_s[2]
    am_dust = 12.011 * au2cgs_m
    an_dust = 6.
    ne_val = 4
    n_dust = s_dust / am_dust * an_dust
    
    args_list = [(n_dust,ne_val,None,Zi,an_dust,am_dust,Mi,Ei,delta_max) for Ei in E]
    r_pd = np.zeros(nE)
    for i in range(0, nE):
        E_imp,r,Earr,distance = compute_energy_deposited(args_list[i])
        r_pd[i] = r
    ax.plot(E/eV2erg,r_pd*1e7,linestyle='-',color='royalblue',label=r'Carbonaceous material',linewidth=2.)
    
    # 2. Now for the silicate
    s_dust = dust_model.basic_s[5]
    am_dust = (24.305 + 55.845 + 28.0855 + 4*15.999) / 7. * au2cgs_m
    an_dust = int((4*8 + 14 + 26 + 12) / 7)
    ne_val = (2+8+4+4*6) / 7
    n_dust = s_dust / am_dust * an_dust
    print(n_dust)
    
    args_list = [(n_dust,ne_val,None,Zi,an_dust,am_dust,Mi,Ei,delta_max) for Ei in E]
    num_cores = 5 #os.cpu_count()
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(tqdm(executor.map(compute_energy_deposited, args_list), 
                            total=nE, 
                            desc=f'    Calculating ranges',
                            unit=' steps'))

    E_imp,r_pd,Earray,distance = zip(*results)
    r_pd = np.array(r_pd)
    
    ax.plot(E/eV2erg,r_pd*1e7,linestyle='-',color='saddlebrown',label=r'Silicate material',linewidth=2.)
    
    Draine_range = 3e-6 / dust_model.basic_s[5] * (E/eV2erg/1e3)
    ax.plot(E/eV2erg,Draine_range*1e7,linestyle='--',color='k',label=r'Draine 1979 ($\rho=3.3$ g cm$^{-3}$)',linewidth=2.)
    
    Draine_range = 3e-6 / dust_model.basic_s[2] * (E/eV2erg/1e3)
    ax.plot(E/eV2erg,Draine_range*1e7,linestyle=':',color='k',label=r'Draine 1979 ($\rho=2.2$ g cm$^{-3}$)',linewidth=2.)
    
    Glauser_2009 = np.array([[0.5,0.8,1.1,1.3,1.9,2.6,3.3,10.,50],
                             [8.3,12.,15.8,17.7,24.4,31.7,38.8,97.2,332.7]])
    ax.plot(Glauser_2009[0]*1e3,Glauser_2009[1],marker='x',linestyle='',color='g',markeredgewidth=3,label=r'Glauser et al. (2009)',linewidth=2.)
    
    ax.hlines(dust_model.basic_a0)
    
    ax.legend(loc='upper left',frameon=False,fontsize=12)
    fig.subplots_adjust(top=0.99,bottom=0.165,left=0.14,right=0.99,hspace=0,wspace=0)
    fig.savefig('testing_ion_range.pdf',format='pdf')
    
def compute_rate_T(args):
    
    T,a_dust,s_dust,Zi,Zd,Md,Mi,nv,delta_max = args
    
    
    # 1. Determine the minimum velocity for convergence
    v_0 = np.sqrt(2.*kb*T/Mi) * (1. - (np.sqrt(3./2.) - 1.))
    while Maxwell_Boltzmann_function(v_0,Mi,T) > 1e-30:
        v_0 = v_0 / 2.
        
    # 2. We set the maximum velocity to the thermal energy of gas at ~1e10 K
    v_max = np.sqrt(2. * kb * 1e10 / Mi) # [cm/s]
    while Maxwell_Boltzmann_function(v_max,Mi,T) < 1e-30:
        v_max = v_max/ 2.0
        
    dist,x = dust_model.grain_charge_dist(1.0,T,0.1,'carbonaceous','100A',gamma=None)
    D = dust_model.cmp_D_WD99(dist,x,Zi,T,a_dust)
    Echarge = 0.
    for i in range(0, len(dist)):
        Echarge += dist[i] * Zi * x[i] * elem_charge**2. / a_dust
    
    # 3. Perform loop over velocity range
    v = np.logspace(np.log10(v_0),np.log10(v_max),nv)
    H = np.zeros(nv)
    
    n_dust = s_dust / Md
    sigma_dust = np.pi * a_dust**2.
    
    for i in range(0, nv):
        vi = v[i] # [cm/s]
        Ei = 0.5 * Mi * vi**2. + Echarge # [erg]
        mb_factor = Maxwell_Boltzmann_function(vi,Mi,T)
        E_imp,_,_,_ = compute_energy_deposited(n_dust,a_dust,Zi,Zd,Md,Mi,Ei,delta_max=delta_max)
        # print(mb_factor,vi,E_imp)
        H[i] = mb_factor * vi * E_imp
    
    # 4. Integrate with the trapezoid method
    H = D * sigma_dust * H
    H = trapezoid(H,v)
    
    return H

def compute_rate_T_nocharge(args):
    
    T,a_dust,s_dust,Zi,Zd,Md,Mi,nv,delta_max = args
    
    
    # 1. Determine the minimum velocity for convergence
    v_0 = np.sqrt(2.*kb*T/Mi) * (1. - (np.sqrt(3./2.) - 1.))
    while Maxwell_Boltzmann_function(v_0,Mi,T) > 1e-30:
        v_0 = v_0 / 2.
        
    # 2. We set the maximum velocity to the thermal energy of gas at ~1e10 K
    v_max = np.sqrt(2. * kb * 1e10 / Mi) # [cm/s]
    while Maxwell_Boltzmann_function(v_max,Mi,T) < 1e-30:
        v_max = v_max/ 2.0
    
    # 3. Perform loop over velocity range
    v = np.logspace(np.log10(v_0),np.log10(v_max),nv)
    H = np.zeros(nv)
    
    n_dust = s_dust / Md
    sigma_dust = np.pi * a_dust**2.
    
    for i in range(0, nv):
        vi = v[i] # [cm/s]
        Ei = 0.5 * Mi * vi**2. # [erg]
        mb_factor = Maxwell_Boltzmann_function(vi,Mi,T)
        E_imp,_,_,_ = compute_energy_deposited(n_dust,a_dust,Zi,Zd,Md,Mi,Ei,delta_max=delta_max)
        # print(mb_factor,vi,E_imp)
        H[i] = mb_factor * vi * E_imp
    
    # 4. Integrate with the trapezoid method
    H = sigma_dust * H
    H = trapezoid(H,v)
    
    return H

def zeta_Dwek(E,a_dust):
    Eth = 3.7e-8 * (a_dust*1e4)**(2./3.)
    
    if E < Eth:
        zeta = 1.
    else:
        zeta = 1. - (1.-(Eth/E)**(3./2.))**(2./3.)
        
    return zeta

def compute_rate_Dwek_T(args):
    
    T,a_dust,s_dust,Zi,Zd,Md,Mi,nv,delta_max = args
    
    
    # 1. Determine the minimum velocity for convergence
    v_0 = np.sqrt(2.*kb*T/Mi) * (1. - (np.sqrt(3./2.) - 1.))
    while Maxwell_Boltzmann_function(v_0,Mi,T) > 1e-30:
        v_0 = v_0 / 2.
    # v_0 = 0.0
        
    # 2. We set the maximum velocity to the thermal energy of gas at ~1e10 K
    v_max = np.sqrt(2. * kb * 1e10 / Mi) # [cm/s]
    while Maxwell_Boltzmann_function(v_max,Mi,T) < 1e-30:
        v_max = v_max/ 2.0
    # 3. Perform loop over velocity range
    v = np.linspace(v_0,v_max,nv)
    H = np.zeros(nv)
    
    n_dust = s_dust / Md
    sigma_dust = np.pi * a_dust**2.
    
    if Mi == 1.00784:
        Eth = 133. * a_dust * 1e4 * 1e3 * eV2erg
    elif Mi == 4.002602:
        Eth = 222. * a_dust * 1e4 * 1e3 * eV2erg
    else:
        Eth = 665. * a_dust * 1e4 * 1e3 * eV2erg
    for i in range(0, nv):
        vi = v[i] # [cm/s]
        Ei = 0.5 * Mi * vi**2. # [erg]
        mb_factor = Maxwell_Boltzmann_function(vi,Mi,T)
        if Ei < Eth:
            zeta = 1.0
        else:
            zeta = Eth / Ei
        # print(mb_factor,vi,E_imp)
        H[i] = mb_factor * vi * Ei * zeta
        
    # 4. Integrate with the trapezoid method
    H = sigma_dust * H
    H = trapezoid(H,v)
    
    return H

def test_rate(a_dust,s_dust,Zi,Zd,Md,Mi,Tmin,Tmax,nT=100,nv=300,delta_max=0.1):
    
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
    
    # 1. Setup the figure
    fig, ax = plt.subplots(1,1, figsize=(5,4),dpi=300,facecolor='w',edgecolor='k',sharey=True)
    ax.set_xlabel(r'$T$ [K]', fontsize=16)
    ax.set_ylabel(r'$H$ [erg cm$^3$ s$^{-1}$]',fontsize=16)
    ax.tick_params(labelsize=14)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.minorticks_on()
    ax.tick_params(which='both',axis="both",direction="in")
    
    # 2. Set the temperature and heating rate arrays
    Md = Md * au2cgs_m
    Mi = Mi * au2cgs_m
    Tgas = np.logspace(np.log10(Tmin),np.log10(Tmax),nT)
    
    num_cores = 5 #os.cpu_count()
    print(f"    Number of cores available: {num_cores}")
    args_list = [(Ti,a_dust,s_dust,Zi,Zd,Md,Mi,nv,delta_max) for Ti in Tgas]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(tqdm(executor.map(compute_rate_T, args_list), 
                            total=nT, 
                            desc=f'    Calculating heating rates',
                            unit=' steps'))

    Hrate = np.array(results)
    
    # 3. Plot the rate
    ax.plot(Tgas,Hrate,label='My full rate')
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(tqdm(executor.map(compute_rate_T_nocharge, args_list), 
                            total=nT, 
                            desc=f'    Calculating heating rates',
                            unit=' steps'))

    Hrate = np.array(results)
    ax.plot(Tgas,Hrate,label='My rate (no charge)')
    for i in range(0,nT):
        dist,x = dust_model.grain_charge_dist(1.0,Tgas[i],0.1,'carbonaceous','100A',gamma=None)
        D = dust_model.cmp_D_WD99(dist,x,Zi,Tgas[i],a_dust)
        Hrate[i] = D * Hrate[i]
    
    # 3. Plot the rate
    ax.plot(Tgas,Hrate,label='My rate (no electrostatic energy)')
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(tqdm(executor.map(compute_rate_Dwek_T, args_list), 
                            total=nT, 
                            desc=f'    Calculating heating rates',
                            unit=' steps'))

    Hrate_dwek = np.array(results)
    
    # 3. Plot the rate
    ax.plot(Tgas,Hrate_dwek,label='Dwek (1987) mine')
    
    if Mi == 1.00784:
        Eth = 133. * a_dust * 1e4 * 1e3 * eV2erg
    elif Mi == 4.002602:
        Eth = 222. * a_dust * 1e4 * 1e3 * eV2erg
    else:
        Eth = 665. * a_dust * 1e4 * 1e3 * eV2erg
    Hrate = np.sqrt(32./np.pi/Mi) * np.pi * a_dust**2. * (kb*Tgas)**(3./2.)
    Hrate = Hrate * (1.-(1.+Eth/(2.*kb*Tgas))*np.exp(-Eth/(kb*Tgas)))
    ax.plot(Tgas,Hrate,label='Dwek (1987)')
    
    ax.legend(loc='best',frameon=False,fontsize=12)
    fig.subplots_adjust(top=0.99,bottom=0.13,left=0.16,right=0.97,hspace=0,wspace=0)
    fig.savefig('testing_collisional_rate.png',format='png')
    
def compute_efficiency(args):
    
    T,a_dust,s_dust,ne_val,Zi,Zd,Md,Mi,nv,delta_max = args
    
    
    # 1. Determine the minimum velocity for convergence
    v_0 = np.sqrt(2.*kb*T/Mi) * (1. - (np.sqrt(3./2.) - 1.))
    while Maxwell_Boltzmann_function(v_0,Mi,T) > 1e-30:
        v_0 = v_0 / 2.
    x_0 = 0.5 * Mi * v_0**2. / (kb*T)
    
    # 2. We set the maximum velocity to the thermal energy of gas at ~1e10 K
    v_max = np.sqrt(2. * kb * 1e10 / Mi) # [cm/s]
    while Maxwell_Boltzmann_function(v_max,Mi,T) < 1e-30:
        v_max = v_max/ 2.0
    
    x_max = 0.5 * Mi * v_max**2. / (kb*T)
    
    # 3. Perform loop over velocity range
    x = np.logspace(np.log10(x_0),np.log10(x_max),nv)
    h = np.zeros(nv)
    
    n_dust = s_dust / Md * Zd
    
    for i in range(0, nv):
        xi = x[i]
        Ei = xi * kb*T
        agg = n_dust,ne_val,a_dust,Zi,Zd,Md,Mi,Ei,delta_max
        E_imp,_,_,_ = compute_energy_deposited(agg)
        h[i] = xi**2. * (E_imp/Ei) * np.exp(-xi)
    
    # 4. Integrate with the trapezoid method
    h = 0.5 * trapezoid(h,x)
    
    return h

def compute_energy_deposited_electron(args):
    
    fit_params,a_dust,E,delta_max = args
    
    # 1. Prepare parameters
    if a_dust == None:
        a_dust = 1e10
        dr = 1e-7
    else:
        dr = a_dust / 10.
    r_pd = 0.
    n = 0
    E_now = E
    
    # 2. Compute the screening length
    Earray = [E_now]
    distance = [0.0]
    
    while E_now > 1e-3*E and r_pd < 4./3.*a_dust:
        # 1. Compute the electron stopping power based on the fitting curve
        S = stopping_fit(E_now,*fit_params) # [eV/Angstrom]
        S = S * 1e8 # [ev/cm]
                
        # 2. Figure out if the step needs to be changed
        if abs(S*dr)/E_now >= delta_max:
            dr = dr / 2.
        else:
            E_now = E_now - S * dr
            Earray.append(E_now)
            r_pd = r_pd + dr
            distance.append(r_pd)
            if r_pd + dr >= 4./3.*a_dust:
                dr = 4./3.*a_dust - r_pd
            elif r_pd + dr == 4./3.*a_dust:
                break
        n += 1
    # 3. Compute the deposited energy
    E_imp = E - E_now
    Earray = np.array(Earray)
    distance = np.array(distance)
    
    return E_imp,r_pd,Earray,distance

def compute_efficiency_electron(args):
    
    T,a_dust,fit_params,nv,delta_max = args
    
    me = 9.10938e-28 # [g]
    
    # 1. Determine the minimum velocity for convergence
    v_0 = np.sqrt(2.*kb*T/me) * (1. - (np.sqrt(3./2.) - 1.))
    while Maxwell_Boltzmann_function(v_0,me,T) > 1e-30:
        v_0 = v_0 / 2.
    x_0 = 0.5 * me * v_0**2. / (kb*T)
    
    # 2. We set the maximum velocity to the thermal energy of gas at ~1e10 K
    v_max = np.sqrt(2. * kb * 1e10 / me) # [cm/s]
    while Maxwell_Boltzmann_function(v_max,me,T) < 1e-30:
        v_max = v_max/ 2.0
    
    x_max = 0.5 * me * v_max**2. / (kb*T)
    
    # 3. Perform loop over velocity range
    x = np.logspace(np.log10(x_0),np.log10(x_max),nv)
    h = np.zeros(nv)
    
    for i in range(0, nv):
        xi = x[i]
        Ei = xi * kb*T
        agg = fit_params,a_dust,Ei/eV2erg,delta_max
        E_imp,_,_,_ = compute_energy_deposited_electron(agg)
        h[i] = xi**2. * (E_imp*eV2erg/Ei) * np.exp(-xi)
    
    # 4. Integrate with the trapezoid method
    h = 0.5 * trapezoid(h,x)
    
    return h

def Dwek87_eff(a,T):
    Eth = 133. * a * 1e3 * eV2erg
    h = 1. - (1.+Eth/(2.*kb*T))*np.exp(-Eth/(kb*T))
    return h
    

def test_efficiency(Zi,Mi,Tmin,Tmax,nT=100,nv=300,delta_max=0.1):
    
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
    
    # 1. Setup the figure
    fig, ax = plt.subplots(1,1, figsize=(5,4),dpi=300,facecolor='w',edgecolor='k',sharey=True)
    ax.set_xlabel(r'$T$ [K]', fontsize=16)
    ax.set_ylabel(r'$h(a,T)$',fontsize=16)
    ax.tick_params(labelsize=14)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.minorticks_on()
    ax.tick_params(which='both',axis="both",direction="in")
    
    # 2. Set the temperature and heating rate arrays
    Mi = Mi * au2cgs_m
    Tgas = np.logspace(np.log10(Tmin),np.log10(Tmax),nT)
    
    # 2. First compute it for carbonaceous material
    s_dust = dust_model.basic_s[2]
    a_dust = dust_model.basic_a0[2]*1e-4
    am_dust = 12.011 * au2cgs_m
    an_dust = 6.
    ne_val = 4
    n_dust = s_dust / am_dust * an_dust
    
    num_cores = 5 #os.cpu_count()
    print(f"    Number of cores available: {num_cores}")
    args_list = [(Ti,a_dust,s_dust,ne_val,Zi,an_dust,am_dust,Mi,nv,delta_max) for Ti in Tgas]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(tqdm(executor.map(compute_efficiency, args_list), 
                            total=nT, 
                            desc=f'    Calculating heating rates',
                            unit=' steps'))

    h_eff = np.array(results)
    
    # 3. Plot the efficiency
    ax.plot(Tgas,h_eff,color='steelblue',linewidth=2,label='smallC')
    ax.plot(Tgas,Dwek87_eff(dust_model.basic_a0[2],Tgas),color='steelblue',linewidth=2,linestyle=':')
    a_dust = dust_model.basic_a0[3]*1e-4
    args_list = [(Ti,a_dust,s_dust,ne_val,Zi,an_dust,am_dust,Mi,nv,delta_max) for Ti in Tgas]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(tqdm(executor.map(compute_efficiency, args_list), 
                            total=nT, 
                            desc=f'    Calculating heating rates',
                            unit=' steps'))

    h_eff = np.array(results)
    
    # 3. Plot the efficiency
    ax.plot(Tgas,h_eff,color='cornflowerblue',linewidth=2,label='largeC')
    ax.plot(Tgas,Dwek87_eff(dust_model.basic_a0[3],Tgas),color='cornflowerblue',linewidth=2,linestyle=':')

    
    # 3. Now for the silicate
    s_dust = dust_model.basic_s[5]
    a_dust = dust_model.basic_a0[5]*1e-4
    am_dust = (24.305 + 55.845 + 28.0855 + 4*15.999) / 7. * au2cgs_m
    an_dust = int((4*8 + 14 + 26 + 12) / 7)
    ne_val = (2+8+4+4*6) / 7
    n_dust = s_dust / am_dust * an_dust
    
    args_list = [(Ti,a_dust,s_dust,ne_val,Zi,an_dust,am_dust,Mi,nv,delta_max) for Ti in Tgas]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(tqdm(executor.map(compute_efficiency, args_list), 
                            total=nT, 
                            desc=f'    Calculating heating rates',
                            unit=' steps'))

    h_eff = np.array(results)
    
    # 3. Plot the efficiency
    ax.plot(Tgas,h_eff,color='saddlebrown',linewidth=2,label='smallSil')
    ax.plot(Tgas,Dwek87_eff(dust_model.basic_a0[5],Tgas),color='saddlebrown',linewidth=2,linestyle=':')

    
    a_dust = dust_model.basic_a0[6]*1e-4
    args_list = [(Ti,a_dust,s_dust,ne_val,Zi,an_dust,am_dust,Mi,nv,delta_max) for Ti in Tgas]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(tqdm(executor.map(compute_efficiency, args_list), 
                            total=nT, 
                            desc=f'    Calculating heating rates',
                            unit=' steps'))

    h_eff = np.array(results)
    
    # 3. Plot the efficiency
    ax.plot(Tgas,h_eff,color='sandybrown',linewidth=2,label='largeSil')
    ax.plot(Tgas,Dwek87_eff(dust_model.basic_a0[6],Tgas),color='sandybrown',linewidth=2,linestyle=':')

    
    ax.legend(loc='best',frameon=False,fontsize=12)
    fig.subplots_adjust(top=0.99,bottom=0.13,left=0.16,right=0.97,hspace=0,wspace=0)
    fig.savefig('testing_efficiency.png',format='png')
    
def stopping_fit(E,a,b,c,d,e,f,g,h):
    E = E/1e3
    S1 = h * np.log(1.+a*E)
    S2 = f * E**g + b * E**d + c * E**e

    return S1/S2
    
    
def fit_e_stopping_silicate():
    
    from unyt import g,cm,eV
    from scipy.optimize import curve_fit
    
    # Data from table 1 in Ashley and Anderson (1981)
    E = np.array([15, 20, 30, 40, 60, 80, 100, 150, 200, 300, 400, 600, 800, 1000, 2000, 4000, 6000, 8000, 10000])
    S_prime = np.array([2.40, 6.64, 26.2, 55.1, 105, 128, 137, 141, 137, 127, 117, 99.6, 87.5, 78.1, 52.8, 33.5, 25.1, 20.4, 17.4])

    data_carbon = pd.read_csv('Carbon_electron_stopping_power_Joy1995.csv',header=None,names=['E','S'])
    
    # Convert S_prime [MeV cm^2 /g] to S in [eV/A]
    rho_Si02 = 2.65 * g /cm**3
    S_prime = S_prime * 1e6 * eV *cm**2/g
    S = S_prime * rho_Si02
    
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
    
    # 1. Setup the figure
    fig, ax = plt.subplots(1,1, figsize=(5,4),dpi=300,facecolor='w',edgecolor='k',sharey=True)
    ax.set_xlabel(r'$E$ [eV]', fontsize=16)
    ax.set_ylabel(r'$S(E)$ [eV / \AA]',fontsize=16)
    ax.tick_params(labelsize=14)
    ax.set_xscale('log')
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.minorticks_on()
    ax.tick_params(which='both',axis="both",direction="in")
    full_E = np.logspace(0,5,100)
    ax.plot(E,S.to('eV/Angstrom'),marker='o',linestyle='',label=r'Ashley et al. (1981)',alpha=0.6,color='goldenrod')
    ax.plot(data_carbon['E']*1e3,data_carbon['S'],marker='o',linestyle='',label=r'Joy (1995)',alpha=0.6,color='cornflowerblue')
    
    popt,pcov = curve_fit(stopping_fit,E,S.to('eV/Angstrom').d)
    print('Silicate: ',popt)
    ax.plot(full_E,stopping_fit(full_E,*popt),linestyle='-.',color='saddlebrown',label=r'$S^{\rm Sil}(E)$',linewidth=2)
    
    popt,pcov = curve_fit(stopping_fit,data_carbon['E']*1e3,data_carbon['S'],
                          p0=[-0.000423375,-3.57429e-11,-3.37861e-7,-3.18688,
                              -0.587928,-0.000232675,1.53851,1.41476])
    print('Carbon: ',popt)
    
    ax.plot(full_E,stopping_fit(full_E,*popt),linestyle='--',color='royalblue',label=r'$S^{\rm C}(E)$',linewidth=2)
    
    ax.legend(loc='best',frameon=False,fontsize=12)
    fig.subplots_adjust(top=0.98,bottom=0.13,left=0.1,right=0.99,hspace=0,wspace=0)
    fig.savefig('fit_stopping_e_silicate.pdf',format='pdf')
    
def DwekWerner81_cooling(a,T):
    """Cooling rate per uncharged grain for electron collisions
    as given in Appendix A of Dwek and Werner (1981).

    Args:
        a (np.float): grain radius [micron]
        T (np.float): gas temperature [K]

    Returns:
        np.float: cooling rate [erg cm^3 / s]
    """    
    
    x = 2.71e8 * a**(2./3.) / T
    xmax = 14000 * a**(2./3.)
    H = np.zeros(len(T))
    for i in range(0, len(T)):
        if x[i] >= xmax:
            H[i] = 0.0
        elif x[i] >= 4.5:
            H[i] = 5.38e-18 * a**2. * T[i]**1.5
        elif x[i] >= 1.5:
            H[i] = 3.37e-13 * a**2.41 * T[i]**0.88
        else:
            H[i] = 6.48e-6 * a**3.
        
    return H

def plot_collisional_cooling(Tmin,Tmax,Td,nT=100,nv=300,delta_max=0.1):
    from unyt import g,cm,eV
    from scipy.optimize import curve_fit
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
    me = 9.10938e-28 # [g]
    # 1. Setup the figure
    fig, ax = plt.subplots(1,1, figsize=(7,5),dpi=300,facecolor='w',edgecolor='k',sharey=True)
    ax.set_xlabel(r'$T$ [K]', fontsize=16)
    ax.set_ylabel(r'$\tilde{H}(a,T)$ [erg cm$^3$/s]',fontsize=16)
    ax.tick_params(labelsize=14)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylim([1e-22,1e-7])
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.minorticks_on()
    ax.tick_params(which='both',axis="both",direction="in")
    
    Tgas = np.logspace(np.log10(Tmin),np.log10(Tmax),nT)
    
    # 2. Plot the fitting function by Dwek and Werner (1981)
    ax.plot(Tgas,DwekWerner81_cooling(0.005,Tgas),linestyle=':',color='k',label=r'DW81 $a=0.005$ $\mu$m')
    ax.plot(Tgas,DwekWerner81_cooling(0.01,Tgas),linestyle='--',color='k',label=r'DW81 $a=0.01$ $\mu$m')
    ax.plot(Tgas,DwekWerner81_cooling(0.1,Tgas),linestyle='-',color='k',label=r'DW81 $a=0.1$ $\mu$m')
    
    # 3. Compute the fitting to the electron stopping power
    # Data from table 1 in Ashley and Anderson (1981)
    E = np.array([15, 20, 30, 40, 60, 80, 100, 150, 200, 300, 400, 600, 800, 1000, 2000, 4000, 6000, 8000, 10000])
    S_prime = np.array([2.40, 6.64, 26.2, 55.1, 105, 128, 137, 141, 137, 127, 117, 99.6, 87.5, 78.1, 52.8, 33.5, 25.1, 20.4, 17.4])

    data_carbon = pd.read_csv('Carbon_electron_stopping_power_Joy1995.csv',header=None,names=['E','S'])
    
    # Convert S_prime [MeV cm^2 /g] to S in [eV/A]
    rho_Si02 = 2.65 * g /cm**3
    S_prime = S_prime * 1e6 * eV *cm**2/g
    S = S_prime * rho_Si02
    
    popt_sil,pcov = curve_fit(stopping_fit,E,S.to('eV/Angstrom').d)
    print('Silicate: ',popt_sil)
    
    popt_car,pcov = curve_fit(stopping_fit,data_carbon['E']*1e3,data_carbon['S'],
                          p0=[-0.000423375,-3.57429e-11,-3.37861e-7,-3.18688,
                              -0.587928,-0.000232675,1.53851,1.41476])
    print('Carbon: ',popt_car)
    
    num_cores = 5 #os.cpu_count()
    print(f"    Number of cores available: {num_cores}")
    
    # 4. Electron cooling for carbonaceous material
    a_dust = dust_model.basic_a0[2]*1e-4
    args_list = [(Ti,a_dust,popt_car,nv,delta_max) for Ti in Tgas]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(tqdm(executor.map(compute_efficiency_electron, args_list), 
                            total=nT, 
                            desc=f'    Calculating electron cooling rate for smallC',
                            unit=' steps'))

    h_electron = np.array(results)
    H_electron = np.sqrt(32./(np.pi*me)) * np.pi * a_dust**2. * np.sqrt(kb*Tgas)*kb*(Tgas-Td) * h_electron
    ax.plot(Tgas,H_electron,linestyle='-',color='steelblue',label=r'e$^{-}$ (smallC)')
    
    a_dust = dust_model.basic_a0[3]*1e-4
    args_list = [(Ti,a_dust,popt_car,nv,delta_max) for Ti in Tgas]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(tqdm(executor.map(compute_efficiency_electron, args_list), 
                            total=nT, 
                            desc=f'    Calculating electron cooling rate for largeC',
                            unit=' steps'))

    h_electron = np.array(results)
    H_electron = np.sqrt(32./(np.pi*me)) * np.pi * a_dust**2. * np.sqrt(kb*Tgas)*kb*(Tgas-Td) * h_electron
    ax.plot(Tgas,H_electron,linestyle='--',color='cornflowerblue',label=r'e$^{-}$ (largeC)')
    
    # 5. Electron cooling for silicate material
    a_dust = dust_model.basic_a0[5]*1e-4
    args_list = [(Ti,a_dust,popt_sil,nv,delta_max) for Ti in Tgas]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(tqdm(executor.map(compute_efficiency_electron, args_list), 
                            total=nT, 
                            desc=f'    Calculating electron cooling rate for smallSil',
                            unit=' steps'))

    h_electron = np.array(results)
    H_electron = np.sqrt(32./(np.pi*me)) * np.pi * a_dust**2. * np.sqrt(kb*Tgas)*kb*(Tgas-Td) * h_electron
    ax.plot(Tgas,H_electron,linestyle='-',color='saddlebrown',label=r'e$^{-}$ (smallSil)')
    
    a_dust = dust_model.basic_a0[6]*1e-4
    args_list = [(Ti,a_dust,popt_sil,nv,delta_max) for Ti in Tgas]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(tqdm(executor.map(compute_efficiency_electron, args_list), 
                            total=nT, 
                            desc=f'    Calculating electron cooling rate for largeSil',
                            unit=' steps'))

    h_electron = np.array(results)
    H_electron = np.sqrt(32./(np.pi*me)) * np.pi * a_dust**2. * np.sqrt(kb*Tgas)*kb*(Tgas-Td) * h_electron
    ax.plot(Tgas,H_electron,linestyle='--',color='sandybrown',label=r'e$^{-}$ (largeSil)')
    
    # 6. Hydrogen cooling for carbonaceous material
    s_dust = dust_model.basic_s[2]
    a_dust = dust_model.basic_a0[2]*1e-4
    am_dust = 12.011 * au2cgs_m
    an_dust = 6.
    ne_val = 4
    Mi = 1.00784 * au2cgs_m
    Zi = 1

    args_list = [(Ti,a_dust,s_dust,ne_val,Zi,an_dust,am_dust,Mi,nv,delta_max) for Ti in Tgas]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(tqdm(executor.map(compute_efficiency, args_list), 
                            total=nT, 
                            desc=f'    Calculating Hydrogen heating rates',
                            unit=' steps'))

    h_eff = np.array(results)
    H_hydrogen = np.sqrt(32./(np.pi*Mi)) * np.pi * a_dust**2. * np.sqrt(kb*Tgas)*kb*(Tgas-Td) * h_eff
    ax.plot(Tgas,H_hydrogen,linestyle='-.',color='steelblue',label=r'H (smallC)')
    
    s_dust = dust_model.basic_s[3]
    a_dust = dust_model.basic_a0[3]*1e-4
    am_dust = 12.011 * au2cgs_m
    an_dust = 6.
    ne_val = 4
    Mi = 1.00784 * au2cgs_m
    Zi = 1

    args_list = [(Ti,a_dust,s_dust,ne_val,Zi,an_dust,am_dust,Mi,nv,delta_max) for Ti in Tgas]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(tqdm(executor.map(compute_efficiency, args_list), 
                            total=nT, 
                            desc=f'    Calculating Hydrogen heating rates',
                            unit=' steps'))

    h_eff = np.array(results)
    H_hydrogen = np.sqrt(32./(np.pi*Mi)) * np.pi * a_dust**2. * np.sqrt(kb*Tgas)*kb*(Tgas-Td) * h_eff
    ax.plot(Tgas,H_hydrogen,linestyle='-.',color='cornflowerblue',label=r'H (largeC)')
    
    # 7. Hydrogen cooling for silicate material
    s_dust = dust_model.basic_s[5]
    a_dust = dust_model.basic_a0[5]*1e-4
    am_dust = (24.305 + 55.845 + 28.0855 + 4*15.999) / 7. * au2cgs_m
    an_dust = int((4*8 + 14 + 26 + 12) / 7)
    ne_val = (2+8+4+4*6) / 7
    Mi = 1.00784 * au2cgs_m
    Zi = 1

    args_list = [(Ti,a_dust,s_dust,ne_val,Zi,an_dust,am_dust,Mi,nv,delta_max) for Ti in Tgas]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(tqdm(executor.map(compute_efficiency, args_list), 
                            total=nT, 
                            desc=f'    Calculating Hydrogen heating rates',
                            unit=' steps'))

    h_eff = np.array(results)
    H_hydrogen = np.sqrt(32./(np.pi*Mi)) * np.pi * a_dust**2. * np.sqrt(kb*Tgas)*kb*(Tgas-Td) * h_eff
    ax.plot(Tgas,H_hydrogen,linestyle='-.',color='saddlebrown',label=r'H (smallSil)')
    
    a_dust = dust_model.basic_a0[3]*1e-4
    args_list = [(Ti,a_dust,s_dust,ne_val,Zi,an_dust,am_dust,Mi,nv,delta_max) for Ti in Tgas]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(tqdm(executor.map(compute_efficiency, args_list), 
                            total=nT, 
                            desc=f'    Calculating Hydrogen heating rates',
                            unit=' steps'))

    h_eff = np.array(results)
    H_hydrogen = np.sqrt(32./(np.pi*Mi)) * np.pi * a_dust**2. * np.sqrt(kb*Tgas)*kb*(Tgas-Td) * h_eff
    ax.plot(Tgas,H_hydrogen,linestyle='-.',color='sandybrown',label=r'H (largeSil)')
    
    # 8. Helium cooling for carbonaceous material
    s_dust = dust_model.basic_s[2]
    a_dust = dust_model.basic_a0[2]*1e-4
    am_dust = 12.011 * au2cgs_m
    an_dust = 6.
    ne_val = 4
    Mi = 4.002602 * au2cgs_m
    Zi = 2

    args_list = [(Ti,a_dust,s_dust,ne_val,Zi,an_dust,am_dust,Mi,nv,delta_max) for Ti in Tgas]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(tqdm(executor.map(compute_efficiency, args_list), 
                            total=nT, 
                            desc=f'    Calculating Helium heating rates',
                            unit=' steps'))

    h_eff = np.array(results)
    H_hydrogen = np.sqrt(32./(np.pi*Mi)) * np.pi * a_dust**2. * np.sqrt(kb*Tgas)*kb*(Tgas-Td) * h_eff
    ax.plot(Tgas,H_hydrogen,linestyle=':',color='steelblue',label=r'He (smallC)')
    
    a_dust = dust_model.basic_a0[3]*1e-4

    args_list = [(Ti,a_dust,s_dust,ne_val,Zi,an_dust,am_dust,Mi,nv,delta_max) for Ti in Tgas]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(tqdm(executor.map(compute_efficiency, args_list), 
                            total=nT, 
                            desc=f'    Calculating Helium heating rates',
                            unit=' steps'))

    h_eff = np.array(results)
    H_hydrogen = np.sqrt(32./(np.pi*Mi)) * np.pi * a_dust**2. * np.sqrt(kb*Tgas)*kb*(Tgas-Td) * h_eff
    ax.plot(Tgas,H_hydrogen,linestyle=':',color='cornflowerblue',label=r'He (largeC)')
    
    # 9. Helium cooling for silicate material
    s_dust = dust_model.basic_s[5]
    a_dust = dust_model.basic_a0[5]*1e-4
    am_dust = (24.305 + 55.845 + 28.0855 + 4*15.999) / 7. * au2cgs_m
    an_dust = int((4*8 + 14 + 26 + 12) / 7)
    ne_val = (2+8+4+4*6) / 7
    Mi = 4.002602 * au2cgs_m
    Zi = 2

    args_list = [(Ti,a_dust,s_dust,ne_val,Zi,an_dust,am_dust,Mi,nv,delta_max) for Ti in Tgas]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(tqdm(executor.map(compute_efficiency, args_list), 
                            total=nT, 
                            desc=f'    Calculating Helium heating rates',
                            unit=' steps'))

    h_eff = np.array(results)
    H_hydrogen = np.sqrt(32./(np.pi*Mi)) * np.pi * a_dust**2. * np.sqrt(kb*Tgas)*kb*(Tgas-Td) * h_eff
    ax.plot(Tgas,H_hydrogen,linestyle=':',color='saddlebrown',label=r'He (smallSil)')
    
    a_dust = dust_model.basic_a0[3]*1e-4
    args_list = [(Ti,a_dust,s_dust,ne_val,Zi,an_dust,am_dust,Mi,nv,delta_max) for Ti in Tgas]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(tqdm(executor.map(compute_efficiency, args_list), 
                            total=nT, 
                            desc=f'    Calculating Helium heating rates',
                            unit=' steps'))

    h_eff = np.array(results)
    H_hydrogen = np.sqrt(32./(np.pi*Mi)) * np.pi * a_dust**2. * np.sqrt(kb*Tgas)*kb*(Tgas-Td) * h_eff
    ax.plot(Tgas,H_hydrogen,linestyle=':',color='sandybrown',label=r'He (largeSil)')
    
    # 10. Carbon cooling for carbonaceous material
    s_dust = dust_model.basic_s[2]
    a_dust = dust_model.basic_a0[2]*1e-4
    am_dust = 12.011 * au2cgs_m
    an_dust = 6.
    ne_val = 4
    Mi = 12.011 * au2cgs_m
    Zi = 6

    args_list = [(Ti,a_dust,s_dust,ne_val,Zi,an_dust,am_dust,Mi,nv,delta_max) for Ti in Tgas]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(tqdm(executor.map(compute_efficiency, args_list), 
                            total=nT, 
                            desc=f'    Calculating Carbon heating rates',
                            unit=' steps'))

    h_eff = np.array(results)
    H_hydrogen = np.sqrt(32./(np.pi*Mi)) * np.pi * a_dust**2. * np.sqrt(kb*Tgas)*kb*(Tgas-Td) * h_eff
    ax.plot(Tgas,H_hydrogen,linestyle=(0, (3, 10, 1, 10)),color='steelblue',label=r'C (smallC)')
    
    a_dust = dust_model.basic_a0[3]*1e-4

    args_list = [(Ti,a_dust,s_dust,ne_val,Zi,an_dust,am_dust,Mi,nv,delta_max) for Ti in Tgas]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(tqdm(executor.map(compute_efficiency, args_list), 
                            total=nT, 
                            desc=f'    Calculating Carbon heating rates',
                            unit=' steps'))

    h_eff = np.array(results)
    H_hydrogen = np.sqrt(32./(np.pi*Mi)) * np.pi * a_dust**2. * np.sqrt(kb*Tgas)*kb*(Tgas-Td) * h_eff
    ax.plot(Tgas,H_hydrogen,linestyle=(0, (3, 10, 1, 10)),color='cornflowerblue',label=r'C (largeC)')
    
    # 11. Carbon cooling for silicate material
    s_dust = dust_model.basic_s[5]
    a_dust = dust_model.basic_a0[5]*1e-4
    am_dust = (24.305 + 55.845 + 28.0855 + 4*15.999) / 7. * au2cgs_m
    an_dust = int((4*8 + 14 + 26 + 12) / 7)
    ne_val = (2+8+4+4*6) / 7
    Mi = 12.011 * au2cgs_m
    Zi = 6

    args_list = [(Ti,a_dust,s_dust,ne_val,Zi,an_dust,am_dust,Mi,nv,delta_max) for Ti in Tgas]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(tqdm(executor.map(compute_efficiency, args_list), 
                            total=nT, 
                            desc=f'    Calculating Carbon heating rates',
                            unit=' steps'))

    h_eff = np.array(results)
    H_hydrogen = np.sqrt(32./(np.pi*Mi)) * np.pi * a_dust**2. * np.sqrt(kb*Tgas)*kb*(Tgas-Td) * h_eff
    ax.plot(Tgas,H_hydrogen,linestyle=(0, (3, 10, 1, 10)),color='saddlebrown',label=r'C (smallSil)')
    
    a_dust = dust_model.basic_a0[3]*1e-4
    args_list = [(Ti,a_dust,s_dust,ne_val,Zi,an_dust,am_dust,Mi,nv,delta_max) for Ti in Tgas]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(tqdm(executor.map(compute_efficiency, args_list), 
                            total=nT, 
                            desc=f'    Calculating Carbon heating rates',
                            unit=' steps'))

    h_eff = np.array(results)
    H_hydrogen = np.sqrt(32./(np.pi*Mi)) * np.pi * a_dust**2. * np.sqrt(kb*Tgas)*kb*(Tgas-Td) * h_eff
    ax.plot(Tgas,H_hydrogen,linestyle=(0, (3, 10, 1, 10)),color='sandybrown',label=r'C (largeSil)')
    
    # 12. Add the HM79 low temperature cooling for carbonaceous grains at Td=2.73K, for a primordial gas
    supp_factor = 1. - 1./(1.+np.exp(-10.*(np.log10(Tgas)-4.)))
    ax.plot(Tgas, supp_factor*low_temp_cooling(Tgas,dust_model.basic_a0[2]*1e-4,Td,12.011,'H'),linestyle=(0, (3, 1, 1, 1, 1, 1)),color='steelblue',label=r'HM79 ($a=0.01$ $\mu$m)')
    ax.plot(Tgas, supp_factor*low_temp_cooling(Tgas,dust_model.basic_a0[3]*1e-4,Td,12.011,'H'),linestyle=(0, (3, 1, 1, 1, 1, 1)),color='cornflowerblue',label=r'HM79 ($a=0.1$ $\mu$m)')

    # 13. Add the HM79 low temperature cooling for silicate grains at Td=2.73K, for a primordial gas
    am_dust = (24.305 + 55.845 + 28.0855 + 4*15.999) / 7.
    ax.plot(Tgas, supp_factor*low_temp_cooling(Tgas,dust_model.basic_a0[5]*1e-4,Td,am_dust,'H'),linestyle=(0, (3, 1, 1, 1, 1, 1)),color='saddlebrown',label=r'HM79 ($a=0.01$ $\mu$m)')
    ax.plot(Tgas, supp_factor*low_temp_cooling(Tgas,dust_model.basic_a0[3]*1e-4,Td,am_dust,'H'),linestyle=(0, (3, 1, 1, 1, 1, 1)),color='sandybrown',label=r'HM79 ($a=0.1$ $\mu$m)')
    
    
    ax.legend(loc='best',frameon=False,fontsize=12,ncol=2)
    fig.subplots_adjust(top=0.99,bottom=0.1,left=0.12,right=0.99,hspace=0,wspace=0)
    fig.savefig('hightemp_collisional_cooling.pdf',format='pdf')
    
def export_collisional_cooling(Tmin,Tmax,nT=100,nv=300,delta_max=0.1):
    from unyt import g,cm,eV
    from scipy.optimize import curve_fit
    
    # 1. Setup the variables and arrays
    me = 9.10938e-28 # [g]
    Tgas = np.logspace(np.log10(Tmin),np.log10(Tmax),nT)
    
    # 2. Compute the fitting to the electron stopping power
    # Data from table 1 in Ashley and Anderson (1981)
    E = np.array([15, 20, 30, 40, 60, 80, 100, 150, 200, 300, 400, 600, 800, 1000, 2000, 4000, 6000, 8000, 10000])
    S_prime = np.array([2.40, 6.64, 26.2, 55.1, 105, 128, 137, 141, 137, 127, 117, 99.6, 87.5, 78.1, 52.8, 33.5, 25.1, 20.4, 17.4])
    
    data_carbon = pd.read_csv('Carbon_electron_stopping_power_Joy1995.csv',header=None,names=['E','S'])
    
    # Convert S_prime [MeV cm^2 /g] to S in [eV/A]
    rho_Si02 = 2.65 * g /cm**3
    S_prime = S_prime * 1e6 * eV *cm**2/g
    S = S_prime * rho_Si02
    
    popt_sil,pcov = curve_fit(stopping_fit,E,S.to('eV/Angstrom').d)
    print('Silicate: ',popt_sil)
    
    popt_car,pcov = curve_fit(stopping_fit,data_carbon['E']*1e3,data_carbon['S'],
                          p0=[-0.000423375,-3.57429e-11,-3.37861e-7,-3.18688,
                              -0.587928,-0.000232675,1.53851,1.41476])
    print('Carbon: ',popt_car)
    
    num_cores = 10 #os.cpu_count()
    print(f"    Number of cores available: {num_cores}")
    
    # 3. Crete the directory for the table data
    table_dir = './collisional_cooling_data'
    if not os.path.exists(table_dir):
        os.mkdir(table_dir)
    
    # 4. Electron cooling for carbonaceous material
    a_dust = dust_model.basic_a0[2]*1e-4
    args_list = [(Ti,a_dust,popt_car,nv,delta_max) for Ti in Tgas]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(tqdm(executor.map(compute_efficiency_electron, args_list), 
                            total=nT, 
                            desc=f'    Calculating electron cooling rate for smallC',
                            unit=' steps'))

    h_electron = np.array(results)
    H_electron = np.sqrt(32./(np.pi*me)) * np.pi * a_dust**2. * np.sqrt(kb*Tgas) * kb * h_electron
    
    # Write to txt file the results of (Tgas, H_electron) in a format suitable for Fortran90
    with open(f'{table_dir}/electron_cooling_{dust_model.basic_a0[2]:.4f}_micron_Gra', 'w') as f:
        f.write(f'{nT}\n')
        for T, H in zip(np.log10(Tgas), np.log10(H_electron)):
            f.write(f'{T:14.6e} {H:14.6e}\n')
    
    
    a_dust = dust_model.basic_a0[3]*1e-4
    args_list = [(Ti,a_dust,popt_car,nv,delta_max) for Ti in Tgas]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(tqdm(executor.map(compute_efficiency_electron, args_list), 
                            total=nT, 
                            desc=f'    Calculating electron cooling rate for largeC',
                            unit=' steps'))

    h_electron = np.array(results)
    H_electron = np.sqrt(32./(np.pi*me)) * np.pi * a_dust**2. * np.sqrt(kb*Tgas) * kb * h_electron
    
    # Write to txt file the results of (Tgas, H_electron) in a format suitable for Fortran90
    with open(f'{table_dir}/electron_cooling_{dust_model.basic_a0[3]:.4f}_micron_Gra', 'w') as f:
        f.write(f'{nT}\n')
        for T, H in zip(np.log10(Tgas), np.log10(H_electron)):
            f.write(f'{T:14.6e} {H:14.6e}\n')
    
    # 5. Electron cooling for silicate material
    a_dust = dust_model.basic_a0[5]*1e-4
    args_list = [(Ti,a_dust,popt_sil,nv,delta_max) for Ti in Tgas]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(tqdm(executor.map(compute_efficiency_electron, args_list), 
                            total=nT, 
                            desc=f'    Calculating electron cooling rate for smallSil',
                            unit=' steps'))

    h_electron = np.array(results)
    H_electron = np.sqrt(32./(np.pi*me)) * np.pi * a_dust**2. * np.sqrt(kb*Tgas) * kb * h_electron
    
    # Write to txt file the results of (Tgas, H_electron) in a format suitable for Fortran90
    with open(f'{table_dir}/electron_cooling_{dust_model.basic_a0[5]:.4f}_micron_Sil', 'w') as f:
        f.write(f'{nT}\n')
        for T, H in zip(np.log10(Tgas), np.log10(H_electron)):
            f.write(f'{T:14.6e} {H:14.6e}\n')
    
    a_dust = dust_model.basic_a0[6]*1e-4
    args_list = [(Ti,a_dust,popt_sil,nv,delta_max) for Ti in Tgas]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(tqdm(executor.map(compute_efficiency_electron, args_list), 
                            total=nT, 
                            desc=f'    Calculating electron cooling rate for largeSil',
                            unit=' steps'))

    h_electron = np.array(results)
    H_electron = np.sqrt(32./(np.pi*me)) * np.pi * a_dust**2. * np.sqrt(kb*Tgas) * kb * h_electron
    
    # Write to txt file the results of (Tgas, H_electron) in a format suitable for Fortran90
    with open(f'{table_dir}/electron_cooling_{dust_model.basic_a0[6]:.4f}_micron_Sil', 'w') as f:
        f.write(f'{nT}\n')
        for T, H in zip(np.log10(Tgas), np.log10(H_electron)):
            f.write(f'{T:14.6e} {H:14.6e}\n')
    
    # 6. Hydrogen cooling for carbonaceous material
    s_dust = dust_model.basic_s[2]
    a_dust = dust_model.basic_a0[2]*1e-4
    am_dust = 12.011 * au2cgs_m
    an_dust = 6.
    ne_val = 4
    Mi = 1.00784 * au2cgs_m
    Zi = 1

    args_list = [(Ti,a_dust,s_dust,ne_val,Zi,an_dust,am_dust,Mi,nv,delta_max) for Ti in Tgas]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(tqdm(executor.map(compute_efficiency, args_list), 
                            total=nT, 
                            desc=f'    Calculating Hydrogen heating rates for smallC',
                            unit=' steps'))

    h_eff = np.array(results)
    H_hydrogen = np.sqrt(32./(np.pi*Mi)) * np.pi * a_dust**2. * np.sqrt(kb*Tgas) * kb * h_eff
    
    # Write to txt file the results of (Tgas, H_hydrogen) in a format suitable for Fortran90
    with open(f'{table_dir}/H_cooling_{dust_model.basic_a0[2]:.4f}_micron_Gra', 'w') as f:
        f.write(f'{nT}\n')
        for T, H in zip(np.log10(Tgas), np.log10(H_hydrogen)):
            f.write(f'{T:14.6e} {H:14.6e}\n')
    
    s_dust = dust_model.basic_s[3]
    a_dust = dust_model.basic_a0[3]*1e-4
    am_dust = 12.011 * au2cgs_m
    an_dust = 6.
    ne_val = 4
    Mi = 1.00784 * au2cgs_m
    Zi = 1

    args_list = [(Ti,a_dust,s_dust,ne_val,Zi,an_dust,am_dust,Mi,nv,delta_max) for Ti in Tgas]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(tqdm(executor.map(compute_efficiency, args_list), 
                            total=nT, 
                            desc=f'    Calculating Hydrogen heating rates largeC',
                            unit=' steps'))

    h_eff = np.array(results)
    H_hydrogen = np.sqrt(32./(np.pi*Mi)) * np.pi * a_dust**2. * np.sqrt(kb*Tgas) * kb * h_eff
    
    # Write to txt file the results of (Tgas, H_hydrogen) in a format suitable for Fortran90
    with open(f'{table_dir}/H_cooling_{dust_model.basic_a0[3]:.4f}_micron_Gra', 'w') as f:
        f.write(f'{nT}\n')
        for T, H in zip(np.log10(Tgas), np.log10(H_hydrogen)):
            f.write(f'{T:14.6e} {H:14.6e}\n')
    
    # 7. Hydrogen cooling for silicate material
    s_dust = dust_model.basic_s[5]
    a_dust = dust_model.basic_a0[5]*1e-4
    am_dust = (24.305 + 55.845 + 28.0855 + 4*15.999) / 7. * au2cgs_m
    an_dust = int((4*8 + 14 + 26 + 12) / 7)
    ne_val = (2+8+4+4*6) / 7
    Mi = 1.00784 * au2cgs_m
    Zi = 1

    args_list = [(Ti,a_dust,s_dust,ne_val,Zi,an_dust,am_dust,Mi,nv,delta_max) for Ti in Tgas]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(tqdm(executor.map(compute_efficiency, args_list), 
                            total=nT, 
                            desc=f'    Calculating Hydrogen heating rates for smallSil',
                            unit=' steps'))

    h_eff = np.array(results)
    H_hydrogen = np.sqrt(32./(np.pi*Mi)) * np.pi * a_dust**2. * np.sqrt(kb*Tgas) * kb * h_eff
    
    # Write to txt file the results of (Tgas, H_hydrogen) in a format suitable for Fortran90
    with open(f'{table_dir}/H_cooling_{dust_model.basic_a0[5]:.4f}_micron_Sil', 'w') as f:
        f.write(f'{nT}\n')
        for T, H in zip(np.log10(Tgas), np.log10(H_hydrogen)):
            f.write(f'{T:14.6e} {H:14.6e}\n')
    
    a_dust = dust_model.basic_a0[3]*1e-4
    args_list = [(Ti,a_dust,s_dust,ne_val,Zi,an_dust,am_dust,Mi,nv,delta_max) for Ti in Tgas]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(tqdm(executor.map(compute_efficiency, args_list), 
                            total=nT, 
                            desc=f'    Calculating Hydrogen heating rates for largeSil',
                            unit=' steps'))

    h_eff = np.array(results)
    H_hydrogen = np.sqrt(32./(np.pi*Mi)) * np.pi * a_dust**2. * np.sqrt(kb*Tgas) * kb * h_eff
    
    # Write to txt file the results of (Tgas, H_hydrogen) in a format suitable for Fortran90
    with open(f'{table_dir}/H_cooling_{dust_model.basic_a0[6]:.4f}_micron_Sil', 'w') as f:
        f.write(f'{nT}\n')
        for T, H in zip(np.log10(Tgas), np.log10(H_hydrogen)):
            f.write(f'{T:14.6e} {H:14.6e}\n')
    
    # 8. Helium cooling for carbonaceous material
    s_dust = dust_model.basic_s[2]
    a_dust = dust_model.basic_a0[2]*1e-4
    am_dust = 12.011 * au2cgs_m
    an_dust = 6.
    ne_val = 4
    Mi = 4.002602 * au2cgs_m
    Zi = 2

    args_list = [(Ti,a_dust,s_dust,ne_val,Zi,an_dust,am_dust,Mi,nv,delta_max) for Ti in Tgas]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(tqdm(executor.map(compute_efficiency, args_list), 
                            total=nT, 
                            desc=f'    Calculating Helium heating rates for smallC',
                            unit=' steps'))

    h_eff = np.array(results)
    H_helium = np.sqrt(32./(np.pi*Mi)) * np.pi * a_dust**2. * np.sqrt(kb*Tgas) * kb * h_eff

    # Write to txt file the results of (Tgas, H_helium) in a format suitable for Fortran90
    with open(f'{table_dir}/He_cooling_{dust_model.basic_a0[2]:.4f}_micron_Gra', 'w') as f:
        f.write(f'{nT}\n')
        for T, H in zip(np.log10(Tgas), np.log10(H_helium)):
            f.write(f'{T:14.6e} {H:14.6e}\n')
    
    a_dust = dust_model.basic_a0[3]*1e-4

    args_list = [(Ti,a_dust,s_dust,ne_val,Zi,an_dust,am_dust,Mi,nv,delta_max) for Ti in Tgas]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(tqdm(executor.map(compute_efficiency, args_list), 
                            total=nT, 
                            desc=f'    Calculating Helium heating rates for largeC',
                            unit=' steps'))

    h_eff = np.array(results)
    H_helium = np.sqrt(32./(np.pi*Mi)) * np.pi * a_dust**2. * np.sqrt(kb*Tgas) * kb * h_eff

    # Write to txt file the results of (Tgas, H_helium) in a format suitable for Fortran90
    with open(f'{table_dir}/He_cooling_{dust_model.basic_a0[3]:.4f}_micron_Gra', 'w') as f:
        f.write(f'{nT}\n')
        for T, H in zip(np.log10(Tgas), np.log10(H_helium)):
            f.write(f'{T:14.6e} {H:14.6e}\n')
    
    # 9. Helium cooling for silicate material
    s_dust = dust_model.basic_s[5]
    a_dust = dust_model.basic_a0[5]*1e-4
    am_dust = (24.305 + 55.845 + 28.0855 + 4*15.999) / 7. * au2cgs_m
    an_dust = int((4*8 + 14 + 26 + 12) / 7)
    ne_val = (2+8+4+4*6) / 7
    Mi = 4.002602 * au2cgs_m
    Zi = 2

    args_list = [(Ti,a_dust,s_dust,ne_val,Zi,an_dust,am_dust,Mi,nv,delta_max) for Ti in Tgas]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(tqdm(executor.map(compute_efficiency, args_list), 
                            total=nT, 
                            desc=f'    Calculating Helium heating rates for smallSil',
                            unit=' steps'))

    h_eff = np.array(results)
    H_helium = np.sqrt(32./(np.pi*Mi)) * np.pi * a_dust**2. * np.sqrt(kb*Tgas)*kb * h_eff

    # Write to txt file the results of (Tgas, H_helium) in a format suitable for Fortran90
    with open(f'{table_dir}/He_cooling_{dust_model.basic_a0[5]:.4f}_micron_Sil', 'w') as f:
        f.write(f'{nT}\n')
        for T, H in zip(np.log10(Tgas), np.log10(H_helium)):
            f.write(f'{T:14.6e} {H:14.6e}\n')
    
    a_dust = dust_model.basic_a0[3]*1e-4
    args_list = [(Ti,a_dust,s_dust,ne_val,Zi,an_dust,am_dust,Mi,nv,delta_max) for Ti in Tgas]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(tqdm(executor.map(compute_efficiency, args_list), 
                            total=nT, 
                            desc=f'    Calculating Helium heating rates for largeSil',
                            unit=' steps'))

    h_eff = np.array(results)
    H_helium = np.sqrt(32./(np.pi*Mi)) * np.pi * a_dust**2. * np.sqrt(kb*Tgas)*kb * h_eff

    # Write to txt file the results of (Tgas, H_helium) in a format suitable for Fortran90
    with open(f'{table_dir}/He_cooling_{dust_model.basic_a0[6]:.4f}_micron_Sil', 'w') as f:
        f.write(f'{nT}\n')
        for T, H in zip(np.log10(Tgas), np.log10(H_helium)):
            f.write(f'{T:14.6e} {H:14.6e}\n')
    
    # 10. Carbon cooling for carbonaceous material
    s_dust = dust_model.basic_s[2]
    a_dust = dust_model.basic_a0[2]*1e-4
    am_dust = 12.011 * au2cgs_m
    an_dust = 6.
    ne_val = 4
    Mi = 12.011 * au2cgs_m
    Zi = 6

    args_list = [(Ti,a_dust,s_dust,ne_val,Zi,an_dust,am_dust,Mi,nv,delta_max) for Ti in Tgas]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(tqdm(executor.map(compute_efficiency, args_list), 
                            total=nT, 
                            desc=f'    Calculating Carbon heating rates for smallC',
                            unit=' steps'))

    h_eff = np.array(results)
    H_carbon = np.sqrt(32./(np.pi*Mi)) * np.pi * a_dust**2. * np.sqrt(kb*Tgas)*kb * h_eff

    # Write to txt file the results of (Tgas, H_carbon) in a format suitable for Fortran90
    with open(f'{table_dir}/C_cooling_{dust_model.basic_a0[2]:.4f}_micron_Gra', 'w') as f:
        f.write(f'{nT}\n')
        for T, H in zip(np.log10(Tgas), np.log10(H_carbon)):
            f.write(f'{T:14.6e} {H:14.6e}\n')
    
    a_dust = dust_model.basic_a0[3]*1e-4

    args_list = [(Ti,a_dust,s_dust,ne_val,Zi,an_dust,am_dust,Mi,nv,delta_max) for Ti in Tgas]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(tqdm(executor.map(compute_efficiency, args_list), 
                            total=nT, 
                            desc=f'    Calculating Carbon heating rates for largeC',
                            unit=' steps'))

    h_eff = np.array(results)
    H_carbon = np.sqrt(32./(np.pi*Mi)) * np.pi * a_dust**2. * np.sqrt(kb*Tgas)*kb * h_eff

    # Write to txt file the results of (Tgas, H_carbon) in a format suitable for Fortran90
    with open(f'{table_dir}/C_cooling_{dust_model.basic_a0[3]:.4f}_micron_Gra', 'w') as f:
        f.write(f'{nT}\n')
        for T, H in zip(np.log10(Tgas), np.log10(H_carbon)):
            f.write(f'{T:14.6e} {H:14.6e}\n')
    
    # 11. Carbon cooling for silicate material
    s_dust = dust_model.basic_s[5]
    a_dust = dust_model.basic_a0[5]*1e-4
    am_dust = (24.305 + 55.845 + 28.0855 + 4*15.999) / 7. * au2cgs_m
    an_dust = int((4*8 + 14 + 26 + 12) / 7)
    ne_val = (2+8+4+4*6) / 7
    Mi = 12.011 * au2cgs_m
    Zi = 6

    args_list = [(Ti,a_dust,s_dust,ne_val,Zi,an_dust,am_dust,Mi,nv,delta_max) for Ti in Tgas]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(tqdm(executor.map(compute_efficiency, args_list), 
                            total=nT, 
                            desc=f'    Calculating Carbon heating rates for smallSil',
                            unit=' steps'))

    h_eff = np.array(results)
    H_carbon = np.sqrt(32./(np.pi*Mi)) * np.pi * a_dust**2. * np.sqrt(kb*Tgas)*kb * h_eff

    # Write to txt file the results of (Tgas, H_carbon) in a format suitable for Fortran90
    with open(f'{table_dir}/C_cooling_{dust_model.basic_a0[5]:.4f}_micron_Sil', 'w') as f:
        f.write(f'{nT}\n')
        for T, H in zip(np.log10(Tgas), np.log10(H_carbon)):
            f.write(f'{T:14.6e} {H:14.6e}\n')

    
    a_dust = dust_model.basic_a0[3]*1e-4
    args_list = [(Ti,a_dust,s_dust,ne_val,Zi,an_dust,am_dust,Mi,nv,delta_max) for Ti in Tgas]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(tqdm(executor.map(compute_efficiency, args_list), 
                            total=nT, 
                            desc=f'    Calculating Carbon heating rates',
                            unit=' steps'))

    h_eff = np.array(results)
    H_carbon = np.sqrt(32./(np.pi*Mi)) * np.pi * a_dust**2. * np.sqrt(kb*Tgas)*kb * h_eff
    
    # Write to txt file the results of (Tgas, H_carbon) in a format suitable for Fortran90
    with open(f'{table_dir}/C_cooling_{dust_model.basic_a0[6]:.4f}_micron_Sil', 'w') as f:
        f.write(f'{nT}\n')
        for T, H in zip(np.log10(Tgas), np.log10(H_carbon)):
            f.write(f'{T:14.6e} {H:14.6e}\n')

    print('Done!')