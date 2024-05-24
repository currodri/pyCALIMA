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
import dust_model
import pandas as pd
import os
from tqdm import tqdm
import concurrent.futures
import time

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
U0               = {'C': 4.0, 'Sil': 5.7} # [eV] - surface binding energy
Ksput            = {'C': 0.65, 'Sil': 0.1} # [] - free parameter fitting to the observed yields
size_correction  = { # [] - fitting to the results of Bocchio et al. (2006) for the size correction
                    #       to the sputtering yields for the semi-infinite target approximation
    'C'          : {
        'p1'     : 4.9,
        'p2'     : 0.55,
        'p3'     : 0.77,
        'p4'     : 4.7,
        'p5'     : 3.0,
        'p6'     : 1.2,
        'alpha_P': -4.73
    },
    'Sil'        : {
        'p1'     : 1.0,
        'p2'     : 0.5,
        'p3'     : 1.0,
        'p4'     : 1.8,
        'p5'     : 2.1,
        'p6'     : 0.76,
        'alpha_P': -3.34
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
    
    f1 = fit_params['p1'] * np.exp(-(np.log(x/fit_params['p2']))**2. / (2. * fit_params['p3']**2.))
    f2 = -fit_params['p4'] * np.exp(-(fit_params['p5']*x - fit_params['p6'])**2.)
    return 1. + f1 + f2

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
            
def screened_Coulomb_function(epsilon):
    """
    Screened Coulomb interaction approximation (Matsunami et al. 1980).

    Args:
        epsilon (float): reduced energy

    Returns:
        float: screened Coulomb factor
    """    
    s1 = 3.441 * np.sqrt(epsilon) * np.log(epsilon + 2.718)
    s2 = 1. + 6.35 * np.sqrt(epsilon) + epsilon * (-1.708 + 6.882 * np.sqrt(epsilon))
    
    return s1 / s2        

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

def sputtering_yield(dust_radius,surface_energy,Kparam,
                        dust_atomic_mass,ion_atomic_mass,
                        dust_atomic_number,ion_atomic_number,ion_charge,
                        bocchio_fit_params,alphaP,ion_energy,
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
        if (E_charge/ion_energy)>0.99:
            if ion_energy + E_charge > E_sp:
                print('Particle moved to high E')
        ion_energy = ion_energy + E_charge
    if ion_energy < E_sp:
        return 0.0
        
    if do_size_correction:
        # 2. Compute the ion penetration depth
        rp = penetration_depth(ion_charge,alphaP,ion_energy)
        
        # 3. Compute the size-dependent correction
        x = dust_radius / (0.7 * rp)
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
    
    from scipy.integrate import trapezoid
    
    dust_radius,surface_energy,Kparam,\
        dust_atomic_mass,ion_atomic_mass,\
        dust_atomic_number,ion_atomic_number,\
        ion_charge,bocchio_fit_params,alphaP,Tgas,\
        nbins_v,do_size_correction,do_dust_charge = args
    
    # 1. Determine the minumum velocity for reaching the threshold energy
    E_sp = threshold_energy(surface_energy,dust_atomic_mass,ion_atomic_mass)
    v_0  = np.sqrt(2. * E_sp * eV2erg) # [cm/s]
    while Maxwell_Boltzmann_function(v_0,ion_atomic_mass*au2cgs_m,Tgas) < 1e-20:
        v_0 = 2 * v_0
    
    # 2. We set the maximum velocity to the thermal energy of gas at ~1e11 K
    v_max = np.sqrt(2. * 1e7 * eV2erg / (ion_atomic_mass*au2cgs_m)) # [cm/s]
    while Maxwell_Boltzmann_function(v_max,ion_atomic_mass*au2cgs_m,Tgas) < 1e-20:
        v_max = v_max/ 2.0
    
    # 3. Perform loop over velocity range
    v = np.logspace(np.log10(v_0),np.log10(v_max),nbins_v)
    Y_v = np.zeros(nbins_v)
    
    # 4. Compute the mean dust charge
    if do_dust_charge:
        if dust_atomic_number == 6:
            Zmean = dust_model.grain_mean_charge(1.0,Tgas,0.1,'carbonaceous',str(int(dust_radius*1e7))+'A')
        else:
            Zmean = dust_model.grain_mean_charge(1.0,Tgas,0.1,'silicates',str(int(dust_radius*1e7))+'A')
    else:
        Zmean = 0.0

    for i in range(0, nbins_v):
        vi = v[i] # [cm/s]
        Ei = 0.5 * ion_atomic_mass * au2cgs_m * vi**2. # [erg]
        Ei = Ei / eV2erg
        mb_factor = Maxwell_Boltzmann_function(vi,ion_atomic_mass*au2cgs_m,Tgas)
        Y = sputtering_yield(dust_radius,surface_energy,Kparam,
                                dust_atomic_mass,ion_atomic_mass,
                                dust_atomic_number,ion_atomic_number,ion_charge,
                                bocchio_fit_params,alphaP,Ei,
                                do_size_correction,Zmean)
        Y_v[i] = mb_factor * vi * Y

    # 5. Integrate Y_v with the trapezoid method
    Y0 = trapezoid(Y_v,v)
    
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
        a_dust = dust_model.basic_a0[2]*1e-4 # [microns]
        m_dust = 4./3. * np.pi * (a_dust)**2. * dust_model.basic_s[2] # [g]
        rho_dust = dust_model.basic_s[2] # [g/cm^3]
        am_dust = 12.011 # [a.u.]
        an_dust = 6
        alphaP = size_correction['C']['alpha_P']
        bocchio_fitparams = size_correction['C']
        Kparam = Ksput['C']
        surface_energy = U0['C']
    elif dust_type == 'largeC':
        a_dust = dust_model.basic_a0[3]*1e-4 # [cm]
        m_dust = 4./3. * np.pi * (a_dust)**2. * dust_model.basic_s[3] # [g]
        rho_dust = dust_model.basic_s[3] # [g/cm^3]
        am_dust = 12.011 # [a.u.]
        an_dust = 6
        alphaP = size_correction['C']['alpha_P']
        bocchio_fitparams = size_correction['C']
        Kparam = Ksput['C']
        surface_energy = U0['C']
    elif dust_type == 'smallSil':
        # Assuming MgFeSiO4 composition (olivine with Iron inclusions as the regular model)
        a_dust = dust_model.basic_a0[5]*1e-4 # [cm]
        m_dust = 4./3. * np.pi * (a_dust)**2. * dust_model.basic_s[5] # [g]
        rho_dust = dust_model.basic_s[5] # [g/cm^3]
        am_dust = (24.305 + 55.845 + 28.0855 + 4*15.999) / 7. # [a.u.]
        an_dust = int((4*8 + 14 + 26 + 12) / 7)
        alphaP = size_correction['Sil']['alpha_P']
        bocchio_fitparams = size_correction['Sil']
        Kparam = Ksput['Sil']
        surface_energy = U0['Sil']
    elif dust_type == 'largeSil':
        # Assuming MgFeSiO4 composition (olivine with Iron inclusions as the regular model)
        a_dust = dust_model.basic_a0[6]*1e-4 # [cm]
        m_dust = 4./3. * np.pi * (a_dust)**2. * dust_model.basic_s[6] # [g]
        rho_dust = dust_model.basic_s[6] # [g/cm^3]
        am_dust = (24.305 + 55.845 + 28.0855 + 4*15.999) / 7. # [a.u.]
        an_dust = int((4*8 + 14 + 26 + 12) / 7)
        alphaP = size_correction['Sil']['alpha_P']
        bocchio_fitparams = size_correction['Sil']
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
        args_list = [(a_dust,surface_energy,Kparam,\
                        am_dust,ion_atomic_masses[i],\
                        an_dust,ion_atomic_numbers[i],\
                        ion_charges[i],bocchio_fitparams,alphaP,Ti,\
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
    table_dir = './thermal_sputtering_data'
    if not os.path.exists(table_dir):
        os.mkdir(table_dir)
    
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
    fig, ax = plt.subplots(1,1, figsize=(7,5),dpi=300,facecolor='w',edgecolor='k',sharey=True)
    
    ax.set_ylabel(r'Sputtering rate $(1/n_{\rm H})da/dt$ [cm$^3 \mu$m yr$^{-1}$]', fontsize=14)
    ax.set_xlabel(r'$T$ [K]',fontsize=14)
    ax.set_ylim([1e-9,1e-4])
    ax.set_xlim([6e3,1e9])
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.tick_params(labelsize=12)
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
    ax.plot(Tgas,Y_Sil,linestyle='--',color='sandybrown',label='Sil: Nozawa et al. (2006)')
    ax.plot(Tgas,Y_C,linestyle='--',color='cornflowerblue',label='C: Nozawa et al. (2006)')
    
    table_dir = './thermal_sputtering_data'
    if not os.path.exists(table_dir):
        os.mkdir(table_dir)
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
        ax.plot(Tgas,Y_smallC,linestyle='-',color='steelblue',label='smallC: No size correction')
        
        a_dust, Tgas, Y_smallC = total_erosion_rate(Tmin,Tmax,'smallC',
                                            ion_atomic_masses,
                                            ion_atomic_numbers,
                                            ion_charges,
                                            ion_abundances,
                                            nT,nbins_v,
                                            True,False)
        ax.plot(Tgas,Y_smallC,linestyle=(0, (3, 1, 1, 1)),color='steelblue',label='smallC: With size correction')
        print(Y_smallC)
        coefficients = np.polyfit(np.log10(Tgas[Y_smallC>0]), np.log10(Y_smallC[Y_smallC>0]), 6)
        poly_fit = np.poly1d(coefficients)
        ax.plot(Tgas,10**poly_fit(np.log10(Tgas)),linestyle=':',color='k',alpha=0.6)
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
        ax.plot(Tgas,Y_largeC,linestyle=':',color='cornflowerblue',label='largeC: With size correction')
        coefficients = np.polyfit(np.log10(Tgas[Y_largeC>0]),  np.log10(Y_largeC[Y_largeC>0]), 6)
        poly_fit = np.poly1d(coefficients)
        ax.plot(Tgas,10**poly_fit(np.log10(Tgas)),linestyle=':',color='k',alpha=0.6)
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
        
        a_dust, Tgas, Y_smallSil = total_erosion_rate(Tmin,Tmax,'smallSil',
                                            ion_atomic_masses,
                                            ion_atomic_numbers,
                                            ion_charges,
                                            ion_abundances,
                                            nT,nbins_v,
                                            False,False)
        ax.plot(Tgas,Y_smallSil,linestyle='-',color='saddlebrown',label='smallSil: No size correction')
        
        
        a_dust, Tgas, Y_smallSil = total_erosion_rate(Tmin,Tmax,'smallSil',
                                            ion_atomic_masses,
                                            ion_atomic_numbers,
                                            ion_charges,
                                            ion_abundances,
                                            nT,nbins_v,
                                            True,False)
        ax.plot(Tgas,Y_smallSil,linestyle=(0, (3, 1, 1, 1)),color='saddlebrown',label='smallSil: With size correction')
        coefficients = np.polyfit(np.log10(Tgas[Y_smallSil>0]),  np.log10(Y_smallSil[Y_smallSil>0]), 6)
        poly_fit = np.poly1d(coefficients)
        ax.plot(Tgas,10**poly_fit(np.log10(Tgas)),linestyle=':',color='k',alpha=0.6)
        file.write("======================\n")
        file.write("Polynomial Coefficients smallSil grains (%.4f microns):\n"%(a_dust/1e-4))
        file.write("f(x) = ")
        for i, coeff in enumerate(coefficients):
            file.write(f"{coeff:.8e}x^{6-i} ")
            if i < len(coefficients) - 1:
                file.write("+ ")
        file.write("\n")
        file.write("\n")
        file.write("Original Data (Tgas, Yield):\n")
        for i in range(len(Tgas)):
            file.write(f"{Tgas[i]:.6e}, {Y_smallSil[i]:.6e}\n")
        
        
        a_dust, Tgas, Y_largeSil = total_erosion_rate(Tmin,Tmax,'largeSil',
                                            ion_atomic_masses,
                                            ion_atomic_numbers,
                                            ion_charges,
                                            ion_abundances,
                                            nT,nbins_v,
                                            True,False)
        ax.plot(Tgas,Y_largeSil,linestyle=':',color='sandybrown',label='largeSil: With size correction')
        
        coefficients = np.polyfit(np.log10(Tgas[Y_largeSil>0]),  np.log10(Y_largeSil[Y_largeSil>0]), 6)
        poly_fit = np.poly1d(coefficients)
        ax.plot(Tgas,10**poly_fit(np.log10(Tgas)),linestyle=':',color='k',alpha=0.6)
        file.write("======================\n")
        file.write("Polynomial Coefficients largeSil grains (%.4f microns):\n"%(a_dust/1e-4))
        file.write("f(x) = ")
        for i, coeff in enumerate(coefficients):
            file.write(f"{coeff:.8e}x^{6-i} ")
            if i < len(coefficients) - 1:
                file.write("+ ")
        file.write("\n")
        file.write("\n")
        file.write("Original Data (Tgas, Yield):\n")
        for i in range(len(Tgas)):
            file.write(f"{Tgas[i]:.6e}, {Y_largeSil[i]:.6e}\n")
            
    print(f"Results written to thermal_sputtering_polynomial_fits{label}.txt")
    
    ax.legend(loc='best', frameon=False, fontsize=10, ncol=2)
    fig.subplots_adjust(top=0.98,bottom=0.1,left=0.15,right=0.99,hspace=0,wspace=0)
    return fig
    