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
import PAHs_model
import pandas as pd
import os
from tqdm import tqdm
import concurrent.futures
import time

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

# Fitting parameters for the friction coefficient scaling law from the
# results of Pusk & Nieminen (1983). Gamma_0 is in terms of (a.u.)^2.
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
    
    from scipy.integrate import trapezoid,quad
    from scipy.optimize import fsolve
    
    def F(E):
        integrand = lambda x: inv_electron_stopping_power(x)
        result, error = quad(integrand, e_sp_min, E)
        return result
    
    # 1. Compute length through the PAH (convert from a.u. to Angstrom)
    l = path_l(R,d,theta) * a_0 / 1e-8

    # 2. Obtain the value of F for E0
    F_0 = F(init_energy)
    
    # 3. Compute F(E_1)
    F_1 = F_0 - l

    # 4. Compute E1 by solving F(E_1) numerically
    E_1 = fsolve(lambda x: F(x) - F_1, x0=e_sp_min)[0]
    
    # 5. The final excitation energy is the difference between E1 and the initial electron energy
    T = init_energy - E_1

    return T

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

def dissociation_probability(binding_energy,Nc,T_av):
    """Normalised PAH dissociation probability.

    Args:
        binding_energy (float): fragment binding energy to PAH molecule in eV
        Nc (int): number of carbon atoms in PAH molecule
        T_av (float): average temperature of the excitation in K

    Returns:
        float: normalised probability for dissociation
    """
    from unyt import h,K,kb    
    # 1. Compute the maximum number of IR photon emissions
    # as suggested by the results of Micelotta et al. (2010b)
    n_max = float(int(Nc / 5))
    
    # 2. Compute the probability based on the value of T_av
    DeltaS = 10 # [cal/K/mol]
    R = 1.98720425864083 # [cal/K/mol]
    k_0 = kb*T_av*K/h * np.exp(1.+DeltaS/R)
    
    P = k_0.to('1/s').d * np.exp(-binding_energy/(8.617e-5*T_av)) / ((k_IR / (n_max + 1.)) + k_0.to('1/s').d * np.exp(-binding_energy/(8.617e-5*T_av)))
    
    return P
    
def electronic_destruction_rate_T(Tgas,particle_type,Nc,
                                  binding_energy=E_0,
                                  nbins_v=100,
                                  nbins_theta=30,
                                  radius_method='Draine21'):
    
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
    if radius_method == 'Draine21':
        R = PAHs_model.size_from_Nc(Nc) * 1e-8 # [cm]
    elif radius_method == 'Omont86':
        R = 0.9 * np.sqrt(Nc) * 1e-8 # [cm]
    else:
        raise NameError(f'This radius_method is not included in this model: {radius_method}')
    R = R / a_0
    d = thickness / a_0
    n_max = float(int(Nc / 5))
    
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
                if Ei > e_sp_min:
                    T_0 = electronic_electron_collision(R,d,theta[j],Ei)
                else:
                    # If the initial electron energy is below the minimum for
                    # the stopping power function, the dissociation probability
                    # is assumed to be equal to zero
                    J_theta[j] = 0.0
                    continue
            else:
                T_0 = electronic_ion_collision(vi,R,d,theta[j],particle_type)

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
        Ti, particle_type, Nc, binding_energy, nbins_v, nbins_theta, radius_method = args
        return Ti, electronic_destruction_rate_T(Ti, particle_type, Nc, binding_energy, nbins_v, nbins_theta, radius_method)
            
def electronic_destruction_rate(Tmin,Tmax,particle_type,Nc,
                                binding_energy=E_0,nT=100,
                                nbins_v=100,nbins_theta=30,
                                radius_method='Draine21'):
    
    T = np.logspace(np.log10(Tmin),np.log10(Tmax),nT)
    
    J = np.zeros(nT)
    
    # Print the number of available cores
    num_cores = os.cpu_count()
    print(f"    Number of cores available: {num_cores}")
    # Create argument list for parallel processing
    args_list = [(Ti, particle_type, Nc, binding_energy, nbins_v, nbins_theta, radius_method) for Ti in T]
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(tqdm(executor.map(wrapper_electronic_rate, args_list), total=nT, desc=f'    Calculating electronic {particle_type} rates', unit=' steps'))

    T, J = zip(*results)
    return np.array(T), np.array(J)

# Nuclear collisions

# Data extracted from the cross sections for H, He and C against C collision
# as given by the results of Micelotta et al. (2010a) in Fig. 2
H2C_cross_section = pd.read_csv('H2C_cross_section.csv',header=1)
He2C_cross_section = pd.read_csv('He2C_cross_section.csv',header=1)
C2C_cross_section = pd.read_csv('C2C_cross_section.csv',header=1)

def nuclear_cross_section_interpolfunc(particle_type):
    """Obtain a linear interpolation function for the nuclear
    cross section for different particle types against carbon.

    Args:
        particle_type (str): particle type, either 'H', 'He' or 'C'

    Returns:
        scipy.interpolate.interpolate.interp1d: interpolation function with extrapolate ON
    """    
    
    from scipy.interpolate import interp1d
    
    # 1. Load data from the .csv files
    if particle_type == 'H':
        sigma = H2C_cross_section
    elif particle_type == 'He':
        sigma = He2C_cross_section
    elif particle_type == 'C':
        sigma = C2C_cross_section
    else:
        raise NameError(f'This particle_type is not included in this model: {particle_type}')
        
    # 2. Construct the linear interpolation function
    interpolation_function = interp1d(np.log10(sigma['Energy Ion (eV)']), np.log10(sigma['Cross section (A^2/atom)']), kind='linear',fill_value='extrapolate')
    
    return interpolation_function

def nuclear_destruction_rate_T(Tgas,M1,M2,Z1,Z2,threshold_E,m_particle,nbins_v=100):
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
    
    for i in range(0, nbins_v):
        vi = v[i]
        Ei = (0.5 * m_particle * v[i]**2.) / eV2erg
        mb_factor = Maxwell_Boltzmann_function(vi,m_particle,Tgas)
        cross_section = energy_transfer_cross_section(M1,M2,Z1,Z2,Ei,threshold_E) * 1e-16 # [cm^2]
        J_v[i] = mb_factor * cross_section * vi

    # 4. Integrate J_v with the trapezoid method
    J = trapezoid(J_v,v)
    
    return J

def wrapper_nuclear_rate(args):
   Tgas,M1,M2,Z1,Z2,threshold_E,m_particle,nbins_v = args
   return Tgas,nuclear_destruction_rate_T(Tgas,M1,M2,Z1,Z2,threshold_E,m_particle,nbins_v) 

def nuclear_destruction_rate(Tmin,Tmax,particle_type,Nc,
                             threshold_energy=7.5,
                             nT=100,nbins_v=1000):
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
    
    T = np.logspace(np.log10(Tmin),np.log10(Tmax),nT)
    
    J = np.zeros(nT)
    
    # Print the number of available cores
    num_cores = os.cpu_count()
    print(f"    Number of cores available: {num_cores}")
    # Create argument list for parallel processing
    args_list = [(Ti, M1, M2, Z1, Z2, threshold_energy, m_particle, nbins_v) for Ti in T]
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(tqdm(executor.map(wrapper_nuclear_rate, args_list), total=nT, desc=f'    Calculating nuclear {particle_type} rates', unit=' steps'))

    T, J = zip(*results)
    return np.array(T), np.array(J)

# Export functions
def export_rates(RPAH,Tmin,Tmax,threshold_energy=7.5,
                 binding_energy=E_0,nT=100,
                 nbins_v=100,nbins_theta=50,
                 radius_method='Draine21',
                 plot_rates=True,nH_plot=1.0,
                 Z_plot=1):
    start_time = time.time()
    print(40*"-")
    print('PAH SPUTTERING IN A HOT GAS')
    print('By: F. Rodriguez Montero (2024)')
    print('1. Obtaining rates...')
    if radius_method == 'Draine21':
        Nc = PAHs_model.Nc_from_size(RPAH)
        RPAH = RPAH * 1e-4  # [micron]
        print(f'The selected PAH radius corresponds to {Nc} carbon atoms')
    elif radius_method == 'Omont86':
        Nc = (RPAH/0.9)**2.
        RPAH = RPAH * 1e-4  # [micron]
        print(f'The selected PAH radius corresponds to {Nc} carbon atoms')
    else:
        raise NameError(f'This radius_method is not included in this model: {radius_method}')

    
    # 1. Obtain the thermal electron destruction rate
    print('    Electron sputtering...')
    T, J_electron = electronic_destruction_rate(Tmin,Tmax,'e',Nc,
                                                binding_energy=binding_energy,
                                                nT=nT,nbins_v=nbins_v,
                                                nbins_theta=nbins_theta,
                                                radius_method=radius_method)

    # 2. Obtain the electronic excitation caused by ions
    J_electronic = {}
    for ptype in friction_params.keys():
        print(f'    {ptype} electronic sputtering...')
        T, J = electronic_destruction_rate(Tmin,Tmax,ptype,Nc,
                                            binding_energy=binding_energy,
                                            nT=nT,nbins_v=nbins_v,
                                            nbins_theta=nbins_theta,
                                            radius_method=radius_method)
        J_electronic[ptype] = J
    
    # 3. Obtain the nuclear collision rate for ions
    J_ion = {}
    for ptype in friction_params.keys():
        print(f'    {ptype} nuclear sputtering...')
        T, J = nuclear_destruction_rate(Tmin,Tmax,ptype,Nc,
                                        threshold_energy=threshold_energy,
                                        nT=nT,nbins_v=nbins_v)
        
        J_ion[ptype] = 0.5 * Nc * J
    
    print(' => Rates done!')
    
    print('2. Preparing data to save in files...')
    
    # 4. Save electronic rate into single file
    PAH_dir = './PAH_sputtering_data'
    if not os.path.exists(PAH_dir):
        os.mkdir(PAH_dir)
    electron_filename = PAH_dir+'/electron_sputtering_%.4f_micron_PAH'%(RPAH)
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
        ion_filename =  PAH_dir+'/%s_sputtering_%.4f_micron_PAH'%(ptype,RPAH)
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