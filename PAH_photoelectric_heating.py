"""
PHOTOELECTRIC HEATING MODEL FOR PAHS

This is based in the modelling of PAH photoelectric heating
by Berne et al. (2022).


By: F. Rodriguez Montero (currodri@gmail.com)
"""

# LIBRARIES
import os
import re
import sys
import csv
import numpy as np
from tqdm import tqdm
import concurrent.futures
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import matplotlib as mpl
import seaborn as sns
from unyt import nm,m,cm,eV,J,s,h,c
from dust_model import basic_a0,basic_amin,basic_amax,basic_sigma,basic_s,LogNormal_Distribution
from dust_oppacity import pah_efficiencies
from PAHs_model import Draine_1978_isrf

os.environ["OMP_NUM_THREADS"] = "1"  # Set it to the desired number of threads
os.environ['OPENBLAS_NUM_THREADS'] = '1'

sns.set_theme(style="white")
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "Computer Modern Roman",
})

BERNEPATH = '/home/currodri/Codes/photoelectric-heating'
sys.path.append(BERNEPATH)
from four_levels_model import HeatingGas

# CONSTANTS
pahneu_filepath = '/home/currodri/Codes/DustRAMSES/li_draine_2001/PAHneu_30'
pahion_filepath = '/home/currodri/Codes/DustRAMSES/li_draine_2001/PAHion_30'
epsilon_0 =  8.8541878188e-21 # Vacuum permittivity [F/nm]
e = 1.602176634e-19           # Elementary charge [C]
partition_coeff = 0.46        # Partition coefficient estimated from Bréchignac et al. 2014

# FUNCTIONS
class PAHDataset:
    def __init__(self, file_path):
        self.file_path = file_path
        self.entries = []
        self._parse_file()

    def _parse_file(self):
        with open(self.file_path, 'r') as file:
            content = file.read()

        # Split the content into individual entries using the UID field as delimiter
        raw_entries = content.split("###############################################################################")[1:]
        
        for entry in raw_entries:
            if not entry.strip():
                continue
            
            # Initialize a dictionary to hold the data for the current entry
            data = {
                'UID': None,
                'COMMENTS': [],
                'PROPERTIES': {},
                'GEOMETRY': [],
                'TRANSITIONS': []
            }

            # Parse UID
            uid_match = re.search(r"UID:\s*(\d+)", entry)
            if uid_match:
                data['UID'] = int(uid_match.group(1))

            # Parse comments
            comments = re.findall(r"#\s+(.+)", entry)
            data['COMMENTS'] = comments

            # Parse properties
            properties = re.findall(r"(\w+):\s*([^\n]+)", entry)
            for prop, value in properties:
                if prop in ['CHARGE', 'SYMMETRY', 'N_SOLO', 'N_DUO', 'N_TRIO', 'N_QUARTET', 'N_QUINTET', 'N_CH2', 'N_CHX']:
                    data['PROPERTIES'][prop] = int(value)
                elif prop in ['WEIGHT', 'TOTAL_E', 'VIB_E']:
                    data['PROPERTIES'][prop] = float(value)
                else:
                    data['PROPERTIES'][prop] = value

            # Parse geometry
            geometry_section = re.search(r"# GEOMETRY:(.+?)# TRANSITIONS:", entry, re.DOTALL)
            if geometry_section:
                geometry_lines = geometry_section.group(1).strip().split("\n")[1:]  # Skip the header line
                for line in geometry_lines:
                    parts = line.split()
                    atom_data = {
                        'POSITION': int(parts[0]),
                        'X': float(parts[1]),
                        'Y': float(parts[2]),
                        'Z': float(parts[3]),
                        'TYPE': int(parts[4])
                    }
                    data['GEOMETRY'].append(atom_data)

            # Parse transitions
            transitions_section = re.search(r"# TRANSITIONS:(.+)", entry, re.DOTALL)
            if transitions_section:
                transitions_lines = transitions_section.group(1).strip().split("\n")[1:]  # Skip the header line
                for line in transitions_lines:
                    parts = line.split()
                    transition_data = {
                        'FREQUENCY': float(parts[0]),
                        'INTENSITY': float(parts[1]),
                        'SYMMETRY': parts[2],
                        'SCALE': float(parts[3])
                    }
                    data['TRANSITIONS'].append(transition_data)

            self.entries.append(data)

    def get_entry_by_uid(self, uid):
        for entry in self.entries:
            if entry['UID'] == uid:
                return entry
        return None

    def get_all_entries(self):
        return self.entries

def ionisation_potential(Z,a):
    """Ionisation potential following the empirical formalism of Weingartner
    and Draine (2001) with the updated parameters from Wenzel et al. (2020).

    Args:
        Z (int): Charge number
        a (np.float): Grain diameter in nanometre

    Returns:
        np.float: ionisation potential in [eV]
    """    
    
    
    if Z == -1:
        IP = 0.0
    else:
        IP = 3.9 + e/(4.*np.pi*epsilon_0) * ((Z + 0.5) / a + (Z+2.)/a * (0.003/a)) 
        
    return IP

def beta_factor(Nc):
    """Beta correction for the cation ionisation yield as obtained by Wenzel et al. (2020)

    Args:
        Nc (int): number of carbon atoms in PAH molecule

    Returns:
        np.float: beta factor
    """    
    
    if 32 <= Nc < 50:
        beta = 0.59 + 8.1e-3 * float(Nc)
    else:
        beta = 1.
    return beta

def ionisation_yield(Nc,Z,photon_energy,IP):
    """Ionisation yields

    Args:
        Nc (int): number of carbon atoms in PAH molecule
        Z (int): PAH charge
        photon_energy (np.float): Photon energy in [eV]
        IP (np.float): Ionisation potential in [eV]

    Returns:
        np.float: yield
    """    
    
    if Z == -1:
        Y = 1.
    elif Z == 0:
        if IP + 9.2 >= photon_energy:
            Y = (photon_energy - IP) / 9.2
        else:
            Y = 1.
    elif Z == 1:
        if photon_energy < IP:
            Y = 0.
        elif IP <= photon_energy <= 11.3:
            Y = 0.3 * (photon_energy - IP) / (11.3-IP)
        elif 11.3 <= photon_energy < 12.9:
            Y = 0.3
        elif 12.9 <= photon_energy < 15.0:
            beta = beta_factor(Nc)
            Y = (beta - 0.3) / 2.1 * (photon_energy - 12.9) + 0.3
        else:
            Y = beta_factor(Nc)
    
    return Y

def recombination_rate_Spitzer(Nc,Z,T):
    """Recombination rate following the Spitzer's formalism (Spitzer 2004) modified
    for cations by Verstraete et al. (1990) and extended to Z>0 by Berne et al. (2022)

    Args:
        Nc (int): Number of carbon atoms
        Z (int): PAH charge number
        T (np.float): Gas (or electron) temperature in [K]

    Returns:
        np.float: Recombination rate in [cm^3/s]
    """    
    phi = 1.85e5 / T / np.sqrt(Nc)
    k_rec = 1.28e-10 * Nc * np.sqrt(T) * (1. + phi * (1.+Z)) 
    return k_rec

def recombination_rate_Tielens21(Nc,T):
    """Recombination rate following Eq. 8.106 in Tielens (2021), which assumes
    a correction factor the the planar geometry of the PAH.

    Args:
        Nc (int): Number of carbon atoms
        T (np.float): Gas (or electron) temperature in [K]

    Returns:
        np.float: Recombination rate in [cm^3/s]
    """    
    k_rec = 1.3e-6 * np.sqrt(Nc) * np.sqrt(300. / T)
    return k_rec

def attachment_rate_Carelli13(T):
    """Electron attachment rate to neutral PAH as obtained for small PAHs
    in experiments by Carelli et al. (2013)

    Args:
        T (np.float): Gas (or electron) temperature in [K]

    Returns:
        np.float: Attachment rate in [cm^3/s]
    """    
    
    # Parameters for coronene (C24H12) from Carelli et al. (2013)
    a = 2.74e-9 # [cm-3]
    b = 0.11
    c = -1.11
    k_att = a * (T/300.)**b * np.exp(-c/T) # [cm^3/s]
    
    return k_att

def attachment_rate_Tielens05(Nc):
    """Electron attachment rate to neutral PAHs as given by Tielens (2005)

    Args:
        Nc (int): Number of carbon atoms in PAH molecule

    Returns:
        np.float: Attachment rate in [cm^3/s]
    """    
    
    # Electron dimensionless sticking coefficient as approximated by Tielens (2005)
    s_e = 1.0
    
    k_att = 1.3e-7 * s_e * np.sqrt(Nc)
    
    return k_att

def absorption_cross_section(distribution,Z):
    """Compute the distribution-averaged cross section for a
    given PAH molecule, considering whether or not the PAH is neutral
    or ionised. NOTE: anions are not allowed for this function.

    Args:
        distribution (LogNormal_Distribution): PAH molecule underlying log-normal distribution
        Z (int): charge of the PAH molecule

    Returns:
        (np.array,np.array): wavelength [microns], absorption cross section [m^2]
    """    
    
    # 1. Load data from the Li & Draine 2001 files
    if Z < 0:
        raise ValueError('Z cannot be negative!') 
    elif Z == 0:
        data,columns,name,nwav = pah_efficiencies(pahneu_filepath)
    else:
        data,columns,name,nwav = pah_efficiencies(pahion_filepath)
    
    # 2. Obtain the distribution averaged absorption cross section
    C_abs_eff = np.zeros(nwav)
    nrad = len(data.keys())
    akeys= list(data.keys())
    for j in range(0, nwav):
        sizes = np.zeros(nrad)
        C_abs = np.zeros(nrad)
        for k in range(0,nrad):
            tmpdt = data[akeys[k]]
            sizes[k] = float(akeys[k])
            C_abs[k] = tmpdt[j,columns.index('Q_abs')] * np.pi * sizes[k]**2.
            w        = tmpdt[:,columns.index('w(micron)')]
        C_abs_eff[j] = distribution.averaged_over_number(C_abs,sizes)
    
    return w, C_abs_eff*1e-12

def ionisation_rate(sigma_ion,I,E):
    """Compute the ionisation rate for a given PAH molecule bathed
    in the interstellar UV radiation field.

    Args:
        sigma_ion (np.array): Ionisation cross section [m^2]
        I (np.array): Local intensity of the UV radiation field [W / m^2 / eV]
        E (np.array): Photon energy [eV]

    Returns:
        np.float: Photo-ionisation rate [s-1]
    """    
    k_pe = sigma_ion * I / E * 6.24150935e+18 # Convert [W] to [eV/s]
    k_pe = np.trapz(k_pe,E)
    
    return k_pe

def power_absorbed(sigma_abs,I,E):
    """Compute the power absorbed by each PAH molecule.

    Args:
        sigma_abs (np.array): Absorption cross section [m^2]
        I (np.array): Local intensity of the UV radiation field [W / m^2 / eV]
        E (np.array): Photon energy [eV]

    Returns:
        np.float: Total radiation power absorbed [W]
    """    
    mask = E<13.6
    P_rad = sigma_abs[mask] * I[mask]
    P_rad = np.trapz(P_rad,E[mask])
    
    return P_rad
    
def power_injected(IP,sigma_ion,I,E):
    """Power injected into the gas by the photoelectrons.

    Args:
        IP (np.float): Ionisation potential of the Z+1 PAH [eV]
        sigma_ion (np.array): Cross section [m^2]
        I (np.array): Local intensity of the UV radiation field [W / m^2 / eV]
        E (np.array): Photon energy [eV]

    Returns:
        np.float: Injected power [W]
    """    
    mask = (E>IP) & (E<13.6)
    P_inj = partition_coeff * (E[mask] - IP) * sigma_ion[mask] * I[mask] / E[mask]
    P_inj = np.trapz(P_inj,E[mask])
    
    return P_inj

def compute_heating_efficiency(args):
    
    # 1. Unpack arguments
    G0,T,ne,dist,attach_model,wav,\
        sigma_abs_anion,sigma_abs_neu,\
            sigma_abs_cation = args
    a0 = dist.a0
    Nc = dist.Nc
    # print('SIZE:',a0*1e3,Nc,(Nc/468)**(1./3.))
    
    # 3. Convert wavelength [micron] to photon energy [eV]
    E = 1.2398 / wav
    # Convert from [photons cm^-2 s^-1 nm^-1] to [W m^-2 eV^-1]
    I = G0 * Draine_1978_isrf(wav*1e3) /1.7 * cm**-2/s/nm
    F = I * E * eV
    f = F *nm/ (1e-9*m) * h * c / (E*eV)**2 * eV / (e*J)
    I = f.to('W/m**2/eV').d
    
    # 4. Compute the e- detachment rate from the anion
    IP_anion = ionisation_potential(-1,a0*1e3)
    yield_anion = np.array([ionisation_yield(Nc,-1,E[i],IP_anion) for i in range(0,len(E))])
    mask = (E>IP_anion) & (E < 13.6)
    k_det = ionisation_rate(yield_anion[mask]*sigma_abs_anion[mask],I[mask],E[mask])
    # print('k_det',k_det)
    
    # 5. e- attachment to a neutral
    if attach_model == 'Berne':
        k_att = attachment_rate_Carelli13(T)
    elif attach_model == 'Tielens':
        k_att = attachment_rate_Tielens05(Nc)
    # print('k_att',k_att)
    
    # 6. Ionisation rate of Z=0 to Z=1
    IP_neutral = ionisation_potential(0,a0*1e3)
    yield_neutral = np.array([ionisation_yield(Nc,0,E[i],IP_neutral) for i in range(0,len(E))])
    mask = (E>IP_neutral) & (E < 13.6)
    # print(IP_neutral,E[mask],I[mask],yield_neutral[mask],sigma_abs_neu[mask])
    k_pe_0 = ionisation_rate(yield_neutral[mask]*sigma_abs_neu[mask],I[mask],E[mask])
    # print('k_pe_0',k_pe_0)
    
    # 7. Recombination rate from Z=1 to Z=0
    if attach_model == 'Berne':
        k_rec_1 = recombination_rate_Spitzer(Nc,1,T)
    elif attach_model == 'Tielens':
        k_rec_1 = recombination_rate_Tielens21(Nc,T)
    # print('k_rec_1',k_rec_1)
    
    # 8. Recombination rate from Z=2 to Z=1
    if attach_model == 'Berne':
        k_rec_2 = recombination_rate_Spitzer(Nc,2,T)
    elif attach_model == 'Tielens':
        k_rec_2 = recombination_rate_Tielens21(Nc,T)
    # print('k_rec_2',k_rec_2)

    
    # 9. Ionisation rate of Z=1 to Z=2
    IP_cation = ionisation_potential(1,a0*1e3)
    yield_cation = np.array([ionisation_yield(Nc,1,E[i],IP_cation) for i in range(0,len(E))])
    mask = (E>IP_cation) & (E < 13.6)
    k_pe_1 = ionisation_rate(yield_cation[mask]*sigma_abs_cation[mask],I[mask],E[mask])
    # print('k_pe_1',k_pe_1)

    
    # 10. Fraction of Z=-1
    f_anion = 1. / (1. + k_det / (k_att*ne) + \
                    k_det * k_pe_0 / (k_att*k_rec_1*ne**2.) + \
                    k_det * k_pe_0 * k_pe_1 / (k_att*k_rec_1*k_rec_2*ne**3.))
    
    # 11. Fraction of Z=0
    f_neutral = 1. / (1. + k_att*ne / k_det + k_pe_0 / (k_rec_1*ne) + \
                    k_pe_0 * k_pe_1 / (k_rec_1*k_rec_2*ne**2.))
    
    # 12. Fraction of Z=1
    f_1 = 1. / (1. + k_rec_1*ne / k_pe_0 + k_pe_1 / k_rec_2 + \
                k_att*k_rec_1*ne**2. / (k_det*k_pe_0))
    
    # 13. Fraction of Z=2
    f_2 = 1. / (1. + k_rec_2*ne / k_pe_1 + k_rec_1*k_rec_2*ne**2. / (k_pe_0*k_pe_1) + \
                k_att*k_rec_1*k_rec_2*ne**3./(k_det*k_pe_1**2.))
    
    # 14. Check that all fractions add up to 1
    f_tot = f_anion + f_neutral + f_1 + f_2
    f_anion, f_neutral, f_1, f_2 = f_anion/f_tot, f_neutral/f_tot, f_1/f_tot, f_2/f_tot
    
    # 15. Compute the total injected power
    Pinj_anion = power_injected(IP_anion,yield_anion*sigma_abs_anion,I,E)
    Pinj_neutral = power_injected(IP_neutral,yield_neutral*sigma_abs_neu,I,E)
    Pinj_cation = power_injected(IP_cation,yield_cation*sigma_abs_cation,I,E)
    
    Pinj = f_anion * Pinj_anion + f_neutral * Pinj_neutral + f_1 * Pinj_cation
    
    # 16. Compute the total absorbed power
    Prad_anion = power_absorbed(sigma_abs_anion,I,E)
    Prad_neutral = power_absorbed(sigma_abs_neu,I,E)
    Prad_cation = power_absorbed(sigma_abs_cation,I,E)
    
    Prad = f_anion * Prad_anion + f_neutral * Prad_neutral + (f_1 + f_2) * Prad_cation
    
    # 17. Compute the heating efficiency as the ratio of injected to absorbed power
    eff = Pinj / Prad
    
    return G0, ne, T, f_anion,f_neutral,f_1,f_2,eff
    
def multiline(xs, ys, c, ax=None, **kwargs):
    """Plot lines with different colorings

    Parameters
    ----------
    xs : iterable container of x coordinates
    ys : iterable container of y coordinates
    c : iterable container of numbers mapped to colormap
    ax (optional): Axes to plot on.
    kwargs (optional): passed to LineCollection

    Notes:
        len(xs) == len(ys) == len(c) is the number of line segments
        len(xs[i]) == len(ys[i]) is the number of points for each line (indexed by i)

    Returns
    -------
    lc : LineCollection instance.
    """
    from matplotlib.collections import LineCollection
    # find axes
    ax = plt.gca() if ax is None else ax

    # create LineCollection
    segments = [np.column_stack([x, y]) for x, y in zip(xs, ys)]
    lc = LineCollection(segments, **kwargs, norm=mpl.colors.LogNorm(vmin=c.min(),
                                                                    vmax=c.max()))

    # set coloring of line segments
    #    Note: I get an error if I pass c as a list here... not sure why.
    lc.set_array(np.asarray(c))

    # add lines to axes and rescale 
    #    Note: adding a collection doesn't autoscalee xlim/ylim
    ax.add_collection(lc)
    return lc

def Berne22_efficiency(G0min,G0max,ne_min,ne_max,T,ax,fig):
    n_ne = 20
    ne_list = np.logspace(np.log10(ne_min),np.log10(ne_max),n_ne)
    eff = np.zeros(n_ne)
    gamma = np.zeros(n_ne)
    for j in range(0, n_ne):
        PEH = HeatingGas('ISRF/draine1978.txt',1,T,ne_list[j],
                            54,0,ISRF=True)
        result = PEH.parameters()
        eff[j] = result[1]
        gamma[j] = result[3]
    ax.plot(gamma,eff,linestyle='-',color='k')    
def my_efficiency(pahtype,attach_model,G0min,G0max,ne_min,ne_max,T,ax,fig,do_colorbar=False):
    n_ne = 100
    n_G0 = 100
    G0_list = np.logspace(np.log10(G0min),np.log10(G0max),n_G0)
    ne_list = np.logspace(np.log10(ne_min),np.log10(ne_max),n_ne)
    xs = []
    ys = []
    if pahtype == 'small':
        dist = LogNormal_Distribution(basic_a0[0],basic_amin[0],basic_amax[0],basic_sigma[0],basic_s[0])
        dist.Nc = 54
        linestyle = '-'
    elif pahtype == 'large':
        dist = LogNormal_Distribution(basic_a0[1],basic_amin[1],basic_amax[1],basic_sigma[1],basic_s[1])
        dist.Nc = 400
        linestyle = '--'
    
    # 2. Compute the distribution-averaged absorption cross sections
    # NOTE: I am using the same cross section for Z=-1 than for Z=0
    wav,sigma_abs_anion = absorption_cross_section(dist,0)
    wav,sigma_abs_neu = absorption_cross_section(dist,0)
    wav,sigma_abs_cation = absorption_cross_section(dist,1)
    
    fig2, ax2 = plt.subplots(1, 1, sharex=True, figsize=(6,5), dpi=300, facecolor='w', edgecolor='k')

    ax2.set_ylabel(r'Population fractions', fontsize=16)
    ax2.set_xlabel(r'$\gamma (G0\sqrt{T}/n_e)$ [K$^{1/2}$ cm$^{-3}$]',fontsize=16)
    ax2.set_xscale('log')
    ax2.tick_params(labelsize=14)
    ax2.xaxis.set_ticks_position('both')
    ax2.yaxis.set_ticks_position('both')
    ax2.minorticks_on()
    ax2.tick_params(which='both',axis="both",direction="in")
    ax2.set_xlim([10,1e+6])
    ax2.set_ylim([0,1])

    for j in range(0, n_ne):
        gamma = np.zeros(n_G0)
        num_cores = 20#min(os.cpu_count(),n_G0)
        args_list = []
        for k in range(0,n_G0):
            args = G0_list[k],T,ne_list[j],dist,attach_model,wav,\
                    sigma_abs_anion,sigma_abs_neu,sigma_abs_cation
            args_list.append(args)
            gamma[k] = G0_list[k] * np.sqrt(T) / ne_list[j]
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
            results = list(tqdm(executor.map(compute_heating_efficiency, args_list), total=n_G0,
                                desc=f'    Computing efficiency for ne={ne_list[j]} cm^-3', unit=' steps'))
        
        f_anion,f_neutral,f_1,f_2,epsilon = zip(*results)
        ax2.plot(gamma,f_anion,color='k')
        ax2.plot(gamma,f_neutral,color='g')
        ax2.plot(gamma,f_1,color='b')
        ax2.plot(gamma,f_2,color='r')

        xs.append(gamma)
        ys.append(epsilon)
    lc = multiline(xs,ys,ne_list,cmap='jet',ax=ax,lw=1,linestyle=linestyle)   
    if do_colorbar:
        axcb = fig.colorbar(lc, orientation="vertical", pad=0.0)
        axcb.set_label(r'$n_e$ [cm$^{-3}$]',fontsize=16)
    
    dummy_lines = [ax2.plot([],[],color='k',linestyle='-',label=r'$Z=-1$')[0],
                   ax2.plot([],[],color='g',linestyle='-',label=r'$Z=0$')[0],
                   ax2.plot([],[],color='b',linestyle='-',label=r'$Z=1$')[0],
                   ax2.plot([],[],color='r',linestyle='-',label=r'$Z=2$')[0]]
    first_legend = ax2.legend(handles=dummy_lines, loc='best', frameon=False, fontsize=14)
    ax2.add_artist(first_legend)
    fig2.subplots_adjust(top=0.98,bottom=0.13,left=0.13,right=0.95)
    fig2.savefig(f'pah_charge_distribution_{attach_model}_{dist.Nc}C_{str(int(T))}K.pdf', format='pdf', dpi=300)

def Tielens2001_efficiency(ax):
    n_gamma = 20
    gamma = np.logspace(np.log10(1),np.log10(1e+15),n_gamma)
    eff = 0.06 / (1.0 + 7e-5*gamma)
    ax.plot(gamma,eff,color='k',linestyle=':',label='Tielens 2001')
    
def Wolfire2003_efficiency(T,ax):
    n_gamma = 20
    gamma = np.logspace(np.log10(1),np.log10(1e+15),n_gamma)
    eff = 4.9e-2/(1.0+2.411e-3*gamma**0.73) + 3.7e-2*(T/1e4)**0.7/(1.0+1e-4*gamma)
    ax.plot(gamma,eff,color='k',linestyle='-.',label='Wolfire et al. 2003')

def compare_eff_curves(G0min,G0max,T,ne_min,ne_max):
    
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(6,5), dpi=300, facecolor='w', edgecolor='k')

    ax.set_ylabel(r'$\epsilon_{\Gamma},\epsilon_{\rm PAH}$', fontsize=16)
    ax.set_xlabel(r'$\gamma (G0\sqrt{T}/n_e)$ [K$^{1/2}$ cm$^{-3}$]',fontsize=16)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.tick_params(labelsize=14)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.minorticks_on()
    ax.tick_params(which='both',axis="both",direction="in")
    ax.set_xlim([10,1e+6])
    ax.set_ylim([1e-4,1])
    
    mydir = os.getcwd()
    # 1. Add Berne+2022 efficiency results
    os.chdir(BERNEPATH)
    Berne22_efficiency(G0min,G0max,ne_min,ne_max,T,ax,fig)
    os.chdir(mydir)
    
    Tielens2001_efficiency(ax)
    
    Wolfire2003_efficiency(T,ax)
    
    my_efficiency('small','Berne',G0min,G0max,ne_min,ne_max,T,ax,fig,do_colorbar=True)
    # my_efficiency('small','Tielens',G0min,G0max,ne_min,ne_max,T,ax,fig)
    
    my_efficiency('large','Berne',G0min,G0max,ne_min,ne_max,T,ax,fig)
    # my_efficiency('large','Tielens',G0min,G0max,ne_min,ne_max,T,ax,fig)
                
    ax.text(0.65, 0.9, r'$T=$ %i K'%int(T),
                        transform=ax.transAxes, fontsize=16,verticalalignment='top',
                        color='black')
    ax.legend(loc='lower left',fontsize=14,frameon=False)

    fig.subplots_adjust(top=0.98,bottom=0.13,left=0.13,right=0.95)
    fig.savefig('dust_heating_efficiency_'+str(int(T))+'K.pdf', format='pdf', dpi=300)

def save_results_to_txt(filename, results, x, y, z):
    with open(filename, 'w', newline='') as txtfile:
        ni,nj,nk = results.shape
        txtfile.write(f'{ni} {nj} {nk}\n')
        for i in range(0,ni):
            for j in range(0, nj):
                for k in range(0, nk):
                    txtfile.write(f'{np.log10(x[i])} {np.log10(y[j])} {np.log10(z[k])} {results[i,j,k]}\n')            

def read_data(filename):
    # Read the data from the file
    data = np.loadtxt(filename,skiprows=1)
    
    # Separate the data into log10 space components and values
    log_G0 = data[:, 0]
    log_ne = data[:, 1]
    log_T = data[:, 2]
    values = data[:, 3]

    # Get unique values for log_G0, log_ne, and log_T
    unique_log_G0 = np.unique(log_G0)
    unique_log_ne = np.unique(log_ne)
    unique_log_T = np.unique(log_T)

    # Reshape the values to reconstruct the matrix
    ni = len(unique_log_G0)
    nj = len(unique_log_ne)
    nk = len(unique_log_T)
    
    # Ensure the reshaping works properly
    values_matrix = values.reshape((ni, nj, nk))
    
    return unique_log_G0, unique_log_ne, unique_log_T, values_matrix

def interpolate_linear(log_G0, log_ne, log_T, values_matrix, G0, ne, T):
    log_G0_query = np.log10(G0)
    log_ne_query = np.log10(ne)
    log_T_query = np.log10(T)

    # Find indices for the interpolation
    i = np.searchsorted(log_G0, log_G0_query) - 1
    j = np.searchsorted(log_ne, log_ne_query) - 1
    k = np.searchsorted(log_T, log_T_query) - 1

    i = np.clip(i, 0, len(log_G0) - 2)
    j = np.clip(j, 0, len(log_ne) - 2)
    k = np.clip(k, 0, len(log_T) - 2)

    # Compute interpolation weights
    x1, x2 = log_G0[i], log_G0[i + 1]
    y1, y2 = log_ne[j], log_ne[j + 1]
    z1, z2 = log_T[k], log_T[k + 1]

    xd = (log_G0_query - x1) / (x2 - x1)
    yd = (log_ne_query - y1) / (y2 - y1)
    zd = (log_T_query - z1) / (z2 - z1)

    # Interpolate
    c00 = values_matrix[i, j, k] * (1 - xd) + values_matrix[i + 1, j, k] * xd
    c01 = values_matrix[i, j, k + 1] * (1 - xd) + values_matrix[i + 1, j, k + 1] * xd
    c10 = values_matrix[i, j + 1, k] * (1 - xd) + values_matrix[i + 1, j + 1, k] * xd
    c11 = values_matrix[i, j + 1, k + 1] * (1 - xd) + values_matrix[i + 1, j + 1, k + 1] * xd

    c0 = c00 * (1 - yd) + c10 * yd
    c1 = c01 * (1 - yd) + c11 * yd

    interpolated_value = c0 * (1 - zd) + c1 * zd

    return interpolated_value

def export_heating(G0_min,G0_max,T_min,T_max,ne_min,ne_max,n_G0,n_T,n_ne,model='Berne'):
    
    # 1. Prepare the parameter space
    G0_values = np.logspace(np.log10(G0_min),np.log10(G0_max),n_G0)
    T_values = np.logspace(np.log10(T_min),np.log10(T_max),n_T)
    ne_values = np.logspace(np.log10(ne_min),np.log10(ne_max),n_ne)
    
    # 2. Prepare arguments for parallel computation
    dist_small = LogNormal_Distribution(basic_a0[0],basic_amin[0],basic_amax[0],basic_sigma[0],basic_s[0])
    dist_small.Nc = 54
    wav,sigma_abs_anion = absorption_cross_section(dist_small,0)
    wav,sigma_abs_neu = absorption_cross_section(dist_small,0)
    wav,sigma_abs_cation = absorption_cross_section(dist_small,1)
    params_small = [(G0, T, ne, dist_small, model, wav, sigma_abs_anion, sigma_abs_neu, sigma_abs_cation)
          for G0 in G0_values for ne in ne_values for T in T_values]

    dist_large = LogNormal_Distribution(basic_a0[1],basic_amin[1],basic_amax[1],basic_sigma[1],basic_s[1])
    dist_large.Nc = 400
    wav,sigma_abs_anion = absorption_cross_section(dist_large,0)
    wav,sigma_abs_neu = absorption_cross_section(dist_large,0)
    wav,sigma_abs_cation = absorption_cross_section(dist_large,1)
    params_large = [(G0, T, ne, dist_large, model, wav, sigma_abs_anion, sigma_abs_neu, sigma_abs_cation)
          for G0 in G0_values for ne in ne_values for T in T_values]


    # 3. Perform parallel computation
    results_small = []
    num_cores = 20
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
            results_small = list(tqdm(executor.map(compute_heating_efficiency, params_small), total=n_G0*n_ne*n_T,
                                desc=f'    Computing efficiency for small PAHs Nc={dist_small.Nc}', unit=' steps'))
    
    results_large = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
            results_large = list(tqdm(executor.map(compute_heating_efficiency, params_large), total=n_G0*n_ne*n_T,
                                desc=f'    Computing efficiency for large PAHs Nc={dist_large.Nc}', unit=' steps'))
                    
            
    # 4. Initialize matrices to store the results
    efficiency_matrix_small = np.zeros((n_G0, n_ne, n_T))
    f_anion_matrix_small = np.zeros((n_G0, n_ne, n_T))
    f_neutral_matrix_small = np.zeros((n_G0, n_ne, n_T))
    f_1_matrix_small = np.zeros((n_G0, n_ne, n_T))
    f_2_matrix_small = np.zeros((n_G0, n_ne, n_T))
    
    efficiency_matrix_large = np.zeros((n_G0, n_ne, n_T))
    f_anion_matrix_large = np.zeros((n_G0, n_ne, n_T))
    f_neutral_matrix_large = np.zeros((n_G0, n_ne, n_T))
    f_1_matrix_large = np.zeros((n_G0, n_ne, n_T))
    f_2_matrix_large = np.zeros((n_G0, n_ne, n_T))
    
    # 5. Fill matrices with the computed results
    for result in results_small:
        G0, ne, T, f_anion, f_neutral, f_1, f_2, eff = result
        i = np.argmin(np.abs(G0_values - G0))
        j = np.argmin(np.abs(ne_values - ne))
        k = np.argmin(np.abs(T_values - T))
        efficiency_matrix_small[i, j, k] = eff
        f_anion_matrix_small[i, j, k] = f_anion
        f_neutral_matrix_small[i, j, k] = f_neutral
        f_1_matrix_small[i, j, k] = f_1
        f_2_matrix_small[i, j, k] = f_2
    for result in results_large:
        G0, ne, T, f_anion, f_neutral, f_1, f_2, eff = result
        i = np.argmin(np.abs(G0_values - G0))
        j = np.argmin(np.abs(ne_values - ne))
        k = np.argmin(np.abs(T_values - T))
        efficiency_matrix_large[i, j, k] = eff
        f_anion_matrix_large[i, j, k] = f_anion
        f_neutral_matrix_large[i, j, k] = f_neutral
        f_1_matrix_large[i, j, k] = f_1
        f_2_matrix_large[i, j, k] = f_2
        
    # 6. Save the results to the data files
    save_results_to_txt('./PAH_PEH_data/peh_efficiency_%s_pah_%.4f_micron.dat'%(model,dist_small.a0), efficiency_matrix_small, G0_values, ne_values, T_values)
    save_results_to_txt('./PAH_PEH_data/f_anion_%s_pah_%.4f_micron.dat'%(model,dist_small.a0), f_anion_matrix_small, G0_values, ne_values, T_values)
    save_results_to_txt('./PAH_PEH_data/f_neutral_%s_pah_%.4f_micron.dat'%(model,dist_small.a0), f_neutral_matrix_small, G0_values, ne_values, T_values)
    save_results_to_txt('./PAH_PEH_data/f_cation_%s_pah_%.4f_micron.dat'%(model,dist_small.a0), f_1_matrix_small, G0_values, ne_values, T_values)
    save_results_to_txt('./PAH_PEH_data/f_dication_%s_pah_%.4f_micron.dat'%(model,dist_small.a0), f_2_matrix_small, G0_values, ne_values, T_values)
    
    save_results_to_txt('./PAH_PEH_data/peh_efficiency_%s_pah_%.4f_micron.dat'%(model,dist_large.a0), efficiency_matrix_large, G0_values, ne_values, T_values)
    save_results_to_txt('./PAH_PEH_data/f_anion_%s_pah_%.4f_micron.dat'%(model,dist_large.a0), f_anion_matrix_large, G0_values, ne_values, T_values)
    save_results_to_txt('./PAH_PEH_data/f_neutral_%s_pah_%.4f_micron.dat'%(model,dist_large.a0), f_neutral_matrix_large, G0_values, ne_values, T_values)
    save_results_to_txt('./PAH_PEH_data/f_cation_%s_pah_%.4f_micron.dat'%(model,dist_large.a0), f_1_matrix_large, G0_values, ne_values, T_values)
    save_results_to_txt('./PAH_PEH_data/f_dication_%s_pah_%.4f_micron.dat'%(model,dist_large.a0), f_2_matrix_large, G0_values, ne_values, T_values)

def check_tables(G0_min,G0_max,ne_test,T_test,model):
    
    dist_small = LogNormal_Distribution(basic_a0[0],basic_amin[0],basic_amax[0],basic_sigma[0],basic_s[0])
    dist_small.Nc = 54

    dist_large = LogNormal_Distribution(basic_a0[1],basic_amin[1],basic_amax[1],basic_sigma[1],basic_s[1])
    dist_large.Nc = 400

    # 7. Test that everything is alright by obtaining a mock curve for a given ne and T
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(6,7), dpi=300, facecolor='w', edgecolor='k')

    axes[0].set_ylabel(r'$\epsilon_{\Gamma},\epsilon_{\rm PAH}$', fontsize=16)
    axes[1].set_ylabel(r'Charge fraction', fontsize=16)
    axes[2].set_ylabel(r'Ionised fraction', fontsize=16)
    axes[2].set_xlabel(r'$\gamma (G0\sqrt{T}/n_e)$ [K$^{1/2}$ cm$^{-3}$]',fontsize=16)
    axes[0].set_yscale('log')
    axes[0].set_xscale('log')
    axes[0].tick_params(labelsize=14)
    axes[0].xaxis.set_ticks_position('both')
    axes[0].yaxis.set_ticks_position('both')
    axes[0].minorticks_on()
    axes[0].tick_params(which='both',axis="both",direction="in")
    axes[0].set_xlim([10,1e+6])
    axes[0].set_ylim([3e-4,1])
    axes[1].set_xscale('log')
    axes[1].tick_params(labelsize=14)
    axes[1].xaxis.set_ticks_position('both')
    axes[1].yaxis.set_ticks_position('both')
    axes[1].minorticks_on()
    axes[1].tick_params(which='both',axis="both",direction="in")
    axes[1].set_xlim([10,1e+6])
    axes[1].set_ylim([0,1])
    axes[2].set_xscale('log')
    axes[2].tick_params(labelsize=14)
    axes[2].xaxis.set_ticks_position('both')
    axes[2].yaxis.set_ticks_position('both')
    axes[2].minorticks_on()
    axes[2].tick_params(which='both',axis="both",direction="in")
    axes[2].set_xlim([10,1e+6])
    axes[2].set_ylim([0,1])
    G0_test = np.logspace(np.log10(G0_min),np.log10(G0_max),200)
    interpolated_eff = np.zeros(200)
    interpolated_f_anion = np.zeros(200)
    interpolated_f_neutral = np.zeros(200)
    interpolated_f_cation = np.zeros(200)
    interpolated_f_dication = np.zeros(200)
    
    log_G0, log_ne, log_T, eff_matrix = read_data('./PAH_PEH_data/peh_efficiency_%s_pah_%.4f_micron.dat'%(model,dist_small.a0))
    log_G0, log_ne, log_T, f_anion_matrix = read_data('./PAH_PEH_data/f_anion_%s_pah_%.4f_micron.dat'%(model,dist_small.a0))
    log_G0, log_ne, log_T, f_neutral_matrix = read_data('./PAH_PEH_data/f_neutral_%s_pah_%.4f_micron.dat'%(model,dist_small.a0))
    log_G0, log_ne, log_T, f_cation_matrix = read_data('./PAH_PEH_data/f_cation_%s_pah_%.4f_micron.dat'%(model,dist_small.a0))
    log_G0, log_ne, log_T, f_dication_matrix = read_data('./PAH_PEH_data/f_dication_%s_pah_%.4f_micron.dat'%(model,dist_small.a0))
    
    for i in range(0, len(G0_test)):
        interpolated_eff[i] = interpolate_linear(log_G0, log_ne, log_T, eff_matrix, G0_test[i], ne_test, T_test)
        interpolated_f_anion[i] = interpolate_linear(log_G0, log_ne, log_T, f_anion_matrix, G0_test[i], ne_test, T_test)
        interpolated_f_neutral[i] = interpolate_linear(log_G0, log_ne, log_T, f_neutral_matrix, G0_test[i], ne_test, T_test)
        interpolated_f_cation[i] = interpolate_linear(log_G0, log_ne, log_T, f_cation_matrix, G0_test[i], ne_test, T_test)
        interpolated_f_dication[i] = interpolate_linear(log_G0, log_ne, log_T, f_dication_matrix, G0_test[i], ne_test, T_test)

        
    axes[0].plot(G0_test*np.sqrt(T_test)/ne_test, interpolated_eff,linestyle='-',color='k')
    axes[1].plot(G0_test*np.sqrt(T_test)/ne_test, interpolated_f_anion, label='$Z=-1$',linestyle='-',color='b')
    axes[1].plot(G0_test*np.sqrt(T_test)/ne_test, interpolated_f_neutral, label='$Z=0$',linestyle='-',color='orange')
    axes[1].plot(G0_test*np.sqrt(T_test)/ne_test, interpolated_f_cation, label='$Z=1$',linestyle='-',color='g')
    axes[1].plot(G0_test*np.sqrt(T_test)/ne_test, interpolated_f_dication, label='$Z=2$',linestyle='-',color='r')
    axes[2].plot(G0_test*np.sqrt(T_test)/ne_test, interpolated_f_cation+interpolated_f_dication,linestyle='-',color='k')
    
    log_G0, log_ne, log_T, eff_matrix = read_data('./PAH_PEH_data/peh_efficiency_%s_pah_%.4f_micron.dat'%(model,dist_large.a0))
    log_G0, log_ne, log_T, f_anion_matrix = read_data('./PAH_PEH_data/f_anion_%s_pah_%.4f_micron.dat'%(model,dist_large.a0))
    log_G0, log_ne, log_T, f_neutral_matrix = read_data('./PAH_PEH_data/f_neutral_%s_pah_%.4f_micron.dat'%(model,dist_large.a0))
    log_G0, log_ne, log_T, f_cation_matrix = read_data('./PAH_PEH_data/f_cation_%s_pah_%.4f_micron.dat'%(model,dist_large.a0))
    log_G0, log_ne, log_T, f_dication_matrix = read_data('./PAH_PEH_data/f_dication_%s_pah_%.4f_micron.dat'%(model,dist_large.a0))
    
    for i in range(0, len(G0_test)):
        interpolated_eff[i] = interpolate_linear(log_G0, log_ne, log_T, eff_matrix, G0_test[i], ne_test, T_test)
        interpolated_f_anion[i] = interpolate_linear(log_G0, log_ne, log_T, f_anion_matrix, G0_test[i], ne_test, T_test)
        interpolated_f_neutral[i] = interpolate_linear(log_G0, log_ne, log_T, f_neutral_matrix, G0_test[i], ne_test, T_test)
        interpolated_f_cation[i] = interpolate_linear(log_G0, log_ne, log_T, f_cation_matrix, G0_test[i], ne_test, T_test)
        interpolated_f_dication[i] = interpolate_linear(log_G0, log_ne, log_T, f_dication_matrix, G0_test[i], ne_test, T_test)

        
    axes[0].plot(G0_test*np.sqrt(T_test)/ne_test, interpolated_eff,linestyle='--',color='k')
    axes[1].plot(G0_test*np.sqrt(T_test)/ne_test, interpolated_f_anion,linestyle='--',color='b')
    axes[1].plot(G0_test*np.sqrt(T_test)/ne_test, interpolated_f_neutral,linestyle='--',color='orange')
    axes[1].plot(G0_test*np.sqrt(T_test)/ne_test, interpolated_f_cation,linestyle='--',color='g')
    axes[1].plot(G0_test*np.sqrt(T_test)/ne_test, interpolated_f_dication,linestyle='--',color='r')
    axes[2].plot(G0_test*np.sqrt(T_test)/ne_test, interpolated_f_cation+interpolated_f_dication,linestyle='--',color='k')

    
    axes[0].text(0.65, 0.9, r'$T=$ %.1e K'%float(T_test)+'\n'+\
                            r'$n_e=$ %.1e ${\rm cm}^{-3}$'%float(ne_test),
                        transform=axes[0].transAxes, fontsize=16,verticalalignment='top',
                        color='black')
    axes[1].legend(loc='lower left',fontsize=14,frameon=False)

    fig.subplots_adjust(top=0.98,bottom=0.1,left=0.13,right=0.95,hspace=0.0)
    fig.savefig('test_peh_table.png', format='png', dpi=300)
    