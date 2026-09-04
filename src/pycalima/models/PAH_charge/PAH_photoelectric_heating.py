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
import swiftascmaps
import matplotlib.pylab as pl
import matplotlib as mpl
import seaborn as sns
from unyt import nm,m,cm,eV,J,s,h,c,erg,K,kb
from models.dust_model import basic_a0,basic_amin,basic_amax,basic_sigma,basic_s,LogNormal_Distribution
from models.PAH_radiation.pah_oppacity import pah_efficiencies
from models.tools.radiation_fields import Draine_1978_isrf

os.environ["OMP_NUM_THREADS"] = "1"  # Set it to the desired number of threads
os.environ['OPENBLAS_NUM_THREADS'] = '1'

sns.set_theme(style="white")
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "Computer Modern Roman",
    "text.latex.preamble": r"\usepackage{xcolor}"
})

# Resolve paths relative to the repository root, independent of cwd.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CALIMA_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..', '..'))
_EXTERNAL_DATA_DIR = os.path.join(_CALIMA_ROOT, 'external_data')
_BERNE_2022_DIR = os.path.join(_CALIMA_ROOT, 'optical_props', 'berne_2022')

# BERNEPATH points to the Berne 2022 PAH cross-section dataset.
# It defaults to optical_props/berne_2022/ inside this repository.
# That directory must contain four subdirectories:
#   anions/, cations/, dications/, neutrals/
# Each holds one .txt file (energy [eV], cross-section [Mb]) per PAH species.
# The neutrals/ subdirectory is required when optical_model='Malloci' is used.
# See optical_props/berne_2022/README.md for the full file list.
# Override with the BERNEPATH environment variable to use a different location.
BERNEPATH = os.environ.get('BERNEPATH', _BERNE_2022_DIR)

# CONSTANTS
PAH_OPTICALS_DIR = os.path.join(_CALIMA_ROOT, 'optical_props', 'li_draine_2001')
pahneu_filepath = os.path.join(PAH_OPTICALS_DIR, 'PAHneu_30')
pahion_filepath = os.path.join(PAH_OPTICALS_DIR, 'PAHion_30')
epsilon_0 =  8.8541878188e-21 # Vacuum permittivity [F/nm]
e = 1.602176634e-19           # Elementary charge [C]
partition_coeff = 0.46        # Partition coefficient estimated from Bréchignac et al. 2014
J_to_erg = 1e7                # Conversion factor from J to erg
kB = 1.380649e-16           # Boltzmann constant [erg/K]
eV2erg = 1.602176634e-12  # Conversion factor from eV to erg
W2ergs = 1e7  # Conversion factor from W to erg/s

# FUNCTIONS
class PAHDataset:
    def __init__(self, file_path):
        file_path = file_path
        entries = []
        _parse_file()

    def _parse_file(self):
        with open(file_path, 'r') as file:
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

            entries.append(data)

    def get_entry_by_uid(self, uid):
        for entry in entries:
            if entry['UID'] == uid:
                return entry
        return None

    def get_all_entries(self):
        return entries

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
        IP = 6.0
    else:
        IP = 3.9 + e/(4.*np.pi*epsilon_0) * ((Z + 0.5) / a + (Z+2.)/a * (0.03/a)) 
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
        if photon_energy < IP:
            Y = 0.
        else:
            Y = 1.
    elif Z == 0:
        if photon_energy < IP:
            Y = 0.
        elif IP + 9.2 >= photon_energy:
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
    elif Z == 2:
        Y = 0.0
    
    return Y

def plot_ionisation_yield(Nc,Z):
    """Plot the ionisation yield as a function of the photon energy

    Args:
        Nc (int): number of carbon atoms in PAH molecule
        Z (int): PAH charge
    """    
    IP = ionisation_potential(Z,(Nc/468)**(1./3.))
    E = np.linspace(4,13.6,100)
    Y = np.array([ionisation_yield(Nc,Z,E[i],IP) for i in range(0,len(E))])
    plt.plot(E,Y)
    plt.xlabel(r'Photon energy [eV]')
    plt.ylabel(r'Yield')
    plt.savefig('ionisation_yield.png', format='png', dpi=300)

def plot_ionisation_potential(Z):
    """Plot the ionisation potential as a function of the number of carbon atoms
    in the PAH molecule.

    Args:
        Nc (int): number of carbon atoms in PAH molecule
        Z (int): PAH charge
    """
    Nc = np.linspace(30,1000,100)
    a = (Nc/468)**(1./3.)
    IP = ionisation_potential(Z,a)
    plt.plot(Nc,IP,'o')
    plt.xlabel(r'Number of carbon atoms $N_c$')
    plt.ylabel(r'Ionisation potential [eV]')
    plt.savefig('ionisation_potential.png', format='png', dpi=300)

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
    c = -1.12
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

def absorption_cross_section(distribution,Z, do_average=True):
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
        nwav,data,columns,name = pah_efficiencies(pahneu_filepath)
    else:
        nwav,data,columns,name = pah_efficiencies(pahion_filepath)
    
    # 2. Obtain the distribution averaged absorption cross section
    if do_average:
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
    else:
        # # Find the closest size to the distribution a0
        # a0 = distribution.a0
        # closest_size = min(data.keys(), key=lambda x: abs(float(x) - a0))
        # tmpdt = data[closest_size]
        # w = tmpdt[:,columns.index('w(micron)')]
        # C_abs_eff = tmpdt[:,columns.index('Q_abs')] * np.pi * float(closest_size)**2.
        a0 = distribution.a0
        if str(a0) in data.keys():
            closest_size = str(a0)
            tmpdt = data[closest_size]
            w = tmpdt[:,columns.index('w(micron)')]
            C_abs_eff = tmpdt[:,columns.index('Q_abs')] * np.pi * float(closest_size)**2.
        else:
            # Interpolate
            C_abs = np.zeros(len(data[list(data.keys())[0]][:,columns.index('Q_abs')]))
            for i in range(len(C_abs)):
                a = np.array([float(r) for r in data.keys()])
                Q_abs = np.array([d[i,columns.index('Q_abs')] for d in data.values()])
                C_abs[i] = 10.**np.interp(np.log10(a0),np.log10(a),np.log10(Q_abs)) * np.pi * a0**2
            w = data[list(data.keys())[0]][:,columns.index('w(micron)')]
            C_abs_eff = C_abs
    return w, C_abs_eff*1e-12

def absorption_cross_section_Berne(Nc):
    """Compute the distribution-averaged cross section for a
    given PAH molecule, considering whether or not the PAH is neutral
    or ionised. NOTE: anions are not allowed for this function.

    Args:
        distribution (LogNormal_Distribution): PAH molecule underlying log-normal distribution
        Z (int): charge of the PAH molecule

    Returns:
        (np.array,np.array): wavelength [microns], absorption cross section [m^2]
    """    
    
    '''==========================|building of the cross section|======================='''
    ''' derives a mean photoabsorption cross section of the molecule considered, in 3 size ranges'''
    
    ''' small size '''
    mb = 1e-18 #1Mb = 1e-18cm2, Mb for Megabarn (unit used to express the cross sectional area of nuclei)
    energy_negative_charged,cross_anion = np.loadtxt(os.path.join(BERNEPATH, 'anions', 'coronene_anion.txt'),unpack=True) #C24
    
    energy_negative_charged,crossa_1_case1 = np.loadtxt(os.path.join(BERNEPATH, 'anions', 'ovalene_anion.txt'),unpack=True) #C32
    energy_neutral,crossn_1_case1 = np.loadtxt(os.path.join(BERNEPATH, 'neutrals', 'ovalene_neutral.txt'),unpack=True) #C32
    energy_charged,crossc_1_case1 = np.loadtxt(os.path.join(BERNEPATH, 'cations', 'ovalene_cation.txt'),unpack=True) #C32
    energy_double_charged,crossdc_1_case1 = np.loadtxt(os.path.join(BERNEPATH, 'dications', 'ovalene_dication.txt'),unpack=True) #C32
    
    energy_negative_charged,crossa_2_case1 = np.loadtxt(os.path.join(BERNEPATH, 'anions', 'tetrabenzocoronene_anion.txt'),unpack=True) #C36
    energy_neutral,crossn_2_case1 = np.loadtxt(os.path.join(BERNEPATH, 'neutrals', 'tetrabenzocoronene_neutral.txt'),unpack=True) #C36
    energy_charged,crossc_2_case1 = np.loadtxt(os.path.join(BERNEPATH, 'cations', 'tetrabenzocoronene_cation.txt'),unpack=True) #C36
    energy_double_charged,crossdc_2_case1 = np.loadtxt(os.path.join(BERNEPATH, 'dications', 'tetrabenzocoronene_dication.txt'),unpack=True) #C36
    
    energy_negative_charged,crossa_3_case1 = np.loadtxt(os.path.join(BERNEPATH, 'anions', 'circumbiphenyl_anion.txt'),unpack=True) #C38
    energy_neutral,crossn_3_case1 = np.loadtxt(os.path.join(BERNEPATH, 'neutrals', 'circumbiphenyl_neutral.txt'),unpack=True) #C38
    energy_charged,crossc_3_case1 = np.loadtxt(os.path.join(BERNEPATH, 'cations', 'circumbiphenyl_cation.txt'),unpack=True) #C38
    energy_double_charged,crossdc_3_case1 = np.loadtxt(os.path.join(BERNEPATH, 'dications', 'circumbiphenyl_dication.txt'),unpack=True) #C38
    
    ''' medium size '''
    energy_negative_charged,crossa_1_case2 = np.loadtxt(os.path.join(BERNEPATH, 'anions', 'circumanthracene_anion.txt'),unpack=True) #C40
    energy_neutral,crossn_1_case2 = np.loadtxt(os.path.join(BERNEPATH, 'neutrals', 'circumanthracene_neutral.txt'),unpack=True) #C40
    energy_charged,crossc_1_case2 = np.loadtxt(os.path.join(BERNEPATH, 'cations', 'circumanthracene_cation.txt'),unpack=True) #C40
    energy_double_charged,crossdc_1_case2 = np.loadtxt(os.path.join(BERNEPATH, 'dications', 'circumanthracene_dication.txt'),unpack=True) #C40
    
    energy_negative_charged,crossa_2_case2 = np.loadtxt(os.path.join(BERNEPATH, 'anions', 'circumpyrene_anion.txt'),unpack=True) #C42
    energy_neutral,crossn_2_case2 = np.loadtxt(os.path.join(BERNEPATH, 'neutrals', 'circumpyrene_neutral.txt'),unpack=True) #C42
    energy_charged,crossc_2_case2 = np.loadtxt(os.path.join(BERNEPATH, 'cations', 'circumpyrene_cation.txt'),unpack=True) #C42
    energy_double_charged,crossdc_2_case2 = np.loadtxt(os.path.join(BERNEPATH, 'dications', 'circumpyrene_dication.txt'),unpack=True) #C42
    
    energy_negative_charged,crossa_3_case2 = np.loadtxt(os.path.join(BERNEPATH, 'anions', 'hexabenzocoronene_anion.txt'),unpack=True) #C42
    energy_neutral,crossn_3_case2 = np.loadtxt(os.path.join(BERNEPATH, 'neutrals', 'hexabenzocoronene_neutral.txt'),unpack=True) #C42
    energy_charged,crossc_3_case2 = np.loadtxt(os.path.join(BERNEPATH, 'cations', 'hexabenzocoronene_cation.txt'),unpack=True) #C42
    energy_double_charged,crossdc_3_case2 = np.loadtxt(os.path.join(BERNEPATH, 'dications', 'hexabenzocoronene_dication.txt'),unpack=True) #C42    
    
    ''' large size '''
    energy_negative_charged,crossa_1_case3 = np.loadtxt(os.path.join(BERNEPATH, 'anions', 'dicoronylene_anion.txt'),unpack=True) #C48
    energy_neutral,crossn_1_case3 = np.loadtxt(os.path.join(BERNEPATH, 'neutrals', 'dicoronylene_neutral.txt'),unpack=True) #C48
    energy_charged,crossc_1_case3 = np.loadtxt(os.path.join(BERNEPATH, 'cations', 'dicoronylene_cation.txt'),unpack=True) #C48
    energy_double_charged,crossdc_1_case3 = np.loadtxt(os.path.join(BERNEPATH, 'dications', 'dicoronylene_dication.txt'),unpack=True) #C48
    
    energy_negative_charged,crossa_2_case3 = np.loadtxt(os.path.join(BERNEPATH, 'anions', 'circumcoronene_anion.txt'),unpack=True) #C54
    energy_neutral,crossn_2_case3 = np.loadtxt(os.path.join(BERNEPATH, 'neutrals', 'circumcoronene_neutral.txt'),unpack=True) #C54
    energy_charged,crossc_2_case3 = np.loadtxt(os.path.join(BERNEPATH, 'cations', 'circumcoronene_cation.txt'),unpack=True) #C54
    energy_double_charged,crossdc_2_case3 = np.loadtxt(os.path.join(BERNEPATH, 'dications', 'circumcoronene_dication.txt'),unpack=True) #C54
    
    energy_negative_charged,crossa_3_case3 = np.loadtxt(os.path.join(BERNEPATH, 'anions', 'circumovalene_anion.txt'),unpack=True) #C66
    energy_neutral,crossn_3_case3 = np.loadtxt(os.path.join(BERNEPATH, 'neutrals', 'circumovalene_neutral.txt'),unpack=True) #C66
    energy_charged,crossc_3_case3 = np.loadtxt(os.path.join(BERNEPATH, 'cations', 'circumovalene_cation.txt'),unpack=True) #C66
    energy_double_charged,crossdc_3_case3 = np.loadtxt(os.path.join(BERNEPATH, 'dications', 'circumovalene_dication.txt'),unpack=True) #C66
    #for each cross section for each state of the molecule, we have an associated energy 
    
    energy_range  = np.where(energy_neutral<13.6)[0]

    pah_cross_a = ( ( (cross_anion/24)   +(crossa_1_case1/32)+(crossa_2_case1/36) +\
                            (crossa_3_case1/38)+(crossa_1_case2/40)+(crossa_2_case2/42) +\
                            (crossa_3_case2/42)+(crossa_1_case3/48)+(crossa_2_case3/54) +\
                            (crossa_3_case3/66)                                         )/10 ) * Nc
    pah_cross_n = ( ( (crossn_1_case1/32)+(crossn_2_case1/36)+(crossn_3_case1/38) +\
                            (crossn_1_case2/40)+(crossn_2_case2/42)+(crossn_3_case2/42) +\
                            (crossn_1_case3/48)+(crossn_2_case3/54)+(crossn_3_case3/66) )/9 ) * Nc
    pah_cross_c = ( ( (crossc_1_case1/32)+(crossc_2_case1/36)+(crossc_3_case1/38) +\
                            (crossc_1_case2/40)+(crossc_2_case2/42)+(crossc_3_case2/42) +\
                            (crossc_1_case3/48)+(crossc_2_case3/54)+(crossc_3_case3/66) )/9 ) * Nc
    pah_cross_dc = ( ((crossdc_1_case1/32)+(crossdc_2_case1/36)+(crossdc_3_case1/38) +\
                            (crossdc_1_case2/40)+(crossdc_2_case2/42)+(crossdc_3_case2/42) +\
                            (crossdc_1_case3/48)+(crossdc_2_case3/54)+(crossdc_3_case3/66) )/9 ) * Nc
    

    pah_cross_a = pah_cross_a[energy_range] * mb * 1e-4
    pah_cross_n = pah_cross_n[energy_range] * mb * 1e-4
    pah_cross_c = pah_cross_c[energy_range] * mb * 1e-4
    pah_cross_dc = pah_cross_dc[energy_range] * mb * 1e-4

    return energy_negative_charged[energy_range], energy_neutral[energy_range], \
        energy_charged[energy_range], energy_double_charged[energy_range], \
            pah_cross_a, pah_cross_n, pah_cross_c, pah_cross_dc
def compare_cross_sections(Nc):
    """Compare the cross sections from the two methods in an individual plot.

    Args:
        Nc (int): Number of carbon atoms in PAH molecule
    """
    # Compute cross sections using the two methods
    if Nc < 100:
        dist = LogNormal_Distribution(basic_a0[0], basic_amin[0], basic_amax[0], basic_sigma[0], basic_s[0])
    else:
        dist = LogNormal_Distribution(basic_a0[1], basic_amin[1], basic_amax[1], basic_sigma[1], basic_s[1])
    wav1, sigma_abs1 = absorption_cross_section(dist, 0, False)
    wav1, sigma_abs2 = absorption_cross_section(dist, 1, False)
    energy_anion,energy_neutral, energy_charged, energy_double_charged,\
        pah_cross_a, pah_cross_n, pah_cross_c, pah_cross_dc = absorption_cross_section_Berne(Nc)

    energy_range = (energy_anion>=0.05) & (energy_anion<=13.6)
    wav2 = 1.2398 / energy_anion[energy_range]
    # Plot the cross sections
    plt.figure(figsize=(10, 6))
    plt.xlim(0.0912,0.6)
    plt.plot(wav1, sigma_abs1, label='Method 1 (Neutral)', linestyle='-', color='blue')
    plt.plot(wav1, sigma_abs2, label='Method 1 (Ion)', linestyle='-', color='red')
    plt.plot(wav2, pah_cross_a[energy_range], label='Method 2 (Anion)', linestyle='-', color='orange')
    plt.plot(wav2, pah_cross_n[energy_range], label='Method 2 (Neutral)', linestyle='--', color='green')
    plt.plot(wav2, pah_cross_c[energy_range], label='Method 2 (Cation)', linestyle='-.', color='red')
    plt.plot(wav2, pah_cross_dc[energy_range], label='Method 2 (Dication)', linestyle=':', color='purple')
    plt.xlabel('Wavelength [microns]')
    plt.ylabel(r'Absorption Cross Section [m$^2$]')
    plt.legend()
    plt.title(f'Comparison of Absorption Cross Sections for Nc={Nc}')
    plt.grid(True, which="both", ls="--")
    plt.savefig(f'cross_section_comparison_Nc_{Nc}.png', format='png', dpi=300)
    

def ionisation_rate(IP,sigma_ion,I,E):
    """Compute the ionisation rate for a given PAH molecule bathed
    in the interstellar UV radiation field.

    Args:
        sigma_ion (np.array): Ionisation cross section [m^2]
        I (np.array): Local intensity of the UV radiation field [W / m^2 / eV]
        E (np.array): Photon energy [eV]

    Returns:
        np.float: Photo-ionisation rate [s-1]
    """
    mask = (E>=IP) & (E<=13.6)    
    k_pe = sigma_ion[mask] * I[mask] / E[mask] * 6.24150935e+18 # Convert [W] to [eV/s]
    k_pe = np.trapezoid(k_pe,E[mask])
    
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
    mask = (E<=13.6)
    P_rad = sigma_abs[mask] * I[mask]
    P_rad = np.trapezoid(P_rad,E[mask])
    
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
    mask = (E>=IP) & (E<=13.6)
    P_inj = (E[mask] - IP) * sigma_ion[mask] * I[mask] / E[mask]
    P_inj = np.trapezoid(P_inj,E[mask]-IP)
    
    return P_inj

def compute_integrated_absorbed_power_G0(a0, amin, amax, sigma, s, Nc, composition='nPAH'):

    # 1. Load the averaged PAH cross sections
    dist = LogNormal_Distribution(a0, amin, amax, sigma, s)
    dist.Nc = Nc
    
    if composition == 'nPAH':
        wav, sigma_abs = absorption_cross_section(dist, 0)
    elif composition == 'iPAH':
        wav, sigma_abs = absorption_cross_section(dist, 1)
    
    wav, sigma_abs = wav[::-1], sigma_abs[::-1]

    # 2. Load the ISRF
    I = Draine_1978_isrf(wav*1e3) / 1.7 # [#/cm^2/s/nm]
    E = h * c / (wav * 1e-4 * cm)
    I = I * E.to('erg').d * 1e7 # [erg/s/cm^2/cm]

    # 3. Define the Habing band (6-13.6 eV))
    habing_mask = (wav>0.0912) & (wav<0.2066)

    # 3. Compute the absorbed power
    P_rad = sigma_abs * 1e4 * I # [erg/s/cm]
    P_rad = np.trapezoid(P_rad[habing_mask],wav[habing_mask]*1e-4) # [erg/s]

    # 4. Compute the integrated ISRF radiation field in [erg/s/cm^2]
    G0 = np.trapezoid(I[habing_mask],wav[habing_mask]*1e-4) # [erg/s/cm^2]

    # 4. Cleaning print to screan the integrated absorbed power for the given PAH
    print(f'For a radiation field of G0={G0:.6e} erg/s/cm^2 in the Habing band')
    print(f'Integrated absorbed power for Nc={Nc}, composition={composition}: {P_rad:.6e} erg/s')


def plot_radiation_fields():
    from astropy.table import Table

    # Read the radiation field in erg s-1 cm-2 nm-1 sr-1
    wave_intensity = _load_isrf_data('Mathis')
    wavelength = wave_intensity['col1']  # in nm
    wavelength_intensity = wave_intensity['col2']  # in erg cm-2 s-1 nm-1 sr-1
    I_file_converted = 2. *np.pi * wavelength_intensity

    # Convert from [photons cm^-2 s^-1 nm^-1] to [W m^-2 eV^-1]
    I_draine =  Draine_1978_isrf(wavelength) 
    I_draine_converted = I_draine * (h * c / (wavelength * nm)).to('erg').d

    # Plot the two radiation fields
    plt.figure(figsize=(10, 6))
    plt.plot(wavelength, I_draine_converted, label='Draine 1978 ISRF')
    plt.plot(wavelength, I_file_converted, label='File-based ISRF', linestyle='--')
    plt.xlabel('Wavelength (nm)')
    plt.yscale('log')
    plt.ylabel(r'Intensity (erg cm-2 s-1 nm-1 sr-1)')
    plt.title('Comparison of Radiation Fields')
    plt.ylim(1e-6,1e-4)
    plt.legend()
    plt.grid(True)
    plt.savefig('radiation_fields_comparison.png', format='png', dpi=300)

def compute_heating_efficiency(args):
    from astropy.table import Table
    
    # 1. Unpack arguments
    G0,T,ne,dist,attach_model,wav,\
        sigma_abs_anion,sigma_abs_neu,\
            sigma_abs_cation, \
                sigma_abs_dication = args
    a0 = (dist.Nc/468)**(1./3.)*1e-3 #dist.a0
    Nc = dist.Nc
    # print('SIZE:',a0*1e3,Nc,(Nc/468)**(1./3.))
    
    # 3. Convert wavelength [micron] to photon energy [eV]
    E = 1.2398 / wav
    # Convert from [photons cm^-2 s^-1 nm^-1] to [W m^-2 eV^-1]
    wave_intensity = _load_isrf_data('Draine')
    wavelength = wave_intensity['col1']  # in nm
    wavelength_intensity = wave_intensity['col2']  # in erg cm-2 s-1 nm-1 sr-1
    I_file_converted = np.interp(wav*1e-3, wavelength, wavelength_intensity)
    I_Draine = I_file_converted / (h * c / (wav*1e-3 * nm)).to('erg').d
    #I_Draine = Draine_1978_isrf(wav*1e3)
    I = G0 * I_Draine /1.7 * cm**-2/s/nm
    F = I * E * eV
    f = F *nm/ (1e-9*m) * h * c / (E*eV)**2 * eV / (e*J)
    I = f.to('W/m**2/eV').d 
    
    # 4. Compute the e- detachment rate from the anion
    IP_anion = ionisation_potential(-1,a0*1e3)
    yield_anion = np.array([ionisation_yield(Nc,-1,E[i],IP_anion) for i in range(0,len(E))])
    mask = (E>=IP_anion) & (E <= 13.6)
    k_det = ionisation_rate(IP_anion,2*np.pi*yield_anion[mask]*sigma_abs_anion[mask],I[mask],E[mask])
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
    mask = (E>=IP_neutral) & (E <= 13.6)
    # print(IP_neutral,E[mask],I[mask],yield_neutral[mask],sigma_abs_neu[mask])
    k_pe_0 = ionisation_rate(IP_neutral,2*np.pi*yield_neutral[mask]*sigma_abs_neu[mask],I[mask],E[mask])
    # print('k_pe_0',k_pe_0)
    
    # 7. Recombination rate from Z=1 to Z=0
    if attach_model == 'Berne':
        k_rec_1 = recombination_rate_Spitzer(Nc,0,T)
    elif attach_model == 'Tielens':
        k_rec_1 = recombination_rate_Tielens21(Nc,T)
    # print('k_rec_1',k_rec_1)
    
    # 8. Recombination rate from Z=2 to Z=1
    if attach_model == 'Berne':
        k_rec_2 = recombination_rate_Spitzer(Nc,1,T)
    elif attach_model == 'Tielens':
        k_rec_2 = recombination_rate_Tielens21(Nc,T)
    # print('k_rec_2',k_rec_2)

    
    # 9. Ionisation rate of Z=1 to Z=2
    IP_cation = ionisation_potential(1,a0*1e3)
    yield_cation = np.array([ionisation_yield(Nc,1,E[i],IP_cation) for i in range(0,len(E))])
    mask = (E>=IP_cation) & (E <= 13.6)
    k_pe_1 = ionisation_rate(IP_cation,2*np.pi*yield_cation[mask]*sigma_abs_cation[mask],I[mask],E[mask])
    # print('k_pe_1',k_pe_1)

    
    # 10. Fraction of Z=-1
    f_anion = 1. / (1. + k_det / (k_att*ne) + \
                    k_det * k_pe_0 / (k_att*k_rec_1*ne**2.) + \
                    k_det * k_pe_0 * k_pe_1 / (k_att*k_rec_1*k_rec_2*ne**3.))
    
    # 11. Fraction of Z=0
    f_neutral = 1. / (1. + k_att*ne / k_det + k_pe_0 / (k_rec_1*ne) + \
                    k_pe_0 * k_pe_1 / (k_rec_1*k_rec_2*ne**2.))
    
    # 12. Fraction of Z=1
    f_1 = 1. / (1. + k_rec_1*ne / k_pe_0 + k_pe_1 / (k_rec_2*ne) + \
                k_att*k_rec_1*ne**2. / (k_det*k_pe_0))
    
    # 13. Fraction of Z=2
    f_2 = 1. / (1. + k_rec_2*ne / k_pe_1 + k_rec_1*k_rec_2*ne**2. / (k_pe_0*k_pe_1) + \
                k_att*k_rec_1*k_rec_2*ne**3./(k_det*k_pe_0*k_pe_0))
    
    # 14. Check that all fractions add up to 1
    f_tot = f_anion + f_neutral + f_1 + f_2
    f_anion, f_neutral, f_1, f_2 = f_anion/f_tot, f_neutral/f_tot, f_1/f_tot, f_2/f_tot
    
    # 15. Compute the total injected power
    Pinj_anion = power_injected(IP_anion,2*np.pi*yield_anion*sigma_abs_anion,I,E)
    Pinj_neutral = partition_coeff * power_injected(IP_neutral,2*np.pi*yield_neutral*sigma_abs_neu,I,E)
    Pinj_cation = partition_coeff * power_injected(IP_cation,2*np.pi*yield_cation*sigma_abs_cation,I,E)
    
    Pinj = f_anion * Pinj_anion + f_neutral * Pinj_neutral + f_1 * Pinj_cation
    
    # 16. Compute the total absorbed power
    Prad_anion = power_absorbed(2*np.pi*sigma_abs_anion,I,E)
    Prad_neutral = power_absorbed(2*np.pi*sigma_abs_neu,I,E)
    Prad_cation = power_absorbed(2*np.pi*sigma_abs_cation,I,E)
    Prad_dication = power_absorbed(2*np.pi*sigma_abs_dication,I,E)
    
    Prad = f_anion * Prad_anion + f_neutral * Prad_neutral + f_1 * Prad_cation + f_2 * Prad_dication
    
    # 17. Compute the heating efficiency as the ratio of injected to absorbed power
    eff = Pinj / Prad

    # print(f'For G0={G0:.6e} erg/s/cm^2, ne={ne:.6e} cm^-3, T={T} K')
    # print(f'Efficiency: {eff:.6e}')
    # print(f'Heating rate: {eff*Prad*(0.1/Nc)*2.7e-4:.6e} W/H')
    
    return G0, ne, T, f_anion,f_neutral,f_1,f_2,eff, Pinj, Prad


def peh_point_using_framework(G0_target, ne, T, Nc, a0, amin, amax, sigma, s, attach_model='Berne',
                              radiation_model='Draine', optical_model='Draine', debug=False):
    """
    Wrapper that uses the existing compute_peh_model framework to compute the
    PAH heating efficiency, charge fractions and powers for a single (G0, ne, T).

    It prepares the same inputs used by compute_peh_model and calls
    compute_heating_efficiency2 directly for a single point. If G0_target is
    provided, the radiation field is scaled so the computed G0 matches it.

    Returns a dict with keys: 'G0','ne','T','Zs','P','efficiency','Pinj','Prad'
    """
    from astropy.table import Table

    # create distribution from provided parameters
    dist = LogNormal_Distribution(a0, amin, amax, sigma, s)
    dist.Nc = Nc

    # optical cross sections
    if optical_model == 'Malloci':
        energy_negative_charged, energy_neutral, \
        energy_charged, energy_double_charged, \
        sigma_abs_anion, sigma_abs_neu, pah_cross_c, pah_cross_dc = absorption_cross_section_Berne(dist.Nc)
    else:
        wav1, sigma_abs_neutral = absorption_cross_section(dist, 0, True)
        wav1, sigma_abs_ion = absorption_cross_section(dist, 1, True)
        energy_negative_charged = 1.2398 / wav1
        energy_neutral = energy_negative_charged
        energy_charged = 1.2398 / wav1
        energy_double_charged = 1.2398 / wav1
        sigma_abs_anion = sigma_abs_neutral
        sigma_abs_neu = sigma_abs_neutral
        pah_cross_c = sigma_abs_ion
        pah_cross_dc = sigma_abs_ion

    anion_data = np.column_stack([energy_negative_charged,sigma_abs_anion])
    neutral_data = np.column_stack([energy_neutral,sigma_abs_neu])
    cation_data = np.column_stack([energy_charged,pah_cross_c])
    dication_data = np.column_stack([energy_double_charged,pah_cross_dc])

    # build radiation field
    if radiation_model == 'Draine':
        draine1978 = _load_isrf_data('Draine')
        rad_field = np.column_stack([draine1978['col1'],draine1978['col2']])
    elif radiation_model == 'Mathis':
        mathis1983 = _load_isrf_data('Mathis')
        rad_field = np.column_stack([mathis1983['col1'],mathis1983['col2']])
    else:
        # fallback to Draine
        draine1978 = _load_isrf_data('Draine')
        rad_field = np.column_stack([draine1978['col1'],draine1978['col2']])

    # optionally scale to requested G0_target
    if G0_target is not None:
        wavelength_intensity = rad_field[:,1]
        wavelength = rad_field[:,0]
        I_rad = wavelength_intensity / (h * c / (wavelength * nm)).to('erg').d
        E = 1.2398 / (wavelength[::-1]*1e-3)
        I = I_rad[::-1] * cm**-2/s/nm
        F = I * E * eV
        f = F *nm/ (1e-9*m) * h * c / (E*eV)**2 * eV / (e*J)
        I_conv = f.to('W/m**2/eV').d
        G0_current = np.trapezoid(2.*np.pi*I_conv[(E<=13.6)&(E>=5.17)],E[(E<=13.6)&(E>=5.17)]) / 1.68e-6
        if G0_current <= 0:
            raise RuntimeError('Could not compute G0 for the selected radiation field')
        scale = float(G0_target) / float(G0_current)
        rad_field[:,1] = rad_field[:,1] * scale

    # prepare args tuple and call the existing worker (use compute_heating_efficiency3 per request)
    args = (T, ne, dist, attach_model, rad_field, anion_data, neutral_data, cation_data, dication_data)
    out = compute_heating_efficiency3(args)
    # out is (G0, ne, T, f_anion, f_neutral, f_1, f_2, eff, Pinj, P_rec, Prad)
    G0_out, ne_out, T_out, f_anion, f_neutral, f_1, f_2, eff, Pinj, P_rec, Prad = out

    Zs = np.array([-1, 0, 1, 2])
    P = np.array([f_anion, f_neutral, f_1, f_2])

    res = {
        'G0': float(G0_out), 'ne': float(ne_out), 'T': float(T_out),
        'Zs': Zs, 'P': P,
        'efficiency': float(eff),
        'Pinj': float(Pinj), 'P_rec': float(P_rec), 'Prad': float(Prad)
    }

    if debug:
        print('[peh_point_using_framework] res:', res)

    return res

def compute_heating_efficiency2(args):
    
    # 1. Unpack arguments
    T,ne,dist,attach_model,radiation_field, \
        anion_data,neutral_data,cation_data, \
                dication_data = args
    a0 = (dist.Nc/468)**(1./3.) #dist.a0
    Nc = dist.Nc
    
    # 2. Get the radiation field and prepare the correct units
    wavelength_intensity = radiation_field[:,1] # in erg cm-2 s-1 nm-1 sr-1
    wavelength = radiation_field[:,0] # in nm


def _prepare_pah_heating_context(dist, attach_model, radiation_field,
                                 anion_data, neutral_data, cation_data, dication_data):
    """Build the invariant PAH photoelectric heating context for one table."""

    wavelength_intensity = radiation_field[:, 1]
    wavelength = radiation_field[:, 0]
    I_rad = wavelength_intensity / (h * c / (wavelength * nm)).to('erg').d
    E = 1.2398 / (wavelength[::-1] * 1e-3)
    I = I_rad[::-1] * cm**-2 / s / nm
    F = I * E * eV
    f = F * nm / (1e-9 * m) * h * c / (E * eV)**2 * eV / (e * J)
    I = f.to('W/m**2/eV').d

    habing_mask = (E <= 13.6) & (E >= 5.17)
    G0 = np.trapezoid(2. * np.pi * I[habing_mask], E[habing_mask]) / 1.68e-6

    sigma_abs_anion = np.interp(E, anion_data[:, 0], anion_data[:, 1])
    sigma_abs_neu = np.interp(E, neutral_data[:, 0], neutral_data[:, 1])
    sigma_abs_cation = np.interp(E, cation_data[:, 0], cation_data[:, 1])
    sigma_abs_dication = np.interp(E, dication_data[:, 0], dication_data[:, 1])

    Nc = dist.Nc
    a0 = (Nc / 468)**(1. / 3.)

    IP_anion = ionisation_potential(-1, a0)
    yield_anion = np.array([ionisation_yield(Nc, -1, E[i], IP_anion) for i in range(len(E))])
    mask_anion = (E >= IP_anion) & (E <= 13.6)
    anion_kernel = 2. * np.pi * yield_anion * sigma_abs_anion

    IP_neutral = ionisation_potential(0, a0)
    yield_neutral = np.array([ionisation_yield(Nc, 0, E[i], IP_neutral) for i in range(len(E))])
    mask_neutral = (E >= IP_neutral) & (E <= 13.6)
    neutral_kernel = 2. * np.pi * yield_neutral * sigma_abs_neu

    IP_cation = ionisation_potential(1, a0)
    yield_cation = np.array([ionisation_yield(Nc, 1, E[i], IP_cation) for i in range(len(E))])
    mask_cation = (E >= IP_cation) & (E <= 13.6)
    cation_kernel = 2. * np.pi * yield_cation * sigma_abs_cation

    Pinj_anion = power_injected(IP_anion, anion_kernel, I, E)
    Pinj_neutral = partition_coeff * power_injected(IP_neutral, neutral_kernel, I, E)
    Pinj_cation = partition_coeff * power_injected(IP_cation, cation_kernel, I, E)

    Prad_anion = power_absorbed(2. * np.pi * sigma_abs_anion, I, E)
    Prad_neutral = power_absorbed(2. * np.pi * sigma_abs_neu, I, E)
    Prad_cation = power_absorbed(2. * np.pi * sigma_abs_cation, I, E)
    Prad_dication = power_absorbed(2. * np.pi * sigma_abs_dication, I, E)

    return {
        'G0': float(G0),
        'E': E,
        'I': I,
        'Nc': Nc,
        'a0': a0,
        'IP_anion': IP_anion,
        'IP_neutral': IP_neutral,
        'IP_cation': IP_cation,
        'mask_anion': mask_anion,
        'mask_neutral': mask_neutral,
        'mask_cation': mask_cation,
        'anion_kernel': anion_kernel,
        'neutral_kernel': neutral_kernel,
        'cation_kernel': cation_kernel,
        'Pinj_anion': Pinj_anion,
        'Pinj_neutral': Pinj_neutral,
        'Pinj_cation': Pinj_cation,
        'Prad_anion': Prad_anion,
        'Prad_neutral': Prad_neutral,
        'Prad_cation': Prad_cation,
        'Prad_dication': Prad_dication,
    }


def _evaluate_pah_heating_context(context, T, ne, attach_model):
    """Evaluate one (T, ne) point using a cached PAH heating context."""

    E = context['E']
    I = context['I']
    Nc = context['Nc']

    if attach_model == 'Berne':
        k_att = attachment_rate_Carelli13(T)
        k_rec_1 = recombination_rate_Spitzer(Nc, 0, T)
        k_rec_2 = recombination_rate_Spitzer(Nc, 1, T)
    elif attach_model == 'Tielens':
        k_att = attachment_rate_Tielens05(Nc)
        k_rec_1 = recombination_rate_Tielens21(Nc, T)
        k_rec_2 = recombination_rate_Tielens21(Nc, T)
    else:
        raise ValueError("attach_model must be 'Berne' or 'Tielens'")

    k_det = ionisation_rate(
        context['IP_anion'],
        context['anion_kernel'][context['mask_anion']],
        I[context['mask_anion']],
        E[context['mask_anion']],
    )
    k_pe_0 = ionisation_rate(
        context['IP_neutral'],
        context['neutral_kernel'][context['mask_neutral']],
        I[context['mask_neutral']],
        E[context['mask_neutral']],
    )
    k_pe_1 = ionisation_rate(
        context['IP_cation'],
        context['cation_kernel'][context['mask_cation']],
        I[context['mask_cation']],
        E[context['mask_cation']],
    )

    f_anion = 1. / (1. + k_det / (k_att * ne) +
                    k_det * k_pe_0 / (k_att * k_rec_1 * ne**2.) +
                    k_det * k_pe_0 * k_pe_1 / (k_att * k_rec_1 * k_rec_2 * ne**3.))

    f_neutral = 1. / (1. + k_att * ne / k_det + k_pe_0 / (k_rec_1 * ne) +
                       k_pe_0 * k_pe_1 / (k_rec_1 * k_rec_2 * ne**2.))

    f_1 = 1. / (1. + k_rec_1 * ne / k_pe_0 + k_pe_1 / (k_rec_2 * ne) +
                k_att * k_rec_1 * ne**2. / (k_det * k_pe_0))

    f_2 = 1. / (1. + k_rec_2 * ne / k_pe_1 + k_rec_1 * k_rec_2 * ne**2. / (k_pe_0 * k_pe_1) +
                k_att * k_rec_1 * k_rec_2 * ne**3. / (k_det * k_pe_0 * k_pe_0))

    f_tot = f_anion + f_neutral + f_1 + f_2
    f_anion, f_neutral, f_1, f_2 = f_anion / f_tot, f_neutral / f_tot, f_1 / f_tot, f_2 / f_tot

    Pinj = W2ergs * (f_anion * context['Pinj_anion'] +
                     f_neutral * context['Pinj_neutral'] +
                     f_1 * context['Pinj_cation'])

    Prad = W2ergs * (f_anion * context['Prad_anion'] +
                     f_neutral * context['Prad_neutral'] +
                     f_1 * context['Prad_cation'] +
                     f_2 * context['Prad_dication'])

    eff = Pinj / Prad
    P_rec = k_att * ne * f_neutral * (3. / 2. * kB * T) + \
            k_rec_1 * ne * f_1 * (3. / 2. * kB * T) + \
            k_rec_2 * ne * f_2 * (3. / 2. * kB * T)

    return context['G0'], ne, T, f_anion, f_neutral, f_1, f_2, eff, Pinj, P_rec, Prad

def compute_heating_efficiency3(args):
    
    # 1. Unpack arguments
    T,ne,dist,attach_model,radiation_field, \
        anion_data,neutral_data,cation_data, \
                dication_data = args
    a0 = (dist.Nc/468)**(1./3.) #dist.a0
    Nc = dist.Nc
    
    # 2. Get the radiation field and prepare the correct units
    wavelength_intensity = radiation_field[:,1] # in erg cm-2 s-1 nm-1 sr-1
    wavelength = radiation_field[:,0] # in nm
    I_rad = wavelength_intensity / (h * c / (wavelength * nm)).to('erg').d
    E = 1.2398 / (wavelength[::-1]*1e-3)
    I = I_rad[::-1] * cm**-2/s/nm
    F = I * E * eV
    f = F *nm/ (1e-9*m) * h * c / (E*eV)**2 * eV / (e*J)
    I = f.to('W/m**2/eV').d 
    G0 = np.trapezoid(2.*np.pi*I[(E<=13.6)&(E>=5.17)],E[(E<=13.6)&(E>=5.17)]) / 1.68e-6

    # 3. Compute the interpolated cross sections
    sigma_abs_anion = np.interp(E,anion_data[:,0],anion_data[:,1])
    sigma_abs_neu = np.interp(E,neutral_data[:,0],neutral_data[:,1])
    sigma_abs_cation = np.interp(E,cation_data[:,0],cation_data[:,1])
    sigma_abs_dication = np.interp(E,dication_data[:,0],dication_data[:,1])
    
    # 4. Compute the e- detachment rate from the anion
    IP_anion = ionisation_potential(-1,a0)
    yield_anion = np.array([ionisation_yield(Nc,-1,E[i],IP_anion) for i in range(0,len(E))])
    mask = (E>=IP_anion) & (E <= 13.6)
    k_det = ionisation_rate(IP_anion,2*np.pi*yield_anion[mask]*sigma_abs_anion[mask],I[mask],E[mask])
    
    # 5. e- attachment to a neutral
    if attach_model == 'Berne':
        k_att = attachment_rate_Carelli13(T)
    elif attach_model == 'Tielens':
        k_att = attachment_rate_Tielens05(Nc)
    
    # 6. Ionisation rate of Z=0 to Z=1
    IP_neutral = ionisation_potential(0,a0)
    yield_neutral = np.array([ionisation_yield(Nc,0,E[i],IP_neutral) for i in range(0,len(E))])
    mask = (E>=IP_neutral) & (E <= 13.6)
    k_pe_0 = ionisation_rate(IP_neutral,2*np.pi*yield_neutral[mask]*sigma_abs_neu[mask],I[mask],E[mask])
    
    # 7. Recombination rate from Z=1 to Z=0
    if attach_model == 'Berne':
        k_rec_1 = recombination_rate_Spitzer(Nc,0,T)
    elif attach_model == 'Tielens':
        k_rec_1 = recombination_rate_Tielens21(Nc,T)
    
    # 8. Recombination rate from Z=2 to Z=1
    if attach_model == 'Berne':
        k_rec_2 = recombination_rate_Spitzer(Nc,1,T)
    elif attach_model == 'Tielens':
        k_rec_2 = recombination_rate_Tielens21(Nc,T)

    
    # 9. Ionisation rate of Z=1 to Z=2
    IP_cation = ionisation_potential(1,a0)
    yield_cation = np.array([ionisation_yield(Nc,1,E[i],IP_cation) for i in range(0,len(E))])
    mask = (E>=IP_cation) & (E <= 13.6)
    k_pe_1 = ionisation_rate(IP_cation,2*np.pi*yield_cation[mask]*sigma_abs_cation[mask],I[mask],E[mask])

    
    # 10. Fraction of Z=-1
    f_anion = 1. / (1. + k_det / (k_att*ne) + \
                    k_det * k_pe_0 / (k_att*k_rec_1*ne**2.) + \
                    k_det * k_pe_0 * k_pe_1 / (k_att*k_rec_1*k_rec_2*ne**3.))
    
    # 11. Fraction of Z=0
    f_neutral = 1. / (1. + k_att*ne / k_det + k_pe_0 / (k_rec_1*ne) + \
                    k_pe_0 * k_pe_1 / (k_rec_1*k_rec_2*ne**2.))
    
    # 12. Fraction of Z=1
    f_1 = 1. / (1. + k_rec_1*ne / k_pe_0 + k_pe_1 / (k_rec_2*ne) + \
                k_att*k_rec_1*ne**2. / (k_det*k_pe_0))
    
    # 13. Fraction of Z=2
    f_2 = 1. / (1. + k_rec_2*ne / k_pe_1 + k_rec_1*k_rec_2*ne**2. / (k_pe_0*k_pe_1) + \
                k_att*k_rec_1*k_rec_2*ne**3./(k_det*k_pe_0*k_pe_0))
    
    # 14. Check that all fractions add up to 1
    f_tot = f_anion + f_neutral + f_1 + f_2
    f_anion, f_neutral, f_1, f_2 = f_anion/f_tot, f_neutral/f_tot, f_1/f_tot, f_2/f_tot
    
    # 15. Compute the total injected power
    pcof = 1 #0.5 * E / (E - IP_anion) # Partition coefficient for the anion
    Pinj_anion = power_injected(IP_anion,pcof*2*np.pi*yield_anion*sigma_abs_anion,I,E)
    pcof = partition_coeff #0.5 * E / (E - IP_neutral) # Partition coefficient for the neutral
    Pinj_neutral = power_injected(IP_neutral,pcof*2*np.pi*yield_neutral*sigma_abs_neu,I,E)
    pcof = partition_coeff #0.5 * E / (E - IP_cation) # Partition coefficient for the cation
    Pinj_cation = power_injected(IP_cation,pcof*2*np.pi*yield_cation*sigma_abs_cation,I,E)

    Pinj = W2ergs*(f_anion * Pinj_anion + f_neutral * Pinj_neutral + f_1 * Pinj_cation) # Convert from W to erg/s
    
    # 16. Compute the total absorbed power
    Prad_anion = power_absorbed(2*np.pi*sigma_abs_anion,I,E)
    Prad_neutral = power_absorbed(2*np.pi*sigma_abs_neu,I,E)
    Prad_cation = power_absorbed(2*np.pi*sigma_abs_cation,I,E)
    Prad_dication = power_absorbed(2*np.pi*sigma_abs_dication,I,E)
    
    Prad = W2ergs*(f_anion * Prad_anion + f_neutral * Prad_neutral + f_1 * Prad_cation + f_2 * Prad_dication)
    
    # 17. Compute the heating efficiency as the ratio of injected to absorbed power
    eff = Pinj / Prad

    # 18. Compute the electron recombination cooling power
    P_rec = k_att * ne * f_neutral * (3./2.* kB * T) + \
            k_rec_1 * ne * f_1 * (3./2.* kB * T ) + \
            k_rec_2 * ne * f_2 * (3./2.* kB * T )

    return G0, ne, T, f_anion,f_neutral,f_1,f_2,eff, Pinj, P_rec, Prad
    
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
                    sigma_abs_anion,sigma_abs_neu,sigma_abs_cation,\
                    sigma_abs_cation
            args_list.append(args)
            gamma[k] = G0_list[k] * np.sqrt(T) / ne_list[j]
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
            results = list(tqdm(executor.map(compute_heating_efficiency, args_list), total=n_G0,
                                desc=f'    Computing efficiency for ne={ne_list[j]} cm^-3', unit=' steps'))
        
        bla1,bla2,bla3,f_anion,f_neutral,f_1,f_2,epsilon = zip(*results)
        ax2.plot(gamma,f_anion,color='k',linestyle=linestyle)
        ax2.plot(gamma,f_neutral,color='g',linestyle=linestyle)
        ax2.plot(gamma,f_1,color='b',linestyle=linestyle)
        ax2.plot(gamma,f_2,color='r',linestyle=linestyle)

        xs.append(gamma)
        ys.append(epsilon)
    lc = multiline(xs,ys,ne_list,cmap='jet',ax=ax,lw=1,linestyle=linestyle)   
    if do_colorbar:
        axcb = fig.colorbar(lc, orientation="vertical", pad=0.0)
        axcb.set_label(r'$n_e$ [cm$^{-3}$]',fontsize=16)
    
    dummy_lines = [ax2.plot([],[],color='k',linestyle=linestyle,label=r'$Z=-1$')[0],
                   ax2.plot([],[],color='g',linestyle=linestyle,label=r'$Z=0$')[0],
                   ax2.plot([],[],color='b',linestyle=linestyle,label=r'$Z=1$')[0],
                   ax2.plot([],[],color='r',linestyle=linestyle,label=r'$Z=2$')[0]]
    first_legend = ax2.legend(handles=dummy_lines, loc='best', frameon=False, fontsize=14)
    ax2.add_artist(first_legend)
    fig2.subplots_adjust(top=0.98,bottom=0.13,left=0.13,right=0.95)
    fig2.savefig(f'pah_charge_distribution_{attach_model}_{dist.Nc}C_{str(int(T))}K.pdf', format='pdf', dpi=300)


def blackbody_radiation(T, lambda_min, lambda_max, num_points=1000):
    """
    Computes the blackbody radiation intensity in erg cm⁻² s⁻¹ nm⁻¹ sr⁻¹ 
    for a given temperature over a range of wavelengths.

    Parameters:
        T (float or unyt_quantity): Temperature in Kelvin.
        lambda_min (float or unyt_quantity): Minimum wavelength in nm.
        lambda_max (float or unyt_quantity): Maximum wavelength in nm.
        num_points (int): Number of wavelength points (default: 1000).

    Returns:
        wavelengths (unyt_array): Wavelengths in nm.
        intensity (unyt_array): Spectral radiance in erg cm⁻² s⁻¹ nm⁻¹ sr⁻¹.
    """
    # Ensure inputs are unyt quantities
    T = T * K if not hasattr(T, "units") else T
    lambda_min = lambda_min * nm if not hasattr(lambda_min, "units") else lambda_min
    lambda_max = lambda_max * nm if not hasattr(lambda_max, "units") else lambda_max


    # Generate wavelength array in nm
    wavelengths = np.linspace(lambda_min.value, lambda_max.value, num_points) * nm

    # Compute the Planck function B_lambda(T) in erg cm⁻² s⁻¹ cm⁻¹ sr⁻¹
    exponent = (h * c) / (wavelengths * kb * T)
    intensity = (2 * h * c**2) / (wavelengths**5 * (np.exp(exponent) - 1))

    # # Create and save the plot
    # plt.figure(figsize=(8, 5))
    # plt.plot(wavelengths.value, intensity.to('erg/cm**2/s/nm').value, label=f"T = {T:.0f}", color='r')
    # plt.xlabel(f"Wavelength ({wavelengths.units})")
    # plt.yscale('log')
    # plt.ylabel(f"Intensity ({intensity.to('erg/cm**2/s/nm').units})")
    # plt.title("Blackbody Radiation Spectrum")
    # plt.legend()
    # plt.grid()

    # # Save the plot
    # save_path = f'BB_{int(T)}K_spectrum.png'
    # plt.savefig(save_path, dpi=300, bbox_inches="tight")
    # plt.close()  # Close the plot to free memory

    # print(f"Plot saved as {save_path}")

    return wavelengths, intensity

def my_efficiency2(pahtype,attach_model,radiation_model,optical_model,ne_min,ne_max,T,fig,axes,n_ne=100,single_ax=False):
    from astropy.table import Table
    ne_list = np.logspace(np.log10(ne_min),np.log10(ne_max),n_ne)
    if pahtype == 'small':
        dist = LogNormal_Distribution(basic_a0[0],basic_amin[0],basic_amax[0],basic_sigma[0],basic_s[0])
        dist.Nc = 54
        linestyle = '-'
        color = 'blue'
    elif pahtype == 'large':
        dist = LogNormal_Distribution(basic_a0[1],basic_amin[1],basic_amax[1],basic_sigma[1],basic_s[1])
        dist.Nc = 418
        linestyle = '--'
        color = 'royalblue'


    if attach_model == 'Berne':
        icol = 0
    elif attach_model == 'Tielens':
        icol = 1
    
    # 2. Compute the distribution-averaged absorption cross sections
    if optical_model == 'Malloci':
        energy_negative_charged, energy_neutral, \
        energy_charged, energy_double_charged, \
        sigma_abs_anion, sigma_abs_neu, pah_cross_c, pah_cross_dc = absorption_cross_section_Berne(dist.Nc)
    elif optical_model == 'Draine':
        wav1, sigma_abs_neutral = absorption_cross_section(dist, 0, True)
        wav1, sigma_abs_ion = absorption_cross_section(dist, 1, True)
        energy_range = (wav1 > 0.0912)
        wav1 = wav1[energy_range]
        sigma_abs_neutral = sigma_abs_neutral[energy_range]
        sigma_abs_ion = sigma_abs_ion[energy_range]
        energy_negative_charged = 1.2398 / wav1 # in eV
        energy_neutral = energy_negative_charged
        energy_charged = 1.2398 / wav1
        energy_double_charged = 1.2398 / wav1
        sigma_abs_anion = sigma_abs_neutral
        sigma_abs_neu = sigma_abs_neutral
        pah_cross_c = sigma_abs_ion
        pah_cross_dc = sigma_abs_ion
            
    anion_data = np.column_stack([energy_negative_charged,sigma_abs_anion])
    neutral_data = np.column_stack([energy_neutral,sigma_abs_neu])
    cation_data = np.column_stack([energy_charged,pah_cross_c])
    dication_data = np.column_stack([energy_double_charged,pah_cross_dc])

    num_cores = 20#min(os.cpu_count(),n_ne)
    args_list = []
    if radiation_model == 'Draine':
        draine1978 = _load_isrf_data('Draine')
        rad_field = np.column_stack([draine1978['col1'],draine1978['col2']])
        rad_name = 'Draine 1978'
        rad_color = '#BBD8B3'
        linestyle= ':'
    elif radiation_model == 'Habing':
        habing1968 = _load_isrf_data('Habing')
        rad_field = np.column_stack([habing1968['col1'],habing1968['col2']])
        rad_name = 'Habing 1968'
        rad_color = '#F3B61F'
        linestyle= ':'
    elif radiation_model == 'Mathis':
        mathis1983 = _load_isrf_data('Mathis')
        rad_field = np.column_stack([mathis1983['col1'],mathis1983['col2']])
        rad_name = 'Mathis+1983'
        rad_color = '#A29F15'
        linestyle= ':'
    elif radiation_model == 'HD200775':
        HD200775 = Table.read(os.path.join(_EXTERNAL_DATA_DIR, 'HD200775_RF.txt'), format='ascii')
        rad_field = np.column_stack([HD200775['col1'],HD200775['col2']])
        distance = 1. # at 20 pc
        star_radius = 10. # in solar radius units
        d_0 = 3.086e18*distance #1pc = 3.086e18cm
        r = star_radius*7e10 #radius of the star in cm
        ''' geometrical dilution '''
        rad_field[:,1] = rad_field[:,1]*(r/d_0)**2
        rad_name = r'HD200775'
        rad_color = '#510D0A'
        linestyle= '--'
    elif radiation_model[:2] == 'BB':
        T_star = float(radiation_model[2:])
        # Obtain the black body radiation field in units of erg cm-2 s-1 nm-1 sr-1 for the 
        # given temperature
        BB = blackbody_radiation(T_star, 23.0, 500, num_points=1000)
        rad_field = np.column_stack([BB[0].to('nm').d,BB[1].to('erg/cm**2/s/nm').d])
        distance = 20. # at 20 pc
        star_radius = 10. # in solar radius units
        d_0 = 3.086e18*distance #1pc = 3.086e18cm
        r = star_radius*7e10 #radius of the star in cm
        ''' geometrical dilution '''
        rad_field[:,1] = rad_field[:,1]*(r/d_0)**2
        rad_name = r'BB $T_{\star}=$'+str(int(T_star))+' K'
        rad_color = '#256EFF'
        linestyle= '--'
    elif radiation_model == 'O6V':
        O6V = Table.read(os.path.join(_EXTERNAL_DATA_DIR, 'kp00_40000'), format='ascii')
        rad_field = np.column_stack([O6V['col1'],O6V['col2']])
        distance = 20. # at 20 pc
        star_radius = 10. # in solar radius units
        d_0 = 3.086e18*distance #1pc = 3.086e18cm
        r = star_radius*7e10 #radius of the star in cm
        ''' geometrical dilution '''
        rad_field[:,1] = rad_field[:,1]*(r/d_0)**2
        rad_name = r'O6V'
        rad_color = '#C33149'
        linestyle= '-.'
    elif radiation_model == 'B0V':
        B0V = Table.read(os.path.join(_EXTERNAL_DATA_DIR, 'kp00_30000'), format='ascii')
        rad_field = np.column_stack([B0V['col1'],B0V['col2']])
        distance = 20. # at 20 pc
        star_radius = 10. # in solar radius units
        d_0 = 3.086e18*distance #1pc = 3.086e18cm
        r = star_radius*7e10 #radius of the star in cm
        ''' geometrical dilution '''
        rad_field[:,1] = rad_field[:,1]*(r/d_0)**2
        rad_name = r'B0V'
        rad_color = '#4B543B'
        linestyle= '-.'
    elif radiation_model == 'A0':
        A0 = Table.read(os.path.join(_EXTERNAL_DATA_DIR, 'kp00_10000'), format='ascii')
        rad_field = np.column_stack([A0['col1'],A0['col2']])
        distance = 2. # at 20 pc
        star_radius = 10. # in solar radius units
        d_0 = 3.086e18*distance #1pc = 3.086e18cm
        r = star_radius*7e10 #radius of the star in cm
        ''' geometrical dilution '''
        rad_field[:,1] = rad_field[:,1]*(r/d_0)**2
        rad_name = r'A0'
        rad_color = '#533A71'
        linestyle= '-.'
    elif radiation_model == 'BPASS_veryyoung_lowz':
        from models.tools.read_ramses_sed import read_sed_tables
        from unyt import Gyr
        metallicities, ages, wavelengths, SEDs = read_sed_tables("/Users/currodri/Documents/Dusty-PRISM/tests/lib/bpass_v221_cha300")
        fixed_age = 0.01 # 10 Myr
        fixed_metallicity = 0.0002 # 0.01 Zsun
        # Find the index of the closest age to the desired fixed age
        age_index = np.argmin(np.abs(ages - fixed_age * Gyr))
        # Find the index of the closest metallicity to the desired fixed metallicity
        metallicity_index = np.argmin(np.abs(metallicities - fixed_metallicity))
        bpass = SEDs[metallicity_index,age_index,:]/1000
        rad_field = np.column_stack([wavelengths.to('nm').d,bpass])
        rad_name = r'BPASS ($t=10$ Myr, $Z=0.01Z_{\odot}$)'
        rad_color = '#258EA6'
        linestyle= '-'
    elif radiation_model == 'BPASS_young_midz':
        from models.tools.read_ramses_sed import read_sed_tables
        from unyt import Gyr
        metallicities, ages, wavelengths, SEDs = read_sed_tables("/Users/currodri/Documents/Dusty-PRISM/tests/lib/bpass_v221_cha300")
        fixed_age = 0.1 # 0.1 Gyr
        fixed_metallicity = 0.01 # 0.5 Zsun
        # Find the index of the closest age to the desired fixed age
        age_index = np.argmin(np.abs(ages - fixed_age * Gyr))
        # Find the index of the closest metallicity to the desired fixed metallicity
        metallicity_index = np.argmin(np.abs(metallicities - fixed_metallicity))
        bpass = SEDs[metallicity_index,age_index,:]/100
        rad_field = np.column_stack([wavelengths.to('nm').d,bpass])
        rad_name = r'BPASS ($t=1$ Gyr, $Z=0.5Z_{\odot}$)'
        rad_color = '#F75590'
        linestyle= '-'
    elif radiation_model == 'BPASS_old_highz':
        from models.tools.read_ramses_sed import read_sed_tables
        from unyt import Gyr
        metallicities, ages, wavelengths, SEDs = read_sed_tables("/Users/currodri/Documents/Dusty-PRISM/tests/lib/bpass_v221_cha300")
        fixed_age = 1 # 1 Gyr
        fixed_metallicity = 0.02 # 1 Zsun
        # Find the index of the closest age to the desired fixed age
        age_index = np.argmin(np.abs(ages - fixed_age * Gyr))
        # Find the index of the closest metallicity to the desired fixed metallicity
        metallicity_index = np.argmin(np.abs(metallicities - fixed_metallicity))
        bpass = SEDs[metallicity_index,age_index,:]/100
        rad_field = np.column_stack([wavelengths.to('nm').d,bpass])
        rad_name = r'BPASS ($t=1$ Gyr, $Z=Z_{\odot}$)'
        rad_color = '#D84A05'
        linestyle= '-'


    context = _prepare_pah_heating_context(
        dist, attach_model, rad_field,
        anion_data, neutral_data, cation_data, dication_data,
    )
    results = [
        _evaluate_pah_heating_context(context, T, float(ne_value), attach_model)
        for ne_value in tqdm(ne_list, desc=f'    Computing efficiency for {rad_name} field', unit=' steps')
    ]

    G0,ne,Tgas,f_anion,f_neutral,f_1,f_2,epsilon,_,_,_ = zip(*results)
    gamma = G0 * np.sqrt(Tgas) / ne
    if single_ax:
        axes.plot(gamma,epsilon,color=rad_color,linewidth=2.5,linestyle=linestyle, 
                    label=rad_name)
    else:
        axes[1,icol].plot(gamma,f_anion,color='k',linewidth=2.5,linestyle=linestyle)
        axes[1,icol].plot(gamma,f_neutral,color='g',linewidth=2.5,linestyle=linestyle)
        axes[1,icol].plot(gamma,f_1,color='b',linewidth=2.5,linestyle=linestyle)
        axes[1,icol].plot(gamma,f_2,color='r',linewidth=2.5,linestyle=linestyle)

        axes[0,icol].plot(gamma,epsilon,color=color,linewidth=2.5,linestyle=linestyle, 
                        label=rf'$N_C=$ {dist.Nc} (This work)')

def compute_peh_model(Nc, a0, amin, amax, sigma, s, attach_model, radiation_model, optical_model, ne_min, ne_max, T, n_ne=100):
    from astropy.table import Table
    ne_list = np.logspace(np.log10(ne_min),np.log10(ne_max),n_ne)
    
    # create distribution from provided parameters
    dist = LogNormal_Distribution(a0, amin, amax, sigma, s)
    dist.Nc = Nc


    if attach_model == 'Berne':
        icol = 0
    elif attach_model == 'Tielens':
        icol = 1
    
    # 2. Compute the distribution-averaged absorption cross sections
    if optical_model == 'Malloci':
        energy_negative_charged, energy_neutral, \
        energy_charged, energy_double_charged, \
        sigma_abs_anion, sigma_abs_neu, pah_cross_c, pah_cross_dc = absorption_cross_section_Berne(dist.Nc)
    elif optical_model == 'Draine':
        wav1, sigma_abs_neutral = absorption_cross_section(dist, 0, False)
        wav1, sigma_abs_ion = absorption_cross_section(dist, 1, False)
        energy_range = (wav1 > 0.0912)
        wav1 = wav1[energy_range]
        sigma_abs_neutral = sigma_abs_neutral[energy_range]
        sigma_abs_ion = sigma_abs_ion[energy_range]
        energy_negative_charged = 1.2398 / wav1 # in eV
        energy_neutral = energy_negative_charged
        energy_charged = 1.2398 / wav1
        energy_double_charged = 1.2398 / wav1
        sigma_abs_anion = sigma_abs_neutral
        sigma_abs_neu = sigma_abs_neutral
        pah_cross_c = sigma_abs_ion
        pah_cross_dc = sigma_abs_ion
            
    anion_data = np.column_stack([energy_negative_charged,sigma_abs_anion])
    neutral_data = np.column_stack([energy_neutral,sigma_abs_neu])
    cation_data = np.column_stack([energy_charged,pah_cross_c])
    dication_data = np.column_stack([energy_double_charged,pah_cross_dc])

    if radiation_model == 'Draine':
        draine1978 = _load_isrf_data('Draine')
        rad_field = np.column_stack([draine1978['col1'],draine1978['col2']])
        rad_name = 'Draine 1978'
        rad_color = '#BBD8B3'
        linestyle= ':'
    elif radiation_model == 'Habing':
        habing1968 = _load_isrf_data('Habing')
        rad_field = np.column_stack([habing1968['col1'],habing1968['col2']])
        rad_name = 'Habing 1968'
        rad_color = '#F3B61F'
        linestyle= ':'
    elif radiation_model == 'Mathis':
        mathis1983 = _load_isrf_data('Mathis')
        rad_field = np.column_stack([mathis1983['col1'],mathis1983['col2']])
        rad_name = 'Mathis+1983'
        rad_color = '#A29F15'
        linestyle= ':'
    elif radiation_model == 'HD200775':
        HD200775 = Table.read(os.path.join(_EXTERNAL_DATA_DIR, 'HD200775_RF.txt'), format='ascii')
        rad_field = np.column_stack([HD200775['col1'],HD200775['col2']])
        distance = 1. # at 20 pc
        star_radius = 10. # in solar radius units
        d_0 = 3.086e18*distance #1pc = 3.086e18cm
        r = star_radius*7e10 #radius of the star in cm
        ''' geometrical dilution '''
        rad_field[:,1] = rad_field[:,1]*(r/d_0)**2
        rad_name = r'HD200775'
        rad_color = '#510D0A'
        linestyle= '--'
    elif radiation_model[:2] == 'BB':
        T_star = float(radiation_model[2:])
        # Obtain the black body radiation field in units of erg cm-2 s-1 nm-1 sr-1 for the 
        # given temperature
        BB = blackbody_radiation(T_star, 23.0, 500, num_points=1000)
        rad_field = np.column_stack([BB[0].to('nm').d,BB[1].to('erg/cm**2/s/nm').d])
        distance = 1. # at 20 pc
        star_radius = 10. # in solar radius units
        d_0 = 3.086e18*distance #1pc = 3.086e18cm
        r = star_radius*7e10 #radius of the star in cm
        ''' geometrical dilution '''
        rad_field[:,1] = rad_field[:,1]*(r/d_0)**2
        rad_name = r'BB $T_{\star}=$'+str(int(T_star))+' K'
        rad_color = '#256EFF'
        linestyle= '--'
    elif radiation_model == 'O6V':
        O6V = Table.read(os.path.join(_EXTERNAL_DATA_DIR, 'kp00_40000'), format='ascii')
        rad_field = np.column_stack([O6V['col1'],O6V['col2']])
        distance = 1. # at 20 pc
        star_radius = 10. # in solar radius units
        d_0 = 3.086e18*distance #1pc = 3.086e18cm
        r = star_radius*7e10 #radius of the star in cm
        ''' geometrical dilution '''
        rad_field[:,1] = rad_field[:,1]*(r/d_0)**2
        rad_name = r'O6V'
        rad_color = '#C33149'
        linestyle= '-.'
    elif radiation_model == 'B0V':
        B0V = Table.read(os.path.join(_EXTERNAL_DATA_DIR, 'kp00_30000'), format='ascii')
        rad_field = np.column_stack([B0V['col1'],B0V['col2']])
        distance = 1. # at 20 pc
        star_radius = 10. # in solar radius units
        d_0 = 3.086e18*distance #1pc = 3.086e18cm
        r = star_radius*7e10 #radius of the star in cm
        ''' geometrical dilution '''
        rad_field[:,1] = rad_field[:,1]*(r/d_0)**2
        rad_name = r'B0V'
        rad_color = '#4B543B'
        linestyle= '-.'
    elif radiation_model == 'A0':
        A0 = Table.read(os.path.join(_EXTERNAL_DATA_DIR, 'kp00_10000'), format='ascii')
        rad_field = np.column_stack([A0['col1'],A0['col2']])
        distance = 1. # at 20 pc
        star_radius = 10. # in solar radius units
        d_0 = 3.086e18*distance #1pc = 3.086e18cm
        r = star_radius*7e10 #radius of the star in cm
        ''' geometrical dilution '''
        rad_field[:,1] = rad_field[:,1]*(r/d_0)**2
        rad_name = r'A0'
        rad_color = '#533A71'
        linestyle= '-.'
    elif radiation_model == 'BPASS_veryyoung_lowz':
        from models.tools.read_ramses_sed import read_sed_tables
        from unyt import Gyr
        metallicities, ages, wavelengths, SEDs = read_sed_tables("/data80/currodri/test_crmhd_dust/G8/lib/bpass_v221_cha300")
        fixed_age = 0.01 # 10 Myr
        fixed_metallicity = 0.0002 # 0.01 Zsun
        # Find the index of the closest age to the desired fixed age
        age_index = np.argmin(np.abs(ages - fixed_age * Gyr))
        # Find the index of the closest metallicity to the desired fixed metallicity
        metallicity_index = np.argmin(np.abs(metallicities - fixed_metallicity))
        bpass = SEDs[metallicity_index,age_index,:]
        rad_field = np.column_stack([wavelengths.to('nm').d,bpass])
        rad_name = r'BPASS ($t=10$ Myr, $Z=0.01Z_{\odot}$)'
        rad_color = '#258EA6'
        linestyle= '-'
    elif radiation_model == 'BPASS_young_midz':
        from models.tools.read_ramses_sed import read_sed_tables
        from unyt import Gyr
        metallicities, ages, wavelengths, SEDs = read_sed_tables("/data80/currodri/test_crmhd_dust/G8/lib/bpass_v221_cha300")
        fixed_age = 0.1 # 0.1 Gyr
        fixed_metallicity = 0.01 # 0.5 Zsun
        # Find the index of the closest age to the desired fixed age
        age_index = np.argmin(np.abs(ages - fixed_age * Gyr))
        # Find the index of the closest metallicity to the desired fixed metallicity
        metallicity_index = np.argmin(np.abs(metallicities - fixed_metallicity))
        bpass = SEDs[metallicity_index,age_index,:]
        rad_field = np.column_stack([wavelengths.to('nm').d,bpass])
        rad_name = r'BPASS ($t=1$ Gyr, $Z=0.5Z_{\odot}$)'
        rad_color = '#F75590'
        linestyle= '-'
    elif radiation_model == 'BPASS_old_highz':
        from models.tools.read_ramses_sed import read_sed_tables
        from unyt import Gyr
        metallicities, ages, wavelengths, SEDs = read_sed_tables("/data80/currodri/test_crmhd_dust/G8/lib/bpass_v221_cha300")
        fixed_age = 1 # 1 Gyr
        fixed_metallicity = 0.02 # 1 Zsun
        # Find the index of the closest age to the desired fixed age
        age_index = np.argmin(np.abs(ages - fixed_age * Gyr))
        # Find the index of the closest metallicity to the desired fixed metallicity
        metallicity_index = np.argmin(np.abs(metallicities - fixed_metallicity))
        bpass = SEDs[metallicity_index,age_index,:]
        rad_field = np.column_stack([wavelengths.to('nm').d,bpass])
        rad_name = r'BPASS ($t=1$ Gyr, $Z=Z_{\odot}$)'
        rad_color = '#D84A05'
        linestyle= '-'


    context = _prepare_pah_heating_context(
        dist, attach_model, rad_field,
        anion_data, neutral_data, cation_data, dication_data,
    )
    results = [
        _evaluate_pah_heating_context(context, T, float(ne_value), attach_model)
        for ne_value in tqdm(ne_list, desc=f'    Computing efficiency for {rad_name} field', unit=' steps')
    ]

    G0,ne,Tgas,f_anion,f_neutral,f_1,f_2,epsilon,Pinj,Prec,Prad = zip(*results)
    gamma = G0 * np.sqrt(Tgas) / ne

    return dist.a0, gamma, epsilon, f_anion, f_neutral, f_1, f_2, Pinj, Prec,Prad

def Tielens2001_efficiency(ax):
    n_gamma = 20
    gamma = np.logspace(np.log10(1),np.log10(1e+15),n_gamma)
    eff = 0.06 / (1.0 + 7e-5*gamma)
    ax.plot(gamma,eff,color='k',linestyle=':',label='Tielens 2001',linewidth=2.5)
    
def Wolfire2003_efficiency(T,ax):
    n_gamma = 20
    gamma = np.logspace(np.log10(1),np.log10(1e+15),n_gamma)
    eff = 4.9e-2/(1.0+2.411e-3*gamma**0.73) + 3.7e-2*(T/1e4)**0.7/(1.0+1e-4*gamma)
    ax.plot(gamma,eff,color='k',linestyle='-.',label='Wolfire et al. 2003',linewidth=2.5)

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

def compare_eff_curves_all(T,ne_min,ne_max,n_ne=100):
    
    fig, axes = plt.subplots(2, 2, sharex=True, figsize=(10,8), dpi=300, facecolor='w', edgecolor='k')

    axes[0,0].set_ylabel(r'$\epsilon_{\Gamma},\epsilon_{\rm PAH}$', fontsize=16)
    axes[1,0].set_ylabel(r'Population fractions',fontsize=16)
    axes[1,0].set_xlabel(r'$\gamma (G0\sqrt{T}/n_e)$ [K$^{1/2}$ cm$^{-3}$]',fontsize=16)
    axes[1,1].set_xlabel(r'$\gamma (G0\sqrt{T}/n_e)$ [K$^{1/2}$ cm$^{-3}$]',fontsize=16)
    axes[0,0].set_yscale('log')
    axes[0,0].set_xscale('log')
    axes[0,1].set_yscale('log')
    axes[0,1].set_xscale('log')
    axes[0,0].tick_params(labelsize=14)
    axes[0,0].xaxis.set_ticks_position('both')
    axes[0,0].yaxis.set_ticks_position('both')
    axes[0,0].minorticks_on()
    axes[0,0].tick_params(which='both',axis="both",direction="in")
    axes[0,1].tick_params(labelsize=14)
    axes[0,1].xaxis.set_ticks_position('both')
    axes[0,1].yaxis.set_ticks_position('both')
    axes[0,1].minorticks_on()
    axes[0,1].tick_params(which='both',axis="both",direction="in")
    axes[1,0].tick_params(labelsize=14)
    axes[1,0].xaxis.set_ticks_position('both')
    axes[1,0].yaxis.set_ticks_position('both')
    axes[1,0].minorticks_on()
    axes[1,0].tick_params(which='both',axis="both",direction="in")
    axes[1,1].tick_params(labelsize=14)
    axes[1,1].xaxis.set_ticks_position('both')
    axes[1,1].yaxis.set_ticks_position('both')
    axes[1,1].minorticks_on()
    axes[1,1].tick_params(which='both',axis="both",direction="in")
    axes[1,1].set_yticklabels([])
    axes[0,1].set_yticklabels([])
    axes[0,0].set_xlim([10,1e+6])
    axes[0,0].set_ylim([1e-4,1])
    axes[0,1].set_xlim([10,1e+6])
    axes[0,1].set_ylim([1e-4,1])
    
    mydir = os.getcwd()
    
    Tielens2001_efficiency(axes[0,0])
    Tielens2001_efficiency(axes[0,1])
    
    Wolfire2003_efficiency(T,axes[0,0])
    Wolfire2003_efficiency(T,axes[0,1])
    
    my_efficiency2('small','Berne',ne_min,ne_max,T,fig,axes,n_ne=n_ne)
    my_efficiency2('small','Tielens',ne_min,ne_max,T,fig,axes,n_ne=n_ne)
    
    my_efficiency2('large','Berne',ne_min,ne_max,T,fig,axes,n_ne=n_ne)
    my_efficiency2('large','Tielens',ne_min,ne_max,T,fig,axes,n_ne=n_ne)
                
    axes[0,0].text(0.65, 0.9, r'$T=$ %i K'%int(T),
                        transform=axes[0,0].transAxes, fontsize=16,verticalalignment='top',
                        color='black')
    axes[0,0].legend(loc='lower left',fontsize=14,frameon=False)

    fig.subplots_adjust(top=0.97,bottom=0.07,left=0.07,right=0.98,wspace=0,hspace=0)
    fig.savefig('dust_heating_efficiency_'+str(int(T))+'K.pdf', format='pdf', dpi=300)

def compare_eff_curves_ISRF(T,ne_min,ne_max,n_ne=100,op_model='Malloci'):
    
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(6,5), dpi=300, facecolor='w', edgecolor='k')

    ax.set_ylabel(r'$\epsilon_{\rm PAH}$', fontsize=16)
    ax.set_xlabel(r'$\gamma (G0\sqrt{T}/n_e)$ [K$^{1/2}$ cm$^{-3}$]',fontsize=16)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.tick_params(labelsize=14)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.tick_params(which='both',axis="both",direction="in")
    ax.tick_params(labelsize=14)
    ax.minorticks_on()
    ax.set_xlim([10,1e+6])
    ax.set_ylim([1e-4,1])
        
    my_efficiency2('small','Berne','Draine',op_model,ne_min,ne_max,T,fig,ax,n_ne=n_ne,single_ax=True)
    my_efficiency2('small','Berne','Habing',op_model,ne_min,ne_max,T,fig,ax,n_ne=n_ne,single_ax=True)
    # my_efficiency2('small','Berne','Mathis',op_model,ne_min,ne_max,T,fig,ax,n_ne=n_ne,single_ax=True)
    # my_efficiency2('small','Berne','HD200775',op_model,ne_min,ne_max,T,fig,ax,n_ne=n_ne,single_ax=True)
    my_efficiency2('small','Berne','O6V',op_model,ne_min,ne_max,T,fig,ax,n_ne=n_ne,single_ax=True)
    my_efficiency2('small','Berne','B0V',op_model,ne_min,ne_max,T,fig,ax,n_ne=n_ne,single_ax=True)
    my_efficiency2('small','Berne','A0',op_model,ne_min,ne_max,T,fig,ax,n_ne=n_ne,single_ax=True)
    my_efficiency2('small','Berne','BPASS_veryyoung_lowz',op_model,ne_min,ne_max,T,fig,ax,n_ne=n_ne,single_ax=True)
    my_efficiency2('small','Berne','BPASS_young_midz',op_model,ne_min,ne_max,T,fig,ax,n_ne=n_ne,single_ax=True)
    my_efficiency2('small','Berne','BPASS_old_highz',op_model,ne_min,ne_max,T,fig,ax,n_ne=n_ne,single_ax=True)


    ax.text(0.65, 0.9, r'$T=10^{%i}$ K'%int(np.log10(T)),
                        transform=ax.transAxes, fontsize=16,verticalalignment='top',
                        color='black')
    ax.legend(loc='lower left',fontsize=12,frameon=False,ncol=2)

    fig.subplots_adjust(top=0.97,bottom=0.12,left=0.12,right=0.96,wspace=0,hspace=0)
    fig.savefig('comparison_ISRF_dust_heating_efficiency_'+str(int(T))+f'K_{op_model}.pdf', format='pdf', dpi=300)



def _load_isrf_data(model_name):
    """Load ISRF data from the repository external_data directory.
    
    Parameters
    ----------
    model_name : str
        Name of the ISRF model ('Draine', 'Mathis', 'Habing')
    
    Returns
    -------
    astropy.table.Table
        ISRF data table with columns col1 (wavelength, nm) and col2 (intensity)
    """
    from astropy.table import Table

    _filename_map = {
        'Draine': 'draine1978.txt',
        'Mathis': 'mathis1983.txt',
        'Habing': 'habing1968.txt',
    }
    if model_name not in _filename_map:
        raise ValueError(f'Unknown ISRF model: {model_name}')

    filepath = os.path.join(_EXTERNAL_DATA_DIR, _filename_map[model_name])
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Could not find ISRF file for model '{model_name}' at '{filepath}'."
        )
    return Table.read(filepath, format='ascii')

def compute_tables_ISRF(Nc, a0, amin, amax, sigma, s, T, ne_min, ne_max, n_ne=100, radiation_model='Draine',
                        op_model='Malloci', attach_model='Berne', output_dir=None, file_prefix=''):

    fig, axes = plt.subplots(1, 2, sharex=True, figsize=(10,4), dpi=300, facecolor='w', edgecolor='k')
    axes[0].text(-0.16, 0.30, r"$P_{\rm inj}$", color='r',
        transform=axes[0].transAxes, rotation=90,
        ha='center', va='center', fontsize=16)

    axes[0].text(-0.16, 0.36, ", ", color='black',
            transform=axes[0].transAxes, rotation=90,
            ha='center', va='center', fontsize=16)

    axes[0].text(-0.16, 0.43, r"$P_{\rm rec}$",
            color='b', transform=axes[0].transAxes,
            rotation=90, ha='center', va='center', fontsize=16)
    axes[0].text(-0.16, 0.61, r"[erg s$^{-1}$]", color='black',
            transform=axes[0].transAxes, rotation=90,
            ha='center', va='center', fontsize=16)
    axes[0].set_xlabel(r'$\gamma (G0\sqrt{T}/n_e)$ [K$^{1/2}$ cm$^{-3}$]',fontsize=16)
    axes[0].set_yscale('log')
    axes[0].set_xscale('log')
    axes[0].tick_params(labelsize=14)
    axes[0].xaxis.set_ticks_position('both')
    axes[0].yaxis.set_ticks_position('both')
    axes[0].tick_params(which='both',axis="both",direction="in")
    axes[0].tick_params(labelsize=14)
    axes[0].minorticks_on()
    axes[0].set_xlim([5,2e+6])
    # axes[0].set_ylim([3e-28,3e-25])

    axes[1].set_ylabel(r'Population fractions', fontsize=16)
    axes[1].set_xlabel(r'$\gamma (G0\sqrt{T}/n_e)$ [K$^{1/2}$ cm$^{-3}$]',fontsize=16)
    axes[1].yaxis.set_label_position("right")
    axes[1].yaxis.tick_right()
    axes[1].set_xscale('log')
    axes[1].tick_params(labelsize=14)
    axes[1].xaxis.set_ticks_position('both')
    axes[1].yaxis.set_ticks_position('both')
    axes[1].tick_params(which='both',axis="both",direction="in")
    axes[1].tick_params(labelsize=14)
    axes[1].minorticks_on()
    axes[1].set_ylim([1e-4,1])

    a0, gamma, epsilon, f_anion, f_neutral, f_1, f_2, Pinj, Prec, Prad = \
        compute_peh_model(Nc, a0, amin, amax, sigma, s, attach_model, radiation_model, op_model, ne_min, ne_max, T, n_ne=n_ne)
    Pinj = np.array(Pinj)
    Prec = np.array(Prec)
    Prad = np.array(Prad)
    efficiency = np.array(epsilon)
    axes[0].plot(gamma, Pinj, color='r', linewidth=2.5, linestyle='-')
    axes[0].plot(gamma, Prec, color='b', linewidth=2.5, linestyle='-')
    axes[1].plot(gamma, f_anion, color='k', linewidth=2.5, linestyle='-',
                 label=r'Anion')
    axes[1].plot(gamma, f_neutral, color='g', linewidth=2.5, linestyle='-',
                    label=r'Neutral')
    axes[1].plot(gamma, f_1, color='b', linewidth=2.5, linestyle='-',
                    label=r'Cation')
    axes[1].plot(gamma, f_2, color='r', linewidth=2.5, linestyle='-',
                    label=r'Dication')
    # Save results to file
    if output_dir is None:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    # Sort in gamma
    sort_index = np.argsort(gamma)
    gamma = np.array(gamma)[sort_index]
    efficiency = np.array(efficiency)[sort_index]
    Prad = Prad[sort_index]
    f_anion = np.array(f_anion)[sort_index]
    f_neutral = np.array(f_neutral)[sort_index]
    f_1 = np.array(f_1)[sort_index]
    f_2 = np.array(f_2)[sort_index]
    
    from models.grain_size_config import get_header_lines
    headers = get_header_lines(
        title=f"PAH Photoelectric heating efficiency (radiation_model={radiation_model}, op_model={op_model}, attach_model={attach_model})",
        script_name="models/PAH_charge/PAH_photoelectric_heating.py",
        bin_info=f"PAH Bin (Nc={Nc}, a0={a0:.4e} micron)",
        val_desc="Columns: log10(gamma), log10(efficiency), log10(Prad), f_anion, f_neutral, f_1, f_2",
        num_lines=6
    )
    header_str = '\n'.join(headers) + f'\n{n_ne}'

    np.savetxt(
        os.path.join(output_dir, f'peh_ISRF_{radiation_model}_{op_model}_{attach_model}_{file_prefix}.dat'),
        np.column_stack([np.log10(gamma), np.log10(efficiency), np.log10(Prad),
                        f_anion, f_neutral, f_1, f_2]),
        header=header_str,
        fmt='%14.6e %14.6e %14.6e %14.6e %14.6e %14.6e %14.6e',
        comments=''
    )
    
    # Add legend for this PAH bin
    axes[0].plot([], [], color='k', linestyle='-', label=rf'$N_{{\rm C}} = {Nc}$')

    axes[0].legend(loc='best',fontsize=14,frameon=False)
    axes[1].legend(loc='best',fontsize=14,frameon=False)

    # Sort in gamma before saving
    sorted_indices = np.argsort(gamma)
    gamma = np.array(gamma)[sorted_indices]
    efficiency = np.array(efficiency)[sorted_indices]
    Prad = np.array(Prad)[sorted_indices]
    f_anion = np.array(f_anion)[sorted_indices]
    f_neutral = np.array(f_neutral)[sorted_indices]
    f_1 = np.array(f_1)[sorted_indices]
    f_2 = np.array(f_2)[sorted_indices]

    # NOTE: results are saved in output_dir above. Avoid writing to legacy
    # hardcoded relative folders that may not exist in export workflows.

    fig.subplots_adjust(top=0.97,bottom=0.135,left=0.085,right=0.935,wspace=0,hspace=0)
    fig.savefig(os.path.join(output_dir, f'{file_prefix}peh_Pinj_ISRF_{radiation_model}_{op_model}_{attach_model}_{int(T)}K.pdf'), format='pdf', dpi=300)

def peh_vs_recombination_ISRF(G0,ne,Tmin,Tmax,nT=100,radiation_model='Draine',
                              optical_model='Malloci',attach_model='Berne'):
    
    from astropy.table import Table
    T_list = np.logspace(np.log10(Tmin*K),np.log10(Tmax*K),nT)
    
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(6,5), dpi=300, facecolor='w', edgecolor='k')

    ax.set_ylabel(r'$P_{\rm inj},P_{\rm rec}$', fontsize=16)
    ax.set_xlabel(r'$T$ [K]',fontsize=16)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.tick_params(labelsize=14)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.minorticks_on()
    ax.tick_params(which='both',axis="both",direction="in")
    ax.set_xlim([Tmin,Tmax])

    # 1. Load the radiation field
    if radiation_model == 'Draine':
        draine1978 = _load_isrf_data('Draine')
        rad_field = np.column_stack([draine1978['col1'],draine1978['col2']])
        rad_name = 'Draine 1978'
        rad_color = '#BBD8B3'
        linestyle= ':'
    elif radiation_model == 'Habing':
        habing1968 = _load_isrf_data('Habing')
        rad_field = np.column_stack([habing1968['col1'],habing1968['col2']])
        rad_name = 'Habing 1968'
        rad_color = '#F3B61F'
        linestyle= ':'
    elif radiation_model == 'Mathis':
        mathis1983 = _load_isrf_data('Mathis')
        rad_field = np.column_stack([mathis1983['col1'],mathis1983['col2']])
        rad_name = 'Mathis+1983'
        rad_color = '#A29F15'
        linestyle= ':'
    elif radiation_model == 'HD200775':
        HD200775 = Table.read(os.path.join(_EXTERNAL_DATA_DIR, 'HD200775_RF.txt'), format='ascii')
        rad_field = np.column_stack([HD200775['col1'],HD200775['col2']])
        distance = 1. # at 20 pc
        star_radius = 10. # in solar radius units
        d_0 = 3.086e18*distance #1pc = 3.086e18cm
        r = star_radius*7e10 #radius of the star in cm
        ''' geometrical dilution '''
        rad_field[:,1] = rad_field[:,1]*(r/d_0)**2
        rad_name = r'HD200775'
        rad_color = '#510D0A'
        linestyle= '--'
    elif radiation_model[:2] == 'BB':
        T_star = float(radiation_model[2:])
        # Obtain the black body radiation field in units of erg cm-2 s-1 nm-1 sr-1 for the 
        # given temperature
        BB = blackbody_radiation(T_star, 23.0, 500, num_points=1000)
        rad_field = np.column_stack([BB[0].to('nm').d,BB[1].to('erg/cm**2/s/nm').d])
        distance = 1. # at 20 pc
        star_radius = 10. # in solar radius units
        d_0 = 3.086e18*distance #1pc = 3.086e18cm
        r = star_radius*7e10 #radius of the star in cm
        ''' geometrical dilution '''
        rad_field[:,1] = rad_field[:,1]*(r/d_0)**2
        rad_name = r'BB $T_{\star}=$'+str(int(T_star))+' K'
        rad_color = '#256EFF'
        linestyle= '--'
    elif radiation_model == 'O6V':
        O6V = Table.read(os.path.join(_EXTERNAL_DATA_DIR, 'kp00_40000'), format='ascii')
        rad_field = np.column_stack([O6V['col1'],O6V['col2']])
        distance = 1. # at 20 pc
        star_radius = 10. # in solar radius units
        d_0 = 3.086e18*distance #1pc = 3.086e18cm
        r = star_radius*7e10 #radius of the star in cm
        ''' geometrical dilution '''
        rad_field[:,1] = rad_field[:,1]*(r/d_0)**2
        rad_name = r'O6V'
        rad_color = '#C33149'
        linestyle= '-.'
    elif radiation_model == 'B0V':
        B0V = Table.read(os.path.join(_EXTERNAL_DATA_DIR, 'kp00_30000'), format='ascii')
        rad_field = np.column_stack([B0V['col1'],B0V['col2']])
        distance = 1. # at 20 pc
        star_radius = 10. # in solar radius units
        d_0 = 3.086e18*distance #1pc = 3.086e18cm
        r = star_radius*7e10 #radius of the star in cm
        ''' geometrical dilution '''
        rad_field[:,1] = rad_field[:,1]*(r/d_0)**2
        rad_name = r'B0V'
        rad_color = '#4B543B'
        linestyle= '-.'
    elif radiation_model == 'A0':
        A0 = Table.read(os.path.join(_EXTERNAL_DATA_DIR, 'kp00_10000'), format='ascii')
        rad_field = np.column_stack([A0['col1'],A0['col2']])
        distance = 1. # at 20 pc
        star_radius = 10. # in solar radius units
        d_0 = 3.086e18*distance #1pc = 3.086e18cm
        r = star_radius*7e10 #radius of the star in cm
        ''' geometrical dilution '''
        rad_field[:,1] = rad_field[:,1]*(r/d_0)**2
        rad_name = r'A0'
        rad_color = '#533A71'
        linestyle= '-.'
    elif radiation_model == 'BPASS_veryyoung_lowz':
        from models.tools.read_ramses_sed import read_sed_tables
        from unyt import Gyr
        metallicities, ages, wavelengths, SEDs = read_sed_tables("/data80/currodri/test_crmhd_dust/G8/lib/bpass_v221_cha300")
        fixed_age = 0.01 # 10 Myr
        fixed_metallicity = 0.0002 # 0.01 Zsun
        # Find the index of the closest age to the desired fixed age
        age_index = np.argmin(np.abs(ages - fixed_age * Gyr))
        # Find the index of the closest metallicity to the desired fixed metallicity
        metallicity_index = np.argmin(np.abs(metallicities - fixed_metallicity))
        bpass = SEDs[metallicity_index,age_index,:]
        rad_field = np.column_stack([wavelengths.to('nm').d,bpass])
        rad_name = r'BPASS ($t=10$ Myr, $Z=0.01Z_{\odot}$)'
        rad_color = '#258EA6'
        linestyle= '-'
    elif radiation_model == 'BPASS_young_midz':
        from models.tools.read_ramses_sed import read_sed_tables
        from unyt import Gyr
        metallicities, ages, wavelengths, SEDs = read_sed_tables("/data80/currodri/test_crmhd_dust/G8/lib/bpass_v221_cha300")
        fixed_age = 0.1 # 0.1 Gyr
        fixed_metallicity = 0.01 # 0.5 Zsun
        # Find the index of the closest age to the desired fixed age
        age_index = np.argmin(np.abs(ages - fixed_age * Gyr))
        # Find the index of the closest metallicity to the desired fixed metallicity
        metallicity_index = np.argmin(np.abs(metallicities - fixed_metallicity))
        bpass = SEDs[metallicity_index,age_index,:]
        rad_field = np.column_stack([wavelengths.to('nm').d,bpass])
        rad_name = r'BPASS ($t=1$ Gyr, $Z=0.5Z_{\odot}$)'
        rad_color = '#F75590'
        linestyle= '-'
    elif radiation_model == 'BPASS_old_highz':
        from models.tools.read_ramses_sed import read_sed_tables
        from unyt import Gyr
        metallicities, ages, wavelengths, SEDs = read_sed_tables("/data80/currodri/test_crmhd_dust/G8/lib/bpass_v221_cha300")
        fixed_age = 1 # 1 Gyr
        fixed_metallicity = 0.02 # 1 Zsun
        # Find the index of the closest age to the desired fixed age
        age_index = np.argmin(np.abs(ages - fixed_age * Gyr))
        # Find the index of the closest metallicity to the desired fixed metallicity
        metallicity_index = np.argmin(np.abs(metallicities - fixed_metallicity))
        bpass = SEDs[metallicity_index,age_index,:]
        rad_field = np.column_stack([wavelengths.to('nm').d,bpass])
        rad_name = r'BPASS ($t=1$ Gyr, $Z=Z_{\odot}$)'
        rad_color = '#D84A05'
        linestyle= '-'

    rad_field[:,1] = rad_field[:,1] * G0  # Scale the radiation field by G0

    # 2. Small PAHs
    dist = LogNormal_Distribution(basic_a0[0],basic_amin[0],basic_amax[0],basic_sigma[0],basic_s[0])
    dist.Nc = 54
    linestyle = '-'
    color = 'blue'

    if optical_model == 'Malloci':
        energy_negative_charged, energy_neutral, \
        energy_charged, energy_double_charged, \
        sigma_abs_anion, sigma_abs_neu, pah_cross_c, pah_cross_dc = absorption_cross_section_Berne(dist.Nc)
    elif optical_model == 'Draine':
        wav1, sigma_abs_neutral = absorption_cross_section(dist, 0, True)
        wav1, sigma_abs_ion = absorption_cross_section(dist, 1, True)
        energy_range = (wav1 > 0.0912)
        wav1 = wav1[energy_range]
        sigma_abs_neutral = sigma_abs_neutral[energy_range]
        sigma_abs_ion = sigma_abs_ion[energy_range]
        energy_negative_charged = 1.2398 / wav1 # in eV
        energy_neutral = energy_negative_charged
        energy_charged = 1.2398 / wav1
        energy_double_charged = 1.2398 / wav1
        sigma_abs_anion = sigma_abs_neutral
        sigma_abs_neu = sigma_abs_neutral
        pah_cross_c = sigma_abs_ion
        pah_cross_dc = sigma_abs_ion

    anion_data = np.column_stack([energy_negative_charged,sigma_abs_anion])
    neutral_data = np.column_stack([energy_neutral,sigma_abs_neu])
    cation_data = np.column_stack([energy_charged,pah_cross_c])
    dication_data = np.column_stack([energy_double_charged,pah_cross_dc])

    args_list = []
    context = _prepare_pah_heating_context(
        dist, attach_model, rad_field,
        anion_data, neutral_data, cation_data, dication_data,
    )
    results = [
        _evaluate_pah_heating_context(context, float(T_value), ne, attach_model)
        for T_value in tqdm(T_list, desc=f'    Computing efficiency for {rad_name} field', unit=' steps')
    ]

    G0,ne_tab,Tgas,f_anion,f_neutral,f_1,f_2,epsilon,Pinj,Prec,Prad = zip(*results)
    gamma = G0 * np.sqrt(Tgas) / ne_tab

    ax.plot(T_list,Pinj,color='r',linewidth=2.5,linestyle=linestyle,
             label=rf'$N_C=$ {dist.Nc}')
    ax.plot(T_list,Prec,color='b',linewidth=2.5,linestyle=linestyle)

    # 3. Large PAHs
    dist = LogNormal_Distribution(basic_a0[1],basic_amin[1],basic_amax[1],basic_sigma[1],basic_s[1])
    dist.Nc = 418
    linestyle = '--'
    color = 'royalblue'

    if optical_model == 'Malloci':
        energy_negative_charged, energy_neutral, \
        energy_charged, energy_double_charged, \
        sigma_abs_anion, sigma_abs_neu, pah_cross_c, pah_cross_dc = absorption_cross_section_Berne(dist.Nc)
    elif optical_model == 'Draine':
        wav1, sigma_abs_neutral = absorption_cross_section(dist, 0, True)
        wav1, sigma_abs_ion = absorption_cross_section(dist, 1, True)
        energy_range = (wav1 > 0.0912)
        wav1 = wav1[energy_range]
        sigma_abs_neutral = sigma_abs_neutral[energy_range]
        sigma_abs_ion = sigma_abs_ion[energy_range]
        energy_negative_charged = 1.2398 / wav1 # in eV
        energy_neutral = energy_negative_charged
        energy_charged = 1.2398 / wav1
        energy_double_charged = 1.2398 / wav1
        sigma_abs_anion = sigma_abs_neutral
        sigma_abs_neu = sigma_abs_neutral
        pah_cross_c = sigma_abs_ion
        pah_cross_dc = sigma_abs_ion

    anion_data = np.column_stack([energy_negative_charged,sigma_abs_anion])
    neutral_data = np.column_stack([energy_neutral,sigma_abs_neu])
    cation_data = np.column_stack([energy_charged,pah_cross_c])
    dication_data = np.column_stack([energy_double_charged,pah_cross_dc])

    args_list = []
    context = _prepare_pah_heating_context(
        dist, attach_model, rad_field,
        anion_data, neutral_data, cation_data, dication_data,
    )
    results = [
        _evaluate_pah_heating_context(context, float(T_value), ne, attach_model)
        for T_value in tqdm(T_list, desc=f'    Computing efficiency for {rad_name} field', unit=' steps')
    ]

    G0,ne,Tgas,f_anion,f_neutral,f_1,f_2,epsilon,Pinj,Prec,Prad = zip(*results)
    gamma = G0 * np.sqrt(Tgas) / ne_tab

    ax.plot(T_list,Pinj,color='r',linewidth=2.5,linestyle=linestyle,
             label=rf'$N_C=$ {dist.Nc}')
    ax.plot(T_list,Prec,color='b',linewidth=2.5,linestyle=linestyle)

    # 4. Add labelling and legend
    ax.text(0.65, 0.9, r'$n_e=$ %i cm$^{-3}$'%int(ne),
                        transform=ax.transAxes, fontsize=16,
                        verticalalignment='top', color='black')
    ax.legend(loc='best', fontsize=14, frameon=False)

    fig.subplots_adjust(top=0.97, bottom=0.12, left=0.12, right=0.96, wspace=0, hspace=0)
    fig.savefig(f'peh_vs_recombination_ISRF_{radiation_model}_{optical_model}_{attach_model}_ne{int(ne)}.pdf', format='pdf', dpi=300)
