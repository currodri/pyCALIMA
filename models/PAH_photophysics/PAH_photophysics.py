"""
PAH PHOTOPHYSICS MODELLING

The functions, data and models included within this Python file are used
for the computation of PAH evolution as they interact with radiation.
The photophysics of PAHs is a complex theoretical framework that goes
from the RRKM and 2nu-RRKM DFT calculations for the PAH density of states
to the chemistry of aromatic and aliphatic bonds.

By: F. Rodriguez Montero (currodri@gmail.com)

"""

# Import libraries
import numpy as np
from models.tools.radiation_fields import Draine_1978_isrf
import pandas as pd
import os
from tqdm import tqdm
import concurrent.futures
import time
from models.PAH_radiation.pah_oppacity import pah_efficiencies
from models.dust_model import LogNormal_Distribution
from models.grain_size_config import get_bins, get_lognormal_parameters
from models.tools.utils import Nc_from_size

from unyt import nm,m,cm,eV,J,s,h,c


# Set OMP_NUM_THREADS to limit the number of threads used by OpenBLAS
os.environ["OMP_NUM_THREADS"] = "1"  # Set it to the desired number of threads
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# Constants
current_dir = os.path.dirname(os.path.abspath(__file__))
_CALIMA_ROOT = os.path.abspath(os.path.join(current_dir, '..', '..'))
EXTERNAL_DATA_DIR = os.path.join(_CALIMA_ROOT, 'external_data')


def _external_data_path(filename):
    if os.path.isabs(filename):
        return filename
    return os.path.join(EXTERNAL_DATA_DIR, filename)


def _get_pah_bin(pah_bin_id=None, pah_bin_rank=0):
    pah_bins = sorted(
        get_bins(is_pah=True),
        key=lambda b: (b['bin_rank'], b.get('index', 0)),
    )
    if not pah_bins:
        raise RuntimeError('No PAH bins found in grain-size configuration.')

    if pah_bin_id is not None:
        for b in pah_bins:
            if b['id'] == pah_bin_id:
                return b
        available = ', '.join([b['id'] for b in pah_bins])
        raise KeyError(f"Unknown PAH bin '{pah_bin_id}'. Available: {available}")

    for b in pah_bins:
        if int(b['bin_rank']) == int(pah_bin_rank):
            return b
    return pah_bins[0]


def _build_pah_distribution(pah_bin_id=None, pah_bin_rank=0):
    pah_bin = _get_pah_bin(pah_bin_id=pah_bin_id, pah_bin_rank=pah_bin_rank)
    p = get_lognormal_parameters(pah_bin['id'], model_name='basic')
    dist = LogNormal_Distribution(p['a0'], p['amin'], p['amax'], p['sigma'], p['s'])
    dist.Nc = Nc_from_size(p['a0'] * 1e4)
    return dist, pah_bin


pahneu_filepath = os.path.join(_CALIMA_ROOT, 'optical_props', 'li_draine_2001', 'PAHneu_30')
pahion_filepath = os.path.join(_CALIMA_ROOT, 'optical_props', 'li_draine_2001', 'PAHion_30')
Delta_epsilon = 0.145 # [eV] - change in internal energy of PAH due to IR photon emission of a typical C-C mode
kb = 1.3806488e-16 # [erg/K] - Boltzmann constant
mh = 1.6735575e-24 # [g] - mass of Hydrogen atom
R = 1.98720425864083 # [cal/K/mol] - R gas constant
e = 1.602176634e-19           # Elementary charge [C]
k_IR          = 100. # [photons / s] - IR emission rate
eV2erg = 1.6021773300241e-12 # [erg] conversion between eV to erg
Delta_epsilon = 0.16 # [eV] - change in internal energy of PAH due to IR photon emission of a typical C-C mode

C96_SH03 = np.array([[33.5000167150172, 1.01081258033674],
                        [77.58778065463976, 2.3648524075083763],
                        [781.2269828902782, 21.887586733274542],
                        [2902.066084868343, 83.96712132836873],
                        [8006.829038010025, 240.16300081523573],
                        [16405.369582436637, 474.0086959462781],
                        [35427.83235619407, 1042.5066595350381],
                        [72587.16691525676, 2025.9962129949902],
                        [101204.75765090082, 2675.545644224014]])
    
C96_SH004 = np.array([[254.19957645875002, 1.0237834351141317],
                        [713.6826194466255, 2.883197024324794],
                        [1514.4183071429334, 6.053628959976971],
                        [3157.981209544873, 12.90885998846291],
                        [6471.0655080199285, 27.10502197499955],
                        [13492.694750187376, 54.33006065975259],
                        [28631.864085102672, 115.85161273226693],
                        [58669.97167412683, 243.2562218316496],
                        [100915.24650146879, 405.06205665802895]])
    
dissociation_parameters = {
    'LePage2001'        : {
        'PAH+_H'        : {'E0':4.8,'S':5.0},
        'PAHH+_H'       : {'E0':2.9,'S':5.0},
        'PAHH+_H2'      : {'E0':3.2,'S':5.0},
        'PAH_H'         : {'E0':4.8,'S':5.0},
        'PAHH_H'        : {'E0':1.2,'S':0.0},
        'PAHH_H2'       : {'E0':1.6,'S':5.0}
    },
    # For Murga, the change in entropy is given in [cal/(mol K)]
    'Murga2020'         : {
        'dehydrogenated': {
            'H(Z<= 0)'  : {'E0':4.3,'S':11.8},
            'H(Z>0)'    : {'E0':4.3,'S':11.8},
            'H2'        : {'E0':3.52,'S':-12.69},
            'C2H2'      : {'E0':4.6,'S':10.0}
        },
        'hydrogenated'  : {
            'H(Z<= 0)'  : {'E0':1.4,'S':13.3},
            'H(Z>0)'    : {'E0':1.55,'S':13.3},
            'H2'        : {'E0':np.nan,'S':np.nan},
            'C2H2'      : {'E0':2.0,'S':10.0}
        }
    }
}

# Galliano data on dust destruction timescales by UV photons
# See https://irfu.cea.fr/Pisp/frederic.galliano/HDR/hdrch6.html#x7-3070004 (Section 4.2.2.1)
names = ['3A_x','3A_y','3.67A_x','3.67A_y','4.48A_x','4.48_y','5.47A_x','5.47A_y']
carbon_subl_time = pd.read_csv(_external_data_path('uv_subl_carbonaceous.csv'),names=names,header=2)

# Allain et al. (1996) C2H2 dissociation timescales by UV photons
# See https://articles.adsabs.harvard.edu/pdf/1996A%26A...305..602A (Eq. 25 and Table 6)
allain_Nc = [6,14,16,24,32,50]
allain_rates = [1.49e-10,1.89e-10,7.13e-11,4.55e-11,4.85e-12,3.56e-18]

# Murga et al. (2019) - SHIVA model prediction for PAH of 5 Angstrom photo-destruction
murga_subl_time = pd.read_csv(_external_data_path('uv_subl_pah_Murga2019.csv'),header=1)

# Micelotta et al. (2010) PAH processing in a hot gas
# See 

thermal_spu = {'50':{'electrons': {'a':-2136.83,'b':1632.17,'c':-499.822,'d':76.4347,'e':-5.82964,'f':0.177174},
                     'H':         {'a':-1896.69,'b':1480.8,'c':-462.733,'d':71.8957,'e':-5.54719,'f':0.169996},
                     'He':        {'a':-971.448,'b':770.259,'c':-245.561,'d':38.8995,'e':-3.05787,'f':0.0954303},
                     'C':         {'a':-704.392,'b':551.313,'c':-175.063,'d':27.6506,'e':-2.16871,'f':0.0675643}},
               
               '100':{'electrons': {'a':-2255.38,'b':1681.45,'c':-503.451,'d':75.4339,'e':-5.64956,'f':0.168959},
                     'H':         {'a':-1645.64,'b':1257.06,'c':-384.309,'d':58.4103,'e':-4.41007,'f':0.132356},
                     'He':        {'a':-945.901,'b':747.921,'c':-237.984,'d':37.6808,'e':-2.96613,'f':0.0928852},
                     'C':         {'a':-711.244,'b':558.48,'c':-177.983,'d':28.2547,'e':-2.23145,'f':0.0701374}},
               
               '200':{'electrons': {'a':-2234.37,'b':1597.71,'c':-459.647,'d':66.332,'e':-4.79841,'f':0.139019},
                     'H':         {'a':-1473.64,'b':1109.01,'c':-334.292,'d':50.1417,'e':-3.74133,'f':0.111164},
                     'He':        {'a':-963.188,'b':765.639,'c':-245.054,'d':39.0745,'e':-3.10123,'f':0.0980047},
                     'C':         {'a':-738.791,'b':584.928,'c':-187.884,'d':30.0819,'e':-2.39748,'f':0.0760639}}}

# H2 formation onto PAHs by Le Page et al. (2009) -
# (https://ui.adsabs.harvard.edu/abs/2009ApJ...704..274L/abstract)
# The mechanism involves the chemical trapping of H atoms on the periphery of the PAH
# carbon skeleton and the subsequent release of H2 through dissociative recombination
# of the hydrogenated ion with an electron.
names = ['C32_x','C32_y','C40_x','C40_y','C50_x','C50_y','C80_x',
         'C80_y','C100_x','C100_y','C120_x','C120_y','C150_x','C150_y']
H2_rate_LePage = pd.read_csv(_external_data_path('H2_formation_rate_PAH_LePage2009.csv'),header=1,names=names)

# FUNCTIONS
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
    return Teff

def Gibbs_dissociation_rate(Te,DeltaS,E0):
    from unyt import h,K,kb    
    # 1. Compute k0
    
    k0 = kb*Te*K/h * np.exp(1.+DeltaS/R)
    
    # 2. Compute the Gibbs microcanonical distribution bond dissociation rate
    kdiss = k0.to('1/s').d * np.exp(-E0 / (8.617e-5*Te))
    
    return kdiss

def dissociation_probability(DeltaS,E0,Nc,T_av):
    
    # 1. Compute the maximum number of IR photon emissions
    # as suggested by the results of Micelotta et al. (2010b)
    n_max = float(int(Nc / 5))
    
    kdiss = Gibbs_dissociation_rate(T_av,DeltaS,E0)
    P = kdiss / (kdiss + k_IR/(n_max+1.))
    
    return P

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
        nwav,data,columns,name = pah_efficiencies(pahneu_filepath)
    else:
        nwav,data,columns,name = pah_efficiencies(pahion_filepath)
    
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

def compute_acetylene_dissociation_rate(args):
    
    # 1. Unpack arguments
    wav,sigma_abs,dist,params = args
    a0 = dist.a0
    Nc = dist.Nc
    n_max = float(int(Nc / 5))
    
    # 2. Convert wavelength [micron] to photon energy [eV]
    E = 1.2398 / wav
    # Convert from [photons cm^-2 s^-1 nm^-1] to [W m^-2 eV^-1]
    I = Draine_1978_isrf(wav*1e3) /1.7 * cm**-2/s/nm
    F = I * E * eV
    f = F *nm/ (1e-9*m) * h * c / (E*eV)**2 * eV / (e*J)
    I = f.to('W/m**2/eV').d
    
    # 3. Compute the integrated rate
    mask = (E>=params['dehydrogenated']['C2H2']['E0']) & (E<=13.6)
    k = []
    for i in range(0, len(E)):
        if params['dehydrogenated']['C2H2']['E0'] <= E[i] <= 13.6:
            T_0 = E[i]
            print(E[i])
            T_nmax = T_0 - n_max * Delta_epsilon
            T_0 = effective_temperature(Nc,T_0,params['dehydrogenated']['C2H2']['E0'])
            T_nmax = effective_temperature(Nc,T_nmax,params['dehydrogenated']['C2H2']['E0'])
            T_av = np.sqrt(T_0 * T_nmax)
            P = dissociation_probability(params['dehydrogenated']['C2H2']['S'],
                                         params['dehydrogenated']['C2H2']['E0'],
                                         Nc,T_av)
            print(sigma_abs[i],I[i],P,E[i],T_av)
            k.append(sigma_abs[i] * I[i] * P / E[i] * 6.24150935e+18)
    k = np.array(k)
    R = np.trapezoid(k,E[mask])
    
    return R

def plot_acetylene_dissociation_rate(G0min,G0max,nHmin,nHmax,pah_bin_id=None,pah_bin_rank=0):
    
    from scipy.optimize import curve_fit
    # 1. Read the Montillaud data
    montillaud_fraction = read_sections('montillaud13_hydrogenation_fraction_C54.csv')
    params, _ = curve_fit(curve_hydro, np.log10(montillaud_fraction[1][0]), np.log10(montillaud_fraction[1][1]), maxfev=10000)           
 
    # 2. Compute the fraction of fully dehydrogenated PAHs at each G0 and nH
    G0 = np.linspace(np.log10(G0min), np.log10(G0max),100)
    nH = np.linspace(np.log10(nHmin), np.log10(nHmax),100)
    f_dehydro = np.zeros((100,100))
    for i in range(0,100):
        for j in range(0, 100):
            f_dehydro[j,i] = find_value(G0[i],nH[j],*params)*10**G0[i]
    
    # 3. Compute the destruction rate [1/s]
    params = dissociation_parameters['Murga2020']
    dist, _ = _build_pah_distribution(pah_bin_id=pah_bin_id, pah_bin_rank=pah_bin_rank)
    wav,sigma_abs_cation = absorption_cross_section(dist,1)
    R = compute_acetylene_dissociation_rate((wav,sigma_abs_cation,dist,params))
    # 4. Rescale by the value of f_dehydro and G0
    diss_rate = R * f_dehydro
    
    with open('acetylene_dissociation_table_%s.dat'%(dist.a0),'w') as f:
        f.write(f"{len(G0)} {len(nH)}\n")
        # Write the X array
        for x in G0:
            f.write(f"{x}\n")
        # Write the Y array
        for y in nH:
            f.write(f"{y}\n")
        
        # Write the Z array
        for i in range(len(G0)):
            for j in range(len(nH)):
                f.write(f"{np.log10(diss_rate[j, i])}\n")
    
    # 5. Plot results
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="white")
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": "Computer Modern Roman",
    })
    fig, ax = plt.subplots(1,1, figsize=(6,5),dpi=300,facecolor='w',edgecolor='k')
    ax.set_xlabel(r'$G0$', fontsize=20)
    ax.set_ylabel(r'$n_{\rm H}$ [cm$^{3}$]', fontsize=20)
    ax.tick_params(labelsize=14)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.minorticks_on()
    ax.tick_params(which='both',axis="both",direction="in")
    ax.set_xscale('log')
    ax.set_yscale('log')
    G0, nH = np.meshgrid(G0, nH)  # 2D grid for interpolation
    pc = ax.contourf(10**G0, 10**nH, np.log10(diss_rate), levels=np.linspace(-24, -11, 40), cmap='RdBu')
    fig.colorbar(pc,label=r'$\log(k_{\rm diss}/[{\rm s}^{-1}])$')
    fig.savefig('C54_integrated_dissociation_rate.png',format='png',dpi=300)
    
def plot_evaporation_rate():
    
    C54H18_4 = np.array([[10.155542455979203, 1326721.2821181505],
                [100.99848964447678, 15280.770324389228],
                [1030.9350963606964, 87.25629056691685],
                [10518.431343866167, 0.22625087563449475]
                ])
    C54H18_12 = np.array([[3888.5930193337376, 2744576.321114021],
                            [40553.078563969466, 2.6530025474682906],
                            [400877.4887208798, 8.197162440732647e-7],
                            [4098320.880197934, 7.100195219251391e-8],
                            [41909220.00407155, 9.535724414677071e-9],
                            ])
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="white")
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": "Computer Modern Roman",
    })
    fig, ax = plt.subplots(1,1, figsize=(6,5),dpi=300,facecolor='w',edgecolor='k')
    ax.set_xlabel(r'$G0$', fontsize=20)
    ax.set_ylabel(r'$\tau_{\rm evap}$ [yr]', fontsize=20)
    ax.tick_params(labelsize=14)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.minorticks_on()
    ax.tick_params(which='both',axis="both",direction="in")
    ax.set_xscale('log')
    ax.set_yscale('log')
    G0 = np.logspace(-1,8,100)
    ax.plot(G0,0.19306/G0,linestyle='--',color='grey')
    
    ax.plot(C54H18_4[:,0],C54H18_4[:,1],linestyle='-',color='b',marker='o',label=r'(C$_{54}$H$_{18}$)$_{4}$')
    ax.plot(C54H18_12[:,0],C54H18_12[:,1],linestyle='-',color='r',marker='o',label=r'(C$_{54}$H$_{18}$)$_{12}$')
    mean_up_x = 0.5*(np.log10(C54H18_4[0,0])+np.log10(C54H18_12[0,0]))
    mean_up_y = 0.5*(np.log10(C54H18_4[0,1])+np.log10(C54H18_12[0,1]))
    mean_down_x = 0.5*(np.log10(C54H18_4[3,0])+np.log10(C54H18_12[1,0]))
    mean_down_y = 0.5*(np.log10(C54H18_4[3,1])+np.log10(C54H18_12[1,1]))
    mean_x, mean_y = np.array([mean_up_x,mean_down_x]),np.array([mean_up_y,mean_down_y])
    ax.plot(10**mean_x,10**mean_y,linestyle='--',color='k',marker='o',label=r'(C$_{54}$H$_{18}$)$_{8}$')
    slope = (mean_down_y-mean_up_y) / (mean_down_x-mean_up_x)
    intercept = mean_down_y - slope * mean_down_x
    print(slope,intercept)
    ax.plot(G0,10**(slope*np.log10(G0)+intercept),linestyle='--',color='k',alpha=0.6)

    ax.legend(loc='best', frameon=False, fontsize=14)
    fig.savefig('PAH_cluster_evaporation_time.png',format='png',dpi=300)

def plot_destruction_timescale_ratio(G0min,G0max,nHmin,nHmax,pah_bin_id=None,pah_bin_rank=0):
    """Plot the ratio of large-PAH evaporation to small-PAH dissociation timescales.

    The evaporation timescale is based on the mean fit used in plot_evaporation_rate,
    while the dissociation rate follows plot_acetylene_dissociation_rate.
    """
    from scipy.optimize import curve_fit
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="white")
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": "Computer Modern Roman",
    })

    # 1. Large-PAH evaporation timescale fit (in years)
    C54H18_4 = np.array([[10.155542455979203, 1326721.2821181505],
                [100.99848964447678, 15280.770324389228],
                [1030.9350963606964, 87.25629056691685],
                [10518.431343866167, 0.22625087563449475]
                ])
    C54H18_12 = np.array([[3888.5930193337376, 2744576.321114021],
                            [40553.078563969466, 2.6530025474682906],
                            [400877.4887208798, 8.197162440732647e-7],
                            [4098320.880197934, 7.100195219251391e-8],
                            [41909220.00407155, 9.535724414677071e-9],
                            ])
    mean_up_x = 0.5*(np.log10(C54H18_4[0,0])+np.log10(C54H18_12[0,0]))
    mean_up_y = 0.5*(np.log10(C54H18_4[0,1])+np.log10(C54H18_12[0,1]))
    mean_down_x = 0.5*(np.log10(C54H18_4[3,0])+np.log10(C54H18_12[1,0]))
    mean_down_y = 0.5*(np.log10(C54H18_4[3,1])+np.log10(C54H18_12[1,1]))
    slope = (mean_down_y-mean_up_y) / (mean_down_x-mean_up_x)
    intercept = mean_down_y - slope * mean_down_x

    # 2. Small-PAH dissociation rate (in 1/s)
    montillaud_fraction = read_sections('montillaud13_hydrogenation_fraction_C54.csv')
    params, _ = curve_fit(curve_hydro, np.log10(montillaud_fraction[1][0]), np.log10(montillaud_fraction[1][1]), maxfev=10000)

    G0 = np.linspace(np.log10(G0min), np.log10(G0max),100)
    nH = np.linspace(np.log10(nHmin), np.log10(nHmax),100)
    f_dehydro = np.zeros((100,100))
    for i in range(0,100):
        for j in range(0, 100):
            f_dehydro[j,i] = find_value(G0[i],nH[j],*params)*10**G0[i]

    params = dissociation_parameters['Murga2020']
    dist, _ = _build_pah_distribution(pah_bin_id=pah_bin_id, pah_bin_rank=pah_bin_rank)
    wav,sigma_abs_cation = absorption_cross_section(dist,1)
    R = compute_acetylene_dissociation_rate((wav,sigma_abs_cation,dist,params))
    diss_rate = R * f_dehydro

    # 3. Ratio of timescales (dimensionless)
    G0_lin = 10**G0
    tau_evap_yr = 10**(slope*np.log10(G0_lin)+intercept)
    tau_evap_s = tau_evap_yr * 3.15576e7
    ratio = (tau_evap_s) * diss_rate
    log_ratio = np.log10(ratio)
    finite_mask = np.isfinite(log_ratio)
    if np.any(finite_mask):
        vlim = np.nanpercentile(np.abs(log_ratio[finite_mask]), 95)
        vlim = max(vlim, 1.0)
    else:
        vlim = 1.0

    # 4. Plot
    fig, ax = plt.subplots(1,1, figsize=(6,5),dpi=300,facecolor='w',edgecolor='k')
    ax.set_xlabel(r'$G0$', fontsize=20)
    ax.set_ylabel(r'$n_{\rm H}$ [cm$^{3}$]', fontsize=20)
    ax.tick_params(labelsize=14)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.minorticks_on()
    ax.tick_params(which='both',axis="both",direction="in")
    ax.set_xscale('log')
    ax.set_yscale('log')
    G0_grid, nH_grid = np.meshgrid(G0_lin, 10**nH)
    levels = np.linspace(-vlim, vlim, 41)
    pc = ax.contourf(G0_grid, nH_grid, log_ratio, levels=levels, cmap='coolwarm')
    ax.contour(G0_grid, nH_grid, log_ratio, levels=[0.0], colors='k', linewidths=1.4)
    fig.colorbar(pc,label=r'$\log(\tau_{\rm evap}/\tau_{\rm diss})$')
    fig.savefig('PAH_destruction_timescale_ratio.png',format='png',dpi=300)

def read_sections(filename):
    filename = _external_data_path(filename)
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    sections = []
    current_section = []

    for line in lines:
        line = line.strip()
        if line == "======================================":
            if current_section:
                sections.append(current_section)
                current_section = []
        else:
            current_section.append(line)
    
    if current_section:  # Add the last section if it exists
        sections.append(current_section)

    # Process each section to separate x and y values
    data = []
    for section in sections:
        x_values = []
        y_values = []
        for item in section:
            x, y = map(float, item.split(','))
            x_values.append(x)
            y_values.append(y)
        data.append((x_values, y_values))
    
    return data

def plot_dissociation_rates():
    
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="white")
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": "Computer Modern Roman",
    })
    
    # 1. Setup the figure
    fig, axes = plt.subplots(1,3, figsize=(10,4),dpi=300,facecolor='w',edgecolor='k',sharex=True,sharey=True)
    axes[0].set_ylabel(r'$k_{\rm diss}$ [s$^{-1}$]', fontsize=20)
    for i in range(0, 3):
        axes[i].tick_params(labelsize=14)
        axes[i].xaxis.set_ticks_position('both')
        axes[i].yaxis.set_ticks_position('both')
        axes[i].minorticks_on()
        axes[i].tick_params(which='both',axis="both",direction="in")
        axes[i].set_yscale('log')
        axes[i].set_xlabel(r'$E_{\rm int}$ [eV]', fontsize=20)
    
    # 1. Load the Andrews et al. (2016) data
    dissH_dehidrogenated = pd.read_csv(_external_data_path('andrews16_Hdissociation_rates_C54H17.csv'),names=['E','k'])
    axes[0].plot(dissH_dehidrogenated['E'],dissH_dehidrogenated['k'],linestyle='-',color='k')
    dissH_hidrogenated = pd.read_csv(_external_data_path('andrews16_Hdissociation_rates_C54H18.csv'),names=['E','k'])
    axes[1].plot(dissH_hidrogenated['E'],dissH_hidrogenated['k'],linestyle='-',color='k')
    dissH2_hidrogenated = pd.read_csv(_external_data_path('andrews16_H2dissociation_rates_C54H18.csv'),names=['E','k'])
    axes[1].plot(dissH2_hidrogenated['E'],dissH2_hidrogenated['k'],linestyle='--',color='k')
    dissH_superhidrogenated = pd.read_csv(_external_data_path('andrews16_Hdissociation_rates_C54H19.csv'),names=['E','k'])
    axes[2].plot(dissH_superhidrogenated['E'],dissH_superhidrogenated['k'],linestyle='-',color='k')
    
    # 2. Load the Castellanos et al. (2018) data
    dissoddH_cation = pd.read_csv(_external_data_path('castellanos18_oddHdissociation_rates_C54H18.csv'),names=['E','k'])
    axes[1].plot(dissoddH_cation['E'],dissoddH_cation['k'],linestyle=':',color='b')
    dissH_cation = pd.read_csv(_external_data_path('castellanos18_Hdissociation_rates_C54H18.csv'),names=['E','k'])
    axes[1].plot(dissH_cation['E'],dissH_cation['k'],linestyle='-',color='b')
    
    # 3. Compute the values for the microcanonical description using the parameters by Murga et al. (2020)
    Eint = np.linspace(1,40,100)
    params = dissociation_parameters['Murga2020']
    Te = effective_temperature(54.,Eint,params['dehydrogenated']['H(Z<= 0)']['E0'])
    kdiss = Gibbs_dissociation_rate(Te,params['dehydrogenated']['H(Z<= 0)']['S'],
                                    params['dehydrogenated']['H(Z<= 0)']['E0'])
    axes[0].plot(Eint,kdiss,linestyle='-',color='r')
    Te = effective_temperature(54.,Eint,params['dehydrogenated']['H2']['E0'])
    kdiss = Gibbs_dissociation_rate(Te,params['dehydrogenated']['H2']['S'],
                                    params['dehydrogenated']['H2']['E0'])
    axes[0].plot(Eint,kdiss,linestyle='--',color='r')
    
    Te = effective_temperature(54.,Eint,params['dehydrogenated']['C2H2']['E0'])
    kdiss = Gibbs_dissociation_rate(Te,params['dehydrogenated']['C2H2']['S'],
                                    params['dehydrogenated']['C2H2']['E0'])
    axes[0].plot(Eint,kdiss,linestyle='-.',color='r')
    
    Te = effective_temperature(54.,Eint,params['hydrogenated']['H(Z<= 0)']['E0'])
    kdiss = Gibbs_dissociation_rate(Te,params['hydrogenated']['H(Z<= 0)']['S'],
                                    params['hydrogenated']['H(Z<= 0)']['E0'])
    axes[2].plot(Eint,kdiss,linestyle='-',color='r')
    
    Te = effective_temperature(54.,Eint,params['hydrogenated']['C2H2']['E0'])
    kdiss = Gibbs_dissociation_rate(Te,params['hydrogenated']['C2H2']['S'],
                                    params['hydrogenated']['C2H2']['E0'])
    axes[2].plot(Eint,kdiss,linestyle='-.',color='r')
    
    
    fig.subplots_adjust(top=0.98,bottom=0.14,left=0.08,right=0.98,hspace=0,wspace=0)
    fig.savefig('circumcoronene_dissociation_rates.png',format='png',dpi=300)
    plt.close(fig)
    
def curve_hydro(x,a,b):
    
    return a*x + b

def find_value(G0_point, nH_point, a, b, width=0.1):
    f_prime = a  # Derivative of the linear function
    y_intercept = a * G0_point + b  # f(nH_point)

    # Distance from point to the line
    distance = abs(y_intercept - nH_point) / np.sqrt(1 + f_prime**2)
    
    if nH_point >= y_intercept:
        value = 0.5 - 0.5/(1.+width*distance**(-2.))
    else:
        value = 0.5 + 0.5/(1.+width*distance**(-2.))
    return value

def plot_integrated_dissociation_rate():
    
    # 1. Setup figure
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="white")
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": "Computer Modern Roman",
    })
    from scipy.interpolate import CloughTocher2DInterpolator as CT
    from scipy.optimize import curve_fit
    fig, ax = plt.subplots(1,1, figsize=(6,5),dpi=300,facecolor='w',edgecolor='k')
    ax.set_xlabel(r'$G0$', fontsize=20)
    ax.set_ylabel(r'$n_{\rm H}$ [cm$^{3}$]', fontsize=20)
    ax.tick_params(labelsize=14)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.minorticks_on()
    ax.tick_params(which='both',axis="both",direction="in")
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # 2. Read the data for the hydrogenated fraction from Montillaud et al. (2013)
    montillaud_fraction = read_sections('montillaud13_hydrogenation_fraction_C54.csv')
    ax.plot(montillaud_fraction[0][0],montillaud_fraction[0][1],linestyle='--',color='k',zorder=30)
    ax.plot(montillaud_fraction[1][0],montillaud_fraction[1][1],linestyle='-',color='k',zorder=30)
    ax.plot(montillaud_fraction[2][0],montillaud_fraction[2][1],linestyle='-.',color='k',zorder=30)
    
    ax.legend(loc='best', frameon=False, fontsize=14)
    fig.subplots_adjust(top=0.98,bottom=0.112,left=0.15,right=0.98,hspace=0,wspace=0)
    fig.savefig('test_hydrogenation.png',format='png',dpi=300)
    plt.close(fig)
    
    # 3. Perform linear interpolation
    x = np.concatenate([montillaud_fraction[0][0],montillaud_fraction[1][0],montillaud_fraction[2][0]])
    x = np.log10(x)
    y = np.concatenate([montillaud_fraction[0][1],montillaud_fraction[1][1],montillaud_fraction[2][1]])
    y = np.log10(y)
    z = np.concatenate([np.full(len(montillaud_fraction[0][0]),0.1),
                        np.full(len(montillaud_fraction[1][0]),0.5),
                        np.full(len(montillaud_fraction[2][0]),0.9)])
    interp = CT(np.c_[x,y],z)
    X, Y = np.linspace(min(x), max(x)), np.linspace(min(y),max(y))
    X, Y = np.meshgrid(X, Y)  # 2D grid for interpolation
    Z = interp(X, Y)
    ax.contourf(10**X, 10**Y, Z, levels=np.linspace(0, 1, 30), cmap='RdBu',zorder=10)
    
    # 4. The fitting of the hydrogenation fraction
    params, _ = curve_fit(curve_hydro, np.log10(montillaud_fraction[1][0]), np.log10(montillaud_fraction[1][1]), maxfev=10000)
    print(params)
    G0 = np.linspace(min(x),max(x),100)
    ax.plot(10**G0,10**curve_hydro(G0,*params),linestyle=':',color='r')
    ax.fill_between(10**G0,10**(curve_hydro(G0,*params)+1),10**(curve_hydro(G0,*params)-1),color='r',alpha=0.4)
    
    G0 = np.linspace(-4, 4,100)
    nH = np.linspace(-4,4,100)
    new_z = np.zeros((100,100))
    for i in range(0,100):
        for j in range(0, 100):
            new_z[j,i] = find_value(G0[i],nH[j],*params)
    G0, nH = np.meshgrid(G0, nH)  # 2D grid for interpolation
    pc = ax.contourf(10**G0, 10**nH, new_z, levels=np.linspace(0, 1, 30), cmap='RdBu')
    fig.colorbar(pc,label='Value')
    fig.savefig('test_hydrogenation.png',format='png',dpi=300)
    plt.close(fig)

def compute_h2_dissociation_rate(args):
    from models.PAH_charge.PAH_photoelectric_heating import ionisation_potential,ionisation_yield
    
    wav,sigma,G0,E0,S,dist,Z = args

    a0 = dist.a0
    Nc = dist.Nc
    n_max = float(int(Nc / 5))
    
    # 1. Compute the radiation field quantities
    # Convert wavelength [micron] to photon energy [eV]
    E = 1.2398 / wav
    # Convert from [photons cm^-2 s^-1 nm^-1] to [W m^-2 eV^-1]
    I = G0 * Draine_1978_isrf(wav*1e3) /1.7 * cm**-2/s/nm
    F = I * E * eV
    f = F *nm/ (1e-9*m) * h * c / (E*eV)**2 * eV / (e*J)
    I = f.to('W/m**2/eV').d
    
    # 2. Compute the ionisation yield
    IP = ionisation_potential(Z,a0*1e3)
    ion_yield = np.array([ionisation_yield(Nc,Z,E[i],IP) for i in range(0,len(E))])
    
    # 3. Loop over energies computing the dissociation rates
    kph = np.zeros(len(E))
    for i in range(0, len(E)):
        if E[i] < 13.6:
            T_0 = E[i]
            T_nmax = T_0 - n_max * Delta_epsilon
            if T_nmax < 0.0:
                kph[i] = 0.0
            else:            
                # 3.A Compute the H dissociation rate
                if T_nmax < E0[0]:
                    kH = 0.0
                else:
                    T_0_h = effective_temperature(Nc,T_0,E0[0])
                    T_nmax_h = effective_temperature(Nc,T_nmax,E0[0])
                    T_av = np.sqrt(T_0_h * T_nmax_h)
                    kH = Gibbs_dissociation_rate(T_av,S[0],E0[0])
                
            
                # 3.B Compute the H2 dissociation rate
                if T_nmax < E0[1]:
                    kH2 = 0.0
                else:
                    T_0_h2 = effective_temperature(Nc,T_0,E0[1])
                    T_nmax_h2 = effective_temperature(Nc,T_nmax,E0[1])
                    T_av = np.sqrt(T_0_h2 * T_nmax_h2)
                    kH2 = Gibbs_dissociation_rate(T_av,S[1],E0[1])

                # 3.C Compute the rate
                k = kH2 / (kH + kH2 + k_IR/(n_max+1.))
                kph[i] = k * sigma[i] * I[i] / E[i] * 6.24150935e+18
        
    kph = np.array(kph)
    R = np.trapezoid(kph,E)
    
    return R

def linear_fit(x,a,b):
    
    return a*x + b

def plot_h2_dissociation_rate(G0min,G0max,n_G0=100,pah_bin_id=None,pah_bin_rank=0):
    from scipy.optimize import curve_fit
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="white")
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": "Computer Modern Roman",
    })
    
    # 1. Setup the figure
    fig, ax = plt.subplots(1,1, figsize=(6,5),dpi=300,facecolor='w',edgecolor='k')
    ax.set_xlabel(r'$G0$', fontsize=20)
    ax.set_ylabel(r'$k_{\rm H\,_{2}}$ [s$^{-1}$]', fontsize=20)
    ax.tick_params(labelsize=14)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.minorticks_on()
    ax.tick_params(which='both',axis="both",direction="in")
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    G0 = np.logspace(np.log10(G0min), np.log10(G0max),n_G0)
    
    # 2. Compute the rate for the selected PAH bin
    dist, pah_bin = _build_pah_distribution(pah_bin_id=pah_bin_id, pah_bin_rank=pah_bin_rank)
    Z = 0
    wav,sigma_abs_neutral = absorption_cross_section(dist,0)
    args_list_neu = []
    E0 = np.array([dissociation_parameters['Murga2020']['dehydrogenated']['H(Z<= 0)']['E0'],
          dissociation_parameters['Murga2020']['dehydrogenated']['H2']['E0']])
    S = np.array([dissociation_parameters['Murga2020']['dehydrogenated']['H(Z<= 0)']['S'],
          dissociation_parameters['Murga2020']['dehydrogenated']['H2']['S']])
    for k in range(0,n_G0):
        args_list_neu.append((wav,sigma_abs_neutral,G0[k],E0,S,dist,Z))
    num_cores = 5

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        results_neu = list(tqdm(executor.map(compute_h2_dissociation_rate, args_list_neu), total=n_G0,
                            desc=f'    Computing H2 dissociation rate for neutral', unit=' steps'))
    R_neutral = np.array(results_neu)
      
    ax.plot(G0,R_neutral,linestyle='-',color='g',label=rf'{pah_bin["id"]} ($N_{{\rm C}}={dist.Nc}$), $Z=0$')
    params_neutral, _ = curve_fit(curve_hydro, np.log10(G0), np.log10(R_neutral), maxfev=10000)
    print('Neutral: ',params_neutral)
    
    # 3. Compute the rate for the circumcoronene cation
    dist, _ = _build_pah_distribution(pah_bin_id=pah_bin_id, pah_bin_rank=pah_bin_rank)
    Z = 1
    wav,sigma_abs_cation = absorption_cross_section(dist,1)
    args_list_ca = []
    E0 = np.array([dissociation_parameters['Murga2020']['dehydrogenated']['H(Z>0)']['E0'],
          dissociation_parameters['Murga2020']['dehydrogenated']['H2']['E0']])
    S = np.array([dissociation_parameters['Murga2020']['dehydrogenated']['H(Z>0)']['S'],
          dissociation_parameters['Murga2020']['dehydrogenated']['H2']['S']])
    for k in range(0,n_G0):
        args_list_ca.append((wav,sigma_abs_cation,G0[k],E0,S,dist,Z))
    num_cores = 5
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        results_ca = list(tqdm(executor.map(compute_h2_dissociation_rate, args_list_ca), total=n_G0,
                            desc=f'    Computing H2 dissociation rate for cation', unit=' steps'))
    R_cation = np.array(results_ca)
      
    ax.plot(G0,R_cation,linestyle='--',color='b',label=rf'{pah_bin["id"]} ($N_{{\rm C}}={dist.Nc}$), $Z=1$')
    params_cation, _ = curve_fit(curve_hydro, np.log10(G0), np.log10(R_cation), maxfev=10000)
    print('Cation: ',params_cation)
    
    # 4. Compute the rate for the circumcoronene dication
    dist, _ = _build_pah_distribution(pah_bin_id=pah_bin_id, pah_bin_rank=pah_bin_rank)
    Z = 2
    wav,sigma_abs_dication = absorption_cross_section(dist,1)
    args_list_di = []
    E0 = np.array([dissociation_parameters['Murga2020']['dehydrogenated']['H(Z>0)']['E0'],
          dissociation_parameters['Murga2020']['dehydrogenated']['H2']['E0']])
    S = np.array([dissociation_parameters['Murga2020']['dehydrogenated']['H(Z>0)']['S'],
          dissociation_parameters['Murga2020']['dehydrogenated']['H2']['S']])
    for k in range(0,n_G0):
        args_list_di.append((wav,sigma_abs_dication,G0[k],E0,S,dist,Z))
    num_cores = 5
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        results_di = list(tqdm(executor.map(compute_h2_dissociation_rate, args_list_di), total=n_G0,
                            desc=f'    Computing H2 dissociation rate for dication', unit=' steps'))
    R_dication = np.array(results_di)
      
    ax.plot(G0,R_dication,linestyle='-.',color='r',label=rf'{pah_bin["id"]} ($N_{{\rm C}}={dist.Nc}$), $Z=2$')
    params_dication, _ = curve_fit(curve_hydro, np.log10(G0), np.log10(R_dication), maxfev=10000)
    print('Dication: ',params_dication)

    
    ax.legend(loc='best', frameon=False, fontsize=14)
    fig.savefig('H2_photodissociation_rate.png',format='png',dpi=300)

def plot_h2_efficiency(G0,ntmin,ntmax,Xe,T,f,n_nt=100,pah_bin_id=None,pah_bin_rank=0):
    from models.PAH_charge.PAH_photoelectric_heating import read_data,interpolate_linear
    from scipy.optimize import curve_fit
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="white")
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": "Computer Modern Roman",
    })
    nt = np.logspace(np.log10(ntmin),np.log10(ntmax),n_nt)
    nH = nt * (1. - f)
    ne = Xe * nH
    # 1. Setup the figure
    fig, ax = plt.subplots(1,1, figsize=(7,5),dpi=300,facecolor='w',edgecolor='k')
    ax.set_xlabel(r'$G_0/n_{\rm tot}$', fontsize=20)
    ax.set_ylabel(r'$\xi_{\rm H\,_{2}}$', fontsize=20)
    ax.tick_params(labelsize=14)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.minorticks_on()
    ax.tick_params(which='both',axis="both",direction="in")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim([1e-5,1e0])
    dist, pah_bin = _build_pah_distribution(pah_bin_id=pah_bin_id, pah_bin_rank=pah_bin_rank)
    
    # 2. Read the Montillaud data
    montillaud_fraction = read_sections('montillaud13_hydrogenation_fraction_C54.csv')
    params_dh, _ = curve_fit(curve_hydro, np.log10(montillaud_fraction[1][0]), np.log10(montillaud_fraction[1][1]), maxfev=10000)
    x_mean = 0.5 * (np.log10(C96_SH03[:,0]) + np.log10(C96_SH004[:,0]))
    y_mean = 0.5 * (np.log10(C96_SH03[:,1]) + np.log10(C96_SH004[:,1]))
    params_sh, _ = curve_fit(curve_hydro, x_mean, y_mean, maxfev=10000)
    print(params_dh)
    print(params_sh)               
    
    models = ['Berne','Tielens']
    linestyles = ['-','--']
    for m in range(0, len(models)):
        # 3. Compute the fraction of fully dehydrogenated PAHs and their charges at each G0
        f_dehydro = np.zeros(n_nt)
        f_sphydro = np.zeros(n_nt)
        f_anion = np.zeros(n_nt)
        f_neutral = np.zeros(n_nt)
        f_cation = np.zeros(n_nt)
        f_dication = np.zeros(n_nt)
        log_G0, log_ne, log_T, f_anion_matrix = read_data('./PAH_PEH_data/f_anion_%s_pah_%.4f_micron.dat'%(models[m],dist.a0))
        log_G0, log_ne, log_T, f_neutral_matrix = read_data('./PAH_PEH_data/f_neutral_%s_pah_%.4f_micron.dat'%(models[m],dist.a0))
        log_G0, log_ne, log_T, f_cation_matrix = read_data('./PAH_PEH_data/f_cation_%s_pah_%.4f_micron.dat'%(models[m],dist.a0))
        log_G0, log_ne, log_T, f_dication_matrix = read_data('./PAH_PEH_data/f_dication_%s_pah_%.4f_micron.dat'%(models[m],dist.a0))
        for i in range(0,n_nt):
            f_dehydro[i] = max(find_value(np.log10(G0),np.log10(nH[i]),*params_dh),0.0)
            f_sphydro[i] = max(find_value(np.log10(nH[i]),np.log10(G0),*params_sh,width=0.05),0.0)
            f_anion[i] = max(interpolate_linear(log_G0, log_ne, log_T, f_anion_matrix, G0, ne[i], T),0.0)
            f_neutral[i] = max(interpolate_linear(log_G0, log_ne, log_T, f_neutral_matrix, G0, ne[i], T),0.0)
            f_cation[i] = max(interpolate_linear(log_G0, log_ne, log_T, f_cation_matrix, G0, ne[i], T),0.0)
            f_dication[i] = max(interpolate_linear(log_G0, log_ne, log_T, f_dication_matrix, G0, ne[i], T),0.0)
        # 4. Compute the H2 photo-dissociation rate for each charge state (this is only for de-hydrogenated PAH)
        wav,sigma_abs_anion = absorption_cross_section(dist,0)
        wav,sigma_abs_neu = absorption_cross_section(dist,0)
        wav,sigma_abs_cation = absorption_cross_section(dist,1)
        wav,sigma_abs_dication = absorption_cross_section(dist,1)
        E0 = np.array([dissociation_parameters['Murga2020']['dehydrogenated']['H(Z<= 0)']['E0'],
            dissociation_parameters['Murga2020']['dehydrogenated']['H2']['E0']])
        S = np.array([dissociation_parameters['Murga2020']['dehydrogenated']['H(Z<= 0)']['S'],
            dissociation_parameters['Murga2020']['dehydrogenated']['H2']['S']])
        k_H2 = np.array([compute_h2_dissociation_rate((wav,sigma_abs_anion,G0,E0,S,dist,-1)),
                        compute_h2_dissociation_rate((wav,sigma_abs_neu,G0,E0,S,dist,0)),
                        compute_h2_dissociation_rate((wav,sigma_abs_cation,G0,E0,S,dist,1)),
                        compute_h2_dissociation_rate((wav,sigma_abs_dication,G0,E0,S,dist,2))])
        
        # 5. Compute the collisional rates
        # nH * np.pi*(dist.a0*1e-4)**2.*np.sqrt(8.*kb*T/(np.pi*mh))*np.exp(-6e-2*eV2erg/(kb*T))
        k_col = np.array([7.8e-10 * nH,
                        nH * np.pi*(dist.a0*1e-4)**2.*np.sqrt(8.*kb*T/(np.pi*mh))*np.exp(-6e-2*eV2erg/(kb*T)),
                        1.4e-10 * nH,
                        1.4e-10 * nH])
        
        # 5. Compute the Eley-Rideal rate
        k_er = 8.7e-13 * np.sqrt(T/100.) * nH
        
        
        # 6. Compute the efficiency of H2 formation
        eps_H2 = np.zeros((n_nt,3,4))
        
        
        for i in range(0,n_nt):
            eps_H2[i,0,1:4] = f_dehydro[i] * k_H2[1:4] / k_col[1:4,i]/(1+np.exp((f_dehydro[i]-0.99)/8e-4))
            eps_H2[i,1,0] = f_sphydro[i] * k_er[i] / k_col[0,i]
            eps_H2[i,1,1] = f_sphydro[i] * k_er[i] / k_col[1,i]
            eps_H2[i,2,:] = f_dehydro[i] * k_H2[:] / k_col[:,i]/(1+np.exp((f_dehydro[i]-0.99)/8e-4))
            eps_H2[i,2,0] = eps_H2[i,2,0] + f_sphydro[i] * k_er[i] / k_col[0,i]
            eps_H2[i,2,1] = eps_H2[i,2,1] + f_sphydro[i] * k_er[i] / k_col[1,i]

        # 7. Compute the PAH density and plot
        eps_tot = np.zeros((n_nt,3))
        for i in range(0, n_nt):
            # eps_tot[i,:] = eps_H2[i,:,1] * f_neutral[i]
            eps_tot[i,:] = eps_H2[i,:,0] * f_anion[i] +\
                                eps_H2[i,:,1] * f_neutral[i]+\
                                eps_H2[i,:,2] * f_cation[i]+\
                                eps_H2[i,:,3] * f_dication[i]

        ax.plot(G0/nt,eps_tot[:,0],linestyle=linestyles[m],linewidth=2.5,color='r',label=r'Photo-dissociation')
        ax.plot(G0/nt,eps_tot[:,1],linestyle=linestyles[m],linewidth=2.5,color='b',label=r'Eley-Rideal')
        ax.plot(G0/nt,eps_tot[:,2],linestyle=linestyles[m],linewidth=2.5,color='k',label=r'Total')
        
    dummy_lines = [ax.plot([],[],color='r',linestyle='-',label=r'Photo-dissociation')[0],
                   ax.plot([],[],color='b',linestyle='-',label=r'Eley-Rideal')[0],
                   ax.plot([],[],color='k',linestyle='-',label=r'Total')[0]]
    first_legend = ax.legend(handles=dummy_lines, loc='upper right', frameon=False, fontsize=14)
    ax.add_artist(first_legend)
    
    dummy_lines = [ax.plot([],[],color='k',linestyle='-',label=r'$k_{\rm rec}$ (V90)')[0],
                   ax.plot([],[],color='k',linestyle='--',label=r'$k_{\rm rec}$ (T21)')[0]]
    second_legend = ax.legend(handles=dummy_lines, loc='lower left', frameon=False, fontsize=14)
    ax.add_artist(second_legend)
    fig.subplots_adjust(top=0.98,bottom=0.12,left=0.11,right=0.98,hspace=0,wspace=0)
    fig.savefig(f'H2_pah_efficiency_{pah_bin["id"]}.pdf',format='pdf',dpi=300)

def plot_h2_formation(model,G0,ntmin,ntmax,Xe,T,f,xPAH=3.3e-5,n_nt=100,pah_bin_id=None,pah_bin_rank=0):
    from models.PAH_charge.PAH_photoelectric_heating import read_data,interpolate_linear,recombination_rate_Spitzer,recombination_rate_Tielens21
    from scipy.optimize import curve_fit
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="white")
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": "Computer Modern Roman",
    })
    nt = np.logspace(np.log10(ntmin),np.log10(ntmax),n_nt)
    nH = nt * (1. - f)
    ne = Xe * nH
    # 1. Setup the figure
    fig, ax = plt.subplots(1,1, figsize=(7,5),dpi=300,facecolor='w',edgecolor='k')
    ax.set_xlabel(r'$G_0/n_{\rm tot}$', fontsize=20)
    ax.set_ylabel(r'$R_{\rm H\,_{2}}^{\rm PAH}$ [cm$^3$ s$^{-1}$]', fontsize=20)
    ax.tick_params(labelsize=14)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.minorticks_on()
    ax.tick_params(which='both',axis="both",direction="in")
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    dist_pah, _ = _build_pah_distribution(pah_bin_id=pah_bin_id, pah_bin_rank=pah_bin_rank)
    
    # 2. Read the Montillaud data
    montillaud_fraction = read_sections('montillaud13_hydrogenation_fraction_C54.csv')
    params, _ = curve_fit(curve_hydro, np.log10(montillaud_fraction[1][0]), np.log10(montillaud_fraction[1][1]), maxfev=10000)           
    
    # 3. Compute the fraction of fully dehydrogenated PAHs and their charges at each G0
    f_dehydro = np.zeros(n_nt)
    f_anion = np.zeros(n_nt)
    f_neutral = np.zeros(n_nt)
    f_cation = np.zeros(n_nt)
    f_dication = np.zeros(n_nt)
    log_G0, log_ne, log_T, f_anion_matrix = read_data('./PAH_PEH_data/f_anion_%s_pah_%.4f_micron.dat'%(model,dist_pah.a0))
    log_G0, log_ne, log_T, f_neutral_matrix = read_data('./PAH_PEH_data/f_neutral_%s_pah_%.4f_micron.dat'%(model,dist_pah.a0))
    log_G0, log_ne, log_T, f_cation_matrix = read_data('./PAH_PEH_data/f_cation_%s_pah_%.4f_micron.dat'%(model,dist_pah.a0))
    log_G0, log_ne, log_T, f_dication_matrix = read_data('./PAH_PEH_data/f_dication_%s_pah_%.4f_micron.dat'%(model,dist_pah.a0))
    for i in range(0,n_nt):
        f_dehydro[i] = max(find_value(np.log10(G0),np.log10(nH[i]),*params),0.0)
        f_anion[i] = max(interpolate_linear(log_G0, log_ne, log_T, f_anion_matrix, G0, ne[i], T),0.0)
        f_neutral[i] = max(interpolate_linear(log_G0, log_ne, log_T, f_neutral_matrix, G0, ne[i], T),0.0)
        f_cation[i] = max(interpolate_linear(log_G0, log_ne, log_T, f_cation_matrix, G0, ne[i], T),0.0)
        f_dication[i] = max(interpolate_linear(log_G0, log_ne, log_T, f_dication_matrix, G0, ne[i], T),0.0)
    # 4. Compute the H2 photo-dissociation rate for each charge state (this is only for de-hydrogenated PAH)
    wav,sigma_abs_anion = absorption_cross_section(dist_pah,0)
    wav,sigma_abs_neu = absorption_cross_section(dist_pah,0)
    wav,sigma_abs_cation = absorption_cross_section(dist_pah,1)
    wav,sigma_abs_dication = absorption_cross_section(dist_pah,1)
    E0 = np.array([dissociation_parameters['Murga2020']['dehydrogenated']['H(Z<= 0)']['E0'],
          dissociation_parameters['Murga2020']['dehydrogenated']['H2']['E0']])
    S = np.array([dissociation_parameters['Murga2020']['dehydrogenated']['H(Z<= 0)']['S'],
          dissociation_parameters['Murga2020']['dehydrogenated']['H2']['S']])
    k_H2 = np.array([compute_h2_dissociation_rate((wav,sigma_abs_anion,G0,E0,S,dist_pah,-1)),
                     compute_h2_dissociation_rate((wav,sigma_abs_neu,G0,E0,S,dist_pah,0)),
                     compute_h2_dissociation_rate((wav,sigma_abs_cation,G0,E0,S,dist_pah,1)),
                     compute_h2_dissociation_rate((wav,sigma_abs_dication,G0,E0,S,dist_pah,2))])
    
    # 5. Compute the collisional rates
    k_col = np.array([7.8e-10 * nH,
                      nH * np.pi*(dist_pah.a0*1e-4)**2.*np.sqrt(8.*kb*T/(np.pi*mh))*np.exp(-60*1e-3*eV2erg/(kb*T)),
                      1.4e-10 * nH,
                      1.4e-10 * nH])
    
    # 5. Compute the Eley-Rideal rate
    k_er = 8.7e-13 * np.sqrt(T/100.) * nH
    
    
    # 6. Compute the efficiency of H2 formation
    eps_H2 = np.zeros((n_nt,3,4))
    
    
    for i in range(0,n_nt):
        eps_H2[i,0,:] = f_dehydro[i] * k_H2[:] / k_col[:,i]
        eps_H2[i,1,:] = (1. - f_dehydro[i]) * k_er[i] / k_col[:,i]
        eps_H2[i,2,:] = (1. - f_dehydro[i]) * k_er[i] / k_col[:,i] +  f_dehydro[i] * k_H2[:] / k_col[:,i]

    # 7. Compute the PAH density and plot
    nPAH = xPAH * nt
    RPAH = np.zeros((n_nt,3))
    for i in range(0, n_nt):
        RPAH[i,:] = 0.5 * (eps_H2[i,:,0] * k_col[0,i] * f_anion[i] +\
                            eps_H2[i,:,1] * k_col[1,i] * f_neutral[i]+\
                            eps_H2[i,:,2] * k_col[2,i] * f_cation[i]+\
                            eps_H2[i,:,3] * k_col[3,i] * f_dication[i])

    ax.plot(G0/nt,nPAH * RPAH[:,0],linestyle='-.',linewidth=2,color='r',label=r'Photo-dissociation')
    ax.plot(G0/nt,nPAH * RPAH[:,1],linestyle='--',linewidth=2,color='b',label=r'Eley-Rideal')
    ax.plot(G0/nt,nPAH * RPAH[:,2],linestyle='-',linewidth=2,color='k',label=r'Total')
        
    ax.legend(loc='best', frameon=False, fontsize=14)
    fig.subplots_adjust(top=0.98,bottom=0.12,left=0.14,right=0.98,hspace=0,wspace=0)
    fig.savefig('H2_pah_formation_rate.png',format='png',dpi=300)
    
def plot_superhydrogenation(width):
    from scipy.optimize import curve_fit
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="white")
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": "Computer Modern Roman",
    })
    
    fig, ax = plt.subplots(1,1, figsize=(7,5),dpi=300,facecolor='w',edgecolor='k')
    ax.set_xlabel(r'$n_{\rm H}$ [cm$^{-3}$]', fontsize=20)
    ax.set_ylabel(r'$G_0$', fontsize=20)
    ax.tick_params(labelsize=14)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.minorticks_on()
    ax.tick_params(which='both',axis="both",direction="in")
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    ax.plot(C96_SH03[:,0],C96_SH03[:,1],linestyle='--',color='k')
    ax.plot(C96_SH004[:,0],C96_SH004[:,1],linestyle=':',color='k')
    
    x_mean = 0.5 * (np.log10(C96_SH03[:,0]) + np.log10(C96_SH004[:,0]))
    y_mean = 0.5 * (np.log10(C96_SH03[:,1]) + np.log10(C96_SH004[:,1]))
    params, _ = curve_fit(curve_hydro, x_mean, y_mean, maxfev=10000)    
    new_x = np.linspace(x_mean.min(),x_mean.max(),100)
    new_y = curve_hydro(new_x,*params)
    print('Super-hydrogenated params: ',params)
    ax.plot(10**new_x,10**new_y,linestyle='-',color='k')
    
    G0 = np.linspace(0, 5,100)
    nH = np.linspace(0,5,100)
    new_z = np.zeros((100,100))
    for i in range(0,100):
        for j in range(0, 100):
            new_z[i,j] = find_value(nH[j],G0[i],*params,width=width)
    G0, nH = np.meshgrid(G0, nH)  # 2D grid for interpolation
    pc = ax.contourf(10**G0, 10**nH, new_z, levels=np.linspace(0, 1, 30), cmap='RdBu')
    fig.colorbar(pc,label='Value')
    
    fig.subplots_adjust(top=0.98,bottom=0.12,left=0.14,right=0.98,hspace=0,wspace=0)
    fig.savefig('test_superhydrogenation.png',format='png',dpi=300)