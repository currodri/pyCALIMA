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
import PAHs_model
import pandas as pd
import os
from tqdm import tqdm
import concurrent.futures
import time


# Set OMP_NUM_THREADS to limit the number of threads used by OpenBLAS
os.environ["OMP_NUM_THREADS"] = "1"  # Set it to the desired number of threads

# Constants
Delta_epsilon = 0.145 # [eV] - change in internal energy of PAH due to IR photon emission of a typical C-C mode
kb = 1.3806488e-16 # [erg/K] - Boltzmann constant
h = 6.62607015e-27 # [erg s] - Planck's constant
R = 1.987 # [cal/(mol K)] - Ideal gas constant

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
            'H(Z<       = 0)' : {'E0':1.4,'S':13.3},
            'H(Z>0)'    : {'E0':1.55,'S':13.3},
            'H2'        : {'E0':np.nan,'S':np.nan},
            'C2H2'      : {'E0':2.0,'S':10.0}
        }
    }
}

# Functions

def pre_exponential_factor(Te,S):
    """Pre-exponential factor for the Arrhenius law 
    in the Gibbs microcanonical distribution.

    Args:
        Te (float): effective temperature of the PAH in [K]
        S (float): change in entropy for the transition in [cal / (mol K)]

    Returns:
        float: pre-exponential factor in [1/s]
    """    
    
    return kb * Te / h * np.exp(1. + S / R)
    
def dissociation_rate(Te,E0,S):
    """Dissociation rate for a particular fragmentation E0 and S
    at a given effective temperature Te.

    Args:
        Te (float): effective temperature of the PAH in [K]
        E0 (float): binding energy of the fragment in [erg]
        S (float): change in entropy for the transition in [cal / (mol K)]

    Returns:
        float: dissociation rate in [1/s]
    """    
    
    k_factor = pre_exponential_factor(Te,S)
    
    return k_factor * np.exp(-E0 / (kb * Te))

def microcanonical_temperature(Eint,Natoms):
    """_summary_

    Args:
        Eint (_type_): _description_
        Natoms (_type_): _description_

    Returns:
        _type_: _description_
    """    
    
    Tm1 = 3750. * (Eint / (3.*Natoms - 6.))**0.45
    Tm2 = 11000. * (Eint / (3.*Natoms - 6.))**0.8
    
    return (Tm1**8. + Tm2**8.)**0.125
    