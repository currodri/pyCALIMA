"""
Physical constants and default parameters for galaxy SAM evolution.
"""

import numpy as np

# Solar abundances (Asplund et al. 2009)
ASPLUND_ABUNDANCES = {
    'H': 0.7381,
    'He': 0.2485,
    'C': 2.39e-3,
    'N': 6.99e-4,
    'O': 5.78e-3,
    'F': 5.09e-7,
    'Ne': 1.27e-3,
    'Mg': 7.16e-4,
    'Si': 6.70e-4,
    'S': 3.10e-4,
    'Fe': 1.30e-3,
}

# Anders & Grevesse (1989) solar abundances (alternative set)
ANDERS_GREVESSE_ABUNDANCES = {
    'H': 1.0,
    'He': 0.70683,
}

# Element names as used in different yield models
ELEMENTS_KOBAYASHI = ['p', 'He', 'C', 'N', 'O', 'F', 'Ne', 'Mg', 'Si', 'S', 'Fe']
ELEMENTS_LC18 = ['H', 'He', 'C', 'N', 'O', 'F', 'Ne', 'Mg', 'Si', 'S', 'Fe']

# Mass separatrix for intermediate/massive star division
MASS_SEPARATRIX = 8.0  # Solar masses

# Time scale (Hubble time in Gyr)
HUBBLE_TIME = 13.5  # Gyr

# Solar metallicity
ZSUN_ASPLUND = 0.01345
ZSUN_ANDERS = 0.01968

# Default photospheric logarithmic solar abundances (log(N/Ntotal) + 12)
LOG_SOLAR_ABUNDANCES = {
    'H': 12.00,
    'He': 10.93,
    'C': 8.43,
    'N': 7.83,
    'O': 8.69,
    'F': 4.56,
    'Ne': 7.93,
    'Mg': 7.60,
    'Si': 7.51,
    'S': 7.12,
    'Fe': 7.46,
}

# Yield models available
YIELD_MODELS = {
    'kobayashi': 'Kobayashi et al. 2006',
    'lc18': 'Limongi & Chieffi 2018',
    'karakas': 'Karakas 2010',
    'karakas_agb': 'Karakas 2010 - AGB only',
    'lc13': 'Limongi & Chieffi 2013',
}

# Default IMF parameters (Chabrier 2003)
IMF_TYPE = 'chabrier'
IMF_ALPHA_SALPETER = -2.35
IMF_BOUNDS_SALPETER = [0.1, 100.0]  # Solar masses
IMF_BOUNDS_CHABRIER = [0.1, 100.0]  # Solar masses

# Default parameters for galaxy SAM evolution
DEFAULT_PARAMS = {
    'nbint': 1000,  # Number of integration time steps
    'tscale_infall': 7.0,  # Infall time scale in Gyr
    'tscale_sfr': 2.2,  # Star formation time scale in Gyr
    'alphaks': 1.0,  # Power law index for Schmidt-Kennicutt relation
    'asnia': 0.05,  # Fraction of stars that become Type Ia SNe
    'agb_fraction': 0.15,  # Fraction of formed stellar mass that contributes to delayed AGB return
    'agb_delay_gyr': 1.0,  # Characteristic AGB delay time in Gyr
    'snia_delay_gyr': 1.5,  # Characteristic SNIa delay time in Gyr
    'snia_yield_model': 'nomoto84',  # Type Ia yield set
    'wind_loading': 0.0,  # Wind mass loading factor
    'mass_return': 0.0,  # Fixed mass return fraction
    'pristine_deuterium': 2.55e-5,  # Primordial D/H ratio
    'prefactor': 1e9,  # Prefactor for accretion or initial gas mass
    'accmodel': 1,  # Accretion model (1=exponential, 2=double exp, 3=no accretion)
    'minmload': 0.0,  # Minimum mass loading
    'maxmload': 10.0,  # Maximum mass loading
}

# SNe core collapse threshold
SNE_CC_THRESHOLD = 8.0  # Solar masses

# SNe Ia progenitor mass range
SNIA_PROGENITOR_MIN = 0.1  # Solar masses
SNIA_PROGENITOR_MAX = 8.0  # Solar masses
