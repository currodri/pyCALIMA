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
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times", "serif"],
})
import re
import time
from scipy.integrate import quad
from scipy.optimize import root_scalar
from models.dust_model import basic_s, build_distribution
from models.dust_radiation.dust_oppacity import dust_efficiencies
from models.PAH_radiation.pah_oppacity import pah_efficiencies
from models.grain_size_config import get_optical_props_path, get_repo_root, load_grain_size_config
from models.tools.radiation_fields import Draine_1978_isrf
from joblib import Parallel, delayed

PATH_OPTICS = str(get_optical_props_path())


def _resolve_collisional_dust_label(dust_type, collisional_dust_bin=None):
    """Resolve collisional-cooling table DustBin label.

    Table names follow `cooling_DustBin_XX_Z_Y`. Users can provide this
    explicitly through `collisional_dust_bin` (e.g. 'DustBin_00' or '00').
    If omitted, we infer the bin index from `dust_type` when available.
    """
    if collisional_dust_bin is not None:
        token = str(collisional_dust_bin).strip()
        m = re.search(r'(\d+)', token)
        if m is None:
            raise ValueError(f'Could not parse DustBin index from: {collisional_dust_bin}')
        return f'DustBin_{int(m.group(1)):02d}'

    dust_token = str(dust_type).lower()
    m = re.search(r'dustbin[_-]?(\d+)', dust_token)
    if m is not None:
        return f'DustBin_{int(m.group(1)):02d}'
    m = re.search(r'_bin_(\d+)', dust_token)
    if m is not None:
        return f'DustBin_{int(m.group(1)):02d}'

    # Backward-compatible fallback when no bin index exists in dust_type.
    return 'DustBin_00'


def _lookup_bin_metadata(dust_type):
    cfg = load_grain_size_config()
    token = str(dust_type).lower()
    for item in cfg['bins']:
        if str(item['id']).lower() == token:
            return dict(item)
    return None


def _resolve_optical_material(dust_type):
    meta = _lookup_bin_metadata(dust_type)
    if meta is not None:
        if meta['is_pah']:
            return 'pah'
        return str(meta['composition']).lower()

    dust_token = str(dust_type).lower()
    if 'silicate' in dust_token or 'sil' in dust_token:
        return 'silicate'
    if 'graphite' in dust_token or 'carbon' in dust_token or 'gra' in dust_token:
        return 'graphite'
    if 'pah' in dust_token:
        return 'pah'
    raise ValueError(f'Unsupported dust type for optical properties: {dust_type}')


def _resolve_distribution_species(dust_type):
    meta = _lookup_bin_metadata(dust_type)
    if meta is not None:
        return meta['id']
    return dust_type


def _read_precomputed_optical_properties(bin_id, optical_dir=None):
    """Read exported optical properties for one bin from model_data/optical_properties."""
    if optical_dir is None:
        optical_dir = os.path.join(str(get_repo_root()), 'model_data', 'optical_properties')

    file_path = os.path.join(optical_dir, f'averaged_cross_section_{bin_id}.txt')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'Optical-property file not found for {bin_id}: {file_path}')

    composition = None
    a0_micron = None
    data_rows = []
    in_table = False

    with open(file_path, 'r', encoding='utf-8') as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith('#'):
                lower_line = line.lower()
                if lower_line.startswith('# composition:'):
                    composition = line.split(':', 1)[1].strip().lower()
                elif lower_line.startswith('# grain size a0:'):
                    token = line.split(':', 1)[1].strip().split()[0]
                    a0_micron = float(token)
                elif lower_line.startswith('# columns:'):
                    in_table = True
                continue

            if in_table:
                values = [float(value) for value in line.split()]
                if len(values) >= 4:
                    data_rows.append(values[:4])

    if a0_micron is None or composition is None or len(data_rows) == 0:
        raise ValueError(f'Could not parse optical-property file for {bin_id}: {file_path}')

    data = np.asarray(data_rows, dtype=float)
    wavelengths = data[:, 0] * 1e-8  # Angstrom -> cm
    C_abs = data[:, 1]
    C_sca = data[:, 2]
    C_rp = data[:, 3]
    a0 = a0_micron * 1e-4  # micron -> cm
    return a0, wavelengths, C_sca, C_abs, C_rp, composition


def _compute_equilibrium_temperature_cheap_from_material(material, a, wavelengths, radiation_field, C_abs):
    abs_power = absorbed_power(wavelengths, radiation_field, C_abs)

    material_token = str(material).lower()
    if material_token == 'silicate':
        C_em = 4. * np.pi * a**2. * 1.3e-6 * (a * 1e4 / 0.1)
    elif material_token == 'graphite':
        C_em = 4. * np.pi * a**2. * 8e-7 * (a * 1e4 / 0.1)
    else:
        raise ValueError(f'Unsupported material for cheap temperature estimate: {material}')

    return (abs_power / (C_em * sigma_sb)) ** (1.0 / 6.0)


def _extract_phi0_rate(table_entry):
    logT = np.asarray(table_entry['logT'], dtype=float)
    logH = np.asarray(table_entry['logH'], dtype=float)

    if logH.ndim == 1:
        return logT, logH

    phi = np.asarray(table_entry.get('phi', []), dtype=float)
    if phi.size == 0:
        raise ValueError('2D collisional table is missing phi grid.')

    phi_idx = int(np.argmin(np.abs(phi)))
    return logT, logH[:, phi_idx]


# Asplund et al. (2009) photospheric abundances in log10 epsilon(X), with log10 epsilon(H)=12.
_ASPLUND2009_LOGEPS = {
    1: 12.00,
    2: 10.93,
    6: 8.43,
    7: 7.83,
    8: 8.69,
    10: 7.93,
    12: 7.60,
    14: 7.51,
    16: 7.12,
    26: 7.50,
}


def _asplund2009_number_abundances():
    """Return number abundances n_X / n_H from Asplund et al. (2009)."""
    return {z: 10.0**(logeps - 12.0) for z, logeps in _ASPLUND2009_LOGEPS.items()}


def _collect_collisional_tables_for_dustbin(coll_tables, dust_label):
    """Collect available collisional tables for one DustBin keyed by atomic number Z."""
    pattern = re.compile(rf'^cooling_{re.escape(dust_label)}_Z_(\d+)$')
    z_to_table = {}
    for key, entry in coll_tables.items():
        match = pattern.match(str(key))
        if match is None:
            continue
        z = int(match.group(1))
        z_to_table[z] = _extract_phi0_rate(entry)

    if len(z_to_table) == 0:
        raise KeyError(f'No collisional tables found for {dust_label}.')

    return z_to_table


def _solar_projectile_densities_from_nH(nH, available_Z):
    """Build projectile number densities from nH using Asplund (2009) abundances.

    Assumption: all species are ionized enough that electron density can be
    estimated from charge neutrality using the included ions.
    """
    abund = _asplund2009_number_abundances()
    nproj = {}

    for z in available_Z:
        if z == 0:
            continue
        nproj[z] = nH * abund.get(z, 0.0)

    n_e = 0.0
    for z, n_z in nproj.items():
        n_e += z * n_z
    nproj[0] = n_e

    return nproj


def _compute_collisional_coupling_multiz(Tgas, z_to_table, z_to_density):
    """Compute K_coll so Hcoll = K_coll * (Tgas - Tdust), summing all available Z."""
    lT = np.log10(Tgas)
    K_coll = 0.0
    for z, (logT, logH) in z_to_table.items():
        density = float(z_to_density.get(z, 0.0))
        if density <= 0.0:
            continue
        rate = 10.0**np.interp(lT, logT, logH)
        K_coll += density * rate
    return K_coll


def _solve_eqT_from_absorption_and_coupling(absorbed, wavelengths_em, C_abs_em,
                                            Tgas, K_coll, method='linearized',
                                            T0=30.0, Tmin=2.7, Tmax=800.0):
    """Solve absorbed + K*(Tgas-T) = emitted(T) for equilibrium dust temperature."""
    method_token = str(method).lower()

    def f(T):
        return absorbed + K_coll * (Tgas - T) - emitted_power(T, wavelengths_em, C_abs_em)

    if method_token == 'linearized':
        # Iterative local linearization of emitted power (re-linearized each iteration).
        T = float(np.clip(T0, Tmin, Tmax))
        for _ in range(100):
            emitted_T = emitted_power(T, wavelengths_em, C_abs_em)
            d_emitted = planck_function_derivative(wavelengths_em, T)
            d_emitted_power = 4.0 * np.pi * np.trapezoid(C_abs_em * d_emitted, x=wavelengths_em)

            denom = K_coll + d_emitted_power
            if (not np.isfinite(emitted_T)) or (not np.isfinite(denom)) or (denom <= 0.0):
                break

            T_new = (
                absorbed + K_coll * Tgas - emitted_T + d_emitted_power * T
            ) / denom
            T_new = float(np.clip(T_new, Tmin, Tmax))
            if abs(T_new - T) / max(abs(T), 1e-12) < 1e-4:
                return T_new
            T = T_new

    elif method_token == 'newton':
        T = float(np.clip(T0, Tmin, Tmax))
        for _ in range(100):
            f_val = f(T)
            d_emitted = planck_function_derivative(wavelengths_em, T)
            d_emitted_power = 4.0 * np.pi * np.trapezoid(C_abs_em * d_emitted, x=wavelengths_em)
            df_val = -K_coll - d_emitted_power

            if (not np.isfinite(f_val)) or (not np.isfinite(df_val)) or (df_val == 0.0):
                break

            T_new = T - f_val / df_val
            T_new = float(np.clip(T_new, Tmin, Tmax))
            if abs(T_new - T) / max(abs(T), 1e-12) < 1e-4:
                return T_new
            T = T_new
    else:
        raise ValueError(f'Unknown solver method: {method}. Use "linearized" or "newton".')

    f_min = f(Tmin)
    f_max = f(Tmax)
    if np.sign(f_min) == np.sign(f_max):
        return Tmin if abs(f_min) <= abs(f_max) else Tmax

    result = root_scalar(f, bracket=[Tmin, Tmax])
    if result.converged:
        return result.root
    raise RuntimeError('Failed to solve multiz collisional equilibrium temperature.')


def _solve_one_cell(nH, Tgas, z_to_table, available_Z,
                   absorbed, wavelengths_em, C_abs_em_interp,
                   method_token, T0_guess):
    """Module-level worker for Parallel: solve one (nH, Tgas) grid cell."""
    z_to_density = _solar_projectile_densities_from_nH(nH, available_Z)
    K_coll = _compute_collisional_coupling_multiz(Tgas, z_to_table, z_to_density)
    Tmax_solver = max(float(Tgas), 800.0)
    try:
        return _solve_eqT_from_absorption_and_coupling(
            absorbed, wavelengths_em, C_abs_em_interp,
            Tgas, K_coll, method=method_token, T0=T0_guess,
            Tmax=Tmax_solver,
        )
    except Exception:
        return np.nan


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
        dust_type (str): The type of dust/bin to be used. This can be:
        - silicate_bin_0 / silicate_bin_1
        - graphite_bin_0 / graphite_bin_1
        - pah_ion_bin_0 / pah_ion_bin_1
        - pah_neutral_bin_0 / pah_neutral_bin_1
        do_average (bool, optional): Whether or not to average of the assumed distribution. Defaults to True.

    Returns:
        np.array,np.array,np.array,np.array: The scattering, absorption and radiation pressure cross sections
    """    
    
    
    # 1. Read the efficiencies
    dust_token = str(dust_type).lower()
    optical_material = _resolve_optical_material(dust_type)

    if optical_material == 'silicate':
        filename = os.path.join(PATH_OPTICS, 'draine_lee_1984', 'suvSil_81')
        nwav,data, columns, name = dust_efficiencies(filename)
    elif optical_material == 'graphite':
        filename = os.path.join(PATH_OPTICS, 'draine_lee_1984', 'Gra_81')
        nwav,data, columns, name = dust_efficiencies(filename)
    elif optical_material == 'pah' and 'ion' in dust_token:
        filename = os.path.join(PATH_OPTICS, 'li_draine_2001', 'PAHion_30')
        nwav,data,columns,dust_type = pah_efficiencies(filename)
    elif optical_material == 'pah':
        filename = os.path.join(PATH_OPTICS, 'li_draine_2001', 'PAHneu_30')
        nwav,data,columns,name = pah_efficiencies(filename)
    else:
        raise ValueError('Dust type not recognised: ', dust_type)

    # 2. Setup the underlying distribution
    dist = build_distribution(_resolve_distribution_species(dust_type))
    
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
    
def interpolate_cross_sections(dust_type, grain_size,efficiency=False,
                               data_table=None):
    # 1. Read the efficiencies
    if data_table is None:
        if dust_type == 'silicate':
            filename = os.path.join(PATH_OPTICS, 'draine_lee_1984', 'suvSil_81')
            nwav,data, columns, name = dust_efficiencies(filename)
        elif dust_type == 'graphite':
            filename = os.path.join(PATH_OPTICS, 'draine_lee_1984', 'Gra_81')
            nwav,data, columns, name = dust_efficiencies(filename)
        elif dust_type == 'iPAH':
            filename = os.path.join(PATH_OPTICS, 'li_draine_2001', 'PAHion_30')
            nwav,data,columns,dust_type = pah_efficiencies(filename)
        elif dust_type == 'nPAH':
            filename = os.path.join(PATH_OPTICS, 'li_draine_2001', 'PAHneu_30')
            nwav,data,columns,name = pah_efficiencies(filename)
        elif dust_type == 'PAH':
            filename = os.path.join(PATH_OPTICS, 'li_draine_2001', 'PAHneu_30')
            nwav,data,columns,name = pah_efficiencies(filename)
        else:
            raise ValueError('Dust type not recognised: ',dust_type)
    else:
        nwav,data,columns,name = data_table
    
    # Compute the cross section by looking for the nearest grain size
    # in the data dictionary. If not found, interpolate
    C_sca = np.zeros(len(data[list(data.keys())[0]][:,columns.index('Q_sca')]))
    C_abs = np.zeros(len(data[list(data.keys())[0]][:,columns.index('Q_abs')]))
    C_rp = np.zeros(len(data[list(data.keys())[0]][:,columns.index('Q_abs')]))
    wavelengths = data[list(data.keys())[0]][:,columns.index('w(micron)')]

    # Check if the size grain_size is in the data dictionary
    if str(grain_size) in data:
        C_sca = data[str(grain_size)][:,columns.index('Q_sca')]
        C_abs = data[str(grain_size)][:,columns.index('Q_abs')]
        g = data[str(grain_size)][:,columns.index('g=<cos>')]
        C_rp = C_abs + (1-g)*C_sca
    else:
        # Interpolate
        for i in range(0,len(C_sca)):
            a = np.array([float(r) for r in data.keys()])
            Q_sca = np.array([d[i,columns.index('Q_sca')] for d in data.values()])
            Q_abs = np.array([d[i,columns.index('Q_abs')] for d in data.values()])
            g = np.array([d[i,columns.index('g=<cos>')] for d in data.values()])
            if efficiency:
                C_sca[i] = 10.**np.interp(np.log10(grain_size),np.log10(a),np.log10(Q_sca))
                C_abs[i] = 10.**np.interp(np.log10(grain_size),np.log10(a),np.log10(Q_abs))
            else:
                C_sca[i] = 10.**np.interp(np.log10(grain_size),np.log10(a),np.log10(Q_sca)) * np.pi * grain_size**2
                C_abs[i] = 10.**np.interp(np.log10(grain_size),np.log10(a),np.log10(Q_abs)) * np.pi * grain_size**2
            g = np.interp(grain_size,a,g)
            C_rp[i] = C_abs[i] + (1-g)*C_sca[i]

    wavelengths = wavelengths*1e-4
    grain_size = grain_size*1e-4
    if not efficiency:
        C_sca = C_sca* 1e-8
        C_abs = C_abs* 1e-8
        C_rp = C_rp* 1e-8

    return grain_size,wavelengths,C_sca,C_abs,C_rp


def planck_function(wavelength, T):
    """This function computes the Planck function for a given wavelength

    Args:
        wavelength (np.array): The wavelength in cm
        T (np.float): The temperature in K

    Returns:
        np.float: Emittance in erg/s/cm^2/cm/steradian
    """    
    x = (h * c / wavelength) / (kb * T)
    x = np.clip(x, 0.0, 700.0)
    return (2. * h * c**2. / wavelength**5.) / np.expm1(x)

def planck_function_derivative(wavelength, T):
    """This function computes the derivative of the Planck function for a given wavelength

    Args:
        wavelength (np.array): The wavelength in cm
        T (np.float): The temperature in K

    Returns:
        np.float: Derivative of emittance in erg/s/cm^2/cm/steradian/K
    """    
    x = (h * c / wavelength) / (kb * T)
    x = np.clip(x, 0.0, 700.0)
    prefactor = (2. * h * c**2. / wavelength**5.)
    expm1_x = np.expm1(x)
    B_lambda = prefactor / expm1_x
    correction = 1.0 + 1.0 / expm1_x
    return B_lambda * (x / T) * correction

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
    absorbed_power = np.trapezoid(radiation_field * C_abs, x=wavelengths)
    
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
    emitted_power = 4. * np.pi * np.trapezoid(C_abs * emp, x=wavelengths)
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
    from models.dust_gas_collisions.dust_collisional_cooling import compute_dust_coll_heating
    
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
    from models.dust_gas_collisions.dust_collisional_cooling import compute_dust_coll_heating

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
    return T_eq

def compute_eqT_withcollisions_newton_linearized(dust_type,a,wavelengths,wavelengths_em,
                                                 radiation_field,C_abs,C_abs_em,
                                                 ne,nH,nHe,nC,Tgas,T_dust_collisional,
                                                 electron_rate_table,H_rate_table,
                                                 He_rate_table,C_rate_table,
                                                 T0=None,dT_frac=1e-2):
    """Compute equilibrium temperature using a linearized collisional heating term.

    The collisional contribution is approximated around `T0` as
    Hcoll(T) ≈ H0 + dH_dT * (T - T0),
    while the dust emission and radiative absorption are still evaluated exactly.
    """
    from models.dust_gas_collisions.dust_collisional_cooling import compute_dust_coll_heating

    if T0 is None:
        T0 = compute_equilibrium_temperature_cheap(dust_type, a, wavelengths, radiation_field, C_abs)

    absorbed = absorbed_power(wavelengths, radiation_field, C_abs)
    H0 = compute_dust_coll_heating(ne, nH, nHe, nC, Tgas, T0, T_dust_collisional,
                                   electron_rate_table, H_rate_table,
                                   He_rate_table, C_rate_table)

    dT = max(abs(T0) * dT_frac, 1e-6)
    H_plus = compute_dust_coll_heating(ne, nH, nHe, nC, Tgas, T0 + dT, T_dust_collisional,
                                       electron_rate_table, H_rate_table,
                                       He_rate_table, C_rate_table)
    H_minus = compute_dust_coll_heating(ne, nH, nHe, nC, Tgas, max(T0 - dT, 1e-8), T_dust_collisional,
                                        electron_rate_table, H_rate_table,
                                        He_rate_table, C_rate_table)
    dH_dT = (H_plus - H_minus) / (2.0 * dT)

    # Iterative local linearization of emitted power while Hcoll remains linearized around T0.
    T = float(np.clip(T0, 2.7, 800.0))
    for _ in range(100):
        Hcoll_linear = H0 + dH_dT * (T - T0)
        emitted_T = emitted_power(T, wavelengths_em, C_abs_em)
        d_emitted = planck_function_derivative(wavelengths_em, T)
        d_emitted_power = 4. * np.pi * np.trapezoid(C_abs_em * d_emitted, x=wavelengths_em)

        denom = d_emitted_power - dH_dT
        if (not np.isfinite(emitted_T)) or (not np.isfinite(denom)) or (denom <= 0.0):
            break

        # Solve linearized balance for the next iterate.
        T_new = T + (absorbed + Hcoll_linear - emitted_T) / denom

        # Keep updates in a physical range; shrink step if needed.
        for _ in range(12):
            if np.isfinite(T_new) and (2.7 <= T_new <= 800.0):
                break
            T_new = 0.5 * (T_new + T)

        if abs(T_new - T) / max(abs(T), 1e-12) < 1e-4:
            return T_new
        T = T_new

    # Robust fallback.
    def f_fallback(Tval):
        Hcoll_linear = H0 + dH_dT * (Tval - T0)
        return absorbed + Hcoll_linear - emitted_power(Tval, wavelengths_em, C_abs_em)

    result = root_scalar(f_fallback, bracket=[2.7, 800.0])
    if result.converged:
        return result.root
    raise RuntimeError("Failed to solve linearized collisional equilibrium temperature")


def compute_collision_only_thermal_equilibration(dust_type, Tgas, Tdust0,
                                                 ne, nH, nHe, nC,
                                                 tolerance=0.01,
                                                 specific_heat=None,
                                                 collisional_dust_bin=None,
                                                 table_dir=None,
                                                 n_steps=400,
                                                 max_time_s=None,
                                                 time_sampling='log'):
    """Estimate how fast a dust grain thermally equilibrates with the gas.

    This assumes no radiative heating/cooling and uses only collisional energy exchange.
    Since the implemented collisional term is linear in (Tgas - Tdust), the solution is
    exponential with characteristic timescale tau = C_grain / K_coll.

    Args:
        dust_type (str): Dust label, e.g. "silicate_bin_00" or "graphite_bin_00".
        Tgas (float): Gas temperature [K], assumed constant.
        Tdust0 (float): Initial dust temperature [K].
        ne, nH, nHe, nC (float): Number densities [cm^-3].
        tolerance (float, optional): |Tdust - Tgas| threshold [K] used to define "equilibrium".
        specific_heat (float, optional): Grain specific heat [erg g^-1 K^-1]. Defaults to 1e7.
        collisional_dust_bin (str, optional): Collisional table bin label or index,
            e.g. 'DustBin_00' or '00'. If None, inferred from `dust_type`.
        table_dir (str, optional): Directory of collisional cooling tables.
        n_steps (int, optional): Number of time samples in the returned history.
        max_time_s (float, optional): Maximum integration time [s]. Auto-set if None.
        time_sampling (str, optional): "log" (default) or "linear" sampling for
            the returned time history.

    Returns:
        dict: {
            'tau_s': characteristic timescale [s],
            'time_to_tolerance_s': time to reach tolerance [s],
            'time_to_tolerance_yr': same in years,
            'times_s': sampled times [s],
            'Tdust_history_K': sampled temperatures [K],
            'Tgas_K': gas temperature [K],
            'Tdust0_K': initial dust temperature [K],
            'tolerance_K': tolerance [K],
            'grain_mass_g': grain mass [g],
            'grain_heat_capacity_erg_per_K': grain heat capacity [erg/K],
            'collisional_coupling_erg_per_s_per_K': coupling coefficient K_coll [erg/s/K],
        }
    """
    from models.dust_gas_collisions.dust_collisional_cooling import load_cooling_tables

    if Tgas <= 0 or Tdust0 <= 0:
        raise ValueError('Tgas and Tdust0 must be positive.')
    if tolerance <= 0:
        raise ValueError('tolerance must be positive.')
    if n_steps < 2:
        raise ValueError('n_steps must be >= 2.')
    if str(time_sampling).lower() not in ('log', 'linear'):
        raise ValueError('time_sampling must be "log" or "linear".')

    # 1) Grain size/mass and heat capacity
    a0, _, _, _, _ = compute_cross_sections(dust_type, do_average=False)
    dust_token = str(dust_type).lower()
    if 'silicate' in dust_token:
        grain_density = 3.5  # g/cm^3
    elif ('graphite' in dust_token) or ('carbon' in dust_token) or ('pah' in dust_token):
        grain_density = 2.2  # g/cm^3
    else:
        raise ValueError(f'Unsupported dust type for equilibration timescale: {dust_type}')

    grain_mass = 4.0 / 3.0 * np.pi * a0**3.0 * grain_density
    if specific_heat is None:
        specific_heat = 1.0e7
    grain_heat_capacity = grain_mass * specific_heat

    # 2) Collisional coupling coefficient K_coll so that Hcoll = K_coll * (Tgas - Tdust)
    dust_label = _resolve_collisional_dust_label(dust_type, collisional_dust_bin=collisional_dust_bin)
    if table_dir is None:
        table_dir = os.path.join(str(get_repo_root()), 'model_data', 'collisional_cooling_data')

    coll_tables = load_cooling_tables(table_dir=table_dir)
    electron_entry = coll_tables[f'cooling_{dust_label}_Z_0']
    H_entry = coll_tables[f'cooling_{dust_label}_Z_1']
    He_entry = coll_tables[f'cooling_{dust_label}_Z_2']
    C_entry = coll_tables[f'cooling_{dust_label}_Z_6']

    T_dust_collisional, electron_rate_table = _extract_phi0_rate(electron_entry)
    _, H_rate_table = _extract_phi0_rate(H_entry)
    _, He_rate_table = _extract_phi0_rate(He_entry)
    _, C_rate_table = _extract_phi0_rate(C_entry)

    lT = np.log10(Tgas)
    electron_cool = 10.0**np.interp(lT, T_dust_collisional, electron_rate_table)
    H_cool = 10.0**np.interp(lT, T_dust_collisional, H_rate_table)
    He_cool = 10.0**np.interp(lT, T_dust_collisional, He_rate_table)
    C_cool = 10.0**np.interp(lT, T_dust_collisional, C_rate_table)

    K_coll = ne * electron_cool + nH * H_cool + nHe * He_cool + nC * C_cool
    if K_coll <= 0:
        raise RuntimeError('Collisional coupling is non-positive; cannot estimate equilibration timescale.')

    tau_s = grain_heat_capacity / K_coll

    # 3) Exponential approach to Tgas and time-to-tolerance
    delta0 = abs(Tdust0 - Tgas)
    if delta0 <= tolerance:
        time_to_tolerance_s = 0.0
    else:
        time_to_tolerance_s = tau_s * np.log(delta0 / tolerance)

    if max_time_s is None:
        if time_to_tolerance_s > 0:
            max_time_s = 1.25 * time_to_tolerance_s
        else:
            max_time_s = 5.0 * tau_s

    if max_time_s <= 0:
        times_s = np.zeros(int(n_steps), dtype=float)
    elif str(time_sampling).lower() == 'linear':
        times_s = np.linspace(0.0, max_time_s, int(n_steps))
    else:
        # Keep t=0 for the exact initial condition and use log spacing afterwards
        # to resolve the early-time approach when plotting on a log-time axis.
        n_tail = int(n_steps) - 1
        tmin = max_time_s * 1.0e-8
        tail = np.logspace(np.log10(tmin), np.log10(max_time_s), n_tail)
        times_s = np.concatenate(([0.0], tail))

    Tdust_history = Tgas - (Tgas - Tdust0) * np.exp(-times_s / tau_s)

    return {
        'tau_s': tau_s,
        'time_to_tolerance_s': time_to_tolerance_s,
        'time_to_tolerance_yr': time_to_tolerance_s / (3600.0 * 24.0 * 365.25),
        'times_s': times_s,
        'Tdust_history_K': Tdust_history,
        'Tgas_K': Tgas,
        'Tdust0_K': Tdust0,
        'tolerance_K': tolerance,
        'grain_mass_g': grain_mass,
        'grain_heat_capacity_erg_per_K': grain_heat_capacity,
        'collisional_coupling_erg_per_s_per_K': K_coll,
    }


def plot_collision_only_thermal_equilibration(dust_type, Tdust0,
                                              Tgas_values,
                                              density_scalings,
                                              ne_ref, nH_ref, nHe_ref, nC_ref,
                                              tolerance=0.01,
                                              specific_heat=None,
                                              collisional_dust_bin=None,
                                              table_dir=None,
                                              n_steps=400,
                                              max_time_s=None,
                                              time_unit='yr',
                                              output_dir=None,
                                              filename=None):
    """Plot collision-only dust temperature relaxation for multiple gas cases.

    Args:
        dust_type (str): Dust label, e.g. "silicate_bin_00".
        Tdust0 (float): Initial dust temperature [K].
        Tgas_values (array-like): Gas temperatures [K] to test.
        density_scalings (array-like): Multiplicative factors applied to
            (ne_ref, nH_ref, nHe_ref, nC_ref).
        ne_ref, nH_ref, nHe_ref, nC_ref (float): Reference number densities [cm^-3].
        tolerance (float, optional): Equilibrium threshold in |Tdust-Tgas| [K].
        specific_heat (float, optional): Grain specific heat [erg g^-1 K^-1].
        collisional_dust_bin (str, optional): Collisional table bin label or index,
            e.g. 'DustBin_00' or '00'. If None, inferred from `dust_type`.
        table_dir (str, optional): Collisional cooling table directory.
        n_steps (int, optional): Number of samples in each temperature history.
        max_time_s (float, optional): Time extent [s] passed to each run.
        time_unit (str, optional): "s", "yr", or "kyr".
        output_dir (str, optional): Directory where figure is saved.
        filename (str, optional): Output filename. Auto-generated if None.

    Returns:
        tuple: (results, output_path)
            - results is a list of dictionaries with one entry per curve.
            - output_path is the saved figure path.
    """

    Tgas_values = np.asarray(Tgas_values, dtype=float)
    density_scalings = np.asarray(density_scalings, dtype=float)

    if Tgas_values.size == 0 or density_scalings.size == 0:
        raise ValueError('Tgas_values and density_scalings must be non-empty.')

    unit = str(time_unit).lower()
    if unit == 's':
        time_scale = 1.0
        time_label = 't [s]'
    elif unit == 'yr':
        time_scale = 3600.0 * 24.0 * 365.25
        time_label = 't [yr]'
    elif unit == 'kyr':
        time_scale = 3600.0 * 24.0 * 365.25 * 1.0e3
        time_label = 't [kyr]'
    else:
        raise ValueError('time_unit must be one of: s, yr, kyr')

    fig, ax = plt.subplots(1, 1, figsize=(7, 5), dpi=300, facecolor='w', edgecolor='k')
    ax.set_xlabel(time_label, fontsize=16)
    ax.set_ylabel(r'$T_{\rm dust}$ [K]', fontsize=16)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.minorticks_on()
    ax.tick_params(which='both', axis='both', direction='in')

    cmap = plt.get_cmap('viridis')
    color_positions = np.linspace(0.1, 0.95, max(len(Tgas_values), 2))
    linestyles = ['-', '--', '-.', ':']

    results = []
    for i, Tgas in enumerate(Tgas_values):
        color = cmap(color_positions[i if len(Tgas_values) > 1 else 0])
        for j, scale in enumerate(density_scalings):
            ne = ne_ref * scale
            nH = nH_ref * scale
            nHe = nHe_ref * scale
            nC = nC_ref * scale

            run = compute_collision_only_thermal_equilibration(
                dust_type=dust_type,
                Tgas=Tgas,
                Tdust0=Tdust0,
                ne=ne,
                nH=nH,
                nHe=nHe,
                nC=nC,
                tolerance=tolerance,
                specific_heat=specific_heat,
                collisional_dust_bin=collisional_dust_bin,
                table_dir=table_dir,
                n_steps=n_steps,
                max_time_s=max_time_s,
            )

            label = f'Tgas={Tgas:.3g} K, n-scale={scale:.3g}'
            ax.plot(run['times_s'] / time_scale,
                    run['Tdust_history_K'],
                    color=color,
                    linestyle=linestyles[j % len(linestyles)],
                    linewidth=2.0,
                    label=label)

            results.append({
                'Tgas_K': Tgas,
                'density_scale': scale,
                'ne_cm3': ne,
                'nH_cm3': nH,
                'nHe_cm3': nHe,
                'nC_cm3': nC,
                'tau_s': run['tau_s'],
                'time_to_tolerance_s': run['time_to_tolerance_s'],
                'time_to_tolerance_yr': run['time_to_tolerance_yr'],
                'times_s': run['times_s'],
                'Tdust_history_K': run['Tdust_history_K'],
            })

            print(
                f"Tgas={Tgas:.6g} K | n-scale={scale:.6g} | "
                f"tau={run['tau_s']:.3e} s | "
                f"t_eq(|Td-Tg|<{tolerance:g}K)={run['time_to_tolerance_s']:.3e} s"
            )

    ax.legend(loc='best', fontsize=9, frameon=False)
    fig.subplots_adjust(top=0.98, bottom=0.13, left=0.12, right=0.98, hspace=0, wspace=0)

    if output_dir is None:
        output_dir = os.path.join(str(get_repo_root()), 'model_data', 'optical_properties')
    os.makedirs(output_dir, exist_ok=True)

    if filename is None:
        filename = f'collision_only_relaxation_{dust_type}.pdf'
    output_path = os.path.join(output_dir, filename)
    fig.savefig(output_path, format='pdf', dpi=300)

    return results, output_path

def compute_equilibrium_temperature_cheap(dust_type,a,wavelengths,radiation_field,C_abs):
    
    # 1. Compute the absorbed power
    abs_power = absorbed_power(wavelengths,radiation_field,C_abs)

    # 2. Compute the emission cross-section based on the approximations by Draine 2008 (eqs. 24.15 and 24.16)
    optical_material = _resolve_optical_material(dust_type)
    if optical_material == 'silicate':
        C_em = 4. * np.pi * (a)**2. * 1.3e-6 * (a*1e4/0.1)
    elif optical_material == 'graphite':
        C_em = 4. * np.pi * (a)**2. * 8e-7 * (a*1e4/0.1)
    else:
        raise ValueError(f'Unsupported dust type for cheap temperature estimate: {dust_type}')

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
        B_lambda = planck_function(wavelength, T)
        u_lambda_optical += (4 * np.pi / c) * W * B_lambda

    # CMB component
    T_CMB = 2.725  # CMB temperature in K
    B_lambda_CMB = planck_function(wavelength, T_CMB)
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
        a0, wavelengths,C_sca,C_abs,C_rp = compute_cross_sections(dust_type,do_average=True)
        C_abs_interp = np.interp(radiation_field[:,0],wavelengths[::-1],C_abs[::-1])
        C_abs_em_interp = np.interp(wavelengths_em,wavelengths[::-1],C_abs[::-1])
        print('Absorption cross section for',dust_type,'computed')
        # --- Precompute Planck-mean opacity table for this dust type ---
        # Determine grain material density for mass calculation
        if 'Sil' in dust_type:
            grain_density = basic_s[5]
        elif 'C' in dust_type:
            grain_density = basic_s[2]
        else:
            grain_density = basic_s[2]
        # a0 is returned in cm by compute_cross_sections
        m_grain = 4.0/3.0 * np.pi * grain_density * (a0)**3.0
        # Compute mass absorption coefficient kappa_abs(λ) [cm^2/g]
        kappa_abs = C_abs / m_grain
        # Temperature grid for Planck-mean table
        planck_temps = np.logspace(np.log10(1.0), np.log10(1000.0), 100)
        # Precompute Planck-mean opacities for this dust type
        planck_kappa = np.zeros_like(planck_temps)
        for ii, Tval in enumerate(planck_temps):
            planck_kappa[ii] = compute_Planck_oppacity(wavelengths, kappa_abs, Tval)
        print('Precomputed Planck-mean table for', dust_type)
        print('Temperatures from', planck_temps[0], 'to', planck_temps[-1], 'K')
        print('Pem from', 4.0 * sigma_sb * planck_kappa[0]*m_grain*planck_temps[0]**4., 'to', 4.0 * sigma_sb * planck_kappa[-1]*m_grain*planck_temps[-1]**4., 'erg/s')
        # 3B. Compute the radiation field averaged cross section
        int_radfield = np.trapezoid(radiation_field[:,1],x=radiation_field[:,0])
        C_abs_avg = np.trapezoid(C_abs_interp * radiation_field[:,1],x=radiation_field[:,0]) / int_radfield /(np.pi*a0**2.)
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
        def compute_temp_fast(i):
            return compute_equilibrium_temperature_planck_table_fast(
                                radiation_field[:,0],
                                G0[i]*radiation_field[:,1],
                                C_abs_interp,
                                planck_temps, planck_kappa, m_grain,
                                Tmin=planck_temps[0], Tmax=planck_temps[-1])

        Teq = Parallel(n_jobs=-1)(delayed(compute_temp)(i) for i in range(nG0))

        Teq_cheap = Parallel(n_jobs=-1)(delayed(compute_temp_fast)(i) for i in range(nG0))
    
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

    # List of line styles for the number of dust types
    linestyles = ['-','--','-.',':']

    import matplotlib as mpl
    import matplotlib.cm as cm

    # Store spectra so we can color them by Teq consistently across dust types
    spectra_to_plot = []
    Teq_values = []

    for idx, dust_type in enumerate(dust_types):
        # 3A. Obtain the absorption cross section and interpolate over the wavelengths
        a0, wavelengths, C_sca, C_abs, C_rp = compute_cross_sections(dust_type, do_average=False)
        C_abs_interp = np.interp(radiation_field[:, 0], wavelengths[::-1], C_abs[::-1])
        C_abs_em_interp = np.interp(wavelengths_em, wavelengths[::-1], C_abs[::-1])
        print('Absorption cross section for', dust_type, 'computed')
        # 3B. Compute the radiation field averaged cross section
        int_radfield = np.trapezoid(radiation_field[:, 1], x=radiation_field[:, 0])
        C_abs_avg = np.trapezoid(C_abs_interp * radiation_field[:, 1], x=radiation_field[:, 0]) / int_radfield / (np.pi * a0**2.)
        print('Average absorption cross section for', dust_type, 'computed')
        print('Given by', C_abs_avg)
        linestyle = linestyles.pop()
        
        # 3C. Compute the equilibrium temperature and store spectra
        for g0 in G0:
            Teq = compute_equilibrium_temperature(radiation_field[:, 0],
                                                  wavelengths_em,
                                                  g0 * radiation_field[:, 1],
                                                  C_abs_interp, C_abs_em_interp)
            # 3D. Compute the emitted power
            L_lambda = np.zeros(len(wavelengths_em))
            for i in range(0, len(wavelengths_em)):
                L_lambda[i] = wavelengths_em[i] * planck_function(wavelengths_em[i], Teq) * C_abs_em_interp[i]
            spectra_to_plot.append((wavelengths_em * 1e4, 4. * np.pi * L_lambda, Teq, linestyle))
            Teq_values.append(Teq)
        # Add legend entry for the dust type with black color
        ax.plot([], [], label=dust_type, color='k', linestyle=linestyle, linewidth=2.5)

    # 4. Plot all spectra using a temperature-dependent color map
    Teq_values = np.asarray(Teq_values)
    positive_Teq = Teq_values[Teq_values > 0.0]
    if len(positive_Teq) == 0:
        raise ValueError('All equilibrium temperatures are non-positive; cannot use LogNorm.')

    Tmin = np.min(positive_Teq)
    Tmax = np.max(positive_Teq)
    if np.isclose(Tmin, Tmax):
        Tmin = Tmin * (1.0 - 1e-6)
        Tmax = Tmax * (1.0 + 1e-6)

    norm = mpl.colors.LogNorm(vmin=Tmin, vmax=Tmax)
    cmap = cm.viridis

    for wavelength_micron, spectrum, Teq, linestyle in spectra_to_plot:
        ax.plot(wavelength_micron, spectrum, label=None,
                color=cmap(norm(Teq)), linestyle=linestyle, linewidth=2.5)

    # Add colorbar for equilibrium temperature
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.02)
    cbar.set_label(r'$T_{\rm eq}$ [K]', fontsize=20)

    # 5. Finalise the figure and save
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
        int_radfield = np.trapezoid(radiation_field[:, 1], x=radiation_field[:, 0])
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
                    luminosity_prima[band['band_name']][g0_idx] = 4. * np.pi * np.trapezoid(L_lambda, x=wavelength)
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

    denominator = np.trapezoid(integrand_denominator,x=wavelengths)
    numerator = np.trapezoid(integrand_numerator,x=wavelengths)

    return numerator / denominator


def compute_Planck_oppacity(wavelengths, kappa_abs, Td):
    """This function computes the Planck mean opacity given the absorption mass
    cross section and the dust temperature.

    Args:
        wavelengths (np.array): The wavelength in cm
        kappa_abs (np.array): The absorption mass cross section in cm^2/g
        Td (np.float): The dust temperature in K

    Returns:
        np.float: The Planck mean opacity in cm^2/g
    """

    # 1. Compute the Planck function for all wavelengths
    B_lambda = planck_function(wavelengths, Td)

    # 2. Compute the Planck mean opacity as the weighted average of kappa_abs
    numerator = np.trapezoid(kappa_abs * B_lambda, x=wavelengths)
    denominator = np.trapezoid(B_lambda, x=wavelengths)

    return numerator / denominator


def compute_equilibrium_temperature_planck_table(wavelengths, radiation_field, C_abs,
                                                planck_temps, planck_kappa, m_grain,
                                                Tmin=2.7, Tmax=800., tol=1e-3, max_iter=100):
    """Compute equilibrium dust temperature using a pre-computed Planck-mean table.

    This function assumes the Planck-mean opacity `planck_kappa` is given as a
    function of temperature `planck_temps` and is (monotonically) increasing.
    It finds the temperature T such that the absorbed power by the grain equals
    the emitted power estimated using the Planck mean opacity via binary search.

    Args:
        wavelengths (np.array): Wavelength grid for the absorption calculation (cm).
        radiation_field (np.array): Spectral radiation field (same length as `wavelengths`),
            in units erg/s/cm^2/cm.
        C_abs (np.array): Absorption cross section for the grain (cm^2), same length as `wavelengths`.
        planck_temps (np.array): Temperatures corresponding to the Planck-mean opacities (K),
            must be sorted in ascending order.
        planck_kappa (np.array): Planck-mean opacities (cm^2/g) at the temperatures in `planck_temps`.
        m_grain (float): Mass of the grain (g).
        Tmin (float): Minimum temperature to search (K).
        Tmax (float): Maximum temperature to search (K).
        tol (float): Relative tolerance on power match (fraction).
        max_iter (int): Maximum number of binary search iterations.

    Returns:
        float: Equilibrium temperature in K.

    Notes:
        Emitted power per grain is approximated as P_emit = 4 * kappa_P(T) * sigma_sb * T^4 * m_grain,
        which follows from P_emit = 4*pi * int kappa_abs B_lambda dlambda = 4 * kappa_P * sigma_sb * T^4 * m_grain.
    """

    # 1. Compute absorbed power (per grain)
    absorbed = np.trapezoid(radiation_field * C_abs, x=wavelengths)

    # 2. Sanity check: planck_kappa monotonicity (user assumed monotonic increasing)
    if not np.all(np.diff(planck_kappa) >= 0):
        print('Warning: provided Planck-mean opacities are not monotonically increasing. Binary search may not be valid.')

    # 3. Binary search over temperature
    low = Tmin
    high = Tmax
    P_low = None
    P_high = None

    for i in range(max_iter):
        mid = 0.5 * (low + high)
        # interpolate kappa_P at mid temperature
        kappa_mid = np.interp(mid, planck_temps, planck_kappa)
        P_mid = 4.0 * kappa_mid * sigma_sb * mid**4. * m_grain

        # initialize endpoints powers if not set
        if P_low is None:
            kappa_l = np.interp(low, planck_temps, planck_kappa)
            P_low = 4.0 * kappa_l * sigma_sb * low**4. * m_grain
        if P_high is None:
            kappa_h = np.interp(high, planck_temps, planck_kappa)
            P_high = 4.0 * kappa_h * sigma_sb * high**4. * m_grain

        # If mid power matches absorbed within tolerance, return
        if abs(P_mid - absorbed) <= tol * max(absorbed, 1e-30):
            return mid

        # Decide which half to keep. If emitted power increases with T, then
        # P_mid < absorbed -> need larger T (move low up), else move high down.
        if P_mid < absorbed:
            low = mid
            P_low = P_mid
        else:
            high = mid
            P_high = P_mid

        # If interval small enough, return midpoint
        if (high - low) / max(mid, 1e-12) < 1e-6:
            return 0.5 * (low + high)

    # If we exit loop without meeting tolerance, return best estimate
    return 0.5 * (low + high)


def fast_solve_Td(T_grid, G_grid, A):
    """Solve G(T) = A using monotonic binary search + single interpolation.

    Assumes `G_grid` is strictly increasing with `T_grid`.
    """
    T_grid = np.asarray(T_grid)
    G_grid = np.asarray(G_grid)

    # Edge cases
    if A <= G_grid[0]:
        return T_grid[0]
    if A >= G_grid[-1]:
        return T_grid[-1]

    lo, hi = 0, len(T_grid) - 1
    # Binary search indices
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if G_grid[mid] < A:
            lo = mid
        else:
            hi = mid

    # Linear interpolation between the bracketing nodes
    T1, T2 = T_grid[lo], T_grid[hi]
    G1, G2 = G_grid[lo], G_grid[hi]
    if G2 == G1:
        return 0.5 * (T1 + T2)
    Td = T1 + (T2 - T1) * (A - G1) / (G2 - G1)
    return Td


def compute_equilibrium_temperature_planck_table_fast(wavelengths, radiation_field, C_abs,
                                                      planck_temps, planck_kappa, m_grain,
                                                      Tmin=None, Tmax=None):
    """Fast equilibrium temperature from a pre-computed Planck-mean table.

    Builds G(T) = 4 * kappa_P(T) * sigma_sb * T^4 * m_grain and solves G(T)=P_abs
    using a single binary-index search + interpolation (fast and robust if G(T)
    is monotonic increasing).
    """

    # 1. Compute absorbed power (per grain)
    P_abs = np.trapezoid(radiation_field * C_abs, x=wavelengths)

    # 2. Build emitted-power grid G(T)
    T_grid = np.asarray(planck_temps)
    kappa_grid = np.asarray(planck_kappa)
    G_grid = 4.0 * kappa_grid * sigma_sb * T_grid**4.0 * m_grain

    # 3. Optional bounds checks
    if Tmin is not None:
        P_tmin = 4.0 * np.interp(Tmin, T_grid, kappa_grid) * sigma_sb * Tmin**4.0 * m_grain
        if P_abs <= P_tmin:
            return Tmin
    if Tmax is not None:
        P_tmax = 4.0 * np.interp(Tmax, T_grid, kappa_grid) * sigma_sb * Tmax**4.0 * m_grain
        if P_abs >= P_tmax:
            return Tmax

    # 4. Check monotonicity
    if not np.all(np.diff(G_grid) >= 0):
        print('Warning: emitted-power grid G(T) is not monotonically increasing. Results may be inaccurate.')

    # 5. Solve using index-based binary search + interpolation
    Td = fast_solve_Td(T_grid, G_grid, P_abs)
    return Td



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
        dust_token = str(dust_type).lower()
        if 'sil' in dust_token:
            dust_s = basic_s[5]
        elif 'pah' in dust_token:
            dust_s = basic_s[0]
        else:
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
    file_path = os.path.join(PATH_OPTICS, 'hensley_draine_2023', 'astrodust+PAH_mean_opacities.dat')
    df = read_HensleyDraine2023_mean_oppacity(file_path)
    ax.plot(df["Temp"], df["kappa_abs_R"], label='Hensley & Draine (2023)', color='b', linestyle='-', linewidth=2.5)

    # 6. Add the power-law scaling in Krumholz & Thompson 2013 paper
    kappa_PLT = 10.**(-1.5) * (Td / 10.)**2.
    ax.plot(Td, kappa_PLT, label='Krumholz & Thompson (2013)', color='r', linestyle='-', linewidth=2.5)   

    # 7. Add the Semenov et al. 2003 data
    file_path = os.path.join(PATH_OPTICS, 'semenov_2003', 'kR.out')
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
        

def plot_eqtemp_withcollision(dust_type,ne,nH,nHe,nC,Tmin,Tmax,nG0=100,nT=10,G0min=1e-1,G0max=1e7,
                              collisional_dust_bin=None,
                              output_dir=None):
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
        collisional_dust_bin (str, optional): Collisional table bin label or index,
            e.g. 'DustBin_00' or '00'. If None, inferred from `dust_type`.
    """
    from models.dust_gas_collisions.dust_collisional_cooling import load_cooling_tables
    
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
    int_radfield = np.trapezoid(radiation_field[:,1],x=radiation_field[:,0])
    C_abs_avg = np.trapezoid(C_abs_interp * radiation_field[:,1],x=radiation_field[:,0]) / int_radfield /(np.pi*a0**2.)
    print('Average absorption cross section for',dust_type,'computed')
    print('Given by',C_abs_avg)

    dust_label = _resolve_collisional_dust_label(dust_type, collisional_dust_bin=collisional_dust_bin)
    table_dir = os.path.join(str(get_repo_root()), 'model_data', 'collisional_cooling_data')

    # 4. Create the figure
    fig, ax = plt.subplots(1,1,figsize=(6,4),dpi=300,facecolor='w',edgecolor='k')
    from matplotlib.lines import Line2D
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
    coll_tables = load_cooling_tables(table_dir=table_dir)
    electron_rate_table = coll_tables[f'cooling_{dust_label}_Z_0']
    H_rate_table = coll_tables[f'cooling_{dust_label}_Z_1']
    He_rate_table = coll_tables[f'cooling_{dust_label}_Z_2']
    C_rate_table = coll_tables[f'cooling_{dust_label}_Z_6']

    T_dust_collisional, electron_rate_table = _extract_phi0_rate(electron_rate_table)
    _, H_rate_table = _extract_phi0_rate(H_rate_table)
    _, He_rate_table = _extract_phi0_rate(He_rate_table)
    _, C_rate_table = _extract_phi0_rate(C_rate_table)

    # 6. Reference Draine (2011) approximate scalings for silicate and graphite
    reference_curves = {}
    for reference_dust_type in ('silicate_bin_00', 'graphite_bin_00'):
        a_ref, ref_wavelengths, _, ref_C_abs, _ = compute_cross_sections(reference_dust_type, do_average=False)
        ref_C_abs_interp = np.interp(radiation_field[:, 0], ref_wavelengths[::-1], ref_C_abs[::-1])
        T_ref = compute_equilibrium_temperature_cheap(
            reference_dust_type,
            a_ref,
            radiation_field[:, 0],
            radiation_field[:, 1],
            ref_C_abs_interp,
        )
        reference_curves[reference_dust_type] = T_ref * G0**(1.0 / 6.0)
    
    # 6. Set the range of temperatures and loop, with increasing temperature
    # having a different color with a colormap
    import matplotlib as mpl
    cmap = plt.get_cmap('viridis')
    T = np.logspace(np.log10(Tmin),np.log10(Tmax),nT)
    for i in range(0,nT):
        
        # 7. Compute the equilibrium temperature
        Teq = np.zeros(nG0)
        
        def compute_temp_coll_newton(j):
            return compute_eqT_withcollisions_newton(
                               dust_type,a0,
                               radiation_field[:,0],
                               wavelengths_em,
                               G0[j]*radiation_field[:,1],
                               C_abs_interp,C_abs_em_interp,
                               ne,nH,nHe,nC,T[i],T_dust_collisional,
                               electron_rate_table,H_rate_table,
                               He_rate_table,C_rate_table)

        def compute_temp_coll_linearized(j):
            return compute_eqT_withcollisions_newton_linearized(
                               dust_type,a0,
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
        

        t_start_newton = time.perf_counter()
        Teq_coll_newton = Parallel(n_jobs=-1)(delayed(compute_temp_coll_newton)(j) for j in range(nG0))
        dt_newton = time.perf_counter() - t_start_newton

        t_start_linearized = time.perf_counter()
        Teq_coll_linearized = Parallel(n_jobs=-1)(delayed(compute_temp_coll_linearized)(j) for j in range(nG0))
        dt_linearized = time.perf_counter() - t_start_linearized

        t_start_cheap = time.perf_counter()
        Teq = Parallel(n_jobs=-1)(delayed(compute_temp)(j) for j in range(nG0))
        dt_cheap = time.perf_counter() - t_start_cheap

        print(
            f"Tgas={T[i]:.6g} K | "
            f"times [s] cheap={dt_cheap:.3f}, newton={dt_newton:.3f}, linearized={dt_linearized:.3f} | "
            f"final T@G0max [K] cheap={Teq[-1]:.6g}, newton={Teq_coll_newton[-1]:.6g}, linearized={Teq_coll_linearized[-1]:.6g}"
        )
        
        # 8. Plot the results with a colormap
        color = cmap(i / nT)
        ax.plot(G0,Teq_coll_newton,color=color,linewidth=2.5,linestyle=':',alpha=0.6)
        ax.plot(G0,Teq_coll_linearized,color=color,linewidth=2.5,linestyle='-.',alpha=0.6)
        ax.plot(G0,Teq,color='k',linewidth=2.5,alpha=0.3)

        ax.plot(G0, reference_curves['silicate_bin_00'], color='0.25', linestyle='--', linewidth=2.5)
        ax.plot(G0, reference_curves['graphite_bin_00'], color='0.25', linestyle=':', linewidth=2.5)
    
    # 9. Add a log colorbar
    norm = mpl.colors.LogNorm(vmin=Tmin, vmax=Tmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.02)
    cbar.set_label(r'$T_{\rm gas}$ [K]', fontsize=20)

    line_handles = [
        Line2D([0], [0], color='k', linestyle=':', linewidth=2.5, label='Computed: collisional equilibrium (Newton)'),
        Line2D([0], [0], color='k', linestyle='-.', linewidth=2.5, label='Computed: collisional equilibrium (linearized)'),
        Line2D([0], [0], color='k', linestyle='-', linewidth=2.5, alpha=0.3, label='Computed: no collisions'),
        Line2D([0], [0], color='0.25', linestyle='--', linewidth=2.5, label='Draine 2011 approx.: silicate'),
        Line2D([0], [0], color='0.25', linestyle=':', linewidth=2.5, label='Draine 2011 approx.: graphite'),
    ]
    ax.legend(handles=line_handles, loc='best', fontsize=10, frameon=False)

    # 10. Save figure
    fig.subplots_adjust(top=0.99, bottom=0.13, left=0.13, right=0.99, hspace=0, wspace=0)
    if output_dir is None:
        output_dir = os.path.join(str(get_repo_root()), 'model_data', 'optical_properties')
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, f'eqtemp_withcollisions_{dust_type}.pdf'), format='pdf', dpi=300)


def plot_eqtemp_tgas_density_grid(dust_bin,
                                  Tgas_min=10.0, Tgas_max=1e6,
                                  nH_min=1e-4, nH_max=1e4,
                                  near_equilibrium_tol=0.1,
                                  method='newton',
                                  output_dir=None,
                                  filename=None):
    """Plot Tdust/Tgas on a (Tgas, nH) grid.

    The equilibrium temperature is computed using the same ingredients as
    `plot_eqtemp_withcollision`: identical radiation field construction,
    interpolated cross sections, and collisional tables loaded from the
    precomputed cooling tables.

    Args:
        dust_bin (str): Bin id used for both optical properties and collisional
            tables, e.g. 'DustBin_01'.
        Tgas_min, Tgas_max (float, optional): Gas-temperature range [K].
        nH_min, nH_max (float, optional): Hydrogen-number-density range [cm^-3].
        near_equilibrium_tol (float, optional): Region criterion for near thermal coupling,
            using |Tdust/Tgas - 1| <= near_equilibrium_tol.
        method (str, optional): 'linearized' (default) or 'newton'.
        output_dir (str, optional): Output directory for the plot.
        filename (str, optional): Output filename. Auto-generated if None.

    Returns:
        dict: {
            'Tgas_grid_K': 1D temperature grid,
            'nH_grid_cm3': 1D nH grid,
            'Tdust_grid_K': 2D equilibrium dust-temperature grid,
            'ratio_grid': 2D Tdust/Tgas grid,
            'near_equilibrium_mask': 2D boolean mask,
            'output_path': figure path
        }
    """
    import matplotlib as mpl
    from models.dust_gas_collisions.dust_collisional_cooling import load_cooling_tables

    if Tgas_min <= 0 or Tgas_max <= 0 or Tgas_max <= Tgas_min:
        raise ValueError('Require 0 < Tgas_min < Tgas_max.')
    if nH_min <= 0 or nH_max <= 0 or nH_max <= nH_min:
        raise ValueError('Require 0 < nH_min < nH_max.')
    if near_equilibrium_tol < 0:
        raise ValueError('near_equilibrium_tol must be non-negative.')

    G0 = 1.0

    method_token = str(method).lower()
    if method_token not in ('linearized', 'newton'):
        raise ValueError('method must be one of: linearized, newton')

    # 1. Build the same radiation field used by the other equilibrium routines.
    wav = np.logspace(np.log10(0.0912 * 1e-4), np.log10(1000 * 1e-4), 100)  # [cm]
    radiation_field = np.zeros((len(wav), 2))
    radiation_field[:, 0] = wav
    radiation_field[:, 1] = modified_mmp83_radiation_field(wav) / wav * c  # [erg cm^-2 s^-1 cm^-1]

    # 2. Cross sections for the selected bin from the exported optical-property files.
    a0, wavelengths, _, C_abs, _, composition = _read_precomputed_optical_properties(dust_bin)
    C_abs_interp = np.interp(radiation_field[:, 0], wavelengths[::-1], C_abs[::-1])

    # Use the full wavelength range of the optical file for emission (covers 10 Å – 1 cm)
    # so that emitted_power is correct even at high temperatures (Tdust ~ Tgas >> 1000 K).
    wavelengths_em = np.logspace(np.log10(wavelengths.min()), np.log10(wavelengths.max()), 2000)
    C_abs_em_interp = np.interp(wavelengths_em, wavelengths[::-1], C_abs[::-1])

    # 3. Load precomputed collisional tables.
    dust_label = _resolve_collisional_dust_label(dust_bin)
    table_dir = os.path.join(str(get_repo_root()), 'model_data', 'collisional_cooling_data')
    coll_tables = load_cooling_tables(table_dir=table_dir)
    z_to_table = _collect_collisional_tables_for_dustbin(coll_tables, dust_label)
    available_Z = sorted(z_to_table.keys())
    print(f'Using collisional channels for {dust_label}: Z={available_Z}')

    # 4. Build the parameter-space grid from min/max bounds only.
    # Keep the auto grid moderately sized so the bounds-only API stays responsive.
    tgas_dex = np.log10(Tgas_max / Tgas_min)
    nh_dex = np.log10(nH_max / nH_min)
    nTgas = int(np.clip(np.ceil(8.0 * tgas_dex) + 8, 16, 36))
    n_nH = int(np.clip(np.ceil(8.0 * nh_dex) + 8, 16, 36))
    Tgas_grid = np.logspace(np.log10(Tgas_min), np.log10(Tgas_max), nTgas)
    nH_grid = np.logspace(np.log10(nH_min), np.log10(nH_max), n_nH)

    Tdust_grid = np.full((len(Tgas_grid), len(nH_grid)), np.nan, dtype=float)

    # Cheap guess is independent of Tgas/density for fixed G0 and dust properties.
    T0_guess = _compute_equilibrium_temperature_cheap_from_material(
        composition,
        a0,
        radiation_field[:, 0],
        G0 * radiation_field[:, 1],
        C_abs_interp,
    )
    absorbed = absorbed_power(radiation_field[:, 0], G0 * radiation_field[:, 1], C_abs_interp)

    # 5. Evaluate equilibrium temperature over the grid.
    print(f'Solving Tgas-nH grid with {nTgas} x {n_nH} points...')
    for i, Tgas_val in enumerate(Tgas_grid):
        print(f'  Row {i + 1}/{len(Tgas_grid)}: Tgas={Tgas_val:.6g} K')
        Tdust_grid[i, :] = Parallel(n_jobs=-1)(
            delayed(_solve_one_cell)(
                nH, Tgas_val, z_to_table, available_Z,
                absorbed, wavelengths_em, C_abs_em_interp,
                method_token, T0_guess,
            )
            for nH in nH_grid
        )

    ratio_grid = Tdust_grid / Tgas_grid[:, None]
    near_equilibrium_mask = np.isfinite(ratio_grid) & (np.abs(ratio_grid - 1.0) <= near_equilibrium_tol)

    # 6. Plot ratio map and highlight near-equilibrium region.
    fig, ax = plt.subplots(1, 1, figsize=(7, 5), dpi=300, facecolor='w', edgecolor='k')
    ax.set_xlabel(r'$n_{\rm H}$ [cm$^{-3}$]', fontsize=16)
    ax.set_ylabel(r'$T_{\rm gas}$ [K]', fontsize=16)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.minorticks_on()
    ax.tick_params(which='both', axis='both', direction='in')

    finite_ratio = ratio_grid[np.isfinite(ratio_grid)]
    if finite_ratio.size == 0:
        raise RuntimeError('No finite Tdust/Tgas values were computed on the requested grid.')

    vmin = max(np.min(finite_ratio), 1e-3)
    vmax = max(np.max(finite_ratio), vmin * 1.001)
    if vmin < 1.0 < vmax:
        norm = mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)
    else:
        norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)

    pcm = ax.pcolormesh(nH_grid, Tgas_grid, ratio_grid,
                        shading='auto', cmap='coolwarm', norm=norm)

    # Draw a contour around cells where dust is close to gas temperature.
    eq_indicator = near_equilibrium_mask.astype(float)
    ax.contour(nH_grid, Tgas_grid, eq_indicator,
               levels=[0.5], colors='k', linewidths=1.8)

    cbar = plt.colorbar(pcm, ax=ax, orientation='vertical', pad=0.02)
    cbar.set_label(r'$T_{\rm dust}/T_{\rm gas}$', fontsize=14)

    frac_near = 100.0 * np.sum(near_equilibrium_mask) / near_equilibrium_mask.size
    ax.set_title(
        f'{dust_label}, G0=1, method={method_token}, solar abund. (Asplund+09), '
        f'near-eq: |Td/Tg-1|<={near_equilibrium_tol:g} ({frac_near:.1f}%)',
        fontsize=11,
    )

    fig.subplots_adjust(top=0.94, bottom=0.13, left=0.13, right=0.98, hspace=0, wspace=0)

    if output_dir is None:
        output_dir = os.path.join(str(get_repo_root()), 'model_data', 'optical_properties')
    os.makedirs(output_dir, exist_ok=True)

    if filename is None:
        filename = f'eqtemp_tgas_density_ratio_{dust_label}.pdf'
    output_path = os.path.join(output_dir, filename)
    fig.savefig(output_path, format='pdf', dpi=300)

    print(
        f'Grid complete for {dust_label}: finite={np.isfinite(ratio_grid).sum()}/{ratio_grid.size}, '
        f'near-equilibrium fraction={frac_near:.2f}%.'
    )

    return {
        'Tgas_grid_K': Tgas_grid,
        'nH_grid_cm3': nH_grid,
        'Tdust_grid_K': Tdust_grid,
        'ratio_grid': ratio_grid,
        'near_equilibrium_mask': near_equilibrium_mask,
        'output_path': output_path,
    }
