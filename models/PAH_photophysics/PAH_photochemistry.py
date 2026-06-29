import sys
from pathlib import Path
import os
# Add models path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import re
import importlib_resources
from amespahdbpythonsuite.amespahdb import AmesPAHdb
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="white")
from scipy.optimize import root_scalar
from scipy.integrate import quad
from scipy.linalg import null_space
from scipy import linalg
from models.tools.radiation_fields import Mathis83_radiation_field
from models.PAH_radiation.pah_oppacity import pah_efficiencies


# Force Matplotlib to use Dejavu Sans for math rendering (supports all superscripts natively)
plt.rcParams['mathtext.fontset'] = 'dejavusans'

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CALIMA_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..', '..'))
_EXTERNAL_DATA_DIR = os.path.join(_CALIMA_ROOT, 'external_data')
PAH_OPTICALS_DIR = os.path.join(_CALIMA_ROOT, 'optical_props', 'li_draine_2001')
pahneu_filepath = os.path.join(PAH_OPTICALS_DIR, 'PAHneu_30')
pahion_filepath = os.path.join(PAH_OPTICALS_DIR, 'PAHion_30')


# CONSTANTS AND PARAMETERS
ME_CGS = 9.1093837015e-28
H_CGS = 6.62607015e-27
C_CGS = 2.99792458e10
KB_CGS = 1.380649e-16
EV2ERG = 1.602176634e-12
E_STATC = 4.8032047e-10
ELECTRON_ESCAPE_LENGTH_CM = 1e-7
TINY = 1e-300

IONISATION_POTENTIAL = {
    'C24H12': {
        '1': 7.20, # Tobita et al. 1994
        '2': 11.50 # Tobita et al. 1994
    },
    'C54H18': {
        '1': 6.14, # Malloci et al. 2007
        '2': 8.91, # Malloci et al. 2007
        '3': 12.94 # Malloci et al. 2007
    },
    'C96H24':{
        '1': 5.68, # Bakes & Tielens 1994
        '2': 8.24, # Bakes & Tielens 1994
        '3': 10.80, # Bakes & Tielens 1994
        '4': 13.36, # Bakes & Tielens 1994
    }
}
ELECTRON_AFFINITY = {
    'C24H12': {
        '1': 0.47 # Duncan et al. 1999
    },
    'C54H18': {
        '1': 1.44 # Malloci et al. 2007
    },
    'C96H24': {
        '1': 0.56, # Bakes & Tielens 1994
        '2': 3.11 # Bakes & Tielens 1994
    }
}



# FUNCTIONS

def extract_transitions(pahdb: AmesPAHdb, Nc: int, charge: str, output_path: str):
    """
    Extract the vibrational transitions from the NASA Ames Database for a given 
    number of carbon atoms and charge.

    Parameters
    ----------
    pahdb : AmesPAHdb
        The NASA Ames Database for PAH Spectra.
    Nc : int
        The number of carbon atoms.
    charge : str
        The charge of the PAH (e.g., "neutral", "anion", "cation").
    output_path : str
        The path to save the extracted transitions.

    Returns
    -------
    list
        A list of file paths to the extracted transitions.
    """
    
    # 1. Find the UIDs of the PAH with the given number of carbon atoms and charge
    uids = pahdb.search(f'c={Nc} {charge} fe=0 mg=0 o=0 si=0 n=0')
    if len(uids) == 0:
        print('ERROR: No PAH found with the given number of carbon atoms and charge')
        return []
    
    file_list = []
    for id in uids:
        pah = pahdb.getspeciesbyuid(id)
        info = pah.print(str=True)
        
        # Parse Formula
        match = re.search(r'^FORMULA\s*:\s*([^\s\n]+)', info, re.MULTILINE)
        if match:
            formula = match.group(1)
            print(f"Extracted Formula: {formula}")
        else:
            print("ERROR: Formula not found.\n", info)
            continue
            
        # Parse Charge 
        match = re.search(r'^CHARGE\s*:\s*([^\s\n]+)', info, re.MULTILINE)
        if match:
            charge_val = match.group(1)
            print(f"Extracted Charge: {charge_val}")
        else:
            print("ERROR: Charge not found.\n", info)
            continue
            
        # Parse N_SOLO
        match = re.search(r'^N_SOLO\s*:\s*([^\s\n]+)', info, re.MULTILINE)
        if match:
            nsolo = match.group(1)
            print(f"Extracted N_SOLO: {nsolo}")
        else:
            print("ERROR: N_SOLO not found.\n", info)
            continue
            
        # Parse N_DUO
        match = re.search(r'^N_DUO\s*:\s*([^\s\n]+)', info, re.MULTILINE)
        if match:
            nduo = match.group(1)
            print(f"Extracted N_DUO: {nduo}")
        else:
            print("ERROR: N_DUO not found.\n", info)
            continue
            
        # 2. Extract and write transitions data out via database stream
        transitions = pah.transitions()
        outfile = os.path.join(output_path, f"{formula}_{charge_val}.dat")
        transitions.write(outfile)

        # 3. Read the file, inject lines after \ SPECIES, and rewrite safely
        with open(outfile, "r") as f:
            file_content = f.read()
        
        # Define the target marker and the lines to inject
        target_marker = "\\ SPECIES"
        metadata_lines = f"\n\\N_SOLO     : {nsolo} \n\\N_DUO      : {nduo} "
        
        # Inject the new lines immediately following the target marker line
        updated_content = file_content.replace(target_marker, target_marker + metadata_lines)
        
        with open(outfile, "w") as f:
            f.write(updated_content)

        file_list.append(outfile)
        
    return file_list

def ionisation_potential_energy(IP0: float, Nh0:int, Nh: int):
    """
    Calculate the ionisation potential energy for a given number of hydrogen atoms.

    Parameters
    ----------
    IP0 : float
        The ionisation potential energy [eV] for the reference number of hydrogen atoms.
    Nh0 : int
        The reference number of hydrogen atoms.
    Nh : int
        The number of hydrogen atoms.

    Returns
    -------
    float
        The ionisation potential energy [eV] for the given number of hydrogen atoms.
    """
    IP = IP0 + 0.1 * (Nh0 - Nh)
    return IP

def electron_affinity_energy(EA0: float, Nh0:int, Nh: int):
    """
    Calculate the electron affinity energy for a given number of hydrogen atoms.

    Parameters
    ----------
    EA0 : float
        The electron affinity energy [eV] for the reference number of hydrogen atoms.
    Nh0 : int
        The reference number of hydrogen atoms.
    Nh : int
        The number of hydrogen atoms.

    Returns
    -------
    float
        The electron affinity energy [eV] for the given number of hydrogen atoms.
    """
    EA = EA0 + 0.1 * (Nh0 - Nh)
    return EA

def ionisation_yield_Jochims1996(IP: float, E: float):
    """
    Calculate the ionisation yield for a given ionisation potential and photon energy.

    Parameters
    ----------
    IP : float
        The ionisation potential [eV].
    E : float
        The photon energy [eV].

    Returns
    -------
    float
        The ionisation yield.
    """
    if E >= IP + 9.2:
        return 1.0
    else:
        return (E - IP) / 9.2

def ionisation_yield_LePage2001(IP: float, IPcoronene: float, E: float):
    """
    Calculate the ionisation yield for a given ionisation potential and photon energy.

    Parameters
    ----------
    IP : float
        The ionisation potential [eV].
    IPcoronene : float
        The ionisation potential of coronene [eV].
    E : float
        The photon energy [eV].

    Returns
    -------
    float
        The ionisation yield.
    """
    c = (14.89 - IPcoronene) / (14.89 - IP)
    return 0.8 * np.exp(-0.00128 * (c * (E - 14.89))**4.)

def photoionisation_rate(sigma_ion: np.ndarray,IP: float, 
                            N: np.ndarray, E: np.ndarray):
    """
    Calculate the photoionisation rate for a given number of hydrogen atoms.

    Parameters
    ----------
    sigma_ion : np.ndarray
        The photoionisation cross-section [cm2].
    IP : float
        The ionisation potential [eV].
    N : np.ndarray
        Photon number flux [# cm-2 s-1 eV-1].
    E : np.ndarray
        The photon energy [eV].

    Returns
    -------
    float
        The photoionisation rate [s-1].
    """
    
    mask = (E>= IP)
    k_ion = sigma_ion[mask] * N [mask]
    k_ion = np.trapezoid(k_ion, E[mask])

    return k_ion

def afromNc(Nc: int):
    """
    Calculcate the molecule radius in cm.

    Parameters
    ----------
    Nc : int
        The number of carbon atoms.

    Returns
    -------
    float
        The molecule radius [cm].
    """
    return 0.9e-8 * np.sqrt(Nc)

def recombination_rate_Spitzer(Nc: int,Z: int, T: float, ne: float):
    """Recombination rate following the Spitzer's formalism (Spitzer 2004) modified
    for cations by Verstraete et al. (1990) and extended to Z>0 by Berne et al. (2022)

    Args:
        Nc (int): Number of carbon atoms
        Z (int): PAH charge number
        T (float): Gas (or electron) temperature in [K]
        ne (float): Electron number density in [cm-3]

    Returns:
        float: Recombination rate in [s-1]
    """    
    phi = 1.85e5 / T / np.sqrt(Nc)
    k_rec = 1.28e-10 * Nc * np.sqrt(T) * (1. + phi * (1.+Z)) 
    return k_rec * ne

def recombination_rate_Tielens21(Nc: int, T: float, ne: float):
    """Recombination rate following Eq. 8.106 in Tielens (2021), which assumes
    a correction factor the the planar geometry of the PAH.

    Args:
        Nc (int): Number of carbon atoms
        T (float): Gas (or electron) temperature in [K]
        ne (float): Electron number density in [cm-3]

    Returns:
        float: Recombination rate in [s-1]
    """    
    k_rec = 1.3e-6 * np.sqrt(Nc) * np.sqrt(300. / T)
    return k_rec * ne

def attachment_rate_Carelli13(T: float, ne: float):
    """Electron attachment rate to neutral PAH as obtained for small PAHs
    in experiments by Carelli et al. (2013)

    Args:
        T (float): Gas (or electron) temperature in [K]
        ne (float): Electron number density in [cm-3]

    Returns:
        float: Attachment rate in [s-1]
    """    
    
    # Parameters for coronene (C24H12) from Carelli et al. (2013)
    a = 2.74e-9 # [cm-3]
    b = 0.11
    c = -1.12
    k_att = a * (T/300.)**b * np.exp(-c/T) # [cm^3/s]
    
    return k_att * ne

def attachment_rate_Tielens05(Nc: int, T: float, ne: float):
    """Electron attachment rate to neutral PAHs as given by Tielens (2005)

    Args:
        Nc (int): Number of carbon atoms in PAH molecule
        T (float): Gas (or electron) temperature in [K]
        ne (float): Electron number density in [cm-3]

    Returns:
        float: Attachment rate in [s-1]
    """    
    
    # Electron dimensionless sticking coefficient as approximated by Tielens (2005)
    s_e = 1.0
    
    k_att = 1.3e-7 * s_e * np.sqrt(Nc)
    
    return k_att * ne

def J_function_DS87(Z: int, q: float, a: float, T: float):
    """Returns the J tilde function from Draine and Sutin 1987
    following equations 3.3-3.5.

    Args:
        Z (int): PAH charge number
        q (int): projectile charge number
        a (float): PAH radius in cm
        T (float): Gas temperature in [K]

    Returns:
        float: J tilde function value
    """
    
    # Parameters
    nu = Z / q
    denom_q2 = (q * q) * (E_STATC * E_STATC)
    tau = (a * KB_CGS * T) / max(denom_q2, TINY)
    
    # Case 1: Z = 0 (neutral PAH)
    if nu == 0:
        return 1.0 + np.sqrt(np.pi / (2.0 * max(tau, TINY)))
    
    # Case 2: Z < 0 (negative PAH)
    elif nu < 0:
        tn = max(tau, TINY)
        inner = max(tn - 2.0 * nu, TINY)  # tau_n = tau - 2Z/q
        return (1.0 - nu / tn) * (1.0 + np.sqrt(2.0 / inner))
    
    # Case 3: Z > 0 (positive PAH)
    else:
        nup = max(nu, TINY)
        tp = max (tau, TINY)
        theta_nu = 1.0 / (1.0 + 1.0 / np.sqrt(nup))
        root_term = 1.0 / np.sqrt(4.0 * tp + 3.0 * nup)
        value = (1.0 + root_term) ** 2 * np.exp(-theta_nu / tp)
        return value if np.isfinite(value) else 0.0

def recombination_rate_Bakes1994(Nc: int, Z: int, se: float, T: float, ne: float):
    """
    Compute the electron-PAH recombination rate coefficients following the
    analytic approximation of Bakes & Tielens (1994).
    
    Args:
        Nc (int): Number of carbon atoms in PAH molecule
        Z (int): Charge number of PAH ion
        se (float): Electron sticking coefficient
        T (float): Gas (or electron) temperature in [K]
        ne (float): Electron number density in [cm-3]
    
    Returns:
        float: Recombination rate in [s-1]
    """

    J = J_function_DS87(Z, -float(Z)*E_STATC, afromNc(Nc), T)
    Verstraete1990_correction = 0.82

    krec = ne * se * np.sqrt(8.0 * KB_CGS * T / (np.pi * ME_CGS)) * J * np.pi * afromNc(Nc)**2

    return krec * Verstraete1990_correction

def se_neutral_Weingartner2001(a: float, Nc: float):
    """
    Compute the electron sticking coefficient for neutral PAHs following the
    analytic approximation of Weingartner & Draine (2001).
    
    Args:
        a (float): Radius of PAH molecule in [cm]
        Nc (int): Number of carbon atoms in PAH molecule
    
    Returns:
        float: Electron sticking coefficient
    """
    
    return 0.5 * (1. - np.exp(-a/ELECTRON_ESCAPE_LENGTH_CM)) / (1. + np.exp(20. - Nc))

def se_anion_Weingartner2001(a: float):
    """
    Compute the electron sticking coefficient for anion PAHs following the
    analytic approximation of Weingartner & Draine (2001).
    
    Args:
        a (float): Radius of PAH molecule in [cm]
    
    Returns:
        float: Electron sticking coefficient
    """
    
    return 0.5 * (1. - np.exp(-a/ELECTRON_ESCAPE_LENGTH_CM))

def compute_thermal_energy_from_file(file_path, t_min=1e2, t_max=1e4, nt=100):
    """
    Reads a NASA Ames transition data file from disk, applies the SCALE factor,
    and computes the relationship between canonical temperature T and the 
    internal energy per mode E / (3N-6).
    
    Parameters:
    -----------
    file_path : str
        The path to the transition data text file on your local system.
    t_min, t_max : float
        The temperature grid boundaries.
    nt : int
        The number of temperature points.
    
        
    Returns:
    --------
    temperatures : ndarray
        1D array of temperatures (K).
    energy_per_mode_cm : ndarray
        1D array containing the mean internal energy per mode in units of cm^-1.
    energy_per_mode_eV : ndarray
        1D array containing the mean internal energy per mode in units of eV.
    """
    # 1. Conversion constants
    # h * c in eV * cm -> yields energy (eV) when multiplied by wavenumber (cm^-1)
    hc_ev = 1.23984193e-4 
    # k_B in eV / K
    kb_ev = 8.61733326e-5 
    
    raw_frequencies = []
    scale_factors = []
    
    # 2. Open and read the file from disk line by line
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            
            # Skip empty lines, metadata headers (\), and structural info blocks (|)
            if not line or line.startswith('\\') or line.startswith('|'):
                continue
                
            tokens = line.split()
            
            # Ensure it is a data row containing our expected columns
            # [UID, FREQUENCY, INTENSITY, SCALE, SYMMETRY]
            if len(tokens) >= 4:
                try:
                    freq = float(tokens[1])
                    scale = float(tokens[3])
                    raw_frequencies.append(freq)
                    scale_factors.append(scale)
                except ValueError:
                    # Skips column descriptor text or footer artifacts safely
                    continue

    if not raw_frequencies:
        raise ValueError(f"No valid transition data could be parsed from the file: {file_path}")

    # 3. Apply the NASA Ames SCALE factor to the raw frequencies
    # This accounts for the systematic overestimation of theoretical DFT methods
    frequencies_raw = np.array(raw_frequencies)
    scales = np.array(scale_factors)
    frequencies_scaled_cm = frequencies_raw * scales  # Scale directly in cm^-1
    
    # Convert scaled frequencies to electron-volts (eV)
    frequencies_ev = frequencies_scaled_cm * hc_ev
    
    # Total number of active vibrational degrees of freedom (3N-6)
    num_modes = len(frequencies_ev)
    
    # 4. Generate the canonical temperature grid (T)
    temperatures = np.logspace(np.log10(t_min), np.log10(t_max), nt)
    total_energy_ev = np.zeros_like(temperatures, dtype=float)
    
    # 5. Compute the canonical internal energy U(T)
    # U(T) = sum( h*nu / (exp(h*nu / k_B*T) - 1) )
    for idx, T in enumerate(temperatures):
        x = frequencies_ev / (kb_ev * T)
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            thermal_occ = np.where(x > 50.0, np.exp(-x) / (1.0 - np.exp(-x)), 1.0 / np.expm1(x))
            thermal_occ = np.where(np.isnan(thermal_occ) | np.isinf(thermal_occ), 0.0, thermal_occ)
        total_energy_ev[idx] = np.sum(frequencies_ev * thermal_occ)
        
    # 6. Normalize by the number of modes to get the energy per mode
    energy_per_mode_eV = total_energy_ev / num_modes
    energy_per_mode_cm = energy_per_mode_eV / hc_ev
    
    return temperatures, energy_per_mode_cm, energy_per_mode_eV

def compute_thermal_ir_rate(file_path, internal_energy_ev):
    """
    Reads a NASA Ames transition file, maps a specific internal energy (E)
    to its operational canonical temperature (T), and computes the 
    total thermal IR emission rate K_thermal(T).
    
    Parameters:
    -----------
    file_path : str
        Path to the NASA Ames TRANSITIONS text file.
    internal_energy_ev : float
        The total internal energy of the PAH molecule in electron-volts (eV).
        Typically set by an absorbed UV photon: E = h * nu_UV.
        
    Returns:
    --------
    dict containing:
        'temperature_K': Calculated canonical temperature (K)
        'ir_rate_s1': Total IR photon emission rate (s^-1) at this energy
    """
    # 1. Physical Constants
    hc_ev = 1.23984193e-4      # h * c in eV * cm
    kb_ev = 8.61733326e-5      # k_B in eV / K
    gamma = 1.2512e-7          # Intensity-to-A coefficient conversion factor
    
    raw_frequencies = []
    intensities = []
    scale_factors = []
    
    # 2. Parse the NASA Ames TRANSITIONS file line by line
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith('\\') or line.startswith('|'):
                continue
                
            tokens = line.split()
            if len(tokens) >= 4:
                try:
                    freq = float(tokens[1])
                    int_val = float(tokens[2])
                    scale = float(tokens[3])
                    
                    raw_frequencies.append(freq)
                    intensities.append(int_val)
                    scale_factors.append(scale)
                except ValueError:
                    continue

    if not raw_frequencies:
        raise ValueError(f"Could not parse valid data from {file_path}")

    # Convert to numpy arrays
    freq_raw = np.array(raw_frequencies)
    ints = np.array(intensities)
    scales = np.array(scale_factors)
    
    # 3. Scale frequencies and calculate Einstein A coefficients
    # Systematically correct DFT calculations via database SCALE factor
    freq_scaled_cm = freq_raw * scales 
    freq_ev = freq_scaled_cm * hc_ev
    
    # Compute A_(i, 1->0) fundamental emission rates (s^-1) 
    # Formula: A = gamma * (nu_scaled)^2 * Intensity
    einstein_A = gamma * (freq_scaled_cm ** 2) * ints

    # 4. Define the Internal Energy Objective Function U(T)
    # This is used by the root-finder to evaluate U(T) - E = 0
    def internal_energy_objective(T):
        if T <= 0:
            return -internal_energy_ev
        x = freq_ev / (kb_ev * T)
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            thermal_occ = np.where(x > 50.0, np.exp(-x) / (1.0 - np.exp(-x)), 1.0 / np.expm1(x))
            thermal_occ = np.where(np.isnan(thermal_occ) | np.isinf(thermal_occ), 0.0, thermal_occ)
        U_T = np.sum(freq_ev * thermal_occ)
        return U_T - internal_energy_ev

    # 5. Numerically invert E -> T
    # We use a broad bracket bounding typical space environments (10K to 4000K)
    try:
        sol = root_scalar(internal_energy_objective, bracket=[1e0,1e5], method='brentq')
        canonical_T = sol.root
    except ValueError as e:
        raise ValueError(
            f"Energy calculation failed. {internal_energy_ev} eV might be out of "
            f"bounds for this molecule's vibrational degrees of freedom."
        ) from e

    # 6. Compute Total IR Emission Rate Constant K_thermal(T)
    # Sum of spontaneous emission rates weighted by thermal occupation numbers
    x = freq_ev / (kb_ev * canonical_T)
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        thermal_occ = np.where(x > 50.0, np.exp(-x) / (1.0 - np.exp(-x)), 1.0 / np.expm1(x))
        thermal_occ = np.where(np.isnan(thermal_occ) | np.isinf(thermal_occ), 0.0, thermal_occ)
    mode_emission_rates = einstein_A * thermal_occ
    
    # Sum over all active vibrational modes to find total photon emission rate (s^-1)
    K_thermal = np.sum(mode_emission_rates)
    
    return canonical_T,internal_energy_ev,K_thermal
    
def compute_rrkm_dissociation_rate(file_path, internal_energy_ev, E_act_ev, dS_cl_jk):
    """
    Computes the unimolecular dissociation rate constant K_dis(E) for a given 
    internal energy using a temperature-mapped TST/Arrhenius formulation.
    
    Parameters:
    -----------
    file_path : str
        Path to the NASA Ames TRANSITIONS text file (to fetch mode frequencies).
    internal_energy_ev : float
        The current internal energy E of the PAH molecule in eV.
    E_act_ev : float
        The activation energy (barrier height) E_act in eV (e.g., ~4.5 eV for C-H loss).
    dS_cl_jk : float
        The change in activation entropy (dS) in J / (mol * K) or J / (K * mol).
        
    Returns:
    --------
    float
        The dissociation rate constant K_dis(E) in s^-1.
    """
    # If the available energy is below the activation barrier, dissociation is impossible
    if internal_energy_ev < E_act_ev:
        return 0.0

    # 1. Physical Constants
    hc_ev = 1.23984193e-4      # h * c in eV * cm
    kb_ev = 8.61733326e-5      # k_B in eV / K
    h_j_s = 6.62607015e-34     # h in Joules * s
    kb_j_k = 1.380649e-23      # k_B in Joules / K
    r_gas = 8.31446261         # Gas constant R in J / (mol * K)

    raw_frequencies = []
    scale_factors = []
    
    # 2. Parse NASA Ames transitions file to build the U(T) function
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith('\\') or line.startswith('|'):
                continue
            tokens = line.split()
            if len(tokens) >= 4:
                try:
                    raw_frequencies.append(float(tokens[1]))
                    scale_factors.append(float(tokens[3]))
                except ValueError:
                    continue

    freq_scaled_cm = np.array(raw_frequencies) * np.array(scale_factors)
    freq_ev = freq_scaled_cm * hc_ev

    # 3. Define the U(T) - E internal energy objective function
    def energy_objective(T):
        if T <= 0:
            return -internal_energy_ev
        x = freq_ev / (kb_ev * T)
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            thermal_occ = np.where(x > 50.0, np.exp(-x) / (1.0 - np.exp(-x)), 1.0 / np.expm1(x))
            thermal_occ = np.where(np.isnan(thermal_occ) | np.isinf(thermal_occ), 0.0, thermal_occ)
        return np.sum(freq_ev * thermal_occ) - internal_energy_ev

    # 4. Invert E -> T(E) using root-finding
    try:
        sol = root_scalar(energy_objective, bracket=[1e0,1e5], method='brentq')
        T_E = sol.root
    except ValueError:
        # Fallback if internal energy corresponds to an extreme temperature environment
        return 0.0

    # 5. Calculate the Arrhenius/TST rate parameter expressions
    # Convert activation entropy from J/mol/K to J/K per molecule by dividing by N_Avogadro,
    # which is equivalent to dividing by the Universal Gas Constant R.
    entropy_exponent = dS_cl_jk / r_gas
    
    # Frequency factor component: (k_B * T / h)
    tst_frequency_factor = (kb_j_k * T_E) / h_j_s
    
    # Exponential energy barrier component: exp(-E_act / (k_B * T))
    boltzmann_factor = np.exp(-E_act_ev / (kb_ev * T_E))
    
    # Combine terms: K_dis = (k_B*T/h) * exp(dS/R) * exp(-E_act/k_BT)
    K_dis = tst_frequency_factor * np.exp(entropy_exponent) * boltzmann_factor
    
    return K_dis


def compute_gd89_temperature_distribution(file_path, radiation_field_func,
                                                cross_section_table,
                                                 t_min=1.0, num_bins=150):
    """
    Computes the steady-state vibrational temperature distribution f(T) of a PAH
    molecule using the stable recursive flux-balance formulation from GD89.
    
    Dynamically constrains the upper boundary (t_max) to the temperature 
    reached when absorbing a Lyman-limit photon (13.6 eV).
    
    Parameters
    ----------
    file_path : str
        Path to the NASA Ames TRANSITIONS text file.
    radiation_field_func : callable
        Function taking frequency (Hz) and returning mean intensity J_nu 
        in photons / (cm^2 * s * Hz * str).
    cross_section_table : ndarray
        A 2D numpy array of shape (N, 2). 
        Column 0: Photon Energy in eV.
        Column 1: Cross section sigma_UV in cm^2.
    t_min : float
        Vibrational temperature grid lower boundary (K). Default is 1.0 K.
    num_bins : int
        Number of discrete temperature intervals (M). Default is 150.
        
    Returns
    -------
    temperatures : ndarray
        Center temperatures of each bin (K).
    f_T : ndarray
        Normalized continuous probability distribution function f(T) (K^-1).
    """
    # =========================================================================
    # 1. PHYSICAL CONSTANTS & FILE PARSING
    # =========================================================================
    hc_ev = 1.23984193e-4      # h * c in eV * cm
    kb_ev = 8.61733326e-5      # k_B in eV / K
    c_cm_s = 2.99792458e10     # c in cm / s
    h_cgs = 6.62607015e-27     # h in erg * s
    gamma = 1.2512e-7          # Database Intensity-to-A conversion factor
    
    raw_frequencies = []
    intensities = []
    scale_factors = []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith('\\') or line.startswith('|'):
                continue
            tokens = line.split()
            if len(tokens) >= 4:
                try:
                    raw_frequencies.append(float(tokens[1]))
                    intensities.append(float(tokens[2]))
                    scale_factors.append(float(tokens[3]))
                except ValueError:
                    continue

    freq_scaled_cm = np.array(raw_frequencies) * np.array(scale_factors)
    freq_ev = freq_scaled_cm * hc_ev
    einstein_A = gamma * (freq_scaled_cm ** 2) * np.array(intensities)

    # Filter out inactive modes for the cooling engine calculations
    active_mask = einstein_A > 0
    cooling_freq_ev = freq_ev[active_mask]
    cooling_A = einstein_A[active_mask]

    # =========================================================================
    # 2. SEPARATING CROSS-SECTION INTERPOLATION CHECKS
    # =========================================================================
    # Unpack columns from user table. Assumed configuration: [Energy_eV, Sigma_cm2]
    cs_energies_ev = cross_section_table[:, 0]
    cs_sigmas_cm2 = cross_section_table[:, 1]
    
    # Helper to interpolate cross sections cleanly at any calculated energy step
    def get_sigma_uv(energy_ev):
        # Interpolate, falling back to 0.0 outside tabulated limits
        return np.interp(energy_ev, cs_energies_ev, cs_sigmas_cm2, left=0.0, right=0.0)

    # =========================================================================
    # 3. DYNAMIC CEILING T_MAX RESOLUTION (U(T_max) = 13.6 eV)
    # =========================================================================
    def energy_ceiling_objective(T):
        x = freq_ev / (kb_ev * T)
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            thermal_occ = np.where(x > 50.0, np.exp(-x) / (1.0 - np.exp(-x)), 1.0 / np.expm1(x))
            thermal_occ = np.where(np.isnan(thermal_occ) | np.isinf(thermal_occ), 0.0, thermal_occ)
        U_T = np.sum(freq_ev * thermal_occ)
        return U_T - 13.6  # Find root where total internal energy is exactly 13.6 eV

    try:
        sol = root_scalar(energy_ceiling_objective, bracket=[50.0, 5000.0], method='brentq')
        t_max = sol.root
    except ValueError as e:
        raise ValueError(
            "Could not dynamically solve for T_max ceiling. Verify that the file parsed "
            "vibrational modes correctly to generate a valid internal energy profile."
        ) from e

    # =========================================================================
    # 4. THERMODYNAMIC GRID & POWER METRICS DISCRETIZATION
    # =========================================================================
    # Construct log-spaced grid spanning from t_min cleanly to dynamic t_max
    t_edges = np.logspace(np.log10(t_min), np.log10(t_max), num_bins + 1)
    t_centers = 0.5 * (t_edges[:-1] + t_edges[1:])
    delta_t = np.diff(t_edges)
    
    def calculate_internal_energy(T):
        x = freq_ev / (kb_ev * T)
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            thermal_occ = np.where(x > 50.0, np.exp(-x) / (1.0 - np.exp(-x)), 1.0 / np.expm1(x))
            thermal_occ = np.where(np.isnan(thermal_occ) | np.isinf(thermal_occ), 0.0, thermal_occ)
        return np.sum(freq_ev * thermal_occ)

    u_edges = np.array([calculate_internal_energy(t) for t in t_edges])
    u_centers = np.array([calculate_internal_energy(t) for t in t_centers])

    # =========================================================================
    # 5. MATRIX POPULATION WITH WAVELENGTH INTERPOLATION
    # =========================================================================
    W_up = np.zeros((num_bins, num_bins))
    W_down_adjacent = np.zeros(num_bins)

    for j in range(num_bins):
        T_j = t_centers[j]
        U_j = u_centers[j]
        
        # --- A. Continuous Total Emitted IR Power Downward Flux ---
        if j > 0:
            x = cooling_freq_ev / (kb_ev * T_j)
            with np.errstate(over='ignore', under='ignore', invalid='ignore'):
                thermal_occ = np.where(x > 50.0, np.exp(-x) / (1.0 - np.exp(-x)), 1.0 / np.expm1(x))
                thermal_occ = np.where(np.isnan(thermal_occ) | np.isinf(thermal_occ), 0.0, thermal_occ)
            mode_rates = cooling_A * thermal_occ
            P_IR_ev_per_s = np.sum(cooling_freq_ev * mode_rates)
            bin_width_ev = u_centers[j] - u_centers[j-1]
            
            # Apply a safe cooling limit anchor floor to protect boundaries
            W_down_adjacent[j] = (P_IR_ev_per_s / bin_width_ev) + 1e-30

        # --- B. Upward Heating Rates (W_{j -> k} for k > j) ---
        for k in range(j + 1, num_bins):
            u_req_min = u_edges[k] - U_j
            u_req_max = u_edges[k+1] - U_j
            
            # Midpoint required photon energy for this state-to-state excitation step
            e_photon_mid = 0.5 * (u_req_min + u_req_max)
            
            if e_photon_mid > 13.6 or e_photon_mid <= 0:
                continue
                
            nu_min = (u_req_min / hc_ev) * c_cm_s
            nu_max = (u_req_max / hc_ev) * c_cm_s
            nu_mid = 0.5 * (nu_min + nu_max)
            delta_nu = nu_max - nu_min
            
            # DYNAMIC LOOKUP: Fetch the precise cross-section for this photon energy
            sigma_uv_dynamic = get_sigma_uv(e_photon_mid)
            
            if sigma_uv_dynamic <= 0.0:
                W_up[j, k] = 0.0
                continue
                
            flux_density = radiation_field_func(nu_mid)
            absorption_rate = 4.0 * np.pi * (flux_density / (h_cgs * nu_mid)) * sigma_uv_dynamic * delta_nu
            W_up[j, k] = max(0.0, absorption_rate)

    # =========================================================================
    # 6. MATHEMATICALLY SOLID LOG-SPACE RECURSION SWEEP
    # =========================================================================
    log_f = np.zeros(num_bins)
    log_f[0] = 0.0  # ln(1.0) = 0.0 baseline ground anchor

    for f in range(1, num_bins):
        log_terms = []
        for j in range(f):
            cumulative_upward_rate = np.sum(W_up[j, f:])
            if cumulative_upward_rate > 0 and log_f[j] > -700:
                log_terms.append(log_f[j] + np.log(cumulative_upward_rate))
        
        if not log_terms:
            log_f[f] = -np.inf
            continue
            
        max_log_term = np.max(log_terms)
        log_flux_sum = max_log_term + np.log(np.sum(np.exp(log_terms - max_log_term)))
        
        log_f[f] = log_flux_sum - np.log(W_down_adjacent[f])

    # =========================================================================
    # 7. EXPORT REGULARIZATION
    # =========================================================================
    max_log_f = np.max(log_f[np.isfinite(log_f)])
    f_discrete = np.exp(log_f - max_log_f)
    
    total_sum = np.sum(f_discrete)
    f_discrete_normalized = f_discrete / total_sum
    f_T = f_discrete_normalized / delta_t

    return t_centers, f_T

def compute_total_dissociation_rate(t_centers, f_T, E_act_ev, dS_cl_jk, t_min=15.0):
    """
    Computes the total time-averaged unimolecular dissociation rate of a PAH 
    by integrating the temperature-dependent rate over the steady-state 
    GD89 temperature distribution f(T).
    
    Automatically handles log-spaced temperature arrays by dynamically 
    reconstructing grid boundaries based on the input length and max values.
    
    Parameters:
    -----------
    t_centers : ndarray
        1D array containing the center temperatures of each bin (K) from the GD89 solver.
    f_T : ndarray
        1D array representing the normalized continuous probability distribution function f(T) (K^-1).
    E_act_ev : float
        The activation energy barrier height (E_act) for the dissociation channel in eV.
    dS_cl_jk : float
        The activation entropy change (dS) in J / (mol * K).
    t_min : float
        The lower boundary used for the temperature grid (K). Default is 15.0 K.
        
    Returns:
    --------
    float
        The total macroscopic dissociation rate (s^-1) of the molecule.
    """
    # 1. Physical Constants
    kb_ev = 8.61733326e-5      # k_B in eV / K
    kb_j_k = 1.380649e-23      # k_B in Joules / K
    h_j_s = 6.62607015e-34     # h in Joules * s
    r_gas = 8.31446261         # Gas constant R in J / (mol * K)
    
    # 2. Reconstruct Grid Edges Dynamically from Input Grid Context
    num_bins = len(t_centers)
    t_max_detected = t_centers[-1] + (t_centers[-1] - t_centers[-2]) / 2.0 # Fallback linear anchor edge
    
    # Extract precise logarithmic boundary ceiling matching our dynamic solver setup
    # Given log-spacing: centers are geometric means or close, recalculating edges based on endpoints:
    t_max = 10**(2 * np.log10(t_centers[-1]) - np.log10(t_centers[-2]))
    
    t_edges = np.logspace(np.log10(t_min), np.log10(t_centers[-1]), num_bins)
    # Re-expand safely to edge boundaries:
    t_edges = np.logspace(np.log10(t_min), np.log10(t_max), num_bins + 1)
    delta_t = np.diff(t_edges)
    
    # 3. Recover Discrete Probabilities from the PDF Array
    # Probability (fraction of time spent in bin) = PDF * bin_width
    f_discrete = f_T * delta_t
    
    # 4. Compute Transition State Theory (TST) Rate Array K_dis(T)
    # Frequency factor component: (k_B * T / h)
    tst_frequency_factors = (kb_j_k * t_centers) / h_j_s
    
    # Entropy factor component: exp(dS / R)
    entropy_factor = np.exp(dS_cl_jk / r_gas)
    
    # Arrhenius activation energy factor component: exp(-E_act / (k_B * T))
    boltzmann_factors = np.exp(-E_act_ev / (kb_ev * t_centers))
    
    # Unified vectorized evaluation
    K_dis_T = tst_frequency_factors * entropy_factor * boltzmann_factors
    
    # 5. Numerical Sum Integration: Sum( P_j * K_dis(T_j) )
    total_dissociation_rate = np.sum(f_discrete * K_dis_T)
    
    return total_dissociation_rate

def mathis83_to_gd89_interface(nu, target_G0=1.0, base_G0=1.0):
    """
    Transforms the output of Mathis83_radiation_field from erg/cm^3/eV 
    to energy mean spectral intensity I_nu in erg/(cm^2*s*Hz*str)
    to match the h*nu division inside the solver loop.
    
    Parameters
    ----------
    nu : float
        Frequency in Hz.
    target_G0 : float
        The target G0 value.
    base_G0 : float
        The base G0 value.
        
    Returns
    -------
    float
        I_nu in erg cm^-2 s^-1 Hz^-1 str^-1.
    """
    import numpy as np

    if nu <= 0.0:
        return 0.0

    # 1. Fundamental Constants
    h_SI = 6.62607015e-34
    eV2J = 1.602176634e-19
    c_cgs = 2.99792458e10       # c in cm / s

    # 2. Convert incoming frequency (Hz) to energy (eV)
    E = (h_SI * nu) / eV2J

    # Bounds check corresponding to the Lyman limit (13.6 eV)
    if E > 13.6:
        return 0.0

    # 3. Fetch u_E from your radiation field function (erg * cm^-3 * eV^-1)
    u_E = Mathis83_radiation_field(E) * (target_G0 / base_G0)

    # 4. Transform energy density distribution: u_E -> u_nu (erg * cm^-3 * Hz^-1)
    u_nu = u_E * (h_SI / eV2J)

    # 5. Convert energy density to mean spectral intensity: u_nu -> I_nu
    # Units: erg * cm^-2 * s^-1 * Hz^-1 * str^-1
    I_nu = (c_cgs / (4.0 * np.pi)) * u_nu

    # CRITICAL CORRECTION: Return I_nu directly. 
    # Do not divide by (h * nu) here, as the solver loop handles it.
    return float(I_nu)

def get_absorption_cross_section(Z,a0):
    """
    Get the absorption cross-section for a given charge and size parameter (a0 in cm).

    Parameters
    ----------
    Z : int
        The charge of the PAH molecule (0 for neutral, otherwise ionized).
    a0 : float
        The size parameter in cm.

    Returns
    -------
    tuple(np.ndarray, np.ndarray)
        Wavelengths in cm and absorption cross-section in cm^2.
    """
    
    if Z == 0:
        nwav,data,columns,name = pah_efficiencies(pahneu_filepath)
    else:
        nwav,data,columns,name = pah_efficiencies(pahion_filepath)

    # Convert a0 from cm to microns for table interpolation (table keys are in microns)
    a0_micron = a0 * 1e4
    
    # Sort keys of data and map float values back to string keys stably
    float_to_str_key = {float(k): k for k in data.keys()}
    keys_sorted = sorted(float_to_str_key.keys())
    a = np.array(keys_sorted)
    
    C_abs = np.zeros(nwav)
    for i in range(nwav):
        Q_abs = np.array([data[float_to_str_key[k]][i, columns.index('Q_abs')] for k in keys_sorted])
        Q_abs_interp = 10.**np.interp(np.log10(a0_micron), np.log10(a), np.log10(Q_abs))
        C_abs[i] = Q_abs_interp * np.pi * a0**2

    w = data[list(data.keys())[0]][:,columns.index('w(micron)')]

    return w*1e-4, C_abs

def compute_base_g0(radiation_field_func):
    """
    Integrates the given radiation field energy density between 6.0 eV 
    and 13.6 eV and converts it to a Habing G0 scaling value.
    
    Returns
    -------
    float
        The intrinsic G0 value of the base radiation function.
    """
    c_cgs = 2.99792458e10  # c in cm / s
    # Standard Habing UV integrated energy flux factor for G0=1 (erg / cm^2 / s)
    habing_flux_norm = 1.6e-3 

    # Objective wrapper to integrate the raw u_E profile from your file
    def integrand(E):
        return radiation_field_func(E)

    # Perform numerical quadrature integration across Habing limits (6.0 to 13.6 eV)
    integrated_u, _ = quad(integrand, 6.0, 13.6)
    
    # Total flux = c * integrated energy density
    mathis_uv_flux = c_cgs * integrated_u
    
    # Calculate base G0 factor
    base_g0 = mathis_uv_flux / habing_flux_norm
    return base_g0


def compute_adaptive_temperature_distribution(file_path, radiation_field_func, cross_section_table,
                                               t_min=15.0, num_bins=150, threshold=0.01):
    """
    Computes steady-state f(T) using an integrated DL01 cooling lifetime check 
    to trigger between single-photon recursion and an extended multi-photon matrix solver.
    """
    # =========================================================================
    # 1. PHYSICAL CONSTANTS & FILE PARSING
    # =========================================================================
    hc_ev = 1.23984193e-4      # h * c in eV * cm
    kb_ev = 8.61733326e-5      # k_B in eV / K
    c_cm_s = 2.99792458e10     # c in cm / s
    h_cgs = 6.62607015e-27     # h in erg * s
    h_SI = 6.62607015e-34      # h in Joules * s
    gamma = 1.2512e-7
    
    raw_frequencies = []
    intensities = []
    scale_factors = []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith('\\') or line.startswith('|'):
                continue
            tokens = line.split()
            if len(tokens) >= 4:
                try:
                    raw_frequencies.append(float(tokens[1]))
                    intensities.append(float(tokens[2]))
                    scale_factors.append(float(tokens[3]))
                except ValueError:
                    continue

    freq_scaled_cm = np.array(raw_frequencies) * np.array(scale_factors)
    freq_ev = freq_scaled_cm * hc_ev
    einstein_A = gamma * (freq_scaled_cm ** 2) * np.array(intensities)

    active_mask = einstein_A > 0
    cooling_freq_ev = freq_ev[active_mask]
    cooling_A = einstein_A[active_mask]

    cs_energies_ev = cross_section_table[:, 0]
    cs_sigmas_cm2 = cross_section_table[:, 1]
    
    def get_sigma_uv(energy_ev):
        if energy_ev <= 0.0 or energy_ev > 13.6:
            return 0.0
        return np.interp(energy_ev, cs_energies_ev, cs_sigmas_cm2, left=0.0, right=0.0)

    # Core Thermodynamics: Internal Energy U(T) and Heat Capacity Cv(T)
    def calculate_internal_energy(T):
        x = freq_ev / (kb_ev * T)
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            thermal_occ = np.where(x > 50.0, np.exp(-x) / (1.0 - np.exp(-x)), 1.0 / np.expm1(x))
            thermal_occ = np.where(np.isnan(thermal_occ) | np.isinf(thermal_occ), 0.0, thermal_occ)
        return np.sum(freq_ev * thermal_occ)

    def calculate_heat_capacity_ev_k(T):
        x = freq_ev / (kb_ev * T)
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            val = np.where(x > 50.0,
                           (x ** 2) * np.exp(-x) / (1.0 - np.exp(-x))**2,
                           (x / np.expm1(x))**2 * np.exp(x))
            val = np.where(np.isnan(val) | np.isinf(val), 0.0, val)
        return np.sum(kb_ev * val)

    # =========================================================================
    # 2. RIGOROUS INTEGRATED HEATING POWER & PHOTON RATES
    # =========================================================================
    eval_energies = np.linspace(6.0, 13.6, 1000)
    dE = eval_energies[1] - eval_energies[0]
    
    R_abs = 0.0
    P_heating_cgs = 0.0
    
    for E_phot in eval_energies:
        nu = (E_phot / hc_ev) * c_cm_s
        I_nu = radiation_field_func(nu)   
        sigma_val = get_sigma_uv(E_phot)  
        
        if sigma_val <= 0.0:
            continue
            
        dnu_step = (dE / hc_ev) * c_cm_s
        
        # Photon Absorption Count Rate Integration (CGS)
        photon_flux_density = 4.0 * np.pi * I_nu / (h_cgs * nu)
        R_abs += photon_flux_density * sigma_val * dnu_step
        
        # Energy Power Integration (CGS)
        P_heating_cgs += 4.0 * np.pi * I_nu * sigma_val * dnu_step

    P_heating_ev_s = P_heating_cgs / 1.602176634e-12

    # =========================================================================
    # 3. DL01 INTEGRATED COOLING LIFETIME (TAU_COOL) IMPLEMENTATION
    # =========================================================================
    sol_1st = root_scalar(lambda T: calculate_internal_energy(T) - 13.6, bracket=[15.0, 5000.0], method='brentq')
    T_max_1st = sol_1st.root
    
    # Solve for the lower temperature limit representing 99% of energy loss (0.136 eV)
    sol_min_cool = root_scalar(lambda T: calculate_internal_energy(T) - 0.136, bracket=[1.0, T_max_1st], method='brentq')
    T_min_cool = sol_min_cool.root
    
    # Numerical quadrature integration of Cv(T) / P_cooling(T) from T_min_cool to T_max_1st
    t_fine_mesh = np.linspace(T_min_cool, T_max_1st, 200)
    dT_mesh = t_fine_mesh[1] - t_fine_mesh[0]
    integrated_tau_cool = 0.0
    
    for T in t_fine_mesh:
        Cv = calculate_heat_capacity_ev_k(T)
        # Compute cooling power at this temperature step stably
        x = cooling_freq_ev / (kb_ev * T)
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            thermal_occ = np.where(x > 50.0, np.exp(-x) / (1.0 - np.exp(-x)), 1.0 / np.expm1(x))
            thermal_occ = np.where(np.isnan(thermal_occ) | np.isinf(thermal_occ), 0.0, thermal_occ)
        P_cooling_ev_s = np.sum(cooling_freq_ev * cooling_A * thermal_occ)
        
        if P_cooling_ev_s > 0:
            integrated_tau_cool += (Cv / P_cooling_ev_s) * dT_mesh

    # Evaluate stacking index using the true integrated cooling lifetime
    stacking_index = integrated_tau_cool * R_abs

    # =========================================================================
    # 4. REGIME BALANCING & DYNAMIC CEILING EXPANSION
    # =========================================================================
    if stacking_index < threshold:
        print(f"Regime: SINGLE-PHOTON STOCHASTIC (Stacking Index = {stacking_index:.4e}, tau_cool = {integrated_tau_cool:.2f} s)")
        t_max = 3. * T_max_1st
        use_multi_photon = False
    else:
        print(f"Regime: MULTI-PHOTON STACKING / CONTINUOUS (Stacking Index = {stacking_index:.4e}, tau_cool = {integrated_tau_cool:.2f} s)")
        
        def equilibrium_objective(T):
            x = cooling_freq_ev / (kb_ev * T)
            with np.errstate(over='ignore', under='ignore', invalid='ignore'):
                thermal_occ = np.where(x > 50.0, np.exp(-x) / (1.0 - np.exp(-x)), 1.0 / np.expm1(x))
                thermal_occ = np.where(np.isnan(thermal_occ) | np.isinf(thermal_occ), 0.0, thermal_occ)
            P_cooling_ev_s = np.sum(cooling_freq_ev * cooling_A * thermal_occ)
            return P_cooling_ev_s - P_heating_ev_s
            
        sol_eq = root_scalar(equilibrium_objective, bracket=[1.0, 5000.0], method='brentq')
        t_eq = sol_eq.root
        print(f"Calculated True Equilibrium Temperature: {t_eq:.2f} K")
        
        # EXTENDED CEILING CORRECTION: Extend the grid capacity to 3x Lyman strikes above equilibrium
        u_ceiling = calculate_internal_energy(t_eq) +  13.6
        sol_max = root_scalar(lambda T: calculate_internal_energy(T) - u_ceiling, bracket=[t_eq, 10000.0], method='brentq')
        t_max = 3. * sol_max.root
        use_multi_photon = True

    # =========================================================================
    # 5. GRID DISCRETIZATION
    # =========================================================================
    t_edges = np.logspace(np.log10(t_min), np.log10(t_max), num_bins + 1)
    t_centers = 0.5 * (t_edges[:-1] + t_edges[1:])
    delta_t = np.diff(t_edges)
    
    u_edges = np.array([calculate_internal_energy(t) for t in t_edges])
    u_centers = np.array([calculate_internal_energy(t) for t in t_centers])

    # =========================================================================
    # 6. TRANSITION RATES MATRIX GENERATION
    # =========================================================================
    W_up = np.zeros((num_bins, num_bins))
    W_down_adjacent = np.zeros(num_bins)

    for j in range(num_bins):
        T_j = t_centers[j]
        U_j = u_centers[j]
        
        if j > 0:
            x = cooling_freq_ev / (kb_ev * T_j)
            with np.errstate(over='ignore', under='ignore', invalid='ignore'):
                thermal_occ = np.where(x > 50.0, np.exp(-x) / (1.0 - np.exp(-x)), 1.0 / np.expm1(x))
                thermal_occ = np.where(np.isnan(thermal_occ) | np.isinf(thermal_occ), 0.0, thermal_occ)
            mode_rates = cooling_A * thermal_occ
            P_IR_ev_per_s = np.sum(cooling_freq_ev * mode_rates)
            bin_width_ev = u_centers[j] - u_centers[j-1]
            W_down_adjacent[j] = (P_IR_ev_per_s / bin_width_ev) + 1e-30

        for k in range(j + 1, num_bins):
            u_req_min = u_edges[k] - U_j
            u_req_max = u_edges[k+1] - U_j
            e_photon_mid = 0.5 * (u_req_min + u_req_max)
            
            if e_photon_mid > 13.6 or e_photon_mid <= 0:
                continue
                
            nu_min = (u_req_min / hc_ev) * c_cm_s
            nu_max = (u_req_max / hc_ev) * c_cm_s
            nu_mid = 0.5 * (nu_min + nu_max)
            delta_nu = nu_max - nu_min
            
            sigma_uv_dynamic = get_sigma_uv(e_photon_mid)
            if sigma_uv_dynamic <= 0.0:
                continue
                
            flux_density = radiation_field_func(nu_mid)  
            # Scale absorption rate back using the proper CGS Planck constant
            absorption_rate = 4.0 * np.pi * (flux_density / (h_cgs * nu_mid)) * sigma_uv_dynamic * delta_nu
            W_up[j, k] = max(0.0, absorption_rate)

    # =========================================================================
    # 7. SOLVER ROUTINGS
    # =========================================================================
    if not use_multi_photon:
        log_f = np.zeros(num_bins)
        log_f[0] = 0.0

        for f in range(1, num_bins):
            log_terms = []
            for j in range(f):
                cumulative_upward_rate = np.sum(W_up[j, f:])
                if cumulative_upward_rate > 0 and log_f[j] > -700:
                    log_terms.append(log_f[j] + np.log(cumulative_upward_rate))
            
            if not log_terms:
                log_f[f] = -np.inf
                continue
                
            max_log_term = np.max(log_terms)
            log_flux_sum = max_log_term + np.log(np.sum(np.exp(log_terms - max_log_term)))
            log_f[f] = log_flux_sum - np.log(W_down_adjacent[f])

        max_log_f = np.max(log_f[np.isfinite(log_f)])
        f_discrete = np.exp(log_f - max_log_f)
    
    else:
        M = np.zeros((num_bins, num_bins))
        for i in range(num_bins):
            total_outflow = 0.0
            if i > 0:
                total_outflow += W_down_adjacent[i]
            total_outflow += np.sum(M_up_elements := W_up[i, :])
            M[i, i] = -total_outflow
            
            if i < num_bins - 1:
                M[i, i+1] = W_down_adjacent[i+1]
                
            for j in range(i):
                M[i, j] = W_up[j, i]
                
        null_vecs = null_space(M)
        if null_vecs.size == 0:
            f_discrete = np.zeros(num_bins)
            f_discrete[np.searchsorted(t_centers, t_eq)] = 1.0
        else:
            f_discrete = np.abs(null_vecs[:, 0])

    total_sum = np.sum(f_discrete)
    f_T = (f_discrete / total_sum) / delta_t

    return t_centers, f_T

import numpy as np

def compute_total_time_averaged_ir_rate(file_path, t_centers, f_T, t_min=15.0):
    """
    Computes the total time-averaged macroscopic IR photon emission rate (s^-1) 
    of a PAH molecule by integrating its temperature-dependent thermal emission rate 
    over the steady-state temperature distribution f(T).
    
    Parameters:
    -----------
    file_path : str
        Path to the NASA Ames TRANSITIONS text file.
    t_centers : ndarray
        1D array containing the center temperatures of each bin (K) from the adaptive solver.
    f_T : ndarray
        1D array representing the normalized continuous probability distribution function f(T) (K^-1).
    t_min : float
        The lower boundary used for the log-spaced temperature grid (K). Default is 15.0 K.
        
    Returns:
    --------
    float
        The total time-averaged macroscopic IR photon emission rate (s^-1).
    """
    # =========================================================================
    # 1. PHYSICAL CONSTANTS & MOLECULAR FILE PARSING (ONCE)
    # =========================================================================
    hc_ev = 1.23984193e-4      # h * c in eV * cm
    kb_ev = 8.61733326e-5      # k_B in eV / K
    gamma = 1.2512e-7          # Database Intensity-to-A factor
    
    raw_frequencies = []
    intensities = []
    scale_factors = []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith('\\') or line.startswith('|'):
                continue
            tokens = line.split()
            if len(tokens) >= 4:
                try:
                    raw_frequencies.append(float(tokens[1]))
                    intensities.append(float(tokens[2]))
                    scale_factors.append(float(tokens[3]))
                except ValueError:
                    continue

    if not raw_frequencies:
        raise ValueError(f"Could not parse valid spectroscopic data from {file_path}")

    freq_raw = np.array(raw_frequencies)
    ints = np.array(intensities)
    scales = np.array(scale_factors)
    
    # Process frequencies and Einstein A channels globally
    freq_scaled_cm = freq_raw * scales 
    freq_ev = freq_scaled_cm * hc_ev
    einstein_A = gamma * (freq_scaled_cm ** 2) * ints

    # =========================================================================
    # 2. DYNAMIC LOG-GRID EDGES RECONSTRUCTION
    # =========================================================================
    num_bins = len(t_centers)
    # Recover precise grid endpoint ceiling from log-spacing structure:
    t_max = 10**(2 * np.log10(t_centers[-1]) - np.log10(t_centers[-2]))
    
    t_edges = np.logspace(np.log10(t_min), np.log10(t_max), num_bins + 1)
    delta_t = np.diff(t_edges)
    
    # Extract discrete probability mass fractions: P_j = f(T_j) * Delta_T_j
    f_discrete = f_T * delta_t

    # =========================================================================
    # 3. VECTORIZED EVALUATION OF EMISSION CONSTANTS K_thermal(T)
    # =========================================================================
    K_thermal_array = np.zeros(num_bins)
    
    for j in range(num_bins):
        T_j = t_centers[j]
        
        # Guard baseline background elements to avoid division by zero or log failures
        if T_j <= 15.0:
            K_thermal_array[j] = 0.0
            continue
            
        # A. Evaluate total internal energy U(T_j) stably
        x = freq_ev / (kb_ev * T_j)
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            thermal_occ = np.where(x > 50.0, np.exp(-x) / (1.0 - np.exp(-x)), 1.0 / np.expm1(x))
            thermal_occ = np.where(np.isnan(thermal_occ) | np.isinf(thermal_occ), 0.0, thermal_occ)
        U_Tj = np.sum(freq_ev * thermal_occ)
        
        # B. Evaluate spontaneous emission channel rates stably
        mode_emission_rates = einstein_A * thermal_occ
        
        # C. Store K_thermal constant = Total photon emission rate (in s^-1)
        K_thermal_array[j] = np.sum(mode_emission_rates)

    # =========================================================================
    # 4. TIME-AVERAGED SUM INTEGRATION
    # =========================================================================
    # Total Macroscopic Rate = Sum( P_j * K_thermal(T_j) )
    total_time_averaged_ir_rate = np.sum(f_discrete * K_thermal_array)
    
    return total_time_averaged_ir_rate

import numpy as np

def compute_total_photon_absorption_rate(radiation_field_func, cross_section_table):
    """
    Computes the total number of UV/visible photons absorbed by a PAH molecule 
    per second (R_abs) by convolving the target radiation field with the 
    molecular absorption cross-section across the accessible spectrum (6.0 to 13.6 eV).
    
    Parameters:
    -----------
    radiation_field_func : callable
        Function taking frequency (Hz) and returning mean energy spectral 
        intensity I_nu in erg / (cm^2 * s * Hz * str).
    cross_section_table : ndarray
        A 2D numpy array of shape (N, 2). 
        Column 0: Photon Energy in eV.
        Column 1: Absolute spatial cross section sigma_UV in cm^2 for the whole molecule.
        
    Returns:
    --------
    float
        The total macroscopic photon absorption rate R_abs (photons / s).
    """
    # =========================================================================
    # 1. PHYSICAL & CONVERSION CONSTANTS
    # =========================================================================
    hc_ev = 1.23984193e-4      # h * c in eV * cm
    c_cm_s = 2.99792458e10     # c in cm / s
    h_cgs = 6.62607015e-27     # h in erg * s (Mandatory for CGS intensity flux)
    
    # =========================================================================
    # 2. TABULATED CROSS-SECTION INTERPOLATOR
    # =========================================================================
    cs_energies_ev = cross_section_table[:, 0]
    cs_sigmas_cm2 = cross_section_table[:, 1]
    
    def get_sigma_uv(energy_ev):
        if energy_ev <= 0.0 or energy_ev > 13.6:
            return 0.0
        return np.interp(energy_ev, cs_energies_ev, cs_sigmas_cm2, left=0.0, right=0.0)

    # =========================================================================
    # 3. NUMERICAL INTEGRATION IN CGS ENERGY SPACE
    # =========================================================================
    # We sample a fine mesh between the operational Habing UV boundaries (6.0 to 13.6 eV)
    eval_energies = np.linspace(6.0, 13.6, 1000)
    dE = eval_energies[1] - eval_energies[0]
    
    R_abs = 0.0  # Accumulator for total photons absorbed per second
    
    for E_phot in eval_energies:
        # Translate the current energy step into active frequency tracking values
        nu = (E_phot / hc_ev) * c_cm_s
        
        # Pull intensity from your scaled CGS wrapper
        I_nu = radiation_field_func(nu)   # erg / (cm^2 * s * Hz * str)
        sigma_val = get_sigma_uv(E_phot)  # cm^2
        
        if sigma_val <= 0.0:
            continue
            
        # 4pi integrated solid-angle energy flux = 4 * pi * I_nu  (erg / cm^2 / s / Hz)
        # Convert energy flux to photon flux density by dividing by single-photon energy:
        # Flux_photon = (4 * pi * I_nu) / (h_cgs * nu)  [photons / cm^2 / s / Hz]
        photon_flux_density = 4.0 * np.pi * I_nu / (h_cgs * nu)
        
        # Translate the current energy interval step (dE) to its frequency span width (dnu)
        dnu_step = (dE / hc_ev) * c_cm_s
        
        # Total rate element = Photon Flux Density * Molecular Cross-Section * Frequency Step Width
        R_abs += photon_flux_density * sigma_val * dnu_step
        
    return R_abs


def load_kurucz_u_E(teff: int):
    """
    Loads the Kurucz stellar atmosphere spectrum for a given effective temperature Teff,
    and returns a function u_E_func(E) that gives the undiluted surface energy density
    u_E in erg cm^-3 eV^-1.
    
    Parameters
    ----------
    teff : int
        Effective temperature in Kelvin (must be one of 10000, 11000, 12500, 15000, 20000, 25000, 30000, 40000).
        
    Returns
    -------
    callable
        Function taking photon energy E (eV) and returning u_E.
    """
    valid_teffs = [10000, 11000, 12500, 15000, 20000, 25000, 30000, 40000]
    if teff not in valid_teffs:
        raise ValueError(f"Teff must be one of {valid_teffs}")
        
    file_path = os.path.join(_EXTERNAL_DATA_DIR, f"kp00_{teff}")
    data = np.loadtxt(file_path, skiprows=3)
    
    wav_nm = data[:, 0]
    I_lam = data[:, 1]
    
    # E (eV) = 1239.84193 / wav_nm (nm)
    E = 1239.84193 / wav_nm
    
    # u_E = (4*pi/c) * I_lam * (d_lambda/d_E) = (4*pi/c) * I_lam * (1239.84193 / E^2)
    c_cgs = 2.99792458e10
    u_E = (4.0 * np.pi / c_cgs) * I_lam * (1239.84193 / E**2)
    
    # Sort by E
    idx = np.argsort(E)
    E_sorted = E[idx]
    u_E_sorted = u_E[idx]
    
    def u_E_func(E_val):
        if E_val <= 0.0 or E_val > 13.6:
            return 0.0
        return float(np.interp(E_val, E_sorted, u_E_sorted, left=0.0, right=0.0))
        
    return u_E_func


def load_kurucz_I_nu(teff: int):
    """
    Loads the Kurucz stellar atmosphere spectrum for a given effective temperature Teff,
    and returns a function I_nu_func(nu) that gives the undiluted surface specific intensity
    I_nu in erg cm^-2 s^-1 Hz^-1 sr^-1.
    
    Parameters
    ----------
    teff : int
        Effective temperature in Kelvin (must be one of 10000, 11000, 12500, 15000, 20000, 25000, 30000, 40000).
        
    Returns
    -------
    callable
        Function taking frequency nu (Hz) and returning I_nu.
    """
    valid_teffs = [10000, 11000, 12500, 15000, 20000, 25000, 30000, 40000]
    if teff not in valid_teffs:
        raise ValueError(f"Teff must be one of {valid_teffs}")
        
    file_path = os.path.join(_EXTERNAL_DATA_DIR, f"kp00_{teff}")
    data = np.loadtxt(file_path, skiprows=3)
    
    wav_nm = data[:, 0]
    I_lam = data[:, 1]
    
    c_cgs = 2.99792458e10
    nu = c_cgs / (wav_nm * 1e-7)
    
    # I_nu = I_lam * (lambda_nm^2 * 1e-7 / c_cgs)
    I_nu = I_lam * (wav_nm**2 * 1e-7) / c_cgs
    
    idx = np.argsort(nu)
    nu_sorted = nu[idx]
    I_nu_sorted = I_nu[idx]
    
    def I_nu_func(nu_val):
        h_SI = 6.62607015e-34
        eV2J = 1.602176634e-19
        E = (h_SI * nu_val) / eV2J
        if E > 13.6 or nu_val <= 0.0:
            return 0.0
        return float(np.interp(nu_val, nu_sorted, I_nu_sorted, left=0.0, right=0.0))
        
    return I_nu_func


if __name__ == "__main__":


    # file_path = importlib_resources.files("amespahdbpythonsuite")
    # xml = file_path / "/Users/currodri/Documents/GitHub/CALIMA/external_data/pahdb-complete-theoretical-v4.00.xml"
    # pahdb = AmesPAHdb(
    #     filename=xml,
    #     check=False,
    #     cache=False,
    #     update=False
    # )
    # extract_transitions(pahdb,54,'neutral','./')

    # T_grid, E_per_mode_cm, E_per_mode_eV = compute_thermal_energy_from_file('/Users/currodri/Documents/GitHub/CALIMA/model_data/PAH_states/C54H18_0.dat')

    # plt.loglog(T_grid, E_per_mode_eV, 'r-')
    # plt.xlabel('Temperature [K]', fontsize=12)
    # plt.ylabel('E/(3N-6) [eV]', fontsize=12)
    # plt.show()


    # E = np.linspace(1e-10, 40, 1000)
    # T_arr = np.zeros(len(E))
    # K_arr = np.zeros(len(E))
    # K_Heven = np.zeros(len(E))
    # K_H2even = np.zeros(len(E))
    # for i, e in enumerate(E):
    #     T_arr[i], E_val, K_arr[i] = compute_thermal_ir_rate('C24H12_0.dat', e)
    #     K_Heven[i] = compute_rrkm_dissociation_rate('C24H12_0.dat', e, 4.5, 44.8)
    #     K_H2even[i] = compute_rrkm_dissociation_rate('C24H12_0.dat', e, 3.52, -53.1)
    # plt.plot(E, np.log10(K_arr), 'r-',label='IR')
    # plt.plot(E, np.log10(K_Heven), 'b-',label='H-loss')
    # plt.plot(E, np.log10(K_H2even), 'g-',label='H2-loss')
    # plt.xlabel('E [eV]', fontsize=12)
    # plt.ylabel('log(k) [s^-1]', fontsize=12)
    # plt.ylim(-10,15)
    # plt.show()

    a0 = afromNc(54)
    w,C_abs = get_absorption_cross_section(0,a0)
    hc_ev = 1.23984193e-4      # h * c in eV * cm
    E = hc_ev / w

    MATHIS_BASE_G0 = compute_base_g0(Mathis83_radiation_field)
    print(f"Calculated base Mathis (1983) field strength: G0 = {MATHIS_BASE_G0:.4f}")

    current_field_wrapper = lambda nu: mathis83_to_gd89_interface(nu, target_G0=1e0, base_G0=MATHIS_BASE_G0)

    T_grid, f_profile = compute_gd89_temperature_distribution(
            file_path="/Users/currodri/Documents/GitHub/CALIMA/model_data/PAH_states/C54H18_0.dat",
            radiation_field_func=current_field_wrapper,
            cross_section_table=np.array([E, C_abs]).T,
            t_min=1.0,
            num_bins=150
        )

    plt.loglog(T_grid, f_profile, label='G0 = 1 (no adaptive)')

    T_grid, f_profile = compute_adaptive_temperature_distribution(
            file_path="/Users/currodri/Documents/GitHub/CALIMA/model_data/PAH_states/C54H18_0.dat",
            radiation_field_func=current_field_wrapper,
            cross_section_table=np.array([E, C_abs]).T,
            t_min=1.0,
            num_bins=150
        )

    plt.loglog(T_grid, f_profile, label='G0 = 1')

    current_field_wrapper = lambda nu: mathis83_to_gd89_interface(nu, target_G0=1e4, base_G0=MATHIS_BASE_G0)

    T_grid, f_profile = compute_adaptive_temperature_distribution(
            file_path="/Users/currodri/Documents/GitHub/CALIMA/model_data/PAH_states/C54H18_0.dat",
            radiation_field_func=current_field_wrapper,
            cross_section_table=np.array([E, C_abs]).T,
            t_min=1.0,
            num_bins=150
        )
    plt.loglog(T_grid, f_profile, label='G0 = 1e4')
    plt.ylim(1e-13,10)
    plt.xlabel('Temperature [K]', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    plt.legend()
    plt.show()

    # Generate your targeted log-spaced G0 parameters array
    g0_grid = np.logspace(0, 5, num=20)  # Spans 10^0 (1) to 10^5 cleanly

    computed_H_rates = np.zeros(len(g0_grid))
    computed_H2_rates = np.zeros(len(g0_grid))
    computed_IR_rates = np.zeros(len(g0_grid))
    computed_abs_rates = np.zeros(len(g0_grid))

    for i, g0 in enumerate(g0_grid):
        current_field_wrapper = lambda nu: mathis83_to_gd89_interface(nu, target_G0=g0, base_G0=MATHIS_BASE_G0)
        
        # 1. Compute the steady-state temperature profile
        T_grid, f_profile = compute_adaptive_temperature_distribution(
            file_path="/Users/currodri/Documents/GitHub/CALIMA/model_data/PAH_states/C54H18_0.dat",
            radiation_field_func=current_field_wrapper,
            cross_section_table=np.array([E, C_abs]).T,
            t_min=1.0,
            num_bins=150
        )
        
        # 2. Integrate the total time-averaged dissociation rate (using Andrews/LePage parameters for H-loss)
        rate = compute_total_dissociation_rate(
            t_centers=T_grid,
            f_T=f_profile,
            E_act_ev=4.8,
            dS_cl_jk=20.92,
            t_min=1.0
        )
        computed_H_rates[i] = rate

        rate = compute_total_dissociation_rate(
            t_centers=T_grid,
            f_T=f_profile,
            E_act_ev=3.52,
            dS_cl_jk=-53.1,
            t_min=1.0
        )
        computed_H2_rates[i] = rate

        rate = compute_total_time_averaged_ir_rate(
            file_path="/Users/currodri/Documents/GitHub/CALIMA/model_data/PAH_states/C54H18_0.dat",
            t_centers=T_grid,
            f_T=f_profile,
            t_min=1.0
        )
        computed_IR_rates[i] = rate

        rate = compute_total_photon_absorption_rate(
            radiation_field_func=current_field_wrapper,
            cross_section_table=np.array([E, C_abs]).T
        )
        computed_abs_rates[i] = rate

    H_rate = computed_H_rates
    H2_rate = computed_H2_rates

    plt.loglog(g0_grid, H_rate, 'r-',label='H-loss')
    plt.loglog(g0_grid, H2_rate, 'b-',label='H2-loss')

    fit_params = [-14.148, 1.962, -0.031, -0.009, 0.003]
    g0_model = np.logspace(0, 5, num=20)
    k_model = fit_params[0] + fit_params[1] * np.log10(g0_model) + fit_params[2] * np.log10(g0_model)**2 + fit_params[3] * np.log10(g0_model)**3 + fit_params[4] * np.log10(g0_model)**4
    plt.loglog(g0_model, 10**k_model, 'k-',label='Fit')
    plt.xlabel('G0', fontsize=12)
    plt.ylabel('log(k) [s-1]', fontsize=12)
    plt.legend()
    plt.show()

