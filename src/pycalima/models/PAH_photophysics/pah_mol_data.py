"""
pah_mol_data.py — PAHdb mode loading, internal energy machinery, RRKM rates.

All functions that need vibrational modes call load_pah_modes() once; results
are cached so subsequent calls for the same file are free.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
import os
import re


import numpy as np
from scipy.optimize import root_scalar

if TYPE_CHECKING:
    # amespahdbpythonsuite is an optional dependency from an external repo
    # (https://github.com/PAHdb/AmesPAHdbPythonSuite). It is referenced here
    # only to annotate extract_transitions()'s caller-supplied `pahdb`
    # argument -- nothing in pyCALIMA ever constructs an AmesPAHdb -- so
    # importing it under TYPE_CHECKING keeps it out of the runtime graph.
    # Install with:  pip install 'pycalima[pahdb]'
    from amespahdbpythonsuite.amespahdb import AmesPAHdb

# ---------------------------------------------------------------------------
# Module-level caches
# ---------------------------------------------------------------------------
_MODE_CACHE: dict = {}   # file_path -> (freq_ev, einstein_A)
_TABLE_CACHE: dict = {}  # table_path -> ndarray

# Physical constants
_HC_EV  = 1.23984193e-4   # h·c in eV·cm
_KB_EV  = 8.61733326e-5   # k_B in eV/K
_KB_J_K = 1.380649e-23    # k_B in J/K
_H_J_S  = 6.62607015e-34  # h in J·s
_R_GAS  = 8.31446261      # gas constant in J/(mol·K)
_GAMMA  = 1.2512e-7       # PAHdb intensity → Einstein-A factor


def load_pah_modes(file_path: str):
    """
    Parse a NASA Ames PAHdb transitions file and return (freq_ev, einstein_A).

    Columns expected: UID  frequency[cm^-1]  intensity[km/mol]  scale  symmetry ...

    Returns
    -------
    freq_ev : ndarray  — scaled vibrational frequencies in eV
    einstein_A : ndarray — Einstein A coefficients in s^-1
    """
    if file_path in _MODE_CACHE:
        return _MODE_CACHE[file_path]

    raw_freq, ints, scales = [], [], []
    with open(file_path, 'r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('\\') or line.startswith('|'):
                continue
            tokens = line.split()
            if len(tokens) >= 4:
                try:
                    raw_freq.append(float(tokens[1]))
                    ints.append(float(tokens[2]))
                    scales.append(float(tokens[3]))
                except ValueError:
                    continue

    if not raw_freq:
        raise ValueError(f"No valid transition data from {file_path}")

    freq_cm  = np.array(raw_freq) * np.array(scales)
    freq_ev  = freq_cm * _HC_EV
    einstein_A = _GAMMA * freq_cm**2 * np.array(ints)

    _MODE_CACHE[file_path] = (freq_ev, einstein_A)
    return freq_ev, einstein_A


def _qho_energy(freq_ev, T):
    """Internal energy U(T) from quantum harmonic oscillator."""
    x = freq_ev / (_KB_EV * T)
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        occ = np.where(x > 50.0, np.exp(-x) / (1.0 - np.exp(-x)), 1.0 / np.expm1(x))
        occ = np.where(np.isnan(occ) | np.isinf(occ), 0.0, occ)
    return np.sum(freq_ev * occ)


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def extract_transitions(pahdb: AmesPAHdb, Nc: int, charge: str, output_path: str):
    """
    Extract vibrational transitions from the NASA Ames PAHdb for a given
    number of carbon atoms and charge, and write them to disk.
    """
    uids = pahdb.search(f'c={Nc} {charge} fe=0 mg=0 o=0 si=0 n=0')
    if len(uids) == 0:
        print('ERROR: No PAH found with the given number of carbon atoms and charge')
        return []

    file_list = []
    for id in uids:
        pah  = pahdb.getspeciesbyuid(id)
        info = pah.print(str=True)

        def _parse(label):
            m = re.search(rf'^{label}\s*:\s*([^\s\n]+)', info, re.MULTILINE)
            if m:
                return m.group(1)
            print(f"ERROR: {label} not found.\n{info}")
            return None

        formula   = _parse('FORMULA')
        charge_val = _parse('CHARGE')
        nsolo     = _parse('N_SOLO')
        nduo      = _parse('N_DUO')
        if None in (formula, charge_val, nsolo, nduo):
            continue

        transitions = pah.transitions()
        outfile = os.path.join(output_path, f"{formula}_{charge_val}.dat")
        transitions.write(outfile)

        with open(outfile, 'r') as f:
            content = f.read()
        meta = f"\n\\N_SOLO     : {nsolo} \n\\N_DUO      : {nduo} "
        content = content.replace("\\ SPECIES", "\\ SPECIES" + meta)
        with open(outfile, 'w') as f:
            f.write(content)

        file_list.append(outfile)

    return file_list


def compute_thermal_energy_from_file(file_path, t_min=1e2, t_max=1e4, nt=100):
    """
    Compute U(T) / (3N-6) on a temperature grid using PAHdb modes.

    Returns (temperatures, energy_per_mode_cm, energy_per_mode_eV).
    """
    freq_ev, _ = load_pah_modes(file_path)
    num_modes  = len(freq_ev)
    temperatures = np.logspace(np.log10(t_min), np.log10(t_max), nt)
    total_energy = np.array([_qho_energy(freq_ev, T) for T in temperatures])
    energy_per_mode_eV = total_energy / num_modes
    energy_per_mode_cm = energy_per_mode_eV / _HC_EV
    return temperatures, energy_per_mode_cm, energy_per_mode_eV


def compute_thermal_ir_rate(file_path, internal_energy_ev):
    """
    Map internal energy E to canonical temperature T and compute K_thermal(T).

    Returns (canonical_T, internal_energy_ev, K_thermal) where K_thermal is in s^-1.
    """
    freq_ev, einstein_A = load_pah_modes(file_path)

    if internal_energy_ev <= 0.0:
        return 0.0, internal_energy_ev, 0.0

    def objective(T):
        return _qho_energy(freq_ev, T) - internal_energy_ev

    try:
        sol = root_scalar(objective, bracket=[1.0, 1e5], method='brentq')
        canonical_T = sol.root
    except ValueError as e:
        raise ValueError(
            f"Energy {internal_energy_ev} eV out of range for molecule in {file_path}"
        ) from e

    x = freq_ev / (_KB_EV * canonical_T)
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        occ = np.where(x > 50.0, np.exp(-x) / (1.0 - np.exp(-x)), 1.0 / np.expm1(x))
        occ = np.where(np.isnan(occ) | np.isinf(occ), 0.0, occ)
    mode_rates = einstein_A * occ
    K_thermal  = np.sum(freq_ev * mode_rates) / internal_energy_ev if internal_energy_ev > 0 else 0.0

    return canonical_T, internal_energy_ev, K_thermal


def compute_rrkm_dissociation_rate(file_path, internal_energy_ev, E_act_ev, dS_cl_jk):
    """
    TST/RRKM dissociation rate K_dis(E) using the effective temperature T_e.

    T_e = T_m * (1 - 0.2 * E_act / E) per Tielens (2005).
    K_dis = e * (k_B T_e / h) * exp(dS/R) * exp(-E_act / k_B T_e)

    Returns K_dis in s^-1, or 0.0 if E < E_act.
    """
    if internal_energy_ev < E_act_ev:
        return 0.0

    freq_ev, _ = load_pah_modes(file_path)

    def objective(T):
        return _qho_energy(freq_ev, T) - internal_energy_ev

    try:
        sol = root_scalar(objective, bracket=[1.0, 1e5], method='brentq')
        T_m = sol.root
    except ValueError:
        return 0.0

    T_e = T_m * (1.0 - 0.2 * E_act_ev / internal_energy_ev)
    if T_e <= 0:
        return 0.0

    K_dis = (np.exp(1.0)
             * (_KB_J_K * T_e / _H_J_S)
             * np.exp(dS_cl_jk / _R_GAS)
             * np.exp(-E_act_ev / (_KB_EV * T_e)))
    return K_dis


def compute_dissociation_rate_from_table(table_path: str, E_ev: float) -> float:
    """
    Log-log interpolation of k(E) from a two-column CSV [E_eV, k_s-1].

    Returns 0.0 if E_ev is outside the table range.
    """
    if table_path not in _TABLE_CACHE:
        _TABLE_CACHE[table_path] = np.loadtxt(table_path, delimiter=',')
    data   = _TABLE_CACHE[table_path]
    E_tab  = data[:, 0]
    k_tab  = data[:, 1]
    if E_ev < E_tab[0] or E_ev > E_tab[-1]:
        return 0.0
    log_k = np.interp(np.log10(E_ev), np.log10(E_tab), np.log10(np.maximum(k_tab, 1e-300)))
    return 10.0**log_k
