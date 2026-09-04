"""
pah_radiation.py — Kurucz stellar atmosphere radiation field loaders.

Only the 15000 K Kurucz field is physically appropriate for Andrews (2016)
comparisons.  The loaders accept other Teff values for generality but the
caller is responsible for using the right temperature.
"""

import os
import numpy as np
from pathlib import Path

_THIS_DIR        = os.path.dirname(os.path.abspath(__file__))
_CALIMA_ROOT     = os.path.abspath(os.path.join(_THIS_DIR, '..', '..'))
_EXTERNAL_DATA_DIR = os.path.join(_CALIMA_ROOT, 'external_data')

_VALID_TEFFS = [10000, 11000, 12500, 15000, 20000, 25000, 30000, 40000]


def load_kurucz_u_E(teff: int):
    """
    Return u_E_func(E) — undiluted surface energy density [erg cm^-3 eV^-1]
    for the Kurucz stellar atmosphere at effective temperature `teff`.

    Parameters
    ----------
    teff : int
        Stellar effective temperature in K.
        Must be one of 10000, 11000, 12500, 15000, 20000, 25000, 30000, 40000.

    Returns
    -------
    callable : E [eV] -> u_E [erg cm^-3 eV^-1]
    """
    if teff not in _VALID_TEFFS:
        raise ValueError(f"teff must be one of {_VALID_TEFFS}, got {teff}")

    file_path = os.path.join(_EXTERNAL_DATA_DIR, f"kp00_{teff}")
    data      = np.loadtxt(file_path, skiprows=3)
    wav_nm    = data[:, 0]
    I_lam     = data[:, 1]

    c_cgs = 2.99792458e10
    E     = 1239.84193 / wav_nm                              # eV
    u_E   = (4.0 * np.pi / c_cgs) * I_lam * (1239.84193 / E**2)

    idx      = np.argsort(E)
    E_s      = E[idx]
    u_E_s    = u_E[idx]

    def u_E_func(E_val):
        if E_val <= 0.0 or E_val > 13.6:
            return 0.0
        return float(np.interp(E_val, E_s, u_E_s, left=0.0, right=0.0))

    return u_E_func


def load_kurucz_I_nu(teff: int):
    """
    Return I_nu_func(nu) — undiluted surface specific intensity
    [erg cm^-2 s^-1 Hz^-1 sr^-1] for the Kurucz atmosphere at `teff`.

    The returned function returns 0 for photon energies above 13.6 eV (Lyman
    limit) or for non-positive frequencies.

    Parameters
    ----------
    teff : int
        Stellar effective temperature in K.

    Returns
    -------
    callable : nu [Hz] -> I_nu [erg cm^-2 s^-1 Hz^-1 sr^-1]
    """
    if teff not in _VALID_TEFFS:
        raise ValueError(f"teff must be one of {_VALID_TEFFS}, got {teff}")

    file_path = os.path.join(_EXTERNAL_DATA_DIR, f"kp00_{teff}")
    data      = np.loadtxt(file_path, skiprows=3)
    wav_nm    = data[:, 0]
    I_lam     = data[:, 1]

    c_cgs = 2.99792458e10
    nu    = c_cgs / (wav_nm * 1e-7)                          # Hz
    I_nu  = I_lam * (wav_nm**2 * 1e-7) / c_cgs              # erg/cm^2/s/Hz/sr

    idx     = np.argsort(nu)
    nu_s    = nu[idx]
    I_nu_s  = I_nu[idx]

    _h_SI  = 6.62607015e-34
    _eV2J  = 1.602176634e-19

    def I_nu_func(nu_val):
        if nu_val <= 0.0:
            return 0.0
        E = (_h_SI * nu_val) / _eV2J
        if E > 13.6:
            return 0.0
        return float(np.interp(nu_val, nu_s, I_nu_s, left=0.0, right=0.0))

    return I_nu_func
