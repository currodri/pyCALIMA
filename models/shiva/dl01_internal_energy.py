"""
dl01_internal_energy.py — DL01 internal energy and microcanonical temperature.

Implements the two-component quantum harmonic oscillator (QHO) approximation
from Draine & Li (2001, ApJ 551, 807), Table 1, for the thermal energy of
graphitic PAH molecules.

The energy grid is parameterised by the number of carbon atoms Nc only;
no PAHdb mode files are required.

References
----------
Draine, B.T. & Li, A. 2001, ApJ, 551, 807 (DL01)
"""

import numpy as np
from scipy.optimize import brentq

# ── DL01 Table 1 parameters for graphitic PAHs ────────────────────────────
# Two characteristic Debye temperatures (K) and mode-count fractions per C atom:
#   group 1: low-frequency bending modes   (~600 cm⁻¹)
#   group 2: high-frequency stretching modes (~1740 cm⁻¹)
_DL01_MODES = [
    {'n': 3.1, 'theta': 863.0},   # θ₁ = 863 K  → E₁ = k_B θ₁ = 0.0744 eV
    {'n': 2.0, 'theta': 2504.0},  # θ₂ = 2504 K → E₂ = k_B θ₂ = 0.2157 eV
]

# ── physical constants (CGS) ──────────────────────────────────────────────
k_B_erg  = 1.380649e-16   # Boltzmann constant [erg K⁻¹]
eV_to_erg = 1.602176634e-12  # conversion factor [erg eV⁻¹]


def U_dl01(T, Nc):
    """
    Thermal internal energy U(T, Nc) [erg] from DL01 eq. 2.

    Uses two Einstein oscillator groups fit to graphitic PAH vibrational
    modes (DL01 Table 1).

    Parameters
    ----------
    T : float or array_like
        Temperature [K].  Clamped to ≥ 2.73 K internally to avoid
        numerical singularities at T→0.
    Nc : int or float
        Number of carbon atoms.

    Returns
    -------
    U : same shape as T, float
        Thermal energy [erg].
    """
    T = np.asarray(T, dtype=float)
    T = np.maximum(T, 2.73)
    scalar = T.ndim == 0
    T = np.atleast_1d(T)
    U = np.zeros_like(T)
    for mode in _DL01_MODES:
        n_i, theta_i = mode['n'], mode['theta']
        x = theta_i / T
        # Einstein oscillator occupation: handle large-x (x > 50) to avoid overflow
        with np.errstate(over='ignore', under='ignore'):
            denom = np.expm1(x)            # exp(x) − 1
            denom = np.where(denom == 0.0, 1e-300, denom)
            U += n_i * Nc * k_B_erg * theta_i / denom
    return float(U[0]) if scalar else U


def dU_dT_dl01(T, Nc):
    """
    Heat capacity dU/dT [erg K⁻¹] from DL01.

    Parameters
    ----------
    T : float or array_like
        Temperature [K].
    Nc : int or float
        Number of carbon atoms.

    Returns
    -------
    C : same shape as T, float
        Heat capacity [erg K⁻¹].
    """
    T = np.asarray(T, dtype=float)
    T = np.maximum(T, 2.73)
    scalar = T.ndim == 0
    T = np.atleast_1d(T)
    C = np.zeros_like(T)
    for mode in _DL01_MODES:
        n_i, theta_i = mode['n'], mode['theta']
        x = theta_i / T
        with np.errstate(over='ignore', under='ignore'):
            ex = np.exp(np.minimum(x, 500.0))
            # x² exp(x) / (exp(x) - 1)²
            fac = np.where(x > 50.0,
                           x**2 * np.exp(-x),     # large-x limit: x² e⁻ˣ
                           x**2 * ex / (ex - 1.0)**2)
            C += n_i * Nc * k_B_erg * fac
    return float(C[0]) if scalar else C


def T_micro_dl01(E_erg, Nc, T_min=2.73, T_max=3000.0):
    """
    Microcanonical temperature T_m such that U(T_m, Nc) = E_erg.

    Inverts U_dl01 via Brent's method.  When E_erg exceeds the classical
    limit, the temperature is extrapolated linearly using dU/dT at T_max.

    Parameters
    ----------
    E_erg : float
        Internal energy [erg].
    Nc : int or float
        Number of carbon atoms.
    T_min : float
        Lower bound for the inversion [K].  Default 2.73 K (CMB).
    T_max : float
        Upper bound for the root search [K].  Default 3000 K; raised
        automatically if U(T_max) < E_erg.

    Returns
    -------
    T_m : float
        Microcanonical temperature [K].
    """
    if E_erg <= 0.0:
        return T_min

    # Extend T_max until U(T_max) > E_erg (classical limit extrapolation)
    U_max = U_dl01(T_max, Nc)
    while U_max < E_erg:
        # linear extrapolation beyond T_max
        C_max = dU_dT_dl01(T_max, Nc)
        return T_max + (E_erg - U_max) / C_max

    U_min = U_dl01(T_min, Nc)
    if E_erg <= U_min:
        return T_min

    return brentq(lambda T: U_dl01(T, Nc) - E_erg, T_min, T_max, xtol=0.1)


def U_to_T_array(E_arr_erg, Nc):
    """
    Vectorised wrapper for T_micro_dl01 over an array of energies.

    Parameters
    ----------
    E_arr_erg : array_like
        Array of internal energies [erg].
    Nc : int or float
        Number of carbon atoms.

    Returns
    -------
    T_arr : ndarray
        Microcanonical temperatures [K], same shape as E_arr_erg.
    """
    E_arr = np.asarray(E_arr_erg, dtype=float)
    return np.array([T_micro_dl01(e, Nc) for e in E_arr.ravel()]).reshape(E_arr.shape)
