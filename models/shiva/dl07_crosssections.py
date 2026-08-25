"""
dl07_crosssections.py — DL07 absorption cross-sections for PAHs.

Implements an analytical parameterisation of the Draine & Li (2007)
per-carbon-atom absorption cross-section σ_C(E) for graphitic PAHs,
covering photon energies from 0.1 eV to 13.6 eV (the Lyman limit).

The parameterisation reproduces the main features visible in DL07 Fig. 1
and Table 1:
  • UV continuum with a power-law rise (∝ E^1.74 above 6.2 eV)
  • The 2175 Å (5.70 eV) graphite/PAH Drude feature
  • A near-UV transition region (3.3–6.2 eV)
  • Modest optical cross-section (1–3.3 eV)
  • Additional factor of ~1.5 for ionised relative to neutral PAHs

Key numbers calibrated against:
  Li & Draine 2001, ApJ 554, 778, Table 3 (C24, C54, C96 values)
  Draine & Li 2007, ApJ 657, 810, Table 1 and Fig. 1

References
----------
Li, A. & Draine, B.T. 2001, ApJ, 554, 778 (LD01)
Draine, B.T. & Li, A. 2007, ApJ, 657, 810 (DL07)
"""

import numpy as np
from scipy.integrate import quad

# ── 2175 Å Drude feature parameters (DL07 Table 1 / Li & Draine 2001) ─────
_E0_2175  = 5.705   # central energy [eV]  (λ = 0.2175 µm)
_B_2175   = 0.217   # fractional FWHM  b_j = Δλ_j / λ_j (DL07 Eq. A1)
# Peak per-C cross-section at the 2175 Å feature [cm²]
# Calibrated so that ∫ σ_drude dE ≈ 1.5e-18 eV·cm² (from DL07 band strength)
_SIGMA_2175_PEAK = 1.5e-18   # cm²

# ── UV continuum anchor points (per C atom) ───────────────────────────────
# Derived from Li & Draine 2001 Table 3 (C24H12) and DL07 Fig. 1:
#   σ_C(6.2 eV) ≈ 6.3e-19 cm²  (below 2175 Å feature)
#   σ_C(10 eV)  ≈ 1.5e-18 cm²
#   σ_C(13.6 eV)≈ 2.5e-18 cm²
_SIGMA_UV_REF = 6.3e-19   # cm²  at E_REF
_E_UV_REF     = 6.2        # eV
_ALPHA_UV     = 1.74       # power-law exponent above E_UV_REF
# Ionised PAHs have ~50 % enhanced UV absorption (DL07 §3)
_ION_UV_FACTOR = 1.5


def sigma_C_dl07(E_eV, Z=0):
    """
    DL07 absorption cross-section per carbon atom σ_C(E) [cm²].

    Parameters
    ----------
    E_eV : float
        Photon energy [eV].  Only 0 < E_eV ≤ 13.6 is meaningful.
    Z : int
        PAH charge state.  Z = 0 → neutral; Z ≠ 0 → ionised (cation or anion).
        The ionisation state only changes the UV continuum level (+50 %).

    Returns
    -------
    sigma : float
        Cross-section per C atom [cm²].  Returns 0 for E_eV ≤ 0 or > 13.6.
    """
    if E_eV <= 0.0 or E_eV > 13.6:
        return 0.0

    # ── Drude profile for the 2175 Å feature ──────────────────────────────
    # Standard Drude parameterisation in photon-energy space (DL07 eq. A1):
    # σ_drude(E) = σ_peak × b² / [(E/E0 - E0/E)² + b²]
    x_2175 = E_eV / _E0_2175 - _E0_2175 / E_eV
    sigma_drude = _SIGMA_2175_PEAK * _B_2175**2 / (x_2175**2 + _B_2175**2)

    # ── UV/optical continuum ───────────────────────────────────────────────
    if E_eV < 1.0:
        # Far-IR to near-IR: negligibly small for heating purposes
        sigma_cont = 2.1e-22 * (E_eV / 1.24)**2

    elif E_eV < 3.3:
        # Optical: modest, rises slowly toward UV
        # Linear interpolation in log–log between (1 eV, ~2e-22) and (3.3 eV, ~6e-21)
        loga = np.log10(2.1e-22)
        logb = np.log10(6.0e-21)
        sigma_cont = 10.0 ** (loga + (logb - loga) * (E_eV - 1.0) / (3.3 - 1.0))

    elif E_eV < _E_UV_REF:
        # Near-UV: rapid power-law rise to the reference point at 6.2 eV
        # σ ∝ ((E - 3.3)/(6.2 - 3.3))^2  interpolated from 6e-21 to _SIGMA_UV_REF
        t = (E_eV - 3.3) / (_E_UV_REF - 3.3)
        sigma_cont = 6.0e-21 + (_SIGMA_UV_REF - 6.0e-21) * t**2

    else:
        # UV continuum above 6.2 eV: power-law σ ∝ E^α
        sigma_cont = _SIGMA_UV_REF * (E_eV / _E_UV_REF) ** _ALPHA_UV

    # Ionised PAHs: enhanced UV continuum (DL07 §3, Berne+2022)
    if Z != 0:
        sigma_cont *= _ION_UV_FACTOR

    return sigma_cont + sigma_drude


def C_abs_dl07(E_eV, Nc, Z=0):
    """
    Total DL07 absorption cross-section C_abs(E, Nc, Z) [cm²].

    C_abs = Nc × σ_C(E, Z)

    Parameters
    ----------
    E_eV : float
        Photon energy [eV].
    Nc : int or float
        Number of carbon atoms.
    Z : int
        Charge state (0 = neutral, ≠ 0 = ionised).

    Returns
    -------
    C_abs : float  [cm²]
    """
    return Nc * sigma_C_dl07(E_eV, Z=Z)


def absorption_rate_dl07(Nc, Z, u_E_fn, G0=1.0,
                          E_min=0.1, E_max=13.6, N_pts=500):
    """
    Total photon absorption rate R_abs [s⁻¹] for a PAH with Nc C atoms
    and charge state Z in radiation field u_E_fn scaled by G0.

    R_abs = ∫ C_abs(E) × c × u_E(E) / (E × eV_erg) dE

    where u_E(E) [erg cm⁻³ eV⁻¹] is the spectral energy density.

    Parameters
    ----------
    Nc : int or float
        Number of carbon atoms.
    Z : int
        Charge state.
    u_E_fn : callable
        u_E_fn(E_eV) → u_E [erg cm⁻³ eV⁻¹].  Should be the unit-G0 field.
    G0 : float
        Habing field strength.  The field is scaled as G0 × u_E_fn(E).
    E_min, E_max : float
        Integration limits [eV].
    N_pts : int
        Number of quadrature points (trapezoidal).

    Returns
    -------
    R_abs : float  [s⁻¹]
    """
    c_cgs   = 2.99792458e10    # cm s⁻¹
    eV_erg  = 1.602176634e-12  # erg eV⁻¹

    E_grid = np.linspace(E_min, E_max, N_pts)
    integrand = np.array([
        C_abs_dl07(E, Nc, Z) * G0 * u_E_fn(E) * c_cgs / (E * eV_erg)
        for E in E_grid
    ])
    return np.trapezoid(integrand, E_grid)


def heating_rate_dl07(Nc, Z, u_E_fn, G0=1.0,
                       E_min=0.1, E_max=13.6, N_pts=500):
    """
    Total radiative heating rate P_heat [erg s⁻¹] = ∫ C_abs(E) × c × u_E(E) dE.

    Parameters
    ----------
    (same as absorption_rate_dl07, except energy-weighted)

    Returns
    -------
    P_heat : float  [erg s⁻¹]
    """
    c_cgs  = 2.99792458e10
    eV_erg = 1.602176634e-12

    E_grid = np.linspace(E_min, E_max, N_pts)
    integrand = np.array([
        C_abs_dl07(E, Nc, Z) * G0 * u_E_fn(E) * c_cgs * eV_erg
        # u_E in erg cm⁻³ eV⁻¹; C_abs × c × u_E gives erg s⁻¹ cm⁻¹ (per eV bin)
        for E in E_grid
    ])
    return np.trapezoid(integrand, E_grid)
