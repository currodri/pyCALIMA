"""
wd01a_yields.py — WD01a photoionization yields and charge-balance rates.

Implements the photoelectric-heating framework of Weingartner & Draine (2001a)
for PAH-sized carbonaceous grains.  This is the "a" paper (ApJS 134, 263),
which gives ionization potentials, photoionization yields, and electron
recombination/attachment rates.

Key choices relative to the existing CALIMA code
-------------------------------------------------
- IP formula: WD01a eq. 2 (Coulomb shift only, graphite work function W = 4.4 eV)
- Yield:  y₁ × (E − IP)² / [(E − IP)² + W²]  with y₁ = 0.5, W = 3.0 eV
          (WD01a eqs. 7–12, PAH limit a ≪ electron mean free path)
- EA:     empirical power law fitted to Tobita+1994 / Malloci+2007 tabulated values
          for C24–C96
- Recombination: Draine & Sutin (1987) Coulomb-focused rate (WD01a §3)
- Attachment:    π a² v̄_e × s_e with s_e from Weingartner & Draine 2001a eq. 15

References
----------
Weingartner, J.C. & Draine, B.T. 2001a, ApJS, 134, 263 (WD01a)
Draine, B.T. & Sutin, B. 1987, ApJ, 320, 803 (DS87)
Tobita, S. et al. 1994, Chem. Phys. 187, 419
Malloci, G. et al. 2007, A&A, 462, 627
"""

import numpy as np
from .dl07_crosssections import C_abs_dl07

# ── Physical constants (CGS) ──────────────────────────────────────────────
k_B  = 1.380649e-16    # Boltzmann constant [erg K⁻¹]
m_e  = 9.1093837e-28   # electron mass [g]
e_sq = 1.44e-7         # e² in CGS → eV·cm  (= 14.4 eV·Å = 1.44e-7 eV·cm)
eV_erg = 1.602176634e-12  # erg eV⁻¹

# ── WD01a graphite work function ──────────────────────────────────────────
W_GRAPHITE = 4.4       # eV (bulk graphite work function, WD01a Table 1)

# ── Yield parameters (WD01a eqs. 7–12, PAH limit) ────────────────────────
Y1_ION  = 0.5   # probability that freed electron escapes (y₁, PAH limit a≪l_e)
W_PDT   = 3.0   # energy width [eV] (WD01a Table 1 "E_low" parameter)
Y1_DET  = 0.5   # detachment probability for anions


# ── Grain radius from Nc ──────────────────────────────────────────────────
def a_from_Nc(Nc):
    """
    Effective grain radius [cm] for a PAH with Nc C atoms.
    From Draine+2021 eq. 8: a[Å] = 0.9 × √Nc.
    """
    return 0.9e-8 * Nc**0.5   # cm


# ── Ionisation potential ──────────────────────────────────────────────────
def ionisation_potential_wd01a(Z, Nc):
    """
    Ionisation potential IP(Z, Nc) [eV] from WD01a eq. 2.

    For Z ≥ 0 (photoionisation to charge Z+1):
        IP(Z) = W + (Z + 0.5) × e²/a

    For Z = −1 (photodetachment threshold):
        returns the electron affinity EA(Nc) of the neutral species
        (the threshold for Z = −1 → Z = 0 is the EA of the neutral).

    Parameters
    ----------
    Z : int
        Current charge state.
    Nc : int or float
        Number of carbon atoms.

    Returns
    -------
    IP : float  [eV]
    """
    if Z == -1:
        return electron_affinity_wd01a(Nc)
    a_cm = a_from_Nc(Nc)
    # WD01a eq. 2: IP(Z) = W + (Z + 0.5) × e²/a  [CGS e² in eV·cm]
    return W_GRAPHITE + (Z + 0.5) * e_sq / a_cm   # eV


def electron_affinity_wd01a(Nc):
    """
    Electron affinity EA(Nc) [eV] for neutral PAHs.

    Empirical power-law fit to experimental/DFT values from
    Tobita et al. (1994) and Malloci et al. (2007):
        EA(C24) ≈ 0.47 eV,  EA(C54) ≈ 0.93 eV,  EA(C96) ≈ 1.35 eV

    fit: EA = 0.20 × Nc^{0.37}  (in eV)

    Parameters
    ----------
    Nc : int or float

    Returns
    -------
    EA : float [eV]  (≥ 0)
    """
    return max(0.0, 0.20 * Nc**0.37)


# ── Photoionization yield ─────────────────────────────────────────────────
def photoion_yield_wd01a(E_eV, Z, Nc):
    """
    Photoionisation (or photodetachment) yield Y(E, Z, Nc) [dimensionless].

    WD01a eqs. 7–12 in the small-grain (PAH) limit where the electron
    escape length l_e ≫ grain radius a:

        Y = y₁ × (E − IP)² / [(E − IP)² + W_PDT²]   for E > IP(Z, Nc)
        Y = 0                                          otherwise

    The squared-energy numerator is used (WD01a eq. 11 with exponent β=2).

    Parameters
    ----------
    E_eV : float
        Photon energy [eV].
    Z : int
        Current charge state.  Allowed: −1, 0, +1, +2.
        Z = −1 → photodetachment; Z ≥ 0 → photoionisation.
    Nc : int or float
        Number of carbon atoms.

    Returns
    -------
    Y : float  ∈ [0, 1]
    """
    if Z > 2:
        return 0.0   # tri-cations not considered

    IP = ionisation_potential_wd01a(Z, Nc)
    if E_eV <= IP:
        return 0.0

    dE = E_eV - IP
    y1 = Y1_DET if Z == -1 else Y1_ION
    return y1 * dE**2 / (dE**2 + W_PDT**2)


# ── Photoionisation / photodetachment rates ───────────────────────────────
def photoion_rate_wd01a(Z, Nc, u_E_fn, G0=1.0,
                         E_min=0.1, E_max=13.6, N_pts=500):
    """
    Photoionisation rate k_pi(Z) [s⁻¹] integrated over the radiation field.

    k_pi = ∫ C_abs(E, Z) × Y(E, Z, Nc) × c × u_E(E) / (E × eV_erg) dE

    Parameters
    ----------
    Z : int
        Current charge state.
    Nc : int or float
        Number of carbon atoms.
    u_E_fn : callable
        u_E_fn(E_eV) → u_E [erg cm⁻³ eV⁻¹] for unit-G0 Habing field.
    G0 : float
        Habing field scaling factor.
    E_min, E_max : float
        Integration limits [eV].
    N_pts : int
        Number of trapezoidal quadrature points.

    Returns
    -------
    k_pi : float  [s⁻¹]
    """
    c_cgs = 2.99792458e10

    # Charge state for the cross-section (ionised vs neutral determines C_abs)
    Z_abs = Z  # use the PAH's current charge state for C_abs

    E_grid = np.linspace(E_min, E_max, N_pts)
    integrand = np.zeros(N_pts)
    for i, E in enumerate(E_grid):
        C = C_abs_dl07(E, Nc, Z=Z_abs)
        Y = photoion_yield_wd01a(E, Z, Nc)
        u = G0 * u_E_fn(E)
        integrand[i] = C * Y * c_cgs * u / (E * eV_erg)
    return np.trapezoid(integrand, E_grid)


# ── Electron recombination rate (Draine & Sutin 1987) ─────────────────────
def recombination_rate_ds87(Z_grain, Nc, T):
    """
    Electron recombination rate coefficient α_rec [cm³ s⁻¹] for a grain
    with charge Z_grain capturing an electron (Z_grain → Z_grain − 1).

    Uses the Draine & Sutin (1987) Coulomb-focused formula:

        α_rec = π a² v̄_e × J̃(ψ)

    where:
        v̄_e = √(8 k_B T / π m_e)    (mean electron speed)
        ψ    = Z_grain × e² / (a k_B T)
        J̃(ψ) = 1 + ψ              for ψ > 0  (attraction: positive grain)
             = exp(−ψ)            for ψ < 0  (repulsion: negative grain → 0)

    Valid for Z_grain ≥ 1 (electron capture by cation).

    Parameters
    ----------
    Z_grain : int
        Charge of the grain BEFORE capturing an electron (must be ≥ 1).
    Nc : int or float
        Number of carbon atoms.
    T : float
        Gas temperature [K].

    Returns
    -------
    alpha : float  [cm³ s⁻¹].  Returns 0 for Z_grain ≤ 0.
    """
    if Z_grain <= 0:
        return 0.0

    a_cm = a_from_Nc(Nc)
    v_bar_e = np.sqrt(8.0 * k_B * T / (np.pi * m_e))  # cm s⁻¹

    # ψ = Z × e² / (a × k_B T)  dimensionless
    kT_erg = k_B * T
    psi = Z_grain * e_sq * eV_erg / (a_cm * kT_erg)   # e² in eV·cm → erg·cm

    # DS87 J̃ function
    J_tilde = 1.0 + psi   # ψ > 0 for Z_grain ≥ 1

    return np.pi * a_cm**2 * v_bar_e * J_tilde   # cm³ s⁻¹


# ── Electron attachment rate ──────────────────────────────────────────────
def attachment_rate_wd01a(Nc, T):
    """
    Electron attachment rate coefficient α_att [cm³ s⁻¹] for a neutral PAH.

    Uses WD01a eq. 14 with the sticking coefficient s_e = 0.5 × (1 − exp(−a/l_0))
    where l_0 = 1 Å (WD01a eq. 15):

        α_att = π a² × v̄_e × s_e

    For PAH-sized grains (a ~ 3–10 Å), s_e ≈ 0.45–0.50.

    Parameters
    ----------
    Nc : int or float
        Number of carbon atoms.
    T : float
        Gas temperature [K].

    Returns
    -------
    alpha_att : float  [cm³ s⁻¹]
    """
    a_cm  = a_from_Nc(Nc)
    a_ang = a_cm * 1e8         # Å
    l0    = 1.0                # Å (WD01a characteristic scale)
    s_e   = 0.5 * (1.0 - np.exp(-a_ang / l0))

    v_bar_e = np.sqrt(8.0 * k_B * T / (np.pi * m_e))
    return np.pi * a_cm**2 * v_bar_e * s_e   # cm³ s⁻¹
