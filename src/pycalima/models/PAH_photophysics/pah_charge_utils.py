"""
pah_charge_utils.py — PAH charge state physics.

Ionisation potentials, electron affinities, photoionisation yields,
recombination rates, electron attachment rates, and sticking coefficients.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Physical constants (CGS)
# ---------------------------------------------------------------------------
ME_CGS               = 9.1093837015e-28
H_CGS                = 6.62607015e-27
C_CGS                = 2.99792458e10
KB_CGS               = 1.380649e-16
EV2ERG               = 1.602176634e-12
E_STATC              = 4.8032047e-10
ELECTRON_ESCAPE_LENGTH_CM = 1e-7
TINY                 = 1e-300

# ---------------------------------------------------------------------------
# Molecular data
# ---------------------------------------------------------------------------
IONISATION_POTENTIAL = {
    'C24H12': {
        '1': 7.20,   # Tobita et al. 1994
        '2': 11.50,  # Tobita et al. 1994
    },
    'C54H18': {
        '1': 6.14,   # Malloci et al. 2007
        '2': 8.91,   # Malloci et al. 2007
        '3': 12.94,  # Malloci et al. 2007
    },
    'C96H24': {
        '1': 5.68,   # Bakes & Tielens 1994
        '2': 8.24,   # Bakes & Tielens 1994
        '3': 10.80,  # Bakes & Tielens 1994
        '4': 13.36,  # Bakes & Tielens 1994
    },
}
ELECTRON_AFFINITY = {
    'C24H12': {
        '1': 0.47,   # Duncan et al. 1999
    },
    'C54H18': {
        '1': 1.44,   # Malloci et al. 2007
    },
    'C96H24': {
        '1': 3.11,   # Bakes & Tielens 1994  (EA(1)=4.4-0.5×25.1/√96)
        '2': 0.56,   # Bakes & Tielens 1994  (EA(2)=4.4-1.5×25.1/√96)
    },
}


# ---------------------------------------------------------------------------
# Molecular geometry
# ---------------------------------------------------------------------------

def afromNc(Nc: int) -> float:
    """PAH effective radius in cm from number of carbon atoms."""
    return 0.9e-8 * np.sqrt(Nc)


# ---------------------------------------------------------------------------
# Ionisation potentials / electron affinities
# ---------------------------------------------------------------------------

def ionisation_potential_energy(IP0: float, Nh0: int, Nh: int) -> float:
    """IP scaled for H-atom count relative to a reference species."""
    return IP0 + 0.1 * (Nh0 - Nh)


def electron_affinity_energy(EA0: float, Nh0: int, Nh: int) -> float:
    """EA scaled for H-atom count relative to a reference species."""
    return EA0 + 0.1 * (Nh0 - Nh)


# ---------------------------------------------------------------------------
# Ionisation yields
# ---------------------------------------------------------------------------

def ionisation_yield_Jochims1996(IP: float, E: float) -> float:
    """Linear ionisation yield ramp (Jochims et al. 1996)."""
    if E >= IP + 9.2:
        return 1.0
    return (E - IP) / 9.2


def ionisation_yield_LePage2001(IP: float, IPcoronene: float, E: float) -> float:
    """Exponential ionisation yield (Le Page et al. 2001)."""
    c = (14.89 - IPcoronene) / (14.89 - IP)
    return 0.8 * np.exp(-0.00128 * (c * (E - 14.89))**4.0)


def photoionisation_rate(sigma_ion: np.ndarray, IP: float,
                         N: np.ndarray, E: np.ndarray) -> float:
    """
    Photoionisation rate [s^-1].

    Parameters
    ----------
    sigma_ion : cross-section array [cm^2]
    IP : ionisation potential [eV]
    N : photon number flux [# cm^-2 s^-1 eV^-1]
    E : photon energy array [eV]
    """
    mask = E >= IP
    return float(np.trapezoid(sigma_ion[mask] * N[mask], E[mask]))


# ---------------------------------------------------------------------------
# Recombination & attachment rates
# ---------------------------------------------------------------------------

def recombination_rate_Spitzer(Nc: int, Z: int, T: float, ne: float) -> float:
    """
    Electron-PAH recombination (Spitzer 2004 / Verstraete 1990 / Berne 2022).

    Returns rate in s^-1.
    """
    phi  = 1.85e5 / T / np.sqrt(Nc)
    k    = 1.28e-10 * Nc * np.sqrt(T) * (1.0 + phi * (1.0 + Z))
    return k * ne


def recombination_rate_Tielens21(Nc: int, T: float, ne: float) -> float:
    """
    Electron-PAH recombination (Tielens 2021, Eq. 8.106).

    Returns rate in s^-1.
    """
    k = 1.3e-6 * np.sqrt(Nc) * np.sqrt(300.0 / T)
    return k * ne


def attachment_rate_Carelli13(T: float, ne: float) -> float:
    """
    Electron attachment to neutral PAH (Carelli et al. 2013, coronene params).

    Returns rate in s^-1.
    """
    a, b, c = 2.74e-9, 0.11, -1.12
    k = a * (T / 300.0)**b * np.exp(-c / T)
    return k * ne


def attachment_rate_Tielens05(Nc: int, T: float, ne: float) -> float:
    """
    Electron attachment to neutral PAH (Tielens 2005).

    Returns rate in s^-1.
    """
    s_e = 1.0
    k   = 1.3e-7 * s_e * np.sqrt(Nc)
    return k * ne


# ---------------------------------------------------------------------------
# Draine & Sutin (1987) J-function and Bakes & Tielens (1994) recombination
# ---------------------------------------------------------------------------

def J_function_DS87(Z: int, q: float, a: float, T: float) -> float:
    """
    J-tilde function from Draine & Sutin (1987), Eqs. 3.3–3.5.

    Parameters
    ----------
    Z : PAH charge number (dimensionless integer, e.g. +1, +2)
    q : projectile charge number (dimensionless, e.g. -1 for electron)
    a : PAH radius [cm]
    T : gas temperature [K]

    Note: q must be the DIMENSIONLESS charge number (not in statcoulombs).
    tau = a*kB*T / (q^2 * e^2), nu = Z / q.
    For electron recombination with a Z=+1 PAH: call J_function_DS87(1, -1, a, T).
    """
    nu        = Z / q if q != 0 else 0.0
    denom_q2  = (q * q) * (E_STATC * E_STATC)
    tau       = (a * KB_CGS * T) / max(denom_q2, TINY)

    if nu == 0:
        return 1.0 + np.sqrt(np.pi / (2.0 * max(tau, TINY)))
    elif nu < 0:
        tn    = max(tau, TINY)
        inner = max(tn - 2.0 * nu, TINY)
        return (1.0 - nu / tn) * (1.0 + np.sqrt(2.0 / inner))
    else:
        nup       = max(nu, TINY)
        tp        = max(tau, TINY)
        theta_nu  = 1.0 / (1.0 + 1.0 / np.sqrt(nup))
        root_term = 1.0 / np.sqrt(4.0 * tp + 3.0 * nup)
        value     = (1.0 + root_term)**2 * np.exp(-theta_nu / tp)
        return value if np.isfinite(value) else 0.0


def recombination_rate_Bakes1994(Nc: int, Z: int, se: float, T: float, ne: float) -> float:
    """
    Electron-PAH recombination (Bakes & Tielens 1994).

    Parameters
    ----------
    Nc : number of carbon atoms
    Z  : PAH charge (e.g. +1, +2)
    se : electron sticking coefficient (typically 0.5)
    T  : gas temperature [K]
    ne : electron number density [cm^-3]

    Returns rate coefficient [cm^3 s^-1] (multiply by ne externally if ne=1 here).
    The 0.82 factor is the Verstraete (1990) empirical correction.
    """
    a    = afromNc(Nc)
    # q=-1: dimensionless electron charge number (Coulomb-focused regime, tau << 1)
    J    = J_function_DS87(Z, -1, a, T)
    krec = ne * se * np.sqrt(8.0 * KB_CGS * T / (np.pi * ME_CGS)) * J * np.pi * a**2
    return krec * 0.82  # Verstraete (1990) correction


def attachment_rate_Bakes1994(Nc: int, se: float, T: float, ne: float) -> float:
    """
    Electron attachment to neutral PAH (Z=0 → Z=-1), Bakes & Tielens (1994).

    Uses the J-function for a neutral grain (nu=0): J = 1 + sqrt(pi/(2*tau)),
    which captures the image-charge (polarization) enhancement.

    Returns rate [s^-1] (already multiplied by ne).
    """
    a   = afromNc(Nc)
    J   = J_function_DS87(0, -1, a, T)   # nu = 0/(-1) = 0
    return ne * se * np.sqrt(8.0 * KB_CGS * T / (np.pi * ME_CGS)) * J * np.pi * a**2


# ---------------------------------------------------------------------------
# Cagliari database polarizabilities and Allamandola sticking coefficients
# ---------------------------------------------------------------------------

# Neutral mean isotropic polarizabilities [Å³] for the two parent molecules
# that are present in the Cagliari Theoretical Spectral Database of PAHs.
_CAGLIARI_ALPHA_NEUTRAL = {24: 47.0, 54: 124.33}

# Power-law fit α = A × Nc^B fitted to all 40 PAHs in the Cagliari database
# (Nc ≤ 54) in log-log space.  Used to extrapolate to Nc = 96 (circumcircum-
# coronene), which is not in the database.
#   α(Nc=54) fit = 136.3 Å³  (Cagliari: 124.3 Å³, 10 % deviation)
#   α(Nc=66) fit = 173.9 Å³  (Cagliari: 162.7 Å³, 7 %  deviation)
#   α(Nc=96) fit = 274.2 Å³  (extrapolated)
_ALPHA_FIT_A = 1.0696
_ALPHA_FIT_B = 1.2152

# Allamandola et al. (1989) se formula: se = SE_C × α^SE_M  (EA < 1 eV).
# Coefficients calibrated so that the charge distributions in Andrews (2016)
# Fig. 8 are reproduced:
#   C24H12 (α = 47.0  Å³, EA = 0.47 eV) → se = 0.075
#   C96H24 (α = 274.2 Å³, EA = 3.11 eV) → se = 1.0   (EA ≥ 1 eV cap)
#   C54H18 (α = 124.3 Å³, EA = 1.44 eV) → se = 1.0   (EA ≥ 1 eV cap)
_SE_C = 1.172e-3
_SE_M = 1.0802


def alpha_neutral_Cagliari(Nc: int) -> float:
    """
    Neutral mean isotropic polarizability [Å³] for a PAH with Nc carbon atoms.

    Returns the Cagliari database value for C24H12 and C54H18 directly, and
    uses the power-law extrapolation for all other sizes (including Nc = 96).
    Andrews et al. (2016) prescribe using the parent-molecule polarizability
    for all derivatives at the same Nc.
    """
    if Nc in _CAGLIARI_ALPHA_NEUTRAL:
        return _CAGLIARI_ALPHA_NEUTRAL[Nc]
    return _ALPHA_FIT_A * Nc ** _ALPHA_FIT_B


def se_neutral_Allamandola1989_full(Nc: int, Nh: int, EA_eV: float,
                                    alpha_Ang3: float, T_K: float = 500.0) -> float:
    """
    Full Allamandola et al. (1989) sticking coefficient via detailed balance.

    S(e) = kr / (kr + kb)

    kf = 2π√(α e²/me)             — Langevin capture rate [s⁻¹]
    ρ_e = me²v/(π²ℏ³)              — free electron DOS at ε = kBT  [erg⁻¹ cm⁻³]
    ρ⁻ = C(n+s-1, n) / hν₀        — quantum harmonic osc DOS at E=EA [erg⁻¹]
      where s = 3(Nc+Nh)-6 modes, n = round(EA/hν₀), ν₀ = 1000 cm⁻¹
    kb = kf × ρ_e / ρ⁻             — autoionization rate (Klots 1967) [s⁻¹]
    kr = ∫(8πν²/c²) C_abs n̄(T_vib) dν — IR radiative stabilization [s⁻¹]
      where T_vib is the vibrational temperature at internal energy U = EA

    NOTE: At T=500 K (PDR conditions) this formula gives S ≈ 0 for small PAHs
    (C24, C96) because the anion DOS is small (few quanta in many modes) so
    kb >> kr.  The result is strongly T-dependent; at T ~ 10 K (cold ISM)
    the formula gives much higher S.  Andrews+16 / Allamandola+89 use an
    empirical Cagliari-α calibration (se_neutral_Andrews2016) that matches
    their Fig. 8 at PDR conditions rather than the literal formula evaluated
    at gas temperature.
    """
    from math import lgamma
    from scipy.optimize import brentq
    # lazy import to avoid circular dependency
    from models.PAH_photophysics.pah_temperature import get_absorption_cross_section

    HBAR_CGS = H_CGS / (2.0 * np.pi)

    # Langevin capture rate kf [s⁻¹]
    alpha_cm3 = alpha_Ang3 * 1e-24
    kf = 2.0 * np.pi * np.sqrt(alpha_cm3 * E_STATC**2 / ME_CGS)

    # Free electron DOS at ε = kBT
    eps_erg = KB_CGS * T_K
    v_e = np.sqrt(2.0 * eps_erg / ME_CGS)
    rho_e = ME_CGS**2 * v_e / (np.pi**2 * HBAR_CGS**3)

    # Quantum harmonic oscillator DOS at E = EA, ν₀ = 1000 cm⁻¹
    s = 3 * (Nc + Nh) - 6
    hnu0_erg = H_CGS * C_CGS * 1000.0
    n = max(1, round(EA_eV * EV2ERG / hnu0_erg))
    # ln C(n+s-1, n) = lgamma(n+s) - lgamma(n+1) - lgamma(s)
    log_rho_m = lgamma(n + s) - lgamma(n + 1) - lgamma(s) - np.log(hnu0_erg)
    rho_m = np.exp(log_rho_m)

    # Autoionization rate
    kb = kf * rho_e / rho_m

    # Radiative stabilization rate kr from Li & Draine IR cross-section
    a0 = afromNc(Nc)
    w_cm, C_abs = get_absorption_cross_section(0, a0)
    nu = C_CGS / w_cm
    hnu_eV = H_CGS * nu / EV2ERG
    ir = (w_cm > 1e-4) & (C_abs > 0)
    nu_ir, C_ir, hnu_ir = nu[ir], C_abs[ir], hnu_eV[ir]

    # T_vib: temperature where the IR-weighted mean photon occupation × s/2 = EA
    def U_of_T(T):
        x = H_CGS * nu_ir / (KB_CGS * T)
        occ = np.where(x > 50, np.exp(-x),
                       1.0 / (np.exp(np.clip(x, 0, 50)) - 1 + TINY))
        return float(np.trapezoid(C_ir * hnu_ir * occ, nu_ir)
                     / np.trapezoid(C_ir, nu_ir)) * s / 2.0

    try:
        T_vib = brentq(lambda T: U_of_T(T) - EA_eV, 10.0, 10000.0, xtol=1.0)
    except ValueError:
        T_vib = 500.0

    x = H_CGS * nu_ir / (KB_CGS * T_vib)
    nb = np.where(x > 50, np.exp(-x),
                  1.0 / (np.exp(np.clip(x, 0, 50)) - 1 + TINY))
    kr = float(np.trapezoid(
        (8.0 * np.pi * nu_ir**2 / C_CGS**2) * C_ir * nb, nu_ir))

    return kr / (kr + kb)


def se_neutral_WR(Nc: int, EA_eV: float, T_K: float = 500.0,
                  kr: float = 10.0) -> float:
    """
    Electron sticking coefficient via Whitten-Rabinovitch (1963) density of states.

    S = kr / (kr + kb),  kb = kf × ρ_e / ρ_WR

    where
      kf   = 2π√(α e²/me),  α = 1.5×10⁻²⁴ Nc cm³  (Allamandola+89)
      ρ_e  = me² v / (π² ℏ³) at ε = kBT             (free electron DOS)
      ρ_WR = (EA + a_WR × E_zpe)^(s-1) / [(s-1)! (hν₀)^s]
           with s = 3(Nc+Nh)-6, Nh = round(√(6·Nc)), hν₀ = 1000 cm⁻¹
      a_WR = −1.3713 ρ² + 0.3802 ρ + 0.7481,  ρ = EA / E_zpe
           (quadratic fit to WR Table I, calibrated to Allamandola+89 Fig. 25
            at EA=0.7 eV, T=10 K, kr=10 s⁻¹)
      kr   = IR radiative stabilisation rate [s⁻¹]; Allamandola+89 use 10 s⁻¹
    """
    from math import lgamma, log, sqrt, exp

    HBAR = H_CGS / (2.0 * np.pi)
    HNU0 = H_CGS * C_CGS * 1000.0          # erg  (ν₀ = 1000 cm⁻¹)

    Nh  = int(round(sqrt(6.0 * Nc)))
    s   = 3 * (Nc + Nh) - 6
    E   = EA_eV * EV2ERG
    E_zpe = s * HNU0 / 2.0
    rho_dim = E / E_zpe

    a_WR  = -1.3713 * rho_dim**2 + 0.3802 * rho_dim + 0.7481
    E_eff = E + a_WR * E_zpe
    if E_eff <= 0.0:
        return 0.0

    log_rho_m = (s - 1) * log(E_eff / HNU0) - lgamma(s) - log(HNU0)

    v        = sqrt(2.0 * KB_CGS * T_K / ME_CGS)
    log_rho_e = log(ME_CGS**2 * v / (np.pi**2 * HBAR**3))

    alpha_cm3 = 1.5e-24 * Nc
    log_kf    = log(2.0 * np.pi * sqrt(alpha_cm3 * E_STATC**2 / ME_CGS))

    log_kb_kr = (log_kf + log_rho_e - log_rho_m) - log(kr)
    if log_kb_kr >  500: return 0.0
    if log_kb_kr < -500: return 1.0
    return 1.0 / (1.0 + exp(log_kb_kr))


def se_neutral_WR_full(Nc: int, EA_eV: float, T_K: float = 500.0) -> float:
    """
    Electron sticking coefficient via Whitten-Rabinovitch DOS with real molecular inputs.

    Same formalism as se_neutral_WR but replaces Allamandola+89 fixed parameters with:
      kf  — Langevin rate using Cagliari polarizability (α_neutral_Cagliari)
      kr  — spontaneous IR emission rate computed from Li & Draine C_abs at the
             vibrational temperature T_vib where <E_vib> = EA (same as
             se_neutral_Allamandola1989_full, but with WR DOS instead of C(n+s-1,n))

    The WR correction factor a_WR is unchanged — it depends only on ρ = EA/E_zpe,
    which is a property of the molecular DOS, not of α or kr.
    """
    from math import lgamma, log, sqrt, exp
    from scipy.optimize import brentq
    from models.PAH_photophysics.pah_temperature import get_absorption_cross_section

    HBAR = H_CGS / (2.0 * np.pi)
    HNU0 = H_CGS * C_CGS * 1000.0

    Nh  = int(round(sqrt(6.0 * Nc)))
    s   = 3 * (Nc + Nh) - 6
    E   = EA_eV * EV2ERG
    E_zpe = s * HNU0 / 2.0
    rho_dim = E / E_zpe

    # WR density of states
    a_WR  = -1.3713 * rho_dim**2 + 0.3802 * rho_dim + 0.7481
    E_eff = E + a_WR * E_zpe
    if E_eff <= 0.0:
        return 0.0
    log_rho_m = (s - 1) * log(E_eff / HNU0) - lgamma(s) - log(HNU0)

    # Free electron DOS at ε = kBT
    v         = sqrt(2.0 * KB_CGS * T_K / ME_CGS)
    log_rho_e = log(ME_CGS**2 * v / (np.pi**2 * HBAR**3))

    # kf from Cagliari polarizability
    alpha_cm3 = alpha_neutral_Cagliari(Nc) * 1e-24
    log_kf    = log(2.0 * np.pi * sqrt(alpha_cm3 * E_STATC**2 / ME_CGS))

    # kr from Li & Draine C_abs: spontaneous IR emission at T_vib where <E_vib>=EA
    a0 = afromNc(Nc)
    w_cm, C_abs = get_absorption_cross_section(0, a0)
    nu   = C_CGS / w_cm
    hnu_eV = H_CGS * nu / EV2ERG
    ir   = (w_cm > 1e-4) & (C_abs > 0)
    nu_ir, C_ir, hnu_ir = nu[ir], C_abs[ir], hnu_eV[ir]

    def U_of_T(T):
        x   = H_CGS * nu_ir / (KB_CGS * T)
        occ = np.where(x > 50, np.exp(-x),
                       1.0 / (np.exp(np.clip(x, 0, 50)) - 1.0 + TINY))
        return float(np.trapezoid(C_ir * hnu_ir * occ, nu_ir)
                     / np.trapezoid(C_ir, nu_ir)) * s / 2.0

    try:
        T_vib = brentq(lambda T: U_of_T(T) - EA_eV, 10.0, 10000.0, xtol=1.0)
    except ValueError:
        T_vib = 500.0

    x_vib = H_CGS * nu_ir / (KB_CGS * T_vib)
    nb    = np.where(x_vib > 50, np.exp(-x_vib),
                     1.0 / (np.exp(np.clip(x_vib, 0, 50)) - 1.0 + TINY))
    kr = float(np.trapezoid(
        (8.0 * np.pi * nu_ir**2 / C_CGS**2) * C_ir * nb, nu_ir))

    log_kb_kr = (log_kf + log_rho_e - log_rho_m) - log(max(kr, 1e-300))
    if log_kb_kr >  500: return 0.0
    if log_kb_kr < -500: return 1.0
    return 1.0 / (1.0 + exp(log_kb_kr))


def se_neutral_Andrews2016(Nc: int, EA_eV: float = 0.47) -> float:
    """
    Electron sticking coefficient for neutral PAH attachment (Z=0 → Z=-1),
    following Andrews et al. (2016) Appendix A / Allamandola et al. (1989).

    Polarizabilities for C24H12 and C54H18 are taken directly from the
    Cagliari PAH database; C96H24 is obtained by extrapolating a power-law
    fit (α = A × Nc^B) to all 40 Cagliari PAHs with Nc ≤ 54.  The parent-
    molecule polarizability is used for all derivatives at the same Nc.

    For EA ≥ 1 eV the anion state is sufficiently stable that se = 1.0.
    Otherwise se = SE_C × α^SE_M, calibrated to match Andrews (2016) Fig. 8.

    Returns se ∈ (0, 1].
    """
    if EA_eV >= 1.0:
        return 1.0
    alpha = alpha_neutral_Cagliari(Nc)
    return min(1.0, _SE_C * alpha ** _SE_M)


# ---------------------------------------------------------------------------
# Electron sticking coefficients (Weingartner & Draine 2001)
# ---------------------------------------------------------------------------

def se_neutral_Weingartner2001(a: float, Nc: float) -> float:
    """Sticking coefficient for neutral PAH."""
    return 0.5 * (1.0 - np.exp(-a / ELECTRON_ESCAPE_LENGTH_CM)) / (1.0 + np.exp(20.0 - Nc))


def se_anion_Weingartner2001(a: float) -> float:
    """Sticking coefficient for anion PAH."""
    return 0.5 * (1.0 - np.exp(-a / ELECTRON_ESCAPE_LENGTH_CM))
