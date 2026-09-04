"""Rate kernels for each dust and PAH chemistry process.

Each function has the signature::

    rate_fn(state, y_gas, y_dust, dydt_gas, dydt_dust) -> kmax

where

* ``state``       — :class:`~pycalima.solvers.chemistry_state.DustChemistryState`
* ``y_gas``       — current element mass densities [g cm⁻³], shape (n_el,)
* ``y_dust``      — current dust/PAH mass densities [g cm⁻³], shape (npah+ndust,)
* ``dydt_gas``    — gas-phase derivative array, modified **in-place**
* ``dydt_dust``   — dust/PAH derivative array, modified **in-place**
* **return**      — ``kmax``: maximum characteristic rate [s⁻¹] for step-size control

This mirrors the Fortran subroutines in ``dust_rates.f90``.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

from .chemistry_state import DustBinParams, DustChemistryState, PAHBinParams
from .grain_dynamics import grain_relative_velocity, sticking_probability_from_velocity

# ---------------------------------------------------------------------------
# Physical constants (CGS)
# ---------------------------------------------------------------------------
KB_CGS: float = 1.3806488e-16   # Boltzmann constant [erg K⁻¹]
YR2SEC: float = 3.1536e7         # s yr⁻¹ (same as RAMSES yr2sec)
MYR2SEC: float = 3.1536e13       # s Myr⁻¹
_E_ESU: float = 4.8032047e-10   # elementary charge [statC = esu]

# ---------------------------------------------------------------------------
# Nozawa+2006 / Hu+2019 polynomial thermal sputtering yield
# (Dubois et al. 2024 RAMSES-CALIMA implementation)
# ---------------------------------------------------------------------------
# 5th-degree polynomial coefficients for log10(Y_th) as a function of
# lT = log10(T * 0.60), where 0.60 is the mean molecular weight of a fully
# ionized solar-composition gas.  Coefficient order: c0 + c1*lT + ... + c5*lT^5.
# Source: Hu et al. (2019) fits to Nozawa et al. (2006) yields.
_NOZAWA_C_COEFFS: tuple = (
    -2.34333937e2, 1.38485732e2, -3.39021615e1,
     4.17705353e0, -2.58281473e-1, 6.38827523e-3,
)
_NOZAWA_SIL_COEFFS: tuple = (
    -2.34790500e2, 1.33208637e2, -3.13027448e1,
     3.71345730e0, -2.21823668e-1, 5.31746427e-3,
)
_NOZAWA_MU_ION: float = 0.60  # mean molecular weight for fully ionized gas


def _nozawa_yield(Tk: float, is_carbonaceous: bool) -> float:
    """Hu+2019 polynomial fit to Nozawa+2006 thermal sputtering yield.

    Parameters
    ----------
    Tk : float
        Gas temperature [K].
    is_carbonaceous : bool
        True for graphite/carbonaceous grains; False for silicate.

    Returns
    -------
    Y_th : float
        Sputtering rate coefficient [µm yr⁻¹ cm³].  The destruction
        timescale (for a grain of radius ``a`` [µm] in gas of density
        ``nH`` [cm⁻³]) is

            t_spu [yr] = a / (3 · nH · Y_th)

        matching the RAMSES-CALIMA Fortran convention where ``asize``
        is kept in µm throughout.
    """
    lT = math.log10(max(Tk * _NOZAWA_MU_ION, 1e-30))
    c = _NOZAWA_C_COEFFS if is_carbonaceous else _NOZAWA_SIL_COEFFS
    log_Y = c[0] + c[1]*lT + c[2]*lT**2 + c[3]*lT**3 + c[4]*lT**4 + c[5]*lT**5
    return 10.0**log_Y   # [µm yr⁻¹ cm³]


# ---------------------------------------------------------------------------
# Coulomb enhancement factor (Weingartner & Draine 1999, RAMSES convention)
# ---------------------------------------------------------------------------

def _coulomb_factor_WD99(
    Z_grain: float,
    Zi: float,
    T: float,
    a_cm: float,
) -> float:
    """Coulomb enhancement (or suppression) factor D for an ion–grain collision.

    Uses the point-charge (mean-Z) approximation:

    * Zi × Z_grain < 0  (attractive): D = 1 - Zi × Z_grain × e²/(a kT)
    * Zi × Z_grain > 0  (repulsive):  D = exp(-Zi × Z_grain × e²/(a kT))
    * Z_grain = 0, Zi ≠ 0 (neutral grain):
                          D = 1 + √(π Zi² e²/(2 kT a))

    Parameters
    ----------
    Z_grain : float
        Mean grain charge in units of electron charges (e).
    Zi : float
        Ion charge (1 for singly ionized, 0 for neutral atoms).
    T : float
        Gas temperature [K].
    a_cm : float
        Grain radius [cm].

    Returns
    -------
    D : float ≥ 1e-10
    """
    if Zi == 0.0:
        return 1.0
    alpha = (Zi * _E_ESU ** 2) / (a_cm * KB_CGS * T)  # dimensionless
    product = Z_grain * Zi
    if product < 0.0:      # attractive
        D = 1.0 - Z_grain * alpha
    elif product > 0.0:    # repulsive
        D = math.exp(-Z_grain * alpha)
    else:                  # neutral grain, charged ion
        D = 1.0 + math.sqrt(math.pi * Zi ** 2 * _E_ESU ** 2 / (2.0 * KB_CGS * T * a_cm))
    return max(D, 1.0e-10)


def _get_coulomb_D(
    db: "DustBinParams",
    Tk: float,
    G0: float,
    ne: float,
    Zi: float = 1.0,
) -> float:
    """Return the Coulomb enhancement factor D for a grain bin given the environment.

    Uses the pre-computed (T, gamma) charge tables if available; otherwise D = 1.

    Parameters
    ----------
    db : DustBinParams
    Tk : float
        Gas temperature [K].
    G0 : float
        FUV field strength [Habing units].
    ne : float
        Electron number density [cm⁻³].
    Zi : float
        Charge of the accreting/sputtering ion (default +1).
    """
    if db.charge_Z_interp is None or ne <= 0.0:
        return 1.0
    # Charging parameter: gamma = G0 × √T / ne  (WD01 convention)
    gamma = G0 * math.sqrt(max(Tk, 1.0)) / ne
    Z_avg = db.charge_Z_interp(Tk, gamma)
    return _coulomb_factor_WD99(Z_avg, Zi, Tk, db.asize_cm)


# ---------------------------------------------------------------------------
# 1.  Grain growth by accretion  (LeBourlot et al. 2012)
# ---------------------------------------------------------------------------

def accretion_rate(
    state: DustChemistryState,
    y_gas: np.ndarray,
    y_dust: np.ndarray,
    dydt_gas: np.ndarray,
    dydt_dust: np.ndarray,
) -> float:
    """Grain growth by accretion of gas-phase metals (Dubois et al. 2024).

    Two regimes follow the RAMSES-CALIMA implementation:

    **Sub-grid cells** (nH > 0.1 cm⁻³ AND T < 10⁴ K, proxy for Jeans unresolved):
      - Constant sticking α_eff = 1/3.
      - Thermal velocity evaluated at fixed T_eff = 100 K.
      - Turbulence clumping boost C_turb = exp(σ_s²), with Mach number
        computed at T_eff = 100 K: σ_s² = ln(1 + b²M²), b = 0.4.
      - C is fully molecular (CO) for nH > 10³ cm⁻³ → no carbonaceous accretion.

    **Resolved cells** (all others):
      - LeBourlot et al. (2012) sticking: S(T) = 1/(1 + 10⁻⁴ T^{1.5}).
      - No clumping boost (C_turb = 1).

    **Coulomb enhancement** E (Dubois et al. 2024):
      - E = 1 everywhere (default).
      - CNM defined as T < 2×10⁴ K AND nH > 10 cm⁻³:
          large carbonaceous → E = 0 (suppressed, grains negatively charged)
          small silicate     → E = 10 (enhanced, positive-ion attraction)

    Modified variables
    ------------------
    dydt_dust[idx]   += rate × y_dust[idx]
    dydt_gas[el_idx] -= rate × y_dust[idx] × el_mfrac[e]
    """
    Tk = state.local_Tk
    nH = state.local_nH
    _MH_CGS = 1.6726219e-24    # proton mass [g]
    _G_CGS  = 6.674e-8         # gravitational constant [cm³ g⁻¹ s⁻²]

    # Accretion requires neutral gas.  Above ~1.8×10⁵ K the gas is fully
    # ionised and there are no neutral species to condense onto grains.
    # This temperature corresponds to the collisional ionisation threshold
    # of hydrogen (kT ~ 15 eV).  Disable accretion above this limit.
    T_ACC_MAX = 1.8e5   # K
    if Tk > T_ACC_MAX:
        return 0.0

    # ── Regime selection (Dubois et al. 2024 Fortran) ─────────────────────────
    # Use LeBourlot2012 (resolved) if ANY of:
    #   1. T > 10^4 K
    #   2. nH < 0.1 cm^{-3}
    #   3. Jeans length > 4 × cell size  (λ_J = sqrt(π kB T / G mH² nH))
    # Otherwise use the subgrid model with turbulent clumping boost.
    _dx = state.local_dx    # cell size [cm]; 0 if unknown
    if _dx > 0.0:
        _lambda_jeans = math.sqrt(math.pi * KB_CGS * Tk / (_G_CGS * _MH_CGS**2 * nH))
        _jeans_resolved = (_lambda_jeans > 4.0 * _dx)
    else:
        _jeans_resolved = False   # cell size unknown: conservative (use subgrid if other criteria met)

    subgrid = (nH >= 0.1) and (Tk <= 1.0e4) and (not _jeans_resolved)

    if subgrid:
        # Fixed T_acc = 100 K for the thermal velocity (accretion sticking prefactor).
        # The Mach number for the lognormal boost uses the LOCAL T (Dubois Fortran T2(i)),
        # not T_acc — only the prefactor sqrt(T_acc) is fixed.
        T_acc = 100.0
        alpha_eff = 1.0 / 3.0
        prefactor = alpha_eff * math.sqrt(T_acc)   # [K^{1/2}]
        # Sound speed at local T with γ=5/3, no µ (Dubois Fortran: sqrt(5/3 kB T / mH))
        c_s_boost = math.sqrt(max((5.0 / 3.0) * KB_CGS * Tk / _MH_CGS, 1.0))
        sigma1d = state.local_sigma
        # Lognormal PDF variance: σ_s² = ln(1 + b²M²),  b = 0.4
        if sigma1d > 0.0:
            Mach2  = (sigma1d / c_s_boost) ** 2
            _sigs2 = math.log1p(0.16 * Mach2)   # σ_s²
            _sigs  = math.sqrt(_sigs2)            # σ_s
        else:
            _sigs2 = 0.0
            _sigs  = 0.0
        # boost is evaluated per-bin because nhmax_acc differs between bins
        _subgrid_params = (_sigs2, _sigs)
    else:
        # Resolved: LeBourlot et al. (2012) sticking, no clumping boost
        prefactor = math.sqrt(Tk) / (1.0 + 1.0e-4 * Tk ** 1.5)
        _subgrid_params = None

    # Coulomb enhancement criterion (Dubois+2024 Fortran):
    # apply when T < 2e4 K  OR  nH > 10 cm^{-3}
    # (Fortran: if(tau.lt.2d4 .or. nh.gt.10.0d0))
    apply_coulomb = (Tk < 2.0e4) or (nH > 10.0)

    kmax = 0.0

    for db in state.dust_bins:
        idx = db.bin_index + state.npah

        if not db.el_indices:
            continue

        # Global density ceiling: accretion suppressed above nhmax_acc
        # (per-bin: nhmax_acc = 1e3 for C grains, 1e4 for Si grains in Dubois+2024)
        if nH > db.nhmax_acc:
            continue

        # ── Turbulence clumping boost (Dubois+2024 Eq. per bin) ───────────────
        # Integral of the lognormal density PDF up to nhmax_acc (density ceiling).
        # boost = 0.5 × exp(σ_s²) × erfc[(1.5 σ_s² − smaxacc) / (√2 σ_s)]
        # Reference: Dubois+2024 Fortran boost_acc with smaxacc = ln(nhmax/nH).
        if _subgrid_params is not None:
            _sigs2, _sigs = _subgrid_params
            if _sigs > 0.0:
                smaxacc = math.log(db.nhmax_acc / max(nH, 1.0e-30))
                _arg = (1.5 * _sigs2 - smaxacc) / (math.sqrt(2.0) * _sigs)
                C_turb = 0.5 * math.exp(_sigs2) * math.erfc(_arg)
                C_turb = max(C_turb, 0.0)
            else:
                C_turb = 1.0
        else:
            C_turb = 1.0

        # Find the limiting element (smallest pseudo-rate)
        limit_rate = math.inf
        for loc, el_idx in enumerate(db.el_indices):
            m_e = state.el_atomic_mass_g[el_idx]
            f_e = db.el_mfractions[loc]
            rho_e = y_gas[el_idx]
            if rho_e <= 0.0 or f_e <= 0.0:
                limit_rate = 0.0
                break
            pseudo = rho_e / (f_e * math.sqrt(m_e))
            if pseudo < limit_rate:
                limit_rate = pseudo

        if limit_rate <= 0.0 or not math.isfinite(limit_rate):
            continue

        # Coulomb enhancement factor (Dubois+2024 Fortran Coulomb_enhance per bin):
        #   SmC = 1.0, LgC = 1e-5 (strongly suppressed), SmSi = 10.0, LgSi = 1.0
        if apply_coulomb:
            is_large_carb = (db.composition == 'graphite') and (db.asize_micron > 0.05)
            is_small_sil  = (db.composition == 'silicate') and (db.asize_micron < 0.05)
            if is_large_carb:
                coulomb_E = 1.0e-5   # strongly suppressed (Fortran Coulomb_enhance(LgC)=1e-5)
            elif is_small_sil:
                coulomb_E = 10.0     # enhanced (Fortran Coulomb_enhance(SmSi)=10)
            else:
                coulomb_E = 1.0
        else:
            coulomb_E = 1.0

        rate = db.k0_acc * prefactor * limit_rate * coulomb_E * C_turb  # [s⁻¹]

        if rate <= 0.0:
            continue

        kmax = max(kmax, rate)
        rate_rho = rate * y_dust[idx]   # [g cm⁻³ s⁻¹]

        dydt_dust[idx] += rate_rho
        for loc, el_idx in enumerate(db.el_indices):
            dydt_gas[el_idx] -= rate_rho * db.el_mfractions[loc]

    return kmax


# ---------------------------------------------------------------------------
# 2.  Thermal sputtering  (table interpolation)
# ---------------------------------------------------------------------------

def thermal_sputtering_rate(
    state: DustChemistryState,
    y_gas: np.ndarray,
    y_dust: np.ndarray,
    dydt_gas: np.ndarray,
    dydt_dust: np.ndarray,
) -> float:
    """Thermal sputtering of dust grains by gas-phase ions.

    Interpolates the pre-computed 2-D T–φ CALIMA tables at ``φ = 0``
    (uncharged grain approximation).  The RAMSES formula is

    .. math::

        \\text{rate} \\; [\\text{s}^{-1}] =
            \\frac{3}{a [\\mu\\text{m}] \\, t_\\text{yr}}
            \\sum_\\text{ions}
            \\left(\\frac{1}{n_\\text{ion}} \\frac{\\mathrm{d}a}{\\mathrm{d}t}\\right)
            \\frac{y_\\text{gas}[e]}{m_e}

    where ``(1/n_ion) da/dt`` is read from the table in [µm yr⁻¹ cm³].

    Modified variables
    ------------------
    dydt_dust[idx]   -= rate × y_dust[idx]
    dydt_gas[el_idx] += rate × y_dust[idx] × el_mfrac[e]   (return to gas)
    """
    Tk = state.local_Tk
    kmax = 0.0

    for db in state.dust_bins:
        idx = db.bin_index + state.npah

        if not db.sputtering_interps:
            continue

        rate_total = 0.0  # [µm yr⁻¹]

        for el_name, interp_fn in db.sputtering_interps.items():
            if el_name not in state.el_names:
                continue
            el_idx = state.el_names.index(el_name)

            rho_ion = y_gas[el_idx]
            if rho_ion < 1.0e-40:
                continue

            m_e_g = state.el_atomic_mass_g[el_idx]
            n_ion = rho_ion / m_e_g  # number density [cm⁻³]

            # Interpolate at (T, φ=0) – uncharged grain approximation
            table_rate = interp_fn(Tk, phi_query=0.0)  # [µm yr⁻¹ cm³]
            rate_total += table_rate * n_ion             # [µm yr⁻¹]

        if rate_total <= 0.0:
            continue

        # Convert to a fractional mass-loss rate [s⁻¹]
        # (1/ρ) dρ/dt = (3/a) |da/dt|,  a in µm,  da/dt in µm/yr
        rate1 = 3.0 * rate_total / (db.asize_micron * YR2SEC)  # [s⁻¹]
        kmax = max(kmax, rate1)

        rate_rho = rate1 * y_dust[idx]  # [g cm⁻³ s⁻¹]

        dydt_dust[idx] -= rate_rho
        for loc, el_idx in enumerate(db.el_indices):
            dydt_gas[el_idx] += rate_rho * db.el_mfractions[loc]

    return kmax


def thermal_sputtering_rate_nozawa(
    state: DustChemistryState,
    y_gas: np.ndarray,
    y_dust: np.ndarray,
    dydt_gas: np.ndarray,
    dydt_dust: np.ndarray,
) -> float:
    """Thermal sputtering using the Nozawa+2006/Hu+2019 polynomial yield.

    Matches the ``nozawa2006`` case of Dubois et al. (2024) RAMSES-CALIMA.
    No ion-by-ion table lookup is performed; instead the combined yield for
    all gas species is given by a 5th-degree polynomial in log10(T × 0.60).

    Timescale formula (identical to RAMSES Fortran):

    .. math::

        t_{\\rm spu} = \\frac{a}{3 \\, n_{\\rm H} \\, Y_{\\rm th}(T)}

    where :math:`Y_{\\rm th}` [cm⁴ yr⁻¹] is the Hu+2019 polynomial yield
    and the fractional mass-loss rate is
    :math:`(1/\\rho) \\, d\\rho/dt = 1/t_{\\rm spu}`.

    Composition is determined from ``db.el_mfractions``: a grain whose
    dominant element (by mass fraction) is C is treated as carbonaceous;
    all others use the silicate coefficients.

    Modified variables
    ------------------
    dydt_dust[idx]   -= rate × y_dust[idx]
    dydt_gas[el_idx] += rate × y_dust[idx] × el_mfrac[e]   (return to gas)
    """
    Tk = state.local_Tk
    nH = state.local_nH
    kmax = 0.0

    for db in state.dust_bins:
        idx = db.bin_index + state.npah

        # Identify carbonaceous vs silicate by the grain's lead element
        is_carb = False
        if db.el_indices:
            # el_mfractions[0] corresponds to el_indices[0]; check whether
            # that element is carbon.  We look for the element with the
            # highest mass fraction and check if it is 'C'.
            lead_loc = int(np.argmax(db.el_mfractions))
            lead_el = state.el_names[db.el_indices[lead_loc]]
            is_carb = (lead_el == 'C')

        Y_th = _nozawa_yield(Tk, is_carb)   # [µm yr⁻¹ cm³]

        # rate [s⁻¹] = 3 nH [cm⁻³] × Y_th [µm yr⁻¹ cm³] / (a [µm] × YR2SEC [s yr⁻¹])
        # Identical formula to the table-based kernel; the polynomial replaces
        # the per-ion table lookup.  asize_micron is already in µm.
        denom = 3.0 * nH * Y_th
        if denom <= 0.0:
            continue
        rate1 = denom / (db.asize_micron * YR2SEC)   # [s⁻¹]
        kmax = max(kmax, rate1)

        rate_rho = rate1 * y_dust[idx]   # [g cm⁻³ s⁻¹]

        dydt_dust[idx] -= rate_rho
        for loc, el_idx in enumerate(db.el_indices):
            dydt_gas[el_idx] += rate_rho * db.el_mfractions[loc]

    return kmax


def thermal_sputtering_rate_nozawa_ramses(
    state: DustChemistryState,
    y_gas: np.ndarray,
    y_dust: np.ndarray,
    dydt_gas: np.ndarray,
    dydt_dust: np.ndarray,
) -> float:
    """Thermal sputtering with the RAMSES-CALIMA Fortran yield assignment.

    Identical physics to :func:`thermal_sputtering_rate_nozawa` but reproduces
    the bin-index-based yield dispatch in ``cooling_module.f90`` (case ``'hu'``,
    ndust=4):

    .. code-block:: fortran

        t_des(1:2) = asize(1:2)/nH / ySi / 3 * year   ! bins 1,2: silicate yield
        t_des(3:4) = asize(3:4)/nH / yC  / 3 * year   ! bins 3,4: carbon yield

    The Fortran bin ordering is SmSi(1), LgSi(2), SmC(3), LgC(4), so bins 1-2
    (silicate) correctly receive ySi and bins 3-4 (carbon) correctly receive yC.
    pyCALIMA uses the reverse ordering SmC(1), LgC(2), SmSi(3), LgSi(4), so to
    match the Fortran the yield must be **swapped relative to composition**:
    carbon grains receive ySi and silicate grains receive yC.

    Use this model (``"nozawa2006_ramses"``) when comparing against Dubois+2024
    RAMSES-CALIMA simulations.  Use ``"nozawa2006"`` for physically correct
    (composition-matched) yields.

    At T > ~7×10⁵ K the silicate yield exceeds the carbon yield (Ys > Yc), so
    this swap accelerates carbon-grain sputtering and reduces f_carb in hot gas,
    matching the simulation.
    """
    Tk = state.local_Tk
    nH = state.local_nH
    kmax = 0.0

    for db in state.dust_bins:
        idx = db.bin_index + state.npah

        # Identify carbonaceous vs silicate by the grain's lead element
        is_carb = False
        if db.el_indices:
            lead_loc = int(np.argmax(db.el_mfractions))
            lead_el = state.el_names[db.el_indices[lead_loc]]
            is_carb = (lead_el == 'C')

        # Swap: carbonaceous grains use silicate yield, silicate grains use
        # carbon yield — reproducing the RAMSES Fortran bin-index assignment.
        Y_th = _nozawa_yield(Tk, not is_carb)   # [µm yr⁻¹ cm³]

        denom = 3.0 * nH * Y_th
        if denom <= 0.0:
            continue
        rate1 = denom / (db.asize_micron * YR2SEC)   # [s⁻¹]
        kmax = max(kmax, rate1)

        rate_rho = rate1 * y_dust[idx]   # [g cm⁻³ s⁻¹]

        dydt_dust[idx] -= rate_rho
        for loc, el_idx in enumerate(db.el_indices):
            dydt_gas[el_idx] += rate_rho * db.el_mfractions[loc]

    return kmax


# ---------------------------------------------------------------------------
# 3.  Thermal sublimation  (GD89 pre-computed erosion tables)
# ---------------------------------------------------------------------------

def sublimation_rate(
    state: DustChemistryState,
    y_gas: np.ndarray,
    y_dust: np.ndarray,
    dydt_gas: np.ndarray,
    dydt_dust: np.ndarray,
) -> float:
    """Thermal sublimation of dust grains (Guhathakurta & Draine 1989).

    The fractional rate  ε = ``|da/dt|`` / a  [s⁻¹] is read from the
    pre-computed table ``sublimation_rate_{bin_id}.dat`` (generated by
    ``models.dust_radiation.dust_sublimation.write_sublimation_rate_tables``).
    The mass loss rate follows from  (1/ρ) dρ/dt = 3 ε.

    The liberated mass is returned to the gas phase distributed over the
    elemental composition of the grain (same bookkeeping as sputtering).

    Modified variables
    ------------------
    dydt_dust[idx]   -= 3 ε × y_dust[idx]
    dydt_gas[el_idx] += 3 ε × y_dust[idx] × el_mfrac[e]
    """
    Tk = state.local_Tk
    kmax = 0.0

    for db in state.dust_bins:
        idx = db.bin_index + state.npah

        if db.erosion_rate_interp is None:
            continue

        # Erosion rate ε [s⁻¹] from the pre-computed table.
        epsilon = float(db.erosion_rate_interp(Tk))
        if epsilon <= 0.0:
            continue

        # Fractional mass-loss rate: (1/ρ) dρ/dt = 3 ε  (since m ∝ a³)
        rate = 3.0 * epsilon          # [s⁻¹]
        kmax = max(kmax, rate)

        rate_rho = rate * y_dust[idx]  # [g cm⁻³ s⁻¹]

        dydt_dust[idx] -= rate_rho
        for loc, el_idx in enumerate(db.el_indices):
            dydt_gas[el_idx] += rate_rho * db.el_mfractions[loc]

    return kmax


# ---------------------------------------------------------------------------
# 4.  Grain coagulation  (Aoyama et al. 2017)
# ---------------------------------------------------------------------------

def coagulation_rate(
    state: DustChemistryState,
    y_gas: np.ndarray,
    y_dust: np.ndarray,
    dydt_gas: np.ndarray,
    dydt_dust: np.ndarray,
) -> float:
    """Grain coagulation: small bins merge into the next larger bin.

    Uses the simplified Aoyama et al. (2017) kernel

    .. math::

        \\text{rate} \\; [\\text{s}^{-1}] =
            k_0^\\text{coa} \\, \\frac{y_\\text{dust}[\\text{small}]}{n_H}

    Active only when ``T < 10⁴ K`` and ``n_H ≥ n_H^\\text{coa}``.

    Modified variables
    ------------------
    dydt_dust[small_idx]   -= rate × y_dust[small_idx]
    dydt_dust[partner_idx] += rate × y_dust[small_idx]
    """
    Tk = state.local_Tk
    nH = state.local_nH
    kmax = 0.0

    # Coagulation is suppressed at high temperature (shock environment)
    if Tk > 1.0e4:
        return kmax

    for db in state.dust_bins:
        if db.coag_partner_index is None:
            continue
        if nH < db.nh_coa:
            continue

        idx = db.bin_index + state.npah
        partner_idx = db.coag_partner_index + state.npah

        rate1 = db.k0_coa * y_dust[idx] / nH  # [s⁻¹]
        if rate1 <= 0.0:
            continue

        kmax = max(kmax, rate1)
        rate_rho = rate1 * y_dust[idx]  # [g cm⁻³ s⁻¹]

        dydt_dust[idx] -= rate_rho
        dydt_dust[partner_idx] += rate_rho

    return kmax


# ---------------------------------------------------------------------------
# 4b.  Dubois et al. (2024) simplified shattering
# ---------------------------------------------------------------------------

def dubois_shattering_rate(
    state: DustChemistryState,
    y_gas: np.ndarray,
    y_dust: np.ndarray,
    dydt_gas: np.ndarray,
    dydt_dust: np.ndarray,
) -> float:
    """Dubois et al. (2024) simplified shattering: large grains → small grains.

    Uses the Aoyama et al. (2017) timescale with a density-dependent velocity:

    .. math::

        t_\\mathrm{sha} = 54 \\,
            \\frac{a_L}{0.1\\,\\mu\\mathrm{m}} \\,
            \\frac{s_i}{3\\,\\mathrm{g\\,cm^{-3}}} \\,
            \\left(\\frac{D_L}{0.01}\\right)^{-1}
            \\left(\\frac{n}{1\\,\\mathrm{cm}^{-3}}\\right)^{-p_\\mathrm{sh}}
            \\mathrm{Myr}

    where :math:`p_\\mathrm{sh} = 1` for :math:`n < 1\\,\\mathrm{cm}^{-3}` and
    :math:`p_\\mathrm{sh} = 1/3` for :math:`1 < n < 10^3\\,\\mathrm{cm}^{-3}`.
    Shattering is inactive for :math:`n \\geq 10^3\\,\\mathrm{cm}^{-3}`.

    Mass is transferred from the **last** (largest) bin to the **first** (smallest)
    bin within each composition group, conserving total dust mass.
    """
    nH = state.local_nH
    rho_gas = state.local_rho

    if nH >= 1.0e3:
        return 0.0   # shattering inactive in dense molecular gas

    p_sh = 1.0 if nH < 1.0 else 1.0 / 3.0
    n_factor = nH ** p_sh

    kmax = 0.0

    for ii1, ii2 in _composition_groups(state.dust_bins):
        if ii2 <= ii1:
            continue   # need ≥ 2 bins per composition group

        db_large = state.dust_bins[ii2]
        db_small = state.dust_bins[ii1]
        idx_large = db_large.bin_index + state.npah
        idx_small = db_small.bin_index + state.npah

        rho_L = y_dust[idx_large]
        if rho_L <= db_large.smallr_dust:
            continue

        D_L = rho_L / rho_gas                   # large-bin dust-to-gas ratio
        # 1/t_sha = D_L × n^{p_sh} / (t_sha_ref × (a_L/0.1µm) × s_3 × 0.01) yr^{-1}
        # Dubois+2024: t_sha = 54 × a_{0.1} × s_3 × (D_L/0.01)^{-1} × n^{-p_sh} Myr
        # s_3 = s_grain / 3 (normalised grain density); factor /3 was missing before.
        # denom [s] = 5.41e5 × (a_L/0.1) × (s_i/3) yr × YR2SEC
        denom = 5.41e5 * (db_large.asize_micron / 0.1) * (db_large.sgrain / 3.0) * YR2SEC
        rate_sha = D_L * n_factor / denom        # [s^{-1}]

        if rate_sha <= 0.0:
            continue

        kmax = max(kmax, rate_sha)
        rate_mass = rate_sha * rho_L             # [g cm^{-3} s^{-1}]

        dydt_dust[idx_large] -= rate_mass
        dydt_dust[idx_small] += rate_mass

    return kmax


# ---------------------------------------------------------------------------
# 4c.  Dubois et al. (2024) simplified coagulation
# ---------------------------------------------------------------------------

def dubois_coagulation_rate(
    state: DustChemistryState,
    y_gas: np.ndarray,
    y_dust: np.ndarray,
    dydt_gas: np.ndarray,
    dydt_dust: np.ndarray,
) -> float:
    """Dubois et al. (2024) simplified coagulation: small grains → large grains.

    Matches the ``coagulation_dispersion='unstable_mc'`` case of the Dubois+2024
    RAMSES Fortran with ``power_coa=0`` and ``power_boost_coa=0`` (no density or
    boost power-law correction):

    .. math::

        t_\\mathrm{coa} = t_\\mathrm{coa,ref} \\,
            \\frac{a_S}{0.005\\,\\mu\\mathrm{m}} \\,
            s_i \\,
            \\left(\\frac{D_S}{0.01}\\right)^{-1}

    where :math:`t_\\mathrm{coa,ref} = 5.42 \\times 10^3\\,\\mathrm{yr}`
    (from the Dubois+2024 namelist ``t_coa_ref = 5.42e5\\,\\mathrm{yr}`` × 0.01).

    Active only when :math:`T < 10^4\\,\\mathrm{K}`, :math:`n_H \\geq 0.1\\,\\mathrm{cm}^{-3}`,
    **and** the Jeans length does not exceed :math:`4\\Delta x` (the
    ``coagulation_dispersion='unstable_mc'`` criterion in Dubois et al. 2024 Fortran).

    Mass is transferred from the **first** (smallest) bin to the **last** (largest)
    bin within each composition group, conserving total dust mass.
    """
    _MH_CGS = 1.6726219e-24
    _G_CGS  = 6.674e-8

    Tk = state.local_Tk
    nH = state.local_nH
    rho_gas = state.local_rho

    if Tk > 1.0e4 or nH < 0.1:
        return 0.0

    # Jeans length criterion (Dubois+2024 Fortran, unstable_mc case):
    # suppress coagulation when gas is not self-gravitating (λ_J > 4Δx)
    _dx = state.local_dx
    if _dx > 0.0:
        _lambda_jeans = math.sqrt(math.pi * KB_CGS * Tk / (_G_CGS * _MH_CGS**2 * nH))
        if _lambda_jeans > 4.0 * _dx:
            return 0.0

    kmax = 0.0

    for ii1, ii2 in _composition_groups(state.dust_bins):
        if ii2 <= ii1:
            continue   # need ≥ 2 bins per composition group

        db_small = state.dust_bins[ii1]
        db_large = state.dust_bins[ii2]
        idx_small = db_small.bin_index + state.npah
        idx_large = db_large.bin_index + state.npah

        rho_S = y_dust[idx_small]
        if rho_S <= db_small.smallr_dust:
            continue

        D_S = rho_S / rho_gas                   # small-bin dust-to-gas ratio
        # 1/t_coa = D_S / (t_coa_ref × (a_S/0.005µm) × s_3 × 0.01) yr^{-1}
        # Dubois+2024: t_coa = 0.27/F × a_{0.005} × s_3 × (D_S/0.01)^{-1} Myr
        # s_3 = s_grain / 3 (normalised grain density); factor /3 was missing before.
        # denom [s] = 5.42e3 × (a_S/0.005) × (s_i/3) yr × YR2SEC
        denom = 5.42e3 * (db_small.asize_micron / 0.005) * (db_small.sgrain / 3.0) * YR2SEC
        rate_coa = D_S / denom                   # [s^{-1}]

        if rate_coa <= 0.0:
            continue

        kmax = max(kmax, rate_coa)
        rate_mass = rate_coa * rho_S             # [g cm^{-3} s^{-1}]

        dydt_dust[idx_small] -= rate_mass
        dydt_dust[idx_large] += rate_mass

    return kmax


# ---------------------------------------------------------------------------
# 4.  PAH accretion of gas-phase carbon
# ---------------------------------------------------------------------------

def pah_accretion_rate(
    state: DustChemistryState,
    y_gas: np.ndarray,
    y_dust: np.ndarray,
    dydt_gas: np.ndarray,
    dydt_dust: np.ndarray,
) -> float:
    """PAH growth by sticking of gas-phase C atoms.

    Uses the same LeBourlot et al. (2012) thermal accretion formula as
    :func:`accretion_rate`, restricted to the carbon element and applied
    to each PAH bin using its effective geometric cross section.

    Modified variables
    ------------------
    dydt_dust[pah_idx] += rate × y_dust[pah_idx]
    dydt_gas[C_idx]    -= rate × y_dust[pah_idx]
    """
    Tk = state.local_Tk
    prefactor = math.sqrt(Tk) / (1.0 + 1.0e-4 * Tk ** 1.5)
    kmax = 0.0

    el_name = "C"
    if el_name not in state.el_names:
        return kmax
    c_idx = state.el_names.index(el_name)
    m_c = state.el_atomic_mass_g[c_idx]
    rho_c = y_gas[c_idx]
    if rho_c <= 0.0:
        return kmax

    for pb in state.pah_bins:
        idx = pb.bin_index
        if y_dust[idx] < pb.smallr_pah or pb.mpah <= 0.0:
            continue

        # Effective PAH radius from mass and material density
        apah = (3.0 * pb.mpah / (4.0 * math.pi * pb.spah)) ** (1.0 / 3.0)  # [cm]
        k0_pah = math.pi * apah ** 2 * math.sqrt(8.0 * KB_CGS / math.pi) / pb.mpah

        rate = k0_pah * prefactor * rho_c / math.sqrt(m_c)  # [s⁻¹]

        if rate <= 0.0:
            continue

        kmax = max(kmax, rate)
        rate_rho = rate * y_dust[idx]  # [g cm⁻³ s⁻¹]

        dydt_dust[idx] += rate_rho
        dydt_gas[c_idx] -= rate_rho

    return kmax


# ---------------------------------------------------------------------------
# Helper: group dust bins by chemical composition
# ---------------------------------------------------------------------------

def _composition_groups(
    dust_bins: List[DustBinParams],
) -> List[Tuple[int, int]]:
    """Return list of (ii1, ii2) index ranges for each composition type.

    All bins within a range share the same ``composition`` string.  Bins are
    assumed to be contiguous within each chemical type (enforced by load order).
    """
    if not dust_bins:
        return []
    groups: List[Tuple[int, int]] = []
    current_comp = dust_bins[0].composition
    ii1 = 0
    for db in dust_bins[1:]:
        if db.composition != current_comp:
            groups.append((ii1, db.bin_index - 1))
            ii1 = db.bin_index
            current_comp = db.composition
    groups.append((ii1, dust_bins[-1].bin_index))
    return groups


# ---------------------------------------------------------------------------
# Helper: effective turbulent velocity (fallback for local_sigma = 0)
# ---------------------------------------------------------------------------

def _effective_sigma(state: DustChemistryState) -> Tuple[float, float]:
    """Return (v_turb [cm/s], inject_L [cm]) for turbulent processes.

    If ``state.local_sigma`` is zero, use the CNM equilibrium estimate from
    the RAMSES dust-equilibrium-test mode:
        σ ≈ 5.67 × 10⁵ (nH/100)^{-0.25}  cm/s
        L ≈ 10 pc × (nH/100)^{-1/3}
    """
    if state.local_sigma > 0.0 and state.local_dx > 0.0:
        return state.local_sigma, state.local_dx
    _PC2CM = 3.085677581e18
    nH = max(state.local_nH, 1.0e-10)
    sigma = 5.67e5 * (nH / 100.0) ** (-0.25)
    L = 10.0 * _PC2CM * (nH / 100.0) ** (-1.0 / 3.0)
    return sigma, L


# ---------------------------------------------------------------------------
# 5.  PAH photolysis  (table interpolation on G0 × nH)
# ---------------------------------------------------------------------------

def pah_photolysis_rate(
    state: DustChemistryState,
    y_gas: np.ndarray,
    y_dust: np.ndarray,
    dydt_gas: np.ndarray,
    dydt_dust: np.ndarray,
) -> float:
    """PAH photodissociation by UV photons.

    For each non-cluster PAH bin, queries the pre-loaded 2-D photolysis table
    to get the dissociation rate per molecule [s⁻¹] at (G₀, nH), then converts
    to a mass flux removing C₂ units (factor 2 × m_C) from the PAH bin and
    returning carbon to the gas phase.

    Mirrors ``pah_photolysis_rate`` in RAMSES ``dust_rates.f90``.
    """
    if state.local_G0 <= 0.0:
        return 0.0

    log_G0 = math.log10(max(state.local_G0, 1.0e-10))
    log_nH = math.log10(max(state.local_nH, 1.0e-10))
    c_idx = state.el_names.index("C") if "C" in state.el_names else 2
    m_C = state.el_atomic_mass_g[c_idx]
    kmax = 0.0

    for pb in state.pah_bins:
        if pb.is_cluster:
            continue
        if pb.dissociation_interp is None:
            continue
        if pb.mpah <= 0.0:
            continue
        pp = pb.bin_index
        rho_pah = y_dust[pp]
        if rho_pah <= pb.smallr_pah:
            continue

        log_rate = pb.dissociation_interp(log_G0, log_nH)
        rate1 = 10.0 ** log_rate  # [s⁻¹] per molecule

        # Mass rate: each dissociation removes 2 C atoms (C₂H₂ ejection)
        rate2 = rate1 * rho_pah / pb.mpah * (2.0 * m_C)  # [g cm⁻³ s⁻¹]
        kmax = max(kmax, rate1)

        dydt_dust[pp] -= rate2
        dydt_gas[c_idx] += rate2

    return kmax


# ---------------------------------------------------------------------------
# 6.  PAH sputtering  (1-D T table per species)
# ---------------------------------------------------------------------------

def pah_sputtering_rate(
    state: DustChemistryState,
    y_gas: np.ndarray,
    y_dust: np.ndarray,
    dydt_gas: np.ndarray,
    dydt_dust: np.ndarray,
) -> float:
    """PAH destruction by thermal sputtering from gas ions and electrons.

    For each non-cluster PAH bin:
        R_total [s⁻¹] = nₑ × Jₑ(T) + Σᵢ nᵢ × Jᵢ(T)
        rate [g cm⁻³ s⁻¹] = R_total × ρ_PAH / m_PAH × m_C

    J(T) [cm³ s⁻¹] is the per-particle sputtering rate constant from the
    pre-loaded 1-D tables.  Mirrors ``pah_sputtering_rate`` in RAMSES.
    """
    Tk = state.local_Tk
    c_idx = state.el_names.index("C") if "C" in state.el_names else 2
    m_C = state.el_atomic_mass_g[c_idx]
    kmax = 0.0

    for pb in state.pah_bins:
        if pb.is_cluster:
            continue
        if not pb.sputtering_interps:
            continue
        if pb.mpah <= 0.0:
            continue
        pp = pb.bin_index
        rho_pah = y_dust[pp]
        if rho_pah <= pb.smallr_pah:
            continue

        R_total = 0.0  # [s⁻¹]

        for species, interp_fn in pb.sputtering_interps.items():
            if species == "electrons":
                n_sp = state.local_ne
            elif species in state.el_names:
                el_idx = state.el_names.index(species)
                rho_sp = y_gas[el_idx]
                m_sp = state.el_atomic_mass_g[el_idx]
                n_sp = rho_sp / max(m_sp, 1.0e-100)
            else:
                continue

            if n_sp < 1.0e-40:
                continue

            J = interp_fn(Tk)  # [cm³ s⁻¹]
            R_total += n_sp * J

        if R_total <= 0.0:
            continue

        rate2 = R_total * rho_pah / pb.mpah * m_C  # [g cm⁻³ s⁻¹]
        kmax = max(kmax, R_total)

        dydt_dust[pp] -= rate2
        dydt_gas[c_idx] += rate2

    return kmax


# ---------------------------------------------------------------------------
# 7a.  PAH coalescence  (Totton et al. 2012)
# ---------------------------------------------------------------------------

def totton2012_pah_coalescence_rate(
    state: DustChemistryState,
    y_gas: np.ndarray,
    y_dust: np.ndarray,
    dydt_gas: np.ndarray,
    dydt_dust: np.ndarray,
) -> float:
    """PAH coalescence using Totton et al. (2012) collision efficiency.

    Smaller PAH bin coagulates into the next larger bin.  The collision
    efficiency C_eff accounts for the probability that two colliding PAHs
    stick:

        C_eff = 1 / (1 + 9.93e-7 × (log₁₀ T)^13.79)

    Mirrors ``pah_coalescence_rate`` (Totton2012 branch) in RAMSES.
    """
    Tk = state.local_Tk
    if Tk <= 0.0:
        return 0.0
    kmax = 0.0
    log10_T = math.log10(max(Tk, 1.0))
    C_eff = 1.0 / (1.0 + 9.92807181e-7 * log10_T ** 13.7933821)

    for i_pp in range(state.npah - 1):
        pb = state.pah_bins[i_pp]
        if pb.is_cluster:
            continue
        if pb.mpah <= 0.0 or pb.apah_cm <= 0.0:
            continue
        pp = pb.bin_index
        rho_pah = y_dust[pp]
        if rho_pah <= pb.smallr_pah:
            continue

        # Self-collision thermal velocity using reduced mass = m/2
        reduced_mass = 0.5 * pb.mpah
        dV_thermal = math.sqrt(8.0 * KB_CGS * Tk / max(reduced_mass, 1.0e-100))
        coll_section = 4.0 * math.pi * pb.apah_cm ** 2

        rate1 = coll_section * dV_thermal * C_eff * rho_pah / pb.mpah  # [s⁻¹]
        if rate1 <= 0.0:
            continue

        rate2 = rate1 * rho_pah  # [g cm⁻³ s⁻¹]
        kmax = max(kmax, rate1)

        dydt_dust[pp] -= rate2
        dydt_dust[pp + 1] += rate2

    return kmax


# ---------------------------------------------------------------------------
# 7b.  PAH coalescence  (Tielens 2021, neutral approximation)
# ---------------------------------------------------------------------------

def tielens2021_pah_coalescence_rate(
    state: DustChemistryState,
    y_gas: np.ndarray,
    y_dust: np.ndarray,
    dydt_gas: np.ndarray,
    dydt_dust: np.ndarray,
) -> float:
    """PAH coalescence using Tielens (2021) rate coefficients.

    Neutral approximation (no ion fractions available in the current solver):
        R₁ = 4e-11 × √(T/10) × √(nc/50)    [cm³ s⁻¹]

    Smaller PAH bin coagulates into the next larger bin.
    """
    Tk = state.local_Tk
    if Tk <= 0.0:
        return 0.0
    kmax = 0.0

    for i_pp in range(state.npah - 1):
        pb = state.pah_bins[i_pp]
        if pb.is_cluster:
            continue
        if pb.mpah <= 0.0:
            continue
        pp = pb.bin_index
        rho_pah = y_dust[pp]
        if rho_pah <= pb.smallr_pah:
            continue

        nc_eff = max(pb.nc, 1)
        R1 = 4.0e-11 * math.sqrt(Tk / 10.0) * math.sqrt(nc_eff / 50.0)  # [cm³ s⁻¹]

        n_pah = rho_pah / pb.mpah  # [cm⁻³]
        rate1 = R1 * n_pah         # [s⁻¹]  (loss rate per molecule × n)
        if rate1 <= 0.0:
            continue

        rate2 = rate1 * rho_pah    # [g cm⁻³ s⁻¹]
        kmax = max(kmax, rate1)

        dydt_dust[pp] -= rate2
        dydt_dust[pp + 1] += rate2

    return kmax


# ---------------------------------------------------------------------------
# 8.  PAH cluster evaporation  (Montillaud et al. 2014 analytic)
# ---------------------------------------------------------------------------

def pah_cluster_evaporation_rate(
    state: DustChemistryState,
    y_gas: np.ndarray,
    y_dust: np.ndarray,
    dydt_gas: np.ndarray,
    dydt_dust: np.ndarray,
) -> float:
    """Photo-evaporation of PAH clusters into monomers.

    Only acts on ``is_cluster=True`` PAH bins.  Each cluster loses one PAH
    monomer at a time (to the bin immediately below it, pp-1).

    Rate coefficients (Montillaud et al. 2014 / RAMSES default):
        k_single = G₀ / 0.19306
        k_multi  = 1 / 10^(-3.169 log₁₀ G₀ + 13.564)
        rate [s⁻¹] = min(k_single, k_multi) / t_yr

    Mirrors ``pah_cluster_evaporation_rate`` in RAMSES.
    """
    G0 = max(state.local_G0, 1.0e-10)
    log_G0 = math.log10(G0)
    k_single = G0 / 0.19306
    k_multi = 1.0 / (10.0 ** (-3.1692061 * log_G0 + 13.5642486))
    rate1_s = min(k_single, k_multi) / YR2SEC  # [s⁻¹] evaporation rate per cluster

    kmax = 0.0

    for pp in range(1, state.npah):
        pb = state.pah_bins[pp]
        if not pb.is_cluster:
            continue
        if pb.mpah <= 0.0:
            continue
        rho_pah = y_dust[pp]
        if rho_pah <= pb.smallr_pah:
            continue

        # Mass leaving cluster bin (as monomers going to pp-1)
        pb_mono = state.pah_bins[pp - 1]
        rate2 = rate1_s * rho_pah / pb.mpah * pb_mono.mpah  # [g cm⁻³ s⁻¹]
        kmax = max(kmax, rate1_s)

        dydt_dust[pp] -= rate2
        dydt_dust[pp - 1] += rate2

    return kmax


# ---------------------------------------------------------------------------
# 9.  PAH freezing onto dust grains
# ---------------------------------------------------------------------------

def pah_freezing_rate(
    state: DustChemistryState,
    y_gas: np.ndarray,
    y_dust: np.ndarray,
    dydt_gas: np.ndarray,
    dydt_dust: np.ndarray,
) -> float:
    """PAH accretion (freezing) onto dust grain surfaces.

    For each PAH bin and each interacting dust bin:
        v_rel from grain_relative_velocity
        σ_coll = π (a_PAH + a_dust)²
        rate1  = √(8/(3π)) × σ_coll × v_rel / m_PAH × ρ_dust / m_dust
        rate2  = p_stick × rate1 × ρ_PAH   [g cm⁻³ s⁻¹]

    The PAH bin loses mass; the dust bin gains mass (PAHs freeze on surface).
    Uses D_av = 1 (no Coulomb correction) since grain charge is not tracked.
    """
    sigma, inject_L = _effective_sigma(state)
    kmax = 0.0
    Tk = state.local_Tk
    rho_gas = state.local_rho
    nH = state.local_nH
    mu = state.local_mu
    model = state.dust_velocity_model

    for pb in state.pah_bins:
        pp = pb.bin_index
        if pb.mpah <= 0.0 or pb.apah_cm <= 0.0:
            continue
        rho_pah = y_dust[pp]
        if rho_pah <= pb.smallr_pah:
            continue

        # Which dust bins does this PAH bin interact with?
        if pb.dust_index_interact < 0 or pb.nd_bins == 0:
            # Default: interact with all dust bins of any graphite type
            dust_range = range(len(state.dust_bins))
        else:
            d_start = pb.dust_index_interact
            dust_range = range(d_start, min(d_start + pb.nd_bins, len(state.dust_bins)))

        for id_d in dust_range:
            db = state.dust_bins[id_d]
            idx_d = db.bin_index + state.npah
            rho_dust = y_dust[idx_d]
            if rho_dust <= db.smallr_dust:
                continue

            v_rel = grain_relative_velocity(
                model, Tk, rho_gas, nH, sigma, mu, inject_L,
                pb.apah_cm, pb.spah, pb.mpah,
                db.asize_cm, db.sgrain, db.mgrain,
            )
            coll_section = (
                math.sqrt(8.0 / (3.0 * math.pi))
                * math.pi * (pb.apah_cm + db.asize_cm) ** 2
            )
            p_stick = sticking_probability_from_velocity(v_rel, db.vthresh_coag)

            # rate1 [s⁻¹]: fractional mass loss rate of PAH bin
            rate1 = coll_section * v_rel / pb.mpah * rho_dust / db.mgrain * pb.mpah * p_stick
            # Simplify: (coll_sec * v_rel) * (n_dust) * p_stick * (m_PAH / m_PAH)
            # = coll_sec * v_rel * rho_dust / m_dust * p_stick   [s⁻¹]
            rate1 = coll_section * v_rel * rho_dust / db.mgrain * p_stick  # [s⁻¹]
            if rate1 <= 0.0:
                continue

            rate2 = rate1 * rho_pah  # [g cm⁻³ s⁻¹]
            kmax = max(kmax, rate1)

            dydt_dust[pp] -= rate2
            dydt_dust[idx_d] += rate2

    return kmax


# ---------------------------------------------------------------------------
# 10.  Fragment distribution helper (shattering)
# ---------------------------------------------------------------------------

def _compute_shattered_fragments(
    state: DustChemistryState,
    id1: int,
    id2: int,
    group_ii1: int,
    group_ii2: int,
) -> Tuple[List[float], List[float], float]:
    """Compute the fragment mass distribution from a shattering collision.

    Parameters
    ----------
    id1, id2 :
        Dust-bin indices of the target and projectile (into ``state.dust_bins``).
    group_ii1, group_ii2 :
        First and last bin indices of the chemical-type group (for fragment
        distribution within the same composition).

    Returns
    -------
    chi_frag : list of float, length == group size
        Fraction of ejected mass going into each dust bin in the group.
    chi_pah : list of float, length == state.npah
        Fraction of ejected mass going into each PAH bin.
    chi_dest : float
        Fraction of ejected mass that is sub-resolution (returned to gas).
    """
    db1 = state.dust_bins[id1]
    db2 = state.dust_bins[id2]
    sigma, inject_L = _effective_sigma(state)

    v_rel = grain_relative_velocity(
        state.dust_velocity_model,
        state.local_Tk, state.local_rho, state.local_nH,
        sigma, state.local_mu, inject_L,
        db1.asize_cm, db1.sgrain, db1.mgrain,
        db2.asize_cm, db2.sgrain, db2.mgrain,
    )

    m1 = db1.mgrain
    m2 = db2.mgrain
    E_imp = 0.5 * m1 * m2 / max(m1 + m2, 1.0e-100) * v_rel ** 2
    phi_factor = E_imp / max(m1 * db1.catastrophic_spec_energy, 1.0e-100)
    m_ej = phi_factor / (1.0 + phi_factor) * m1
    # m_remnant = m1 - m_ej  (goes into nearest bin; handled in caller)

    n_group = group_ii2 - group_ii1 + 1
    chi_frag = [0.0] * n_group
    chi_pah = [0.0] * state.npah
    chi_dest = 0.0

    if m_ej <= 0.0:
        return chi_frag, chi_pah, chi_dest

    # Fragment power-law mass distribution:  dN/dm ∝ m^{-α},  α = slope+1
    # Integrate: ∫ m × m^{-α} dm = m^{1-α}/(1-α)
    # → cumulative mass ∝ m^{1-α} / (1-α) = m^β / β  with β = 1-α = 1 - (1+slope)
    # RAMSES uses  dM/dm ∝ m^{-slope}  (mass-weighted power law)
    # ∫_{m1}^{m2} m^{-slope} dm = [m^{1-slope}/(1-slope)]
    # We use α = state.slope_frag_func (= 1.3/3 ≈ 0.433)
    alpha = state.slope_frag_func   # exponent in dM/dm ∝ m^{-alpha}
    beta = 1.0 - alpha              # integration exponent

    m_max_frag = 0.02 * m_ej
    m_min_frag = 1.0e-6 * max(m_max_frag, 1.0e-100)

    # Normalization: ∫_{m_min}^{m_max} m^{-alpha} dm = [m^beta/beta]
    norm_denom = (m_max_frag ** beta - m_min_frag ** beta) / beta
    if abs(norm_denom) < 1.0e-100:
        return chi_frag, chi_pah, chi_dest
    prefactor = m_ej / norm_denom

    # --- Assign to dust bins within the same composition group ---
    total_assigned = 0.0
    for loc, gidx in enumerate(range(group_ii1, group_ii2 + 1)):
        db = state.dust_bins[gidx]
        m_lo = max(db.mgrain_min, m_min_frag)
        m_hi = min(db.mgrain_max, m_max_frag)
        if m_lo >= m_hi:
            continue
        chi_frag[loc] = prefactor * (m_hi ** beta - m_lo ** beta) / beta
        total_assigned += chi_frag[loc]

    # --- Assign to PAH bins (only if interact_pah=True for this dust type) ---
    if db1.interact_pah and state.npah > 0:
        for i_pp, pb in enumerate(state.pah_bins):
            if pb.mpah_min <= 0.0 or pb.mpah_max <= 0.0:
                continue
            m_lo = max(pb.mpah_min, m_min_frag)
            m_hi = min(pb.mpah_max, m_max_frag)
            if m_lo >= m_hi:
                continue
            chi_pah[i_pp] = prefactor * (m_hi ** beta - m_lo ** beta) / beta
            total_assigned += chi_pah[i_pp]

    # --- Sub-resolution mass → gas (chi_dest) ---
    # Any mass below the smallest tracked grain (min of all grain bins)
    all_m_min = min((db.mgrain_min for db in state.dust_bins if db.mgrain_min > 0),
                    default=m_min_frag)
    if state.npah > 0:
        all_m_min = min(all_m_min,
                        min((pb.mpah_min for pb in state.pah_bins if pb.mpah_min > 0),
                            default=all_m_min))
    m_dest_hi = min(all_m_min, m_max_frag)
    if m_dest_hi > m_min_frag:
        chi_dest = prefactor * (m_dest_hi ** beta - m_min_frag ** beta) / beta
        total_assigned += chi_dest

    # Normalize so that chi_frag + chi_pah + chi_dest = 1
    total = total_assigned + max(m_ej - total_assigned, 0.0) / m_ej
    if total > 0.0:
        scale = 1.0 / total
        chi_frag = [c * scale for c in chi_frag]
        chi_pah = [c * scale for c in chi_pah]
        chi_dest *= scale

    return chi_frag, chi_pah, chi_dest


# ---------------------------------------------------------------------------
# 11.  Turbulent shattering – self-collisions
# ---------------------------------------------------------------------------

def turbulent_shattering_rate(
    state: DustChemistryState,
    y_gas: np.ndarray,
    y_dust: np.ndarray,
    dydt_gas: np.ndarray,
    dydt_dust: np.ndarray,
) -> float:
    """Turbulent grain shattering: each bin collides with itself.

    For each dust bin within a composition group:
        v_rel = grain_relative_velocity(bin, bin)
        rate1 = √(8/(3π)) × 4π a² × v_rel × ρ_bin / m_grain   [s⁻¹]
        rate2 = rate1 × ρ_bin   [g cm⁻³ s⁻¹]

    Ejected mass is redistributed via the power-law fragment distribution.
    The remnant mass goes into the nearest bin by mass within the same group.

    Mirrors ``turbulent_shattering_rate`` in RAMSES.
    """
    sigma, inject_L = _effective_sigma(state)
    if sigma <= 0.0:
        return 0.0

    kmax = 0.0
    Tk = state.local_Tk
    rho_gas = state.local_rho
    nH = state.local_nH
    mu = state.local_mu
    model = state.dust_velocity_model
    alpha = state.slope_frag_func
    beta = 1.0 - alpha

    for ii1, ii2 in _composition_groups(state.dust_bins):
        for jj in range(ii1, ii2 + 1):
            db = state.dust_bins[jj]
            idx = db.bin_index + state.npah
            rho_d = y_dust[idx]
            if rho_d <= db.smallr_dust:
                continue

            v_rel = grain_relative_velocity(
                model, Tk, rho_gas, nH, sigma, mu, inject_L,
                db.asize_cm, db.sgrain, db.mgrain,
                db.asize_cm, db.sgrain, db.mgrain,
            )
            if v_rel <= 0.0:
                continue

            coll_factor = (
                math.sqrt(8.0 / (3.0 * math.pi))
                * 4.0 * math.pi * db.asize_cm ** 2 * v_rel
            )
            rate1 = coll_factor * rho_d / db.mgrain  # [s⁻¹]
            rate2 = rate1 * rho_d                    # [g cm⁻³ s⁻¹]
            kmax = max(kmax, rate1)

            chi_frag, chi_pah, chi_dest = _compute_shattered_fragments(
                state, jj, jj, ii1, ii2
            )

            # Remove from source bin
            dydt_dust[idx] -= rate2

            # Remnant → bin closest to m_remnant within the group
            # (handled as fraction 1 - sum(chi_frag) - sum(chi_pah) - chi_dest)
            chi_rem = 1.0 - sum(chi_frag) - sum(chi_pah) - chi_dest
            if chi_rem > 0.0:
                # Find nearest bin by mass
                m_rem = (1.0 - chi_rem) * db.mgrain  # approximate remnant mass
                best = ii1
                best_diff = abs(state.dust_bins[best].mgrain - m_rem)
                for gg in range(ii1, ii2 + 1):
                    diff = abs(state.dust_bins[gg].mgrain - m_rem)
                    if diff < best_diff:
                        best_diff = diff
                        best = gg
                dydt_dust[best + state.npah] += chi_rem * rate2

            # Fragments → dust bins in group
            for loc, gidx in enumerate(range(ii1, ii2 + 1)):
                if chi_frag[loc] > 0.0:
                    dydt_dust[gidx + state.npah] += chi_frag[loc] * rate2

            # Fragments → PAH bins
            for i_pp, pb in enumerate(state.pah_bins):
                if chi_pah[i_pp] > 0.0:
                    dydt_dust[i_pp] += chi_pah[i_pp] * rate2

            # Sub-resolution → gas (carbon only for graphite)
            if chi_dest > 0.0:
                c_idx = state.el_names.index("C") if "C" in state.el_names else 2
                for loc, el_idx in enumerate(db.el_indices):
                    dydt_gas[el_idx] += chi_dest * rate2 * db.el_mfractions[loc]

    return kmax


# ---------------------------------------------------------------------------
# 12.  Turbulent shattering – all pairs within a composition group
# ---------------------------------------------------------------------------

def turbulent_all_shattering_rate(
    state: DustChemistryState,
    y_gas: np.ndarray,
    y_dust: np.ndarray,
    dydt_gas: np.ndarray,
    dydt_dust: np.ndarray,
) -> float:
    """Turbulent grain shattering: all bin pairs within each composition type.

    For each ordered pair (ii, jj) with jj ≥ ii:
        mass_rate_ii = coll_factor_ij × (ρ_jj/m_jj) × ρ_ii  [g cm⁻³ s⁻¹]
        For self-collisions (ii == jj) multiply by 0.5.

    Fragment distribution applied to both bins in the pair.
    Mirrors ``turbulent_all_shattering_rate`` in RAMSES.
    """
    sigma, inject_L = _effective_sigma(state)
    if sigma <= 0.0:
        return 0.0

    kmax = 0.0
    Tk = state.local_Tk
    rho_gas = state.local_rho
    nH = state.local_nH
    mu = state.local_mu
    model = state.dust_velocity_model

    for ii1, ii2 in _composition_groups(state.dust_bins):
        for ii in range(ii1, ii2 + 1):
            db_ii = state.dust_bins[ii]
            idx_ii = db_ii.bin_index + state.npah
            rho_ii = y_dust[idx_ii]
            if rho_ii <= db_ii.smallr_dust:
                continue

            for jj in range(ii, ii2 + 1):
                db_jj = state.dust_bins[jj]
                idx_jj = db_jj.bin_index + state.npah
                rho_jj = y_dust[idx_jj]
                if rho_jj <= db_jj.smallr_dust:
                    continue

                v_rel = grain_relative_velocity(
                    model, Tk, rho_gas, nH, sigma, mu, inject_L,
                    db_ii.asize_cm, db_ii.sgrain, db_ii.mgrain,
                    db_jj.asize_cm, db_jj.sgrain, db_jj.mgrain,
                )
                if v_rel <= 0.0:
                    continue

                coll_factor = (
                    math.sqrt(8.0 / (3.0 * math.pi))
                    * math.pi * (db_ii.asize_cm + db_jj.asize_cm) ** 2
                    * v_rel
                )
                sym = 0.5 if ii == jj else 1.0

                rate_ii = sym * coll_factor * rho_jj / db_jj.mgrain * rho_ii
                rate_jj = sym * coll_factor * rho_ii / db_ii.mgrain * rho_jj

                kmax = max(kmax, sym * coll_factor * rho_jj / db_jj.mgrain)

                chi_frag_ii, chi_pah_ii, chi_dest_ii = _compute_shattered_fragments(
                    state, ii, jj, ii1, ii2
                )
                chi_frag_jj, chi_pah_jj, chi_dest_jj = _compute_shattered_fragments(
                    state, jj, ii, ii1, ii2
                )

                # Apply destruction for bin ii
                dydt_dust[idx_ii] -= rate_ii
                chi_rem_ii = 1.0 - sum(chi_frag_ii) - sum(chi_pah_ii) - chi_dest_ii
                if chi_rem_ii > 0.0:
                    dydt_dust[idx_ii] += chi_rem_ii * rate_ii
                for loc, gidx in enumerate(range(ii1, ii2 + 1)):
                    if chi_frag_ii[loc] > 0.0:
                        dydt_dust[gidx + state.npah] += chi_frag_ii[loc] * rate_ii
                for i_pp in range(state.npah):
                    if chi_pah_ii[i_pp] > 0.0:
                        dydt_dust[i_pp] += chi_pah_ii[i_pp] * rate_ii
                if chi_dest_ii > 0.0:
                    for loc, el_idx in enumerate(db_ii.el_indices):
                        dydt_gas[el_idx] += chi_dest_ii * rate_ii * db_ii.el_mfractions[loc]

                # Apply destruction for bin jj (skip if self-collision: already done)
                if ii != jj:
                    dydt_dust[idx_jj] -= rate_jj
                    chi_rem_jj = 1.0 - sum(chi_frag_jj) - sum(chi_pah_jj) - chi_dest_jj
                    if chi_rem_jj > 0.0:
                        dydt_dust[idx_jj] += chi_rem_jj * rate_jj
                    for loc, gidx in enumerate(range(ii1, ii2 + 1)):
                        if chi_frag_jj[loc] > 0.0:
                            dydt_dust[gidx + state.npah] += chi_frag_jj[loc] * rate_jj
                    for i_pp in range(state.npah):
                        if chi_pah_jj[i_pp] > 0.0:
                            dydt_dust[i_pp] += chi_pah_jj[i_pp] * rate_jj
                    if chi_dest_jj > 0.0:
                        for loc, el_idx in enumerate(db_jj.el_indices):
                            dydt_gas[el_idx] += chi_dest_jj * rate_jj * db_jj.el_mfractions[loc]

    return kmax


# ---------------------------------------------------------------------------
# 13.  Turbulent coagulation – self-collisions
# ---------------------------------------------------------------------------

def turbulent_coagulation_rate(
    state: DustChemistryState,
    y_gas: np.ndarray,
    y_dust: np.ndarray,
    dydt_gas: np.ndarray,
    dydt_dust: np.ndarray,
) -> float:
    """Turbulent coagulation: each bin self-collides and merges upward.

    Rate:
        v_rel from grain_relative_velocity
        p_stick = sticking_probability_from_velocity(v_rel, v_coag)
        rate1 = k0_coa × v_rel × ρ_bin × p_stick   [s⁻¹]

    The next larger bin (coag_partner_index) receives the merged mass.
    Active only when T < 10⁴ K.
    Mirrors ``turbulent_coagulation_rate`` in RAMSES.
    """
    Tk = state.local_Tk
    if Tk > 1.0e4:
        return 0.0

    sigma, inject_L = _effective_sigma(state)
    if sigma <= 0.0:
        return 0.0

    kmax = 0.0
    rho_gas = state.local_rho
    nH = state.local_nH
    mu = state.local_mu
    model = state.dust_velocity_model

    for db in state.dust_bins:
        if db.coag_partner_index is None:
            continue
        idx = db.bin_index + state.npah
        rho_d = y_dust[idx]
        if rho_d <= db.smallr_dust:
            continue

        v_rel = grain_relative_velocity(
            model, Tk, rho_gas, nH, sigma, mu, inject_L,
            db.asize_cm, db.sgrain, db.mgrain,
            db.asize_cm, db.sgrain, db.mgrain,
        )
        p_stick = sticking_probability_from_velocity(v_rel, db.vthresh_coag)

        rate1 = db.k0_coa * v_rel * rho_d * p_stick  # [s⁻¹]
        if rate1 <= 0.0:
            continue

        kmax = max(kmax, rate1)
        rate2 = rate1 * rho_d  # [g cm⁻³ s⁻¹]

        partner_idx = db.coag_partner_index + state.npah
        dydt_dust[idx] -= rate2
        dydt_dust[partner_idx] += rate2

    return kmax


# ---------------------------------------------------------------------------
# 14.  Turbulent coagulation – all pairs within a composition group
# ---------------------------------------------------------------------------

def turbulent_all_coagulation_rate(
    state: DustChemistryState,
    y_gas: np.ndarray,
    y_dust: np.ndarray,
    dydt_gas: np.ndarray,
    dydt_dust: np.ndarray,
) -> float:
    """Turbulent coagulation: all bin pairs within each composition group.

    For each ordered pair (ii, kk) with kk ≥ ii in the same group:
        coll_factor = √(8/(3π)) π (a_ii + a_kk)² × v_rel
        rate_from_ii = sym × coll_factor × (ρ_kk / m_kk) × ρ_ii × p_stick  [g cm⁻³ s⁻¹]
        rate_from_kk = sym × coll_factor × (ρ_ii / m_ii) × ρ_kk × p_stick  [g cm⁻³ s⁻¹]

    The destination bin is the one whose mass is closest to m_ii + m_kk,
    capped at the largest bin in the group.
    Active only when T < 10⁴ K.
    Mirrors ``turbulent_all_coagulation_rate`` in RAMSES.
    """
    Tk = state.local_Tk
    if Tk > 1.0e4:
        return 0.0

    sigma, inject_L = _effective_sigma(state)
    if sigma <= 0.0:
        return 0.0

    kmax = 0.0
    rho_gas = state.local_rho
    nH = state.local_nH
    mu = state.local_mu
    model = state.dust_velocity_model

    for ii1, ii2 in _composition_groups(state.dust_bins):
        # Build list of masses for this group
        group_masses = [state.dust_bins[g].mgrain for g in range(ii1, ii2 + 1)]

        for ii in range(ii1, ii2):  # ii goes up to ii2-1 (largest has no partner)
            db_ii = state.dust_bins[ii]
            idx_ii = db_ii.bin_index + state.npah
            rho_ii = y_dust[idx_ii]
            if rho_ii <= db_ii.smallr_dust:
                continue

            for kk in range(ii, ii2 + 1):
                db_kk = state.dust_bins[kk]
                idx_kk = db_kk.bin_index + state.npah
                rho_kk = y_dust[idx_kk]
                if rho_kk <= db_kk.smallr_dust:
                    continue

                v_rel = grain_relative_velocity(
                    model, Tk, rho_gas, nH, sigma, mu, inject_L,
                    db_ii.asize_cm, db_ii.sgrain, db_ii.mgrain,
                    db_kk.asize_cm, db_kk.sgrain, db_kk.mgrain,
                )
                # Use minimum of the two thresholds
                v_coag = min(db_ii.vthresh_coag, db_kk.vthresh_coag)
                p_stick = sticking_probability_from_velocity(v_rel, v_coag)

                coll_factor = (
                    math.sqrt(8.0 / (3.0 * math.pi))
                    * math.pi * (db_ii.asize_cm + db_kk.asize_cm) ** 2
                    * v_rel
                )
                sym = 0.5 if ii == kk else 1.0

                # Loss from ii: number density of kk = rho_kk / m_kk
                rate_from_ii = sym * coll_factor / db_kk.mgrain * rho_kk * rho_ii * p_stick
                # Loss from kk: number density of ii = rho_ii / m_ii
                rate_from_kk = sym * coll_factor / db_ii.mgrain * rho_ii * rho_kk * p_stick

                if rate_from_ii <= 0.0:
                    continue

                kmax = max(kmax, sym * coll_factor / db_kk.mgrain * rho_kk * p_stick)

                # Destination: bin whose mass is closest to m_ii + m_kk
                m_target = db_ii.mgrain + db_kk.mgrain
                best_dest = ii2  # cap at largest bin
                best_diff = abs(group_masses[ii2 - ii1] - m_target)
                for gg in range(ii1, ii2 + 1):
                    diff = abs(group_masses[gg - ii1] - m_target)
                    if diff < best_diff:
                        best_diff = diff
                        best_dest = gg

                dydt_dust[idx_ii] -= rate_from_ii
                if ii != kk:
                    dydt_dust[idx_kk] -= rate_from_kk
                    dydt_dust[best_dest + state.npah] += rate_from_ii + rate_from_kk
                else:
                    dydt_dust[best_dest + state.npah] += rate_from_ii

    return kmax
