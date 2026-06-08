"""Rate kernels for each dust and PAH chemistry process.

Each function has the signature::

    rate_fn(state, y_gas, y_dust, dydt_gas, dydt_dust) -> kmax

where

* ``state``       — :class:`~solvers.chemistry_state.DustChemistryState`
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
    """Grain growth by accretion of gas-phase metals.

    Implements the LeBourlot et al. (2012) prescription used in RAMSES:

    .. math::

        \\text{rate} \\; [\\text{s}^{-1}] =
            k_0^\\text{acc} \\,
            \\frac{\\sqrt{T_k}}{1 + 10^{-4}\\, T_k^{1.5}}
            \\, \\frac{y_\\text{gas}[e]}{f_e \\sqrt{m_e}}

    where the rate is limited by the element with the smallest
    pseudo-density ratio.

    Modified variables
    ------------------
    dydt_dust[idx]   += rate × y_dust[idx]
    dydt_gas[el_idx] -= rate × y_dust[idx] × el_mfrac[e]
    """
    Tk = state.local_Tk
    prefactor = math.sqrt(Tk) / (1.0 + 1.0e-4 * Tk ** 1.5)
    kmax = 0.0

    for db in state.dust_bins:
        idx = db.bin_index + state.npah

        if not db.el_indices:
            continue

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

        rate = db.k0_acc * prefactor * limit_rate  # [s⁻¹]

        if rate <= 0.0:
            continue

        kmax = max(kmax, rate)
        rate_rho = rate * y_dust[idx]  # [g cm⁻³ s⁻¹]

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

    The fractional rate  ε = |da/dt| / a  [s⁻¹] is read from the
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
        σ_coll = √(8/(3π)) π (a_ii + a_kk)² / m_ii
        rate1   = σ_coll × v_rel × ρ_kk × p_stick   [s⁻¹]  (loss from ii)
        rate2   = rate1 × ρ_ii   [g cm⁻³ s⁻¹]

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

                sigma_coll = (
                    math.sqrt(8.0 / (3.0 * math.pi))
                    * math.pi * (db_ii.asize_cm + db_kk.asize_cm) ** 2
                    / db_ii.mgrain
                )
                sym = 0.5 if ii == kk else 1.0
                rate1 = sym * sigma_coll * v_rel * rho_kk * p_stick  # [s⁻¹]
                if rate1 <= 0.0:
                    continue

                kmax = max(kmax, rate1)
                rate2 = rate1 * rho_ii  # [g cm⁻³ s⁻¹]

                # Destination: bin whose mass is closest to m_ii + m_kk
                m_target = db_ii.mgrain + db_kk.mgrain
                best_dest = ii2  # cap at largest bin
                best_diff = abs(group_masses[ii2 - ii1] - m_target)
                for gg in range(ii1, ii2 + 1):
                    diff = abs(group_masses[gg - ii1] - m_target)
                    if diff < best_diff:
                        best_diff = diff
                        best_dest = gg

                dydt_dust[idx_ii] -= rate2
                if ii != kk:
                    dydt_dust[idx_kk] -= rate2  # symmetric loss
                dydt_dust[best_dest + state.npah] += (2.0 * rate2 if ii != kk else rate2)

    return kmax
