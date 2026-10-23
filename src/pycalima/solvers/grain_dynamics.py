"""Grain relative-velocity models and sticking probability.

Implements the functions used by shattering, turbulent coagulation, and
PAH-freezing rate kernels in RAMSES-CALIMA.

Models for grain relative velocity
------------------------------------
``'Ormel2007'``
    Ormel & Cuzzi (2007) – Brownian motion plus turbulence in the Epstein drag
    regime, from their full closed-form expression (Eqs. 16-18), continuous
    across the tightly coupled, intermediate and heavy regimes.

``'Hirashita2019'``
    Simplified Mach-number scaling from Hirashita & Aoyama (2019), Appendix C.
    Deterministic result (mean cos θ = 0, i.e. RMS relative velocity).

Any other string
    Falls back to the turbulent velocity ``v_turb`` directly.
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Physical constants (CGS)
# ---------------------------------------------------------------------------

KB_CGS: float = 1.3806488e-16    # Boltzmann constant  [erg K⁻¹]
MH_CGS: float = 1.6726219e-24    # Proton mass          [g]
PI: float = math.pi
E2_CGS: float = 2.3070779e-19    # e²  [erg cm]  (e = 4.803e-10 esu)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def grain_relative_velocity(
    model: str,
    T: float,
    rho_gas: float,
    nH: float,
    v_turb: float,
    local_mu: float,
    inject_L: float,
    target_a: float,
    target_s: float,
    target_m: float,
    projectile_a: float,
    projectile_s: float,
    projectile_m: float,
    x_ion: float = 0.0,
) -> float:
    """Relative collision velocity of two dust grains [cm s⁻¹].

    Parameters
    ----------
    model :
        ``'Ormel2007'``, ``'Hirashita2019'``, or any string (→ *v_turb*).
    T :
        Gas temperature [K].
    rho_gas :
        Gas mass density [g cm⁻³].
    nH :
        Hydrogen number density [cm⁻³].
    v_turb :
        Turbulent 1-D velocity dispersion [cm s⁻¹].
    local_mu :
        Mean molecular weight [m_H].
    inject_L :
        Turbulence injection (driving) scale [cm].
    target_a, projectile_a :
        Grain radii [cm].
    target_s, projectile_s :
        Grain material (bulk) densities [g cm⁻³].
    target_m, projectile_m :
        Grain masses [g].
    x_ion :
        Ionised fraction n_ion / n_H, for the gas viscosity of ``'Ormel2007'``
        (0: neutral gas).

    Returns
    -------
    float
        Relative velocity magnitude [cm s⁻¹].
    """
    if model == "Ormel2007":
        return _ormel2007(
            T, rho_gas, nH, v_turb, local_mu, inject_L,
            target_a, target_s, target_m,
            projectile_a, projectile_s, projectile_m, x_ion,
        )
    if model == "Hirashita2019":
        return _hirashita2019(
            T, nH, v_turb, local_mu,
            target_a, target_s,
            projectile_a, projectile_s,
        )
    # Fallback: all turbulent kinetic energy goes into relative motion
    return v_turb


def sticking_probability_from_velocity(v_rel: float, v_coag: float) -> float:
    """Smooth sticking probability centred on *v_coag*.

    Uses the Fermi–Dirac-like sigmoid from RAMSES::

        p_stick = 1 / (1 + exp(4 × (v_rel / v_coag − 1)))

    so that ``p_stick → 1`` when ``v_rel ≪ v_coag`` and
    ``p_stick → 0`` when ``v_rel ≫ v_coag``.

    Parameters
    ----------
    v_rel :
        Relative grain velocity [cm s⁻¹].
    v_coag :
        Threshold coagulation velocity [cm s⁻¹].
    """
    if v_coag <= 0.0:
        return 0.0
    x = 4.0 * (v_rel / v_coag - 1.0)
    if x > 50.0:
        return 0.0
    if x < -50.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(x))


# ---------------------------------------------------------------------------
# Ormel & Cuzzi (2007)  — Appendix B / Kawasaki & Machida (2023) formulation
# ---------------------------------------------------------------------------

SIGMA_NEUTRAL = 2.0e-15   # H/H2 kinetic cross-section [cm^2] (as Ormel et al. 2009)
COULOMB_LOG = 20.0        # ln Lambda of the ion-ion collisions (Braginskii 1965)


def _ormel2007(
    T: float,
    rho_gas: float,
    nH: float,
    v_turb: float,
    local_mu: float,
    inject_L: float,
    target_a: float,
    target_s: float,
    target_m: float,
    projectile_a: float,
    projectile_s: float,
    projectile_m: float,
    x_ion: float = 0.0,
) -> float:
    # --- Brownian (thermal) contribution: (8 k T / pi m_red)^1/2 ---
    m_sum = target_m + projectile_m
    m_prod = target_m * projectile_m
    dV_thermal = math.sqrt(8.0 * KB_CGS * T * m_sum / (PI * max(m_prod, 1.0e-100)))

    if v_turb <= 0.0 or inject_L <= 0.0:
        return dV_thermal

    # --- Gas thermal (Maxwellian mean) speed: no adiabatic index in it ---
    v_th = math.sqrt(8.0 * KB_CGS * T / (PI * MH_CGS * max(local_mu, 1.0e-3)))
    v_th = max(v_th, 1.0)

    # --- Kinematic viscosity of the neutral + ionised mixture, each species
    # weighted by its mass fraction: nu = sum_s f_s v_th mfp_s / 3, with
    # neutral collisions at SIGMA_NEUTRAL (neutral-neutral and neutral-ion) and
    # ion-ion Coulomb collisions (Braginskii 1965). Neutral gas: v_th / (3 n sigma);
    # fully ionised: the Braginskii ion viscosity. ---
    x = min(max(x_ion, 0.0), 1.0)
    rc = E2_CGS / (KB_CGS * T)               # Coulomb distance [cm]
    sigma_C = PI * rc ** 2 * COULOMB_LOG
    nu_n = (1.0 - x) / SIGMA_NEUTRAL
    nu_i = x / (x * sigma_C + (1.0 - x) * SIGMA_NEUTRAL)
    nu = v_th * (nu_n + nu_i) / (3.0 * max(nH, 1.0e-30))

    # --- Turbulence timescales: Re = v L / nu ---
    tau_L = inject_L / v_turb                 # injection timescale [s]
    Re = v_turb * inject_L / nu
    Re = max(Re, 1.0)
    tau_eta = tau_L / math.sqrt(Re)           # dissipation timescale [s]

    # --- Epstein stopping times ---
    rho_s = max(rho_gas, 1.0e-30)
    ts_t = target_s * target_a / (rho_s * v_th)
    ts_p = projectile_s * projectile_a / (rho_s * v_th)

    # Normalise: target is the larger grain
    if ts_t < ts_p:
        ts_t, ts_p = ts_p, ts_t

    # --- Stokes numbers ---
    St_t = ts_t / tau_L
    St_p = ts_p / tau_L
    Stmin = tau_eta / tau_L                   # Re^-1/2

    # --- Turbulent relative velocity: the full closed form of Ormel & Cuzzi
    # (2007, Eqs. 16-18), continuous across the tightly coupled, intermediate
    # and heavy regimes (their Eqs. 26-29 are its limits; switching between
    # those leaves jumps of up to x2.4 at t_eta and t_L). The class I / II
    # boundary eddy: t* = y_a t_s with y_a = 1.6 (their Sect. 3.2.1), t* = t_L
    # once t_s >= t_L, and t_eta <= t*_12 <= t_L. ---
    St_star = min(max(1.6 * St_t, Stmin), 1.0) if St_t < 1.0 else 1.0
    sumSt = St_t + St_p
    if sumSt > 0.0:
        dV_I2 = (St_t - St_p) / sumSt * (
            St_t ** 2 / (St_star + St_t) - St_t ** 2 / (1.0 + St_t)
            - St_p ** 2 / (St_star + St_p) + St_p ** 2 / (1.0 + St_p))
    else:
        dV_I2 = 0.0
    dV_II2 = (2.0 * (St_star - Stmin)
              + St_t ** 2 / (St_t + St_star) - St_t ** 2 / (St_t + Stmin)
              + St_p ** 2 / (St_p + St_star) - St_p ** 2 / (St_p + Stmin))
    dV_turb = math.sqrt(3.0 / 2.0) * v_turb * math.sqrt(max(dV_I2 + dV_II2, 0.0))

    return math.sqrt(dV_thermal ** 2 + dV_turb ** 2)


# ---------------------------------------------------------------------------
# Hirashita & Aoyama (2019)  — Appendix C, deterministic (cos θ = 0)
# ---------------------------------------------------------------------------

def _hirashita2019(
    T: float,
    nH: float,
    v_turb: float,
    local_mu: float,
    target_a: float,
    target_s: float,
    projectile_a: float,
    projectile_s: float,
) -> float:
    cs_gas = math.sqrt(5.0 / 3.0 * KB_CGS * T / (MH_CGS * max(local_mu, 1.0)))
    cs_gas = max(cs_gas, 1.0)
    Mach = v_turb / cs_gas
    nH_safe = max(nH, 1.0e-10)
    Mach_fac = Mach ** 1.5
    T_fac = (T / 1.0e4) ** 0.25
    nH_fac = nH_safe ** (-0.25)

    v_t = (
        1.1e5 * Mach_fac
        * math.sqrt(target_a / 1.0e-5)
        * T_fac * nH_fac
        * math.sqrt(target_s / 3.5)
    )
    v_p = (
        1.1e5 * Mach_fac
        * math.sqrt(projectile_a / 1.0e-5)
        * T_fac * nH_fac
        * math.sqrt(projectile_s / 3.5)
    )
    return math.sqrt(v_t ** 2 + v_p ** 2)
