"""Grain relative-velocity models and sticking probability.

Implements the functions used by shattering, turbulent coagulation, and
PAH-freezing rate kernels in RAMSES-CALIMA.

Models for grain relative velocity
------------------------------------
``'Ormel2007'``
    Ormel & Cuzzi (2007), Appendix B – Brownian motion plus turbulence in the
    Epstein drag regime.  Three sub-regimes depending on the stopping-time
    ratio relative to the turbulent dissipation and injection timescales.

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

    Returns
    -------
    float
        Relative velocity magnitude [cm s⁻¹].
    """
    if model == "Ormel2007":
        return _ormel2007(
            T, rho_gas, nH, v_turb, local_mu, inject_L,
            target_a, target_s, target_m,
            projectile_a, projectile_s, projectile_m,
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
) -> float:
    # --- Brownian (thermal) contribution ---
    m_sum = target_m + projectile_m
    m_prod = target_m * projectile_m
    dV_thermal = math.sqrt(8.0 * KB_CGS * T * m_sum / max(m_prod, 1.0e-100))

    if v_turb <= 0.0 or inject_L <= 0.0:
        return dV_thermal

    # --- Gas sound speed and thermal velocity ---
    cs_gas = math.sqrt(5.0 / 3.0 * KB_CGS * T / (MH_CGS * max(local_mu, 1.0)))
    cs_gas = max(cs_gas, 1.0)
    v_th = math.sqrt(8.0 / PI) * cs_gas

    # --- Coulomb mean free path ---
    rc = E2_CGS / (KB_CGS * T)               # Coulomb distance [cm]
    mfp = 1.0 / max(nH * rc ** 2, 1.0e-30)  # [cm]

    # --- Turbulence timescales ---
    tau_L = inject_L / v_turb                 # injection timescale [s]
    Re = 3.0 * v_turb * inject_L / (cs_gas * mfp)
    Re = max(Re, 1.0)
    tau_eta = tau_L / math.sqrt(Re)           # dissipation timescale [s]

    # --- Epstein stopping times ---
    rho_s = max(rho_gas, 1.0e-30)
    v_th_s = max(v_th, 1.0)
    ts_t = target_s * target_a / (rho_s * v_th_s)
    ts_p = projectile_s * projectile_a / (rho_s * v_th_s)

    # Normalise: target is the larger grain
    if ts_t < ts_p:
        ts_t, ts_p = ts_p, ts_t

    # --- Stokes numbers ---
    St_t = ts_t / tau_L
    St_p = ts_p / tau_L
    Stmin = tau_eta / tau_L

    # --- Turbulent relative velocity (three regimes) ---
    if ts_t < tau_eta:
        # Both grains well-coupled to turbulence
        dSt = St_t - St_p  # ≥ 0 since ts_t ≥ ts_p
        sumSt = St_t + St_p
        if sumSt > 0.0:
            frac1 = St_t ** 2 / (St_t + Stmin) - St_p ** 2 / (St_p + Stmin)
            arg = (dSt / sumSt) * max(frac1, 0.0)
            dV_turb = math.sqrt(3.0 / 2.0) * v_turb * math.sqrt(arg)
        else:
            dV_turb = 0.0
    elif ts_t < tau_L:
        # Intermediate regime for the larger grain
        x = St_p / max(St_t, 1.0e-30)
        OC07 = 3.2 - (1.0 + x) + 2.0 / (1.0 + x) * (1.0 / 2.6 + x ** 3 / (1.6 + x))
        dV_turb = math.sqrt(3.0 / 2.0) * v_turb * math.sqrt(max(OC07 * St_t, 0.0))
    else:
        # Large Stokes number
        dV_turb = math.sqrt(3.0 / 2.0) * v_turb * math.sqrt(
            1.0 / (1.0 + St_t) + 1.0 / (1.0 + St_p)
        )

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
