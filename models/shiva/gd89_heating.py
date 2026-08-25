"""
gd89_heating.py — GD89 / Pavlyuchenkov+2012 stochastic heating distribution.

Computes the steady-state vibrational temperature probability distribution
P(T) for a PAH using the Guhathakurta & Draine (1989) log-space forward
recursion.  The approach follows the simplified version described in
Pavlyuchenkov et al. (2012) and used in Murga et al. (2019, 2020).

Differences from the existing ``pah_temperature.compute_gd89_temperature_distribution``:
- U(T) and dU/dT from the DL01 two-component model (no PAHdb files needed).
- C_abs from the DL07 analytical parameterisation.
- IR cooling P_IR from the DL01 characteristic frequencies plus
  parameterised Einstein A coefficients (see below).
- Self-contained: only needs Nc and an ISRF callable.

IR cooling model
----------------
The downward transition rate is:
    W_down(j → j−1) = P_IR(T_j) / ΔU_j

where P_IR(T) is the total IR emission power.  We use the DL01 two-component
model for the phonon occupation numbers, scaled by effective Einstein A
coefficients.  The DL01 model for graphitic material covers two groups
(bending at θ₁=863 K and C-C stretching at θ₂=2504 K).  For PAHs, the
C-H stretching modes at ~3300 cm⁻¹ (θ₃≈4750 K) dominate the cooling at
T > 500 K.  We fold all high-frequency contributions into an effective A₂:

    A₁ = 1.0 s⁻¹  (low-freq bending,   θ₁ = 863 K)
    A₂ = 30.0 s⁻¹ (C-C + C-H modes effective, θ₂ = 2504 K)

This gives τ_cool ~ 40–100 ms for Nc=24–96 at T~1500 K, consistent with
Allain et al. (1996) and GD89 Fig. 3 for small PAHs.

References
----------
Guhathakurta, P. & Draine, B.T. 1989, ApJ, 345, 230 (GD89)
Pavlyuchenkov, Ya.N. et al. 2012, Astron. Rep. 56, 476
Murga, M.S. et al. 2019, MNRAS 487, 3004
Murga, M.S. et al. 2020, A&A 644, A89
Draine, B.T. & Li, A. 2001, ApJ, 551, 807 (DL01)
Draine, B.T. & Li, A. 2007, ApJ, 657, 810 (DL07)
"""

import numpy as np

from .dl01_internal_energy import U_dl01, _DL01_MODES, k_B_erg, eV_to_erg
from .dl07_crosssections import C_abs_dl07

# ── IR cooling Einstein A coefficients per DL01 mode group ────────────────
# Calibrated to reproduce τ_cool ~ 50 ms for C24 at T ~ 1000 K
_A_MODES = [1.0, 30.0]   # s⁻¹ for groups 1 and 2 (group 2 includes effective C-H contribution)

# Speed of light and Planck's constant
_c_cgs  = 2.99792458e10   # cm s⁻¹
_h_cgs  = 6.62607015e-27  # erg s


def ir_cooling_rate_dl01(T, Nc, A1=1.0, A2=30.0):
    """
    Total IR emission power P_IR(T, Nc) [erg s⁻¹].

    P_IR = Nc × Σᵢ nᵢ × Eᵢ × Aᵢ × f_occ(Eᵢ, T)

    where f_occ = 1 / (exp(θᵢ/T) − 1) is the Bose–Einstein occupation.

    Parameters
    ----------
    T : float
        Vibrational temperature [K].
    Nc : int or float
        Number of carbon atoms.
    A1, A2 : float
        Einstein A coefficients [s⁻¹] for DL01 mode groups 1 and 2.

    Returns
    -------
    P_IR : float  [erg s⁻¹]
    """
    A_list = [A1, A2]
    P = 0.0
    for mode, A_i in zip(_DL01_MODES, A_list):
        n_i, theta_i = mode['n'], mode['theta']
        E_i = k_B_erg * theta_i       # mode energy [erg]
        x   = theta_i / max(T, 2.73)
        if x > 500.0:
            f_occ = np.exp(-x)         # very low T, occupation → 0
        else:
            f_occ = 1.0 / np.expm1(x)
        P += Nc * n_i * E_i * A_i * f_occ
    return P


def compute_PT_shiva(Nc, Z, u_E_fn, G0=1.0,
                     N_T=150, T_min=2.73, T_max_eV=27.2,
                     E_phot_min=0.1, E_phot_max=13.6):
    """
    Steady-state temperature distribution P(T) via the GD89 forward recursion.

    Follows the approach of Pavlyuchenkov+2012 / Murga+2019/2020 using:
      - DL01 two-component U(T) for the energy grid
      - DL07 C_abs(E) for upward (photon absorption) transition rates
      - DL01-calibrated P_IR(T) for downward (IR cooling) transition rates

    The recursion (log-spaced temperature bins) is:

        log P[f] = log Σ_{j<f} (P[j] × Σ_k W_up[j,k≥f]) − log W_down[f]

    which avoids floating-point underflow by working in log space.

    Parameters
    ----------
    Nc : int or float
        Number of carbon atoms.
    Z : int
        Charge state of the PAH (used for DL07 C_abs).
    u_E_fn : callable
        u_E_fn(E_eV) → u_E [erg cm⁻³ eV⁻¹] for the unit-G0 radiation field.
    G0 : float
        Habing field scaling.
    N_T : int
        Number of temperature bins.
    T_min : float
        Minimum temperature [K].
    T_max_eV : float
        Sets the maximum temperature via U(T_max) = T_max_eV × eV_erg.
        Default 27.2 eV corresponds to two Lyman-limit photons.
    E_phot_min, E_phot_max : float
        Photon energy integration range [eV].

    Returns
    -------
    T_centers : ndarray, shape (N_T,)
        Bin centre temperatures [K].
    P_T : ndarray, shape (N_T,)
        Normalised probability density [K⁻¹] such that ∫ P(T) dT = 1.
    """
    # ── Energy grid from DL01 ─────────────────────────────────────────────
    # T_max from U(T_max) = T_max_eV × eV_erg
    from scipy.optimize import brentq
    U_target_max = T_max_eV * eV_to_erg
    try:
        t_max = brentq(lambda T: U_dl01(T, Nc) - U_target_max,
                       100.0, 20000.0, xtol=1.0)
    except ValueError:
        t_max = 6000.0   # fallback if root not in bracket

    t_edges   = np.logspace(np.log10(T_min), np.log10(t_max), N_T + 1)
    t_centers = 0.5 * (t_edges[:-1] + t_edges[1:])
    delta_t   = np.diff(t_edges)
    u_edges   = np.array([U_dl01(t, Nc) for t in t_edges])
    u_centers = np.array([U_dl01(t, Nc) for t in t_centers])

    # ── Downward rates W_down(j → j−1) [s⁻¹] via DL01 P_IR ──────────────
    # W_down[j] = P_IR(T_j) / ΔU_j  [erg s⁻¹ / erg = s⁻¹]
    # At very low T (T ≪ θ₁=863 K), both P_IR and ΔU are exponentially small
    # but their ratio equals the effective Einstein A coefficient analytically
    # (P_IR = A_eff × U(T); ΔU ≈ U(T_j) for log-spaced grid at low T).
    # We compute the ratio directly.  If bw underflows to 0 we use the
    # low-T limiting value A₁ = _A_MODES[0].
    W_down = np.zeros(N_T)
    for j in range(1, N_T):
        P_IR = ir_cooling_rate_dl01(t_centers[j], Nc)
        bw   = u_centers[j] - u_centers[j - 1]
        if bw > 0.0:
            W_down[j] = P_IR / bw
        else:
            # Low-T limit: bw underflows but ratio P_IR/bw → A_eff
            W_down[j] = _A_MODES[0]   # s⁻¹

    # ── Upward rates W_up[j, k] (absorption from bin j to bin k > j) [s⁻¹]
    # Photon energy needed to jump from bin j to bin k: ΔE_jk = u_edges[k] − U_j
    W_up = np.zeros((N_T, N_T))
    E_grid_phot = np.linspace(E_phot_min, E_phot_max, 400)
    dE_phot = E_grid_phot[1] - E_grid_phot[0]

    # Precompute C_abs × F_phot (photon flux density) on the photon energy grid
    F_phot_arr = np.zeros(len(E_grid_phot))   # [s⁻¹ eV⁻¹] per grain = C_abs × c × u_E / (E × eV_erg)
    for i_E, E in enumerate(E_grid_phot):
        u_ph = G0 * u_E_fn(E)               # erg cm⁻³ eV⁻¹
        C    = C_abs_dl07(E, Nc, Z=Z)       # cm²
        F_phot_arr[i_E] = C * _c_cgs * u_ph / (E * eV_to_erg)   # s⁻¹ eV⁻¹

    # For each source bin j, find destination bins k reached by each photon E
    for j in range(N_T):
        U_j = u_centers[j]
        # Energy of the PAH after absorbing photon E: U_j + E × eV_erg
        for i_E, E in enumerate(E_grid_phot):
            dU = E * eV_to_erg   # energy deposited [erg]
            U_after = U_j + dU
            if U_after >= u_edges[-1]:
                continue
            # Find which upper bin k receives this population
            k = np.searchsorted(u_edges, U_after) - 1
            if 0 <= k < N_T and k > j:
                W_up[j, k] += F_phot_arr[i_E] * dE_phot

    # ── GD89 log-space forward recursion ─────────────────────────────────
    # Stable recursion in log space to avoid underflow:
    #   log f[i] = log(Σ_{j<i} f[j] × R[j→i]) − log W_down[i]
    log_f    = np.full(N_T, -np.inf)
    log_f[0] = 0.0   # seed the lowest bin

    # Row sums W_up_cumsum[j, i] = Σ_{k≥i} W_up[j, k]:
    # population in bin j that absorbs a photon and jumps to any k ≥ i
    # will cascade through bin i, so contributes to P[i].
    W_up_from_j_to_ge_i = np.cumsum(W_up[:, ::-1], axis=1)[:, ::-1]  # shape (N_T, N_T)

    for i in range(1, N_T):
        log_terms = []
        for j in range(i):
            # Rate at which population in bin j feeds into bin i (via any k ≥ i)
            rate = W_up_from_j_to_ge_i[j, i]
            if rate > 0.0 and np.isfinite(log_f[j]):
                log_terms.append(log_f[j] + np.log(rate))
        if not log_terms:
            continue
        mx       = np.max(log_terms)
        log_sum  = mx + np.log(np.sum(np.exp(np.array(log_terms) - mx)))
        W_d      = W_down[i]
        if W_d > 0.0:
            log_f[i] = log_sum - np.log(W_d)

    # ── Normalise to probability density ─────────────────────────────────
    finite_mask = np.isfinite(log_f)
    if not np.any(finite_mask):
        # Fallback: delta-function at T_min (essentially no heating)
        P_T = np.zeros(N_T)
        P_T[0] = 1.0 / delta_t[0]
        return t_centers, P_T, delta_t

    mx_lf  = np.max(log_f[finite_mask])
    f_disc = np.exp(log_f - mx_lf)
    f_disc[~finite_mask] = 0.0

    # The recursion computes discrete bin occupations f[i] (probability of
    # being in bin i, NOT a density).  Normalise with Σ f so that
    # Σ P_T[i] × delta_t[i] = 1:
    #   p_i = f[i] / Σ f  (bin probability)
    #   P_T[i] = p_i / delta_t[i]  (probability density [K⁻¹])
    norm = np.sum(f_disc)
    if norm == 0.0:
        norm = 1.0
    P_T = f_disc / norm / delta_t   # probability density [K⁻¹]

    return t_centers, P_T, delta_t
