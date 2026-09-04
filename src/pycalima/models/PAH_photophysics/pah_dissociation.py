"""
pah_dissociation.py — PAH photodissociation rate integrators.

All methods return H-loss and H2-loss rates Y_H, Y_H2 [s^-1] as a function of
radiation field strength G0.  Five integration strategies are implemented:

Method A — RRKM branching applied globally to the total absorption rate
    Y_H = R_abs × k_H / (k_H + k_H2 + k_IR)
    where each rate is the temperature-average over f(T).  Neglects the
    energy-dependent competition between dissociation and IR cooling.

Method B (default) — GD89 temperature distribution with RRKM rates
    Y_H = ∫ f(T) × k_H(T) × W_down(T) / (k_H + k_H2 + W_down) dT

    f(T)   : GD89 single-photon energy cascade temperature distribution [K^-1 s^-1]
    k_H    : RRKM Arrhenius rate (Tielens 2005, eq. below) [s^-1]
    W_down : effective cascade rate through one energy bin [s^-1]
             W_down = k_IR × U(T) / ΔU   (GD89 convention)

    RRKM formula used for k_H (and analogously k_H2):
        k(E) = e × (k_B T_e / h) × exp(ΔS/R) × exp(−E_act / k_B T_e)
        T_e  = T_m × (1 − 0.2 × E_act / E)   (Tielens 2005 effective temperature)
        T_m  : canonical temperature, U_QHO(T_m) = E  (from PAHdb vibrational modes)

    Accuracy note: T_m depends on the vibrational mode set.  Our code uses PAHdb
    modes; Andrews (2016) uses B3LYP/6-31G* DFT modes.  Because k_H enters as
    exp(−E_act / k_B T_e), even a 5–10 % difference in mode frequencies changes
    k_H by orders of magnitude.  Consequently Method B agrees with Andrews only
    for C54H18 neutral (fortuitous cancellation); for C24 and C96 discrepancies
    reach 1–2 dex.

Method C — GD89 temperature distribution with Andrews k(E) tables
    Same integral as Method B but with k_H(E), k_H2(E), k_IR(E) supplied as
    lookup tables digitised from Andrews (2016) figures.  Bypasses the mode-set
    sensitivity of Method B.  Limited by the table energy range (8.88–14.5 eV
    for the 5-point H-loss table; use andrews16_Hdissociation_rates_C54H18.csv
    which extends to 40 eV to avoid an artificial k_H = 0 cutoff at high G0).

Method D — Bakes (2001) / Dwek (1986) multi-photon Volterra equation
    Solves the Volterra integral I(U) = R_above(U) + ∫ I(U')/P_IR(U') R_above(U−U') dU'
    for the cumulative energy injection rate, then weights by k_H/P_IR.
    Reduces to the single-photon Bakes model when the stacking index SI ≪ 1.

Method E — direct energy-weighted single-photon branching (Andrews 2016)
    Y_H = ∫ dR/dE × k_H(E) / (k_H + k_H2 + k_IR)(E) dE
    Strictly valid only in the single-photon regime (G0 ≲ 1000 for C54H18).

Public functions
----------------
  compute_total_photon_absorption_rate   — R_abs [photons s^-1] over 6–13.6 eV
  compute_total_photoionisation_rate     — R_ion [s^-1] above IP
  compute_total_dissociation_rate        — Method A component rate [s^-1]
  compute_branching_integrated_rates     — Methods B and C [s^-1]
  compute_andrews_direct_branching       — Method E [s^-1]
  compute_bakes_direct_branching         — Method D single-photon [s^-1]
  compute_bakes_dwek_branching           — Method D Volterra [s^-1]
  compare_dissociation_methods           — run all five over a G0 grid
  print_method_comparison                — tabular printout vs Andrews data
"""

import sys
from pathlib import Path
import os

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

from models.PAH_photophysics.pah_mol_data import (
    load_pah_modes,
    compute_thermal_ir_rate,
    compute_rrkm_dissociation_rate,
    compute_dissociation_rate_from_table,
)
from models.PAH_photophysics.pah_temperature import compute_adaptive_temperature_distribution
from models.PAH_photophysics.pah_charge_utils import ionisation_yield_Jochims1996

_HC_EV  = 1.23984193e-4
_KB_EV  = 8.61733326e-5
_KB_J_K = 1.380649e-23
_H_J_S  = 6.62607015e-34
_R_GAS  = 8.31446261


# ---------------------------------------------------------------------------
# Photon absorption and photoionisation rates
# ---------------------------------------------------------------------------

def compute_total_photon_absorption_rate(radiation_field_func, cross_section_table):
    """
    Total UV photon absorption rate R_abs [photons s^-1] over 6–13.6 eV.
    """
    hc_ev = _HC_EV
    c_cgs = 2.99792458e10
    h_cgs = 6.62607015e-27
    cs_E   = cross_section_table[:, 0]
    cs_sig = cross_section_table[:, 1]

    def get_sigma(E):
        if E <= 0.0 or E > 13.6:
            return 0.0
        return float(np.interp(E, cs_E, cs_sig, left=0.0, right=0.0))

    E_arr = np.linspace(6.0, 13.6, 1000)
    dE    = E_arr[1] - E_arr[0]
    R_abs = 0.0
    for E_phot in E_arr:
        nu  = (E_phot / hc_ev) * c_cgs
        I   = radiation_field_func(nu)
        sig = get_sigma(E_phot)
        if sig <= 0.0:
            continue
        dnu   = (dE / hc_ev) * c_cgs
        flux  = 4.0 * np.pi * I / (h_cgs * nu)
        R_abs += flux * sig * dnu
    return R_abs


def compute_total_photoionisation_rate(radiation_field_func, cross_section_table, IP=6.14):
    """
    Total photoionisation rate R_ion [s^-1] from IP to 13.6 eV.
    Uses the Jochims (1996) ionisation yield.
    """
    hc_ev = _HC_EV
    c_cgs = 2.99792458e10
    h_cgs = 6.62607015e-27
    cs_E   = cross_section_table[:, 0]
    cs_sig = cross_section_table[:, 1]

    def get_sigma(E):
        if E <= 0.0 or E > 13.6:
            return 0.0
        return float(np.interp(E, cs_E, cs_sig, left=0.0, right=0.0))

    E_arr = np.linspace(IP, 13.6, 1000)
    dE    = E_arr[1] - E_arr[0]
    R_ion = 0.0
    for E_phot in E_arr:
        if E_phot < IP:
            continue
        nu    = (E_phot / hc_ev) * c_cgs
        I     = radiation_field_func(nu)
        sig   = get_sigma(E_phot)
        if sig <= 0.0:
            continue
        yield_ = ionisation_yield_Jochims1996(IP, E_phot)
        dnu    = (dE / hc_ev) * c_cgs
        flux   = 4.0 * np.pi * I / (h_cgs * nu)
        R_ion  += flux * sig * yield_ * dnu
    return R_ion


# ---------------------------------------------------------------------------
# Method A helper: temperature-averaged RRKM rate (independent channels)
# ---------------------------------------------------------------------------

def compute_total_dissociation_rate(file_path, t_centers, f_T, E_act_ev, dS_cl_jk,
                                    t_min=15.0, k_table_path=None):
    """
    Method A component: ∫ f(T) × K_dis(T) dT  [s^-1].

    Computes the temperature-averaged dissociation rate for a single channel
    (H-loss or H2-loss).  The three channel rates (k_H, k_H2, k_IR) are
    combined in compare_dissociation_methods to form the Method A branching.

    Parameters
    ----------
    file_path    : path to PAHdb modes file (.dat)
    t_centers    : temperature bin centres [K] from GD89 solver
    f_T          : temperature distribution f(T) [K^-1 s^-1] matching t_centers
    E_act_ev     : activation energy [eV]
    dS_cl_jk     : activation entropy [J K^-1 mol^-1]
    t_min        : minimum temperature for integration [K]
    k_table_path : if given, look up k(U) from this table instead of RRKM formula
    """
    freq_ev, _ = load_pah_modes(file_path)

    num_bins = len(t_centers)
    t_max    = 10**(2 * np.log10(t_centers[-1]) - np.log10(t_centers[-2]))
    t_edges  = np.logspace(np.log10(t_min), np.log10(t_max), num_bins + 1)
    delta_t  = np.diff(t_edges)
    f_disc   = f_T * delta_t

    K_arr = np.zeros(num_bins)
    for j in range(num_bins):
        T_j = t_centers[j]
        if T_j <= 0:
            continue
        x = freq_ev / (_KB_EV * T_j)
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            occ = np.where(x > 50.0, np.exp(-x) / (1.0 - np.exp(-x)), 1.0 / np.expm1(x))
            occ = np.where(np.isnan(occ) | np.isinf(occ), 0.0, occ)
        U_j = np.sum(freq_ev * occ)

        if k_table_path is not None:
            K_arr[j] = compute_dissociation_rate_from_table(k_table_path, U_j)
        else:
            if U_j < E_act_ev:
                continue
            T_e = T_j * (1.0 - 0.2 * E_act_ev / U_j)
            if T_e <= 0:
                continue
            K_arr[j] = (np.exp(1.0)
                        * (_KB_J_K * T_e / _H_J_S)
                        * np.exp(dS_cl_jk / _R_GAS)
                        * np.exp(-E_act_ev / (_KB_EV * T_e)))

    return float(np.sum(f_disc * K_arr))


# ---------------------------------------------------------------------------
# Method B (default): f(T)-integrated RRKM branching rates
# ---------------------------------------------------------------------------

def compute_branching_integrated_rates(file_path, t_centers, f_T,
                                       E_act_H, dS_H, E_act_H2, dS_H2,
                                       t_min=15.0,
                                       k_H_table=None, k_H2_table=None,
                                       k_IR_table=None, u_centers=None):
    """
    Methods B and C: energy-cascade-corrected H-loss and H2-loss rates [s^-1].

    At each temperature bin j the competition between dissociation and IR
    cascade is resolved via the GD89 W_down prescription, then integrated:

        Y_H = ∫ f(T) × [k_H × W_down / (k_H + k_H2 + W_down)] dT

    where  W_down(T) = k_IR(T) × U(T) / ΔU(T)   [s^-1]
    is the rate at which the cascade moves through one energy bin of width ΔU.
    This accounts for the finite time spent at each energy before IR cooling
    drains the molecule below the dissociation threshold.

    Mode B (default, no table arguments)
        k_H, k_H2 from RRKM Arrhenius formula (Tielens 2005):
            k = e × (k_B T_e / h) × exp(ΔS/R) × exp(−E_act / k_B T_e)
            T_e = T × (1 − 0.2 × E_act / U(T))
        k_IR from PAHdb Einstein A coefficients:
            k_IR = Σ_i ε_i A_i ⟨n_i(T)⟩ / U(T)

    Method C (all three table arguments supplied)
        k_H, k_H2, k_IR from Andrews (2016) digitised look-up tables
        (same f(T) from GD89, only the per-energy rates differ).

    Parameters
    ----------
    file_path    : path to PAHdb modes file (.dat) — used for U(T) and k_IR
    t_centers    : temperature bin centres [K]
    f_T          : GD89 temperature distribution [K^-1 s^-1]
    E_act_H      : H-loss activation energy [eV]
    dS_H         : H-loss activation entropy [J K^-1 mol^-1]
    E_act_H2     : H2-loss activation energy [eV]  (set large to suppress channel)
    dS_H2        : H2-loss activation entropy [J K^-1 mol^-1]
    t_min        : lower temperature cutoff for integration [K]
    k_H_table    : path to Andrews k_H(E) table (Method C); None → RRKM
    k_H2_table   : path to Andrews k_H2(E) table (Method C); None → RRKM
    k_IR_table   : path to Andrews k_IR(E) table (Method C); None → PAHdb
    u_centers    : precomputed U_QHO at t_centers [eV]; recomputed if None

    Returns
    -------
    (Y_H, Y_H2) : H-loss and H2-loss rates [s^-1]
    """
    freq_ev, einstein_A = load_pah_modes(file_path)

    num_bins = len(t_centers)
    t_max    = 10**(2 * np.log10(t_centers[-1]) - np.log10(t_centers[-2]))
    t_edges  = np.logspace(np.log10(t_min), np.log10(t_max), num_bins + 1)
    delta_t  = np.diff(t_edges)
    f_disc   = f_T * delta_t

    # Precompute U at all edges and bin centres (vectorised QHO)
    def _u_arr(T_arr):
        result = np.zeros(len(T_arr))
        for idx, T in enumerate(T_arr):
            if T <= 0:
                continue
            x = freq_ev / (_KB_EV * T)
            with np.errstate(over='ignore', under='ignore', invalid='ignore'):
                occ = np.where(x > 50.0, np.exp(-x) / (1.0 - np.exp(-x)), 1.0 / np.expm1(x))
                occ = np.where(np.isnan(occ) | np.isinf(occ), 0.0, occ)
            result[idx] = np.sum(freq_ev * occ)
        return result

    if u_centers is not None:
        u_centers_arr = np.asarray(u_centers, dtype=float)
        u_lo  = np.interp(t_edges[:-1], t_centers, u_centers_arr, left=0.0, right=u_centers_arr[-1])
        u_hi  = np.interp(t_edges[1:],  t_centers, u_centers_arr, left=0.0, right=u_centers_arr[-1])
        delta_u = np.maximum(u_hi - u_lo, 1e-6)
    else:
        u_edges_arr = _u_arr(t_edges)
        delta_u     = np.diff(u_edges_arr)
        u_centers_arr = None

    Y_H_arr  = np.zeros(num_bins)
    Y_H2_arr = np.zeros(num_bins)

    for j in range(num_bins):
        T_j = t_centers[j]
        if T_j <= 15.0:
            continue

        # Internal energy at this bin
        if u_centers_arr is not None:
            U_Tj = float(u_centers_arr[j])
        else:
            x = freq_ev / (_KB_EV * T_j)
            with np.errstate(over='ignore', under='ignore', invalid='ignore'):
                occ = np.where(x > 50.0, np.exp(-x) / (1.0 - np.exp(-x)), 1.0 / np.expm1(x))
                occ = np.where(np.isnan(occ) | np.isinf(occ), 0.0, occ)
            U_Tj = float(np.sum(freq_ev * occ))

        if U_Tj <= 0:
            continue

        # IR cooling rate (inline to avoid re-reading file)
        if k_IR_table is not None:
            k_IR = compute_dissociation_rate_from_table(k_IR_table, U_Tj)
        else:
            x = freq_ev / (_KB_EV * T_j)
            with np.errstate(over='ignore', under='ignore', invalid='ignore'):
                occ = np.where(x > 50.0, np.exp(-x) / (1.0 - np.exp(-x)), 1.0 / np.expm1(x))
                occ = np.where(np.isnan(occ) | np.isinf(occ), 0.0, occ)
            k_IR = float(np.sum(freq_ev * einstein_A * occ) / U_Tj)

        # H-loss rate
        if k_H_table is not None:
            k_H = compute_dissociation_rate_from_table(k_H_table, U_Tj)
        else:
            if U_Tj >= E_act_H:
                T_e = T_j * (1.0 - 0.2 * E_act_H / U_Tj)
                k_H = (np.exp(1.0) * (_KB_J_K * T_e / _H_J_S)
                       * np.exp(dS_H / _R_GAS)
                       * np.exp(-E_act_H / (_KB_EV * T_e))) if T_e > 0 else 0.0
            else:
                k_H = 0.0

        # H2-loss rate
        if k_H2_table is not None:
            k_H2 = compute_dissociation_rate_from_table(k_H2_table, U_Tj)
        else:
            if U_Tj >= E_act_H2:
                T_e2 = T_j * (1.0 - 0.2 * E_act_H2 / U_Tj)
                k_H2 = (np.exp(1.0) * (_KB_J_K * T_e2 / _H_J_S)
                        * np.exp(dS_H2 / _R_GAS)
                        * np.exp(-E_act_H2 / (_KB_EV * T_e2))) if T_e2 > 0 else 0.0
            else:
                k_H2 = 0.0

        # Cascade-corrected branching (GD89 W_down convention)
        du_j     = delta_u[j] if j < len(delta_u) else delta_u[-1]
        W_down_j = (k_IR * U_Tj / du_j) if du_j > 0 else k_IR
        denom    = k_H + k_H2 + W_down_j
        if denom > 0:
            Y_H_arr[j]  = k_H  * W_down_j / denom
            Y_H2_arr[j] = k_H2 * W_down_j / denom

    return float(np.sum(f_disc * Y_H_arr)), float(np.sum(f_disc * Y_H2_arr))


# ---------------------------------------------------------------------------
# Reference methods (require Andrews k(E) tables)
# ---------------------------------------------------------------------------

def compute_andrews_direct_branching(radiation_field_func, cross_section_table,
                                     k_H_table, k_H2_table, k_IR_table,
                                     E_min=6.0, E_max=13.6, n_eval=800):
    """
    Method E: direct energy-weighted single-photon branching (Andrews 2016).

    Assumes every absorbed photon is processed independently (no multi-photon
    heating), so the rate is just the photon absorption rate spectrum weighted
    by the instantaneous branching fraction at that photon energy:

        Y_H = ∫_{E_min}^{E_max} (dR/dE) × k_H(E) / (k_H + k_H2 + k_IR)(E) dE

    where dR/dE = (4π J_ν / hν) × σ_abs(E) × (eV→erg conversion).

    Valid in the single-photon regime (SI ≪ 1), i.e., G0 ≲ 10^3 for C54H18.
    At higher G0 the molecule absorbs a second photon before cooling; Method B/D
    account for this via the temperature distribution or Volterra equation.

    Requires Andrews (2016) k(E) tables for k_H, k_H2, k_IR.
    """
    hc_ev = _HC_EV
    h_cgs = 6.62607015e-27
    c_cgs = 2.99792458e10
    eV_erg = 1.602176634e-12

    E_eval   = np.linspace(E_min, E_max, n_eval)
    dE       = E_eval[1] - E_eval[0]
    sig_eval = np.interp(E_eval, cross_section_table[:, 0], cross_section_table[:, 1],
                         left=0.0, right=0.0)
    nu_eval  = (E_eval / hc_ev) * c_cgs
    J_nu     = np.array([radiation_field_func(nu) for nu in nu_eval])
    phot_flux = 4.0 * np.pi * J_nu / (h_cgs * nu_eval)
    dR_dE    = phot_flux * sig_eval * (eV_erg / h_cgs)

    Y_H_arr  = np.zeros(n_eval)
    Y_H2_arr = np.zeros(n_eval)
    for i, E in enumerate(E_eval):
        k_H  = compute_dissociation_rate_from_table(k_H_table,  E)
        k_H2 = compute_dissociation_rate_from_table(k_H2_table, E)
        k_IR = compute_dissociation_rate_from_table(k_IR_table, E)
        denom = k_H + k_H2 + k_IR
        if denom > 0:
            Y_H_arr[i]  = k_H  / denom
            Y_H2_arr[i] = k_H2 / denom

    return float(np.sum(dR_dE * Y_H_arr) * dE), float(np.sum(dR_dE * Y_H2_arr) * dE)


def compute_bakes_direct_branching(radiation_field_func, cross_section_table,
                                   k_H_table, k_H2_table, k_IR_table,
                                   E_min=6.0, E_max=13.6, n_eval=800):
    """
    Method D (single-photon limit): Bakes (2001) / Andrews (2016) microcanonical model.

    Integrates over internal energy U rather than photon energy.  For a molecule
    at energy U that is cooling by IR emission at rate P_IR(U) [eV s^-1], the
    time spent near U is ~1/P_IR(U), and R_above(U) [s^-1] is the rate at which
    photons inject energy ≥ U.  The single-photon H-loss rate is then:

        Y_H = ∫ [k_H(U) / P_IR(U)] × R_above(U) dU
        R_above(U) = ∫_{U}^∞ (dR/dE) dE,   P_IR(U) = k_IR(U) × U

    This is Method D without the Dwek (1986) multi-photon correction.
    Use compute_bakes_dwek_branching when multi-photon heating matters.
    """
    hc_ev  = _HC_EV
    h_cgs  = 6.62607015e-27
    c_cgs  = 2.99792458e10
    eV_erg = 1.602176634e-12

    E_eval   = np.linspace(E_min, E_max, n_eval)
    dE       = E_eval[1] - E_eval[0]
    sig_eval = np.interp(E_eval, cross_section_table[:, 0], cross_section_table[:, 1],
                         left=0.0, right=0.0)
    nu_eval  = (E_eval / hc_ev) * c_cgs
    J_nu     = np.array([radiation_field_func(nu) for nu in nu_eval])
    phot_flux = 4.0 * np.pi * J_nu / (h_cgs * nu_eval)
    dR_dE    = phot_flux * sig_eval * (eV_erg / h_cgs)
    R_above  = np.cumsum(dR_dE[::-1])[::-1] * dE

    k_H_arr  = np.array([compute_dissociation_rate_from_table(k_H_table,  U) for U in E_eval])
    k_H2_arr = np.array([compute_dissociation_rate_from_table(k_H2_table, U) for U in E_eval])
    K_th_arr = np.array([compute_dissociation_rate_from_table(k_IR_table,  U) for U in E_eval])
    P_IR_arr = K_th_arr * E_eval

    with np.errstate(divide='ignore', invalid='ignore'):
        intg_H  = np.where(P_IR_arr > 0, k_H_arr  / P_IR_arr * R_above, 0.0)
        intg_H2 = np.where(P_IR_arr > 0, k_H2_arr / P_IR_arr * R_above, 0.0)

    return float(np.sum(intg_H) * dE), float(np.sum(intg_H2) * dE)


def compute_bakes_dwek_branching(radiation_field_func, cross_section_table,
                                 k_H_table, k_H2_table, k_IR_table,
                                 E_min=6.0, E_max=13.6, n_eval=800):
    """
    Method D + Dwek (1986) multi-photon correction via Volterra integral equation.

    Extends compute_bakes_direct_branching to multi-photon heating by replacing
    R_above(U) with the total energy injection rate I_total(U) that accounts for
    the probability that a molecule at energy U arrived there via multiple photons.
    I_total satisfies the Volterra equation of the second kind:

        I_total(U) = R_above(U)
                     + ∫_0^U [I_total(U') / P_IR(U')] × R_above(U − U') dU'

    The convolution term represents molecules that first absorbed a photon
    bringing them to U', cascaded to U' by IR emission, and then absorbed another
    photon of energy U − U'.  Solved numerically by forward substitution.

    At low G0 (SI ≪ 1), I_total ≈ R_above and the result reduces to
    compute_bakes_direct_branching.  At high G0, I_total > R_above, boosting
    the dissociation rate.
    """
    hc_ev  = _HC_EV
    h_cgs  = 6.62607015e-27
    c_cgs  = 2.99792458e10
    eV_erg = 1.602176634e-12

    E_eval   = np.linspace(E_min, E_max, n_eval)
    dE       = E_eval[1] - E_eval[0]
    sig_eval = np.interp(E_eval, cross_section_table[:, 0], cross_section_table[:, 1],
                         left=0.0, right=0.0)
    nu_eval  = (E_eval / hc_ev) * c_cgs
    J_nu     = np.array([radiation_field_func(nu) for nu in nu_eval])
    phot_flux = 4.0 * np.pi * J_nu / (h_cgs * nu_eval)
    dR_dE    = phot_flux * sig_eval * (eV_erg / h_cgs)
    R_above  = np.cumsum(dR_dE[::-1])[::-1] * dE

    K_th_arr = np.array([compute_dissociation_rate_from_table(k_IR_table, U) for U in E_eval])
    P_IR_arr = np.maximum(K_th_arr * E_eval, 1e-40)
    k_H_arr  = np.array([compute_dissociation_rate_from_table(k_H_table,  U) for U in E_eval])
    k_H2_arr = np.array([compute_dissociation_rate_from_table(k_H2_table, U) for U in E_eval])

    R_abs_total  = float(R_above[0])
    N_min        = int(round(E_min / dE))
    R_above_conv = np.full(n_eval, R_abs_total)
    for m in range(N_min, n_eval):
        idx = m - N_min
        R_above_conv[m] = R_above[idx] if idx < n_eval else 0.0

    I_total = np.zeros(n_eval)
    for k in range(n_eval):
        conv = 0.0
        if k > 0:
            conv = np.dot(I_total[:k] / P_IR_arr[:k], R_above_conv[k:0:-1]) * dE
        I_total[k] = R_above[k] + conv

    with np.errstate(divide='ignore', invalid='ignore'):
        intg_H  = np.where(P_IR_arr > 0, k_H_arr  / P_IR_arr * I_total, 0.0)
        intg_H2 = np.where(P_IR_arr > 0, k_H2_arr / P_IR_arr * I_total, 0.0)

    return float(np.sum(intg_H) * dE), float(np.sum(intg_H2) * dE)


# ---------------------------------------------------------------------------
# Multi-method comparison utilities
# ---------------------------------------------------------------------------

def compare_dissociation_methods(
    pah_file,
    cross_section_table,
    field_factory,
    g0_grid,
    E_act_H=4.6,
    dS_H=44.8,
    E_act_H2=3.52,
    dS_H2=-53.1,
    andrews_H_table=None,
    andrews_H2_table=None,
    andrews_IR_table=None,
    num_bins=150,
    t_min=1.0,
):
    """
    Run all five dissociation methods for each G0 in g0_grid.

    Parameters
    ----------
    field_factory : callable
        A function ``field_factory(g0) -> callable`` where the returned callable
        takes frequency ``nu`` (Hz) and returns mean intensity I_nu
        in erg cm^-2 s^-1 Hz^-1 sr^-1.
    Methods C/D/E are only computed when all three Andrews table paths are provided;
    otherwise their arrays are returned as zeros.

    Returns
    -------
    dict with keys:
      "g0", "R_abs",
      "A_H", "A_H2",  (Method A: RRKM scaled by R_abs)
      "B_H", "B_H2",  (Method B: GD89 f(T) + RRKM — default)
      "C_H", "C_H2",  (Method C: GD89 f(T) + Andrews k(E) tables)
      "D_H", "D_H2",  (Method D: Bakes direct / Volterra)
      "E_H", "E_H2",  (Method E: direct energy-weighted Andrews)
    """
    n = len(g0_grid)
    out = {k: np.zeros(n) for k in
           ("g0", "R_abs",
            "A_H", "A_H2", "B_H", "B_H2",
            "C_H", "C_H2", "D_H", "D_H2", "E_H", "E_H2")}
    out["g0"] = np.asarray(g0_grid, dtype=float)

    have_tables = (andrews_H_table is not None
                   and andrews_H2_table is not None
                   and andrews_IR_table is not None)

    for i, g0 in enumerate(g0_grid):
        field = field_factory(g0)

        T_grid, f_prof = compute_adaptive_temperature_distribution(
            file_path=pah_file,
            radiation_field_func=field,
            cross_section_table=cross_section_table,
            t_min=t_min, num_bins=num_bins,
        )

        R_abs = compute_total_photon_absorption_rate(field, cross_section_table)
        out["R_abs"][i] = R_abs

        # Method A
        k_H_A  = compute_total_dissociation_rate(pah_file, T_grid, f_prof,
                                                  E_act_H, dS_H, t_min=t_min)
        k_H2_A = compute_total_dissociation_rate(pah_file, T_grid, f_prof,
                                                  E_act_H2, dS_H2, t_min=t_min)
        from models.PAH_photophysics.pah_temperature import compute_total_time_averaged_ir_rate
        k_IR_A = compute_total_time_averaged_ir_rate(pah_file, T_grid, f_prof, t_min=t_min)
        denom_A = k_H_A + k_H2_A + k_IR_A
        if denom_A > 0:
            out["A_H"][i]  = R_abs * k_H_A  / denom_A
            out["A_H2"][i] = R_abs * k_H2_A / denom_A

        # Method B
        Y_H, Y_H2 = compute_branching_integrated_rates(
            pah_file, T_grid, f_prof,
            E_act_H, dS_H, E_act_H2, dS_H2, t_min=t_min,
        )
        out["B_H"][i]  = Y_H
        out["B_H2"][i] = Y_H2

        if have_tables:
            # Method C
            Y_H_C, Y_H2_C = compute_branching_integrated_rates(
                pah_file, T_grid, f_prof,
                E_act_H, dS_H, E_act_H2, dS_H2, t_min=t_min,
                k_H_table=andrews_H_table,
                k_H2_table=andrews_H2_table,
                k_IR_table=andrews_IR_table,
            )
            out["C_H"][i]  = Y_H_C
            out["C_H2"][i] = Y_H2_C

            # Method D (Dwek/Volterra)
            Y_H_D, Y_H2_D = compute_bakes_dwek_branching(
                field, cross_section_table,
                andrews_H_table, andrews_H2_table, andrews_IR_table,
            )
            out["D_H"][i]  = Y_H_D
            out["D_H2"][i] = Y_H2_D

            # Method E
            Y_H_E, Y_H2_E = compute_andrews_direct_branching(
                field, cross_section_table,
                andrews_H_table, andrews_H2_table, andrews_IR_table,
            )
            out["E_H"][i]  = Y_H_E
            out["E_H2"][i] = Y_H2_E

    return out


def print_method_comparison(results, andrews_H_csv=None, andrews_H2_csv=None):
    """
    Print tabular comparison of method rates vs Andrews (2016) digitised data.

    Parameters
    ----------
    results : dict   — output of compare_dissociation_methods
    andrews_H_csv  : path to Andrews G0-dependent H-loss CSV  [G0, rate]
    andrews_H2_csv : path to Andrews G0-dependent H2-loss CSV [G0, rate]
    """
    g0 = results["g0"]
    lg = np.log10(g0)

    H_ref = H2_ref = None
    if andrews_H_csv is not None:
        dat     = np.loadtxt(andrews_H_csv,  delimiter=',')
        H_ref   = 10**np.interp(lg, np.log10(dat[:,0]),  np.log10(dat[:,1]),
                                left=np.nan, right=np.nan)
    if andrews_H2_csv is not None:
        dat2    = np.loadtxt(andrews_H2_csv, delimiter=',')
        H2_ref  = 10**np.interp(lg, np.log10(dat2[:,0]), np.log10(dat2[:,1]),
                                 left=np.nan, right=np.nan)

    for meth, key_H, key_H2 in [
        ("A (RRKM scaled)",      "A_H",  "A_H2"),
        ("B (GD89 + RRKM)",      "B_H",  "B_H2"),
        ("C (GD89 + Andrews k)", "C_H",  "C_H2"),
        ("D (Bakes/Dwek)",       "D_H",  "D_H2"),
        ("E (direct energy)",    "E_H",  "E_H2"),
    ]:
        rates_H  = results[key_H]
        rates_H2 = results[key_H2]
        if np.all(rates_H == 0) and np.all(rates_H2 == 0):
            continue

        print(f"\n--- Method {meth}: H-loss ---")
        print(f"{'G0':>10}  {'Model':>12}  {'Andrews':>12}  {'Ratio':>8}")
        for i in range(len(g0)):
            ref = H_ref[i] if H_ref is not None else np.nan
            if rates_H[i] > 0 and (H_ref is None or (not np.isnan(ref) and ref > 0)):
                ratio = rates_H[i] / ref if (not np.isnan(ref) and ref > 0) else np.nan
                ref_s = f"{ref:12.3e}" if not np.isnan(ref) else f"{'N/A':>12}"
                rat_s = f"{ratio:8.3f}" if not np.isnan(ratio) else f"{'N/A':>8}"
                print(f"{g0[i]:10.1f}  {rates_H[i]:12.3e}  {ref_s}  {rat_s}")

        print(f"\n--- Method {meth}: H2-loss ---")
        print(f"{'G0':>10}  {'Model':>12}  {'Andrews':>12}  {'Ratio':>8}")
        for i in range(len(g0)):
            ref2 = H2_ref[i] if H2_ref is not None else np.nan
            if rates_H2[i] > 0 and (H2_ref is None or (not np.isnan(ref2) and ref2 > 0)):
                ratio2 = rates_H2[i] / ref2 if (not np.isnan(ref2) and ref2 > 0) else np.nan
                ref2_s = f"{ref2:12.3e}" if not np.isnan(ref2) else f"{'N/A':>12}"
                rat2_s = f"{ratio2:8.3f}" if not np.isnan(ratio2) else f"{'N/A':>8}"
                print(f"{g0[i]:10.1f}  {rates_H2[i]:12.3e}  {ref2_s}  {rat2_s}")
