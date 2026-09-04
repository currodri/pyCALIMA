"""
pah_temperature.py — PAH vibrational temperature distribution solvers.

Provides:
  - compute_gd89_temperature_distribution    (GD89 stable recursion)
  - compute_adaptive_temperature_distribution (GD89 + multi-photon switch)
  - compute_bakes_temperature_distribution   (Bakes/Dwek Poisson model, Tielens E-T table)
  - compute_andrews_temperature_distribution (Andrews/Bakes G̃(T), fully QHO-consistent)
  - compute_total_time_averaged_ir_rate
  - helper utilities: get_absorption_cross_section, compute_base_g0,
                      mathis83_to_gd89_interface
"""

from pathlib import Path
import os


import numpy as np
from scipy.optimize import root_scalar
from scipy.linalg import null_space
from scipy.integrate import quad

from pycalima.models.tools.radiation_fields import Mathis83_radiation_field
from pycalima.models.PAH_radiation.pah_oppacity import pah_efficiencies
from pycalima.models.PAH_photophysics.pah_mol_data import load_pah_modes

_THIS_DIR       = os.path.dirname(os.path.abspath(__file__))
_CALIMA_ROOT    = os.path.abspath(os.path.join(_THIS_DIR, '..', '..'))
PAH_OPTICALS_DIR = os.path.join(_CALIMA_ROOT, 'optical_props', 'li_draine_2001')
pahneu_filepath  = os.path.join(PAH_OPTICALS_DIR, 'PAHneu_30')
pahion_filepath   = os.path.join(PAH_OPTICALS_DIR, 'PAHion_30')

_HC_EV = 1.23984193e-4
_KB_EV = 8.61733326e-5


def _qho_energy(freq_ev, T):
    x = freq_ev / (_KB_EV * T)
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        occ = np.where(x > 50.0, np.exp(-x) / (1.0 - np.exp(-x)), 1.0 / np.expm1(x))
        occ = np.where(np.isnan(occ) | np.isinf(occ), 0.0, occ)
    return np.sum(freq_ev * occ)


def _qho_cv(freq_ev, T):
    x = freq_ev / (_KB_EV * T)
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        val = np.where(x > 50.0,
                       (x**2) * np.exp(-x) / (1.0 - np.exp(-x))**2,
                       (x / np.expm1(x))**2 * np.exp(x))
        val = np.where(np.isnan(val) | np.isinf(val), 0.0, val)
    return np.sum(_KB_EV * val)


# ---------------------------------------------------------------------------
# Radiation field helpers
# ---------------------------------------------------------------------------

def mathis83_to_gd89_interface(nu, target_G0=1.0, base_G0=1.0):
    """
    Convert Mathis83 radiation field to I_nu [erg cm^-2 s^-1 Hz^-1 sr^-1].
    """
    if nu <= 0.0:
        return 0.0
    h_SI  = 6.62607015e-34
    eV2J  = 1.602176634e-19
    c_cgs = 2.99792458e10
    E     = (h_SI * nu) / eV2J
    if E > 13.6:
        return 0.0
    u_E  = Mathis83_radiation_field(E) * (target_G0 / base_G0)
    u_nu = u_E * (h_SI / eV2J)
    return float((c_cgs / (4.0 * np.pi)) * u_nu)


def get_absorption_cross_section(Z, a0):
    """
    Return (wavelength_cm, C_abs_cm2) for a neutral (Z=0) or ionised PAH
    of radius a0 [cm], interpolated from the Li & Draine (2001) tables.
    """
    if Z == 0:
        nwav, data, columns, name = pah_efficiencies(pahneu_filepath)
    else:
        nwav, data, columns, name = pah_efficiencies(pahion_filepath)

    a0_micron      = a0 * 1e4
    float_to_str   = {float(k): k for k in data.keys()}
    keys_sorted    = sorted(float_to_str.keys())
    a              = np.array(keys_sorted)

    C_abs = np.zeros(nwav)
    for i in range(nwav):
        Q_abs = np.array([data[float_to_str[k]][i, columns.index('Q_abs')]
                          for k in keys_sorted])
        Q_abs_interp = 10.0**np.interp(np.log10(a0_micron),
                                        np.log10(a), np.log10(Q_abs))
        C_abs[i] = Q_abs_interp * np.pi * a0**2

    w = data[list(data.keys())[0]][:, columns.index('w(micron)')]
    return w * 1e-4, C_abs


def compute_base_g0(radiation_field_func):
    """
    Integrate the radiation field energy density over 6–13.6 eV and
    convert to the Habing G0 normalisation.
    """
    c_cgs         = 2.99792458e10
    habing_norm   = 1.6e-3
    integrated, _ = quad(radiation_field_func, 6.0, 13.6)
    return c_cgs * integrated / habing_norm


# ---------------------------------------------------------------------------
# GD89 single-photon temperature distribution
# ---------------------------------------------------------------------------

def compute_gd89_temperature_distribution(file_path, radiation_field_func,
                                          cross_section_table,
                                          t_min=1.0, num_bins=150):
    """
    Steady-state f(T) via the GD89 log-space recursion (single-photon regime).
    """
    freq_ev, einstein_A = load_pah_modes(file_path)
    active_mask    = einstein_A > 0
    cool_freq      = freq_ev[active_mask]
    cool_A         = einstein_A[active_mask]

    cs_E   = cross_section_table[:, 0]
    cs_sig = cross_section_table[:, 1]
    c_cgs  = 2.99792458e10
    h_cgs  = 6.62607015e-27

    def get_sigma(E):
        return np.interp(E, cs_E, cs_sig, left=0.0, right=0.0)

    # T_max: temperature at 2×Lyman-limit energy
    sol   = root_scalar(lambda T: _qho_energy(freq_ev, T) - 27.2,
                        bracket=[50.0, 12000.0], method='brentq')
    t_max = sol.root

    t_edges  = np.logspace(np.log10(t_min), np.log10(t_max), num_bins + 1)
    t_centers = 0.5 * (t_edges[:-1] + t_edges[1:])
    delta_t  = np.diff(t_edges)
    u_edges  = np.array([_qho_energy(freq_ev, t) for t in t_edges])
    u_centers = np.array([_qho_energy(freq_ev, t) for t in t_centers])

    W_up            = np.zeros((num_bins, num_bins))
    W_down_adjacent = np.zeros(num_bins)

    for j in range(num_bins):
        U_j = u_centers[j]
        T_j = t_centers[j]
        if j > 0:
            x = cool_freq / (_KB_EV * T_j)
            with np.errstate(over='ignore', under='ignore', invalid='ignore'):
                occ = np.where(x > 50.0, np.exp(-x) / (1.0 - np.exp(-x)), 1.0 / np.expm1(x))
                occ = np.where(np.isnan(occ) | np.isinf(occ), 0.0, occ)
            P_IR = np.sum(cool_freq * cool_A * occ)
            bw   = u_centers[j] - u_centers[j - 1]
            W_down_adjacent[j] = (P_IR / bw) + 1e-30

        for k in range(j + 1, num_bins):
            u_min = u_edges[k]     - U_j
            u_max = u_edges[k + 1] - U_j
            e_mid = 0.5 * (u_min + u_max)
            if e_mid <= 0 or e_mid > 13.6:
                continue
            nu_min  = (u_min / _HC_EV) * c_cgs
            nu_max  = (u_max / _HC_EV) * c_cgs
            nu_mid  = 0.5 * (nu_min + nu_max)
            dnu     = nu_max - nu_min
            sig     = get_sigma(e_mid)
            if sig <= 0.0:
                continue
            flux = radiation_field_func(nu_mid)
            W_up[j, k] = max(0.0, 4.0 * np.pi * (flux / (h_cgs * nu_mid)) * sig * dnu)

    log_f    = np.zeros(num_bins)
    log_f[0] = 0.0
    for f in range(1, num_bins):
        log_terms = []
        for j in range(f):
            rate = np.sum(W_up[j, f:])
            if rate > 0 and log_f[j] > -700:
                log_terms.append(log_f[j] + np.log(rate))
        if not log_terms:
            log_f[f] = -np.inf
            continue
        mx          = np.max(log_terms)
        log_f[f]    = mx + np.log(np.sum(np.exp(log_terms - mx))) - np.log(W_down_adjacent[f])

    mx_lf     = np.max(log_f[np.isfinite(log_f)])
    f_discrete = np.exp(log_f - mx_lf)
    f_T        = (f_discrete / np.sum(f_discrete)) / delta_t
    return t_centers, f_T


# ---------------------------------------------------------------------------
# Adaptive distribution (GD89 + multi-photon matrix solver)
# ---------------------------------------------------------------------------

def compute_adaptive_temperature_distribution(file_path, radiation_field_func,
                                              cross_section_table,
                                              t_min=15.0, num_bins=150, threshold=0.01):
    """
    Steady-state f(T) with automatic single-photon / multi-photon regime detection.

    In the single-photon regime uses the GD89 log-space recursion.
    In the multi-photon regime solves the steady-state rate matrix (null-space).
    """
    freq_ev, einstein_A = load_pah_modes(file_path)
    active_mask = einstein_A > 0
    cool_freq   = freq_ev[active_mask]
    cool_A      = einstein_A[active_mask]

    cs_E   = cross_section_table[:, 0]
    cs_sig = cross_section_table[:, 1]
    c_cgs  = 2.99792458e10
    h_cgs  = 6.62607015e-27
    eV_erg = 1.602176634e-12

    def get_sigma(E):
        if E <= 0.0 or E > 13.6:
            return 0.0
        return float(np.interp(E, cs_E, cs_sig, left=0.0, right=0.0))

    # --- Absorption rate and heating power ---
    eval_E = np.linspace(6.0, 13.6, 1000)
    dE     = eval_E[1] - eval_E[0]
    R_abs  = 0.0
    P_heat = 0.0
    for E_phot in eval_E:
        nu  = (E_phot / _HC_EV) * c_cgs
        I   = radiation_field_func(nu)
        sig = get_sigma(E_phot)
        if sig <= 0.0:
            continue
        dnu  = (dE / _HC_EV) * c_cgs
        flux = 4.0 * np.pi * I / (h_cgs * nu)
        R_abs  += flux * sig * dnu
        P_heat += 4.0 * np.pi * I * sig * dnu
    P_heat_eV = P_heat / eV_erg

    # --- Integrated cooling lifetime ---
    sol_1ph  = root_scalar(lambda T: _qho_energy(freq_ev, T) - 13.6,
                           bracket=[15.0, 5000.0], method='brentq')
    T_max_1ph = sol_1ph.root
    sol_min   = root_scalar(lambda T: _qho_energy(freq_ev, T) - 0.136,
                            bracket=[1.0, T_max_1ph], method='brentq')
    T_min_cool = sol_min.root

    t_mesh = np.linspace(T_min_cool, T_max_1ph, 200)
    dT_mesh = t_mesh[1] - t_mesh[0]
    tau_cool = 0.0
    for T in t_mesh:
        Cv = _qho_cv(freq_ev, T)
        x  = cool_freq / (_KB_EV * T)
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            occ = np.where(x > 50.0, np.exp(-x) / (1.0 - np.exp(-x)), 1.0 / np.expm1(x))
            occ = np.where(np.isnan(occ) | np.isinf(occ), 0.0, occ)
        P_cool = np.sum(cool_freq * cool_A * occ)
        if P_cool > 0:
            tau_cool += (Cv / P_cool) * dT_mesh

    stacking_index = tau_cool * R_abs
    print(f"Stacking Index = {stacking_index:.4e}, tau_cool = {tau_cool:.2f} s")

    if stacking_index < threshold:
        sol_2ph = root_scalar(lambda T: _qho_energy(freq_ev, T) - 27.2,
                              bracket=[T_max_1ph, 15000.0], method='brentq')
        t_max         = sol_2ph.root
        use_multi_photon = False
    else:
        def eq_obj(T):
            x  = cool_freq / (_KB_EV * T)
            with np.errstate(over='ignore', under='ignore', invalid='ignore'):
                occ = np.where(x > 50.0, np.exp(-x) / (1.0 - np.exp(-x)), 1.0 / np.expm1(x))
                occ = np.where(np.isnan(occ) | np.isinf(occ), 0.0, occ)
            return np.sum(cool_freq * cool_A * occ) - P_heat_eV
        sol_eq = root_scalar(eq_obj, bracket=[1.0, 5000.0], method='brentq')
        t_eq   = sol_eq.root
        print(f"Equilibrium Temperature: {t_eq:.2f} K")
        u_ceil  = _qho_energy(freq_ev, t_eq) + 13.6
        sol_max = root_scalar(lambda T: _qho_energy(freq_ev, T) - u_ceil,
                              bracket=[t_eq, 10000.0], method='brentq')
        t_max            = 3.0 * sol_max.root
        use_multi_photon = True

    t_edges  = np.logspace(np.log10(t_min), np.log10(t_max), num_bins + 1)
    t_centers = 0.5 * (t_edges[:-1] + t_edges[1:])
    delta_t  = np.diff(t_edges)
    u_edges  = np.array([_qho_energy(freq_ev, t) for t in t_edges])
    u_centers = np.array([_qho_energy(freq_ev, t) for t in t_centers])

    W_up            = np.zeros((num_bins, num_bins))
    W_down_adjacent = np.zeros(num_bins)

    for j in range(num_bins):
        U_j = u_centers[j]
        T_j = t_centers[j]
        if j > 0:
            x = cool_freq / (_KB_EV * T_j)
            with np.errstate(over='ignore', under='ignore', invalid='ignore'):
                occ = np.where(x > 50.0, np.exp(-x) / (1.0 - np.exp(-x)), 1.0 / np.expm1(x))
                occ = np.where(np.isnan(occ) | np.isinf(occ), 0.0, occ)
            P_IR = np.sum(cool_freq * cool_A * occ)
            bw   = u_centers[j] - u_centers[j - 1]
            W_down_adjacent[j] = (P_IR / bw) + 1e-30

        for k in range(j + 1, num_bins):
            u_min = u_edges[k]     - U_j
            u_max = u_edges[k + 1] - U_j
            e_mid = 0.5 * (u_min + u_max)
            if e_mid <= 0 or e_mid > 13.6:
                continue
            nu_min = (u_min / _HC_EV) * c_cgs
            nu_max = (u_max / _HC_EV) * c_cgs
            nu_mid = 0.5 * (nu_min + nu_max)
            dnu    = nu_max - nu_min
            sig    = get_sigma(e_mid)
            if sig <= 0.0:
                continue
            flux = radiation_field_func(nu_mid)
            W_up[j, k] = max(0.0, 4.0 * np.pi * (flux / (h_cgs * nu_mid)) * sig * dnu)

    if not use_multi_photon:
        log_f    = np.zeros(num_bins)
        log_f[0] = 0.0
        for f in range(1, num_bins):
            log_terms = []
            for j in range(f):
                rate = np.sum(W_up[j, f:])
                if rate > 0 and log_f[j] > -700:
                    log_terms.append(log_f[j] + np.log(rate))
            if not log_terms:
                log_f[f] = -np.inf
                continue
            mx       = np.max(log_terms)
            log_f[f] = mx + np.log(np.sum(np.exp(log_terms - mx))) - np.log(W_down_adjacent[f])
        mx_lf      = np.max(log_f[np.isfinite(log_f)])
        f_discrete  = np.exp(log_f - mx_lf)
    else:
        M = np.zeros((num_bins, num_bins))
        for i in range(num_bins):
            out = (W_down_adjacent[i] if i > 0 else 0.0) + np.sum(W_up[i, :])
            M[i, i] = -out
            if i < num_bins - 1:
                M[i, i + 1] = W_down_adjacent[i + 1]
            for j in range(i):
                M[i, j] = W_up[j, i]
        null_vecs  = null_space(M)
        f_discrete = np.abs(null_vecs[:, 0]) if null_vecs.size > 0 else np.zeros(num_bins)

    total = np.sum(f_discrete)
    f_T   = (f_discrete / total) / delta_t
    return t_centers, f_T


# ---------------------------------------------------------------------------
# Bakes / Dwek Poisson temperature distribution
# ---------------------------------------------------------------------------

def compute_bakes_temperature_distribution(
    file_path, radiation_field_func, cross_section_table, et_table_path,
    Nc=54, T0=2.73, n_internal=1000, num_bins=300, n_dwek_max=20, tol=1e-6,
):
    """
    Bakes (2001) / Andrews (2016) Poisson f(T) with Dwek (1986) multi-photon
    corrections.  Uses the Tielens (2005) E-T table for Cv(T); PAHdb modes for
    P_IR(T).
    """
    freq_ev, einstein_A = load_pah_modes(file_path)
    A_i = einstein_A

    et     = np.loadtxt(et_table_path, delimiter=',')
    T_et   = et[:, 0]
    U_et   = et[:, 1] * float(Nc)
    Cv_et  = np.gradient(U_et, T_et)

    def U_of_T(T):
        return float(np.interp(T, T_et, U_et, left=0.0, right=U_et[-1]))

    def T_of_U(U):
        return float(np.interp(U, U_et, T_et, left=T0, right=T_et[-1]))

    def Cv_of_T(T):
        return float(np.interp(T, T_et, Cv_et, left=Cv_et[0], right=Cv_et[-1]))

    hc_ev  = _HC_EV
    h_cgs  = 6.62607015e-27
    eV_erg = 1.602176634e-12
    c_cgs  = 2.99792458e10

    E_eval   = np.linspace(6.0, 13.6, 800)
    dE       = E_eval[1] - E_eval[0]
    sig_eval = np.interp(E_eval, cross_section_table[:, 0],
                         cross_section_table[:, 1], left=0.0, right=0.0)
    nu_eval  = (E_eval / hc_ev) * c_cgs
    J_nu     = np.array([radiation_field_func(nu) for nu in nu_eval])
    phot_flux = 4.0 * np.pi * J_nu / (h_cgs * nu_eval)
    dnu_dE   = eV_erg / h_cgs
    dR_dE    = phot_flux * sig_eval * dnu_dE
    R_abs    = np.sum(dR_dE) * dE
    if R_abs <= 0:
        t_out = np.logspace(np.log10(max(T0, 10.0)), 4.0, num_bins)
        return t_out, np.zeros(num_bins)
    mean_E = np.sum(E_eval * dR_dE) * dE / R_abs

    U_peak = U_of_T(T0) + mean_E
    T_max  = max(T_of_U(min(U_peak, U_et[-1])), T0 + 50.0)

    T_int  = np.linspace(T0, T_max, n_internal)
    dT_int = T_int[1] - T_int[0]

    dTdt_int = np.zeros(n_internal)
    for k in range(n_internal):
        T = T_int[k]
        if T < 5.0:
            continue
        x = freq_ev / (_KB_EV * T)
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            occ = np.where(x > 50.0, np.exp(-x) / (1.0 - np.exp(-x)), 1.0 / np.expm1(x))
            occ = np.where(np.isnan(occ) | np.isinf(occ), 0.0, occ)
        P_IR = np.sum(A_i * freq_ev * occ)
        Cv   = Cv_of_T(T)
        dTdt_int[k] = P_IR / max(Cv, 1e-30)
    dTdt_int = np.maximum(dTdt_int, 1e-30)

    inv_dTdt = 1.0 / dTdt_int
    tau_int  = np.zeros(n_internal)
    for k in range(n_internal - 2, -1, -1):
        tau_int[k] = tau_int[k + 1] + 0.5 * (inv_dTdt[k] + inv_dTdt[k + 1]) * dT_int

    U_int        = np.array([U_of_T(T) for T in T_int])
    U_peak_int   = np.minimum(U_int + mean_E, U_et[-1])
    T_peak_int   = np.array([T_of_U(U) for U in U_peak_int])
    k_peak_int   = np.searchsorted(T_int, T_peak_int).clip(0, n_internal - 1)
    tau_at_Tpeak = tau_int[k_peak_int]

    tau_offset = tau_at_Tpeak[0]
    G_1 = np.zeros(n_internal)
    k_max1 = k_peak_int[0]
    if k_max1 > 0:
        exp_arg  = -R_abs * (tau_int[:k_max1] - tau_offset)
        G_1[:k_max1] = (1.0 / dTdt_int[:k_max1]) * np.exp(np.clip(exp_arg, -500.0, 0.0))
    G_1 = np.where(np.isfinite(G_1), G_1, 0.0)

    G_n     = G_1.copy()
    G_total = G_1.copy()

    SI = R_abs * (tau_int[0] - tau_offset)
    if SI > 1e-4:
        U_thresh_j   = np.maximum(0.0, U_int - mean_E)
        k_thresh_arr = np.searchsorted(U_int, U_thresh_j).clip(0, n_internal - 1)
        for _ in range(n_dwek_max):
            exp_arg  = np.clip(R_abs * tau_at_Tpeak, 0.0, 500.0)
            weight   = np.where(np.isfinite(G_n * np.exp(exp_arg)), G_n * np.exp(exp_arg), 0.0)
            cumW     = np.cumsum(weight) * dT_int
            integ    = cumW.copy()
            mask     = k_thresh_arr > 0
            integ[mask] -= cumW[k_thresh_arr[mask] - 1]
            G_next = ((1.0 / dTdt_int)
                      * np.exp(np.clip(-R_abs * tau_int, -500.0, 0.0))
                      * integ)
            G_next = np.where(np.isfinite(G_next), G_next, 0.0)
            G_next[0] = 0.0
            G_total += G_next
            if np.max(np.abs(G_next)) < tol * max(np.max(G_total), 1e-30):
                break
            G_n = G_next

    integral_hot = np.trapezoid(G_total, T_int)
    P_cold = max(0.0, 1.0 - integral_hot)
    if P_cold > 0 and dT_int > 0:
        G_total[0] += P_cold / dT_int
    norm = np.trapezoid(G_total, T_int)
    if norm > 0:
        G_total /= norm
    G_total = np.where(np.isfinite(G_total), G_total, 0.0)

    t_min_out    = max(T0, 10.0)
    t_edges_out  = np.logspace(np.log10(t_min_out), np.log10(T_max), num_bins + 1)
    t_out        = np.sqrt(t_edges_out[:-1] * t_edges_out[1:])
    delta_t_out  = np.diff(t_edges_out)
    f_out = np.zeros(num_bins)
    for k in range(num_bins):
        mask_k = (T_int >= t_edges_out[k]) & (T_int < t_edges_out[k + 1])
        if mask_k.any():
            f_out[k] = np.trapezoid(G_total[mask_k], T_int[mask_k]) / delta_t_out[k]
    norm2 = np.sum(f_out * delta_t_out)
    if norm2 > 0:
        f_out /= norm2
    return t_out, f_out


def compute_andrews_temperature_distribution(
    file_path, radiation_field_func, cross_section_table,
    T_floor=15.0, n_T=2000, n_E=800,
    k_H_table=None, k_H2_table=None,
):
    """
    Andrews (2016) / Bakes (2001) single-photon temperature probability
    distribution G̃(T) computed consistently from PAHdb QHO modes.

    Uses QHO vibrational modes for U(T), Cv(T) = dU/dT, and P_IR(T)
    throughout — no external E-T table.

    The cascade cooling rate is:
        dT/dt = P_IR(T) / Cv(T)   [K s^-1]

    The minimum cooling time from energy E down to temperature T is:
        τ_min(T, E) = ∫_T^{T_E} Cv(T') / P_IR(T') dT'

    The un-normalised spectrum-averaged distribution including survival probability:

        G̃(T) = (Cv/P_IR)(T) × exp(-K_tot(T,T_E)) × ∫_{E>U(T)} exp(-R_abs τ_min) n̂(E) dE

    where K_tot(T,T_E) = ∫_T^{T_E} k_tot(T') × Cv(T')/P_IR(T') dT'
    and k_tot = k_H + k_H2 + k_IR (survival probability against dissociation during cascade).

    If k_H_table and k_H2_table are None, K_tot = 0 (Bakes approximation — only k_IR
    in k_tot, which simplifies the computation). In the full Andrews formula, all three
    channels are included.

    Physical meaning of G̃:
        P_H = R_abs × ∫ k_H(T) G̃(T) dT  [s^-1] — time-averaged H-loss rate

    NOTE: G̃ is returned UN-NORMALISED (units [s K^-1]).
    The rate is:   rate_i = R_abs × trapezoid(k_i * G_tilde, T_int)

    Parameters
    ----------
    file_path             : path to PAHdb .dat modes file
    radiation_field_func  : callable(nu [Hz]) → I_ν [erg cm^-2 s^-1 Hz^-1 sr^-1]
    cross_section_table   : ndarray shape (N,2), columns [E_eV, σ_cm2]
    T_floor               : lower temperature bound [K]
    n_T                   : number of T grid points for internal integration
    n_E                   : number of photon-energy grid points
    k_H_table             : ndarray shape (M,2) [U_eV, k_H s^-1], or None
    k_H2_table            : ndarray shape (M,2) [U_eV, k_H2 s^-1], or None

    Returns
    -------
    T_centers : ndarray (n_T,) — temperature grid [K]
    G_tilde   : ndarray (n_T,) — G̃(T) [s K^-1], UN-NORMALISED
    R_abs     : float — photon absorption rate [s^-1]
    U_arr     : ndarray (n_T,) — QHO internal energy at each T [eV]
    """
    freq_ev, A_i = load_pah_modes(file_path)

    hc_ev  = _HC_EV
    h_cgs  = 6.62607015e-27
    c_cgs  = 2.99792458e10
    eV_erg = 1.602176634e-12
    kb_ev  = _KB_EV

    # ── 1. Build T grid and precompute QHO quantities ─────────────────────────
    T_max_guess = 3500.0
    for _ in range(3):
        x = freq_ev / (kb_ev * T_max_guess)
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            occ = np.where(x > 50, np.exp(-x) / (1 - np.exp(-x)), 1 / np.expm1(x))
            occ = np.where(np.isnan(occ) | np.isinf(occ), 0, occ)
        U_test = float(np.sum(freq_ev * occ))
        if U_test > 14.0:
            break
        T_max_guess *= 1.5

    T_int = np.linspace(T_floor, T_max_guess, n_T)
    dT    = T_int[1] - T_int[0]

    U_arr    = np.zeros(n_T)
    Cv_arr   = np.zeros(n_T)
    P_IR_arr = np.zeros(n_T)

    for k, T in enumerate(T_int):
        if T < 5.0:
            continue
        x = freq_ev / (kb_ev * T)
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            occ = np.where(x > 50, np.exp(-x) / (1 - np.exp(-x)), 1 / np.expm1(x))
            occ = np.where(np.isnan(occ) | np.isinf(occ), 0, occ)
        U_arr[k]    = float(np.sum(freq_ev * occ))
        ex          = np.exp(np.minimum(x, 500.0))
        em1sq       = np.maximum((ex - 1.0) ** 2, 1e-30)
        Cv_arr[k]   = float(np.sum((freq_ev / (kb_ev * T)) ** 2 * ex / em1sq) * kb_ev)
        P_IR_arr[k] = float(np.sum(A_i * freq_ev * occ))

    Cv_arr   = np.maximum(Cv_arr,   1e-30)
    P_IR_arr = np.maximum(P_IR_arr, 1e-30)
    dTdt_arr = P_IR_arr / Cv_arr          # K s^-1
    inv_dTdt = Cv_arr / P_IR_arr          # s K^-1

    # ── 2. τ(T) = ∫_T^{T_max} Cv/P_IR dT'  (cumulative from above) ───────────
    tau = np.zeros(n_T)
    for k in range(n_T - 2, -1, -1):
        tau[k] = tau[k + 1] + 0.5 * (inv_dTdt[k] + inv_dTdt[k + 1]) * dT

    # ── 3. Survival integral K(T) = ∫_T^{T_max} k_tot × Cv/P_IR dT' ──────────
    # k_tot = k_H + k_H2 + k_IR.  k_IR = P_IR/U → k_IR × Cv/P_IR = Cv/U.
    # The k_IR contribution to K is ∫Cv/U dT = ∫dU/U = ln(U_max/U(T)).
    k_IR_arr = P_IR_arr / np.maximum(U_arr, 1e-10)  # [s^-1]

    k_diss_arr = np.zeros(n_T)  # k_H + k_H2 on T grid
    if k_H_table is not None:
        k_diss_arr += np.interp(U_arr, k_H_table[:, 0], k_H_table[:, 1],
                                left=0.0, right=k_H_table[-1, 1])
    if k_H2_table is not None:
        k_diss_arr += np.interp(U_arr, k_H2_table[:, 0], k_H2_table[:, 1],
                                left=0.0, right=k_H2_table[-1, 1])

    k_tot_arr = k_diss_arr + k_IR_arr  # [s^-1]

    # K_arr[k] = ∫_{T_k}^{T_max} k_tot × Cv/P_IR dT'  (cumulative from above)
    K_arr = np.zeros(n_T)
    for k in range(n_T - 2, -1, -1):
        K_arr[k] = K_arr[k + 1] + 0.5 * (
            k_tot_arr[k] * inv_dTdt[k] + k_tot_arr[k + 1] * inv_dTdt[k + 1]
        ) * dT

    # ── 4. Photon spectrum ────────────────────────────────────────────────────
    E_eval   = np.linspace(6.0, 13.6, n_E)
    dE       = E_eval[1] - E_eval[0]
    sig_eval = np.interp(E_eval, cross_section_table[:, 0],
                         cross_section_table[:, 1], left=0.0, right=0.0)
    nu_eval   = (E_eval / hc_ev) * c_cgs
    J_nu      = np.array([radiation_field_func(nu) for nu in nu_eval])
    phot_flux = 4.0 * np.pi * J_nu / (h_cgs * nu_eval)
    dR_dE     = phot_flux * sig_eval * (eV_erg / h_cgs)
    R_abs     = float(np.sum(dR_dE) * dE)
    if R_abs <= 0:
        return T_int, np.zeros(n_T), 0.0, U_arr
    n_ph = dR_dE / R_abs                  # normalised photon spectrum [eV^-1]

    # ── 5. T_E(E): temperature reached after absorbing photon of energy E ─────
    T_E_arr = np.interp(E_eval, U_arr, T_int, left=T_floor, right=T_max_guess)
    tau_TE  = np.interp(T_E_arr, T_int, tau)   # τ(T_E)
    K_TE    = np.interp(T_E_arr, T_int, K_arr)  # K(T_E)

    # ── 6. G̃(T_k) with full survival probability ──────────────────────────────
    # The combined exponent for each (T_k, E) pair is:
    #   R_abs*(τ_TE - τ_k) + (K_TE - K_arr_k)
    #   = R_abs*(τ_TE - τ_k) - K_diss(T_k, T_E)
    #
    # Both terms are ≤ 0 for T_E > T_k (physically: multi-photon suppression
    # and survival suppression against dissociation). Computing the combined
    # exponent avoids overflow when K_diss >> 500.
    G_tilde = np.zeros(n_T)
    for k in range(n_T):
        T_k  = T_int[k]
        mask = T_E_arr > T_k
        if not mask.any():
            break
        exponent = (R_abs * (tau_TE[mask] - tau[k])   # multi-photon term (≤ 0)
                    + (K_TE[mask] - K_arr[k]))          # survival term (≤ 0)
        exp_vals  = np.exp(np.clip(exponent, -500.0, 0.0))
        G_tilde[k] = float(np.sum(exp_vals * n_ph[mask]) * dE) / dTdt_arr[k]

    G_tilde = np.where(np.isfinite(G_tilde), G_tilde, 0.0)
    # Return un-normalised G̃ [s K^-1] and U_arr for convenience.
    # Rate: rate_i = R_abs × trapezoid(k_i * G_tilde, T_int)
    return T_int, G_tilde, R_abs, U_arr


# ---------------------------------------------------------------------------
# Time-averaged IR emission rate
# ---------------------------------------------------------------------------

def compute_total_time_averaged_ir_rate(file_path, t_centers, f_T, t_min=15.0):
    """
    Time-averaged macroscopic IR emission rate k_IR [s^-1]:
        k_IR = ∫ f(T) × K_thermal(T) dT
    """
    freq_ev, einstein_A = load_pah_modes(file_path)

    num_bins = len(t_centers)
    t_max    = 10**(2 * np.log10(t_centers[-1]) - np.log10(t_centers[-2]))
    t_edges  = np.logspace(np.log10(t_min), np.log10(t_max), num_bins + 1)
    delta_t  = np.diff(t_edges)
    f_disc   = f_T * delta_t

    K_arr = np.zeros(num_bins)
    for j in range(num_bins):
        T_j = t_centers[j]
        if T_j <= 15.0:
            continue
        x = freq_ev / (_KB_EV * T_j)
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            occ = np.where(x > 50.0, np.exp(-x) / (1.0 - np.exp(-x)), 1.0 / np.expm1(x))
            occ = np.where(np.isnan(occ) | np.isinf(occ), 0.0, occ)
        U_Tj = np.sum(freq_ev * occ)
        if U_Tj <= 0:
            continue
        K_arr[j] = np.sum(freq_ev * einstein_A * occ) / U_Tj

    return float(np.sum(f_disc * K_arr))


# ---------------------------------------------------------------------------
# DustEM-equivalent: spectral P_IR + DL07 U(T) variants
# ---------------------------------------------------------------------------

def _planck_lambda_arr(T, wav_cm):
    """Planck B_λ(T) [erg/s/cm²/cm/sr] for an array of wavelengths."""
    _h = 6.62607015e-27
    _c = 2.99792458e10
    _k = 1.380649e-16
    x = _h * _c / (wav_cm * _k * T)
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        bnu = np.where(x > 500.0, 0.0,
                       2.0 * _h * _c**2 / wav_cm**5 / (np.expm1(np.minimum(x, 500.0)) + 1e-300))
    return bnu


def _pir_spectral_fast(T, wav_cm, C_abs_cm2):
    """4π ∫ C_abs(λ) B_λ(T) dλ [erg/s]. Sorts by wavelength for trapezoid."""
    B = _planck_lambda_arr(T, wav_cm)
    idx = np.argsort(wav_cm)
    return float(4.0 * np.pi * np.trapezoid(C_abs_cm2[idx] * B[idx], wav_cm[idx]))


def _build_dl07_tables(T_dl07, U_dl07_ev):
    """
    Return (U_of_T, C_of_T, t_max_ev27) helpers from DL07 tabulated data.

    U_of_T(T)  → internal energy [eV] via log-linear interpolation + power-law extrapolation
    C_of_T(T)  → dU/dT [eV/K], the volumetric heat capacity per grain, via finite differences
    t_max_ev27 → temperature where U(T) = 27.2 eV (2× Lyman limit); extrapolated if needed
    """
    # Work in log-log space for smooth interpolation
    logT = np.log10(np.maximum(T_dl07, 1e-6))
    logU = np.log10(np.maximum(U_dl07_ev, 1e-40))

    def U_of_T(T):
        lT = np.log10(max(T, 1e-6))
        if lT <= logT[0]:
            # C ∝ T³ at very low T (Debye) → U ∝ T⁴; use power law from first two points
            slope = (logU[1] - logU[0]) / (logT[1] - logT[0])
            return float(10 ** (logU[0] + slope * (lT - logT[0])))
        if lT >= logT[-1]:
            slope = (logU[-1] - logU[-2]) / (logT[-1] - logT[-2])
            return float(10 ** (logU[-1] + slope * (lT - logT[-1])))
        return float(10 ** np.interp(lT, logT, logU))

    # Numerical dU/dT on the tabulated grid (central differences)
    dU_dT = np.gradient(U_dl07_ev, T_dl07)

    def C_of_T(T):
        if T <= T_dl07[0]:
            # extrapolate with power law; same slope as U_of_T
            slope = (logU[1] - logU[0]) / (logT[1] - logT[0])
            return float(dU_dT[0] * (T / T_dl07[0]) ** (slope - 1.0))
        if T >= T_dl07[-1]:
            return float(dU_dT[-1])
        return float(np.interp(T, T_dl07, dU_dT))

    # T_max where U = 27.2 eV
    if U_dl07_ev[-1] >= 27.2:
        t_max = float(np.interp(27.2, U_dl07_ev, T_dl07))
    else:
        # Extrapolate log-log
        slope = (logU[-1] - logU[-2]) / (logT[-1] - logT[-2])
        t_max = float(T_dl07[-1] * (27.2 / U_dl07_ev[-1]) ** (1.0 / slope))

    return U_of_T, C_of_T, t_max


def compute_spectral_gd89_distribution(
    radiation_field_func,
    cross_section_table,
    wav_cm,
    C_abs_cm2,
    T_dl07,
    U_dl07_ev,
    t_min=10.0,
    num_bins=150,
    freq_ev_qho=None,
):
    """
    GD89 log-space single-photon recursion with spectral P_IR.

    Two modes depending on ``freq_ev_qho``:

    * ``freq_ev_qho=None`` (default): use tabulated T_dl07/U_dl07_ev for the bin
      structure U(T).  The Tielens (2005) or DL07 table must cover the requested
      t_min range.

    * ``freq_ev_qho=<array>`` : use the QHO mode-sum U(T) from the supplied
      vibrational frequencies for the bin structure (same as the standard
      GD89 solver), but switch P_IR to the spectral formula.  This lets t_min
      go as low as ~1 K and isolates the effect of spectral P_IR alone.

    In the single-photon regime the result is functionally equivalent to
    DustEM's power-iteration solver (Desert et al. 1986).

    Parameters
    ----------
    radiation_field_func : callable
        I_nu(nu [Hz]) → erg cm^-2 s^-1 Hz^-1 sr^-1
    cross_section_table  : (N,2) ndarray  [E_eV, sigma_cm2]
    wav_cm               : (M,) ndarray  wavelengths for spectral P_IR [cm]
    C_abs_cm2            : (M,) ndarray  absorption cross-section [cm²]
    T_dl07               : (K,) ndarray  temperature grid from heat-capacity table [K]
    U_dl07_ev            : (K,) ndarray  internal energy at each T [eV / molecule]
    t_min                : float  minimum temperature bin edge [K]
    num_bins             : int   number of temperature bins
    freq_ev_qho          : (N_modes,) ndarray or None
        If given, QHO mode frequencies [eV] are used for the U(T) bin structure
        (allowing low t_min) while P_IR stays spectral.

    Returns
    -------
    t_centers : (num_bins,) ndarray  bin centres [K]
    f_T       : (num_bins,) ndarray  f(T) [K^-1]
    """
    _eV2erg = 1.602176634e-12
    c_cgs   = 2.99792458e10
    h_cgs   = 6.62607015e-27

    if freq_ev_qho is not None:
        # Use QHO U(T) for bin structure — same as standard GD89
        def U_of_T(T):
            return _qho_energy(freq_ev_qho, T)
        t_max = float(T_dl07[-1]) if T_dl07 is not None else 1.2e4
    else:
        U_of_T, _C_of_T, t_max = _build_dl07_tables(T_dl07, U_dl07_ev)

    t_edges   = np.logspace(np.log10(t_min), np.log10(t_max), num_bins + 1)
    t_centers = 0.5 * (t_edges[:-1] + t_edges[1:])
    delta_t   = np.diff(t_edges)

    u_edges   = np.array([U_of_T(t) for t in t_edges])
    u_centers = np.array([U_of_T(t) for t in t_centers])

    cs_E   = cross_section_table[:, 0]
    cs_sig = cross_section_table[:, 1]

    W_up            = np.zeros((num_bins, num_bins))
    W_down_adjacent = np.zeros(num_bins)

    for j in range(num_bins):
        T_j = t_centers[j]

        # Spectral cooling power [eV/s]
        if j > 0:
            P_IR_ev = _pir_spectral_fast(T_j, wav_cm, C_abs_cm2) / _eV2erg
            bw = u_centers[j] - u_centers[j - 1]
            W_down_adjacent[j] = max(P_IR_ev, 1e-40) / max(bw, 1e-40)

        # Upward absorption rates
        for k in range(j + 1, num_bins):
            u_min = u_edges[k]     - u_centers[j]
            u_max = u_edges[k + 1] - u_centers[j]
            e_mid = 0.5 * (u_min + u_max)
            if e_mid <= 0.0 or e_mid > 13.6:
                continue
            nu_min = (u_min / _HC_EV) * c_cgs
            nu_max = (u_max / _HC_EV) * c_cgs
            nu_mid = 0.5 * (nu_min + nu_max)
            dnu    = nu_max - nu_min
            sig    = float(np.interp(e_mid, cs_E, cs_sig, left=0.0, right=0.0))
            if sig <= 0.0 or nu_mid <= 0.0 or dnu <= 0.0:
                continue
            flux = radiation_field_func(nu_mid)
            W_up[j, k] = max(0.0, 4.0 * np.pi * (flux / (h_cgs * nu_mid)) * sig * dnu)

    # GD89 log-space forward recursion (unchanged from QHO version)
    log_f    = np.zeros(num_bins)
    log_f[0] = 0.0
    for f in range(1, num_bins):
        log_terms = []
        for j in range(f):
            rate = np.sum(W_up[j, f:])
            if rate > 0 and log_f[j] > -700:
                log_terms.append(log_f[j] + np.log(rate))
        if not log_terms:
            log_f[f] = -np.inf
            continue
        mx       = np.max(log_terms)
        log_f[f] = mx + np.log(np.sum(np.exp(log_terms - mx))) - np.log(W_down_adjacent[f])

    mx_lf     = np.max(log_f[np.isfinite(log_f)])
    f_discrete = np.exp(log_f - mx_lf)
    f_T        = (f_discrete / np.sum(f_discrete)) / delta_t
    return t_centers, f_T


def compute_dustem_poweriter_distribution(
    radiation_field_func,
    cross_section_table,
    wav_cm,
    C_abs_cm2,
    T_dl07,
    U_dl07_ev,
    t_min=10.0,
    num_bins=150,
    n_iter=80,
    freq_ev_qho=None,
):
    """
    Desert et al. (1986) power-iteration solver — the algorithm used by DustEM.

    Uses the same spectral/DL07 physics as compute_spectral_gd89_distribution,
    but adds the τ multi-photon correction to the cascade kernel, making it
    valid in both the single-photon and multi-photon regimes.

    Physics inputs (identical to DustEM GET_TDIST):
      - Heating:  L[i,j] = ∫ nbrpho(ν) σ(E) dν for each (i→j) state pair
      - Cooling:  g[j]   = 4π ∫ C_abs(λ) B_λ(T_j) dλ
      - τ corr:   τ = R_abs × ΔU / g   (photons absorbed during cascade through bin)

    Algorithm (DustEM eq. 24–25):
      1. Build emission-cascade matrix F[i,j]:
             F[i,j] = (L[i,j]×ΔU_j + F[i,j+1]×g[j+1]) × exp(−τ_j) / g[j]
         Backward recursion from j = ndist−1 down to 0.
      2. Iterate  p ← F^T · p  for n_iter steps.
      3. Normalise and convert from p(U) to f(T) = p(U) × C(T).

    Returns
    -------
    t_centers : (num_bins,) ndarray  [K]
    f_T       : (num_bins,) ndarray  f(T) [K^-1]
    """
    _eV2erg = 1.602176634e-12
    c_cgs   = 2.99792458e10
    h_cgs   = 6.62607015e-27

    if freq_ev_qho is not None:
        # QHO bin structure (valid down to very low T) + spectral P_IR
        def U_of_T(T):
            return _qho_energy(freq_ev_qho, T)
        # Heat capacity for final f(T) conversion
        def C_of_T(T):
            return _qho_cv(freq_ev_qho, T)
        t_max = float(T_dl07[-1]) if T_dl07 is not None else 1.2e4
    else:
        U_of_T, C_of_T, t_max = _build_dl07_tables(T_dl07, U_dl07_ev)

    t_edges   = np.logspace(np.log10(t_min), np.log10(t_max), num_bins + 1)
    t_centers = 0.5 * (t_edges[:-1] + t_edges[1:])
    delta_t   = np.diff(t_edges)

    u_edges   = np.array([U_of_T(t) for t in t_edges])
    u_centers = np.array([U_of_T(t) for t in t_centers])
    delta_u   = np.diff(u_edges)                              # bin widths in U-space

    cs_E   = cross_section_table[:, 0]
    cs_sig = cross_section_table[:, 1]

    # --- Spectral cooling power g[j] = P_IR(T_j) [eV/s] ---
    g = np.array([max(_pir_spectral_fast(t, wav_cm, C_abs_cm2) / _eV2erg, 1e-40)
                  for t in t_centers])

    # --- Upward transition matrix L[i,j] [s^-1] ---
    # Same as W_up in GD89: rate of absorbing a photon that takes grain from
    # state i (energy u_centers[i]) to state j (energy u_centers[j]).
    L = np.zeros((num_bins, num_bins))
    for i in range(num_bins - 1):
        for j in range(i + 1, num_bins):
            u_min = u_edges[j]     - u_centers[i]
            u_max = u_edges[j + 1] - u_centers[i]
            e_mid = 0.5 * (u_min + u_max)
            if e_mid <= 0.0 or e_mid > 13.6:
                continue
            nu_mid = (e_mid / _HC_EV) * c_cgs
            dnu    = ((u_max - u_min) / _HC_EV) * c_cgs
            sig    = float(np.interp(e_mid, cs_E, cs_sig, left=0.0, right=0.0))
            if sig <= 0.0 or nu_mid <= 0.0 or dnu <= 0.0:
                continue
            flux   = radiation_field_func(nu_mid)
            L[i, j] = max(0.0, 4.0 * np.pi * (flux / (h_cgs * nu_mid)) * sig * dnu)

    # Total photon absorption rate R_abs [s^-1] — used for τ normalization
    # DustEM: l0 = ∫ lij(i, ndist) dU_i  (rate from any state to the top state)
    # Numerically: R_abs = Σ_i Σ_{j>i} L[i,j] × delta_u[i]
    R_abs = float(np.sum(L * delta_u[:, np.newaxis]))      # sum over (i,j) pairs

    # --- Emission-cascade kernel F (Desert+86 backward recursion) ---
    # F[i, j] = probability that grain in state j cascades to state i before the
    #           next photon absorption (accounts for possible photon interruptions).
    #
    # Recursion (backward in j from ndist-1 to 0):
    #   F[i, ndist-1] = 0
    #   τ_j = R_abs × ΔU_j / g[j]           (photons per cascade-through-bin time)
    #   F[i, j] = (L[i,j]×ΔU_j + F[i,j+1]×g[j+1]) × exp(−τ_j) / g[j]
    F = np.zeros((num_bins, num_bins))
    for i in range(num_bins):
        F[i, num_bins - 1] = 0.0
        for j in range(num_bins - 2, -1, -1):
            du  = delta_u[j]
            tau = R_abs * du / g[j]
            exp_tau = np.exp(-min(tau, 500.0))
            val = (L[i, j] * du + F[i, j + 1] * g[j + 1]) * exp_tau / g[j]
            F[i, j] = max(val, 0.0)

    # --- Find equilibrium bin nequi (G0-dependent initialization) ---
    # nequi = bin where cooling power equals heating power from cold state.
    # Absorbed power from cold state (bin 0): P_abs = Σ_k L[0,k] × (u_k - u_0)
    # Find bin nequi where g[nequi] ≈ P_abs (energy balance).
    P_abs = float(np.sum(L[0, :] * (u_centers - u_centers[0])))   # eV/s
    nequi = 0
    if P_abs > 0 and np.any(g > 0):
        # find bin whose cooling power is closest to P_abs
        nequi = int(np.argmin(np.abs(g - P_abs)))

    # --- Power iteration: p ← F^T · p, n_iter times ---
    # Use max-norm (like DustEM) so ratios between bins are preserved and the
    # G0-dependent shape of f(T) is not washed out.  Integral-norm at each
    # step makes p converge to the G0-independent principal eigenvector of F^T.
    p = np.zeros(num_bins)
    p[nequi] = 1.0   # DustEM: p(nequi) = 1.0

    F_T = F.T   # F_T[j,i] = F[i,j]: correct DustEM direction
    for _ in range(n_iter):
        # DustEM: fdist = TRANSPOSE(fdist); p = MATMUL(fdist, p)
        # ↔ p_new[j] = Σ_i F[i,j] × p[i] = (F.T @ p)[j]
        p = F_T @ p
        # Max-norm: prevents overflow while preserving inter-bin ratios.
        pmax = float(np.max(p))
        if pmax > 0:
            p /= pmax

    # --- Convert p(U) → f(T) = p(U) × C(T) where C(T) = dU/dT ---
    # Normalisation: ∫ f(T) dT = ∫ p(U) C(T) dT = ∫ p(U) dU = 1 ✓
    C_arr = np.array([C_of_T(t) for t in t_centers])      # eV/K
    f_T_raw = p * C_arr
    norm2   = float(np.trapezoid(f_T_raw, t_centers))
    if norm2 > 0:
        f_T_raw /= norm2

    return t_centers, f_T_raw
