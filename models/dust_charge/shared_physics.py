"""Shared cgs charging/photoelectric helper physics.

This module centralizes low-level routines used by both dust charging and
photoelectric heating implementations.
"""

import numpy as np

# Shared constants
GRAPHITE_WORK_FUNCTION = 4.4
SILICATE_WORK_FUNCTION = 8.0
SILICATE_BAND_GAP = 5.0
ELECTRON_ESCAPE_LENGTH_CM = 1e-7
ME_CGS = 9.1093837015e-28
H_CGS = 6.62607015e-27
C_CGS = 2.99792458e10
KB_CGS = 1.380649e-16
EV2ERG = 1.602176634e-12
E_STATC = 4.8032047e-10
TINY = 1e-300


def _coulomb_energy_over_a(Z, a):
    return (E_STATC ** 2.0) * (Z + 1.0) / np.maximum(a, TINY) / EV2ERG


def ionisation_potential_valence_vec(W, Z, a):
    return W + (E_STATC ** 2.0) / a * ((Z + 0.5) + (Z + 2.0) * (0.3e-8 / a)) / EV2ERG


def electron_affinity_graphite_vec(Z, a):
    return GRAPHITE_WORK_FUNCTION + (E_STATC ** 2.0) / a * ((Z - 0.5) - (4e-8 / (a + 7e-8))) / EV2ERG


def electron_affinity_silicate_vec(Z, a):
    return SILICATE_WORK_FUNCTION - SILICATE_BAND_GAP + (E_STATC ** 2.0) / a * (Z - 0.5) / EV2ERG


def min_energy_ejection_vec(Z, a):
    att = 1.0 + np.power(27e-8 / a, 0.75)
    emin_neg = -(Z + 1.0) * (E_STATC ** 2.0) / (a * att) / EV2ERG
    return np.where(Z >= 0, 0.0, emin_neg)


def photodetachment_energy_graphite_vec(Z, a):
    return electron_affinity_graphite_vec(Z + 1, a) + min_energy_ejection_vec(Z, a)


def photodetachment_energy_silicate_vec(Z, a):
    return electron_affinity_silicate_vec(Z + 1, a) + min_energy_ejection_vec(Z, a)


def photodetachment_cross_section_vec(E, E_det, Z):
    x = (E - E_det) / 3.0
    sigma = 1.2e-17 * np.abs(Z) * x / np.power(1.0 + (x * x) / 3.0, 2.0)
    return np.where(x < 0.0, 0.0, sigma)


def min_photon_energy_vec(IPV, Z, a):
    emin = min_energy_ejection_vec(Z, a)
    return np.where(Z >= -1, IPV, IPV + emin)


def parameter_theta_vec(E, Emin_ej, Z, a):
    coul = _coulomb_energy_over_a(Z, a)
    return E - Emin_ej + np.where(Z >= 0, coul, 0.0)


def escape_fraction_attempting_electrons_vec(hnu, Emin_ej, Z, a):
    elow = -_coulomb_energy_over_a(Z, a)
    ehigh = hnu - Emin_ej
    denom = np.maximum((ehigh - elow) ** 3.0, TINY)
    y2_pos = (ehigh ** 2.0) * (ehigh - 3.0 * elow) / denom
    return np.where(Z >= 0, np.clip(y2_pos, 0.0, 1.0), 1.0)


def photon_attenuation_length_graphite_vec(wav, Imperp, Impar):
    l_inv = (4.0 * np.pi / wav) * ((2.0 / 3.0) * Imperp + (1.0 / 3.0) * Impar)
    return 1.0 / np.maximum(l_inv, TINY)


def photon_attenuation_length_silicate_vec(wav, Im):
    return wav / np.maximum(4.0 * np.pi * Im, TINY)


def Watson73_y1_vec(a, la, le):
    beta = a / la
    alpha = a / le + a / la
    num = (beta / alpha) ** 2.0 * (alpha ** 2 - 2.0 * alpha + 2.0 - 2.0 * np.exp(-alpha))
    den = np.maximum(beta ** 2 - 2.0 * beta + 2.0 - 2.0 * np.exp(-beta), TINY)
    return num / den


def BT94_y0_graphite_vec(theta, W):
    x = np.maximum(theta / np.maximum(W, TINY), 0.0)
    return (9e-3 * x ** 5) / (1.0 + 3.7e-2 * x ** 5)


def y0_silicate_vec(theta, W):
    x = np.maximum(theta / np.maximum(W, TINY), 0.0)
    return 0.5 * x / (1.0 + 5.0 * x)


def DS87_J_function_vec(Z, q, a, T):
    Z = np.asarray(Z, dtype=float)
    q = np.asarray(q, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        nu = Z / q

    denom_q2 = (q ** 2) * (E_STATC ** 2)
    tau = (a * KB_CGS * T) / np.maximum(denom_q2, TINY)
    try:
        tau = np.broadcast_to(tau, np.shape(nu))
    except Exception:
        tau = np.asarray(tau)

    tau_safe = np.maximum(tau, TINY)
    J = np.zeros_like(nu, dtype=float)

    nu_zero = nu == 0.0
    nu_neg = nu < 0.0
    nu_pos = nu > 0.0

    if np.any(nu_zero):
        J[nu_zero] = 1.0 + np.sqrt(np.pi / (2.0 * tau_safe[nu_zero]))

    if np.any(nu_neg):
        tn = tau_safe[nu_neg]
        nun = nu[nu_neg]
        inner = np.maximum(tn - 2.0 * nun, TINY)
        J[nu_neg] = (1.0 - nun / tn) * (1.0 + np.sqrt(2.0 / inner))

    if np.any(nu_pos):
        tp = tau_safe[nu_pos]
        nup = np.maximum(nu[nu_pos], TINY)
        theta_nu = 1.0 / (1.0 + 1.0 / np.sqrt(nup))
        root_term = 1.0 / np.sqrt(4.0 * tp + 3.0 * nup)
        J[nu_pos] = (1.0 + root_term) ** 2 * np.exp(-theta_nu / tp)

    return np.where(np.isfinite(J), J, 0.0)


def autoionisation_potential_graphite(a):
    a = np.asarray(a, dtype=float)
    return 3.9 + 1.2e7 * a + 2.0e-8 / np.maximum(a, TINY)


def autoionisation_potential_silicate(a):
    a = np.asarray(a, dtype=float)
    return 2.5 + 7.0e6 * a + 8.0e-8 / np.maximum(a, TINY)


def most_negative_allowed_charge_graphite(a):
    a = np.asarray(a, dtype=float)
    return np.floor(-autoionisation_potential_graphite(a) / 14.4 * 1.0e8 * a)


def most_negative_allowed_charge_silicate(a):
    a = np.asarray(a, dtype=float)
    return np.floor(-autoionisation_potential_silicate(a) / 14.4 * 1.0e8 * a)


def most_positive_allowed_charge(a, W, hnu_max):
    a = np.asarray(a, dtype=float)
    W = np.asarray(W, dtype=float)
    a_safe = np.maximum(a, TINY)
    return np.floor(((hnu_max - W) / 14.4 * 1.0e8 * a_safe + 0.5 - 3.0e-9 / a_safe) / (1.0 + 3.0e-9 / a_safe))


def electron_sticking_coefficient_graphite(Z, a):
    Z = np.asarray(Z, dtype=float)
    a = np.asarray(a, dtype=float)
    Nc = 468.0 * (a / 1e-7) ** 3
    base = 0.5 * (1.0 - np.exp(-a / ELECTRON_ESCAPE_LENGTH_CM))
    factor = 1.0 / (1.0 + np.exp(20.0 - Nc))
    zmin = most_negative_allowed_charge_graphite(a)

    s = np.zeros_like(Z)
    mask0 = Z == 0
    mask_neg = (Z < 0) & (Z > zmin)
    mask_pos = Z > 0

    bf = base * factor
    if np.ndim(base) == 0 or base.shape == ():
        s[mask0] = bf
        s[mask_neg] = bf
        s[mask_pos] = base
    else:
        s[mask0] = bf.reshape((-1,))[0] if np.size(base) == 1 else bf
        s[mask_neg] = bf.reshape((-1,))[0] if np.size(base) == 1 else bf
        s[mask_pos] = base.reshape((-1,))[0] if np.size(base) == 1 else base
    return s


def electron_sticking_coefficient_silicate(Z, a):
    Z = np.asarray(Z, dtype=float)
    a = np.asarray(a, dtype=float)
    Nc = 468.0 * (a / 1e-7) ** 3
    base = 0.5 * (1.0 - np.exp(-a / ELECTRON_ESCAPE_LENGTH_CM))
    factor = 1.0 / (1.0 + np.exp(20.0 - Nc))
    zmin = most_negative_allowed_charge_silicate(a)

    s = np.zeros_like(Z)
    mask0 = Z == 0
    mask_neg = (Z < 0) & (Z > zmin)
    mask_pos = Z > 0

    bf = base * factor
    if np.ndim(base) == 0 or base.shape == ():
        s[mask0] = bf
        s[mask_neg] = bf
        s[mask_pos] = base
    else:
        s[mask0] = bf.reshape((-1,))[0] if np.size(base) == 1 else bf
        s[mask_neg] = bf.reshape((-1,))[0] if np.size(base) == 1 else bf
        s[mask_pos] = base.reshape((-1,))[0] if np.size(base) == 1 else base
    return s


def collisional_rates_electrons_vector(a, Zs, n_e, T_e, s_e_func):
    vth = np.sqrt(8.0 * KB_CGS * T_e / (np.pi * ME_CGS))
    cross = np.pi * a * a
    Jtilde = DS87_J_function_vec(Zs, np.array([-1.0]), a, T_e)
    s = s_e_func(Zs, a)
    return np.asarray(s * cross * n_e * vth * Jtilde, dtype=float)


def collisional_rates_ions_vector(a, Zs, ion_species):
    Zs = np.asarray(Zs, dtype=float)
    a = np.asarray(a, dtype=float)
    cross = np.pi * a * a

    J_total = np.zeros_like(Zs, dtype=float)
    for ion in ion_species:
        n_i = float(ion.get("n", 0.0))
        T_i = float(ion.get("T", 1.0))
        m_g = float(ion.get("m", 1.0))
        z_i = float(ion.get("z", 1.0))

        vth_i = np.sqrt(8.0 * KB_CGS * T_i / (np.pi * m_g))
        Jtilde_i = DS87_J_function_vec(Zs, np.array([z_i]), a, T_i)
        J_total = J_total + cross * n_i * vth_i * Jtilde_i

    return np.asarray(J_total, dtype=float)
