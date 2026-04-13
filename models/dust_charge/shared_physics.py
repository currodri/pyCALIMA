"""Shared cgs charging/photoelectric helper physics.

This module centralizes low-level routines used by both dust charging and
photoelectric heating implementations.
"""

import numpy as np

try:
    from numba import njit
except Exception:  # pragma: no cover - optional acceleration
    njit = None

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


def _ds87_j_scalar_impl(Z, q, a, T):
    nu = Z / q
    denom_q2 = (q * q) * (E_STATC * E_STATC)
    tau = (a * KB_CGS * T) / max(denom_q2, TINY)

    if nu == 0.0:
        return 1.0 + np.sqrt(np.pi / (2.0 * max(tau, TINY)))

    if nu < 0.0:
        tn = max(tau, TINY)
        inner = max(tn - 2.0 * nu, TINY)
        return (1.0 - nu / tn) * (1.0 + np.sqrt(2.0 / inner))

    tp = max(tau, TINY)
    nup = max(nu, TINY)
    theta_nu = 1.0 / (1.0 + 1.0 / np.sqrt(nup))
    root_term = 1.0 / np.sqrt(4.0 * tp + 3.0 * nup)
    value = (1.0 + root_term) ** 2 * np.exp(-theta_nu / tp)
    return value if np.isfinite(value) else 0.0


if njit is not None:
    _ds87_j_scalar_impl = njit(cache=True)(_ds87_j_scalar_impl)


def _electron_sticking_coefficient_graphite_scalar_impl(Z, a):
    Nc = 468.0 * (a / 1e-7) ** 3
    base = 0.5 * (1.0 - np.exp(-a / ELECTRON_ESCAPE_LENGTH_CM))
    factor = 1.0 / (1.0 + np.exp(20.0 - Nc))
    zmin = np.floor(-(3.9 + 1.2e7 * a + 2.0e-8 / max(a, TINY)) / 14.4 * 1.0e8 * a)

    if Z == 0:
        return base * factor
    if (Z < 0) and (Z > zmin):
        return base * factor
    if Z > 0:
        return base
    return 0.0


def _electron_sticking_coefficient_silicate_scalar_impl(Z, a):
    Nc = 468.0 * (a / 1e-7) ** 3
    base = 0.5 * (1.0 - np.exp(-a / ELECTRON_ESCAPE_LENGTH_CM))
    factor = 1.0 / (1.0 + np.exp(20.0 - Nc))
    zmin = np.floor(-(2.5 + 7.0e6 * a + 8.0e-8 / max(a, TINY)) / 14.4 * 1.0e8 * a)

    if Z == 0:
        return base * factor
    if (Z < 0) and (Z > zmin):
        return base * factor
    if Z > 0:
        return base
    return 0.0


if njit is not None:
    _electron_sticking_coefficient_graphite_scalar_impl = njit(cache=True)(_electron_sticking_coefficient_graphite_scalar_impl)
    _electron_sticking_coefficient_silicate_scalar_impl = njit(cache=True)(_electron_sticking_coefficient_silicate_scalar_impl)


def DS87_J_function_scalar(Z, q, a, T):
    return float(_ds87_j_scalar_impl(float(Z), float(q), float(a), float(T)))


def DS87_lambda_scalar(Z, q, a, T):
    nu = float(Z) / float(q)
    denom = (float(q) ** 2.0) * (E_STATC ** 2.0)
    tau = (float(a) * KB_CGS * float(T)) / max(denom, TINY)
    if nu == 0.0:
        return float(2.0 + 1.5 * np.sqrt(np.pi / (2.0 * max(tau, TINY))))
    if nu < 0.0:
        tn = max(tau, TINY)
        inner = max(tn - nu, TINY)
        return float((2.0 - nu / tn) * (1.0 + 1.0 / np.sqrt(inner)))
    tp = max(tau, TINY)
    nup = max(nu, TINY)
    theta_nu = 1.0 / (1.0 + 1.0 / np.sqrt(nup))
    return float((2.0 + nup / tp) * (1.0 + 1.0 / np.sqrt(1.5 / tp + 3.0 * nup)) * np.exp(-theta_nu / tp))


def electron_sticking_coefficient_graphite_scalar(Z, a):
    return float(_electron_sticking_coefficient_graphite_scalar_impl(float(Z), float(a)))


def electron_sticking_coefficient_silicate_scalar(Z, a):
    return float(_electron_sticking_coefficient_silicate_scalar_impl(float(Z), float(a)))


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


def min_energy_ejection_scalar(Z, a):
    att = 1.0 + np.power(27e-8 / float(a), 0.75)
    if float(Z) >= 0.0:
        return 0.0
    return float(-(float(Z) + 1.0) * (E_STATC ** 2.0) / (float(a) * att) / EV2ERG)


def photodetachment_energy_graphite_vec(Z, a):
    return electron_affinity_graphite_vec(Z + 1, a) + min_energy_ejection_vec(Z, a)


def photodetachment_energy_graphite_scalar(Z, a):
    return float(electron_affinity_graphite_vec(float(Z) + 1.0, float(a)) + min_energy_ejection_scalar(float(Z), float(a)))


def photodetachment_energy_silicate_vec(Z, a):
    return electron_affinity_silicate_vec(Z + 1, a) + min_energy_ejection_vec(Z, a)


def photodetachment_energy_silicate_scalar(Z, a):
    return float(electron_affinity_silicate_vec(float(Z) + 1.0, float(a)) + min_energy_ejection_scalar(float(Z), float(a)))


def photodetachment_cross_section_vec(E, E_det, Z):
    x = (E - E_det) / 3.0
    sigma = 1.2e-17 * np.abs(Z) * x / np.power(1.0 + (x * x) / 3.0, 2.0)
    return np.where(x < 0.0, 0.0, sigma)


def min_photon_energy_vec(IPV, Z, a):
    emin = min_energy_ejection_vec(Z, a)
    return np.where(Z >= -1, IPV, IPV + emin)


def parameter_theta_vec(E, Emin_ej, Z, a, coulomb_over_a=None):
    coul = _coulomb_energy_over_a(Z, a) if coulomb_over_a is None else coulomb_over_a
    return E - Emin_ej + np.where(Z >= 0, coul, 0.0)


def escape_fraction_attempting_electrons_vec(hnu, Emin_ej, Z, a, coulomb_over_a=None):
    hnu = np.asarray(hnu, dtype=float)
    Emin_ej = np.asarray(Emin_ej, dtype=float)
    Z = np.asarray(Z, dtype=float)
    a = np.asarray(a, dtype=float)

    if hnu.size == 1 and Emin_ej.size == 1 and Z.size == 1 and a.size == 1:
        if coulomb_over_a is None:
            elow = -float(_coulomb_energy_over_a(Z.reshape(-1)[0], a.reshape(-1)[0]))
        else:
            elow = -float(np.asarray(coulomb_over_a, dtype=float).reshape(-1)[0])
        ehigh = float(hnu.reshape(-1)[0] - Emin_ej.reshape(-1)[0])
        if float(Z.reshape(-1)[0]) < 0.0:
            return 1.0
        denom = max((ehigh - elow) ** 3.0, TINY)
        y2_pos = (ehigh ** 2.0) * (ehigh - 3.0 * elow) / denom
        return float(min(max(y2_pos, 0.0), 1.0))

    coul = _coulomb_energy_over_a(Z, a) if coulomb_over_a is None else coulomb_over_a
    elow = -coul
    ehigh = hnu - Emin_ej
    denom = np.maximum((ehigh - elow) ** 3.0, TINY)
    y2_pos = (ehigh ** 2.0) * (ehigh - 3.0 * elow) / denom
    result = np.ones_like(y2_pos, dtype=float)
    mask = np.broadcast_to(Z >= 0, y2_pos.shape)
    if np.any(mask):
        result[mask] = np.clip(y2_pos[mask], 0.0, 1.0)
    return result


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
    a = np.asarray(a, dtype=float)
    T = np.asarray(T, dtype=float)

    if Z.size == 1 and q.size == 1 and a.size == 1 and T.size == 1:
        return np.asarray(
            [DS87_J_function_scalar(float(Z.reshape(-1)[0]), float(q.reshape(-1)[0]), float(a.reshape(-1)[0]), float(T.reshape(-1)[0]))],
            dtype=float,
        ).reshape(np.broadcast(Z, q, a, T).shape)

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
    if Z.size == 1 and a.size == 1:
        value = electron_sticking_coefficient_graphite_scalar(float(Z.reshape(-1)[0]), float(a.reshape(-1)[0]))
        return np.asarray(value, dtype=float).reshape(np.broadcast(Z, a).shape)
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
    if Z.size == 1 and a.size == 1:
        value = electron_sticking_coefficient_silicate_scalar(float(Z.reshape(-1)[0]), float(a.reshape(-1)[0]))
        return np.asarray(value, dtype=float).reshape(np.broadcast(Z, a).shape)
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
    a = np.asarray(a, dtype=float)
    Zs = np.asarray(Zs, dtype=float)

    if a.size == 1 and Zs.size == 1:
        return np.asarray(
            [collisional_rates_electrons_scalar(float(a.reshape(-1)[0]), float(Zs.reshape(-1)[0]), n_e, T_e, s_e_func)],
            dtype=float,
        )

    vth = np.sqrt(8.0 * KB_CGS * T_e / (np.pi * ME_CGS))
    cross = np.pi * a * a
    Jtilde = DS87_J_function_vec(Zs, np.array([-1.0]), a, T_e)
    s = s_e_func(Zs, a)
    return np.asarray(s * cross * n_e * vth * Jtilde, dtype=float)


def collisional_rates_electrons_scalar(a, Z, n_e, T_e, s_e_func):
    if s_e_func is electron_sticking_coefficient_graphite:
        s = electron_sticking_coefficient_graphite_scalar(Z, a)
    elif s_e_func is electron_sticking_coefficient_silicate:
        s = electron_sticking_coefficient_silicate_scalar(Z, a)
    elif s_e_func is electron_sticking_coefficient_graphite_scalar:
        s = electron_sticking_coefficient_graphite_scalar(Z, a)
    elif s_e_func is electron_sticking_coefficient_silicate_scalar:
        s = electron_sticking_coefficient_silicate_scalar(Z, a)
    else:
        s = float(np.asarray(s_e_func(np.array([Z], dtype=float), np.array([a], dtype=float)), dtype=float).reshape(-1)[0])

    vth = np.sqrt(8.0 * KB_CGS * T_e / (np.pi * ME_CGS))
    cross = np.pi * a * a
    Jtilde = DS87_J_function_scalar(Z, -1.0, a, T_e)
    return float(s * cross * n_e * vth * Jtilde)


def collisional_rates_ions_vector(a, Zs, ion_species):
    Zs = np.asarray(Zs, dtype=float)
    a = np.asarray(a, dtype=float)

    if not ion_species:
        return np.zeros_like(Zs, dtype=float)

    if a.size == 1 and Zs.size == 1:
        return np.asarray([collisional_rates_ions_scalar(float(a.reshape(-1)[0]), float(Zs.reshape(-1)[0]), ion_species)], dtype=float)

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


def collisional_rates_ions_scalar(a, Z, ion_species):
    if not ion_species:
        return 0.0

    cross = np.pi * a * a
    total = 0.0
    for ion in ion_species:
        n_i = float(ion.get("n", 0.0))
        T_i = float(ion.get("T", 1.0))
        m_g = float(ion.get("m", 1.0))
        z_i = float(ion.get("z", 1.0))

        vth_i = np.sqrt(8.0 * KB_CGS * T_i / (np.pi * m_g))
        total += cross * n_i * vth_i * DS87_J_function_scalar(Z, z_i, a, T_i)

    return float(total)
