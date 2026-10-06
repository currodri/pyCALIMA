"""Infrared cooling of a vibrationally hot PAH.

For a harmonic molecule at total vibrational energy E, the rate of emitting a
photon in mode i (fundamental A_i, the v -> v-1 rate being v A_i) is exactly

    k_i(E) = A_i sum_{v>=1} rho(E - v h nu_i) / rho(E),

because the probability that mode i holds v quanta is rho_{-i}(E - v h nu_i)/rho(E)
and sum_v v rho_{-i}(E - v h nu) = sum_{v>=1} rho(E - v h nu). Each emission
lowers E by h nu_i: this is the discrete IR cascade.

The thermal approximation (Andrews et al. 2016, eq. 3; the legacy
``pah_mol_data.compute_thermal_ir_rate``) replaces the ratio by the
canonical occupation n(T) at T = T_can(E): k_i = A_i n_i(T). It is the
large-molecule limit of the expression above.

Energies and frequencies are in cm^-1, on a ``dos.beyer_swinehart`` grid.
"""

from __future__ import annotations

import numpy as np

from pycalima.pahchem import dos


def _window_sum(values, half):
    total = np.concatenate([[0.0], np.cumsum(values)])
    i = np.arange(values.size)
    lo = np.clip(i - half, 0, values.size)
    hi = np.clip(i + half + 1, 0, values.size)
    return total[hi] - total[lo]


def mode_emission_rates(energy_cm, counts, freqs_cm, einstein_a, window_cm=100.0, at_cm=None):
    """Microcanonical photon-emission rate of each mode, k_i(E) [s^-1].

    Both sums are averaged over ``window_cm``, as for rho in ``dos``. With
    ``at_cm``, rates are returned only at those energies (nearest grid point),
    which keeps memory at n_modes x len(at_cm) for a fine grid.

    Returns
    -------
    ndarray, shape (n_modes, n_grid) or (n_modes, len(at_cm))
    """
    grain = energy_cm[1] - energy_cm[0]
    half = max(int(round(0.5 * window_cm / grain)), 1)
    n = counts.size
    pick = (np.arange(n) if at_cm is None
            else np.clip(np.rint(np.asarray(at_cm) / grain).astype(int), 0, n - 1))
    rho_w = _window_sum(counts, half)[pick]
    out = np.zeros((len(freqs_cm), pick.size))
    for m, (nu, A) in enumerate(zip(freqs_cm, einstein_a)):
        f = max(int(round(nu / grain)), 1)
        # S(E) = sum_{v>=1} c(E - v nu) obeys S(E) = c(E - nu) + S(E - nu)
        S = np.zeros(n)
        for start in range(f, n, f):
            stop = min(start + f, n)
            S[start:stop] = counts[start - f:stop - f] + S[start - f:stop - f]
        with np.errstate(divide="ignore", invalid="ignore"):
            out[m] = np.where(rho_w > 0, A * _window_sum(S, half)[pick] / rho_w, 0.0)
    return out


#: 8 pi^2 e^2 / (m_e c) [cm^2 s^-1]: A_n0 = this x nu_n^2 [cm^-2] x f_0n x g_0 / g_n
RF_A_PER_F = 8.0 * np.pi**2 * 4.80320471e-10**2 / (9.1093837e-28 * 2.99792458e10)


def recurrent_fluorescence_rates(energy_cm, counts, levels_cm, oscillator_strengths, window_cm=100.0, at_cm=None):
    """Recurrent-fluorescence rate of each electronic level, k_n(E) [s^-1].

    A molecule at vibrational energy E (in the ground electronic state's
    reckoning) is thermally in level n, E_n above, with probability
    (g_n/g_0) rho(E - E_n)/rho(E), taking the excited state's vibrational
    density as the ground state's (as Leger+88, Martin+13, Bernard+23). The
    level emits at A_n0 = RF_A_PER_F nu_n^2 f_0n g_0/g_n, so the degeneracies
    cancel:

        k_n(E) = RF_A_PER_F nu_n^2 f_0n rho(E - E_n) / rho(E),

    with f_0n the absorption oscillator strength. Each emission lowers E by
    E_n. Densities are averaged over ``window_cm`` as in
    ``mode_emission_rates``.
    """
    grain = energy_cm[1] - energy_cm[0]
    half = max(int(round(0.5 * window_cm / grain)), 1)
    n = counts.size
    pick = (np.arange(n) if at_cm is None
            else np.clip(np.rint(np.asarray(at_cm) / grain).astype(int), 0, n - 1))
    rho_w = _window_sum(counts, half)
    out = np.zeros((len(levels_cm), pick.size))
    for m, (e_n, f) in enumerate(zip(levels_cm, oscillator_strengths)):
        s = max(int(round(e_n / grain)), 1)
        shifted = np.zeros(n)
        shifted[s:] = rho_w[:n - s]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(rho_w > 0, shifted / rho_w, 0.0)
        out[m] = RF_A_PER_F * e_n**2 * f * ratio[pick]
    return out


def thermal_mode_rates(freqs_cm, einstein_a, temperature_k):
    """Thermal approximation k_i = A_i / (exp(h nu_i / kT) - 1) [s^-1]."""
    x = np.asarray(freqs_cm, dtype=float) / (dos.KB_CM * temperature_k)
    return np.asarray(einstein_a) / np.expm1(np.minimum(x, 700.0))


def cooling_rate(freqs_cm, mode_rates):
    """Radiated power dE/dt = sum_i h nu_i k_i [cm^-1 s^-1]."""
    return np.asarray(freqs_cm) @ np.asarray(mode_rates)
