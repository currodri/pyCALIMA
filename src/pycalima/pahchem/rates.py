"""Unimolecular dissociation rate constants k(E).

Four forms, from the most to the least physical:

- ``rrkm``: k(E) = sigma N_TS(E - E0) / (h rho(E)), with the sum of states of
  an explicit transition state (its s - 1 frequencies).
- ``inverse_laplace_arrhenius``: k(E) = A rho(E - E0) / rho(E), the exact
  microcanonical inverse of a canonical Arrhenius law A exp(-E0/kT)
  (Montillaud et al. 2013, eq. 3).
- ``arrhenius_effective_temperature``: k = e (k T_e/h) exp(dS/R)
  exp(-E0/k T_e), with T_e = T(E)(1 - 0.2 E0/E) and T(E) the canonical
  temperature of energy E (Tielens 2005; Andrews et al. 2016). It is an
  approximation to the inverse Laplace form, kept to reproduce papers that
  use it.
- ``rrk``: classical k(E) = A ((E - E0)/E)^(s - 1).

``transition_state_for_entropy`` builds a transition state with a given
activation entropy dS (at 1000 K by convention), the parameter that iPEPICO
experiments report.

Energies are in cm^-1 above the zero point, as in ``pahchem.dos``; the
exact forms work on a Beyer-Swinehart grid (``energy_cm``, ``counts``).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from pycalima.pahchem import dos

C_CM_S = 2.99792458e10        # speed of light [cm s^-1]; 1/h = c in cm^-1 s
H_J_S = 6.62607015e-34
KB_J_K = 1.380649e-23
R_J_MOL_K = 8.314462618


def _sum_of_states(energy_cm, counts):
    return np.cumsum(counts)


def _shift(values, n_shift):
    """values[i - n_shift], zero where i < n_shift."""
    out = np.zeros_like(values)
    if n_shift < values.size:
        out[n_shift:] = values[:values.size - n_shift]
    return out


def rrkm(energy_cm, counts, ts_freqs_cm, E0_cm, degeneracy=1.0, window_cm=100.0):
    """RRKM rate constant on the reactant's grid [s^-1].

    Parameters
    ----------
    energy_cm, counts : ndarray
        Beyer-Swinehart grid of the reactant (``dos.beyer_swinehart``).
    ts_freqs_cm : array_like
        The s - 1 real frequencies of the transition state [cm^-1].
    E0_cm : float
        ZPE-corrected barrier [cm^-1].
    degeneracy : float
        Reaction-path degeneracy sigma.
    window_cm : float
        Window over which rho(E) is averaged.
    """
    grain = energy_cm[1] - energy_cm[0]
    _, ts_counts = dos.beyer_swinehart(ts_freqs_cm, energy_cm[-1], grain)
    n_ts = _shift(np.cumsum(ts_counts), int(round(E0_cm / grain)))
    rho = np.exp(dos.smoothed_log_density(energy_cm, counts, window_cm))
    with np.errstate(divide="ignore", invalid="ignore"):
        k = degeneracy * C_CM_S * n_ts / rho
    return np.where(rho > 0, k, 0.0)       # no reactant states in the window


def inverse_laplace_arrhenius(energy_cm, counts, A_s, E0_cm, window_cm=100.0):
    """k(E) = A rho(E - E0) / rho(E) [s^-1] on the reactant's grid."""
    grain = energy_cm[1] - energy_cm[0]
    log_rho = dos.smoothed_log_density(energy_cm, counts, window_cm)
    i0 = int(round(E0_cm / grain))
    k = np.zeros_like(log_rho)
    k[i0:] = A_s * np.exp(log_rho[:log_rho.size - i0] - log_rho[i0:])
    return k


def arrhenius_effective_temperature(freqs_cm, energy_cm, E0_cm, dS_J_mol_K):
    """Tielens (2005) k = e (k T_e/h) exp(dS/R) exp(-E0/k T_e) [s^-1].

    T_e = T(E)(1 - 0.2 E0/E), with T(E) the canonical temperature at which
    the thermal energy U(T) equals E. Zero for E <= E0.
    """
    E = np.atleast_1d(np.asarray(energy_cm, dtype=float))
    k = np.zeros_like(E)
    above = E > E0_cm
    T_e = dos.canonical_temperature(freqs_cm, E[above]) * (1.0 - 0.2 * E0_cm / E[above])
    k[above] = (np.e * KB_J_K * T_e / H_J_S * np.exp(dS_J_mol_K / R_J_MOL_K)
                * np.exp(-E0_cm / (dos.KB_CM * T_e)))
    return k if np.ndim(energy_cm) else k[0]


def rrk(energy_cm, n_modes, A_s, E0_cm):
    """Classical RRK k(E) = A ((E - E0)/E)^(s - 1) [s^-1]."""
    E = np.asarray(energy_cm, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        k = A_s * np.clip((E - E0_cm) / E, 0.0, None) ** (n_modes - 1)
    return np.where(E > E0_cm, k, 0.0)


def vibrational_entropy(freqs_cm, temperature_k):
    """Harmonic vibrational entropy S/R (dimensionless) at T."""
    x = np.asarray(freqs_cm, dtype=float) / (dos.KB_CM * temperature_k)
    return float(np.sum(x / np.expm1(x) - np.log(-np.expm1(-x))))


def transition_state_for_entropy(freqs_cm, coordinate_cm, dS_J_mol_K, temperature_k=1000.0,
                                 n_scaled=None):
    """A transition state with activation entropy dS at ``temperature_k``.

    Removes the reaction coordinate (the mode nearest ``coordinate_cm``, e.g.
    a C-H stretch for H loss) and scales the ``n_scaled`` lowest remaining
    frequencies (all of them if None) by one factor, chosen so that
    S_TS - S_reactant = dS. ``n_scaled=5`` is the convention of the iPEPICO
    fits of West et al. (2018), whose dS are quoted at 1000 K.

    Returns
    -------
    ndarray
        The s - 1 transition-state frequencies [cm^-1], sorted.
    """
    freqs = np.sort(np.asarray(freqs_cm, dtype=float))
    rest = np.delete(freqs, np.argmin(np.abs(freqs - coordinate_cm)))
    n = rest.size if n_scaled is None else int(n_scaled)
    target = vibrational_entropy(freqs, temperature_k) + dS_J_mol_K / R_J_MOL_K

    def scaled(log_f):
        out = rest.copy()
        out[:n] *= np.exp(log_f)
        return out

    def mismatch(log_f):
        return vibrational_entropy(scaled(log_f), temperature_k) - target

    return np.sort(scaled(brentq(mismatch, np.log(1e-4), np.log(20.0))))


def thermal_rate(energy_cm, counts, k_E, temperature_k):
    """Canonical average <k>(T) = sum k(E) rho(E) exp(-E/kT) / Q(T) [s^-1].

    For an exact microcanonical k(E) this reproduces transition-state theory
    (RRKM) or the Arrhenius law (inverse Laplace form) at every T.
    """
    w = counts * np.exp(-(energy_cm - energy_cm[0]) / (dos.KB_CM * temperature_k))
    return float(np.sum(k_E * w) / np.sum(w))
