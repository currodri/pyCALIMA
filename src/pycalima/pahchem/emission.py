"""Infrared emission of a PAH from the microcanonical cascade (M7).

The cascade of ``energy_me`` already holds every IR emission rate k_i(E); this
module counts the photons:

- ``cascade_photons``: N_i(E_j), the photons emitted in mode i by a molecule
  that starts at E_j and cascades with no further absorption until it can no
  longer emit (or dissociates). It is exact for the discrete cascade of
  ``energy_me``, split jumps included, so sum_i h nu_i N_i(E_j) is E_j less
  the energy left when the cascade stops.
- ``single_photon_emission``: N_i averaged over the absorbed photon energies of
  a heating spectrum, for a molecule cold before each absorption (the
  single-photon limit).
- ``steady_state_emission``: R_i = sum_j P_j k_i(E_j) [photons s^-1] from a
  multiphoton ``energy_me.SteadyState``; sum_i h nu_i R_i is its radiated
  power.
- ``spectrum``: bands of a given width and shift on a wavenumber grid.

Energies and frequencies are in cm^-1.
"""

from __future__ import annotations

import numpy as np

from pycalima.pahchem import dos


def cascade_photons(model, channel_rates=None) -> np.ndarray:
    """N[i, j]: photons emitted in mode i from bin j of ``model`` (a CascadeModel).

    Rows follow ``model.freqs_cm``: the IR modes, then any recurrent-
    fluorescence levels (``model.n_ir`` IR rows; the rest are optical photons
    of energy E_n). ``channel_rates`` (name -> k_c on the grid) are dissociation sinks: a
    molecule that dissociates stops emitting (the fragment is not followed).
    """
    names, K = model._channels(channel_rates or {})
    k_out = K.sum(axis=0)
    N = np.zeros((model.freqs_cm.size, model.n))
    for j in range(model.n):
        km = model.k_modes[:, j]
        lo, hi, w = model._lo[:, j], model._hi[:, j], model._w[:, j]
        to_self = km * ((1.0 - w) * (lo == j) + w * (hi == j))
        denom = k_out[j] + km.sum() - to_self.sum()
        if denom <= 0.0:
            continue                                   # cannot emit: the cascade ends here
        lo_part = np.where(lo < j, (1.0 - w) * km, 0.0)
        hi_part = np.where(hi < j, w * km, 0.0)
        N[:, j] = (km + N[:, lo] @ lo_part + N[:, hi] @ hi_part) / denom
    return N


def single_photon_emission(model, photon_energy_ev, heating_spectrum, channel_rates=None) -> np.ndarray:
    """Photons emitted per mode per absorbed heating photon, from a cold start.

    The absorbed energy is split between the two grid bins around it, as
    ``energy_me`` does for up-jumps.
    """
    N = cascade_photons(model, channel_rates)
    e = np.asarray(photon_energy_ev, dtype=float) * dos.CM_PER_EV / model.de_cm
    h = np.asarray(heating_spectrum, dtype=float)
    lo = np.floor(e).astype(int)
    w = e - lo
    if lo.max() + 1 >= model.n:
        raise ValueError("photon energies reach the top of the grid; raise e_max_cm")
    weights = np.trapezoid(h, photon_energy_ev)
    return np.trapezoid(h * ((1.0 - w) * N[:, lo] + w * N[:, lo + 1]), photon_energy_ev, axis=1) / weights


class ThermalCascade:
    """The thermal-approximation cascade of the AmesPAHdb suites (Bauschlicher+18).

    A molecule given energy E reaches T_max with int_{2.73}^{T_max} C(T) dT = E
    (harmonic C), then cools through T, mode i radiating the fraction
    I_i nu_i^3 n_i(T) / sum_j I_j nu_j^3 n_j(T) of C(T) dT, with n the Planck
    occupation and I the band intensity (A_i / nu_i^2). It is the model of
    ``Transitions.cascade`` and of Maragkoudakis+20 eqs. A1-A4; pahchem's own
    physics is ``cascade_photons``. ``energy_per_mode(E)`` [cm^-1] sums to E.
    """

    def __init__(self, freqs_cm, einstein_a, t_min=2.73, t_max=6000.0, n_t=4000):
        nu = np.asarray(freqs_cm, dtype=float)
        intensity = np.asarray(einstein_a, dtype=float) / nu**2
        T = np.geomspace(t_min, t_max, n_t)
        x = nu[None, :] / (dos.KB_CM * T[:, None])
        with np.errstate(over="ignore"):
            occ = 1.0 / np.expm1(np.minimum(x, 700.0))
            heat = np.sum(x**2 * occ * (1.0 + occ), axis=1) * dos.KB_CM                    # C(T) [cm^-1/K]
        w = intensity * nu**3 * occ
        total = w.sum(axis=1, keepdims=True)
        frac = np.divide(w, total, out=np.zeros_like(w), where=total > 0)
        dT = np.diff(T)
        mid = 0.5 * (heat[1:, None] * frac[1:] + heat[:-1, None] * frac[:-1]) * dT[:, None]
        self.freqs_cm, self.temperature = nu, T
        self.cumulative = np.vstack([np.zeros(nu.size), np.cumsum(mid, axis=0)])   # [n_t, modes]
        self.energy = self.cumulative.sum(axis=1)                                 # U(T) - U(t_min)

    def energy_per_mode(self, energy_cm):
        """Energy emitted in each mode [cm^-1] after absorbing ``energy_cm``."""
        if energy_cm > self.energy[-1]:
            raise ValueError("energy beyond the temperature grid; raise t_max")
        k = np.searchsorted(self.energy, energy_cm)
        f = (energy_cm - self.energy[k - 1]) / (self.energy[k] - self.energy[k - 1])
        return (1 - f) * self.cumulative[k - 1] + f * self.cumulative[k]


def steady_state_emission(model, steady) -> np.ndarray:
    """R_i = sum_j P_j k_i(E_j): photon emission rate of each mode [s^-1].

    Rows follow ``model.freqs_cm`` (IR modes, then recurrent-fluorescence levels).
    """
    return model.k_modes @ steady.probability


def spectrum(freqs_cm, band_strength, grid_cm, fwhm_cm=15.0, shift_cm=-15.0, profile="lorentzian"):
    """Sum of bands of strength ``band_strength`` (any unit) per cm^-1 on ``grid_cm``.

    Defaults are the PAHdb convention (Bauschlicher+18): Lorentzian, FWHM
    15 cm^-1, redshift 15 cm^-1 for anharmonicity. Each band integrates to its
    strength over an infinite grid.
    """
    nu0 = np.asarray(freqs_cm, dtype=float)[:, None] + shift_cm
    x = np.asarray(grid_cm, dtype=float)[None, :] - nu0
    if profile == "lorentzian":
        g = 0.5 * fwhm_cm / np.pi / (x**2 + (0.5 * fwhm_cm) ** 2)
    elif profile == "gaussian":
        s = fwhm_cm / np.sqrt(8.0 * np.log(2.0))
        g = np.exp(-0.5 * (x / s) ** 2) / (s * np.sqrt(2.0 * np.pi))
    else:
        raise ValueError(f"profile must be 'lorentzian' or 'gaussian', not {profile!r}")
    return np.asarray(band_strength, dtype=float) @ g
