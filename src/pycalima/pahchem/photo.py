"""Photon absorption by PAHs: cross sections, ionisation yields, heating.

An absorbed photon of energy E either ionises the PAH, with probability
Y_ion(E), or heats it. The rates are

  r_abs = int sigma F dE,   k_ion = int sigma Y F dE,
  heating spectrum h(E) = sigma (1 - Y) F   [s^-1 eV^-1],

where h is what the internal-energy master equation (``energy_me``) receives.
An ionising photon is taken to leave no internal energy, as in Montillaud et
al. (2013). Andrews et al. (2016) do not say; this is the one place where the
energy given to the electron would enter.

Cross sections [cm^2] on an energy grid in eV:

- ``sigma_li_draine``: Li & Draine (2001), neutral tables for Z <= 0 and
  ionised tables for Z > 0, at the LD01 radius a = 10 A (N_C/468)^(1/3). The
  legacy PAH_photophysics code used the disc radius 0.9 sqrt(N_C) A, which
  overestimates sigma by 1.7-3.3x for C24-C96.
- ``sigma_malloci_coronene``: Malloci et al. (2007) TD-DFT for C24H12 at
  Z = -1, 0, +1 (bundled files).
"""

from __future__ import annotations

import numpy as np

from pycalima import _paths

HC_EV_UM = 1.23984193

_MALLOCI = {-1: "malloci_coronene_anion_sigma.dat",
            0: "malloci_coronene_neutral_sigma.dat",
            1: "malloci_coronene_cation_sigma.dat"}


def li_draine_radius_um(nc: int) -> float:
    """Li & Draine (2001) PAH radius: 10 A (N_C/468)^(1/3), in micron."""
    return 1e-3 * (nc / 468.0) ** (1.0 / 3.0)


def sigma_li_draine(nc, charge, energy_ev) -> np.ndarray:
    """Li & Draine (2001) absorption cross section [cm^2]."""
    from pycalima.models.PAH_radiation.pah_oppacity import interpolate_pah_cross_sections_2d

    E = np.asarray(energy_ev, dtype=float)
    kind = "iPAH" if charge > 0 else "nPAH"
    _, _, _, c_abs, _ = interpolate_pah_cross_sections_2d(
        kind, li_draine_radius_um(nc), target_wavelengths=HC_EV_UM / E)
    return np.asarray(c_abs, dtype=float)


def sigma_malloci_coronene(charge, energy_ev) -> np.ndarray:
    """Malloci et al. (2007) C24H12 cross section [cm^2]; zero outside 0-30 eV."""
    if charge not in _MALLOCI:
        raise ValueError(f"Malloci coronene sigma exists for Z = -1, 0, 1, not {charge}")
    data = np.loadtxt(_paths.get_external_data_path(_MALLOCI[charge]), comments="#")
    return np.interp(energy_ev, data[:, 0], data[:, 1], left=0.0, right=0.0) * 1e-18


def yield_jochims1996(energy_ev, ip_ev) -> np.ndarray:
    """Linear rise from IP to unity at IP + 9.2 eV (Jochims et al. 1996)."""
    return np.clip((np.asarray(energy_ev, dtype=float) - ip_ev) / 9.2, 0.0, 1.0)


def yield_lepage2001(energy_ev, ip_ev, ip_coronene_ev=7.02) -> np.ndarray:
    """Y = 0.8 exp{-0.00128 [c (E - 14.89)]^4}, c = (14.89 - IP_cor)/(14.89 - IP).

    Le Page et al. (2001) as used by Montillaud et al. (2013, eq. 1); zero
    below IP.
    """
    E = np.asarray(energy_ev, dtype=float)
    c = (14.89 - ip_coronene_ev) / (14.89 - ip_ev)
    return np.where(E >= ip_ev, 0.8 * np.exp(-0.00128 * (c * (E - 14.89)) ** 4), 0.0)


def ionisation_rate(field, sigma_cm2, yield_, energy_ev) -> float:
    """k_ion = int sigma Y F dE [s^-1]."""
    return field.photon_rate(np.asarray(sigma_cm2) * np.asarray(yield_), energy_ev)


def heating_spectrum(field, sigma_cm2, yield_, energy_ev) -> np.ndarray:
    """h(E) = sigma (1 - Y) F [s^-1 eV^-1]: absorptions that heat, per photon energy."""
    return np.asarray(sigma_cm2) * (1.0 - np.asarray(yield_)) * field.flux_at(energy_ev)
