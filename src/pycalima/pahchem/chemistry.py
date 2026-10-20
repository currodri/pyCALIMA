"""Collisional rate coefficients for the (Z, N_H) network [cm^3 s^-1].

Multiply by the partner's density to get a rate in s^-1.

- ``thermal_speed``: mean speed sqrt(8 k T / pi m).
- ``cross_section_rate``: sigma v exp(-E_b/kT), e.g. H addition over a
  barrier, or Eley-Rideal abstraction. With sigma = 0.06 A^2 (Mennella et al.
  2012) this gives Andrews et al. (2016) eq. 10, 8.7e-13 sqrt(T/100) n_H.
- ``electron_capture``: Bakes & Tielens (1994) capture of an electron by a
  PAH of charge Z, s_e pi a^2 v_e J~(Z) (Draine & Sutin 1987 J~). Defaults:
  a = 0.9 sqrt(N_C) A and the 0.82 disc factor of Verstraete et al. (1990), as
  in Andrews et al. (2016, eq. A.1).
- ``lab_recombination_factor``: the measured PAH-cation recombination
  coefficient over BT94, as a power law in N_C fitted to the FALP data
  (pah_electron_recombination.csv, N_C 6-16, 300 K).
"""

from __future__ import annotations

import numpy as np

KB_CGS = 1.380649e-16
AMU_G = 1.66053907e-24
ME_G = 9.1093837e-28
KB_EV = 8.617333262e-5


def thermal_speed(temperature_k, mass_g):
    """Mean thermal speed [cm s^-1]."""
    return np.sqrt(8.0 * KB_CGS * np.asarray(temperature_k) / (np.pi * mass_g))


def cross_section_rate(temperature_k, sigma_a2, barrier_ev=0.0, partner_mass_amu=1.00794):
    """sigma v exp(-E_b / kT) [cm^3 s^-1] for a cross section in A^2."""
    v = thermal_speed(temperature_k, partner_mass_amu * AMU_G)
    return sigma_a2 * 1e-16 * v * np.exp(-barrier_ev / (KB_EV * np.asarray(temperature_k)))


def disc_radius_cm(nc):
    """PAH radius 0.9 sqrt(N_C) A used for charging (Bakes & Tielens 1994)."""
    return 0.9e-8 * np.sqrt(nc)


def electron_capture(nc, charge, temperature_k, sticking=1.0, disc_factor=0.82, radius_cm=None):
    """Electron capture coefficient by a PAH of charge Z [cm^3 s^-1]."""
    from pycalima.models.PAH_photophysics.pah_charge_utils import J_function_DS87

    a = disc_radius_cm(nc) if radius_cm is None else radius_cm
    J = J_function_DS87(int(charge), -1, a, float(temperature_k))
    return disc_factor * sticking * np.pi * a**2 * thermal_speed(temperature_k, ME_G) * J


_LAB_FIT = {}


def lab_recombination_fit():
    """(a, b, n) of log10(k_measured / k_BT94) = a + b log10(N_C) over the FALP data at 300 K.

    Every measurement in pah_electron_recombination.csv counts once (the two
    naphthalene values, 3x apart, both). BT94 is ``electron_capture`` with
    sticking 1 at 300 K.
    """
    if not _LAB_FIT:
        import pandas as pd

        from pycalima._paths import get_external_data_path

        t = pd.read_csv(get_external_data_path("pahchem/pah_electron_recombination.csv"), comment="#")
        bt = np.array([electron_capture(n, 1, 300.0) for n in t.n_c])
        b, a = np.polyfit(np.log10(t.n_c.to_numpy(float)), np.log10(t.k_cm3_s.to_numpy() / bt), 1)
        _LAB_FIT.update(a=float(a), b=float(b), n=int(len(t)))
    return _LAB_FIT["a"], _LAB_FIT["b"], _LAB_FIT["n"]


def lab_recombination_factor(nc, cap=None):
    """k_measured / k_BT94 at N_C from ``lab_recombination_fit``, optionally capped at ``cap``.

    The fit spans N_C 6-16 only. Beyond that it is an extrapolation (about 2
    at N_C = 100), and ``cap`` bounds it. The factor is T-independent: the data
    are at 300 K.
    """
    a, b, _ = lab_recombination_fit()
    f = 10.0 ** (a + b * np.log10(np.asarray(nc, dtype=float)))
    return np.minimum(f, cap) if cap is not None else f
