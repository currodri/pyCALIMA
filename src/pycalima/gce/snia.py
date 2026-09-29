"""Type Ia supernovae: W7 ejecta and a power-law delay-time distribution."""

from __future__ import annotations

import numpy as np

from .stellar_yields import ELEMENTS

# Nomoto+1984 / Iwamoto+1999 W7 isotopic ejecta [Msun per SN], in the order
# of cmp_typeia_yield_release.pro (12C, 13C, 14N, 15N, 16O, 17O, 18O, 19F,
# 20Ne, 21Ne, 22Ne, 23Na, 24Mg, ..., 67Zn, 68Zn).
_W7_ISOTOPES = np.array([
    4.83e-02, 1.40e-06, 1.16e-06, 1.32e-09, 1.43e-01, 3.54e-08, 8.25e-10, 5.67e-10,
    2.02e-03, 8.46e-06, 2.49e-03, 6.32e-05, 8.50e-03, 4.05e-05, 3.18e-05, 9.86e-04,
    1.50e-01, 8.61e-04, 1.74e-03, 4.18e-04, 8.41e-02, 4.50e-04, 1.90e-03, 3.15e-07,
    1.34e-04, 3.98e-05, 1.49e-02, 1.06e-03, 1.26e-08, 8.52e-05, 7.44e-06, 1.23e-02,
    3.52e-05, 1.03e-07, 8.86e-06, 1.99e-09, 7.10e-12, 2.47e-07, 1.71e-05, 6.04e-07,
    2.03e-04, 1.69e-05, 1.26e-05, 8.28e-09, 5.15e-05, 2.71e-04, 5.15e-03, 7.85e-04,
    1.90e-04, 8.23e-03, 1.04e-01, 6.13e-01, 2.55e-02, 9.63e-04, 1.02e-03, 1.28e-01,
    1.05e-02, 2.51e-04, 2.66e-03, 1.31e-06, 1.79e-06, 6.83e-07, 1.22e-05, 2.12e-05,
    1.34e-08, 1.02e-08,
])
_W7_SLICES = {"C": slice(0, 2), "N": slice(2, 4), "O": slice(4, 7), "Mg": slice(12, 15),
              "Si": slice(16, 19), "S": slice(20, 24), "Fe": slice(50, 54)}

W7_TOTAL = float(_W7_ISOTOPES.sum())
W7_EJECTA = np.array([_W7_ISOTOPES[_W7_SLICES[el]].sum() if el in _W7_SLICES else 0.0
                      for el in ELEMENTS])


class PowerLawDTD:
    """Delay-time distribution R(t) ∝ t^-slope for t >= t_min.

    Normalised so that ``n_per_msun`` SNe Ia occur per Msun formed by
    ``t_norm``: 1.3e-3 per Msun over a Hubble time with slope ~1
    (Maoz & Graur 2017).
    """

    def __init__(self, n_per_msun=1.3e-3, slope=1.0, t_min=4e7, t_norm=1.37e10):
        self.n_per_msun, self.slope = n_per_msun, slope
        self.t_min, self.t_norm = t_min, t_norm

    def _primitive(self, t):
        t = np.maximum(np.asarray(t, dtype=float), self.t_min)
        if np.isclose(self.slope, 1.0):
            return np.log(t / self.t_min)
        p = 1.0 - self.slope
        return (t ** p - self.t_min ** p) / p

    def cumulative(self, t):
        """Number of SNe Ia per Msun formed that have exploded by age ``t`` [yr]."""
        return self.n_per_msun * self._primitive(t) / self._primitive(self.t_norm)
