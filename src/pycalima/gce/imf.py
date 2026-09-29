"""Initial mass functions, normalised to unit *mass* formed.

``imf(m)`` returns dN/dm [Msun^-1 per Msun formed], so that
``∫ m imf(m) dm = 1`` over ``[m_low, m_up]``. Every quantity integrated
against it is then "per unit stellar mass formed", which is what an SSP
table needs.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad

LN10 = np.log(10.0)


class IMF:
    """Base class: subclasses define the unnormalised shape ``_shape(m)``."""

    name = "base"

    def __init__(self, m_low: float = 0.1, m_up: float = 100.0):
        if not 0.0 < m_low < m_up:
            raise ValueError(f"need 0 < m_low < m_up, got {m_low}, {m_up}")
        self.m_low = float(m_low)
        self.m_up = float(m_up)
        breaks = [b for b in self._breaks() if self.m_low < b < self.m_up]
        edges = [self.m_low, *breaks, self.m_up]
        mass = sum(
            quad(lambda m: m * self._shape(np.array(m)), a, b, limit=200)[0]
            for a, b in zip(edges[:-1], edges[1:])
        )
        self._norm = 1.0 / mass

    def _breaks(self) -> list[float]:
        return []

    def _shape(self, m: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, m):
        m = np.asarray(m, dtype=float)
        inside = (m >= self.m_low) & (m <= self.m_up)
        safe = np.where(inside, m, 1.0)
        return np.where(inside, self._norm * self._shape(safe), 0.0)


class Salpeter(IMF):
    """Salpeter (1955): dN/dm ∝ m^-2.35."""

    name = "salpeter"

    def __init__(self, m_low=0.1, m_up=100.0, slope=2.35):
        self.slope = slope
        super().__init__(m_low, m_up)

    def _shape(self, m):
        return m ** -self.slope


class Chabrier(IMF):
    """Chabrier (2003) single-star IMF.

    Lognormal below 1 Msun (A=0.158, m_c=0.079, σ=0.69 in log10) joined
    continuously to dN/dlogm ∝ m^-1.3 above.
    """

    name = "chabrier"
    _A, _MC, _SIGMA = 0.158, 0.079, 0.69

    def _breaks(self):
        return [1.0]

    def _lognormal(self, m):
        return self._A * np.exp(-((np.log10(m) - np.log10(self._MC)) ** 2)
                                / (2.0 * self._SIGMA ** 2))

    def _shape(self, m):
        # dN/dlogm, converted to dN/dm by 1/(m ln10)
        xi = np.where(m < 1.0, self._lognormal(m), self._lognormal(1.0) * m ** -1.3)
        return xi / (m * LN10)


class Kroupa(IMF):
    """Kroupa (2001): dN/dm ∝ m^-α with α = 0.3, 1.3, 2.3 (breaks 0.08, 0.5)."""

    name = "kroupa"
    _EDGES = (0.08, 0.5)
    _SLOPES = (0.3, 1.3, 2.3)

    def _breaks(self):
        return list(self._EDGES)

    def _shape(self, m):
        a0, a1, a2 = self._SLOPES
        b0, b1 = self._EDGES
        # continuity constants at the breaks
        c1 = b0 ** (a1 - a0)
        c2 = c1 * b1 ** (a2 - a1)
        return np.where(m < b0, m ** -a0,
                        np.where(m < b1, c1 * m ** -a1, c2 * m ** -a2))


_IMFS = {cls.name: cls for cls in (Salpeter, Chabrier, Kroupa)}


def create_imf(name: str = "chabrier", **kwargs) -> IMF:
    """Build an IMF by name: ``'chabrier'``, ``'kroupa'`` or ``'salpeter'``."""
    try:
        return _IMFS[name.lower()](**kwargs)
    except KeyError:
        raise ValueError(f"unknown IMF {name!r}; choose from {sorted(_IMFS)}") from None
