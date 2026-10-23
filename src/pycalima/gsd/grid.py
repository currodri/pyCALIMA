"""Fixed grain-size grid, geometric in grain mass."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MassGrid:
    """Bins geometric in grain mass for one material of bulk density ``rho``.

    ``m_edges`` [g] has n_bins + 1 entries with a constant ratio
    eta = m_edges[i+1] / m_edges[i]; ``a_edges`` [cm] are the matching radii.
    """

    m_edges: np.ndarray
    rho: float

    @classmethod
    def from_radii(cls, a_min_cm: float, a_max_cm: float, bins_per_mass_decade: float,
                   rho: float) -> "MassGrid":
        """Grid from a_min to a_max with ``bins_per_mass_decade`` bins per
        decade in mass (3x that per decade in radius is 1/3 as many)."""
        m_lo = 4.0 / 3.0 * np.pi * rho * a_min_cm ** 3
        m_hi = 4.0 / 3.0 * np.pi * rho * a_max_cm ** 3
        n = int(np.ceil(np.log10(m_hi / m_lo) * bins_per_mass_decade))
        return cls(np.geomspace(m_lo, m_hi, n + 1), float(rho))

    @property
    def n_bins(self) -> int:
        return self.m_edges.size - 1

    @property
    def eta(self) -> float:
        return float(self.m_edges[1] / self.m_edges[0])

    @property
    def a_edges(self) -> np.ndarray:
        return self.radius(self.m_edges)

    @property
    def m_centres(self) -> np.ndarray:
        return np.sqrt(self.m_edges[1:] * self.m_edges[:-1])

    @property
    def a_centres(self) -> np.ndarray:
        return self.radius(self.m_centres)

    def radius(self, m):
        return (3.0 * np.asarray(m) / (4.0 * np.pi * self.rho)) ** (1.0 / 3.0)

    def mass(self, a):
        return 4.0 / 3.0 * np.pi * self.rho * np.asarray(a) ** 3

    def bin_of_radius(self, a):
        """Index of the bin containing radius a (-1 below the grid, n_bins above)."""
        return np.searchsorted(self.a_edges, a, side="right") - 1
