"""Fixed grain-size grid, geometric in grain mass (or graded: geometric per segment)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MassGrid:
    """Bins in grain mass for one material of bulk density ``rho``.

    ``m_edges`` [g] has n_bins + 1 entries; ``a_edges`` [cm] are the matching
    radii. A uniform grid has one ratio eta = m_edges[i+1] / m_edges[i]; a
    graded one (``graded``) has a ratio per bin, ``etas``, and ``eta`` raises.

    With ``overflow`` the last bin is the composition's overflow bin (``with_overflow``): one wide bin above the
    grid's top (1-10 um, decided 2026-11-23), an ordinary bin for every operator, which the per-cell operator
    computes only once it holds grains.
    """

    m_edges: np.ndarray
    rho: float
    overflow: bool = False

    def with_overflow(self, a_top_cm: float = 1e-3) -> "MassGrid":
        """This grid plus the overflow bin, from its top edge to ``a_top_cm`` (default 10 um)."""
        if self.overflow:
            raise ValueError("the grid already has an overflow bin")
        m_top = 4.0 / 3.0 * np.pi * self.rho * a_top_cm ** 3
        if not m_top > self.m_edges[-1]:
            raise ValueError("the overflow bin must end above the grid's top edge")
        return MassGrid(np.append(self.m_edges, m_top), self.rho, overflow=True)

    @classmethod
    def from_radii(cls, a_min_cm: float, a_max_cm: float, bins_per_mass_decade: float,
                   rho: float) -> "MassGrid":
        """Grid from a_min to a_max with ``bins_per_mass_decade`` bins per
        decade in mass (3x that per decade in radius is 1/3 as many)."""
        m_lo = 4.0 / 3.0 * np.pi * rho * a_min_cm ** 3
        m_hi = 4.0 / 3.0 * np.pi * rho * a_max_cm ** 3
        n = int(np.ceil(np.log10(m_hi / m_lo) * bins_per_mass_decade))
        return cls(np.geomspace(m_lo, m_hi, n + 1), float(rho))

    @classmethod
    def graded(cls, m_breaks, n_bins, rho: float) -> "MassGrid":
        """Grid geometric within each segment [m_breaks[k], m_breaks[k+1]] [g],
        with ``n_bins[k]`` bins in segment k."""
        m_breaks = np.asarray(m_breaks, float)
        if m_breaks.size != len(n_bins) + 1 or np.any(np.diff(m_breaks) <= 0):
            raise ValueError("m_breaks must be increasing, one more than n_bins")
        edges = [np.geomspace(m_breaks[k], m_breaks[k + 1], int(n) + 1)[:-1]
                 for k, n in enumerate(n_bins)]
        return cls(np.concatenate(edges + [m_breaks[-1:]]), float(rho))

    @property
    def n_bins(self) -> int:
        return self.m_edges.size - 1

    @property
    def uniform(self) -> bool:
        """True if every bin has the same mass ratio."""
        r = self.m_edges[1:] / self.m_edges[:-1]
        return bool(np.allclose(r, r[0], rtol=1e-9, atol=0.0))

    @property
    def eta(self) -> float:
        """The mass ratio of every bin (uniform grids only)."""
        if not self.uniform:
            raise ValueError("graded grid: bins differ in mass ratio, use etas")
        return float(self.m_edges[1] / self.m_edges[0])

    @property
    def etas(self) -> np.ndarray:
        """Mass ratio of each bin; on a uniform grid every entry is ``eta``."""
        if self.uniform:
            return np.full(self.n_bins, self.eta)
        return self.m_edges[1:] / self.m_edges[:-1]

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
