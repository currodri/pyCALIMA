"""The carbon carriers on one graded grid in carbon mass.

All carbon (PAHs, small a-C:H, aromatic a-C, large grains) shares one grid,
so size changes (growth, erosion, shattering, coagulation) never cross from
one grid to another; only state changes (aromatisation, aliphatisation) move
mass between populations, at fixed size. The grid coordinate is the carbon
mass, so a bin's edges are exact carbon-atom counts, N_C = m / m_C; radii use
the material's bulk density.

- Bins below ``nc_pah_max`` are PAH bins: their aromatic population uses the
  molecular (pahchem) tables, the grain tables elsewhere.
- Bins below ``nc_ali_max`` (default: ``nc_pah_max``) also carry the
  aliphatic a-C:H population, each population with its own (N, M).
- The PAH range has ``n_pah_bins`` bins (3-4 keep the 3.3 um band ratios to
  a few per cent); the grains up to ``a_max_cm`` have ``n_grain_bins``.

Optical carriers (``CARRIERS``): a bin's M counts carbon only, and each
population carries H at the fixed fraction of its band gap, X_H = E_g / 4.3 eV
(Jones 2012). A grain of carbon mass m therefore has the total mass
m (1 + f_H) and, at its carrier's density, its own radius, which is what its
optics see (``CarbonCarrier.per_gram_carbon``, per gram of carbon):

- ``aromatic``: THEMIS CM20 structure (a-C with E_g = 0.1 eV below 20 nm, an
  a-C:H core in a 20 nm a-C mantle above), 1.6 g/cm^3, with the large grains
  recomputed by coated-sphere Mie (``themis_optics.mie_efficiencies``), which
  matches DustEM's Q_CM20 where that table is sound and replaces it where it is
  broken (a >~ 0.5 um). The large grains' core H is not counted (their carbon is
  overestimated by up to ~10%).
- ``aromatic_dustem``: the same with DustEM's Q_CM20 tables throughout, the
  benchmark for comparisons with DustEM.
- ``aliphatic``: a-C:H with E_g = 2.5 eV from the size-dependent Jones (2012)
  tables (``jones2012_optics``), 1.33 g/cm^3; a grain is 1.10 times the radius of
  an aromatic grain of the same carbon mass.

Heat capacities (``CarbonCarrier.heat_capacity``) are harmonic sums over the
atoms (``carbon_heat_capacity``), DustEM's C_CM20.DAT for ``aromatic_dustem``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pycalima.models.dust_radiation.jones2012_optics import density as _jones_density

from .grid import MassGrid
from .materials import get_material

M_C = 12.011 * 1.66053906660e-24      # g
M_H = 1.008 * 1.66053906660e-24       # g


@dataclass(frozen=True)
class CarbonLayout:
    """Size ranges of the carbon carriers and the grid that holds them."""

    nc_min: float = 20.0
    nc_pah_max: float = 400.0
    n_pah_bins: int = 4
    a_max_cm: float = 1e-4
    n_grain_bins: int = 7
    nc_ali_max: float | None = None
    material: str = "themis_ac"

    def grid(self) -> MassGrid:
        rho = get_material(self.material).rho
        m_top = 4.0 / 3.0 * np.pi * rho * self.a_max_cm ** 3
        if not self.nc_min * M_C < self.nc_pah_max * M_C < m_top:
            raise ValueError("need nc_min < nc_pah_max < N_C at a_max_cm")
        g = MassGrid.graded([self.nc_min * M_C, self.nc_pah_max * M_C, m_top],
                            [self.n_pah_bins, self.n_grain_bins], rho)
        if self.nc_ali_max is not None:
            nc_e = g.m_edges / M_C
            if (self.nc_ali_max < self.nc_pah_max
                    or not np.any(np.isclose(nc_e, self.nc_ali_max, rtol=1e-9))):
                raise ValueError(f"nc_ali_max must be a bin edge >= nc_pah_max; edges: {nc_e}")
        return g

    def pah_bins(self, grid=None) -> np.ndarray:
        """True for bins whose aromatic population is molecular (PAH tables)."""
        g = grid or self.grid()
        return g.m_edges[1:] <= self.nc_pah_max * M_C * (1 + 1e-9)

    def aliphatic_bins(self, grid=None) -> np.ndarray:
        """True for bins that also carry the a-C:H population."""
        g = grid or self.grid()
        top = self.nc_pah_max if self.nc_ali_max is None else self.nc_ali_max
        return g.m_edges[1:] <= top * M_C * (1 + 1e-9)


@dataclass(frozen=True)
class CarbonCarrier:
    """Optics of one carbon population on the carbon-mass grid."""

    name: str
    e_gap: float                   # band gap [eV]; sets X_H = E_g / 4.3
    optics: str                    # 'cm20_mie', 'themis_cm20' (DustEM tables) or 'jones2012'
    rho: float                     # bulk density [g cm^-3]
    charging: str = "ac"           # wdb06_yields material of its charge tables

    @property
    def h_per_carbon_mass(self) -> float:
        """f_H: grams of H per gram of C."""
        x = self.e_gap / 4.3
        return x * M_H / ((1.0 - x) * M_C)

    def radius_cm(self, m_carbon):
        """Radius of a grain of carbon mass ``m_carbon`` [g]."""
        m = np.asarray(m_carbon, float) * (1.0 + self.h_per_carbon_mass)
        return (3.0 * m / (4.0 * np.pi * self.rho)) ** (1.0 / 3.0)

    def cross_sections(self, m_carbon, lam_um):
        """C_abs, C_sca, C_rp per grain [cm^2], each (nlam, n_m)."""
        a_um = np.atleast_1d(self.radius_cm(m_carbon)) * 1e4
        if self.optics == "themis_cm20":
            from pycalima.models.dust_radiation import themis_optics

            return themis_optics.cross_sections(a_um, lam_um)
        if self.optics == "cm20_mie":
            from pycalima.models.dust_radiation import themis_optics

            qabs, qsca, g = themis_optics.mie_efficiencies(a_um, lam_um)
            area = np.pi * (a_um * 1e-4) ** 2
            return {"C_abs": qabs * area, "C_sca": qsca * area, "C_rp": (qabs + (1.0 - g) * qsca) * area}
        if self.optics == "jones2012":
            from pycalima.models.dust_radiation import jones2012_optics

            return jones2012_optics.cross_sections(a_um, lam_um, self.e_gap)
        raise ValueError(f"unknown optics {self.optics!r}")

    def per_gram_carbon(self, m_carbon, lam_um):
        """(C_abs, C_sca, C_rp) per gram of carbon, shape (n_m, nlam, 3)."""
        m = np.atleast_1d(np.asarray(m_carbon, float))
        c = self.cross_sections(m, lam_um)
        return np.stack([c[q].T / m[:, None] for q in ("C_abs", "C_sca", "C_rp")], axis=-1)

    def heat_capacity(self, m_carbon, T):
        """Heat capacity per grain [erg/K], (nT, n_m): the harmonic sum over atoms
        (``carbon_heat_capacity``; CM20 structure for the aromatic carrier), or
        DustEM's C_CM20.DAT for the benchmark carrier."""
        a_um = np.atleast_1d(self.radius_cm(m_carbon)) * 1e4
        if self.optics == "themis_cm20":
            from pycalima.models.dust_radiation import themis_optics

            return themis_optics.heat_capacity(a_um, T)
        from pycalima.models.dust_radiation import carbon_heat_capacity as chc

        if self.optics == "cm20_mie":
            return chc.heat_capacity_cm20(a_um, T)
        m_tot = np.atleast_1d(np.asarray(m_carbon, float)) * (1.0 + self.h_per_carbon_mass)
        return chc.grain_heat_capacity(m_tot / (chc.mean_atomic_mass(self.e_gap) * chc.AMU), T, self.e_gap)

    def per_gram_on(self, grid, lam_um):
        """``per_gram(a_cm)`` for ``gsd.optics.kappa_tables`` on ``grid``, whose radii
        convert to carbon masses at the grid's density."""
        return lambda a_cm: self.per_gram_carbon(grid.mass(np.asarray(a_cm, float)), lam_um)


CARRIERS = {
    "aromatic": CarbonCarrier("aromatic", 0.1, "cm20_mie", 1.6, "ac"),
    "aromatic_dustem": CarbonCarrier("aromatic_dustem", 0.1, "themis_cm20", 1.6, "cm20"),
    "aliphatic": CarbonCarrier("aliphatic", 2.5, "jones2012", float(_jones_density(2.5)), "ach"),
}
