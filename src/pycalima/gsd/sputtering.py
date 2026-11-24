"""Thermal sputtering on the GSD: grains shrink along da/dt(a) < 0.

Ions of a thermal gas erode a grain of radius a at

    da/dt = -(m_d / 2 rho) n_H sum_i x_i <Y_i v>_T,

with m_d the mean atomic mass of the grain material, x_i the abundance of
ion i per H and <Y_i v>_T the yield of ``dust_sputtering.sputtering_yield``
(Nozawa et al. 2006 form) averaged over the ion's Maxwellian. This is the
rate of ``dust_sputtering.total_erosion_rate`` (the CALIMA tables), here as a
function of a: with ``size_correction`` the Bocchio et al. (2016) factor for
ions that cross small grains makes it size-dependent, so it is tabulated on
radii and interpolated in ln a. Grains are neutral.

``sputter`` moves (N, M) along the characteristics with ``remap.remap``;
grains that shrink below the grid are removed, and all mass lost from the
grains is returned for the caller to put back into the gas.
"""

from __future__ import annotations

import numpy as np

from ..models.dust_gas_collisions import dust_sputtering as ds
from .remap import advance, remap, velocity_table

_AMU = 1.66053906660e-24

# (atomic mass, atomic number, abundance per H) of a fully ionised solar gas
# (Asplund et al. 2009; C and O undepleted)
SOLAR_IONS = ((1.008, 1, 1.0), (4.0026, 2, 0.0851), (12.011, 6, 2.69e-4), (15.999, 8, 4.90e-4))


def erosion_rate(material, T, n_H, rho=None, ions=SOLAR_IONS, size_correction=True,
                 a_lo=1e-8, a_hi=1e-3, n_a=33, nbins_v=100):
    """da/dt(a) [cm s^-1], a vectorised callable (negative), for grains of
    ``material`` ('graphite' or 'silicate') in gas at T [K] and n_H [cm^-3].

    ``rho`` is the bulk density used to turn the eroded atoms into a radius
    loss (default: the sputtering module's); pass the grid's so that the mass
    returned to the gas is the mass the grains lose. Without
    ``size_correction`` the rate does not depend on a.
    """
    radii = np.geomspace(a_lo, a_hi, n_a) if size_correction else np.array([1e-5])
    rates = np.empty(radii.size)
    for k, a in enumerate(radii):
        s = ds._get_composition_sputtering_setup(material, a * 1e4)
        E_exec = s["bocchio_fitparams"]["E_exec"]
        yv = sum(x * ds.average_yields_T((a, s["surface_energy"], s["Kparam"], s["rho_dust"],
                                          s["am_dust"], A, s["an_dust"], Z, Z,
                                          s["bocchio_fitparams"], E_exec, float(T), nbins_v,
                                          size_correction, False))
                 for A, Z, x in ions)
        rates[k] = -s["am_dust"] * _AMU / (2.0 * (rho or s["rho_dust"])) * n_H * yv
    if not size_correction:
        return lambda a: np.full(np.shape(a), rates[0])
    la = np.log(radii)
    return lambda a: np.interp(np.log(np.maximum(a, 1e-30)), la, rates)


def sputter(grid, N, M, adot, dt, table_points=2000, method=None, knots=None, H=None):
    """Shrink (N, M) over dt along da/dt = adot(a) <= 0, with the in-bin closure
    of ``method`` (``closures``; default ``closures.DEFAULT_METHOD``).
    ``knots``: extra radii of the velocity table, e.g. a pair on each side of a
    jump of adot, which the log-spaced table alone would smear over one interval.

    Returns (N, M, dM_gas): the new bins and the mass the grains lost,
    including grains eroded below the grid, which goes back to the gas. With
    ``H`` (the bins' hydrogen mass, ``remap.remap_hydrogen``: eroding grains
    keep their H/C) also (H_new, dH_gas).
    """
    table = velocity_table(adot, grid.a_edges[0] * 0.3, grid.a_edges[-1] * 3.0, table_points)
    if knots is not None and len(knots):
        ta = np.unique(np.concatenate([table[0], np.asarray(knots, dtype=float)]))
        table = (ta, np.asarray(adot(ta), dtype=float))
    N1, M1, _lost, _oN, _oM = remap(grid, N, M, lambda a: advance(table, a, dt), method=method)
    if H is None:
        return N1, M1, float(np.sum(M) - M1.sum())
    from .remap import remap_hydrogen

    H1, h_gas, h_over = remap_hydrogen(grid, N, M, H, lambda a: advance(table, a, dt), N1, method=method)
    return N1, M1, float(np.sum(M) - M1.sum()), H1, h_gas + h_over
