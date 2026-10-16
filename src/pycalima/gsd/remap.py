"""Size-space transport: flow maps, conservative remap and nucleation injection.

Growth and sublimation move every grain along a characteristic
da/dt = adot(a), so a grain at a ends a step at X(a). The velocity is
tabulated (``velocity_table``) and the characteristics are integrated
exactly for a velocity linear between table points (``_kernels.advance``):
no time steps, no stiffness limit, and equilibrium radii (roots of adot) are
approached but never crossed.

``remap`` takes the two-moment power-law profile of each source bin
(``pl2m``), splits it into sub-intervals whose images are interpolated
linearly in a (exact when adot is constant), cuts the images at the fixed
bin edges and integrates number and mass of each piece analytically. Number
moves along characteristics exactly; mass is the grains' new mass. Grains
whose image falls below the smallest edge are removed (their mass goes back
to the gas through the caller's closure); grains above the largest edge are
reported as overflow.

``inject`` lays down a nucleation source of rate J [cm^-3 s^-1] over a step:
grains born at the critical radius a* at time tau into the step end where
the characteristic from a* takes them in the remaining time, so the number
is spread with density J along that trajectory (NK13's integral over
formation time, binned).
"""

from __future__ import annotations

import numpy as np

from . import _kernels as K
from .pl2m import BinProfile


def velocity_table(adot, a_lo, a_hi, n=4000):
    """(radii, adot(radii)) on ``n`` log-spaced points from a_lo to a_hi."""
    ta = np.geomspace(a_lo, a_hi, n)
    return ta, np.asarray(adot(ta), dtype=float)


def advance(table, a, dt):
    """Radii after dt along the characteristics of a velocity table."""
    ta, tf = table
    a = np.asarray(a, dtype=float)
    return K.advance(ta, tf, a.ravel(), float(dt)).reshape(a.shape)


def flow_map(adot, a, dt, n_table=8000, span=1e3):
    """Positions after dt of grains at radii a under da/dt = adot(a); grains
    that shrink to the bottom of the table return 0 (vanished)."""
    a = np.asarray(a, dtype=float)
    pos = a[a > 0]
    if pos.size == 0:
        return np.zeros_like(a)
    table = velocity_table(adot, pos.min() / span, pos.max() * span, n_table)
    return advance(table, a, dt)


def trajectory(adot, a0, dt, n_nodes=33, table=None):
    """Radius along the characteristic from a0 at times 0..dt (n_nodes)."""
    s = np.linspace(0.0, dt, n_nodes)
    if table is None:
        table = velocity_table(adot, a0 / 1e3, a0 * 1e4, 8000)
    ta, tf = table
    return s, np.array([K.advance_one(ta, tf, float(a0), float(si)) for si in s])


def remap(grid, N, M, image, n_pieces=4, profile=None):
    """Remap (N, M) per bin along the flow map ``image(a) -> X(a)``.

    ``image`` is a vectorised callable giving the new radius of grains now at
    radius a; it must be monotone non-decreasing. Returns (N_new, M_new,
    lost_N, overflow_N, overflow_M): the number of grains that vanished below
    the grid, and the number and mass carried above it. ``profile`` reuses a
    ``BinProfile`` of (N, M) already built (e.g. across root-finding calls).
    """
    N = np.asarray(N, float)
    M = np.asarray(M, float)
    prof = profile if profile is not None else BinProfile(grid, N, M)
    a_e = grid.a_edges
    t = np.linspace(0.0, 1.0, n_pieces + 1)
    nodes = a_e[:-1, None] * (a_e[1:, None] / a_e[:-1, None]) ** t[None, :]
    # spikes move as a whole: evaluate their image in column 0
    nodes = np.where(prof.delta[:, None], (prof.aL * prof.u_delta)[:, None], nodes)
    X = np.asarray(image(nodes.ravel()), float).reshape(nodes.shape)
    N1, M1, lost, oN, oM = K.remap_profiles(a_e, grid.rho, N, prof.A, prof.lam, prof.delta,
                                            prof.u_delta, nodes, X)
    return (*realisable(grid, N1, M1, tiny=1e-30 * max(N.sum(), 1e-300)), lost, oN, oM)


def realisable(grid, N, M, tiny=0.0):
    """Clip round-off: bins with N <= tiny are emptied, and the mean mass of
    the others is held inside the bin (the analytic pieces always land
    inside, so this only removes cancellation error)."""
    keep = N > tiny
    N = np.where(keep, N, 0.0)
    M = np.where(keep, np.clip(M, N * grid.m_edges[:-1], N * grid.m_edges[1:]), 0.0)
    return N, M


def inject(grid, J, positions):
    """Bin a nucleation source along a characteristic.

    ``positions`` are the end-of-step radii of grains born at evenly spaced
    times through the step (``trajectory`` read backwards, y[::-1]), and
    ``J`` [cm^-3] is the number formed in the step (rate times dt). Between
    successive positions the number is spread uniformly in radius; pieces
    below the grid are dropped (they never become grains). Returns (dN, dM).
    """
    return K.inject_segments(grid.a_edges, grid.rho, float(J),
                             np.asarray(positions, dtype=float))
