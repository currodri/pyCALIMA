"""Size-dependent sinks that remove grains without shrinking them, per moment.

A sink at a rate k(a) [s^-1] (Coulomb explosion; in RAMSES also molecular carbon loss and freeze-out treated as
sinks) does not move grains along the size axis; it removes them where they are. Held fixed over a step, its exact
solution is dN/da(t + dt) = dN/da(t) exp(-k(a) dt) at every size. The two moments of each bin are therefore the
in-bin integrals of that factor over the bin's reconstruction:

    N' = int exp(-k dt) dN,        M' = int m exp(-k dt) dN.

Each bin's N and M thus decay at their own effective rates, the number- and mass-weighted means of k for small
k dt, which is the per-moment rule of the plan (2026-11-17), the same as the drift's. The removed grains keep a
realisable mean mass, and the in-bin state s evolves. ``charging.destroy`` instead applies the factor at the bin's
rate nodes.
"""
from __future__ import annotations

import numpy as np

from . import closures

#: nodes per bin (n_nodes + 1 points, uniform in a / a_L) between which exp(-k dt) is taken linear (second order in
#: the spacing). With 32, a bin of eta ~19 across which k dt runs from ~0.5 to ~4 loses the wrong grains for ~2e-4 of
#: its content; a constant k is exact.
N_NODES = 32


def survival_weights(grid, N, M, k_of_a, dt, method=None, n_nodes=N_NODES):
    """The fractions (f_N, f_M) of each bin's number and mass that survive a sink k(a) [s^-1, vectorised] held fixed
    over dt [s]: the number- and mass-weighted means of exp(-k dt) over the bin's reconstruction (``closures``).
    Empty bins return 1.

    Both means are normalised by the weights' own sum, which is 1 only to round-off. Where most of a bin survives,
    the fraction is 1 minus the mean lost fraction 1 - exp(-k dt) (``expm1``). There a zero rate leaves the bin
    exactly as it was, and small k dt keeps its precision. Where most is lost, it is the mean of exp(-k dt) itself,
    which keeps the precision of the few survivors."""
    P = closures.pieces(grid, N, M, method)
    t = np.linspace(0.0, 1.0, n_nodes + 1)
    u = 1.0 + (grid.etas[:, None] ** (1.0 / 3.0) - 1.0) * t[None, :]
    k = np.maximum(np.asarray(k_of_a(P.aL[:, None] * u), float), 0.0)
    kept, lost = np.exp(-k * dt), -np.expm1(-k * dt)
    out = []
    for measure in ("number", "mass"):
        w = closures.bin_weights(grid, P, u, measure)
        s = w.sum(axis=1)
        s = np.where(s > 0.0, s, 1.0)
        f_kept = np.einsum("ij,ij->i", w, kept) / s
        f_lost = np.einsum("ij,ij->i", w, lost) / s
        out.append(np.clip(np.where(f_kept < 0.5, f_kept, 1.0 - f_lost), 0.0, 1.0))
    empty = np.asarray(N, float) <= 0.0
    return np.where(empty, 1.0, out[0]), np.where(empty, 1.0, out[1])


def sink_per_moment(grid, N, M, k_of_a, dt, method=None, n_nodes=N_NODES):
    """Remove grains at the size-dependent rate k(a) [s^-1] over dt [s], exactly for k held fixed (module doc).

    Returns (N', M', dN, dM): the new bins and the number and mass removed from each bin, which the caller routes
    (to the gas, or as fragments)."""
    N = np.asarray(N, float)
    M = np.asarray(M, float)
    fN, fM = survival_weights(grid, N, M, k_of_a, dt, method, n_nodes)
    N1, M1 = N * fN, M * fM
    return N1, M1, N - N1, M - M1


def deposit_power_law(grid, mass, a_lo, a_hi, alpha=3.3):
    """Number and mass per bin of ``mass`` [g cm^-3] of grains with dN/da ~ a^-alpha on [a_lo, a_hi] [cm], exactly
    (the fragments of shattering and of Coulomb explosion). Returns (dN, dM, below, above): the bins' shares and the
    mass below the grid's lowest edge (for the gas) and above its top edge."""
    z = np.zeros(grid.n_bins)
    if mass <= 0.0:
        return z, z.copy(), 0.0, 0.0
    if a_hi <= grid.a_edges[0]:                                  # every fragment below the grid
        return z, z.copy(), float(mass), 0.0
    if a_hi <= a_lo:
        raise ValueError("a_hi must exceed a_lo")
    if abs(alpha - 1.0) < 1e-9 or abs(alpha - 4.0) < 1e-9:
        raise ValueError("alpha = 1 or 4 is not supported (log integrals)")
    eN, eM = 1.0 - alpha, 4.0 - alpha
    mfac = 4.0 / 3.0 * np.pi * grid.rho
    norm = mass * eM / (mfac * (a_hi ** eM - a_lo ** eM))            # dN/da = norm a^-alpha
    x = np.clip(grid.a_edges, a_lo, a_hi)
    dN = norm * np.diff(x ** eN) / eN
    dM = norm * mfac * np.diff(x ** eM) / eM
    below = norm * mfac * (min(grid.a_edges[0], a_hi) ** eM - a_lo ** eM) / eM if a_lo < grid.a_edges[0] else 0.0
    above = norm * mfac * (a_hi ** eM - max(grid.a_edges[-1], a_lo) ** eM) / eM if a_hi > grid.a_edges[-1] else 0.0
    return dN, dM, float(below), float(above)
