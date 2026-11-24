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
    Empty bins return 1."""
    P = closures.pieces(grid, N, M, method)
    t = np.linspace(0.0, 1.0, n_nodes + 1)
    u = 1.0 + (grid.etas[:, None] ** (1.0 / 3.0) - 1.0) * t[None, :]
    k = np.maximum(np.asarray(k_of_a(P.aL[:, None] * u), float), 0.0)
    g = np.exp(-k * dt)
    fN = np.einsum("ij,ij->i", closures.bin_weights(grid, P, u, "number"), g)
    fM = np.einsum("ij,ij->i", closures.bin_weights(grid, P, u, "mass"), g)
    empty = np.asarray(N, float) <= 0.0
    return np.where(empty, 1.0, np.clip(fN, 0.0, 1.0)), np.where(empty, 1.0, np.clip(fM, 0.0, 1.0))


def sink_per_moment(grid, N, M, k_of_a, dt, method=None, n_nodes=N_NODES):
    """Remove grains at the size-dependent rate k(a) [s^-1] over dt [s], exactly for k held fixed (module doc).

    Returns (N', M', dN, dM): the new bins and the number and mass removed from each bin, which the caller routes
    (to the gas, or as fragments)."""
    N = np.asarray(N, float)
    M = np.asarray(M, float)
    fN, fM = survival_weights(grid, N, M, k_of_a, dt, method, n_nodes)
    N1, M1 = N * fN, M * fM
    return N1, M1, N - N1, M - M1
