"""Coagulation of one population (perfect sticking), on the general
collision operator of ``collisions``: each node pair merges into one grain
of mass m_p + m_q. See ``collisions`` for the scheme (two-node bin
representation, MPRK22 mass transfers, Patankar number).

Collisions are within one population only: grains of different materials
do not coagulate with each other here.
"""

from __future__ import annotations

from .collisions import CoagulationModel, bin_nodes, collide

__all__ = ["bin_nodes", "coagulate"]


def coagulate(grid, N, M, kernel, dt, errmax=1e-3, atol=1e-8, h0=None, max_steps=100000):
    """Coagulate (N, M) over dt with kernel(m1, m2) [cm^3 s^-1] (vectorised
    in grain masses). Returns (N, M, overflow_N, overflow_M, n_steps)."""
    r = collide(grid, N, M, CoagulationModel(kernel), dt, errmax=errmax, atol=atol, h0=h0,
                max_steps=max_steps)
    return r["N"], r["M"], r["overflow_N"], r["overflow_M"], r["n_steps"]
