"""Discretised grain size distributions (GSD) and their evolution.

A standalone solver for the full grain size distribution of several
compositions under growth and sublimation (signed, size-dependent da/dt),
nucleation, coagulation and shattering, coupled to the gas. The design and
the literature behind it are in docs/development/gsd_solver_review.md.

State: number N_i [cm^-3] and mass M_i [g cm^-3] per bin of a fixed grid
geometric in grain mass, per composition. Within a bin the distribution is
rebuilt from (N_i, M_i) by a closure (``closures``): by default the
edge-aware 'ea' (a power law whose exponent M_i/N_i fixes, as ``pl2m``,
with front bins truncated and continuing their neighbour's exponent, and
isolated bins as spikes); 'pl2m', 'eaf', 'affine' and 'exp' are selectable
with ``method=`` on every operator.

Modules
-------
grid
    the fixed mass grid
closures
    the in-bin closures and ``GSDMethod`` (closure, nodes, kernel averaging)
pl2m
    the two-moment power-law reconstruction and its exact integrals
remap
    size-space transport: flow maps, conservative remap of the
    distribution along characteristics, and nucleation injection
"""
