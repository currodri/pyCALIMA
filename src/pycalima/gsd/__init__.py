"""Discretised grain size distributions (GSD) and their evolution.

A standalone solver for the full grain size distribution of several
compositions under growth and sublimation (signed, size-dependent da/dt),
nucleation, coagulation and shattering, coupled to the gas. The design and
the literature behind it are in docs/development/gsd_solver_review.md.

State: number N_i [cm^-3] and mass M_i [g cm^-3] per bin of a fixed grid
geometric in grain mass, per composition. Within a bin dN/dm is a power law
whose exponent is fixed by the bin's mean mass M_i/N_i (two-moment power
law, ``pl2m``), so the representation is positive and local.

Modules
-------
grid      the fixed mass grid
pl2m      the two-moment power-law reconstruction and its exact integrals
remap     size-space transport: flow maps, conservative remap of the
          distribution along characteristics, and nucleation injection
"""
