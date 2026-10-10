(api-collapse)=
# `pycalima.collapse` — one-zone collapse with evolving dust

A one-zone gravitational collapse along a prescribed track, with the grain
size distribution evolving on a dense logarithmic grid, to measure how dust
evolution changes the efficiency of gas-grain collisional cooling. The
size-distribution numerics follow Marchand et al. (2021, A&A 649, A50,
appendix B): a power law in mass inside each bin and a mass-conserving
two-collector Smoluchowski scheme.

## Modules

```{eval-rst}
.. autosummary::
   :toctree: generated
   :template: autosummary/module.rst

   pycalima.collapse.grid
   pycalima.collapse.coagulation
   pycalima.collapse.velocities
   pycalima.collapse.charge
   pycalima.collapse.tracks
   pycalima.collapse.materials
   pycalima.collapse.optics
   pycalima.collapse.gas_grain
   pycalima.collapse.dust_temperature
   pycalima.collapse.accretion
   pycalima.collapse.integrator
   pycalima.collapse.ics
   pycalima.collapse._jit
```
