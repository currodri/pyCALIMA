(api-gsd)=
# `pycalima.gsd` — discretised grain size distributions

A standalone solver for the full grain size distribution of several
compositions, under growth and sublimation (signed, size-dependent
$\dot a$), nucleation, coagulation and shattering, coupled to the gas. The
design and the literature behind it are in
{doc}`/development/gsd_solver_review`.

The state is the number $N_i$ and mass $M_i$ per bin of a fixed grid
geometric in grain mass. Inside a bin $dN/dm \propto m^\beta$ with $\beta$
fixed by $M_i/N_i$ (`pl2m`), a positive and local reconstruction; bins whose
mean mass lies at an edge are held as spikes. Growth and sublimation move
grains along characteristics and remap the profiles analytically onto the
grid (`remap`): number is exact, there is no CFL limit, and a narrow
population stays within a few bins (a spike inside a bin is reconstructed
as a power law of the same $N$ and $M$, which costs ~0.5% in mass over a
factor of 10 of growth). Nucleation is injected along the characteristic
from the critical size.

Status: transport (phase 2a of the plan). Gas coupling, coagulation,
shattering and per-bin charging follow.

## Modules

```{eval-rst}
.. autosummary::
   :toctree: generated
   :template: autosummary/module.rst

   pycalima.gsd.grid
   pycalima.gsd.pl2m
   pycalima.gsd.remap
```
