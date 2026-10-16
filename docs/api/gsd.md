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

The size-space velocity is tabulated and the characteristics are integrated
exactly for a velocity linear between table points (`_kernels.advance`), so
there are no time steps and equilibrium radii (roots of $\dot a$) are
approached but never crossed. The hot loops are compiled with numba when it
is installed.

`exchange` couples populations of one material (a
`sublimation_kinetics` model, any dataset) to the gas: recondensation,
sublimation with the Kelvin term and steady-state nucleation, implicit in
the gas density (backward Euler on the key species, solved for the
end-of-step density), so steps of any length are stable and every key
element is conserved to round-off. While nucleation is active the step is
halved until the gas density changes by less than 0.5% per sub-step;
otherwise one exchange step on 136 bins takes ~1.5 ms. New grains are
injected past the nucleation barrier (the Zeldovich width above n*).
Verified against Wagner's (LSW) self-similar ripening and against Nozawa &
Kozasa (2013) dust formation in expanding gas (within ~1-4% in
$a_{\rm ave}$; see the review, s. 7.6).

`coagulation` collides grains of one population: each bin's profile is
represented by two nodes (its number halves) that reproduce $N$ and $M$
exactly, node pairs deposit their products in the bin containing
$m_1 + m_2$, and the mass transfers are integrated with MPRK22 (mass
conserved to round-off and positive for any step; number Patankar-implicit),
with adaptive sub-steps from the first/second-order stage difference.
Verified against the constant, additive (Golovin) and multiplicative
Smoluchowski solutions; ~2 ms per sub-step on 150 bins. Different
materials do not coagulate with each other.

Status: transport, gas exchange and coagulation (phases 2a-2c of the
plan). Shattering and per-bin charging follow.

## Modules

```{eval-rst}
.. autosummary::
   :toctree: generated
   :template: autosummary/module.rst

   pycalima.gsd.grid
   pycalima.gsd.pl2m
   pycalima.gsd.remap
   pycalima.gsd.exchange
   pycalima.gsd.coagulation
   pycalima.gsd._kernels
```
