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

`collisions` is the general operator behind coagulation and shattering:
a model returns, per node pair and outcome channel, a rate coefficient and
the products of each grain (a remnant, deposited spread over its parent
node's width, and fragments $dN/da\propto a^{-\alpha_f}$ integrated exactly
over the bins with difference arrays, so a pair costs $O(1)$). Fragments
below the grid go to a collector (kept in the smallest bin, HY09, or
removed, HK13); an optional constant source feeds the step. Large problems
run on up to 4 threads.

`velocities` gives continuous relative velocities for collisions: for
every node pair the rms random speed from Brownian motion and the full
Ormel & Cuzzi (2007) turbulence (`collapse.velocities.ormel_cuzzi_dv`),
plus an optional systematic drift between sizes (e.g. radiation pressure),
with the skewed-Maxwellian speed distribution of Guillet et al. (2020).
Shattering is split into equal-flux channels above $v_{\rm shat}$ at their
flux-quantile speeds and sticking takes the flux below $v_{\rm coag}$, so
both thresholds are smooth in grain size and gas state (the HK13 step and
the HY09 four directions remain available for their benchmarks).

`shattering` provides the physics: HY09 Table 2 materials, the KT10 and
JTH96 ejecta laws in the form of Hirashita & Kobayashi (2013), fragments up
to $(0.0204 f)^{1/3} a$, HY09's four velocity directions and the Chokshi
sticking threshold (HY09 eq. 8). Velocity laws should be continuous: a node
sitting on a jump makes the step control chatter. Verified against the
Dohnanyi/Tanaka steady cascade ($a^{-3.5}$ for any fragment slope), the
HK13 worked example and their $P_1/t$ scaling
(`diagnostics/gsd/check_shattering.py`).

`charging` gives every radius its charge distribution with the RAMSES-CALIMA
group solver (WDB06 photoemission including inner shells and Auger
electrons, DS87 capture, hot-gas secondaries), on photon groups extended to
10 keV and potentials to 30 kV for AGN illumination (`AGN_GROUPS`; tables
cached on disk). `PhotonGroups` converts any spectrum, and
`xray_power_law` adds the corona that the SKIRTOR SED lacks. The
`ChargeState` provides:

* $\langle Z\rangle$, $\sigma_Z$ and three Gauss-Hermite charge points,
  for the Coulomb focusing or barrier of grain-grain collisions in
  `velocities` (closed-form skewed-Maxwellian integrals with
  $\sigma_{\rm eff} = \sigma(1 - v_C^2/v^2)$; shattering on the impact speed,
  sticking on the relative speed at infinity, as Guillet et al. 2020);
* the ion-capture factor $\langle\tilde J\rangle$ over the exact $P(Z)$, which
  `exchange` applies to the ionised fraction of a key species;
* Coulomb-explosion rates: the first-passage flux into the charge at which
  $(U/a)^2/4\pi$ reaches the tensile strength (Waxman & Draine 2000;
  Tazaki, Ichikawa & Kokubo 2020), zero where ion field emission caps the
  charge first; `destroy` applies such a rate to a population.

`GrainCollisions(table_ppd=...)` tabulates the collision rates per
environment on a log-radius grid (linear on triangles split along the
diagonal, where Ormel & Cuzzi's velocity has a cusp) and interpolates
them, which matters once the rates are averaged over charge points.

Status: transport, gas exchange, coagulation, shattering and per-size
charging (phases 2a-2e of the plan). The AGN rim on the solver follows.

## Modules

```{eval-rst}
.. autosummary::
   :toctree: generated
   :template: autosummary/module.rst

   pycalima.gsd.grid
   pycalima.gsd.pl2m
   pycalima.gsd.remap
   pycalima.gsd.exchange
   pycalima.gsd.collisions
   pycalima.gsd.coagulation
   pycalima.gsd.shattering
   pycalima.gsd.velocities
   pycalima.gsd.charging
   pycalima.gsd._kernels
```
