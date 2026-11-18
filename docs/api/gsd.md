(api-gsd)=
# `pycalima.gsd` — discretised grain size distributions

A standalone solver for the full grain size distribution of several
compositions, under growth and sublimation (signed, size-dependent
$\dot a$), nucleation, coagulation and shattering, coupled to the gas. The
design and the literature behind it are in
{doc}`/development/gsd_solver_review`.

The state is the number $N_i$ and mass $M_i$ per bin of a fixed grid
geometric in grain mass, or graded: geometric within each segment, with a
mass ratio per bin (`MassGrid.graded`, `MassGrid.etas`). Inside a bin the
distribution is rebuilt from $(N_i, M_i)$ by a closure (see
{ref}`Methods <gsd-methods>`); by default the edge-aware EA, which is the power law
$dN/dm \propto m^\beta$ with $\beta$ fixed by $M_i/N_i$ (`pl2m`) except in front
bins (which continue their neighbour's exponent up to a cut-off inside the bin)
and isolated bins. Growth and sublimation move grains along characteristics and
remap the profiles analytically onto the grid (`remap`): number is exact and
there is no CFL limit. Nucleation is injected along the characteristic from the
critical size.

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

`sputtering` erodes grains in hot gas along $\dot a(a) < 0$ through the same
remap: the yield of `dust_sputtering` (the CALIMA tables) averaged over each
ion's Maxwellian, with the Bocchio et al. (2016) size correction that spares
nm grains, and all mass lost from the grains returned to the gas.

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

`materials` holds one record per grain material (bulk density, optics set,
shattering constants, sputtering, sublimation and charging inputs, with
sources), so every operator reads the same density; constants borrowed from
another material are listed as proxies, and unknown names raise. `carbon`
lays out the carbon carriers on one graded grid in carbon mass: a few bins
for the PAH range (N_C 20-400 by default), the grains above, and the bins
that also carry the a-C:H population. Its `CARRIERS` give the optics of the
aromatic (THEMIS CM20) and aliphatic (Jones 2012 a-C:H) populations per gram
of carbon, each grain carrying its band gap's H at its own density; see
{doc}`/physics/carbon-optics`. With a tracked hydrogen mass (rung M1),
`HydrogenatedCarbon` gives the optics of the covalent bins at any band gap,
`bin_h_per_c` and `bin_h_limits` map a bin's M_H/M_C to it, and `hydrogen`
evolves that H/C under the laboratory UV-loss and H-addition laws with the exact
step RAMSES would take; see {doc}`/physics/grain-hydrogen`.

`GrainCollisions(table_ppd=...)` tabulates the collision rates per
environment on a log-radius grid (linear on triangles split along the
diagonal, where Ormel & Cuzzi's velocity has a cusp) and interpolates
them, which matters once the rates are averaged over charge points.

(gsd-methods)=
## Methods

Every operator (`remap`, `exchange.Population`, `sputtering.sputter`,
`collisions.bin_nodes`/`collide`, `coagulation.coagulate`, `charging.destroy`,
`optics`) takes `method=`: a name of `closures.METHODS` or a
`closures.GSDMethod(closure, eps, front_exponent, nodes, rate_nodes,
kernel_average)`; `None` means `closures.DEFAULT_METHOD`. Unknown names raise.
All methods store the same two scalars per bin and keep $N$, $M$, positivity
and $m_L \le M/N \le m_R$ exactly.

| `closure` | in-bin profile |
|---|---|
| `ea` (default) | edge-aware: a bin whose neighbour is empty (mean $dn/da$ below `eps` = 1e-6 of its own) is a front, continuing its other neighbour's exponent on a truncated support (exact for power-law segments with sharp cut-offs); a bin with both neighbours empty is a spike at its mean mass; elsewhere PL2M |
| `eaf` | as `ea` with a fixed front exponent, $dN/da \propto a^{-3.5}$ (`front_exponent`), so that its shapes depend on $s$ alone |
| `pl2m` | one power law per bin (the method before 2026-11-18) |
| `affine` | $dn/da$ linear in $a$, a triangle with partial support beyond (local) |
| `exp` | $dN/dm \propto e^{-\kappa m}$, the maximum-entropy profile (local) |

`nodes` (collisions) and `rate_nodes` (rates at nodes, `charging.destroy`) are
`median` (number halves at their mean masses) or `gauss` (2-point Gauss rule
in radius: $N$, surface and $M$ exact). `DEFAULT_METHOD` is `ea` with
`median` collision nodes and `gauss` rate nodes; `method="pl2m"` reproduces
the earlier results bit for bit. `kernel_average = n > 1` averages each node
pair's kernel and products over $n \times n$ sub-slices (sharp kernel
thresholds; costs $n^2$ kernel evaluations per pair, channel models only).
Per-moment drift velocities (`closures.moment_velocities`) always use the
local base closure, never the isolated-bin spike.

**Tables for any closure: node tables.** EA's fronts depend on the
neighbour's exponent as well as on $s$, so it cannot use $s$ tables.
`optics.node_table` stores a per-gram quantity at a few radii per bin
(uniform or geometric in $u = a/a_L$, the cheapest layout per bin whose bin
averages meet a tolerance over power laws at every in-bin state and truncated
fronts of every slope, `optics.probe_profiles`), and `optics.node_average`
integrates that piecewise-linear $k(u)$ exactly over a cell's own
reconstruction. The same table serves `ea`, `eaf`, `pl2m`, `affine` and
`exp`; for the A2 carbon grid it is about the size of the PL2M $s$ tables
(`diagnostics/gsd/check_node_tables.py`).

**EAF tables.** Each EAF component is a function of the in-bin state $s$
alone: the PL2M profile (base; `optics.profile_weights` as before), the right
front ($a^{-3.5}$ on $[a_L, a_c]$), the left front (on $[a_c, a_R]$) and the
spike. `optics.component_kappa_tables` tabulates the four per bin, and a
bin's value is $\sum_c f_c\,\kappa_c(s)$ with the fractions $f_c$ from its
neighbours' $N$ (`closures.component_fractions`, `optics.combine_components`).

Benchmarks (8 bins): growth-test mass error 14 % (PL2M) to 1.6 % (EA; 2.2 %
EAF, whose fixed front slope also puts peaks at the edges of flat profiles);
worst A0 observable 12 % to 4.4 %, at +22 % compiled cost of the growth step;
`results/gsd_overview/variants/` and `diagnostics/gsd/bench_pl2m_variants.py`.

## Performance

`diagnostics/gsd/benchmark_gsd.py` times the main workloads (wall and CPU).

* **Collision sums** run in compiled code without the GIL on Python
  threads (`collisions.run_rows`), not on numba's threading layers. Those are
  limited on Apple silicon: the pip wheels have no TBB layer, and the OpenMP
  layer needs Homebrew's keg-only `libomp`, so numba falls back to
  `workqueue`.
* **Threads:**
  * Workers are marked user-interactive (macOS QoS), so the scheduler keeps
    them on the performance cores.
  * `default_threads()` takes the performance cores less the current load
    (or `PYCALIMA_GSD_THREADS`), because oversubscribing a busy machine is
    slower than serial.
  * Threads pull row-chunks from a shared counter.
* **`GrainCollisions` with a `RelativeVelocity`** tabulates rates and impact
  speeds per environment (`table_ppd = 24` by default; the table rebuilds
  when the gas, turbulence or charge provider changes). Its pair sum is
  fused (`_shatter_rows`): table interpolation, ejecta and deposits in one
  compiled loop, with no intermediate arrays. Pairs below 1e-12 of the total
  number and mass rates are skipped (`prune`).
* **`n_speed`:** the default of 4 is converged to ~1% against 8. With
  `n_speed = 2` it costs ~3% and halves the time.

Measured on an idle M-series machine (10 performance + 4 efficiency cores):

| Workload | Wall | CPU |
|---|---|---|
| Exchange step, 136 bins | 1.9 ms | 1.9 ms |
| Collision rate evaluation, 120 bins (MRN, charges, table) | 1.1 ms | 6 ms |
| WIM shattering + sticking, 10 Myr (253 steps) | 1.1-1.3 s | 7 s |
| Coagulation, 150 bins, one collision time | 0.94 s | 1.5 s |
| Charging solve, 22 radii | 1.9 ms | 1.9 ms |

Shattering scales 5.5x from 1 to 10 threads and gains nothing on the
efficiency cores (12-14 threads). Coagulation sums are small and run best
on 3-4 threads, so `threads_for(work)` gives one thread per 15k
channel x pair evaluations.

Status: transport, gas exchange, coagulation, shattering and per-size
charging (phases 2a-2e of the plan). The AGN rim on the solver follows.

## Modules

```{eval-rst}
.. autosummary::
   :toctree: generated
   :template: autosummary/module.rst

   pycalima.gsd.grid
   pycalima.gsd.materials
   pycalima.gsd.carbon
   pycalima.gsd.hydrogen
   pycalima.gsd.closures
   pycalima.gsd.pl2m
   pycalima.gsd.remap
   pycalima.gsd.exchange
   pycalima.gsd.sputtering
   pycalima.gsd.collisions
   pycalima.gsd.coagulation
   pycalima.gsd.shattering
   pycalima.gsd.velocities
   pycalima.gsd.charging
   pycalima.gsd.optics
   pycalima.gsd._kernels
```
