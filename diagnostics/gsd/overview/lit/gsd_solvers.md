# How hydrodynamics codes evolve the dust grain-size distribution

Literature comparison by a Sonnet subagent, 2026-11-17, for the GSD overview (point 5 of the user's review). Saved by
the main session. Method sections read from arXiv PDFs (pdftotext); equation numbers are the arXiv versions'.
[inference] marks the subagent's own reasoning. Not read in full: Hopkins & Lee 2016, Aoyama+17/18, Hou+17/19 (grep
only), Lombart & Laibe 2021 (introduction), Mattsson 2016 (closure parts). Journal references marked "from memory" in
the subagent's notes are not checked. The 2026 preprints (Choban+26, Parente+26, Narayanan+26) were not checked
against journal versions.

**Headline.** Every full-GSD hydro code stores one moment per bin plus a fixed in-bin shape, except McKinnon+18 (AREPO)
and its re-implementation in Choban+26 (FIRE), which store N and M per bin with a shape *linear in a*. All descend from
Hirashita & Yan 2009 (via Asano+13 and McKinnon+18).

## AREPO

- **McKinnon, Vogelsberger, Torrey, Marinacci & Kannan 2018** (MNRAS 478, 2851; 1805.04521). 16 log bins, 1 nm–1 µm,
  one composition per dust particle. N_i and M_i per bin; ∂n/∂a piecewise linear in a with the slope fixed by M_i
  (Eqs. 31, 39, 44); piecewise-constant option. Growth/sputtering: shift by ȧΔt plus exact intersection with the fixed
  bins (Eqs. 34–37); top-boundary rebin (Eq. 46). Slope limiter sets ∂n/∂a = 0 at an edge (keeps n ≥ 0, conserves M).
  Shattering/coagulation: HY09 kernels with a piecewise-linear generalisation (Eqs. 57–58), first-order explicit update
  plus mass rescaling (Eq. 59); galaxy runs drop the slopes inside the collision integral ("no significant effect").
  Dust as Lagrangian particles with semi-implicit drag and one effective stopping time per GSD (does "not allow grains
  of different sizes to properly segregate"). Sub-cycling with Δt < 0.1 τ_GSD (Eq. 68). Tests: growth convergence
  (piecewise constant first order, linear second order; ~40 % vs ~1 % mass error at N = 64), sinusoidal ȧ, shattering
  N = 8 and coagulation N = 32 against a many-bin integrator, dusty box/wave/shock/Sedov.
- **Li, Narayanan, Torrey, Davé & Vogelsberger 2021** (2012.03978; GIZMO/Simba): dust particles with drag; 41 bins,
  0.1 nm–1 µm, piecewise constant; collisions after McKinnon+18.
- **Narayanan+23** (ApJ 951, 100; 2301.07136): AREPO/SMUGGLE idealised galaxies, 16 bins ("converged"), silicate and
  carbonaceous, aromatic/aliphatic conversion; built on McKinnon+18 and Li+21. **Narayanan+26** (2606.20809):
  cosmological PAH lifecycle, 16 bins per species, superparticles, "not treated as a passive scalar".
- **Parente, Narayanan & Torrey 2026** (2604.06314; L-Galaxies SAM): 32 bins; bin edges advected along accretion
  characteristics and remapped by overlap in log size (number conserved).
- McKinnon+19 (1912.02825, dust–radiation) and Hu+19 (1902.01368, fixed GSD) do not evolve a GSD.

## GIZMO / FIRE

- **Choban, Kereš, Hopkins & Sandstrom 2022** (MNRAS 514, 4506; 2201.12369) and Choban+24 (2401.04766, 2408.08962):
  species only, no GSD; dust is a per-cell mass fraction with turbulent diffusion; growth timescales averaged over an
  assumed GSD.
- **Choban+26, "Ashes of FIRE"** (2603.08504): the first FIRE GSD. 16 log bins, 5 nm–1 µm, per gas cell and per
  species (silicate, carbonaceous, Fe). N_i and M_i, McKinnon's piecewise-linear method (Eqs. 2–5), slope limiter
  (App. E1). Growth/sputtering: shift plus intersection remap (App. E2). Collisions HY09-type, explicit, with
  mass- and number-limited sub-cycling (App. D; without it "the abundance of large grains was heavily reduced");
  clumping factors and an ad hoc coagulation enhancement. GSD moves with the gas, dust and gas assumed coupled.
  Convergence (App. C): GSD at N = 16, extinction curves nearly at N = 8. Stated limits: "high computational and memory
  overhead, and ... strict time-stepping requirements"; no population of very small carbonaceous grains.
- **Hopkins & Lee 2016** (1510.02477): dust super-particles of fixed size, semi-implicit drag (basis of Li+21).
  **Li, Narayanan & Davé 2019** (1906.09277, Simba): single size, passively advected.

## GADGET (Osaka, MUPPI)

- **Hirashita 2015** (MNRAS 447, 2937; 1412.3866): two-size approximation (split at 0.03 µm), checked against Asano+13.
- **Aoyama+17** (1609.07547), **Aoyama+18** (1802.04027), **Hou+17** (1704.01769), **Hou+19** (1901.02886),
  **Granato+21** (2010.05919), **Parente+22** (2204.11884): two-size on gas particles.
- **Hirashita & Aoyama 2019** (MNRAS 482, 2555; 1810.07962): full GSD, 128 points over 0.3 nm–10 µm, piecewise
  constant; growth/sputtering as first-order upwind advection in mass (App. A–B); Smoluchowski sums with the loss term
  at the new time ("semi-implicit ... stabilizes"); Δt = 0.3 τ_coll with sub-cycling. N = 16 vs 128: bump position and
  width differ at ~1 Gyr from "numerical diffusion which is unavoidable in our simple discretization method".
- **Aoyama, Hirashita & Nagamine 2020** (MNRAS 491, 3844; 1906.01917): full GSD on each SPH particle, 32 points; HA19
  scheme. **Romano, Nagamine & Hirashita 2022** (MNRAS 514, 1441; 2202.05521): Gadget4-Osaka, 30 radii, turbulent
  dust diffusion.

## RAMSES and methods

- **Dubois+24** (2402.18515): RAMSES two-size, carbonaceous and silicate, passive variables "in the exact same vein as
  for the transport of metals"; no differential motion.
- **Rodríguez Montero, Dubois, Katz, Slyz & Devriendt 2026 (CALIMA)** (2602.21790): two-size, two compositions,
  log-normal populations plus PAH bins. §5.2: processes evaluated "at the central grain size of each bin"; Smoluchowski
  cost ∝ N_bins²; proposes moment methods (Mattsson 2016); notes that large grains decouple in hot, low-density gas.
- **Lebreuilly, Commerçon & Laibe 2019** (A&A 626, A96; 1905.01948): RAMSES multigrain, fixed sizes, monofluid with
  terminal-velocity drift per species, second-order MUSCL.
- **Lebreuilly+23** (2211.01891): 1-D collapse with coagulation/fragmentation, Kovetz–Olund-style collector; needs
  N > 100 bins to converge.
- **Lombart & Laibe 2021** (MNRAS 501, 4298; 2011.12298): discontinuous-Galerkin coagulation, mass to machine
  precision, few bins; surveys two-moment sectional methods (Gelbard/Landgrebe, Pratsinis) and a piecewise-exponential
  scheme (Soong 1974). Companions 2208.02964, 2109.11065.
- **Mattsson 2016** (1606.02272): method of moments with interpolation closure; one-zone, not adopted by hydro codes.

## Comparison

| code | bins | moments | in-bin shape | growth/sputtering | collisions | transport | arXiv |
|---|---|---|---|---|---|---|---|
| AREPO, McKinnon+18 | 16 | N, M | linear in a (limiter) | shift + intersection remap | HY09, explicit + rescale, sub-cycled | dust particles, one stopping time | 1805.04521 |
| GIZMO-Simba, Li+21 | 41 | M (+N), constant | constant | not detailed | McKinnon-style | dust particles + drag | 2012.03978 |
| AREPO-SMUGGLE, Narayanan+23/26 | 16 per species | as McKinnon | as McKinnon | as McKinnon | as McKinnon | superparticles | 2301.07136; 2606.20809 |
| GIZMO-FIRE, Choban+26 | 16 per species | N, M | linear in a (limiter) | shift + intersection remap | HY09, explicit, sub-cycled | gas cell, turbulent diffusion | 2603.08504 |
| GIZMO-FIRE, Choban+22/24 | — | mass | fixed GSD | timescales | — | passive scalar | 2201.12369 |
| GADGET-Osaka, HA19/Aoyama+20/Romano+22 | 128/32/30 | M | constant | first-order upwind | Smoluchowski, semi-implicit loss | SPH particle | 1810.07962; 1906.01917; 2202.05521 |
| GADGET two-size (Osaka, MUPPI) | 2 | M | two-size | timescales | timescales | SPH particle | 1412.3866 ... |
| RAMSES-CALIMA, Dubois+24, Rodríguez Montero+26 | 2 per composition | M | two-size (log-normal) | timescales | timescales | passive scalar | 2402.18515; 2602.21790 |
| RAMSES multigrain, Lebreuilly+19 | fixed | M | — | — | — | monofluid + TVA drift | 1905.01948 |

## Where PL2M sits

- Closest: McKinnon+18 and Choban+26 (two moments, shape from M/N, shift + conservative remap); Parente+26 for the
  growth step alone.
- Differences: power law in m instead of linear in a (never negative, no limiter needed [inference]); positive,
  conservative MPRK22 instead of explicit first-order updates with mass rescaling and sub-cycling (no dust paper found
  that does this); two nodes per bin keep in-bin information in the collision kernel (McKinnon drops the slopes there).
- New in the set: per-composition GSDs with only 8 two-moment bins (others 16+, or 30–128); N and M as passive scalars
  plus a terminal-velocity drift with an evolving GSD in an Eulerian code (TVA exists only for fixed sizes,
  Lebreuilly+19) [inference: seems unprecedented].
- To check: an accretion-front test comparable to McKinnon's (≈ 1 % mass error at N = 64) and HA19's N = 16 diffusion;
  Lebreuilly+23's N > 100 for single-collector coagulation as motivation for the two-node kernel.

## Added by the main session (2026-11-17, after the user's pointers): McKinnon+18 against FIRE and GADGET4

Read from the arXiv full texts in `reference_data/gsd_solvers_2026-11-17/` (mckinnon2018, choban2026, gadget4_2026,
diallo2026).

- **Choban+26 (FIRE, MNRAS 549, stag1020; 2603.08504)**, Sect. 3.3 and App. C–E: "largely follows the second-order
  piecewise linear method developed in McKinnon et al. (2018)". Same N_i + slope per bin (Eq. 2), slope from M_i,
  slope limiter keeping M_i (App. E1), the same shift-and-intersect update (Eqs. E5–E8) and the same top-edge
  rebinning (Eqs. E9–E11 = McKinnon Eqs. 44–46); collisions after McKinnon App. A. Differences from McKinnon: the GSD
  lives in each gas cell (moves with the gas, turbulent diffusion as for metals) instead of on dust particles; one GSD
  per species (silicate, carbonaceous, Fe) instead of one; a_min = 5 nm, 16 bins; explicit sub-cycling with
  Δt = 0.3 δa_1/ȧ for accretion/sputtering and, for collisions, both a mass-moved and a number-lost criterion (0.1 each;
  "without subcycling, the abundance of large grains was heavily reduced"); sub-resolution clumping factors
  C2 = 1 + b²M², a Coulomb enhancement of accretion, and an ad hoc coagulation density enhancement (C_coag below
  T_cutoff). Convergence (App. C): GSD at 16 bins, extinction nearly at 8.
- **Matsumoto+26 (GADGET4-OSAKA, A&A 705, A75; 2508.21157)**, Sect. 2.1: 30 bins over 3e-4–10 µm, MW-like isolated
  galaxy, 10 Gyr; the dust model of Aoyama+20 and Romano+22a (i.e. the HA19 discretisation: mass per bin, piecewise
  constant, first-order upwind growth, semi-implicit Smoluchowski loss); one GSD with no composition (composition
  assigned in post-processing for SKIRT); a two-phase sub-grid model in every gas particle (f_dense = min(α n_H, 1),
  α = 0.12, dense phase at 50 K and 1e3 cm^-3) with processes split between the phases; dust carried by the SPH
  particles with turbulent diffusion (Romano+22b).
- **Diallo+26 (RAMSES-MCR; 2602.21147)**, Sect. 2.2: cosmic-ray protons in momentum bins with a power law per bin and
  two moments (number, energy), the slope interpolated from tabulated e/n (Eq. 20); bin-edge fluxes integrated over the
  power law across the momentum swept in Δt (Eqs. 33–37), with Δt limited to 0.1 of the bin's log width; in-bin
  loss terms by quadrature with a semi-implicit update (Eqs. 38–40); open boundaries extrapolating the neighbour's
  slope; and, after Hopkins (2023), separate effective diffusion coefficients for the number and energy moments,
  because one bin-centred coefficient keeps the inner slope fixed (Sect. 2.2.5).
