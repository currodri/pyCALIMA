# Literature survey: two-moment, power-law-in-bin schemes for size/momentum distributions

Sonnet subagent, 2026-11-17, for the PL2M-variants study. Saved by the main session.

**Reliability (read first).** The subagent read no full text. Entries marked [V] were verified by search (identifier
and abstract only); [M] are from the model's memory and NOT checked; [I] are its inferences. Treat [M] items as leads to
verify before relying on them. Note from the main session: the per-moment (number/energy) diffusion coefficients of
Diallo et al. (2026, RAMSES-MCR, Sect. 2.2.5, after Hopkins 2023) WERE verified by the main session from the full text
(`reference_data/gsd_solvers_2026-11-17/diallo2026/`), although this survey could not confirm them.

Verified identifiers: Lombart & Laibe 2021 (arXiv:2011.12298); Lombart et al. 2022 (arXiv:2208.02964), 2024
(arXiv:2404.11851); Diallo et al. 2026 (arXiv:2602.21147); Rosdahl et al. grey two-moment CR (arXiv:2509.19447);
Girichidis et al. 2020 (arXiv:1909.12840), 2023 (arXiv:2303.03417); Hopkins et al. 2021/22 CR spectra
(arXiv:2109.09762); CRESCENDO II (arXiv:2607.14363); Kumar et al. 2006 cell average technique (Chem. Eng. Sci. 61,
3327); Laurent, Sibra & Doisneau 2016 TSM (Commun. Comput. Phys. 20(4), no arXiv); Sibra et al. 2017 (J. Comput.
Phys.); CRESP II (arXiv:2505.15614). Not located: Jones & Kang 2005 (CGMV), Ogrodnik 2021, Böss 2023, a Hopkins
2022/2023 numerics paper; no identifiers checked for Tzivion, Chen & Lamb, Bott, Jacobson, Soong, Kumar & Ramkrishna
1996, Gelbard, Simmel/Wurzler, Kah.

## A. Two-moment sectional methods
- **Tzivion, Feingold & Levin 1987, 1989 [M]:** number and mass per bin; collision transfers as exact integrals of a
  linear in-bin profile f = a + b m over the part of the product landing in each bin; both moments conserved.
  Defects: a linear profile goes negative near the edges (clipped), edge pile-up, diffusion in condensation. Closest
  prior to PL2M's defect 1; PL2M is a positivity-guaranteed variant of the same closure [I].
- **Chen & Lamb 1994 [M]:** prognostic mean mass in bin schemes; guidance on partial bins.
- **Gelbard, Tambour & Seinfeld 1980 [M]:** sectional, piecewise constant.
- **Landgrebe & Pratsinis 1990 [M]:** moving sectional model.
- **Kumar & Ramkrishna 1996 [M]:** fixed pivot (products split to keep two moments; overpredicts number at the tail on
  geometric grids [V via the CAT abstract]); moving pivot (pivot = stored mean size).
- **Kumar et al. 2006, cell average technique [V abstract]:** newborn particles averaged in each cell, then assigned
  to neighbouring nodes keeping chosen moments; fixes the fixed-pivot overprediction, usable near gelation.
- **Jacobson moving centre [M]:** low diffusion in condensation; rebinning when crossing centres.
- **Bott 1998, 2000 [M]:** flux method for collection with a linear-in-mass profile and positive-definite limiting; much
  less diffusive than first-order upwind.
- **Simmel et al. 2002, Wurzler, Khain et al. 2015 [M]:** scheme differences come mostly from the in-bin kernel
  treatment and growth diffusion; tails converge only at tens of bins per mass doubling.
- **Laurent, Sibra & Doisneau 2016, TSM [V abstract]:** two size moments per section with a piecewise affine
  reconstruction, analysed for stability and accuracy; fewer sections than one-moment methods. [M] A positivity
  condition on (N, M) defines the realisable domain; at its boundary the profile is switched to a reduced form —
  directly relevant to defect 1.
- **Kah et al. 2012, EMSM [M]:** entropy-maximising closure; for two moments the maximum-entropy profile is exponential
  (a different closure with a theoretical basis); tabulated exponent vs moment ratio.

## B. Power-law / exponential reconstructions; cosmic-ray spectral schemes
- **Soong 1974 [M]:** piecewise exponential drop spectra.
- **Miniati 2001, COSMOCR [M]:** first piecewise power-law CR scheme; slope solver trouble at the steep end.
- **Jones & Kang 2005, CGMV [M, not located]:** same structure as PL2M; reported spurious accumulation at a bin edge
  when a cut-off lies inside a bin, slope limits for steep slopes, first-order in momentum for advection.
- **Girichidis et al. 2020 [V abstract]:** piecewise power law with local amplitudes and slopes, more accurate than
  piecewise constant at few bins; [M] energy-loss flux through bin edges from the upwind power law.
- **CRESP (Piernik) [M]:** non-linear slope iteration; problems with steep spectra and empty bins reported.
- **General pattern [I]:** slope by bisection/Newton on a monotone ratio; steep limit capped or a spike; first-order
  edge fluxes diffuse cut-offs; explicit schemes have a momentum CFL (PL2M's remap avoids it).

## C. Sharpening discontinuities / reducing remap diffusion without extra memory ([M]/[I] only)
1. Harten sub-cell resolution: reconstruct a discontinuity inside a cell from neighbours — replace the power law by a
   partial-support profile when a neighbour is empty (defect 1).
2. Limited-downwind / anti-diffusive remaps (THINC, SLIC, MPDATA corrections).
3. ENO/WENO in size space from neighbour moments, with a correction to keep each bin's N and M.
4. Moment-preserving neighbour reconstructions: slope from neighbours plus a correction restoring N and M; risk to
   positivity/realisability, cured by clipping to the realisable domain (as TSM).
5. PPM in size space (semi-Lagrangian remap with a limited parabola): much less diffusion in repeated remaps.
6. Trade-off: non-local reconstructions need neighbour reads (local to the cell in a hydro code) and can lose
   positivity or the exactness of the in-bin moments.

## D. Collision quadrature with in-bin structure
- More Gauss nodes per bin (cost ∝ nodes² per pair); exact integration of a step kernel over the in-bin power law
  (needs the threshold location); adaptive nodes (not available if nodes are fixed by N, M alone).
- DG (Lombart & Laibe 2021 [V]; 2022 [V]; 2024 [V]: 0.1–1 % with 20 bins over nine decades): order-1 DG stores two
  coefficients per bin, the same memory as PL2M, with a linear profile [I]; a natural benchmark.
- QMOM/DQMOM [M]: different closure (four moments for two nodes); realisability issues.
- Kovetz–Olund / Brauer two-collector splitting [M]: Ohtsuki-type artificial acceleration on coarse grids; remedies:
  finer grids or cell-average weighting.

## E. Hybrids
- Moving pivot for growth + sectional collisions [M] (pivot = stored M/N used as a point).
- ALE practice [I]: remap only when distortion exceeds a threshold (needs stored edges: not allowed); a free variant is
  to remap only the part that crossed an edge.

## F. Stress tests in the literature ([M]/[I])
Constant/additive/product kernels (Scott 1968); coagulation–fragmentation steady states (Dohnanyi; Birnstiel+11;
Lombart); narrow spectra under prolonged constant or size-dependent growth; condensation–evaporation to a fixed point;
CR cooling with a moving exponential cut-off; mixing of populations with different mean sizes.

## Remedies vs defects (subagent's table; benefits are inferences)
| defect | remedy | source | memory | cost | expected benefit |
|---|---|---|---|---|---|
| 1 edge pile-up | partial-support power law when a neighbour is (near) empty | Harten SCR [M]; TSM [V abstract] | none | small | removes the overshoot; needs a design |
| 1 | neighbour-informed slope + moment correction | [I] | none | few reads | smoother; realisability needs clipping |
| 2 remap diffusion | remap only what crossed an edge | ALE [I] | none | small | large, but hard without a stored shape |
| 2 | Bott-type limited flux | Bott [M] | none | moderate | less diffusion than first order |
| 2 | mass fix-up | [I] | none | trivial | conservation only |
| 3 kernel threshold | analytic integration over the power law | [I] | none | small | removes the threshold-bin error if the threshold is known |
| 3 | extra nodes only in threshold bins | [I] | none | small | moderate |
| 3 | cell-average product assignment | Kumar+06 [V abstract] | none | moderate | better product placement |
| 4 bin-centre rates | integrate over the in-bin profile | TFL87, Gelbard [M] | none | small | removes tens-of-per-cent errors |
| 5 edge discontinuity | inherent to two dof per bin (also order-1 DG) | Lombart [V] | — | — | — |
| 6 overflow | open outflow with power-law extrapolation, or top-bin assignment with number from the closure | [I] | none | trivial | no mass inflation |

Ranked under the constraint: (1) rates over the in-bin profile; (2) analytic/extra-node treatment of kernel thresholds;
(3) mass fix-up / single exact step; (4) support-limited power law for partly filled bins; (5) neighbour-informed or
PPM reconstruction with moment correction; (6) cell-average product assignment; (7) order-1 DG as a benchmark.

Stress tests suggested: narrow spectrum under 1e3 growth steps; da/dt ∝ a^p; a sharp edge at several positions inside
a bin; a cut-off under cooling-like advection; condensation–evaporation to a fixed point; constant/Golovin/product
kernels vs analytic; coagulation–fragmentation steady state; a threshold kernel swept through a bin; overflow;
mixing at per-bin drift speeds; convergence at eta 2–19; order-1 DG at equal memory.
