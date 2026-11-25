# GSD modelling: status and what is left to do (as of 2026-11-24)

The canonical to-do list of the grain-size-distribution (GSD) work: what to do next, the decisions taken and still
open, and the build (the plan's work packages A and B). Keep it in step with the work.

**State.** PR #19 (the WDB06rt/LG5 charge solve ported from RAMSES), PR #20 (the GSD work) and PR #22 (the collide fix)
are merged into pyCALIMA main (89ee18a); the GSD checks were rerun on it on 2026-11-22 (section 1). On 2026-11-23 the
carbon mass transfer was settled before A3c (report section "How carbon moves ..."; notes `overview/lit/carbon_transfer_*.md`,
outputs `overview/tables/carbon_transfer/`, scripts `check_carbon_heating_share.py`, `check_carbon_transfer_timescales.py`,
`check_carbon_transfer_options.py`, `check_seam_decomposition.py`). A3c planning resumes with those decisions.

**A3c (23-24 November), branch `feature/gsd-a3c` (off main 89ee18a; this status branch merged into it).** The per-cell
operator `gsd.cell` is built and tested: `precool` and `update`, the two calls RAMSES will mirror, with every phase of
the plan but the hot-gas join (its tables are being built on megatron). Summary in the report's section "The per-cell
operator (A3c)"; the list below is in section 3. Working rules (user, 2026-11-23): `pytest -m gsd_quick`
(`diagnostics/gsd/run_quick_checks.sh`, ~1.5 min; `CORES=` on megatron) and the golden outputs
(`tests/data/gsd_golden.npz`) at every commit, one logical change per commit named by module, the full suite before
each push (with `CALIMA_MODEL_DATA` pointing at a complete model_data: a partial `model_data/` in a worktree makes
two legacy solver tests fail instead of skipping).

**Where the work lives (since the cleanup of 2026-11-25).** The GSD line is the branch `feature/pahrt`, pushed:
- It holds `feature/gsd-a3c`, this status branch and `study/pahrt`, all three deleted as branches since then.
- Its worktree is `../pyCALIMA-pahrt`, to be renamed `../pyCALIMA-gsd`. The ignored `results/` there holds the
  outputs of every earlier GSD worktree.
- `reference_data/` is a single link to the main checkout's.

pahchem changes are made in `../pyCALIMA-pahchem` and reach this line only through main. After each merge of main,
rerun the GSD checks that read pahchem.

**Where the detail lives:**
- the status report: `overview/gsd_overview.tex` (its figures and the PDF are built in the git-ignored
  `results/gsd_overview/`; see `overview/README.md`), and the variant study `overview/variants/REPORT.md`;
- the numbers behind the report as of 2026-11-22: `overview/tables/`;
- the working plan and the charge port's study (`RAMSES_dev/studies/pycalima_port_20261118/`) are local, outside
  this repository.

The sections are in order: what to do next, then the decisions only the user can make, then the build (the plan's
work packages).

## 1. Next (small, unblocked)

- [x] **Rerun the GSD checks on main** (done 2026-11-22, main 89ee18a = #19 charge port + #22 collide fix; no code
  changes). Outputs in the git-ignored `results/rerun_2026-11-22/`
  (old outputs in `before/` and `old/`, comparison scripts `cmp_*.py`, `attrib.py`, `coag_bins.py`); new A0 runs
  installed in `results/binconv_v4/` and `results/gsd_overview/variants/a0_prod/`; PDF updated ("Rerun on 22 November" in the summary,
  Sects. 4, 6.7, 7.1-7.3, Table 7, the next steps).
  - A0: accretion/sputtering bit-identical; SN/coagulation <= 0.3 points. **Shattering at 8 bins: EA 4.4 -> 7.1 %,
    PL2M 7.8 -> 5.7 %.** The coarse runs moved, not the reference (0.24 %); converged in time (errmax 1e-5 equal).
    EA worst at 8 bins 7.1 % (was 4.4); 12 bins 3.9 %, 32 bins 0.6 % (shattering now converges steadily).
  - Rates at 8 bins (all the rise from shattering): area node 8.8 / 7.3 % (was 7.9 / 6.6), EA exact 4.4 / 4.6 / 6.2,
    Gauss 5.4-7.1, mean mass 13-14, centre 42-50 %; other four cases 0.7-2.0 % (area node).
  - In-bin (single bin, #19 only): area node median 1.4 %, p95 7.0 % (was 1.6 / 7.8), worst 9.7 % unchanged; Gauss
    worst 4.2 -> 1.6 %. Changes sit in the large carbon bins (53 nm-1 um) of the PDR cell.
  - McKinnon coagulation at 32 bins: worst bin 16 -> 24 % (0.039-0.049 um). The old 16 % was a step artefact (old code
    at errmax 1e-5 also gives 23 %). Shattering test unchanged.
  - Unchanged: the seam (grain side <= 0.3 %), `ac` vs DustEM, the AGN a_CE, RT pseudo-bins (IR 1.3 -> 1.4 %).
  - Not rerun: the variants bench (`bench_pl2m_variants.py`), broken since the GSDMethod API (patches
    `bin_samples` / `bin_nodes` with old signatures). See the item below.
- [ ] **Re-measure the collision variants through `GSDMethod`** (kernel averaging, Gauss nodes; McKinnon shattering/
  coagulation, threshold sweeps, constant/additive kernels) on the fixed collide. Needed before decision 2 (KA
  default); the PDF's variants section is marked as a dated study.
- [ ] **`docs/api/gsd.md:183`** on main still says "worst A0 observable 12 % to 4.4 %": now 7.1 %. Deferred (user,
  2026-11-22): review the docs once the GSD work is done, in one pass.
- [x] **8 bins after the collide fix:** EA misses 5 % at 8 bins only in shattering (7.1 %, graphite mean size); under
  10 % everywhere; 12 bins would meet 5 % in every case. **Decided 2026-11-22 (user): keep 8 bins** ("if we can limit
  as much the number of operations and memory the better").
- [ ] **megatron** (needs the VPN):
  - verify `RAMSES_dev/studies/pycalima_port_20261118/src_recon/c1..c6` against `calima/charge-lg5` 7f842bb4 and
    the commits before it (7f842bb4 stays the reference after the RAMSES side's guard-fix commit);
  - build the new bench driver (lgx_bench plus wdbz_bench's columns 33-47, both modes, cores 74-95 until 64-73 are
    clear) and compare:
    - d/d ln T, d/d ln S, d/d ln N;
    - warm starts on the Fortran side;
    - alpha and D not wanted (wants = 0).
- [ ] **Review** the relaxed test in `tests/test_gsd_methods.py::test_remap_round_trip_and_conservation`. The
  identity remap is now checked to 1e-13 of the largest bin, not exact zeros in empty bins: the affine/exp closures
  left 5e-17 on Linux CI.
- [ ] **Side finding, not fixed:** a collapse test writes `model_data/collapse/` into the checkout. On the next run
  two `test_solvers` sputtering tests then fail instead of skipping (their skip only checks that model_data
  exists).

- [ ] **Hot-gas join (A3c phase 6, the last open item):** the `--hot-gas` hydrogen tables (axes log G0 -1..6, log n_H
  -1..6, log T 4..7.5, log x_e -1..0; 77 molecules, deep super-H chain +8) are being built on megatron
  (`~/pycalima_runs/hydrogen_tables/run_hot_k8.sh`). Then `diagnostics/gsd/check_hot_erosion_join.py --hot <dir>
  --cold <deep_k8>`: the carbon loss per C of the top molecular bin against classical sputtering at N_C 400 and
  1000 over 1e5-1e8 K, and the two table sets at 1e4 K; beyond x2 at the seam, a labelled interpolation towards
  N_C ~ 1000 (Bocchio+12). Then a hot cell in `run_cell.py demo`.
- [ ] **Calibrate Y_pd (A5).** Thermal release is negligible at 8 bins (the smallest host bin's area node is 1.5 nm, the
  smallest molecular bin's mean N_C ~35), so the photodesorption yield decides freeze-out: in the demo, Y_pd 1e-7-1e-6
  keeps 0.12-0.14 of the carbon in molecules in the CNM (0.002 at Y_pd 0) and 0.10 in the PDR; dense cores freeze out
  at any Y_pd.
- [ ] **Dense-core band gap (for the dense-gas gates):** the covalent bins sit at the UV floor (0.22 eV) even at
  G0 = 0.01, because the lab re-hydrogenation law's activation energy (Mennella+06) cuts k_H ~1e3-fold at 10 K. What
  the adopted laws give; check against dense-core a-C:H mantles in A5.
- [ ] **Open from the carbon-transfer study (2026-11-23):**
  - ~~hot-gas erosion~~: tables extended (A3c phase 6), join pending (item above);
  - Coulomb explosion: fragments (a^-3.3, labelled) or ion loss, set by the assumed tensile strength (crossover S_max
    ~ 8e10 erg cm^-3 against ion field emission at ~3e8 V/cm);
  - P1's a-C:H end and the bare end (pahchem bare states or a-C physics);
  - redo the band test (3-4 bins over N_C 20-400; no script survived) with the extended N_C,max test;
  - ~~implement the pair rule and the release term~~: done in A3c (`gsd.pairs`, `gsd.release`).

## 2. Decisions waiting for the user

1. **Bottom-up PAH channel** for the molecular bins.
   - Recommended: option D, a Form-A source calibrated on TMC-1 plus one reaction-limited growth efficiency
     eps <= 0.15 calibrated on the translucent q_PAH rise (Zhang+25). PDF Sect. on growth.
2. **Kernel averaging (KA)** in the collisions as a default: test it in the full collision model first (5.5x the
   collision cost; it fixes sharp shattering thresholds).
3. ~~**a_H,max**~~ **Decided 2026-11-22 (user): option B.** The four covalent bins below 53 nm take their band gap from
   the local equilibrium (x_eq = (k_H x_max + k_UV x_min)/(k_UV + k_H)), nothing tracked by default; the M_H block is
   built but compile-time 0-4 bins (default 0). A5's ablation against Orion Bar / Horsehead / NGC 7023 decides whether
   to switch tracking on, bin by bin (20-53 nm first for FIR/extinction, 1-3 nm for 3.4/3.3 emission). Why: heating
   and charging are safe in equilibrium (FUV path budget <= 0.03 / 0.19); the failures (visible, 3.4 um, FIR for cloud
   material leaving clouds) are mostly post-processing observables; tracking costs 16 B/cell + 1 transported scalar
   per bin, while the E_g-axis tables are needed either way.
   Rung M1 itself (tracked M_H in the covalent bins) is adopted only if it fixes an observable M0 misses.
4. **Super-H physical stop for large PAHs** (a pahchem question): barriers past +24, the Eley-Rideal bracket, a check
   against the 3.4 um band. Until then the H2 and r_ali tables of PAH bins 3-4 are provisional, and the production
   tables need the user's choice of super-H chemistry and cap (decision of 2026-11-15).
5. ~~**The PAH/grain seam**~~ **Decided 2026-11-23:** the gap is the electron sticking on cations (x1.3-2.7 of
   the x1.4-2.5 heating difference), unmeasured beyond N_C 16 (range 0.3-1). One labelled law: molecules on pahchem's
   lab_cold, `ac` taking the same value at N_C 400 and tapering to WD01 by ~3 nm; A5 runs the brackets (WD01
   everywhere; molecular everywhere).
6. **Associative detachment on intact PAH anions**: it decides the dense-gas PAH charge.
7. **A finer G0 axis above 1e4** for the carbon-loss tables of PAH bins 2-3, if runs reach fields near massive stars.
8. **Default N_C,max** (400) after the extended band test; whether to use a xi(a) transition band for the DL07
   comparison.
9. **Smaller choices:**
   - H2 release from cold grains: diffusion-limited or immediate;
   - whether sputtering removes H first.
10. ~~**Carbon material**~~ **Decided 2026-11-22 (user): THEMIS a-C(:H) (rho 1.6) is the default, graphite the control**
    at every rung. A material is a table set (`gsd/materials.py`), so switching stays
    possible; graphite proxies for shattering, sputtering and sublimation stay labelled. Revisit if a-C does worse than
    graphite on T3 (diffuse ISM). A0 and the rate study used graphite 2.2; A5 reruns them on a-C.
11. **Which observational datasets to fetch first** for T3/T4 (Gordon+23 extinction, tau_3.4/A_V, the JWST PDR
    nano-grain results, Pilleri+15 maps, ...); the full list is in the plan's "Data".
12. **The RAMSES feature branch** for phase B: ask before creating it.
13. **RAMSES's truncated pi** (constants.f90: 6.2831853/2, 1.1e-9 off). pyCALIMA mirrors it (`PI_RAMSES`); a fix
    would be its own RAMSES PR and change D by ~5e-10.

Decided (for reference):
- carbon mass transfer (2026-11-23): no PAH clusters (rung M2's cluster part and P4 removed); legacy CALIMA
  PAH-PAH / PAH-grain methods not reused; molecule + molecule no outcome; molecule + larger particle by impact speed
  (labelled, C60/cluster surface-impact data): stick < ~4 km/s, rebound between, molecule destroyed > ~7 km/s
  (carbon to the gas; destroyed in warm turbulent gas, frozen out in cold gas); sticking times S_eff = 1/(1 + R_abs
  (P_th + Y_pd) t_bury), Y_pd a labelled calibrated photodesorption yield (default 0, calibrated in A5 on
  molecular-cloud parity and dense-core depletion); the hydrocarbon-family identity below N_C,max (2026-10-30) stands;
- photo-release on every host (2026-11-24, user: "the photodesorption should consistently act across the whole
  population"): S_eff applies to molecules on nano-grains and grains alike (was nano-grains only);
- A3c (2026-11-23): an exploding grain's mass goes to a^-3.3 fragments (gas selectable); the hot-gas extension of the
  molecular tables inside A3c; Track O targets G23 + the HD23 model (HD21 has no tables);
- two moments, EA closure, 8 bins (re-confirmed 2026-11-22 after the collide fix);
- grain hydrogen: equilibrium band gap, nothing tracked by default; M_H block compile-time 0-4 bins (2026-11-22);
- carbon material: THEMIS a-C(:H) default, graphite control (2026-11-22);
- per-moment TVA drift and sinks;
- RT optics: 5 PL2M pseudo-bins per bin;
- grain rates: one charge solve per bin at the area node;
- the overflow bin: one wide bin per composition over 1-10 um (23 November; was open-ended above 1 um), computed only
  above 1e-6 of the cell's dust mass, warning at 1 %;
- PAH hydrogen tabulated in equilibrium;
- the predictor RAMSES-only;
- gsd.charging on LG5, with lattice=False for the full solve.

## 3. Build: pyCALIMA (phase A)

**A3b, the static per-bin tables:**
- [x] Grain rates over the in-bin profile with the adopted area node: charge kernels at 3-9 radii per bin (A3c phase
  3, `check_area_node_tables.py`; 79 tables, ~28 MiB in RAMSES format for the 15 grain bins). Still to measure: the
  memory against the planned coarsening (real*4, a cut potential range).
- [x] Optics with a band-gap (E_g) axis for the covalent bins, with Planck means (`optics.PseudoBinOptics`, 49 E_g
  nodes; Rosseland means not needed by the operator).
- [ ] For RAMSES, the E_g axis multiplies the RT pseudo-bins: 5 s x ~6 E_g on the 4 covalent bins is ~30 MB of
  SED_table per node (needed with or without M_H, since the equilibrium band gap reads the same axis). Measure fewer E_g nodes, or a coarser age axis, before B6/B7.
- [x] Molecular-bin charge and photoelectric heating per GSD bin through `pahchem.group_tables`, on the GSD's 12
  photon groups (`build_molecular_group_tables.py`; not yet conditional on the H state). The top bin (N_C 189-400) has
  only 5 molecules complete in -1..+2 in PAHdb.
- [ ] Molecular-bin band emission per bin, for post-processing (SKIRT), not for RAMSES.
- [ ] Re-gridded, real*4 cooling and sputtering tables on (log T, phi/kT). Each "to verify" of the table-memory
  proposal is a test: build full, rebuild coarse, compare the interpolated rates at random conditions.
- [x] Charging along E_g: the 5 `acx<E_g>` nodes, linear in H/C (decided 2026-10-30), wired in
  `charging.AreaNodeCharging(gap_bins=)`.

**A3c, the `ramses` mode of `gsd`** (the per-cell operator RAMSES will mirror), built on `feature/gsd-a3c`:
- [x] the operator `gsd.cell`: `precool` (charges, heating, cooling, explosion rates, grain area, T_d, group absorption,
  band gaps, molecular charge and H2, S_eff) and `update` (collisions with the pair rule, Coulomb explosion as a
  per-moment sink feeding a^-3.3 fragments, erosion with the molecular carbon loss, accretion and sublimation, the H
  step); the A0 oracle (`run_cell.py oracle`) agrees to round-off with the features off;
- [x] EA with Gauss rate nodes (the default method);
- [x] the pair rule (`gsd.pairs`) and the photo-release (`gsd.release`; on every host since 2026-11-24);
- [x] equilibrium band gap per covalent bin (x_eq from k_UV, k_H) feeding the E_g-axis optics and charging (default,
  decision 3);
- [x] the M_H tracer with the exact H step, carried at the donor H/C by every operator (growing grains keep their H,
  shrinking ones their H/C); tested on, off by default; not for the 'pl2m' closure;
- [x] the overflow bin (passive below 1e-6 of the composition's mass; share logged, warning at 1 %, strict mode);
- [x] per-moment size-dependent sinks (`gsd.sinks`);
- [x] area-node charge-dependent rates (`AreaNodeCharging`, seam sticking law) and the grain area from the exact second
  moment;
- [x] pseudo-bin optics (`optics.PseudoBinOptics`, `material_optics`), T_d per bin from the energy balance;
- [x] the molecular bins: carbon loss as erosion (`gsd.molecular`, process 'carbon_loss'), charge/heating/R_abs from
  the group tables, H2 from the hydrogen tables;
- [ ] the hot-gas join (section 1).

Track O (can the decomposition meet the basic constraints?):
- [x] O1, the binning alone (`check_o1_binning.py`): <= 0.4 % on THEMIS J13, <= 2 % on DL07; the N_C >= 20 cut costs
  DL07's 3.3 um band 17 %.
- [x] O2, our materials (`check_o2_fit.py`, smooth parametric size distributions): a-C(:H) + astrosilicate fit the
  Milky Way (G23 extinction, the HD23 model's emission) at chi2/point 1.97, 2.02 as the GSD carries it; extinction
  within 9 %, 8-12 um emission 33-42 % low (monochromatic points), silicate at its abundance limit; graphite control
  5.4; LMC/SMC curve shapes within 9-26 %.

Conservation tests: carbon and hydrogen to round-off, H/C within bounds (done). To try there:
- [ ] a third collision node, or bin edges at kernel thresholds (the coagulation threshold bin);
- [ ] a thermal H-loss sensitivity run for the 1-3 nm bin.

**A4, the advection prototype:**
- [ ] N following M_C through the ratio pass, and M_H as a CMA child of M_C;
- [ ] the per-moment TVA drift, with the CFL on w(a_R);
- [ ] the six tests of the plan:
  - MUSCL drift with 0 clips;
  - AMR front with C and H conserved;
  - Q_pr / Epstein drifts with size-dependent direction;
  - drift + growth + collisions;
  - integral against two-node w^N, w^M;
  - M_H at the donor H/C.
- [ ] then choose between the integral and the two-node velocities.

**A5, accept or reject:**
- [ ] T2: our tables against DustEM's SED/EXT (THEMIS), and the graphite control against DL07.
- [ ] T3: the diffuse ISM per H, the 2175 A bump, tau_3.4/A_V, the carbon budget, the growth gates.
- [ ] T4: one-zone and slab runs for NGC 7023, the Orion Bar and the Horsehead, plus the TMC-1, phase-parity,
  translucent, metallicity and Sextans A gates.
- [ ] The full rate-convergence test: 6, 8, 12, 16 bins and a reference, rates coupled to the gas.
- [ ] The ladder ablations that adopt or reject rungs M1-M3.
- [ ] The growth gates once the bottom-up channel is chosen.

**A6, the exporter `models/export_gsd_tables.py`** (with provenance headers; estimate the memory with 1 worker first):
- [ ] node tables for pyCALIMA and post-processing;
- [ ] for RAMSES: the pseudo-bin sigma(lambda), Planck and Rosseland tables (5 per bin, Chebyshev s);
- [ ] the charging kernels at radii;
- [ ] the re-gridded cooling and sputtering tables;
- [ ] the PAH group and hydrogen tables.

**pahchem (new modules only):**
- [ ] P1: the hydrocarbon family (states up to N_H ~ 1.4 N_C).
- [ ] P3: the H-coordinate tables.
- ~~P4: the cluster population~~ (dropped 2026-11-23).
- [ ] P5: a-C(:H) photo-fragmentation (Jones & Ysard 2025).
- [ ] The production PAH hydrogen tables wait on decision 4.

## 4. Build: RAMSES (phase B; branch off staging, ask first)

- [ ] **B1, the state:** the M_C block, then N, then M_H (its length from a_H,max: compile-time 0-4, default 0), the overflow bin (2 scalars per
  composition). NGSD=0 must stay byte-identical.
- [ ] **B2, the sources per cell in `dust_solver.f90`:**
  - the exact H step;
  - the carbon-moving processes carrying H/C;
  - per-moment sinks;
  - area-node rates;
  - the overflow bin with its diagnostic.
- [ ] **B4, the TVA drift:** recomputed, not stored (two velocities per bin from the bin's s and the drift at two
  sizes).
- [ ] **B6/B7, the tables per GSD bin:**
  - the RT optics as 5 pseudo-bins (ordinary dust bins for SED_table, star polling and the IR means; linear blend
    in s once per cell-call; css dropped);
  - charging, molecular or grain type from each table's header (band emission goes to the post-processing
    exporter, not to RAMSES);
  - an E_g axis for the 4 covalent bins (equilibrium band gap by default, the tracked H/C where M_H is carried).
- [ ] **Transport:**
  - the ratio pass at the CMA hook, with N outside the CMA groups;
  - M_H as a CMA child (slopes on H/C), and donor-ratio TVA fluxes;
  - an HLLC, exact or acoustic Riemann solver;
  - the realisability clip as a counted safety net (target 0).
- [ ] **One copy of the tables per node** (MPI-3 shared memory); real*4 tables.
- [ ] **Targets:**
  - a CALIMA step at most 1.5x legacy;
  - Godunov at most +35 %;
  - tables at most 46 MB per rank;
  - the nightly suite byte-identical with NGSD=0.
- [ ] RAMSES's `calima/dust_charging.f90` header comments (l. 50, 60, 77 at 7f842bb4) to update now that the port has
  landed (RAMSES side).

## 5. Known limitations and side findings (recorded, not fixed)

- **A3c, labelled in the code:**
  - the burial time of S_eff is capped at 1 Gyr (`Options.t_bury_max`): without growth it is infinite, and release
    rates that underflow to ~1e-300/s would otherwise empty 10 nm hosts; burial counts accretion growth only;
  - the molecular charge above 1e5 K is the group tables' clamped value (no hot-gas collisions);
  - the UV background's absorptions are split over the groups by the group-averaged cross-sections;
  - the hydrogen tables' H2 is for their Draine field shape scaled by G0;
  - astrosilicate optics are extrapolated below 1e-3 um (X-ray groups);
  - every grain temperature is an equilibrium one, also for the smallest grains;
  - the x <-> E_g map of a covalent bin is its mid state's (s = 0.5).
- **Python cost of the operator** (one-zone demo, 21 bins, all features): ~165 ms per precool and 25-420 ms per
  update; RAMSES writes its own kernels.

- **wide_moments:** a wide P(Z) of a large grain with a spectral break inside +-1.5 sigma. In the 162 nm graphite
  PDR cell sigma is +12 % (Gaussian -11 %), a ~1e-4 effect on the rates (RAMSES side: within the bar, not tuned).
- **LG5 in pyCALIMA** is pure Python (10.7 ms per 22-bin cell against the Fortran's 13 us). A numba path, if it
  matters.
- **The trimmed tables of `tests/data/rtgroup_parity_lg5.npz`** reproduce the full ones to ~3e-15 in g. alpha_fit is
  not in the fixture: its break charges read the whole potential grid.
- **Legacy code:**
  - `dust_sputtering.py` setup bugs (`smallSil` reads the large-silicate entries; `largeSil` indexes past 6);
  - carbon density appears five ways (2.0 / 2.2 / 2.24 / 2.26 / 2.27);
  - `grain_distributions.py` puts `np.log` in a `logspace` start;
  - graphite Mie uses `CpaD03_0.01` against `Im_n` at 0.10;
  - the silicate MieTheory fast path has a C_sca jump at |m|x ~ 1000;
  - unused `att_len_dust` and `compute_shattered_fragments`;
  - `wdb06_yields._material`: any non-graphite label falls back to silicate outside the GSD path.
