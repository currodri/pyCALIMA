# PAHrt: carrying pahchem's PAH physics into RAMSES (study of 24 November 2026)

How much of pahchem's PAH model (`pycalima.pahchem`) can enter RAMSES as bin-averaged kernels and a per-cell solver,
the way WDB06rt carries the grain charge. The PAH bins are the GSD's molecular bins (`gsd.carbon.CarbonLayout`:
N_C 20-42, 42-89, 89-189, 189-400). This note records what was measured, the issues found, how they could be fixed,
and what was decided. It ends with the decisions of 24 November.

## Setup and reproduction

- **Branch:** `study/pahrt`, off main, with pahchem's open PR #17 (`Population.joint_state_observables`) merged in.
  That PR gives each hydrogenation state its own thresholds. pahchem's own files are otherwise untouched.
- **Reference ("truth"):** each molecule's full joint (Z, N_H) chain (`Params.hydrogen_chain = "joint"`, Andrews super-H
  to N_H0+4) with every output RAMSES would take: f(Z), photoelectric heating, recombination cooling, absorbed power,
  carbon loss, H2, H uptake, the net electron source, C+/H+ recombination on PAHs, the hydrogenation shares
  (`common.py`; the electron bookkeeping checks against charge conservation to machine precision).
- **Molecules:** 5 per bin, evenly in N_C, from C22H12 to C268H52 (bin 4 has only 5 complete molecules in PAHdb).
- **Run** from the worktree root with `CALIMA_WORKERS=1` (pahchem otherwise spawns pools), PAHdb at `$CALIMA_PAHDB_DIR`
  and the population-basis cache in `model_data/pahchem_population_basis`. The figures and npz go to the git-ignored
  `results/pahrt_study/`; each script rebuilds them:

  ```bash
  python diagnostics/pahchem/pahrt/s1_drivers.py --workers 4 && python diagnostics/pahchem/pahrt/s1_drivers.py --analyse
  python diagnostics/pahchem/pahrt/s3_states.py --workers 4 && python diagnostics/pahchem/pahrt/s3_states.py --analyse
  python diagnostics/pahchem/pahrt/s7_tracking.py --workers 4 && python diagnostics/pahchem/pahrt/s7_tracking.py --analyse
  python diagnostics/pahchem/pahrt/s2_binchain.py --workers 4 && python diagnostics/pahchem/pahrt/s2_binchain.py --analyse
  ```

  The full set takes 1-2 h on 4 laptop cores (S2 reads S1's and S7's results). S7 needs the Murga et al. (2020)
  profiles (dataset `murga2020-figshare`, or `--murga2020 DIR`).

## What the equilibrium state depends on (S1, `s1_drivers.py`)

One input at a time around five illustrative phases (WNM, CNM, translucent, an Orion-Bar-like PDR, a dense core; 214
conditions per molecule). Figures: `s1_<phase>.png`, `s1_scaling.png`, `s1_fields.png`, `s1_high_charges.png`.

- **G0 and n_e** set charge, heating, cooling and C+ recombination (2-8 dex over +-2 dex).
- **The field shape at fixed G0 matters as much:**
  - carbon loss moves 4.7-8 dex, H2 up to 2.3 dex, heating 0.7-1 dex, <Z> up to 1.4;
  - a soft 10 kK star: heating x0.2-0.5, carbon loss 1e-4-1e-7;
  - hot 30-40 kK stars: carbon loss x3-60;
  - old-star optical light in dense gas: <Z> up to +0.66 (anion photodetachment).

  So the solver must use the RT group spectrum, not G0.
- **n(H I)** drives the hydrogenation (4-20 H atoms) and H2 (5.7-8.4 dex), and bin 1's carbon loss. **T** acts mainly on
  cooling and H2. The **ions** cap the anion share through C+ mutual neutralisation.
- **Scaling variables:**
  - <Z> collapses on gamma = G0 sqrt(T)/n_e on the cation side only;
  - f(-1) also needs n_e/G0 and the ions;
  - the hydrogenation is not a function of G0/n(H I) alone (T through the addition rate; coupled to the charge in bins
    2-4).
- **Charges above +2:**
  - 4-17 % of bins 3-4 at the PDR base (gamma ~7e4); nearly all of them at gamma ~1e6, with <Z> 3-5;
  - bins 2-4 need them; bin 1 does not (<= 2 %);
  - suspect: C89H27 reaches +6.8 against +2.8 (C72H26) and +3.4 (C90H24), possibly pahchem's ladder extension above its
    top computed charge (a pahchem question).
- One condition failed: C268H52, dense core with old-star light, the multiphoton loss cascade not converged at 150 eV.

## The smallest state set (S3, `s3_states.py`)

Each molecule's chain has 4 charges x (N_H0 + 5) states (110 for C43H17). Figures: `s3_classes_bin<b>.png` (shares of
the bin's totals per state class, grey where its carbon loss is slower than 10 Gyr), `s3_errors.png`,
`s3_deep_bin<b>.png`.

- **Charge +2 must stay:** dropping it costs heating x35-150.
- **The super-H cap at +2** keeps charge, heating and carbon within 1-17 %, but H2 is x3-4 off: every level +1 ... +4
  carries H2 somewhere (the +4 cap binds in shielded gas; super-H decision 4 open).
- **The deep segment** (N_H 1 ... N_H0-4) collapsed into effective transitions to the bare skeleton (splitting
  probabilities of a charge-averaged 1-D chain):
  - exact for bins 3-4, which never fill those states;
  - it fails for bin 1, whose partly stripped molecules hold up to 96 % of the population and make the H2, and stripping
    then sets in too early;
  - it also fails for bin 2 above G0 ~2e5.
- **Two numerical traps** (they would hit a Fortran version too):
  - 1 - p for a tiny climbing probability rounds to 0 and makes the bare state a trap: compute both exit probabilities
    by GTH state reduction;
  - effective rates ~1e-300 underflow GTH pivots: drop transitions below 1e-200.
- **Carbon loss in bin 2** comes from intact catacondensed molecules (C2H2 loss, C60H26 and C72H26: 4e3-1e5 yr in the
  PDR); compact molecules lose carbon only through their (negligible) bare skeleton. Within a bin, carbon loss is set by
  structure more than by size.

## Track the hydrogenation or take it in equilibrium? (S7, `s7_tracking.py`)

Each molecule's full chain is evolved exactly along time-dependent histories:
- PDR crossings: NGC 7023, Orion Bar, Horsehead at 0.3-3 km/s, in and out;
- square cycles: CNM<->WNM, and G0 x10 across the bins' transitions, periods 1e3-1e5 yr;
- random histories: Ornstein-Uhlenbeck in log G0 and log n(H I), correlation times 1e2-1e5 yr;
- three mixtures of two parcels.

Schemes: E (equilibrium at each step); TC (9 tracked class populations, each class rebuilt from its local equilibrium);
TN (the tracked N_H distribution, the charge in equilibrium per N_H). Figures: `s7_paths_bin<b>.png`, `s7_errors.png`,
`s7_mixing.png`.

- **The charge is fast enough to sit in equilibrium:** TN is within a few % everywhere, H2 included.
- **E** is within 1-4 % (time-averaged) on the PDR crossings, but in fluctuating gas:
  - H2 is off in every bin, x100-260 time-averaged in bins 1-2;
  - on cycles of 1e4 yr or less: carbon loss x11 (bin 2) and 0.7 (bin 1), the electron source 70-90 %;
  - heating only in bin 1, 10-12 % (worst 20-43 %);
  - mixing: H2 x60 at 1 yr in bin 1.
- **TC** fixes nearly all of it (<= 2-5 %), except bin 1's H2 (x1.4-3.5: its single deep class is too coarse).

This revisits the November study (docs/physics/pah-hydrogen.md), which tracked one scalar <N_H> and found it no better
than equilibrium.

## One averaged network per bin (S2, `s2_binchain.py`)

Each molecule's chain is lumped onto (Z, class) aggregates (local equilibrium inside each class), and the bin network is
the molecules' lumped rates and per-state outputs averaged state by state, as `group_tables` averages kernels. Figures:
`s2_equilibrium_bin<b>.png`, `s2_tracking_bin<b>.png`.

- **Charge, heating, cooling and the electron source** follow the molecules' mean:
  - bins 3-4: <= 10-13 % worst, ~0.5 % mean;
  - bins 1-2: mostly a few %, except <Z> about 2.5 too high at the PDR's strongest fields (G0 >~ 1e5) in bin 2.
- **The hydrogenation fails in bins 1-2:** the averaged network strips to -16/-22 H where the molecules average -2/-7.
  Carbon loss is off x100-1e7 and H2 x10-300, in equilibrium and when tracked.
- **Bins 3-4** lose only H2 (x1-2 mean, x26-86 worst).
- **Cause:** the bin's molecules strip at rates spanning orders of magnitude (bin 1 over ~2.5 dex of G0/n(H I)). The mean
  of their rates is dominated by the fastest, so the average molecule strips like the most fragile one, while the real
  bin is a mixture of intact and stripped molecules.

## Issues left open, and how they could be fixed

| issue | where | possible fix (not done) |
|---|---|---|
| hydrogenation smeared by averaging rates | bins 1-2 (carbon loss, H2, <N_H>) | split each bin into 2-3 response groups (by the G0/n(H I) where each molecule strips, or by structure: C2H2-capable or not), each with its own averaged network; test at equal cost (e.g. 2 groups x 4-5 classes); or a few representative molecules solved per cell (cost x K) |
| equilibrium misses fast fluctuations | bins 1-2 (H2, carbon, e source); H2 everywhere | track 9 class populations in bins 1-2 (18 scalars; bin 1's deep class split for H2) |
| charges above +2 | bins 2-4 at gamma >~ 1e4-1e5 | a top charge per bin, or one lumped ">= +3" state with its own kernels |
| C89H27 at <Z> +6.8 | pahchem's ladder above its top computed charge | check with the pahchem session |
| super-H cap (+4) binds for H2 | bins 3-4 in shielded gas | the deep super-H chain (PRs #17/#18) and decision 4 |
| multiphoton heating (G0^2 dehydrogenation for N_C >~ 50, G0^2-3 bare carbon loss), tabulated for one shape only | all bins, strong fields | S4 (not run): a field proxy (absorption rate and mean absorbed energy, or two-photon group kernels) tested on other shapes |
| the in-bin shape s | all bins | S5 (not run): kernels at the 5 pseudo-bin states, blended in s; the tables so far weight molecules equally |
| the multiphoton cascade fails for C268H52 with old-star light at G0 0.01 | pahchem | report to the pahchem session |

## S6: the per-cell equilibrium solver (from 24 November; `s6_solver.py`)

After the decisions below, the solver was built on `feature/pahrt` (off feature/gsd-a3c with this study merged in):

- **`pahchem/pahrt_kernels.py`.** Every rate of a molecule's joint chain and every per-state observable is read from
  pahchem itself, on basis environments; nothing is re-implemented.
  - Photon bases: one-group power laws, 7 in-group slopes per group, pahchem's 13 G0 nodes, per unit group photon
    flux.
  - Collision bases: one unit of n_e, n(H I) or each ion, on log T from 3 K.
  - Each basis is a difference from a solvable regulariser (n_e = n(H I) = 1 at G0 1e-30).
  - `Params.cosmic_rays` (at zeta = 0) makes every charge live in every basis; at zeta = 0 it adds no process and
    changes no rate.
- **`gsd/pahrt.py`.**
  - `MoleculeSolver` folds the kernels with the cell's groups (slope from the group's mean photon energy), T, n_e,
    n(H I) and ions, then solves the full (Z, N_H) steady state.
  - `PAHrtBin` weights a bin's 5 molecules with the in-bin PL2M profile at s.
- **Run.**
  - Laptop, bins 1-2: `s6_solver.py kernels --molecules all --bins 0 1 --workers 3 --kdir results/pahrt_study/s6/kernels_cr`,
    then `v1`, `v2`, `v5`, `sweeps` with the same `--bins/--kdir`.
  - Bins 3-4 on megatron (a top-bin molecule needs 49 shapes at up to 13 min and 6.5 GB each): `--prepare` first
    builds pahchem's bases per (molecule, shape) in parallel. Output in `~/pycalima_runs/pahrt/kernels_cr`.
- **Figures** (`results/pahrt_study/s6/`):
  - `v1_kernels.png`: per process and output, at S1's phases x field shapes;
  - `v2_molecules_full.png`, `v2_molecules_full_power.png`, `v2_molecules_nested.png` (the last on the first
    kernels): S1's 214 nodes;
  - `v2_sweeps_PDR.png`: dnh and carbon loss along the PDR's G0 and n(H I) sweeps;
  - `v5_cost.txt`.

Findings, bins 1-2 (10 molecules, 25 November):

- **Collision kernels are exact:** within 1e-3 of pahchem, and H addition within 2 % from the T interpolation.
- **Fields exactly representable** (a power law at a slope node in every group): single-photon ionisation within
  0.2 %, H loss within 0.5 %.
  - Between slope nodes, the linear blend in mean energy costs H loss 9 %. A geometric blend doubles it.
- **Multiphoton proxy.** Each group's multiphoton kappa can be read at its own flux ("self"), or where the group alone
  would give the cell's absorbed "power" (or absorption rate).
  - The truth lies between the two.
  - "self" lags the stripping transitions of C43H17 and C50H18 by about one G0 step (0.25-0.5 dex): carbon there is
    off by 2-2.6 dex at one node, and dnh by up to 9.6 H.
  - "power" sits on those transitions, but overstates carbon by about x2-3 elsewhere.
  - Default "self"; the choice is the user's (decision 3 of the plan).
- **Full or nested steady state.** Nested (charge balance per N_H, then the ladder) fails where H addition is as fast
  as charging. In shielded gas, H2 is off by up to 2.9 dex and H uptake by 2.1 dex, against 1.4 and 0.6 in full.
  Default full.
- **V2 (full, "self", every charge live), median and worst over S1's nodes:**

  | output | median | worst |
  |---|---|---|
  | charge fractions, <Z> | <= 0.01 | 0.06-0.17 |
  | heating, cooling | 1-5 % | 0.16-0.4 dex and 0.04-0.2 dex |
  | H2 | 0.02-0.055 dex | 0.26-1.4 dex |
  | dnh | | <= 2.3 H, except at the transitions above |

  - Every worst heating case comes from the soft 10 kK star: a power law in each group cannot follow its Wien tail.
    This is the in-group shape, which any group scheme has.
  - Making every charge live and starting the T grid at 3 K fixed the absorption at low n_e (0.4 to 0.18 dex) and the
    cooling at 4.5 K (0.39 to 0.2 dex).
- **Cost (V5).** Per molecule, a lean fold takes 2e4-1.8e5 flops and a banded full solve 1.7e4-9e5 flops (bandwidth
  13-39), with 0.7-5.8 MB of kernels. Over the 20 molecules of 4 bins that is several Mflop per cell, about 1000x
  WDB06rt's budget. Candidate reductions:
  - store only the distinct deep levels (pahchem maps them);
  - skip the zero kernel entries;
  - a block-banded or nested-plus-Gauss-Seidel solve;
  - fewer molecules;
  - recompute only when the inputs change.

Open issues found in S6:

- **pahchem's per-N_H threshold shifts are ratios per field shape** (`_threshold_factors`: "1 where the intact state
  has none"). A band entirely below the intact threshold therefore loses the stripped states' ionisation. An absolute
  shifted rate (a pahchem accessor) would make the band sums exact.
- **Old-star fields at G0 < 1** lie below pahchem's k_H grid, where pahchem extrapolates its multiphoton tables. This
  explains the H and H2 loss differences there.
- **A hard 1-100 eV field at G0 >= 1e4** puts deposits and H2 off by up to 3 dex (EUV groups; unexplained).
- **Not in the bases yet:**
  - cosmic-ray passages (pahchem blends two spectra with zeta-dependent weights, so they need a zeta grid);
  - hot gas (tier T3);
  - C 1s fields (wait for pahchem's K-shell fix).

## Decisions (24 November 2026, the user)

1. **Tracking first chosen, then shelved.** After S7, the user chose to track the 9 class populations in bins 1-2 only
   (18 scalars; all four bins are unaffordable). Bin 1's TC error was accepted, as bin 1 PAHs should exist only briefly
   and in shielded gas.
2. **Equilibrium instead.** After S2 showed that one averaged network per bin cannot carry the hydrogenation of bins 1-2,
   the user set the hydrogenation problem aside (this note records it) and chose equilibrium for all PAH bins:
   - charge, hydrogenation, H2 formation, photoelectric heating, ion recombination, recombination cooling,
     photodissociation, fragmentation, sputtering and the rest are taken in equilibrium;
   - computed per cell from the local gas and RT radiation field as self-consistently as possible;
   - the errors above are accepted and documented.
