(physics-pahchem-status)=
# PAHchem: current state (2026-11-25)

This page is the entry point to the state of `pycalima.pahchem`. It says where
each kind of information lives, which pages and figures are dated snapshots,
and what is still open. It does not repeat the numbers: they stay in the pages
it points to.

These older pages are kept as dated records, and each says so at the top:
- {doc}`pahchem-briefing` and its figures: 2026-10-20, figures rebuilt on
  2026-10-26;
- {doc}`pahchem-audit`: the audit of 2026-10-20;
- {doc}`pahchem-pdrs4all-comparison`: 2026-10-21.

Results and plots are not in git. Paths below that start with `results/` are
under the results directory (`calima-paths` shows where it resolves). Every
folder there is labelled in `results/pahchem/README.md` and
`results/plots/pahchem/README.md`, and superseded runs are in
`results/pahchem/archive/`.

## 1. Where to look

| Question | Current source | Notes |
|---|---|---|
| What each part computes, and every gate's result | {doc}`pahchem-spec`, "Status" | milestone by milestone, M1–M7, then the cosmic rays |
| Charge, gas coupling, hot gas, the joint (Z, N$_{\rm H}$) chain | {doc}`pahchem-gas-coupling` §2.2b–2.2i and §4 | up to 2026-11-11 |
| The verdict of every benchmark | `diagnostics/pahchem/briefing_benchmarks.csv` | 172 rows with `status`, `cause` and `needs`; last edited 2026-11-05 |
| Disagreements with published models and data | {doc}`pahchem-briefing` §D; {doc}`pahchem-spec`, M5 and the extended targets | summary in §5 |
| Uncertainties still to model, decisions still open | this page, §6 | |
| Data we need and cannot get | {doc}`pahchem-data-requests` | summary in §7 |
| UV cross sections | {doc}`pahchem-uv-absorption` | band positions still an open decision (§5) |
| Carrying the physics into RAMSES | `diagnostics/pahchem/pahrt/README.md`, branch `feature/pahrt` | 2026-11-24; frozen 2026-11-25, when the GSD line restarted from main (§3) |
| Audits after 2026-10-20 | `results/pahchem/audit_2026-11-04/REPORT.md`, `results/pahchem/audit_open_2026-11-06/README.md` | the fixes are on main (PRs #11 and #12) |
| The external audit of 2026-11-25 | its claims were checked against the code; the verified fixes are PRs #18 (cache format), #24 (C 1s live states) and #25–#29, all on main since 2026-11-25 (§2) | its spin-degeneracy, attachment-temperature and solver-trap claims did not hold |

To list the open benchmarks with their cause and what they need:

```bash
python -c "import csv; r=[l for l in open('diagnostics/pahchem/briefing_benchmarks.csv') if l[0] != '#']; [print(x['id'], '|', x['cause'], '|', x['needs']) for x in csv.DictReader(r) if x['status'] == 'OPEN']"
```

Replace `'OPEN'` with `'PARTIAL'` for the partial ones.

## 2. The defaults on main, and when they changed

A result is current only if it was computed after the last default change
that can move it. The figure dates of §8 are read against this table.

| Since | Default (`Params`, {mod}`pycalima.pahchem.baseline`) | Described in |
|---|---|---|
| 2026-10-25 | `detachment = "valence"` | {doc}`pahchem-gas-coupling` §4b |
| 2026-11-01 | `uv_cross_section = "octopus_library"`; the neutrals' bands shifted +0.4 eV, labelled ad hoc | {doc}`pahchem-uv-absorption` §4e, §5 |
| 2026-11-02 | `capture_model = "lab_cold"`, `attachment_model = "size"`, both stated as extrapolations | {doc}`pahchem-gas-coupling` §2.2b |
| 2026-11-02 | `charge_ladder = "cold_limit"` | §2.2c |
| 2026-11-03 | `kshell = "auger"`, `recombination_dissociation = True`, `ionisation_heating = True` | §2.2c |
| 2026-11-04 | `photoelectron_reference = "neutral"`; proton emission, C$_2$H$_2$ loss and the destruction of bare skeletons | §2.2c, §2.2e, §2.2f |
| 2026-11-05/06 | the fixes of the 2026-11-04 audit: collisions, hot gas, the CR-induced UV, the ladder top | the audit reports of §1 |
| 2026-11-11 | `threshold_shifts = "dft_ladder"`, `threshold_calibration = "measured"` | §2.2h, §2.2i |
| 2026-11-25 | every present charge live in fields reaching the C K edge (PR #24; before, C 1s Auger steps out of the top live state were booked as Coulomb explosions); stored super-H tables rebuilt (`SUPERH_VERSION` 2, PR #18) | §2.2c, §2.2h |
| 2026-11-25 | the audit's fixes. The charge ladder repeats a computed step only within 25 % of the conductor fit, else it takes the fit (PR #25; 28 molecules, of which C$_{115}$H$_{29}$ reached +16 below 13.6 eV). Odd-N$_{
m H0}$ parents (57 π radicals) count H-loss values and super-H barriers from the parent (PR #26; their intact H loss was far too fast). Cells with old-star light above their G$_0$ (`optical_isrf` > G$_0$) get multiphoton tables on G$_0$ nodes that follow their field shape (PR #29; before, the old stars were scaled with every node, and cells below G$_0$ = 1 were extrapolated). The group tables' anion absorbs with `state_sigma` (PR #27; +0.04 % on the bin K) | §2.2c, §2.2h, §4d |

Off by default (opt-in): `cosmic_rays`, `hot_gas`, `coulomb_escape`,
`double_ionisation`, `hydrogen_chain = "joint"` (since 2026-11-25 an odd-N$_{\rm H0}$ parent's super-H thresholds take
the ladder one H later, from a ΔSCF ladder of C$_{31}$H$_{15}$; since 2026-11-26 the first super-H IP0 is fitted to C$_{27}$H$_{13}$, C$_{31}$H$_{15}$ and C$_{43}$H$_{17}$, and the EA past +2 H is known to fall 0.1–0.5 eV below the law; {doc}`pahchem-gas-coupling` §2.2h), the deep super-hydrogenated chain (`superh_chemistry = "deep"`,
PR #18; molecules with N$_{\rm C}$ ≳ 100 still reach its +24 cap in shielded and dense gas), and every band convention
other than `legacy` (`Population(band_convention=...)`). Two brackets of PR #28: anions with the cations' super-H
barriers (`superh_anion_barriers = "cation"`), and `associative_detachment = "all"`, now weighed by its
thermochemistry (EA < D).

## 3. Work that is not on main

| Where | What | State |
|---|---|---|
| `feature/gsd` | the GSD implementation, restarted from main on 2026-11-25 | its design is to be set first |
| `feature/pahrt` (holds `study/pahrt` and the GSD branch `feature/gsd-a3c`) | how much of pahchem can enter RAMSES as bin-averaged kernels with a per-cell solver | frozen at the restart (5ff3c12), kept as the record. It had decided, on 2026-11-24, equilibrium per cell for every PAH bin with nothing tracked |

## 4. What passes

The benchmark file (2026-11-05) holds 73 PASS, 57 PARTIAL, 31 OPEN and 11
WITHDRAWN rows:

| Milestone | PASS | PARTIAL | OPEN | WITHDRAWN |
|---|---|---|---|---|
| M1, statistical mechanics | 3 | 1 | 0 | 0 |
| M2, rates and laboratory gates | 9 | 0 | 2 | 0 |
| M3, photophysics and the master equation | 8 | 1 | 1 | 0 |
| M4, the (Z, N$_{\rm H}$) network | 4 | 2 | 1 | 0 |
| M5, reproductions of published models | 21 | 12 | 15 | 0 |
| M6, RAMSES tables | 1 | 0 | 1 | 0 |
| M7, emission and observations | 11 | 6 | 10 | 10 |
| CR, cosmic-ray collisions | 7 | 3 | 1 | 1 |
| GC, PAH–gas coupling | 9 | 32 | 0 | 0 |

The foundations hold: M1–M4, the emission of one molecule and of a family
(M7a, M7b), and the cosmic-ray gates G1 and G4 pass ({doc}`pahchem-spec`,
"Status"). Most of the gas-coupling rows are PARTIAL for the reasons in §5.

How current these verdicts are:
- The GC rows, the PDRs4All fits, the cosmic-ray and the hot-gas rows run
  pahchem's own defaults, so they predate the threshold calibration of
  2026-11-11. {doc}`pahchem-gas-coupling` marks its CR and hot-gas numbers
  "to be rerun".
- The reproductions take each paper's stated inputs where given
  (`src/pycalima/pahchem/method_cards/`). Where an input is unstated they fall
  back on pahchem's choices, so only a rerun shows whether a later default
  moved them.

## 5. Disagreements with published models and data

Each line is one or more rows of the benchmark file, with its section in
{doc}`pahchem-spec`. The briefing's figures D01–D08 show the first group, as
of 2026-10-26.

**Published models (M5).** Most of these gaps trace to inputs the papers do not
state; each has an author question in {doc}`pahchem-data-requests`.
- **Montillaud+13 Fig. 10:** our contours of total dehydrogenation sit
  +1.0 to +1.5 dex higher in G$_0$ with parent modes, and +0.3 to +0.8 dex
  with modes stripped per missing H. E$_{\rm th}$(C$_{54}$H$_{18}$) is 19.5 eV
  against their 17 eV.
- **Andrews+16, NGC 7023:**
  - we keep C$_{54}$H$_{18}$ ≥ 97 % normally hydrogenated, while their Fig. 7
    needs a k$_{\rm IR}$ ~10³ smaller;
  - the charge has the right shape but is 2–3× off in γ;
  - their C$_{96}$ k(E) is matched only to ~0.4 dex, unexplained.
- **Murga+20/22 (SHIVA):**
  - partial destruction is 1–4 dex high (rms 2.5 dex);
  - the two papers need opposite readings of SHIVA's p(E);
  - model b of Murga+22 is largely resolved (PAHs ≥ 6 Å within 0.05), apart
    from PDR 2 (+0.12 to +0.17);
  - Murga+20 Fig. 3 charges match in 48–57 of 66.
- **Allain+96:**
  - the dissociation of PAH 50 C is −1.4 dex from their paper I Table 4;
  - our network destruction times are 1–2 dex shorter than theirs;
  - our small PAHs are too dehydrogenated.
- **Le Page+01/03:** naphthalene$^+$ needs E$_0$ ≈ 4.2 eV, not their 4.5 eV.
  The charge fractions are off by up to 0.07.
- **Laboratory against laboratory:** the West+18 and Jochims+94 H-loss
  thresholds differ by 0.7–1.4 eV. This is carried as an uncertainty on E$_0$.
- **Gate G2, electron-impact ionisation:** our model is 0.99–1.80× our own
  BEB(HF) over 15 eV–5 keV, depending on the molecule. The fix is a decision,
  not more modelling (§6).

**Gas balances (GC).**
- **Diffuse ISM (Wolfire+08 C$^+$ neutralisation and C I/C II; Lepp+88
  metals):** these pass only if the PAH carbon sits mostly in
  N$_{\rm C}$ ≳ 50, because of the size-dependent attachment. This is read as a
  constraint on the PAH size distribution, not as a failure.
- **Tielens 2021 PDR charge:** pahchem is less ionised, because its electron
  capture is 2.2× Tielens'.
- **EUV charge against WDB06:** pahchem charges higher above ψ ~ 10$^5$,
  through the unit yield of multiply charged ions above 20 eV. No yield has
  been measured above +1 → +2.
- **Hot gas against Micelotta+10b:**
  - for N$_{\rm C}$ = 50, carbon loss by electrons is 0.05–0.4× theirs above
    3×10$^5$ K;
  - ion knock-outs are 1.7–2.8× theirs.

**Observations (M7).**
- **PDRs4All, Orion Bar:**
  - with C$_{54}$H$_{18}$, 3.3/11.2 is ×1.7–2.1 high in the H II and atomic
    templates, and 12.7/11.2 is ×1.5–1.7 low;
  - the fitted edge-structure ratios differ from Khan+25's, but the band
    assignment is model-dependent, so this is open rather than a
    contradiction;
  - the stated conditions are too wide to test the charge: every observed
    ratio lies inside the model range.
- **The 3.4 µm complex:** PAHdb's aliphatic species give 3.4/3.29 ≈ 0.01–0.02,
  against 0.35–0.51 observed. CH$_2$ and CH$_3$ side groups on the fitted
  population give it with 2.3–4.5 % aliphatic carbon.
- **The Demyk+26 laboratory laws:** applied to the population, they put the
  3.29 µm band ~11 cm$^{-1}$ too blue and too narrow.
- **ERE:** the efficiency follows the cation fraction, which is the wrong
  trend with U.
- **Not reproducible from the papers:** the binned results of
  Maragkoudakis+20 and Rigopoulou+21.

## 6. Uncertainties still to model, and open decisions

**Decisions open.**
- **Band shapes and positions** (`pahchem.band_conventions`). The `legacy`
  default puts the 11.2 µm band at 11.39 µm. The sourced conventions change
  11.0/11.2 by about 2×. Until a default is chosen, band-ratio tests of the
  charge depend on the convention. The choice is tied to a PDRs4All refit,
  which waits until the physics settles.
- **UV band positions:** the neutrals' +0.4 ± 0.1 eV shift is ad hoc and is
  not applied to the ions ({doc}`pahchem-uv-absorption` §5b).
- **Gate G2:** adopt BEB for the electron dz = +1 rate, or scale the
  Li & Draine dipole strength.
- **EUV yields of multiply charged ions:** the default is a unit yield above
  max(20 eV, IP); the `coulomb_escape` bound is opt-in.
- **Hot gas:** it stays opt-in. The per-bin sputtering tables computed through
  pahchem are deferred.

**Physics resting on stand-ins or extrapolations.**
- **Electron capture and attachment** are extrapolated beyond the laboratory
  sizes: C$_{16}$–C$_{54}$ for cations, C$_{24}$–C$_{60}$ for attachment. No
  single attachment fraction fits every observation
  (`results/pahchem/capture_study_2026-10-26/RESULTS.md`).
- **Charges above the computed ones (+2 to +3):** they take the IR and UV of the
  top computed state.
- **Bare skeletons:**
  - their IR cooling comes from a size law (rms ×2);
  - the fission barriers are fitted to five PAHs no larger than C$_{24}$.
- **Cosmic rays:**
  - the secondary electrons are not wired in (the CRISpy spectra are in
    `results/pahchem/cr_secondary_electrons/`);
  - the CR-induced UV is Heays+17's.
- **Super-hydrogenation:**
  - the +4 cap binds in shielded and dense gas;
  - the deep chain (PR #18) has no physical stop for large PAHs.
- **Attachment to radicals:** their attachment fraction has not been measured.
- **Missing carriers:**
  - aliphatic side groups for the 3.4 µm complex;
  - clusters or VSGs for the 11.25 µm component;
  - cations or N-substituted species for the blue shift of the CC bands.
- **Sensitivity ranking** ({doc}`pahchem-spec`, M5). Four inputs each move
  C$_{54}$H$_{18}$'s dehydrogenation by ~1 dex: per-state modes, the UV cross
  section, the H/H$_2$-loss barriers and the IR/RF cooling. The ranking was
  computed in early October, before the UV library and the new sticking. Its
  ordering is what sets the data requests.

**Not done yet.**
- M6: the end-to-end check of the RAMSES tables. It is now part of the
  per-cell solver of `feature/pahrt`, frozen on 2026-11-25. Its README lists the open issues:
  - the averaged hydrogenation of bins 1–2;
  - fast fluctuations;
  - charges above +2;
  - multiphoton heating and the field shape;
  - the in-bin shape.
- The WD01 cross-check of the M4 charge balance.
- The storage-ring RF decay curves.
- The audit's item m13 (the baseline's folding).
- `results/pahchem/octopus_library/` is the 57-state build of 2026-10-29. It is
  named in {doc}`pahchem-uv-absorption` §0 and read by
  `diagnostics/pahchem/octopus/shift_sensitivity.py`, but it is older than the
  shipped anchor file (92 states, rebuilt on 2026-11-06). It needs a rebuild
  with `diagnostics/pahchem/octopus/build_library.py`.

## 7. Missing data

{doc}`pahchem-data-requests` holds the list, ranked by the sensitivity ranking
(it was last edited on 2026-11-11):
- **T1, spectra-style DFT (ask now).** Per-state modes along the hydrogenation
  ladder, and IP/EA ladders. Our own B3LYP ladders cover
  N$_{\rm H0}$ − 4 … N$_{\rm H0}$ + 2 since 2026-11-10; higher-level values and
  the deeper and bare states are still wanted.
- **T2, reaction paths.** The H- and H$_2$-loss barriers.
- **T3, excited states.** UV cross sections per state, and the electronic
  states behind RF and the ERE.
- **Asked of others:**
  - the recombination of large cations (no measurement above C$_{16}$);
  - ML-MD spectra of ions;
  - CH-stretch intensities at N$_{\rm C}$ ~ 100;
  - the aliphatic carriers;
  - the Demyk+26 laws for larger PAHs;
  - the Mackie+18 VPT2 intensities.
- **Questions to authors:** Montillaud/Joblin, Candian/Tielens (Andrews+16),
  Murga/Wiebe (SHIVA) and Allain/Léger.

The `needs` column of the benchmark file names the measurement each PARTIAL or
OPEN row lacks. Examples:
- the EUV yields of PAH ions above +1;
- absolute C 1s ion yields;
- the dissociative-recombination branching of PAH cations;
- photoelectron spectra of PAH cations;
- anion photodetachment yields;
- the size distribution of diffuse-ISM PAHs.

The literature still to be supplied is in {doc}`pahchem-gas-coupling` §5.

## 8. Figures: which are current

| Figures | Made | Status |
|---|---|---|
| `results/plots/pahchem/briefing/` (sets A–E) | 2026-10-20/26 | the snapshot {doc}`pahchem-briefing` describes; it predates every default change from 2026-11-01. Its scorecard A1 is older than the benchmark file |
| `results/plots/pahchem/pahchem_*.png`, the gas-coupling checks | 2026-10-24 to 11-05 | the latest output of each check script, newer than the briefing's E copies; they predate the threshold calibration (2026-11-11) |
| `results/plots/pahchem/cr_impact_*.png` | 2026-11-01 | predate the audit fixes, which made the CR lifetimes 2–5× longer; to be rerun |
| `results/plots/pahchem/pahchem_hot_gas_*.png` | 2026-11-05 | the day of the audit fixes; to be rerun with the CR figures |
| `results/plots/pahchem/pahchem_kshell_auger.png` | 2026-11-03 | X-ray fields: predates the live-state fix (PR #24, on main 2026-11-25) |
| `results/plots/pahchem/` reproductions (`demyk2026_reproduction`, `sidegroups_34um*`, `sizelaw_33um`, `saykally_pyrene`, ...) | 2026-10-19/21 | each paper's own inputs; not rerun since |
| the PAHrt study (`results/pahrt_study/`, made on `feature/pahrt`) | 2026-11-24 | the record of the frozen `feature/pahrt`; predates the audit's fixes of 2026-11-25 |

Nothing has rebuilt the briefing set since 2026-10-26. Doing so takes
`diagnostics/pahchem/make_briefing.sh` about 3 h with 4 workers, and any
numbers that move would then need checking in the pages that quote them.
