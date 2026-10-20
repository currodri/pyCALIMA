(physics-pahchem-spec)=
# PAHchem: theory and data specification

:::{note}
This is the design specification of `pycalima.pahchem`, the PAH photochemistry
engine that will replace the research modules in `models/PAH_photophysics/`
and the heuristic photolysis table of {doc}`pah`. The engine is still being
built: this page states what it must compute, from which data, and against
which benchmark each step is validated. {doc}`pahchem-data-requests` lists the
quantum-chemistry inputs we need from collaborators.
:::

## Goal and order of work

The engine follows PAHs from molecular data to observables:

$$
\text{molecular data} \to \rho(E) \to k(E) \to P(E)\ \text{per state}
\to f(N_{\rm C}, N_{\rm H}, Z) \to \text{RAMSES tables} \to I_\lambda
$$

Tables for RAMSES-CALIMA come first, and the JWST forward model after. The
first paper is limited to:

- the coronene family (C$_{24}$H$_{12}$, C$_{54}$H$_{18}$, C$_{96}$H$_{24}$),
  with pyrene for laboratory validation;
- states $(N_{\rm C}, N_{\rm H}, Z)$ plus site classes (solo, duo, trio; even
  or odd $N_{\rm H}$; number of extra H);
- the gas phase only.

Ices, grain surfaces, explicit isomers, Bayesian inference and extragalactic
benchmarks come later.

## Why the previous attempt stalled

The audit at the start of this work found that the microcanonical rates were
**not** the problem. $k_{\rm H}(E)$, $k_{\rm H_2}(E)$ and $k_{\rm IR}(E)$ built
from PAHdb modes agree with the digitised {cite:t}`Andrews2016` curves to
within 0.25 dex for the three parents. The failures had four causes.

**1. The benchmark quantity was misread.** {cite:t}`Andrews2016` Table B (and
their Fig. 6, in s$^{-1}$) gives $k_{\rm H}(G_0 = 1) = 1.6\times10^{-6}$
s$^{-1}$ for neutral C$_{24}$H$_{12}$ and $6\times10^{3}$ s$^{-1}$ for
C$_{24}$H$_{13}$. Neutral coronene absorbs a photon only every
$7\times10^{7}$ s at $G_0 = 1$ ($r_{\rm abs} = 1.4\times10^{-8}$ s$^{-1}$), so
neither number can be a per-molecule loss rate. They are eq. (7) of that paper,
$k_i = \int k_i(T)\,G_n(T)\,dT$, which has no dissociation sink and so counts
$k_i \times t_{\rm cool}$ "events" per absorbed photon. The rate entering their
network is $r_{\rm abs}P_i$ with $P_i = k_i/k_{\rm tot}$ (their eq. 8).

`diagnostics/pahchem/check_andrews16_tableB_definition.py` recomputes the
sink-free integral from Andrews' own inputs and reproduces Table B.1:

| State | Table B.1 [s$^{-1}$] | eq. (7), no sink | With a sink, $r_{\rm abs}P$ |
|---|---|---|---|
| C$_{24}$H$_{12}$, H loss | $1.6\times10^{-6}$ | $1.8\times10^{-6}$ (+0.06 dex) | $5.4\times10^{-9}$ |
| C$_{24}$H$_{13}$, H loss | $6.2\times10^{3}$ | $1.0\times10^{4}$ (+0.22 dex) | $1.4\times10^{-8}$ |
| C$_{24}$H$_{3}$, H loss | $1.4\times10^{-2}$ | bracketed: $-1.9$ (C$_{24}$H$_{12}$ modes) to $+0.8$ dex (C$_{24}$ modes) | $\le r_{\rm abs}$ |

Consequences:
- The legacy scripts that fed Table B into the network compared against the
  wrong quantity (`compare_nh_rrkm_vs_andrews_poly.py`, `compare_rrkm_rates.py`).
- Andrews Figs. 7–9, the hydrogenation and charge fractions, remain valid
  targets. To reproduce them, implement eqs. (7)–(8) as written, not a physical
  sink.

**2. The cross section was too large.** Li & Draine $Q_{\rm abs}$ was evaluated
at the disc radius $a = 0.9\sqrt{N_{\rm C}}$ Å instead of the LD01 radius
$10\,{\rm Å}\,(N_{\rm C}/468)^{1/3}$. That makes $\sigma$ 1.7–3.3× too large.
The neutral $\sigma$ was also used for every charge.

**3. The cascade had no sink.** Method B of `pah_dissociation.py` adds
$k \gg k_{\rm IR}$ bins without a dissociation sink, so it can exceed
$r_{\rm abs}$. Ionisation never removed IP + electron energy from the internal
energy.

**4. The charge network was incomplete, and fudged.**
- Anion H addition kept $Z = -1$. Andrews (Sect. 2.4.4) uses associative
  detachment, which produces the neutral $(0, N_{\rm H}+1)$.
- Mutual neutralisation is missing.
- The electron sticking coefficient was calibrated to Andrews Fig. 8, the
  figure being reproduced.

**Lesson.** Every step gets its own gate, inputs are verified before the
outputs are compared, and a published model is reproduced by running *its*
inputs, recorded in a method card. Tuning our inputs toward its figure is not a
reproduction.

## Method cards

`src/pycalima/pahchem/method_cards/*.toml` record, value by value, the inputs
of each published model:
- the source of every number (section, table or equation);
- a confidence grade: **A** experiment, **B** benchmarked theory, **C** theory,
  **D** extrapolation, **E** assumption;
- what the paper leaves unspecified.

The engine runs in a card's configuration to reproduce that paper. A failed
reproduction then points at one card entry. Cards exist for
{cite:t}`Andrews2016` and {cite:t}`Montillaud2013`. Still to write:
{cite:t}`Allain1996a`, {cite:t}`LePage2001`, {cite:t}`Murga2020`,
{cite:t}`Murga2022`.

The two existing cards already differ in ways that matter:

| | Andrews16 | Montillaud13 |
|---|---|---|
| $k(E)$ | Arrhenius at $T_e$; H: 4.6/4.1 eV, 44.8/55.6 J K$^{-1}$ mol$^{-1}$ | $A\rho(E-E_0)/\rho(E)$; $A = 6.8\times10^{17}$ s$^{-1}$, $E_0$ = 4.8/3.2 eV |
| Super-H loss | 1.4–1.55 eV | $k(N_{\rm H0}+1) = k(N_{\rm H0}-1)$ |
| Cascade | $G_n(T)$, eqs. (7)–(8) | explicit $P(E)$, 0.25 eV bins, sink |
| Charges | all with IP < 13.6 eV, plus anions | 0, +1 |
| H addition, cation | $1.4\times10^{-10}$ cm$^3$ s$^{-1}$, every step | $(y/x)\,1.4\times10^{-10}$; $5\times10^{-11}$ for odd $N_{\rm H}$ |
| $N_{\rm H}$ range | 0 to $N_{\rm H0}+4$ | 0 to $N_{\rm H0}+1$ |

## Benchmark ladder

A layer is not built on until the layer below passes. Each gate is a pytest
test with a recorded tolerance, or a `diagnostics/pahchem/check_*.py` overlay
for figure comparisons.

| Level | Milestone | Gate |
|---|---|---|
| 0, numerical | M1 `dos.py` | Beyer–Swinehart = direct count (≤ 10 modes); Whitten–Rabinovitch within 5 % above the ZPE; $T_{\rm micro}\to T_{\rm canonical}$ at large $N$ |
| 1, molecular | M2 `rates.py` | Montillaud Fig. 3 $k(E)$ from their $(A, E_0)$; Andrews Fig. 5 $k(E)$ (already within 0.25 dex) |
| 2, laboratory | M2 | {cite:t}`West2018` iPEPICO $k(E)$ for ~10 cations from their $E_0$, $\Delta S^\ddagger$; {cite:t}`Jochims1994` $E(k = 10^4\,{\rm s^{-1}})$ vs $N_{\rm C}$ |
| 2, laboratory | M3 `photo.py`, `cooling.py`, `energy_me.py` | ionisation branching of {cite:t}`Zhen2016` and {cite:t}`Wenzel2020`; recurrent fluorescence of anthracene$^+$ {cite:p}`Martin2013` and tetracene$^+$ {cite:p}`Bernard2023` within ×2; Monte Carlo = master equation |
| 2, laboratory | M4 `network.py` | addition and abstraction cross sections of {cite:t}`Mennella2012`; $f(Z)$ vs $\gamma$ of {cite:t}`BakesTielens1994` with no fitted $s_e$ |
| 4, historical models | M5 | Montillaud Figs. 6–7; Andrews Figs. 7–9; Allain critical size; Murga Orion Bar ratios |
| 5, observations | M7 | {cite:t}`Chown2024` templates; PDRs4All charge and size profiles; NGC 7023 (`NWPDR_NGC7023/`) |

### Status

**Audit fixes (2026-10-20).** A code audit ({doc}`pahchem-audit`) found and
fixed seven bugs that moved benchmarks:
- C1, C–H mode stripping;
- C2, the SHIVA post-absorption yield;
- C3, the Murga+22 model-b rates;
- C4, the Li & Draine size clamp;
- C5, the electron-impact ionisation denominator;
- C6, the emission shares;
- C7, the lab CH-stretch scale in the families.

The before/after numbers of every affected benchmark are in its section 5.
Where a paragraph below quotes a pre-fix number, that table supersedes it.
The paragraphs most affected are updated in place: Montillaud's stripped
variant, Allain's dehydrogenated columns, Murga+22 ≥ 6 Å, the PDRs4All
families, and gate G2.

**Correction: mode frequencies (2026-10-06).** PAHdb stores its frequencies
already scaled; the SCALE column only records the factor that was applied.
- Coronene's duo C–H out-of-plane bend is 861.4 cm$^{-1}$ in PAHdb v4.00,
  against 857 in matrix isolation.
- The legacy loader `pah_mol_data.load_pah_modes`, which
  `PAHSpecies.from_states_file` used until this date, multiplies by SCALE a
  second time. So every PAHdb mode in pahchem was 2.1 % too low (C–H
  stretches 4.0 %), and every Einstein A 4–8 % too low.
- `PAHSpecies.from_states_file` now parses the files itself. The legacy
  loader is unchanged and still double-scales.
- What moved: the gates of this ladder were re-run. Those that moved:
  - the Tielens E–T table, 2–7 % → 1–12 %;
  - Andrews Fig. 5 C$_{96}$, now a strict xfail;
  - coronene sticking, −30 %.

  Every other test passed unchanged within its recorded tolerance. Numbers in
  this page measured before the correction and not re-quoted here (the
  sensitivity ranking, the M5 offsets, the M6 spot check) carry that ~2 %
  frequency error.

**M1 (statistical mechanics) passes** (`tests/test_pahchem.py`;
{mod}`pycalima.pahchem.dos`, {mod}`pycalima.pahchem.molecule`):

- **Beyer–Swinehart vs direct count:** identical for frequencies on the grid.
- **Whitten–Rabinovitch accuracy:** within 2.2 % of Beyer–Swinehart above the
  zero-point energy for 30 PAH-like modes. Its error grows with the number of
  modes, reaching −3 % (C$_{24}$H$_{12}$) and −9 % (C$_{54}$H$_{18}$) at
  $E = E_z$. Beyer–Swinehart is therefore the state count for rates;
  Whitten–Rabinovitch only serves as a check. The 5 % target set before
  measuring was not achievable for large PAHs.
- **Microcanonical vs canonical temperature:** $T_{\rm micro}/T_{\rm can}-1$
  is positive, below 3 % from 1 eV up, and falls monotonically to
  $\approx 1.1/s$ at 20 eV for C$_{24}$H$_{12}$, C$_{54}$H$_{18}$ and
  C$_{96}$H$_{24}$, as the classical $1/(s-1)$ predicts.
- **Thermal energy vs Tielens:** $U(T)/(3N-6)$ for C$_{54}$H$_{18}$ matches
  the Tielens E–T table to 1–12 % (it was 2–7 % before the frequency
  correction above), which confirms that the table is energy per
  mode.

**M2 (unimolecular rates): level 0–1 passes; the laboratory gates are
blocked on data.** {mod}`pycalima.pahchem.rates` implements:

- RRKM with an explicit transition state;
- the inverse-Laplace Arrhenius form of Montillaud;
- the effective-temperature Arrhenius form of Andrews and Tielens;
- RRK;
- a transition state built from a measured $\Delta S^\ddagger$, scaling either
  all modes or the 5 lowest, as {cite:t}`West2018` do.

What passes:

- **Thermal identities.** The canonical average of each exact form gives back
  its thermal law: $A e^{-E_0/kT}$ to 0.2 %, and transition-state theory to
  0.5 %, at 500–2000 K. The RRKM identity holds only when $Q$ is evaluated on
  the grid-rounded frequencies, because rounding a ~30 cm$^{-1}$ mode moves
  its $Q$ by ~1 %.
- **Montillaud Fig. 3** (coronene, $A = 6.8\times10^{17}$ s$^{-1}$,
  $E_0 = 4.8$ eV). The curve is matched to within a near-constant −0.11 to
  −0.26 dex. The offset is expected: they used Malloci modes, we use PAHdb
  neutral modes.
- **Andrews Fig. 5** (H and H$_2$ loss, C$_{24}$, C$_{54}$, C$_{96}$), more
  than 1.5 eV above threshold:
  - C$_{24}$ and C$_{54}$: median $|\Delta|$ = 0.05–0.17 dex, maximum 0.58.
  - C$_{96}$ sits uniformly ~0.4 dex high (strict xfail). It was ~0.2 dex
    before the frequency correction, an accident of modes that were 2.1 %
    too low.
  - PAHdb v2.00 modes (B3LYP/4-31G × 0.958, the level the card states) give
    0.03–0.06 dex more, so the mode version is not the cause.
  - On the same modes the values agree with the legacy `pah_mol_data` to
    $10^{-6}$ dex.

The same $(E_0, \Delta S)$ gives very different $k(E)$ depending on the form.
For coronene H loss ($E_0 = 4.6$ eV, $\Delta S = 44.8$ J K$^{-1}$
mol$^{-1}$) at 10 eV:

| Form | $k$ [s$^{-1}$] |
|---|---|
| Effective-temperature form (Andrews) | 93 |
| Inverse-Laplace form, $A = (ekT/h)e^{\Delta S/R}$ at 1000 K | 3 |
| RRKM, TS with that $\Delta S$ | 0.2 |

So the form is part of a model's definition. Each method card records it, and
it must never be swapped silently. Andrews' Fig. 7–9 fractions rest on the
effective-temperature form, which is much faster than RRKM near threshold.

**The M2 laboratory gates pass** (`tests/test_pahchem.py`,
{mod}`pycalima.pahchem.labdata`).

- **The laboratory data.** West+18 Table 1 and Jochims+94 Tables 1–2 are
  transcribed into `external_data/pahchem/`. Transcribing Jochims Table 1 was
  checked with their own eq. (1), $E_{\rm int} = {\rm AP} - {\rm IP} +
  \langle E_{\rm th}\rangle$, which holds to ≤ 0.05 eV.
- **The West ESI.** Its frequencies and $k(E)$ are the manual dataset
  `west2018-esi`.
- **RRKM against West's own inputs.** Fed their B3LYP frequencies (reactant
  and transition state) and their Table 1 $E_0$, `rates.rrkm` reproduces their
  $k(E)$ for 7 reactions, to a median of 0.0002–0.01 dex over 10 eV. The
  integer degeneracies come out as 1, 2 and 6. The remaining reactions'
  spreadsheet columns imply an $E_0$ 0.02–1.7 eV away from Table 1, so they are
  not used for the gate.
- **West's transition-state convention.** Their TS is the reactant minus one
  C–H stretch, with the 5 lowest modes scaled to give $\Delta S^\ddagger$ at
  1000 K. `transition_state_for_entropy(..., n_scaled=5)` reproduces their TS
  frequencies to 2 %.
- **RRK.** With $\nu = 10^{16}$ s$^{-1}$ and $E_0 = 2.8$ eV, `rates.rrk`
  reproduces every $E_{\rm int}$ and $E_{\rm crit}$ of Jochims Table 2 to the
  printed 0.01 eV.

The two experiments disagree on the H-loss threshold
(`diagnostics/pahchem/check_west18_vs_jochims94.py`). West's $k(E)$ reaches
$10^4$ s$^{-1}$ at an internal energy higher than Jochims' measured
$E_{\rm int}$:

| Molecule | Offset |
|---|---|
| Pyrene | +1.42 eV |
| Fluoranthene | +0.73 eV |
| Coronene | +1.22 eV |

At Jochims' $E_{\rm int}$, West's rates are 10$^2$–10$^3$ s$^{-1}$, not the
assumed 10$^4$. This ~1 eV is the size of the laboratory systematic on the
threshold, larger than either paper's quoted errors. The sensitivity analysis
of M5 should carry it as the uncertainty of the H-loss $E_0$.

**M3a (photophysics and the internal-energy master equation) passes.**

- {mod}`pycalima.pahchem.fields`: one field type (4$\pi$ photon flux per eV)
  and one $G_0$ (Habing, 6–13.6 eV). Draine (1978) has $G_0 = 1.67$ and
  Mathis et al. (1983) 1.13.
- {mod}`pycalima.pahchem.photo`: Li & Draine $\sigma$ at the LD01 radius, the
  Malloci coronene $\sigma$, and the Jochims and Le Page yields. Heating plus
  ionisation equals absorption.
- {mod}`pycalima.pahchem.cooling`: the exact harmonic microcanonical emission
  rate $k_i(E) = A_i\sum_{v\ge1}\rho(E-vh\nu_i)/\rho(E)$. For one oscillator
  it gives $nA$ exactly. It falls below the thermal approximation by 5 % at
  1 eV and 0.1 % at 40 eV, and the thermal approximation equals the legacy code.
  $(dE/dt)/E$ matches Andrews Fig. 5 $k_{\rm IR}$ to 0.26 dex.
- {mod}`pycalima.pahchem.energy_me`: the internal-energy master equation, with
  discrete IR down-jumps, photon up-jumps and dissociation sinks; split jumps
  conserve energy.
  - **Per-photon yields** come from an exact absorbing-cascade recursion and
    are never above 1.
  - **Multiphoton rates** come from a renewal steady state.
  - **Energy balance** is exact: absorbed = radiated + carried off, to
    $10^{-6}$.
  - **Low-$G_0$ limit:** the steady state equals the single-photon result to
    1 %.

Results under the Andrews H-loss parameters:

- **Coronene** stays single-photon to 2 % up to $G_0 = 10^5$, as Andrews state.
  Its H-loss probability per heating photon is 0.18. Their eqs. (7)–(8) give
  ≈ 0.97 with the same $k(E)$.
- **Circumcoronene** becomes multiphoton above $G_0 \sim 10$, with enhancement
  1.5, 6 and 56 at $G_0 = 10^2$, $10^3$, $10^4$, scaling as $\sim G_0^2$.
  Andrews Table B already has slope ≈ 2 at $G_0 = 1$. M5 has to show whether
  that comes from the sink-free eq. (7).

**M3b: the Monte Carlo check passes.** {mod}`pycalima.pahchem.montecarlo`
follows single molecules by Gillespie sampling, with continuous energy and no
binning or jump splitting. It agrees with `energy_me` within Poisson errors in
two regimes:

- **per-photon yields:** |z| < 1;
- **deep multiphoton** (8 eV barrier, rate $4\times10^5$ times the
  single-photon value): MC/ME = 0.95–0.99 ($z = -0.8$, $-0.1$).

**M3c: the cation-ionisation gate passes** (2026-10-06;
`tests/test_pahchem_wenzel2020.py`). {cite:t}`Wenzel2020` measured absolute
ionisation cross sections $\sigma_I$ of four cations going to the dication:
C$_{30}$H$_{14}^+$, C$_{32}$H$_{14}^+$, C$_{36}$H$_{18}^+$ and
C$_{48}$H$_{20}^+$. Their Zenodo record is bundled as
`wenzel2020_cation_ionisation.csv`.
- **Their yield law.** `photo.yield_wenzel2020` (their eqs. 4–5) reproduces
  every vertex of their vector Fig. 9 to $2\times10^{-3}$.
  - Its $\beta$ equals the measured 20 eV branching ratio to 0.03.
  - The Jochims curves they compare with are exactly
    `yield_jochims1996` shifted to IP2.
  - Their text gives the eq. 4 integral from IP2 to 13.6 eV as +19 % at
    $N_{\rm C} = 34$ and −14 % at 60 against Jochims. Their plotted curves
    integrate to +25 % and −11 %.
- **The rate.** The gate compares $\int\sigma_{\rm LD01}\,Y\,F\,dE$ with the
  measured $\int\sigma_I F\,dE$ (9.5–13.6 eV; Draine and Kurucz 15 000 and
  40 000 K):

  | Yield law | ours / measured | rms |
  |---|---|---|
  | Jochims ramp at IP2 (the Andrews+16 recipe) | 0.56–0.97 | 0.14 dex |
  | Wenzel eq. 4 | 0.66–1.30 | 0.09 dex |

  - The agreement is partly compensating. LD01 $\sigma$ is 1.1–1.5× their
    TD-DFT $\sigma_{\rm abs}$, while eq. 4 integrates 8–18 % below their
    Fig. 8 yields under 13.6 eV.
  - Their measured $\sigma_{I+F}$ is only 0.3–0.5 of the TD-DFT absorption
    below 13.6 eV. Most absorbed photons neither ionise nor fragment the
    cation on the trap time scale, which is what our cascade assumes.
- **Baseline.** The baseline now takes the Wenzel yield for $Z = +1$ where it
  applies ($N_{\rm C} \ge 32$, IP2 < 11.3 eV), and Jochims elsewhere.
  - Coronene$^+$ keeps Jochims: Andrews' IP2 = 11.50 eV is outside eq. 4, and
    the {cite:t}`Zhen2016` coronene$^+$ yields exist only as a raster figure.
  - Switching from Jochims moves $f_+$ by ≤ 0.011 and $f_{\rm dehydro}$ by
    ≤ 0.08 dex (C$_{54}$ and C$_{96}$, NGC 7023 and the Orion Bar).

**M3d: recurrent fluorescence (RF) is in** (2026-10-06;
`cooling.recurrent_fluorescence_rates`, `CascadeModel(electronic=...)`,
`tests/test_pahchem_recurrent_fluorescence.py`).
- **The rate.** $k_n(E) = 0.667\,\tilde\nu_n^2 f_{0n}\,\rho(E-E_n)/\rho(E)$
  s$^{-1}$ ($\tilde\nu$ in cm$^{-1}$), with the excited state taking the
  ground-state density. This is the form of Léger+88 and of
  {cite:t}`Subramani2025`; the degeneracies cancel through the absorption
  oscillator strength.
  - The levels are radiative down-jumps of the cascade, so the steady state,
    the per-photon yields and the Monte Carlo all include them.
- **Gates:**
  - detailed balance $\langle k_n\rangle_T = A_n (g_n/g_0) e^{-E_n/kT}$ to
    $2\times10^{-3}$ at 600–2000 K;
  - energy closure with the optical channel;
  - the Monte Carlo dissociation yield.
- **Against {cite:t}`Borja2026`** (Fig. 8: Np$^+$, An$^+$, Py$^+$; their
  Table I gaps and $f$, PAHdb cation modes; method card `borja2026.toml`):

  | States | Agreement at 4–13 eV |
  |---|---|
  | Allowed (D2, D3) | within 0.18 dex |
  | Forbidden D1 (pure Herzberg–Teller) | −0.14 to +0.49 dex |

  Their full treatment adds the excited states' own modes, Duschinsky
  rotation and the Herzberg–Teller dipoles.
- **In the baseline** (`baseline.RF_SPECTRA`, `Params.recurrent_fluorescence`,
  default on):
  - **The levels** come from the {cite:t}`Malloci2007` TD-DFT cross sections of
    the cations, via `photo.rf_levels_from_cross_section`. Each band between
    successive minima of σ from 0.5 to 3.1 eV becomes one level, at its peak,
    with its integrated $f$ ($f \ge 10^{-4}$). One level per band keeps the
    spectrum's line broadening from moving strength to low $E_n$.

    | Cation | Levels ($E_n$ eV, $f$) |
    |---|---|
    | C$_{24}$H$_{12}^+$ | (0.60, 0.0005), (1.70, 0.068), (2.10, 0.008), (2.85, 0.068) |
    | C$_{54}$H$_{18}^+$ (circumcoronene$^+$ spectrum, fetched from the Cagliari database and bundled) | (1.05, 0.076), (2.40, 1.08) |

    - These are equilibrium-geometry values. Subramani+25 and Stockett+23
      find that equilibrium $f$ underestimates measured RF, so these are a
      lower bound.
    - **The weakest, lowest state matters most.** Without the 0.60 eV band,
      coronene$^+$'s RF share of the absorbed power falls from 32 % to 10 %:
      $\rho(E - E_n)/\rho(E)$ favours low $E_n$ at low internal energy.
  - **Coverage.**
    - No spectrum exists for C$_{96}$H$_{24}^+$ or for the dications (T3
      requests), so RF is off there.
    - Neutrals have no low-lying excited states and get no RF.
    - RF is not applied to the bare skeleton's carbon loss.
  - **Effects:**

    | Quantity | C$_{24}^+$, NGC 7023 edge | C$_{24}^+$, Orion Bar | C$_{54}^+$, NGC 7023 edge | C$_{54}^+$, Orion Bar |
    |---|---|---|---|---|
    | H-loss rate (fully hydrogenated) | 2.3× slower | 1.7× slower | 68× slower | 50× slower |
    | RF share of absorbed power | 32 % | 43 % | 6 % | 15 % |
    | 3.29/6.2 | 0.063 → 0.046 | — | 0.051 → 0.046 | — |

    - C$_{54}$ needs multiphoton energies to lose H, and there the strong
      2.4 eV band makes RF fast. Most of the energy is still radiated in the
      IR at lower internal energy.
    - **Family effects:**
      - the C$_{54}$ dehydrogenated fraction at the Orion Bar halves
        (0.115 → 0.052);
      - C$_{54}$ 6.2/11.2 goes 5.86 → 5.05 at the Orion Bar and 1.15 → 1.09
        at NGC 7023;
      - C$_{24}$ stays fully dehydrogenated through its neutrals.
  - **Cost.** RF widens the cascade's band from 16 to 115 bins, so a cation
    solve takes 0.66 s instead of 0.09 s (C$_{24}$, de = 200 cm$^{-1}$). This
    is worth optimising before the RAMSES tables are regenerated.
- **Not yet done:** a storage-ring benchmark. The measured cooling rates
  (Martin+13 anthracene$^+$, Bernard+23 tetracene$^+$, Subramani+25
  C$_{12}$H$_8^+$) need a decay-curve model and the authors' inputs, which
  are behind paywalls. No RF measurement exists for coronene$^+$.

Still open:
- **photodetachment of anions.** The baseline still uses the Jochims ramp with
  IP = EA. No laboratory yield exists for PAH anions.

**M4 (the (Z, N$_{\rm H}$) network) passes.**

- {mod}`pycalima.pahchem.network`: a generic transition network over charge ×
  hydrogenation. Each process is (ΔZ, ΔN$_{\rm H}$) with a rate per source
  state; a rate into a state outside the network is an error. It gives the
  steady state, time evolution (dense matrix exponential), per-process fluxes,
  H$_2$ formation, and charge and hydrogenation fractions.
- **Which processes exist is the caller's choice.** Anion H addition, for
  example, is added explicitly as associative detachment to (0,
  N$_{\rm H}$+1), the Andrews+16 choice. The legacy network had it wrong.
- {mod}`pycalima.pahchem.chemistry`: $\sigma v e^{-E_b/kT}$ collision
  coefficients and Bakes & Tielens electron capture with the DS87
  $\tilde J$.

Gates:
- the Mennella+12 abstraction cross section reproduces Andrews eq. (10) to 1 %;
- capture is ordered attractive > neutral > repulsive, and a cation goes as
  $T^{-1/2}$ at low $T$;
- the generator conserves probability;
- time evolution reaches the steady state;
- the H-atom flux balances at steady state;
- the pure charge chain equals `four_state_charge_fractions`;
- the charge balance depends only on $G_0/n_e$ when photo-rates ∝ $G_0$ and
  capture ∝ $n_e$;
- H$_2$ formation equals H$_2$ loss plus Eley–Rideal.

Still open:
- cross-checking the charge balance against the WD01 solver in `dust_charge`
  for the same rates;
- per-state photo-rates from `energy_me` assembled for every (Z,
  N$_{\rm H}$). This is the method-card runner, and it is M5.

**M5 (reproductions), first result: Montillaud+13 Fig. 10.**
{mod}`pycalima.pahchem.montillaud13` runs the Montillaud method card through
the engine (`diagnostics/pahchem/check_montillaud13_fig10.py`). The quantity
is the fraction of C$_{54}$H$_{18}$ totally dehydrogenated on a ($G_0$,
$n_{\rm H}$) grid.

**Which modes the dehydrogenated states use.** The paper does not say so
explicitly, but implies the parent's (Malloci) modes for every state.
Photodissociation depends "on the size of the species through their
vibrational density of states, and ... on their hydrogenation state through
the binding energy $E_0$" (Sect. 3.7). The faithful reproduction therefore
uses parent modes. Stripping C–H modes per missing H is a pahchem sensitivity
variant, not their model:

| Mode set for each state | 10 % contour | 50 % contour | 90 % contour |
|---|---|---|---|
| Parent C$_{54}$H$_{18}$ modes (their model) | +1.5 dex | +1.2 dex (+1.24 at $n_{\rm H} = 1276$) | +1.0 dex |
| C–H modes stripped per missing H (variant) | +0.60 | +0.35 | +0.00 |
| the same, with proportional stripping (2026-10-20 audit fix C1) | +0.84 | +0.59 | +0.28 |

(Offsets are median log $G_0$, ours minus theirs, at the same $n_{\rm H}$.)

The stripped variant's apparent improvement was partly an artefact: the old
stripping removed the most intense C–H modes first, which in PAHdb carry
nearly all a window's intensity. The C–H cooling of the stripped states fell
far below N$_H$/N$_{H0}$ of the parent's, so they dehydrogenated too easily
({doc}`pahchem-audit`, C1).

**Where the 1.2 dex gap comes from.** The cleaner diagnostic is their
threshold energy, $E_{\rm th}$, where $k_{\rm diss} = k_{\rm IR}$ (Sect. 6.1).
With their $A$ and $E_0$:

| Molecule | Theirs | Ours (PAHdb neutral modes) |
|---|---|---|
| Coronene | 9.6 eV | 10.25 eV |
| C$_{54}$H$_{18}$ | 17 eV | 19.5 eV |

- Their C$_{54}$ $k(E)$ is therefore ~10× faster than ours at the same $E$,
  or their $k_{\rm IR}$ is slower. That is not a constant offset: for
  coronene we match their Fig. 3 to within −0.2 dex.
- PAHdb cation modes make the gap larger, not smaller: they radiate ~4×
  faster at threshold, giving $E_{\rm th} = 20.3$ eV.
- The difference must lie in inputs the paper does not publish: their Malloci
  frequencies (scaled or not) and their Monte Carlo $k_{\rm IR}(E)$. Those are
  exactly the items under "asked of others" in {doc}`pahchem-data-requests`.

Other observations:
- The contour slope is reproduced (≈ 1.4; theirs ≈ 1.5), so the multiphoton
  physics behaves.
- The mode set moves the even-N$_{\rm H}$ rates by ~100×. With bare-skeleton
  modes, $k_{\rm eff}$ at $G_0 = 100$ rises 110×.

Tests:
- a regression bound on the Fig. 10 gap passes;
- the coronene $E_{\rm th}$ test passes;
- strict xfails record the missing Fig. 10 agreement and the C$_{54}$
  $E_{\rm th}$.

What closes the gap is the mode sets and $k_{\rm IR}$ of Montillaud et al.
Physically, the per-state modes of {doc}`pahchem-data-requests` T1.1 matter
just as much, since the mode set changes the result by ~1 dex.

**M5, second result: Andrews+16.** {mod}`pycalima.pahchem.andrews16`
emulates their photo-rate recipe (eqs. 4–8) and network (Appendix C).

*Eq. 7, the Table B quantity, is reproduced.* $G_n(T)$ is taken to be the
sink-free stationary distribution of `energy_me`, which includes multiphoton
heating. With a Kurucz 15000 K field and **all absorptions heating**, Table B
is reproduced to −0.2 dex (C$_{24}$H$_{12}$), +0.1 (C$_{24}$H$_{13}$),
−0.3 to +0.4 (C$_{54}$H$_{18}$) and −0.5 (C$_{96}$H$_{24}$). This is a test.
With the ionising photons removed from the heating, the match degrades to
−0.5 to −1.5 dex. Table B itself is therefore the evidence that their
$G(T,E)$ is heated by ionising photons too. One state is off:
C$_{96}$H$_{25}$, +2.3 dex. That cannot change the network, because
$P_{\rm H} = 1$ there either way.

*Eq. 8 is the problem* (`diagnostics/pahchem/check_andrews16_ngc7023.py`).
Here $k_{\rm IR}$ in $P_i = k_i/(k_{\rm H}+k_{\rm H_2}+k_{\rm IR})$ is taken
along their thermal cooling path down to $T_i$ ("$T_{\rm peak}$ where $dT/dt
= \bar r$"), which gives $k_{\rm IR} = \bar r\,\ln(E/U(T_i))$.

- **Hydrogenation.** C$_{54}$H$_{18}$ then stays ≥ 97 % normally
  hydrogenated through NGC 7023 NW. Their Fig. 7 has N$_{\rm H}$ = 16 at
  0.19 at the edge and 0.11 at $A_V = 2$. That needs an effective
  H$_2$-loss rate of ~$10^{-9}$ s$^{-1}$ at $A_V = 2$, against
  $2\times10^{-11}$ from their own Table B there. Their network behaves as if
  $k_{\rm IR}$ in eq. 8 were ~$10^3$ smaller. That is what one gets if $G$
  carries its weight near $T_i$ = 3–20 K, where the thermal $k_{\rm IR}$
  vanishes.
- **Why it cannot be settled.** Eq. 8 is not specified tightly enough: how
  the cold part of $G$ enters decides whether C$_{54}$ comes out fully
  hydrogenated or substantially dehydrogenated.
- **Charge** (Fig. 8). The shape is right, but at every $\gamma$ ours is more
  neutral and anionic than theirs, resembling theirs at 2–3× lower $\gamma$.
  The candidates are Malloci against Li & Draine $\sigma$ for ionisation, and
  their computed Allamandola $s_e$ against our $s_e = 1$ for the C$_{54}$
  neutral.

*Interpretation.* The two published PDR models that pahchem reproduces
disagree with each other, and each disagrees with pahchem, through inputs
they do not publish:
- Montillaud: modes and $k_{\rm IR}$;
- Andrews: the weighting of $G$ in eq. 8, the $s_e$ values and $\sigma$.

This is the clearest argument for the physical cascade of `energy_me`, whose
per-photon yield (coronene: 0.18) has a sink and no free normalisation.
Their *figures* should be used as order-of-magnitude benchmarks, not as
targets to tune to. The questions to send Candian/Tielens are: how $G_n$ is
normalised; which $T$ range enters $k_{\rm IR}$ in eq. 8; the $s_e$ values;
and whether ionising photons heat in $G(T,E)$.

**M5, third result: Murga & Wiebe 2022, Fig. 1**
(`diagnostics/pahchem/check_murga_wiebe22.py`). Their four rate forms for
C$_{66}$H$_{20}$ H loss, with their Table 1 parameters, are put through
pahchem's sink-free multiphoton $P(E)$ (log $R_{\rm H}$ in yr$^{-1}$, ours
against theirs):

| Form | $\chi = 1$ | $\chi = 10^4$ |
|---|---|---|
| RRK | −7.7 vs ≈ −7.0 | +0.7 vs ≈ 0.0 |
| Arrh. 1 | −2.5 vs ≈ −4.1 | +3.2 vs ≈ +0.3 |
| Arrh. 2 | −6.1 vs ≈ −9.3 | +1.4 vs ≈ −3.6 |
| Laplace | −7.1 vs ≈ −3.1 | +0.6 vs ≈ +1.7 |

- **RRK agrees within 0.7 dex.** It is fixed completely by ($s$, $k_0$,
  $E_0$), so this validates our multiphoton $P(E)$ against Draine & Li's
  $dp/dE$.
- **The other three differ by 1.6–5 dex, in both directions, and in a
  different order from theirs.** Implementations of the *same named* method
  differ by orders of magnitude, through conventions the paper does not
  state. Two candidates: whether $\rho$ includes the ZPE, and whether the
  Arrhenius form has the $T_e$ correction.
- So their 6-dex spread is part rate-form physics and part implementation.
- With a dissociation sink, the rates fall a further 1–1.5 dex at
  $\chi = 10^4$.

**M5, sensitivity ranking** (`diagnostics/pahchem/check_sensitivity.py`,
2026-10-05).

{mod}`pycalima.pahchem.baseline` is pahchem's own model of C$_{54}$H$_{18}$,
with inputs from `method_cards/pahchem_baseline.toml`:
- RRKM from the West+18 transition states;
- the physical cascade, with sinks;
- Li & Draine $\sigma$;
- BT94 capture with the literature-default $s_e$;
- the Andrews+16 H chemistry.

Each input was moved one at a time across its literature range at two sites:
- **NGC 7023 NW edge:** $G_0 = 4100$, $n({\rm H}) = 1.35\times10^4$, Kurucz
  15000 K.
- **Orion Bar atomic PDR:** $G_0 = 2\times10^4$, $n({\rm H}) =
  5\times10^4$, Kurucz 40000 K.

The baseline fractions are:

| | Dehydrogenated | Cationic |
|---|---|---|
| NGC 7023 edge | 0.04 % | 38 % |
| Orion Bar | 8.4 % | 86 % |

The ranking, by the largest $|\Delta \log f_{\rm dehydro}|$; $\Delta f_+$ is
the change in cation fraction:

| Input (low / high) | NGC 7023: $\Delta\log f_{\rm dehydro}$ | Orion Bar: $\Delta\log f_{\rm dehydro}$ | Charge |
|---|---|---|---|
| Per-state mode set (stripped vs parent) | +0.03 | **+1.08** | $\Delta f_+ = -0.13$ (Orion) |
| UV cross section ×0.5 / ×2 | −0.80 / +0.82 | −0.99 / +0.97 | **$\Delta f_+$ ∓0.16–0.20** |
| H$_2$-loss $E_0$ ∓0.5 eV | +0.96 / −0.26 | +0.56 / −0.12 | – |
| Odd-H loss $E_0$ ∓0.5 eV | +0.66 / −0.04 | +0.96 / −0.44 | – |
| IR cooling ×3 (RF proxy) | −0.77 | −0.89 | – |
| Even-H loss $E_0$ ∓0.5 eV | +0.84 / −0.08 | +0.55 / −0.08 | – |
| H-loss $\Delta S^\ddagger$ ∓40 J K$^{-1}$ mol$^{-1}$ | −0.08 / +0.75 | −0.07 / +0.50 | – |
| Ionising photons heat | +0.47 | +0.31 | – |
| Electron capture ÷6.5 | +0.38 | +0.28 | **$\Delta f_+ = +0.48$** (NGC 7023) |
| Neutral H addition ÷3 / ×3 | +0.25 / −0.38 | +0.10 / −0.23 | – |
| Cation H addition ×0.5 / ×2 | +0.08 / −0.12 | +0.37 / −0.40 | – |
| Rate form: Andrews Arrhenius-$T_e$ | +0.20 | −0.22 | – |
| Le Page instead of Jochims yields | −0.17 | −0.04 | $\Delta f_+ \le 0.04$ |
| Jochims instead of Wenzel cation yield (2026-10-06) | −0.01 | −0.02 | $\Delta f_+ \le 0.004$ |
| Neutral $s_e = 10^{-2}$ | +0.05 | +0.01 | ≈ 0 |
| Super-H $E_0$ ±0.3 eV; Eley–Rideal $\sigma$ ×/÷3 | 0 | 0 | 0 |

What this means for the data:

- **Hydrogenation.** Four inputs each move it by ~1 dex: the per-state
  vibrational modes, the UV cross sections, the H- and H$_2$-loss barriers
  of the even/odd and dehydrogenated states, and the IR/RF cooling. A
  −0.5 eV error matters far more than a +0.5 eV one, because a lower barrier
  opens a channel while a higher one hands the flux to the other channel.
  The 0.7–1.4 eV laboratory disagreement on $E_0$ (M2) is therefore
  decisive.
- **Charge.** $\sigma$ (ionisation) and the electron-capture rates dominate.
  The ionisation yield model and the neutral sticking barely matter, because
  anions are rare at these $G_0$.
- **Inputs that do not move these observables.** H-addition barriers and
  rates move the result by at most 0.4 dex. Super-H energetics and
  Eley–Rideal leave $f_{\rm dehydro}$ and $f_+$ unchanged (they matter for
  the super-H fraction and for H$_2$ formation, which this ranking does not
  score).

Caveats: one at a time, C$_{54}$ only, two sites, and the Andrews H chemistry
throughout. Interactions between inputs, and other sizes, still need a grid.

**Electron sticking** (literature search, 2026-10-04). No prescription better
constrained than Bakes & Tielens 1994 / Weingartner & Draine 2001 has been
published since. Recent models reuse a classic law plus the
{cite:t}`Carelli2013` attachment rates.

- **Cations.** Measured recombination for C$_6$–C$_{16}$ cations
  (`external_data/pahchem/pah_electron_recombination.csv`; pyrene$^+$
  $4.1\times10^{-6}$ cm$^3$ s$^{-1}$) is 0.03–0.35 of capture with
  $s_e = 1$, rising with $N_{\rm C}$ and reaching 1 near
  $N_{\rm C} \approx 20$.
  - So $s_e^+ = 1$ is the default for C$_{24}$–C$_{96}$, as an extrapolation.
  - WD01's $s_e^+ \approx 0.1$ already underestimates pyrene$^+$ by 2.4×.
  - Sensitivity range: ~6.5× (WD01 at the low end, Montillaud's
    pyrene/Spitzer mean in between).
  - No data exist for any cation above C$_{16}$, or for any dication.
- **Neutrals.**
  - Coronene: default $k \approx 2.7\times10^{-9}$ cm$^3$ s$^{-1}$
    ({cite:t}`Carelli2013`, an upper bound; $s_e \approx 4\times10^{-3}$). The
    range spans 4 decades, from $10^{-5}$ (statistical, as in Andrews+16) to
    0.15 (WD01).
  - C$_{54}$H$_{18}$ and C$_{96}$H$_{24}$: default $s_e = 1$, on two grounds:
    C$_{60}$ attaches at ~Langevin, and circumcoronene binds a non-valence
    anion state that acts as a capture doorway (Voora & Jordan 2015; Mulvey &
    Jordan 2024). Range down to $10^{-2}$.

**Statistical neutral sticking: implemented and gated** (2026-10-06;
{mod}`pycalima.pahchem.attachment`, `tests/test_pahchem_attachment.py`).
- **The model.** A captured electron leaves the anion at
  $E = {\rm EA} + E_n + \epsilon$. The anion either autodetaches or radiates
  below EA, and $s_e$ is the probability of radiating, averaged over the
  capture flux.
  - $k_d(E)$ follows from the capture cross section by detailed balance
    (Klots), with no free parameter. The cross section is the one behind
    `chemistry.electron_capture`, so $s_e$ multiplies that capture rate
    consistently.
  - IR stabilisation is the harmonic cascade of the anion's PAHdb modes.
  - Recurrent fluorescence is not included. It could stabilise anions with
    bound excited states much faster (C$_6^-$: Chandrasekaran+17).
- **Gate (exact).** For a 30-mode molecule the thermal average of $k_d$
  equals the capture rate times the Saha-like equilibrium constant to
  $6\times10^{-4}$. Densities are averaged over a window as for the IR rates;
  without that, a sparse comb of states loses up to 70 % of the flux.
- **Results.** In the ISM the neutral sits at its radiative P(E), with
  $\langle E\rangle$ = 60 cm$^{-1}$ for coronene at $G_0 = 1$. The baseline
  takes the P(E) at $G_0 = 1$, which lowers $s_e$ by ≤ 9 % up to $10^4$.
  - C$_{24}$H$_{12}$ (EA 0.47 eV): $s_e = 4.2\times10^{-3}$,
    $2.5\times10^{-3}$ and $2.7\times10^{-4}$ at 30, 100 and 1000 K. This
    brackets the literature: {cite:t}`Carelli2013`'s ratio is
    ~$4\times10^{-3}$, and a 300 K thermal neutral gives $4.6\times10^{-6}$,
    the "statistical" $10^{-5}$ of Andrews+16.
  - C$_{54}$H$_{18}$ (1.44 eV): $s_e = 1$ to $10^{-3}$ up to 3000 K, which
    replaces the assumed value. C$_{96}$H$_{24}$ at Andrews' 0.56 eV is
    ≥ 0.97 at and below 1000 K.
  - **Impact on the baseline.** For C$_{24}$ in the CNM ($G_0 = 1$,
    $n_{\rm H} = 30$, 100 K) the anion fraction falls from 0.40 to 0.001;
    $f_+$ is unchanged. C$_{54}$ is unchanged.
- **The laboratory data cannot calibrate it directly.**
  - Every measured attachment rate is at ≥ 1 Torr (FALP, CRESU, swarms),
    where collisions stabilise the anion. They measure the formation of the
    transient anion, $\eta\,k_{\rm cap}$, not IR-stabilised attachment.
  - Anthracene gives $\eta \approx$ 2–6$\times10^{-3}$ of Langevin, and C$_{60}$
    gives $\eta \approx 1$ (Viggiano+10).
  - Detailed balance makes $k_d \propto \eta$. So where autodetachment wins
    (C$_{24}$ and smaller), the interstellar rate $\eta k_{\rm cap} k_{\rm IR}/
    (\eta k_d)$ does not depend on $\eta$. Where $s_e \to 1$ (C$_{54}$,
    C$_{96}$), the rate is $\eta k_{\rm cap}$, and $\eta = 1$ is taken from
    C$_{60}$.
- **Autodetachment lifetimes** (Khatymov+11, near-0 eV electrons) are a
  consistency check, not a gate: the decays are non-exponential, and τ
  depends on the observation window.
  - Tetracene$^-$ (371 K): 3–5 ms against 12 ms measured.
  - Pyrene$^-$ (391 K): 4 ns at $\eta = 1$ against 29 µs; $\eta \approx
    2\times10^{-4}$ from its laboratory rate would bring it to ~20 µs.

**M6, RAMSES tables** ({mod}`pycalima.pahchem.export_pah_tables`, export
stage `pahchem_dissociation_tables`, solver key
`models.photolysis_model = "pahchem"`).

Physics. The table holds the carbon-loss rate per molecule in C$_2$ units,
$k = \tfrac12 \sum_Z f(Z, N_{\rm H} = 0)\, r_{\rm C}(Z)$. The solver and
RAMSES convert it to mass with $2m_{\rm C}$.
- $f$ is the steady state of the baseline $(Z, N_{\rm H})$ network.
- $r_{\rm C}$ is the number of carbon atoms the bare skeleton loses per
  second, $\sum n_{\rm C} k_{\rm frag}$. Its fragments are competing sinks of
  the cascade (`baseline.CARBON_LOSS`), computed on the bare skeleton's own
  PAHdb modes (C$_{24}$, C$_{54}$, C$_{96}$), otherwise on the stripped
  parent.
- Carbon is lost from the bare skeleton only: compact PAHs lose their H
  first (Zhen+14). Carbon loss is not fed back into $f$.

**Carbon-loss parameters (revised 2026-10-06).** The default is
{cite:t}`Leger1989` Table 2: C, C$_2$ and C$_3$ loss with $(E_0, A)$ =
(7.37 eV, $6.2\times10^{15}$ s$^{-1}$), (8.49, $3.5\times10^{17}$) and (7.97,
$1.5\times10^{18}$), in inverse-Laplace form. These come from graphite
sublimation and are applied to bare PAHs by assumption, as Visser+07 and
Montillaud+13 (Sect. 7.2) do. Confidence **C**.
- **Why the old default was dropped.** The previous set (4.6 eV, 10 cal
  K$^{-1}$ mol$^{-1}$; Micelotta+10b, Murga & Wiebe "Arrh. 2") is not a
  bare-skeleton value.
  - It is C$_2$H$_2$ loss from small cations: the entropy is from Ling &
    Lifshitz (1998), and $E_0$ = 3.65 eV is fitted to the Jochims+94 C$_2$H$_2$
    appearance energies.
  - Micelotta+10b then raised $E_0$ to 4.6 eV so that $N_{\rm C} \approx 50$
    PAHs last $10^8$ yr. The same section says the bare-carbon channels are
    "for all practical purposes closed".
  - Murga+22 (MNRAS 517, 3732) call their rates from it an upper limit.
  - It stays as the variant `micelotta10b`, the fast bound.
- **The slow bound** is the variant `c60_cage`: C$_2$ loss of C$_{60}^+$,
  10.0 eV and $5\times10^{19}$ s$^{-1}$ (Matt+01).
  - Bare C$_{54}$ and C$_{96}$ flakes probably fold into cages first:
    Lebedeva+12 find 4.0 eV and $10^{15}$ s$^{-1}$, with no C$_2$ shed while
    folding.
  - Bare C$_{24}$ instead isomerises to rings that lose C$_3$ (Ferrari+23,
    Panchagnula+24).
  - No $E_0$ has been measured for any bare planar C$_{24}$–C$_{96}$.
- **Size of the choice.** Bare skeleton, neutral, Draine field, carbon atoms
  s$^{-1}$:

  | Set | C$_{24}$, $G_0 = 10^2$ | C$_{24}$, $10^4$ | C$_{54}$, $10^2$ | C$_{54}$, $10^4$ |
  |---|---|---|---|---|
  | `micelotta10b` (old default) | $1.2\times10^{-6}$ | $1.2\times10^{-4}$ | $2.6\times10^{-11}$ | $6.1\times10^{-8}$ |
  | `leger89` (default) | $1.0\times10^{-10}$ | $4.1\times10^{-8}$ | $5.6\times10^{-17}$ | $1.5\times10^{-12}$ |
  | `c60_cage` | $3.9\times10^{-13}$ | $3.9\times10^{-9}$ | $5.7\times10^{-21}$ | $5.2\times10^{-15}$ |

  - With Léger the carbon loss is multiphoton: it scales as $G_0^2$ for
    C$_{24}$ and $G_0^3$ for C$_{54}$, because $k = k_{\rm IR}$ is reached
    only at 15.5 and 30.7 eV.
  - C$_3$ dominates, as in graphite sublimation.

Assumptions:
- the field has the Draine 1978 shape, scaled by $G_0$ (Habing);
- $n({\rm H}) = n_{\rm H}$, i.e. the gas is atomic;
- each PAH bin maps to the nearest parent in $\log N_{\rm C}$, with
  $N_{\rm C} = 418\,(a_0/10\,\text{Å})^3$;
- bins outside $16 \le N_{\rm C} \le 150$, i.e. the cluster bins, are
  skipped.

Spot check (C$_{54}$, $G_0 = 10^4$, $n_{\rm H} = 100$): $f_{\rm bare} = 0.99$ and
$k = 10^{-11.5}$ s$^{-1}$ (it was $10^{-7.2}$ with `micelotta10b`). A companion `pahchem_state_<bin>.npz` stores, on
the same grid, $f_+$, $f_-$, $f_{\rm dehydro}$, $f_{\rm bare}$ and
$R_{\rm H_2}$ per PAH.

File format `pahchem-nd v1`, written by `write_nd_table` and read by
`solvers.table_io.read_pah_nd_table`. One value per line:

```
# ...                       any number of header lines starting with '#'
4                           ndim
n1 n2 n3 n4                 axis lengths: log10 G0, log10 nH, log10 T, log10 x_e
<n1 values> ... <n4 values> the axis values, ascending
<n1*n2*n3*n4 values>        log10 k [s^-1], C order (x_e fastest); floor -100
```

A reader evaluates the table by multilinear interpolation, clamping each
coordinate to its axis range, with $x_e = n_e/n_{\rm H}$. The RAMSES Fortran
reader is **not written yet**; it belongs in the RAMSES repository.

Open items:
- **Cluster bins.** The cluster approach in pyCALIMA needs revision before
  its bins can get pahchem tables. PAHdb has theoretical cluster data
  (`pahdb-complete-clusters_theoretical`), including for C$_{24}$H$_{12}$.
- **Precision floor of the network solve** (found 2026-10-05, fixed
  2026-10-06).
  - A general linear solve resolves populations only down to ~$10^{-16}$ of
    the peak; below that it returns noise (local balance residuals of order
    1).
  - So the bare fraction $f(Z, N_{\rm H} = 0)$ at low $G_0$ or high
    $n_{\rm H}$ came out as 0 or ~$2\times10^{-16}$ instead of e.g.
    $10^{-92}$, and the table corner was a non-monotonic plateau at
    ~$10^{-16}\,k_{\rm C_2}$.
  - `PAHNetwork.steady_state` now uses the Grassmann–Taksar–Heyman
    algorithm (`network.gth_stationary`). It is subtraction-free, so every
    population comes out to relative precision: an exact 150-decade chain is
    reproduced to $2\times10^{-15}$, and every baseline state balances to
    $3\times10^{-16}$.
  - The table is now monotonic in $G_0$ at every $n_{\rm H}$. It reaches
    the −100 floor only where the true rate is below $10^{-100}$ s$^{-1}$.
  - Checked and unaffected: the `energy_me` and SHIVA $p(E)$ solves balance
    every bin to ≤ $10^{-12}$ down to $P \sim 10^{-60}$, and equal GTH on
    the same chains. The WD01 charge chain is solved in log space.
- **Gate not yet run.** The planned gate is an end-to-end check: the
  `calima-run` rates with `photolysis_model = "pahchem"` should match the
  engine, followed by a RAMSES single-cell test.

**M7a, IR emission of one molecule: passes** (2026-10-06;
{mod}`pycalima.pahchem.emission`, `tests/test_pahchem_emission.py`, figure
`emission_cascade`).
- **The model.** The cascade of `energy_me` already holds every emission
  rate $k_i(E)$; `emission` counts the photons.
  - `cascade_photons` gives $N_i(E)$, the photons emitted per mode on the way
    down from $E$. It is exact for the discrete cascade, split jumps
    included, with optional dissociation sinks.
  - `single_photon_emission` averages $N_i$ over a heating spectrum, for a
    molecule cold before each absorption.
  - `steady_state_emission` gives $R_i = \sum_j P_j k_i(E_j)$ for the
    multiphoton steady state.
  - `spectrum` broadens the bands with the PAHdb convention: Lorentzian,
    FWHM 15 cm$^{-1}$, redshift 15 cm$^{-1}$.
- **Gates:**
  - **Energy.** $\sum_i h\nu_i N_i(E) = E$ less < 150 cm$^{-1}$ left when no
    mode can emit (66 cm$^{-1}$ for C$_{24}$H$_{12}$ at 8 eV).
  - **Steady state.** $\sum_i h\nu_i R_i$ equals the radiated power to
    $10^{-9}$, and the absorbed power to $10^{-3}$.
  - **Monte Carlo.** 3000 Gillespie cascades of C$_{24}$H$_{12}$ from 6 eV
    give every band's photon count within 1.5 % (`MonteCarlo.cascade_photons`).
  - **AmesPAHdb suite.** `Transitions.cascade(E)`, its thermal-approximation
    emission model, was run on the same PAHdb v4.00 modes; the frequencies
    agree to 0.005 cm$^{-1}$ after the correction above. Its band fractions
    (not the PAHdb modes, which are never bundled) are
    `ames_suite_cascade_band_fractions.csv`.
    - At 4–12 eV every band but 3.3 µm agrees to 1 % (far-IR 4 %).
    - At 3.3 µm the exact cascade is below the thermal one and converges to
      it as $E$ grows:

      | Species | 2 eV | 4 eV | 8 eV | 12 eV |
      |---|---|---|---|---|
      | C$_{24}$H$_{12}$ | 0.84 | 0.96 | 0.99 | 0.995 |
      | C$_{54}$H$_{18}$ | 0.75 | 0.90 | 0.97 | 0.99 |

      The thermal approximation overestimates high-frequency emission from a
      small energy content. This is the expected physical difference, not a
      tolerance.
- **Not yet in.**
  - The emission spectrum of each $(Z, N_{\rm H})$ state, weighted by the
    network populations (M7b). Its prerequisite is per-charge modes: the
    baseline uses the neutral parent's modes for every charge, but the
    cation's C–C bands are ~10× stronger, which is what makes 6.2/11.2 trace
    charge.
  - Temperature-dependent band shifts and widths (anharmonicity).
  - Recurrent fluorescence.
  - The comparison with observations (M7c).

**M7a, published emission models recovered** (2026-10-06). Two papers'
results were reproduced from their stated inputs, each with a method card;
the PAHdb v3.20 species are read by {mod}`pycalima.pahchem.ames_suite` (manual
dataset, never bundled).

- **Their model.** Both papers use the thermal-approximation cascade of the
  AmesPAHdb suites. {mod}`pycalima.pahchem.emission` has it as
  `ThermalCascade`, equal to the Python suite's `Transitions.cascade` to
  $5\times10^{-4}$ per band.
- **{cite:t}`Maragkoudakis2020`** ({mod}`pycalima.pahchem.maragkoudakis2020`,
  `tests/test_pahchem_maragkoudakis2020.py`, figure `maragkoudakis2020_fig2`):
  - **The inputs.** Their 308 species (Tables H1–H3); single photons of
    6–12 eV, or the Mathis ISRF; Lorentzian bands with FWHM 15 cm$^{-1}$ and
    no shift; band fluxes summed in the Table 1 wavebands.
  - **Fig. 2, molecule by molecule** (ISRF, every marker read from the
    vector figure): 120 neutrals, 116 cations and 35 anions, median
    $\log$(ours/paper) = +0.001 each, at most 0.002–0.004.
    - This needs the suite's own cross section (`photo.sigma_amespahdb`:
      Draine & Li 2007 UV terms, Mattioda+05 cutoff and ion near-IR bands),
      which is what they state.
    - With the Li & Draine 2001 tables the ratios are 0.03–0.05 dex higher,
      and weighting by absorbed energy instead of photon number gives
      −0.12 dex. So both were ruled out.
  - **Size calibrations** (eq. 3, Tables F9–F12). Our per-molecule
    11.2/3.3 at 6–12 eV sits −0.007 to −0.028 dex (median, $N_{\rm C} \ge 50$)
    from their fits of binned data.
  - **Not reproduced:** the charge–size grid and the binned points, whose
    bin edges and mixture weighting are not stated.
  - **Figure-side issues.** Fig. 2 has a cation at $N_{\rm C} = 28$, which is
    not in the sample, and one C$_{22}$ cation at 2.23 where the sample's
    give 0.9–1.2.
- **{cite:t}`Rigopoulou2021`** ({mod}`pycalima.pahchem.rigopoulou2021`,
  `tests/test_pahchem_rigopoulou2021.py`, figure `rigopoulou2021_fig1`):
  - **Fig. 1** (PAHdb UID 742, C$_{210}$H$_{36}$, at 5 and 12.4 eV) is the
    thermal cascade on the stored modes with FWHM **15** cm$^{-1}$. That
    fits the 12 band peaks to 1.6 % rms; the 30 cm$^{-1}$ the text states
    gives 12.8 %.
  - **The binned diagnostics (Figs. 5–11) cannot be reproduced** from what
    is stated:
    - the bands come from CASSIS Lorentzian fits;
    - the bins differ between figures;
    - Table A1 has wrong UIDs;
    - 17 spectra are unpublished DFT;
    - the figures contradict each other (method card).
  - **A window-based check of Fig. 8:**
    - small-bin 11.3/3.3 agrees (0.64/0.43/0.33/0.27 against
      0.68/0.52/0.35/0.27 at 6–12 eV);
    - small-bin 11.3/7.7 is 2–4× theirs;
    - the large bin is off by factors 0.75–3.
  - The questions for the authors are in the card and in
    {doc}`pahchem-data-requests`.
- **The physics difference.** pahchem's exact microcanonical cascade on the
  same inputs:
  - **Maragkoudakis ISRF ratios:** 11.2/3.3 (and CH$_{\rm oop}$/3.3) are
    raised by +0.016 to +0.020 dex (4–5 %) for every charge. At
    $N_{\rm C} \approx 100$ that moves the eq. 3 size by ~2–3 C.
  - **Rigopoulou Fig. 1:** 4.8 % rms instead of 1.6 %, through a lower
    3.3 µm band at 5 eV.

  Both come from the thermal approximation over-emitting at 3.3 µm when the
  energy content is small. The published size diagnostics stand to a few per
  cent.

**M7b, emission of a PAH family: passes** (2026-10-06;
{mod}`pycalima.pahchem.family_emission`, `tests/test_pahchem_family_emission.py`,
figure `family_emission`).
- **The model.** `FamilyEmission` weights the emission of every
  $(Z, N_{\rm H})$ state by its fraction $f$ from the baseline network.
  - **Modes.** Each charge emits with its own PAHdb v4.00 modes: UIDs
    29/18/19/31 (C$_{24}$H$_{12}$, −1 to +2), 46/37/38/44/45 (C$_{54}$H$_{18}$,
    −1 to +3) and 114/108/111/112/113 (C$_{96}$H$_{24}$, −1 to +3). They are in
    `model_data/PAH_states` as `<formula>_<Z>.dat`; dications are triplets.
  - Below $N_{\rm H0}$ the C–H modes are stripped per missing H.
  - **Super-H states** emit with the DFT modes of the genuine
    super-hydrogenated isomer when PAHdb has it, so the aliphatic bands come
    with the molecule (see below).
  - **Heating.** Each state has its own multiphoton steady state under the
    baseline's absorption: σ of charge $Z$, and ionising photons do not heat.
  - The emission does not feed back on $f$.
- **Gates (C$_{54}$H$_{18}$):**
  - **Energy.** The family emits what it absorbs, to $6\times10^{-16}$.
  - **Charge contrast.** Each charge alone at the NGC 7023 edge:

    | Charge | 6.2/11.2 | 3.3/11.2 |
    |---|---|---|
    | Neutral | 0.34 | 0.72 |
    | Cation | 5.2 | 0.28 |
    | Dication | 9.5 | 0.07 |
    | Anion | 20 | — |

  - **G0 trend.** At $n_{\rm H} = 10^4$, $n_e = 3$, 300 K the family
    6.2/11.2 is 0.57, 0.69 and 3.0 at $G_0 = 10^2$, $10^3$, $10^4$ (cation
    fraction 0.01, 0.15, 0.72). At $10^5$ it is 249, because the family is
    dehydrogenated and the stripped modes lose the C–H out-of-plane band.
- **Family values:**
  - **NGC 7023 edge:** C$_{54}$ 1.15 and C$_{96}$ 1.20 (6.2/11.2).
  - **Orion Bar atomic PDR:** 5.9 and 5.6. The baseline puts 85 % (C$_{54}$)
    and 92 % (C$_{96}$) of the family in cations and dications there. That is
    the charge balance to test against observations (M7c).
- **Two features of the model:**
  - **Anions.** At low $G_0$ ($\le 30$ for C$_{54}$ at $n_e = 3$),
    6.2/11.2 rises again: anions are frequent, and their PAHdb spectra carry
    strong C–C bands, as cations do.
  - **Coronene.** C$_{24}$H$_{12}$ has no solo H, so its C–H out-of-plane band
    (~11.7–11.9 µm) falls outside the 10.5–11.6 µm window. Its "6.2/11.2" is
    meaningless; Maragkoudakis+20 excluded such PAHs for this reason.
- **Super-hydrogenated emission** (2026-10-06).
  - **The isomers.** Each extra H makes an sp$^3$ CH$_2$ on the parent
    skeleton. PAHdb v4.00 has 13 such isomers in the network's range
    $N_{\rm H0}+1$ to $+4$, all B3LYP/6-31G* as the parents, with one CH$_2$
    per added H; the lowest-energy one is taken per formula and charge.
    - They are in `model_data/PAH_states/superh/` (README).
    - The other CnHm with more H in PAHdb are different skeletons, so the
      selection uses $n_{\rm CH_2}$.
    - **Coverage:**

      | Family | Isomers |
      |---|---|
      | C$_{24}$ | H$_{13}^\pm$, H$_{16}$ |
      | C$_{54}$ | H$_{19}^\pm$, H$_{20}$, H$_{21}^+$ |
      | C$_{96}$ | H$_{25}$ and H$_{28}$, −1/0/+1 |

    - A missing state takes the nearest $N_{\rm H}$ of its charge, else the
      nearest charge (`superh_choice`).
  - **What the DFT gives.**
    - The aliphatic C–H stretch peaks at 2954–2962 cm$^{-1}$ (3.38 µm),
      37 km/mol per aliphatic H against 24 per aromatic H in C$_{24}$H$_{16}$.
      So $A_{3.4}/A_{3.3} \approx 1.5$ per C–H, as B3LYP studies find.
    - The aromatic 3.3 and out-of-plane bands weaken with each converted H.
    - The CH$_2$ scissoring at 6.9 µm is weak. The 1400–1500 cm$^{-1}$
      intensity changes little on super-hydrogenation: C$_{24}$ 0 → 8 km/mol
      (H$_{16}$), C$_{54}$ 5 → 14 (H$_{20}$), C$_{96}$ 61 → 45 (H$_{28}$). Even
      a 96 % super-H family does not raise 6.9/6.2. The 3.4 µm band is the only
      extra band that matters. The interstellar 6.85/7.25 µm bands need
      methyl or HAC-like carriers, which the network does not have.
  - **Bands.** "3.3" (3.1–3.5 µm, Maragkoudakis+20) holds both stretches, so
    it is split into "3.29" (3.20–3.36, aromatic) and "3.4" (3.36–3.60,
    aliphatic); "6.9" is 6.80–7.00 µm.
    - With 15 cm$^{-1}$ Lorentzians the aromatic band's wing alone puts
      3.4/3.29 at 0.027 (C$_{54}$) to 0.029 (C$_{96}$). That is the floor of
      this band definition, not aliphatic emission.
  - **Results.**
    - Single states: C$_{54}$H$_{20}$ 0.41 and C$_{96}$H$_{28}$ 0.62.
    - Cations differ by molecule: C$_{54}$H$_{19}^+$ is at the floor (0.039),
      while C$_{96}$H$_{28}^+$ gives 0.59.
    - **The family 3.4/3.29 traces super-hydrogenation:**

      | Environment | C$_{54}$ f(super-H) | C$_{54}$ 3.4/3.29 | C$_{96}$ 3.4/3.29 |
      |---|---|---|---|
      | $G_0 = 10$, $n_{\rm H} = 10^4$ | 0.96 | 0.39 | 0.47 |
      | CNM ($G_0 = 1$, $n_{\rm H} = 30$) | 0.50 | 0.20 | 0.09 |
      | NGC 7023 edge, Orion Bar | 0.01–0.03 | ≈ floor | ≈ floor |

  - **Caveats.**
    - Harmonic DFT cannot give the 3.4–3.6 µm substructure (Fermi
      resonances; anharmonic spectra of hydrogenated PAHs, Mackie+15/18).
    - The fallback states carry their neighbour's number of CH$_2$.
- **Not yet in:** per-state (DFT) modes of dehydrogenated states (T1);
  temperature-dependent band shifts and widths; the comparison with
  observations (M7c).

**M7c, first comparison with the PDRs4All Orion Bar** (2026-10-06;
`diagnostics/pahchem/compare_pdrs4all.py`; observed ratios in
`pdrs4all_orionbar_band_ratios.csv`; input conditions in
`environments/orion_bar.toml`).
- **The local conditions are inputs, not predictions.**
  - {mod}`pycalima.pahchem.conditions` reads them from a conditions card in
    which every quantity has a source, an A–E confidence and `stated`: whether
    a paper gives it for that template.
  - **Stated:**
    - $G_0$ at the IF, 2.2–7.1×10$^4$ (Peeters+24);
    - $G_0$ at DF3, 1.3–5×10$^3$ (Zannese+26);
    - $n_{\rm H}$: atomic PDR $(5$–$10)\times10^4$ (Habart+24), DFs from
      Meudon fits to H$_2$ (Peeters+24);
    - $T$ from H$_2$ rotational temperatures (Van De Putte+24);
    - $n_e$ = 10–100 cm$^{-3}$ in the atomic PDR, from C and S
      recombination lines (Berné+22, citing Wyrowski+97, Cuadrado+19,
      Goicoechea+21).
  - **Not stated:** $G_0$ at DF1 and DF2 (the DF3 range is used); $n_e$ in
    the DFs (bounded by C$^+$ + S$^+$, $x_e$ = 1.4–1.6×10$^{-4}$); the H II
    template's background PDR (the atomic-PDR ranges are used).
  - Each template is run at the two corners of its ranges, least and most
    ionising, and the model is reported as that range.
- **Observed.** The Van De Putte+25 templates were measured here with one
  recipe: Chown+24 continuum anchors, then windows (Khan+26 for 6.2–11.2 µm).
  No per-template mid-IR fluxes are published. 3.4/3.29 is taken from
  Peeters+24 (Gaussian decomposition).
- **Model.** The baseline C$_{54}$ and C$_{96}$ families (charge ladders,
  super-H modes, RF) under a 40 000 K Kurucz field, measured with the same
  windows.
- **Results:**
  - **The input ranges dominate.** The stated conditions span
    $\gamma = G_0\sqrt{T}/n_e \approx 6\times10^3$–$2\times10^5$ in the
    atomic PDR and $5\times10^3$–$2.6\times10^5$ in the DFs. The model's
    cation fraction runs from 0.2–0.3 to 0.95–1.0 across that range, and
    6.2/11.2 from ~1 to 20–500. Every charge-sensitive observed ratio falls
    inside the model range. **So the present inputs cannot test the charge
    balance**; per-template conditions (maps, PDR-model fits) are needed.
  - **At the least-ionising corner the atomic PDR is matched** (C$_{96}$,
    $\gamma = 6\times10^3$):

    | Ratio | Model | Observed |
    |---|---|---|
    | 6.2/11.2 | 1.0 | 0.96 |
    | 8.6/11.2 | 0.75 | 0.87 |
    | 11.0/11.2 | 0.10 | 0.12 |
    | 3.3/11.2 | 0.32 | 0.29 |

    An earlier pass that assumed $n_e = 1.4\times10^{-4}\,n_{\rm H}$
    (7 cm$^{-3}$) found the atomic PDR 15–40× off. That was the assumption,
    not the model.
  - **Robust misses, independent of the inputs:**
    - 7.7 too weak relative to 6.2 (6.2/7.7 = 0.6–0.8 against 0.43–0.57;
      7.7/11.2 1.7× low at matched 6.2/11.2);
    - 12.7/11.2 2–2.5× low (no trio H in C$_{54}$/C$_{96}$);
    - C$_{54}$'s 3.3/11.2 1.6–2× high.

    These are properties of the two compact PAHdb parents, and a structure
    and size mix is the remedy.

    **Superseded on 2026-10-20** ({doc}`pahchem-audit`, C7 and section 5):
    - **The stretch scale.** These family numbers used the raw DFT CH-stretch
      intensities, while the population fits apply the lab scale. With the
      scale (and C1), C$_{54}$'s 3.3/11.2 matches at the DFs (0.35–0.52), but
      stays ×1.7–2.1 high in H II and Atomic. C$_{96}$'s becomes ×1.5–1.7 low
      at the DFs (0.18–0.20).
    - **The observable.** These rows also measure the observation over the
      Chown spline, the mismatch (C1 of the review) that withdrew the
      band-ratio fits. They should be re-scored against the spectral-fit
      decomposition before they are quoted.
  - **3.4/3.29.** C$_{96}$ stays at the floor in the atomic PDR (0.02–0.04
    against 0.09). In the DFs, super-hydrogenation gives the observed rise.
    Chown+24 and Pasquini+24 attribute the Bar's 3.4 band to methyl groups.

**PDRs4All: what the papers give for a model comparison** (reviewed
2026-10-06; arXiv sources or the HAL PDF, since aanda.org blocks automated
access).

| Paper | PAH band data | Local conditions | Theory |
|---|---|---|---|
| Khan+25, XIV, A&A 699, A133 (10–15 µm CH oop) | maps and a cut, windows and Gaussian sub-components; only mosaic means tabulated; no uncertainties | none (quotes $G_0$ and $n_{\rm H}$) | DFT band strengths per CH group (Bauschlicher+09) give the number ratios of edge H: solo/duo ≈ 14–17, solo/trio ≈ 1.5, solo/quartet 22–46 (Table 2) |
| Van De Putte+25, XVI, A&A 701, A111 | PAHFIT "PDR pack" (CDS Table B.1; v2 YAML on the drvdputt/pahfit `pdrpack` branch); profile diagnostics in figures only | none | qualitative: broadening as small hot PAHs and anharmonicity; 7.7 broad component as VSGs or clusters |
| Maragkoudakis+26, XVIII, A&A 709, A38 | windows 6.2/7.7/8.6/11.2; the CDS maps (J/A+A/709/A38) are only colour-map PDFs, with no FITS or WCS | none: γ is inferred as 2.66 $f_{\rm cat}/f_{\rm neut}$ from the fit itself | pyPAHdb on PAHdb v4.00-α (cascade at 7 eV, 15 cm$^{-1}$ Gaussians, 5–15 µm): cations 47–63 % of the emission, anions up to 17 %, $\bar N_{\rm C}$ 61–71 (Table 1) |
| Khan+26, XIX, A&A 709, A39 (6–9 µm) | maps and a cut; windows in Table A.1; nothing tabulated; no uncertainties | none | qualitative: 8.6 and 11.0 from large compact cations; 6.2/11.2 the best charge tracer |

What follows for pahchem:
- **No paper measures local conditions.** They stay those of
  `environments/orion_bar.toml`.
  - Maragkoudakis+26's γ comes from their own pyPAHdb charge fractions, so
    it cannot serve as an input to test a charge balance.
- **The continuum dominates the observed ratios.** On the same templates:
  - atomic PDR 6.2/11.2 is 0.96 (Chown+24 anchors, here), 1.24 (Khan+26
    local spline) and 1.58 (Maragkoudakis+26 pybaselines, from their
    figures);
  - DF3 8.6/11.2 runs from 0.28 to 1.35 between baselines.
  - So the comparison applies one recipe to both the model and the
    templates, and a ~30 % (×5 for 8.6) continuum systematic belongs on the
    observed side.
- **The model's edge structure is wrong.** Our parents have solo/duo = 0.5
  (C$_{54}$H$_{18}$) and 1 (C$_{96}$H$_{24}$) and no trio H, while
  Khan+25 infer solo/duo ≈ 15 and solo/trio ≈ 1.5. This, not the charge
  balance, explains the robust misses in 12.7/11.2 and 7.7/6.2. The remedy
  is a PAHdb population with that edge structure, built with the machinery
  of the Maragkoudakis+20 reproduction, rather than tuning the coronene
  family.
- **Sub-band diagnostics** (3.3A, 3.4B, 5.7A/B/C, 6.2 FWHM, 7.7
  broad/total; Van De Putte+25) need anharmonic band shifts and widths and
  a VSG or cluster continuum, which pahchem does not have yet.

**M7d: which PAH population explains the Orion Bar bands (preliminary,
2026-10-06; {mod}`pycalima.pahchem.population`,
`diagnostics/pahchem/fit_pdrs4all_population.py`; figures `population_fit`,
`population_structure`).**
- **Set-up.**
  - Library: the 133 Maragkoudakis+20 PAHdb v3.20 molecules (N_C 22–216, the
    Ricca+18 species included), 38 with their anion.
  - IP and EA from DFT total energies at one level of theory (120 molecules).
    The other 13 have pairs at different levels and take a fit,
    IP = a + b/√N_C.
  - Charge (−1/0/+1) and the first H loss come from the conditions card at
    both corners; only the abundances are fitted.
  - The fit is NNLS in ratio space on 9 band ratios to 11.2 (15 % and 0.01
    of 11.2 in quadrature), with a 200-draw bootstrap.
- **The library's own data needed fixing.**
  - 17 Table H1 cation UIDs in `maragkoudakis2020_sample_uids.csv` were
    rotated by one within a column. They are corrected, and a test now ties
    every row to its v3.20 formula. The M20 reproduction is unaffected,
    since the set of 308 is the same.
  - PAHdb mixes levels of theory between charge states (B3LYP/6-31G* neutral
    against a 4-31G cation; BP86 entries), so total-energy differences are
    taken only within one level.
- **New observed bands.** 12.0, 13.5 and 14.2 µm, measured with pahchem's
  windows after Hony+01 (`pdrs4all_orionbar_bands.csv`), each over two
  continua:
  - the Chown+24 spline, which counts the 10–15 µm plateau in;
  - a local line across each out-of-plane window, which takes it off.
  The two differ by ~1.7× in every ratio to 11.2. The model is measured the
  same way.
- **Results.**
  - **The stated conditions are not all allowed.** At the least-ionising
    corner (γ ≈ 5–6×10³) the population fits the PDR templates with
    χ² = 6–31 for 9 ratios (spline continuum). At the most-ionising corner
    (γ ≈ 1–3×10⁵) χ² rises to 32–148. There the cation fraction is ≈ 0.9 and
    6.2/11.2 and 7.7/11.2 come out too strong for any mix. This is the first
    constraint on the charge balance from the data, but it is conditional on
    the library and the yields.
  - **The size is robust.** Carbon-weighted ⟨N_C⟩ ≈ 90–110 in every template
    and continuum at the least-ionising corner. The weight sits in
    C98–C142 molecules, many of them the Ricca+18 species (UIDs 32xx–33xx),
    plus 10–25 % in C48.
  - **Misses that persist.**
    - 12.7/11.2 is under-predicted by 1.3–2×, and by ~3× with the local
      continuum.
    - 8.6/11.2 is over-predicted by 10–35 %.
    - 3.3/11.2 runs 10–25 % high (spline) and 40–80 % high (local).
  - **The edge structure disagrees with Khan+25.** The fitted populations
    have solo/duo ≈ 0.6–1.2, solo/trio ≈ 4–9, solo/quartet ≳ 5. Khan+25
    infer 14–17, 1.5 and 22–46. Part of this is the band assignment: in
    PAHdb, compact PAHs put their duo out-of-plane intensity mostly inside
    the 11.2 window. Of the 650–1000 cm⁻¹ intensity, the share in the 11.2
    window against the 12.0 window is:
    - ovalene (12 duo, 2 solo): 51 % against 25 %;
    - C54H18: 58 % against 4 %.

    So "11.2 = solo, 12.0 = duo" does not hold for this library, and edge
    ratios inferred band by band are model-dependent. The comparison of edge
    statistics is therefore open, not a contradiction.
- **Limits.**
  - The chain stops at +1 (no dications in the sample).
  - No dehydrogenated or super-H states beyond the first-step intact
    fraction: 116–121 of 133 molecules pass ≥ 0.5 at the most-ionising
    corner.
  - No RF (no spectra for the library).
  - Harmonic positions with the fixed −15 cm⁻¹ shift; no plateau or
    continuum carrier.
  - NNLS returns ≤ 9 species, so individual species are not conclusions;
    only ⟨N_C⟩ and the edge statistics are.

**Maragkoudakis+26 recovered, and why pahchem differed (2026-10-07;
{mod}`pycalima.pahchem.maragkoudakis2026`, method card `maragkoudakis2026.toml`,
`diagnostics/pahchem/check_maragkoudakis2026.py`, `explore_maragkoudakis2026.py`).**
- **Their pipeline, from public pieces.**
  - pyPAHdb's NNLS (GitHub source) with the public v4.00-alpha matrix, on the
    MIRI v2 additive templates.
  - The continuum is pybaselines morphological then imodpoly. Its unstated
    half-window and order are inferred as 0.8 µm and 3, from their own
    digitised band ratios (rms 0.046 dex).
  - Against Table 1: f_neut within 0.04, f_cat 0.03–0.07 low, N̄_C within 1–5.
    Table 1 averages per-pixel fits over zones, while the templates are pixel
    averages, so this is as close as the templates allow.
- **What their numbers are.**
  - Every matrix column integrates to 7 eV: one molecule after one absorbed
    7 eV photon. The weight fractions are therefore shares of the emitted
    energy, and N̄_C is emission-weighted. pahchem's earlier "cation fraction
    ≈ 0.3" was a share of carbon; as an emission share it is 0.4–0.5.
  - "Cation" means charge > 0. Dications and trications carry 11–18 % of the
    emission.
  - The matrix mixes levels of theory. Of 40 random columns, pahchem rebuilds
    20 exactly from PAHdb v3.20 (4-31G) spectra and 9 from final v4.00
    (6-31G*, newest species). The other 11, mostly dications, match neither.
    pahchem's 7 eV thermal cascade, Gaussian profile and XML reader reproduce
    the columns to < 1e-3 of the peak (`test_a_v320_column_is_rebuilt_by_pahchem`).
- **One choice at a time** (emission shares; templates; cube-like sampling):

  | change from their pipeline | f_cat | f_an | N̄_C | fit error |
  |---|---|---|---|---|
  | none (R0) | 0.42–0.56 | 0.06–0.23 | 56–68 | 0.23–0.37 |
  | no dications (R1) | −0.08 to −0.11 | **+0.06 to +0.14** | ≈ | ≈ |
  | final v4.00, 6-31G* (R2) | −0.16 to +0.01 | +0.07 to +0.23 | +6 to +12 | ≈ |
  | pahchem's 133-molecule v3.20 library (R3) | −0.2 to −0.3 | **+0.3 to +0.4** | ≈ | **×2** |
  | exact cascade; Kurucz-field heating (R4, R5) | ≤ 0.03 | ≤ 0.03 | ≤ 1 | ≈ |
  | Lorentzian, −15 cm⁻¹ (R6, with v4.00) | +0.1 to +0.3 | **−0.2 to −0.4** | −5 to −10 | +0.1 |
  | Chown+24 spline continuum (R7) | −0.07 to −0.21 | −0.05 to −0.18 | ±5 | ≈ |

- **Findings.**
  - **No bug in pahchem's emission physics.** The exact cascade and
    field heating change nothing at 5–15 µm.
  - **The library was pahchem's main defect.** The 133 v3.20 molecules have no
    dications, and they double the full-spectrum fit error. With PAHdb v4.00
    (``population.library_v400``: 1396 molecules, −1 to +3, 6-31G*) the
    error matches pyPAHdb's. The physics-constrained band-ratio fits
    reach χ² = 0.6–7 for 8 ratios (D3).
  - **Free fits use anions as spectral fillers.** Removing dications, or
    using a smaller library, moves weight into anion columns. The anion share
    also depends on the band profile (Lorentzian −15 cm⁻¹ removes most of it).
    At the stated conditions the charge balance allows ≤ 3 % of the emission
    from anions, so Table 1's 11–17 % anions in the DFs is not supported by
    the physics. It is a degeneracy of the free decomposition.
  - **Cation shares agree; the γ calibration does not.** Constrained, pahchem
    finds cations 0.36–0.51 of the emission, as M26 does (0.47–0.60). But
    pahchem's charge balance reaches that at γ ≈ 5×10³, the low edge of the
    stated ranges (the v4.00 scan, Δχ² < 9: γ ≲ 1–3×10⁴). M26's Eq. 1 maps it
    to γ ≈ 4×10⁴. The disagreement is in how ionising a given γ is: the
    adiabatic B3LYP IPs (coronene 6.79 eV against ≈ 7.2 measured), the yields,
    and BT94 capture are the candidates.
  - The continuum remains the largest observational systematic (R7), as in
    the band-ratio comparison.
- **Data notes.**
  - The M20 sample file had 17 rotated cation UIDs, now fixed.
  - PAHdb v4.00 holds 1522 skeletons with a neutral and a dication but no
    cation. pyPAHdb fits them freely; a charge chain cannot reach them.

**Charge inputs, lab band strengths and plateau carriers (2026-10-07;
`diagnostics/pahchem/calibrate_charge_inputs.py`, `check_lab_band_strengths.py`,
`check_plateau_carriers.py`, `fit_pdrs4all_population.py --scan --library v400`
with `--ip-shift/--yield-model/--detachment/--capture-scale`).**
- **IPs and EAs** (PAHdb v4.00 against the Cagliari database, Malloci+07 and measured):
  - IP1: v4.00 − Malloci adiabatic = −0.22 eV, with scatter 0.01 over the
    unambiguous molecules (coronene, circumcoronene, circumpyrene, hexacene).
    v4.00 − measured = −0.62 eV (−0.48 to −0.70) over 16 molecules.
  - EA: v4.00 − measured = −0.39 eV (−0.35 against Malloci).
  - IP2 (cation → dication): v4.00 − Malloci = −0.14 eV, and Malloci −
    measured ≈ −0.5 eV (wide scatter).
  - Calibrated shifts: +0.37 (EA), +0.60 (IP1), +0.64 (IP2, IP3)
    (``population.shift_thresholds``). 6-31G* without diffuse functions
    underbinds the electron, as expected.
- **Recombination.** Measured PAH⁺ + e⁻ coefficients at 300 K are 0.03–0.37
  of BT94 (median 0.15; N_C ≤ 16). The ratio rises as N_C^1.15 and would
  reach ~2 at N_C ≈ 100 if extrapolated; no measurement exists there.
- **What moves the charge balance** (v4.00 scans, spline continuum; best γ,
  and the Δχ² < 9 range):

  | input | best γ (Atomic, DF1) | allowed γ, all templates |
  |---|---|---|
  | as before | 5–6e3 (low edge) | ≲ 1–3e4 |
  | IPs calibrated (+0.6 eV; +1 eV alike) | 8–9e3 (interior) | ≲ 1.5–3e4 |
  | + Le Page+01 yields | low edge | unchanged |
  | + unit photodetachment | as calibrated | unchanged |
  | + capture × 0.3 (the small-PAH lab trend) | low edge, χ² worse | ≲ 0.5–1e4 |
  | + **capture × 3** | **1.8–2.7e4** | **≲ 3–10e4** |

  - The γ tension is a recombination question. Capture about 3× faster than
    BT94 for N_C ~ 100 puts every fit inside the stated ranges and reaches
    M26's γ ≈ 4e4.
  - The lab size trend points that way, but it stops at N_C = 16. A measured
    recombination coefficient for a large PAH cation would settle it.
  - IPs and yields are second order.
- **Cation band strengths** (DFT v4.00 against matrix isolation, experimental
  v3.10, 59 species matched by bond graph; ratios within a spectrum):
  - CC/oop: DFT/lab = 1.2× (neutrals, n 35) and 1.3× (cations, n 19).
  - The cation/neutral contrast of CC/oop agrees to −0.02 dex (19 pairs).
  - CH in-plane / CC (1100–1500 cm⁻¹) relative to oop is 0.16 dex weaker in
    DFT cations.
  - The CH stretch / oop ratio is **2× too strong in DFT** (+0.29 dex, all
    neutrals). So the model's 3.3/11.2 is ~2× high per molecule, which pushes
    fits that use 3.3 µm towards larger PAHs: a candidate for pahchem's
    ⟨N_C⟩ > M26's.
  - DFT cations are not too bright at 6–9 µm, so band strengths do not
    explain the γ tension.
- **Plateau carriers.** The metric is the share of 10.9–14.45 µm PAH power
  under a grey opening (0.3 / 0.4 / 0.6 µm). It is applied to the
  spline-subtracted templates and to model spectra alike:

  | | 0.3 µm | 0.4 µm | 0.6 µm |
  |---|---|---|---|
  | templates | 0.70–0.84 | 0.62–0.77 | 0.57–0.70 |
  | best-fit v4.00 populations | 0.89–0.91 | 0.84–0.85 | 0.68–0.74 |
  | the whole v4.00 library | 0.93 | 0.89 | 0.80 |
  | clusters mixed (4 monomers) | 0.97 | 0.93 | 0.82 |

  - Harmonic PAH mixtures with 15 cm⁻¹ Lorentzians already carry at least as
    much broad emission as observed, so no plateau carrier is missing at
    10–15 µm.
  - Clusters (neutral, Ricca+13) only smooth the spectrum further.
  - The local-continuum misfits are therefore band positions against window
    edges, not a missing continuum.
  - Anharmonic spectra (PAHdb anharmonic v1.00, C10–C18) move 19–36 % of the
    intensity into overtones and combinations. They shift fundamental
    centroids by −5 to −15 cm⁻¹ in neutrals, consistent with the −15 cm⁻¹
    convention, and leave the plateau metric unchanged (≤ 0.03) as cold
    spectra.

**Stated inputs for CH stretch and recombination (2026-10-07).**
- **CH stretch.** `Params.stretch_scale = population.LAB_STRETCH_SCALE`
  (10^−0.29) scales the Einstein A of the 2900–3200 cm⁻¹ modes by the lab
  stretch/oop ratio (above). The cascade conserves energy, so the power
  leaves through the other modes (test).
- **Recombination.** `Params.capture_model = "lab_trend"` multiplies BT94
  capture by cations and dications by
  `chemistry.lab_recombination_factor` = 10^−2.09 N_C^1.15. That is the power
  law fitted to the 10 FALP measurements: 0.79 at C54, 1.5 at C96, 2.5 at
  C150. `capture_cap` bounds it; capture by neutrals is unchanged.
- **Scans** (v4.00, IPs calibrated, spline continuum; ⟨N_C⟩ carbon-weighted):

  | inputs | χ² (5 templates) | best γ | allowed γ (Δχ² < 9) | ⟨N_C⟩ |
  |---|---|---|---|---|
  | IPs calibrated | 0.6–16 | 5–9e3 | ≲ 1–3e4 | 106–118 |
  | … without 3.3 µm | 0.3–5 | 5–8e3 | ≲ 1–3e4 | 51–110 |
  | + stretch correction | 0.3–9 | 5–8e3 | ≲ 1–3e4 | 90–112 |
  | + stretch, without 3.3 µm | 0.3–5 | 5–8e3 | ≲ 1–3e4 | 52–111 |
  | + lab recombination trend | 0.8–14 | 5e3–1.9e4 | ≲ 1.8–5.5e4 | 111–117 |
  | + lab trend and stretch | 0.4–8 | 5e3–1.3e4 | ≲ 1.5–3e4 | 87–96 |
  | + capture × 3 and stretch | 0.3–8 | 6e3–2.4e4 | ≲ 3–10e4 | 90–110 |

- **Findings.**
  - The stretch correction improves every fit and lowers ⟨N_C⟩ by up to 16
    (most in DF2, DF3), but does not remove the 3.3 µm pull. In DF3 the
    population is C52 without 3.3 and C90 with it. Either the corrected 3.3
    is still too strong per molecule at N_C ~ 50, or the DF3 3.3/11.2 needs a
    component pahchem does not have (super-H, aliphatics).
  - The lab recombination trend alone moves the Atomic and DF1 fits into the
    interior. Their allowed γ then reaches 4–5.5e4, M26's values there. DF2
    and DF3 stay at the low edge unless capture is ≈ 3× BT94.
  - Without 3.3 µm, pahchem's sizes against M26 (emission-weighted, no 3.3):
    HII 78 / 70, Atomic 111 / 71, DF1 100 / 69, DF2 86 / 63, DF3 52 / 62.
    The Atomic and DF1 excess remains, and it comes from the
    physics-constrained charges (D1–D3 above).
- A memory-cache key in ``Population`` omitted the yield, detachment and
  stretch settings, so two populations sharing molecules could reuse each
  other's bases. It is fixed. All the scans above ran one setting per
  process and were not affected.

**Review findings and withdrawn conclusions (2026-10-07).** Two independent
reviews of pahchem were run, one on software and numerics and one on the
physics (reports and scripts in the session scratchpad, review_A and
review_B).
- **C1, measurement mismatch** (confirmed).
  - The band-ratio population fits compared observed bands over the Chown+24
    spline (every band below 10 µm, and all bands in "spline" mode) with raw
    model windows.
  - Put through the same spline, the model keeps 24 % of its 6.2 µm flux and
    37–51 % at 7.7–8.6 µm. 1274 of 1396 molecules get a negative band, the
    fits become exact in every cell, and γ is unconstrained.
  - **Withdrawn:** every γ result of the band-ratio fits:
    - "best γ at the low edge";
    - "the stated conditions are not all allowed";
    - "the γ tension is a recombination question", and capture ×3;
    - the lab-trend and IP-calibration γ tables;
    - D1–D3 of the M26 comparison;
    - the band-ratio ⟨N_C⟩.
- **Also withdrawn:** "harmonic mixtures already have the plateau". The
  opening metric scores a single Lorentzian line at 0.48, so it counts line
  wings as plateau.
- **Open items from the reviews**, to be fixed in order:
  - the single-step intact filter (wrong near threshold: C54H18 0.87 against
    0.00 from the network);
  - recombination heating, missing (neutral band power +40–48 %);
  - the G0 normalisation (the Kurucz shape scaled to 6–13.6 eV is 1.24× the
    Peeters+24 F_ν(1027/1040 Å) definition);
  - capture above the classical limit (the lab trend extrapolated);
  - an uncalibrated error model;
  - ⟨N_C⟩ degeneracy;
  - bond-graph hash collisions (114 non-isomorphic pairs);
  - M20 partner isomers;
  - result caches keyed without their inputs;
  - dications missing from reported cation shares;
  - recurrent fluorescence and photons below 5 eV, absent from the population.
- **Step 1, the observable** ({mod}`pycalima.pahchem.spectral_fit`,
  `diagnostics/pahchem/fit_pdrs4all_spectra.py`).
  - The templates are now fitted as spectra: I_ν = Σ w_s P_s(conditions) +
    a cubic B-spline continuum in λ (knots 1.0 / 1.5 / 2.5 µm, per segment).
  - Fitted over 3.1–3.65 and 5–15 µm on a 2.5 cm⁻¹ grid, Chown+24 artefacts
    masked, all coefficients ≥ 0. Data and model are treated alike, and a
    synthetic population plus continuum is recovered to 3 % in band ratios
    (tests).
  - Results with the current inputs (v4.00, IPs calibrated, lab stretch,
    BT94, the old intact filter):
    - The best cells lie at γ ≈ 5e4–2e5, at or near the upper edge in the
      Atomic PDR and DF1.
    - Cations (Z ≥ 1) carry 0.89–0.97 of the PAH emission and anions 0.
    - Carbon-weighted ⟨N_C⟩ is 75–86, the same for all knot spacings.
  - The fits are poor (rms 15–32 %) with structured residuals:
    - the model 3.3 µm band peaks ~25 cm⁻¹ blue of 3.29 µm and is too weak;
    - 11.2 and 12.7 µm are too weak and too broad, and the continuum takes a
      third to two thirds of the 10–15 µm power;
    - 6–9 µm fits well, and it drives the cation share.
  - With a consistent observable, the conditions inference is controlled by
    the band-profile model: one fixed −15 cm⁻¹ shift and one 15 cm⁻¹
    Lorentzian for every band. Δχ² ranges are meaningless with structured
    residuals like these.

**Step 2 corrections and the corrected spectral fits (2026-10-07;
`fit_pdrs4all_spectra.py --fit-shape`, figure `population_spectral_fit_shape.png`).**
- **Corrections in place:**
  - recombination heating;
  - G0 definitions per card entry (Peeters+24 F_ν(1027/1040 Å), 1.236× the
    Habing integral for 40 kK);
  - capture capped at the classical limit;
  - the dehydrogenation chain per cell, replacing the one-step filter;
  - multiphoton loss cascades to 45 eV, where the 16 eV grid had cut the
    two-photon tail off;
  - the stable-partition bond-graph hash (v4.00: 1431 molecules) and
    graph-matched M20 partners;
  - gas-line masks;
  - result caches keyed on the code version.
- **Shape fitting.** Each start (default, broad CH) goes through the whole
  alternation, and its first shape fit is made at the best cell of a scan
  with the start shape. Fitted at a fixed high-γ cell, every start fell into
  the narrow-band branch (rms 0.10–0.23).
- **Run time** (2026-10-20). The full `--fit-shape` refit has gone from 37 min
  to ~1 min on 14 cores, with the same result (shapes to 2e-10, weights to
  1e-11):
  - the profiles are summed by a numba kernel over modes packed once per
    round (`spectral_fit.ModeTable`); Gaussians are cut at ±8σ;
  - the NNLS is Lawson–Hanson on Gram columns built only as they enter
    (`spectral_fit._nnls`, 12× scipy, falling back to it);
  - the IR down-jump rates (`cooling`), the electron-sticking recursion
    (`attachment`) and the H-loss chain have numba kernels too;
  - the basis cache key hashes arrays by their bytes, not by `repr` (old
    files are linked to the new keys on first use);
  - the 20 runs go one template per process (`--workers`, or any
    `concurrent.futures` executor, e.g. dask's `Client.get_executor()`).

  Without numba (the `accel` extra) each kernel falls back to NumPy/scipy
  with the same numbers.
- **Result** (v4.00, IPs calibrated, lab CH-stretch scale, BT94; Gaussian
  profile best):

  | | HII | Atomic | DF1 | DF2 | DF3 |
  |---|---|---|---|---|---|
  | rms | 0.048 | 0.102 | 0.091 | 0.067 | 0.055 |
  | best γ | 9.8e3 | 1.0e4 | 1.0e4 | 9.9e3 | 1.1e4 |
  | cation emission share (Z ≥ 2) | 0.55–0.61 (0.05); 0.48–0.51 (0.04) of the emitted power, audit C6 | | | | |
  | ⟨N_C⟩, carbon-weighted | 85 | 87 | 85 | 81 | 83 |
  | CH shift / FWHM | −28 / 40 | −29 / 35 | −29 / 36 | −28 / 35 | −28 / 34 |
  | CC shift | +15 | +17 | +18 | +17 | +19 |
  | oop shift / rest FWHM | −1 / 7 | 0 / 6 | 0 / 7 | −3 / 9 | +1 / 12 |

  Lorentzian fits agree (rms 0.049–0.109).
- **Reading.**
  - The band shape is the same in all five templates.
  - The best γ is ≈ 1e4 everywhere, inside every template's stated range.
  - Cations (mostly +1) carry ≈ 0.48–0.51 of the PAH emission, anions ≤ 0.01. That is
    a share of the emitted power, recombination heating included. The ≈ 0.55–0.6 first reported
    counted photon heating only ({doc}`pahchem-audit`, C6); the refit with the fixed code
    (2026-10-20) leaves rms, γ, ⟨N_C⟩ and the band shape unchanged.
  - The continuum takes 0.5–0.6 of the 5–15 µm power.
  - The γ and ⟨N_C⟩ uncertainties are not yet calibrated (step 3). The
    Δχ² < 9 ranges use the fit's own residual scale and are narrow because
    the residuals are structured.
- **Remaining residuals.**
  - The 3.40 µm aliphatic band, everywhere: there are no super-H or
    aliphatic species in the library (see "The 3.4 µm aliphatic complex").
  - A 3.25 µm shoulder.
  - The Atomic 8.6 µm band.
  - Atomic is the worst fit (rms 0.10).

**Temperature-dependent band profiles: Mai et al. 2025 assessed (2026-10-07;
analysis scripts in the session scratchpad, mai2025/).**
- **What it is.** X. Mai, Z. Wang et al., MNRAS 541, 3073 (2025;
  arXiv:2503.05120).
  - Machine-learning MD (NeuralIL force field plus a dipole network, trained
    on B3LYP/4-31G, PAHdb v3.20 geometries).
  - Absorption spectra from dipole autocorrelations at 50, 300 and 600 K for
    1704 **neutral** PAHs.
  - Frequencies scaled by 0.9578; quantum correction D = 1.
  - Data: Zenodo 10.5281/zenodo.15771437, peak-normalised spectra,
    CC BY-NC-SA 4.0.
  - The paper gives no per-band numbers. The shifts and widths below are
    fitted to their data (600 K and 300 K against 50 K):
    - CH stretch: −0.07 cm⁻¹/K, FWHM +0.06 /K;
    - 6.2 µm: ≈ −0.06 /K;
    - 7.7 µm: −0.03 to −0.05 /K;
    - 11.2 µm (compact, N_C ≥ 50): −0.01 /K, FWHM +0.01 /K.
  - Pyrene 11.9 µm agrees with the de Bentzmann+26 lab slopes to ~25 %.
- **What it means for the fitted band shape.**
  - The CH-stretch shift (−28 cm⁻¹, FWHM 37–47) and the oop shift (−3 to +1,
    FWHM 7–11) are consistent with these classical T ≈ 300–600 K laws.
  - The CC **blue** shift (+15 to +18) cannot be thermal: MLMD shifts the CC
    bands red, and v4.00 neutrals are within +2.5 cm⁻¹ of matrix. It points
    to the carriers (cations, N substitution), as in the long-standing 6.2 µm
    position problem.
- **Why it cannot replace the fitted shape.**
  - Neutrals only.
  - The absolute CH-stretch positions sit +26 cm⁻¹ off matrix data (4-31G).
  - Intensities are biased by D = 1.
  - Classical equipartition: 600 K classical holds the thermal energy of a
    ~1150 K quantum PAH, so the temperature mapping is uncertain by ~2×.
  - Rotation is not removed, and the 600 K spectra are noisy.
- **Use.** As a prior on the temperature-dependent part only: per-band
  δ(T), w(T) added to the PAHdb v4.00 positions in the cascade, bracketed by
  T-matching and E-matching. To be cross-checked against CosmicPAH-IRDB lab
  slopes. Questions for the authors are in the data requests.

**The 3.4 µm aliphatic complex (2026-10-07;
`diagnostics/pahchem/fit_34um_profiles.py`, `fit_pdrs4all_spectra.py
--aliphatic free`; figures `aliphatic_34um_survey`,
`cascade_profile_prototype`, `refit_3um_twostep_<template>`).** Can PAHdb
carriers and pahchem's emission physics give the complex?
- **What is observed.**
  - Components (Peeters+24 Table G.1, Gaussian): 3.395, 3.403, 3.424,
    3.465, 3.516 and 3.561 µm on a plateau ({cite:t}`Peeters2024`).
  - The 3.40 band is asymmetric, with a steep blue edge and a red wing.
  - Over a linear continuum through 3.10–3.17 and 3.60–3.66 µm,
    (3.36–3.60)/(3.20–3.36) is 0.37 (HII), 0.35 (Atomic), 0.38 (DF1), 0.44
    (DF2) and 0.51 (DF3).
- **Library changes.**
  - The aromatic v4.00 library held seven C$_{96}$H$_{25}$ super-H isomers
    (UIDs 684–690): PAHdb's edge tags miss their extra H. The default library
    now keeps only molecules whose H all sit on aromatic edges (1424).
  - `include_superh=True` adds the species with off-edge H: 28 with a
    neutral and a cation at N_C ≥ 21, the seven C$_{96}$H$_{25}$ among them.
    Most carry 1–3 such H; C$_{96}$H$_{28}$
    (8 CH$_2$ H) is the most aliphatic. `edges` holds "aliphatic" and "ch3".
  - The lab CH-stretch scale now covers 2600–3200 cm⁻¹, so the aliphatic
    stretches take it too. This is an assumption: the scale was measured on
    aromatic stretches only. The CH mode class of `BandShape` is
    2600–3300 cm⁻¹.
- **Where PAHdb puts the stretches.**
  - Below the aromatic stretch: isolated CH$_2$ ~170 cm⁻¹ (3.48 µm after the
    fitted −28 cm⁻¹); adjacent CH$_2$ pairs (asymmetric, ~2962) ~100 cm⁻¹
    (3.41 µm); methyl peaks at ~3.46 µm.
  - The anharmonic indene of PAHdb anharmonic v1.00 agrees on the CH$_2$
    offset (171 against 170 cm⁻¹ harmonic) and is 1.5× stronger.
  - Per CH bond, neutral CH$_2$ and CH$_3$ carry 0.6–1.3× the aromatic
    stretch. In cations the CH$_2$ stretch nearly vanishes.
- **The full fit with the aliphatic species free (`--aliphatic free`, no
  hydrogenation constraint)** gives them no weight: off-edge H ≤ 0.3 %, the
  3.36–3.62 µm residual unchanged (0.20–0.33 of the mean). Two traps make the
  segment unfit for this question, and the intermediate aliphatic fractions
  (3–23 % of H) found on the way are withdrawn:
  - the segment's continuum (one cubic of four non-negative B-splines) sags
    under the 3.3–3.5 µm bands and takes part of the complex; it also lets the
    full fit's 3.29 band run ~20 % above what a straight continuum through the
    band-free edges allows;
  - fitted alone on 3.1–3.65 µm, NNLS lets the neutral aliphatic molecules
    carry the 3.29 band, because the cation-dominated population has weak CH
    stretches.
- **What the physics allows (baseline network, coronene family, at the
  corners of each template's stated conditions).** The extra-H share of
  C$_{54}$/C$_{96}$:

  | | HII | Atomic | DF1 | DF2 | DF3 |
  |---|---|---|---|---|---|
  | least ionising | 0.014–0.018 | 0.016–0.020 | 0.11–0.15 | 0.11–0.15 | 0.09–0.12 |
  | most ionising | ≤ 1e-4 | ≤ 1e-4 | 4–7e-4 | 4–7e-4 | ≤ 1e-4 |

  The super-H loss rate (~3e-5 to 3e-3 s⁻¹) sits close to H addition, so the
  share spans up to five dex inside one template: the stated conditions
  neither require nor exclude super-H. C$_{24}$ keeps none.
- **Band profiles from the cascade.** `emission.emission_by_energy` splits
  each mode's photons by the cascade bin they leave from (exact against
  `cascade_photons`; energy closes to 0.1 %). Each photon then takes
  ν$_i$ + χ$_1'$T(E) and FWHM χ$_0''$ + χ$_1''$T(E).
  - This gives the observed asymmetry: a steep blue edge at the cold
    position and a red wing from hot emission.
  - With the Mai+25 classical laws, the classical T(E) = E/(s k) puts the
    3.29 band near the fitted −28 cm⁻¹; the quantum (canonical) T doubles the
    shift.
  - Lab laws of {cite:t}`Demyk2026` (Table 4, gas phase, rotation removed):
    aromatic CH −0.0216 cm⁻¹/K (pyrene), asymmetric CH$_2$ −0.0115
    (6H-pyrene, 523–673 K), so aliphatic bands shift half as fast. PAHdb
    pyrene and 6H-pyrene sit ~6 cm⁻¹ above their 0 K intercepts for both
    classes; that offset is applied as a stated input.
- **Two-step refit of 3.1–3.65 µm.** (1) The aromatic population of the full
  fit (weights and charges of the best Gaussian cell) as one column with a
  free scale, plus a linear continuum, on 3.1–3.36 and 3.60–3.65 µm; (2) the
  aliphatic species by NNLS on the whole residual, so they pay for their own
  3.29 emission. 3.36–3.60 µm residual / mean, aromatic only → with the
  aliphatic species (off-edge H):

  | Law | HII | Atomic | DF1 | DF2 | DF3 |
  |---|---|---|---|---|---|
  | symmetric −28/37 | 0.37 → 0.30 (4.2 %) | 0.56 → 0.46 (3.5 %) | 0.63 → 0.50 (4.0 %) | 0.68 → 0.56 (4.7 %) | 0.67 → 0.54 (5.3 %) |
  | Mai+25, classical T | 0.34 → 0.27 (1.3 %) | 0.49 → 0.37 (1.2 %) | 0.58 → 0.46 (1.2 %) | 0.67 → 0.57 (1.2 %) | 0.65 → 0.54 (1.4 %) |
  | Demyk+26 lab laws | 0.28 → 0.24 (1.1 %) | 0.41 → 0.34 (1.0 %) | 0.48 → 0.41 (1.0 %) | 0.56 → 0.50 (1.0 %) | 0.56 → 0.50 (1.1 %) |
  | Demyk, aromatic law for all | 0.28 → 0.24 (1.1 %) | 0.41 → 0.34 (1.0 %) | 0.48 → 0.42 (1.0 %) | 0.56 → 0.49 (2.1 %) | 0.56 → 0.48 (2.6 %) |

  The aliphatic species carry 9–18 % of the 3.2–3.36 µm window.
- **The library's ceiling.** Charge-weighted at the DF3 cell, the aliphatic
  molecules give (3.36–3.60)/(3.20–3.36) of 0.008 (median, lab laws; 0.02
  symmetric). Only C$_{96}$H$_{28}$ (0.64), C$_{22}$H$_{16}$ (0.39) and
  C$_{25}$H$_{14}$ (0.32) reach the observed 0.35–0.51, and the aromatic
  population, fixed by the 5–15 µm bands, gives almost nothing there. The
  complex would need most of the emitting PAHs to be strongly aliphatic.
- **Reading.**
  - **The carriers are missing.** PAHdb v4.00 has few large PAHs with
    methyl or longer side groups, or with several CH$_2$, that also have a
    cation, and its per-bond aliphatic intensities are about half of
    {cite:t}`Yang2020`'s. This, not the band profile, bounds the fit.
  - **The cascade gives the asymmetry,** and the aromatic population alone
    fills more of 3.36–3.60 µm with it (lab laws: DF3 residual 0.56 against
    0.67 symmetric).
  - **The pyrene laws do not transfer to the population.** They make the
    3.29 band ~11 cm⁻¹ too blue and too narrow (the observed FWHM is
    37–42 cm⁻¹, Peeters+24), so the segment rms is worse than symmetric.
  - **The 3.42–3.56 µm structure** is Fermi-resonance bands, which harmonic
    spectra miss ({cite:t}`Maltseva2018`).
  - **Unresolved:** {cite:t}`Mackie2018` find anharmonic cascade peaks
    unshifted from their 0 K positions, with red wings only; the
    temperature-law approach of Pech+02 and {cite:t}`Demyk2026` shifts them.
  - **Context.** {cite:t}`Thatte2026` split the band into 3.395 µm
    (robust, tracks 3.3) and 3.405 µm (fragile, tracks H$_2$; shielded gas),
    and find with harmonic B3LYP coronene + methyl/ethyl/vinyl cascades that
    methyl and ethyl groups move the peak to 3.405. Side-group bond energies
    4.14 (CH$_3$), 3.94 (ethyl), 4.51 eV (vinyl). Chown+24 and Pasquini+24
    attribute the band to methyl groups; neither models a profile.
- **Next.** Emit side groups from the population itself: each methyl or
  CH$_2$ group on a molecule radiates with that molecule's cascade
  temperatures, at the lab 0 K positions ({cite:t}`Maltseva2018`,
  {cite:t}`Demyk2026`) with the aliphatic laws and a per-bond A(3.4)/A(3.3)
  from DFT. One parameter, the aliphatic H fraction, then has a physical
  meaning, and 3.395/3.405 can be separated. The carrier spectra to ask for
  are in {doc}`pahchem-data-requests`.

**Demyk+26 emission model reproduced (2026-10-20; `pahchem.cosmicpah`,
`pahchem.demyk2026`, `diagnostics/pahchem/check_demyk2026.py`, figure
`demyk2026_reproduction`).** Before the laws of {cite:t}`Demyk2026` go into the
population, their single-photon model was rerun on pahchem's cascade and
compared with their figures (digitised from the arXiv source).
- **Inputs.** Band laws from CosmicPAH-IRDB by the paper's dataset IDs
  (fetch dataset `cosmicpah-irdb`; Table 4 checked), Table 5 intensities,
  and B3LYP/4-31G modes computed here: their mode lists are not published
  and PAHdb has no methylpyrene. Scaled against PAHdb pyrene, our
  6H-pyrene matches PAHdb UID 336 to 8 cm⁻¹ rms.
- **Reproduced.**
  - 3 µm profiles of 6H- and Me-pyrene at 4 and 6 eV (Fig. 5): main CH
    band positions within 0.5–1.5 cm⁻¹, FWHMs within 1 cm⁻¹, sub-bands
    within ~5 %.
  - Emission temperatures of the 6H-pyrene 3.4 µm photons after 4 eV
    (Fig. 11): 2.68 against 2.67 photons per UV photon, mean 924 against
    917 K, each 100 K bin within 7 %.
  - T(U) is their harmonic curve to 1 % (Fig. 10).
- **Not reproduced.** Their 6.9/3.3 µm ratios (1.08 for 6H-pyrene at 4 eV,
  0.19 for Me-pyrene at 6 eV; ours 0.90 and 0.16). The 6.27 and 7.92 µm
  bands match, so the cascade is not the cause: their 6.9 µm CH$_2$/CH$_3$
  deformation band carries 15–20 % more energy (strict xfail).
- **Reading.** The aliphatic 3.4 µm profiles of their method are what
  pahchem computes, so their laws can be carried to the population's
  cascades (the "Next" item above).

**Aliphatic side groups on the fitted population (2026-10-20;
`diagnostics/pahchem/fit_34um_sidegroups.py`, figure `sidegroups_34um`).**
The "Next" item of the 3.4 µm study: the population of the full fit (fit
`392313363fe8`, best Gaussian cell) is given CH$_2$ and CH$_3$ side groups
that radiate from each molecule's own cascade.
- **Model.**
  - Per state: the residence time in each energy bin, from the CH modes'
    photons over their rates (the same for every mode), under the bases'
    field and recombination heating.
  - Each side-group band emits from each bin at its harmonic rate on the
    state's density of states. Its photons take the CosmicPAH-IRDB law
    position and FWHM at the bin's microcanonical T, as in the Demyk+26
    reproduction above.
  - Bands: the aliphatic CH stretches of 6H-pyrene (CH$_2$) and
    Me-pyrene (CH$_3$), {cite:t}`Demyk2026` Table 5. Intensity per
    aliphatic H = the lab ratio to the aromatic CH stretch per H (CH$_2$
    1.63, CH$_3$ 0.95) times the state's own aromatic CH stretch per H.
    Using the neutral's value instead changes the amounts by ≤ 15 %.
  - First order: the groups do not change the cascade, and the aromatic H
    they replace are not removed (the free aromatic scale takes that).
  - Check: the tables' CH-stretch power equals the bases' to 0.3 %.
- **Fit of 3.1–3.65 µm:** observed = a × (the full fit's 3 µm PAH spectrum)
  + x$_2$ CH$_2$ + x$_3$ CH$_3$ + a linear continuum, all ≥ 0.

  | | HII | Atomic | DF1 | DF2 | DF3 |
  |---|---|---|---|---|---|
  | 3.36–3.60 µm residual / mean, aromatic only | 0.27 | 0.38 | 0.44 | 0.48 | 0.48 |
  | with CH$_2$ + CH$_3$ | 0.09 | 0.14 | 0.15 | 0.14 | 0.13 |
  | (3.36–3.60)/(3.20–3.36), observed / model | 0.36 / 0.31 | 0.31 / 0.27 | 0.35 / 0.30 | 0.40 / 0.38 | 0.48 / 0.44 |
  | aliphatic H per aromatic H, CH$_2$ / CH$_3$ | 0.06 / 0.16 | 0.05 / 0.15 | 0.06 / 0.17 | 0.09 / 0.22 | 0.11 / 0.26 |
  | aliphatic C fraction | 2.5 % | 2.3 % | 2.6 % | 3.8 % | 4.5 % |

  (The observed ratio is over the fitted continuum. In the aromatic-only fit
  that continuum rises under the complex and the ratio drops to 0.15–0.23.)
- **Reading.**
  - **The complex appears.** One to two methyl and about one CH$_2$ group per
    molecule (N$_C$ ≈ 72) give the 3.4 µm complex in strength and place. The
    3.36–3.60 µm residual falls by 3–4×. The PAHdb aliphatic species could
    not do this: the earlier library ceiling was a median ratio of 0.008.
  - **The trend is physical.** The aliphatic share rises from the HII and
    atomic zones (2.3–2.6 % of C) to DF2–DF3 (3.8–4.5 %), into shielded
    gas, as the observed ratio does.
  - **What is missing.**
    - The model's 3.40 µm peak sits at ~3.405 µm, narrower than the
      observed flat-topped 3.395–3.40 µm band with its steep blue edge.
    - A CH$_3$ bump at 3.355 µm (2981 cm⁻¹) is not observed.
    - The 3.516 µm component is too weak: the 2846 cm⁻¹ CH$_2$ band.
    - All are the pyrene-family lab bands carried to larger PAHs, where
      the side-group environments (and their Fermi resonances) differ.

**Cold photons and the 3.40 µm peak (2026-10-20;
`fit_34um_sidegroups.py --below extrapolate`).** {cite:t}`Mackie2018` find
cascade bands with a steep blue wall at the 0 K position and a red wing.
- **Laws continued to 0 K below the lab data** (instead of held at 423 K)
  change nothing: 1 % of the CH$_2$ 3.40 µm photons are emitted below 423 K
  and ~75 % above 673 K, the top of the lab data. The complex is emitted
  hot, so its profile is set by the extrapolation of the laws upwards.
- **The 3.40 µm peak** (HII / DF3, µm): observed 3.3973 / 3.4002 (blue
  half-maximum 3.3858 / 3.3887). With the lab laws the CH$_2$ peak is at
  3.4016 / 3.4018 (3.3917). Frozen at the 0 K intercepts it is at 3.3932
  (3.3835).
- **Reading.** The observed band lies between the two: in the cascade it
  shifts about half as much as the thermal lab laws extrapolated past 673 K
  predict, as {cite:t}`Mackie2018` argue. The observed peak also moves
  0.003 µm red from HII to DF3, which the CH$_2$ band alone does not do
  (Thatte+26's 3.405 µm component). The test of the extrapolation is
  Mackie+18's fully anharmonic cascade of 6H-pyrene (supplementary material).

**The Extended Red Emission test (preliminary, 2026-10-06;
`diagnostics/pahchem/check_ere.py`).** Can RF of PAH cations make the ERE?
- **What it should match** (Lai+17, Gordon+98, Smith & Witt 2002, Berné+08,
  Witt & Vijh 2004):
  - a band at 540–900 nm peaking at 600–800 nm;
  - photon efficiency ≥ 10 % in the diffuse ISM, falling ~100× with U in
    reflection nebulae (IC 63 4.7 %, NGC 7023 0.2–0.6 %, Orion 0.4–1.3 %);
  - ERE/mid-IR ≲ 6 %;
  - no emission at 350–500 nm;
  - in NGC 7023 the ERE follows the neutrals (Berné+08), in IC 63 the
    cations (Lai+17).
- **The model (C$_{24}$ family):**
  - only coronene$^+$'s 1.70 eV (730 nm) level falls in the band;
  - C$_{54}^+$ emits at 1.18 µm and gives none, consistent with the
    ≲ 36 C carriers of Léger+88 and Duley 09;
  - blue/ERE ≈ 0.2–0.35 %, so no blue emission, as observed.

  | Environment | Model η (per absorbed PAH heating photon) | Observed η | Model ERE/IR | Observed ERE/IR |
  |---|---|---|---|---|
  | Diffuse ISM | 0.35 % | ≥ 10 % | 0.08 % | – |
  | IC 63-like ($G_0 = 650$) | 22 % | 4.7 % | 4.8 % | – |
  | NGC 7023 edge | 9 % | 0.2–0.6 % | 2.3 % | 1.6 % (N) |
  | Orion Bar | 100 % | 0.4–1.3 % | 29 % | ≲ 6 % |

- **The verdict: the wrong trend.**
  - Model η follows the cation fraction, so it rises with U where the
    observed one falls. This also contradicts NGC 7023, where ERE goes with
    the neutrals.
  - The magnitudes are of the right order at intermediate U. The observed η
    is per photon absorbed by all dust, so the model's would be diluted by
    the PAH share of the absorption.
- **It all rests on one unknown.** In these environments the C$_{24}$ family
  is almost fully dehydrogenated, and every N$_{\rm H}$ takes the parent
  cation's levels. With RF only for $N_{\rm H} \ge N_{\rm H0}$, η < $10^{-4}$
  everywhere.
  - So whether PAH-cation RF can make the ERE depends on the low-lying
    excited states of dehydrogenated cations, which no data constrain (T3
    request).
  - It also depends on the baseline's coronene dehydrogenation itself.
- **Not modelled:**
  - the RF band shapes and their Stokes shift, so the observed peak shift
    with U cannot be tested;
  - cation absorption below 5 eV, which also heats;
  - the dust share of the UV absorption.

**Benchmark figures** (`diagnostics/pahchem/plot_benchmarks.py`, 2026-10-05).
The script makes one PNG per benchmark, in `<plots dir>/pahchem/`, and runs
the same calculations as the gated tests:
- West+18;
- Jochims+94;
- Montillaud Figs. 3 and 10;
- Andrews Fig. 5, Table B and NGC 7023;
- Murga & Wiebe Fig. 1;
- the sensitivity tornado.

**Extended targets: method cards drafted 2026-10-05.** None of these models
has a runner yet.
- **`allain96.toml`** (Allain+96 I and II).
  - The model: one Jochims RRK law for every PAH, with single-photon
    branching at $E = h\nu$ and no cascade.
  - Tables 4–5 (I) and 2–3 and 5 (II) are transcribed; they are the cheapest
    targets.
  - Errors in the papers themselves:
    - two I Table 4 entries break their own $k_{\rm C_2H_2}/k_{\rm H_2} = 0.1$;
    - II eq. 10 is 3.3× too low;
    - I eq. 19 does not join eq. 18.
- **`lepage01.toml`** (Le Page+01, +03 and +09).
  - The "Le Page" coronene $k(E)$ in Montillaud's Fig. 3 is Le Page+01
    Fig. 9: a two-frequency RRKM with $E_0 = 4.8$ eV and $\Delta S = 5$ cal
    K$^{-1}$ mol$^{-1}$. It is reproduced to ±0.12 dex only with path
    degeneracy 12, which the paper does not state.
  - The 2003 Fig. 4 charge fractions and Table 2 rates are transcribed.
  - **The yield-law units.** Le Page's eq. 14 is in µm$^{-1}$. Montillaud+13
    eq. 1 converts $d$ to eV but keeps $b$, which lowers $Y$ 9× at 7.5 eV.
    `photo.yield_lepage2001` now takes `form="lepage01"` or
    `form="montillaud13"` (the default, which the Montillaud card uses).
  - The "Le Page yield" row of the sensitivity ranking used the Montillaud
    form.
- **`murga2020.toml`** (SHIVA, Murga+2019/2020).
  - The H loss is `arrhenius_effective_temperature` with Murga & Wiebe's
    "Arrh. 2" set, with $T_m$ from Draine & Li 2001 rather than from modes.
  - The Figshare data (doi:10.6084/m9.figshare.12597092) tabulate the Fig. 2
    profiles and the C$_2$H$_2$ production rates.
  - The unmerged branch `worktree-shiva-model` is not a starting point:
    - it cites a non-existent A&A paper;
    - its conditions are invented;
    - its cooling and charge physics are hand-made.
- **`murga2022.toml`** (Orion Bar).
  - The full model output is on Figshare (doi:10.6084/m9.figshare.14892612):
    158 positions × charge × hydrogenation × size × time, for models b, c and
    d. It reproduces their Fig. 8, so the targets are tabulated.
  - Model b (H loss only) is the closest to the $(Z, N_{\rm H})$ network.
  - Their "Andrews+16" H-loss parameters (4.3 eV, 11.8 cal) are not
    Andrews' Table 2 values.

**Allain+96 runner** ({mod}`pycalima.pahchem.allain96`,
`tests/test_pahchem_allain96.py`, `check_allain96.py`, 2026-10-05).
- **Inputs it substitutes, because the papers don't give them:**
  - Malloci+07 cross sections stand in for the Joblin (1992) spectra;
  - PAHdb modes stand in for the unstated Léger+89b Einstein A;
  - PAH 50 C and PAH 100 C are PAHdb C$_{50}$H$_{20}$ and C$_{100}$H$_{26}$.
- **Level 0 passes:**
  - the RRK convention;
  - $k_{\rm C_2H_2}/k_{\rm H_2} = 0.1$;
  - the eq. 11 yield;
  - the eq. 9 recombination (benzene$^+$ $1.0\times10^{-6}$);
  - Set 1 equals `fields.draine1978` to 0.13 %.
- **Photoionisation (I Table 5):** −0.11 to +0.03 dex for C$_{14}$–C$_{32}$.
- **Dissociation (I Table 4):** C$_{14}$–C$_{24}$ within 0.09 dex and
  C$_{32}$ at −0.22 dex. PAH 50 C is −1.35 dex, a strict xfail: its
  $N_{\rm H0}$ and $k_{\rm IR}$ convention are not stated.
- **Network:** the II Table 5 single-state destruction times agree to ≤ 0.37
  dex. The critical sizes are 39–46 C in the UIR media and 64 C in the warm
  diffuse medium, against their 40–55 C.
- **Network-mean destruction times:** 1–2 dex shorter than II Table 5, a
  strict xfail, because our steady state is more dehydrogenated. Their own
  text agrees with ours, not with their Table 5.
- **Paper II rate tables (II Tables 2–3), in the Set 2 field** (Gondhalekar+80,
  digitised from I Fig. 5 into `allain96_set2_field.csv`; $G_0 = 0.95$):
  - C$_{14}$–C$_{24}$ agree within 0.23 dex in every column;
  - C$_{32}$H$_{14}$ is −0.17 dex in its normal states and within 0.04 dex
    when dehydrogenated;
  - PAH 50 C and 100 C are −1.2 to −1.4 dex in their normal states but within
    0.2 dex when dehydrogenated: the same unstated $N_{\rm H0}$ and
    $k_{\rm IR}$ convention as I Table 4.
  - **Superseded on 2026-10-20** ({doc}`pahchem-audit`, C1, section 2).
    - **The old agreement was an artefact.** The dehydrogenated "within
      0.2 dex" came from the old stripping, which removed the most intense
      C–H modes first. With the card's own rule (band A scaled by the number
      of C–H bonds), the 50 C and 100 C dehydrogenated columns are
      −0.5 to −1.1 dex, like the normal ones.
    - **What does not explain it:** N$_{H0}$. Closing the gap would need
      N$_{H0}$ ≈ 15–16, below the compact minimum.
    - **What likely does:** reading "multi-quanta neglected" as keeping the
      occupation factor, which brings PAH 50 C to −0.1 to −0.34 dex. That is
      a convention question for the authors, not applied.
  - With this field the single-state destruction times agree to ≤ 0.21 dex,
    and the critical sizes move by < 0.7 C.
- **Ionised fraction (II Fig. 7)**, digitised for the Orion Bar and NGC 7027
  (`allain96_fig7_alpha_ion.csv`):
  - C$_{10}$–C$_{32}$: within 1–5 points in the Orion Bar and 1–2.5 points in
    NGC 7027.
  - In the Orion Bar the residual, as log odds, equals our I Table 5
    photoionisation residual to 0.06 dex. So the recombination side (II eq.
    9) reproduces and what remains is the cross section.
  - PAH 50 C and 100 C are +9 and +10 points (strict xfail): Li & Draine
    $\sigma$ makes their photoionisation 0.2 dex high. Removing that leaves
    1 point.
- **Hydrogenation (II Fig. 10)**, digitised for the Orion Bar cations and
  neutrals and for NGC 7027 (`allain96_fig10_alpha_h.csv`). $\alpha_{\rm H}$
  is normalised within each charge state.
  - From 24 C up the agreement is within 1.6 points in the Orion Bar and 3.4
    points in NGC 7027, including the full hydrogenation at 50 and 100 C.
  - In NGC 7027 the 10–32 C PAHs are bare, as theirs are. Their transition
    lies between 32 and 50 C, where we have no species.
  - In the Orion Bar the 10–16 C PAHs are 15–19 points (neutral) and 10–16
    points (cation) too dehydrogenated (strict xfail), as are the network
    means of II Table 5.
    - A factor ~2 in the H-loss / H-addition balance closes it. The tabulated
      states agree to 0.15 dex, so the suspects are the intermediate states,
      whose rates rest on the unstated C–H mode removal and band-$A$ scaling.
    - $\alpha_{\rm H}$ is steep here, which turns a 0.15–0.3 dex rate offset
      into 15 points.

**Le Page+01/03 runner** ({mod}`pycalima.pahchem.lepage01`,
`tests/test_pahchem_lepage01.py`, `check_lepage01.py`, 2026-10-05).
- **Dissociation rates:** the two-frequency transition state gives
  $\Delta S = 5.03$ cal K$^{-1}$ mol$^{-1}$ at every size.
  - Coronene $k(E)$ is within −0.07 to +0.15 dex above threshold (path
    degeneracy 12).
  - Benzene$^+$ and anthracene$^+$ are within 0.14 dex.
  - Naphthalene$^+$ is a strict xfail: Table 2's 4.5 eV does not reproduce
    it, and ~4.2 eV would.
- **The Appendix C$_{50}$ example** comes out at 0.578/0.268/0.155. It does
  so only if the recombination of C$_{50}$H$_{21}^+$ is dissociative.
- **Charge (LP03 Fig. 4):**
  - From LP03's own Table 2 rates it is reproduced within 0.015.
  - Our recombination, attachment and photodetachment rates match Table 2
    within 1.5 %.
  - With their own $\sigma$ (LP01 Fig. 11, digitised into
    `lepage01_fig11_sigma_per_c.csv`), our ionisation rate is 1.06–1.34× high
    and the fractions are off by up to 0.07 (strict xfail; 0.13 with Li &
    Draine).
  - The remaining offset is the yield units and the unstated IPs. With the
    eV form of the yield (Montillaud's transcription) the rate is 0.96–0.99×
    for C$_{16}$–C$_{32}$, but 0.72–0.82× for C$_{50}$–C$_{100}$. So the
    figures cannot tell which form Le Page computed.
- **Photodissociation (LP01 Fig. 10, digitised into `lepage01_fig10_kph.csv`):**
  - Fig. 10 and its caption fix two inputs the card listed as "not stated":
    eq. 11 integrates over the whole field, not from IP, and neutrals carry
    $(1 - Y_{\rm ion})$. These are now the defaults.
  - With them, PAH, PAH$^+$, PAHH and PAHH$^+$ agree within 0.21 dex over
    6–150 C.
  - The XdPAH classes (one H left) agree within 0.13 dex up to 24 C. They
    fall through $10^{-12}$ s$^{-1}$ within 1–2 C of theirs.
  - The PAHH$^+$ sp$^3$ H-loss path degeneracy is $N_{\rm H} - y$:
    $2(N_{\rm H} - y)$ is +0.22–0.29 dex high from 50 C up, where the rate
    is no longer saturated.
- **H and carbon loss against size (LP03 Figs. 6–7, digitised):**
  - Fig. 7's H loss is the cation rate, within 0.13 dex over 14–32 C.
  - Fig. 6's C$_2$ loss is the bare skeleton: with one H left, H loss wins.
    With 8 eV, the default 5 cal K$^{-1}$ mol$^{-1}$ and an assumed
    transition state (one C–C mode removed, two scaled), the neutral reaches
    C$^+$ insertion at 16.2 C, against 18.1 C on the figure and 18–20 C in
    the text.
  - The C$_2$H$_2$ curves are underdetermined, because the per-size $E_0$
    and the charge are not stated. 4.5 eV fits C$_{14}$–C$_{16}$, 6.0 eV the
    C$_{28}$ cation, but not pyrene at 5.5 eV. Recorded, not gated.
- **Hydrogenation (LP03 Fig. 9, card readings ±0.03; the figure is too hard
  to read for a proper digitisation):**
  - C$_{50}$H$_{20}^+$ is 0.60 against ~0.6.
  - C$_{80}$H$_{24}^+$ and C$_{80}$H$_{25}^+$ are 0.58/0.26 against
    0.6/0.27.
  - C$_{32}$H$_{14}^+$ and C$_{32}$H$_{14}$ are 0.50/0.33 against 0.45/0.37.
  - With benzene recombination instead, C$_{80}$H$_{25}^+$ drops to 0.08, so
    Fig. 9 used the naphthalene value.
- **LP01 Table 1:** the two-frequency column is reproduced within 0.25 dex by
  the neutral PAHH channel. The RRK column needs $E_0 \approx 0.7$ eV, not
  the stated value.
- **Not done:** LP09 (H$_2$ formation).

**Murga+2020/2022: data** ({mod}`pycalima.pahchem.murga_data`,
`tests/test_pahchem_murga.py`, 2026-10-05).
- **Fetchable datasets.** The Figshare model outputs are registered as the
  fetchable datasets `murga2020-figshare` and `murga2022-figshare`. Their
  fetchers check Figshare's MD5. `calima-fetch-data verify` now treats an
  unfetched `fetch` dataset as optional.
- **Parsers.** {mod}`~pycalima.pahchem.murga_data` reads:
  - the 2020 profiles and C$_2$H$_2$ rates;
  - the 2022 (location, $Z$, $t$, $X_{\rm H}$, size) populations of models
    b–d.
- **Figures read from the vector source.** The arXiv sources'
  matplotlib vector figures are read exactly with
  `diagnostics/pahchem/digitise_vector_figure.py` (tick calibration to
  $\le 10^{-4}$ dex):
  - M20 Fig. 3 (⟨Z⟩, integers);
  - M20 Fig. 4 (C$_2$H$_2$ per molecule, 20 parents × 2 $A_V$ × 2 $Z$ × 2
    sites);
  - M20 Fig. 5;
  - M22 Fig. 6 (the MARION profile).
- **Checks on the extraction:**
  - Fig. 4 reproduces the stated full destruction at Orion $A_V = 0.1$
    ($N_{\rm C}/2$ per neutral, to 0.01 %).
  - The 158 Figshare locations are placed on the Fig. 6 distance axis by
    their $\chi$. The Fig. 6 $T_{\rm gas}$ and $n_{\rm H}$ at those distances
    reproduce the Figshare headers (median −0.001 and −0.003 dex).
  - The $n_e$, $n_{\rm H_2}$, $n_{\rm C}$ and $n_{\rm C^+}$ that the
    Figshare files lack are now tabulated per location
    (`murga2022_fig6_profile.csv`).

**Murga: charge** ({mod}`pycalima.pahchem.murga_charge`,
`tests/test_pahchem_murga_charge.py`, `check_murga_charge.py`). WD01 chain
on pyCALIMA's WD01 rate functions.
- **M22 Figshare f(a, Z)** (158 locations × 10 sizes, $\chi$, $T$ and
  $n_e$ per location):

  | Region | median $\Delta\langle Z\rangle$ | max $\lvert\Delta f\rvert$ | modal Z agrees |
  |---|---|---|---|
  | PDR 1 | −0.04 | 0.13 | 92 % |
  | PDR 2 | −0.03 | 0.05 | 96 % |
  | DF | −0.00 | 0.03 | 100 % |
  | MC | −0.00 | 0.03 | 100 % |

  - **Inferred, not tested.** The files fix choices Murga do not state, so
    this is an inference of their configuration:
    - $s_e = 1$ for every $Z$ (the WD01 law gives +0.5–0.8 in the PDR);
    - the MMP83 shape for $\chi$, although MARION uses Draine;
    - no C$^+$ collisions;
    - WD01 eq. 24 $Z_{\rm min}$ truncated as printed;
    - the chain truncated at $Z = -2..9$.
  - **H II region:** −0.97 without EUV (strict xfail). M22 Sect. 3.2 adds
    photons above 13.6 eV "with photon flux corresponding to an O6-type
    star", without stating the spectrum or the normalisation; our field
    stopped at 13.6 eV.
    - With a Kurucz 40 000 K EUV, matched at 6–13.6 eV and scaled ×4 (the
      one inferred number), every size up to 14.2 Å is within 0.06 in ⟨Z⟩
      (max $\lvert\Delta f\rvert$ 0.11; gate
      `test_m22_hii_region_with_o6_euv`).
    - One factor fitting all ten sizes is a real test, since the sizes
      respond very differently to EUV. A lower $n_e$ alone cannot fit them
      all: $n_e/8$ leaves 19.9 Å at −1.1, while $n_e/16$ puts 6–12 Å at
      +0.3–0.4.
- **M20 Fig. 3** (66 integer ⟨Z⟩): 48/66 with M22's configuration and at
  best 57/66 (strict xfail).
  - The Horsehead C$_{66}$H$_{20}$ anion needs the floor $Z_{\rm min}$,
    which the M22 files exclude, so the two papers ran different charge
    conventions.
  - Orion C$_{128}$H$_{28}$ at +2 is not reproduced by any variant tried.
- **pyCALIMA bugs found** in `models/dust_charge`, not fixed here:
  - its solver overflows at $\chi \lesssim 0.01$ and $T \approx 13$ K;
  - its $Z_{\rm min}$ rounding goes one charge below WD01 eq. 24 as printed.

**Murga: the SHIVA rate engine** ({mod}`pycalima.pahchem.shiva`,
`tests/test_pahchem_shiva.py`, `check_shiva.py`).

The engine is built from these pieces:
- the DL01 synthetic modes and their $T(E)$;
- Tielens' Arrhenius-$T_e$ law;
- the thermal IR photon-emission rate from LD01 $C_{\rm abs}$ (DL07 is not
  bundled);
- the WD01a yield;
- a stationary $p(E)$ under continuous cooling, both time-averaged and
  post-absorption (M20 does not say which);
- the M20 eq. 4–6 rate.

Gates:
- **Inputs.** DL01 $T(E)$ reproduces DL01 Fig. 7 to 0.004 dex. The WD01a
  yield reproduces M19 Fig. 2 to 0.0012. Both figures are read from the
  vector source.
- **MW22 Fig. 1** (exact, `murga_wiebe22_fig1.csv`; the eye-read gate is
  retired):
  - **RRK.** RRK has no $T(E)$, so it tests $p(E)$ and absorption: within
    0.33 dex at $\chi \ge 100$.
    - At $\chi = 1$ it is −1.4 dex (strict xfail). There two-photon events
      are needed, so the point probes the tail of $p(E)$, which differs
      between continuous cooling and their Monte Carlo.
  - **The Arrhenius forms** are reproduced only with Tielens'
    $T_m = 2000\,{\rm K}\,(E/N_{\rm C})^{0.4}$: H +0.6 dex, C ≤ 0.1 dex at
    $\chi = 100$.
    - With the DL01 relation that M20 states, they are 1.6–3.4 dex off
      (strict xfail). DL01 runs 5–10 % hotter at 10–14 eV, and
      $E_0/kT \approx 45$ turns that into 2–3 dex.
    - **This is the 3–5 dex "rate-form" gap found in M5: it was the $T(E)$
      relation all along.**
  - **Laplace** remains −2 to −4 dex: a $\rho(E)$ convention (fewer modes,
    or the ZPE included). It is recorded as a bound test.
- **Which $T(E)$ SHIVA ran.** The two choices shift the parent rates by 2–3
  dex. The M22 Figshare chains say DL01, as M20 states (see the next
  section), so MW22's Tielens-like curves are the exception.

**Murga+2022 network, model b** ({mod}`pycalima.pahchem.murga2022`,
`tests/test_pahchem_murga2022.py`, `check_murga2022.py`). H loss and
addition only, as bin-to-bin fluxes on the 5 $X_{\rm H}$ bins, per size and
charge class, at the 158 locations.

What the Figshare data reveal:
- **Output times:** linear, 0 to $10^5$ yr. The pure-loss chains of 3.3 Å
  neutrals in the deep MC fit exactly.
- **$T(E)$:** DL01, as M20 states. Their second-to-first step ratio is 1.250
  at every $\chi$; ours is 1.257 with DL01 and 0.787 with Tielens.
  - So MW22 (Tielens-like) and M20/M22 (DL01) used different $T(E)$
    relations within SHIVA.
  - The post-absorption yield reading is the right one. The time-averaged
    reading is ~12 dex off.
- **$R_{\rm H}$ includes the hydrogen of C$_2$H$_2$ loss** (2026-10-06).
  M22 Sect. 3.1: $R_{\rm H}$ is "the net rate of detachment of all hydrogen
  atoms which might be detached ... both separately and also inside
  hydrogen or acetylene molecules". Model b keeps the carbon. This reading
  takes the deep-MC rate from 1.5–2.5× to 0.87–1.44× ours, and the step ratio
  from 1.278 to 1.257.
- **Channels are independent.** Competing channels (M19 eq. B2) would put
  the Figshare deep-MC rate 2.8–4.6× above ours.
- **What M22 say changed relative to M20,** and so what may differ between
  the two configurations (`murga2022.toml`, `stated_changes_vs_m20`):
  - the super-H $E_0$ law (Table 3);
  - the bin-to-bin network;
  - the $R_{\rm H}$ definition;
  - the $X_{\rm H}$ grid;
  - the carbon-loss cases a–d;
  - no H$_2$ addition;
  - the emission cross sections.

  Not stated as changed: $p(E)$, branching, H addition, $T(E)$ and the field
  shape.

Gates:
- **Deep-MC step rate:** theirs is 0.87–1.44× ours (DL01; 1.5–2.5× without
  the C$_2$H$_2$ hydrogen). The factor grows with depth, which suggests a
  field shape that changes inward (not stated).
- **3.3 Å dehydrogenation** at $10^5$ yr: within 0.03 in every region (0.07
  without the C$_2$H$_2$ hydrogen).
- **PAHs ≥ 6 Å, resolved 2026-10-20** ({doc}`pahchem-audit`, C3).
  - The Figshare chains fix M22's rates. With the rates read from them
    (bin flux step 0.5 N$_{H0}$, a 2-H Eley–Rideal event, and the addition
    laws in `murga2022.h_addition`), ⟨X$_H$/X$_{H0}$⟩ is within +0.00 to
    +0.05 of Figshare in PDR 1, DF and MC.
  - PDR 2 keeps +0.12 to +0.17 at 6–8.7 Å (strict xfail): the neutral top
    bin fills there.
  - The record below is the pre-fix diagnosis, with M20's rates as printed.
- **Before the fix:** PAHs ≥ 6 Å ended 60–90 % super-hydrogenated where
  Figshare keeps them normal.
  - The M20 H-addition rates as printed outrun the super-H photo-loss by
    $10^3$–$10^4$.
  - The Figshare super-H steps are ~$2\times10^{-13}$ s$^{-1}$ whatever
    $n_{\rm H}$.
  - So M22's addition or super-H loss differs from M20 as printed.
  - The one change M22 state there makes it worse. The Table 3 law puts
    the neutral super-H H-loss $E_0$ at 3.3 eV at $X_{\rm H}/X_{\rm H0} =
    1.37$, against 1.4 eV in M20.
  - So no stated update explains it. **Open (authors).**
- **Not attempted:** models c and d (C loss and addition), because model b's
  super-H balance is not in hand.

**Murga+2020 trajectories** ({mod}`pycalima.pahchem.murga2020`,
`tests/test_pahchem_murga2020.py`, `check_murga2020.py`). Eqs. 1–3 of M20:
real-valued $N_{\rm C}(t)$ and $N_{\rm H}(t)$ at fixed $Z$, with photo-rates
tabulated on $(N_{\rm C}, N_{\rm H})$ and Goumans/Fig. 1 H addition. The
integrator is stiff LSODA; it matches M20's explicit Euler to $10^{-8}$ where
Euler is stable.
- **Full destruction (Orion $A_V = 0.1$, $Z = 0$):** all 20 parents give
  $N_{\rm C}/2$ to 0.1 %. Goumans addition super-hydrogenates the neutrals,
  and the 2.0 eV super-H C$_2$H$_2$ channel destroys them.
  - C$_{150}$H$_{30}$ dies at $1.6\times10^3$ yr with DL01, against the
    $10^4$ yr end of production in Figshare. Tielens gives 24 yr.
- **Partial destruction:** Orion cations, Orion $A_V = 2$ and the Horsehead
  are 1–4 dex above Fig. 4 with independent channels (140 points: rms 2.53
  dex, median +2.45; strict xfail).
  - Under any H addition the real-valued $N_{\rm H}$ drifts above
    $X_{\rm H0}$ and opens the super-H channel, which runs ~$10^{12}$×
    faster than the normal channels.
  - Fig. 4 needs a super-H contribution $10^2$–$10^4$× smaller. M20's
    threshold handling is not published.
- **Channel competition** (M19 eq. B2; 2026-10-06):
  - It brings the 140 partial points to 1.16 dex rms (median +0.06).
  - But the Orion $A_V = 0.1$ neutrals with $N_{\rm C} \ge 110$ then stop
    short of full destruction by 0.6–1.3 dex (strict xfail).
  - The M22 deep-MC chains also reject competition (above).
  - The default stays independent. **Open (authors).**
  - $[1 - Y_{\rm ion}]$ in the rate changes the Fig. 4 rms by < 0.1 dex;
    eq. 6 as printed is kept.
- **$p(E)$ verdicts conflict between the papers.**
  - M20 Fig. 4 needs the time-averaged reading: post-absorption makes the
    Horsehead coronene lose C$_2$H$_2$ ~$10^4$× too fast.
  - The M22 Figshare chains need the post-absorption yield: time-averaged is
    ~12 dex off.
  - Either SHIVA changed between the papers, or the two used a definition
    neither states.
  - Channel competition and $[1 - Y_{\rm ion}]$ do not move either verdict:
    M20 time 1.16 dex against post 4.40 with competition.
  - M22 state no change to SHIVA's $p(E)$. **Open (authors).**

What the remaining targets need:
- **Raster figures the user is digitising:**
  - for Allain: II Fig. 9, and Figs. 7 and 10 for the other four media (I Fig. 5 and the Orion Bar and NGC 7027 curves of Figs. 7 and 10 are done);
  - for Le Page: LP01 Figs. 10–11 and LP03 Figs. 6–7 are done. LP03
    Fig. 9 is not digitisable; the card readings stand.
- **Murga:**
  - its figures are extracted (above);
  - the network needs a size ($N_{\rm C}$) axis, time integration through a
    profile, and a WD01 charge mode.

Invariants hold at every level:
- each per-molecule photo-rate is at most $r_{\rm abs}$, and each yield per
  photon is at most 1;
- the cascade conserves energy;
- steady-state populations sum to 1;
- detailed balance holds where it applies.

After M5, a one-at-a-time sensitivity analysis at Orion Bar and NGC 7023
conditions ranks the uncertain inputs by their effect on $f(N_{\rm H})$ and
$f(Z)$. That ranking decides the second round of requests in
{doc}`pahchem-data-requests`.

## Cosmic-ray collisions

This section covers PAH excitation, ionisation and destruction by direct
cosmic-ray (CR) collisions: the primary nuclei and electrons, and later their
secondary electrons. The model and results are dated 2026-10-08.

Two things are left for later:
- **The CR-induced UV field.** By the estimate in the plan it out-heats direct
  impacts about 30× at any ζ.
- **H$_2$ line shielding of PAHs**, the shielded regime of {cite:t}`Thatte2026`.

So direct collisions matter for **destruction and ionisation**, which only
large single deposits do, and not for the IR brightness.

**Inputs (sourced, not tuned).**
- **Interstellar spectra.** {cite:t}`Padovani2018` L and H protons and their
  electrons, with nuclei in the {cite:t}`PDG2022` ratios at fixed energy per
  nucleon, in groups He … Fe–Ni.
- **Attenuation.** Continuous slowing down through N(H$_2$), with the NIST
  {cite:p}`NISTSTAR` loss functions for H$_2$.
- **Secondary electrons: the spectrum exists, but the collisions do not use it
  yet.** It comes from CRISpy's Spencer–Fano degradation in H$_2$ (2026-10-19)
  and is extracted by `diagnostics/pahchem/extract_crispy_secondary_spectrum.py`,
  run with CRISpy's interpreter. The output is 4πj$_{\rm sec}$/ζ in
  cm⁻² eV⁻¹, written to `results/pahchem/cr_secondary_electrons/` for:
  - spectra L and H;
  - x$_e$ = 0–10⁻³;
  - log N(H$_2$) = 20–24;
  - 1 eV – 100 keV, 10 bins per decade;
  - primary electrons excluded.

  Checks:
  - 4πj L$_{\rm ESTAR}$ / S(>ε) = 0.98–1.00 at 10–30 keV, and 1.18–1.26 at
    1 keV (cascade electrons, and CRISpy's approximate low-energy loss);
  - ζ$_{\rm sec}$/ζ$_{\rm prim}$ = 0.75–0.94 (L) and 0.85–0.95 (H) at
    10$^{21}$–10$^{24}$ cm⁻², against Ivlev et al. (2021, ApJ 909, 107)
    0.8–1.1;
  - CRISpy's B/C excitation per ionisation is within 4 % of Padovani+2026.

  Properties and caveats:
  - The raw 0.02 eV bins below 40 eV carry a comb from the discrete losses
    (50–80 % rms), so only rebinned values are meaningful.
  - Per ζ, the spectrum below ~300 eV hardly depends on N; at keV it rises up
    to 30× from 10$^{20}$ to 10$^{24}$ cm⁻².
  - Coulomb losses suppress it below 10 eV at x$_e$ ≳ 10⁻⁵.
  - The medium is pure H$_2$, with no He.
  - Secondaries born above 100 keV are lumped into the top bin.

  Still to do: bundle the table, then use it in `impact_rates` and
  `check_cr_impact.py`.

**Energy transfer** ({mod}`pycalima.pahchem.impact`). Bethe theory is written
over the PAH's own optical oscillator strength df/dW, in the manner of
{cite:t}`Liljequist1983`, with two kinds of collision:
- **distant collisions:** a virtual photon of energy W, whose outcome is that
  of an absorbed photon;
- **close collisions:** an ionisation, with a δ electron that leaves and a
  hole of energy max(W, IP) that is kept.

df/dW is Li & Draine below 40 eV and {cite:t}`Henke1993` atomic C + H above.
It meets the TRK sum rule to 1–4 % (C$_{14}$–C$_{96}$) and gives I = 59 eV.

**Gate G1 passes.** With no correction, the stopping per electron is within
1–7 % of NIST polystyrene, the measured aromatic CH stopping:
- 1–100 MeV/u for protons and alphas;
- 10 keV – 1 GeV for electrons, against ESTAR with the condensed-medium density
  effect removed.

Two corrections, both from measurements, apply only where Bethe fails:
- **Below 1 MeV/u**, multiply by the polystyrene STAR/Bethe ratio (0.93 at
  0.3 MeV). The model stops at 0.1 MeV/u.
- **For electrons below 10 keV**, apply {cite:t}`Kim2001`'s binary-encounter
  factor T/(T + IP + W). Stopping is then within 15 % of Joy (1995) carbon
  above 80 eV, 1.35× at 50 eV and 2.3× at 30 eV (confidence D).

**Gate G2 (electron-impact ionisation of PAHs): improved by the 2026-10-20
audit fixes C5 and C4, not closed.** The fixes use Kim's ionisation scaling,
T/(T + B + U) with U = 2B, for the ionising collisions only, and the volume
scaling of σ below N$_C$ = 21. With them, model/BEB over 15 eV–5 keV is:

| | before | after |
|---|---|---|
| anthracene | 1.36–2.30 | 0.99–1.41 |
| coronene | 1.28–2.21 | 1.10–1.55 |
| C$_{54}$H$_{18}$ | 1.40–2.58 | 1.39–1.80 |

The remaining ~1.2–1.4× at keV is the Li & Draine dipole strength
({doc}`pahchem-audit`, P2). The record below is the pre-fix state
(2026-10-19).

*The reference.* `diagnostics/pahchem/beb_hf_pah.py` computes BEB (Kim & Rudd
1994) from RHF orbitals: B = −ε (Koopmans) and U = ⟨T⟩, using PySCF on ideal
planar geometries. PySCF runs in its own environment; it is not a pycalima
dependency.
- It reproduces the BEB(HF) peaks of Krishnadas et al. (2024, Phys. Scr. 99,
  095403, Table 4): anthracene 28.5 against 28.1 Å², pyrene 31.6 against
  31.0 Å².
- 6-31G and cc-pVDZ agree to 0.1 %.
- The OVGF-based peaks of Appl. Sci. 15, 9562 (2025, Table 2; naphthalene to
  hexacene) are 12–15 % higher. That spread is the systematic of the
  reference.
- No laboratory cross section exists for these PAHs, so G2 is a comparison
  against theory.
- Results are in `results/pahchem/beb_hf/` (σ(T) and the orbital B, U, N):
  C$_{10}$H$_8$ to C$_{26}$H$_{16}$, C$_{16}$H$_{10}$, C$_{24}$H$_{12}$ and
  C$_{54}$H$_{18}$.
- The peak is σ$_{\max}$ ≈ 0.43 Å² per valence electron (0.415 for
  C$_{54}$H$_{18}$), at 72–75 eV for every size.

*The model against it.* `impact` (σ$_{\rm ion}$ = collisions with dz ≥ 1, the
Jochims yield) over BEB(HF):

| T (eV) | 15 | 20–30 | 50 | 72 | 1000–5000 |
|---|---|---|---|---|---|
| C$_{10}$–C$_{26}$, C$_{24}$H$_{12}$ | 1.6 | 2.2–2.4 | 1.9–2.1 | 1.75–1.95 | 1.26–1.47 |
| C$_{54}$H$_{18}$ | 2.1 | 2.6–2.8 | 2.2 | 2.0 | 1.40–1.44 |

The model also peaks at ~50 eV instead of ~72 eV.

*Diagnosis so far:*
- **High energy.** The 1/W-weighted ionising dipole strength is too large:
  M²$_{\rm ion}$ = ∫(R/W) y df/dW dW = 35.7 for anthracene, against BEB's
  Σ N R/2B = 27.9 (×1.28). It is not yet known whether this comes from the
  Li & Draine σ at 10–30 eV or from the yield.
- **Near the peak.** The likely extra factor is the slow-electron denominator:
  T + IP + W here, against BEB's T + B + U, with U ≈ 2.1 B (median U ≈ 38 eV
  for the valence orbitals). This has not been tested.
- **Ruled out.** Restricting close collisions to Q ≥ IP changes the peak by
  only ~3 %.
- **Unaffected.** Stopping (G1) still passes, so the deposited energy is
  sound. The number of collisions counted as ionisations is high.

*Open decision.* There are two options:
1. use BEB(HF) for the dz = +1 rate of electrons, and keep the Bethe df/dW
   for the energy partition;
2. repair the model's dipole strength and denominator from physics, not by
   fitting.

Until then, electron-impact ionisation rates are high by these factors.

**Collision outcomes.** Each collision is mapped to a charge change and to the
internal energy E* left in the product:
- Ionising distant collisions follow the photoionisation yield. The ion keeps
  E* uniform on [0, W − IP], bracketed by 0 and by W − IP (`partition`).
- K-shell holes decay by Auger to dz = +2, with E* = 27 eV − IP₁ − IP₂.

Several collisions in one passage are summed:
- Poisson statistics across the disc of {cite:t}`Micelotta2011`;
- n-fold convolutions truncated to the grid, so there is no wrap;
- tested to conserve crossings and collisions.

Iron at 10 MeV/u makes about 60 collisions crossing C$_{54}$H$_{18}$ and always
ends beyond any survivable charge (Coulomb explosion, {cite}`Chabot2020`).
Per passage with a collision, protons and electrons leave a mean of about
9 eV (19 eV at 0.3 MeV), with a power-law tail. CNO and the Fe group extend
to hundreds of eV (`cr_impact_deposition.png`). About 76 % of passages with a
collision ionise the molecule.

**Fragmentation.** E* goes into a sequential statistical fragmentation chain:
- Whitten–Rabinovitch RRKM rates for H, H$_2$ and C-loss channels, branching
  k_c/Σk, and a Klots kinetic release;
- the survival probability from the IR cascade below 40 eV.

The dissociation onset, where P(diss) = 0.5, is 10.5, 19.4 and 31 eV for
C$_{24}$H$_{12}$, C$_{54}$H$_{18}$ and C$_{96}$H$_{24}$. Two C-loss channels
bracket the carbon loss:
- Micelotta+10b C$_2$ (4.6 eV, 41.8 J K⁻¹ mol⁻¹);
- West+18 acenaphthylene$^+$ C$_2$H$_2$ (5.21 eV, 46 J K⁻¹ mol⁻¹).

Benzenoid PAH cations show no C$_2$H$_2$ channel at iPEPICO energies. Daughter
molecules keep the parent's modes, so the H loss is an upper limit at very
high E*.

**Destruction.** A molecule is destroyed when any of these happens:
- its charge reaches 4 N$_C$/60 (Chabot+20, scaled from C$_{60}$);
- its E* leaves the grid (512 eV);
- its expected carbon loss reaches N$_C$/3, Micelotta's criterion. Gradual
  carbon loss counts towards that third.

The Chabot criterion makes C$_{24}$H$_{12}$ dications explode, and lab
coronene dications are stable. That threshold needs a laboratory anchor.

**Gate G4 passes.** {mod}`pycalima.pahchem.micelotta2011`, run from
`method_cards/micelotta2011.toml`, reproduces both of their testable numbers:
- The CR-versus-shock crossover sizes of their Sect. 7.1: N$_C$ = 152 for
  f = 0.5 (stated ~150), and 319 for f = 1 (stated ~280).
- Their Table 2 electron lifetimes, to 4 %, but only without the 0.5
  orientation factor printed in their eq. 31. That is recorded as a card
  discrepancy.

**Results (primary CRs, neutral parents; `check_cr_impact.py`).** Lifetimes
are bracketed by the two C-loss channels:

| | unattenuated, L | unattenuated, H | N(H$_2$) = 10$^{22}$, L | N(H$_2$) = 10$^{22}$, H |
|---|---|---|---|---|
| C$_{24}$H$_{12}$ | 3–5 Myr | 0.5–0.6 Myr | 150–220 Myr | 12–16 Myr |
| C$_{54}$H$_{18}$ | 16–60 Myr | 0.8–1.0 Myr | 300–450 Myr | 25–35 Myr |
| C$_{96}$H$_{24}$ | 35–180 Myr | 1.4–1.7 Myr | 460–650 Myr | 37–50 Myr |

- **Against Micelotta+11** (their spectra, f = 1): 20 Myr at C$_{54}$ and
  40 Myr at C$_{96}$.
- **Against {cite:t}`Chabot2020`.** They count Coulomb explosion by heavy CRs
  alone, scaling the C$_{60}$ threshold Q/N$_C$ = 4/60 to every size, and
  find "billions of years" for C$_{54}$. Here explosions alone give
  C$_{54}$H$_{18}$ 5×10⁸ yr (L, N(H$_2$) = 10$^{22}$). Most of the shorter
  total lifetime is the statistical carbon loss of the sub-explosion deposits.
- **Which projectiles matter.** At the cloud edge, primary electrons dominate
  for spectrum L. That rests on the extrapolation of the Padovani+18 electron
  spectrum to keV energies, where it is unmeasured. Deeper in, the Fe group
  and CNO dominate for L; protons, He and CNO for H.
- **Impact ionisation** of C$_{54}$H$_{18}$: 4×10⁻¹⁵ (L) to 4×10⁻¹⁴ s⁻¹ (H)
  at N(H$_2$) = 10$^{22}$.

**IR emission.** Most collisions are UV-like. Bethe's f(W)/W weighting puts
them at the 5–20 eV resonances: the mean deposit is ~10 eV, though the
projectile loses ~37 eV per collision, mostly to the δ electron. Half the
deposited energy, however, arrives in the 15 % of passages that leave more
than 13.6 eV, and 20 % of it arrives above 40 eV (C$_{54}$H$_{18}$,
N(H$_2$) = 10$^{22}$).

Those deposits fragment until the last daughter can cool, and that daughter
radiates `FragmentationChain.residual(E*)`. For C$_{54}$H$_{18}$ this is
16–17 eV whatever E*: fragmentation is a thermostat, the evaporative-ensemble
ceiling.

The yield caps the neutral products: it reaches 1 at IP + 9 eV, so every hot
deposit ends as an ion. With CR/UV band ratios taken at the same charge:
- **Neutral products** keep the UV ratios: 3.3/11.2 is ×1.04–1.06, the others
  are within 2 %.
- **The products the collisions ionise** are about three quarters of the
  excited molecules. Their 3.3/11.2 is ×1.65–1.78 (0.56–0.60 against 0.335),
  6.2/11.2 ×1.09–1.13 and 7.7/11.2 ×1.04–1.07.

So CR-excited emission carries a hot-cation signature, a cation spectrum with
a strong 3.3 µm band, even where the PAH population is neutral or anionic.

An earlier version stopped the emission at the first dissociation and dropped
deposits above 40 eV. That removed the hot tail and gave only ×1.3–1.4. The
"uniform" ionisation-energy partition is the middle case: with "all" the ions
are hotter still.

The heating is 5×10⁻¹⁴ (L) to 5×10⁻¹³ eV s⁻¹ (H) per C$_{54}$H$_{18}$ at
N(H$_2$) = 10$^{22}$. Under G$_0$ = 1 the same molecule absorbs
~10⁻⁷ eV s⁻¹.

**Next.**
- Fix gate G2 (see above), then put the secondary-electron spectrum into the
  collisions.
- The CR-UV field, as a radiation field with H$_2$ line blanketing.
- The aliphatic side groups of the 3.4 µm work under the same deposits.
- Charge-resolved modes for the dications and above.

## Process specification

| Process | Rate | Parameters | Best current source | Conf. | Gap | Owner | Priority |
|---|---|---|---|---|---|---|---|
| UV absorption | $r_{\rm abs} = \int \sigma(E)F(E)dE$ | $\sigma(E, Z, N_{\rm H})$ | {cite:t}`Malloci2007` (≤ C$_{66}$, Z = −1…+2) | C | $N_{\rm C} > 66$, H ladders | Malloci data now; TD-DFT later (T3) | 1 |
| Photoionisation | $\int \sigma Y_{\rm ion} F dE$ | IP, $Y_{\rm ion}(E)$, electron energy | {cite:t}`Jochims1996`, {cite:t}`Zhen2016`, {cite:t}`Wenzel2020` | A | neutrals > C$_{24}$; photodetachment | literature | 1 |
| IR cooling | discrete modes; thermal limit | $\nu_i$, $A_i$ | PAHdb v4.00 {cite:p}`Ricca2025` | C | H ladders × charge at one level | Oxford T1 | 1 |
| Recurrent fluorescence | $k_{\rm RF}(E)$ | $D_n$ energies, $f$ | {cite:t}`Martin2013`, {cite:t}`Bernard2023` (small cations) | A | $N_{\rm C} \ge 24$ cations | TD-DFT (T3) | 2 |
| H loss (aromatic) | RRKM, loose TS | $E_0$, $\Delta S^\ddagger$ | {cite:t}`West2018`, {cite:t}`Jochims1994`, {cite:t}`Castellanos2018` | A/B | neutral, dication, sites | iPEPICO $\Delta S^\ddagger$ + relaxed scans (T2) | 1 |
| H loss (super-H) | RRKM | $E_0$ per site | assumed 1.4–1.55 eV {cite:p}`Andrews2016,Murga2020` | C/E | everything | reverse-addition TS (T2) | 1 |
| H$_2$ loss, H roaming | RRKM, tight TS | TS energies, frequencies | {cite:t}`Castellanos2018`, {cite:t}`West2018` | B | neutral, large | T2 | 1 |
| C$_2$H$_2$, C$_2$ loss | RRKM / parametric | multi-step barriers | {cite:t}`Allain1996a`, {cite:t}`West2019` | D | compact large PAHs | parametric | 2 |
| H addition | $\sigma v$, barrier | per site and charge | {cite:t}`Boschman2015`, {cite:t}`Cazaux2016` (coronene$^+$) | B | site × charge × $n$ | T2 | 1 |
| H abstraction (Eley–Rideal) | $\sigma_{\rm er} v n_{\rm H}$ | $\sigma_{\rm er}$, barriers | {cite:t}`Mennella2012` | A | site and charge | T2 | 1 |
| Recombination, attachment | DS87 $\tilde J$ capture | $s_e$, EA, product branching | {cite:t}`BakesTielens1994` | B | products after recombination | RRKM on the neutral | 1 |
| Mutual neutralisation | Langevin | — | — | E | — | us | 2 |
| CR collisions: energy transfer | Bethe over df/dW, distant + close | df/dW to 30 keV, IP | Li & Draine + {cite:t}`Henke1993`; NIST STAR (G1) | B | slow electrons (< 100 eV); PAH EII | literature, lab | 2 |
| CR collisions: fragmentation | sequential RRKM | $E_0$, $\Delta S^\ddagger$ of C loss; Coulomb limit | {cite:t}`West2018`, Micelotta+10b, {cite}`Chabot2020` | D | C loss of benzenoid PAHs; $q_{\rm CE}$ | T2, lab | 2 |
| CR spectra at the PAH | CSDA through N(H$_2$) | L/H, nuclei, secondaries | {cite:t}`Padovani2018`, {cite:t}`PDG2022` | C | secondary e⁻ spectrum | literature | 2 |

## Literature reviewed, and what it is used for

These are papers read for this design beyond the ones the tables above cite.
They are recorded here so that the same papers are not reassessed later.
"Cite only" means the paper supports an assumption but provides no number
we use.

| Paper | What it provides | Use |
|---|---|---|
| {cite:t}`Blanquart2007` | G3MP2B3 thermochemistry of 46 neutral PAHs and aryl radicals up to coronene. $\Delta_f H(298)$ of pyrene is 226.1 kJ mol$^{-1}$ (experiment 225.7) and of coronene 292.4 (experiment 302.0). B3LYP frequency scale factor 0.966. | Check of the collaborators' neutral-parent thermochemistry. Do **not** take C–H bond energies from its radical table: they give ≈ 5.5 eV for pyrenyl, against the ≈ 4.7 eV usually quoted. |
| {cite:t}`Gatchell2021` | Hot coronene-cation fragments stored in DESIREE (13 K) stop decaying after ~10 ms, attributed to radiative cooling; power-law decays. | A qualitative constraint on IR and recurrent-fluorescence cooling of coronene$^+$ (M3). It becomes quantitative only if the supplement gives the decay curves and initial energies. |
| {cite:t}`Lee2021` | XUV-pumped fluorene, phenanthrene and pyrene ions: electronic relaxation in < 100 fs for PAH$^+$ and PAH$^{2+}$. Dications prefer losing even numbers of C atoms. | Cite only: it supports the statistical (internal conversion first) basis of RRKM. |
| {cite:t}`Cilesiz2025` | $\Delta_f H$ of 8 neutral PAHs with 9 functionals; double hybrids are best (DSD-PBEP86, MUE 0.6 kcal mol$^{-1}$). | Cite only: supports recommending double-hybrid single points ({doc}`pahchem-data-requests`). |
| Fernández 2020 (Chem. Sci. 11, 3769); Su et al. 2023 (ChemPhysChem 24, e202200638); Mao et al. 2021 (Proc. Combust. Inst. 38, 729) | Diels–Alder reactivity; atmospheric nitration of neutrals; C$_5$H$_5$ combustion kinetics. | Not relevant. |
| {cite:t}`NavarroNavarrete2023` | 1-cyanonaphthalene$^+$ in DESIREE: the first measured energy-loss rate $-dE/dt(E)$ of a PAH cation (their Fig. 11) and $k_{\rm diss}(E)$ (Fig. 9). RF inputs: $h\nu_{\rm el} = 1.10$ eV and $A_{\rm RF} = 2\pi\nu^2e^2f/(\epsilon_0 m_e c^3)$. Their calculated $f = 0.011$ needs Herzberg–Teller coupling (without it, $10^{-4}$); fitted $f = 0.0074$. HCN loss $E_a = 3.16$ eV, $A = 2\times10^{13}$ s$^{-1}$; $k_c = 302.9$ s$^{-1}$. At 3–5 eV the loss rate is up to 100× the IR-only value. | **The first RF gate.** Reproduce Fig. 11 with `dos` + `cooling` + RF (M3). The full model is in Stockett et al. 2023 (Nat. Commun. 14, 395; arXiv:2209.05229). |
| {cite:t}`Subramani2025` | Acenaphthylene$^+$ and biphenylene$^+$ in DESIREE: $k_c = 1230$ and 1050 s$^{-1}$. Master equation with the inverse-Laplace $k$, an anharmonic Beyer–Swinehart density of states, the same harmonic cascade as `cooling`, and RF as $A_j\rho(E-E_j)/\rho(E)$. RF states from AIMD (Tables 2–3). Their inverse-Laplace refit of the West+18 ACY$^+$ H loss: $E_a = 4.68$ eV, $A = 6.1\times10^{15}$ s$^{-1}$. $E_c$ (RF = dissociation) ≈ 5.8, 7.6, 10.6 eV for C$_{10}$, C$_{12}$, C$_{18}$–C$_{20}$. | RF benchmark and a cross-check of the M2 West gate. The $E_c$ size trend is a sanity target for coronene. |
| {cite:t}`Lee2023` | Naphthalene$^+$ and azulene$^+$ in DESIREE. Isomerisation and C$_2$H$_2$-loss barriers (CCSD(T)//ωB97X-D: 3.72–4.75 eV), branching H : C$_2$H$_2$ ≈ 1 : 1 at 5.1 eV, RF states ($f$ up to 0.094). Their model overestimates $k_c$ by 1.7–2× and underestimates absolute rates by ~10×. | RF benchmark and C$_2$H$_2$ pathway structure. A realistic error budget for a state-of-the-art model. |
| {cite:t}`MurgaWiebe2022` | $k(E)$ by RRK, inverse-Laplace and two Arrhenius parameter sets (Table 1), pushed through multiphoton heating. For C$_{66}$H$_{20}$ the H- and C-loss rates spread by ~6 dex; lifetimes spread by up to ~20 dex at $N_{\rm C} \approx 100$. The minimum surviving size is 45–100 C depending on the set. | **M5 target.** Reproduce their Figs. 1–2 from the Table 1 parameters, then place the gated microcanonical rates inside that envelope. It measures the same rate-form dependence as the M2 finding. |
| {cite:t}`Stockett2025` | An anharmonic IR cascade (Morse levels, $\nu^3$ correction, overtones and combination bands) for 2-cyanoindene. Overtones and combinations carry ~1/3 of the photons and ~1/2 of the power near threshold. | Method for an optional anharmonic `cooling` mode; an estimate that the harmonic cascade may underestimate cooling by up to ~2× at UV energies. |
| {cite:t}`Allamandola1989` | The origin of the exact microcanonical emission rate used in `cooling`, and the demonstration that the thermal approximation fails near thresholds. | Cite for `cooling`. |
| {cite:t}`Sehring2025`; {cite:t}`DouglasWalker2025` | PM6 quartic force fields for neutral PAHs up to C$_{30}$; 2-cyanoindene$^+$ spectroscopy. | Cite only (a cheap route to anharmonic constants); not relevant. |

The DESIREE papers agree that equilibrium-geometry TD-DFT underestimates RF
oscillator strengths by 10–100×. Vibronic (Herzberg–Teller) coupling must be
included, and $f$ may depend on $E$. So the T3 request in
{doc}`pahchem-data-requests` asks for vibronic $f$, and pahchem treats $f$ as
an uncertain parameter with a confidence grade.

None of these closes a gap in the process table. The H addition and
abstraction barriers per site and charge, super-hydrogenated H loss, C$_2$H$_2$
loss in compact PAHs, and IP/EA along the hydrogenation ladders still come from
{doc}`pahchem-data-requests`.

## Known legacy issues

These stay in the frozen `models/PAH_photophysics/` code and are not fixed
there; the new engine must not repeat them.

| Issue | Where |
|---|---|
| LD01 $Q_{\rm abs}$ at the disc radius | callers of `pah_temperature.get_absorption_cross_section` with `afromNc` |
| Tielens E–T table scaled by $N_{\rm C}$ instead of $3N-6$ | `pah_temperature.py:378`, `diagnose_temperature_distribution.py` |
| Method B overcounts without a sink | `pah_dissociation.py` |
| Anion "associative detachment" keeps $Z = -1$ | `pah_network_solver.py` |
| $s_e$ and $a_{\rm WR}$ fitted to the target figures | `pah_charge_utils.py` |
| Andrews Table B used as network rates | `compare_nh_rrkm_vs_andrews_poly.py`, `compare_rrkm_rates.py` |

These were fixed in the production path:
- the $f_2$ term of the four-state PEH charge balance (`four_state_charge_fractions`);
- the all-NaN photolysis table for $N_{\rm C} = 418$ (see {doc}`pah`).

The `ramses_source/` reader `init_pah_dissociation_tables` reads the grid size
from line 1 and so cannot read the exporter's 6-line header. The current
RAMSES-CALIMA source skips those 6 lines, so only this copy is stale.
