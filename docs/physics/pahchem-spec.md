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
