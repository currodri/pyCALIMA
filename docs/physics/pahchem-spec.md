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
  the Tielens E–T table to 2–7 %, which confirms that the table is energy per
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
- **Andrews Fig. 5** (H and H$_2$ loss, C$_{24}$, C$_{54}$, C$_{96}$). The
  median $|\Delta| \le 0.22$ dex and the maximum is 0.51 dex, at more than
  1.5 eV above threshold. C$_{96}$ sits uniformly ~0.2 dex high. The values
  agree with the legacy `pah_mol_data` to $10^{-6}$ dex.

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

Still open:
- recurrent fluorescence, which needs TD-DFT inputs (T3) and a storage-ring
  benchmark;
- the ionisation-branching gate against Zhen+2016 and Wenzel+2020, which needs
  their tabulated yields.

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
- a physical electron sticking coefficient for neutrals (it is 1 by default;
  the legacy fits to the target figures must not return);
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

(Offsets are median log $G_0$, ours minus theirs, at the same $n_{\rm H}$.)

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
- **First-principles option.** Compute $s_e^0 = \langle k_{\rm IR}/(k_{\rm IR}
  + k_{\rm autodetach})\rangle$ with `energy_me` at $E = {\rm EA} + \epsilon$,
  and calibrate on the anthracene (~$10^{-9}$) and C$_{60}$ (~$10^{-6}$)
  attachment rates. That needs the T1 electron affinities.

**M6, RAMSES tables** ({mod}`pycalima.pahchem.export_pah_tables`, export
stage `pahchem_dissociation_tables`, solver key
`models.photolysis_model = "pahchem"`).

Physics. The table holds the carbon-loss rate per molecule,
$k = \sum_Z f(Z, N_{\rm H} = 0)\, k_{\rm C_2}(Z)$:
- $f$ is the steady state of the baseline $(Z, N_{\rm H})$ network.
- $k_{\rm C_2}$ is RRKM C$_2$ loss from the bare skeleton, with the cascade
  sinks:
  - $E_0 = 4.6$ eV and $\Delta S^\ddagger = 10$ cal K$^{-1}$ mol$^{-1}$
    (Micelotta+10b; the Murga & Wiebe "Arrh. 2" set), confidence **D**;
  - the bare skeleton's own PAHdb modes (C$_{24}$, C$_{54}$, C$_{96}$),
    otherwise the stripped parent.
- C$_2$ loss is taken from the bare skeleton only: compact PAHs lose their
  H before any carbon (Zhen+14). Carbon loss is not fed back into $f$.

Assumptions:
- the field has the Draine 1978 shape, scaled by $G_0$ (Habing);
- $n({\rm H}) = n_{\rm H}$, i.e. the gas is atomic;
- each PAH bin maps to the nearest parent in $\log N_{\rm C}$, with
  $N_{\rm C} = 418\,(a_0/10\,\text{Å})^3$;
- bins outside $16 \le N_{\rm C} \le 150$, i.e. the cluster bins, are
  skipped.

Spot check at $G_0 = 10^4$, $n_{\rm H} = 100$: $f_{\rm bare} = 0.99$ and
$k = 10^{-7.2}$ s$^{-1}$. A companion `pahchem_state_<bin>.npz` stores, on
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
- **Gate not yet run.** The planned gate is an end-to-end check: the
  `calima-run` rates with `photolysis_model = "pahchem"` should match the
  engine, followed by a RAMSES single-cell test.

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
- **Blocked:** the II Tables 2–3 gates need the Set 2 field (I Fig. 5,
  raster). With Set 1/1.6 the small PAHs already agree to 0.2 dex.

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
  - Our ionisation rate is 1.2–1.7× high with the substitute Li & Draine
    $\sigma$, which makes the cation fraction 0.02–0.13 too high (strict
    xfail, pending LP01 Fig. 11).
- **Hydrogenation (LP03 Fig. 9, card readings ±0.03):**
  - C$_{50}$H$_{20}^+$ is 0.64 against ~0.6.
  - C$_{80}$H$_{24}^+$ and C$_{80}$H$_{25}^+$ are 0.59/0.26 against
    0.6/0.27.
  - C$_{32}$H$_{14}^+$ and C$_{32}$H$_{14}$ are 0.55/0.30 against 0.45/0.37.
  - With benzene recombination instead, C$_{80}$H$_{25}^+$ drops to 0.08, so
    Fig. 9 used the naphthalene value.
- **LP01 Table 1:** the two-frequency column is reproduced within 0.25 dex by
  the neutral PAHH channel. The RRK column needs $E_0 \approx 0.7$ eV, not
  the stated value.
- **Not done:** LP09 (H$_2$ formation).

What the remaining targets need:
- **Raster figures the user is digitising:**
  - for Allain: I Fig. 5 and II Figs. 7, 9 and 10;
  - for Le Page: LP01 Figs. 10–11 and LP03 Figs. 6, 7 and 9.
- **Murga:**
  - its figures are vector and can be extracted here;
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
