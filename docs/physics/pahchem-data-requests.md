(physics-pahchem-data-requests)=
# PAHchem: quantum-chemistry data requests

This page is the list of molecular data that `pycalima.pahchem` needs and
cannot get from the literature, in the form we would send to collaborators.
{doc}`pahchem-spec` gives the reason for each item. It is tiered by the
capability required:

- **T1:** spectra-style DFT (minima, frequencies, intensities, energies). The
  Rigopoulou group has done this up to $N_{\rm C} \approx 450$
  {cite:p}`Rigopoulou2021,Kerkeni2022`.
- **T2:** reaction paths and transition states, from the chemists working with
  them.
- **T3:** excited states. This is a new capability for the group.

T1 and the T2 validation item can be asked for now.

**Priority, from the M5 sensitivity ranking** ({doc}`pahchem-spec`). The
observables are C$_{54}$H$_{18}$ dehydrogenation and charge at the NGC 7023
edge and in the Orion Bar.

1. **Per-state vibrational modes along the hydrogenation ladder** (T1.1):
   ~1 dex on dehydrogenation in the Orion Bar.
2. **UV absorption cross sections per charge and hydrogenation state** (T3):
   ~1 dex on dehydrogenation, and the largest effect on charge.
3. **H- and H$_2$-loss barriers of the even, odd and dehydrogenated states**
   (T2.1–T2.2): 0.5–1 dex per −0.5 eV.
4. **Electronic states for recurrent fluorescence** (T3), together with
   anharmonic cooling: ~0.8 dex if cooling is 3× faster.
5. Below that: H-addition barriers (≤ 0.4 dex). Super-H and Eley–Rideal
   energetics do not move these observables; they matter only for the super-H
   fraction and for H$_2$ formation.

Outside DFT, the electron-capture (recombination) rates of large cations move
the charge by up to 0.5 in $f_+$, and no measurement exists above C$_{16}$.

## Common specification for every deliverable

**Output.** The raw output (Gaussian `.log` and `.fchk`, or the ORCA
equivalent) plus one summary record per stationary point. The record holds:
- formula, charge, spin multiplicity and $\langle S^2\rangle$;
- functional, basis and code version;
- the optimised geometry;
- the electronic energy and ZPE;
- **unscaled** harmonic frequencies, with the scaling factor they recommend;
- IR intensities (km mol$^{-1}$);
- the isomer or site label (which carbon carries the extra H, or which H is
  missing).

For transition states, the record also holds:
- the imaginary frequency;
- the reactant and product it connects, verified by IRC.

**Level of theory.**
- **Minima:** B3LYP/6-31G(d,p), or 6-31G*, the PAHdb v4.00 level
  {cite:p}`Ricca2025`, so that every species overlapping PAHdb is a direct
  cross-check. Not 4-31G.
- **EAs and anions:** add diffuse functions, 6-31+G(d).
- **Barriers:** a dispersion-corrected or range-separated functional (M06-2X or
  ωB97X-D)/6-311G(d,p), with DLPNO-CCSD(T) single points on a subset.

**Spin.** Use unrestricted references for open-shell species. For
dehydrogenated species, report the two lowest multiplicities.

**Species.**

| Species | Role |
|---|---|
| Pyrene, C$_{16}$H$_{10}$ | protocol validation against iPEPICO {cite:p}`West2018` and DFT {cite:p}`Trinquier2017` |
| Coronene, C$_{24}$H$_{12}$ | Andrews and Montillaud benchmarks; laboratory data |
| Circumcoronene, C$_{54}$H$_{18}$ | Andrews and Montillaud benchmarks |
| Circumcircumcoronene, C$_{96}$H$_{24}$ | Andrews benchmark; no UV data exist |

## T1: minima ladders (ask now)

1. **States.** Each parent at $Z = -1, 0, +1, +2$, with $N_{\rm H}$ from
   $N_{\rm H0}-4$ to $N_{\rm H0}+6$. For super- and dehydrogenated states, give
   the 2–3 lowest-energy site isomers. These states are the whole hydrogenation
   ladder that the benchmark models assume shares the parent's modes.
2. **Outputs per state.** Frequencies, intensities, total energy and ZPE. They
   give the density of states for every state, the per-state IR spectra, and
   the reaction energies along the ladder.
3. **Ionisation and attachment energies.** Adiabatic and vertical IP$_1$–IP$_3$
   and EAs (ΔSCF) over the same ladder, calibrated against experimental IP$_1$
   for pyrene and coronene. The benchmark models extrapolate these with a
   constant 0.1 eV per missing H.
4. **Optional.** If the ML-MD route is available, temperature-dependent band
   shifts and widths for the three parents.

Count: about 4 charges × 11 hydrogenation states × 2–3 isomers × 3 parents,
plus pyrene. That is roughly 300 optimisations and frequency jobs, nearly all at
or below C$_{96}$.

## T2: reaction paths

**Validation first (ask now).** Pyrene$^+$ and coronene$^+$ H loss and H$_2$
loss, comparing $E_0$ with {cite:t}`West2018` and the pathways with
{cite:t}`Trinquier2017`. This fixes the protocol before it is scaled up.

**Then, in priority order.** This order follows the M5 sensitivity ranking.
An earlier draft put H addition first.

1. **H$_2$ loss and H roaming.** Tight transition states, following the
   protocol of {cite:t}`Castellanos2018`, for cations and neutrals, along the
   even and dehydrogenated ladder. A −0.5 eV change in this $E_0$ moves C$_{54}$
   dehydrogenation by up to 1 dex.
2. **Aromatic H loss along the ladder, even and odd (radical) states.**
   Relaxed C–H dissociation scans, **not** a TS search: the channel has no
   saddle point and needs variational TST. A −0.5 eV change moves
   dehydrogenation by 0.5–1 dex. The laboratory thresholds disagree by
   0.7–1.4 eV ({doc}`pahchem-spec`, M2).
3. **H addition.**
   - *What:* transition states for H addition per site class (outer-edge
     solo, duo and trio, and sites next to an sp$^3$ carbon), for
     $Z = 0, +1, -1$ and $n$ = 1–6 extra H, for the coronene family.
   - *With:* IRC, TS frequencies, and the imaginary frequency for a
     tunnelling correction.
   - *Why:* this moves dehydrogenation by ≤ 0.4 dex, but sets the super-H
     fraction and the H$_2$-formation route. The only systematic set is for
     coronene$^+$ {cite:p}`Cazaux2016`.
4. **H abstraction (Eley–Rideal), H + HPAH → H$_2$ + PAH, and sp$^3$ H loss
   from super-H states.**
   - Transition states for abstraction from sp$^3$ CH$_2$ groups and from
     aromatic C–H, per charge.
   - The sp$^3$ H loss is the reverse of item 3; the models assume
     1.4–1.55 eV.
   - Neither moves dehydrogenation or charge. Both matter for H$_2$
     formation on PAHs. The current models use one cross section measured on
     coronene films {cite:p}`Mennella2012`.

## T3: excited states (later)

- **TD-DFT for recurrent fluorescence.** The lowest ~10 doublet states and their
  oscillator strengths for the cations, at CAM-B3LYP or ωB97X-D/6-31+G(d) on the
  D$_0$ geometry. Validate first on anthracene$^+$ and tetracene$^+$, whose
  rates were measured in storage rings {cite:p}`Martin2013,Bernard2023`.
- **Vibronic oscillator strengths.** Equilibrium-geometry $f$ is not enough:
  storage-ring studies find RF needs vibronically enhanced, effective $f$
  10–100× larger. Examples are a Franck–Condon–Herzberg–Teller $f = 0.011$
  against $10^{-4}$ without it for 1-cyanonaphthalene$^+$, and AIMD-averaged
  $f$ 30–65× the TD-DFT value for biphenylene$^+$
  {cite:p}`NavarroNavarrete2023,Subramani2025`. Ask for Herzberg–Teller $f$
  (Gaussian `freq=(FC,HT)` on the D$_0$/D$_n$ pair), or $f$ averaged along MD
  at 2–10 eV of vibrational energy, for at least coronene$^+$. For degenerate
  (Jahn–Teller) D$_0$ states, report the degeneracies as well.
- **UV absorption.** $\sigma(E)$ up to 13.6 eV for species not covered by
  {cite:t}`Malloci2007`: C$_{96}$H$_{24}$ and the hydrogenation ladders.

## Not requested

- **Photoionisation yields.** These are set by super-excited-state dynamics that
  DFT does not give. Use the measured yields and parametrisations
  {cite:p}`Jochims1996,Zhen2016,Wenzel2020`, with the DFT IPs as thresholds.
- **Electron recombination rate coefficients.** Capture theory and experiment
  give these; DFT is needed only for the products, via the T1/T2 energies.
- **C$_2$H$_2$ and C$_2$ skeleton loss.** These are multi-step and
  multi-configurational. Keep them parametric, with the constraint that compact
  PAHs strip their H first {cite:p}`West2019`.

## Asked of others

- **Joblin/Montillaud:** the mode sets (and whether they were scaled) and the
  Monte Carlo $k_{\rm IR}(E)$, $e_{\rm IR}(E)$ used in {cite:t}`Montillaud2013`.
  With their published $A$ and $E_0$ and PAHdb modes we get $E_{\rm th}$ =
  19.5 eV for C$_{54}$H$_{18}$ against their 17 eV, and their Fig. 10 needs
  1.2 dex less $G_0$ than ours. These two inputs are the only unpublished ones
  left in that comparison ({doc}`pahchem-spec`, M5).
- **Mayer/West:** the iPEPICO $k(E)$ curves behind {cite:t}`West2018`, as
  tables. (The ESI spreadsheet gives them for 7 of 25 reactions consistently
  with Table 1; see {doc}`pahchem-spec`, M2.)
- **Candian/Tielens:** for {cite:t}`Andrews2016`:
  - how $G_n(T)$ is normalised;
  - which temperature range enters $k_{\rm IR}$ in their eq. 8;
  - the neutral electron sticking coefficients they computed;
  - whether ionising photons are included in $G(T,E)$.

  Their Table B is reproduced only if they are, and their Fig. 7
  hydrogenation needs a $k_{\rm IR}$ in eq. 8 ~$10^3$ smaller than the
  cooling-path value ({doc}`pahchem-spec`, M5).
