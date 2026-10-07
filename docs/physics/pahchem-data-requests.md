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
- **Excited states of dehydrogenated cations.** The same lowest ~10 doublet
  (or singlet/triplet) states and $f$ for coronene$^+$ with 2, 4, 6 and 12 H
  removed. In the model the Extended Red Emission hinges on them: coronene$^+$
  at 1.70 eV emits at 730 nm, but the family is almost fully dehydrogenated
  where ERE is seen ({doc}`pahchem-spec`, ERE test).
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

- **Electron recombination of large PAH cations (experimental groups).** This
  is the input that now decides the PDR charge balance ({doc}`pahchem-spec`,
  "Charge inputs").
  - The data: measured PAH⁺ + e⁻ coefficients stop at pyrene (N_C = 16).
    There they are 0.03–0.37 of the BT94 capture theory, and the ratio rises
    as N_C^1.15 (`chemistry.lab_recombination_factor`).
  - The constraint: the PDRs4All population fits fall inside the stated
    conditions only if capture by N_C ≈ 100 cations is ≈ 3× BT94. The lab
    power law gives 1.5× there; extrapolated, it is a guess.
  - The ask: one or two coefficients for a compact cation with N_C ≥ 24, at
    100–1000 K. Coronene⁺ is the minimum, circumcoronene⁺ the ideal.
  - Who could measure it: FALP or CRESU flow tubes (Biennier and Rowe,
    Rennes, who measured the small PAHs); cryogenic storage rings with merged
    electron beams (DESIREE Stockholm, CSR Heidelberg); ion traps (PIRENEA
    Toulouse).
  - Also the dication (Z = +2 → +1) coefficient for one of them, which pahchem
    now sets with the same factor.
- **Temperature-dependent anharmonic spectra (Mai, Z. Wang et al. 2025,
  MNRAS 541, 3073; Z. Wang is in the Rigopoulou group's circle).** Their
  ML-MD neutral spectra already constrain the thermal band shifts and widths
  ({doc}`pahchem-spec`, "Temperature-dependent band profiles"). What would
  make them usable directly:
  - the models or spectra for **cations and dications**;
  - runs at 800–1500 K, or NVE runs at fixed internal energy (the
    classical-temperature mapping is the largest uncertainty);
  - several trajectories per molecule, with rotation removed;
  - unnormalised cross-sections or the raw dipole autocorrelations, so the
    quantum correction can be applied here (D = 1 in the release);
  - per-mode shifts and widths;
  - a retrain at 6-31G* on PAHdb v4.00 geometries;
  - whether derived per-band tables may be redistributed (the data are
    CC BY-NC-SA).
- **CH-stretch intensities of large PAHs (Rigopoulou group, T1 item 3, or the
  laboratory).** Against 35 matrix-isolation neutrals (PAHdb experimental
  v3.10, N_C ≤ 50), B3LYP/6-31G* puts 2× too much intensity in the CH stretch
  relative to the CH out-of-plane bands. pahchem applies the lab ratio as a
  stated input (`population.LAB_STRETCH_SCALE`).
  - The ask: whether this holds at N_C ~ 100, and for cations (3 lab cations
    only). Anharmonic (VPT2) or ML-MD spectra of two or three compact PAHs
    with N_C ≥ 54 would show it.
  - Why it matters: 3.3/11.2 drives the inferred size.

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
- **Murga/Wiebe:** for {cite:t}`Murga2020` and {cite:t}`Murga2022` (SHIVA);
  each item is a strict xfail or an inferred choice in {doc}`pahchem-spec`, M5:
  - the O6-star EUV added above 13.6 eV in the H II region of Murga+2022
    (Sect. 3.2): which spectrum, and how it is normalised. Their H II charges
    are reproduced with a 40 000 K Kurucz EUV matched at 6–13.6 eV and scaled
    ×4; the factor of 4 is our inference;
  - which $p(E)$ reading SHIVA uses, and whether it changed between the
    papers: Murga+2020 Fig. 4 needs the time-averaged one, the Murga+2022
    Figshare chains the post-absorption one;
  - whether the H, H$_2$ and C$_2$H$_2$ channels compete (as in Murga+2019
    eq. B2) or are independent;
  - how the super-H threshold is applied when $N_{\rm H}$ is real-valued
    (Murga+2020 Fig. 4 needs a super-H contribution $10^2$–$10^4$× below
    eq. 6 as printed);
  - the H-addition and super-H loss rates of Murga+2022, under which PAHs
    $\ge$ 6 Å stay normally hydrogenated (with the Murga+2020 rates they end
    60–90 % super-hydrogenated);
  - the charge settings: the electron sticking coefficient, the
    $Z_{\rm min}$ rounding, C$^+$ collisions, and the $\chi$ field shape.
    The Murga+2022 files imply $s_e = 1$, eq. 24 truncated and no ions,
    but Murga+2020 Fig. 3 needs the floor $Z_{\rm min}$;
  - the $T(E)$ relation of Murga & Wiebe 2022, which behaves like Tielens'
    rather than the DL01 relation of Murga+2020/2022.
- **Rigopoulou group:** for {cite:t}`Rigopoulou2021` (method card
  `rigopoulou2021.toml`). Their Fig. 1 is reproduced exactly with the thermal
  cascade and FWHM 15 cm$^{-1}$; the binned diagnostics are not. The
  questions:
  - the spectra of the 17 molecules computed independently, and whether they
    were scaled;
  - the code and the band FWHM actually used: 15 cm$^{-1}$ fits Fig. 1,
    against the 30 cm$^{-1}$ stated;
  - the bin edges and how the neutral–cation mixtures of Figs. 8–11 were
    built;
  - how "ISRF" and "$10^3\times$ISRF" were turned into photon energies;
  - the CASSIS fit constraints behind 11.3/7.7: our window ratios for the
    small bin are 2–4× theirs;
  - the Table A1 UID corrections listed in the card.
- **Borja, Calvo, Parneix & Falvo** (ISMO Orsay, LIPhy Grenoble): for
  {cite:t}`Borja2026` (method card `borja2026.toml`). Their data and tools are
  "available from the corresponding author upon reasonable request".
  - **What we have done.** We fed their Table I inputs into the standard RF
    rate, which uses the ground-state density of states and fixed $f$ with
    PAHdb cation modes. Against their Fig. 8:
    - allowed D2/D3 states: within 0.18 dex at 4–13 eV;
    - symmetry-forbidden D1 states: within −0.14 to +0.49 dex.
  - **What we would ask for**, for naphthalene$^+$, anthracene$^+$ and
    pyrene$^+$:
    1. the $k_{\rm RF}(E)$ of each state as tables: the Fig. 8 curves, and the
       total RF rate;
    2. the ωB97XD/cc-pVTZ frequencies of D0 and of each excited state
       (D1–D3), with the Duschinsky matrices $J$ and shifts $K$;
    3. the transition dipoles $d^0_e$ and their derivatives $d^1_e$ (the
       Herzberg–Teller terms), or $f_{\rm HT}(T)$ on a temperature grid
       rather than at 0 and 2000 K;
    4. the emission spectra of Fig. 9 as tables.
  - **Two questions:**
    - whether "6.41 eV" (the Np$^+$ D1/D2 crossover in the text) or the
      6.09 eV of the revised Fig. 8 is the published value;
    - whether the coth($\beta\hbar\omega$) in their eqs. 28 and 30 should be
      coth($\beta\hbar\omega/2$), the standard thermal result their App. E
      uses.
  - **What it would let us test.** With 1–2 we could check our
    approximations one at a time against the same molecules:
    - the excited state's own density of states;
    - the Duschinsky rotation;
    - the temperature dependence of $f_{\rm HT}$.

    This would tell us where the simple RF rate is good enough for coronene-
    to C$_{96}$-size cations, for which no such calculation exists.
  - **A further question, if they are open to it:** whether they plan the
    same treatment for larger cations (coronene$^+$ or bigger), the sizes our
    RAMSES tables use.
