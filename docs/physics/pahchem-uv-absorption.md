# PAH UV–visible absorption for pahchem: TD-DFT, Malloci+07 and experiment

Status on 2026-10-26, branch `feature/pahchem-anharmonic`. pahchem's default σ(E, Z) is now the Malloci average of
{cite:t}`Berne2022` (`photo.sigma_uv`: nine compact PAHs at 5–30 eV, {cite:t}`LiDraine2001` outside). That is a
placeholder until our own Octopus library replaces it. The library is charge-resolved real-time TD-DFT, checked
against {cite:t}`Malloci2007` and against laboratory data. **No band profile or position correction is a default
yet**; both are open decisions for the user (§4, §5).

Where things are:
- **Runs:**
  - megatron `~/pahspec`: coronene, pyrene and ovalene, and the convergence tests;
  - DIaL3 `/lustre/dirac3/scratch/dp016/dc-rodr4/pahspec`: the library, C54 to C216 at charges −1 to +2.
- **Status log:** `RAMSES_dev/pahspec/STATUS.md`.
- **Scripts:** `diagnostics/pahchem/octopus/`:
  - `make_run.py`, `make_dial3_library.py`, `queue.py` (making and queueing runs);
  - `compare_malloci.py`, `plot_progress.py` (megatron runs), `plot_dial3_library.py` (the library), `check_damping.py`.
- **Laboratory spectra:** `RAMSES_dev/pahspec/experiment/`.
  - `atlas/`: vapour spectra from the MPI-Mainz Spectral Atlas: pyrene, anthracene, naphthalene and others.
  - `vuv/`: synchrotron spectra of benzene and naphthalene.

## 1. Method

**Octopus settings** (real time, real space; Octopus 16.4):
- **Functional and pseudopotentials:** LDA (Perdew–Zunger) with the standard norm-conserving set.
- **Grid:** 0.25 Å spacing. This is converged: within 1 % of 0.20 Å, whereas Malloci's 0.30 Å is not, with these
  pseudopotentials.
- **Box:** every atom at least 4 Å from the edge, plus a 1 Å absorbing boundary.
- **Propagation:** a δ kick of 0.01 Å⁻¹ along each axis; AETRS, dt = 0.002 ħ/eV; T = 20 ħ/eV.
- **Open shells** are spin-polarised, with ExtraStates = 4 and their occupations fixed per spin channel
  (`make_run.py`).
  - PAHdb's triplet dications run as triplets.
  - C216H36⁺ runs with its hole shared 0.5/0.5 over a pair 0.8 meV apart; with whole electrons it never converged.

**The spectrum.** It is the sine transform of the dipole, d(t), times a window D(t):

$$\sigma(E) \propto E \int_0^T d(t)\, D(t) \sin(Et)\, dt .$$

The window sets the band profile:
- polynomial (Octopus's default): compact bands that ring, with negative lobes between bands at finite T;
- exp(−γt): Lorentzian bands of half-width γ;
- exp(−s²t²/2): Gaussian bands of standard deviation s.

A clean Gaussian needs exp(−s²T²/2) ≲ 10⁻², i.e. s ≥ 3.0/T. At T = 20 that means **s ≥ 0.15 eV (FWHM ≥ 0.36 eV)**;
T = 40 allows 0.076 eV, T = 80 0.038 eV.

**Agreement with Malloci+07 at 0.25 Å**: oscillator strength per band within 1–6 % above 5 eV for coronene, pyrene and
ovalene (all charges) and circumcoronene (C54H18, DIaL3), once their window is applied (§2).

**DIaL3 build.** It reproduces megatron's coronene run: ground-state energy and all 54 eigenvalues identical to the
printed digits, and the dipole equal to 4 × 10⁻¹³ over 500 steps.

## 2. Malloci's spectra carry an exponential window

{cite:t}`Malloci2007` give T = 20 ħ/eV and "a resolution ħ/T = 0.05 eV", but no window (Sect. 2 of
arXiv:0707.3045, [read]). `check_damping.py` recovers it, using the fact that the inverse transform of σ/E
gives back d(t)·D(t):
- **Method check:** recovering our own polynomial window this way reproduces it to ≤ 0.03.
- **Malloci's window:** exp(−γt) cut at t = 20. γ = 0.21 eV for neutral coronene and 0.148 eV for its cation;
  γ = 0.16 eV for circumcoronene neutral and anion, 0.21 eV for its cation.
  - That makes their bands Lorentzians of FWHM 0.3–0.4 eV.
  - It is very likely Octopus's own old default: `oct-sf` used exponential damping with SpecDampFactor = 0.15 until
    September 2005 (source-code history).
- **Forward test:** our runs with their window match Malloci.
  - Coronene⁺, exp(−0.15 t): every band from 0.5 to 30 eV within 3 %.
  - Circumcoronene: 1–5 eV goes from 0.82–0.83 to 0.95–0.97, and the 8–13 eV fine structure largely disappears.

**Consequence.** Malloci's absorption floor below a neutral's first band is the Lorentzian wing of their window, not
something the molecule absorbs. It carries about a quarter of the neutral's Mathis 1–5 eV rate. The cations' visible
bands are real.

## 3. What experiment provides

Literature surveys of 2026-10-24 and 2026-10-26. [read] means seen in full or in the stated tables; [read-data] means
measured by us on digitised data; [abs] means abstract only; [sec] means through a citing paper.

### 3a. Data sets

| Molecule, charge | Phase, temperature | Range | Absolute | Reference |
|---|---|---|---|---|
| pyrene, anthracene, naphthalene 0 | vapour, 296–873 K | 3.5–6.2 eV | yes; scales differ up to 2× between sources | Thöny & Rossi 1997, JPPA 104, 25; Ferguson+ 1957; Lyons & Morris 1960; Grosch+ 2015 (Spectral Atlas) |
| benzene, naphthalene 0 | gas, synchrotron | 5–36 eV | yes | Rennie+ 1998, Chem. Phys. 229, 107; Suto+ 1992, JQSRT 48, 79 |
| pyrene, anthracene, PAH mixtures 0 | gas, 570–630 K | 1.8–15 eV | yes | Joblin, Léger & Martin 1992, ApJ 393, L79 [read] |
| pyrene, coronene and others 0 | gas EELS, 0.6 eV resolution | 0–30 eV | relative | Keller, Coplan & Goruganthu 1992, ApJ 391, 872 [read] |
| pyrene 0 | jet | S₂ vibronic groups, widths | calibrated | Rouillé+ 2004; Gredel+ 2011, A&A 530, A26 [read] |
| coronene, ovalene, C30–C48 PAHs 0, ± | Ne matrix, 4 K | UV–visible–NIR, f for some | f only | Ehrenfreund+ 1992, A&A 259, 257; Ruiterkamp+ 2002, A&A 390, 1153 [read] |
| HBC 0, +1 | Ne/Ar matrix; beam | 1.5–6 eV | f (cation) | Rouillé+ 2009, JCP 131, 204311; Steglich+ 2011, ApJ 742, 2 [read] |
| coronene⁺ | gas, He-tagged, ~10 K; TPES | D₁, D₃; D₂ | no | Hardy, Rice & Maier 2017, ApJ 836, 37; Bréchignac+ 2014, JCP 141, 164325 [read] |

Matrix-to-gas shifts are ~0.25 % in energy for neutrals and ~0.5 % for ions (Ruiterkamp+02). No gas-phase spectra of
anions or dications were found.

### 3b. Band widths

| Case | Width | Source |
|---|---|---|
| single vibronic components, cold (jet, He-tagged, Ne) | 0.002–0.06 eV FWHM | Gredel+11; Hardy+17; Ehrenfreund+92; Rouillé+09 |
| whole band systems, cold C24–C48 in Ne (coronene α/p/β, ovalene p/β) | **σ = 0.07–0.12 eV** (second moment) | Ehrenfreund+92; Ruiterkamp+02 [read] |
| pyrene S₂, computed Franck–Condon (PBE0, 0 K) | σ = 0.093 eV | Benkyi+ 2019, PCCP 21, 21094 [read] |
| benzene, TDLDA plus Franck–Condon | FWHM 0.35 eV computed vs 0.29–0.42 measured | Bertsch, Schnell & Yabana 2001, JCP 115, 4051 [read] |
| small-PAH vapour, 3.8–5.9 eV bands | σ = 0.12–0.18 eV at 300–420 K; 0.15–0.22 at 770–870 K | Spectral Atlas [read-data] |
| thermal growth (N_C 10–16) | σ² rises by (1.7–3.1) × 10⁻⁵ eV² K⁻¹ | same |
| cation bands | similar or broader (×1–2); widths equal in Ne and Ar, so intrinsic | Steglich+11 [read] |
| matrices | negligible for UV bands; they broaden only the narrow S₁, S₂ and low D–D bands (4–5×) | Biennier+03; Rouillé+09 |
| lifetimes above 5 eV | 10–80 fs, i.e. Lorentzian FWHM 0.01–0.07 eV (theory 0.06–0.12 eV) | Marciniak+15, Hervé+21, Lee+21 [read] |

**Above 6 eV.**
- Benzene and naphthalene have real Rydberg and autoionisation lines.
- PAHs of C14 and up show only shoulders and plateaus at 0.15–0.6 eV resolution, and the structure decreases with
  size (Jochims+ 1996, A&A 314, 1003 [read]).
- **Above ~13 eV every measured spectrum is one smooth hump:** peak 17–18 eV, ~25 Mb per C, FWHM 12–15 eV, residual
  structure ≤ 3 %.
- The smoothness is congestion of states, not short lifetimes.

**The raw TD-DFT spikes at high energy are box artefacts.** Above the LDA ionisation threshold (−ε_HOMO), Octopus
puts the continuum in a finite box, which makes it discrete. Wenzel+ 2020 (A&A 641, A98, App. E [read]) and Zhen+ 2016
(ApJ 822, 113 [read]) say so, and Malloci+07 called them "spurious structures". Wenzel+20:
- judged Malloci's database, which uses the same 4 + 1 Å box as ours, converged to ~13 eV, "suitable for
  astronomical modeling purposes involving photons up to the Lyman limit";
- needed 8 Å spheres and a 0.18 Å grid for spectra up to ~20 eV, and averaged two box sizes to smooth the continuum.

The bump at 217.5 nm (FWHM ~1 eV) is a property of mixtures. It does not constrain single-molecule widths
(Joblin+92; Cecchi-Pestellini+08; Lin, Yang & Li 2023).

### 3c. How others broaden computed spectra

None derives a width.
- Malloci: the exponential window above.
- Steglich+ 2010, 2011: Lorentzians of 5000 and 3000 cm⁻¹, or a width growing with the number of lower transitions.
- Lin+23: 0.2 eV per transition.
- Dubosq+20 (TD-DFTB): 0.01–0.2 eV.
- Li & Draine 2001: Drude profiles fitted to laboratory mixtures and to extinction.

## 4. The band profile

The user's decision of 2026-10-24: long propagation, then an explicitly stated envelope, compared with experiment.
That envelope cannot be derived from first principles here: TD-DFT gives vertical excitations, not the vibronic
envelope. It is therefore an empirical choice, labelled as such and opt-in, and reported with its range (the
no-ad-hoc-factors rule).

**The calibration target is the spread of each band system**, not its sharp vibronic peaks. A vertical excitation
lies at the band's centroid (its first moment); its spread is the second moment. Absorption rates in broad fields
depend on these, not on the fine structure.

**Our pyrene calibration** (2026-10-26, `results/pahchem/pyrene_width_2026-10-26/`):
- **Data:** Octopus pyrene neutral against the vapour at 313, 423 and 873 K.
- **Single lines:** each of our bands is one state, so the Gaussian needed equals the laboratory spread:

| pyrene vapour | S₂ | S₃ | 5.3 eV band | mean s (FWHM) |
|---|---|---|---|---|
| 313 K (Ferguson 1957) | 0.150 | — | — | 0.15 (0.35) |
| 423 K (Thöny & Rossi 1997) | 0.162 | 0.148 | 0.178 | 0.16 (0.38) |
| 873 K (Thöny & Rossi 1997) | 0.185 | 0.145 | 0.199 | 0.18 (0.42) |

- **Edge sensitivity:** moving the band edges by ±0.05 eV changes s by ±0.02 eV.
- **Shape:** at 873 K a Gaussian reproduces each band's red edge (the share below the peak). At 313 K the S₂ origin is
  sharp: 1 % of the band lies below it, but a Gaussian puts 10 % there.

**Proposed profile** (not a default; the user decides):
1. **Below ~7.5 eV,** a Gaussian of s set by temperature:
   - cold molecules: s ≈ 0.08 eV (range 0.06–0.12), from the Ne-matrix band systems and the pyrene Franck–Condon
     value;
   - warm ones: 0.15–0.20 eV, from the vapours.
2. **From 7.5 to 14 eV,** s rising linearly to ~0.8 eV, then flat to 30 eV (range 0.6–1.0).
   - This removes the box artefacts to ≤ 3 % and widens the 17 eV hump by < 2 %.
   - The 8–13.6 eV part is only weakly constrained by data.
3. **No Lorentzian part,** except perhaps for anion resonances above the detachment threshold. Lifetime widths are far
   below s, and Lorentzian wings put absorption where the molecule is transparent.
4. **Construction:** a Gaussian window of the smallest s, then a convolution in energy with the remaining width at
   each energy. One window alone cannot give an energy-dependent width.

**What this implies for the runs.** The cold width at 2–5 eV is below what T = 20 can give cleanly (0.15 eV).
- **Test, 2026-10-27** (`results/pahchem/width_sensitivity_2026-10-27/compare_width.py`): coronene neutral at T = 40,
  production settings, cold (s = 0.08 eV) against warm (0.15 eV) widths.
  - Every rate changes by ≤ 2.4 %: absorption at 0.1–5 and 5–13.6 eV, and above the ionisation energy. The fields:
    the Mathis ISRF, Draine 1978, a 10 000 K Kurucz photosphere and a 6000 K blackbody.
  - The mean absorbed photon energy moves by ≤ 0.02 eV.
  - At equal s, T = 40 and T = 20 agree to 0.75 %.
- **So T = 20 with the warm width suffices for pahchem's rates;** the cold width is a ≤ 2.4 % systematic.
- **Caveat:** this is one neutral. The ions' near-IR bands sit where the fields change fastest; untested.

**Guards, for any profile that ships:**
1. σ ≥ −10⁻³ max σ on the grid.
2. f to 30 eV unchanged to < 0.5 %, and the f of any band wider than a few s unchanged.
3. Below the lowest band of a neutral, σ falls at least as fast as the Gaussian. The diagnostic: the share of the
   Mathis 1–5 eV rate carried by energies more than 3 s below the lowest peak.
4. Every rate pahchem uses is recomputed at both ends of the width range:
   - absorption in 1–5 and 5–13.6 eV;
   - photoionisation and photodetachment;
   - the photoelectric efficiency ε.
5. The onset may not lie below the measured gas-phase origin by more than the position error (§5).

**Box test** (2026-10-26/27, `results/pahchem/octopus_box_2026-10-26/compare_box.py`, figure
`box_test_coronene.png`). Coronene neutral and anion at T = 20 in an 8 Å box (7 Å + 1 Å absorber); the inputs and
geometries are the megatron runs', and only the box changes.
- **Integrated quantities agree.**
  - Band integrals: 8 Å / 5 Å = 1.0015, 1.003 and 0.996 (neutral; 1–5, 5–13.6, 13.6–30 eV); 1.018, 1.004 and 0.999
    (anion).
  - Mathis-field rate at 1–5 eV: 0.9995 (neutral) and 1.024 (anion).
  - Mean absorbed photon energy: 5.91 → 5.91 eV and 4.17 → 4.14 eV.
  - The anion's 2–3 eV strength is +12 %, but it is small (f 0.060 → 0.068).
- **The shape is box-independent only at low energy.** The neutral's spectrum below 6.5 eV is identical in the two
  boxes, and so is the anion's below 5 eV.
- **Above ~7 eV the details move**, even with the profile above: smoothed features at 7.5–10 and 16–22 eV shift by
  10–15 % locally.
  - The anion's strong 5.5–6 eV band moves 0.1–0.2 eV lower and splits. An anion's LDA detachment threshold is low,
    so nearly its whole spectrum lies above it, where the box matters.
- **Conclusion:** the production box (4 + 1 Å) needs no rerun for anything pahchem integrates. The spectral shape
  above ~7 eV (above ~5 eV for anions) carries a ~15 % local box uncertainty. Removing it would need much wider
  smoothing there, or an average over two box sizes, as Wenzel+20 did.

**Does a two-box average help? Four-box check** (2026-10-27, `results/pahchem/octopus_box_2026-10-26/compare_box4.py`,
figure `box_test_four_boxes.png`). Coronene neutral and anion in 5, 6.5, 8 and 10 Å boxes, all cut at T = 10 with a
0.3 eV Gaussian window, then smoothed to the proposed s(E) above 9 eV. The 10 Å box is the reference.
- **Above 13.6 eV the shape converges with box size.** The 5 Å box is the outlier: 6.8 % rms from 10 Å, 14–15 %
  locally. The 8 Å box alone is within 2.5–3 %, and the 5 + 8 Å average within 3 %. The average adds nothing over the
  larger box.
- **At 9–13.6 eV no box converges.** The boxes differ by 4–12 % rms, and by up to 26 % locally, with no trend in box
  size. The planned average avg(5, 8) and the other average avg(6.5, 10) differ by 5.6 % (neutral) and 8.8 % (anion).
  That is as much as two single boxes differ, 6.5 % and 9.0 % on average.
- **Rates barely move.** Absorption over 7–13.6 eV, 5 Å against 10 Å:
  - neutral: 1.001 (Mathis), 1.007 (Draine 1978), 1.023 (Kurucz 10 000 K);
  - anion: 1.016, 1.020, 1.045.
  - The split between 7–9 and 9–13.6 eV moves more, but in opposite directions: +5 to +12 % below 9 eV, −1 to −12 %
    above.
- **Conclusion for neutral gas:** a second 8 Å box would converge only the shape above 13.6 eV and would not fix
  9–13.6 eV. The box systematic on pahchem's rates is ≤ 2 % for ISRF-like fields and ≤ 5 % for a 10 000 K star.

**Ionising fields (H II regions)** (2026-10-27, `results/pahchem/octopus_box_2026-10-26/compare_box_euv.py`, output
in `compare_box_euv.log`). Coronene 0/−1 to 150 eV from the same multipoles, at T = 5.
- **Most of the absorption is above 30 eV, where the library plots stop.**
  - Of coronene's 108 valence electrons, f = 68.6 lies below 30 eV, 92.4 below 54.4 eV, 103.3 below 100 eV and
    106.8 below 150 eV.
  - The dipoles are written every step (dt = 0.002 ħ/eV), so extending the spectra needs no rerun: only the energy
    grid of the post-processing changes.
- **The 0.25 Å grid holds to 100 eV:** within 1.7 % (30–54.4 eV) and 2.5 % (54.4–100 eV) rms of 0.2 Å. A 0.3 Å grid
  fails there, at 24–32 %.
- **The box barely changes rates in O-star fields.** Kurucz 30 000 and 40 000 K and a 40 000 K blackbody, 5 Å over
  10 Å:
  - 13.6–140 eV: 1.000–1.003;
  - 13.6–24.6 eV: within 0.1 %;
  - 24.6–54.4 eV: +2.3 to +5.1 %;
  - 54.4–140 eV: −3 to +1 %, or within 1 % if s widens to 4 eV by 40 eV.
  - The local shape still differs by 11–14 % rms above 30 eV at s = 0.8 eV, and by 1–3 % at s = 4 eV.
- **T = 5 would suffice for a second box above 12.5 eV:** at the same s(E), T = 5 and T = 10 agree to 0.4–1.3 % rms.
- **Against the atomic sum** (Verner phfit2 C 2s + 2p, hydrogenic H 1s), TD-LDA lies 30–70 % above it at 20–40 eV,
  where molecular structure persists. It meets it within 10–15 % from 80 eV. f(30–54.4 eV) is 24.0 against 17.9 for
  the atoms, f(54.4–100 eV) 11.2 against 10.4.
- **Conclusion:** the production box serves ionising fields too, to ≤ 5 % in any sub-band and 0.3 % over 13.6–140 eV,
  so no second box. What H II regions do need is the library extended above 30 eV. The C 1s edge (285 eV) lies
  outside the pseudopotential calculation. Untested: cations and dications, which are more compact than the anion
  and so presumably less box-sensitive.

**The library to 150 eV** (`diagnostics/pahchem/octopus/plot_dial3_library.py`, figure `library_euv.png`; 37 states
on 2026-10-27, raw T = 20 spectra):
- f to 150 eV is 98.8–98.9 % of the valence electrons in every state.
- Octopus over the atomic sum, per band:

  | 13.6–30 eV | 30–54.4 eV | 54.4–150 eV |
  |---|---|---|
  | 1.33–1.41 | 1.32–1.35 | 1.00 ± 0.01 |

  - Below ~50 eV the molecule absorbs a third more than its atoms.
  - Above 100 eV, TD-LDA falls 10–20 % below the atoms, likely because the pseudo-wavefunctions are smooth near the
    nuclei, where fast photoelectrons are made.
- **X-rays:** the atomic sum serves, and pahchem already has it. Li & Draine 2001 (graphite above 30 eV) matches the
  atomic sum to 3–10 % from 40 eV to 1 keV, and it includes the C K edge (graphite's π* onset at 285 eV).
  - A splice of the library onto it belongs at 54–100 eV, where the band integrals agree.
- **pahchem's default steps down at 30 eV:** 7.05 → 4.85 Mb per C, where the Malloci average hands over to Li &
  Draine. The library has no step there; Li & Draine lies 35 % below it at 30 eV and 18 % at 40 eV. For ionising
  fields the hand-over must move to ~54 eV, or use the library. The user's decision (2026-10-27): Li & Draine will be
  dropped; the library replaces it.

**The assembled library, 0.02 eV to 10 keV** (2026-10-27, `diagnostics/pahchem/octopus/build_library.py`; output in
`results/pahchem/octopus_library`, figures `final_*.png`; the 39 states finished by then).

The stages:
1. a Gaussian window of s0 = 0.15 eV at T = 20;
2. widening to s(E): 0.15 eV to 7.5 eV, 0.8 eV at 14–30 eV, 4 eV above 40 eV.
   - It is a conservative diffusion in energy, which keeps f exactly and σ ≥ 0.
   - A convolution whose width follows the source energy also drifts strength down the width gradient. That drift
     piled strength into the grid point where the widening starts, and moved up to 5 % of f below 13.6 eV.
3. positions stay TD-LDA's (§5);
4. above 80 eV, the atoms take over:
   - C 2s + 2p and H 1s, scaled by k: TD-LDA's integral over theirs at 80–130 eV;
   - blended in with a cos² weight over the same range;
   - plus C 1s, unscaled, from 291 eV.

**Measured benzene supports the hand-over** (Feng+ dipole (e,e), absolute, f = 29.3 of 30 valence electrons to 200
eV):

| band | benzene / atoms | library / atoms |
|---|---|---|
| 13.6–30 eV | 1.50 | 1.33–1.41 |
| 30–54.4 eV | 1.30 | 1.32–1.35 |
| 54.4–100 eV | 0.96 | — |
| 100–200 eV | 0.78–0.79, flat | k = 0.885–0.891 |

- The molecular excess at 30–54 eV is real.
- Above ~80 eV both lie a constant factor below the atomic sum. So the atomic shape is right there, and its
  normalisation is the molecule's.
- The unscaled atoms would put 100–290 eV 13 % above TD-LDA and 27 % above benzene.
- Two differences from benzene: the library is 10–25 % above it at 40–200 eV, and it keeps bumps at ~40 and ~55 eV,
  in every state, that benzene does not show.

**Checks, all 39 states:**
- f to 30 eV is unchanged to ≤ 0.17 %.
- σ ≥ −3 × 10⁻⁴ of the maximum: the T = 20 ringing below the first band of neutrals, at the 10⁻³ Mb level.
- f to 130 eV is 97.8 % of the valence electrons. To 10 keV it is 97.2–97.4 % of valence plus 2 per C (the K shell
  absorbs < 2).
- Against Octopus's own spectrum:
  - f below 13.6 eV: −0.1 to +3.9 %. This is the 0.75 eV width smoothing the steep rise into the 17 eV hump; it lies
    within the 9–13.6 eV box uncertainty.
  - rates: Mathis within ±0.9 %; Draine 1978 −0.6 to +3.3 %; Kurucz 40 000 K −2.2 to +0.3 %.

The C K edge is the atomic step: no π* resonance at 285 eV (open).

## 5. Band positions

**The errors, against vapour centroids** (vertical energies are compared with centroids; peaks sit 0.08–0.13 eV
lower):

| | Octopus TD-LDA | measured | error |
|---|---|---|---|
| pyrene S₂ (La) | 3.38 eV | 3.98 (vapour, 423 K) | −0.60 |
| pyrene S₃ | 4.15 | 4.81 | −0.66 |
| pyrene 5.3 eV band (Bb) | 5.02 | 5.48 | −0.45 |
| coronene⁺ D₁ | 1.30 | 1.314 (gas) | −0.01 |
| coronene⁺ D₂ | 1.75 | 1.80 ± 0.05 (TPES), 1.823 (Ne) | −0.05 to −0.07 |
| coronene⁺ D₃ | 2.85 | 2.713 (gas) | +0.14 |

The ordering of the coronene⁺ band strengths (ours: D₂ > D₃) disagrees with the Ne-matrix values (D₃ > D₂). Those
values are uncertain by ×2.

**The literature agrees** (2026-10-26 survey; signed error = calculated − measured):

| Method | La | Lb | Bb/β | Cation D_n ← D₀ | Source |
|---|---|---|---|---|---|
| RT-TD-LDA (Octopus) | −0.64 (acenes; grows with size) | −0.08 | — | — | Peng & Chai 2014, PCCP 16, 21564 [read] |
| BLYP | −0.56 (acenes) | — | −0.33 (−0.16 … −0.47) | mean −0.05 (−0.47 … +0.31) | Malloci+ 2007, Chem. Phys. 340, 43 [read] |
| B3LYP | −0.28 to −0.42 | +0.18 | −0.06 (±0.1) | mean +0.11 | Malloci+07; Wong & Hsieh 2010 [read] |
| CAM-B3LYP / LC-ωPBE | −0.10 / +0.04 | +0.36 | overshoots +0.3 … +0.4 | worse for small cations | Wong & Hsieh 2010; Weidlich & Dreuw 2025 [read] |
| GGA, PAH radical ions (51) | — | — | — | within ~0.3, size-independent | Hirata+ 2003 [sec] |

- **Neutrals below ~6 eV are too low**, in La (−0.5 to −0.7 eV) and also in the strong Bb bands (−0.2 to −0.5 eV).
- **Ions are within ~±0.3 eV with no systematic sign.**
- **Above ~6–7 eV** TD-LDA agrees with gas-phase data, including the 17 eV hump.
- **No single functional** gets La, Lb and Bb all within 0.2 eV.
- **No astrophysical model shifted its TD-LDA spectra.** Mulas+ 2006 (A&A 460, 93) measured the effect on Mathis-field
  absorption rates: ratios of 0.93–1.45. The mean absorbed photon energy came out up to ~40 % low.

**Effect on our library** (`plot_dial3_library.py`, figure `library_isrf.png`):
- **Mathis-field mean absorbed photon energy:** 2.0–4.8 eV for our states, against 4.6–5.8 eV for pahchem's default,
  Li & Draine and the AmesPAHdb suite. The suite is Maragkoudakis+20's cross section; it exceeds the f-sum rule (5.2
  per C to 30 eV), so only its shape is meaningful.
- **Why:** the difference comes from real low-energy bands, cation near-IR bands and the visible bands of large
  neutrals.
- **The LDA shift is not the cause:** shifting our whole spectrum up by 0.5 eV raises the mean by only 0.8–0.9 eV.
- **The error lands mainly on the neutrals**, while most low-energy absorbed photons go to the ions.

**Fix options**, ranked for positions at 1–6 eV, C24–C216, all charges:
1. **Simplified TD-DFT** (sTD-DFT / XsTD-DFT, open-source `std2`) on B3LYP and range-separated-hybrid orbitals, for
   every species, spliced onto the RT-LDA spectra below ~6 eV.
   - Reported for 500–1000 atoms and thousands of states; the cost is one hybrid ground state per species.
   - Needs ORCA (or another code that writes molden files) and `std2`. Open-shell support is unverified.
2. **Full linear-response TD-DFT on a calibration subset** (C24–C54, all charges): B3LYP, CAM-B3LYP or ωB97X-D, and
   tuned LC-ωPBE. It would validate option 1 and give state-dependent shifts.
   - Davidson solvers cannot reach the ~10³ states below 6 eV of C96 or larger.
3. **Anchors for a few molecules:** DFT/MRCI (validated to C96), IP-ADC(3) for cations, or GW-BSE.
4. **LC-TD-DFTB:** cheap, but its accuracy for large compact PAHs and ions is not established.
5. **Octopus with a hybrid** (`TheoryLevel = generalized_kohn_sham`, ACE on by default since 16.0) as a like-for-like
   check on one molecule.
   - Range-separated hybrids need the FFT Poisson solver without a cutoff, i.e. periodic images.
   - The estimated cost is 10²–10⁴× LDA.
6. **Not recommended:** a uniform scissor shift (`TDScissor`). It would move Lb and the cation bands the wrong way and
   is an ad hoc factor.

**First tests of option 1** (2026-10-27, PySCF def2-SVP + std2; `results/pahchem/stda_coronene_2026-10-27`,
`results/pahchem/stda_splice_2026-10-27`):

| | TD-LDA | B3LYP sTD-DFT | ωB97X-D3 XsTD-DFT | measured |
|---|---|---|---|---|
| pyrene, three bands | 3.38, 4.15, 5.03 | 3.65, 4.53, 5.41 | — | 3.98, 4.81, 5.48 |
| coronene, strong bands | 3.68, 5.93 | 4.01, 6.40 | 4.95, — | not in our data set |
| coronene⁺ D₁, D₂, D₃ | 1.30, 1.75, 2.85 | 1.46–1.53, 1.93, 2.92 | nothing bright below 3.5 | 1.314, 1.80, 2.713 |

- **Open shells work:** std2 reads PySCF's unrestricted Molden files.
- **ωB97X-D3 fails here.** It places the strong coronene band 0.9 eV above B3LYP. For the cation it loses D₁–D₃, with
  12 eV CSF windows (1742 and 2282 CSFs).
- **B3LYP** halves the neutral error on pyrene (−0.6 → −0.3 eV). On coronene⁺ it is worse than TD-LDA (+0.13 to
  +0.2 eV against −0.05 to +0.14).
  - So a correction would apply to neutrals only.
  - This matches the literature: ions in TD-LDA are within ±0.3 eV with no sign.
- **Cost on the laptop** (M4 Max, 36 GB), one B3LYP ground state: C54H18, 900 functions, 7 min, 4.3 GB. C96 is
  feasible; C150 and C216 need DIaL3 or long out-of-core runs.
- **The splice, a negative result:** moving TD-LDA onto the sTD-DFT positions by matching cumulative oscillator
  strength fails. The two methods share f differently between bands, and the map shifted pyrene's S₃ by −0.26 to
  −0.40 eV, depending on the cut-off. A band-by-band assignment works on pyrene, but is fragile for the dense spectra
  of C96–C216.
- **Band-by-band test on pyrene** (2026-10-28, `results/pahchem/stda_splice_2026-10-27/band_shift_pyrene.py`). Each
  TD-LDA band is moved rigidly, so its shape and f are kept. Centroid errors against the 423 K vapour (3.98, 4.81,
  5.48 eV), bands 1–3; TD-LDA is −0.61, −0.66, −0.45 (mean |error| 0.57 eV):

  | hybrid sticks | shift from | errors [eV] | mean \|error\| |
  |---|---|---|---|
  | sTDA | stick cluster of the same rank | −0.17, −0.09, +0.15 | 0.14 |
  | sTD-DFT | same | −0.34, −0.28, −0.07 | 0.23 |
  | sTDA | cumulative-f window | −0.25, −0.59, −0.33 | 0.39 |
  | sTD-DFT | cumulative-f window | −0.41, −0.70, −0.45 | 0.52 |
  | sTDA, one shift for all (+0.49) | mean of the above | −0.12, −0.17, +0.04 | 0.11 |
  | one shift fitted to the vapour (+0.61), ad hoc | | | 0.07 |

  - Matching by rank works. The cumulative-f window barely helps, because the methods share f differently between
    bands. The library needs rank matching, with band edges set automatically.
  - **sTDA does better than sTD-DFT** (0.14 against 0.23 eV). The library's std2 runs used sTD-DFT (`-rpa`); sTDA
    needs a rerun from the same Molden files, cheap except for C216.
  - The per-band shifts differ (sTDA +0.44, +0.57, +0.60, +0.36 eV; sTD-DFT +0.27, +0.38, +0.38, +0.18), so a
    uniform shift fits pyrene about as well as per-band ones do. Pyrene is one molecule with three bands. It cannot
    show whether per-band shifts pay off in the dense spectra of C54 and above.
  - The +0.55–0.65 eV for the larger molecules above is the alignment of the strongest band, not this method. On
    pyrene that method gives +0.66 at the 6.4 eV band against +0.27 to +0.38 at the three lower ones, so it may
    overstate the shift of the lower bands.
- **Per-band matching does not carry over to the dense spectra** (2026-10-28, sTDA of all 13 neutrals on megatron, minutes
  each except C150/C216). Bright sTDA clusters (f > 0.05, within 0.3 eV) against TD-LDA peaks:
  - coronene: TD-LDA 3.67 and 5.93 eV; sTDA 4.19, 6.02, 6.68 and 7.20. The strongest sTDA cluster (6.68) is 0.75 eV
    above the strongest TD-LDA band, the one of the same rank (6.02) only 0.09 eV.
  - C54H18: TD-LDA 2.50, 4.31, 5.25, 6.14; sTDA 2.86, 4.40, 4.96, 6.30 (the last holds 18 of the f). The ranked shifts
    are +0.36, +0.09, −0.29, +0.16 eV.
  - C96H24: TD-LDA 1.82, 2.87, 4.09, 4.97, 6.43; sTDA 2.09, 3.33, 3.96, 5.97. The ranked shifts are +0.27, +0.46,
    −0.13, +1.00 eV. TD-LDA has more bands than sTDA has clusters.
  - An automatic overlap alignment of each band fails even on pyrene (band 1 hits the search limit). So only rank matching
    works, and only where the bands are well separated.
  - The shifts therefore cannot be set band by band for C54 and up without a hand-made assignment per molecule.

- **Validation against larger molecules** (2026-10-28; hybrid runs in `RAMSES_dev/pahspec/stda_validation_2026-10-28`):
  - **Ovalene S₁** (B₂ᵤ, the lowest bright state): origin 465 nm = 2.67 eV in solution and jet (Wu+ 2024 [read]; the
    measured Ne-matrix table of Ruiterkamp+02 is behind the publisher's block and still to get). TD-LDA puts it at
    2.40 eV, sTDA at 2.68, sTD-DFT at 2.56, and full TD-B3LYP/6-311++G(2d,2p) at 2.50 (Wu+ 2024). Vertical energies lie
    above the origin by ~0.1 eV, so TD-LDA is ~0.3–0.4 eV low and sTDA is within ~0.1.
  - **Hexabenzocoronene** (Steglich+ 2011 [read]): full TD-B3LYP/6-31+G(d) puts S₁ at 536 nm against 540 nm (origin)
    measured, within 0.02 eV, and S₂ at 448 nm against 434 nm. std2 does not reproduce that: its S₁ is at 2.74 eV (sTDA)
    and 2.73 eV (sTD-DFT) against 2.31 eV (full TD-B3LYP) and 2.30 eV measured, a 0.4 eV overshoot. For the strong
    E₁ᵤ band std2 is close: 3.39 and 3.28 eV against 3.41 eV (full TD-B3LYP).
  - **So std2 overshoots the weak, low-lying states** (HBC S₁ by 0.43 eV, ovalene S₁ by 0.18 eV over full TD-B3LYP) and
    does well on the strong ones. The ovalene agreement of sTDA with the measured origin (2.68 against 2.67 eV) is
    therefore partly that overshoot. Full TD-B3LYP is the faithful reference for the weak bands: it is 0.17–0.4 eV
    below the measured ovalene S₁ and agrees to 0.02 eV for HBC.
  - **Consequence:** std2 shifts are reliable for the strong bands. For the weak visible and near-UV bands, which carry
    the neutral absorption in stellar fields, the shift needs full linear-response B3LYP (PySCF/megatron) for a few
    molecules. The sensitivity (+30–55 % in stellar-field rates) sits on those bands.

- **How much the choice matters** (2026-10-28, `shift_sensitivity.py`): the 12 neutrals with the part below 7 eV moved
  rigidly up by d, rate over the unmoved rate (min..max over the molecules):

  | d [eV] | Mathis | Draine 1978 | Kurucz 10 000 K | 6000 K blackbody | mean Mathis photon [eV] |
  |---|---|---|---|---|---|
  | 0.3 | 0.68–0.78 | 1.03–1.27 | 0.68–0.95 | 0.67–0.75 | 4.04 |
  | 0.5 | 0.52–0.67 | 1.03–1.42 | 0.56–0.91 | 0.50–0.61 | 4.45 |
  | 0.6 | 0.46–0.63 | 1.04–1.46 | 0.53–0.87 | 0.44–0.54 | 4.66 |

  The unmoved mean is 3.46 eV. A 0.1 eV difference between shifts, the spread of pyrene's per-band shifts, changes the
  rates by 3–15 %. So the correction is a 30–55 % effect on neutral absorption in stellar fields, and "leave TD-LDA
  positions" is not a neutral option.

- **The user's choice (2026-10-27): B3LYP positions for the neutrals, band by band; ions stay on TD-LDA; computed on
  megatron.**
  - All 12 library neutrals and coronene: B3LYP/def2-SVP, then sTD-DFT with a 10 eV CSF window. Every SCF converged.
    C216H36 took 7.5 h, with integral-direct SCF from C96 up.
  - Location: `~/pahspec/stda_neutrals_2026-10-27` on megatron, mirrored to
    `RAMSES_dev/pahspec/stda_neutrals_2026-10-27/results`.
  - A std2 bug: `print_tdadat` declares its file name `character*80`. Under gfortran 15 that garbles the name of
    `tda.dat`. Patched locally to `character(len=*)`.
  - **The strongest band at 3.5–7 eV moves up by 0.55–0.65 eV** for the compact family, C24–C216 (§7), and by
    0.56–0.79 eV for the isomers. Two isomers are ambiguous: C54H20_3176 (+0.15) and C54H24_3870, where the alignment
    and the strongest maximum disagree.

## 6. Cost and computing

- **Production layout** (C96H24 scaling tests): 1 node with ParStates 4 × ParDomains 32, 235 core-h per T = 20 kick of
  C96H24.
- **Kick time** grows as (N_C/96)^p, with p = 1 measured below C96 and 2.5 assumed above.
- **Spin-polarised runs** cost ×3.
- **The library** (48 states, −1 to +2) was ~8 × 10⁴ core-h. The triplet and doublet reruns add ~5 × 10⁴.

## 7. Comparison with Lin, Yang & Li 2023 (the 2175 Å bump)

Lin, Yang & Li 2023 (MNRAS 525, 2380 [read]) and Lin, Yang, Li & Witstok 2025 (A&A 694, A84 [read]) model the bump
with Octopus spectra of compact PAHs (2023: 30 molecules, C16–C96, −1/0/+1; 2025: 48 neutrals to C150). The user
chose Lin+23 as the reference to compare against (2026-10-28).
- **Set-up:** "B3LYP", a 0.3 Å grid, 3 Å spheres (5 Å for anions), T = 20; geometries B3LYP/6-31+G(d).
- **Width:** 0.2 eV per transition. Our spectra reproduce theirs with exp(−0.2 t), i.e. a Lorentzian of half width
  0.2 eV.
- **Fit:** each spectrum per C over 0–25 µm⁻¹ with a Drude profile (the π* ← π bump: peak x₀, width γ) plus a Fano
  profile (the σ band). Mixtures are equal-weight means.
- **Ours:** `results/pahchem/lin25_bump_2026-10-27/compare_lin.py`, log `compare_lin.log`, figure
  `lin23_compare.png`. Their numbers are read off their figures.

**Per species, the five molecules we share** (x₀ and γ in µm⁻¹, with their processing):

| | x₀ ours (Lin+25, neutral) | γ ours, −1 / 0 / +1 | γ Lin+23 Fig. 8 |
|---|---|---|---|
| pyrene | 5.01 (5.00) | 1.56 / 1.13 / 1.17 | 1.63 / 1.01 / 1.15 |
| coronene | 4.80 (4.76) | 0.63 / 0.53 / 0.50 | 0.57 / 0.55 / 0.55 |
| ovalene | 4.52 (4.48) | 1.20 / 0.99 / 1.01 | 1.01 / 1.00 / 1.00 |
| C54H18 | 4.30 (4.28) | 0.78 / 0.77 / 0.79 | 0.80 / 0.61 / 0.71 |
| C96H24 | 4.08 (4.11) | 1.15 / 1.12 / 1.13 | 1.16 / 1.01 / 1.04 |

- Positions agree to 0.05 µm⁻¹, widths to ≤ 0.2 µm⁻¹. C150H30 (Lin+25 only) has the same x₀, 3.93.
- **The exception is C150H30's width:** 1.42 in ours against their 0.48. Our spectrum has a second strong band at
  4.7 µm⁻¹, which the Drude term spans. Open: their spectrum is not published.
- **Their grid:** on coronene, 0.3 Å against our 0.25 Å raises x₀ by 0.08 µm⁻¹ and γ by 10 %, and lowers the band
  strength by 6 %.

**So their band positions are TD-LDA's,** whatever functional they name:
- B3LYP sTD-DFT moves the bump band of every compact molecule up by 0.55–0.65 eV (§5).
- Their own anthracene test, a bump 0.4 eV below the gas phase, is TD-LDA's error size. Our pyrene's 5.0 eV band is
  0.45 eV low. B3LYP's is −0.06 ± 0.1 in the literature, and −0.07 in our sTD-DFT.

**Size and charge, as they analyse them:**
- **Size trend.** Our compact molecules, charges averaged: x₀ = 4.52, 4.30 and 4.07 at N_C = 32, 54 and 96. Their
  size groups give 4.53, 4.31 and 4.18. It continues to 3.93 (C150) and 3.84 (C216).
- **Charge.** Over our 12 molecules with all of −1/0/+1, x₀ = 4.17, 4.19 and 4.20 and the strength falls 7 % from
  anion to cation. Their Table 1 gives 4.35, 4.41 and 4.45 and 5 %. Our mixture sits lower because it is larger
  (mean N_C ~90 against 43) and includes non-compact isomers.
- Like theirs, our bump barely weakens on ionisation, which experiment disputes (Lee & Wdowiak 1993; Robinson+97).
- **Non-compact isomers:** bumps at 3.75–4.10 µm⁻¹, broad (γ 1.5–3.7). For C54H30_3835 the Drude profile fails: γ
  reaches the fit bound.

**What B3LYP positions would do to their conclusions.** The compact family's x₀ rises by 0.44–0.52 µm⁻¹:

| | C54H18 | C96H24 | C150H30 | C216H36 |
|---|---|---|---|---|
| x₀, B3LYP positions [µm⁻¹] | 4.77 | 4.59 | 4.45 | 4.36 |

- **Lin+23** assumed +0.3 eV, which put their mixture (mean N_C 43) at 4.64 µm⁻¹, on the Milky Way bump. With
  +0.55–0.65 eV it would sit at ~4.85–4.9. The Milky Way's 4.60 would then need N_C ~100.
- **Lin+25** fitted the JADES-GS-z6-0 bump (4.42 µm⁻¹) with unshifted positions, dominated by C52–C64. With the
  shift those molecules peak at ~4.75–4.9; 4.42 needs N_C ~150.
- **Caveat:** sTD-DFT is checked against experiment only on pyrene's 5 eV band. The larger molecules have no gas-phase
  data.

**How firm each result is** (2026-10-28):
- **Firm:** their positions are reproduced; their positions are not B3LYP-level; the charge trends are shared.
- **Approximate:** the size trend. Their groups mix several molecules, ours are single molecules.
- **Open:** C150H30's width.
- **Rough:** the sizes the bumps would need (N_C ~100 and ~150).
  - x₀ + shift assumes the Drude peak moves with the strongest band. In pyrene the bands move by different amounts:
    +0.38 eV at 5.0 eV, +0.66 eV at 6.4 eV.
  - The widths depend strongly on the assumed per-transition width. With exp(−0.1 t) instead of exp(−0.2 t), γ falls
    from 1.13 to 0.52 (pyrene), from 0.53 to 0.35 (coronene) and from 1.12 to 0.67 (C96H24).
  - Lin+25's pre-selection (γ < 1.0) and its narrow JADES fit therefore partly reflect their 0.2 eV choice.

**To check with the final library** (the user's request, 2026-10-28):
1. **Redo their bump fits:** the Milky Way bump (Lin+23) and JADES-GS-z6-0 (Lin+25). Use the final spectra twice,
   once with TD-LDA positions and once with the spliced B3LYP positions, in place of the x₀ + shift estimate.
2. **Recheck C150H30's width** with the final processing. Ask the authors for their spectrum if it still disagrees.
3. **Report γ for both profiles:** their 0.2 eV Lorentzian and our final profile.
4. **Add the remaining states** to the charge analysis: C216H36 −1, +1 and +2, the other dications, and the rerun
   anions.
