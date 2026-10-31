# PAH UV–visible absorption for pahchem: the Octopus TD-DFT library

Status on 2026-10-29, branch `feature/pahchem-anharmonic`. pahchem's default σ(E, Z) is still the Malloci average of
{cite:t}`Berne2022` (`photo.sigma_uv`: nine compact PAHs at 5–30 eV, {cite:t}`LiDraine2001` outside), a placeholder.
Li & Draine is to be dropped. Its replacement is our own charge-resolved real-time TD-DFT library, assembled from 0.02 eV
to 10 keV for 48 states (§4): C54–C216, charges −1 to +2, with TD-DFT below 80 eV and atomic cross sections above, and
the measured carbon K edge, and a rule that turns its 57 states into σ for any N_C and charge (§4e). It is checked against {cite:t}`Malloci2007`, laboratory data and Lin, Yang & Li (§7).

**Two choices ship with the library, both labelled.** The neutrals' bands below 6 eV are moved up by 0.4 eV, because TD-LDA
puts them 0.3–0.65 eV too low (§5; ad hoc, with TD-LDA's positions kept in the same file), and the carbon K edge is the
measured one for pyrene, used for every PAH (§4c). The library is available in pahchem as `Params(uv_cross_section="octopus_library")` (§4e), not yet the default.

```{figure} ../_static/pahchem_uv/library_overview.png
:width: 100%
:alt: Cross section per carbon of four PAHs at 0–20 eV in four charge states.

The library at 0–20 eV for four molecules (σ per carbon). The visible and near-UV bands (1–7 eV) depend on the charge and
the molecule; above ~10 eV every state follows one smooth hump, peaking at 15–17 eV at ~25 Mb per carbon. The neutral is
shown with the 0.4 eV shift of its bands below 6 eV (solid) and with TD-LDA's positions (dotted).
```

## 0. Where things are

| What | Where |
|---|---|
| Scripts | `diagnostics/pahchem/octopus/` (below) |
| Runs, megatron | `~/pahspec`: coronene, pyrene, ovalene and the convergence tests; hybrid runs in `stda_*` |
| Runs, DIaL3 | `/lustre/dirac3/scratch/dp016/dc-rodr4/pahspec/library` (account dp016) |
| Local mirrors and laboratory spectra | `RAMSES_dev/pahspec/` (`results/dial3_library`, `experiment/{atlas,vuv}`, `STATUS.md`) |
| Assembled library | `results/pahchem/octopus_library/` (`pah_sigma_library.npz` with `sigma_cm2` and `sigma_cm2_tdlda`, `sigma_<state>.dat`, `checks.csv`) |
| Figures | `docs/_static/pahchem_uv/`, made by `make_library_figures.py` |
| The rule σ(E, N_C, Z) | `pycalima.pahchem.uvlib`; anchors `src/pycalima/data/external_data/pahchem/pah_uv_library.npz` (from `build_library.py --anchors`) |

| Script | Role |
|---|---|
| `make_run.py` | one Octopus run: inputs for the ground state, three kicks and the spectrum |
| `make_dial3_library.py` | the 48-state library on DIaL3: geometries from PAHdb, Slurm jobs, time limits |
| `queue.py`, `plot_progress.py`, `compare_malloci.py`, `check_damping.py` | megatron runs, progress, comparison with Malloci |
| `plot_dial3_library.py` | the library against Malloci, pahchem's default and the atomic sum |
| `build_library.py` | the final assembly (§4) and its checks |
| `make_library_figures.py` | the figures of this page |
| `check_uvlib_rule.py` | the accuracy of the size and charge rule (§4e) |
| `run_pyscf.py`, `run_tddft.py` | hybrid ground state for std2; full TD-B3LYP (§5) |
| `shift_sensitivity.py`, `compare_lin.py` | effect of a band shift on rates; the Lin+23 comparison (§5, §7) |

## 1. Method and how to rerun it

**Octopus 16.4** (real time, real space), LDA (Perdew–Zunger), the standard norm-conserving pseudopotentials:

| Setting | Value | Why |
|---|---|---|
| Grid | 0.25 Å | within 1 % of 0.20 Å. Malloci's 0.30 Å is not converged with these pseudopotentials |
| Box | atoms ≥ 4 Å from the edge, plus a 1 Å absorbing boundary | §4 |
| Kick | δ, 0.01 Å⁻¹, along x, y and z | linear response |
| Propagation | AETRS, dt = 0.002 ħ/eV, T = 20 ħ/eV | resolution ħ/T ≈ 0.05 eV; dt gives a spectrum to > 150 eV |
| Open shells | spin-polarised, ExtraStates = 4, occupations fixed per spin channel | closed-shell runs never converge; PAHdb's triplet dications run as triplets |
| Quasi-degenerate holes | C216H36⁺ (hole), C54H30(3835)⁻ (electron) shared 0.5/0.5 over the pair | whole electrons did not converge (`--shared-hole`, `--shared-electron`) |

**Reproduce one state:** `make_run.py <xyz> --charge Z --out <dir> [--time T] [--multiplicity M]`, then run the ground
state, the three kicks (`td_x/y/z`) and `oct-propagation_spectrum`. **The library:**

1. `make_dial3_library.py --out <dir>` writes the inputs, geometries (PAHdb v4.00, B3LYP) and `submit.sh` for the 12
   molecules (C54–C216, compact and non-compact isomers) at −1…+2. One node, ParStates 4 × ParDomains 32 for the kicks
   (not for the ground state). PAHdb itself stays on this machine.
2. `plot_dial3_library.py --sync dial3` mirrors the multipoles, then plots and checks.
3. `build_library.py --benzene RAMSES_dev/pahspec/experiment/vuv/bz_feng.dat` assembles the library (§4).

**The spectrum** is the sine transform of the dipole d(t) times a window D(t), σ(E) ∝ E ∫₀ᵀ d D sin(Et) dt
(`plot_progress.octopus_sigma`, equal to Octopus's `cross_section_tensor` to 6 × 10⁻⁸). The dipoles are written every
step, so a spectrum to any energy needs no rerun. The window sets the band profile: polynomial (Octopus's default)
rings; exp(−γt) gives Lorentzians; exp(−s²t²/2) gives Gaussians. A clean Gaussian needs s ≥ 3/T, i.e. **s ≥ 0.15 eV at
T = 20**.

**DIaL3 reproduces megatron:** the same coronene run agrees in ground-state energy and all eigenvalues, and in the
dipole to 4 × 10⁻¹³.

## 2. Malloci's spectra carry an exponential window

{cite:t}`Malloci2007` give T = 20 ħ/eV and "a resolution ħ/T = 0.05 eV", but no window. `check_damping.py` recovers it
(the inverse transform of σ/E gives d·D; the method returns our own polynomial window to ≤ 0.03): **exp(−γt) cut at
t = 20, with γ = 0.15–0.21 eV**, i.e. Lorentzians of FWHM 0.3–0.4 eV. It is probably Octopus's old default (`oct-sf`
before September 2005). With it our runs match Malloci's, within 3 % for coronene⁺ and 5 % for circumcoronene, in every
band above 0.5 eV.

**Consequence.** Malloci's absorption floor below a neutral's first band is the Lorentzian wing of the window, not
something the molecule absorbs. It carries about a quarter of a neutral's Mathis 1–5 eV rate. The cations' visible
bands are real.

## 3. What experiment provides

| Molecule, charge | Phase | Range | Reference |
|---|---|---|---|
| pyrene, anthracene, naphthalene 0 | vapour, 296–873 K | 3.5–6.2 eV, absolute (scales differ up to 2×) | Thöny & Rossi 1997; Ferguson+ 1957; Grosch+ 2015 (MPI-Mainz Spectral Atlas) |
| benzene, naphthalene 0 | gas, synchrotron | 5–200 eV, absolute | Rennie+ 1998; Suto+ 1992; Feng+ (benzene, to 200 eV) |
| pyrene, anthracene, mixtures 0 | gas, 570–630 K | 1.8–15 eV | Joblin, Léger & Martin 1992 |
| coronene, pyrene and others 0 | gas EELS, 0.6 eV resolution | 0–30 eV, relative | Keller+ 1992 |
| coronene, ovalene, C30–C48 PAHs 0, ± | Ne matrix, 4 K | UV–NIR, f for some | Ehrenfreund+ 1992; Ruiterkamp+ 2002 |
| HBC 0, +1 | Ne, Ar matrix | 1.5–6 eV | Steglich+ 2010, 2011; Rouillé+ 2009 |
| coronene⁺ | gas, He-tagged | D₁–D₃ | Hardy+ 2017; Bréchignac+ 2014 |
| ovalene 0 | jet, para-H₂ | S₁, S₂ origins | Amirav+; Wu+ 2024 |

Neon matrices shift neutrals by ~0.25 % and ions by ~0.5 % in energy. No gas-phase spectra of anions or dications, and
none of neutrals above C32, were found.

**Widths.** The spread of a whole band system is what absorption rates depend on, not its vibronic peaks:

| Case | σ [eV] |
|---|---|
| single vibronic lines, cold | 0.002–0.06 (FWHM) |
| band systems, cold C24–C48 in Ne | 0.07–0.12 |
| vapours of small PAHs | 0.12–0.18 at 300–420 K; 0.15–0.22 at 770–870 K |
| lifetime broadening above 5 eV | Lorentzian FWHM 0.01–0.07 |

Above ~13 eV every measured spectrum is one smooth hump (peak 17–18 eV, ~25 Mb per C, FWHM 12–15 eV): congestion of
states, not short lifetimes. The 217.5 nm interstellar bump is a property of mixtures and constrains no single width.
**The raw TD-DFT spikes at high energy are box artefacts:** above the ionisation threshold the box makes the continuum
discrete (Wenzel+ 2020; Zhen+ 2016). Malloci's database, with the same 4 + 1 Å box, was judged converged only to
~13 eV.

## 4. Building the library

### 4a. Width and the choice of profile

TD-DFT gives vertical excitations, not the vibronic envelope, so the envelope is an explicitly stated, empirical choice
(the no-ad-hoc-factors rule). Calibration target: the second moment of each band system. For pyrene every Octopus band
is one state, so the Gaussian needed equals the laboratory spread: s = 0.15 eV at 313 K, 0.16 at 423 K and 0.18 at
873 K (±0.02 for the choice of band edges). The final profile:

1. **Gaussian window of s₀ = 0.15 eV** at T = 20 (the narrowest clean width, and the warm-vapour value).
2. **Widening to s(E):** s₀ to 7.5 eV, rising to 0.8 eV at 14 eV and flat to 30 eV, then rising to 4 eV at 40 eV and flat.
   - Why: it removes the box artefacts of the discretised continuum (the local shape then agrees between boxes to ≤ 3 %
     at 14–30 eV; with 4 eV above 40 eV, O-star rates agree to ≤ 1 %) and widens the 17 eV hump by < 2 %. The 8–13.6 eV
     part is only weakly constrained by data.
   - How: conservative diffusion in energy to D = (s² − s₀²)/2. It keeps f exactly and σ ≥ 0. A convolution whose width
     follows the source energy drifts strength down the width gradient (it moved up to 5 % of f below 13.6 eV).
   - No Lorentzian part: lifetime widths are far below s, and Lorentzian wings put absorption where the molecule is
     transparent.
3. **Positions:** TD-LDA's (§5).
4. **Atoms above 80 eV** (§4c).

```{figure} ../_static/pahchem_uv/processing_stages.png
:width: 100%
:alt: Steps from Octopus's spectrum to the library for C96H24.

From Octopus's spectrum to the library (C96H24 neutral). (a) 0–14 eV: Octopus's polynomial window rings; the Gaussian window of 0.15
eV removes it, the widening to s(E) smooths the rise above 10 eV, and the shift moves the neutral's bands up. (b) 12–150 eV: the box's
ripples in Octopus's spectrum are smoothed away, and the spectrum blends into the scaled atomic sum between 80 and 130 eV.
```

**Is T = 20 enough?** Coronene neutral at T = 40, cold (s = 0.08 eV) against warm (0.15 eV): every rate (Mathis, Draine
1978, a 10 000 K Kurucz star, a 6000 K blackbody; 0.1–13.6 eV and above the ionisation energy) changes by ≤ 2.4 %, and the
mean absorbed photon energy by ≤ 0.02 eV. At equal s, T = 40 and T = 20 agree to 0.75 %. Coronene⁺ (spin-polarised, T = 40, same
settings; `compare_width_ion.py`) agrees: the same run cut at T = 20 gives rates within 0.05 %, and cold against warm
widths changes the rates by ≤ 1 % in the Mathis and Draine fields, ≤ 4.6 % for a 10 000 K star (0.1–5 eV; 1.8 % in total)
and by ≤ 0.04 eV in the mean absorbed photon energy. So **T = 20 and s₀ = 0.15 eV suffice** for the two species tested.

**Decision (2026-10-29): the library keeps T = 20 ħ/eV and s₀ = 0.15 eV.** The kick restart files are not kept, and the
ground-state restarts are. Cold widths (s ≈ 0.08 eV) remain a ≤ 5 % systematic on the rates.

### 4b. Box and grid

| Test | Result |
|---|---|
| 8 Å against 5 Å, coronene 0/−1, T = 20 | band integrals within 0.4 % (anion 1–5 eV 1.8 %); the shape is box-independent below 6.5 eV (anion 5 eV) |
| 5, 6.5, 8 and 10 Å at T = 10, smoothed | above 13.6 eV the shape converges (5 Å is the outlier, 6.8 % rms; 8 Å within 3 %); at 9–13.6 eV **no box converges** (4–12 % rms, no trend), and a two-box average adds nothing |
| rates, 5 Å over 10 Å | ≤ 2 % for ISRF-like fields, ≤ 5 % for a 10 000 K star (7–13.6 eV); O-star fields 13.6–140 eV 1.000–1.003 |
| grid, 0.25 against 0.20 Å | to 100 eV within 2.5 % rms. 0.30 Å fails above 30 eV (24–32 %) |
| T = 5 against T = 10 above 12.5 eV | 0.4–1.3 % rms |

So the production box (4 + 1 Å) serves neutral gas and ionising fields: **no second box**. The shape at 9–13.6 eV (above
~5 eV for anions) keeps a ~15 % local uncertainty. Untested: cations and dications, which are more compact than the
anion.

### 4c. The ionising and X-ray part

Most absorption is above 30 eV. For coronene (108 valence electrons) f = 68.6 below 30 eV, 92.4 below 54.4, 103.3 below 100
and 106.8 below 150 eV. Against the **atomic sum** (Verner phfit2 C 2s 2p, hydrogenic H 1s; C 1s from 291 eV), Octopus over
atoms is:

| 13.6–30 eV | 30–54.4 eV | 54.4–150 eV |
|---|---|---|
| 1.33–1.41 | 1.32–1.35 | 1.00–1.01 |

- **Below ~50 eV the molecule absorbs a third more than its atoms.** Measured benzene agrees (1.50 and 1.30; Feng+ to
  200 eV, f = 29.3 of 30).
- **Above ~80 eV both TD-LDA and benzene lie a constant factor below the atoms:** 0.88–0.89 (library) and 0.78–0.79
  (benzene, flat to 200 eV). The atomic shape is right and its normalisation is the molecule's.
- **Assembly:** TD-LDA to 80 eV; a cos² blend from 80 to 130 eV into the atomic valence sum scaled by k (TD-LDA's integral
  over theirs at 80–130 eV; k = 0.885–0.891 in every state); C 1s unscaled from 291 eV. Unscaled atoms would sit 13 %
  above TD-LDA and 27 % above benzene at 100–290 eV. The library itself is 10–25 % above benzene at 40–200 eV, with
  bumps at ~40 and ~55 eV that benzene lacks.
- **The C K edge** (optional, `build_library.py --ck-edge`). Octopus's pseudopotentials freeze the 1s core, so the TD-DFT part has
  no K-shell absorption, and Verner's atomic C 1s is a plain step at 291 eV. Real molecules start earlier and show 1s → π*
  resonances. The library therefore replaces the atomic C 1s from 282 to 296 eV by the measured gas-phase C 1s oscillator
  strength per carbon atom of **pyrene** (ISEELS, 0.55 eV resolution, absolute: Gordon, Tulumello, Cooper, Hitchcock+ 2003, Fig. 1,
  digitised from the bitmap; `RAMSES_dev/pahspec/experiment/ck_edge`). From 296 to 310 eV the excess over the atomic value fades
  out with a cos² weight.
  - **What it holds:** the π* peak at 285.1 eV (1.95 Mb per C; 1s → π* integral 0.029 per C over 283–286.4 eV), Rydberg
    and σ* structure at 287–290 eV (0.8–0.9 Mb) and a broad hump to 1.7 Mb at 293.5 eV, about 1.8 times the atomic value. At 291.5 eV
    it agrees with Verner (0.97 against 0.935 Mb). The f to 10 keV over valence plus two per C rises from 0.972 to 0.991.
  - **Positions, high resolution:** neutral coronene has π* peaks at 284.5 and 285.2 eV (the second, from carbons without H, is
    stronger), then 286.0–286.5 and 286.7–287.1 eV, σ* from ~287.5 eV, and a C 1s binding energy of 289.6–289.95 eV
    (Fronzoni+ 2014; ion yield, no absolute scale). The coronene cation adds a 282.5 eV peak into the singly occupied orbital
    (Huo+ 2022).
  - **Uncertainties:** (a) pyrene stands for all PAHs, per carbon; the 1s → π* integral is 0.020–0.031 per C for the fused-ring
    molecules of Gordon+ (benzene 0.044), so ±20 %. (b) The ISEELS resolution (0.55 eV) is coarser than the structure of
    Fronzoni's 0.05 eV spectra: peak heights are too low, the area is right. (c) The 296–310 eV interpolation misses the
    σ*(C–C) structure at 299.6 and 306.2 eV. (d) Cations, anions and dications use the neutral curve; the cation's 282.5 eV hole peak
    is not included. (e) The data are not bundled with the package: without `--ck-edge` the library has the atomic step.

```{figure} ../_static/pahchem_uv/library_xray.png
:width: 100%
:alt: The library from 1 eV to 10 keV and the carbon K edge.

(a) The library for C96H24 from 1 eV to 10 keV against the sum of atomic cross sections. (b) The carbon K edge: Verner's atomic step
at 291 eV against the measured gas-phase C 1s spectrum of pyrene, which carries the 1s → π* resonance at 285 eV and the broad
σ* hump, and hands over to the atomic value by 310 eV.
```

```{figure} ../_static/pahchem_uv/library_euv_atoms.png
:width: 80%
:alt: The library over the sum of atomic cross sections from 12 to 250 eV, with measured benzene.

The library over the sum of atomic cross sections at 12–250 eV, for all 48 states, against measured benzene (Feng+). Both lie 30–80 %
above the atoms at 20–50 eV and settle at a constant factor below them above ~80 eV; the library's factor (0.89) is fitted to its
own TD-LDA at 80–130 eV, benzene's is 0.78.
```

### 4d. Checks (all 48 states, `checks.csv`)

| Check | Result |
|---|---|
| f to 30 eV, final over raw | within 0.2 % |
| f to 150 eV over valence electrons | 98.8–98.9 % |
| most negative σ over the maximum | −3 × 10⁻⁴ (T = 20 ringing below the first band of neutrals) |
| f below 13.6 eV, final over raw | −0.1 to +4.8 % (the 0.75 eV smoothing of the rise to the 17 eV hump; inside the box uncertainty) |
| rates, final over Octopus's own spectrum, charged states | Mathis −0.9 to +0.3 %; Draine 1978 −0.6 to +3.4 %; Kurucz 40 000 K −2.5 to +0.3 % |
| the same, neutrals (shifted by 0.4 eV) | Mathis 0.59–0.72; Draine 1978 1.02–1.39; Kurucz 40 000 K 0.98–1.00 |
| f to 10 keV over valence + 2 per C | 97.2–97.4 % with the atomic C 1s, 99.0–99.1 % with the measured K edge (shipped) |

### 4e. From 57 states to σ(E, N_C, Z)

pahchem needs σ for any size and charge: its PAHdb library holds 1424 molecules with N_C = 22–268 (most between 24 and 100) and
charges −1 to +3. `pycalima.pahchem.uvlib` (`sigma_library(nc, charge, energy_ev)`, `per_carbon`) applies one rule:

- **Size:** σ = N_C s(E; N_C, Z), with the per-carbon spectrum s interpolated linearly in ln N_C between the anchor sizes 16, 24, 32,
  42, 54, 96, 150, 216, and held at the nearest anchor outside.
- **Charge:** the charge itself for −1 to +2, and the dication for +3. Pyrene, coronene and ovalene were run as dications on megatron
  (2026-10-30; the neutral's geometry; coronene a triplet with fixed occupations, pyrene and ovalene closed-shell singlets), so every
  anchor is a computed state.
- **Structure:** the anchor is the equal-weight mean over every library molecule of that size, `structure="mixed"` (default): C24 (coronene
  and one isomer), C32 (ovalene and three isomers), C42 (four isomers) and C54, C96, C150 (five, four and two isomers, compact and not);
  C16 is pyrene alone. `"compact"` keeps the circumcoronene family alone; no compact C42 was run, so that anchor is interpolated from
  C32 and C54. Eight non-compact isomers of C24–C42 were run at all four charges (DIaL3, 2026-10-29/31; C42H24_3973, whose +1 state
  had not converged, is left out until it has, so that every charge averages the same molecules).
- **Hydrogen:** N_H does not enter. It cannot be modelled from the data: the five C54 isomers (N_H = 18–30) do not order by N_H.
- **Positions:** `positions="shifted"` (default, the neutrals' 0.4 eV shift) or `"tdlda"`.

```{figure} ../_static/pahchem_uv/size_rule.png
:width: 100%
:alt: Per-carbon spectra of the seven anchors and the Mathis-field absorption rate against N_C.

The rule. (a) The per-carbon spectrum of the neutral at the seven anchors and at N_C = 70 (dashed), interpolated. The bands move down and
spread with size. (b) The photon absorption rate per carbon in the Mathis field against N_C, for four charges with mixed (solid) and
compact (dotted) anchors: it grows fivefold from C16 to C216 as the bands move into the visible, and the mixed anchors lie 13–47 % above
the compact ones at C54 (more low-lying bands in the isomers) and 4–10 % below at C96 and C150.
```

**How well it works** (`check_uvlib_rule.py`; per-carbon absorption rates over 0.1–13.6 eV, ratios):

| test | Mathis | Draine 1978 | Kurucz 10 000 K | blackbody 6000 K |
|---|---|---|---|---|
| a held-out anchor from its two neighbours, C54 and up | 0.71–1.22 | 0.91–1.07 | 0.88–1.10 | 0.80–1.14 |
| the same, C24 and C32 (compact; C32 from C24 and C54) | 1.00–1.12 | 0.95–1.02 | 0.87–1.21 | 0.99–1.38 |
| mixed over compact anchors, C24–C42 | 0.96–1.43 | 0.85–1.02 | 0.92–1.04 | 0.89–1.29 |
| mixed over compact anchors, C54 | 1.13–1.47 | 0.88–0.94 | 0.90–1.20 | 1.08–1.24 |
| mixed over compact anchors, C96 and C150 | 0.90–1.01 | 0.96–1.04 | 0.94–0.97 | 0.88–0.98 |
| dication over cation, per carbon | 1.07–1.23 | 0.93–0.98 | 1.00–1.05 | 1.01–1.22 |

- The interpolation error is at most ~15 % for the visible-weighted fields (Mathis, blackbody) in the largest case and a few % for the
  far-UV ones, comparable to the shift's uncertainty (§5). The test is a stringent one: it removes an anchor from a sparse set.
- **The structure is the larger systematic:** 0.9–1.5 in the Mathis rate at one size. Between C24 and C42 it is now sampled by eight
  isomers (C24 one, C32 three, C42 four), but the C54 mean, with five, is the only one where the isomers were chosen as a set; below C54
  the mixed anchor is the mean of however many were run, not of a representative sample of the 350 molecules of pahchem's 1424 that
  sit there. The default there is thus an isomer mean, 1–2 % richer in oscillator strength per carbon (≤ 30 eV) than the compact one.
- The dications of C16, C24 and C32 were first replaced by the cations' spectra. Running them showed that proxy to be 4 % (pyrene), 22 %
  (coronene) and 16 % (ovalene) low in the Mathis field, within 3 % in 13.6–60 eV, and 8–19 % high in the visible-weighted 40 000 K
  blackbody below 13.6 eV; the real dications keep the cations' main bands within 0.15–0.3 eV but have sharper structure above 10 eV.
- The library's energy range is 0.02 eV–10 keV; outside it the rule returns zero.

**Using it** (wired 2026-10-29). `Params(uv_cross_section="octopus_library")` selects the rule; `photo.sigma_uv`, the population bases, the group
tables and the family emission then use it. The suffixes `_tdlda` and `_compact` select TD-LDA's positions and the circumcoronene family
(`octopus_library_tdlda_compact` both). The basis cache keys carry the model name and the anchor file's fingerprint, so a rebuilt library does
not reuse old bases. It is **not the default yet**: that needs the anchors file bundled and the gas-coupling review (the 0.65 → 0.88 hydrogenation
shift of the Malloci default already moved the chain tests).

A cross-check through the pipeline: the photoionisation rate of C54H18 in a 40 000 K field (`Population.basis`, ratio to the Malloci average)

| model | −1 | 0 | +1 | +2 |
|---|---|---|---|---|
| `octopus_library` (isomer mean, shifted) | 1.065 | 1.197 | 1.224 | 1.202 |
| `octopus_library_tdlda` | 1.065 | 1.181 | 1.224 | 1.202 |
| `octopus_library_compact` | 1.008 | 1.000 | 1.008 | 1.050 |

The compact anchors reproduce the Malloci average (nine compact PAHs, C32–C66) to 0–5 %: two independent TD-DFT sets agree. The default
anchor, the mean over the C54 isomers, absorbs 6–22 % more, because the non-compact isomers have bands at lower energy.

## 5. Band positions: the open decision

**TD-LDA puts neutrals' bands too low** (signed error, calculated − measured):

| Band | TD-LDA | Source |
|---|---|---|
| pyrene S₂, S₃, 5.3 eV | −0.60, −0.66, −0.45 eV | vapour centroids at 423 K |
| ovalene S₁ | ~−0.15 to −0.25 eV | S₂ origin 2.61 eV in para-H₂ (Wu+ 2024), S₁ ≈ 0.06 eV below it (Kropp & Stanley); vertical ≈ origin + 0.1; TD-LDA 2.40 |
| ovalene, strongest band of the 270–340 nm system | −0.5 eV | Ne-matrix peak at 326.97 nm = 3.79 eV (Ruiterkamp+ 2002, Table 3); TD-LDA 3.29; TD-B3LYP/def2-SVP 3.70, sTDA 3.74, sTD-DFT 3.55 |
| ovalene, broad 215.8 nm band | −0.25 eV | 5.75 eV measured (FWHM 0.8 eV); TD-LDA 5.50 |
| acenes La | −0.64, growing with size | Peng & Chai 2014 |
| coronene⁺ D₁, D₂, D₃ | −0.01, −0.06, +0.14 eV | gas phase |
| HBC S₁ (weak), E₁ᵤ | ~−0.3 to −0.4, ~−0.5 eV | S₁ measured 2.86 eV (434 nm, vibronic peak); TD-LDA onset 2.46, E₁ᵤ peak 2.92 against 3.41–3.45 (TD-B3LYP) |

Ions are within ±0.3 eV with no sign, in the literature (Hirata+ 2003) and in ours. Above ~6–7 eV TD-LDA agrees with
gas-phase data, including the 17 eV hump. No astrophysical model has shifted its TD-LDA spectra; Mulas+ 2006 measured
Mathis-field rate ratios of 0.93–1.45 from the band positions.

**Why it matters** (`shift_sensitivity.py`: the part of each neutral below 7 eV moved up rigidly by d, f kept, 12 neutrals):

| d [eV] | Mathis | Draine 1978 | Kurucz 10 000 K | 6000 K blackbody | mean absorbed Mathis photon |
|---|---|---|---|---|---|
| 0 | 1 | 1 | 1 | 1 | 3.46 eV |
| 0.3 | 0.68–0.78 | 1.03–1.27 | 0.68–0.95 | 0.67–0.75 | 4.04 |
| 0.5 | 0.52–0.67 | 1.03–1.42 | 0.56–0.91 | 0.50–0.61 | 4.45 |
| 0.6 | 0.46–0.63 | 1.04–1.46 | 0.53–0.87 | 0.44–0.54 | 4.66 |

Moving the visible and near-UV bands up costs photons, because the fields fall steeply. **Leaving TD-LDA positions is not a
neutral choice**, and a 0.1 eV difference between shifts changes the rates by 3–15 %. For comparison, pahchem's default,
Li & Draine and the AmesPAHdb suite absorb Mathis photons at a mean 4.6–5.8 eV, against 2.0–4.8 eV for our states. The
cause is real: low-energy cation bands and the visible bands of large neutrals.

```{figure} ../_static/pahchem_uv/band_positions.png
:width: 80%
:alt: Calculated minus measured band positions for TD-LDA, sTDA, sTD-DFT and full TD-B3LYP.

Calculated minus measured band positions. TD-LDA (circles) is 0.15–0.65 eV too low; the hybrid methods (sTDA, sTD-DFT, full
TD-B3LYP/def2-SVP) are within ~0.25 eV. Measured values: pyrene's vapour centroids at 423 K; ovalene's strongest peak in neon and its
S₁ origin (estimated from S₂ in para-H₂); HBC's S₁ vibronic peak in neon. Pyrene's full TD-B3LYP is PySCF's TDA.
```

**The user's choice (2026-10-27):** hybrid-functional positions for the neutrals; ions stay on TD-LDA. Per-band assignment proved
impossible for the dense spectra (§5a), so the shipped correction is one uniform shift (below).

### 5a. What was tried

All calculations: PySCF B3LYP/def2-SVP (Cartesian, density fitting; integral-direct from C96), then `std2` (simplified
TDA, "sTDA", and simplified TD-DFT, "sTD-DFT", with `-ax 0.2`, a 10 eV CSF window), run on megatron for the 12 library
neutrals plus coronene, ovalene and HBC. Full linear-response TD-B3LYP (`run_tddft.py`) is the reference for a few.

| Method | Finding |
|---|---|
| ωB97X-D3 XsTD-DFT | fails: puts coronene's strong band 0.9 eV above B3LYP and loses coronene⁺ D₁–D₃ |
| B3LYP sTD-DFT/sTDA on coronene⁺ | worse than TD-LDA (+0.13 to +0.2 eV): a correction applies to neutrals only |
| sTDA against sTD-DFT on pyrene | mean error 0.14 against 0.23 eV (TD-LDA 0.57) |
| **std2 against full TD-B3LYP, same basis (def2-SVP)** | within 0.15 eV, no overshoot: ovalene S₁ 2.56 (sTD-DFT), 2.68 (sTDA) against 2.70; HBC S₁ 2.73, 2.74 against 2.88, E₁ᵤ 3.28, 3.39 against 3.45; coronene β 3.98, 4.19 against 4.12 |
| basis set and geometry | def2-SVPD moves ovalene's full TD-B3LYP by only −0.03 eV (S₁ 2.699 → 2.672; bright band 3.70 → 3.66), so the basis is not the issue. TD-B3LYP/6-311++G(2d,2p) in Wu+ 2024 gives S₁ at 2.50 eV, 0.17 eV lower, probably from their own optimised geometry (ours is Malloci's). HBC S₁ (B₂ᵤ, weak) is measured at 434 nm = 2.86 eV (a vibronic peak; Steglich+ 2011) against 2.88 eV in def2-SVP. Steglich's TD-B3LYP/6-31+G(d) places HBC's E₁ᵤ at 364 nm = 3.41 eV. (Their 536 against 540 nm agreement is for dibenzorubicene, not HBC) |
| **C54H18, full TD-B3LYP/def2-SVP, 60 states** | first strong band 2.89 eV (f = 0.93 per state, degenerate pair) against TD-LDA 2.50 (+0.39) and sTDA 2.86 (std2 agrees to 0.03 eV); the 60 states stop at 4.99 eV, so the strongest band (TD-LDA 5.25 eV) is not reached |
| matching bands by rank | works on pyrene (sTDA +0.44, +0.57, +0.60 eV for the three bands); ambiguous for C54 and up |
| matching by cumulative f | fails: the methods share f differently between bands |
| automatic overlap alignment of each band | fails, even on pyrene |

For C54H18, C96H24 and coronene TD-LDA has more bands than the hybrid has bright clusters, and the ranked shifts scatter
(C96H24: +0.27, +0.46, −0.13, +1.00 eV). The shift of the *strongest* band against sTD-DFT is +0.55–0.65 eV for the compact family
(+0.56–0.79 eV for isomers), but pyrene shows the lower bands move less (+0.27–0.38) than that, so it may overstate them.

**The shipped shift: +0.4 eV for the neutrals** (`build_library.py --shift 0.4`, the default; 0 keeps TD-LDA's positions). The
neutrals' spectrum below 6 eV is transported up by 0.4 eV, the shift fading out (cos²) by 10 eV. f is conserved exactly, σ stays
positive, and ions and the K edge are untouched. The file carries both variants (`sigma_cm2` and `sigma_cm2_tdlda`). Over the 12
neutrals the shift changes the absorption rates as follows, and lifts the mean absorbed Mathis photon from 3.46 eV (2.2–4.9) to
4.24 eV (2.8–5.7), closer to the 4.6–5.8 eV of pahchem's earlier default and the AmesPAHdb suite:

| field | rate, shifted over unshifted |
|---|---|
| Mathis ISRF | 0.59–0.72 |
| Draine 1978 (5–13.6 eV) | 1.02–1.35 |
| Kurucz 40 000 K (13.6–136 eV) | 1.000 |

```{figure} ../_static/pahchem_uv/shift_effect.png
:width: 100%
:alt: Mean absorbed photon energy and absorption rates of the neutrals against the band shift.

The effect of the shift on the 12 neutrals (mean and range). (a) The mean absorbed Mathis photon energy rises by ~2.0 eV per eV of shift,
towards the 4.6–5.8 eV of the earlier default and the AmesPAHdb suite (grey). (b) Absorption rates relative to the unshifted spectrum:
a 0.1 eV difference in the shift changes the rates by 3–15 %. The dotted line marks the shipped 0.4 eV.
```

The stretching of the spectrum between 6 and 10 eV (up to ~19 %) moves a small amount of strength in a valley of the spectrum.

### 5b. State of the decision

- **Evidence:** the first strong bands need +0.27 to +0.5 eV from C16 to C54 (ovalene +0.27 and +0.37, coronene +0.44, HBC +0.4 to
  +0.5, C54H18 +0.39, pyrene +0.4 to +0.6 against measurement), with no clear trend in size. The hybrid positions are stable
  against the basis (def2-SVP → SVPD: −0.03 eV); the geometry is the larger uncertainty (~0.15 eV for ovalene). Nothing measured
  constrains the strongest bands of C54 and larger.
- **Shipped (2026-10-29):** the uniform +0.4 ± 0.1 eV, labelled ad hoc, with TD-LDA's positions in the same file.
- **Not tested:** the 0.1 eV spread changes neutral rates by 3–15 % (figure above); the correction is not applied to ions
  (hybrid methods are worse than TD-LDA for coronene⁺), which shifts the neutral/ion balance.
- **To do:** the strongest bands of C54+ against full TD-B3LYP (>150 states), and the 2175 Å bump as a consistency check (§7).

### 5c. Reproducing the hybrid runs

- **Environment:** a separate venv with PySCF 2.14 (`~/opt/qc-venv`); `std2` from github.com/grimme-lab/std2, built with meson
  (`-Dla_backend=custom -Dcustom_libraries=openblas` where LAPACK sits inside OpenBLAS; megatron's gfortran 15 needs
  `print_tdadat`'s file-name argument changed from `character*80` to `character(len=*)`, or `tda.dat` is written with a garbled
  name).
- **Steps:** `run_pyscf.py <xyz> --charge Z --xc b3lyp --basis def2-svp [--no-df]` writes a Molden file (coordinates in
  Ångström, which std2 needs). Then `std2 -f <molden> -sty 1 -ax 0.2 -e 10 [-rpa]`, run from bash, with
  `OMP_STACKSIZE=512M`. `run_tddft.py <xyz> --nstates N` gives the full TD-B3LYP.
- **Cost:** a C54 ground state takes ~4 min on megatron and 4 GB (7 min, 4.3 GB on the laptop); C96 ~11 min; C216 ~75 min
  (density fitting needs 17 GB for C54 and ~930 GB for C216, hence integral-direct from C96). sTDA takes seconds to ~20 s
  for C96 and ~13 min for C150; sTD-DFT ~1 h for C150. Full TD-B3LYP of coronene, 40 states, 22 min.

## 6. Cost and computing

- **Production layout** (C96H24 scaling tests): one node, ParStates 4 × ParDomains 32, 235 core-h per T = 20 kick.
- **Kick time** grows as (N_C/96)^p, with p = 1 measured below C96 and 2.5 assumed above; spin-polarised runs cost ×3.
- **The library** (48 states, −1…+2): ~8 × 10⁴ core-h, plus ~5 × 10⁴ for the triplet and doublet reruns.
- **Time limits** follow 235 core-h × (N_C/96)^2.5, ×2 for spin-polarised, ×2.5 margin; a kick can still overrun (the
  C54H24(3870)⁻ one did at 0.95 s/step), and resuming from a checkpoint written at the kill can hang, so rerun it.

## 7. Comparison with Lin, Yang & Li 2023 (the 2175 Å bump)

Lin, Yang & Li 2023 (MNRAS 525, 2380) and Lin, Yang, Li & Witstok 2025 (A&A 694, A84) model the bump with Octopus spectra
of compact PAHs (2023: 30 molecules, C16–C96, −1/0/+1; 2025: 48 neutrals to C150). Their set-up: "B3LYP", a 0.3 Å grid,
3 Å spheres (5 Å for anions), T = 20, and a width of 0.2 eV per transition. **Our spectra reproduce theirs with exp(−0.2 t)**
(a Lorentzian of half-width 0.2 eV). They fit each per-C spectrum over 0–25 µm⁻¹ with a Drude profile (the π* ← π bump:
peak x₀, width γ) plus a Fano profile (the σ band). `compare_lin.py` does the same on ours.

```{figure} ../_static/pahchem_uv/lin23_compare.png
:width: 100%
:alt: Bump widths and peak positions of our spectra against Lin et al.

(a) The bump width γ of our spectra with their processing against their Fig. 8, for five molecules in three charge states. (b) The bump
peak x₀ against the number of carbons: their size groups (squares) and neutrals (crosses), ours (blue), and our neutrals with a
hybrid shift (orange arrows; the estimate of §7).
```

| | x₀ ours (Lin+25) [µm⁻¹] | γ ours, −1/0/+1 | γ Lin+23 Fig. 8 |
|---|---|---|---|
| pyrene | 5.01 (5.00) | 1.56 / 1.13 / 1.17 | 1.63 / 1.01 / 1.15 |
| coronene | 4.80 (4.76) | 0.63 / 0.53 / 0.50 | 0.57 / 0.55 / 0.55 |
| ovalene | 4.52 (4.48) | 1.20 / 0.99 / 1.01 | 1.01 / 1.00 / 1.00 |
| C54H18 | 4.30 (4.28) | 0.78 / 0.77 / 0.79 | 0.80 / 0.61 / 0.71 |
| C96H24 | 4.08 (4.11) | 1.15 / 1.12 / 1.13 | 1.16 / 1.01 / 1.04 |

- **Firm:** positions agree to 0.05 µm⁻¹, widths to ≤ 0.2. **Their positions are TD-LDA's** whatever functional they name:
  their anthracene bump lies 0.4 eV below the gas phase, TD-LDA's error size, and B3LYP would move it 0.4–0.6 eV (§5).
  The charge trend is shared (x₀ 4.17, 4.19, 4.20 and strength −7 % from anion to cation for us; 4.35, 4.41, 4.45 and −5 %
  for them; both disagree with experiment, which sees the bump weaken on ionisation). Their 0.3 grid moves coronene by
  +0.08 µm⁻¹.
- **Approximate:** the size trend. Our compact molecules, charges averaged, give x₀ = 4.52, 4.30, 4.07, 3.93, 3.84 at N_C = 32,
  54, 96, 150, 216; their groups give 4.53, 4.31, 4.18 at 32, 54 and 96. Their groups mix several molecules.
- **Open:** C150H30's width (1.42 ours, 0.48 theirs; ours has a second strong band at 4.7 µm⁻¹). Non-compact isomers have
  bumps at 3.75–4.10 µm⁻¹ and broad widths (γ 1.5–3.7; for C54H30(3835) the Drude profile fails).
- **Widths depend strongly on the assumed width per transition:** with exp(−0.1 t), γ falls from 1.13 to 0.52 (pyrene), 0.53 to
  0.35 (coronene), 1.12 to 0.67 (C96H24). Lin+25's pre-selection (γ < 1.0) and its narrow fit partly reflect their 0.2 eV.
- **What hybrid positions would change (rough):** x₀ → 4.77, 4.59, 4.45, 4.36 µm⁻¹ at C54, C96, C150, C216. That puts C96H24 on
  the Milky Way bump (4.60) and C150H30 on JADES-GS-z6-0 (4.42). Lin+23 assumed +0.3 eV and landed on 4.6 with a 43-carbon
  mean; with +0.55–0.65 eV that mean would reach ~4.85–4.9, and Lin+25's best-fit molecules (C52–C64) would peak at
  ~4.75–4.9. These estimates assume the Drude peak moves with the strongest band, and rest on def2-SVP hybrid positions
  (def2-SVP; ±0.15 eV from geometry) and a shift taken from the strongest band, so they are indicative only.

**To do with the final library:** redo their Milky Way and JADES fits with TD-LDA and with corrected positions; recheck
C150H30's width (ask the authors for their spectrum if it persists); report γ for their 0.2 eV Lorentzian and for our
profile; add the states missing from the charge analysis.
