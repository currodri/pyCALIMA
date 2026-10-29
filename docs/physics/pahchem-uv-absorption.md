# PAH UV–visible absorption for pahchem: the Octopus TD-DFT library

Status on 2026-10-28, branch `feature/pahchem-anharmonic`. pahchem's default σ(E, Z) is still the Malloci average of
{cite:t}`Berne2022` (`photo.sigma_uv`: nine compact PAHs at 5–30 eV, {cite:t}`LiDraine2001` outside), a placeholder.
Li & Draine is to be dropped. Its replacement is our own charge-resolved real-time TD-DFT library, assembled from 0.02 eV
to 10 keV for 48 states (§4). It is checked against {cite:t}`Malloci2007`, laboratory data and Lin, Yang & Li (§7).

**Two things are not decided.** The library is not yet wired into pahchem, and **band positions below ~7 eV are
TD-LDA's, which are 0.3–0.6 eV too low for neutrals** (§5). That bias alone changes neutral absorption in stellar
fields by 30–55 %.

## 0. Where things are

| What | Where |
|---|---|
| Scripts | `diagnostics/pahchem/octopus/` (below) |
| Runs, megatron | `~/pahspec`: coronene, pyrene, ovalene and the convergence tests; hybrid runs in `stda_*` |
| Runs, DIaL3 | `/lustre/dirac3/scratch/dp016/dc-rodr4/pahspec/library` (account dp016) |
| Local mirrors and laboratory spectra | `RAMSES_dev/pahspec/` (`results/dial3_library`, `experiment/{atlas,vuv}`, `STATUS.md`) |
| Assembled library | `results/pahchem/octopus_library/` (`pah_sigma_library.npz`, `sigma_<state>.dat`, `checks.csv`) |

| Script | Role |
|---|---|
| `make_run.py` | one Octopus run: inputs for the ground state, three kicks and the spectrum |
| `make_dial3_library.py` | the 48-state library on DIaL3: geometries from PAHdb, Slurm jobs, time limits |
| `queue.py`, `plot_progress.py`, `compare_malloci.py`, `check_damping.py` | megatron runs, progress, comparison with Malloci |
| `plot_dial3_library.py` | the library against Malloci, pahchem's default and the atomic sum |
| `build_library.py` | the final assembly (§4) and its checks |
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
- **Open: the C K edge.** Octopus's pseudopotentials freeze the 1s core, so the TD-DFT part has no K-shell absorption and
  the library has a plain atomic step at 291 eV. Real molecules and graphite start about 6 eV earlier with a sharp 1s → π*
  resonance near 285 eV. This matters only for soft-X-ray sources at 280–300 eV, and needs measured PAH carbon K-edge
  spectra.

### 4d. Checks (all 48 states, `checks.csv`)

| Check | Result |
|---|---|
| f to 30 eV, final over raw | within 0.2 % |
| f to 150 eV over valence electrons | 98.8–98.9 % |
| most negative σ over the maximum | −3 × 10⁻⁴ (T = 20 ringing below the first band of neutrals) |
| f below 13.6 eV, final over raw | −0.1 to +4.8 % (the 0.75 eV smoothing of the rise to the 17 eV hump; inside the box uncertainty) |
| rates, final over raw | Mathis −0.9 to +0.4 %; Draine 1978 −0.6 to +3.4 %; Kurucz 40 000 K −2.5 to +0.3 % |
| f to 10 keV over valence + 2 per C | 97.2–97.4 % |

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

**The user's choice (2026-10-27):** hybrid-functional positions for the neutrals, band by band; ions stay on TD-LDA.

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
| basis set | def2-SVP puts ovalene S₁ at 2.70 eV, against 2.50 for TD-B3LYP/6-311++G(2d,2p) (Wu+ 2024): **~0.2 eV too high**. HBC S₁ (B₂ᵤ, weak) is measured at 434 nm = 2.86 eV (a vibronic peak; Steglich+ 2011) against 2.88 eV in def2-SVP. Steglich's TD-B3LYP/6-31+G(d) places HBC's E₁ᵤ at 364 nm = 3.41 eV. (Their 536 against 540 nm agreement is for dibenzorubicene, not HBC) |
| matching bands by rank | works on pyrene (sTDA +0.44, +0.57, +0.60 eV for the three bands); ambiguous for C54 and up |
| matching by cumulative f | fails: the methods share f differently between bands |
| automatic overlap alignment of each band | fails, even on pyrene |

For C54H18, C96H24 and coronene TD-LDA has more bands than the hybrid has bright clusters, and the ranked shifts scatter
(C96H24: +0.27, +0.46, −0.13, +1.00 eV). The shift of the *strongest* band against sTD-DFT is +0.55–0.65 eV for the compact family
(+0.56–0.79 eV for isomers), but pyrene shows the lower bands move less (+0.27–0.38) than that, so it may overstate them.

### 5b. State of the decision

- **Evidence so far:** the lowest bands need roughly +0.15–0.25 eV (ovalene, uncertain by ±0.1) to +0.45–0.65 eV (pyrene centroids);
  HBC (TD-LDA run done) is 0.3–0.5 eV low for both its weak S₁ and its strong E₁ᵤ band. The hybrid positions carry a ~0.2 eV basis-set uncertainty (def2-SVP is too high; def2-SVPD or TZVP is still to test). For the strongest bands of C54+ nothing measured exists.
- **Options:** (1) a smooth shift curve for the neutrals set by anchors from full TD-B3LYP; (2) one uniform sTDA-based shift,
  labelled ad hoc (it fits pyrene as well as per-band does, 0.11 against 0.14 eV); (3) hand-assigned bands for a few
  molecules; (4) no correction, stating the bias. Whatever ships, the unshifted spectra stay available as an opt-in switch.
- **Next:** full TD-B3LYP of coronene, ovalene, HBC and C54H18 (running; coronene done: bright pair at 4.12 eV),
  TD-LDA of HBC, the Ne-matrix tables of Ruiterkamp+ 2002 (needs the PDF), and the 2175 Å bump as a consistency check
  (§7).

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
  (~0.2 eV too high) and a shift taken from the strongest band, so they are indicative only.

**To do with the final library:** redo their Milky Way and JADES fits with TD-LDA and with corrected positions; recheck
C150H30's width (ask the authors for their spectrum if it persists); report γ for their 0.2 eV Lorentzian and for our
profile; add the states missing from the charge analysis.
