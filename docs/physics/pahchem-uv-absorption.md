# PAH UV–visible absorption for pahchem: TD-DFT, Malloci+07 and experiment

Status on 2026-10-24, branch `feature/pahchem-anharmonic`. pahchem's default σ(E, Z) is still
{cite:t}`LiDraine2001`. That cross section is charge-blind above 4 eV and 1.57× too high at 17 eV. Its
replacement is charge-resolved real-time TD-DFT from our own Octopus runs, checked against
{cite:t}`Malloci2007` and against laboratory data. **Nothing here is a default yet.** Switching pahchem's
cross section, and starting production runs, both wait for the user's decision.

Where things are:
- **Runs:** megatron `~/pahspec`, and DIaL3 `/lustre/dirac3/scratch/dp016/dc-rodr4/pahspec`.
- **Status log:** `RAMSES_dev/pahspec/STATUS.md`.
- **Scripts:** `diagnostics/pahchem/octopus/`:
  - `make_run.py`, `queue.py` (making and queueing runs);
  - `compare_malloci.py`, `plot_progress.py`, `check_damping.py` (comparisons and plots).

## 1. Method

**Octopus settings** (real time, real space; Octopus 16.4):
- **Functional and pseudopotentials:** LDA (Perdew–Zunger) with the standard norm-conserving set.
- **Grid:** 0.25 Å spacing. This is converged: within 1 % of 0.20 Å, whereas Malloci's 0.30 Å is not, with these
  pseudopotentials.
- **Box:** every atom at least 4 Å from the edge, plus a 1 Å absorbing boundary.
- **Propagation:** a δ kick of 0.01 Å⁻¹ along each axis; AETRS, dt = 0.002 ħ/eV; T = 20 ħ/eV.

**The spectrum.** It is the sine transform of the dipole, d(t), times a window D(t):

$$\sigma(E) \propto E \int_0^T d(t)\, D(t) \sin(Et)\, dt .$$

The window sets the band profile:
- polynomial (Octopus's default): compact bands that ring, with negative lobes between bands at finite T;
- exp(−γt): Lorentzian bands of half-width γ;
- exp(−s²t²/2): Gaussian bands of standard deviation s.

**Agreement with Malloci+07 at 0.25 Å** (coronene; pyrene and ovalene are queued):
- **Neutral:** oscillator strength to 30 eV (f30) is 1.03× theirs; the bands sit within 0.05 eV.
- **Cation:** f30 is 1.035× theirs.
- **Anion:** f30 = 69.6, the value predicted from the other states. That confirms the served Malloci anion file is
  circumcoronene's (`malloci2007.INCONSISTENT`).
- **Open:** the 14–20 eV peak of the neutral sits at 16.85 eV, against Malloci's 16.2.

**DIaL3 build.** It reproduces megatron's coronene run: ground-state energy and all 54 eigenvalues identical to the
printed digits, and the dipole equal to 4 × 10⁻¹³ over 500 steps.

## 2. Malloci's spectra carry an exponential window

{cite:t}`Malloci2007` give T = 20 ħ/eV and "a resolution ħ/T = 0.05 eV", but no window (Sect. 2 of
arXiv:0707.3045, [read]). `check_damping.py` recovers it, using the fact that the inverse transform of σ/E
gives back d(t)·D(t):
- **Method check:** recovering our own polynomial window this way reproduces it to ≤ 0.03.
- **Malloci's window**, as the rms envelope of their signal over ours (insensitive to phase): exp(−γt) with
  γ = 0.21 eV (neutral coronene, our T = 10) and 0.148 eV (cation, T = 19.5). Their signal is cut at t = 20.
  That makes their bands Lorentzians of FWHM 0.3–0.4 eV.
- **Forward test:** our runs with the same window match Malloci.
  - Cation, exp(−0.15 t): every band from 0.5 to 30 eV within 3 %. With the polynomial window, 4.2–5 eV comes out
    at 0.06×.
  - Neutral, exp(−0.20 t): Mathis 1–5 eV rate 0.93×, f(0.5–2.5 eV) 0.83×, f(4.2–5 eV) 0.91×. With the polynomial
    window these are 0.75×, 0.009× and 0.17×.

**Consequence.** Malloci's absorption floor below a neutral's first band (e.g. coronene below ~3 eV) is the
Lorentzian wing of their window, not something the molecule absorbs. It carries about a quarter of the neutral's
Mathis 1–5 eV rate. So the "Malloci absorbs 2.2–3.6× more than Li & Draine at 1–5 eV" in
{doc}`pahchem-gas-coupling` §4d overstates the neutrals' visible absorption. The cations' visible bands are real:
their 1–5 eV rate is 0.98× Malloci's whatever the window.

## 3. What experiment provides

Literature survey of 2026-10-24. [read] means seen in full or in the stated tables; [abs] means abstract or a
citing paper only. Nothing in it is a digital, absolute, gas-phase cross section for cold coronene or ovalene, or
for any PAH ion.

| Molecule, charge | Phase, temperature | Range | Absolute | Data | Reference |
|---|---|---|---|---|---|
| pyrene 0 | vapour, 423–873 K | 3.5–6.2 eV | yes | MPI-Mainz Spectral Atlas (digital) | Thöny & Rossi 1997, J. Photochem. Photobiol. A 104, 25 |
| pyrene 0 | vapour, 313 K | 3.8–5.0 eV, with f | yes | Spectral Atlas (digital) | Ferguson, Reeves & Schneider 1957, Can. J. Chem. 35, 1117 |
| pyrene 0 | jet-cooled | S₂ (321 nm), f and widths | calibrated | Gredel+11 Table 1 | Rouillé+ 2004, JCP 120, 6028; Gredel+ 2011, A&A 530, A26 |
| pyrene 0, coronene 0 | gas, synchrotron | photoionisation σ to ~20 eV | yes (≤ σ_abs) | figures | Verstraete+ 1990, A&A 237, 436 [abs] |
| coronene 0 | gas | 12.4–18.6 eV | yes | figures | Léger+ 1989, IAU Symp. 135, 173 [sec, via Li & Draine 2001] |
| coronene⁺ | gas, He-tagged, ~10 K | 4570, 9438 Å bands | no (positions, widths) | figures | Hardy, Rice & Maier 2017, ApJ 836, 37 |
| pyrene⁺ | gas, cold jet | D₅ band at 436.2 nm | — | — | Biennier+ 2004, CPL 387, 287 [abs] |
| coronene⁺, ovalene 0/⁺ | Ne matrix, 4 K | UV–visible–NIR, f for coronene⁺ | f only | tables | Ehrenfreund+ 1992, A&A 259, 257; Ruiterkamp+ 2002, A&A 390, 1153 |
| ovalene 0 | jet; solid para-H₂ | S₁, S₂ origins | no | — | Amirav+ 1981, JCP 74, 3745; Weber+ 2024, 2026, JPCL |

The local copies of the Spectral Atlas files are in `RAMSES_dev/pahspec/experiment/atlas/`.

**Matrix-to-gas shifts.** Spectra in neon matrices are blue-shifted relative to the gas by ~0.25 % in energy for
neutrals and ~0.5 % for ions (Ruiterkamp+02). Examples: coronene⁺ −27 Å (A band) and −20 Å (D₃) from neon to gas
(Hardy+17). Matrices also broaden bands 4–5× (Biennier+ 2003, JCP 118, 7863).

**Anions.** No gas-phase electronic spectra were found.

**Band widths:**
- **Cold molecules: homogeneous widths are narrow.**
  - Pyrene S₂ (jet) clusters: ≈ 50 cm⁻¹ FWHM.
  - Naphthalene⁺ D₂: Lorentzian Γ = 25 cm⁻¹.
  - Pyrene⁺ D₅: 145 cm⁻¹ ≈ 0.018 eV [sec].
  - Coronene⁺ D₃: lifetime ~50 fs, i.e. ~0.013 eV.
  - So ≲ 0.02 eV for the low bands, growing to ~0.1 eV for higher ones.
- **Warm molecules: wider envelopes** (pyrene vapour).
  - At 423 K: the S₂ envelope is 0.043 eV FWHM, the 4.67 eV band 0.11 eV, the 5.36 eV band 0.28 eV.
  - At 873 K the S₂ red half-width is 0.076 eV.
  - The 0.1–0.3 eV envelopes are vibronic (Franck–Condon) structure. TD-DFT has none of it.
- **The absolute scale is uncertain.** The integrated f of pyrene over 3.4–4.2 eV is 0.25 (Thöny & Rossi, 423 K),
  0.10 (Ferguson) and 0.13 (jet), a 2× spread between sources.

**Below the first allowed band.** No measured upper limit was found for cold neutrals. Coronene's S₁ origin is
symmetry-forbidden: only Herzberg–Teller vibronic bands appear (Kunishige+ 2017, JCP 146, 044309). The other
contributions are hot bands in warm molecules and the spin-forbidden S₀→T₁, which is not quantified. Ovalene's S₁
was recently placed at ≈ 2.41 eV in the gas phase (Weber+ 2026), about 0.2 eV below earlier assignments.

**How others broaden computed spectra:**
- Malloci: the window above.
- Steglich+ 2010, 2011: Lorentzians of 5000 and 3000 cm⁻¹, stated to be for vibronic patterns, i.e. ad hoc.
- Li & Draine 2001: Drude profiles fitted to laboratory mixtures and to extinction.
- Hrodmarsson+25, Zhen+16, Montillaud+13: Malloci's spectra as served.

None derives a width.

## 4. The band profile: an explicit, empirical choice

The user's decision of 2026-10-24: long propagation, then an explicitly stated envelope, compared with experiment.
That envelope's width **cannot at present be derived from first principles**. TD-DFT gives vertical excitations,
not the Franck–Condon envelope. So it is an empirical choice, labelled as such, opt-in, and reported with its
range, under the no-ad-hoc-factors rule.

**What is planned:**
- **The profile:** a Gaussian time window, D(t) = exp(−s²t²/2), applied to the undamped dipole. A small
  Lorentzian part is allowed only where a measured lifetime width calls for it. No exponential wings.
- **Clean truncation:** exp(−s²T²/2) < 10⁻³, i.e. s ≥ 3.7/T: s ≥ 0.19 eV at T = 20, or 0.09 eV at T = 40. Finer
  resolution than the vibronic envelope has no physical meaning in TD-DFT, so T = 20 suffices unless the
  calibration needs a smaller s.
- **The width:** set from the pyrene vapour envelopes and the measured cation bands (§3), possibly as a function of
  energy, with the calibration data and the fitted width both stated.
- **What is compared** with experiment: integrated f per band and band onsets, not peak σ. Peak σ depends on the
  chosen width, while f and onsets test the TD-DFT itself.
- **Band positions:** TD-DFT/LDA position errors are reported as they are and are not absorbed into the width.

**Guards against unphysical results.** These are the conditions for any profile to ship; each becomes a test or a
diagnostic:
1. **Non-negativity by construction.** Im α(E) ≥ 0 for a stable ground state, and a Gaussian window convolves it
   with a positive kernel. So σ ≥ 0 wherever the truncation condition holds. The test: σ ≥ −10⁻³ max σ on the grid.
2. **Sum rule unchanged.** The window must not change f to 30 eV beyond the truncation error (< 0.5 %). It must also
   not change the integrated f of any band wider than a few s.
3. **No absorption created below a band.** Below the lowest computed band of a neutral, σ must fall at least as fast
   as the Gaussian. The diagnostic: the fraction of the Mathis 1–5 eV rate carried by energies more than 3 s below
   the lowest band peak. This is what separates the choice from Malloci's floor.
4. **Width sensitivity reported.** Every rate pahchem uses is recomputed with s at both ends of the measured range:
   absorption in 1–5 and 5–13.6 eV, photoionisation, photodetachment, and the PE heating efficiency ε. A rate that
   moves by more than its other uncertainties is flagged. Photodetachment near threshold is expected to be the
   sensitive case.
5. **Cold-molecule onset.** The onset may not lie below the measured gas-phase origin by more than the TD-DFT
   position error. The origins: coronene⁺ 9438 Å, ovalene S₁ 2.41 eV, pyrene S₂ 3.86 eV.

**What could later derive the envelope:**
- vibronic (Franck–Condon and Herzberg–Teller) calculations of the low S₁/S₂ and D_n bands from DFT geometries and
  frequencies;
- for higher bands, lifetime broadening from the measured internal-conversion times (~100 fs for cations).

Either would replace the empirical width band by band.

## 5. Cost and computing

The DIaL3 build is described above. A strong-scaling test on C₉₆H₂₄ (1, 2 and 4 nodes, and three layouts of
states × domains) is in progress, and its result sets the production cost for C₂₄–C₂₁₆.

A first number, C₉₆H₂₄ neutral on 1 node with states-only parallelisation: 1.5 s per step, about 190 core-s per
step. A T = 20 run would take about 530 core-h per kick.
