# PAHchem: briefing figures (2026-10-20)

```{note}
Updated after the code audit of 2026-10-20 ({doc}`pahchem-audit`):
- **Fixes applied.** Seven bugs were fixed (C1–C7), and the figures and
  numbers below come from the fixed code.
- **The emission cation share changed.** It was counted from photon heating
  only (C6); it is 0.48–0.51, not the 0.57–0.60 of the first version.
- **The other fit results are unchanged** by the fixes: rms, γ, ⟨N$_C$⟩ and
  the band shape.
```

This page goes with the figure set in `<plots dir>/pahchem/briefing/` (run
`calima-paths` to see where the plots dir resolves). It covers four things:
- what pahchem can already reproduce;
- how a model spectrum is computed and fitted to the PDRs4All Orion Bar spectra;
- what those fits imply;
- what is still wrong or unexplained.

The figures are in presentation order:
- **A**: overview;
- **B**: spectral fits;
- **C**: reproduced benchmarks;
- **D**: open problems.

Each entry gives the result, then the points to discuss. The full record is
{doc}`pahchem-spec`. The questions for collaborators are in
{doc}`pahchem-data-requests`.

```{admonition} Regenerating the set
`diagnostics/pahchem/make_briefing.sh <PAH_states dir> <West+18 ESI .xlsx>`,
run from the worktree root with `$CALIMA_PAHDB_DIR` set, rebuilds every
figure in pyCALIMA's style (`pycalima.plotting_style`).
- The benchmark runners take ~30 min.
- The spectral fits are not refitted: they are read from the cached
  `fit_pdrs4all_spectra.py --fit-shape` result (key `392313363fe8`, refitted
  2026-10-20 after the audit fixes). A refit needs the Van De Putte+25
  templates (CDS J/A+A/701/A111, in `results/pahchem/data/vdp25/`) and takes
  ~1 min with numba installed (`--workers`, one template per process).
- Figures superseded before this date are kept in
  `<plots dir>/pahchem/archive_2026-10-08/`.
```

## A. Overview

**A1 — benchmark scorecard.** These are all 131 comparisons recorded in
{doc}`pahchem-spec`, grouped by milestone. Verdicts are taken from the spec
(table: `diagnostics/pahchem/briefing_benchmarks.csv`, one row each, with the
metric, cause and spec line).

| Verdict | Count | Meaning |
|---|---|---|
| Reproduced | 63 | within the stated tolerance |
| Partly | 25 | some of it, or within a looser bound |
| Fails or open | 32 | |
| Withdrawn | 11 | |

- **The statistical-mechanics, rate and photophysics layers (M1–M3)** pass
  almost everywhere.
- **The published models (M5)** are where the open items concentrate. Almost
  every one traces back to an input the authors did not publish (section D).
- **The withdrawn items (M7)** are the band-ratio fits of 2026-10-06/07.
  Review finding C1 removed them: observed bands measured over the Chown+24
  spline were compared with raw model bands. The spectral fits of section B
  replace them.

## B. The Orion Bar spectral fits

### B1 — how a model spectrum is built

The four panels follow one molecule (C$_{119}$H$_{29}$, PAHdb UID 3300, the
largest single contributor in DF2) through the chain. These are the steps:

1. **Excitation and cascade, per charge state.**
   - Each state absorbs the stellar field: a 40 000 K Kurucz atmosphere for
     θ¹ Ori C, times the Li & Draine (2001) cross section.
   - Every absorbed photon starts the exact microcanonical IR cascade on that
     state's PAHdb v4.00 modes (`emission.cascade_photons`), folded with the
     state's heating spectrum.
   - Recombination heating is included: each recombination leaves the
     ionisation energy in vibrations.
   - Single-photon heating holds at these G$_0$, so the result is a fixed
     basis at G$_0$ = 1, scaled by G$_0$.
   - Panel 1 shows the IR power per mode for the neutral and the cation. The
     cation puts its power into the 6–9 µm CC modes, the neutral into the
     11.2 µm CH out-of-plane modes.
2. **Band profile.** Each mode becomes a band with a shift and a width per
   mode class (CH stretch, CC, CH out-of-plane, other).
   - The shifts and widths stand for anharmonicity at the emission
     temperature. One set is shared by every molecule and fitted per
     template (`spectral_fit.BandShape`).
   - Gaussians fit better than Lorentzians in all five templates.
   - The DFT CH-stretch intensities are scaled down by the laboratory factor
     $10^{-0.29}$ (matrix-isolation band strengths, 59 species).
3. **Charge balance from the stated local conditions.** The charges are
   computed from conditions that pahchem does not predict: G$_0$, n$_{\rm H}$,
   n$_e$ and T are read from `environments/orion_bar.toml`, each with its source.
   - A chain over charge states balances photoionisation (Jochims+96 yield;
     Wenzel+20 for cations with N$_{\rm C}$ ≥ 32) against electron capture
     (BT94; statistical sticking for neutrals).
   - The IPs are calibrated: the v4.00 IPs and EAs are low against measured
     values, so they are shifted by +0.37 eV (anion), +0.60 eV (neutral) and
     +0.64 eV (cation and dication).
   - The molecule's spectrum is the f-weighted sum of its states (panel 3).
   - Each molecule is also passed through a dehydrogenation chain at the same
     conditions. It is kept only where its intact fraction is ≥ 0.5.
4. **Fit.** One non-negative least squares over the whole library: 1424
   PAHdb v4.00 molecules, N$_{\rm C}$ = 22–268.
   - Only the abundances are fitted.
   - The continuum is fitted at the same time: positive cubic B-splines with
     knots 1.5 µm apart, set separately in each segment.
   - Fitted range: 3.1–3.65 µm (NIRSpec) and 5–15 µm (MRS), on a uniform
     2.5 cm$^{-1}$ grid, so every cm$^{-1}$ weighs the same.
   - Gas lines are removed with running medians and masked. The Chown+24
     artefacts (12.2, 13.7–13.95, 14.3 µm) are masked.
   - This is repeated over a 4 × 3 × 4 grid of each template's stated G$_0$,
     n$_{\rm H}$ and n$_e$ (or x$_e$) ranges. The band shape is refitted at
     the best cell and the scan repeated.

The point of this design is that the observed spectrum and every model go
through the same continuum rule. Band ratios over an observer's spline could
not do that (finding C1).

### B2–B6 — the five templates

Each figure has three parts:
- **Spectrum.** The observed spectrum (black, gas lines removed), the fitted
  continuum (grey) and the PAH emission stacked by charge: anions, neutrals,
  cations, dications and higher. The thin dark line is the model total.
- **Residual**, in per cent of the segment's mean intensity.
- **Header:** the best cell of the stated conditions and the fitted band
  shape.

| | H II | Atomic | DF1 | DF2 | DF3 |
|---|---|---|---|---|---|
| rms residual (% of mean I$_\nu$) | 4.8 | 10.2 | 9.1 | 6.4 | 5.5 |
| best γ = G$_0$T$^{1/2}$/n$_e$ | 9.8e3 | 1.0e4 | 1.0e4 | 9.9e3 | 1.1e4 |
| cation share of the PAH emission (Z ≥ +1) | 0.48 | 0.51 | 0.50 | 0.49 | 0.51 |
| of which dications and higher | 0.04 | 0.04 | 0.04 | 0.04 | 0.04 |
| ⟨N$_{\rm C}$⟩, carbon-weighted | 84 | 87 | 84 | 85 | 83 |
| continuum / (continuum + PAH), 5–15 µm | 0.71 | 0.39 | 0.44 | 0.55 | 0.63 |
| molecules with w > 0 | 81 | 80 | 79 | 62 | 55 |
| CH stretch shift / FWHM (cm$^{-1}$) | −29 / 43 | −29 / 36 | −29 / 37 | −29 / 32 | −28 / 34 |
| CC shift, CH oop shift, other FWHM (cm$^{-1}$) | +17, +2, 8 | +18, +1, 6 | +18, 0, 7 | +20, +3, 11 | +19, +1, 12 |

What to look at:
- **The 6.2, 7.7, 8.6 and 11.2 µm bands are carried by cations and neutrals
  as expected.** The 7.7 µm complex is mostly cations; 11.2 µm is mostly
  neutrals.
- **The 3.4 µm complex is not in the model.** It is the large positive
  residual at 3.40–3.6 µm in every template. The PAHdb aliphatic species
  cannot produce the observed 3.4/3.29 (section D, D08).
- **The 3.25 µm shoulder** of the 3.29 band is over-predicted (negative
  residual).
- **The Atomic PDR 8.6 µm band** is under-predicted. This is the worst fit
  (10 %).
- **The residuals at 5–6 µm** are larger, but the intensity there is low.

### B7 — what the fits imply

The panels show, per template:
- the rms;
- γ: the scanned range (grey), the allowed cells (blue) and the best cell;
- the cation share of the emitted PAH power;
- ⟨N$_{\rm C}$⟩;
- the continuum share;
- the number of molecules in the fit.

Results:
- **One band shape and one physics fit all five templates** to 5–10 %.
- **The best γ is ≈ 1 × 10⁴ everywhere**, inside the scanned range
  (5 × 10³ to 1.5–2.6 × 10⁵), and the Δχ² < 9 cells are narrow.
- **Cations carry 0.48–0.51 of the emission and ⟨N$_{\rm C}$⟩ ≈ 83–87 in every
  template.** The share counts all the emitted power, photon heating plus
  recombination heating. Counted from photon heating alone (as first
  reported) it was 0.57–0.60, because recombination heats neutrals much more
  than cations (audit fix C6).

Caveats to raise:
- **The best G$_0$ sits at the low end of the stated range** in H II, Atomic,
  DF1 and DF2 (1.8 × 10⁴ in the PDR templates, 10³ at the fronts). So the
  constancy of γ partly reflects the fit wanting less ionisation than the
  stated G$_0$ allows.
- **Δχ² < 9 is an approximate scale.** It uses s² = sse$_{\min}$/(n − p),
  with no calibrated uncertainties (step 3 of the review). So the allowed
  ranges are indicative only.
- **The continuum is fitted, not measured.** Its share (0.4–0.7) is degenerate
  with the band wings. The knot spacing is the recipe systematic.
- **The CC shift of +17 to +20 cm$^{-1}$ is a blue shift.** It cannot be
  thermal (Mai+25 MLMD laws give red shifts), so it points at the carriers:
  cations or N substitution. The CH and out-of-plane shifts are consistent
  with the MLMD temperature laws.
- **These fits do not test the charge physics by themselves.** Over the stated
  conditions the model's charge-sensitive ratios already span all the
  observed values. The test is that the γ the fit picks agrees with the
  stated conditions.

## C. Reproduced benchmarks

Each figure overlays pahchem on the published curve. The thick translucent
curve is the paper's, the dashed line or marker is pahchem's, computed from
the paper's own stated inputs.

- **C01 — West+18 iPEPICO RRKM** (7 reactions of PAH cations). This is a code
  check: with their B3LYP frequencies, transition states and Table 1 E$_0$,
  `rates.rrkm` gives their k(E) to a median 0.0002–0.01 dex.
- **C02 — Montillaud+13 Fig. 3** (coronene k(E) with their A and E$_0$).
  - The match is to a near-constant −0.11 to −0.26 dex.
  - That offset is expected: they used Malloci modes, we use PAHdb neutral
    modes.
- **C03 — Andrews+16 Fig. 5** (H and H$_2$ loss and k$_{\rm IR}$ for
  C$_{24}$H$_{12}$, C$_{54}$H$_{18}$, C$_{96}$H$_{24}$).
  - C$_{24}$ and C$_{54}$ are within a median 0.05–0.17 dex.
  - C$_{96}$ sits ~0.4 dex high, a strict xfail with no explanation yet.
  - k$_{\rm IR}$ is within 0.26 dex.
- **C04 — Andrews+16 Table B is their sink-free eq. 7.**
  - Table B and Fig. 6 count events per absorbed photon. They are not
    per-molecule rates; the network uses r$_{\rm abs}$P ≤ r$_{\rm abs}$.
  - This is why every earlier reproduction failed.
  - Emulating eq. 7 gives Table B to ~0.5 dex, but only if ionising photons
    heat as well.
- **C05 — Allain+96 II Fig. 10** (hydrogenation against size, Orion Bar and
  NGC 7027).
  - Within 1.6 and 3.4 points from 24 C up.
  - The 10–16 C Orion Bar PAHs are 10–19 points too dehydrogenated (strict
    xfail).
- **C06 — Le Page+01 Fig. 10** (H-loss photodissociation rate of six classes
  against size). Within 0.21 dex for 6–150 C.
- **C07 — Murga+22 charge along the Orion Bar** against their Figshare f(a, Z).
  - The modal charge agrees in 92–100 % of the points.
  - It reproduces only with an inferred configuration: s$_e$ = 1, MMP83
    field, no ions.
  - The H II region needs an EUV component (Kurucz 40 kK × 4, our inference).
- **C08 — exact cascade against the AmesPAHdb thermal cascade.**
  - All bands but 3.3 µm agree to 1 % at 4–12 eV.
  - The 3.3 µm difference is expected physics: the thermal approximation
    over-predicts it at low energy.
- **C09 — Maragkoudakis+20 Fig. 2**, molecule by molecule (120 neutrals, 116
  cations, 35 anions).
  - The median log(ours/paper) is +0.001.
  - This needs the suite's own cross section.
- **C10 — recurrent fluorescence against Borja+26** (naphthalene⁺,
  anthracene⁺, pyrene⁺).
  - The allowed transitions are within 0.18 dex.
  - The forbidden D1 is −0.14 to +0.49 dex off. Their Herzberg–Teller and
    Duschinsky treatment is beyond the band rule used here.
- **C11–C13 — cosmic-ray collisions (primary CRs).** Gates G1 (stopping
  against NIST polystyrene, 1–7 %) and G4 (Micelotta+11: crossover sizes and
  Table 2 lifetimes) pass. The figures show three things:
  - **C11, the energy left per passage:** ~9 eV mean for protons and
    electrons; CNO and Fe-group nuclei reach hundreds of eV.
  - **C12, lifetimes against N(H$_2$):** C$_{54}$H$_{18}$ 17–66 Myr (L) and
    0.8–1.0 Myr (H) unattenuated; 300–450 Myr (L) at 10$^{22}$ cm$^{-2}$.
  - **C13, the CR-excited IR spectrum:** ionised products show a hot-cation
    signature, 3.3/11.2 × 1.65–1.78 against UV excitation at the same charge.
- **C14 — the secondary-electron spectrum per ζ** from CRISpy's Spencer–Fano
  degradation in H$_2$.
  - It passes the CSDA check against NIST ESTAR (0.98–1.00 at 10–30 keV).
  - Its ζ$_{\rm sec}$/ζ$_{\rm prim}$ = 0.75–0.95 matches Ivlev+21's 0.8–1.1.
  - Coulomb losses empty the sub-10 eV electrons for x$_e$ ≳ 10⁻⁵.
  - It is not yet used by the collision code.

## D. Errors and inconsistencies

Most of these are not errors in pahchem. They come from inputs the published
models did not state, and each has a request in {doc}`pahchem-data-requests`.
D01, D02 and D04 should be read as order-of-magnitude benchmarks, not as
tuning targets.

- **D01 — Montillaud+13 Fig. 10** (where C$_{54}$H$_{18}$ is totally
  dehydrogenated, in G$_0$–n$_{\rm H}$).
  - With parent modes for every state, which is their stated model, our
    contours sit +1.0 to +1.5 dex higher in G$_0$.
  - With modes stripped per missing H, they sit +0.3 to +0.8 dex higher
    (+0.28/+0.59/+0.84 at 90/50/10 %). The earlier +0.0 to +0.6 came from the
    old stripping, which removed the most intense C–H modes first (audit fix
    C1).
  - E$_{\rm th}$(C$_{54}$H$_{18}$) is 19.5 eV against their 17 eV. With
    k$_{\rm IR}$ taken as the energy-loss rate (dE/dt)/E and cation modes, it
    matches theirs to 0.25 eV for C$_{24}$, C$_{54}$ and C$_{66}$. That
    definition explains 0.4–0.55 dex of the Fig. 10 gap
    ({doc}`pahchem-audit`, section 2).
  - *Needs:* whether their k$_{\rm IR}$ is the energy e-folding rate, and
    their C$_{54}$ modes (request to Joblin/Montillaud).
- **D02 — Andrews+16 NGC 7023** (Fig. 7, hydrogenation; Fig. 8, charge).
  - We keep C$_{54}$H$_{18}$ ≥ 97 % normally hydrogenated. Their Fig. 7 needs
    a k$_{\rm IR}$ ~10³ smaller in their eq. 8.
  - The charge has the right shape but is shifted 2–3× in γ.
  - *Needs:* the G$_n$ normalisation, the temperature range in eq. 8, their
    s$_e$ values (request to Candian/Tielens).
- **D03 — Murga+20 Fig. 4** (C$_2$H$_2$ produced per molecule).
  - Full destruction (Orion A$_V$ = 0.1, neutrals) is exact.
  - Partial destruction is 1–4 dex high (rms 2.5 dex).
  - With channel competition (dotted) it improves to 1.2 dex rms, but
    Murga+22 contradicts that.
  - The two papers also need opposite readings of SHIVA's p(E).
  - *Needs:* author answers.
- **D04 — Murga+22 model b** (hydrogenation through the Orion Bar),
  **largely resolved** by audit fix C3.
  - 3.3 Å PAHs agree within 0.03.
  - **The fix.** The Figshare chains fix M22's rates: the bin step is
    0.5 N$_{H0}$, Eley–Rideal removes 2 H, and the addition laws differ from
    M20's printed ones. With these rates, PAHs ≥ 6 Å stay normal as in
    Figshare: ⟨X$_H$/X$_{H0}$⟩ within +0.00 to +0.05 in PDR 1, DF and MC.
    With M20's rates as printed they were 60–90 % super-hydrogenated.
  - **What remains:**
    - PDR 2, +0.12 to +0.17 at 6–8.7 Å;
    - 5 Å in DF and MC, 0.18 more dehydrogenated.
  - *Needs:* the authors to confirm the bin step and the 2-H Eley–Rideal
    event, and the PDR 2 super-H loss.
- **D05 — West+18 against Jochims+94 H-loss thresholds.** The two
  laboratories differ by 0.7–1.4 eV at k = 10⁴ s⁻¹. That is a laboratory
  systematic on the H-loss E$_0$, larger than either's quoted error, and it is
  carried as an uncertainty.
- **D06 — gate G2, electron-impact ionisation** (improved, not closed).
  - Reference: our BEB(HF). It reproduces Krishnadas+24's 28.1 and 31.0 Å²
    peaks to 2 %; the Appl. Sci. OVGF values are 12–15 % higher.
  - **Fixed.** The slow-electron denominator was Kim's excitation form,
    T + IP + W, used for ionisation too. Ionising collisions now use Kim's
    ionisation form, T + B + U with U = 2B (audit C5), and σ below N$_C$ = 21
    scales with volume (C4).
  - **Effect.** pahchem/BEB over 15 eV–5 keV is now 0.99–1.41 (anthracene),
    1.10–1.55 (coronene) and 1.39–1.80 (C$_{54}$H$_{18}$), against
    1.28–2.58 before.
  - **What remains** at keV (×1.2–1.4) is the Li & Draine dipole strength,
    which is 1.2–1.5× the TD-DFT cross section at the ionising energies.
  - Energy deposition (G1) is unaffected.
  - *Decision pending:* adopt BEB for the ionisation rate, or scale the
    Li & Draine σ.
- **D07 — sensitivity ranking** (C$_{54}$H$_{18}$, NGC 7023 edge and Orion Bar
  atomic PDR; one input varied at a time).
  - Four inputs each move dehydrogenation by ~1 dex: per-state modes, UV
    cross section, H/H$_2$-loss barriers (lowering them matters more than
    raising them), IR/RF cooling.
  - Charge is set by the cross section and electron capture.
  - This ranking orders the DFT requests (T1.1 per-state modes, T2 barriers).
- **D08 — the 3.4 µm aliphatic complex** (kept in its original style: the
  templates are needed to redraw it).
  - The PAHdb v4.00 aliphatic species give 3.4/3.29 ≈ 0.01–0.02 (library
    median; the best single species, C$_{96}$H$_{28}$, gives 0.64), against
    the observed 0.35–0.51.
  - This is a gap in the library, not a band-profile problem.
  - *Needs:* aliphatic carrier spectra (Oxford group, Thatte+26).

Other open items without a figure here: the ERE from cation RF has the wrong
trend with U; the end-to-end check of the RAMSES tables (M6) has not been
run; and the WD01 charge cross-check has not been done.

## Points for discussion

1. **Montillaud, Andrews and Murga.** Do we accept order-of-magnitude
   agreement where the inputs are unpublished, and close those three with
   the author questions rather than more modelling?
2. **The Orion Bar fits.**
   - Is γ ≈ 10⁴ with G$_0$ at the low end of the stated range a result, or a
     sign that the stated G$_0$ (from O I fluorescence for the Atomic PDR,
     assumed for the DFs) is too high for the PAH layer?
   - Next step: calibrated uncertainties (step 3), then per-pixel fits.
3. **The CC blue shift and the 3.4 µm gap** both point at missing carriers.
   They set the priority of the DFT requests: aliphatic side groups on large
   PAHs, cations, N-substituted species.
4. **Gate G2.** Adopt BEB(HF) for the electron dz = +1 rate, or fix the
   model's dipole strength and denominator?
5. **The secondary electrons.** When should they go into the collision rates,
   given that G2 sets how many of them ionise?
