# PAHchem: audit of the failed benchmarks (2026-10-20)

The question was: before asking collaborators and authors for missing inputs,
does pahchem itself contain bugs or physical errors behind the benchmarks that
fail ({doc}`pahchem-spec`; 33 open and 24 partial in
`diagnostics/pahchem/briefing_benchmarks.csv`)?

**Method.** Seven read-only audits were run in parallel. Each covered a code
area shared by many benchmarks, not a single paper:
- unimolecular rates;
- the energy master equation and cooling;
- cross sections and charge;
- the H network;
- emission and the spectral fits;
- cosmic-ray collisions;
- constants and conventions.

Every finding marked confirmed was shown numerically. The main ones were
checked a second time, either by a second audit reaching the same result
independently or by a re-run here. The audit scripts are kept in
`results/pahchem/audit_2026-10-20/<area>/`. Fixes C1–C7 were applied the same day; their
effect on every benchmark is in section 5.

**Summary.**
- The core engine is sound:
  - state counting, the rate forms and their constants;
  - the master equation and its numerics;
  - Einstein A coefficients;
  - field integrals and every G$_0$ conversion;
  - electron capture;
  - the 4π conventions;
  - the units from PAHdb intensities to I$_\nu$.
- There are, however, **seven confirmed bugs or inconsistencies** that move
  benchmarks (section 1), and **several stated causes of failures that are
  wrong** (section 2).
- Some author questions can therefore be answered or narrowed before they are
  sent (section 4).

## 1. Confirmed bugs and inconsistencies

**C1. Dehydrogenated states lose their most intense C–H modes first.**
- Where: `molecule.py` (`PAHSpecies.stripped`) and `allain96.strip_h_modes`.
- **Cause.** These functions remove the most intense C–H modes first. In
  PAHdb a few symmetric combinations carry nearly all the C–H stretch
  intensity: in coronene the top two stretches carry 95 %.
- **Effect on intensity.** Taking away two H makes the C–H stretch A go from
  303 to 16 s⁻¹, against 252 if A were scaled with N$_H$; total A goes from
  339 to 29 s⁻¹. For C$_{54}$H$_{18}$ at N$_H$ = 12 the stretch emission is
  zero.
- **Effect on rates.** Dehydrogenated states cool 2–6× too slowly. Multiphoton
  H loss is too fast, by 3.5× for C$_{54}$H$_{16}$ at G$_0$ = 10³ (Mathis).
- Found by two audits independently and re-checked here.
- **Affects:**
  - the population intact fractions (the H-loss tables for N$_H$ = N$_{H0}$−1
    to −3), and so which molecules enter the Orion Bar fits;
  - the baseline and Montillaud "stripped" variants, where about 0.35 dex of
    the variant's apparent improvement on Fig. 10 is this artefact;
  - Allain's dehydrogenated k$_{\rm IR}$. The Allain runner also breaks its
    own card, which states Allain's rule: band A scaled by the number of C–H
    bonds.
- **Fix:** keep the remaining modes, but rescale each C–H window to the parent
  window sum × N$_H$/N$_{H0}$, as the Allain card says. Per-state DFT modes
  remain the real answer (request T1.1).

**C2. SHIVA's post-absorption yield is paired with the wrong absorption rate.**
- Where: `shiva.py`.
- **Cause.** The yield `p_post` is built from every heating photon, from
  0.5 eV up. It is then multiplied by the absorption rate above 5 eV only.
  Yet 47–64 % of the heating absorptions are below 5 eV.
- **Effect:** every "post" rate is 1.8–2.8× too low (C$_{17}$–C$_{161}$).
- Found by two audits independently; the code pairing was confirmed here.
- **Affects:**
  - murga20 partial destruction (normal channels only);
  - the M20/M22 p(E) conflict, which shifts by 0.3–0.45 dex;
  - murga22_deepmc_rate, which **gets worse** (theirs/ours 0.4–0.7) unless
    C3 is fixed with it.
- **Fix:** build `p_post` from the absorptions above 5 eV, or pair it with
  the all-photon rate.

**C3. The Murga+22 model-b rates differ from the Figshare chains.**
- **The files fix the rates.** M22's addition and loss rates fit the
  published chains to 1–2 % at every location and size from 6 to 19.9 Å.
  Re-run here for the cations at three locations: fitted/law = 1.00–1.04
  wherever the bins are populated.
- They differ from `murga2022.py` in three places:

  | | Figshare chains | `murga2022.py` |
  |---|---|---|
  | Bin step | Δ = 0.5 N$_{H0}$ | 0.371 N$_{H0}$ |
  | Ion normal → super-H addition | 0.5(1.4e-10 N$_H$/N$_C$ + 5e-11) n$_H$/N$_C$ | ~0.82 N$_C$ times faster |
  | H removed per Eley–Rideal event | 2 | 1 |

- **Neutrals.** Their addition law is R = c n$_H$[0.5·½(k$_1$+k$_4$) +
  N$_C$k$_{3a1}$], where k$_1$, k$_4$ and k$_{3a1}$ are the Goumans site rates
  used by `murga2022.py`. Compared with that, pahchem's is N$_{H0}$ times too
  fast below ~1000 K and 2–3× too slow above ~2000 K.
- **Effect:** with these laws the ≥ 6 Å super-hydrogenation excess (strict
  xfail) closes, from +0.32 to +0.49 in ⟨X$_H$/X$_{H0}$⟩ down to 0.00–0.05.
  PDR 2 is the exception at +0.05–0.21: the top bin of the 5–8.7 Å neutrals.
- **The module docstring is wrong** where it says the files do not reveal
  these rates.
- **Fix:** adopt the laws as readings of the files (confidence D), not as
  M20's printed equations.

**C4. The Li & Draine cross section is clamped in size below N$_C$ = 21.**
- Where: `pah_oppacity.py:150` (reached through `photo.sigma_li_draine`,
  `shiva`, `murga_charge`).
- **Cause.** Q$_{\rm abs}$ is held at the smallest table radius (N$_C$ =
  20.9), but C$_{\rm abs}$ = Q πa² still scales with a².
- **Effect:** σ per C is too high by (20.9/N$_C$)$^{1/3}$:
  - C$_{10}$ 1.27;
  - C$_{14}$ 1.14;
  - C$_{16}$ 1.09.

  Re-checked here, against an expected 1.279/1.143/1.093.
- **Affects:**
  - gate G2 for C$_{10}$–C$_{20}$, where it is 1.13 of the 1.28 dipole excess
    of anthracene;
  - LP03 with the LD option at C$_{16}$ (+9 %);
  - the Murga 3.3 Å grains;
  - small-N$_C$ chains.

  The population, baseline and Andrews runs (N$_C$ ≥ 22) are not affected.
  It is hidden from the TRK test, which covers C$_{24}$ and above.
- **Fix:** σ = N$_C$ σ$_{\rm per\,C}$(a$_{\min}$) below a$_{\min}$, and a TRK
  test at C$_{14}$.

**C5. The electron-impact denominator uses the excitation form for ionisation.**
- Where: `impact.py:227`.
- **Cause.** T/(T + IP + W) is Kim's excitation scaling. For ionisation Kim's
  form is T/(T + B + U), with U ≈ 2B for the PAH valence orbitals (HF). The
  two forms coincide for H (U = B), which is why the scheme passes for H.
- **Effect of the fix.** Applying T/(T + B + U(B)) to ionising collisions only:

  | | before | after |
  |---|---|---|
  | model/BEB (C$_{10}$–C$_{26}$, coronene), 15 eV–5 keV | 1.3–2.4 | 0.9–1.5 |
  | model/BEB (C$_{54}$H$_{18}$) | 1.40–2.77 | 1.12–1.64 |
  | peak position | 54 eV | 67 eV (BEB 72 eV) |
  | stopping vs Joy carbon at 50 eV | 1.26 | 0.98 |

- Applied to all collisions it over-corrects the stopping, so the split
  matters.
- **The keV residual.** What remains above 1 keV (×1.3) is not a method bug:
  the method is within 0.6–8 % of an exact first-Born calculation for atomic
  H, while BEB(Q = 1) is itself 1.22–1.34× above Born for H. That residual is
  the Li & Draine strength (section 3).

**C6. The cation share of the emission leaves out recombination heating.**
- Where: `fit_pdrs4all_spectra.py`, `explore_maragkoudakis2026.py`.
- **Cause.** The share uses photon-absorbed power, but the emitted power also
  contains recombination heating: neutrals 0.49× of their photon heating,
  anions ~1×, cations 0.17×.
- **Effect:** at the best cells the emitted-power cation share is 0.49–0.51,
  **not 0.57–0.60** as stated in {doc}`pahchem-briefing` (B table, B7) and in
  the spec's spectral-fit result. The neutral share is 0.48–0.51.
- This also changes the M26 cation-share comparison.
- **Fix:** use `basis.emitted` instead of `basis.absorbed`.

**C7. The family benchmarks use the raw DFT CH-stretch intensities.**
- Where: `family_emission.py`, `baseline.py`.
- **Cause.** The population fits apply the laboratory scale 10$^{-0.29}$ to
  the CH stretch; these benchmarks do not.
- **Effect on 3.3/11.2:**

  | | neutral | cation |
  |---|---|---|
  | C$_{54}$H$_{18}$, DFT → lab-scaled | 1.17 → 0.68 | 0.57 → 0.30 |
  | C$_{96}$H$_{24}$, DFT → lab-scaled | 0.49 → 0.27 | 0.30 → 0.16 |

- **This flips both PDRs4All family conclusions.** The "C$_{54}$ 3.3/11.2
  1.6–2× high" miss disappears, and the C$_{96}$ "0.32 vs 0.29" match becomes
  ~1.7× low.
- These benchmarks also still measure the observation over the Chown spline,
  the C1 mismatch that withdrew the band-ratio fits. Against the
  spectral-fit decomposition:
  - the Atomic 6.2/11.2 "match" becomes 1.0 vs 1.4;
  - the 7.7/11.2 shortfall grows to ~2.3×;
  - the 12.7/11.2 shortfall shrinks to 1.6–2×.

  Both rows (pdrs4all_least_ionising, pdrs4all_robust_misses) should be
  re-scored, and m7d_edge_structure is stale. From the spectral-fit weights,
  solo/trio is 1.2–1.4 (Khan 1.5) and solo/quartet 13–20 (Khan 22–46); only
  solo/duo (0.7 vs 14–17) remains a real disagreement.

## 2. Stated causes that the audit does not support

**Montillaud+13 (E$_{\rm th}$ and Fig. 10).**
- **The E$_{\rm th}$ gate uses the wrong k$_{\rm IR}$.** It takes k$_{\rm IR}$
  as the photon-emission rate. With the energy-loss rate (dE/dt)/E and cation
  modes (the card says their k$_{\rm IR}$ is for cations), E$_{\rm th}$
  matches theirs to 0.25 eV:

  | | ours | Montillaud |
  |---|---|---|
  | C$_{24}$ | 9.35 eV | 9.6 eV |
  | C$_{54}$ | 17.15 eV | 17.0 eV |
  | C$_{66}$ | 19.5 eV (neutral modes) | 20.0 eV |

  With the photon rate the values are 10.15, 19.15 and 22.75 eV. Their
  implied k$_{\rm IR}$ falls with size, which no photon rate does.
  E$_{\rm th}$ is therefore **not evidence** that their C$_{54}$ k(E) is
  faster.
- **The Fig. 3 numbers have changed.** Our coronene k(E) now matches their
  Fig. 3 to −0.04…+0.02 dex, not the stale −0.11 to −0.26 in the spec.
- **About 0.4–0.55 dex of the Fig. 10 gap is the k$_{\rm IR}$ definition.** A
  k$_{\rm IR}$ ~6× lower moves the 50 % contour from +1.2 to +0.8 dex.
- **The rest is unexplained.** About +0.7 dex remains, and per C1 the
  stripped variant is +0.5–0.7 dex once its artefact is removed.
- **Question for the authors, now narrower:** is your k$_{\rm IR}$ the energy
  e-folding rate, and which C$_{54}$ modes did you use?

**Allain+96, PAH 50 and 100 C.**
- The xfail reason (an unstated N$_{H0}$) does not hold: closing the gap
  would need N$_{H0}$ ≈ 15–16, below the compact minimum of 18.
- The cause is the reading of "multi-quanta neglected" above 24 C, which
  currently takes k$_{\rm IR}$ = ΣA literally. Keeping the occupation factor
  at all sizes, plus the card's proportional stripping (C1), gives
  (dex, ours − theirs):

  | Column | before | after |
  |---|---|---|
  | normal cation | −1.34 | −0.29 |
  | α$_H$ = 0.5 | −0.13 | −0.10 |
  | α$_H$ = 0.25 | +0.02 | +0.16 |
  | I Table 4 | −1.39 | −0.34 |

- The cost is the PAH 100 C dehydrogenated columns, which become +0.4 to
  +0.7 dex.
- **Allain II Fig. 10, small PAHs.** Neither mode stripping nor band-A
  scaling explains it (≤ 0.06 moved).

**MW22 Fig. 1.**
- **Laplace form.** The −2 to −4 dex is MW22's convention: their density of
  states counts energy from the potential minimum, (E + E$_z$). With that
  form we reproduce them to −0.4…+0.5 dex (a = 0.95: −0.03 to −0.40). Our
  exact form is right; Whitten–Rabinovitch confirms it.
- **RRK at χ = 1.** The stated cause, continuous cooling, is refuted. A
  discrete-photon cascade moves it by only +0.22 dex. The real mismatch is a
  slope: theirs ∝ χ$^{1.0}$, ours ∝ χ$^{2.0}$ (two-photon). The cause is
  unknown.

**Le Page.**
- The size trend of the LP03 k$_{\rm ion}$ (1.34 → 1.06 from C$_{16}$ to
  C$_{100}$) is the same for every IP law (±0.04), so it is not "unstated
  IPs". The naphthalene⁺ E$_0$ of 4.2 eV remains needed: their ΔS adds only
  +0.1 dex.

**Andrews+16.**
- **Fig. 7 (k$_{\rm IR}$ ~10³).** No bookkeeping error. Neither alternative
  reading of eq. 8 works; it remains open.
- **Fig. 5, C$_{96}$ (+0.37 dex).** No code error found. The offset grows with
  size, and C$_{96}$ also sits +0.22 dex high in k$_{\rm IR}$, which hints at a
  y-calibration offset in that digitised panel.
- **Fig. 8, charge (2–3× in γ).** Two effects found independently, which
  together remove most of it (C$_{54}$ cation fraction at A$_V$ = 0.05: 0.45 vs
  their 0.48):
  - the edge G$_0$ of their Fig. 3 profile is 1.64× the Table 1 value, close
    to the Draine/Habing ratio of 1.67. **Question for the authors:** which
    G$_0$ unit scales the Kurucz field?
  - photodetachment counts photons only from 5 eV and uses the ionisation
    ramp from the EA (finding P1 in section 3).

## 3. Physics and input issues (not code bugs)

These are measured, but their resolution is a modelling choice.

- **P1. Photon grid and photodetachment.**
  - Photons start at 5 eV everywhere (`E_PH`).
  - **Absorption missed below 5 eV:** 10 % of absorptions under a
    40 000 K Kurucz field, 45 % under 15 000 K, and more than 100 % extra
    under the Mathis field.
  - **Detachment.** The yield is the ionisation ramp counted from the EA. The
    alternative, Y ≈ 1 from the EA, gives 2.3× the detachment rate for
    C$_{54}^-$ under 15 000 K.
  - **Andrews C$_{96}$ ladder.** Its EA of 0.56 eV is most likely the
    Z = −2 step of the BT94 disc law; the −1 step is 3.12 eV. That is
    ~0.1 dex in the C$_{96}$ anion fraction. Re-read Table A.1.
- **P2. Li & Draine σ against TD-DFT.**
  - **The size of the excess.** Li & Draine σ is 1.2–1.5× Malloci TD-DFT at
    the ionising energies (2× for C$_{54}^+$ at 8–10 eV), and 1.25–1.5× in
    k$_{\rm ion}$.
  - **What it explains:**
    - the keV G2 residual, where the anthracene dipole strength goes from
      35.7 to ≈ 27.5 against BEB's 27.9;
    - static polarisabilities 18–30 % above the literature values the audit
      recalled (to be confirmed);
    - over-ionisation of 1.25–1.5× in γ, in the direction of the spectral
      fits' G$_0$ sitting at the low edge.
  - **Next step:** a σ-scale ≈ 0.75 variant, or TD-DFT where it exists.
- **P3. Leftover photoionisation energy.** The ion receives none of
  E − IP − KE. With ⟨E − IP⟩/2 the deposit into Z ≥ 1 rises by ~20 %, and
  fewer cations are needed for the same spectrum.
- **P4. Small items:**
  - **Micelotta eq. 31.** Dropping the factor 0.5 is physically right: the
    McKinley–Feshbach cross section is per nucleus, so it isn't projected.
  - **Heavy ions.** Their effective charge should be taken relative to the
    proton's (×1.42 at 0.1 MeV/u).
  - **Malloci coronene anion file.** It breaks the TRK sum rule by 2.1×. It
    is unused, but must not be used for detachment as it stands.
  - **Coronene⁺ RF level at 0.6 eV.** Probably TD-DFT ringing; the comment in
    `baseline.py` lists 3 levels while the code yields 4.
  - **ERE efficiency.** It uses a different denominator from the
    observations (3–10×). That rescales every environment but does not fix
    the wrong trend.
  - **3.4 µm target.** The observed 0.35–0.51 includes the plateau. Against
    the band-only target (0.09–0.18) the library falls short by 10–20×, not
    40–60×.
  - **Spectral-fit bounds.** The Lorentzian fits sit at the `fwhm_rest`
    bound, and DF2's CC shift at its +20 cm⁻¹ bound.
  - **Odd-N$_H$ library species.** They get the even-H rate in the population
    basis.
  - **PDG He/H abundance.** Worth checking against the printed table.
- **Confirmed real, not artefacts:**
  - **CC blue shift.** The sign is right, both halves of the CC class want
    the same +17–20 cm⁻¹, and it is additive rather than a scale-factor
    error.
  - **3.25 µm shoulder.** It comes from the blue DFT stretches under a single
    CH shift.

## 4. What this changes before writing to authors

**Questions that can be dropped or narrowed:**
- **Murga, super-H at ≥ 6 Å.** The rates are fixed by the files (C3). Ask
  only for confirmation of the bin step and of the 2-H Eley–Rideal event.
- **Montillaud.** Ask whether k$_{\rm IR}$ is the energy e-folding rate, and
  which C$_{54}$ modes they used. Leave out the k(E) question for coronene,
  which now matches.
- **Allain.** Ask about the meaning of "multi-quanta neglected", not
  N$_{H0}$.

**Questions to add:**
- **Andrews:** the G$_0$ unit of the NGC 7023 profile, and the C$_{96}$ EA.

**Still open after the fixes, needing author inputs:**
- Andrews Fig. 7 and Fig. 5 C$_{96}$;
- the rest of Montillaud Fig. 10 (~0.7 dex);
- the M20 partial destruction of cations and small PAHs (the p(E) conflict);
- Allain's small-PAH hydrogenation;
- the MW22 RRK slope;
- the Le Page k$_{\rm ion}$ size trend and the naphthalene E$_0$;
- the carriers (7.7/12.7 µm, the 3.4 µm complex, the CC shift, ERE).

**Proposed order of fixes** (none applied):
1. C1, C2, C3, C4, C6 and C7, which are mechanical, then re-run the affected
   gates and the briefing figures;
2. C5 together with a decision on gate G2;
3. P1–P3, as variants with their effect recorded.

## 5. The fixes applied (2026-10-20) and what they changed

C1–C7 were applied as proposed. One regression they exposed was fixed with
them:
- **GTH overflow.** `network.gth_stationary` builds the stationary vector
  relative to state 0, and it overflowed (nan) once proportional stripping
  (C1) kept a 100 C PAH fully hydrogenated in Allain's HI region (state 0
  ~1e-600 of the last).
- **The fix:** it rescales on the way, which is exact. A test chain spanning
  1e-600 now covers it.

How the comparison was made:
- **Before:** every check script was run on a frozen copy of the code from
  before the fixes; the whole pahchem suite was run on the unchanged
  worktree (374 passed, 21 xfailed, plus the Murga set: 34 passed, 9
  xfailed).
- **After:** the same scripts and suite were run on the fixed code. Outputs
  were diffed line by line.

**Test outcomes.** Only `test_pahchem_murga2022.py` changed:
- the three strict xfails "≥ 6 Å stay normal as in Figshare" (PDR 1, DF, MC)
  now pass, so their marker is removed;
- PDR 2 is a new strict xfail that records the residual;
- the deep-MC calibration was re-recorded.

No other test changed outcome.

| Benchmark | Before | After | Fix |
|---|---|---|---|
| σ per C (6–13.6 eV), C$_{10}$ / C$_{14}$ / C$_{16}$ against C$_{24}$ | 1.27 / 1.14 / 1.09 | 1.00 / 1.00 / 1.00 | C4 |
| C$_{24}$H$_{10}$ C–H stretch A (parent 303 s⁻¹) | 16 s⁻¹ | 252 s⁻¹ | C1 |
| C$_{54}$H$_{12}$ C–H stretch A (parent 595 s⁻¹) | 0 | 397 s⁻¹ | C1 |
| G2 model/BEB, anthracene, 15 eV–5 keV | 1.36–2.30 | 0.99–1.41 | C5, C4 |
| G2 model/BEB, coronene | 1.28–2.21 | 1.10–1.55 | C5 |
| G2 model/BEB, C$_{54}$H$_{18}$ | 1.40–2.58 | 1.39–1.80 | C5 |
| Montillaud Fig. 10, parent modes | +1.2 dex (50 %) | unchanged | – |
| Montillaud Fig. 10, stripped modes, 10/50/90 % | +0.46 / +0.25 / −0.08 | +0.84 / +0.59 / +0.28 | C1 |
| Allain II Tables 2–3, PAH 50 C, α$_H$ = 0.5 (H, H$_2$) / α$_H$ = 0.25 (H) | −0.13 / −0.17 / +0.02 | −1.02 / −1.07 / −0.64 | C1 |
| Allain II Tables 2–3, PAH 100 C, same columns | −0.24 / −0.24 / −0.07 | −0.91 / −0.91 / −0.52 | C1 |
| Allain C$_{14}$–C$_{32}$, same columns | −0.16…+0.06 | −0.21…+0.02 | C1 |
| SHIVA "post" rates (M20 eq. 4), C$_{24}$ / C$_{66}$ / C$_{128}$ | – | ×1.9 / ×2.5 / ×2.7 | C2 |
| Murga+20 Fig. 4 (default time-averaged reading) | – | ≤ 0.03 dex | C2 |
| M22 deep-MC step rate, theirs/ours | 0.87–1.44 | 0.70–1.16 | C2, C3 |
| M22 ⟨X$_H$/X$_{H0}$⟩, ≥ 6 Å, PDR 1 / DF / MC | +0.12…+0.49 | +0.00…+0.05 | C3 |
| M22 ⟨X$_H$/X$_{H0}$⟩, 6–8.7 Å, PDR 2 | +0.41…+0.44 | +0.12…+0.17 | C3 |
| M22 5 Å, DF / MC | −0.10 / −0.11 | −0.19 / −0.18 | C2 |
| Murga charge, 3.3 Å | – | ≤ 0.015 | C4 |
| PDRs4All families, C$_{54}$ 3.3/11.2, DFs | 0.62–0.89 (×1.8–2.0 high) | 0.35–0.52 (ok) | C7, C1 |
| PDRs4All families, C$_{96}$ 3.3/11.2, DFs | 0.33–0.36 (ok) | 0.18–0.20 (×1.5–1.7 low) | C7 |
| PDRs4All families, C$_{54}$ 3.3/11.2, H II and Atomic | 0.38–0.94 | 0.50–0.55 (×1.7–2.1 high) | C1, C7 |
| PDRs4All families, C$_{54}$ 12.7/11.2, H II and Atomic | 0.18–0.60 (ok) | 0.18–0.35 (×1.5–1.7 low) | C1 |
| Le Page; Andrews NGC 7023; Murga+22 charge | – | unchanged | – |

What the table says:
- **Two "agreements" were partly the stripping artefact.** Montillaud's
  stripped variant and Allain's dehydrogenated 50 and 100 C columns now
  disagree as the audit predicted. Allain's still need the "multi-quanta
  neglected" reading (section 2), which is a convention, not a confirmed bug,
  and is not applied.
- **The Murga+22 super-hydrogenation failure is resolved** outside PDR 2.
- **The two PDRs4All family conclusions flip** at the dissociation fronts, as
  predicted. In the H II and Atomic templates, C$_{54}$'s ratios also move,
  because it is now less dehydrogenated there (C1).
