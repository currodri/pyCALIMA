# PAHchem: coupling the PAHs to the gas (specification, 2026-10-21; status to 2026-10-24)

PAHs carry a large share of the free charge in neutral gas. They neutralise C⁺, H⁺ and the metal ions,
and they dominate the photoelectric heating. So the C chemistry, the electron abundance and the thermal
balance of a PDR are partly set by what the PAHs do. This page specifies how pahchem provides these
couplings, from the same steady state f(Z, N_H) that gives its IR emission. It is stage P of the calima-pdr
roadmap (`calima-pdr/docs/design.md`), and it is modelled here, in pyCALIMA, before any gas code uses it.

- **Literature:** {doc}`pahchem-gas-coupling-literature` holds the formulas, lab data, the Wolfire+22
  checklist, the Thi+20 network, the H₂ routes and the bibliography. Every source there is marked [read],
  [abs], [sec] or [derived].
- **Derived numbers:** `diagnostics/pahchem/check_pah_ion_rates.py` (DS87 against Wolfire+08, the lab and
  Langevin).
- **Unchanged rule:** pahchem does not predict the local conditions. The ion densities needed here
  (H⁺, He⁺, C⁺, S⁺, Si⁺, Mg⁺, Fe⁺, HCO⁺, H₃O⁺, H₃⁺) become inputs, like n_e and T, taken from a sourced
  card or a PDR model.

## 1. What pahchem had, and what was missing (2026-10-21, before stage P)

Most of the missing items are now implemented; their status is in §4b.

**Has:**
- photoionisation (Jochims+96; Wenzel+20 for cations with N_C ≥ 32; IPs shifted +0.6 / +0.64 eV) and
  photodetachment (EA +0.37 eV);
- electron capture on cations, `chemistry.electron_capture`: 0.82 πa² v_e J̃, a = 0.9 √N_C Å, s = 1;
  the FALP lab trend is optional;
- statistical electron sticking on neutrals (`attachment.py`);
- recombination energy put into the vibrations;
- the H-loss and H-addition chain, with photolytic H₂ loss and Eley–Rideal abstraction
  (`network.PAHNetwork.h2_formation_rate`).

**Missing:**
- **Ion–PAH collisions.** There are none (`murga_charge.py`: "no ion collisions"). Now §4b step 3.
- **Photoelectron kinetic energy.** pahchem treats every ionising photon as depositing no internal energy,
  which implicitly gives the electron all of E − IP. That contradicts the measured partition (§2.1). Now
  §4b steps 1 and 1c. Since 2026-11-03 the ion keeps its share, in the product's cascade (§2.2c,
  `Params.ionisation_heating`).
- **Gas heating and cooling.** No gas photoelectric heating, no recombination cooling. Now §4b steps 1 and 2.
- **Rates for a gas network.** Per-ion neutralisation, the net electron source and H₂ formation per H are
  not exported. Now §4b step 4, except H₂ formation.
- **Three PAH charge models in pyCALIMA:** `pahchem`, `models/PAH_charge` (the Berné+22 table builder)
  and `models/PAH_photophysics`. This page makes pahchem the owner of the PAH terms.

## 2. Processes and recommended treatment

All rates are per molecule m (N_C, N_H, IP_m(Z), EA_m) in charge state Z. They use pahchem's one geometry,
g πa² with g = 0.82 and a = 0.9 √N_C Å, and v_X = (8kT/πμ_X)^½.

### 2.1 Photoelectric emission, with the photoelectron energy

- **Mean photoelectron energy:** ⟨E_k⟩ = γ_e (E − IP_m(Z)) for Z ≥ 0. The default is γ_e = 0.46 ± 0.06; the
  classical value is 0.5 (V90, BT94), and Lepp & Dalgarno (1988a) gave the electron all of it (γ_e = 1).
  - 0.46 is not a measured constant. It is Berné+22's unweighted average over 7.1–10.1 eV of γ_e(E),
    which they derived from the coronene photoelectron spectra of Bréchignac+14 (Fig. 2). Bréchignac+14
    itself gives no partition.
  - Per photon energy, γ_e(E) runs from ≈ 1 (direct ionisation to D0, 7.5–7.8 eV) to 0.05–0.37
    (autoionisation, 9.1–9.6 eV; Bréchignac+14 Table I). Nothing constrains 10.1–13.6 eV, where most
    of the heating photons of an O-star field are.
  - So the ±0.06 is the spread of the mean. An option γ_e(E), digitised from Bréchignac+14 for coronene,
    is to be offered and compared against the default; neither choice is tuned to a fit.
  - The rest, (1 − γ_e)(E − IP), stays in the PAH as vibrational energy.
  - For photodetachment, E_k = E − EA.
- **Hrodmarsson+25 (A&A 698, A202) measures 10.1–13.6 eV**, on nine neutrals of C13–C20 from 7 to 22 eV (SOLEIL
  DESIRS, i²PEPICO; data on Zenodo 15271348, CC BY 4.0, `pahchem.hrodmarsson2025`;
  `diagnostics/pahchem/check_hrodmarsson25.py`). The `PARTITION_NODES` above 13.6 eV come from these data
  (cited as "Verstraete+25" until 2026-10-22).
  - **Electron share below 13.6 eV:** the median γ_e is 0.77, 0.64, 0.61, 0.55 and 0.51 at 9, 10, 11, 12 and
    13 eV, against pahchem's 0.46.
    - Perylene (C20) is the lowest, at 0.50–0.59.
    - Coronene's 0.46 is the only measurement above C20. So the default is kept for N_C ≥ 21, and the
      C13–C20 values are not extrapolated.
    - The paper's single fit, ⟨KE⟩ = 0.55 (hν − 7.74), describes the 13.6–22 eV range better than the range
      below it.
  - **Ionisation cross section:** the total ion yield, scaled to 22 Mb/C on the 17 eV peak as the paper does,
    against σ × Y_Jochims.
    - The Mathis-weighted 6–13.6 eV rate is 0.70× the data with Li & Draine (median of 8 molecules) and
      0.66× with Malloci (4).
    - Independent of the assumed scale, the rate per 17 eV peak is 0.44× and 0.60×. The linear Jochims yield
      rises too slowly above threshold for these molecules.
    - Li & Draine's 17 eV peak is 1.57× the 22 Mb/C (34 against 22), so its error partly offsets the yield's.
  - **Perylene's maximum photoelectric efficiency** in the Mathis ISRF (their eq. 5): 3.0 % (Li & Draine) and
    2.6 % (Malloci) with pahchem's ingredients, but 4.6 % with the measured σ_ion and ⟨KE⟩ over either
    absorption. The paper's 1.5 % (eK16) includes WD01's free-space factor Y2, which removes the slow electrons
    that the ion's Coulomb field would pull back.
    - Their argument is that the lab's extraction field (440 V/cm) removes this attraction.
    - It does not. That field equals the ion's Coulomb field only at ≈ 1800 Å, where the potential is 8 meV,
      and it lowers the threshold by 2(e³F)^½ ≈ 16 meV.
    - So measured molecular yields and energies are already free-space values, and Y2 on top of a measured
      IE counts the attraction twice. pahchem does not apply it.
  - These results are for C13–C20 only. Changing the yield law or γ_e for the library (N_C ≥ 21) needs either
    data on larger PAHs or a size trend; neither is assumed here.
- **Gas heating:**

      Γ_PE = Σ_m x_m Σ_Z f_m(Z) ∫ σ Y F ⟨E_k⟩ dE

  with the yields and shifted thresholds pahchem already uses. Its rate is then exactly pahchem's
  photoionisation rate, so the charge balance and the heating are consistent by construction.
- **Implementation:**
  - an energy-weighted twin of `photo.ionisation_rate`, stored in `StateBasis` (this bumps
    `BASIS_VERSION`);
  - the (1 − γ_e)(E − IP) deposit into the cascade is **opt-in**. It changes the existing H-loss of cations
    in strong fields and the IR emission. Before it becomes the default, its effect on the
    PDRs4All fits and the benchmark gates is reported.

### 2.2 Electron capture, attachment and recombination cooling

- **e + PAH^Z (Z ≥ 1):** keep `electron_capture` with s = 1.
  - Lab check: the FALP data at 300 K (N_C 6–16) are 0.03–0.37 × pahchem's capture (§D of the
    diagnostic). The ratio rises with N_C, but where it extrapolates to 1 depends on the fit: N_C ≈ 25 for the
    eight PAH points (N_C 10–16, slope 2.95), N_C ≈ 87 for `chemistry.lab_recombination_fit`, which includes
    benzene (9 points, slope 1.01). Neither extrapolation is constrained above N_C = 16.
  - Fluoranthene is one experiment (Rebrion-Rowe+03: (2.5 ± 1.5)×10⁻⁶ for a mixture of C₁₆H₁₀₋ₙ⁺ and N-bearing
    ions made by charge transfer with 6.6–7.7 eV to spare); a second row quoting the same experiment was
    removed on 2026-10-22. Their own "physical limit" for the monomer, 5.3×10⁻⁶, is 0.47 × pahchem's capture.
    Phenanthrene (Hassouna+03, conference proceedings) could not be obtained and stays as quoted by Biennier+06.
  - Every point with N_C ≥ 10 is at 300 K, from two methods: FlAPI photo-ions, and FALP-MS ions made by
    charge transfer, which can be internally hot (Biennier+06). The method is now a column of
    `pah_electron_recombination.csv`.
  - No PAH has a measured T dependence. The small aromatics give T^−0.69 (C₆D₆⁺) and T^−0.83 (C₆D₇⁺)
    (Hamberg+11, storage ring). So the T^−½ of the capture formula is an assumption.
  - Tielens (2021, eq. 8.106), 1.3×10⁻⁶ N_C^½ (300/T)^½, is the Spitzer form with an ad hoc 0.8 planarity
    factor: 0.46 × pahchem's capture at every N_C. The data reach it near N_C ≈ 19 (PAH-only fit) or 40
    (pahchem's fit), so they cannot tell the two forms apart. This is the factor-2 uncertainty of this rate.
- **e + PAH⁰:** keep the statistical sticking (the low-density, radiatively stabilised limit). Compare it with:
  - anthracene: 1.0×10⁻⁹ ± 30 % at 300 K (FALP, Canosa+94), and 0.84–2.6×10⁻⁹ over 48–300 K with no trend
    (CRESU, normalised to that value; Moustefaoui+98). These are not radiative: in N₂ at 0.6–2.7×10¹⁶ cm⁻³,
    collisions (≈ 10⁷ s⁻¹) stabilise the hot anion far faster than autodetachment or IR emission (≈ 2 s⁻¹), so
    the missing pressure dependence means saturated collisional stabilisation (corrected 2026-10-26; before, read
    as "so radiative"). They constrain the share of classical captures that form the anion, not the
    zero-density sticking (`results/pahchem/investigations_2026-10-25/electrons/`);
  - C₆₀: 1.2×10⁻⁶ at 400 K.
  - Bounds:
    - Flower & Pineau des Forêts (2026) recommend a constant 10⁻⁶ from classical theory. Their own
      model ignores the anthracene value, 0.002 of Langevin, so this is at most a labelled upper bound for
      large, high-EA PAHs.
    - Tielens (2021, §8.7.2) notes that capture needs low-energy s-wave anion resonances, which not every
      PAH has. The Langevin form of his eq. 8.107 is 1.27×10⁻⁷ N_C^½ (printed with N_C, a typo), with s = 1
      for N_C > 30.
- **Recombination cooling:** each capture removes ⟨E⟩ = kT Λ̃/J̃ from the gas (DS87; already in
  `shared_physics.DS87_lambda_scalar`). That is ≈ kT on cations and 1.5–2 kT on neutrals:

      Λ_rec = Σ_m x_m Σ_Z f_m(Z) n_e k_e(m, Z) ⟨E⟩

### 2.2b The two sticking factors, constrained (2026-11-01/02)

Figures: `check_wd01_ingredients.py` (`pahchem_wd01_ingredients_T*.png`) for the decomposition;
`check_sticking_constraints.py` for the lab fits and anchors (`pahchem_sticking_lab.png`), the PDRs4All rms over the
card's cells (`pahchem_pdrs4all_sticking.png`) and Wolfire+08 against density (`pahchem_w08_attachment.png`).

**Against WD01, one ingredient at a time** (`check_wd01_ingredients.py`; per grain, C24–C150, T = 100 and 1000 K,
ψ = 10²–10⁵). Both models on one charge chain, validated against each code (≤ 4×10⁻¹⁶ and ≤ 8×10⁻⁴), nine rate blocks
swapped cumulatively:
- ψ ≥ 10⁴: the cation sticking (s = 1 here, WD01's 0.5(1 − e^(−a/10 Å)) ≈ 0.16–0.25) is the largest term in every case:
  ⟨Z⟩ +0.39 to +0.65, α(H⁺) −0.23 to −1.04 dex. The charge ladder beyond +2 matters only at ψ = 10⁵ for N_C ≥ 96.
- ψ ≤ 10³: the neutral attachment (statistical sticking ≈ 1 here, WD01's ≈ 0.2) is the largest term for N_C ≥ 54.
- The photon physics (absorption, thresholds, yields) moves ⟨Z⟩ by ≤ 0.1 for N_C ≥ 54. The two models differ almost
  only in the sticking, on cations and on neutrals.
- WD01's sticking is a grain-physics estimate: on cations it is 0.14–0.15 of BT94 at C10–C16, inside the C10–C14 data
  (0.03–0.23) but below pyrene's 0.37, and it stays flat with size (0.19 at C300); its neutral fit was calibrated on N_C ≤ 20 and on C60/C70 values that
  Viggiano+10 superseded (in WD01's normalisation C60 sticks at 0.97 at 400 K, against its fit's 0.20).

**Cation sticking: lab fits** (`chemistry.lab_sticking`, `capture_model="lab_cold"` / `"lab_pahs"`). s(N_C) = 1 −
exp[−(N_C/N_s)^b] to the photo-ion points (FlAPI, VENDAMS) or every PAH point with N_C ≥ 10. Benzene is left out (its
recombination is dissociative); with it, the power law flattens from slope ~2.5–3 to 1 (the older `lab_trend`). N_s =
23–24, b = 2.6–3.1: s ≈ 0.3 at C16, 0.64–0.68 at C24, 1 from C54 up, an extrapolation beyond C16. The internally hot
charge-transfer ions change little: s reaches 1 near N_C ≈ 22–27 in every PAH subset.

**Neutral attachment: lab and theory.** f ≈ 0.001–0.005 for anthracene, perylene and coronene (the anthracene flow
tubes at 48–300 K; Carelli+13's scattering at 10–1000 K, upper bounds), f ≈ 0.72 and 0.47 for C60 at 400 and 625 K
(Viggiano+10, each a factor 3). Nothing is measured between C24 and C60 (`check_carelli13_attachment.py`, and a
literature search of 2026-11-01). `attachment_model="size"`: f = f_low + (1 − f_low)/(1 + (N_half/N_C)^b) with
f_low = 0.003; the anchors need b ≳ 7 and N_half ≈ 45–57 (defaults 50, 10). An electron-affinity logistic that meets
the same anchors (midpoint 2.35–2.6 eV, set by perylene's 0.97 eV and C60's 2.68 eV) leaves the library's 514 anions
of N_C 40–120 (EA 0.9–3.3 eV, median 1.8) at a median f of 0.02–0.06, a quarter of them above 0.1–0.3.

**Against the PDRs4All Orion Bar spectra** (`fit_pdrs4all_spectra.py`, the stated card conditions; best profile per
template; `results/pahchem/pdrs4all_capture_2026-11-01/`):

| variant | HII / Atomic rms | best γ (HII, Atomic) | DF1 / DF2 / DF3 rms | f⁻ in the DFs |
|---|---|---|---|---|
| s = 1, f = 1 | 0.046 / 0.097 | 1.4e4, 1.5e4 (interior) | 0.074 / 0.052 / 0.047 | 0.43–0.49 |
| `lab_cold`, `lab_pahs` | the same | the same | the same | the same |
| `lab_trend` (with benzene) | 0.048 / 0.101 | 1.0e4, 1.1e4 | the same | the same |
| `wd01` cation capture | 0.087 / 0.177 | 9.5e3, 1e4 (low edge) | 0.073 / 0.052 / 0.046 | 0.42–0.47 |
| f = 0.3 / 0.1 | — | — | 0.095–0.123 / 0.067–0.088 / 0.064–0.080 | 0.07–0.28 |
| f ≤ 0.03 | 0.047 / 0.098 | 1.1e4, 1.5e4 | 0.128 / 0.091 / 0.078 (γ at the high edge) | 0 |
| f size logistic, N_half 50 | — | — | 0.073 / 0.053 / 0.049 | 0.44–0.47 |
| f size logistic, N_half 80 | — | — | 0.081 / 0.060 / 0.055 | 0.33–0.38 |
| f EA logistic, midpoint 2.5 eV | — | — | 0.092 / 0.065 / 0.062 | 0.29–0.32 |

- The ionised templates exclude WD01's cation capture (too ionised everywhere in the stated γ, rms ×1.8–1.9) and do not
  separate s = 1 from the lab fits: the emitting population (⟨N_C⟩ 84–87) has s ≈ 1 in both. WD01's rms still falls
  towards the card's lowest γ (9.5×10³), so the exclusion holds within the card, not beyond it.
- The dissociation fronts want efficient attachment on the emitting PAHs (⟨N_C⟩ 76–82). The size logistic that meets
  the lab anchors fits as well as f = 1; the EA logistic that meets them does not. Every DF fit sits at an edge of the
  stated γ range, so the card's G0 and n_e bound the test; but f = 1 fits better than f ≤ 0.1 at every cell of the
  card, not only at its edge.
- The f = 0.1 and 0.3 and the logistic rows hold the band shape of the s = 1 run (`--shape-from`), which reproduces the
  full fit to ≤ 0.003 in rms with the same γ and charge shares.
- The "anion" signal rests on PAHdb's DFT anion spectra; it may stand in for something else the fit lacks.

**Against Wolfire+08's C⁺ neutralisation** (W08 at Φ_PAH = 0.4, the cold neutral medium, the same PAH carbon per H):
α(C⁺)/W08 = 0.78 (f = 1), 0.23 (size logistic), 0.20 (f = 0.003) for W08's N_C 32–40, and 0.88, 0.83 and 0.14 for
N_C 50–100. The size logistic meets W08's sight-line calibration if the PAH carbon is mostly in N_C ≳ 50.

**The defaults since 2026-11-02: `capture_model="lab_cold"`, `attachment_model="size"`** (N_half 50, b 10,
f_low 0.003), in the population (gas rates, group tables, the PDRs4All fits) and, since a later commit of the same
day, in `BaselineModel` (`capture_coefficients`; before, it took s = 1 and `attachment_fraction` whatever the model).
Before, s = 1 and f = 1.
- In the baseline's three parents (the RAMSES photodestruction tables, H₂ formation), C96 is unchanged.
- C24's neutral sticking halves (f × s_e(f): a lower f also slows the anion's autodetachment, so 280× in f is 2× in
  sticking), and its cation fraction rises (0.13 → 0.19 in a PDR cell, 0.35 → 0.46 in the Orion atomic one). Its carbon
  loss rises ≤ 4 % and its H₂ formation 6–15 %.
- C54's anion fraction falls (0.174 → 0.126 diffuse). Its carbon loss (10⁻⁸⁶–10⁻³⁸ s⁻¹) stays negligible.

Both functions are **extrapolations** where the PAHs that matter most are:
- **Cation sticking** is measured at N_C 10–16 only. Its rise to 1 by N_C ≈ 40 is the fitted form's, with saturation at
  the molecular limit assumed. The astronomical tests (PDRs4All, ⟨N_C⟩ 76–87) see s = 1 and cannot test the rise.
- **Neutral attachment** is measured at N_C ≤ 24 (f ≈ 0.003) and at C60 (0.5–0.7). The midpoint (N_half 50) and the
  steepness (b 10) in between are set by those two ends, not by data. The PDRs4All fronts and Wolfire+08 constrain f
  only for N_C ≳ 50, where it is ≈ 1. Below N_C ≈ 45 the logistic follows the lab, not any astronomical test.
- The rows of §4 were measured with the earlier defaults.

### 2.2c The charge ladder to the cold Coulomb limit (2026-11-02, stage 1)

`Params.charge_ladder = "cold_limit"` (the default; `"library"` keeps the computed charges, the top one holding the
higher charges, as before). In the population every molecule runs from its lowest computed charge to ⌊0.2 N_C⌋
(`impact.coulomb_limit`; +4 for C24, +10 for C54, +30 for C150), `population.extend_ladder`:
- **Thresholds** continue the last computed step, constant per charge as for a charged conductor. They reproduce
  PAHdb's +2 → +3 of C54H18 and C96H24 to 0.07 eV. With the calibrated shifts they give 13.4–14.1 eV for the isomers
  of C30H14, C32H14 and C36H18, against Wenzel+20's measured 13.9–14.1 eV. A molecule with no computed dication
  (676 of v4.00's 1424) takes its +1 → +2 step from a fit to the 743 that have one: 0.464 + 16.91/√N_C eV (rms
  0.24 eV). The reach of a field cut at 13.6 eV is WD01's (their eq. 22) and Andrews+16's (Fig. 4): +2, +3 and +4
  for C24, C54 and C96.
- **Stand-ins.** The states above the highest computed charge take its modes. Their UV cross section is +2's.
  - The literature (search of 2026-11-02, quotes checked against the full texts): computed IR spectra exist at +3
    for only two molecules in print (Bakes, Tielens & Bauschlicher 2001, Table 1) and the 7 of PAHdb v4.00. There
    are none above +3, and no computed or measured UV cross section of any PAH at +3.
  - Multiply charged spectra are cation-like, with small shifts: Bakes+01 §2.2, Malloci+07 (A&A 462, 627; 40
    dications), and the only lab dication spectra (Banhatti+21).
  - The total IR intensity keeps rising through +2 (dications 2.5 ± 1.0× their cations, Malloci+07; ×1.7 in
    PAHdb v4.00). From +2 to +3 it scatters both ways: ×0.93 and ×1.63 in Bakes+01, ×0.57–2.3 in PAHdb.
  - Models give every ion one UV cross section: WD01 ("the distinction is not critical"), Draine & Li 2007 (every
    multiply charged PAH has the cation's), Bakes+01. The one documented trend, Malloci's TD-DFT and our Octopus
    library, is a drop of 4–10 % per charge at 6–13.6 eV through +2, as the σ* onset moves blueward. It is not
    extrapolated.
  - So the stand-in is a choice, not a measurement. Where stand-in states carry the population is reported
    (`Molecule.stand_ins`), and the UV sensitivity is for stage 2.
- **Yields** are unity above Y_UNITY_EV (20 eV) only above the state's threshold. A state with a higher threshold
  (C54 from +5, C150 from +9) absorbs below it without ionising.
- **Live states** (`Population._live`). Above +1 only photoionisation raises the charge, so a state is live if the
  one below ionises in that field shape. The others hold f = 0 exactly and get no basis. In fields cut at 13.6 eV the
  cold-limit ladder adds only the states that field reaches.
- **Top state.** Ionisation out of ⌊0.2 N_C⌋ is the molecule's Coulomb explosion (`chain_overflow`, a destruction
  rate).
- **C 1s absorption and Auger decay** (stage 3, 2026-11-03; `Params.kshell = "auger"`, the default; `"single"` keeps the
  treatment before):
  - *Before:* the library's C 1s absorption (the measured pyrene edge from 282 eV, Gordon+03, then the atomic C 1s) was
    counted as one ionisation at the valence yield up to E_PH's top (500 eV), and lost above it.
  - *The split* (`uvlib.sigma_k_shell`): the C 1s part exactly as `build_library.py` adds it to every anchor (the bundled
    curve `gordon03_pyrene_kedge.dat`); the rest of the library stays valence. Above 500 eV it is 96 % of the
    absorption. The C 1s rates run on their own grid to 10 keV (`population.K_GRID`), outside the bases.
  - *Per absorption* (`population.kshell_branching`), the same rule as the valence electrons:
    - The 1s electron lands at E′ = E − E_K(Z), with E_K(Z) = 291 eV + IP(Z) − IP(0): the core level shifts with the
      charge as the valence does (coronene⁺'s 1s threshold ≈ 293 eV against ≈ 290 eV, Huo+22).
    - Above 0 it escapes. In the Coulomb window (−D ≤ E′ < 0, D = (Z+1) times the IP step) it is bound and escapes as
      the valence electrons do there (`Params.coulomb_escape`; all of them by default). Below the window, the
      1s → π* resonances from 284 eV, it stays: resonant Auger decay leaves +1 (Jayadev, Jagau & Krylov 2025).
    - The hole fluoresces with ω_K = 2.8×10⁻³ (Krause 1979, Table 3) or Auger-decays. 9 % of the decays eject two
      electrons (7–12 %): the C 1s hydrocarbons of Sci. Rep. 8, 16405 (2018), CH₄ 6.3, C₂H₂ 8.2, C₂H₄ 8.5 and C₂H₆
      8.8 %, at an aromatic carbon's count of valence electrons within 2 Å (13–16). Their all-molecule fit gives 11 %,
      pulled up by the halogenated molecules. Small PAH cations at the K edge end "predominantly" at net +2 with "a
      small fraction" at +3 (Huo+22, +23), which agrees.
    - So above the edge dz = +2 (90.7 %), +3 (9.0 %) or +1 (0.3 %), at every charge: up to 0.2 N_C a molecule loses
      ≤ 5 % of the valence electrons near any carbon, and the Auger electron (≈ 265 eV from a neutral C216, ≈ 200 eV at
      +40, one IP step less per charge) has far more energy than the window is deep (63 eV for C216 at +40).
  - *Energies:* the photoelectron takes E − E_K(Z) (a window electron autoionises slowly, ~0); an Auger electron
    E_K − IP(Z) − IP(Z+1) − 17.3 eV, a double Auger pair that less IP(Z+2). The 17.3 eV is the dication's mean excitation
    after the decay, "an average excitation of about 17.3 eV" over the benzene Auger spectrum (Huo+23, Sect. 3, citing
    Carniato+20); the spectrum's peak alone gives ~14 eV (Jayadev+23), Lange+25 quote ~20 eV, and no PAH Auger spectrum
    exists. It has one owner, `impact.K_AUGER_EXCITATION_EV`, also for the cosmic-ray collisions' K holes (27 eV above
    the neutral less IP1 + IP2, clipped at 0, before).
  - *The outgoing electrons* (`population.escape_table`, `impact.escape_outcomes`): the photoelectron and the Auger
    electrons collide with the molecule on their way out, as `impact.py` collides a cosmic-ray electron with it (Bethe on
    its own oscillator strengths, the polystyrene low-speed correction, Kim's binary-encounter scaling for slow electrons;
    a loss is at most T/2, so below 1 eV none is possible). Poisson collisions along escape paths from a uniform start
    in Micelotta+11's disc; each one's extra charge and E* from `impact.deposition` (a distant ionising collision leaves
    the photoionisation's mean hole, partition "measured", as the photons do since 2026-11-04; no E* before). The
    outcomes are convolved over the electrons, on grid points E* = k ΔE with
    each bin split between its edges (`impact._edge_split`, exact in the mean; the cosmic-ray passages use it too since
    2026-11-03, their bin centres having lost ΔE/2 per extra collision). Per K event (F_ν ∝ ν⁻¹) the collisions raise ⟨dz⟩
    to 2.45–3.54 at Z = 0 and +1 (C24–C216, rising with size; 2.36–2.78 at +5) against 2.09 without them. The product
    keeps 31.7–40.9 eV, the 17.3 eV plus 14–24 eV, at and above Micelotta+10b's and Lange+25's "15–35 eV". The measured
    partition adds 0.3–5 eV to that over "none" and leaves ⟨dz⟩ alone. Nothing leaves the grids silently: a probability of more than 10⁻⁴
    beyond 320 eV or 12 extra charges per electron raises.
  - *The chain:* steps of +1 up to +21 (`population._chain_fractions`, exact against a dense solve); steps past the cold
    limit are Coulomb explosions (`chain_overflow`). Fields that do not reach 282 eV keep the plain chain, bit for bit.
    Not in the group tables, which follow single ionisations.
  - *Fields:* `Environment.field_powerlaw` (F_ν ∝ ν^−α from `field_e_min_ev` to `field_e_max_ev`, `fields.powerlaw`) for
    X-ray-bright sources that keep the far UV that sets G0. Fields with no far UV (shielded X-ray-dominated gas, where
    C 1s is ~97 % of the absorption) need a normalisation other than G0.
  - *Check* (`check_kshell_auger.py`; C24–C216, F_ν ∝ ν⁻¹ and ν⁻¹·⁵ from 1 eV to 10 keV, G0 = 10⁴, T = 10⁴ K, against
    one electron per absorption and WDB06 graphite): with the far UV present, C 1s makes 0.2–0.5 % of the ionisation
    electrons at ψ ≤ 10⁴ and 1.3–6.4 % at ψ = 10⁶. ⟨Z⟩ rises by ≤ 0.24 (C150 at ψ = 10⁶, 19.55 against 19.30); Γ_PE by
    0.5 % at low ψ, and by −1.9 % (C150) to +1.2 % (C24) at ψ = 10⁶, where the outgoing electrons leave part of their
    energy in the molecule (−2.7 to +0.7 % before the measured partition and the "neutral" reading, 2026-11-04). The ion's
    excitation after the decay is 1–7×10⁻⁴ of the power a state absorbs. So C 1s matters where the far and extreme UV
    are gone.
  - *X-rays only* (`RadiationField.intensity`): a field with no flux at 6–13.6 eV takes G0 as its whole energy flux in
    Habing units; `Environment(field_powerlaw=α, field_e_min_ev=300)` is a crude shielded field (a sharp cut, not a
    column's attenuation: that needs Morrison & McCammon 1983's or MHT96's cross sections, not in the repo). There no
    absorption heats without ionising: the bases have no photon deposits and no photon-driven loss (`build_state_basis`
    returns zeros; bases stored earlier held NaN there and are read as 0).
- **Deposits the bases do not hold: dissociative recombination and the Auger excitation** (2026-11-03,
  `Params.recombination_dissociation`, on by default):
  - *Before:* each capture deposited the threshold into the state it made, but only into the IR (at the rate k_up(z),
    exact for a chain of single steps without ions); the loss tables held the photon deposits only. So no capture ever
    drove H or H₂ loss, and the C 1s Auger decays heated nothing.
  - *Now* (`population.deposit_response`, field-independent; cached in `cache_dir/responses` and shared by every
    Population of a process): for each state, its IR per mode and P(H), P(H₂) per deposit, in two sets: "rec", its
    capture energy, everywhere; "aug", 17.3 eV plus each of 21 heat nodes (0–320 eV, `HEAT_NODES_EV`), only in fields
    reaching the C K edge, all from one multi-start cascade pass (`_emission_many`) and the loss cascade's single-event
    yields (`CascadeModel.photon_yields`). At run time (`_deposit_flows`) the captures into z run at f(z+1) α(z+1) n_e
    and the C 1s products z+j keep 17.3 eV plus their outgoing electrons' E* on the nodes (mean-conserving split): they
    add to the hydrogenation chain's loss rates (`_deposit_losses`), the band fluxes (`_deposit_bands`, replacing the
    captures the bases emit) and the spectra (`deposit_spectra`, also in the fits' band shape). A plain chain keeps its
    band fluxes to 10⁻⁹ and builds no deposit rows.
  - *Per capture* into the intact molecule (`check_dissociative_recombination.py`): coronene⁺ + e⁻ → coronene keeps
    its H (P = 2×10⁻⁶), as Le Page's statistical model expects for pyrene⁺ and larger (Biennier+06, who also note that
    no laboratory set-up has measured PAH dissociative recombination products). The deposit grows with the charge, and
    so does the loss: C24 0.84 into +1 (11.2 eV); C54 0.32 at +5, ~1 at +9; C96 5×10⁻³ at +10, ~1 at +18; C216 0.06 at
    its cold limit. Per Auger decay: C24 ~1 (H 0.36, H₂ 0.64), C54 4×10⁻³–3×10⁻², C96 and larger < 10⁻⁸.
  - *In fields:* PDRs (Kurucz 40 kK, G0 10³–10⁵, n_H 10⁴): no change (⟨H lost⟩ ≤ +0.04, for C24, which the photons
    strip anyway). EUV (step 4's field): C96 f(N_H0) 0.980 → 0.967 at ψ = 10⁵, ⟨H lost⟩ 21.4 → 23.5 at 10⁶. X-rays only
    (F_ν ∝ ν⁻¹, 300 eV–10 keV, n_H 10⁴, n_e 10⁻²), where photon-driven loss is exactly 0: at G0-equivalent 10⁴ the C 1s
    products alone (32–41 eV each, the tail to hundreds of eV) leave f(N_H0) 0.959, 0.900, 0.869, 0.895, 0.927 for
    C24–C216; with dissociative recombination 0.889, 0.773, 0.820, 0.894, 0.927 (rerun 2026-11-04 with the measured
    partition: ≤ 0.006 lower). Both strip PAHs that photons do not.
  - *Not yet:* the deposits in `state_powers`; the tdep band convention takes each state's own temperature shares for
    them.
  - *Guards:* the probabilities and rates bounded by construction are checked before any clip (`population._round_off`,
    `photo.yield_coulomb_window`): the recursions stray 1–2 ulp past [0, 1] (≤ 4.4×10⁻¹⁶ over every bin of C24–C216),
    and anything further, or non-finite, raises.
- **The energy a photoionisation leaves its product** (2026-11-03, `Params.ionisation_heating`, on by default):
  - *Before:* the bases deposited nothing for an ionising absorption, so the ion's share of the excess, E − IP − ⟨E_k⟩
    (54 % below 13.6 eV, the Hrodmarsson+25 shares to 21.9 eV, then constant: `photo.photoelectron_energy`), was lost.
    It was neither IR nor gas heating, a leak in the energy budget. The opt-in `ionising_photons_heat` (Montillaud's
    convention, kept for the BaselineModel) put it into the absorbing state's cascade, not the product's.
  - *Now* (`population.ionisation_deposits`, `ionisation_products`): each valence ionisation out of Z leaves the
    product Z + 1 its E_int at every photon node, with the same σ (less C 1s where that has its own branching), yield
    and partition as the charge chain and Γ_PE. Double ionisations ("knockout") leave Z + 2 the share over IP(Z) +
    IP(Z+1). The anion's detachment leaves the neutral E − EA − ⟨E_k⟩, the mean over its prompt, valence and delayed
    branches. Per field shape, each product state gets one response row per parent (`deposit_response` with
    weights). This is exact: the cascade and the single-event yields are linear in the starts. The rows run at
    f(Z − dz) G0 k (`_deposit_flows`) into the chain's losses, band fluxes and spectra. Absorbed energy = deposits +
    electrons (as Γ_PE takes them) + ionisation work, to 10⁻⁹ (test). Not with `ionising_photons_heat`, which would
    count it twice (raises).
  - *Per event* (`check_ionisation_heating.py`; C24–C216; Kurucz 40 kK):
    - Cut at 13.6 eV: ⟨E_int⟩ is 2.4–3.4 eV into +1, falling with the charge to 0.2–0.8 eV at the last live state,
      and 4.9–6.4 eV into the neutral from the anion.
    - To 100 eV: 4.3–4.7 eV into +1 and 6.2–7.5 eV into the neutral.
    - P(H) on N_H0 ≤ 2×10⁻⁵ everywhere except neutral C24 from its anion (0.03). These deposits heat; they do not strip.
  - *In fields* (off → on):
    - PDRs at G0 = 10³ (⟨Z⟩ 0.5–1.1): IR power per molecule ×1.04–1.08, 3.3/11.2 −4 to −8 %, 7.7/11.2 +2 to
      +4 %. At 10⁴ and 10⁵ ×1.01 and ×1.001, the higher charges keeping less.
    - EUV, ψ = 10⁴: ×1.26–1.28, 3.3/11.2 and 6.2/11.2 −5 %. At 10⁵ ×1.03; at 10⁶ ×1.000, as measured with the
      "photon" reading (see below). The "neutral" default adds ×1.006–1.009, ×1.010 and ×1.003 on top
      (`check_photoelectron_reference.py`); PDRs are unchanged.
    - X-rays only: ×1.003–1.006.
    - f(N_H0) and ⟨H lost⟩ are unchanged in every field.
  - *The hole of a charged state* (`Params.photoelectron_reference`; `check_photoelectron_reference.py`, 2026-11-04).
    - *The problem.* `photo.photoelectron_energy` keeps the ion's E_int at 21.9 eV *photon* energy,
      (1 − 0.58)(21.9 − IP). So a state with IP above 21.9 eV keeps nothing: Z ≥ +6 for C54, ≥ +8 for C96 in the
      ladder. Its electron then takes the whole excess, a share of 1 even 1 eV above threshold. That is why ψ = 10⁶
      gains nothing above.
    - *The data.* Hrodmarsson's nine neutrals (IE 7.00–7.91 eV) do not decide between hν and X = hν − IE. X lines
      them up better at 11.5–14 eV, hν at 14.5–19.5 eV. Their dip in the share, the σ onset's slow electrons, sits
      at hν ≈ 14 eV in all nine, perylene included.
    - *The ladders* (Octopus LDA, C24–C216, −1 to +2): with the charge, every occupied level shifts alike. C54H18
      moves −2.66 eV per charge, the same to ±0.03 eV from the HOMO to > 16 eV below it, close to the DFT IP step;
      C24 ±0.05, C216 ±0.01; the anion as much the other way. Among neutrals, the density of states per C lines up
      better in absolute energy than from the HOMO (std 0.012 against 0.066 at −7 eV), while the HOMO rises 0.84 eV
      from C24 to C216. The σ levels are anchored.
    - *The consequence.* A pure excess reading, one curve for every molecule, is wrong across sizes. Across the
      charges of one molecule the ladder is its neutral's, shifted. "neutral" (opt-in) therefore reads a state of
      threshold T at hν − (T − IP(0)), so every charge keeps its neutral's hole (5.5–6.5 eV for IP(0) 5.4–7.4 eV).
      Near threshold the neutral share is, if anything, an upper limit for an ion: the Coulomb field of the ion does
      not suppress slow electrons, which favour deep holes.
    - *Effect on Γ_PE* ("neutral"/"photon"):
      - Neutrals, and cations in fields cut at 13.6 eV, are unchanged exactly (test), since their shifted energies
        stay below 13.6 eV, where the share is constant.
      - V9 at γ = 10² changes 1.008–1.023 and at 10³ 1.000–1.012, through the anions' valence branch only; at
        ≥ 10⁴ and in PDRs ≤ 1.003.
      - EUV: ψ = 10⁴ 0.955–0.975, 10⁵ 0.77–0.79, 10⁶ 0.49–0.54.
      - ν⁻¹ to 10 keV: ψ = 10⁴ 0.93–0.95, 10⁶ 0.74–0.82.
      - X-rays only: ≥ 0.9994.
      - ⟨Z⟩ is unchanged and the IR rises 0.3–1.7 %.
      - So "photon" overestimates the EUV photoelectric heating by up to 2×.
    - "neutral" is the default since 2026-11-04 (the user's choice); "photon" keeps the reading before.
- **Validation, step 1** (`check_charge_ladder.py`; C24–C216 per molecule, Draine field G0 = 10⁴, T = 100 and 1000 K,
  n(H⁺) = n_e, against the library ladder and WD01 graphite spheres of the same carbon in the Mathis field):
  - Regression: identical where the field reaches no state above the computed ones (C24 and C54 at every ψ, every
    molecule at ψ ≤ 3×10³); ⟨Z⟩ moves ≤ 0.006 at ψ = 10⁴.
  - At ψ = 10⁶ pahchem now reaches WD01's charge for N_C ≥ 96: 2.69, 3.50, 4.16 against 2.81, 3.27, 3.70 (C96,
    C150, C216). The library ladder stopped near +1.9. Below ψ ≈ 3×10⁵ pahchem stays 0.4–0.9 below WD01, the cation
    sticking of §2.2b.
  - α(H⁺)/WD01 at ψ = 10⁶ falls from 486 to 40 (C150) and from 1725 to 41 (C216): the pile-up at +2 had kept
    neutrals. Γ_PE/WD01 is 7.2–7.5 for N_C ≥ 96 there (9.3 before); Λ_rec rises ×1.6–1.9.
  - Stand-ins carry 6 % (C96), 92 % (C150) and 98 % (C216) of the molecules and of their IR power at ψ = 10⁶,
    19–30 % for N_C ≥ 150 at ψ = 10⁵. Above ψ ≈ 10⁵ the emission of the largest PAHs rests on the stand-in modes.
  - Lowering the stand-ins' 5–13.6 eV cross section by 5–10 % per charge moves ⟨Z⟩ by ≤ 0.09 and Γ_PE by ≤ 3 %.
  - No Coulomb explosions in a field cut at 13.6 eV: the top is out of reach.
- **Effect, Draine field (T = 1000 K):**
  - ψ ≤ 10⁴: ⟨Z⟩ is unchanged to 0.01 for C24–C150.
  - ψ = 10⁵: C150 goes from +1.26 to +1.59 (18 % above +2).
  - ψ = 10⁶: C150 goes from +1.89 to +3.49 (91 % above +2), C96 from +2.60 to +2.68.
  - The total IR power is unchanged to 1 %.
- **Effect, Kurucz 40 kK field to 100 eV** (G0 10⁵, n_e 100, 10⁴ K): C54's charge peaks at +3/+4 with a tail to +9.
  Its fraction at +10 is 8×10⁻⁸ and its Coulomb-explosion rate 4×10⁻¹³ s⁻¹.
- **Validation, step 3: the PDRs4All refit** (`fit_pdrs4all_spectra.py --fit-shape` with every default of
  2026-11-02; `results/pahchem/validation_2026-11-02/pdrs4all_default.txt`), against the same capture and attachment
  on the library ladder and BASIS 8 (`results/pahchem/pdrs4all_capture_2026-11-01/att_size_50.txt`):
  - Every template fits as before: rms 0.047, 0.097, 0.073, 0.053, 0.048 (HII, Atomic, DF1–DF3, best profile;
    0.049 for DF3 before). The best γ (1.4–1.5×10⁴, interior; 260–330, the card's low edge) and ⟨N_C⟩ (77–87) are
    unchanged.
  - In HII and Atomic, Z ≥ 2 emits 9–10 % of the PAH power at the best cell (3–4 % before), +3 0.9 %. Stand-in
    states emit 5 %: mostly the dications of the 676 molecules with no computed one, on their cation's modes, which
    the library ladder had stopped at +1. The DFs (γ ≈ 300) hold nothing above +1.
  - So the spectra do not tell the two ladders apart; what changes is the dications' share, which rests on stand-in
    modes for half the library.
- **Validation, step 4: EUV fields** (`check_charge_ladder_euv.py`; C24–C216, Kurucz 40 kK at G0 = 10⁴ cut at
  13.6 eV and extended to 100 eV, T = 10⁴ K, n(H⁺) = n_e, ψ = 10²–10⁶). It is compared with WDB06 graphite spheres of
  the same carbon in the same field (pyCALIMA's `yield_model="WDB06"`):
  - At 100 eV every state to the cold limit is live. EUV photons raise ⟨Z⟩ in both models. At ψ = 10⁶ pahchem goes
    from 2.4 to 6.9 (C54) and from 4.7 to 12.7 (C216); WDB06 from 2.3 to 5.7 and from 4.0 to 9.6. For N_C ≥ 54
    pahchem stays below WDB06 up to ψ ≈ 10⁵ (the cation sticking, §2.2b) and rises above it beyond.
  - Per state, pahchem's photoionisation rate is 1.2–1.9× WDB06's for Z ≤ +1 (absorption). It is 1.6–9× at +2 to +4,
    4–35× at +5 and +6, and 50× up to 3×10⁷× from +10 up. Two causes:
    - The thresholds climb more slowly: the DFT step, 1.54 eV per charge for C216, against the sphere's 1.94 eV.
      They are 0.6–2 eV lower at +3 to +5 and 3–8.5 eV lower at +10 to +20.
    - Mostly the yields. pahchem's mean yield above threshold is 0.4–0.7 while IP < 20 eV and 1 above
      (Y_UNITY_EV). WDB06's falls from 0.7 at Z = 0 to 0.08–0.32 at +3 to +5 and ≤ 0.07 from +10 up: its electrons
      must escape over (Z+1)e²/a (y2).
  - On WDB06's photoionisation rates (pahchem's chain, capture and ion rates kept; exact on its own), pahchem's ⟨Z⟩ at
    ψ = 10⁶ falls to 4.1 (C54) and 7.1 (C216), below WDB06 at every ψ for N_C ≥ 54. Its higher charge comes from the
    photon side alone and outweighs its higher sticking.
  - Γ_PE is 0.9–1.9× WDB06's at ψ ≤ 10³ and 5–13× at ψ ≥ 10⁴ (C24 3.5–21×). Stand-ins emit nearly all the IR at
    ψ ≥ 10⁵ for N_C ≥ 150.
  - Coulomb explosions: C24 reaches its cold limit (+4) and explodes at 1.6×10⁻⁸ s⁻¹ (ψ = 10⁴) to 1.2×10⁻⁵ s⁻¹
    (ψ = 10⁶) at G0 = 10⁴; C54 at 2×10⁻⁹ s⁻¹ (ψ = 10⁶); the larger PAHs never.
  - So above +3 the charge in EUV fields rests on unit yields above max(20 eV, IP) and on the continued DFT step.
    Neither is measured; WDB06's bulk-grain yield is not measured for molecules either.
- **The Coulomb-escape bound** (opt-in, `Params.coulomb_escape`, 2026-11-03; `photo.yield_coulomb_window`):
  - *The literature* (two searches of 2026-11-03, quotes checked against the full texts):
    - No yield of any PAH ion beyond +1 → +2 is measured or computed. The C60^q+ series (Scully+05, q = 1–3) is
      from ions of ≥ 100 eV internal energy (ECR source; Douix+17, whose cold-ion C60⁺ cross section reaches the
      computed absorption at 22 eV).
    - WD01's escape factor y2 is bulk physics: its parabolic energy distribution comes from bulk photoemission
      (their footnote 3, Draine 1978), and their yields are fitted at Z = 0 only, on coronene. WDB06's eq. 15
      (a ≤ 6 Å) restores WD01's y0 and keeps y2. Only codes that copy WD01 use it for PAHs (SHIVA, Murga+22, Cloudy);
      the molecular models shift the lab ramp to IP(Z) "due to the lack of available data" (Sidhu+22).
    - At Z = +1 the measured yields are about twice what y2 allows. At 15 eV, C30–C36 cations give 0.83–0.88 and
      C48H20⁺ 0.98 (Wenzel+20), against y2 ≤ 0.35–0.48; cold C60⁺ gives ≈ 1 at 14–25 eV (Douix+17, ±21 %) against
      0.41–0.68.
  - *The physics.* An absorption above IP(Z) makes a particle–hole excitation. The electron's energy at infinity,
    E′, lies between −D and E − IP(Z), with D = (Z+1) B and B the ion's IP step per charge (e²/C; the IPs are
    linear in q, Holm+11). Electrons with E′ > 0 escape. Those with E′ < 0 are bound by the ion's Coulomb field in
    superexcited states, whose energy is above IP(Z), so they can still autoionise. WD01's y2 is the limit where
    none of them does; pahchem's unit yield the limit where all do. This does not double-count the Coulomb term in
    IP(Z): E − IP(Z) uses it once, and D bounds the states it binds.
    - A bound state at fractional depth u = −E′/D turns at r = a/u at every Z. It returns to the molecule with a
      period ∝ (a/u)^{3/2} / √(Z+1) and crosses it faster at higher Z. So its autoionisation rate does not fall with
      Z at fixed u: it rises ∝ √(Z+1) while the molecule's inner potential sets the crossing speed, and is flat
      once the Coulomb drop does.
    - The competitor is relaxation into vibrations. Excited PAH cations relax by internal conversion "within a
      few tens of fs", 29–55 fs from naphthalene to tetracene, slower with size (Marciniak+15). Then
      p(u) = 1 / (1 + (u_c/u)^{3/2}) has the same u_c at every Z, or a smaller one.
    - u_c = 0.354 (`photo.ESCAPE_DEPTH`) is the largest the cation data allow. It reproduces C30H14⁺'s 0.83 at
      15 eV less the 21 % calibration error, taking every shortfall (fragmentation too) as the window's.
    - The yield is then a function of (E − IP)/D alone: 0.58 at threshold (every electron starts bound), 0.55 at
      0.14 D above it (the parabola's dip), 0.73 at D, 1 far above. WD01's y2 gives ≤ 0.2 over the first
      0.3 D. "window" caps the lab yields of Z ≥ +2 (and the top state's) at this; "y2" at u_c = ∞, for comparison.
      Z ≤ +1 keep their measured laws.
  - *Not included:* the threshold law of the autoionising channel, the energy of the autoionised electrons (the
    photoelectric heating keeps `electron_energy`), and multiple ionisation.
  - *Step 4 with the bound* (`check_charge_ladder_euv.py`; `results/pahchem/validation_2026-11-03/`):
    - In fields cut at 13.6 eV "window" changes nothing: below 13.6 eV the lab ramps of Z ≥ +2 lie under the cap.
      "y2" lowers ⟨Z⟩ by ≤ 0.37.
    - To 100 eV, "window" scales the photoionisation rates by 0.91–0.95 at +2, 0.63–0.94 at +4 and 0.56–0.57 from
      +10 up; "y2" by 0.56–0.87, 0.20–0.60 and 0.002–0.09.
    - At ψ = 10⁶, ⟨Z⟩ for C24–C216 is 3.84, 6.89, 8.70, 10.85, 12.74 by default, 3.75, 6.18, 7.80, 9.77, 11.47 with
      "window" (0.1–1.3 lower) and 3.29, 4.49, 5.74, 7.23, 8.42 with "y2". WDB06 gives 4.14, 5.66, 7.04, 8.34, 9.57.
      With the bound pahchem still exceeds WDB06 at ψ = 10⁶ for N_C ≥ 54, and stays below it up to ψ ≈ 10⁵.
    - Γ_PE/WDB06 at ψ = 10⁶ falls from 21.5 to 14.4 (C24) and from 10.7–12.6 to 10.5–11.8 (N_C ≥ 54). These are with
      the "photon" photoelectron reading; the "neutral" default (2026-11-04) multiplies Γ_PE at ψ = 10⁶ by 0.49–0.54.
    - C24's Coulomb explosions halve (1.2×10⁻⁵ → 6.0×10⁻⁶ s⁻¹); C54's fall from 2.2×10⁻⁹ to 1.6×10⁻¹⁰ s⁻¹.
    - So the escape physics, held to the cation data, lowers the EUV charge by ≲ 10 %. pahchem's excess over WDB06 at
      the highest ψ comes from the bound electrons that autoionise, which WD01 drop and the cation data require: with
      "y2" pahchem falls below WDB06 at every ψ for N_C ≥ 54.
  - *Why it stays an opt-in.* On their own, the central yields allow u_c ≤ 0.12 (C30H14⁺), 0.11, 0.08 (C32, C36),
    0.013 (C48H20⁺) and 0.023 (C60⁺), each an upper limit since fragmentation lowers them too. At u_c ≤ 0.02 the
    high-charge yield is ≥ 0.97, unit yield within the errors. 0.354 needs every measurement lowered by its full
    calibration error. The bound is the systematic of the default, not a better default.
- **Double ionisation below 291 eV** (opt-in, `Params.double_ionisation = "knockout"`, 2026-11-03):
  - *The data* (full texts):
    - Neutral PAHs (Hartman+13, PRA 87, 063403): the parent-ion ratio M²⁺/M_tot of coronene is 5, 10, 15 and 16.5 %
      at 12, 18, 27 and 33 eV above its double threshold of 18.8 eV (read from their Fig. 8). Pentacene's M²⁺/M⁺ is
      ≈ 30 % at 30–130 eV above (Fig. 9). Parent ions are a lower limit: dications fragment before detection, and
      benzene's ratio at 75 eV is 5× larger counted by photoelectrons.
    - C60 (Kafle+08, JPSJ 77, 014302; every charge-2 product): σ(2+) peaks at 24.8 Mb at 50 eV; read from their
      Fig. 2 (±20 %), double over all ionisations is ≈ 0.14 at 50 eV, 0.25 at 80 eV, 0.3 at 100–120 eV.
    - C60⁺ (Baral+16, Tables I–III): 0.9 % at 35 eV, 11 % at 65, 15 % at 75, 19–22 % at 90–140 eV; threshold
      30.2 ± 2 eV.
  - *The curve* (`photo.double_fraction`): P2 = 0.25 x²/(x² + 22²), x = E − IP(Z) − IP(Z+1) [eV], through all four within
    ~40 % (0.06, 0.10, 0.15, 0.22 at x = 12, 18, 27, 60 eV). Nothing is measured above +1; the same curve is taken at
    every charge.
  - *The chain* (`population._chain_fractions`): +2 steps make it no longer a birth–death chain; its steady state is
    still exact as a recursion over cuts, f(i+1) down(i+1) = f(i)(k1 + k2)(i) + f(i−1) k2(i−1), checked against a dense
    solve. A double step out of the state below the cold limit is an explosion (`chain_overflow`). The two electrons
    share the excess over IP(Z) + IP(Z+1) as one would; the bases (ionising photons deposit nothing) are unchanged.
    Not in the group tables (they refuse it) or in the recombination heating's per-state rate.
  - *Step 4:* 0.5–1 % more electrons per second at Z = 0 and +1, none above: the star has few photons 20–30 eV above
    the double thresholds. ⟨Z⟩ moves ≤ 0.01 and Γ_PE ≤ 1 %, alone or with "window". In power laws F_ν ∝ ν⁻¹ to ν⁻¹·⁵
    over 13.6–284 eV a molecule gives 1.006–1.034 electrons per ionisation (C24–C216, Z = 0 to +20), because its
    absorption sits at 13.6–30 eV where P2 ≈ 0. So below 291 eV double ionisation is a ≤ 3 % effect; the recursion is
    what stage 3's Auger dz = +2 needs.

**Energies above 21 eV, which the ladder needs:**
- **Deposits.** A state that absorbs without ionising above E_DEP's top (21 eV) gets the photon nodes above it on
  its deposit grid (`deposit_grid`). Its emission cascade then reaches 0.5 eV above its hardest deposit and its
  recombination energy.
- **State counts** overflow float64 for large PAHs at high energy: C268 at 30 eV, C150 and C216 at 45 eV, C96 at
  60–90 eV. They are now tilted where they would, c(E) e^(−βE) (`dos.count_tilt`). The recursion stays exact, every
  rate takes the same β and the tilt cancels: forced on a 40-mode set, every rate agrees to ≤ 10⁻¹⁰. Where no tilt is
  needed (β = 0) the arithmetic is unchanged: C54H18⁺'s basis rebuilds bit-identical to the cached one.
- **Multiphoton H loss.** Until now the loss ceiling dropped from 45 to 30 or 20 eV for N_C ≳ 150, where the counts
  overflowed. That cut multiphoton heating short: C150's k_H at G0 = 10⁶ was 46× low (×2.3 at 10⁴, ×5.8 at 10⁵).
  The ceiling now starts at 45 eV (or the same headroom above a state's hardest deposit). It rises by 15 eV until
  the jumps clamped at it at G0 = 10⁶ are below 1 % of the loss rate, up to 150 eV. C54 and C96 converge at 45 eV,
  C150 at 60 eV and C268 at 75 eV (C268's k_H at G0 = 10⁶ is 2.3× its 45 eV value). `BASIS_VERSION` 9.

### 2.2d Cosmic rays at the constant zeta of RAMSES-RTZ (2026-11-04)

- *What RTZ takes.* RTZ takes one input, the primary ionisation rate per H, `rtz_primary_cosmic_ray_ionization_rate`
  (ζ_H). Its total is ζ_H (1 + φ_s(x_e)), with φ_s the secondaries (0.67 in neutral gas), and ζ(H₂) = 2 ζ_H
  (`rtz/rtz_cooling_module.f90`). The CR-induced UV rates of the metals scale as ζ(H₂)/10⁻¹⁶.
- *The standard spectrum* (`cosmic_rays.standard_spectra`): Padovani+18 protons with the PDG nuclei and primary
  electrons, attenuated (CSDA) through N(H₂) and scaled so that its primary ζ(H₂) is 2 ζ_H. The H₂ cross sections are
  Padovani+09's, as CRISpy has them (Rudd+85 protons, Rudd 1991 electrons; nuclei the proton's at equal speed ×
  z_eff²).
  - An attenuated shape is necessary: unattenuated, primary electrons below ~1 keV (j ∝ E⁻¹·³) give ζ(H₂) =
    2.8×10⁻¹⁴ s⁻¹.
  - Through 10²¹ cm⁻² the shapes give ζ(H₂) = 3.0×10⁻¹⁷ (model L) and 6.0×10⁻¹⁶ s⁻¹ (H). ≤ 2 % of that comes from
    below `impact`'s passages (1 keV electrons, 100 keV/u).
  - *The blend* (the user's choice, 2026-11-04; `cosmic_rays.blend_weights`). L and H share the measured high-energy
    spectrum: both tend to C E⁻²·⁷ with the same C. They differ only by H's low-energy excess. A cell of given ζ_H
    takes L plus as much of that excess as gives its ζ, (1 − w) L + w H. Outside [ζ(L), ζ(H)] = [1.5×10⁻¹⁷,
    3.0×10⁻¹⁶] per H the end member is scaled. Every rate is linear in j, so the population keeps the passages of
    the two unscaled shapes and combines them per run (`Population._cr_rates`). Its rates per ζ therefore depend on
    ζ, fixed in a run as RTZ's ζ is.
- *In the population* (`Params.cosmic_rays`, `Environment.zeta_cr`; `population.cr_passages`):
  - **Passages.** Each state's passages per unit ζ_H come from `impact.passage_spectrum`: Poisson collisions per
    passage, distant ionisations leaving the photons' mean hole (`impact.deposition`, partition "measured", the
    default since 2026-11-04 instead of "uniform"; the C 1s escape tables use it too).
  - **Charge.** Their +1 … +15 steps join the charge chain.
  - **Heat.** Their E* goes on `HEAT_NODES_EV`, a "heat" response per product, dz = 0 included, so it adds to the IR
    and the H loss.
  - **Not followed.** Steps past the cold limit are Coulomb explosions (`chain_overflow`). Passages beyond 320 eV or
    15 charges are not followed (`cr_beyond`): an upper bound on destruction until the carbon loss is in.
  - **Live states.** Every charge is live (more bases).
  - **G0 below 1.** The k_H tables continue their first log-log slope below G0 = 1, for shielded gas.
- *Per unit ζ_H* (`check_cosmic_rays.py`, C24–C216):
  - **Ionising passages.** Neutral C54H18 gets 395 s⁻¹ per s⁻¹ with L through 10²¹ cm⁻², ⟨dz⟩ 1.13, 3.9 keV of
    heat; C24 172, C216 1624. Model H gives 0.55–0.64× the ionisations and 0.70–0.77× the heat, but 2–5× more
    passages beyond.
  - **Column dependence.** Within one model, the rates change by ±15–20 % from 10²⁰ to 10²² cm⁻².
  - **Shape.** For PAH rates per ζ the shape matters at a factor ~1.7. H's ζ is carried by sub-MeV protons, which
    ionise H₂ well but PAHs less: their low-speed stopping is below Bethe.
- *In clouds* (dark: Draine G0 10⁻³, n_H 10⁴, n_e 10⁻⁴, T 10 K; translucent: G0 1, n_H 10³, n_e 0.1, T 50 K):
  - **Charge, hydrogenation, IR.** Direct passages change ⟨Z⟩ by ≤ 0.003 and f(N_H0) not at all. Their heat adds
    0.2–0.3 % of the IR at ζ_H = 10⁻¹⁶ in the dark cloud (2–3 % at 10⁻¹⁵) and nothing at G0 = 1. Attachment
    (~10⁻⁹ s⁻¹) beats CR ionisation (~4×10⁻¹⁴) by orders of magnitude.
  - **Destruction** (the blend). Lifetimes are 1.1–2.8×10⁹ yr at ζ_H = 10⁻¹⁷ (L alone, scaled),
    0.64–1.6×10⁸ yr at 10⁻¹⁶ (w ≈ 0.3) and 0.61–1.5×10⁷ yr at 10⁻¹⁵ (H scaled). Coulomb explosions dominate for C24
    and C54, passages beyond 320 eV for N_C ≥ 96, and H's low-energy nuclei shorten them. This is Micelotta+11's
    order.
  - **Missing piece.** In dark gas the CR-induced UV (H₂ Lyman–Werner emission) matters more for the charge and
    heating; it is not in yet.
- *Open:* the column of the blend (10²¹ cm⁻²); the secondary electrons (CRISpy); the induced UV.

### 2.2e Proton emission by hot ions and C₂H₂ loss (2026-11-04)

- *Sources:* Holm+11 (J. Chem. Phys. 134, 044301), West+14, West+18, Ling & Lifshitz 1998, Jochims+94, all read in full.
- **C₂H₂ loss** (`Params.c2h2_loss`, `population.loss_extras`):
  - **Measured.** RRKM fits exist only for small catacondensed cations: anthracene⁺ 4.50 eV and +12.8 eu, phenanthrene⁺
    3.99 eV and +12.3 eu, both loose transition states (Ling98 Table 2); acenaphthylene⁺ 5.21 eV (West+18).
    "Perylene (PER) and pyrene (PYR) cations do not lose hydrocarbon fragments" (West+18), and coronene⁺ only
    dehydrogenates. DFT puts the pericondensed C₂H₂ dissociation energies at 6–8 eV against anthracene's 2.7–3.3
    (Holm+11 Table V). Pyrene⁺'s C₂H₂ share grows as it is stripped: 1.3 %, 8 % at C₁₆H₈⁺, 50 % at C₁₆H₅⁺ (West+14).
  - **Model.** A channel only for skeletons with catacondensed end rings (PAHdb's quartet H): Ling98's anthracene⁺
    parameters, path degeneracy ½ per end ring, the C–C stretch as reaction coordinate. None for pericondensed parents.
    It destroys the molecule as the library holds it (`Population.loss_rates`).
  - **Not yet.** The deeply dehydrogenated states, where West+14 see it open, are part of the bare-skeleton item.
- **Proton emission** (`Params.fission`):
  - **Barrier.** Holm+11's DFT fission barriers for H⁺ emission (Fig. 7, five PAHs, q = 3–10, read from the figure)
    collapse on q/√N_C, the charge over the radius: B = 6.30 − 2.88 q/√N_C eV, rms 0.28 eV (q/N_C: 0.53).
    `impact.fission_barrier`.
  - **Threshold.** max(D, B), with D = the H loss's E0 + IE(H) − IP(q − 1) from the ladder (within 0.1–1 eV of Holm's
    Table V), and the H-loss transition state (an assumption: Holm computed barriers, not transition states).
    The proton takes the charge (q → q − 1) and an H, so it is a down-step of the charge chain and an H loss of the
    hydrogenation chain. The two are solved together (`charge_matrix`: per-molecule Anderson mixing to 10⁻¹²,
    6–86 iterations).
  - **Not modelled.** C₂H₂⁺ emission, computed for naphthalene only.
- **The top of the ladder** (the user's choice, after a literature check): min(⌊0.2 N_C⌋, the highest q with B > 0,
  2.19 √N_C), so C150 has +26 (was +30) and C216 +32 (was +43). Both the thermodynamic and the barrier limits scale with
  the radius. No astrophysical model prescribes either law:
  - PAH codes truncate (Murga+22: −2 … +8; Montillaud+13: 0, +1).
  - Grain physics (WDB06 §10.2.1, Hoang+15: Coulomb explosion or ion field emission) scales as a² ∝ N_C^(2/3), C216
    ~12–14. It puts coronene and C₆₀ at ~3, against coronene⁴⁺ and C₆₀¹²⁺ seen intact, so for molecules it is a lower
    bound.
- *Thresholds* (`check_fission_c2h2.py`): proton emission falls below the 4.41 eV H loss from +4 in C24 (3.95 eV), +5 in
  C54, +8 in C96, +12 in C216, and to 0.03–0.7 eV at the tops of C96–C216.
- *Effects* (off → on):
  - **PDRs and EUV.** ⟨Z⟩ moves ≤ 0.01. Only highly charged molecules that are still intact are stripped: C96 at
    ψ = 10⁵, f(N_H0) 0.967 → 0.463; C150 at 10⁶, 1.000 → 0.292. The small ones are bare there already.
  - **X-rays only, G0-equivalent 10⁴.** Hot C 1s products shed protons, with proton emission outrunning electron capture
    by up to 40×. ⟨Z⟩ falls by 0.16 (C24), 0.91 (C54), 1.18 (C96), 0.99 (C150) and 0.74 (C216). f(N_H0) rises for
    C54 and C96 (0.773 → 0.872, 0.820 → 0.865): lower charges deposit less on recombination.
  - **C₂H₂ loss.** Catacondensed C22H14 isomers lose C₂H₂ at ~5×10⁻⁸ s⁻¹ in X-ray gas (G0-equivalent 10⁴), from the
    C 1s products' heat: the loose transition state beats H loss at high energy, as for anthracene⁺. In PDRs it is
    10⁻¹²–10⁻¹⁰ s⁻¹ there, because those molecules are stripped and their fate is the bare skeleton's.

### 2.2f The bare skeleton and the destruction of PAHs (2026-11-04)

- *The skeleton's modes:*
  - **PAHdb's own.** PAHdb v4.00 holds 812 N_H = 0 entries. 204 of its 3107 H-bearing neutrals have their own skeleton
    computed, matched by carbon framework (`ames_suite` `carbon_graph`, `Molecule.bare`): N_C 14–34, 54, 66, 96. In
    `library_v400` that covers 19 molecules, among them C24, C54, C66 and C96, often at several charges.
  - **Calibrated, for the others.** The parent's `stripped(0)` keeps its frequencies, which set the state counts, but
    it lacks the C–C bands that carry a skeleton's emission and so cools 4–12× too slowly. Its Einstein A are therefore
    multiplied by `BARE_IR_LAW`, R = 54.7 N_C^−0.52: PAHdb skeleton over `stripped(0)` thermal IR power at 800 K,
    166 pairs, rms ×2.0 about the law (C54 5.5, C66 3.4, C96 4.4). The cooling of an uncomputed skeleton is uncertain
    by that factor.
- *Its destruction* (`bare_channels`, `bare_loss_table`):
  - **Carbon loss.** Léger, d'Hendecourt, Boissel & Désert 1989 (A&A 213, 351, Table 2, checked on the scan): C, C₂
    and C₃ loss in the inverse-Laplace form (their eq. 8), A and B from graphite sublimation at 2000–4000 K (C 7.37 eV,
    6.2×10¹⁵ s⁻¹; C₂ 8.49, 3.5×10¹⁷; C₃ 7.97, 1.5×10¹⁸), as `CARBON_LOSS` held them.
  - **Folding.** For N_C ≥ 60, folding into a cage (Berné, Montillaud & Joblin 2015 Table 2: 4.0 eV, 10¹⁵ s⁻¹): the
    PAH becomes a fullerene.
  - **How it runs.** The channels compete in the skeleton's multiphoton cascade, heated as the charge state, with
    single-event yields for the deposits the tables do not hold.
  - **Output.** `Population.destruction_rates` weights them by the skeleton's share f(N_H = 0) of the hydrogenation
    chain and adds C₂H₂ loss, Coulomb explosions and cosmic-ray passages beyond 320 eV. It gives the events
    (1/lifetime) and the carbon lost. The chain is quasi-static: these are slow leaks from its steady state.
- *Lifetimes* (`check_bare_skeletons.py`; C24–C216):
  - **PDRs** (Kurucz 40 kK, n_H 10⁴, n_e 1). Bare C24 lasts 22 yr at G0 = 10³, 1.2 yr at 10⁴ and 11 days at 10⁵. C54,
    10 % bare at 10⁵, lasts 1.9×10³ yr. N_C ≥ 96 stays hydrogenated and survives. That is the critical size of Jochims+94:
    below ~30–40 C, PAHs dissociate in the far UV.
  - **EUV.** C54 lasts 230, 10 and 0.07 yr at ψ = 10⁴, 10⁵, 10⁶. Stripped C96 folds into a cage in ~5 days at ψ = 10⁶.
  - **X-rays only.** The molecules stay hydrogenated, and Coulomb explosions from C 1s multi-ionisation dominate: at
    G0-equivalent 10⁴, C24 1.2 yr, C54 20 yr, C96 7×10³ yr, C216 9×10⁵ yr.
  - **Cosmic rays** (§2.2d): 0.6–1.6×10⁸ yr at ζ_H = 10⁻¹⁶.
- *Not yet:*
  - C₂H₂ loss in the deeply dehydrogenated states (West+14);
  - the skeleton's IR in the spectra;
  - the ionisation threshold of the bare skeleton (it takes the parent's).

### 2.2g Hot gas: thermal electrons and ions (2026-11-04; steps 1–2 of replacing `PAH_sputtering`'s dissociation)

- *Why.* `models/PAH_gas_collisions/PAH_sputtering.py` (Micelotta+10a,b) computes the energy T₀ a collision leaves. It
  then turns T₀ into a dissociation probability with an outdated recipe: an Arrhenius rate at T_eff = 2000 K
  (T₀/N_C)^0.4, for C₂H₂ loss over the tuned 4.6 eV, against k_IR = 100 s⁻¹ and n_max = N_C/5 photons. The user chose
  to move hot-gas collisions into pahchem, so that collisions (`impact`) and dissociation (the population's cascades,
  §2.2e–f) have one owner each.
- *Thermal electrons* (`Params.hot_gas`, opt-in):
  - **Passages.** `impact.maxwellian_flux` gives j(T) = n v f(T)/4π at `Environment.temperature_k` and n_e, through
    `impact.passage_spectrum`. The energies run from 1 eV to 40 kT, and the collisions are Bethe plus Kim's binary-encounter
    scaling for slow electrons, as the C 1s escape tables use.
  - **Coulomb focusing.** Classical: on a charge-Z PAH the passage rate is ×(1 − qZe²/RT) and the impact energy T − qZe²/R
    (`impact.coulomb_focus`), as `PAH_sputtering` shifts its energies.
  - **In the population.** `population.hot_passages` feeds charge steps, heat on the nodes ("heat" responses) and passages
    beyond, as the cosmic rays do (`Population._passages` sums both). Every charge is then live.
- *Per unit n_e* (C24–C216, Z = 0): collision rates of 0.7–12×10⁻⁶ cm³ s⁻¹, 40–80 % of them ionising, and
  ⟨E*⟩ = 8–14 eV per passage, at 10⁵–10⁷ K.
- *Hot, ionised gas* (n_e = 1, n(H) = 10⁻⁴, a weak G0 = 1 field):
  - **Effect.** The electrons raise ⟨Z⟩ by 0.3–2.4 and strip C24–C96 completely. Their bare skeletons then lose carbon,
    and C96 folds into a cage: lifetimes 0.02–0.4 yr, ∝ 1/n_e. Without electrons, 10⁴–10²⁵ yr.
- *Against Micelotta+10b's model* (A&A 510, A37, "PAH processing in a hot gas"; its companion on shocks, A36, is
  Micelotta+10a or MJT; `check_hot_gas_micelotta.py`, `pahchem_hot_gas_micelotta.png`). Step 1 compared the electrons
  alone; step 2 below adds the ions.
  - **Reading the published fits.** Their Table 2 fits of the electrons' carbon-loss rate constant reproduce their
    Table 3 lifetimes to 4–10 % (M82 156 vs 162 yr; the 2 C per event are in the fits). Their N_C = 50 electron fit is
    printed with a = +2136.83; with −2136.83, as for N_C 100 and 200, it gives their Fig. 7.
  - **This repo's implementation** (`PAH_sputtering`) runs 0.2–0.76× of their published rates.
  - **N_C ≈ 50 (C54H18).** Above 3×10⁵ K pahchem loses carbon at 0.05–0.4× Micelotta's rate (lifetimes 1–11 yr against
    0.2–2 yr at n_e = 1), and 2.7× faster at 10⁵ K. It is destroyed as a PAH in 0.05–0.6 yr, 1–4× sooner than
    Micelotta's carbon-loss lifetime.
  - **N_C ≈ 100–200 (C96H24, C216H36).** Once stripped, the bare skeleton folds into a cage, which keeps its carbon.
    Carbon is lost at only 10⁻³–10⁻⁴ of Micelotta's rate, but the PAH is gone in 0.02–0.6 yr, 2–100× sooner than
    Micelotta's 0.4–60 yr (most so for N_C 200 at high T).
  - **At 3×10⁴ K** both destroy far faster than Micelotta, whose electrons do almost nothing there.
  - **The pathways differ.** Micelotta's is C₂H₂ loss from the intact molecule over the tuned 4.6 eV. pahchem's is
    stripping (measured H loss, proton emission), then the bare skeleton's carbon loss (Léger+89) or folding (Berné+15).
    pahchem's electrons also ionise (⟨Z⟩ 1–2) and are focused.
  - **pahchem needs inputs Micelotta's model does not.** Its rates depend on the re-hydrogenation (n(H) = 10⁻⁴ n_e here)
    and on the field (G0 = 1). Their 2/3-of-the-carbon criterion becomes "carbon lost" or "destroyed as a PAH".
- *Thermal ions* (step 2, 2026-11-04; `Params.hot_gas`, the ions of `Environment.ions`):
  - **Electronic friction.** `impact.thermal_ion_passages` ports Micelotta+10b's slow-ion stopping (eqs 1–7). The disc
    holds a jellium of density 0.15 exp(−x²/2.7) a.u. at height x; its column is graphene's four valence electrons per C.
    The friction is Puska & Nieminen's, γ = Γ₀ exp(−(r_s − 1.5)/R₂), and an ion loses v ∫γ ds along its path, linear in
    the speed (v ≲ 1 a.u.).
    - Micelotta print R₂ only (H 2.28, He 0.88, C 0.90). Γ₀ (0.33, 0.75, 1.68; O 1.62, 0.57) is `PAH_sputtering`'s and
      has no source: Puska & Nieminen 1983 (PRB 27, 6121) is needed to check it. Other ions get no friction.
    - `impact.disc_chords` follows every chord at every angle and offset, so the mean deposit is exact (int γ dV, to
      8×10⁻⁴). Micelotta take the central chord at each angle, 4/π too long edge-on, with no spread.
  - **Elastic collisions below threshold are heat.** `impact.knock_on` gives the ZBL stopping of transfers below the
    knock-out threshold, S_n0 min(1, T₀/T_m)^(1−m) (the complement of MJT eq. 14), per C and per H atom. MJT leave it out
    ("radiated away"), but in pahchem every deposit is heat. On C54H18 it is 6–10× the friction's for He and C at
    10⁴–10⁵ K, 1.6–3× at 10⁶ K and 0.4–0.8× at 10⁷ K; for H, 2.6×, 0.5× and 0.1× at 10⁴, 10⁶ and 10⁷ K. The H atoms take
    10–45 % of it; they sit on the rim but are spread over the disc like the C atoms.
  - **Knock-outs** remove C atoms (one C each in `destruction_rates["knockout"]`, a molecule leaving the library) and H
    atoms (an H loss in the hydrogenation chain, N_H times the rate per atom, `_deposit_losses`).
    - **Cross section.** MJT eq. 15, zero below E₀ₙ = T₀/γ.
    - **Thresholds.** T₀(C) = 23.3 eV, Stockett+15's MD energy transfer at their measured He + PAH⁺ thresholds (their MD
      alone 27.0; Postma+14 27; graphene 23.6). MJT took 7.5 (4.5–15) eV, below every later number. T₀(H) = 9 eV, Chen+14
      after Postma+14's MD.
    - **No factor 1/2.** MJT's eq. 22 multiplies by 1/2, their Appendix C average of the projected area, ⟨cos θ⟩. A
      per-atom cross section already samples every orientation: the atom column an ion crosses grows as 1/cos θ where
      the projection shrinks as cos θ. No later work uses the factor.
    - **Shadowing** of the molecule's atoms by each other is not followed. Chen+14's Monte Carlo puts whole-molecule
      cross sections at 0.44 (H) and 0.74 (He) of the atoms' sum, for their own per-atom cross section.
  - **Coulomb focusing** on the charged PAH as for the electrons.
  - **Population.** `population.hot_ion_passages` (heat at dz = 0, knock-outs) per unit ion density, summed in
    `_hot_rates`.
- *Ions against Micelotta+10b:*
  - **Implementation check.** Table 2's ion fits give Table 3's Orion lifetime, set by the ions, to 1.8×10⁷ vs 1.3×10⁷ yr.
    At MJT's T₀ = 7.5 eV, pahchem's knock-outs are 1.7–2.8× their ion fits at 3×10⁴–10⁵ K, where the nuclear part
    dominates, for H, He and C on N_C 50 and 200. That is their factor 1/2: the ZBL cross section is reproduced.
  - **`PAH_sputtering`'s nuclear rate** applies T₀ to the ion's energy where MJT eq. 15 has E₀ₙ = T₀/γ. It
    overestimates H knock-outs ×3000 at 3×10⁴ K and ×24 at 10⁵ K. He and C are close (γ = 0.75, 1).
  - **In gas of n_H = n_e = 1** (Micelotta's ions: H⁺ 1, He⁺ 0.1, C⁺ 10⁻⁴), the ions change pahchem's carbon loss
    against the electrons alone:
    - N_C 50: +1 % at 10⁶ K, +56 % at 10⁷ K, ×25 at 10⁸ K (0.79× Micelotta's electrons + ions there);
    - N_C 100–200 at 10⁶–10⁷ K: ×3–30. The knock-outs (T₀ 23.3 eV) are 50–98 % of the carbon lost there, because the
      stripped skeletons fold with their carbon.
  - **Knock-outs at T₀ = 23.3 eV** are 10⁻³ of MJT's at 3×10⁴ K, and 0.4–0.8 of them above 10⁶ K.
  - **Destruction as a PAH** stays electron-driven (stripping, folding): 0.03–0.4 yr for N_C 50 above 10⁵ K, at n_e = 1.
  - **At 10⁸ K** the friction's linear rise is used past v = 1 a.u. in the Maxwellian's tail. H's true stopping peaks
    near 100 keV, so the ions' heat there is an overestimate, up to ~2× in the tail.
- *Found while doing it (step 3):*
  - Electron capture (`capture_coefficient`, DS87 × sticking 1 on cations) is used at any temperature. A 0.1–10 keV
    electron crosses a PAH losing ~10 eV and cannot be captured. At 10⁶–10⁸ K the capture rate (10⁻⁵–3×10⁻⁴ cm³ s⁻¹)
    competes with or beats the passages' ionisation, and C216H36 comes out at ⟨Z⟩ −0.4 to −0.9 at 10⁷–10⁸ K. The
    hot-gas ⟨Z⟩ above are lower bounds until capture requires the electron to be stopped (its passages' loss ≥ its
    energy).
  - The ions take no electrons from cations (`ion_up_rate`: a Coulomb barrier, valid below 10⁴ K), although at 10⁶ K
    kT ≫ qZe²/R.
- *Not yet:*
  - `PAH_sputtering`'s per-bin tables through pahchem (step 4);
  - the electron-impact ionisation's 1.3–2.3× excess over BEB (gate G2), inherited here;
  - the friction's Γ₀ against Puska & Nieminen.

### 2.3 Ion–PAH collisions (new)

**Starting point: Wolfire+08** (ApJ 680, 384), Table 2.
- It is the Wolfire+03 rate set, with Φ_PAH ≈ 0.4 fitted to C I/C II along 42 sight lines.
- DS87 for a sphere of radius (N_C/1.222)^½ Å with s = 1 reproduces every entry to ≤ 2.6 % (§A of the
  diagnostic).
- pahchem's geometry gives 0.79–0.82 × those values.

**Mutual neutralisation, X⁺ + PAH^Z (Z ≤ −1) → X + PAH^(Z+1):**

    k_MN = P_MN g πa² v_X J̃(τ_X, Z)

- No PAH⁻ + X⁺ rate has been measured. Three estimates of P_MN, in pahchem's geometry for C₅₀H₁₈⁻ at
  300 K, for every ion (§B of the diagnostic):

  | law | P_MN |
  |---|---|
  | Tielens 2021 eq. 8.112 ("by lack of better": the electron form with the cation mass) | 0.46 |
  | Shuman+14 (VENDAMS), 2.7×10⁻⁷ μ^−½ at 300 K, measured on polyatomic cations with halide anions (≥ 4 atoms) | 0.58 |
  | Hickman 1979 eq. 28, 5.35×10⁻⁷ μ^−½ (T/300)^−½ EA^−0.4 (EA 2.5–1.5 eV; fitted for EA 0.75–4 eV, ±30 %) | 0.80–0.98 |

  Lepp & Dalgarno (1988b), after Omont (1986), quote "of the order" 1.3×10⁻⁶ T^−½ scaling as m^−½ with no stated
  reference mass, so it cannot be placed in this table.

- Proposed default: P_MN = 0.58, the measured polyatomic law. The range 0.46–0.98 is its uncertainty. It
  is labelled an assumption, since it transfers a small-anion law to PAH anions.
  - Both lab laws are size-independent. Relative to DS87, P_MN therefore falls as N_C^−½. A constant P_MN
    quietly assumes DS87's N_C^½ scaling, and that is stated wherever it is used.
  - The T dependence follows DS87 (T^−½), as the C₆₀⁺ + C₆₀⁻ Landau–Zener calculation does (Gatchell+25).
    The T^−0.9 over 300–550 K is Miller+12's (JCP 136, 204306: polyatomic anions with Ar⁺ and Kr⁺), not Shuman+14's,
    which is 300 K only; pahchem uses neither exponent.
  - Molecular ions take 50 % dissociative products (Thi+20).
  - For comparison, Bakes & Tielens (1998) use a flat 10⁻⁷ (T/300)^−½ for every ion, which is 0.18–1.27 ×
    DS87.

**Ion–neutral reactions, X⁺ + PAH⁰ → products with ΔZ = +1, only if IP(X) > IP_m(0):**

    k = P_react · 2π e (α_m/μ)^½

- α_m = 1.75 N_C Å³ by default. Tielens (2021, Table 6.5) gives 125 Å³ for circumcoronene (2.31 per C), with
  α → N_C^{3/2} for large PAHs, so 1.75 N_C is ≈ 1.3× low at N_C ≥ 50. DFT polarisabilities are used where
  available. P_react ≈ 0.9 is the **total** reaction efficiency measured on anthracene at 293 K
  (Canosa+95 Table 1): He⁺ 5.1–5.4×10⁻⁹ and C⁺ 3 ± 1.5×10⁻⁹, i.e. 0.90 and 0.88 of Langevin with their
  α = 25.4 Å³.
- **It is not all charge transfer.** For C⁺ the products are 60 % C + C₁₄H₁₀⁺ and 40 % C₁₅H₉⁺ + H, carbon
  accretion (p. 412). For the charge balance both give ΔZ = +1. For the gas network, the second does not
  return C to the gas. He⁺ gives the parent ion, fragments and possibly the dication, with the branching not
  measured.
- **The efficiency is ion-dependent.** Ar⁺, exothermic by ≈ 8 eV, reacts at only ≈ 0.5 of Langevin. Applying
  0.9 to H⁺, N⁺, O⁺ and S⁺ is an extrapolation, labelled as such; no T dependence is measured.
- This is Langevin, not DS87: DS87's neutral sphere is Langevin with α = a³. With P_react = 0.9 the ratio to
  DS87 falls from 0.74–0.78 (N_C = 24) to 0.51 (N_C = 100) (§C).
- The threshold is decided per molecule. Mg⁺ (7.65 eV), Fe⁺ (7.90 eV) and Si⁺ (8.15 eV) transfer only to
  low-IP PAHs, which pahchem's per-molecule IPs handle directly. No rate has been measured for these three.
- **Below the threshold, no reaction is not assumed silently.** Two unmeasured alternatives are offered as
  labelled options:
  - [M–PAH]⁺ adducts at near-Langevin rates for low-IP metal ions (Tielens 2021, §8.7.3). The binding
    energies are Na⁺/K⁺ 1–1.2 eV on benzene and Fe⁺ ≈ 2.4–2.5 eV on benzene and coronene. This removes
    Fe⁺, Mg⁺ and Na⁺ from the gas even where charge transfer is closed. Fe–coronene⁻ is stable (EA ≈ 1 eV).
  - H abstraction, X⁺ + PAH → XH⁺ + PAH′, as Lepp & Dalgarno (1988) assume for Na⁺, K⁺, Ca⁺ and Al⁺.

**Smaller channels:**
- X⁺ + PAH^(Z≥1): zero below 10⁴ K (Coulomb barrier).
- He⁺ + PAH⁰ ≈ k_L, split between PAH⁺ + fragments and PAH²⁺ + e; the branching is unknown.
- Proton transfer from H₃⁺, HCO⁺ and H₃O⁺ (≈ k_L) gives HPAH⁺.
- PAH⁺ + PAH⁻ ≈ 3×10⁻⁸ (300/T)^½. It is second order in x_PAH.

**In the chain:**
- Ions only add up-transitions: k_up(Z) gains Σ_X n_X k(X, Z). The nearest-neighbour steady state
  f(Z+1)/f(Z) = k_up(Z)/k_down(Z+1) keeps its form.
- Δz = +2 (He⁺) and proton transfer (ΔZ = +1, ΔN_H = +1) go into `BaselineModel.network` only.
- Mutual neutralisation competes with photodetachment already in the CNM: n(C⁺) k_MN ≈ 2×10⁻⁹ s⁻¹
  against 1.2×10⁻⁸ G₀ s⁻¹. In dark gas it dominates. So f(Z) depends on n_ion/n_e as well as on γ.

**No global Φ_PAH.** A single `phi_pah` scale on all collisional rates may exist for calibration studies
only, labelled ad hoc. The test is whether pahchem's rates, with its PAHdb population, give W08's
calibrated C⁺ neutralisation within ×2 untuned (V2–V4). If they don't, that is a finding about the size
distribution or the abundance, not a reason to tune.

### 2.4 Outputs for a gas network

From `PAHNetwork.fluxes`, extended with named per-ion processes. All are per H nucleus, for given inputs:
1. **α_PAH(X):** the neutralisation coefficient per ion, split into its mutual-neutralisation and
   charge-transfer parts, with product channels. dn(X⁺)/dt = −α_PAH n(X⁺) n_H. It replaces WD01b's α_g.
2. **The net electron source:** photoionisation and detachment minus capture and attachment.
3. **Γ_PE and Λ_rec.**
4. **R_f,PAH = Σ x_m h2_per_pah_per_s / n(H),** with n(H) an input. `export_pah_tables` currently sets
   n(H) = n_H, which holds only in atomic gas.
5. **f(Z), ⟨Z⟩ and f(N_H) per molecule and per population.** Table axes: ψ = G₀ √T/n_e, T, and
   n_ion/n_e (or explicit n(C⁺) and n(H⁺)).
6. **Diagnostics of who carries the negative charge:**
   - x(PAH⁻)/x_e: PAH⁻ out-neutralise the electrons only when this exceeds ≈ 43 (Flower & Pineau des Forêts
     2026, dark clouds, which needs x_PAH ≳ 10⁻⁶);
   - Tielens (2021, §10.5.3): PAH⁻ dominate the negative charge when x_e < 2×10⁻⁷ (x_PAH/10⁻⁷).
   - Lepp & Dalgarno (1988b): in dense clouds PAH⁻ carry most of the negative charge for x_PAH > 10⁻⁸, which
     raises C and C₃H₂ by two orders of magnitude at x_PAH = 10⁻⁷; DCO⁺/HCO⁺ limits x_PAH to ≲ 6×10⁻⁶.
   - In diffuse gas, H⁺ + PAH⁻ can remove most H⁺ (Tielens 2021 §9.5.3). A ζ derived from OH⁺, H₂O⁺ or HD
     depends on this rate.

### 2.5 H₂ formation on PAHs

These follow from the H₂ section of the literature page and the primary papers (Andrews+16, Mennella+12,
Thrower+12). They are **opt-in and labelled until derived**:
- **Eley–Rideal per aliphatic C–H bond.** Mennella+12 measured 0.06 ± 0.02 Å² per aliphatic C–H bond, on
  coronene films at 300 K. An extra H on an edge carbon makes a CH₂ with two abstractable H, so the
  per-molecule rate lies between 1× and 2× (N_H − N_H0) σ(Z) v_H n(H). Cations take ≥ 0.45 Å² (Foley+18).
  - pahchem now applies 8.7×10⁻¹³ (T/100)^½ n(H) once per molecule, as Andrews+16 (eqs. 9–10) do.
  - Tielens (2021, §11.6.3) calls the ER route unimportant.
- **Photolytic H₂ loss from superhydrogenated states: the barrier is disputed.**
  - The proposed RRKM uses E₀ = 1.6 eV (neutral) and 3.2 eV (cation), after Boschman+15 and Le Page.
  - Andrews+16 (p. 7) reject the channel, with DFT barriers of 3.89 eV (duo) and 2.2 eV (solo),
    ΔS = −53.1 J K⁻¹ mol⁻¹.
  - Thrower+12's DFT gives H₂ release at nearly no cost for 19, 20, 23 and 24 extra H.
  - So any super-H H₂ channel is run with both brackets and reported, never with one chosen value.
- **H addition.** The measured rates are below the Andrews+16 prescription, geometric × exp(−0.06 eV/kT):
  - ≈ 3× lower at 300 K (Mennella+12: 1.1 Å²);
  - ≈ 40× lower at 2000 K (Thrower+12: 0.7 ± 0.4 Å² per molecule);
  - with no Arrhenius rise between the two (surface-bound molecules; flux uncertain by 2×).
  - The lab reaches +24 H (peak near +20), against the N_H0 + 4 cap used by Andrews+16.
  - Alternating cation addition barriers follow Boschman+15. The first extra-H barriers are 10–30 meV
    (cation) and 60 meV (neutral) (Tielens 2021).
- **The pyrene⁺ H₂-loss barrier differs between sources.** Tielens (2021, Table 8.1) gives E₀ = 3.5 eV,
  ΔS = −12.7 J K⁻¹ mol⁻¹; pahchem uses West+18's 4.58 eV and 40. To be reconciled, not tuned.
- **`h2_formation_rate` made net:** today H₂-consuming channels are not subtracted (e.g. `lepage01.py`).
- Any H₂ produced through dissociative recombination stays labelled: the H against H₂ branching is not
  measured (Hamberg+11).

Targets:
- Castellanos+18 efficiency of 1–2 %;
- Andrews+16:
  - Fig. 10: ξ_ER < 1 %, and ξ_photo < 10⁻⁴ for normally hydrogenated C₅₄;
  - eq. 13: R_PAH ≈ 10⁻¹⁹–10⁻¹⁸ cm³ s⁻¹;
  - eq. 12: the grain rate, 6×10⁻¹⁸;
- Mennella+12's upper limit, 1.2×10⁻¹⁶ cm³ s⁻¹ at 330 K;
- Habart+04 PDR rates of 3×10⁻¹⁷ to 1.5×10⁻¹⁶ cm³ s⁻¹;
- the upper bound R_PAH ≲ ½ κ_add x_PAH ≈ 3.5×10⁻¹⁷ (f_C/0.1)(50/N_C) for cations.

## 3. Double counting: one owner per term

When pahchem supplies these terms, each has one owner. Drop the overlapping fit, or restrict it to grains
larger than the PAHs:
- **Ion neutralisation.** WD01b α_g includes PAHs down to 3.5 Å. This matters for Gong+17, TIGRESS-NCR,
  UCLCHEM 4.0 and HYACINTH, and for any network that uses α_g.
  - pyCALIMA has no grain-assisted recombination of its own: `dust_ion_recombination.py` was removed on
    2026-10-04 (6a56802).
- **Photoelectric heating and recombination cooling.** The BT94, WD01 eq. 44/45 and W22 fits are
  PAH-dominated: half the heating comes from grains < 15 Å.
  - RAMSES `pah_pe_heating` (Berné+22) and pahchem's Γ_PE on the same bins must not both run.
- **H₂ formation.** The standard R = 3×10⁻¹⁷ is an empirical total; 55–81 % of a WD01 grain area is PAH
  surface. Use the grain bins without the PAHs.
- **Electrons.** Remove the network's own grain-electron terms when pahchem's net electron source is used.
- **Φ inside γ.** W03, W08 and W22 heating use γ′ = γ/Φ. Never apply Φ on top of pahchem's f(Z).
- **Φ and ζ are calibrated together** (Liszt 2011: ζ_H = 10⁻¹⁶ with PAHs, 10⁻¹⁷ without). Comparisons with
  W08 or Liszt use their ζ.

## 4. Validation

The exact numbers are in the literature page, section 8.

**Rerun on 2026-11-02 with the current defaults** (`results/pahchem/validation_2026-11-02/`; the table below
keeps each row's earlier text, the current verdicts are in `briefing_benchmarks.csv`). The rows were last measured
on 2026-10-25/26. Since then the UV cross section changed twice (Berné+22's Malloci average on 10-26, our Octopus
library on 11-01), and on 11-02 the sticking (§2.2b), the ladder (§2.2c) and the anion's valence step.
`check_default_changes.py` adds them one at a time from the 2026-10-25 setup, which reproduces the documented
numbers:

| quantity | 10-25 | + Malloci UV | + Octopus UV | + sticking | + ladder (now) |
|---|---|---|---|---|---|
| V2 α(C⁺)/W08, N_C 32–40 | 0.84 | 0.92 | 0.79 | 0.23 | 0.23 |
| V14 diffuse f(−1), C54 / C96 (Tielens 0.13 / 0.13) | 0.093 / 0.155 | 0.112 / 0.192 | 0.110 / 0.206 | 0.080 / 0.206 | the same |
| V14 PDR ⟨Z⟩, C54 / C96 (0.55 / 0.78) | 0.37 / 0.52 | 0.26 / 0.36 | 0.31 / 0.34 | 0.33 / 0.34 | the same |
| V14 cavity ⟨Z⟩, C96 (2.57) | 2.46 | 2.31 | 2.36 | 2.36 | 2.40 |
| V9 ε, Tielens' diffuse example, C54 / C96 (~0.05) | 0.045 / 0.052 | 0.042 / 0.049 | 0.036 / 0.032 | 0.033 / 0.032 | the same |
| V9 Γ / Berné+22, diffuse ISM | 1.86 | 1.53 | 1.72 | 1.54 | the same |

- **The size distribution now decides the diffuse-ISM rows.** V2, V3 and V13 take W08's single size, N_C 32–40,
  where the size logistic gives f = 0.014–0.10. There pahchem's C⁺ neutralisation is 0.23 of W08's, the effective Φ
  is 0.18–0.19 (W08's 0.2–0.4) and the observed C I/C II is 1.8× the model's (median), and Lepp+88's ion ratios
  need x_PAH 6–11× theirs. With W08's carbon (2.1×10⁻⁵ C per H) in N_C 50–100 instead (`check_diffuse_ionisation.py
  --nc-range 50 100`):
  - V2: 0.83–0.91 of W08 over n = 3–300 (§2.2b).
  - V3: Φ = 0.36 (valence; 0.33–0.47 over the detachments), observed/model C I/C II 0.87: passes.
  - V13: x_PAH 4.1–5.8×10⁻⁸ toward ζ Per, against 4.8×10⁻⁸ from Lepp+88's own model here and their quoted 1×10⁻⁷:
    passes. 1.9–2.8×10⁻⁷ toward ζ Oph, against their 6×10⁻⁷ (×2.4 low, χ² 5.9).
  - V4's x_e at n = 30: 2.6–3.0×10⁻⁴ (4.3–4.5×10⁻⁴ at N_C 32–40), against 3.12×10⁻⁴ from WD01b's published α_g.

  The new attachment holds these calibrations only if the diffuse ISM's PAH carbon is mostly in N_C ≳ 50.
- **V4 (α against WD01b, group tables):** pahchem/WD01(PAH) at ψ ≤ 10³ is 1.4–1.5 at 100 K and 0.9–1.15 at 1000 K (2.0
  and 1.2–1.5 before: fewer anions), and 7–11 at ψ ≥ 10⁵. The group tables stop at +2.
- **V6:** the script takes f = 1 and is unchanged. With the default f, C60 is 1.19× Viggiano+10 (1.39× at f = 1), and
  anthracene's collisional limit is 1.6×10⁻⁹, inside the lab's 0.84–2.6×10⁻⁹ (f's anchor).
- **V9 and V14** moved mostly with the UV library: C96's diffuse ε from 0.052 to 0.032 and f(−1) from 0.155 to 0.206.
  C54 also moves with f (0.68). The ladder adds +0.04 to C96's ⟨Z⟩ in the cavity, where 3 % now sit at +4.
  `check_tielens21_fz.py` summed ⟨Z⟩ over Z ≤ +3 only, and now takes every charge.
- **V10** is unchanged (the default's s at C54 is 0.9998). **V15** too: without associative detachment of N_H0 anions,
  PAH⁻ overtake electrons at n(H₂) = 1.4×10², and the attachment's median is 1.55×10⁻⁶ (1.65×10⁻⁶ at f = 1).
- **V11:** NGC 7023's R_PAH is 7.5×10⁻²⁰ (C54) and 4.9×10⁻²⁰ (C96), from 6.4 and 3.3×10⁻²⁰: the UV library; the
  sticking does not move them. Coronene's largest R_PAH on the grid is 1.4–1.5×10⁻¹⁷ in every configuration of the
  chain, including 2026-10-25's: the 2×10⁻¹⁸ quoted in the V11 row is not reproduced by these changes. An earlier
  change moved it; not traced.

| | target | status |
|---|---|---|
| V1 | Wolfire+08 Table 2 from DS87 (and Hollenbach+12 at N_C = 100) | passes (§A, ≤ 2.6 %) |
| V2 | W03/W08 three-state charge balance (N_C = 35; G₀ = 1.7, T = 100, n = 30): with W08's rates exactly, then with pahchem's, giving the effective Φ (expected 0.4–0.6) | **passes** (2026-10-22, `check_wolfire08_cplus.py`): untuned, pahchem's α(C⁺) for the 72 v4.00 molecules with N_C 32–40 at x_PAH = 6×10⁻⁷ is 0.99× W08's calibrated value (Φ = 0.4), 0.86× with statistical detachment; f(PAH⁻, PAH⁰, PAH⁺) = 0.14–0.17, 0.81–0.83, 0.03 against W08's 0.22, 0.75, 0.03. **Rerun 2026-10-24** on the 1 eV grid with the solar-circle old stars (§4d): 0.91× (Jochims) and 0.73× (statistical), f(PAH⁻) 0.15 and 0.11 ; **valence detachment (2026-10-25):** 0.72×, f(PAH⁻) 0.11; over n = 3–300 at W08's G₀, T and fractional abundances, 0.65–1.10× (Jochims 0.69–1.28×; figure `pahchem_v2_w08_cplus.png`) ; **without associative detachment of the parent anions (2026-10-26, below the table):** 0.84× (valence; f(PAH⁻) 0.137), 1.16× (Jochims; f(PAH⁻) 0.205 against W08's 0.217), 0.84–1.15× over n = 3–300 |
| V3 | the W08 C I/C II regimes (their eq. B15), on both C I data sets: W08's compilation and Burgh+10, which is ×4–21 lower on three of four common sight lines (Liszt 2011 Table 1) | **passes** (2026-10-25, `check_diffuse_ionisation.py`, N_C 32–40 at 6×10⁻⁷): pahchem's effective Φ is 0.37–0.39 (Jochims) and 0.32–0.35 (statistical, valence) over T = 50–300 K, inside W08's 0.2–0.4. Through eqs. B15 + B17 with their H₂-derived ⟨G₀⟩/n, the 28 W08 lines with A_V > 0.25 imply Φ = 0.26 (16–84 %: 0.09–0.54), and pahchem's own C I/C II gives obs/model 0.89–1.01 (0.42–3.4). On the four Burgh+10 lines Φ drops from 0.18–0.57 to 0–0.19 and obs/pahchem from 0.57–4.5 to 0.22–0.59: pahchem lies between the two C I data sets ; **2026-10-26** (no associative detachment of parent anions): Φ_eff 0.43–0.45 (Jochims), 0.35–0.38 (statistical, valence); obs/pahchem median 0.68 / 0.84 / 0.85 (16–84 %: 0.35–2.2 / 0.43–2.7 / 0.43–2.7) |
| V4 | α_PAH against WD01b for ψ = 10²–10⁶, T = 10–10³ K (C⁺, H⁺, S⁺, Mg⁺, Fe⁺); WD01b §7, x_e 8.0 → 4.5×10⁻⁴ | **partial** (2026-10-23, `check_wd01b_alpha.py`; T = 10² / 10³ / 6×10³ K, 7 ions; see below). **x_e (2026-10-25, `check_diffuse_ionisation.py`): partial.** At n = 30, T = 100 K, with ζ fixed by WD01b's grain-free 8.0×10⁻⁴, pahchem gives x_e = 2.9–3.1×10⁻⁴ (α(H⁺) 3.35–4.0×10⁻¹⁴ per H) for the three detachments, against 3.12×10⁻⁴ from WD01b's published α_g in the same set-up. Their printed 4.5×10⁻⁴ is recovered only with the arXiv v2 Table 2 H⁺ coefficients (3.436, 1.909×10⁻⁶, 1.496, …), whose α_g is 2.7× smaller than the published ones (12.25, 8.074×10⁻⁶, 1.378, …); whether the journal §7 kept 4.5×10⁻⁴ is not checked. **α rerun (2026-10-25):** the WD01 reference (`_alpha_full`, from `dust_ion_recombination.py`, removed on main in 6a56802) restored in the script; the Jochims numbers are unchanged, and the valence detachment moves every ratio by ≤ 7 % (C⁺ 1.86–1.96× at 100 K, 1.20–1.43× at 10³ K, 0.88–1.05× at 6000 K, ψ ≤ 10³; 6–9× at ψ ≥ 10⁵) ; x_e **2026-10-26** (no associative detachment of parent anions): 2.7–2.9×10⁻⁴. The ψ-sweep runs at n_H = 0 and does not move. **Cause of the α, Γ and Λ excess (2026-10-25, `results/pahchem/investigations_2026-10-25/charge_bias/`):** pahchem's electron capture is 5.5–6.8× WD01's (sticking 1 against WD01's 0.5(1 − e^(−a/10 Å)) ≈ 0.2, ×4.5–5.2; geometry ×1.1–1.3); with WD01's capture alone ⟨Z⟩, Γ and Λ come within ≈ 20 % at ψ ≤ 10⁴. IPs, yields and absorption per C (0.93–1.14×) do not explain it |
| V5 | FALP e + PAH⁺ (10 points, attributions and methods checked against Abouelaziz+93 and Biennier+06); the fits' extrapolations (§2.2) | §D lists them |
| V6 | attachment: anthracene 1.0×10⁻⁹ ± 30 % (300 K) and 0.84–2.6×10⁻⁹ (48–300 K); C₆₀ 1.2×10⁻⁶ (400 K); Tielens 2021's autodetachment, eqs. 8.101–8.103 | **partial** (2026-10-25, `check_attachment_v6.py`). C₆₀ (PAHdb UIDs 717/723, EA 2.68 eV): s_e = 1, k = 1.66×10⁻⁶ (400 K), 1.39× Viggiano+10: passes. Anthracene: 1.4×10⁻¹¹ (300 K) with the neutral's modes standing for the anion's, 5.5×10⁻¹¹ with the cation's (no anthracene anion in PAHdb), i.e. 0.005–0.18 of the lab and falling with T where the lab is flat. s_e ~ 10⁻⁴ because autodetachment (10⁴–10⁵ s⁻¹) beats IR emission (1–4 s⁻¹); the flow tubes' buffer gas stabilises collisionally (strong collisions: 5.2×10⁻⁷), so the lab lies between pahchem's zero-density limit and the collisional one, not a clean test of it. Tielens' eqs. 8.101–8.103 (from calima-pdr notes; T_i, T_f not defined there) give s_e ≪ 1 for anthracene and 1 for C₆₀ as well |
| V7 | mutual neutralisation: the Shuman, Hickman and Tielens laws; C₆₀⁺ + C₆₀⁻ (2.5×10⁻⁸ at 300 K, 1.35×10⁻⁷ at 10 K) | §B |
| V8 | ion–neutral: anthracene (Canosa+95 Table 1, with the C⁺ 60/40 products) | §C |
| V9 | PE heating: BT94 ε(γ); Berné+22 ε_PAH(γ) (N_C = 54) and its diffuse-ISM ε = 0.037, Γ = 2.7×10⁻²⁶ erg s⁻¹ H⁻¹; the W22 ratios of WD01 to BT94 | **partial** (2026-10-23, `check_pe_efficiency.py`; see below). Berné+22's own ε(γ) is matched by `check_berne22_pe_heating.py`; the W22 ratios compare WD01 with BT94 and do not involve pahchem ; **valence (2026-10-25):** ε = 0.045 (C₅₄H₁₈) and 0.052 (C₉₆H₂₄) at Tielens' diffuse example (his ~0.05), 0.10–0.12 at γ = 10², Berné+22's diffuse Γ 1.86×. The step-by-step comparison is drawn in `pahchem_berne22_steps.png`; its last step (valence) equals `Population.photoelectric_heating` to ≤ 4 % (the rest is the 1 eV grid and the old stars) |
| V10 | Thi+20 Table B.3 (C₅₄ recombination) | **partial** (2026-10-25, `check_thi20_recombination.py`). 1+: pahchem/Thi = 2.1 (10–300 K) to 2.3 (3×10³ K), 1.3–1.4 with `capture_model = "lab_trend"`: an effective φ of 0.83 (0.51) against their 0.4, ×1.41 from the radius (6.61 against 4.69 Å); within the ×2–2.5 of R1. 2+, 3+: pahchem/Thi = 10⁻³ (10 K) to 0.9 (10⁴ K). Thi's (1 + W/kT) multiplies an already Coulomb-focused rate, which gives 195× (100 K) to 2×10³× (10 K) the Spitzer limit at their own radius; pahchem's k(2+)/k(1+) = 1.71 and k(3+)/k(1+) = 2.37 are the DS87 limits. Attributed to Thi's multiply-charged scaling, as transcribed in the literature page (to be checked against the paper) |
| V11 | H₂: the targets of §2.5 (Andrews+16 Fig. 10 and eqs. 12–13, Mennella+12, Castellanos+18, Boschman+15, Habart+04, Thi+20 R_max) | **partial** (2026-10-25, `check_pah_h2_formation.py`; C₂₄, C₅₄, C₉₆ baseline). Passes: the Eley–Rideal constant (0.9995×), 2·h₂ ≤ the H-addition flux (0.97), Andrews Fig. 10 (ξ_ER plateaus 0.08–0.5 %, anions highest and size-independent, as `Andrews16Model`), ξ_photo < 10⁻⁴ for normal C₅₄; NGC 7023 R_PAH = 3–6×10⁻²⁰ (Andrews: ≲ a few 10⁻¹⁹). Fails: Castellanos' ξ = 1–2 % (coronene⁺ 30 %, circumcoronene⁺ 0.09–3 % rising with G₀). Partial: Boschman Table A.2 within 0.7–3.6×, no H₂ from odd or super-H states. R_PAH ≤ 2×10⁻¹⁸, ≤ 0.7–2.4 % of Habart+04's R_f and 10⁻⁴–2×10⁻² of Le Bourlot+12: pahchem's PAHs cannot supply the PDR excess. Thi's printed R_max is ¼ of its own σ v x_PAH x̄ |
| V12 | energy closure per process: absorbed = IR + photoelectron + ionisation work, returned on recombination | **passes** per state to < 10⁻⁴ on the 5–500 eV grid (`test_euv_grid_leaves_lyman_cut_fields_unchanged_and_closes_the_energy`) |
| V13 | Lepp, Dalgarno, van Dishoeck & Black 1988 (ApJ 329, 418) Table 2: neutral-to-ion ratios of C, Mg, S, Ca and Fe toward ζ Per and ζ Oph against x_PAH (they infer 1×10⁻⁷ and 6×10⁻⁷, ×2) | **passes** (2026-10-25, `check_diffuse_ionisation.py`). In a uniform slab at their models' central conditions, the observed ratios give x_PAH = 1.2–1.4×10⁻⁷ (ζ Per) and 5.6–7.0×10⁻⁷ (ζ Oph) for the three detachments; the same fit of their own models gives 4.8×10⁻⁸ and 6.7×10⁻⁷. The fall of each ratio with x_PAH follows theirs within ≈ 0.1 at 10⁻⁷. pahchem has no X⁺ + PAH → XH⁺ channel, which they assume for Ca⁺ ; **2026-10-26** (no associative detachment of parent anions): x_PAH = 0.7–1.1×10⁻⁷ (ζ Per) and 3.5–4.8×10⁻⁷ (ζ Oph). The element-to-element misses are already there without PAHs (gas-phase rates and the 1988 columns; Welty+03 revises Fe I ×8 lower): `results/pahchem/investigations_2026-10-25/observations/` |
| V14 | Tielens 2021 Fig. 12.34: f(Z) of C₅₄H₁₈ and C₉₆H₂₄ for a diffuse cloud (G₀ 2, n 50, T 80), the NGC 7023 PDR (2600, 2×10⁴, 500) and its cavity (2×10⁴, 5×10³, 500), x_e = 1.6×10⁻⁴; and its ε examples (0.05 at γ ≈ 10³, 0.008 at 10⁵) | **diffuse and cavity pass, PDR partial** (2026-10-23, `check_tielens21_fz.py`, values read from the figure ±0.01; ε examples in V9; see below) ; **by detachment (2026-10-25):** C₉₆H₂₄ f(−1) diffuse / PDR = 0.11 / 0.025 (Jochims), 0.75 / 0.53 (statistical: fails), 0.155 / 0.034 (valence), against 0.13 / 0.015. With valence (the default), the diffuse cloud is within 0.07 in every f(Z) (⟨Z⟩ −0.04 / −0.07 against −0.04 / 0.00) and the PDR ⟨Z⟩ is +0.37 / +0.53 against +0.55 / +0.78 (figure `pahchem_v14_tielens21_fz.png`) ; **2026-10-26** (Wenzel+20 on the cation only): C₉₆H₂₄ cavity ⟨Z⟩ +2.46 against +2.57, PDR +0.52; n_H = n and n_H = 0 now agree. The PDR gap is the electron capture: Tielens' recombination is 0.46× pahchem's, and alone it moves ⟨Z⟩ to +0.65 / +0.89 |
| V15 | dense clouds: Tielens 2021 eq. 10.77, x_e ≃ 2×10⁻⁷ (ζ/3×10⁻¹⁷)^½ (10⁴/n)^½, and Fig. 10.22 (PAH⁻ take over at n ≈ 3×10³); Flower & Pineau des Forêts 2026 (n_H = 10⁴, T = 10 K) | **partial** (2026-10-25, `check_dense_cloud_charge.py`; 349 v4.00 molecules N_C 50–100, 10 K, one molecular ion plus Mg). G₀ = 0 runs; f is the same for the three detachments to 2×10⁻¹⁶; attachment 1.65×10⁻⁶ against FPdF's 10⁻⁶. At x_PAH = 10⁻⁷ PAH⁻ overtake the electrons at n(H₂) = 8.3×10³ (Fig. 10.22: ~3×10³), or 1.3×10²–10³ without associative detachment by H, and lower x_e 14–19×. Eq. 10.77 fails: 0.03–0.24× it at 10⁴, slope −0.7 to −1.2 (x_e → ζ/(k_att x_PAH n)). FPdF's x(PAH⁻)/x_e ≥ 43 needs x_PAH ~ 10⁻⁶ without associative detachment; with it (n(H) ≈ ζ/R_gr), f(PAH⁻) ≤ 0.15. `chemistry.IONS` has no molecular ions; the script adds HCO⁺ ; **2026-10-26** (default: no associative detachment of these parents): the crossover falls to n(H₂) = 1.3×10² (x_PAH = 10⁻⁷), FPdF's ratio 66 at 10⁻⁶. It rests on the attachment, the full classical capture here (s_e = 1), whose capture-to-anion share is not measured above N_C = 14 |
| — | pahchem's existing gates and the PDRs4All fits unchanged with ions and the opt-in deposit off | **passes** (2026-10-23): `tests/test_pahchem_*.py` pass, every xfail a marked one; the Orion Bar shape fit recomputed from scratch equals the 2026-10-22 cache (`pahchem_spectral_shape_4154c0c24afa.pkl`) to ≤ 5×10⁻⁸ in every number |

**Two fixes (2026-10-26).**
- **Associative detachment** (PAH⁻ + H → PAH(0, N_H+1) + e, 7.8×10⁻¹⁰): Demarais+12 measured it on dehydrogenated
  (deprotonated) anions only. pahchem had applied it to every anion; it now applies to N_H < N_H0 only
  (`Params.associative_detachment = "dehydrogenated"`), so the population's charge chain and the group tables, which
  hold N_H0, have none; `"all"` restores the old behaviour (as Andrews+16, whose reproduction keeps it). This moves V2,
  V3, V4 (x_e), V13 and V15 where n(H) > 0 (rows above).
- **Wenzel+20 yield:** measured on cation → dication; `population._yield` had applied it from the cation up, so to the
  dication too (`baseline` already used it for z = 1 only). The dication now takes Jochims'. This moves the cavity
  (V14) and the high-γ phases.
- The PDRs4All result cache did not key on the physics defaults (`RESULT_VERSION` 8 now), so the Orion Bar gate had
  returned the Jochims-era fit since the valence switch of 2026-10-25.

**V4, V9 and V14 (2026-10-23).** Nothing tuned; v4.00 molecules with the calibrated IP shifts.
- **V4, α_PAH against WD01b.**
  - **Set-up:** the v4.00 molecules with charges −1..+2 (N_C 22–268) in four N_C bins (`group_tables`, Mathis
    background), each carrying the WD01 graphite grains of the same carbon count (a = 3.5–8.3 Å). These are
    compared with pyCALIMA's WD01-physics charging integrated over the same grains, over all sizes, and with
    the WD01b fit.
  - **PAH share of WD01b:** PAH sizes carry 0.8–0.93 of WD01's α at ψ ≤ 10⁴. At ψ = 10⁶ they carry 0.3–0.7
    for H⁺, He⁺, C⁺ and S⁺, and all of it for Mg⁺, Si⁺ and Fe⁺.
  - **pahchem against WD01 on the same carriers (ψ ≤ 10³):** 2.0× at T = 100 K, 1.2–1.5× at 10³ K and
    0.9–1.1× at 6000 K.
  - **At ψ ≥ 10⁵:** 6–9× (T ≤ 10³ K). The excess is in f(Z), not in the collision rates. For N_C 40–70 at
    100 K, pahchem has f(−) = 0.60 against WD01's 0.18 at ψ = 10², and f(0) = 0.155 against 0.016 at 10⁵.
    pahchem's IPs are higher (C₅₄H₁₈: 6.5 against WD01's 6.24 eV, eq. 2 at a = 4.87 Å; 5.7 eV, quoted here
    before 2026-10-26, is Berné+22's) and its yields are Jochims'.
  - **Truncation:** above ψ ≈ 10⁵ the chain's +2 cap makes the ratio partly an upper bound.
  - **pyCALIMA's WD01 physics against the WD01b fit:** 0.55–2.5×.
- **V9, photoelectric efficiency** (C₅₄H₁₈ and C₉₆H₂₄, Draine field, no ions). The numbers below are from the
  first run on the old 5–13.6 eV grid and normalisation.
  - **γ ≥ 10⁵:** pahchem is Tielens eq. 12.26 with pahchem's own capture (3.2×10⁻⁵ γ), to 3 % at 10⁵ and
    20 % at 10⁶. It is 3–5× BT94, the direction Berné+22 found against WD01.
  - **γ ≤ 10³, Jochims detachment (the default until 2026-10-25):** ε = 0.19–0.59, 4–12× BT94 and eq. 12.26, and Berné+22's
    diffuse-ISM Γ is exceeded 6×.
    - C₅₄H₁₈ is 81–94 % anionic at γ = 10², and every detached electron takes E − EA.
  - **With `detachment = "statistical"`:** ε = 0.024–0.044 for C₅₄H₁₈, and Γ = 0.81× Berné+22's
    diffuse-ISM value.
    - C₉₆H₂₄ (detachment threshold 2.14 eV) is then 93 % anionic in the diffuse cloud (23 % with Jochims),
      with ε = 0.005–0.009.
  - **Rerun 2026-10-24** (1 eV grid, old stars, ε over all absorption). The old stars detach anions, and ε's
    denominator gains the optical absorption.
    - Jochims: ε = 0.15–0.40 at γ ≤ 10³; the Berné+22 diffuse Γ is exceeded 5.8×.
    - Statistical: ε = 0.028–0.034. At the Berné+22 diffuse point, Γ = 0.90× theirs and ε = 0.032 against
      their 0.037, now on the same normalisation. C₉₆H₂₄ is 81 % anionic.
    - γ = 10⁵: ε = 0.0100, against 0.0143 for eq. 12.26 with pahchem's capture (its normalisation is not
      all-absorption).
  - **So Jochims detachment is not usable for gas heating below γ ≈ 10⁴.** Which default to use is a
    decision, not a fit; it is `valence` since 2026-10-25 (below).
  - **Neither alternative passed everything either (2026-10-25).** Statistical detachment fixes C₅₄H₁₈'s heating
    but leaves a large anion holding its electron below ~10 eV, where thermionic emission loses to IR emission:
    C₉₆H₂₄ is 0.75 anionic in Tielens' diffuse cloud (his 0.13) and 0.53 in the NGC 7023 PDR (his 0.015).
  - **The valence channel (`detachment = "valence"`, the default since 2026-10-25).** Statistical plus the prompt removal of a valence
    electron, which leaves the neutral excited: WD01's valence-band photoemission of a negative grain, with
    pahchem's yield above `population.anion_valence_threshold` (the molecule's own ladder one Coulomb step down,
    2 IP(0) − IP(+1): 3.55, 3.75, 3.88 eV for C₂₄H₁₂, C₅₄H₁₈, C₉₆H₂₄) and the photoelectron share of the excess.
    A molecule with no computed IP(+1) takes the fitted DFT step of §2.2c since 2026-11-02, WD01's e²/a(1 + 0.3 Å/a)
    before, which is 0.78× the DFT steps. That is 5 molecules of v4.00 (all 38 anions of v3.20): their valence
    threshold drops ~0.5 eV, f(−1) by 2–4 % and ε rises 6–11 % at γ = 300–1000 (C73–C92); the C34s barely move.
    - It is the only option that passes the charge tests for both molecules: V14 f(−1) 0.155 / 0.034 against
      0.13 / 0.015 for C₉₆H₂₄, and V2 α(C⁺) 0.72× W08.
    - Its ε matches Tielens' diffuse example (0.045 and 0.052 against ~0.05), and is 0.10–0.12 at γ = 10², 2–3× BT94.
      Berné+22's diffuse Γ is exceeded 1.86×, for the reason found before: their coronene attachment law leaves out
      the anions (f(−1) = 0.32 here), and an anion's valence electron leaves from below the neutral IP, so each
      detachment heats more, as WD01's negative grains do (`check_berne22_pe_heating.py` steps 6–7).
    - The unmeasured input is the anion's UV photodetachment yield and electron energy (the neutral's yield shape is
      taken). At the PDRs4All conditions (γ ≈ 10⁴) anions carry ~1 % of the power, so the choice does not move
      those fits.
  - **γ = 10⁵:** C₅₄H₁₈ is less ionised than in Berné+22: {f⁺⁺, f⁺, f⁰} = {0.31, 0.51, 0.17} against
    {0.5, 0.4, 0.1}. Their values are for T_eff = 3×10⁴ K (their Fig. 6), not the Draine field used here, so
    this is not yet a like-for-like comparison. In the Draine field their code gives {0.36, 0.53, 0.11} at 100 K.
  - **Berné+22 recovered** (`check_berne22_pe_heating.py`, run against their code on 2026-10-23).
    - Their code gives their Table 4 point, ε 0.0364 and Γ 2.71×10⁻²⁶ (published 0.037 and 2.7×10⁻²⁶).
      For Habing it gives 0.0574 and for Mathis 0.0391, as their Table 4; the text of their §5.2.1 swaps
      these two. The published A&A version (667, A159) has the same content as arXiv v2.
    - Rebuilt from their inputs, the chain equals their code to ≤ 0.3 % for γ = 0.6 to 6×10⁴.
    - Of the ingredients swapped one at a time, two move the result:
      - **Attachment on the neutral:** pahchem's, 1.5×10⁻⁶ for C₅₄H₁₈ (EA 1.52 eV), is ~600× their
        coronene law (EA 0.47 eV). Alone, it takes f(−) from 0.001 to 0.44 and Γ to 6.9× at γ = 513.
      - **Cation recombination:** pahchem's DS87 rate is 2.0× their Spitzer form, which makes the PAHs more
        neutral at γ ≳ 10⁴ (Γ 1.5× at 6×10⁴).
    - pahchem's attachment is in line with Tielens' s = 1 for N_C > 30 and with C₆₀'s 1.2×10⁻⁶. Applying
      coronene's law to C₅₄ is Berné+22's assumption.
    - So the gap is the attachment, combined with each detached electron taking E − EA.
- **V14, Tielens Fig. 12.34** (n_H = 0, as his balance).
  - **Diffuse cloud:** every f(Z) within 0.04 on the old grid, within 0.05 with the old stars (2026-10-24; C₅₄H₁₈
    f(−1) 0.11 against 0.13). The PDR and cavity cells do not change at their G₀.
  - **C₉₆H₂₄ cavity:** ⟨Z⟩ +2.62 against +2.57.
  - **NGC 7023 PDR:** pahchem is less ionised, ⟨Z⟩ +0.36 against +0.55 (C₅₄H₁₈) and +0.54 against +0.78
    (C₉₆H₂₄). This is the same direction as V4 and V9 at high γ.
  - **The figure itself:** its C₅₄H₁₈ cavity panel repeats the C₉₆H₂₄ PDR panel bar for bar. Its caption
    swaps the PDR and cavity columns; the panel labels and the γ order are taken.

**Closed 2026-10-23: W08 against WD01b.** At CNM conditions, W08 (Φ = 0.4, x_PAH = 6×10⁻⁷) is 1.61× WD01b
for both C⁺ and H⁺ (`check_pah_ion_rates.py` §E).
- WD01b's H⁺/C⁺ is 3.46, the DS87 mass scaling √12.
- The ×6 for H⁺ quoted before came from a transcribed H⁺ coefficient (C2 = 1.578 for 1.378).
- Codes print the C⁺ row of WD01b differently (DESPOTIC's C4 is 10× pyCALIMA's, giving 5× lower α). The
  paper's Table 2 is the reference for V4.

**A grain-side bug, fixed 2026-10-21.** `models/dust_chemistry/dust_h2_formation.py` computed
0.5 n_H (2kT/m)^½ n_d 2πa². Cazaux & Spaans 2004 (eqs. 1, 2 and 4) and Tielens 2021 (eq. 9.9: R = ½ S η n_d σ_d
n(H) v̄_H, with the mean speed v̄_H = 1.5×10⁴ T^½) give a rate 1.77× lower. It now uses πa² and the mean speed
(`mean_h_speed`) in the rate and in the H flux; `tests/test_models_physics.py` reproduces their eq. 4. Nothing in
pyCALIMA consumes the rate, so no pyCALIMA benchmark moves.
- **RAMSES:** the same fix is e7b5165a in part 02 of the PR stack (#16) and all later parts, but not in `staging`
  or `calima_tva`. The recommendation (land #16, re-check results made without it) is in
  `RAMSES_dev/OPEN_ISSUES.md`. pyCALIMA's Fortran mirror `solvers/ramses_source/dust_surface_chemistry.f90`
  already has the fix (πa² and the mean speed √(8kT/πm_H), lines 133–142).

**The Andrews+16 reproduction's neutral sticking, fixed 2026-10-21.** `pahchem/andrews16.py` used sticking 1 for
every neutral. Andrews+16 (App. A.3) take an EA-dependent sticking from Allamandola+89 (≈ 10⁻⁵ at EA = 0.47 eV,
unity for EA ≳ 1 eV), N_H-dependent through eq. 1, but give no formula. `Andrews16Model` now uses pahchem's
statistical sticking (`attachment.Attachment`, as `BaselineModel` does) per (N_H, EA(N_H)), with the neutral's
radiative P(E) at G₀ = 1; `sticking_neutral=1.0` restores the old behaviour.
- Coronene (EA 0.47 eV): 4.5×10⁻³ at 100 K and 1.0×10⁻³ at 500 K, so its attachment falls 200–1000×. That is
  still above Allamandola's ≈ 10⁻⁵, and the difference is pahchem's physics against a formula not given; it is
  recorded, not tuned.
- C₅₄ (EA 1.44 eV): 1, so the NGC 7023 benchmark (`check_andrews16_ngc7023.py`) is identical.
- C₉₆ (EA 0.56 eV at N_H0): ≈ 1, falling to 0.56 at 500 K for two extra H (EA 0.36 eV).
- Still departing from the paper: C₉₆ Z = −2 is missing (unpopulated in their Fig. 8); Li & Draine cross sections
  replace Malloci's; the × N_C/24 scaling of the geometric addition cross section is pahchem's own.

## 4b. Implementation status (2026-10-22)

| step | what | where | status |
|---|---|---|---|
| 1 | photoelectric heating per state, from the charge balance's rates | `photo.photoelectron_power`, `Population.photoelectric_heating` | done; reproduces Berné+22's code to ≤ 0.3 % with their inputs (`check_berne22_pe_heating.py`; rerun 2026-10-23 after fixing its yield grid for the 5–500 eV `E_PH`) |
| 1b | anion photodetachment: prompt (WD01 σ_pdt) plus delayed (detailed-balance autodetachment, thermionic electrons); with `"valence"` also the prompt removal of a valence electron above `anion_valence_threshold` | `Params.detachment = "statistical"` or `"valence"`, `population.statistical_detachment`, `Attachment.delayed_detachment` | done, opt-in; C₅₄H₁₈ ε 0.03 (statistical) or 0.045–0.12 (valence) at γ ≤ 10³, instead of 0.15–0.47; only valence passes V14 for C₉₆H₂₄ |
| 1c | the EUV: 5–500 eV grid (1–500 eV since 2026-10-24, §4d), Y = 1 above 20 eV, measured photoelectron share to 21.9 eV, constant ion internal energy above, deposits into the cascades | `population.E_PH`, `photo.photoelectron_energy`, `population.deposit_spectrum`, `Environment.field_e_max_ev` | done; Lyman-cut fields unchanged (≤ 3×10⁻⁷); top-state ionisations reported by `chain_overflow` |
| 2 | recombination cooling, kT Λ̃/J̃ per capture | `Population.recombination_cooling` | done |
| 3 | ion–PAH collisions: mutual neutralisation, ion–neutral reactions, optional adducts | `chemistry.mutual_neutralisation`, `chemistry.ion_neutral_rate`, `population.ion_up_rate`, `Environment.ions` | done |
| 4 | outputs for a gas network: α_PAH(X), carbon accretion, net electron source, Γ_PE, Λ_rec, f(Z) | `Population.gas_rates` | done, except H₂ formation (needs the hydrogenation network) |
| 4c | the same outputs per PAH bin for RAMSES-RT photon groups (§4c) | `pahchem.group_tables` | Python side done; RAMSES side not started |

## 4c. Per-bin tables for RAMSES-RT photon groups (2026-10-22)

`pahchem.group_tables` follows the WDB06rt grain tables (`models/dust_charge/rt_group_charging.py`). It uses
the same 8 RTZ groups and the same family of blackbody in-group shapes (T_s = 3×10³–3×10⁶ K, chosen by the run's
group mean energies). It also carries the same UV background: the Mathis ISRF at G₀ = 1 Habing, 1–13.6 eV.

**A bin** is a set of v4.00 molecules with weights. `nc_range_weights` gives equal weights over an N_C range.
The CALIMA bin scheme is still open, so any other weights (e.g. a size distribution) work too.

**What is tabulated.** For the charges −1, 0, +1 and +2, averaged over the bin's molecules:
- the photon kernels ⟨σY⟩_g, ⟨σY E_k⟩_g and ⟨σ⟩_g, per group and per T_s;
- the same three for the background;
- in T: the electron capture coefficients and the energy they remove;
- mutual neutralisation of the anion by each RTZ ion;
- the ion–neutral rates;
- associative detachment (k_ad = 0 since 2026-10-26: the bins hold parent anions; `Params.associative_detachment`).

The physics is `Population`'s, through shared functions:
- `_yield` and `statistical_detachment`, now on any photon grid;
- `electron_energy`, `capture_coefficient` and `capture_energy`, factored out of `Population`.

**Per cell** (`PAHGroupTable.solve`, the algorithm RAMSES would run):
- one 4-state detailed-balance chain per bin, in closed form;
- outputs per PAH: f(Z), Γ_PE, Λ_rec, the net electron source, α_mn and α_ct per ion, and the carbon accretion;
- photoionisation out of +2 is reported as overflow, as `Population` does for its top states.

A cell with no RT photons uses the background kernels only. A 4-state chain is cheap enough that no separate
WDB06isrf-style table is needed.

**Against `Population`** (`diagnostics/pahchem/check_pah_group_tables.py`):
- **Setup:**
  - Kurucz 40 000 K field cut at 13.6 eV;
  - G₀ = 1–10⁴, T = 100 and 1000 K, γ = 10²–10⁵, n(C⁺) = n_e;
  - four test bins: N_C 22–40, 40–70, 70–120 and 120–270, with 82–291 molecules each.
- **In-group shape** (blackbody in place of the true spectrum): ≤ 0.7 % in f(Z) and ≤ 1.1 % in Γ_PE, Λ_rec and
  α(C⁺).
- **One chain per bin** (rates averaged over the molecules before the balance is solved): ≤ 4 % in f(Z), ≤ 1.5 %
  in Γ_PE, ≤ 2.5 % in Λ_rec, ≤ 2.9 % in α(C⁺).
- So even wide bins lose little in charge physics. The number of PAH bins has to be set by the size evolution
  (photodestruction, growth, coagulation), not by the charge.

**Photons below 5 eV.** The tables integrate from each group's lower edge, so the 1–5.6 eV group detaches anions.
Until 2026-10-24 the population's grid started at 5 eV, so the population missed this (it now starts at 1 eV, §4d).
In the Mathis ISRF, photons between the electron affinity and 5 eV raise the detachment rate of N_C 32–40 anions:
- by ≈ 35 % with the Jochims yields;
- by ≈ 90 % with the statistical detachment.

In the bin tables (T = 100 K, n(C⁺) = n_e, γ = 10²–10⁵), switching the 1–5.6 eV group on changes:
- **Mathis ISRF:**
  - α(C⁺): ×0.88–0.98 with the Jochims yields; ×0.78–0.97 with the statistical detachment for N_C < 120;
  - N_C 120–270, statistical: ×0.71 at γ = 10⁴, where f(−1) falls from 0.52 to 0.36, and ×0.39 at γ = 10⁵;
  - Γ_PE: up to +30 % with the statistical detachment.
- **Kurucz 40 000 K:** ≤ 8 %, except α(C⁺) ×0.75 for N_C 120–270 at γ = 10⁵.

This matters for the anions in cold neutral gas under a starlight-like field, and therefore for V2. V2 used the
Draine field, which then had no photons below 5 eV.

The tables' UV background is the whole Mathis field times `G0_bg`, optical included. In the population the old-star
light below 5 eV no longer scales with G₀ (§4d); the tables do not follow this yet.

**Open.**
- The RAMSES reader and per-cell solve (Fortran), replacing the Berné+22 γ-tables of `init_pah_peh_tables`.
- The bin scheme and its weights.
- ~~Whether the population's grid should start below 5 eV.~~ Done 2026-10-24 (§4d).
- The tables' background: scale only its far UV with `G0_bg` and keep the old stars fixed, as the population now does.
- Li & Draine anions use the neutral cross section (`photo.sigma_li_draine`), which is 7× below the ion one at
  2 eV. That sets the delayed detachment by optical photons.

## 4d. Photons below 5 eV in the population (2026-10-24)

The user's decisions: the grid starts at 1 eV; ε is normalised to all absorbed power; the optical ISRF is old-star
light that does not scale with G₀.

**The grid.** `population.E_PH` now runs:
- 1–5 eV in the baseline's 0.025 eV steps, with a node 10⁻⁶ eV below 5 eV;
- then the old 5–13.6 eV grid;
- then the EUV grid up to 500 eV.

Every 1–5 eV node is also a node of the cascades' deposit grid. Because the cache key hashes the grid, every
stored basis is rebuilt on first use.

**What reaches below 5 eV.** Only the anions' detachment, since EA ≈ 0.5–2.5 eV. The neutral and cation thresholds
lie above 5 eV. Every state absorbs more, the cations especially in the visible, so their cascades and IR emission
change.

**Reproducing the old results.** `Environment.field_e_min_ev = 5.0` gives the rates of the old grid. Two tests
check this:
- `test_photons_below_5ev_detach_anions_and_leave_the_higher_thresholds_alone`: rates equal to 10⁻⁶;
- `test_old_star_light_below_5ev_does_not_scale_with_g0`.

**The fields below 5 eV**, following the PDR codes (checked in their sources):
- **Cloudy:** `table Draine` stops at 5 eV (Draine & Bertoldi 1996, eq. 23), and `table ISM` (Black 1987) is the
  full-spectrum field.
- **Meudon PDR 7.1:** Draine's polynomial is extrapolated, and a separate stellar component is added that is not
  scaled by G₀ (three diluted blackbodies after Mathis+83, refitted to DIRBE). Meudon's default shape is Mathis.
- **3D-PDR:** no spectrum, only an FUV scale and parametric photorates.

The extrapolated Draine polynomial gives 10–170× fewer 1–5 eV photons than the old-star light. So in pahchem:
- **`fields.draine1978`** evaluates its fit from 5 eV only. Its `old_stars` argument adds, without scaling by G₀,
  `fields.old_stars_flux`: the published Mathis+83 blackbodies (7500, 4000 and 3000 K) below their 5.04 eV break.
- **`fields.mathis1983(fixed_optical=True)`** scales only its far UV. The default still scales the whole field, as
  the reproduced papers (Murga+20, Montillaud+13, Maragkoudakis+20, SHIVA) do.
- **In the population**, a Draine cell gets `Environment.optical_isrf` (default 1) × the solar-circle old stars.
  - The shape key carries `optical_isrf / G0`, so each G₀ is its own shape, with exact bases.
  - Between the nodes of the multiphoton k_H table, the old stars scale with the node's G₀. They add little energy
    per photon, so the H loss barely feels this.
- **Kurucz fields** keep the star's whole spectrum, now from 1 eV.

**ε** is now the heating over all the power absorbed (1–13.6 eV, or up to 500 eV with the EUV). That is Berné+22's
and WD01's definition, and what the observed I_gas / (I_PAH + I_gas) measures. BT94's 6–13.6 eV convention differs.

**Still Li & Draine below 5 eV, by choice.** Since 2026-10-26 the default UV cross section is Berné+22's Malloci
average at 5–30 eV (`photo.sigma_uv`), with Li & Draine kept below 5 eV. Malloci+07 absorbs 2.2–3.6× more at
1–5 eV (Mathis field, median of 27 molecules), so the sub-5 eV effects are lower bounds until our own Octopus
library replaces both. For neutrals, part of Malloci's excess is the Lorentzian wing of the
exponential time window their spectra carry, not absorption: about a quarter of coronene's Mathis 1–5 eV rate
({doc}`pahchem-uv-absorption` §2). The cations' visible bands are real.

**Rerun on the new grid** (2026-10-24): V2, V9 and V14 (Draine cells, which gain the old stars; see §4) and the
PDRs4All fits (below), which meant rebuilding the Kurucz 40 000 K bases of all 1424 v4.00 molecules. The Berné+22
comparison is still to rerun.

**PDRs4All rerun (2026-10-24).** The bases (4315 states) took 14 min with 8 workers. The comparison script is
`diagnostics/pahchem/check_sub5ev_pdrs4all.py`. The Orion Bar fits hardly feel the sub-5 eV light:
- 9 of 10 fits keep their best cell, and γ stays ≈ 10⁴.
- rms improves by ~1 %.
- Charge shares of the emitted power move by ≤ 0.006.
- 3.3/11.2 falls by 1.3–3.2 % and 6.2/11.2 by 1.3–2.5 %; 12.7/11.2 rises by 0.8–2.3 %. The emission is slightly
  cooler.

The reason is the hot star: little of its absorbed power lies below 5 eV. With Malloci-level cation absorption the
ratio changes might reach ~3–13 %. Cooler fields (the ISRF, B-star nebulae) are where the sub-5 eV light matters.
`population.BASIS_VERSION` is now 6, so the result caches of the fit scripts no longer return 5 eV results.

## 5. Items for the user to supply

Supplied and read on 2026-10-21. Their corrections are applied above:
- Tielens 2021, chapters 6–12;
- Bakes & Tielens 1998; Lepp & Dalgarno 1988; Flower & Pineau des Forêts 2026; Liszt 2011; Burgh+10;
  Hickman 1979;
- Bréchignac+14; Canosa+94/95; Moustefaoui+98; Abouelaziz+93; Biennier+06; Hamberg+11;
- Andrews+16; Mennella+12; Thrower+12.

**Tielens 2021: what the book confirms and corrects.**
- Eq. 12.26, ε = 0.06/(1 + 7×10⁻⁵ γ), is confirmed. Its 7×10⁻⁵ rests on eq. 8.106, which is 0.46 ×
  pahchem's capture, so pahchem's own estimate would give ≈ 3.2×10⁻⁵ γ.
- Eq. 8.106 is confirmed; its meaning is in §2.2.
- Eq. 8.107 (attachment) is printed with the wrong power of N_C: it should be N_C^½.
- The book has internal inconsistencies, so it is cited by equation number:
  - Y = 0.3 at 10 eV against its own Y = (hν − IP)/9 = 0.44;
  - f(0) = 0.85 against 0.92 from eq. 12.25;
  - a work function of 4.2 against 4.4 eV;
  - a coronene⁺ IP of 10.58 against 11.4 eV.
- Tielens says the photoelectron energy "has not been quantified" and that half the excess energy is
  assumed "rather arbitrarily". This supports the γ_e(E) option of §2.1.
- H₂ formation on PAHs: no rate coefficient is given anywhere in chapters 7, 9, 11 or 12. The
  "2×10⁻¹⁷ (T/100)^½" attributed to it second-hand is unverified.

**Still to supply** (section 13 of the literature page):
- Tielens 2005 §3.3, §6.3, §8.7.
- Hassouna+03 (conference proceedings) is not obtainable; Rebrion-Rowe+03 and both Lepp & Dalgarno 1988 papers
  (ApJ 324, 553; ApJ 335, 769) were supplied on 2026-10-22.
