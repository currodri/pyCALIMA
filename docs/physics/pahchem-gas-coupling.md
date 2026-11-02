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
  §4b steps 1 and 1c (the deposit is opt-in).
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
f_low 0.003), in the population (gas rates, group tables, the PDRs4All fits). `BaselineModel` keeps s = 1 and
`attachment_fraction`, as it always ignored `capture_model`. Before, s = 1 and f = 1. Both functions are
**extrapolations** where the PAHs that matter most are:
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
- **Not yet:** K-shell absorption above 291 eV with Auger dz = +2 (stage 3), and the validation (stage 2).
- **Effect, Draine field (T = 1000 K):**
  - ψ ≤ 10⁴: ⟨Z⟩ is unchanged to 0.01 for C24–C150.
  - ψ = 10⁵: C150 goes from +1.26 to +1.59 (18 % above +2).
  - ψ = 10⁶: C150 goes from +1.89 to +3.49 (91 % above +2), C96 from +2.60 to +2.68.
  - The total IR power is unchanged to 1 %.
- **Effect, Kurucz 40 kK field to 100 eV** (G0 10⁵, n_e 100, 10⁴ K): C54's charge peaks at +3/+4 with a tail to +9.
  Its fraction at +10 is 8×10⁻⁸ and its Coulomb-explosion rate 4×10⁻¹³ s⁻¹.

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

**Measured with the sticking defaults before 2026-11-02** (s = 1 on cations, f = 1 on neutrals). Since then the
defaults are `lab_cold` and the size logistic (§2.2b), and the rows have not been rerun. Expected to move most are
the rows on N_C 32–40 (V2, V3, V13: f = 0.014–0.10 there instead of 1; on the current code V2 is 0.78 of W08 with
f = 1 and 0.23 with the size logistic, §2.2b) and the WD01 per-size comparison at C24 (f = 0.0036, s = 0.63). C54 (V9,
V14, V15: f = 0.68, s = 1) moves less; C66 and C96 (f ≥ 0.94) hardly. They also predate the cold-limit ladder and
the converged multiphoton H loss (§2.2c), which change nothing below ψ ≈ 10⁴ and move the largest PAHs above it.

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
