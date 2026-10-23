# PAHchem: coupling the PAHs to the gas (specification, 2026-10-21)

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

## 1. What pahchem has, and what is missing

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
- **Ion–PAH collisions.** There are none (`murga_charge.py`: "no ion collisions").
- **Photoelectron kinetic energy.** pahchem treats every ionising photon as depositing no internal energy,
  which implicitly gives the electron all of E − IP. That contradicts the measured partition (§2.1).
- **Gas heating and cooling.** No gas photoelectric heating, no recombination cooling.
- **Rates for a gas network.** Per-ion neutralisation, the net electron source and H₂ formation per H are
  not exported.
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
- **e + PAH⁰:** keep the statistical sticking (the low-density, radiatively stabilised limit). Validate it on:
  - anthracene: 1.0×10⁻⁹ ± 30 % at 300 K (FALP, Canosa+94; no pressure dependence, so radiative), and
    0.84–2.6×10⁻⁹ over 48–300 K with no trend (CRESU, normalised to that value; Moustefaoui+98);
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
  | Shuman+14 (VENDAMS), 2.7×10⁻⁷ μ^−½ (T/300)^−0.9, measured on small polyatomic anions | 0.58 |
  | Hickman 1979 eq. 28, 5.35×10⁻⁷ μ^−½ (T/300)^−½ EA^−0.4 (EA 2.5–1.5 eV; fitted for EA 0.75–4 eV, ±30 %) | 0.80–0.98 |

  Lepp & Dalgarno (1988b), after Omont (1986), quote "of the order" 1.3×10⁻⁶ T^−½ scaling as m^−½ with no stated
  reference mass, so it cannot be placed in this table.

- Proposed default: P_MN = 0.58, the measured polyatomic law. The range 0.46–0.98 is its uncertainty. It
  is labelled an assumption, since it transfers a small-anion law to PAH anions.
  - Both lab laws are size-independent. Relative to DS87, P_MN therefore falls as N_C^−½. A constant P_MN
    quietly assumes DS87's N_C^½ scaling, and that is stated wherever it is used.
  - The T dependence follows DS87 (T^−½), as the C₆₀⁺ + C₆₀⁻ Landau–Zener calculation does (Gatchell+25).
    The Shuman law gives T^−0.9 over 300–550 K.
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
  - pyCALIMA's `dust_ion_recombination.py` already excludes the PAH bins, and nothing uses it.
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

| | target | status |
|---|---|---|
| V1 | Wolfire+08 Table 2 from DS87 (and Hollenbach+12 at N_C = 100) | passes (§A, ≤ 2.6 %) |
| V2 | W03/W08 three-state charge balance (N_C = 35; G₀ = 1.7, T = 100, n = 30): with W08's rates exactly, then with pahchem's, giving the effective Φ (expected 0.4–0.6) | **passes** (2026-10-22, `check_wolfire08_cplus.py`): untuned, pahchem's α(C⁺) for the 72 v4.00 molecules with N_C 32–40 at x_PAH = 6×10⁻⁷ is 0.99× W08's calibrated value (Φ = 0.4), 0.86× with statistical detachment; f(PAH⁻, PAH⁰, PAH⁺) = 0.14–0.17, 0.81–0.83, 0.03 against W08's 0.22, 0.75, 0.03 |
| V3 | the W08 C I/C II regimes (their eq. B15), on both C I data sets: W08's compilation and Burgh+10, which is ×4–21 lower on three of four common sight lines (Liszt 2011 Table 1). At fixed n/G₀, eq. B15 then implies a Φ lower by 2.1–4.6, unless n is pinned independently. Burgh+10 measure no C II. | to do |
| V4 | α_PAH against WD01b for ψ = 10²–10⁶, T = 10–10³ K (C⁺, H⁺, S⁺, Mg⁺, Fe⁺); WD01b §7, x_e 8.0 → 4.5×10⁻⁴ | to do |
| V5 | FALP e + PAH⁺ (10 points, attributions and methods checked against Abouelaziz+93 and Biennier+06); the fits' extrapolations (§2.2) | §D lists them |
| V6 | attachment: anthracene 1.0×10⁻⁹ ± 30 % (300 K) and 0.84–2.6×10⁻⁹ (48–300 K); C₆₀ 1.2×10⁻⁶ (400 K); Tielens 2021's autodetachment, eqs. 8.101–8.103 | to do |
| V7 | mutual neutralisation: the Shuman, Hickman and Tielens laws; C₆₀⁺ + C₆₀⁻ (2.5×10⁻⁸ at 300 K, 1.35×10⁻⁷ at 10 K) | §B |
| V8 | ion–neutral: anthracene (Canosa+95 Table 1, with the C⁺ 60/40 products) | §C |
| V9 | PE heating: BT94 ε(γ); Berné+22 ε_PAH(γ) (N_C = 54) and its diffuse-ISM ε = 0.037, Γ = 2.7×10⁻²⁶ erg s⁻¹ H⁻¹; the W22 ratios of WD01 to BT94 | to do |
| V10 | Thi+20 Table B.3 (C₅₄ recombination) | to do |
| V11 | H₂: the targets of §2.5 (Andrews+16 Fig. 10 and eqs. 12–13, Mennella+12, Castellanos+18, Boschman+15, Habart+04, Thi+20 R_max) | to do |
| V12 | energy closure per process: absorbed = IR + photoelectron + ionisation work, returned on recombination | **passes** per state to < 10⁻⁴ on the 5–500 eV grid (`test_euv_grid_leaves_lyman_cut_fields_unchanged_and_closes_the_energy`) |
| V13 | Lepp & Dalgarno 1988 Table 2: neutral-to-ion ratios of C, Mg, S, Ca and Fe toward ζ Per and ζ Oph against x_PAH (they infer x_PAH 1×10⁻⁷ and 6×10⁻⁷, ×2) | to do |
| V14 | Tielens 2021 Fig. 12.34: f(Z) of C₅₄H₁₈ and C₉₆H₂₄ for a diffuse cloud (G₀ 2, n 50, T 80), the NGC 7023 PDR (2600, 2×10⁴, 500) and its cavity (2×10⁴, 5×10³, 500), x_e = 1.6×10⁻⁴; and its ε examples (0.05 at γ ≈ 10³, 0.008 at 10⁵) | to do |
| V15 | dense clouds: Tielens 2021 eq. 10.77, x_e ≃ 2×10⁻⁷ (ζ/3×10⁻¹⁷)^½ (10⁴/n)^½, and Fig. 10.22 (PAH⁻ take over at n ≈ 3×10³); Flower & Pineau des Forêts 2026 (n_H = 10⁴, T = 10 K) | to do |
| — | pahchem's existing gates and the PDRs4All fits unchanged with ions and the opt-in deposit off | to do |

**Open item.** W08 (Φ = 0.4, x_PAH = 6×10⁻⁷) against WD01b at CNM conditions gives ×1.8 for C⁺ but ×6 for
H⁺, and the H⁺ factor is unexplained (WD01b's H⁺ ≈ its C⁺, where the DS87 mass scaling gives √12×). Check
it against WD01b Fig. 1 before V4.

**A grain-side bug, fixed 2026-10-21.** `models/dust_chemistry/dust_h2_formation.py` computed
0.5 n_H (2kT/m)^½ n_d 2πa². Cazaux & Spaans 2004 (eqs. 1, 2 and 4) and Tielens 2021 (eq. 9.9: R = ½ S η n_d σ_d
n(H) v̄_H, with the mean speed v̄_H = 1.5×10⁴ T^½) give a rate 1.77× lower. It now uses πa² and the mean speed
(`mean_h_speed`) in the rate and in the H flux; `tests/test_models_physics.py` reproduces their eq. 4. Nothing in
pyCALIMA consumes the rate, so no pyCALIMA benchmark moves.
- **RAMSES:** the same fix is e7b5165a in part 02 of the PR stack (#16) and all later parts, but not in `staging`
  or `calima_tva`. The recommendation (land #16, re-check results made without it, refresh pyCALIMA's Fortran
  mirror `solvers/ramses_source/dust_surface_chemistry.f90`) is in `RAMSES_dev/OPEN_ISSUES.md`.

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
| 1 | photoelectric heating per state, from the charge balance's rates | `photo.photoelectron_power`, `Population.photoelectric_heating` | done; reproduces Berné+22's code to ≤ 0.3 % with their inputs (`check_berne22_pe_heating.py`) |
| 1b | anion photodetachment: prompt (WD01 σ_pdt) plus delayed (detailed-balance autodetachment, thermionic electrons) | `Params.detachment = "statistical"`, `population.statistical_detachment`, `Attachment.delayed_detachment` | done, opt-in; C₅₄H₁₈ efficiency 0.02–0.04 instead of 0.3–0.6 |
| 1c | the EUV: 5–500 eV grid, Y = 1 above 20 eV, measured photoelectron share to 21.9 eV, constant ion internal energy above, deposits into the cascades | `population.E_PH`, `photo.photoelectron_energy`, `population.deposit_spectrum`, `Environment.field_e_max_ev` | done; Lyman-cut fields unchanged (≤ 3×10⁻⁷); top-state ionisations reported by `chain_overflow` |
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
- associative detachment.

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
The population's grid starts at 5 eV, so the population misses this. In the Mathis ISRF, photons between the
electron affinity and 5 eV raise the detachment rate of N_C 32–40 anions:
- by ≈ 35 % with the Jochims yields;
- by ≈ 90 % with the statistical detachment.

In the bin tables (T = 100 K, n(C⁺) = n_e, γ = 10²–10⁵), switching the 1–5.6 eV group on changes:
- **Mathis ISRF:**
  - α(C⁺): ×0.88–0.98 with the Jochims yields; ×0.78–0.97 with the statistical detachment for N_C < 120;
  - N_C 120–270, statistical: ×0.71 at γ = 10⁴, where f(−1) falls from 0.52 to 0.36, and ×0.39 at γ = 10⁵;
  - Γ_PE: up to +30 % with the statistical detachment.
- **Kurucz 40 000 K:** ≤ 8 %, except α(C⁺) ×0.75 for N_C 120–270 at γ = 10⁵.

This matters for the anions in cold neutral gas under a starlight-like field, and therefore for V2. V2 used the
Draine field, which has no photons below 5 eV.

**Open.**
- The RAMSES reader and per-cell solve (Fortran), replacing the Berné+22 γ-tables of `init_pah_peh_tables`.
- The bin scheme and its weights.
- Whether the population's grid should start below 5 eV. That would change its cascades too: the visible
  absorption of the cations.
- Li & Draine anions use the neutral cross section (`photo.sigma_li_draine`), which is 7× below the ion one at
  2 eV. That sets the delayed detachment by optical photons.

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
