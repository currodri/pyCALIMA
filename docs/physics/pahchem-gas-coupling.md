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

- **Mean photoelectron energy:** ⟨E_k⟩ = γ_e (E − IP_m(Z)) for Z ≥ 0, with γ_e = 0.46 ± 0.06. This was
  measured on coronene (Bréchignac+14, analysed by Berné+22); the classical value is 0.5 (V90, BT94).
  - The rest, (1 − γ_e)(E − IP), stays in the PAH as vibrational energy.
  - For photodetachment, E_k = E − EA.
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
    diagnostic). The ratio rises steeply with N_C and extrapolates to 1 near N_C ≈ 26. No measurement
    exists above N_C = 16, and none gives a T dependence.
  - Uncertainty: a factor of 2 between the DS87 image-potential form and the Spitzer form.
- **e + PAH⁰:** keep the statistical sticking. Validate it on anthracene (1–3×10⁻⁹, flat over 48–300 K) and
  C₆₀ (1.2×10⁻⁶ at 400 K).
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

- No PAH⁻ + X⁺ rate has been measured.
- The polyatomic law of Shuman+14 (VENDAMS), 2.7×10⁻⁷ μ^−½ (T/300)^−0.9, is **0.58 × DS87** in pahchem's
  geometry for C₅₀H₁₈⁻ at 300 K, for every ion (§B).
- Proposed default: P_MN = 0.58, the value implied by that law. It is labelled an assumption, since it
  transfers a small-anion law to PAH anions.
  - The T dependence follows DS87 (T^−½), as the C₆₀⁺ + C₆₀⁻ Landau–Zener calculation does (Gatchell+25).
    The lab law gives T^−0.9 over 300–550 K.
  - Molecular ions take 50 % dissociative products (Thi+20).
  - Uncertainty: ×2–3.

**Charge transfer, X⁺ + PAH⁰ → X + PAH⁺, only if IP(X) > IP_m(0):**

    k_CT = P_CT · 2π e (α_m/μ)^½

- α_m = 1.75 N_C Å³, and P_CT = 0.85 is measured on anthracene (Canosa+95: He⁺ 5.2×10⁻⁹, C⁺ 3×10⁻⁹).
- This is Langevin, not DS87: DS87's neutral sphere is Langevin with α = a³. Their ratio falls from 0.71
  (N_C = 24) to 0.48 (N_C = 100) (§C).
- The threshold is decided per molecule. Mg⁺ (7.65 eV), Fe⁺ (7.90 eV) and Si⁺ (8.15 eV) transfer only to
  low-IP PAHs, which pahchem's per-molecule IPs handle directly. No rate has been measured for these three.

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

### 2.5 H₂ formation on PAHs

These follow from the H₂ section of the literature page. They are **opt-in and labelled until derived**:
- Eley–Rideal per excess H, (N_H − N_H0) σ(Z) v_H n(H), with 0.06 Å² for Z ≤ 0 and ≥ 0.45 Å² for cations
  (Foley+18). pahchem now applies it once per molecule.
- Photolytic H₂ loss from superhydrogenated states: RRKM with E₀ = 1.6 eV (neutral) and 3.2 eV (cation),
  after Boschman+15 and Le Page.
- Alternating cation addition barriers (Boschman+15).
- `h2_formation_rate` made net: today H₂-consuming channels are not subtracted (e.g. `lepage01.py`).

Targets:
- Castellanos+18 efficiency of 1–2 %;
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

## 4. Validation

The exact numbers are in the literature page, section 8.

| | target | status |
|---|---|---|
| V1 | Wolfire+08 Table 2 from DS87 (and Hollenbach+12 at N_C = 100) | passes (§A, ≤ 2.6 %) |
| V2 | W03/W08 three-state charge balance (N_C = 35; G₀ = 1.7, T = 100, n = 30): with W08's rates exactly, then with pahchem's, giving the effective Φ (expected 0.4–0.6) | to do |
| V3 | the W08 C I/C II regimes (their eq. B15) | to do |
| V4 | α_PAH against WD01b for ψ = 10²–10⁶, T = 10–10³ K (C⁺, H⁺, S⁺, Mg⁺, Fe⁺); WD01b §7, x_e 8.0 → 4.5×10⁻⁴ | to do |
| V5 | FALP e + PAH⁺ (10 points; trend −3.94 + 2.78 log N_C against DS87) | §D lists them |
| V6 | attachment: anthracene and C₆₀ | to do |
| V7 | mutual neutralisation: the Shuman law, and C₆₀⁺ + C₆₀⁻ (2.5×10⁻⁸ at 300 K, 1.35×10⁻⁷ at 10 K) | §B |
| V8 | charge transfer: anthracene (Canosa+95) | §C |
| V9 | PE heating: BT94 ε(γ); Berné+22 ε_PAH(γ) (N_C = 54) and its diffuse-ISM ε = 0.037, Γ = 2.7×10⁻²⁶ erg s⁻¹ H⁻¹; the W22 ratios of WD01 to BT94 | to do |
| V10 | Thi+20 Table B.3 (C₅₄ recombination) | to do |
| V11 | H₂: the ER constant, Castellanos+18, Boschman+15, Habart+04, Thi+20 R_max | to do |
| V12 | energy closure per process: absorbed = IR + photoelectron + ionisation work, returned on recombination | to do |
| — | pahchem's existing gates and the PDRs4All fits unchanged with ions and the opt-in deposit off | to do |

**Open item.** W08 (Φ = 0.4, x_PAH = 6×10⁻⁷) against WD01b at CNM conditions gives ×1.8 for C⁺ but ×6 for
H⁺, and the H⁺ factor is unexplained (WD01b's H⁺ ≈ its C⁺, where the DS87 mass scaling gives √12×). Check
it against WD01b Fig. 1 before V4.

**A possible grain-side bug, not changed.** `models/dust_chemistry/dust_h2_formation.py` and RAMSES
`dust_surface_chemistry.f90` use 2πa² with v = (2kT/m)^½. Cazaux & Spaans 2004 use ½ v_H πa² with the
mean speed, so the code may be 1.77× high.

## 5. Items for the user to supply

The full list is in section 13 of the literature page. The main items:
- **Tielens:** 2005 §3.3, §6.3, §8.7; 2021 §8.6–8.7, §12.8.
- **Rate theory:** Bakes & Tielens 1998; Lepp+88; Liszt 2011; Burgh+10.
- **Lab papers:**
  - Moustefaoui+98; Canosa+94/95; Biennier+06; Abouelaziz+93;
  - Hamberg+11; Hickman 1979; Bréchignac+14;
  - Flower & Pineau des Forêts 2026;
  - Andrews+16; Mennella+12; Thrower+12.
