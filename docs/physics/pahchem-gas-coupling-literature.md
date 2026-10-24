# PAH–gas coupling for pahchem: the literature behind the specification

Compiled on 2026-10-21 for the branch `feature/pahchem-anharmonic`. The specification built on it is {doc}`pahchem-gas-coupling`. The companion notes and derivation scripts named below are kept in `calima-pdr/docs/literature/pah_gas/` (the calima-pdr repository), and the arXiv PDFs read in `~/Documents/GitHub/pdr_refs/literature/pah_gas/pdfs/` (local only). The reproducible check of the derived rates is `diagnostics/pahchem/check_pah_ion_rates.py`.

**Status markers**
- [read]: full text, or the stated full sections, read here or by a sub-agent.
- [abs]: abstract or a snippet only.
- [sec]: the number comes from a secondary source that was read; the primary was not read.
- [derived]: my own calculation from a quoted formula. The scripts are in `calima-pdr/docs/literature/pah_gas/scripts/`.

**Companion notes** (in `calima-pdr/docs/literature/pah_gas/`)
- `notes_core.md`: W08, W03, WD01, WD01b, BT94, V90, Berne+22, H12, Thi+20 and W22, with every formula.
- `labdata.md`: laboratory and theory rates. It includes `labdata_topic3_5.md` (mutual neutralisation) and `labdata_topic4.md` (ion–molecule reactions).
- `citations.md` and `citations_raw.tsv`: 106 citers of W08.
- `h2_formation.md`: H2 formation on grains and PAHs.
- `code_notes.md`: the pahchem and pyCALIMA code paths.

**Conventions**
- Units are cm³ s⁻¹ unless stated. T2 = T/100 K.
- G0 is in Habing units unless stated.
- The charging parameter is γ = ψ = G0 T^½ / n_e, in K^½ cm³.
- The Draine & Sutin 1987 reduced quantities are τ = a kT/q² and ν = Ze/q.
- pahchem's capture geometry is 0.82 πa² with a = 0.9 √N_C Å (BT94/V90 disc). The absorption radius is 10 Å (N_C/468)^⅓ (Li & Draine 2001).

---------------------------------------------------------------------------------------------------

## 0. Summary of the recommended rate set

Section 9 has the details. Each rate below is per molecule m in charge state Z.

| # | process | recommended form | default parameter | uncertainty |
|---|---|---|---|---|
| R1 | e + PAH^Z (Z ≥ 1) | s_rec · 0.82 πa² v_e J̃(τ_e, −Z), as pahchem does now | s_rec = 1 for N_C ≳ 25. Optional capped lab trend below that (FALP, 300 K). | ×2 from the image-potential form (DS87 vs Spitzer). For N_C ≤ 16, ×0.03–0.4 against the data. T law ∝ T^−½ is assumed (measured T^−0.7…−1.1 for small aromatics). |
| R2 | e + PAH0 → PAH− | statistical sticking (`attachment.py`) × capture | as now | Lab values span 1e-9 (anthracene, EA 0.53 eV) to 1.2e-6 (C60, EA 2.68 eV). pahchem must reproduce both. |
| R3 | X⁺ + PAH^Z (Z ≤ −1) → X + PAH^(Z+1) (mutual neutralisation) | P_MN · 0.82 πa² v_X J̃(τ_X, Z), with v_X from the reduced mass (≈ m_X) | P_MN = 0.58 (Shuman+14 law / DS87 in this 0.82 πa² geometry, C₅₀H₁₈⁻, 300 K; 0.48 without the 0.82); Hickman 1979 gives 0.8–1.0 | ×2–3. No PAH⁻ measurement exists. T^−½ below 300 K (theory); T^−0.9 measured for 300–550 K. |
| R4 | X⁺ + PAH^0 → X + PAH⁺ (charge transfer), only if IP(X) > IP_m(0) | P_CT · k_L, with k_L = 2.34e-9 (α_m/Å³)^½ (μ/amu)^−½ and α_m ≈ 1.75 N_C Å³ | P_react = 0.9 total (ΔZ = +1), measured for He⁺ and C⁺ only; for C⁺ the products are 60 % charge transfer (C + PAH⁺) and 40 % carbon accretion (C₁₅H₉⁺ + H) (Canosa+95 p. 412) [read] | ×2. Ar⁺ reacts at only ≈ 0.5 k_L, so applying 0.9 to H⁺, N⁺, O⁺ and S⁺ is an extrapolation. For Mg⁺, Fe⁺ and Si⁺ the energetics are decided per molecule and no rate has been measured. |
| R5 | X⁺ + PAH^Z (Z ≥ 1) | 0 below 10⁴ K (Coulomb barrier, exp(−θ/τ) ≲ e^−100 at 100 K) | — | — |
| R6 | He⁺ + PAH^0 | dissociative or double charge transfer; total ≈ k_L | Tielens 2005 (via Thi+20): He + PAH²⁺ + e at 1.1e-8 Φ (N_C/50)^½ | Product branching is unknown. |
| R7 | H3⁺ / HCO⁺ / H3O⁺ + PAH^0 → HPAH⁺ + … (proton transfer) | ≈ k_L | P = 1 | Non-dissociative for aromatics (Milligan+02 [abs]). |
| R8 | PAH⁺ + PAH⁻ | ≈ 3e-8 (300/T)^½ (C60 Landau–Zener; Coulomb estimate for two C50) | — | ×5 high, ×1 low. This is second order in x_PAH. |
| E1 | photoelectron kinetic energy | ⟨E_k⟩ = γ_e (hν − IP_m(Z)) for Z ≥ 0, with the remainder (1 − γ_e)(hν − IP) left in the PAH. Detachment: E_k = hν − EA. | γ_e = 0.46 ± 0.06: Berné+22's unweighted average over 7.1–10.1 eV of the coronene photoelectron spectra of Bréchignac+14 (not a number in Bréchignac+14). Per photon energy it runs from ≈ 1 (7.5–7.8 eV) to 0.05–0.37 (9.1–9.6 eV). The classical value is 0.5 (V90, BT94). | ±0.06 is the spread of the mean; γ_e(E) spans 0.05–1 |
| E2 | recombination cooling | ⟨E⟩ per capture = kT Λ̃(τ, ν)/J̃(τ, ν) (DS87; `shared_physics.DS87_lambda_scalar`) | ≈ kT for Coulomb-focused capture on cations, ≈ 1.5–2 kT on neutrals | small |
| E3 | MN energy release IP(X) − EA_m | goes mostly into X* (radiated) and kinetic energy | 0 deposited in the PAH by default. Optional: half (C60 analogue). | unknown |
| H | H2 from PAHs | existing channels plus the fixes in §6 | — | see §6 |

**Effective Φ_PAH (main conclusion).** Three independent routes give an effective efficiency of about 0.4–0.5 relative to Draine & Sutin with s = 1, for ions on PAH⁻ and PAH⁰:
- the W08 observational calibration;
- the universal polyatomic mutual-neutralisation law (Shuman+14: 0.47–0.48 × DS87 for all cations at 300 K) [derived];
- for neutral PAHs, Langevin with the measured polarisability × the measured ≈ 0.9 total reaction efficiency (Canosa+95; for C⁺ only 60 % of it is charge transfer), against DS87's conducting sphere (about 0.5) [derived].

For electrons on large PAH⁺ (N_C ≳ 25), the data are consistent with s ≈ 1. pahchem's capture is already 0.81 × W08 at N_C = 35 [derived]. So a pahchem that implements R1–R4 physically should reproduce W08's calibrated C⁺ neutralisation without a free Φ_PAH. That is the key validation (§8, V1–V4).

---------------------------------------------------------------------------------------------------

## 1. Wolfire, Tielens, Hollenbach & Kaufman 2008 (W08) and its key references

### 1.1 W08: ApJ 680, 384; 2008ApJ...680..384W; arXiv:0803.0138; doi:10.1086/587688 [read]

**What it is.** A semi-empirical calibration. W08 fits N(C I)/N(C II) and f_H2 on 42 Copernicus, GHRS, STIS and FUSE sight lines (0.03 ≤ AV ≤ 2.13). It uses a two-sided version of the Kaufman+99 PDR code, with three free parameters: Φ_PAH, the H2 formation rate R, and G0/n. The PAH physics is not new; it is the rate set of Wolfire+03, written out in Table 2.

**PAH model (App. A)**
- Draine & Sutin 1987 rates for "disk" PAHs with N_C = 35 and a = (N_C/1.222)^½ Å = 5.35 Å.
- Abundance n_PAH = 6e-7 n_H, which puts 22e-6 C per H in PAHs (W03, from Tielens+99).
- Φ_PAH multiplies every collisional rate. It "include[s] a wide range of unknowns ... PAH size, geometry, and abundance".
- The full DS87 formalism runs in the code. Table 2 gives fits for diffuse-ISM conditions.

**Table 2 rates.** Collisional rates are in cm³ s⁻¹ and scale as m^−½ for other collision partners. Photo rates are in s⁻¹, for G0 in Habing units.

| reaction | rate |
|---|---|
| PAH⁻ + H⁺ → PAH0 + H | 8.3e-7 Φ T2^−½ |
| PAH0 + H⁺ → PAH⁺ + H | 3.1e-8 Φ (W03 eq. C7 adds T2^−½, which is inconsistent with Table 2) |
| PAH⁺ + e → PAH0 | 3.5e-5 Φ T2^−½ |
| PAH0 + e → PAH⁻ | 1.3e-6 Φ (W03: 1.34e-6) |
| PAH⁻ + C⁺ → PAH0 + C | 2.4e-7 Φ T2^−½ |
| PAH0 + C⁺ → PAH⁺ + C | 8.8e-9 Φ |
| PAH0 + hν → PAH⁺ + e | 4.6e-9 G0 exp(−1.8 AV) |
| PAH⁻ + hν → PAH0 + e | 1.2e-8 G0 exp(−1.8 AV) |
| C + hν → C⁺ + e | 2.1e-10 G0 exp(−2.6 AV) |
| C⁺ + e → C (RR + DR) | 1.8e-11 T2^−0.83 (Badnell/Altun+04; within 8 % for 20–300 K; Nahar & Pradhan 97 differs by 2× at 100 K) |

**Reproduction [derived, `scripts/ds87_check.py`].** DS87 J̃ for a sphere of radius 5.35 Å, with sticking 1 for both electrons and ions and no disc factor, reproduces every collisional entry to ≤ 2 %. At 100 K:

| reaction | reproduced value |
|---|---|
| e + PAH⁺ | 3.50e-5 |
| e + PAH0 | 1.29e-6 |
| C⁺ + PAH⁻ | 2.37e-7 |
| C⁺ + PAH0 | 8.75e-9 |
| H⁺ + PAH⁻ | 8.17e-7 |
| H⁺ + PAH0 | 3.02e-8 |

- W08's "disk" is therefore a sphere of the disc radius.
- Charged-partner rates scale ∝ a ∝ N_C^½ (τ ≪ 1). Neutral-PAH rates scale ∝ a^1.5 ∝ N_C^¾.
- Hollenbach+12 (N_C = 100) is reproduced by the same script.
- pahchem's own convention (0.82 πa², a = 0.9√N_C Å) gives 0.81 × these numbers at N_C = 35.

**Fit results**
- AV > 0.25: R ≈ 3.5e-17 and Φ_PAH ≈ 0.4.
- AV < 0.25: R ≈ 1e-17 and Φ_PAH ≈ 0.2–0.4.
- Φ is degenerate with R: the χ² trough follows Φ_PAH = 0.22 (R/1e-17) − 0.40 for R ≳ 3e-17, so Φ ∝ R.
- 23 Ori and π Sco are outliers, explained by time dependence. Grazing sight lines explain the low-AV points.

**Analytic relations (from W03)**
- n(PAH⁻) = 4.8e-5 n_e n Φ/G0 cm⁻³, valid while n(PAH⁻)/n(PAH0) < 1; n(PAH0) ≈ 0.7 n_PAH.
- When PAH⁻ dominates C⁺ destruction: G0/n = 4.1e-3 Φ T2^−¼ (N_CI/N_CII)^−½, with x_e = 3e-4. Then N_CI/N_CII ∝ Φ² (n/G0)².
- When PAH0 and e dominate: G0/n = (1.8e-5 Φ + 2.6e-5 T2^−0.83)(N_CI/N_CII)^−1. Then N_CI/N_CII ∝ Φ n/G0.
- PAH0 charge exchange takes over for G0/n ≳ T2^−½ Φ² / (1.2 Φ + 1.6 T2^−0.83), i.e. N_CI/N_CII ≲ 3e-4 for Φ = 0.5.
- Timescale: C I/C II reaches equilibrium in about 62 yr at n = 100 (Φ = 0.5), against about 5e4 yr for H2.

**H2.** W08 has no H2 formation on PAHs. It uses a single grain rate R n n_HI with a constant R.

**Uncertainties named by W08.** Φ absorbs PAH size, geometry and abundance. The C⁺ + e rate is uncertain by a factor of 2. W08 discusses no lab data. W03 §5 does: collision rates with Φ = 1 "consistently overestimate the laboratory measured rates for electron attachment to neutral PAHs and for electron recombination with PAH⁺" (citing Allamandola+89, Salama+96 and WD01a).

### 1.2 Key references of W08: rate prescription in each

- **Lepp & Dalgarno 1988b**, ApJ 324, 553 ("PAHs in interstellar chemistry", dense clouds) [read, supplied 2026-10-22]
  - For x_PAH > 1e-8 the free electrons attach to PAHs and PAH⁻ carry most of the negative charge. Mutual
    neutralisation with PAH⁻ then replaces dissociative recombination as the main loss of positive ions, and
    it is non-destructive.
  - Rates, all "of the order" of Omont (1986): attachment ≈ 6e-7; M⁺ + PAH⁻ ≈ 1.3e-6 T^−½, depending on M⁺ only
    as m^−½ (the reference mass is not stated); X⁺ + PAH charge transfer near Langevin, with the polarisability
    of Omont (1986). PAH⁻ + O detachment is neglected in cold clouds (barrier).
  - x_PAH = 1e-7 raises the steady-state C and C₃H₂ by two orders of magnitude, and the chemistry no longer
    depends on the metal abundance.
  - Upper limit from DCO⁺/HCO⁺ (H₂D⁺ + PAH⁻): x_PAH ≲ 6e-6 at most.
- **Lepp & Dalgarno 1988a**, ApJ 335, 769 ("Heating of interstellar gas by large molecules or small grains")
  [read, supplied 2026-10-22]
  - A single N_C = 50 molecule with IP = 6.8 eV and a constant σ = 1e-16 cm²; the photoelectron carries **all**
    of E − 6.8 eV (γ_e = 1). Ionisation from deeper levels, which gives slower electrons, is set aside as smaller
    than the uncertainty of a single IP.
  - Photodetachment of PAH⁻ can be a significant heat source in diffuse clouds.
  - For x_LM > 2e-7 the PAH heating (ionisation plus detachment) exceeds the standard grain heating. With the
    abundances inferred from C⁺/C toward ζ Oph and ζ Per, it balances the cooling in the cloud cores but not at
    the boundaries.

- **Wolfire, McKee, Hollenbach & Tielens 2003**, ApJ 587, 278 (astro-ph/0207098) [read §5, App. C]
  - Single N_C = 35 PAH, the "mean size of the 3–15 Å distribution", n_PAH/n = 6e-7, Φ_PAH = 0.5 fitted to CI/CII ≲ 3e-3.
  - Photoionisation and detachment from BT94: 7.85e-9 G0′ and 2.00e-8 G0′, with G0′ in Draine-like units.
  - BT94 photoelectric heating and recombination cooling rescaled ×1.3 for the higher PAH abundance, with Φ inside γ:
    - nΓ = 1.3e-24 n ε G0, with ε = 4.9e-2/[1 + 4.0e-3 (G0T^½/(n_e Φ))^0.73] + 3.7e-2 (T/1e4)^0.7/[1 + 2.0e-4 (G0T^½/(n_e Φ))].
    - n²Λ = 4.65e-30 T^0.94 (G0T^½/(n_e Φ))^β n_e Φ n, with β = 0.74/T^0.068.
  - App. C2:
    - n_e = 1.6e-10 (ζ′ T2^½ n/(Φ n_PAH⁻))^½ when H⁺ is destroyed by PAH⁻.
    - n_PAH⁻ = 2.8e-5 n_e n Φ Zd′/G0′.
    - n_e = 2.4e-3 ζ′^½ T2^¼ G0′^½ Zd′^−½ Φ^−1.
    - C6 says "N_C = 25", an apparent typo for 35.
- **Bakes & Tielens 1994**, ApJ 427, 822 (ADS scan) [read pp. 823–829, 835–836]. Formulas are in §2.1 and §3.1. In short:
  - s = 1 for electrons and ions; the 0.8 disc factor of V90 is ignored.
  - IP_Z = 4.4 + (Z+½) 25.1/N_C^½ eV for a disc.
  - Photoelectron energy partition g = ½ (hν − IP)/hν.
  - DS87 Λ̃ for recombination cooling.
  - Fits: ε(γ) (their eqs. 42–43) and Λ = 3.49e-30 T^0.944 γ^β n_e n_H with β = 0.735/T^0.068 (their eq. 44).
  - MRN-like N_C^−11/6 distribution, 14 ≤ N_C ≤ 4.6e5.
- **Bakes & Tielens 1998**, ApJ 499, 258 [read, supplied 2026-10-21]. One PAH size (N_C = 50, x_PAH = 2e-7; PAH, PAH⁺, PAHH⁺, PAH⁻), 45 species and 464 reactions.
  - Table 3: PAH⁺ + e 2.4e-5 (T/300)^−½; attachment 1.6e-6 (s = 1, EA > 1 eV for N_C > 25); PAH⁻ + X⁺ 1.0e-7 (T/300)^−½ for every atomic ion, with no mass scaling; molecular ions dissociative at 1e-7 per channel; PAH⁺ + PAH⁻ and PAHH⁺ + PAH⁻ 3.2e-8 (T/300)^−½; PAHH⁺ + e 2.4e-5. Table 2: photoionisation 7.9e-9 G0 e^(−1.8 A_V).
  - Table 1 ("charge exchange") is proton transfer only, 3.9e-9 for every donor. BT98 has **no** X⁺ + PAH⁰ charge transfer for atomic ions, and no Φ.
  - Results: anions dominate the PAH charge at A_V ≈ 3, and PAH⁻ becomes the main negative carrier once x_e < 1e-7; metals are neutralised at k_PAH/k_RR ≈ 3e-3/X(e) (neutral metals ×10, up to ×100 deep in); the C⁺/C/CO transition moves ≈ 1 mag toward the surface; [C I] lines rise ×2–3 (Orion Bar [C I] 609 µm 1.3e-5 against 1e-5 observed). C I peaks at A_V = 3.5 from C⁺ + PAH⁻ and at A_V = 5 from S + C⁺ → S⁺ + C.
- **Draine & Sutin 1987**, ApJ 320, 803 [read by sub-agent from the ADS scan]
  - J̃(ν=0) = 1 + (π/2τ)^½. This is the Langevin limit with α = a³.
  - J̃(ν<0) = (1 − ν/τ)[1 + (2/(τ − 2ν))^½]. This is 2× the Spitzer (1+φ) form for τ ≪ 1 (image-potential focusing).
  - J̃(ν>0) = [1 + (4τ + 3ν)^−½]² exp(−θ_ν/τ), with θ_ν = ν/(1 + ν^−½).
  - Λ̃: (ν=0) 2 + (3/2)(π/2τ)^½; (ν<0) ≈ (2 − ν/τ)[1 + (τ − ν)^−½]; (ν>0) ≈ (2 + ν/τ)[1 + (3/(2τ) + 3ν)^−½] exp(−θ_ν/τ). These are as quoted in BT94 eqs. 32–34.
- **Weingartner & Draine 2001b**, ApJ 563, 842 (astro-ph/0105237) [read]
  - Grain-assisted ion removal: dn_i/dt = −Σ n_gr f(a,Z) J_i(Z) θ[IP(X⁰) − IP(a,Z)], with J_i = n_i (8kT/πm_p)^½ A^−½ πa² J̃.
  - The sum runs over the WD01a R_V = 3.1, b_C = 6e-5 size distribution (PAHs included down to 3.5 Å), with WD01 charging.
  - Fit: α_g = 1e-14 C0 / {1 + C1 ψ^C2 [1 + C3 T^C4 ψ^(−C5 − C6 ln T)]} cm³ s⁻¹ per H, accurate to 20 % for 10–1e3 K and ψ = 1e2–1e6.
  - Table 2 has 12 ions; the coefficients are copied into `notes_core.md`.
  - C⁺ has C0 = 45.6 against 1.6–3.4 for the other ions at low ψ, because C⁺ can also take an electron from neutral grains and PAHs.
  - Ion sticking after transfer: s ≈ 1 for Ca, s ≈ 0 for C, Mg and S.
  - In CNM conditions (ψ ≈ 400, x_e ≈ 1e-3) the grain-assisted rate is comparable to radiative recombination. In the WNM it is ≲ 10 %.
  - H⁺ and He⁺ removal lowers x_e at n = 30, T = 100 from 8.0e-4 to 4.5e-4.
- **Weingartner & Draine 2001 (WD01)**, ApJS 134, 263 (astro-ph/9907251) [read §§2–3, 5–6]. See §2.
- **Lepp, Dalgarno, van Dishoeck & Black 1988**, ApJ 329, 418 [not read]: atomic ions in diffuse clouds, PAH-assisted neutralisation of metals. For the user to supply.
- **Lepp & Dalgarno 1988**, ApJ 324, 553 [read by sub-agent from ADS OCR]
  - M⁺ + PAH⁻ mutual neutralisation ≈ 1.3e-6 T^−½ ∝ M^−½ (from Omont 1986 with Y ≈ 1).
  - Attachment ≈ 6e-7, for N_C = 30–50 and IP ≈ 6.5 eV.
- **Omont 1986**, A&A 164, 159 [read by sub-agent from ADS scan]
  - k_n = 1.3e-7 (N_C/50)^½ (10/T)^½ (Y_r/0.3) (10/M_i)^½.
  - PAH⁺ + e: 3e-6 (a/1 Å)(100/T)^½ Y_r.
  - Attachment: 1.8e-6 Y_a (N_C/50)^¾.
  - This is the origin of a = 0.9 √N_C Å.
- **Verstraete, Léger, d'Hendecourt, Dutuit & Défourneau 1990**, A&A 237, 436 (ADS scan) [read]
  - Measured photoionisation of coronene and pyrene: σ_ion ∝ N_C; yield rises to 0.75 at 13.6 eV.
  - k_rec = 1.28e-10 N_C T^½ (1 + Φ), with Φ = 1.85e5/(N_C^½ T); disc a = 0.9√N_C Å and disc factor Y_rec = 0.80.
  - Partition γ = 0.5. Energy lost per recombination f kT, with f = (2 + Φ)/(1 + Φ).
  - For N_C = 80 in the Draine field: R_ion = 3.2e-8 s⁻¹ and ⟨E_e⟩ = 2.2 eV.
  - This is the source of pahchem's 0.82 disc factor.
- **Flower & Pineau des Forêts 2003**, MNRAS 343, 390 [read by sub-agent from the OUP text; tables not seen]
  - C54H18 with a = 6.6 Å.
  - PAH⁻ + H⁺: 7.5e-8 (300/T)^½, with probability 0.3. Other ions ÷ M^½ (including PAH⁺, which is wrong by about 10×).
  - PAH0 + H⁺: 4.4e-9 (probability 0.1).
  - Attachment: 1.0e-7 (s_e = 0.06).
  - PAH⁺ + e: 3.3e-6 (300/T)^½ (s_e = 0.3).
  - Collisional detachment: 3.0e-9 exp(−0.5 eV/kT).
  - Used by Meudon (Goicoechea+09: M⁺ + PAH⁻ = 1e-8 (300/T)^½).
- **Wakelam & Herbst 2008**, ApJ 680, 371 [read by sub-agent]
  - Dense-cloud PAH network: X⁺ + PAH⁻ at k = πa² v (1 + e²/(a kT)).
  - 50 % dissociative channel for H-bearing ions; X⁺ + PAH0 at Langevin.
  - Attachment ∝ N_C^¾ (Omont).
  - Caution: it is in the same ApJ issue as W08, and many lab papers cite it rather than W08.
- **Liszt 2003**, A&A 398, 621 (astro-ph/0211369) [abs]. **Liszt 2011**, A&A 527, A45 [read, supplied]
  - Without grain neutralisation, models cannot reproduce the C I columns.
  - The Burgh+10 C I columns are ×4–21 lower than W08's compilation for three of the four sight lines in common (HD 210839 agrees; Liszt Table 1). Liszt concludes that "at least one of the sets … is simply wrong" and wants sub-mm [C I] to decide; he does **not** ask for Φ to be refitted.
  - He uses Φ = 0.33 throughout: the W08 data fit with n = 256 cm⁻³, G0 = 1.7; the Burgh data fit either with Φ = 0.33 at n = 32 or with no PAHs at n ≳ 512.
  - His ζ_H is 1e-16 with PAHs and 1e-17 without, so Φ and ζ are calibrated together. Burgh+10 measure no C II, so any C I/C II needs an assumed C/H.
- **Welty+03** (ApJS 147, 61), **Jenkins & Tripp 2001/2011**, **Sofia+04**: the observational inputs (not read).

---------------------------------------------------------------------------------------------------

## 2. Process 1: photoelectric emission with photoelectron energies (gas PE heating)

### Physics and formulas

Heating by one PAH m in charge state Z:

  Γ_m,Z = ∫_{IP_m,Z}^{13.6 eV} σ_abs,m,Z(E) Y_m,Z(E) F(E) ⟨E_k⟩(E) dE  (+ the detachment term for Z < 0)

There are three choices for ⟨E_k⟩:

1. **Mean partition.** ⟨E_k⟩ = γ_e (E − IP_Z).
   - γ_e = 0.5 classically (V90 and BT94, from Terenin & Vilessov 1964 on benzene).
   - γ_e = 0.46 ± 0.06 is Berné+22's (App. C.4, Fig. C.2) unweighted average of γ_e(E) from the coronene photoelectron spectra of Bréchignac+14 (Fig. 2), which cover 7.1–10.1 eV for the monomer (cold beam, 120 K translational, 200–350 K vibrational) [read, both]. Bréchignac+14 itself gives no partition.
   - What the data imply: below 8.6 eV direct ionisation goes to the D0 0–0 band, so the electron carries almost all of hν − IP; autoionisation dominates the yield, and from Bréchignac Table I the electron's share is ≈ 1 at 7.5–7.8 eV, 0.3–0.7 at 8.31 eV and 0.05–0.37 at 9.06–9.58 eV. The ±0.06 is the spread of the mean, not of γ_e(E), and nothing constrains 10.1–13.6 eV.
   - Coronene IP = 7.310 ± 0.005 eV; cation states D0 7.310, D1 8.615, D2 9.11, D3 10.04, D4 10.21 eV (Bréchignac Table III).
   - IP_Z already contains the Coulomb term, so E − IP_Z is the energy at infinity.
2. **WD01 parabolic distribution** (WD01 eq. 10) [read]
   - f_E0(E) = 6 (E − E_low)(E_high − E)/(E_high − E_low)³.
   - For Z ≥ 0: E_low = −(Z+1)e²/a and E_high = hν − hν_pet. The escaping fraction is y2 = E_high²(E_high − 3E_low)/(E_high − E_low)³ (eq. 11).
   - For a molecule with a measured IP_Z, the escape barrier is already in IP. The consistent choice is then E_low = 0 and E_high = E − IP_Z, which gives a mean of ½(E − IP), i.e. equivalent to choice 1 with γ = 0.5.
   - pyCALIMA implements the WD01 form as `attempting_electron_energy_dist` (dust_photoelectric_heating.py l.569).
3. **BT94 partition** (eq. 29) [read]: g(N_C, IP_Z) = ½ (hν − IP_Z)/hν of the photon energy, i.e. the same as γ = 0.5.

**Photodetachment (Z = −1).** E_k = E − EA, all of it to the electron (Berne+22). WD01 adds a separate photodetachment cross section, σ_pdt = 1.2e-17 |Z| x/(1 + x²/3)², x = (E − E_pdt)/3 eV, with E_k = hν − hν_pdt + E_min (WD01 eq. 40; `shared_physics` l.161).

**Internal energy.** Energy conservation (Berne+22 eq. 21) gives E − IP = E* + E_K. So (1 − γ_e)(E − IP) stays in the PAH as vibrational energy.

**Net heating per H**

  Γ_PE = Σ_m x_m Σ_Z f_m(Z) Γ_m,Z − Λ_rec   (Λ_rec from §3)

Two efficiency definitions are in use:
- BT94/W03: ε relative to the 6–13.6 eV absorbed photons, nΓ = 1e-24 ε G0 n_H.
- WD01/Berne+22: relative to all absorbed power. ε_PAH = P_e/P_rad (Berne eq. 25).

### Reference numbers

- **BT94** [read]
  - ε = 4.87e-2/[1 + 4e-3 γ^0.73] + 3.65e-2 (T/1e4)^0.7/[1 + 2e-4 γ].
  - Heating per grain: H/(f_y N_C) = 1.4e-25 (13.6 − IP)^2.987 erg s⁻¹ (T_eff = 3e4 K).
  - Photoejection: J_pe/(N_C f_y) = 2.5e-13 (13.6 − IP)^1.983 s⁻¹.
- **WD01 eq. 44** [read]
  - Γ_pe = 1e-26 G n_H (C0 + C1 T^C4)/{1 + C2 ψ^C5 [1 + C3 ψ^C6]}.
  - ISRF, R_V = 3.1, b_C = 6e-5: 9.30, 0.248, 0.00697, 0.01848, 0.365, 0.633, 0.509.
  - B0 star: 3.66, 7.66, 0.00661, 0.01552, 0.094, 0.697, 0.482.
  - 96 % of the heating comes from a < 100 Å at ψ = 1e2.
- **Berne+22**, A&A 667, A159 (arXiv:2208.08762) [read §3–4, App. C]
  - Single N_C (54), charges −1 to +2.
  - ε_PAH ≈ a/(1 + bγ) + c/(1 + dγ) (Table 5). For T_eff = 3e4 K: a = 0.947, b = 0.646, c = 0.119, d = 6.77e-5.
  - Γ_PAH = 3.0 ± 0.6e-31 ε_PAH G0 f_C W/H.
  - Diffuse ISM (T = 80, n_e = 3e-2, f_C = 0.1, Draine field): ε = 0.037 and Γ = 2.7e-33 W/H (2.7e-26 erg s⁻¹ H⁻¹).
  - At γ = 100 Berne agrees with WD01 for a = 4.9 Å. At γ ≳ 1e5 Berne is up to 10× higher.
  - Charge fractions F = {f++, f+, f0, f−}: at γ = 1e5 and T_eff = 3e4 K (their Fig. 6), Berne gives {0.5, 0.4, 0.1, 0} and WD01 gives {0.7, 0.2, 0.1, 0}.
- **Tielens 2021** eq. 12.26 [read]: ε_PAH = 0.06/(1 + 7e-5 γ), using Y(10 eV) = 0.3 (see §12).
- **V90** [read]: ⟨E_e⟩ = 2.2 eV and R_ion = 3.2e-8 s⁻¹ (N_C = 80, Draine field). Cut-off temperature T_c = ⟨E_e⟩/2k = 1.3e4 K.
- **ProDiMo (Woitke+11; Thi+19/20)** [read via the citations agent]
  - Explicit per-charge-state heating with E_k = hν − IP_k (no partition).
  - IP_k from the WD01 formula: 3.10, 6.24, 9.38 and 12.5 eV for C54, k = −1 to +2.
  - Recombination cooling 1.5 kT per capture.
- **Wolfire+22 ARA&A** [read] (see §10)
  - Uses the BT94 ε, and states that eps is at most about 5 % for γ ≲ 1e2.
  - Half of the heating comes from grains < 15 Å.
  - WD01 (B0) is 1/1.7 of BT94 at γ = 1e5, 1/4.1 at 1e6, and 2.6× BT94 at 1e2.
  - Observed [C II]/FIR lies below the BT94 curve (Salas+19, Pabst+22).
  - **Typo:** W22 eq. 8 prints "1 + 2e-2 γ"; BT94 and W03 have 2e-4.
- **Weingartner, Draine & Barr 2006**, ApJ 645, 1188 [not read]: EUV and X-ray photoemission, Auger electrons and secondary electrons. Already used as the WDB06 option in pyCALIMA's grain charging; relevant only for E > 13.6 eV, where pahchem's grid ends.

### How it enters pahchem

- `photo.ionisation_rate` (photo.py l.217) integrates σ Y F on `E_PH` (5–13.6 eV). Add an energy-weighted twin, `photo.photoelectron_power`, that integrates σ Y F γ_e (E − IP_Z). Store it as `StateBasis.k_up_E` (this needs a `BASIS_VERSION` bump).
- **Consistency fix.** pahchem currently assumes ionising photons deposit no internal energy (`Params.ionising_photons_heat = False`; `photo.heating_spectrum` uses σ(1 − Y)F). That implicitly gives the electron all of E − IP, which contradicts γ_e = 0.46. With PE heating on, deposit (1 − γ_e)(E − IP_Z) into the IR cascade for ionising absorptions. This affects H loss from cations in strong fields, so keep it opt-in and labelled.
- Use the shifted thresholds (`population.shift_thresholds`, +0.6 eV for IP1), the same thresholds as the yields.
- Output Γ_PE per PAH per charge state, and the ensemble value per H.

---------------------------------------------------------------------------------------------------

## 3. Process 2: electron recombination and attachment, with recombination cooling

### Physics and formulas

**Cations.** k_e(Z) = s_rec · 0.82 πa² v_e J̃(τ_e, ν = −Z), with v_e = (8kT/πm_e)^½ (pahchem `chemistry.electron_capture`, l.44).
- For τ ≪ 1, J̃ → 2Z/τ. So k ∝ a T^−½ and is 2× the Spitzer/V90 form (2.17× Tielens 2021 eq. 8.106, which adds a 0.8 planarity factor).
- V90/Spitzer: k = 1.28e-10 N_C T^½ (1 + Φ), with Φ = 1.85e5/(N_C^½ T).
- Tielens 2021 eq. 8.106 [read]: 1.3e-6 N_C^½ (300/T)^½, 0.46 × pahchem's capture (see §12).
- W08: 3.5e-5 Φ T2^−½ at N_C = 35. H12: 3.4e-5 Φ (T/300)^−½ at N_C = 100.
- Thi+20: 4.1e-5 φ f(a) (N_C/50)^½ (100/T)^½ with φ ≈ 0.2–0.4. Multiply charged ions are scaled ×(1 + W_PAH/kT) [read].
- FPdF03: 3.3e-6 (300/T)^½ (s_e = 0.3).
- WD01 sticking: s_e(Z>0) = 0.5 (1 − e^(−a/l_e)), with l_e = 10 Å.

**Neutrals.** k_att = s_e(T, m) × capture with J̃(ν = 0).
- pahchem computes s_e statistically: Klots detailed balance, autodetachment against the IR cascade (`attachment.py`).
- WD01 eq. 28: s_e = 0.5 (1 − e^(−a/l_e))/(1 + e^(20 − N_C)).
- Carelli+13 (theory, upper bounds): k = α (T/300)^β exp(−γ/T). Coronene: 2.74e-9, 0.11, −1.12. Anthracene: 1.31e-9, 0.21, −1.12.
- Tielens 2005 [sec]: 1.3e-7 s_e N_C^½.
- W08: 1.3e-6 Φ. H12: 3.0e-6 Φ.
- Omont/WH08: 1e-7 N_C^¾ to 1.8e-6 (N_C/50)^¾.

**Recombination cooling** (energy taken from the gas per capture)
- BT94 eq. 31 and WD01 eq. 42: Λ′ = Σ_i n_i s_i (8kT/πm_i)^½ πa² Λ̃(τ_i, ν_i) kT.
- So ⟨E⟩ = kT Λ̃/J̃.
  - For Coulomb-focused capture on cations (τ ≪ 1), Λ̃/J̃ → 1, so ⟨E⟩ ≈ kT. The V90 form (2 + Φ)/(1 + Φ) kT agrees.
  - For neutrals, ⟨E⟩ ≈ 1.5 kT; it tends to 2 kT at τ ≫ 1.
- Ion captures carry their kinetic energy too. They are ≲ 1 % of the electron term, since k_ion n_ion ≪ k_e n_e for cations; ignore them.
- BT94 eq. 44 fit: Λ = 3.49e-30 T^0.944 γ^(0.735/T^0.068) n_e n_H. W03 scales it ×1.3 (to 4.65e-30) with γ → γ/Φ and n_e → Φ n_e.
- WD01 eq. 45: Λ = 1e-28 n_e n_H T^(D0 + D1/x) exp(D2 + D3 x − D4 x²), with x = ln ψ. ISRF, b_C = 6: 0.3632, 2.937, −7.601, 1.742, 0.06228.
- Wolfire+22: this cooling becomes significant for T ≳ 3e3 K and γ ≳ 1e4.

### Lab data (labdata.md §1–2)

**PAH⁺ + e at 300 K** (FALP/FLAPI). No PAH with N_C ≥ 10 has a measured T dependence.

| species (N_C) | k (cm³ s⁻¹) | source |
|---|---|---|
| naphthalene (10) | 0.3 ± 0.1e-6 | Abouelaziz+93 [abs]; Biennier+06 |
| naphthalene (10) | 8e-7 | Hassouna+03 [sec] |
| naphthalene (10) | 9 ± 3e-7 | VENDAMS, Wiens+15 [read] |
| azulene (10) | 1.1 ± 0.3e-6 | Biennier+06 [sec] |
| acenaphthene (12) | 0.5 ± 0.2e-6 | Biennier+06 [sec] |
| anthracene (14) | 2.4 ± 0.8e-6 | Novotny+05 JCP 123, 104303 [read] |
| phenanthrene (14) | 1.7 ± 0.5e-6 | Hassouna+03 / Rowe+95 [sec] |
| fluoranthene (16) | 3.0 ± 0.9e-6; 2.5 ± 1.5e-6 | Biennier+06; Rebrion-Rowe+03 [abs] |
| pyrene (16) | 4.1 ± 1.2e-6 | Novotny+05 [read] |

**Smaller aromatics** (T dependence)
- C6H6⁺: 1.2e-6 at 300 K, about T^−1.1±0.6 (Fournier+13 [read]).
- C6D6⁺: 1.3e-6 (T/300)^−0.69. C6D7⁺: 2.0e-6 (T/300)^−0.83. Both CRYRING, ring kept intact (Hamberg+11 [abs]).
- No storage-ring measurement exists for any PAH⁺ with N_C ≥ 10, nor for C60⁺. No ab initio theory exists.

**Trend** [derived by the lab agent, `scripts/rec_compare.py`]
- Measured / (pahchem BT94-DS87) = 10^−4.13 N_C^2.95 over the 8 PAH points with N_C 10–16 (10^−3.94 N_C^2.78 before one fluoranthene experiment counted twice was removed, 2026-10-22). The ratio is 0.03–0.37 for N_C = 10–16 and extrapolates to 1 at N_C ≈ 25. `chemistry.lab_recombination_fit`, which includes benzene, gives 10^−1.96 N_C^1.01 (9 points) and reaches 1 only at N_C ≈ 87.
- At N_C = 50 and 300 K, the classical estimates span 3.3e-6 (FPdF03) to 2.0e-5 (DS87). The data favour 1–2e-5, i.e. BT94 with s ≈ 1. They argue against WD01 sticking (3.4e-6) and FPdF03.

**Attachment**
- Anthracene: 1e-9 at 300 K (FALP, Canosa+94 [sec]; the ±30 % is in neither paper read). CRESU (Moustefaoui+98 Table 3 [read]): 1.7e-9 (47.7 K), 8.4e-10 (74.5 K), 2.6e-9 (169.7 K), 1.0e-9 (300 K), normalised ×2.7 to the 300 K FALP value; the one independent absolute value is 1.0e-9 at 75 K (Langmuir probe). No systematic T trend (×3 scatter; "preliminary").
  - Caveat (Moustefaoui+98): the lab buffer is N₂ at 0.6–2.7e16 cm⁻³, where collisions can stabilise the anion; in space it must be radiative. pahchem's statistical sticking (the low-density limit) should match the lab value or fall below it.
  - Earlier data they quote [sec]: anthracene ≈ 2e-9 at 380–480 K (flat); naphthalene 3.1e-13 at 400 K. A large EA need not mean fast low-T attachment (p. 81).
- C60: 1.2e-6 at 400 K and 8e-7 at 625 K, a factor-3 absolute uncertainty (Viggiano+10 [read]). This supersedes the 1990s values (3e-9) on which WD01's "C60 sticking is low" rests.
- Tobita+92: no long-lived parent anion when EA ≲ 0.5 eV (naphthalene, phenanthrene, triphenylene) [abs].
- Electron Langevin capture is 1.0e-7 (α/Å³)^½: 5.2e-7 for anthracene, 6.5e-7 for coronene.
- Flower & Pineau des Forêts 2026, MNRAS [abs]: Carelli-type PAH attachment is 2–3 orders of magnitude too low and should approach Langevin above about 100 K. Not verified.

### How it enters pahchem

- `chemistry.electron_capture` stays as it is. The default is `capture_model = "bt94"`, s_rec = 1.
- The capped `lab_trend` is the option for N_C < 26. The baseline network ignores `capture_model`, which should be fixed or documented.
- Add `chemistry.recombination_cooling(z, a, T)`: the capture rate × kT Λ̃/J̃, using `shared_physics.DS87_lambda_scalar`.
- For neutral attachment, take the energy moment of the same weight that `Attachment.sticking` already uses (σ_c ε e^(−ε/kT)). That gives the s_e-weighted ⟨ε⟩, which differs from 1.5 kT because s_e depends on ε.
- `recombination_heating` deposits exactly IP_Z into vibrations. The correct internal deposit is IP_Z + ⟨E⟩, with ⟨E⟩ ≈ kT. This is a negligible correction below 1e3 K and should be documented.

---------------------------------------------------------------------------------------------------

## 4. Process 3: ion–PAH collisions

### 4.1 Mutual neutralisation X⁺ + PAH⁻ → X + PAH0 (X = H, C, S, Si, Mg, Fe, and molecular ions)

**Physics.** Coulomb-focused capture followed by long-range electron transfer at a curve crossing (Landau–Zener). The reaction is always exothermic: IP(X) − EA_m ≥ 5.1 − 1.5 > 0 for every ion of interest. There is no threshold test.

**Formula.** k_MN(X, m, Z<0) = P_MN · 0.82 πa² ⟨v_rel⟩ J̃(τ_X, ν = Z), with ⟨v_rel⟩ = (8kT/πμ)^½ and μ ≈ m_X.

**Prescriptions** (labdata §3). At 300 K for C⁺ + PAH⁻ with N_C ≈ 50 [derived]:

| source | formula | value |
|---|---|---|
| DS87 (s = 1) | as above | 1.65e-7 |
| W08 (Φ = 1, N_C = 35) | 2.4e-7 Φ T2^−½ (12/m_X)^½ | 1.4e-7 |
| Thi+20 | 2.9e-7 φ (12 amu/m_X)^½ (100/T)^½ (N_C/50)^½, with φ = 0.2 | 3.3e-8 |
| Hickman 1979 (alternative quoted by Thi+20) | 5.33e-7 (T/300)^−½ μ^−½ EA^−0.4 | — |
| Omont86 | 1.3e-7 (N_C/50)^½ (10/T)^½ (Y_r/0.3) (10/M)^½ | — |
| FPdF03 | 7.5e-8 (300/T)^½ M^−½ (probability 0.3) | 2.2e-8 |
| WH08 | πa² v (1 + e²/(a kT)) | 5–8e-8 |
| Walsh+09 / Harada & Herbst 08 (any anion + cation) | 7.5e-8 (T/300)^−½ | — |

**Lab** [labdata §3, read]
- No PAH⁻ + X⁺ rate has ever been measured.
- Polyatomic MN at 300–550 K (AFRL VENDAMS) follows k = 2.7e-7 μ^−½ (T/300)^−0.9 to 10–20 % (Shuman+14, JCP 140, 224309). The Miller+12 fit is (2.8 ± 1.0)e-7 (T/300)^(−0.9±0.2) μ^(−0.5±0.1) EA^(−0.13±0.04); Hickman's EA^−0.4 is not confirmed.
- Naphthalene⁺ + Cl⁻, Br⁻, I⁻ at 300 K: 9.5, 6.1 and 2.9e-8 (Wiens+15).
- Atomic pairs can be 10–100× lower (C⁺ + Cl⁻ = 4.2e-9; Sawyer+19). PAH⁻, with its dense final-state manifold, should follow the polyatomic law, but this has not been tested.
- DESIREE, C60⁺ + C60⁻ (Gatchell+25, A&A 693, A43; arXiv:2409.11851):
  - The final states were measured; the absolute rate comes from Landau–Zener theory.
  - k = 2.52e-8 (300 K), 4.31e-8 (100 K), 1.35e-7 (10 K), going as T^−½.
  - C60* carries about 3.6 eV of the 4.9 eV exothermicity.
  - The authors warn that planar PAHs could be "markedly different".

**Key derived result.** At 300 K, the Shuman law / DS87 (s = 1) ratio is 0.47–0.48 for all cations: H⁺ 2.7e-7 vs 5.7e-7, C⁺ 7.9e-8 vs 1.65e-7, Fe⁺ 3.8e-8 vs 7.9e-8. Both scale as μ^−½. So P_MN ≈ 0.5, independently of W08's Φ_PAH ≈ 0.4–0.5.

**Temperature.** Use T^−½ below 300 K (theory, and the DESIREE C60 result). Do not extrapolate T^−0.9: it would give 5.8e-6 for H⁺ + PAH⁻ at 10 K.

**Products.**
- Atomic ions: X (possibly excited) + PAH0.
- Molecular AH⁺: 50 % A + H + PAH and 50 % AH + PAH. This is the Thi+20/WH08 assumption; the branching is unknown.

**Energy.**
- Default: deposit nothing in the PAH. With the Landau–Zener crossing at large R, the electron lands in an excited state of X and the energy defect is small.
- In the symmetric C60 case each partner carries about 0.65 eV of kinetic energy. The kinetic energy of the products (about 1 eV) heats the gas, which is negligible per H.

### 4.2 Charge transfer X⁺ + PAH^0 → X + PAH⁺ (and on PAH⁺)

**Physics.** Ion–induced-dipole (Langevin) capture, followed by charge transfer if it is exothermic: IP(X) > IP_m(0).
- CT is exothermic with every PAH for H⁺ (13.60 eV), O⁺ (13.62), N⁺ (14.53), He⁺ (24.59), C⁺ (11.26) and S⁺ (10.36).
- Si⁺ (8.15), Fe⁺ (7.90) and Mg⁺ (7.65) decide per molecule. IP: naphthalene 8.14, pyrene 7.43, coronene 7.310 eV (Bréchignac+14), C54 about 5.9–6.2 eV.
- pahchem's per-molecule IP (shifted DFT) makes this a natural, library-resolved threshold. WD01b uses the same θ test.

**Rates**
- k_L = 2πe (α/μ)^½ = 2.34e-9 (α/Å³)^½ (μ/amu)^−½, independent of T.
- Polarisabilities: α = 17.4 (naphthalene), 26.7 (anthracene), 42.5 (coronene), 79 (C60) Å³. That is about 1.75 N_C Å³ per carbon.
- k_L for a C50 PAH (α ≈ 88 Å³): H⁺ 2.2e-8, He⁺ 1.1e-8, C⁺ 6.4e-9, S⁺ 4.0e-9, Fe⁺ 3.1e-9.
- Measured on anthracene, 293 K (Canosa+95 Table 1 [read]): He⁺ 5.4 ± 2.1 / 5.1 ± 1.5e-9 (high / low n_e), C⁺ 3 ± 1.5e-9 (one measurement), Ar⁺ 0.8 ± 0.7 / 1.2 ± 0.4e-9. With the paper's α = 25.4 Å³ these are 0.90 (He⁺), 0.88 (C⁺) and ≈ 0.5 (Ar⁺) of Langevin.
  - These are **total** reaction efficiencies. For C⁺: 60 % C14H10⁺ (charge transfer) and 40 % C15H9⁺ (carbon accretion with H loss) (p. 412), so charge transfer alone is ≈ 0.53 k_L. He⁺ gives the parent ion, fragments and possibly the dication (branching not measured).
  - Ar⁺ is exothermic by ≈ 8 eV yet reacts at only 0.5 k_L: the efficiency is ion-dependent. Nothing on Mg⁺, Fe⁺, Si⁺, S⁺, N⁺ or O⁺, and no T dependence.
- CT and proton transfer to naphthalene and other aromatics run at about the collision rate (Midey+00, Milligan+02 [abs]).

**Comparison with DS87.** The DS87 neutral J̃ is exactly Langevin with α = a³. For N_C = 35, a³ = 153 Å³ against the measured α ≈ 61 Å³, so DS87 is 1.6× too high. That again gives an effective Φ ≈ 0.9/1.6 ≈ 0.55 for the total rate [derived].

**Recommendation.** P_react · k_L for ΔZ = +1, with α_m = 1.75 N_C Å³ (or DFT polarisabilities if available) and P_react = 0.9 (He⁺, C⁺ measured; other ions extrapolated and labelled), split into products: for C⁺, 0.6 C + PAH⁺ and 0.4 carbon accretion (the C is not returned to the gas).

**Other cases**
- **He⁺:** dissociative CT or PAH²⁺ formation. Tielens 2005 [sec, via Thi+20] gives He⁺ + PAH → He + PAH²⁺ + e at 1.1e-8 Φ (N_C/50)^½. This is a Δz = +2 step, which the nearest-neighbour chain cannot hold; put it in `BaselineModel.network` only.
- **X⁺ + PAH^Z (Z ≥ 1):** the Coulomb barrier gives exp(−θ/τ) ≈ e^−167 at 100 K for a = 5 Å. Negligible below 1e4 K. Thi+20 finds T_min = 16 710 (N_C/468)^−⅓ K.
- **X + PAH⁺ → X⁺ + PAH:** only for IP(X) < IP_m(0). That means Na and K, and Mg/Fe with small PAHs. Neutral metal densities are inputs, and the channel is usually negligible.
- **Proton transfer** (H3⁺, HCO⁺, H3O⁺, N2H⁺ + PAH0 → HPAH⁺ + …): at about the collision rate and non-dissociative for aromatics (Milligan+02 [abs]). This is a Δz = +1, ΔN_H = +1 step that links the charge and hydrogenation chains in dense gas.

### 4.3 Other ion and neutral channels

- **PAH⁺ + PAH⁻:** ≈ 3e-8 (300/T)^½ (C60 Landau–Zener; Coulomb estimate for two C50: 3.3e-8 at 300 K, 1.75e-7 at 10 K) [derived/read]. It needs two-population coupling (f_m(+) f_m′(−)), i.e. a nonlinear iteration. It is second order and minor wherever PAH⁻ or PAH⁺ is rare.
- **Collisional and associative detachment.**
  - PAH⁻ + H → PAH-H + e: 7.8e-10 to 1.0e-9 for dehydrogenated anions (Demarais+12 [read]). pahchem already uses 7.8e-10.
  - Collisional detachment PAH⁻ + X → PAH + X + e: 1.5e-8 φ (1 amu/m_X)^½ (N_C/50)^½ exp(−5500/T) (Thi+20, from FPdF03). Matters only in hot gas.
- **PAH⁺ + H, O, N** (FA-SIFT, Bierbaum/Snow; labdata §4)
  - PAH⁺ + H: 1.3–2.2e-10 (pahchem uses 1.4e-10).
  - PAH⁺ + O: 1.0–1.3e-10 (addition, plus a CO channel for small PAHs). This is a C-chemistry and oxidation route not in pahchem.
  - PAH⁺ + N: falls from 1.2e-10 (benzene) to < 1e-12 (coronene).
  - PAH⁺ + H2: ≤ 5e-13.

### 4.4 How it enters pahchem

- **Environment.** `baseline.Environment` and the population inputs need ion densities n_X for H⁺, He⁺, C⁺, S⁺, Si⁺, Mg⁺, Fe⁺, HCO⁺, H3O⁺, H3⁺. These are inputs from sourced environment cards, consistent with "conditions are inputs"; pahchem must never predict them.
- **Up rates (z → z+1).**
  - Z ≥ 0: Σ_X n_X P_CT k_L θ[IP(X) − IP_m(Z)]. This is in practice Z = 0 only, plus Z = 1 for H⁺ and He⁺ with small PAHs when IP2 < 13.6 eV, which then needs the Coulomb factor.
  - Z < 0: Σ_X n_X k_MN.
  - Add these to k_up(z) in `population.charge_fractions` (l.371) and `charge_matrix` (l.754), and as Δz = +1 transitions in `BaselineModel.network` (l.307).
  - Δz = +2 (He⁺) and the proton-transfer Δz = +1, ΔN_H = +1 step go in the network only.
- **Detailed balance in the chain.** Ions only add to k_up, so the nearest-neighbour form f(z+1)/f(z) = k_up(z)/k_down(z+1) still holds. The down rate stays n_e α(z+1), plus the rare X + PAH⁺ → X⁺ + PAH.
- **Reuse grain-side code.**
  - `shared_physics.DS87_J_function_vec(Z, z_i, a, T)` and `collisional_rates_ions_vector` (ion mass and charge).
  - `dust_ion_recombination.ATOMIC_RADII` and `IONIZATION_POTENTIALS`, and its `case_a` θ test. The `case_b` image-energy correction changes WD01b rates by ≲ 10 %.
- **Geometry.** Keep pahchem's 0.82 πa² with a = 0.9√N_C Å for every capture: electrons, mutual neutralisation, and the DS87 option for neutrals. One geometry factor then enters all rates. The Langevin form for neutrals needs no radius.

---------------------------------------------------------------------------------------------------

## 5. Process 4: outputs to a gas chemical network

`PAHNetwork.fluxes(f)` (network.py l.194) already returns events per PAH per s for each named process. `h2_formation_rate` (l.198) returns H2. What is needed:

1. **Per-ion neutralisation coefficient per H nucleus**, the drop-in replacement for WD01b α_g:

     α_PAH(X; inputs) = Σ_m x_m Σ_Z f_m(Z) k_X(m, Z)   (cm³ s⁻¹ per H; dn_X⁺/dt = −α_PAH n_X⁺ n_H)

   Split it into MN (PAH⁻) and CT (PAH0) parts, with product channels; for C⁺ + PAH⁻ the product is C + PAH0. This needs named processes per ion (`mn_C+`, `ct_C+`, …).
2. **Net PAH electron source per H:** Σ_m x_m Σ_Z f_m(Z) [k_photo,up(Z) − n_e k_e(Z)], with the attachment part separated.
3. **Photoelectric heating Γ_PE and recombination cooling Λ_rec per H** (§§2–3), plus the internal-energy bookkeeping.
4. **H2 formation rate per H:** R_f,PAH = Σ_m x_m h2_per_pah_per_s / n(H) (§6). n(H) must be an input; `export_pah_tables` currently sets n(H) = n_H, which is valid only in atomic gas.
5. **f(Z) and ⟨Z⟩ per bin.** For RAMSES or the solver this means tables over (G0, T, n_e) plus ion fractions. Mutual neutralisation competes with photodetachment for PAH⁻. For example, with C⁺ at 1.6e-4 n_H and n_H = 100: k_MN n_C⁺ ≈ 0.5 × 2.4e-7 × 0.016 ≈ 2e-9 s⁻¹, against 1.2e-8 G0 for photodetachment [derived]. So f(Z) depends on n_C⁺/G0 in the CNM, and in dark gas mutual neutralisation dominates. The table axes must include ψ = G0 √T/n_e, T, and n_ion/n_e (or explicit n_C⁺, n_H⁺). Alternatively, fit to a WD01b-like form per ion.

Gas-network consistency:
- C⁺ + e uses Badnell (W08 eq. 3).
- H⁺ + e case B: 3.5e-12 (T/300)^−0.75 (WD01b, from Liszt 2001).
- H⁺ is affected the most (H12: PAH rates are "extremely important in deriving cosmic ray ionization rates" from OH⁺/H2O⁺).
- pyCALIMA has no Python gas network. `solvers/rhs.py` returns element mass changes only, and the RAMSES flag `H2onpah` is declared but unused (code_notes §3). The outputs above are therefore an export interface (tables or fit coefficients), not a solver coupling, until a network exists.

---------------------------------------------------------------------------------------------------

## 6. H2 formation on grains and PAHs

Full notes are in `h2_formation.md` (23 papers read; sections 0, A–F).

### 6.0 What pahchem counts now (h2_formation.md §0; code_notes §1)

**Bookkeeping.** `PAHNetwork.h2_formation_rate` (network.py l.198–200) sums `h2_released` × flux. It is exported as `h2_per_pah_per_s` (export_pah_tables.py l.112).

**H2 channels in the baseline**
- **Photolytic H2 loss** (baseline.py l.316): RRKM, West+18 E0 = 4.58 eV, ΔS = 40 J K⁻¹ mol⁻¹, degeneracy 2. It is open only for even N_H ≤ N_H0 with N_H > n_solo.
- **Eley–Rideal** (l.331–332): 0.06 Å² × v_H n_H = 8.7e-13 (T/100)^½ n_H. It applies to all charges with N_H > N_H0, once per molecule rather than per excess H.

**Channels that feed them**
- Cation H addition: 1.4e-10 n_H, no barrier.
- Neutral H addition: geometric 35.7 Å² × N_C/24 (the × N_C/24 scaling is pahchem's assumption, not Andrews'), with the 7 % efficiency for dehydrogenated neutrals only (below N_H0), and barriers of 0.06 eV (at N_H0) and 0.035 eV (at N_H0+2).
- Associative detachment: 7.8e-10.

**Gaps**
- No photolytic H2 loss from superhydrogenated states (Le Page / Boschman use 1.6 eV neutral and 3.2 eV cation).
- No dissociative-recombination H2.
- `lepage01.py` l.408 "H2 addition" consumes H2 but is not subtracted, so the rate is gross, not net.
- `murga2022` removes 2 H per ER event while `murga2020` removes 1.

### 6.1 PAH routes: rates and dependences

**ER abstraction on superhydrogenated PAHs.** k = σ_abs v_H per excess H.
- σ = 0.06 ± 0.02 Å² per **aliphatic C–H bond**, measured on coronene films only at 300 K (Mennella+12 Fig. 5 [read]; the aC:H values 0.03 Å², Mennella 2008, and 0.05 Å², Biener+95, are comparisons quoted there). An extra H on an edge carbon makes a CH₂ with two abstractable H, so the per-molecule rate lies between 1× and 2× (N_H − N_H0) σ v n(H). Thi+20 uses it as 1.5e-12 (T/300)^½ x cm³ s⁻¹ [read].
- σ = 0.01 Å² at 2000 K (Skov+14 [abs]).
- 17 Å² (Zecho+02, graphite, low coverage) and 4 Å² (Pasquini+16, quasi-classical) [sec].
- Cations: abstraction:attachment ≈ 1:2, σ ≳ 0.45 Å² against 0.9 Å² for attachment (Foley+18 [read by agent]). That is ≥ 7× the neutral value.
- Cation abstraction per Montillaud+13: 1.4e-10 (N_H/12)(N_C/24)^−1 [read].
- T dependence is ∝ T^½ (thermal flux). It depends on G0 through the superhydrogenated population and on charge through σ.

**Hydrogenation barriers**
- Neutral coronene: first H at 0.06 eV (696 K; Rauls & Hornekær 08 [sec]). Boschman+15 tabulates per-state barriers alternating 28–64 meV, with a 10 meV abstraction barrier [read].
- Cations: 72 ± 6 meV (second H) and 40 ± 10 meV (fourth H) (Boschman+12 [read]). Cazaux+16 gives magic numbers +5, +11, +17, and 0.01 eV for the first H on coronene⁺ [abs].
- Thi+20: neutral hydrogenation 2.78e-11 (T/300)^½ N_C Q_Bell(E_act) (1.1 Å² per reactive C) [read].
- Cations: about 1–2e-10, roughly size-independent (Snow+98; Demarais+14 [read]).

**Photolytic H2 loss** (Castellanos+18b) [read by agent]
- Efficiency ξ = Σ 2 k_H2 f/(κ_add n_H): 1–2 % for coronene⁺ and circumcoronene⁺ at every G0, and ≥ 10× higher for ovalene⁺.
- R_PAH = ½ ξ κ_add x_PAH ≈ 4e-18 (f_C/0.1)(50/N_C)(ξ/0.1).
- Ovalene in NGC 7023 gives about 1e-17, around 20–50× Andrews+16.

**Boschman+15** (A&A 579, A72, "H2 formation on PAHs in PDRs: a high-temperature pathway") [read]
- The dominant PDR route is photolytic: a bare skeleton picks up 2 H and loses H2.
- R up to a few × 1e-16, only for T > 200 K.

**Andrews, Candian & Tielens 2016** (A&A 595, A23) [sec, from the repo method card; A&A returned 403]: ER is negligible in NGC 7023 NW.

**Tielens 2021** "optimum": ~2e-17 (T/100)^½ (Wolfire+22, private communication) [sec, unverified].

**Habart+04 PDR constraints** (Table 3) [read by agent]: R_f = 1.5e-16 for Oph W, S140 and IC 63; 3e-17 for NGC 2023 and the Orion Bar; 4e-17 for Chamaeleon. R_f scales with PAH abundance.

### 6.2 Grain routes

**Standard rate.** R = 3e-17 (T/100)^½ S(T, T_d) η(T_d) Z_d′ (Hollenbach & McKee 79; Wolfire+22 eq. 7).
- S_phys = [1 + 0.4 ((T + T_d)/100)^½ + 0.2 T/100 + 0.08 (T/100)²]^−1.
- Jura's 3e-17 is an empirical diffuse-cloud total.
- Cazaux & Tielens 2002/2004 efficiency ε(T_d) (physisorbed + chemisorbed): ~1 at 6–25 K, ~0.2 at 100 K, dropping near 300 K. Cazaux & Spaans 2004 parameters match pyCALIMA's [read by agent].

**Le Bourlot+12 ER** [read by agent]: k = 2.8e-17 √T · α e^(−300/T)/(1 + α e^(−300/T)), with α = 1/(1 + (T/464)^1.5). This gives 1.2e-17 at 100 K and 1.3e-16 at 500 K, i.e. 2–5× the standard rate in bright PDRs.

**Bron+14**: stochastic small-grain temperatures raise the low-G0 rate ~4× and move the H/H2 transition from AV 4e-2 to 8e-3, but change H2 lines by ≤ 2.7×.

**Thi+20 (ProDiMo)**: physisorption and chemisorption rate equations with Bell tunnelling; details in §11.

**Recent efficiencies**: ε ≈ 0.2 up to 250 K on carbonaceous grains (Grieco+23 Nat. Astron. [abs]); 0.06 (10 K), 0.28 (20–80 K), 0.19 (> 150 K) (Das+26) [read by agent].

**Wakelam+17 review** (Mol. Astrophys. 9, 1), Table 1: sticking and barriers, including PAH 60 meV and PAH⁺ 10 meV chemisorption barriers and σ_ER = 0.06 Å².

**Röllig+13 (KOSMA-τ)**: several H2 formation options; grain area per H from WD01 includes PAHs (5.4 of 6.7e-21 cm²).

### 6.3 Coupling to pahchem's hydrogenation chain

The recommended changes are all opt-in and labelled ad hoc until derived. None has been applied.
1. Apply ER per aliphatic C–H bond: between 1× and 2× (N_H − N_H0) σ_abs(Z) v_H n(H), with 0.06 Å² for Z ≤ 0 and ≥ 0.45 Å² for Z > 0.
2. Add a superhydrogenated photolytic H2 channel: RRKM with E0 = 1.6 eV (neutral) and 3.2 eV (cation), and the West+18-type ΔS.
3. Add alternating cation addition barriers (Boschman+15 Table A.1).
4. Make `h2_formation_rate` net by subtracting H2-consuming channels (`h2_released = −1`).
5. Settle 1 vs 2 H per ER event in murga2022.
6. Export R_f,PAH = Σ x_m h2_per_pah_per_s / n(H), with n(H) as an input.

Upper bound [derived by the agent]: R_PAH ≤ ½ κ_add x_PAH ≈ 3.5e-17 (f_C/0.1)(50/N_C) for cations.

### 6.4 Possible pyCALIMA grain-side issue (not changed)

`dust_h2_formation.py` and `dust_surface_chemistry.f90` l.133–142 use 2πa² with v = (2kT/m)^½. Cazaux & Spaans 2004 eq. 2 uses ½ v_H πa² with the mean speed. The code may be 1.77× high relative to that.

---------------------------------------------------------------------------------------------------

## 7. Process 5: double counting

1. **WD01b α_g already contains PAHs.** It integrates the WD01a distribution (b_C = 6e-5, PAH log-normals down to 3.5 Å).
   - Any external network using α_g (Gong+17, Kim+23 TIGRESS-NCR, UCLCHEM 4.0, HYACINTH, Hu+21) must drop it, or restrict it to a ≳ 10–15 Å, when pahchem's α_PAH is supplied.
   - Gong+17 and UCLCHEM multiply WD01 by 0.6 "to match W08" (Φ = 0.4, x_PAH = 2e-7) [read by agent].
   - W08 Table 2 with Φ = 0.4 and x_PAH = 6e-7 against WD01b, for G0 = 1.7, T = 100, n = 30, x_e = 4.4e-4 (ψ = 1.3e3) [derived, `diagnostics/pahchem/check_pah_ion_rates.py` §E]:

     | ion | W08 | WD01b α_g | ratio |
     |---|---|---|---|
     | C⁺ | 1.41e-14 | 8.74e-15 | 1.61 |
     | H⁺ | 4.88e-14 | 3.02e-14 | 1.61 |

     The PAH⁻ fraction is 0.22. WD01b's H⁺/C⁺ is 3.46, the DS87 mass scaling √(m_C/m_H) = 3.45, as in W08.
     - Until 2026-10-23 this table gave H⁺ 9.1e-15 and a ratio of ~6. That value is reproduced only with C2 = 1.578 for H⁺; WD01b's coefficient is 1.378, as pyCALIMA, 3D-PDR and DESPOTIC/Gong+17 all print.
     - The C⁺ row is not printed the same everywhere: DESPOTIC's `GOW.py` has C4 = 0.4845 where pyCALIMA and 3D-PDR have 4.845e-2, giving 1.8e-15 here (5× lower). 3D-PDR's second copy has C3 = 411.3 for 433.1 (+5 %). Not yet checked against the WD01b paper.
   - pyCALIMA itself does not use WD01b. `dust_ion_recombination.py` uses single-radius grain bins (≥ 50 Å) and excludes PAH bins, and nothing consumes it (code_notes §3).
2. **Photoelectric heating fits.** BT94, W03, W22-ε and WD01 eq. 44 are dominated by PAHs and small grains: half the heating comes from < 15 Å (W22), and 75–96 % from < 100 Å (WD01 h_s).
   - If pahchem supplies Γ_PE, drop these fits or rebuild them for a > a_PAH,max.
   - RAMSES `pah_pe_heating` (Berne+22, from `PAH_photoelectric_heating.py`) and a pahchem Γ_PE on the same PAH bins would double count. Choose one owner.
   - WD01 heating on DustBin_01/03 is fine as long as only a0 ≥ 50 Å is used.
3. **Recombination cooling.** BT94 eq. 44, W03, WD01 eq. 45 and Berne's 3/2 kT P_rec against pahchem Λ_rec. Same rule: one owner.
4. **H2 formation.** Do not add pahchem's PAH H2 to a fixed R = 3e-17 (Jura/HM79), which is an empirical total.
   - A WD01-based grain area per H is 55–81 % PAH surface (CS09: 1.6 of 2.8e-21; Röllig+13: 5.4 of 6.7e-21).
   - Meudon and Bron+14 ER includes 1–3 nm PAH-like grains.
   - Fix: use silicate (1e-21) + amorphous carbon (1.7e-22) area per H, as Boschman+15 does, or the explicit `ndust` bins.
5. **Electron budget.** If the gas network counts PAH attachment through α_g-type terms and pahchem also removes electrons, the free electrons are counted twice. Export the net electron source (§5 item 2) and remove the network's own grain electron terms.
6. **Φ_PAH inside γ.** W03/W08/W22 heating uses γ′ = γ/Φ. Do not apply Φ again on top of pahchem's physically derived f(Z).

---------------------------------------------------------------------------------------------------

## 8. Process 6: validation targets (exact numbers to reproduce)

**V1. W08 Table 2 from DS87.** Sphere a = (N_C/1.222)^½ Å, N_C = 35, s = 1, 100 K, all to ≤ 2 % (`scripts/ds87_check.py`):
- e + PAH⁺ 3.50e-5; e + PAH0 1.29e-6.
- C⁺ + PAH⁻ 2.37e-7; C⁺ + PAH0 8.75e-9.
- H⁺ + PAH⁻ 8.17e-7; H⁺ + PAH0 3.02e-8.

Hollenbach+12, N_C = 100 at 300 K: 3.46e-5, 3.00e-6, 2.34e-7, 2.03e-8, 8.06e-7, 6.99e-8. The paper gives 3.4e-5, 3.0e-6, 2.3e-7, 2.0e-8, 8.1e-7, 7.0e-8.

**V2. W03/W08 three-state charge balance.** Single N_C = 35 PAH, G0 = 1.7, T = 100, n = 30, x_e = 3e-4:
- n(PAH0) ≈ 0.7 n_PAH.
- n(PAH⁻) = 4.8e-5 n_e n Φ/G0.
- n_e = 2.4e-3 ζ′^½ T2^¼ G0′^½ Φ^−1 (W03 C15) when H⁺ + PAH⁻ dominates.

pahchem run with W08's rates should reproduce these. pahchem run with its own rates should give the effective Φ (expected 0.4–0.6).

**V3. W08 C I/C II regimes.** With the analytic eqs. B15 and B17, the observed N_CI/N_CII for the AV > 0.25 sight lines (Table 1) should come out for Φ ≈ 0.4. At minimum, reproduce eq. B15: G0/n = 4.1e-3 Φ T2^−¼ (N_CI/N_CII)^−½.

**V4. WD01b comparison.** At ψ = 1e2–1e6 and T = 10–1e3 K, compare α_PAH(C⁺, H⁺, S⁺, Mg⁺, Fe⁺) with WD01b eq. 8 for the PAH-only part. Expected: pahchem/WD01b ≈ 1–2 for C⁺. Also reproduce WD01b §7: x_e 8.0e-4 → 4.5e-4 at n = 30, T = 100.

**V5. Lab recombination.**
- The ten FALP points above (all at 300 K).
- The trend log10(k/k_DS87) = −4.13 + 2.95 log10 N_C (8 PAH points; −1.96 + 1.01 log10 N_C with benzene).
- At N_C = 50, 300 K: k between 1e-5 and 2e-5.

**V6. Attachment.** Anthracene 1–3e-9 (48–300 K, flat); C60 1.2e-6 at 400 K. Both from `attachment.py` with the right EA.

**V7. Mutual neutralisation.**
- The Shuman law at 300 K: 2.7e-7 μ^−½. For C50⁻: H⁺ 2.7e-7, C⁺ 7.9e-8, O⁺ 6.8e-8, HCO⁺ 5.1e-8, Fe⁺ 3.8e-8.
- The C60⁺ + C60⁻ Landau–Zener law: 2.52e-8 (300 K), 1.35e-7 (10 K).
- Naphthalene⁺ + Cl⁻ 9.5e-8 as a sanity check of the polyatomic law.

**V8. Charge transfer.** Anthracene at 293 K: He⁺ 5.2e-9, C⁺ 3e-9, Ar⁺ 1e-9 (Canosa+95).

**V9. Photoelectric heating.**
- BT94 ε(γ) (eq. 43) and Λ (eq. 44) for the BT94 size distribution.
- Berne+22 Table 5 ε_PAH(γ) for N_C = 54, T = 500 K.
- Berne+22 Table 4: diffuse ISM ε = 0.037, Γ = 2.7e-26 erg s⁻¹ H⁻¹ (Draine field, T = 80, n_e = 3e-2, f_C = 0.1).
- Berne+22 charge fractions at γ = 1e5, T_eff = 3e4 K: {f++, f+, f0, f−} = {0.5, 0.4, 0.1, 0}.
- V90: R_ion = 3.2e-8 s⁻¹ and ⟨E_e⟩ = 2.2 eV for N_C = 80 in the Draine field.
- W22 ratios: WD01 is 1/1.7 of BT94 at γ = 1e5, 1/4.1 at 1e6, and 2.6× at 1e2.

**V10. Thi+20 Table B.3** (C54, φ = 0.4): e + C54H18⁺ = 1.7e-5 (100/T)^½, ×(1 + 36 438.8/T) for 2+, ×(1 + 72 877.6/T) for 3+. Fig. B.1, the FALP comparison with φ = 0.2.

**V11. H2** (h2_formation.md §D)
- ER constant 8.7e-13 (T/100)^½.
- Castellanos ξ = 1–2 %.
- Boschman+15 Fig. 3 and Table A.2.
- Habart+04 Table 3.
- Le Bourlot+12 k_ER(T).
- Thi+20 Fig. C.1: σ = 0.06/0.6/6 Å²; the maximum is R_max ≈ 1e-16 f_PAH (σ/10 Å²)(T/100)^½ (x̄/10) s⁻¹.

**V12. Energy closure.** For each absorbed ionising photon: γ_e(E − IP) to the gas, (1 − γ_e)(E − IP) to the PAH, and IP returned on recombination as IP + ⟨E⟩ (with ⟨E⟩ taken from the gas). The sum over a steady cycle must close.

---------------------------------------------------------------------------------------------------

## 9. Recommended rate set for pahchem, with uncertainties

All rates are per molecule m (N_C, N_H, IP_m(Z), EA_m, α_m) in charge state Z. The radius is a = 0.9√N_C Å with geometry factor g = 0.82; v_i = (8kT/πμ_i)^½.

| process | rate | parameters | uncertainty | basis |
|---|---|---|---|---|
| photoionisation and detachment | ∫σYF dE | existing | existing | — |
| PE heating per event | γ_e (E − IP_m(Z)); detachment E − EA | γ_e = 0.46 (Berné+22 unweighted mean) | γ_e(E) spans 0.05–1 (coronene, 7.1–10.1 eV); 0.5 classical | Berne+22 from Bréchignac+14, V90, BT94 |
| e + PAH^Z⁺ | g πa² v_e J̃(τ_e, −Z) s_rec | s_rec = 1 (N_C ≥ 25); lab trend 10^−4.13 N_C^2.95 below (PAH-only fit; the code's `lab_trend` is the benzene-inclusive 10^−1.96 N_C^1.01) | ×2 (image form); T law assumed | FALP, DS87 |
| e + PAH0 | statistical s_e × g πa² v_e J̃(τ_e, 0) | EA_m (+0.37 eV shift) | must span 1e-9 (EA 0.5) to 1e-6 (EA 2.7) | Moustefaoui, Viggiano |
| recombination cooling | kT Λ̃/J̃ per capture | — | ±20 % | DS87, WD01 eq. 42 |
| X⁺ + PAH^Z⁻ | P_MN g πa² v_X J̃(τ_X, Z) | P_MN = 0.58 (Shuman+14 in this geometry; Hickman 1979: 0.8–1.0) | ×2–3; T^−½ (theory) vs T^−0.9 (lab, 300–550 K); both laws are size-independent, so a constant P_MN assumes DS87's N_C^½ scaling | Shuman+14, Hickman 1979, Gatchell+25, W08 Φ |
| X⁺ + PAH0 (IP(X) > IP_m) | P_CT · 2.34e-9 (α_m/μ)^½ | α_m = 1.75 N_C Å³; P_react = 0.9 total, for C⁺ 60 % charge transfer + 40 % accretion | ×2; ion-dependent (Ar⁺ 0.5); thresholds for Mg⁺, Fe⁺, Si⁺ | Canosa+95, Langevin |
| He⁺ + PAH0 | ≈ k_L; split between PAH⁺ + fragments and PAH²⁺ + e | 1.1e-8 Φ (N_C/50)^½ for the double channel | branching unknown | Tielens 05 [sec] |
| proton transfer from H3⁺, HCO⁺, H3O⁺ | k_L, giving HPAH⁺ | P = 1 | ×1.5 | Milligan+02 [abs] |
| X⁺ + PAH^Z⁺ | 0 | — | — | DS87 Coulomb barrier |
| PAH⁺ + PAH⁻ | 3e-8 (300/T)^½ | — | ×0.3–5 | Gatchell+25 |
| PAH⁻ + H (associative detachment) | 7.8e-10 | existing | ±50 % | Demarais+12 |
| H2 channels | §6.3 | — | — | — |

There is **no global Φ_PAH**. Keep a single optional `phi_pah` scale on all collisional rates for calibration studies only, documented as ad hoc. Expected outcome: pahchem's C⁺ neutralisation per H at W08 conditions should fall within ×2 of W08 with Φ = 0.4 (V2–V4). If it does not, that is a finding about the PAHdb size distribution or abundance, not a reason to tune P_MN.

---------------------------------------------------------------------------------------------------

## 10. Wolfire, Vallini & Chevance 2022: what it says about PAHs, and a process checklist

ARA&A 60, 247; 2022ARA&A..60..247W; arXiv:2202.05867 [read].

### 10.1 PAH and small-grain chemistry and heating in W22

- **Grain-assisted recombination (§3.2.6)**
  - "mainly light ions (H⁺, He⁺ and C⁺) recombine with electrons on PAH⁻ or charge exchange with PAH0". This reduces the free electrons and increases the neutrals.
  - Without it, CO production would be inhibited for ζ_p ≳ 1e-17/n2 s⁻¹ (n2 = n/100 cm⁻³).
  - Modelers use DS87 against N_C, integrated over the distribution or multiplied by a typical size and abundance.
  - W03/W08 Φ_PAH = 0.5. The Burgh+10 C I columns are lower than W08's compilation; Liszt (2011) asks for the discrepancy between the two data sets to be settled, not for a refit of Φ (see §1.2). Hollenbach+12 has fits near 300 K.
  - "the rates and required integrals are quite uncertain and urge that further work is needed".
- **Metal ions.** S photoionisation electrons at AV ≈ 1–5 suppress ion–neutral chemistry. This is mitigated by recombination on PAHs or metal freeze-out (Hollenbach+12).
- **Electron abundance.** x_e ≈ 1e-4 from C⁺ at the surface, ≈ 1e-7 deep inside. PAH-contributed electrons are insignificant (n_PAH/n ≈ 1e-7 against 1e-4 for C⁺).
- **Photoelectric heating (§3.3)**
  - nΓ_PE = 1e-24 ε G n with the BT94 ε (eq. 8; note the "2e-2" typo).
  - Maximum about 5 % for γ ≲ 1e2. It drops at higher γ because of Coulomb losses and fewer ionising photons.
  - Small grains and PAHs dominate: half the heating comes from < 15 Å.
  - The Φ_PAH modification γ′ = γ/Φ_PAH, with Φ ≈ 0.5.
  - Good agreement in the diffuse ISM (T ≈ 100, n_e ≈ 0.01, G0 ≈ 1.7, γ ≈ 1.7e3).
  - WD01 against BT94 by factors 1/1.7 to 2.6. The WD01 ε ≈ Meudon at the surface.
  - [C II]/FIR observations fall below BT94 (Salas+19, Pabst+22).
  - Reddening lowers ε by about 25 % at AV = 1, γ = 1e4. Spaans+94 gives a correction for cooler spectra.
- **Recombination cooling.** Significant for T ≳ 3e3 K and γ ≳ 1e4. ε is often quoted as net heating.
- **H2 formation (§3.2.3)**
  - k_H2 = 3e-17 (T/100)^½ S(T, T_gr) η(T_gr) Z_d′.
  - PDRs need 1–5× the diffuse rate (Habart+04).
  - ER chemisorption (Le Bourlot+12; Bron+14, barrier 300 K).
  - Bron+14 fluctuating very-small-grain/PAH temperatures: important for G0 < 200 and n > 1e3.
  - Superhydrogenated-PAH abstraction (Thrower+12) is inefficient if σ is as low as Mennella+12, given the photodesorption of the extra H (Andrews+16).
  - Photodesorption route (Castellanos+18); optimum ~2e-17 (T/100)^½ (Tielens 2021, private communication).
  - Observations require k_H2 ≳ 3e-17 even at T ≈ 1000 K, T_gr ≈ 75 K (Orion Bar).
- **Future Issues #2.** "rates for ion recombination on grains", and H2 formation at high gas and grain temperatures, need laboratory, theoretical or observational refinement.

### 10.2 Wolfire+22 checklist: process → recommended treatment → references

| process | W22 recommended treatment | references in W22 | pahchem/pyCALIMA status |
|---|---|---|---|
| thermal balance | nΓ = n²Λ, iterate G, T, n_e for the PE term | TH85; Tielens 2005; Draine 2011 | not in pahchem (no gas T solve; conditions are inputs) |
| FUV radiative transfer | k = α G0 exp(−γ_exp AV) or E2(γ_E2 AV); isotropic vs normal incidence; full RT for grain charge | Heays+17; Le Petit+06; Röllig+13; van Hoof+04 | fields are inputs |
| photoionisation and photodissociation rates | Heays+17 tables (Draine field) | Heays+17; Leiden database | — |
| H2 photodissociation and self-shielding | D = 5.8e-11 χ f_shield exp(−3.1 AV); Draine & Bertoldi 96 self-shielding | Sternberg+14; Abgrall+92; DB96 | — |
| CO shielding | Visser+09 | — | — |
| C photoionisation shielding | Rollins & Rawlings 12 | — | — |
| H2 formation | 3e-17 (T/100)^½ S η Z_d′ (HM79); LH + ER (chemisorption); fluctuating T_d; PAH routes uncertain | Jura 74; HM79; Cazaux & Tielens 04; Le Bourlot+12; Bron+14; Habart+04; Thrower+12; Mennella+12; Andrews+16; Castellanos+18; Wakelam+17 | pahchem PAH H2 (§6); pyCALIMA grain H2 |
| excited H2* chemistry | C⁺ + H2* → CH⁺ (ΔE 4640 K), S⁺ + H2* → SH⁺ (9860 K) | Agúndez+10; Nagy+13; Zanchet+19 | — |
| cosmic-ray ionisation | ζ_p ≈ 2e-16 diffuse; ζ_H = 1.5 ζ_p, ζ_H2 = 2.3 ζ_p; attenuation with N; secondary electrons | Indriolo+15; Neufeld & Wolfire 17; Padovani+09; Glassgold+12 | inputs (`cosmic_rays.py`) |
| grain-assisted recombination (PAH⁻ MN, PAH0 CT) | DS87 against N_C × Φ_PAH = 0.5; integrate over the distribution; recalibrate (Liszt 2011) | Lepp+88; WD01a; W03; W08; H12; Liszt 2011 | **new: §4** |
| metal ions and S electrons | include S, Si, Fe, Mg; PAH recombination or freeze-out | H12 | ion densities as inputs |
| freeze-out and surface chemistry | adsorption, LH/ER, thermal, photo- and CR desorption | H09; Cuppen+17; Garrod+08 | out of scope |
| gas-phase rates | KIDA, UMIST; Badnell DR/RR | Wakelam+12; McElroy+13 | gas network not in Python |
| photoelectric heating | BT94 ε(γ) (or WD01), γ′ = γ/Φ_PAH; size distribution and R_V; spectrum shape (Spaans+94) | BT94; WD01; W03; Abel+08; van Hoof+04 | **new: §2** (Berne+22 module exists) |
| recombination cooling | include at T ≳ 3e3 K, γ ≳ 1e4 | BT94; WD01 | **new: §3** |
| H2 formation heating | ER: about 0.6 eV kinetic + 2.7 eV internal; LH: equipartition about 1.5 eV each; collisional de-excitation above n_cr = 1.1e5/√T3 | Sizun+10; Röllig+06 | not in pahchem (could add kinetic energy per H2 from the PAH channels) |
| H2 FUV pumping (vibrational) heating | dominates at the edge for n > n_cr | Sternberg & Dalgarno 89; Burton+90; Röllig+06 | — |
| photodissociation heating | about 0.5 eV per dissociated H2 | Stephens & Dalgarno 73; Abgrall+00 | — |
| CR and X-ray heating | Q ≈ 10 eV (diffuse) to 13 eV (molecular) per ion pair | Glassgold+12 | — |
| gas–grain collisional heating and cooling | T_gr from RT or a fit | Hollenbach+91; Hocuk+17; Röllig+13 | — |
| IR-pumped fine-structure heating | [O I] 63 µm | — | — |
| line cooling | [C II] 158, [O I] 63/145, [C I] 370/609, CO, [Si II] 35, H2; escape probability (microturbulent or LVG); LAMBDA/BASECOL rates; o/p H2 | TH85; Stoerzer+96; Goldsmith+12; van der Tak+20; Lique+18; Wiesenfeld & Goldsmith 14 | — |

**Where W22 places the main uncertainties**
- **H2 formation** at high T and T_gr (a factor of a few; PDRs require 1–5× the diffuse rate).
- **Ion recombination on grains/PAHs** (Φ_PAH calibration, size integrals, lab data).
- **The PE heating efficiency** (×2–4 between BT94, WD01 and Meudon at high γ; [C II]/FIR below the theory).
- **Gas temperatures between codes:** ×2–3 at AV ≈ 1, and ×4–10 for unconstrained microphysics, from the Röllig+07 code comparison.
- **The cosmic-ray ionisation rate** and its depth dependence.
- **Endothermic CH⁺/SH⁺ chemistry:** turbulent dissipation or warm H2 is needed.

---------------------------------------------------------------------------------------------------

## 11. The user's paper: Thi et al. 2020, A&A 634, A42 (2020A&A...634A..42T)

Thi, Hocuk, Kamp, Woitke, Rab, Cazaux & Caselli, "Warm dust surface chemistry. H2 and HD formation". doi:10.1051/0004-6361/201731746; arXiv:1812.06730 [read in full: main text and Appendices A–F].

**Code.** ProDiMo (PROtoplanetary DIsk MOdel). Tests are 0D clouds (n = 2e4, T_d = T_g = 10–700 K, AV = 10, ζ = 1.7e-17, r = 0.1 µm, x_PAH = 3e-7) plus disks. The PAH charge network is shared with Thi+19, A&A 632, A44 (arXiv:1811.08663).

### PAH charge chemistry (App. B)

**The molecule**
- A single circumcoronene C54H18, charges −1 to +3, with hydrogenation states PAH-H_x (x = 0–18), neutral and cationic, plus D analogues.
- a = 10 Å (N_C/468)^⅓ = 4.69 Å.
- IP from the WD01 formula: 6.2, 9.4, 12.5 eV (measured 5.9, 8.8, 12.9). EA = 1.3 eV.

**Photoionisation and detachment**
- Li & Draine cross sections × the Jochims+96 yield, in the local RT field, with PAH self-shielding.
- CR- and X-ray-induced UV (H2 Rydberg fluorescence): k = 0.15 ζ n_H g_PAH y (f_Ryd = 0.15, FPdF03).

**e + PAH⁺**
- 4.1e-5 φ f(a) (N_C/50)^½ (100/T)^½.
- f(a) = (1 − e^(−12a/l_e))/(1 + e^(12 − N_C)), a correction for N_C < 20.
- φ = σ_disc/σ_sphere, between 0.1 and 0.8. φ = 0.2 fits Biennier+06 and FPdF03; C54 adopts φ = 0.4, i.e. 1.7e-5 (100/T)^½.
- Multiply charged: ×(1 + W_PAH/kT), with W_PAH = IP(Z) − IP(0).
- Recombination of PAH-H⁺ is dissociative (PAH + H). Recombination is dissociative for N_C ≤ 10.

**e + PAH0 attachment**
- k = S · 8.5e-7 φ (N_C/50)^½, with S the WD01 form without the 0.5: (1 − e^(−a/l_e))/(1 + e^(20 − N_C)). T-independent.
- They note it is about 10× below WH08/Omont at N_C = 30, and consistent with anthracene at 1e-9 (Canosa+94; Moustefaoui+98).
- Inconsistency: eq. B.16 writes √N_C while eq. B.17 writes (N_C/50)^½.

**Mutual neutralisation (B.19)**
- 2.9e-7 φ (12 amu/m_X)^½ (100/T)^½ (N_C/50)^½ for all atomic and molecular ions, with φ = 0.2 in their Fig. B.4.
- The Hickman alternative is quoted.
- AH⁺: 50 % A + H + PAH, 50 % AH + PAH.

**Charge exchange X⁺ + PAH^n⁺ (B.26–B.29)**
- Allowed if IP(X) > IP(PAH^n⁺): 2.9e-8 φ (1 amu/m_X)^½ (100/T)^½ (N_C/50)^½ (Tielens 2005). FPdF03 used probability 0.1 per collision.
- For n > 0: × max(0, 1 − n e²/(a kT)) or max(0, 1 − W/kT), which is zero for C54 at disk temperatures.
- He⁺ + PAH → He + PAH²⁺ + e: 1.1e-8 φ (N_C/50)^½.
- Only Na can neutralise PAH⁺ (X + PAH⁺ → X⁺ + PAH).

**Other channels**
- Collisional detachment: 1.5e-8 φ (1 amu/m_X)^½ (N_C/50)^½ exp(−5500/T).
- PAH freeze-out: E_des/k = 482 (N_C − N_H) + 946 N_H K (34 380 K for C54H18).

### H2 on PAHs (§2.6–2.7, App. C)

- **Neutral hydrogenation:** 2.78e-11 (T/300)^½ N_C Q_Bell(E_act). The cross section is 1.1 Å² per reactive C (Boschman+15). Barriers are Boschman's alternating sequence (Table 1): first H 324 K, then 0, 533, 0, 742, … K.
- **Cation hydrogenation:** 2e-10 (T/300)^−3/2 Q_Bell(116 K) (Cazaux+16), size-independent.
- **ER abstraction**
  - Neutrals: 1.5e-12 (T/300)^½ x (0.06 Å² per aliphatic C–H bond, Mennella+12, applied by Thi+20 per excess H), barrier neglected.
  - Cations: 1.4e-10 (N_H/12)(N_C/24)^−1 (Montillaud+13).
- **Dehydrogenation**
  - UV RRK: T_e ≈ 2000 (hν/N_C)^0.4 (1 − 0.2 E0/hν), R = (kT_e/h) e^(1 + ΔS/R) exp(−E0/kT_e), ΔS = 55.6 J K⁻¹ mol⁻¹, against IR relaxation at 1 s⁻¹. E0 ≈ 1.4 eV for the extra H.
  - Thermal unimolecular dissociation at T_PAH = T_d.
- **H2 dissociative chemisorption on PAH edges:** barrier 3481 K.
- **Not included:** photolytic H2 (Castellanos+18) and physisorbed H on PAHs.
- **Results**
  - The PAH route is efficient above 80 K but small next to silicate chemisorption with σ = 0.06 Å².
  - With 6 Å² it dominates above about 200 K.
  - Maximum R ≈ 1e-16 f_PAH (σ/10 Å²)(T/100)^½ (x̄/10) s⁻¹.

### H2 on grains (§2.1–2.5, 3)

- Rate equations, LH and ER, on physisorbed sites (E_b = 600 K, E_diff = 408 K, tunnelling width 3.9 Å) and chemisorbed sites (E_b = 35 T_d, between 1e4 and 2.5e4 K; E_act = E_b − 9100 K, ≥ 900 K; Bell tunnelling width 0.5 Å).
- Diffusion-mediated chemisorption through ice mantles.
- Sticking: S_phys from HM79; S_chem = (1 + 5e-2 (T_g + T_d)^½ + 1e-14 T_g⁴)^−1.
- Reduces to Cazaux & Tielens: R = ½ ε 4πr² v_th S n_d. For 0.1 µm grains and ε = 1 this gives k_H2 = 3.74e-17 at 80 K (Jura 3e-17; Gry+02 3.1–4.5e-17).
- H2 forms efficiently from 10 to about 700 K once chemisorption is included.

### Comparison with W08, W22 and pahchem

| item | Thi+20 | W08 / W22 | pahchem now | recommendation |
|---|---|---|---|---|
| PAH | one C54, −1…+3, x = 0…18 | one N_C = 35, −1/0/+1 | PAHdb library, per-molecule IP/EA | keep library |
| e + PAH⁺ | 4.1e-5 φ (N_C/50)^½ T2^−½, φ = 0.4 (C54) → 1.7e-5 T2^−½ | 3.5e-5 Φ T2^−½ (N_C = 35), Φ = 0.4–0.5 | DS87 × 0.82 (2.86e-5 at N_C = 35) | same family; Thi's (N_C/50)^½ scaling is W08 rescaled (2.9e-7 = 2.4e-7 (50/35)^½) |
| neutral rates' N_C scaling | (N_C/50)^½ for CT and attachment | DS87 ∝ N_C^¾ | DS87 | Thi's ½ power is wrong for neutral targets (should be ¾ in DS87, ½ for Langevin) |
| charge exchange T dependence | (100/T)^½ | none | none | none (Langevin is T-independent); Thi inherits W03 eq. C7's slip |
| attachment | WD01 S × 8.5e-7 | 1.3e-6 Φ | statistical | statistical; validate on anthracene and C60 |
| mutual neutralisation | φ = 0.2 × W08-scaled | Φ × DS87 | none | P_MN = 0.58 (§4.1) |
| ion CT on PAH⁺ | Coulomb-suppressed to zero | not included | none | zero below 1e4 K |
| He⁺ double CT | 1.1e-8 φ | — | none | network-only option |
| CR-induced UV ionisation | 0.15 ζ n_H g y | — | none (`cosmic_rays.py` exists) | adopt as an option in dark gas |
| H2 on PAHs | addition (Boschman barriers), ER 0.06 Å², cation abstraction (Montillaud) | not in W08; W22 qualitative | ER 0.06 Å² once per molecule, photolytic H2 loss | per-excess-H ER, charge-dependent σ, add super-H photolysis (§6.3) |
| H2 on grains | full LH/ER with chemisorption | 3e-17 standard (W08 fits R) | pyCALIMA CS04 | keep; consider Le Bourlot ER; check the 1.77 factor |

**What Thi+20 adds**
- The most complete published PAH charge and hydrogenation network coupled into a full gas network. It includes every ion through mutual neutralisation and charge transfer, multiply charged states, CR-induced photo-processes, collisional detachment, and H/D isotopologues.
- A clean link between the hydrogenation state, the ER cross section and the H2 rate, with the theoretical maximum.

**What to adopt**
- The process list and product branching: mutual neutralisation for all ions with a 50 % dissociative channel for molecular ions; CT only when IP(X) > IP(PAH); He⁺ double ionisation; CR-UV.
- The R_max formula as a validation bound.

**What not to adopt**
- Their φ factors: physical P_MN and P_CT replace them.
- The (N_C/50)^½ scaling and the T^−½ on charge transfer.
- Their low attachment formula: pahchem's statistical sticking is better founded.

---------------------------------------------------------------------------------------------------

## 12. Tielens textbooks (contents paywalled; tables of contents from the Cambridge front matter [read])

**Tielens 2005, *The Physics and Chemistry of the Interstellar Medium***, CUP, ISBN 9780521826341, doi:10.1017/CBO9780511819827
- Ch. 3.3 photoelectric heating, pp. 66–71.
- Ch. 6.3 PAH charge, pp. 190–198.
- Ch. 6.4 photochemistry of PAHs, pp. 198–209.
- Ch. 8.2 physical processes in atomic gas, pp. 266–271.
- Ch. 8.7 chemistry of diffuse clouds, pp. 287–299.
- Ch. 9.2–9.3 PDR ionisation and energy balance, pp. 319–324.
- Ch. 10.2 degree of ionisation in molecular clouds, pp. 349–351.

**Tielens 2021, *Molecular Astrophysics***, CUP, ISBN 9781107169289, doi:10.1017/9781316718490
- §6.8 dissociative electron recombination, p. 194.
- §6.10 electron attachment, p. 197.
- §6.11 associative detachment, p. 200.
- §8.6 photoionisation of PAHs, pp. 298–302.
- §8.7 gas-phase chemical processes of PAHs, pp. 302–308.
- §9.2 H2 formation, p. 325.
- §9.3 diffuse-cloud chemistry, pp. 329–338.
- §10.6 gas–grain interactions, pp. 418–439.
- §11.6 PDRs, pp. 509–532.
- §12.8 PAHs and photoelectric heating, pp. 615–620.

**Tielens 2021, checked against the book (chapters 6–12, supplied and read 2026-10-21)**
- Eq. 8.106 (§8.7.1, p. 302), k_rec = 1.3e-6 N_C^½ (300/T)^½: confirmed. It is eq. 6.103, 1.8e-6 (300/T)^½ (r_min/Å), with r_min = 0.9√N_C Å and an ad hoc 0.8 planarity factor. That is 0.40 × DS87 for a sphere with s = 1, and 0.46 × pahchem's capture (not "½ of DS87").
- Eq. 8.107 (§8.7.2, p. 302), attachment: printed as "1.3e-7 N_C". It is the Langevin rate with α ≃ 1.6e-24 N_C cm³, which gives 1.27e-7 N_C^½, so the printed linear N_C is a typo (×7 at N_C = 50). There is no s_e: sticking is 1 for N_C > 30 (EA > 1 eV).
- Eq. 8.112 (p. 305), mutual neutralisation: k ≃ 3e-8 N_C^½ (300/T)^½ (amu/μ)^½, "by lack of better". That is 0.46 × DS87 in pahchem's geometry (`check_pah_ion_rates.py` §B).
- Eq. 8.109: k_L = 6e-9 √(N_C/50) √(12 amu/m_X), with α = 1.5 N_C Å³ (1.6 in §8.7.2). Table 6.5 gives circumcoronene α = 125 Å³ (2.31 per C), with α → N_C^{3/2} for large PAHs.
- Eq. 12.26 (§12.8.2, p. 618), ε = 0.06/(1 + 7e-5 γ): confirmed. It follows from ε = Y f(0) E_kin/hν (eqs. 12.22–12.25), with Y = 0.3 at 10 eV, IP = 6 eV, E_kin = ½(hν − IP), two charge states and N_C = 50. The 7e-5 rests on eq. 8.106; with pahchem's capture it would be ≈ 3.2e-5.
- Eq. 8.98: k_ion ≃ 2.8e-12 (13.6 − IP)² N_C G0 s⁻¹ (7.9e-9 for C₅₀).
- Eq. 9.9: R = ½ S η n_d σ_d n(H) v̄_H with v̄_H = 1.5e4 T^½ (mean speed), which confirms the 1.77× flag on pyCALIMA's grain H₂ rate.
- H₂ on PAHs: no rate coefficient is given in chapters 7, 9, 11 or 12. The "~2e-17 (T/100)^½" stays [sec, unverified].
- Internal inconsistencies of the book: Y = 0.3 at 10 eV against its own Y = (hν − IP)/9 = 0.44; f(0) = 0.85 against 0.92 from eq. 12.25; work function 4.2 against 4.4 eV; coronene⁺ IP 10.58 against 11.4 eV (Table 8.3). Several section cross-references are wrong, so cite by equation number.
- Full notes: `calima-pdr/docs/literature/pah_gas/tielens2021_notes.md`.

**Still known only from secondary sources [sec]**
- Tielens 2005: k_att = 1.3e-7 s_e N_C^½.
- Tielens 2005: charge transfer 2.9e-8 Φ (1 amu/m_X)^½ (100/T)^½ (N_C/50)^½; He⁺ double ionisation 1.1e-8 Φ (N_C/50)^½ (via Thi+20). Tielens 2021 §8.7.3 discusses the He⁺ channel only qualitatively.

The user should supply Tielens 2005 §3.3, §6.3 and §8.7 to confirm these.

The free lecture-note route was not pursued. A third-party PDF of the 2005 book appeared in search results; it was not downloaded.

---------------------------------------------------------------------------------------------------

## 13. Paywalled or unread items for the user to supply

**Supplied by the user and read (2026-10-21):** Bakes & Tielens 1998; Bréchignac+14; Canosa+94; Canosa+95; Moustefaoui+98; Wolfire+22 (journal version); Lepp & Dalgarno 1988 (ApJ 329, 418; ApJ 324, 553; ApJ 335, 769); Rebrion-Rowe+03; Flower & Pineau des Forêts 2026 (MNRAS 552, 1); Liszt 2011; Burgh+10; Hickman 1979; Hamberg+11; Abouelaziz+93; Biennier+06; Andrews+16; Mennella+12; Thrower+12.

**Requests from that reading:** Lepp & Dalgarno 1988a (ApJ 335, 769) and Rebrion-Rowe et al. 2003 (Int. J. Mass Spectrom. 223–224, 237) were supplied on 2026-10-22, with Lepp & Dalgarno 1988b (ApJ 324, 553). Hassouna et al. 2003 (Kluwer conference proceedings), the source of the phenanthrene row of `pah_electron_recombination.csv`, is not obtainable; that row stays as quoted by Biennier+06. Local copies: `~/Documents/GitHub/pdr_refs/literature/pah_gas/supplied/`. The Tielens 2021 book was supplied as a copy outside OneDrive.

**Rate theory and models**
- Draine B. T., Sutin B. 1987, ApJ 320, 803: read by a sub-agent from the ADS scan. A clean copy is advisable.
- Weingartner J. C., Draine B. T., Barr D. K. 2006, ApJ 645, 1188: EUV and X-ray photoemission.
- Millar T. J., Walsh C., Field T. A. 2017, Chem. Rev. 117, 1765: interstellar anions.
- Tielens 2005 and 2021 chapters (§12 above).

**Laboratory papers**
- Rebrion-Rowe C. et al. 2003, Int. J. Mass Spectrom. 223–224, 237 (supplied and read 2026-10-22).
- Hassouna M. et al. 2003, in *Dissociative Recombination of Molecular Ions with Electrons* (Kluwer), pp. 49–57.
- Rowe B. R. et al. 1995 (book chapter).
- Adams N. G., Babcock L. M. 2005, J. Phys. Conf. Ser. 4, 38.
- Novotny O. et al. 2005, J. Phys. Conf. Ser. 4, 198. The IOP bot wall blocked it.
- McLain J. L. et al. 2004, J. Phys. Chem. A 108, 6704.
- Tobita S. et al. 1992, Chem. Phys. 161, 501.
- Carelli F., Gianturco F. A. 2012, MNRAS 422, 3643; Sanz A. G. et al. 2013, New J. Phys. 15, 013018.
- Rienstra-Kiracofe J. C. et al. 2002, Chem. Rev. 102, 231 (electron affinities).
- Milligan D. B. et al. 2002 (proton transfer to aromatics); Midey A. J. et al. 2000 (CT with naphthalene).

**H2 formation** (full list in h2_formation.md §E)
- Rauls E., Hornekær L. 2008, ApJ 679, 531.
- Skov A. L. et al. 2014, Faraday Discuss. 168, 223.
- Le Page V. et al. 2009, ApJ 704, 274.
- Snow T. P. et al. 1998, Nature 391, 259.
- Cazaux S., Tielens A. G. G. M. 2004, ApJ 604, 222, and the 2010 erratum, ApJ 715, 698.
- Grieco F. et al. 2023, Nat. Astron. 7, 541.
- Habart E. et al. 2003, A&A 397, 623.
- Joblin C. et al. 2018, A&A 615, A129.
- Jensen P. A. et al. 2019; Bauschlicher C. W. 1998, ApJ 509, L125; Hirama M. et al. 2004. These citations are from memory and should be verified.

---------------------------------------------------------------------------------------------------

## 14. Bibliography (key items)

Primary set:

| reference | identifiers | status |
|---|---|---|
| Wolfire M. G., Tielens A. G. G. M., Hollenbach D., Kaufman M. J. 2008, ApJ 680, 384 | 2008ApJ...680..384W; doi:10.1086/587688; arXiv:0803.0138 | [read] |
| Wolfire M. G., McKee C. F., Hollenbach D., Tielens A. G. G. M. 2003, ApJ 587, 278 | 2003ApJ...587..278W; astro-ph/0207098 | [read §5, App. C] |
| Wolfire M. G., Vallini L., Chevance M. 2022, ARA&A 60, 247 | 2022ARA&A..60..247W; arXiv:2202.05867 | [read] |
| Bakes E. L. O., Tielens A. G. G. M. 1994, ApJ 427, 822 | 1994ApJ...427..822B | [read, ADS scan, pp. 823–829, 835–836] |
| Bakes E. L. O., Tielens A. G. G. M. 1998, ApJ 499, 258 | 1998ApJ...499..258B | [read, supplied] |
| Bréchignac P. et al. 2014, J. Chem. Phys. 141, 164325 | doi:10.1063/1.4900427 | [read, supplied; supplementary material not seen] |
| Canosa A. et al. 1995, Chem. Phys. Lett. 245, 407 | ScienceDirect PII 000926149501040G | [read, supplied] |
| Moustefaoui T. et al. 1998, Faraday Discuss. 109, 71 | doi:10.1039/a800490k | [read, supplied] |
| Draine B. T., Sutin B. 1987, ApJ 320, 803 | 1987ApJ...320..803D | [read by sub-agent, ADS scan] |
| Weingartner J. C., Draine B. T. 2001, ApJS 134, 263 | 2001ApJS..134..263W; astro-ph/9907251 | [read relevant sections] |
| Weingartner J. C., Draine B. T. 2001, ApJ 563, 842 | 2001ApJ...563..842W; astro-ph/0105237 | [read] |
| Verstraete L., Léger A., d'Hendecourt L., Dutuit O., Défourneau D. 1990, A&A 237, 436 | 1990A&A...237..436V | [read, ADS scan] |
| Berné O., Foschino S., Jalabert F., Joblin C. 2022, A&A 667, A159 | arXiv:2208.08762 | [read §3–4, App. C] |
| Hollenbach D., Kaufman M. J., Neufeld D., Wolfire M., Goicoechea J. R. 2012, ApJ 754, 105 | arXiv:1205.6446 | [read PAH parts] |
| Thi W.-F., Hocuk S., Kamp I., Woitke P., Rab Ch., Cazaux S., Caselli P. 2020, A&A 634, A42 | 2020A&A...634A..42T; doi:10.1051/0004-6361/201731746; arXiv:1812.06730 | [read] |
| Thi W.-F. et al. 2019, A&A 632, A44 | arXiv:1811.08663 | [read §2.3, App. C by sub-agent] |
| Flower D. R., Pineau des Forêts G. 2003, MNRAS 343, 390 | doi:10.1046/j.1365-8711.2003.06716.x | [read text by sub-agent] |
| Wakelam V., Herbst E. 2008, ApJ 680, 371 | 2008ApJ...680..371W | [read by sub-agent] |
| Lepp S., Dalgarno A. 1988, ApJ 324, 553 | 1988ApJ...324..553L | [read OCR by sub-agent] |
| Omont A. 1986, A&A 164, 159 | 1986A&A...164..159O | [read scan by sub-agent] |
| Gong M., Ostriker E. C., Wolfire M. G. 2017, ApJ 843, 38 | arXiv:1610.09023 | [read by sub-agent] |
| Kim J.-G., Gong M., Kim C.-G., Ostriker E. C. 2023, ApJS 264, 10 | arXiv:2210.08024 | [read by sub-agent] |
| Montillaud J., Joblin C., Toublanc D. 2013, A&A 552, A15 | arXiv:1301.6507 | [read by sub-agent] |
| Novotny O. et al. 2005, J. Chem. Phys. 123, 104303 | doi:10.1063/1.2000927 | [read by sub-agent] |
| Wiens J. P., Shuman N. S., Viggiano A. A. 2015, J. Chem. Phys. 142, 114304 | doi:10.1063/1.4913829 | [read] |
| Shuman N. S. et al. 2014, J. Chem. Phys. 140, 224309 | doi:10.1063/1.4879780 | [read partial] |
| Miller T. M. et al. 2012, J. Chem. Phys. 136, 204306 | doi:10.1063/1.4720499 | [abs] |
| Gatchell M. et al. 2025, A&A 693, A43 | doi:10.1051/0004-6361/202452303; arXiv:2409.11851 | [read] |
| Fournier J. A. et al. 2013 | — | [read by sub-agent] |
| Viggiano A. A. et al. 2010 | — | [read by sub-agent] (C60 attachment) |
| Carelli F., Grassi T., Gianturco F. A. 2013, A&A 549, A103 | — | [read by sub-agent] |
| Demarais N. J. et al. 2012, ApJ 746, 32 | — | [read by sub-agent] |
| Demarais N. J. et al. 2014 | — | [read via thesis by sub-agent] |
| Bierbaum V. M. et al. 2011 (EAS 46) | — | [read by sub-agent] |
| Le Page V., Snow T. P., Bierbaum V. M. 2001, ApJS 132, 233; Le Page et al. 2003 | — | [read by sub-agent] |
| Woitke P. et al. 2011 | arXiv:1103.5309 | [read by sub-agent] |
| Röllig M. et al. 2013, A&A 549, A85 | arXiv:1211.3546 | [read by sub-agent] |
| Godard B. et al. 2014, A&A 570, A27 | arXiv:1408.3716 | [read by sub-agent] |
| Goicoechea J. R. et al. 2009 | arXiv:0902.2748 | [read by sub-agent] |
| Bisbas T. et al. 2012 | arXiv:1209.1091 | [read by sub-agent] |
| Balashev S., Kosenko D. 2024 | arXiv:2402.00714 | [read by sub-agent] |
| Khatri et al. 2024 (HYACINTH) | arXiv:2402.11023 | [read by sub-agent] |
| Hu C.-Y., Sternberg A., van Dishoeck E. F. 2021 | arXiv:2103.03889 | [read by sub-agent] |
| Bialy S., Sternberg A. 2019 | arXiv:1902.06764 | [read by sub-agent] |
| Vermariën et al. 2026 (UCLCHEM 4.0) | arXiv:2606.20265 | [read by sub-agent] |
| Boschman L. et al. 2012, ApJL 761, L33; Boschman L. et al. 2015, A&A 579, A72 | — | [read by sub-agent] |
| Castellanos P. et al. 2018b | — | [read by sub-agent] |
| Habart E. et al. 2004, A&A 414, 531 | — | [read by sub-agent] |
| Le Bourlot J. et al. 2012, A&A 541, A76 | — | [read by sub-agent] |
| Bron E. et al. 2014, A&A 569, A100 | — | [read by sub-agent] |
| Wakelam V. et al. 2017, Mol. Astrophys. 9, 1 | — | [read by sub-agent] |
| Foley et al. 2018 | — | [read by sub-agent] |

The remaining entries, with arXiv ids, are in `citations.md`, `labdata.md` and `h2_formation.md`.
