# Sublimation kinetics compilation for pyCALIMA: notes

> In pyCALIMA: the CSV is `compilation.csv` in this directory, the scripts
> referred to below as `scripts/` are in
> `diagnostics/dust_radiation/sublimation_compilation_provenance/`, and the
> raw research notes (verbatim quotes) and paper text dumps stayed in the
> working directory where the compilation was made (they are not bundled).

Files in this directory:
- `sublimation_parameters.csv`: 107 rows, one per material and source. UTF-8, comma separated, with a header row.
- `scripts/`: the Python used for every conversion and derived fit (`conv.py`, `freeevap.py`, `runfits.py`, `build.py`), plus the intermediate JSON.
  - `freeevap.py` and `parse_w18.py` read the pdftotext dump of arXiv:1712.01010 (Woitke et al. 2018). That dump was made with `pdftotext -layout` and sits outside this directory, in the session scratchpad under `dl/1712.01010.txt`.
- Raw research notes are in the parent scratchpad, not in this directory: `raw_silicates.md`, `raw_metals_sulfides.md`, `raw_carbon_oxides.md`. They contain verbatim quotes and how each number was verified.
  - The silicate, metal/sulfide/salt/ice and carbon/oxide literature searches were run as three parallel sub-searches. Their notes are merged here.

## 1. Conventions and CSV columns

- `A10`, `B10_K`: the common convention **log10 P_sat[dyn cm^-2] = A10 − B10/T**, with T in K.
- `Bk_K` = B10 · ln10. This is the Arrhenius temperature that appears in exp(−B/kT), i.e. GD89's B/k.
- `mu_unit_amu`: mass of the evaporating unit used in the Hertz–Knudsen–Langmuir (HKL) flux.
  - For "effective" rows (`W18-*+FE`, `N03+conv`) it is the mass of one formula unit (f.u.).
  - For partial-pressure rows it is the mass of that gas molecule (C2, C3, S2, …).
  - For van Lieshout rows it is the value the source used (the formula mass).
- `P0_cm2s_at_1500K` = 10^A10 / sqrt(2π m k T) evaluated at T = 1500 K, in units of [units of the chosen mass] cm^-2 s^-1.
  - It scales as T^-1/2. For the GD89 rows it is the source's T-independent prefactor.
- `nu0_s_at_1500K_alpha1` = (m/ρ)^(2/3) · P0, with α = 1. This is the attempt frequency in the GD89 / Waxman & Draine (2000) form da/dt = −(m/ρ)^(1/3) ν0 exp(−B/kT).
  - Multiply by α to include the evaporation coefficient.
- `alpha`: the evaporation coefficient, defined by J = α P_sat / sqrt(2π m k T), with conditions in `alpha_conditions`.
  - **α must be paired with the same reference vapour pressure it was measured against.** See SiO2 in §5.
- `verification` tags:
  - [V-txt] / [P-txt]: read in the primary PDF text.
  - [V-img] / [P-img]: read from a rendered page image of the primary source.
  - [ABS]: abstract only.
  - [HTML-x]: publisher HTML read through a summariser.
  - [2nd]: quoted by a later paper.
  - "derived": computed here from opened tables.
  - [unverified] / [doubtful]: as stated.
- Source IDs are listed in §8.

## 2. Conversions performed

Pressure-unit offsets in log10, to reach dyn cm^-2:

| From | Add |
|---|---|
| bar | +6 |
| atm | +6.00572 |
| Torr | +3.12490 |
| Pa | +1 |

Natural-log fits ln P = B − A/T become A10 = B/ln10 and B10 = A/ln10.

1. **van Lieshout et al. 2014a (vL14a)**, Table 3: p_v = exp(−A/T + B) dyn cm^-2, so A10 = B/ln10 and B10 = A/ln10.
2. **Kobayashi et al. 2011 (K11)** and **van Lieshout et al. 2014b (vL14b)**: P = P0 exp(−μ m_u H / kT), so A10 = log10 P0 and B10 = μ m_u H / (k ln10).
3. **Nozawa et al. 2003 (N03)**, Table 2: ΔG0/kT = −A/T + B per key species for the *condensation* reaction (gas → solid).
   - At equilibrium (S = 1), Σ ν_i ln P_i = ΔG0/kT per key species. Per formula unit, Π P_i^{n_i} = K = exp[ν_key (B − A/T)], where ν_key is the number of key species per f.u. (2 for Al2O3 and Mg2SiO4, 3 for Fe3O4, otherwise 1).
   - Units: N03 do not state the pressure unit. I assumed **bar** (the JANAF 1985 standard state). This is [inferred], but N03 Fe matches JANAF and CRC (1 Pa at 1728 K) only with bar.
   - To get one effective pressure I assumed congruent Langmuir evaporation into vacuum, with only the N03 reaction products and equal α for all products:
     - Stoichiometric fluxes, J_i = n_i J_fu with J_i = P_i / sqrt(2π m_i k T), give P_i = n_i sqrt(m_i) X.
     - X^N Π (n_i sqrt m_i)^{n_i} = K, with N = Σ n_i.
     - **P_eff ≡ J_fu sqrt(2π m_fu k T) = K^{1/N} sqrt(m_fu) / Π (n_i sqrt m_i)^{n_i/N}.**
   - Hence **B10 = ν_key A / (N ln10)** and **A10 = ν_key B / (N ln10) + 6 + log10[sqrt(m_fu) / Π (n_i sqrt m_i)^{n_i/N}]**.
   - Physically: when a solid decomposes into N gas particles, the flux scales as exp(−ΔH_fu / (N kT)). For Mg2SiO4 → 2Mg + SiO + 3O, ΔH_fu/k ≈ 3.7e5 K but the effective B/k is ≈ 6.2e4 K. This agrees with the measured value (Nagahara et al. 1994 via vL14a: 65,308 K).
4. **Woitke et al. 2018 (W18) "free evaporation" rows (`W18-D2+FE`, `W18-D4+FE`)**, derived here. W18 gives:
   - Table D.2: either p_vap of the *same-formula molecule* (fit 3; e.g. FeS(g), MgO(g), SiO2(g) — usually minor species!), or ΔGf from free atoms (fits 1, 2, 5).
   - Table D.4: SUPCRTBL −ΔGf/RT from atoms, at 1 bar, 100–2500 K.
   - Table D.1: kp fits for 552 molecules.

   I solved the full equilibrium vapour over each solid (atoms plus all neutral fit-4 molecules made only of the solid's elements). The constraints were:
   - Π (p_e/1 bar)^{n_e} = K_solid.
   - Element fluxes Σ_s ν_{e,s} p_s / sqrt(2π m_s k T) are in stoichiometric ratio (congruent, vacuum, equal α).

   I then least-squares fitted log10 P_eff = A10 − B10/T on a grid (T range and maximum deviation are in the CSV; all deviations < 0.06 dex).
   - Checks: monatomic metals reproduce W18/JANAF p_vap exactly; SiO reproduces Gail et al. 2013; graphite agrees with vL14a C3 within the scatter.
   - The vapour composition at mid-range T is listed in `vapour_species`. Examples:
     - Mg2SiO4: Mg 0.38, SiO 0.25, O2 0.23, O 0.13.
     - SiO2: SiO 0.68, O2 0.22.
     - FeS: Fe 0.65, S2 0.34.
     - Al2O3: O 0.52, Al 0.44.
     - TiO2: TiO2 0.81, TiO 0.12.
   - D.2 and D.4 versions give P_eff agreeing within ≤0.2 dex over the fitted ranges.
   - Fit 1 data (Sharp & Huebner) are given in cal/mol at 1 atm and were converted to bar.
5. **JANAF rows** (fits made by the metals/sulfides sub-search): log10 P[bar] = log10 Kf(gas) − n log10 Kf(solid), fitted linearly on the JANAF 100 K grid.
6. **Graphite, Zavitsanos 1970**: log P[atm] = A − M·10^4/T, so A10 = A + 6.00572 and B10 = M·10^4.
7. **Water ice, W18 fit 7** (Ackerman & Marley 2001): linearised over 110–200 K (maximum deviation 0.044 dex).

## 3. Mapping onto the GD89 framework used in pyCALIMA

pyCALIMA uses J = α_N P0 exp(−B/kT) [monomers cm^-2 s^-1]. With HKL and log10 P_sat = A10 − B10/T:

- J = α · 10^A10 · exp(−B10 ln10 / T) / sqrt(2π m k T).
- Therefore **B/k = B10 ln10** and **P0(T) = 10^A10 / sqrt(2π m k T)**. The weak T^-1/2 dependence can be frozen at the operating T (the CSV uses 1500 K), or kept explicitly.
- GD89 avoided the T^-1/2 by writing p_sat ∝ T^1/2 e^(−B/kT).
- The "monomer" must match the m used for P0:
  - For effective rows the monomer is one formula unit. Multiply J by the number of atoms per f.u. if the code counts atoms. GD89 silicate uses R_N = 7 R_Si.
  - For partial-pressure rows (C3, S2) multiply by atoms per molecule.
- Grain-radius rate: da/dt = −(m_fu/ρ) J_fu = −(m/ρ)^(1/3) ν0 e^(−B/kT), with ν0 = (m/ρ)^(2/3) α P0.
- Small-grain correction (GD89 eq. 3.8–3.11): B(N) = B_∞ − θ_∞ N^(−1/3), with θ_∞ = (32π/3)^(1/3) (m_X/ρ)^(2/3) σ/k.
  - θ_∞ per unit computed from the N03 σ values: graphite 13,900 K; Fe 21,800; Mg2SiO4 (per f.u.) 17,700; MgSiO3 13,100; SiO2 15,900; Al2O3 19,700; FeS 8,600; SiC 31,700; MgO 18,100; TiO2 (σ = 480.6) 11,200; H2O (σ = 109) 2,600.
  - GD89 adopted θ_∞ = 2×10^4 K for both graphite and silicate. This implies σ ≈ 2000 (graphite, ρ = 2.24) and ≈ 1900 erg cm^-2 (silicate per atom). Both are derived here.
  - With the calorimetric σ for crystalline forsterite (4410 erg cm^-2), θ_∞ would be ≈ 1.8×10^5 K. That value is a room-temperature surface enthalpy, not a high-T surface free energy; do not use it blindly.

**Check against GD89 graphite** (GD89 eqs. 3.12–3.13, read from the scan):
- p_sat(C1) = 6e11 T^0.5 e^(−81200/T) dyn cm^-2 gives P0 = 6e11 / sqrt(2π m_C k) = **4.56e30** cm^-2 s^-1. This reproduces GD89's 4.6e29 at α = 0.1 exactly.
- C1 alone, against other sources:
  - W18 C-atom p_vap at 3000 K: 52 dyn cm^-2, vs GD89's 58.
  - Zavitsanos 1970 C1: 48 dyn cm^-2.
  - JANAF C1: B/k = 85,700 K.
- Total C-atom flux using Zavitsanos 1970 C1/C2/C3 pressures, compared with GD89 at α = 0.1:

  | α set used | 2000 K | 2500 K | 3000 K |
  |---|---|---|---|
  | Thorn & Winslow species α (0.37 / 0.34 / 0.08) | 2.5× | 5.3× | 10× |
  | Zavitsanos 1966 species α (0.24 / 0.50 / 0.023) | 1.4× | 2.8× | 5.5× |

  - W18 free evaporation with equal α: 1.5× GD89 at 2000 K.
  - **Conclusion: GD89's graphite rate is reproduced within a factor 1.4–3 at 2000–2500 K, but it is increasingly too low at ≥3000 K.** GD89 counted only C1 (B/k = 81,200 K), while the C2/C3 channels have B/k ≈ 92,000–103,000 K.
  - For T ≲ 2500 K (AGN dust), GD89's (P0, B) is adequate within a factor of a few.
  - Fitting all channels gives B/k ≈ 92,400 K and P0 ≈ 2e33 (α = 1, per C atom). Equivalently, α_eff ≈ 0.1–0.2 with the C3-weighted B.

**Check against GD89 silicate** (GD89 eq. 3.17: R_Si = 7e30 ⟨α⟩ e^(−68100/T), per Si, i.e. per Mg2SiO4 f.u.):
- Comparison of free-evaporation fluxes per f.u. at α = 1:

  | Source | 1800 K | 2500 K |
  |---|---|---|
  | GD89 | 2.6e14 | 1.0e19 |
  | W18 forsterite | 4.5e15 | 5.9e19 |
  | N03 | 2.5e15 | 3.3e19 |
  | Nagahara 1994 lab (vL14a) | 5.9e15 | 1.3e20 |

- That is, GD89 is **lower by ×7–40 at 1500–2000 K and ×3–12 at 2500 K**.
- The B values differ: B/k(free evaporation) ≈ 62,000 K (W18, N03) to 65,300 K (lab), vs GD89's 68,100 K.
- Why: GD89 computed the vapour assuming **cosmic Mg:Si:O abundances**. That vapour is very O-rich and, by mass action, suppresses p(Mg) and p(SiO). It is not Langmuir evaporation into vacuum.
- GD89's P0 = 7e30 per f.u. is reproduced to within a factor ~2 by the free-evaporation P0 (2.6e30 for N03, 4.3e30 for W18). The rate difference is almost entirely in B.
- Waxman & Draine 2000 quote ν0 ≈ 2e15 s^-1 (Mg2SiO4) and 2e14 s^-1 (graphite). With α = 0.1 I recover 2.2e15 and 2.0e14, so their ν0 values include α = 0.1.
- **Recommendation**: for a vacuum or low-density AGN environment use the forsterite free-evaporation numbers (B/k ≈ 62,000–65,000 K, P0 ≈ 4e30–4e31 per f.u.) with α ≈ 0.1. GD89's numbers are a more conservative (slower) option.

## 4. (i) Which materials have laboratory kinetics?

- **Solid (both P_sat and α measured, in vacuum):**
  - Forsterite: P_sat from Nagahara et al. 1994 (Knudsen) [2nd via vL14a, Costa et al. 2017]. α = 0.09–0.14 (Hashimoto 1989) and ≈ 0.1 within ×3 over 1200–1800 °C, in vacuum and H2 (Tsuchiyama et al. 1998, compiling seven datasets). Crystal-face anisotropy: Vc > Va ≥ Vb.
  - Metallic Fe: α(T) from Tachibana et al. 2011; P_sat from JANAF, Ferguson et al. 2004 and Smithells.
  - Corundum: α = 0.02–0.2 with face dependence (Takigawa et al. 2015); P_sat from JANAF.
  - SiO2 (cristobalite): α = 0.011–0.015 against total vapour (Hashimoto 1989).
  - Solid SiO: P_sat and α(T) (Gail et al. 2013).
  - Graphite: species pressures (Zavitsanos 1970; Thorn & Winslow 1957) and species α values (which disagree by ×2–4).
  - Water ice: P_sat (Murphy & Koop 2005), α(T) (Gundlach et al. 2011).
  - Cr, NaCl, KCl: α from Pound 1972 compilation; P_sat from JANAF.
- **Partial (P_sat or rate measured, α missing or only adopted):**
  - Enstatite: rate constants and incongruency measured; α not found.
  - Fayalite: P_sat from Nagahara et al. 1994 via vL14a; α adopted at 0.1.
  - Fe-bearing olivine: Costa et al. 2017 KEMS partial pressures; Nagahara & Ozawa 1998 rates.
  - FeS: α_S2, α_H2S and rates measured in H2; vacuum only qualitative.
  - MgO: α = 0.12–0.22 measured; P_sat from thermochemistry.
  - SiC: incongruency known; vL14a gives a P_sat (Lilov 1993); α adopted.
- **Thermodynamics only (no laboratory α found):** amorphous carbon (not even a separate P_sat); TiC; MgAl2O4; CaTiO3; TiO2; FeO; Fe3O4; Fe2O3; FeSiO3; intermediate Mg–Fe pyroxenes and olivines; amorphous silicates; Si; MnS; ZnS; Na2S.
  - For these use the W18-derived free-evaporation rows with α ≈ 0.1. This is adopted, not measured; measured α values for oxides span about 0.01–0.2.

## 5. (ii) Evidence for incongruent Fe release (and other incongruency)

- **Fe-bearing olivine**:
  - Nagahara & Ozawa 1998 (LPSC 1641) evaporated San Carlos olivine (≈Fo90) in vacuum at 1400–1500 °C. The residue stays stoichiometric olivine but becomes more magnesian (FeO and NiO depleted, MgO and CaO enriched). Mg–Fe fractionation is limited by solid-state diffusion (fractionation factor K < 0.06–0.13, face dependent). Rates are about 5× faster than pure forsterite.
  - Their model has Fo50 grains of 1, 10 and 100 µm becoming pure forsterite in 0.2, 2 and 20 days at 1400 °C.
  - Costa, Jacobson & Fegley 2017 (Knudsen-effusion mass spectrometry, Fo95Fa5, 1750–2250 K) found Fe to be the most abundant gas (Fe > SiO > Mg > O2 > O) and (Fe/Mg)gas/(Fe/Mg)olivine > 1. The sample went from Fo95 to Fo96. Their fits are in the CSV.
  - Older work (Nagahara et al. 1988 Nature; Ozawa & Nagahara 2000 GCA) showed fractional vaporisation [2nd, quoted by Costa et al. 2017].
  - Fayalite's vapour pressure is far higher than forsterite's (triple points 6.3e-8 bar at 1490 K vs 5.2e-5 bar at 2163 K; Nagahara et al. 1994 [2nd]). In the W18-derived rows, P_eff(Fa) exceeds P_eff(Fo) by ~1.4 dex at 1400 K.
  - **Implication**: Fe-bearing silicate grains near the sublimation front should lose Fe first, evolving toward forsterite. The rate is limited by Mg–Fe interdiffusion; small grains (≲1 µm) homogenise quickly.
  - A minimal model: treat the Fe component with its own P_sat (fayalite or FeO activity) until depleted, then forsterite kinetics.
  - Whether Fe leaves as gas or as metallic Fe inclusions (reduction in H2) depends on the environment. In H2, olivine reduction to Fe metal is known, but I did not open quantitative sources on it.
- **Enstatite**: incongruent in both vacuum and H2. The SiO2 component is lost preferentially, leaving a polycrystalline forsterite layer (Tachibana et al. 2002; Tachibana & Tsuchiyama 1999). The equilibrium gas has Mg/Si ≈ 0.05. The Fo layer stays ≤0.1 µm in nebular conditions, so ≲1 µm grains can change Mg/Si strongly.
  - The same is expected for Fe-bearing pyroxenes, but I found no data.
- **FeS (troilite/pyrrhotite)**: strongly incongruent. S leaves (as H2S in H2, as S2 in vacuum or low pH2), leaving a porous metallic Fe residue (Tsuchiyama et al. 1997; Tachibana & Tsuchiyama 1998).
  - Thermodynamically, congruent FeS evaporation in vacuum would require p(Fe) above P_sat of Fe metal (~1 dex at 1200 K, from JANAF), so Fe metal must precipitate. Tsuchiyama et al. 1997 state the same.
  - **Recommended model**: S flux = 2 α_S2 q P_S2(Fe/FeS buffer) / sqrt(2π m_S2 k T), with α_S2 = 0.23 and q = 0.609 (Tachibana & Tsuchiyama 1998 via Tachibana et al. 1999), and log10 P_S2[dyn] = 12.106 − 16289/T. The residual Fe then evaporates with the Fe row and α_Fe(T).
  - S release starts at much lower T than Fe release.
- **SiC**: Si is lost and a carbon (graphene) residue remains (Yazdi et al. 2016). In H2–H2O gas SiC oxidises actively.
- **Fe2O3 and Fe3O4**: these lose O2 (Fe2O3 → Fe3O4 + O2; 1386 °C in air, lower in vacuum) before any congruent sublimation. Their W18 rows are formal only.
- **Congruent in the lab**: forsterite (vacuum and H2), corundum, cristobalite, SiO, Fe, ice.
- **Silica α caveat**: the measured cristobalite flux matches α ≈ 0.7–0.9 relative to p(SiO2(g)), or α ≈ 0.012 relative to the total equilibrium vapour (mostly SiO + O2). vL14a uses the first; Gail & Sedlmayr 1999 the second. Use either pair consistently, never "W18 SiO2 row with α = 1".

## 6. Per-material caveats and recommended rows

- **Graphite**: use Z70 species pressures with Thorn & Winslow (or Zavitsanos 1966) species α, or the W18 total with α_eff ≈ 0.1.
  - vL14b's C3 P0 is ~25× below vL14a and Z70 although both cite the same source. Treated as [inconsistent].
  - K11 "carbon" (B/k = 105,000 K) has no stated source.
  - Pound 1972 lists α = 1 (Marshall & Norton 1950, total mass loss). This conflicts with the ~0.1 found by species-resolved work.
- **Amorphous carbon**: no data. Graphite parameters are the standard proxy. Amorphous C is metastable, so its true P_sat is ≥ graphite's; the magnitude is unknown.
- **Astronomical silicate**: no "astrosilicate" sublimation data exist. Use forsterite, or MgFeSiO4 handled as in §5.
  - K11 "olivine" has μ = 169.1 and B/k = 65,285 K, essentially identical to the vL14a forsterite B. K11 "pyroxene" uses μ = 60.1, i.e. the SiO2 unit.
- **Forsterite**: lab (vL14a, from Nagahara 1994) and free-evaporation thermochemistry agree within ≤0.2 dex at 1500–2000 K. α ≈ 0.1. Recommended.
- **Enstatite**: vL14a (Mysen & Kushiro 1988; Kushiro & Mysen 1991) is ~0.6 dex above the W18-derived value at 1700 K. The real process is incongruent.
- **Fayalite**: vL14a (Nagahara 1994) is ~1.6 dex above the W18-derived value at 1400 K. This is unresolved: it may reflect a different reaction or reference basis, or a non-stoichiometric vapour. [doubtful either way]
- **Ferrosilite and intermediate compositions**: thermochemistry only (W18 Table D.4 ferrosilite). No rows with numbers for Mg0.8Fe0.2SiO3, etc.
- **Fe**: JANAF, W18, N03 and Visscher et al. 2010 agree within 0.1 dex. vL14a (Ferguson et al. 2004) is ~0.5 dex lower. Lamy 1974 (Smithells) is consistent with JANAF.
  - The **K11 iron H/P0 are wrong**: they come from a mis-transcription of the Lamy formula. Do not use.
  - α_Fe = exp(1.44 − 2595/T), 0.6–0.9 (Tachibana et al. 2011).
- **Si**: JANAF and N03 agree. No α found. The Wikipedia "SMI" solid-Si fit is 2 dex too high and was rejected.
- **FeS**: see §5. The W18 and N03 "congruent" rows are formal upper bounds only.
- **Corundum**: vL14a (Lihrmann 2008) is ~2.4 dex above JANAF-based values at 1800 K [doubtful; possibly from the Al–C–O system]. Use the W18 or N03 rows with Takigawa et al. 2015 α.
- **SiC**: vL14a (SiC2 only) vs W18 total differ by ~0.9 dex at 1800 K. The real process is incongruent.
- **TiO2**: vapour is mostly TiO2(g), so the Lee et al. 2018 fit (TiO2 molecule) and the W18 total agree within ~0.15 dex at 1600 K.
- **MgO**: decomposes to Mg + O/O2. α = 0.12–0.22 (Hashimoto 1989).
- **FeO**: nominal only (real FeO is Fe1−xO).
- **Spinel, perovskite, TiC**: thermochemistry only.
- **MnS, ZnS, Na2S**: the Morley et al. 2012 / Visscher et al. 2006 expressions include the H2S of a solar gas, so they are *not* vacuum P_sat. They are kept for cloud-style use and condensation checks.
  - Vacuum P_eff exists for MnS (W18 fit 1) and Na2S (W18 and JANAF agree within 0.03 dex once P(Na) is converted to the same per-f.u. P_eff). Nothing exists for ZnS.
- **NaCl, KCl**: the vapour contains ~15–20% dimers (W18 total includes them). α = 0.4 (NaCl) and 0.63–0.72 (KCl).
- **Water ice**: Murphy & Koop 2005 is the reference. α(T) from Gundlach et al. 2011 falls from ~1 to ~0.15 above ~200 K.
- **General**:
  - All W18 and N03 numbers are thermochemical equilibrium (JANAF 1985/1998 or SUPCRTBL) for bulk crystals.
  - They assume vacuum. In gas containing H2 or O, mass action changes the net flux; use J = α (P_sat,i − p_i,ambient) per species, as in Tachibana et al. 2011 eq. 1.
  - Some fits are extrapolated beyond their stated T range.
- **Surface energies**:
  - The N03 σ values are mostly melt surface tensions from the 1950s–70s.
  - Calorimetric surface enthalpies of nanocrystals are 3–10× higher: forsterite 4410, enstatite 4790, fayalite 2470, quartz 1000, corundum 2040, rutile 2060, ZnS 1760 erg cm^-2 (Householder et al. 2025 poster [V-img]; original papers not opened).
  - Choose σ consciously; θ_∞ scales linearly with σ.
- **Condensation temperatures**: Lodders 2003 values came from a single HTML extraction [HTML-x], and forsterite 1354 K is [unverified]. Morley 2012 values are at 1 bar. Use them only as sanity checks.

## 7. Sources I could not open (so their values are second-hand or missing)

- Paywalled or unavailable:
  - Hashimoto 1990 (Nature): the LPSC 1989 abstract was used instead.
  - Nagahara & Ozawa 1996; Nagahara et al. 1994; Wang et al. 1999; Ozawa & Nagahara 2000/2001; Tachibana & Tsuchiyama 1998 (full text).
  - Takigawa et al. 2009 (forsterite) and Takigawa et al. 2015 (full text; numbers via HTML).
  - Kimura et al. 2002 (Icarus); Gail 2010 (LNP 815); the Gail & Sedlmayr 2013 book; Kozasa & Hasegawa 1987.
  - Zavitsanos & Carlson 1973 (the 1970 GE report was used); Leider et al. 1973; Abrahamson 1974; Lilov 1993; Lihrmann 2008; Ferguson et al. 2004 (abstract only).
  - Mann et al. 2006 review; Kimura et al. 1997; Fray & Schmitt 2009; Lodders 2003 (HTML only).
- Woitke et al. 2018 has no vapour-pressure "table of A, B" for multi-species solids. Its D.2 p_vap entries refer to the same-formula gas molecule. That is why I derived the free-evaporation rows instead of quoting D.2 directly.

## 8. References (source IDs used in the CSV)

- **GD89**: Guhathakurta, P. & Draine, B. T. 1989, ApJ 345, 230 — eqs. 3.5–3.19, pp. 235–237 [V-img, ADS scan].
- **WD00**: Waxman, E. & Draine, B. T. 2000, ApJ 537, 796 (arXiv:astro-ph/9909020) — eq. 10 and text [V-txt].
- **vL14a**: van Lieshout, R., Min, M. & Dominik, C. 2014, A&A 572, A76 (arXiv:1410.3494) — Tables 2–3, eq. 13 [V-txt].
- **vL14b**: van Lieshout, R. et al. 2014, A&A 571, A51 (arXiv:1404.3271) — Sect. 2.2.1 [V-txt].
- **K11**: Kobayashi, H., Kimura, H., Watanabe, S., Yamamoto, T. & Müller, S. 2011, Earth Planets Space 63, 1067 (arXiv:1104.5627) — Table 1 [V-txt].
- **N03**: Nozawa, T., Kozasa, T., Umeda, H., Maeda, K. & Nomoto, K. 2003, ApJ 598, 785 (arXiv:astro-ph/0307108) — Table 2, eq. 4 [V-txt]. σ original references are in that table.
- **W18**: Woitke, P. et al. 2018, A&A 614, A1 (arXiv:1712.01010) — Tables D.1, D.2, D.4, D.5, eqs. 8–13 [V-txt].
- **Z70**: Zavitsanos, P. D. 1970, GE Report SC-CR-70-6126 (OSTI 4717386, doi:10.2172/4717386) — Table III [P-img].
- **TW57**: Thorn, R. J. & Winslow, G. H. 1957, J. Chem. Phys. 26, 186 [ABS].
- **Z66**: Zavitsanos, P. D. 1966, DTIC AD0633611 [ABS].
- Clarke, J. T. & Fox, B. R. 1969, J. Chem. Phys. 51, 3231 [ABS]; Hoch & Yamawaki 1972, DTIC AD0759203 [ABS].
- **Ph89**: Phinney, E. S. 1989, in Theory of Accretion Disks (NATO ASI), Sect. 6, eq. 9 [P-img via NED; pagination unverified].
- **Pound 1972**: J. Phys. Chem. Ref. Data 1, 135 — Table 1 [P-img].
- **Lamy74**: Lamy, P. L. 1974, A&A 35, 197 — eqs. 2–3, Fe formula, Tables 5 and 7 [P-img]. Quotes Schick 1960, Centolanzi & Chapman 1966 and Smithells 1967.
- **Hash89**: Hashimoto, A. 1989, LPSC XX, 385 [V-img]. Precursor of Hashimoto 1990, Nature 347, 53.
- **Ts98**: Tsuchiyama, A., Takahashi, T. & Tachibana, S. 1998, Mineral. J. 20, 113 — abstract, Fig. 5, Table 2 [V-img].
- Kuroda, D. & Hashimoto, A. 2002, Antarct. Meteorite Res. 15, 152 — eq. 3 [V-txt].
- Yokoyama, M. et al. 2007, LPSC XXXVIII, 1724 [V-txt]; Takigawa, A. et al. 2008, LPSC XXXIX, 1523 [V-txt].
- **Tach02**: Tachibana, S., Tsuchiyama, A. & Nagahara, H. 2002, GCA 66, 713 [abs]; Tachibana, S. & Tsuchiyama, A. 1999, LPSC XXX, 1532 [V-txt].
- Nagahara, H. & Ozawa, K. 1998, LPSC XXIX, 1641 [V-txt].
- **Costa17**: Costa, G. C. C., Jacobson, N. S. & Fegley, B. 2017, Icarus 289, 42 — Figs. 4C and 6, Table 4 [V-img, NSF-PAR manuscript].
- Nagahara, H., Kushiro, I. & Mysen, B. O. 1994, GCA 58, 1951 [2nd].
- **G13**: Gail, H.-P., Wetzel, S., Pucci, A. & Tamanai, A. 2013, A&A 555, A119 (arXiv:1305.2879) — eqs. 3, 4, 6, 7 [V-txt]. **FN08**: Ferguson & Nuth 2008 [2nd via G13].
- **G&S99**: Gail, H.-P. & Sedlmayr, E. 1999, A&A 347, 594 — eqs. 10 and 14, Table 1 [V-txt / V-img].
- **Tach11**: Tachibana, S. et al. 2011, ApJ 736, 16 — eqs. 1, 3–5 [V-txt].
- Tachibana, S., Tsuchiyama, A. & Watanabe 1999, Antarct. Meteorite Res. 12, 213 — pp. 220–221 [V-img], quoting Tachibana & Tsuchiyama 1998 (GCA 62, 2005).
- **Ts97**: Tsuchiyama, A., Uyeda, C. & Makoshi, Y. 1997, Geochem. J. 31, 289 [V-txt].
- Ferguson, F. T., Nuth, J. A. & Johnson, N. M. 2004, J. Chem. Eng. Data 49, 497 [ABS].
- **V10**: Visscher, C., Lodders, K. & Fegley, B. 2010, ApJ 716, 1060 (arXiv:1001.3639) — eq. 6 [V-txt]. Visscher, Lodders & Fegley 2006, ApJ 648, 1181 [V-txt].
- **MO12**: Morley, C. V. et al. 2012, ApJ 756, 172 (arXiv:1206.4313) — §2.5, eqs. 5, 9, 12, 15, 18 [V-txt].
- **Lee18**: Lee, E. K. H., Blecic, J. & Helling, Ch. 2018, A&A 614, A126 (arXiv:1801.08482) — Tables A.1–A.3 [P-txt]. Lee, E. K. H. et al. 2015, A&A 575, A11 (arXiv:1410.6610) — TiO2 σ [P-txt].
- **Tak15**: Takigawa, A., Tachibana, S., Nagahara, H. & Ozawa, K. 2015, ApJS 218, 2 [ABS + HTML-x].
- Yamamoto, Takigawa & Tachibana 2023, LPSC 54, #1129 (SiC in H2–H2O) [P-txt]; Yazdi, Iakimov & Yakimova 2016, Crystals 6, 53 [ABS].
- **JANAF**: Chase, M. W. 1998, NIST-JANAF Thermochemical Tables, 4th ed. (janaf.nist.gov) — fits made by the metals/sulfides sub-search.
- **MK05**: Murphy, D. M. & Koop, T. 2005, QJRMS 131, 1539 — eqs. 5 and 7 [V-txt]. **G11**: Gundlach, B., Skorov, Yu. & Blum, J. 2011, Icarus 213, 710 (arXiv:1101.2518) — Table 1, eq. 20 [V-img].
- **L03**: Lodders, K. 2003, ApJ 591, 1220 — Table 8 [HTML-x].
- Householder, M. et al. 2025, NExScI workshop poster (calorimetric surface energies, citing Chen & Navrotsky 2010, Lilova et al. 2018, Parks 1990, Chiang et al. 1996, McHale et al. 1997, Castro et al. 2006, Subramani et al. 2022) [V-img].
- Anand et al. 2024/25, steel research int. 96, doi:10.1002/srin.202400200 (hematite decomposition) [ABS].
