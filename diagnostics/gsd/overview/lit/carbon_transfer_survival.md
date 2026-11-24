# Literature: observational constraints on PAH / small-grain survival versus density and phase

Sonnet subagent, read-only, 2026-11-23 (resumed after a network error). No paper read in full: local PDFs text-extracted,
abstracts plus grep-located passages (L = local); W = web abstract or snippet only (unverified in text). Not checked:
Boulanger+90, Bernard+93, Egorov+23, Baron+24, Sandstrom+23, Whitcomb+ molecular content, Aniano+20 molecular result,
Ritchey+26, Hardegree-Ullman+19, Sales+. OBS = observation, MOD = model statement.

## 1. q_PAH versus phase and density
- LMC/SMC phase parity (OBS, L): Chastenet+19 (arXiv:1904.02705) abstract: "the PAH fraction in the diffuse neutral
  medium is higher than in the ionized gas, but similar to the molecular gas." Table 2, q_PAH (%): diffuse neutral LMC
  4.1 (+0.6/-0.8), SMC 1.1 (+0.2/-0.3); molecular LMC 4.3 (+1.3/-0.9), SMC 1.1 (+0.1/-0.2): ratio ~1.05 (LMC), 1.0
  (SMC), ~+-30 % in the LMC. "Molecular" = 3-sigma NANTEN CO cut at ~3' (Sigma_mol 7.5 / 22.8 Msun/pc^2): GMC scale, not
  cores. Parity "consistent either with formation in the diffuse ISM and incorporation into molecular clouds or its
  inverse"; accretion/coagulation onto big grains "could also lead to a decrease" - none found.
- SMC (OBS, L): Sandstrom+10 (arXiv:1003.4516): q_PAH ~1-2 % in molecular clouds vs 0.6 % diffuse, correlated with CO
  and Sigma_dust; reduced to parity by Chastenet+19 with a clean diffuse neutral phase.
- Nearby galaxies, JWST (OBS, L): Chastenet+23 (arXiv:2301.00578): R_PAH = (F770W+F1130W)/F2100W "relatively constant
  across other galactic environments", drop only in H II regions above I_Ha/Sigma_gas ~ 10^37.5; "the abundance of PAHs
  is not particularly sensitive to the molecular fraction"; NGC 7496 R_PAH rises ~20 % then falls; medians vary by "a few
  tens of percent"; 40-90 pc scales.
- KINGFISH (OBS, L): Aniano+20 (arXiv:1912.04914): q_PAH suppressed where dust is warm (nuL_nu(70) > 0.4 L_dust), rises
  with Z above 0.15 Zsun; molecular vs atomic comparison not extracted.
- Translucent MW clouds (OBS + MOD, L): Zhang, Hensley & Green (arXiv:2410.23171): R(V) anticorrelates with q_PAH; "the
  range of observed q_PAH indicates PAH growth by a factor of ~2 between A_V ~ 1 and 3"; q_PAH ~4-8 %, example 6 -> 11 %
  (+54 ppm C vs ~200 ppm gas C). Growth model-inferred; caveats W3 decomposition, line contamination, 2-D mixing. App. C:
  mass doubling 1.3-2 Myr for C10-C24 PAHs by C accretion at 20 K (after Krasnokutski+17).
- Low metallicity (OBS, L): Tarantino+26 (arXiv:2512.04060): Sextans A PAH emission in compact 3-10 pc clumps, "active
  sites of in-situ PAH growth within a dense, well-shielded phase". Whitcomb+24/26: inhibited growth, 3.3 um fraction up
  3x as Z falls. Not directly about freeze-out.

## 2. Depletion in dense filaments and cores
- Taurus filament (OBS, W): Stepnik+03: no 60 um emission from transiently heated small particles inside the filament,
  T_d 12.1 K; small grains removed from the densest region, submm emissivity x3.4; interpreted as coagulation into fluffy
  aggregates; quantitative deficit unverified.
- Köhler+15 (L, second-hand): mid-IR emission "decreases by 80-100 % from diffuse to denser regions (e.g., Laureijs+91;
  Stepnik+03; Juvela+12)"; confounded with radiation-field attenuation.
- Taurus L1506 (OBS, W): Ysard+13 (arXiv:1309.6489): coagulation into aggregates dominates at "a few 10^3 H/cm^3" (A_V
  ~2-3); 250 um opacity x1.8-2.2, grain size x~5; no explicit PAH/VSG number.
- The ~1500 cm^-3 threshold is an assumption (L): Köhler+15 Sect. 2: "we assume to occur in relatively dense
  interstellar environments (nH > 1500 H/cm^3, as in Ysard+13)".
- Taurus, Spitzer (OBS, W): Flagey+09 (arXiv:0906.5025): variations "by a factor of a few" in both radiation field and
  PAH/VSG abundance; "a significant fraction of the dust mass cycles in and out the small size end" (partial, reversible);
  FIR opacity ~2x diffuse.
- PAHs in ices (OBS, W): Chiar+21 (Mon R2 IRS 3, Elias 16): 3.25 um absorption matched by coronene:H2O; frozen neutral PAHs
  up to ~12 % of cosmic C (Mon R2), < 14.5 % (Elias 16); earlier 5-9 %; lab band strengths good to ~x3; YSO sightlines.
- Gas-phase PAHs in TMC-1 (OBS, L): Wenzel+24 (arXiv:2410.00657): pyrene "up to 0.03-0.3 % of the carbon budget in the
  dense ISM". McGuire+21 (arXiv:2103.09984): naphthalene n/n(H2) = 1e-7, "roughly 1 % of the total carbon" (a model
  initial condition, MOD). Only small PAHs (<= 24 C) seen; possibly bottom-up.
- JWST ice cores (W): McClure+23 (arXiv:2301.09140): ices carry 2-19 % of bulk C, O, N, S in two starless-core
  sightlines. No JWST 3.3 um PAH-vs-density result found.

## 3. Gas-phase carbon
- Jenkins 09 (L) Table 4 for C: A_C = -0.101 +- 0.229, B_C = -0.193 +- 0.060, [C_gas/H]_0 = -0.112, [C_gas/H]_1 =
  -0.213; with solar 8.46 (~288 ppm): ~223 ppm gas at F* = 0, ~177 ppm at F* = 1 (agent's arithmetic). Sect. 10.1.2: at
  F* ~ 0.8 "the amount of C available for interstellar dust (and small molecules) is only 1e-4 times H", about half of
  what extinction needs; Sofia & Parvathi 2009 line strength could raise dust C by ~90 %. Diffuse and translucent only.
- Sofia+11, Parvathi+12 (W): strong-line method gives lower gas-phase C (more dust C); values not read.
- Dense clouds: budget bounded, not measured (ices 2-19 % of C; CO presumably dominant, unverified; frozen PAHs <~12 % of
  C; TMC-1 PAHs ~1 % model). Nothing verified requires or forbids PAH delivery to grains.

## 4. Timescales
- Agent's estimate (not literature): grain cross-section per H ~5e-22 cm^2, PAH thermal speed 0.027 km/s (C50, 20 K),
  unit sticking: t ~ 2 Myr at n_H 1e4, ~200 Myr at 1e2 (gas species like CO ~10x faster, ~2e5 yr at 1e4). The one-zone
  95 % loss in 1 Myr needs speeds above thermal, Coulomb focusing, or small-grain area (it has the nano-grains' area).
- Observations do not show freeze-out of PAH-sized material in GMC-scale gas (parity; constrains the net balance, not
  sticking). Cold-core data show small-grain loss and coagulation at A_V >~ 2-3 (n ~ few 1e3 and above). Unit sticking
  to all grains at n_H 1e4 is allowed only for dense, low-field cores; for CNM and GMC-averaged gas it conflicts with
  parity.

## Gates
1. q_PAH(molecular)/q_PAH(diffuse neutral) = 1.0-1.1 (+-30 % LMC, 15-25 % SMC) at GMC scale (n_H ~1e2-1e3, Sigma_mol
   >~7-23 Msun/pc^2). Chastenet+19 Table 2. Medium-high (CO-defined phases, ~45 pc).
2. R_PAH changes by only a few tens of percent across environments, no strong dependence on molecular fraction, 40-90
   pc. Chastenet+23. Medium-high, few galaxies.
3. Translucent (A_V ~1-3): q_PAH does not drop, may rise up to ~2x. Zhang+25. Low-medium; an upper bound on loss.
4. Dense filaments/cores (n_H >~ few 1e3, A_V >~ 2-3, low field): small-grain mid-IR emission falls by a factor of a few
   to 80-100 %, large-grain opacity ~x2. Stepnik+03, Flagey+09, Köhler+15, Ysard+13. Low-medium (degenerate with field
   attenuation).
5. Frozen PAHs <~12-15 % of total C on YSO sightlines; ices 2-19 % of C in cores. Chiar+21, McClure+23 (both W).
   Low-medium.
6. Gas-phase C 177-223 ppm over F* = 0-1 bounds the movable PAH-sized C. Jenkins 09 (diffuse only). Medium (diffuse).
7. Cold cores (TMC-1): PAH-sized species survive at >= 0.03-1 % of C. Wenzel+24, McGuire+21. Medium (detection), low
   (carbon fraction).
