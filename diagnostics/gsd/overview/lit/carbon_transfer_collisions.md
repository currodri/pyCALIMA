# Literature: grain collisions (shattering, cratering, coagulation) and their products

Sonnet subagent, read-only, 2026-11-23.

Source key. Full text (local): Hirashita & Murga 2020 (HM20, 2001.01968), Hirashita 2022 (H22, 2211.09982),
Narayanan+23 (2301.07136), Narayanan+26 (2606.20809), Seok+14 (1401.3943), Rau+19 (1909.02725), Murga+16
(1612.00419), Murga+19 (1906.11308), Jones+13 (1411.6293), Jones+14 (1411.5877), Jones+25 (2511.04750), Schirmer+22
(2209.04282), Köhler+15 (1506.01533), Weingartner & Draine 1999 (WD99), Rapacioli+06, Choban+26, McKinnon+18. Web /
fetched PDF: Hirashita & Kobayashi 2013 (HK13, arXiv 1303.3958), Hirashita & Yan 2009 (HY09, arXiv 0812.3451), Anderl+13
(1408.0140). Abstract / snippet only (unverified): Guillet+11, Ormel+09, Wada+09/13, Köhler+12, Draine & Sutin 87,
Akimkin+20, pyrene-cluster paper (1809.09770), Dubois+24, Granato+21, CALIMA (2602.21790). Not obtained: Jones+96,
Tielens+94, Kobayashi & Tanaka 2010 (KT10), Guillet+09, Hirashita+21 porosity, Aoyama+, Kirchschlager+, Ysard+16.

## A. Shattering and cratering
A1. Fragment size law
- Index (computed): alpha_f = 3.3 in dN/da ~ a^-alpha. HK13 Sect. 2.2: Jones+96 "argue that alpha_f = 3.0-3.4, based
  on an analysis of the flow associated with the formation of a crater". HM20 Eq. 17, McKinnon+18, Choban+26 Eq. 24 use 3.3.
- Lab (secondhand via HK13 Sect. 2.4.3): impact experiments give a wide range; HK13 tested 2.3 and 4.3 (Takasawa+11).
- Jones+13 footnote 16 "1.8-2.0" is very probably a mass index: for dN/dm ~ m^-beta, beta = (alpha+2)/3 = 1.77 for
  alpha = 3.3 (agent's derivation, not stated in the footnote). Schirmer+22 Sect. 6.3: gamma = -2.1 to -1.2 for
  aggregate fragmentation (Güttler+10, secondhand).
- Sensitivity (HK13 abstract): "minor effects as long as alpha_f <~ 3.5"; Choban footnote 9: "< 4". Cascades converge
  to a^-3.5 (KT10 via HK13).
- Cutoffs: a_f,max = (0.0204 f)^(1/3) a1 is HK13 Eq. 5 (from Guillet+09), with a_min = 3 A. HM20 and Choban: m_f,max =
  0.02 m_ej; HM20 m_f,min = 1e-6 m_f,max; Choban a_min = 0.01 a_max. Agree within a few % on a_max; differ on the lower
  cutoff (fixed 3 A vs 0.01 in radius).
- Ejected mass: Jones+96 from a critical shock pressure P1; KT10 M_ej = phi/(1+phi) m1, phi = E_imp/(m1 Q*_D), HK13 Eqs.
  6-9 Q*_D = P1/(2 rho); graphite (P1 = 4e10 dyn cm^-2, rho = 2.2) ~9e9 cm^2 s^-2 (as HM20). HK13: P1 the dominant
  uncertainty; KT10 vs Jones+96 changes n(a) by 1.3 (silicate), 1.6 (carbonaceous).
- Material: every galaxy model uses graphite P1 and v_shat; no P1 or Q*_D for a-C or a-C:H found.

A2. What the fragments are
- Murga+16 Sect. 3.3 (assumed): M_shat = 0.4 M_sh, a^-3.3 fragments inheriting the parent's E_gap; above 200 A,
  layered: surface fragments take the outer layer's value, otherwise the inner layer's.
- Rau+19 and HM20 shatter aromatic and non-aromatic species separately: fragments keep the parent's class. HM20 Sect.
  3.2: shattering "helps to convert the aromatic grains with a > 50 A to small PAHs"; accretion "does not directly
  produce PAHs".
- Direct production of PAH molecules: no lab or MD work. Seok+14 Sect. 1: "PAHs (or PAH-like carbonaceous fragments)"
  (assumption; cites Jones 2009). Jones+25 Sect. 4: "no experimental data or theoretical models" for a-C(:H)
  fragmentation pathways. Narayanan+26 Sect. 2.2: PAH as a-C grains "an effective approximation".
- Vaporisation (computed): threshold ~19 km/s (Anderl+13 Sect. 1, citing Tielens+94, Jones+96); fraction vs velocity
  not verified; HY09 and galaxy models omit it (HY09 Sect. 2.1).
- Collisional dehydrogenation or amorphisation of fragments: not treated anywhere found.

A3. Where and how fast
- Thresholds (computed): 1.2 km/s carbon, 2.7 km/s silicate (Jones+96 via HY09 Table 2, HK13, Choban Table 2);
  Esmerian+26 may update (Narayanan+26, unverified).
- Velocities: gyroresonance ~20 km/s for a >~ 0.1 um in the WIM (HK13 Sect. 2.4.4); CNM 1-2 km/s, near threshold
  (HY09 Sect. 2.1.3).
- SN velocities destroy PAH-sized products, so Seok+14 argue for turbulence (~10 km/s). Guillet+11 (abstract):
  dense-cloud C-shock shattering only above ~1e5 cm^-3.
- Seok+14 Table 2 (garbled text, agent's reading): PAH production ~30-44 Myr (WIM), 1.3-2.3 Gyr (CNM).
- Bottom line A: alpha = 3.3, KT10 ejecta with Q*_D = P1/(2 rho), HK13 Eq. 5 a_max best supported; alternatives alpha
  = 3.0-3.5 and the 0.01 a_max lower cutoff. Fragments inheriting the parent's H state and aromaticity (Murga+16) is
  the only published choice, followed by UV equilibration.

## B. Coagulation
B4. PAH + PAH
- Neutral PAHs van der Waals only (computed): Rapacioli+06 ~1 eV per molecule vs > 4 eV chemical bonds; dimer
  nucleation > 50 % below 3.6 eV collision energy; clusters photoevaporate much faster than they form in NGC 7023.
- Covalent channels (abstract/snippet only): lab pyrene dimer cations under laser excitation form a single C32H16
  aromatic cation (arXiv 1809.09770); MD: PAH-H radicals bond to a neighbour in ps (source not confirmed). No
  astrophysical rate. Dropping clusters is defensible.

B5. PAH or nano-grain onto big grains
- Seok+14 unit sticking (Sect. 4.3); Table 2 coagulation timescales ~0.3-0.5 Myr (dense clouds), 10-17 Myr (molecular
  clouds), 100-140 Myr (CNM); dense-cloud value floored at the 3 Myr cloud lifetime (Sect. 2.1.2).
- WD99 Eq. 20 (unit sticking): tau ~ 1e7 yr (1 km/s / v)(30 cm^-3 / n_H) for a = 10 A; drift speeds ~50 times smaller
  than 1 km/s.
- Jones+14 Eq. 12, Fig. 5: tau^-1 = pi (a1+a2)^2 X1 n_H dv; for 5 nm onto 160 nm, small-grain coagulation "always lags
  behind C atom accretion" as a mantling route in cloud interiors.
- Fate: the aromatic-rich a-C mantle, ~5 nm thick (THEMIS core/mantle/mantle, Köhler+15 Sect. 2). Release by top-down
  photo-fragmentation or fragmentation in the diffuse ISM (Jones+13 Sect. 7.3; Jones+25; Schirmer+22). Desorption by
  CRs, UV or shocks: no quantified rate.
- Coulomb factors: Murga+19 Eq. 3: F_C = 1 - 2 Z1 Z2 e^2/[mu (a1+a2) v^2] (0 if negative), repulsion only, ~1 at
  shattering speeds. Draine & Sutin 87 (abstract) adds polarisation; Akimkin+20 (abstract): electrostatic barrier for
  micron grains. No published PAH-grain Coulomb treatment found.

B6. Nano + nano (1-20 nm): no dedicated study; unit sticking assumed (HY09, HM20); covalent vs loose unknown. H22 Sect.
1: suppressing small-grain processing is "just a hypothesis (without firm confirmation by experiments or theory)".

B7. Large-grain thresholds and structure: HY09 Eq. 8 Chokshi/Dominik & Tielens form with F_stick = 10 (factor 10 moves
the largest coagulating size by only 2-3); Choban Sect. 3.6.2: 1-80 m/s; Wada+09/13 (search): ~8 (r/0.1 um)^-5/6 m/s
silicate aggregates, ~60 m/s ice. Compact spheres assumed (Choban, HM20, H22); porosity shifts the 2175 A bump (H22);
Ormel+09 (abstract): fluffy at low velocity; Schirmer+22: aggregates break at 1 m/s (Güttler+10, secondhand).
- Bottom line B: unit sticking below the HY09 v_coag with a geometric kernel; PAH-sized particles coagulating like
  small grains with Coulomb factor 1; optionally a lower cap on the dense-cloud timescale (Seok+14). Nothing better
  supported.

## C. Galaxy models
| Model | Shattering fragments to | PAH coagulation onto grains |
|---|---|---|
| Seok+14 | 3-20 A fragments are PAHs directly (graphite properties) | unit sticking, a loss term |
| Rau+19, HM20 | small carbonaceous grains, aromatised in UV; PAH = aromatic grains 3-50 A | applied; high density suppresses PAHs |
| H22 | same | switched off for carbon below 20 A |
| Narayanan+23 | aromatic fraction counted as PAH, no size cut | included |
| Narayanan+26 | PAH = grains below 13 A (< 1000 C), no aromatic/aliphatic split | included; off lowers q_PAH |
| Choban+26 | no explicit PAH population | n/a |
| Dubois+24 | mass to the 5 nm bin; no PAHs | none |
| Granato+21 | two-size small carbon grains | none (search level) |
- H22 reasons for switching processing off: shattering + coagulation drive n(a) to MRN; coagulation porosity would shift
  the bump; WD01/LD01/Jones+13 need excess small carbon grains. A hypothesis; coagulation off matters most.
- Narayanan+26 Sect. 3.3: coagulation off lowers q_PAH (fewer large grains to shatter); sputtering their dominant PAH
  sink (changes q_PAH by ~2).
- Choban+26: fast accretion kills small carbonaceous grains; "top-down" PAH formation may be needed.
- Seok+14 and HM20: PAHs not destroyed directly by coagulation in their one-zone models.

## Open questions
1. No P1, Q*_D or v_shat for a-C or a-C:H.
2. State of fragments: H content after impact, vaporised fraction vs velocity, aromaticity.
3. Whether any 0.4-1 nm shattering fragment is a PAH molecule rather than a nano-grain.
4. Sticking and Coulomb factor for PAH + grain and nano + nano.
5. Covalent vs loose bonding after nano + nano coagulation.
6. Release rate of coagulated carbon (CR, UV, shock desorption).
7. Compact-sphere v_coag vs aggregate thresholds (1 m/s).
