# Literature: the PAH <-> a-C(:H) boundary at N_C ~ 20-400 (growth, conversions, fragments)

Sonnet subagent, read-only, 2026-11-23. "Full" = relevant sections of the full text (local pdftotext copy); "abstract" =
abstract or search snippet; "notes" = from the 2026-11-17 notes in diagnostics/gsd/overview/lit/ (not re-verified).
M = measured, C = computed, A = assumed.

**Bottom line.** No paper gives a verified, environment-dependent rule for moving carbon between "PAH" and "a-C".
Galaxy models either assume small carbon is PAH (Seok+14, Rau+19, Narayanan+26, Parente+26) or carry an aromatic
fraction with timescales (Hirashita & Murga 2020 = HM20, Narayanan+23). THEMIS denies that PAHs exist at all. A size
rule plus local equilibrium is defensible for the aromatic/H state. "Fragments of N_C < 400 are instantly PAH" is not
defensible without a stated assumption.

## 1. Identity at N_C ~ 20-400
Options: (a) planar PAH; (b) 3-D a-C(:H) "arophatic" particles; (c) a mixture.
- (b) THEMIS (full). Jones+13 (1411.6293) footnote 3: aromatised a-C nano-particles are "three-dimensional networks
  ... The term PAH cannot be applied to them because they are not purely aromatic carbon structures"; a continuous
  distribution of carbon properties "rather than separate and disconnected populations of small carbon grains and
  PAHs" (C).
  - Jones 2025 (2509.13119, full): "small, pure, perfect, planar PAHs, with less than ~30 C atoms, may not be the
    dominating nanoparticle dust component"; planar PAHs cannot explain the 3.4-3.5 um bands without sp3 carbon,
    which deforms them (argument).
  - Jones & Ysard 2025 (2511.04750, full) Sect. 3.2: for a < 1 nm (N_C <~ 300) the largest aromatic domain has
    N_R(max) <~ 60 rings; for a = 0.4 nm (N_C ~ 40), N_R(max) <~ 8. Open or shell-like structures at ~100 C (C/model).
- (a)/(c) observations.
  - Pilleri+15 (1502.04941, full): I3.4/I3.3 falls from 0.13 (G0 ~ 150) to 0.03 (G0 > 1e4) in NGC 7023 (M);
    "photo-processing of eVSGs leads to the production of PAHs with attached aliphatic sidegroups ... very small
    grains of mixed aromatic and aliphatic composition"; aliphatic carriers "not efficiently regenerated" (M).
  - Rapacioli+05/06 (notes): eVSG minimum ~400 C (the origin of N_C,max = 400).
  - Croiset+16 (notes): N_C ~ 70 in the NGC 7023 cavity, ~50 at the PDR surface.
  - Lemmens+23 (secondhand, search): Orion Bar upper size limit 37-52 C; Orion Bar 3.3/11.2 drops ~40 % from the
    dissociation fronts to the H II side (PDRs4All IV, abstract).
  - Draine+21 (notes): 3.3 um from N_C <~ 100, 7.7 um ~1e2, 17 um >~ 1e3. Ricca+12 (notes): compact PAHs above ~150 C
    stop matching the 6.3 and 7.7/8.6 bands.
  - Elyajouri+24 (2401.01221, full): THEMIS fits, a_min ~ 0.4 nm in the Orion Bar, nano-grains depleted 15x,
    hydrogenation varying across the bar. Schirmer+22 (2209.04282, full): a_min ~ 0.7-0.8 nm.
- Size diagnostics constrain the emitter size (~40-100 C), not planarity; THEMIS and PAH-database fits both reproduce
  the bands. Tokunaga+25 (2510.26970, abstract): a few PAH species cannot explain the band uniformity (unverified).
- How models treat the transition:
  - DL07/HD23: xi(a) = 0.01 for a <= 50 A, then 0.01 + 0.99[1 - (50 A/a)^3] (HD23 Eq. 15-16); "entirely ad hoc" (DL07,
    notes); HD23 PAHs at < 1000 C (qPAH = 5.9 %).
  - Seok+14 (full): PAH = carbonaceous grains with a = 3-20 A, by size only; fragments assumed PAH-like (cites Jones+96;
    Jones 2009 a-C:H -> PAH precursor argument).
  - Rau+19 (full): small carbon aromatised "much shorter than ... shattering and accretion", so PAH = small carbon;
    aromatic grains lose MIR features above ~100 A (Murga+16).
  - HM20 (full): state variable f_ar(m); PAH = aromatic carbon with a <= 0.005 um.
  - Narayanan+23 (full): aromatics and aliphatics as HM20, no PAH size cut. Narayanan+26 (full): PAH = carbon < 1e3 C,
    "we do not include a model for aliphatic and aromatic grains separately".
  - Parente+26 (2609.35533, full): PAH = carbon < 15 A in the diffuse gas, scaled by (1 - f_H2).
  - Murga+19 Shiva (full) footnote 4: small a-C:H grains "are volumetric in a general case, though in some specific
    cases they can be PAHs"; dissociation properties set equal to PAHs (A).
  - Choban+26 (2603.08504, full): no < 1 nm population survives (fast accretion); calls for top-down PAH formation.
- Recommendation: treat the identity of N_C ~ 50-400 carbon as unresolved; keep N_C,max = 400 as an assumed,
  Rapacioli-motivated threshold; do not claim planarity. Open: no discriminating measurement of planar vs 3-D at
  100-400 C found.

## 2. Conversions at fixed size
- a-C:H -> a-C (UV dehydrogenation):
  - Jones+12 II (notes): tau ~ 1e4/(Q_abs eps) yr, ~1e5 yr in the diffuse ISM.
  - Jones & Habart 15 Eq. 4-8 (notes): tau = (1e4/G0){2.7 + 6.5/a^1.4 + 0.04 a^1.3} yr (a in nm), minimum ~4e4/G0 yr
    at 3-10 nm; ~6.5e4/G0 yr at 1.5 nm (C, agent's arithmetic).
  - HM20 Eq. 21: tau_ar/yr = U^-1 [3 (a/um)^-2 + 6.6e7 (a/um)] (coefficient layout garbled in the extracted text:
    verify): ~3e6/U yr at 1 nm, ~7e5/U at 10 nm, ~7e6/U at 100 nm (agent's arithmetic); ~20x longer than Jones &
    Habart at 1.5 nm. HM20: one-zone equilibrium, tau12 = 1e7 yr phase exchange (A).
  - Murga+16 (notes): one-way, no re-hydrogenation; N_C < 2e4 loses H in several 1e5 yr at U = 1 (C).
- a-C -> planar PAH: needs heating/annealing; evidence thin. Scott & Duley 96 (abstract, secondhand): HAC decomposition
  leaves 1-5 nm protographitic islands. Micelotta+12 (full): arophatic clusters (C104H80) -> fullerenes under UV.
  Berné & Tielens 2012 (abstract): PAH -> graphene -> C60, ~1 Myr. These are aromatic/aliphatic -> fullerene routes,
  not a-C -> planar PAH. Murga+19: SN shocks aromatise grains, small ones destroyed first. No study of a-C -> planar
  PAH timescales vs G0, n_H, T found.
- PAH -> a-C (aliphatisation): Murga+19 fit (via HM20): tau_al,H = 1.6e5 (a/um) yr, plus tau_al = tau_acc/xi; origin
  untraceable. Omont & Bettinger 2025 (full): C+ accretion onto neutral PAHs >= 2e-9 cm3 s-1 (~2e-4 yr-1 per PAH, from
  the Canosa+95 anthracene measurement); for N_C <~ 50 C2H2 returned quickly; above N_lim ~ 40-50 "continuous growth
  ... cannot be excluded"; 5C-ring and chain-adduct products (B3LYP). Jones+14 (notes): H incorporation slow.
- Recommendation: the a-C:H <-> a-C hydrogen state through the Jones-type band gap; the PAH tables an assumption.
  Open: HM20 vs Jones & Habart differ ~20x at 1 nm; neither has thermal H loss for the smallest grains.

## 3. Growth by gas carbon
- Onto grains: Jones+13 Sect. 7.2: a-C mantles in the diffuse ISM; a-C:H mantles at A_V < 3, ices above. Bossion+24
  (abstract): MD sticking on a-C, 50-2250 K. Esmerian+26 (2607.14237, full): chemisorption sticking > 0.2 for all
  species and temperatures (computed), an upper limit for a-C:H. Choban+26: fast growth removes < 1 nm grains.
- Onto PAHs: reaction-limited (Omont & Bettinger); no C+ accretion onto cations.
- PAH past a few hundred C: uncertain (C2H2 loss vs growth with loosely bound rings and chains); no study shows a PAH
  grown past ~300 C stays planar.
- Recommendation: grown mass is a-C(:H) above N_C,max; below it, accretion returns carbon as small hydrocarbons for
  N_C < ~50 (Omont & Bettinger), uncertain for 50-400. Open: C(3P) rates (Hickson & Loison) not read.

## 4. Fragments
- Instant PAH: Seok+14, Rau+19, Narayanan+26, Parente+26, by definition; Rau+19 reference fast aromatisation; Seok+14
  cite Jones 2009 (A).
- a-C that aromatises on tau_ar: HM20, Narayanan+23.
- a-C that never becomes a PAH: Jones+13: disintegration yields "small free-flying, PAH-like species and short
  hydrocarbon chains", destroyed in the diffuse ISM; Jones & Ysard 2025: photo-fragments mostly C4H6-C5H7 bridge
  pieces, lifetimes 1e6-1e7 yr at 40-60 C (diffuse), 1e2-1e4 times shorter in PDRs. Schirmer+22: nano-grains formed by
  fragmentation of large grains in radiation-pressure collisions. Pilleri+15: aliphatic-bearing PAH-like products.
- Recommendation: shatter fragments are a-C(:H) of their parents' H state; PAH-table properties only through the size
  rule; "instant PAH" for all fragments is the least defensible choice.

## 5. Extra state variable?
- Aromatic fraction (HM20-like): not needed if the H equilibrium is applied locally and the timescale (>~ 1e5/G0 yr,
  Jones; ~1e6/G0 yr at 1 nm, HM20) is short against dynamics; needed if dynamics are faster. Schirmer advection times:
  1.8e3 yr (Orion Bar), 2e4 yr (IC 63), 5e4 yr (Horsehead) (notes), shorter than tau_pd.
- Montillaud+13 (notes): PAH steady state in < 1 yr to ~1e4 yr, not for large PAHs; Jones+17: out of equilibrium in PDRs.
- Recommended hybrid: size rule for tables, plus an H-state variable only for N_C ~ 50-400 in PDR and shock gas. A
  separate PAH/a-C label is a decision on the planarity assumption, not on timescales. Open: a thermal H-loss and
  annealing sensitivity run for the smallest a-C bin.

Web sources: Berné & Tielens 2012 (arXiv:1111.0839), Tokunaga+25 (arXiv:2510.26970). Local: reference_data/
{carbon_growth_2026-10-30, ac_photochem_2026-10-29/models, ac_literature_2026-10-24/optics/dl}, diagnostics/gsd/overview/lit/.
