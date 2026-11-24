# Literature: carbon erosion products and routing options, by process

Sonnet subagent, read-only, 2026-11-23.

Provenance. Full: Jones & Ysard 2025 (2511.04750), Jones 2025 (2509.13119), Jones & Habart 2015 (1507.08534),
Micelotta+10a (0910.2461), Micelotta+10b (0912.1595), Micelotta+11 (1012.1599), Montillaud+13 (1301.6507), Murga+16
(1612.00419), Murga+19 (1906.11308, parts), Micelotta+12 (1207.5817, parts), Hirashita & Murga 2020 (2001.01968,
parts), Bocchio+12 (A&A 545, A124; abstract and Sect. 3). Web summary only: Berné+15 (1503.03698), Tazaki+20
(2002.08023). Abstract only: Berné & Tielens 12, Zhen+14, Guhathakurta & Draine 89, Bocchio+14, Serra Díaz-Cano &
Jones 08, Hoang & Tram. Not read (unverified): Draine & Salpeter 79, Tielens+94, Nozawa+06, Allain+96, Le Page+03,
Andrews+16, Murga+22/23, Pilleri+15, Jones+12 III, Jones+13/14. Note: local boc14.pdf and sdc08.pdf under
ac_literature_2026-10-24/optics/dl/ are HTML error pages; arxiv_1404.2107 is Montillaud & Joblin, not Bocchio+14.

## 1. Sputtering (thermal and non-thermal)
- Products: atoms. Bocchio+12 Sect. 3: "a collisional cascade contributes to the sputtering yield allowing the
  possibility of ejecting more than one target atom"; no cluster ejection (computed, SRIM-based).
- Size dependence: the classical size correction breaks down for N_C below ~1000 (a < 3 nm); classical treatment
  ignores electron collisions and electronic excitation; "small hydrocarbon nano-particles (N_C < 1000 or a < 3 nm)
  cannot be abundant in a hot coronal-type gas" (Bocchio+12 abstract).
- For PAH and nano-grain bins use the molecular approach of Micelotta+10b, not Draine & Salpeter / Tielens yields.
  Classical and molecular agree for N_C >~ 1000 at 1e5-1e6 K, within ~2 at 1e7 K (Bocchio+12).
- Preferential H sputtering: unverified; the a-C yield only slightly affected by H content (Bocchio+14 citing Serra
  Díaz-Cano & Jones 08); a-C:H yields exceed graphite (abstract level).
- Bocchio+14 carbon dust lifetime in SN shocks ~(6.2 +- 5.6)e7 yr (abstract; also quoted by Jones & Ysard 2025).
- Sputtered-down a-C below ~400 C: nobody models it. Murga+16/19 treat dehydrogenated grains below N_C ~ 1000 as
  "analogous to PAHs" (Murga+19 Sect. 2.1, assumption).
- Routing options: (a) carbon to gas as atoms (and H), grain shifts down (Murga+19: "the loss of carbon atoms does not
  change the grain aromatisation state"); (b) below N_C ~ 1000, switch to the PAH molecular rates.
- Where: SN shocks above ~100 km/s, gas above ~1e5 K.

## 2. PAH destruction by ions and electrons (computed only)
- Threshold T0 = 7.5 eV (4.5-15 eV explored; Micelotta+10a Sect. 2.2.1).
- Products: single atoms and C2, not pieces. Micelotta+10a Sect. 4: two limiting modes, "instantaneous and random
  removal" or "removal only from the periphery in order to preserve aromatic domain"; reality "somewhere between".
  Micelotta+10b abstract: "erosion of PAHs occurs via C2 loss from the periphery of the molecule, thus preserving the
  aromatic structure". No fragment-size distribution.
- Destruction criterion: 10a: all carbon lost (20-40 % loss "severely denatures" at 75-100 km/s); Micelotta+11: one
  third of the carbon lost (Eqs. 34, 37).
- Rates: SN shocks: N_C = 50 destroyed above 100 km/s, N_C = 200 above 125 km/s; lifetimes t_SNR = 1.6e8 / 4.0e8 yr
  (electrons, N_C 50 / 200), 1.8e8 / 1.4e8 yr (ions) (10a); Galactic lifetime a few 1e8 yr. Hot gas (10b): He below
  3e4 K, electrons above; at n_H = 0.01, T = 1e7 K lifetime ~1e3 yr.
- Routing: "shrink by one C2 per event" consistent with 10b; if stopping at 1/3 loss, the remainder as a carbon
  cluster or small a-C.
- Where: shocks, hot halo gas, ICM.

## 3. PAH photolysis
- Channels: H/H2 loss first; C2H2/C2 loss second and the only carbon-removing route. Murga+16 Sect. 3.1.1: RRK k0 =
  1e16 (H), 1e16 (H2), 1e15 s^-1 (C2); the second stage "plays an important role only for macromolecules with less than
  50 atoms".
- Size limit: Montillaud+13: PAHs < 50 C "are not expected to survive" in NGC 7023 NW or the diffuse ISM; transition
  N_C ~ 50-60 (A_V = 0), 50-70 (A_V = 1); fragmentation thresholds ~15, 37, 54 eV for C24, C66, C96 (Sect. 6.2).
  Jones & Ysard 2025 cite Jochims+94, Allain+95/96: < 30-40 C readily dissociated; > 50 C more than 1e9 yr in the
  diffuse ISM. Berné+15: < 60 C destroyed; 60-66 C can form C60 quickly; C66H20 -> C60 at ~10" from the star in ~1e5 yr.
- Products: Montillaud+13: "Carbon clusters turn out to be end products of PAH photodissociation"; can isomerise into
  cages (C60). Zhen+14 (lab, ns laser): > 60 C PAHs -> C60 by H then C2 loss. Fate of < 60 C unspecified.
- Options: below ~60 C -> "carbon cluster/cage" species (to gas as C, or a small a-C / fullerene sink); 60-66 C -> C60,
  survives at G0 < 1e4; do not shift PAHs down one bin at a time: fragmentation is fast once the threshold is crossed.
- Where: PDRs, HII regions, diffuse ISM below 50 C.

## 4. a-C(:H) nano-grain photo-processing
- (a) Aromatisation: Murga+16/19: H loss with sigma ~ 1e-20 cm^2 at 10-13.6 eV, E_g 2.67 -> 0.1 eV, E_g = 4.3 X_H;
  grains < 50 C "should be fully aromatized in the general ISM". HM20 Eq. 21 (see lit_identity.md; ~3e6 yr at 1 nm,
  U = 1, agent's arithmetic); aromatisation much shorter than shattering or accretion for most radii. Jones & Habart
  15 Eq. 8: >= ~4e4/G0 yr at a ~ 3-10 nm, >= 1e5/G0 yr for the smallest and largest.
- (b) Photo-fragmentation (Jones & Ysard 2025): loss of aliphatic-olefinic bridges -> C_nH_k fragments of at most
  C4H6-C5H7, mostly smaller, plus freed aromatic domains. Lifetimes of a ~ 0.4-0.5 nm (40-60 C): 1e6-1e7 yr diffuse,
  1e2-1e4 shorter in PDRs; larger grains stable; in HII regions only a > 0.7 nm (>~ 150 C) survive. "no experimental
  data or theoretical models" (Sect. 4): order of magnitude, k0 and E0 uncertain. Eq. 29: t_pf = 2 N_ar tau_abs /
  P(n_max). Jones & Habart 15 Sect. 3.3 cite Duley+15 lab: primary products < 4 C (C3H2, C3H4), phenyl, benzene.
- (c) Route to planar PAH: none quantitative. Jones 2025: dehydrogenated a-C(:H) nanoparticles end as "conjugated,
  polycyclic, olefinic sp2 systems resembling open loop, bowl, broad ring, or shell structures"; small planar PAHs
  (< ~30 C) may not dominate. Micelotta+12: UV/STIR dehydrogenation of arophatic clusters (N_C >~ 100) -> giant
  fullerenic cages shrinking by C2 loss to C60/C70 (Table 1: C144 cage 0.8 s at 200 K, 27 yr at 100 K; E0 = 0.36 eV a
  lower limit). Pilleri+15 not read here.
- Can nano-grains become indistinguishable from PAHs below N_C = 400? Not defensibly; Jones & Ysard: smaller particles
  fragment first (t_pf falls steeply with N_C); large dehydrogenated grains PAH-like in thermodynamic behaviour (Murga),
  not in structure.
- Options: (i) flag-based (Murga): E_g -> 0.1 eV and N_C < 1000 switch rates to PAH ones, no mass transfer; (ii) mass
  transfer to the PAH family at the same N_C once E_g is at its minimum (assumption); (iii) shift small a-C down and
  route the lost mass to gas small hydrocarbons.

## 5. Sublimation and PAH thermal dissociation
- Graphite: Guhathakurta & Draine 89 (abstract): detailed-balance sublimation; N_crit ~ 23 atoms (graphite), 37
  (silicate) for the solar-neighbourhood ISRF (1e13 s lifetime). A later paper quotes prefactor 2e14 s^-1 and B/k =
  81200 - 20000 N^(-1/3) K (unverified). Tazaki+20: T_sub = 1322 K (graphite), 1072 K (silicate).
- a-C differences: not found. Products: atoms or small clusters (unverified); route to gas C or C2/C3.
- PAH thermal dissociation = multi-photon photolysis (Sect. 3).
- Where: near stars and AGN (Tazaki+20: in the dense torus sublimation dominates).

## 6. Coulomb explosion and rotational disruption
- Tazaki, Ichikawa & Kokubo 2020 (arXiv 2002.08023): S = (1/4 pi)(U/a)^2 >= S_max, S_max = 1e9 (graphite,
  conservative), 1e10 erg cm^-3 (silicate); a_CE ~ 0.01-0.1 um for U_ion = 1e2-1e4, T_g = 1e4-1e6 K; size floor 0.005
  um. "Coulomb explosion will produce fragments of smaller grains", no sizes, possible cascade. Bulk S_max at nm sizes
  is a stated limitation.
- Jones & Ysard 2025: Coulomb fragmentation unimportant except for extremely large charges on small grains (aromatic
  domains >= +2 or +3).
- Rotational disruption (Hoang+19 RATD; not read): smallest nanoparticles (~2 nm in C-shocks) per a search summary
  (unverified).
- Options: fragments to smaller bins by a mass-conserving shattering kernel (alpha_f = 3.3), or ignore below ~1e4 K.

## 7. Cosmic rays (Micelotta+11)
- Single-C (electrons) or C2 (ions) ejection; destroyed at 1/3 loss. Electron tau0 = 1.2e13 / 2.0e13 / 4.2e13 yr for T0
  = 4.5 / 7.5 / 15 eV, independent of N_C: negligible. Ions: a few 1e9 yr (Fe-Co-Ni), small PAHs faster; matter only
  for cold shielded PAHs.

## Open questions
1. Product of a PAH below 60 C: gas atoms, a small a-C, or a cage?
2. When sputtering or photolysis takes an a-C grain below ~400 C, which family keeps the mass?
3. Are the Murga flag-switch rates (E_g = 0.1 eV at N_C < 1000) acceptable for N_C 20-400 PAH bins?
4. Unread sources to verify next: Draine & Salpeter, Tielens+94, Nozawa+06, Allain+96, Andrews+16, Murga+22/23, Pilleri+15.
5. Hydrogen-preferential sputtering in Serra Díaz-Cano & Jones 08 / Bocchio+14?
6. a-C vs graphite sublimation: not found.
7. Tensile strengths for nm-scale a-C: only bulk values.
