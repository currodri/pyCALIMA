# Carbon mass transfer: options per channel (2026-11-23, first synthesis)

Sources: lit_collisions.md, lit_erosion.md, lit_identity.md (three literature reviews, this folder), the code maps of
2026-11-23 (gsd collisions and legacy CALIMA PAH processes), and the PAH/grain seam check (docs/physics/charging-cm20.md).

Carbon species on the one grid: gas (C, C+, C2/C2H2, small C_nH_k, CO); "molecular" carbon N_C 20-400 (bins 1-4,
0.39-1.06 nm); a-C(:H) nano-grains 1.06-20 nm (bins 5-7); grains 20 nm - 1 um (bins 8-11) and the 1-10 um overflow.
Decided: no PAH clusters; legacy CALIMA PAH-PAH / PAH-grain methods not reused; grain H from the local equilibrium
band gap (tracked M_H built in, off); PAH H from equilibrium tables.

## The original design, checked against the new literature (2026-11-23)

Original records: the plan's carriers and processes tables (2026-10-24/30), the overview report, `gsd/carbon.py` and
`gsd/materials.py` docstrings, docs/physics (charging-cm20.md seam, grain-hydrogen.md, pah-hydrogen.md), memory notes.
History of the identity: 10-24 the user proposed three populations (large H-poor a-C, small aromatic grains following
the PAHs, small a-C:H) -> the "middle option" (one graded grid, aromatic and aliphatic populations with their own N, M
in the small bins) -> 10-30 one population per bin with the hydrogen coordinate (hydrocarbon family below N_C,max,
E_g-mapped a-C(:H) above; M_H carried) -> 11-15 PAH H tabulated in equilibrium -> 11-22 grain H from the equilibrium
band gap (M_H built in, off).

| process | original design (date) | new literature (23 Nov) | verdict |
|---|---|---|---|
| identity N_C < 400 | hydrocarbon family along H (10-30); structure at equal H/C a labelled assumption; P1 to N_H ~ 1.4 N_C | unresolved: size diagnostics cannot tell planar from 3-D; THEMIS has no PAHs; galaxy models PAH by size or an aromatic fraction | design stands; P1's a-C:H end and the bare end still to build; the seam is first-order for heating (check 1: x1.4-2.5), so the seam decision moves up |
| N_C,max = 400 | run-time knob; no observation needs molecular properties above 300-400 C (10-24, verified 11-17) | "assumed, Rapacioli-motivated" | stands, but its main anchor (eVSGs = clusters >= 400 C) leaves with the clusters; bands come from N_C <~ 100-150; the extended band test (400 vs ~1000) still open |
| 3-4 two-moment bins over N_C 20-400 | 3.3 um ratios within 2-7 % (10-24) | - | the script and output are not in the repo or results: not reproducible; redo with the extended band test |
| shattering | fragments at the parent's H/C (x_min for CM20 large grains: mantle a-C); graphite strength as proxy; fragments below the grid to the collector (10-30) | alpha = 3.3 (HK13/Jones+96); fragments inheriting the parent's state is the only published choice (Murga+16); no a-C strength data; vaporisation > 19 km/s omitted by everyone | confirmed; vaporisation omitted (labelled) |
| coagulation, freeze-out | PAH and nano-grain freeze-out onto large grains (Köhler+15); PAH+PAH off unless clusters (10-30) | unit sticking below v_coag is the practice; PAH+PAH van der Waals only; coagulated carbon joins the mantle, no release rate | confirmed; PAH+PAH now off for good; the pair-class rule is not in the code |
| growth by gas carbon | grains classical, carbon only; molecules reaction-limited, f_acc = 0 (10-30) | Omont & Bettinger 2025: C2H2 returned below ~50 C | confirmed |
| molecular carbon loss | C2H2 loss as a shift (pahchem tables) | < 50-60 C destroyed fast once below the threshold; products clusters/cages (C60 from 60-66 C) | confirmed as an approximation (bin 1's high rates empty it); fullerenes omitted; products below 60 C open |
| a-C photo-fragmentation | opt-in (P5), C4H_k to the gas | Jones & Ysard 2025 confirmed; matters only below ~150 C | overlaps the molecular bins: it applies only if the family's bare/aliphatic members are treated as a-C (ties to P1 and the bare end) |
| sputtering | Nozawa/Bocchio, a-C:H Serra Díaz-Cano & Jones; H in proportion or first (10-30) | classical valid only for N_C >~ 1000 (Bocchio+12); smaller: Micelotta+10b C2 loss; H-first unverified | revise: molecular sputtering for N_C < 1000 (bins 1-5) in hot gas; H-first stays open |
| sublimation | graphite vapour pressure as the a-C proxy | no a-C data | confirmed |
| Coulomb explosion | per-moment sink, grains removed (B2) | Tazaki+20: "fragments of smaller grains", sizes unknown | open: removal or fragments with the shattering law |
| hydrogen rule | every carbon move carries H at the source bin's H/C (10-30) | consistent (Murga+16) | stands; matters only with M_H on (decision B) |
| clusters | M2 option | dropped 11-23 | records still list them (report Sect. ladder, infographic, STATUS P4): update |

Stale records found: the `gsd/carbon.py` module docstring and `diagnostics/gsd/check_carbon_carriers.py` still describe
the 10-24 two-population layout (aromatic and aliphatic, each with N and M; `nc_ali_max`), superseded on 10-30; the
report says "The PAH bins and the smallest a-C bins may describe the same carriers" (TEX:1704).

## The central question: what is carbon at N_C 20-400? (D1)

**Correction (user, 2026-11-23): this was decided on 2026-10-30 and is still the design.** The molecular bins hold one
carbon population along a hydrogen coordinate, the hydrocarbon family: PAHs and their H-modified forms, from fully
dehydrogenated (bare carbon, a-C-like) through normal PAHs and super-hydrogenated states to aliphatic-rich, a-C:H-like
nano-particles (plan: carriers table and P1, N_H from 0 to ~1.4 N_C). So a species is a PAH unless its hydrogen makes it
a-C (none) or a-C:H (fully hydrogenated); the structure at equal H/C is a labelled assumption fixed by the network.
- Since 2026-11-15 the H state is tabulated in equilibrium (from pahchem's joint (Z, N_H) chain), not tracked: a bin's
  properties average over the equilibrium H distribution at the local conditions, so its identity follows the
  conditions, not the carbon's origin.
- Built: the joint chain from bare/dehydrogenated through normal to super-H (deep chain caps 8/12/24, pahchem PR #18
  unmerged; the super-H physical stop is an open decision), and the per-bin equilibrium tables
  (`pahchem/hydrogen_tables.py`).
- Not built: P1's a-C:H end (N_H up to ~1.4 N_C, aliphatic CH2/CH3 modes); the bare end uses pahchem's dehydrogenated
  states (carbon loss), not a-C grain physics.
- Consequence for the transfer channels: carbon entering a molecular bin (shattering fragments, eroded nano-grains)
  takes the family's equilibrium H state there; no "instant PAH" rule is needed beyond the size rule, but the
  structural assumption stays labelled.
- What remains open under this design: (i) building P1's a-C:H end; (ii) whether fully dehydrogenated molecular-size
  carbon should use a-C physics (Jones & Ysard 2025 photo-fragmentation, `ac` charging) rather than pahchem's bare
  states; (iii) the seam at N_C,max between the family's tables and `ac`/`acx`.

The options below (D1a-c) were written before this correction; D1a extended along the H coordinate IS the design.

Evidence: the identity of N_C ~ 50-400 carbon is unresolved. Size diagnostics constrain the emitter size (~40-100 C),
not planarity; PAH-database fits and THEMIS a-C fits both reproduce the bands. THEMIS (Jones+13/17, Jones 2025) has no
PAHs (3-D aromatic/aliphatic networks); most galaxy models call all small carbon PAH by size (Seok+14, Rau+19,
Narayanan+26, Parente+26) or carry an aromatic fraction (HM20, Narayanan+23). No lab or MD route turns a-C into planar
PAHs at fixed size; the known top-down routes go to fullerenes (Berné & Tielens 2012, Zhen+14, Micelotta+12).
Why it matters: the molecular-bin physics differs. PAH tables heat the gas 0.9-3.6x more per C than `ac` at the same
N_C (seam check, partly pahchem calibration: DFT IPs 0.4-0.6 eV low, cation recombination 3-30x high); PAH destruction
is pahchem's multiphoton network (C2H2 loss; < 50-60 C destroyed in PDRs and the diffuse ISM), a-C destruction is Jones
& Ysard 2025 photo-fragmentation (C_nH_k bridge loss; 40-60 C live 1e6-1e7 yr diffuse, 1e2-1e4 x shorter in PDRs).

Options:
- **D1a. Size rule, PAH identity** (current design): every bin below N_C,max uses the PAH tables, whatever the origin
  of its carbon. Simple, no extra state, the dominant galaxy-model practice. Its weak step: shattering fragments and
  eroded nano-grains become PAHs instantly (no lab/MD support; "the least defensible choice" of the published ones).
- **D1b. Size rule, a-C identity** (THEMIS-like): the molecular bins use a-C(:H) tables (Jones optics, `ac`/`acx`
  charging, JY25 photo-fragmentation). No PAH molecular physics in RAMSES; pahchem stays for post-processing/benchmarks.
- **D1c. Two populations below N_C,max** (PAH and a-C, each N and M): +8 scalars; needs an a-C -> PAH conversion rate
  that no study quantifies. Not recommended unless D1a and D1b both fail the A5 tests.
- Under a size rule, D1a vs D1b is a choice of TABLES for bins 1-4, not of state or code: both can be built, selected at
  run time and confronted with observations in A5 (T2-T4: band ratios, q_PAH trends, heating). That keeps D1 open at no
  cost to A3c.

## D2. Shattering and cratering: what fragments are and where they go
- Fragment law: KT10 ejected mass with Q*_D = P1/(2 rho) (graphite P1 as the a-C proxy, labelled), dN/da ~ a^-3.3
  (alternatives 3.0-3.5; minor effect below 3.5), a_f,max from HK13 Eq. 5. Lower cutoff: the grid bottom (0.39 nm, N_C
  20); mass below it to the gas (as carbon; the gas model holds elements).
- Identity: fragments are the parent's material (a-C; the CM20 mantle for large grains, Murga+16's layered choice);
  below N_C,max they take the molecular-bin identity of D1 (instantly, by the size rule).
- H state: the local equilibrium (decided), so no rule is needed; with M_H tracking on, fragments inherit the parent's
  H/C (Murga+16, the only published choice), then relax.
- Vaporisation above ~19 km/s (Tielens+94, Jones+96): omitted by all galaxy models; in the WIM large grains reach ~20
  km/s, so it is near threshold. Option: omit and label (no verified fraction-vs-velocity law found), or add the Jones+96
  law once read.
- Open: no P1/Q*_D/v_shat for a-C or a-C:H; aggregate (porous) grains break at ~1 m/s vs compact 1.2 km/s.

## D3. Coagulation: pair rules
- Molecular + molecular: no sticking (van der Waals only, ~1 eV per molecule vs > 4 eV chemical bonds; decided).
- Molecular + nano-grain, molecular + grain (freeze-out), nano + nano, nano + grain, grain + grain: unit sticking below
  the HY09/Chokshi v_coag, geometric kernel, Brownian + turbulent + drift velocities (the literature's practice; nothing
  better supported). The coagulated carbon joins the larger partner (THEMIS: the aromatic-rich a-C mantle); it is not
  released except by later fragmentation (no desorption rate exists).
- Coulomb factor: ours is charge-averaged over both grains' charge points (beyond the literature, which uses 1 or
  Murga+19's repulsion-only form); fed from the precool charges.
- How PAH-sized carbon moves up: only by sticking to a larger partner (freeze-out), and by accretion of gas carbon
  where enabled (off by default for molecules); a molecule grown past N_C,max becomes a nano-grain by the size rule.
  Freeze-out timescales: ~0.3-0.5 Myr in dense clouds, 10-17 Myr in molecular clouds, 100-140 Myr in the CNM (Seok+14,
  unit sticking); WD99 ~1e7 yr (1 km/s / v)(30 / n_H) for 1 nm.
- Option: molecular + nano-grain sticking off as well (only grains >= 20 nm collect molecules), if a PAH adsorbed on a
  nano-grain is judged to be photodesorbed at once; no rate known.

## D4. Erosion: routing per process
- Grain sputtering (N_C >~ 1000, a > 3 nm): classical yields; C atoms to the gas; the grain shifts down (remap); H in
  proportion (H-preferential sputtering unverified).
- Small particles (N_C < 1000) in hot gas and shocks: Micelotta+10b molecular approach (C2 loss from the periphery;
  destroyed at 1/3 or all carbon lost): shift down by C2 per event, below the grid to the gas. Needs the Micelotta rates
  (not in the code).
- PAH photolysis (D1a): H loss in the equilibrium tables; C2H2 loss as a shift down with pahchem's per-bin carbon-loss
  rates; bin 1's high rates empty it to the gas quickly, which stands in for the literature's "destroyed fast once below
  ~50-60 C". Fullerenes (C60 from 60-66 C, Berné+15) are not a species here: omitted, labelled.
- a-C photo-processing: aromatisation = the equilibrium band gap (decided); photo-fragmentation (JY25, opt-in): shift
  down, products (C_nH_k, n <= 5) to the gas. It matters only below ~150 C, i.e. in the molecular bins: under D1a it has
  almost no effect (nano-grains above 400 C are stable), under D1b it replaces pahchem's carbon loss there.
- Sublimation: grains to the gas (graphite vapour pressure as the a-C proxy); for molecules it is photolysis.
- Coulomb explosion: options: removal to the gas (current `charging.destroy`) or fragments with the shattering law
  (Tazaki+20: "fragments of smaller grains", sizes unknown). Recommended: fragments (mass-conserving), labelled.
- Cosmic rays: negligible (electrons, ~1e13 yr); ions only for shielded PAHs: omitted.

## D5. Growth
- Gas carbon onto grains: classical, the accreted carbon is a-C(:H) (a-C:H mantles at A_V < 3, ices above, Jones+13);
  sticking from Bossion+24 / Esmerian+26 (upper limit for a-C:H).
- Onto molecules: reaction-limited, off by default (decided 2026-10-30); for N_C < ~50 C2H2 is returned (Omont &
  Bettinger 2025).

## Timescales that decide whether rules can be instantaneous
- PAH H state: < 1 yr to ~1e4 yr (Montillaud+13): equilibrium tables fine except near transitions (decided).
- a-C aromatisation: >= 4e4/G0 yr at 3-10 nm (Jones & Habart 15) vs ~3e6/U yr at 1 nm (HM20; 20x apart); advection
  through PDR fronts 2e3-5e4 yr (Schirmer): the equilibrium band gap lags there (the A5 tracking test).
- Fragment production: ~30-44 Myr (WIM), 1-2 Gyr (CNM) for PAH-sized carbon (Seok+14).
- Freeze-out: 0.3-0.5 Myr (dense clouds) to 1e8 yr (CNM).
- Photo-destruction of < 50-60 C: fast in PDRs and the diffuse ISM; > 150 C survives HII regions (JY25 for a-C).

## Proposed quantitative checks (to inform D1-D4)
1. Population heating: the share of PE heating and recombination cooling carried by N_C < 400 carbon in the diffuse ISM
   and PDR cells, with PAH vs a-C tables: how much the identity choice moves the gas heating.
2. Timescale maps per bin (1-7) across the environment cards: fragment supply (shattering), destruction (photolysis vs
   JY25), freeze-out, aromatisation; which bins are supply-limited and which are in equilibrium.
3. One-zone runs on the full carbon grid with each option set (D1a/D1b; D3 nano sticking on/off; D4 Coulomb explosion
   routing): q_PAH, the small-carbon size distribution, heating, the carbon budget.

## Results of the quantitative checks (2026-11-23)

**Check 1, heating share** (`diagnostics/gsd/check_carbon_heating_share.py`, heating_share.txt/.pdf). THEMIS J13 carbon
(43 % of it below N_C = 400) in the seam cells: N_C < 400 carbon carries 70-76 % of the grain PE heating and 73-81 % of
the recombination cooling (ac tables). Charging it with the PAH tables instead of `ac` multiplies the total heating by
1.42 (CNM, dense) to 2.32-2.53 (WNM, PDR), the cooling by 1.33-2.18. With 10 % (20 %) of the carbon below N_C = 400 the
factor is 1.16-1.62 (1.27-2.01). The identity of that carbon, i.e. the seam calibration (which side moves), is a
first-order input of RAMSES's thermal balance.

**Check 2, timescales** (`diagnostics/gsd/check_carbon_transfer_timescales.py`, timescales.txt; yr):
- Shattering supply (WIM/WNM, graphite strength, KT10): every bin from 1 to 8 is refilled in 0.8-1.2e7 yr; large grains
  are lost in 2-7e6 yr.
- Freeze-out of the molecular bins onto grains alone (trace molecules): 1.3-2.5e8 (WIM/WNM), ~3e7 (CNM, Larson
  turbulence), 3-5e5 (dense cloud; Seok+14: 0.3-0.5 Myr).
- With the code as it is, molecules also collide with molecules under grain physics: loss of the molecular bins 2-7e6
  (CNM) and 2-8e4 (dense) with sticking, 2-5e7 (WIM/WNM) by shattering (PAH thermal speeds ~3 km/s at 8000 K exceed
  graphite's 1.2 km/s). Molecule + molecule collisions would dominate the molecular bins by 10-25x: the pair rule (no
  molecule + molecule outcome, molecules only stick to grains) is essential, not optional.
- PAH carbon loss (deep +8 tables): bin 1 (20-42 C) 2e6 (diffuse), 3e2 (PDR), 2e9 (dense); bin 2 6e9 (diffuse), 2e6
  (PDR); bins 3-4 effectively stable. Bin 1 is destruction-dominated in diffuse and PDR gas (supply ~1e7).
- a-C(:H) hydrogen relaxation (lab laws): 3e3 (diffuse), 3 (PDR), 3e5 (dense), against the HM20 aromatisation time 3
  dex slower (the known 3.5-dex spread). With the lab laws the equilibrium band gap is consistent in UV-lit gas
  (relaxation << supply and destruction); in dense gas it is comparable to freeze-out and dynamics (where tracking
  M_H, built in but off, would matter).
- Classical sputtering at 3e6 K, n_H 1: 3e3-1e4 for bins 1-6 (invalid below N_C ~1000; Micelotta+10b: PAHs live ~1e3
  yr at n_H 0.01, 1e7 K, i.e. faster): hot gas removes all small carbon quickly either way.

**Check 3, one-zone runs** (`diagnostics/gsd/check_carbon_transfer_options.py`, options_runs.txt). J13 start (molecular
46.3 %, nano-grains 1-20 nm 27.2 %, grains 26.6 % of the carbon); collisions (graphite strength, KT10, HY09 sticking,
neutral grains) and the molecular carbon loss; WNM and CNM for 30 Myr, a dense cloud for 1 Myr. Carbon shares at the
end:

| env | rule | molecular | nano 1-20 nm | grains >= 20 nm | gas | heating PAH/ac |
|---|---|---|---|---|---|---|
| WNM | O0 code as is | 0.283 | 0.165 | 0.002 | 0.550 | 2.77 |
| WNM | O1 no mol+mol, mol+grain stick only | 0.348 | 0.309 | 0.014 | 0.329 | 2.62 |
| WNM | O2 = O1, mol stick only to >= 20 nm | 0.348 | 0.309 | 0.014 | 0.329 | 2.62 |
| CNM | O0 | 0.038 | 0.542 | 0.256 | 0.164 | 1.10 |
| CNM | O1 | 0.087 | 0.444 | 0.259 | 0.210 | 1.25 |
| CNM | O2 | 0.246 | 0.273 | 0.261 | 0.220 | 1.57 |
| dense | O0 | 0.004 | 0.670 | 0.326 | 0.000 | 1.01 |
| dense | O1 | 0.028 | 0.649 | 0.323 | 0.000 | 1.07 |
| dense | O2 | 0.398 | 0.251 | 0.351 | 0.000 | 1.43 |

Findings:
- O0 (molecules colliding with molecules under grain physics) sweeps the molecular carbon away: in warm gas by PAH-PAH
  "shattering" (gas 55 % against 33 %), in cold gas by PAH-PAH sticking into nano-grains. The pair rule is needed.
- **Whether molecules stick to nano-grains (O1 against O2) decides how much PAH-sized carbon survives in cold and
  dense gas: x2.8 in the CNM after 30 Myr, x14 in a dense cloud after 1 Myr.** Nano-grains are numerous, so with unit
  sticking they collect the molecules much faster than the big grains do. In warm gas it makes no difference.
- In the HY09 WNM, the grains >= 20 nm are shattered almost completely in 30 Myr under every rule (the known
  aggressiveness of that turbulence; A0's sn_diffuse case uses it).
- Limits: no Coulomb factors (PAH-grain charges would change sticking in warm, ionised gas), no accretion or re-supply,
  J13 start, HY09 turbulence, graphite strength for a-C.

## Decisions this study puts to the user
1. **The seam (identity of N_C < 400 carbon):** which side moves (pahchem thresholds and recombination, or `ac`);
   first-order for the gas heating (check 1). STATUS decision 5, now urgent.
2. **Molecule + nano-grain sticking:** O1 (unit sticking, the literature's practice) or O2 (molecules stick only to
   grains >= 20 nm; a PAH adsorbed on a nano-grain is van der Waals bound like a cluster and would evaporate in UV light,
   but stays in shielded gas). A physically founded middle: sticking times survival against photo-evaporation
   (depends on binding energy and G0; needs the cluster-evaporation physics the user dropped).
3. **Pair rule (O1 core):** no molecule + molecule outcome; molecules never shatter or crater; confirmed by checks 2-3.
4. **Smaller:** molecular sputtering (Micelotta) for N_C < 1000 in hot gas; Coulomb explosion as removal or fragments;
   vaporisation omitted; P1's a-C:H end and the bare-end physics; redo the band test (missing evidence).

## Status of the decisions (2026-11-23)
- Pair rule (no molecule + molecule outcome; molecules never shatter or crater, only stick to larger particles):
  adopted (follows from dropping clusters; checks 2-3).
- Molecule + nano-grain sticking: pending; the user asked first for the observational constraints on how much
  PAH-sized carbon survives free against frozen out or lost, versus density and phase (literature check running).
- The seam: the user chose to study it next, before A3c.

## Observational constraints on PAH survival against the sticking rules (lit_survival.md)
- Gate 1, phase parity (Chastenet+19, LMC/SMC, GMC scale, n_H ~1e2-1e3): q_PAH(molecular)/q_PAH(diffuse) = 1.0-1.05
  (+-30 %); JWST nearby galaxies (Chastenet+23): PAH abundance insensitive to molecular fraction (a few tens of %);
  translucent clouds (Zhang+25): no drop from A_V 1 to 3, maybe x2 growth.
- Gate 4, dense filaments and cores (n_H >~ few 1e3, A_V >~ 2-3, low field): small-grain mid-IR emission falls by a
  factor of a few up to 80-100 % (Stepnik+03, Flagey+09, Ysard+13, Köhler+15), degenerate with field attenuation;
  Köhler+15's 1500 cm^-3 threshold is an assumption, not a measurement.
- Others: frozen PAHs <~12-15 % of C on YSO sightlines (Chiar+21); TMC-1 small PAHs survive at >= 0.03-1 % of C.
- Against check 3 (bins 2-4, i.e. without bin 1's photolysis): CNM 30 Myr O1 -72 %, O2 -21 %; dense 1 Myr O1 -94 %,
  O2 -14 % (bins 1-4). O1 conflicts with parity unless re-supply or release offsets it; O2 passes parity and may give
  too little loss in cores (where the observed drop is partly attenuation and partly the nano-grains themselves).

## Decided (user, 2026-11-23): UV-dependent sticking of molecules onto nano-grains (labelled)
- Molecule + grain >= 20 nm: unit sticking (freeze-out into the mantle), everywhere.
- Molecule + nano-grain (1-20 nm): sticking only where the adsorbed molecule survives photo-release, from a release rate
  built on the van der Waals binding energy and the photon heating of the molecule + nano-grain aggregate (statistical
  evaporation, as for clusters). A labelled assumption until derived; inputs gathered next (binding energies, release
  physics, any incorporation process). It reproduces the observed pattern: PAHs kept in UV-lit cloud envelopes, lost
  in dark cores.
- Molecule + molecule: no outcome (pair rule, adopted).

## The UV-dependent sticking, worked through (lit_release.md)
Binding energies ~0.05 eV per C (Zacharia+04 TPD; DFT agrees): ~1.2 eV for C24, 2.6 eV for C50. A UV photon heats a
1 nm host + PAH to ~400-640 K but a 3 nm host only to ~100-140 K: release is a single-photon, small-PAH, ~1 nm-host
effect (C24 always released from 1 nm hosts; C50 stays below G0 ~ 0.1-1e3, very uncertain); on hosts >= 3 nm a PAH
stays at any G0 below ~1e7 (agent's estimates; MJ14's cluster photostability and Lange+23's "> 54 C stay frozen out"
point the same way). So the decided rule acts almost like unit sticking onto nano-grains >= ~3 nm (O1 for bins 6-7,
partial for bin 5), and by check 3 that conflicts with molecular-cloud parity unless something else releases or
re-supplies PAHs (none quantified: no photodesorption yield, no incorporation rate).

## Seam step 1: decomposition with today's pahchem (seam_decomposition.txt, check_seam_decomposition.py)
Four-state chains reproduce the full solves (PAH 1.00, ac 1.00-1.11). Heating PAH/ac per C: bin 1 0.91-2.71, bins 2-4
1.86-3.60. Factor on the ac heating when one ingredient is taken from the PAH table:
- photoemission rate (absorption x yield; the split is spectral bookkeeping: PAHs absorb ~4x more per C in the Mathis
  field, mostly non-ionising photons): x0.95-1.4;
- energy per photoelectron: x0.68-1.27;
- **electron capture: x1.34-2.70 in bins 2-4 (the dominant ingredient); bin 1 x0.62-1.69** (PAH neutral attachment lower);
- ion channels: x1.0-1.16.
The capture gap is the electron sticking on cations: both use Draine & Sutin focusing (with the image charge) on similar
effective radii (pahchem's disc x0.82 vs ac's sphere), but pahchem's lab-fitted sticking (lab_cold) saturates at 1 by
N_C ~ 35 while WD01's grain sticking 0.5(1 - exp(-a/1 nm)) gives 0.2-0.3 for these sizes. Both fit the lab cation
recombination (N_C 6-16) within ~2x and diverge beyond it (alpha(+1, 300 K): PAH 1.5-4.2e-5, ac 3.4-9.5e-6 for N_C
35-216). Evidence beyond the lab range: the measured/ac ratio rises with N_C (0.5 at <= 12 C to 1.2-2 at 14-16);
C60 neutral attachment s_e ~ 0.8 (Viggiano+10) against WD01's 0.19; statistical (molecular) capture of the electron's
energy argues for high sticking in molecules, WD01's electron escape (l_e = 1 nm) is a grain argument.

## Release channels and electron sticking (lit_release_channels.md, lit_sticking.md)
- No quantified channel releases a PAH from a >= 3 nm host. The one that could act: non-thermal UV photodesorption,
  unmeasured for PAHs on carbon or ice; in the CNM (one absorption per ~0.5 yr for a C50) a yield of only ~1e-7 per
  absorbed photon balances freeze-out, while dense clouds would need ~1e-3 (unlikely). A small yield would thus give the
  observed pattern by itself: PAHs kept in UV-lit gas, lost in dark cores. Turbulent mixing (~10-30 Myr, assumed) and CR
  track sputtering (~30-100 Myr, estimate) are slower.
- Electron sticking on PAH-sized cations: no measurement beyond N_C 16; pahchem's saturation at ~1 (by N_C ~35) and
  WD01's 0.2-0.3 are both extrapolations (WD01's 0.5 ceiling and l_e = 1 nm are assumptions; its cation law has no
  molecular term). Only C60 neutral attachment (~1, Viggiano+10) lies above N_C 16. Range 0.3-1 at N_C 35-400, ~x3:
  the seam's capture gap (x1.3-2.7) lies inside it.

## Decided (user, 2026-11-23, after the release and sticking reviews)
- **Molecule + nano-grain sticking with photo-release:** S_eff = 1 / (1 + R_abs (P_th + Y_pd) t_bury), where R_abs is the
  adsorbed molecule's photon absorption rate, P_th the single-photon thermal release probability (derived; matters only
  for small PAHs on ~1 nm hosts), Y_pd the non-thermal photodesorption yield (unmeasured: an explicit input labelled
  "calibrated", default 0 until A5 calibrates it against molecular-cloud parity and dense-core depletion, expected
  1e-7 to 1e-6), t_bury the host's burial time (its accretion/coagulation). Molecule + grain >= 20 nm: unit sticking.
- **Seam electron sticking: one labelled law, bracketed.** Molecules keep pahchem's lab-fitted sticking (lab_cold);
  'ac' grains take the same value at N_C 400 and taper to WD01's 0.5 (1 - exp(-a/1 nm)) at larger sizes (taper scale
  a labelled choice, ~3 nm). A5 runs the brackets (WD01 everywhere; the molecular value everywhere) to show the
  heating sensitivity.

## Still open from this study
- PAH sputtering by ions/electrons for N_C < ~1000 in hot gas (Micelotta+10b, C2 loss) instead of classical yields.
- Coulomb explosion: removal to the gas or fragments with the shattering law.
- P1's a-C:H end of the hydrocarbon family and the bare-end physics (pahchem bare states or a-C).
- Redo the band test (3-4 bins over N_C 20-400, evidence missing) with the extended N_C,max test (400 vs ~1000).
- Labelled omissions: vaporisation above ~19 km/s, fullerenes, cosmic-ray processing.

## Against galaxy-model practice (2026-11-23, user's question)
- One-zone WNM, 30 Myr, turbulence-only velocities (Brownian off, as most galaxy models): all pairs colliding
  (O0) leaves molecular 0.298 / nano 0.167 / gas 0.533, against 0.283 / 0.165 / 0.550 with Brownian; our rule (O1)
  0.348 / 0.309 / 0.329 either way (options_runs_nobrownian.log). So in warm gas the extra loss of O0 is not PAH-PAH
  collisions: it is molecules shattered as targets (and cratering) in collisions with fast grains, under graphite's
  bulk strength (Q*_D ~ 0.1 eV per C, far below molecular bond energies).
- CNM and dense runs without Brownian fail: pairs with zero relative velocity make velocities.py divide by zero
  (u = v/sigma with sigma = 0; overflow at line 77). Side finding, not fixed.
- Implication: "molecules never shatter" is too strong for fast impacts. A molecule hit by a large grain at several
  km/s receives hundreds of eV; fullerene/PAH surface-impact experiments suggest molecules survive up to ~100-200 eV
  and fragment above (to verify). A molecular destruction threshold (stick below, destroyed above) would sit between
  O1 and O0 (closer to O1, since molecules are far harder than graphite's bulk strength implies).

## Molecule-grain impacts (molecule_grain_speeds.txt, options_runs_O3.*; literature: C60 and cluster surface impacts)
- Data (abstract level; no PAH-specific measurements): C60 rebounds intact from graphite and diamond up to ~100-250 eV
  (MD onsets ~100-200 eV; Busmann+91/93 intact scattering 140-450 eV with metastable fragmentation; Kolodney: sharp onset
  ~150 eV on gold), i.e. ~2-4 eV per atom; cluster deposition: soft landing < ~0.1 eV/atom, break-up from ~1 eV/atom.
  Products: C2 loss, then statistical multifragmentation away from the surface. A flat PAH physisorbs at ~0.07 eV per C.
- Rule (inference, labelled): stick below ~1 eV per atom (~4 km/s, size-independent), rebound intact between, destroyed
  above ~3 eV per atom (~7 km/s; 5-10 km/s for a factor 2), carbon to the gas. ~30x the bulk Q*_D of the shattering law.
- Speeds: molecules against grains >= 20 nm, rms 9-30 km/s in WIM/WNM (80-100 % of the flux above 7 km/s); 0.05-1.7
  km/s in CNM and dense gas (no flux above 4 km/s). Destroyed in warm turbulent gas, frozen out in cold gas.
- One-zone with this rule (O3, no release term): WNM 30 Myr molecular 0.259 / nano 0.381 / grains 0.010 / gas 0.349
  (O1: 0.348 / 0.309 / 0.014 / 0.329); CNM as O1. The destruction itself adds only ~0.02 of the carbon to the gas in 30
  Myr (molecule-large grain collisions are rare: ~1e8 yr), a minor sink next to photolysis (bin 1: 2 Myr) and shattering
  supply (~1e7 yr). The larger change is more sticking onto nano-grains in warm gas (the 4 km/s trapping limit is more
  permissive than the grain-grain Chokshi threshold), which the decided photo-release term would largely undo in UV-lit
  gas.
- Answer to the user's question: PAH abundances are set by the large grains' shattering (supply), photolysis, freeze-out
  (cold gas) and hot-gas destruction; molecule-molecule collisions (excluded) and molecule destruction by fast grains
  (minor) do not control them. The galaxy-model channel worth keeping is the velocity-dependent molecule-grain rule.

## Decided (user, 2026-11-23): velocity-dependent molecule + grain rule (labelled)
Molecule + non-molecule collisions: stick below ~4 km/s (~1 eV per atom; onto nano-grains still subject to the
photo-release term), rebound intact between, molecule destroyed above ~7 km/s (~3 eV per atom; carbon to the gas);
thresholds labelled inputs with per-atom (size-independent) scaling, destruction bracketed 5-10 km/s in A5. Molecule +
molecule: no outcome. Replaces "molecules never shatter" and the unconditional freeze-out onto grains >= 20 nm.
