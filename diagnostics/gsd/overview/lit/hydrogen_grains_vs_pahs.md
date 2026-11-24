# Why small a-C(:H) grains are tracked while PAH hydrogenation is tabulated: what the literature says

Literature reading by a Sonnet subagent, 2026-11-17, for the GSD overview (point 3 of the user's review). Saved by the
main session (subagents cannot write report files). [S] = stated in the paper, [I] = the subagent's inference.
Papers: worktree `reference_data/ac_photochem_2026-10-29/` and `results/carbon_plan/pah_hydrogenation_review/papers/`;
Montillaud+13 and Andrews+16 from `pdr_refs/literature/pah_gas/`.

Identification: 1511.01673/82/84 Jones 2012 I/II/III; 1411.6293 Jones+13; 1411.5877 Jones+14 (Faraday); 1507.08534
Jones & Habart 15; 1703.00775 Jones+17 (THEMIS); 1612.00419 Murga+16; 1906.11308 Murga+19 (Shiva); 2001.01968
Hirashita & Murga 20; 2212.04993 Murga+23 (HAC); 2209.04282 Schirmer+22; 2003.05902 Schirmer+20; 2401.01221
Elyajouri+24; 2510.24573 Elyajouri+25; 2511.04750 Jones & Ysard 25; 1502.04941 Pilleri+15; 1404.2107 Montillaud &
Joblin 14.

## Headline

The literature does not state "PAHs in equilibrium, small a-C(:H) not". On the PAH side it says nearly the opposite:
Montillaud+13 warn that large PAHs may never reach steady state; Andrews+16 assume steady state without testing it.
For a-C(:H), the Jones/THEMIS group states that PDR dust is out of equilibrium and must be followed in time; Murga's
Shiva and HAC codes do so. The only equilibrium treatment is the galaxy-scale one-zone model of Hirashita & Murga 20.
No paper gives a size criterion for when the H state relaxes faster than the dynamics.

## 1. a-C(:H): equilibrium or time-dependent?

- **Jones 2012 III, §5.2, Fig. 18.** [S] "Carbonaceous dust evolution must therefore be treated dynamically ... also
  time-dependently in terms of the evolution of its chemical composition and structure". τ_UV,pd × G0 ≈ 1e5 yr for
  10–100 nm and ≳ 1e6 yr for a ≲ 1 nm (ε = 0.1); ≈ 1e3 (1e2) yr at G0 ≈ 1e3 (1e4). Thermal processing during
  stochastic heating "could be important but has not yet been evaluated". Re-hydrogenation not discussed.
- **Jones 2012 II, §5.1–5.2.** [S] τ_UV,pd ≈ 1e4/(Q_abs ε) yr, ≈ 1e5 yr in the diffuse ISM (σ_CH = 1e-19 cm² over
  10–13.6 eV, ε = 0.1). H loss stops at X_H ≈ 0.05 (E_g ≈ 0.2 eV), "a natural block". H re-addition "not clear how and
  under what circumstances".
- **Jones+13, §7.1, §7.5.** [S] Aromatisation ≃ 1e5/G0 yr in the diffuse ISM, so small particles (≲ 20 nm) are a-C.
  Footnote 13: particles ≲ 10 nm undergo temperature excursions that drive dehydrogenation, "time-scales not yet
  determined". §7.5: "The dust processing time-scale is generally longer than the local dynamical time-scale and so the
  dust will therefore not be in equilibrium with the local conditions."
- **Jones+17 (THEMIS), §1.** [S] In diffuse and dense clouds "the dust is generally in equilibrium with the local
  conditions"; "in PDRs and shocked regions this is no longer the case ... the dust is often out of equilibrium and
  constantly evolving".
- **Jones & Habart 15, Eqs. 4–8.** [S] τ_UV,pd(a) = (1e4/G0){2.7 + 6.5/a^1.4 + 0.04 a^1.3} yr (a in nm): minimum
  ≈ 4e4/G0 yr at 3–10 nm. ε = 1 below 2 nm. Thermal H2 desorption not included.
- **Jones+14 (Faraday), Eqs. 10–11.** [S] H incorporation t_Hin = [ξ Σ n v S]^-1 with ξ ≪ 1. For A_V < 0.01–0.7
  photolysis wins and small grains are "rapidly transformed (t_pd ≥ 4e4 yr) to aromatic-rich a-C"; cores "cannot be
  re-hydrogenated" in the diffuse ISM.
- **Murga+16.** [S] One-way photo-processing, no re-hydrogenation: grains with N_C < 2e4 (a < 35 Å) lose essentially all
  H in several 1e5 yr at U = 1.
- **Murga+19 (Shiva).** [S] Band gap followed in time (CVODE), Y_CH = 1 below 20 Å, E_g floor 0.1 eV. Multiphoton RRK
  only for dehydrogenated/PAH-like grains, for C loss. [I] No H accretion channel.
- **Hirashita & Murga 20.** [S] "the aromatization time-scale is much shorter than the other relevant processes ...
  the aromatic fraction converges to an equilibrium value quickly" — at galaxy scale, two-phase exchange 1e7 yr.
  τ_al,H = 1.6e5 (a/µm) yr "based on Murga+19" (origin not traceable).
- **Murga+23 (HAC).** [S] Time-dependent, 20 mass × 5 H bins, at fixed PDR conditions and ages. Re-addition efficiency ξ
  "undetermined", scanned 0–1. Alata yields extended to 13.6 eV. No comparison with steady state.
- **JWST.** Elyajouri+24: static E_g fits per Orion Bar zone, 0.03 eV in the atomic zone (below the 0.1 eV floor);
  "equilibrium between formation and destruction ... beyond the scope". Elyajouri+25: Horsehead edge E_g = 0.1 eV.
  Schirmer+20/22 compare processing times with advection times L/v (1.8e3 yr Orion Bar, 2e4 IC 63, 5e4 Horsehead).

## 2. Thermal (multi-photon) channels for nano-grains

- Jones 2012 II/III and Jones+13 flag thermal processing of small grains as important but unevaluated; Jones & Habart 15
  omit it.
- **Jones & Ysard 25, §4.** [S] k_diss = k0 exp(−E/kT), k0 = 1e16 s^-1, E = 6.4 eV (C4H_k bridge loss), k_IR = 0.02 T
  s^-1; lifetimes 1e6–1e7 yr in the diffuse ISM for 40–60 C, "likely 1e2–1e4 times shorter in PDRs". C–C, not C–H loss.
  [I] With a C–H energy (3.1 eV) the same machinery gives ~10 % H loss per ≥ 10 eV photon for sub-nm grains, negligible
  for ~8 eV photons: a plausible speed-up for the smallest bin, not quantified in the literature.
- **Murga+23.** [S] Thermal desorption of H, ν exp(−E/T_d) averaged over P(T_d), E_H ≈ 3.1 eV (mantle H).
- [I] No a-C(:H) H-state calculation except Murga+23 has a thermal channel; it would shorten the relaxation of the
  smallest grains.

## 3. PAHs: steady state or time-dependent?

- **Montillaud+13.** [S] "Steady state for hydrogenation is reached on timescales ranging from less than a year for
  small PAHs, up to 1e4 years for large PAHs at A_V = 1." "These timescales become long when multiphoton events are
  required to dissociate PAHs, as each dehydrogenation step behaves as a limiting step". "large PAHs may never reach
  their steady state abundances". G0/n_H cannot parametrise the problem. N_H ≤ N_H0 + 1; C24–C96.
- **Andrews+16.** [S] "We solve the system of equations assuming steady state". No timescale validation. Rates: cations
  1.4e-10 cm³ s^-1 per step; neutrals with Rauls & Hornekær barriers; σ_ER = 0.06 Å².
- **Murga+22.** Time-dependent at fixed environments for 1e5 yr; no steady-state comparison; ⟨X_H/X_H0⟩ 1.1–1.2.
- No PAH paper shows that equilibrium holds for the state.

## 4. Why a molecule and a nano-grain relax differently

- **H inventory.** A PAH exchanges ~10–20 edge H (N_H0 to N_H0 + 1…+4); an a-C(:H) grain carries H on a large fraction
  of its C, X_H 0.62 → 0.05, a bulk process.
- **Per-photon loss.** PAHs: multiphoton/threshold-limited, very small per photon except rare events (M13 Fig. 8).
  Grains: bond-by-bond lab yields (Y_CH = 1, ε = 0.1 in Jones/Murga; 0.02 per photon from Alata).
- **Re-addition.** PAH cations 1.4e-10 cm³ s^-1 per molecule (≈ 6e-12 per C for coronene); the a-C(:H) σ_f =
  1.9e-18 cm² (Mennella+02) is a net incorporation efficiency, 1e3–1e4 below geometric. [I] For PAHs the slow step is
  photo-loss near the transition; for grains it is re-addition.
- **Definition.** Jones's τ_UV,pd is a one-way per-bond photolysis time; pyCALIMA's 2900 yr (1–3 nm, G0 = 1) is a
  relaxation time with re-addition, ~20–30× shorter than Jones & Habart's ≈ 6.5e4/G0 yr at 1.5 nm. [I] Likely the photon
  band: 6.8–13.6 eV (Alata) vs 10–13.6 eV (Jones). To check before quoting.

## 5. Synthesis

Consistent with caveats, not literature-endorsed:
- Supports: Jones/THEMIS say PDR dust is out of equilibrium; Murga follows a-C(:H) H in time; the only equilibrium
  treatment is galaxy scale; one-way processing with a floor has memory.
- Must be argued: (1) the PAH state is not shown to be in equilibrium — the justification is the observables test, so
  say "tabulated observables", not "PAHs in equilibrium"; (2) different criteria (PAH observables vs grain state); H2 and
  3.4/3.3 do not converge for large PAHs in shielded gas; (3) missing thermal H loss for grains could move the smallest
  bin toward tabulation; (4) re-addition weakly constrained (ξ undetermined); (5) Elyajouri+24's E_g = 0.03 eV breaches
  the floor; (6) tables must be multi-axis (M13 §6.2), which ours are; (7) PAH/a-C continuity: the PAH bins and the
  smallest a-C bins may be the same carriers.

Open items: a thermal H-loss sensitivity run for the 1–3 nm bin; the origin of Hirashita & Murga's τ_al,H; reconcile
2900 yr with Jones's 6.5e4/G0 yr.
