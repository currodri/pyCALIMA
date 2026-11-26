(physics-pah-hydrogen)=
# Hydrogen of PAHs: tabulate or track?

The molecular bins of the carbon grid (N_C 20–400) could store a hydrogen mass $M_{\rm H}$ next to
$M_{\rm C}$ and $N$, as the covalent bins do ({doc}`grain-hydrogen`). A PAH's hydrogen, though, is a
distribution over charge $Z$ and hydrogen count $N_{\rm H}$, not one number. This page answers three
questions for RAMSES:

- Is the equilibrium enough (**E**)?
- Is one tracked scalar, $\langle N_{\rm H}\rangle$ with a closure, enough (**M**), or neither (**F**)?
- Does the charge need to be solved with the hydrogen, or can charge tables ignore the hydrogen state?

It covers the PAH half of the plan's P2 tests.

## Networks and tools

**The networks** are pahchem's own, `baseline.BaselineModel` with the default `Params`. Each is the
full $(Z, N_{\rm H})$ network:

- RRKM H and H₂ loss from West et al. (2018), in a multiphoton cascade;
- states from bare to $N_{\rm H,0}+4$, with Andrews et al. (2016) H addition and Eley–Rideal
  abstraction;
- thresholds away from $N_{\rm H,0}$ from B3LYP ΔSCF ladders (`Params.threshold_shifts = "dft_ladder"`,
  the default since 2026-11-11; see Caveats for the earlier +0.1 eV per missing H);
- charges from the anion up.

Three parents stand for three of the four molecular bins of the default `CarbonLayout`:

| bin | N_C | parent |
|---|---|---|
| 1 | 20–42 | C₂₄H₁₂ |
| 2 | 42–89 | C₅₄H₁₈ |
| 3 | 89–189 | C₉₆H₂₄ |
| 4 | 189–400 | no parent of its own |

**The tools** are in `pahchem.hydrogen_tracking`:

- relaxation rates;
- settling times after a step in the conditions;
- charge elimination, which averages the hydrogen rates over the charge distribution at each $N_{\rm H}$.
  The FC0 variant uses the intact molecule's charge for every $N_{\rm H}$, which is what `population`
  does and what a charge table with no $N_{\rm H}$ axis implies;
- the one-scalar closure. It rebuilds the distribution from $\langle N_{\rm H}\rangle$ as the steady
  state of the same network with H addition rescaled.

The tests are in `diagnostics/gsd/check_pah_hydrogen_tracking.py`.

## Gates

**Montillaud et al. (2013), Fig. 9.** The test molecule is C₆₆H₂₀ in NGC 7023 NW, run through
`montillaud13` with their card.

- **Charge** settles in 5–16 h, against their "few hours".
- **Hydrogen, with their inputs:** C₆₆H₂₀ stays intact. Our multiphoton H loss needs ~10× more UV than
  theirs; this is the Fig. 10 offset, `check_montillaud13_fig10.py`. The intact population settles in
  10–100 yr.
- **Hydrogen, with the transition placed where theirs is** (stripped modes, G0 × 2.5): stripping takes
  10² yr at $A_V$ = 0 and 10⁴ yr at $A_V$ = 1, their numbers.
- **With their cation + H₂ reaction:** 60–100 yr, against their 10²–10³.

Their long times are those of a parcel sitting near the dehydrogenation transition. Each H lost there
needs a multiphoton event.

**Murga et al. (2022), model b**, read straight from Figshare, starting from the normal state:

- Their own populations have settled by 10⁴ yr in PDR 1, PDR 2 and the DF.
- In their molecular-cloud locations, 3–8 % of the change is still to come at 10⁵ yr (e-folding
  6×10⁴–2.5×10⁵ yr), and 10 % for the 50 Å grains.

## T1: how fast charge and hydrogen relax

Conditions: Draine field, 100 K, $n_e = 1.4\times10^{-4}\,n$(H) (`_T1.png`).

- **Lit PDR gas.** Far from its transition, a PAH's hydrogen settles within ~10 yr of a ×10 step in G0.
- **At the transition** the settling time grows roughly as $1/n$(H): $\sim10^4$ yr at $n$(H) = 10,
  $\sim10^2$ yr at $10^3$, and ~1 yr at $10^5$. This is the slowing that Montillaud et al. describe.
  - C₂₄ dehydrogenates at G0/$n$(H) ≈ 10⁻², which includes the CNM: C₂₄ is mostly bare there.
  - C₅₄ dehydrogenates at ≈ 30–100.
  - C₉₆ dehydrogenates at ≈ 10³–10⁴.
- **Shielded gas** (G0 ≲ 10⁻², $n$(H) ≲ 10): the super-hydrogenated states take 10⁴–3×10⁴ yr.
- **Charge** relaxes in hours in PDRs, months in the CNM, and up to 10³ yr in dark gas.

## T2: can charge be eliminated?

The question is whether the hydrogen rates can be averaged over a charge distribution computed
without them. That holds wherever charge is faster than the hydrogen chemistry. It fails in a band
at G0/$n$(H) ≈ 10⁻⁵–10⁻³ (the third row of `_T1.png`). There, H addition to neutrals and Eley–Rideal
abstraction on the super-hydrogenated states run faster than charge exchange:

- the charge and the hydrogen state are then correlated;
- the steady state of the fast-charge chain misses the full network's by up to 0.86 for C₂₄, 0.34 for
  C₅₄ and 0.46 for C₉₆, mostly in the super-hydrogenated share.

The equilibrium of the full network is not affected. Two conclusions follow:

- An equilibrium table must come from the full network offline. It cannot be rebuilt in RAMSES from
  charge-averaged rates.
- On the charge tables' $N_{\rm H}$ axis: the intact-state charge (FC0) adds a cation-fraction error of
  0.07 for C₂₄ at the NGC 7023 edge (0.08 over the T1 grid where it is bare), and ≤ 0.03 for C₅₄
  and C₉₆.

## T3–T6: parcels

Parcels are carried at 0.3, 1 and 3 km/s into and out of three PDRs:

- NGC 7023 NW (the Meudon profile, digitised);
- the Orion Bar (isobaric) and Horsehead (Murga et al. 2020).

The reference is the full network with conditions held over 50 $A_V$ cells. It is compared with the
equilibrium (E), the closure (C), and RAMSES-like steps of the closure with its rate frozen at each
step (exp). The worst absolute error over $x=\langle N_{\rm H}\rangle/N_{\rm H,0}$ and the shares
(intact, bare, super-hydrogenated, cation) is:

| parent | where | E | C, exp |
|---|---|---|---|
| C₅₄, C₉₆ | every path | ≤ 0.002 | ≤ 0.002 |
| C₂₄ | NGC 7023, Horsehead, CNM ↔ WNM | ≤ 0.02 | ≤ 0.005 |
| C₂₄ | deep Orion Bar (G0 1–100 at $n_{\rm H}\sim10^6$) | 0.23–0.51 | 0.23–0.34 (the super-H share) |

- **The step** adds ≤ 0.03 to the closure's error, for steps of 10²–10⁴ yr.
- **C₅₄ and C₉₆** stay intact through all three PDRs under this physics. Their transitions lie at
  G0/$n$(H) above the PDR fronts' 0.2–0.3.
- **C₂₄ in the deep Orion Bar.** Bare C₂₄ meets a high-density layer near its transition and is carried
  through it faster than it re-hydrogenates. The closure follows $x$ but not the share of
  super-hydrogenated states (F).

**Cycling across a transition.** G0 alternates by ×10 across each molecule's own transition, at T1's
conditions:

- C₅₄: G0 10² ↔ 10³ at $n$(H) = 10;
- C₉₆: G0 10³ ↔ 10⁴ at $n$(H) = 1.

| parent | period | E | C |
|---|---|---|---|
| C₅₄ | 10³ yr | 0.89 | 0.89 |
| C₅₄ | 10⁴ yr | 0.64 | 0.64 |
| C₅₄ | 10⁵ yr | 0.00 | 0.00 |
| C₉₆ | 10³ yr | 0.92 | 0.52 |
| C₉₆ | 10⁴ yr | 0.55 | 0.52 |
| C₉₆ | 10⁵ yr | 0.24 | 0.24 |

The hydrogenation is all-or-nothing. The true population stays intact (or stripped) and moves at the
rate of its first multiphoton step, $\sim 10^3$–$10^5$ yr here.

- **The closure** rebuilds intermediate distributions that relax orders of magnitude faster. It does
  no better than the equilibrium.
- **A two-basin closure was tried and dropped.** It took the basins from the network's slowest mode and
  relaxed their mixing weight at that rate. It gave 0.70 for C₅₄ and no gain for C₉₆: there are several
  slow modes along the ladder, not one intact ↔ stripped exchange.
- **Near a transition, one scalar is not enough (F)** when the conditions change faster than ~10⁵ yr.
  It would take a second moment, such as the intact share beside $\langle N_{\rm H}\rangle$.

## T5: mixing

Two parcels are mixed 1:1, and the closure's distribution at the mean $\langle N_{\rm H}\rangle$ is
compared with the true mixture:

- **Bare + intact:** off by 0.30–0.49 in the bare share. That lasts 60–600 yr in the CNM.
- **PDR edge + cloud material:** only the cation fraction is off (0.10–0.20), and only for hours.

## Treatment map (PAHs, pahchem baseline physics)

| bins | where | treatment |
|---|---|---|
| 2–3 (C₅₄, C₉₆) | PDR fronts and shielded layers (NGC 7023, Orion Bar, Horsehead), diffuse phases | **E** (errors ≤ 0.002) |
| 2–3 | within ×10 of their transitions (G0/$n$(H) ≈ 30–100 for C₅₄, 10³–10⁴ for C₉₆) when conditions change on ≲ 10⁵ yr | **F** (E and one scalar off by 0.2–0.9) |
| 1 (C₂₄) | lit PDR layers, the Horsehead, CNM ↔ WNM | **E** (≤ 0.02) |
| 1 | dense transition layers (deep Orion Bar) | **F** for the super-H share; E and C off by 0.2–0.5 |

For RAMSES this means:

- No $M_{\rm H}$ for the molecular bins. A tracked scalar does not fix the F cases: one scalar does no
  better than the equilibrium, and the F cases are confined to transitions crossed faster than they
  relax.
- **Equilibrium tables of the full network** for what RAMSES needs from the hydrogen state: the bare share
  (carbon loss), H₂ formation, and the charge thresholds.
- **The charge tables** need a dehydrogenated node for bin 1 only.

## The bins with the full pahchem model

The tests above use one `BaselineModel` parent per bin. The RAMSES bins are mixtures of molecules, and their
physics is `Population`'s. `diagnostics/gsd/check_pah_bins_hydrogen.py` builds each bin as pahchem does:

- PAHdb v4.00, pure aromatic CH, from 21 C;
- the calibrated threshold shifts;
- the molecules with charges −1…+2, at equal weights;
- default `Params`.

Each molecule's hydrogen is `Population`'s chain (`hydrogen_tracking.population_chains`). That chain carries the
intact molecule's charge for every $N_{\rm H}$ and has no super-hydrogenated states. The bin is the mixture
of its molecules.

The run (2026-11-08, megatron, 20 workers, 22 min) covered all 711 molecules. C₃₀H₁₄ (UID 7277) was dropped
because its anion tripped `population._round_off` (−2.8×10⁻¹⁰ eV, against a 10⁻¹² eV tolerance): its delayed
detachment gave the electron more than the excess above the EA. pahchem fixed that on 2026-11-12 (#13), so a
rerun includes it. Results:
`results/carbon_plan/pah_bins_full.txt` and `_T1.png`.

| bin (N_C) | molecules | E: worst | C, exp: worst | where they fail |
|---|---|---|---|---|
| 20–42 | 97 | 0.59 | 0.28 | Horsehead (E 0.42–0.59, C 0.05–0.28); Orion Bar at 1–3 km/s (E 0.21–0.48, C 0.17–0.26); CNM ↔ WNM at 10³ yr (E 0.48, C 0.17–0.20) |
| 42–89 | 384 | 0.24 | 0.21 | only cycles across its transition, G0 10³–10⁵ at $n$(H) = 10 |
| 89–189 | 225 | 0.06 | 0.06 | the G0 10⁴ ↔ 10⁵ cycle |
| 189–400 | 5 | 0.00 | 0.00 | none; intact up to G0 ~10⁵ at $n$(H) < 1 |

What changes against the single parents:

- **Bin 1 has no single transition.** Its 97 molecules strip at G0/$n$(H) values spread over ~2.5 dex
  (intact share 0.9 → 0.1). Some molecules are always near their own threshold, so wherever $n$(H) ≲ 10 the bin
  settles in 10⁴–10⁵ yr.
- **Tracking pays off in bin 1.** One tracked ⟨N_H⟩ cuts the error from 0.4–0.6 to 0.05–0.28. Inward through
  the Horsehead it falls to 0.05–0.10; leaving the cloud and in the Orion Bar it stays at 0.2–0.28.
- **Bins 2–4** follow equilibrium through the PDRs and the diffuse phases. They fail only where conditions cycle
  across their transitions faster than ~10⁴ yr, and tracking does not fix that.

**Tracking bin 1 alone.** `diagnostics/gsd/analyse_pah_bins_tracking.py` rebuilds every scheme from the same
per-molecule results and scores the whole PAH population. The population's shares (intact, emitting, bare) are
carbon-weighted across the bins, with equal carbon per bin or DL07-like (a lognormal, a₀ = 0.4 nm, σ = 0.4: 0.20,
0.30, 0.30, 0.21). The worst absolute error over all paths, by G0 class (≥ 100 / 1–100 / < 1):

| scheme | equal carbon | DL07-like |
|---|---|---|
| all bins at equilibrium | 0.06 / 0.12 / 0.15 | 0.07 / 0.10 / 0.12 |
| **bin 1 tracked**, bins 2–4 at equilibrium | 0.06 / 0.07 / 0.07 | 0.07 / 0.05 / 0.06 |
| the same with RAMSES-like 10³ yr steps | 0.06 / 0.06 / 0.07 | 0.07 / 0.05 / 0.06 |
| every bin tracked | 0.05 / 0.07 / 0.07 | 0.06 / 0.05 / 0.06 |
| bin 1 split at its median N_C (36), both halves tracked | 0.06 / 0.06 / 0.07 | 0.07 / 0.05 / 0.05 |

- **Tracking bin 1 halves the population's error.**
  - Entering the Horsehead: 0.11–0.15 → 0.01–0.03.
  - The Orion Bar at 3 km/s: 0.12 → 0.05.
  - CNM ↔ WNM at 10³ yr: 0.12 → 0.04.
- **Tracking bins 2–4 as well adds nothing.** The one gain is bin 2's extreme cycle (G0 10⁴ ↔ 10⁵ at $n$(H) = 10,
  10³ yr), 0.061 → 0.054.
- **The RAMSES step costs nothing** against the exact closure (≤ 0.002).
- **Splitting bin 1 barely helps,** at N_C 30 or at its median. What remains, 0.05–0.07, is the one-scalar
  closure's shape for material leaving the cloud (Horsehead outward) plus bin 2's extreme cycle. Neither finer
  bins nor more tracked bins remove it; a second moment would.

**Treatment for RAMSES (full pahchem, charge-averaged chain):**

- $M_{\rm H}$ in bin 1 only, one scalar per cell, with E for bins 2–4.
- The population's error is then ≤ 0.07, against 0.12–0.15 with no tracking.

This conclusion does not survive the joint chain (next section).

### With the joint (Z, N_H) chain (2026-11-13)

pahchem's joint chain (`Params.hydrogen_chain = "joint"`, #14/#16) adds what the charge-averaged chain lacks:

- super-hydrogenated states up to $N_{\rm H,0}+4$;
- a charge for every $N_{\rm H}$.

`check_pah_bins_hydrogen.py --chain joint` (the default) takes each molecule's network from
`Population.joint_networks` and its steady state from `steady_state_closed`. It scores the super-H share as a
fifth quantity and reports the share at the cap. The run (2026-11-13, megatron, 8 workers, 4.4 h) covered all 712
molecules, C₃₀H₁₄ included. Results: `results/carbon_plan/pah_bins_joint/` (`pah_bins_joint.txt`, `_T1.png`,
`analyse_pah_bins_joint.txt`).

**The cap binds in shielded gas.** The share at $N_{\rm H,0}+4$ in the T1 grid:

| bin | ≥ 0.93 at the cap for G0/$n$(H) ≤ | at the cap along the paths |
|---|---|---|
| 1 | 3×10⁻⁴ | ≤ 0.10 (Orion Bar) |
| 2 | 3×10⁻⁴ | ≤ 0.20 (Orion Bar) |
| 3 | 10⁻³ | 0.64 (Orion Bar), 0.56–0.59 (Horsehead), 0.35 (G0 0.1 ↔ 1 at $n$(H) = 10) |
| 4 | 10⁻² | 0.79–0.94 (NGC 7023), 0.96 (Orion Bar), 0.99 (Horsehead), 0.78 (CNM ↔ WNM) |

Wherever these shares are large, the +4 cap sets the super-H content, not the chemistry. That holds where bins
3–4 fail, below. pahchem labels the cap valid for PDRs and diffuse gas; a deeper chain to ~+12 is a separate
pahchem step.

| bin (N_C) | molecules | E: worst | C: worst | where they fail |
|---|---|---|---|---|
| 20–42 | 98 | 0.47 | 0.47 | CNM ↔ WNM at 10³ yr (0.47; 0.13 at 10⁴ yr); G0 cycles at $n$(H) = 10 at 10³ yr (0.09–0.32). PDR paths ≤ 0.08 |
| 42–89 | 384 | 0.24 | 0.23 | as before: cycles across its transition |
| 89–189 | 225 | 0.25 | 0.24 | Orion Bar (0.16–0.25, intact and super-H shares); G0 0.1 ↔ 1 at 10³ yr (0.18) |
| 189–400 | 5 | 0.76 | 0.72 | Orion Bar (0.65–0.76), Horsehead (0.13–0.55), NGC 7023 at 1–3 km/s (0.07–0.16) |

- **Bin 1 now follows equilibrium through the PDRs.** Its errors there were 0.36–0.59 with the averaged chain.
  Only the fast diffuse cycles still fail, and the closure no longer improves them. C₃₄H₁₆ and C₃₆H₁₆ are off by
  0.8–0.9 on CNM ↔ WNM at 10³ yr, under E and under the closure alike, while C₂₂H₁₂ and C₄₂H₂₄ are exact. This
  is not yet understood.
- **Bins 3–4 fail in the shielded PDR layers through their super-H states.** For C₁₉₀H₃₄ at the Orion Bar's
  shielded end (G0 3–10, atomic $n$(H) 2–8×10³ cm⁻³, 33 K), the slowest mode takes 300–1100 yr. The profile
  crosses those cells in ~180 yr even at 0.3 km/s, so the error does not fall with speed over 0.3–3 km/s. The
  steady state agrees with the long-time evolution (residual 10⁻¹⁷): the lag is kinetic.

**The whole population** (carbon-weighted; worst error by G0 class ≥ 100 / 1–100 / < 1):

| scheme | equal carbon | DL07-like |
|---|---|---|
| all bins at equilibrium | 0.06 / 0.24 / 0.13 | 0.07 / 0.21 / 0.11 |
| bin 1 tracked | 0.06 / 0.24 / 0.13 | 0.07 / 0.21 / 0.11 |
| every bin tracked | 0.06 / 0.21 / 0.11 | 0.07 / 0.19 / 0.09 |

Tracking bin 1 gains nothing. The error comes from the super-H share of bins 3–4, in the Orion Bar (0.21–0.24) and
the Horsehead (0.11–0.13), and the one-scalar closure barely reduces it.

**Why the closure fails on the super-H ladder.** H addition alternates between two speeds:

- slow onto the closed-shell states (the parent, +2);
- fast onto the odd-H radicals (+1, +3): about 600 yr⁻¹ against 37 yr⁻¹ for C₁₉₀H₃₄ in the cell above.

On top of that, +1 ↔ +2 cycle fast through addition and Eley–Rideal abstraction, which is H₂ formation with no
net change. One λ on every dN_H > 0 process leaves the rebuilt state's radicals undrained. At the true ⟨N_H⟩ its
d⟨N_H⟩/dt is 25–75× the truth's, so the closure relaxes like E. The parent's charge distribution is the same in
both (96 % anion), and a family that keeps associative detachment's charge flow gives the same rate.

A family that scales only the H gain out of even ($N_{\rm H}-N_{\rm H,0}$) states, leaving radical additions at
their true rates, was tested on single molecules (scratch, not in `hydrogen_tracking` yet):

- C₁₉₀H₃₄, Orion Bar inward at 0.3 km/s: 0.84 → 0.04;
- C₁₉₀H₃₄, outward at 3 km/s: 0.94 → 0.15;
- C₃₄H₁₆ and C₃₆H₁₆ on CNM ↔ WNM at 10³ yr: no change (0.83, 0.90).

**Status.** The bin-1-only decision is on hold. With the joint chain the population's worst error is 0.21–0.24
under every scheme tested. It is superseded by the next section.

## Equilibrium tables, judged on observables (2026-11-15)

**Decision (the user, 2026-11-15).** It follows a literature review of how other codes treat PAH hydrogenation
(`results/carbon_plan/pah_hydrogenation_review/REPORT.md` in the GSD worktree).
- RAMSES tabulates the hydrogenation state of the molecular bins from pahchem's joint chain, computed offline. Nothing
  is tracked for now.
- A treatment is judged on what the hydrogenation changes, not on state shares.

No code in the review carries a hydrogenation state through moving gas: Le Page+01/03, Andrews+16 and Boschman+15 solve
the steady state, and Montillaud+13 and Murga+20/22 run fixed environments in time. What the hydrogenation controls,
strongest first:
1. carbon loss, through the bare share and, at G0 ≳ 10, the super-H share;
2. the charge and photoelectric heating of small, dehydrogenated molecules;
3. H₂ formation, small next to grains and uncertain by about 3 dex between papers;
4. the 3.3 / 3.4 µm bands, a post-processing quantity weighted by the absorbed UV.

**Observables** (`pahchem.hydrogen_observables`). Each is a weight W[Z, N_H] per molecule on its `joint_networks` network,
so O = Σ W F for the steady state and for a state evolved along a path:
- carbon: C atoms lost (bare-skeleton channels, the C₂H₂ of the emitting states, explosions, passages beyond the chain);
- heating and cooling: photoelectric heating and recombination cooling;
- H₂ released, and H atoms taken from the gas;
- absorbed UV power, times the aromatic and the aliphatic CH counts (each extra H turns an aromatic CH into a CH₂).
  These are the first moments the 3.3 and 3.4 µm bands follow (Murga+22 App. A); their ratio is r_ali.

Carbon, heating, cooling and absorbed come from pahchem's `Population.joint_state_observables` (branch
feature/pahchem-joint-observables, answering the GSD request of 2026-11-15).
- It takes heating, cooling and explosions from the network's own rates: each N_H's shifted threshold and the per-N_H
  attachment. Its steady-state functions are computed from the same weights (equal to ≤ 4e-16).
- Until that branch is merged, a provisional version is built from Population internals with the intact molecule's
  thresholds. It overstates the heating of stripped states that the raised thresholds hold at high charge. In the
  smoke test, stripped C₉₀H₂₄ at Z = +3 and G0 10⁵, n(H) = 10 got 36× the heating, so no heating result is reported
  from that version.
- H₂, H uptake, the CH moments and the shares come from the network and the states themselves.

**Tables** (`pahchem.hydrogen_tables`). Per bin, the number-weighted mean over its PAHdb v4.00 molecules, on the
pahchem-nd v1 axes of `export_pah_tables`:
- log G0 −1 … 6 (Draine 1978 shape, with the solar circle's old-star light by default since 2026-11-25, `optical_isrf` 1
  as `Environment` and {doc}`pahchem-gas-coupling` §4d; `--optical-isrf 0` drops it);
- log n(H) −1 … 6;
- log T 1 … 4;
- log x_e −5 … −1.

Two files per bin:
- `<bin>_carbon.dat`: the carbon-loss rate per molecule in `export_pah_tables`' unit and format, read by
  `solvers.table_io.read_pah_nd_table`;
- `<bin>.npz`: every observable, plus the at_cap share that flags the nodes where the +4 cap sets the answer.

`export_pah_tables` builds its file from one parent molecule's BaselineModel network; these come from the bin's
molecules and their whole joint chain. `--superh-chemistry` and `--superh-levels` select pahchem's super-H chemistry and
cap; the next section says which of the observables depend on them.

### Scores and the super-H cap (2026-11-16)

**Runs.** `check_pah_bins_hydrogen.py` was run on megatron with all 712 molecules, from pahchem PRs #17 and #18 (63e4d4f,
not merged). Two super-H chemistries:
- Andrews+16's (pahchem's default) at cap +4: 3.9 h on 16 workers;
- the deep chain (`superh_chemistry = "deep"`, the user's choices of 2026-11-15) at caps +8, +12 and +24: 5.5, 7.0 and
  11.8 h.

The reports are in `results/carbon_plan/pah_bins_obs/`: one `<run>.txt` per run, and `compare_caps.txt` from
`compare_pah_bins_caps.py`, which scores every run against the +24 run's evolution.

**E against its own truth** (deep chain at +24). Factor errors max(E/T, T/E) − 1, as the worst cell / path budget:

| bin | heating | cooling | carbon | H₂ | H uptake | r_ali |
|---|---|---|---|---|---|---|
| 1 (20–42 C) | 0.19 / 0.08 | 0.11 / 0.11 | 1.7 / 1.2 | 17 / 1.1 | 1.1 / 0.72 | 0.20 / 0.86 |
| 2 (42–89) | 0.09 / 0.01 | 0.05 / 0.01 | 38 / 1.4 | 13 / 5.4 | 0.00 / 0.00 | 0.00 / 0.00 |
| 3 (89–189) | 0.02 / 0.01 | 0.01 / 0.00 | 0.06 / 0.06 | 0.87 / 0.86 | 0.02 / 0.00 | 0.26 / 0.07 |
| 4 (189–400) | 0.00 / 0.00 | 0.00 / 0.00 | – | 48 / 0.42 | 31 / 0.16 | 90 / 0.10 |

Where the errors are, by path budget:
- **PDR paths** (NGC 7023, Orion Bar and Horsehead at 0.3–3 km/s, in and out): ≤ 0.08 in every bin and observable.
- **Cycles with 1e4- and 1e5-yr periods:** ≤ 0.17, except bin 2's carbon (0.30) and H₂ (0.49).
- **Cycles with 1e3-yr periods:** every larger error is here.
  - carbon ×2.2–2.4 in bins 1–2 (bin 1 at G0 0.1 ↔ 1, bin 2 at 10³ ↔ 10⁴);
  - H₂ up to ×6.4 (bin 2 at G0 100 ↔ 1000);
  - bin 1's r_ali (0.86) and H uptake (0.72) on CNM ↔ WNM.
- **Large cell errors (×17–90)** are in cells where the observable barely passes the 1e-3 cut, e.g. bin 4's H₂, H uptake
  and r_ali on the Horsehead paths, whose budgets are ≤ 0.08.

The H₂ errors were traced by process on the laptop for one molecule per bin:
- Large PAHs lose their hydrogen more slowly than a 500-yr bright half-cycle. C₉₀H₂₄ keeps ⟨N_H⟩ ≈ 20 at G0 10⁵,
  n(H) = 10, against 2.4 in equilibrium, and loses H₂ from intact molecules: 2.6e-11 s⁻¹ against 9.4e-14.
- The H uptake is the same in both (1.4e-9 H s⁻¹); only the share leaving as H₂ rather than H differs.
- For 10 ppm of carbon in the bin (an assumed abundance, not a model output), the time-resolved PAH H₂ is about 1 % of
  grain formation (R = 3e-17 cm³ s⁻¹). At those G0 the H₂ is photodissociated at once.

**The cap.** The truth at +K is scored against the truth at +24 (path budget, worst over paths):

| bin | Andrews +4: H₂ / H uptake / r_ali | deep +8 | deep +12 | share at the +24 cap |
|---|---|---|---|---|
| 1 | 0.05 / 0.22 / 2.95 | ≤ 0.01 | ≤ 0.01 | – |
| 2 | 0.88 / 1.45 / 3.53 | 0.03 / 0.00 / 0.02 | ≤ 0.01 | – |
| 3 | 5.2 / 4.5 / 7.4 | 0.99 / 0.13 / 1.03 | 0.52 / 0.06 / 0.55 | 0.12 (Orion Bar) |
| 4 | 19 / 1.6 / 8.0 | 2.6 / 1.1 / 4.0 | 1.2 / 0.53 / 1.8 | 0.76 (Orion Bar); > 0.05 on 26 of 38 paths |

- **Heating, cooling and carbon** do not depend on the cap or the chemistry: the truth moves by ≤ 0.05 (cell) and
  ≤ 0.02 (budget) in every bin.
- **Bins 1–2:** the deep chain changes their H₂, H uptake and r_ali from Andrews+16's, through its chemistry, and is
  converged by +8.
- **Bins 3–4 depend on the cap,** and bin 4 is not converged at +24. Bin 4's T1 map shows why:
  - At G0/n(H) ≲ 1e-2 every bin-4 molecule sits at the cap: ⟨N_H⟩/N_H0 = 1.20 at +8 and 1.61 at +24.
  - Between 1e-2 and 1, a third of the molecules are super-hydrogenated, and about half of those are at the cap.
  - Once past +4, the deep chain's additions run on, and the cap is what stops them.
- **r_ali along the paths:** the UV-weighted ratio of aliphatic to aromatic CH, path-averaged.
  - Bins 1–2: ≤ 0.03.
  - Bin 3: 0.02–0.17 (peak 1.4 in the Orion Bar).
  - Bin 4 at +24: 1.0–1.1 on every path, CNM ↔ WNM included, with peaks up to 2.8. That is 4–5× its value at +8
    (0.23–0.31) and with Andrews +4 (0.17–0.21).
  - Whether so aliphatic a large-PAH population is allowed needs the band emission, not this proxy. The 3 µm bands come
    mostly from the small PAHs.

**What it means for the tables.**
- For heating, cooling and carbon, E holds on every PDR path in every bin, and these do not depend on the super-H
  chemistry or the cap. E fails only on the 1e3-yr cycles.
- H₂, H uptake and r_ali of bins 3–4 depend on the cap and the chemistry until the super-H runaway has a physical stop.
  That stop is a pahchem question: barriers beyond +24, the Eley-Rideal bracket (×2, ×7.5) and the 3.4 µm band.

**Across the tables' domain.** The paths do not reach every corner of the table. A pilot build of 81 nodes compares
+8 with +24 over all 712 molecules, on these values:
- G0 0.1 / 300 / 10⁶;
- n(H) 0.1 / 300 / 10⁶;
- T 10 / 316 / 10⁴ K;
- x_e 1e-5 / 1e-3 / 0.1.

The results:
- **Heating and cooling:** ≤ 0.3 % in every bin.
- **Carbon:** ≤ 0.13 % in bins 2–4. In bin 1, the cap changes carbon by 33 % at one node: G0 300, n(H) 10⁶,
  10⁴ K, x_e 0.1.
  - It is one molecule, C₂₄H₁₄ (UID 538), whose carbon comes from a dication share of ~1e-11.
  - The deeper super-H states raise its thresholds, so the anion share goes from 0.71 to 0.79 and the dications drop.
  - Those threshold shifts at +24 are pahchem's size law extrapolated.
- **H₂ and the aliphatic CH:** ×3 in every bin, exactly the cap ratio, at the warm shielded nodes (G0 0.1, T ≥ 316 K),
  where every bin, bin 1 included, sits at the cap. At 10 K the addition barriers keep the molecules intact.

So the H₂ and r_ali tables of every bin are set by the cap wherever warm, shielded atomic gas reaches them. The npz's
at_cap share marks those nodes.

**Production build (2026-11-16, the user's choice):** deep chain at +8 on the full pahchem-nd v1 grid (7875 nodes),
built on megatron in 37 min with 32 workers. The tables are in `~/pycalima_runs/hydrogen_tables/deep_k8/`, copied to
`results/carbon_plan/hydrogen_tables/deep_k8/`.

Checks:
- The `.dat` files read back through `read_pah_nd_table` and equal the npz.
- Every value is finite, and the shares sum to 1 within 1e-15.
- The full build equals the corner pilot at their shared nodes.

Interpolation error, log-linear on the grid against nodes solved half a step away: 286 nodes, half-steps in G0 and n(H)
at 100 K, and in T, all at x_e 1e-4:
- **Heating and cooling:** ≤ 9 % everywhere.
- **Carbon:** scored where the bin loses its carbon within 1 Gyr.
  - Bin 1: ≤ 13 %.
  - Bins 2–3 up to G0 10^4.5: p90 ≤ 0.2, worst ×1.7.
  - Above G0 10^4.5 interpolation is poor: bin 2 up to ×2.3, bin 3 up to ×24 (p90 ×2.2). There the carbon loss
    changes by up to 3.6 dex per 0.5-dex step.
  - A finer G0 axis above 10^4 would fix it; the format takes any axis values.
- **H₂:** p99 0.5–2.9, worst ×9, at the 10 → 32 K step, where the addition barriers open (up to 10 dex per step).
- Heating, cooling and carbon: final.
- H₂ and r_ali: provisional wherever at_cap is large. Rebuild them if pahchem gives the super-H runaway a physical stop.
- The n(H) axis is the atomic H density (`hydrogen_tables.environment`). RAMSES must pass n(H I), not n_H.

## Caveats

- **The physics is pahchem's baseline.** Its multiphoton H loss needs ~10× more UV than Montillaud's,
  so the transitions of C₅₄ and C₉₆ sit at higher G0/$n$(H) than in their models. Where the transitions
  lie decides where tracking matters.
- **Field shapes** are held with depth (no reddening). Kurucz 15000 / 40000 / 30000 K for the three
  PDRs, Draine for the diffuse phases.
- **Bin 4** (189–400 C) has no parent. The bins' in-bin size spread is not sampled: one parent per bin.
- **The super-hydrogenated ladder** is Andrews et al.'s, to $N_{\rm H,0}+4$, in the single-parent tests and
  pahchem's joint chain's, also to +4, in the joint bins run. In shielded gas the cap binds (see the at-cap
  table). pahchem's deep chain (2026-11-16, above) extends it to +24, where bin 4 still sits at the cap.
- **Thresholds of the parents' other N_H states.** The 2026-11-07 run used Andrews et al.'s IP rise of
  0.1 eV per missing H. The 2026-11-12 rerun (`check_pah_hydrogen_tracking.txt`; the earlier one kept as
  `_2026-11-07`) uses pahchem's B3LYP ΔSCF ladders (`pahchem.threshold_shifts.LADDER`). Their first IP rises by
  about 0.075 / 0.045 / 0.03 eV per missing H for C₂₄ / C₅₄ / C₉₆, and the EA of the dehydrogenated states by
  up to 1.3 / 0.6 / 0.06 eV.
  The gates are unchanged, and the paths move by ≤ 0.03: the intact-state charge error of C₂₄ falls from
  0.09 to 0.07, and its deep Orion Bar errors rise by 0.01–0.03.
