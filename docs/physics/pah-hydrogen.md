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
- an IP that rises by 0.1 eV per missing H;
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
- the steady state of the fast-charge chain misses the full network's by up to 0.86 for C₂₄ and ~0.5
  for C₅₄ and C₉₆, mostly in the super-hydrogenated share.

The equilibrium of the full network is not affected. Two conclusions follow:

- An equilibrium table must come from the full network offline. It cannot be rebuilt in RAMSES from
  charge-averaged rates.
- On the charge tables' $N_{\rm H}$ axis: the intact-state charge (FC0) adds a cation-fraction error of
  0.09 for C₂₄ where it is bare (the NGC 7023 edge), and ≤ 0.03 for C₅₄ and C₉₆.

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
| C₅₄, C₉₆ | every path | ≤ 0.003 | ≤ 0.002 |
| C₂₄ | NGC 7023, Horsehead, CNM ↔ WNM | ≤ 0.02 | ≤ 0.005 |
| C₂₄ | deep Orion Bar (G0 1–100 at $n_{\rm H}\sim10^6$) | 0.21–0.50 | 0.21–0.31 (the super-H share) |

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
| 2–3 (C₅₄, C₉₆) | PDR fronts and shielded layers (NGC 7023, Orion Bar, Horsehead), diffuse phases | **E** (errors ≤ 0.003) |
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
because its anion trips `population._round_off` (−2.8×10⁻¹⁰ eV, against a 10⁻¹² eV tolerance). Results:
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

**Treatment for RAMSES (full pahchem):**

- $M_{\rm H}$ in bin 1 only, one scalar per cell, with E for bins 2–4.
- The population's error is then ≤ 0.07, against 0.12–0.15 with no tracking.

## Caveats

- **The physics is pahchem's baseline.** Its multiphoton H loss needs ~10× more UV than Montillaud's,
  so the transitions of C₅₄ and C₉₆ sit at higher G0/$n$(H) than in their models. Where the transitions
  lie decides where tracking matters.
- **Field shapes** are held with depth (no reddening). Kurucz 15000 / 40000 / 30000 K for the three
  PDRs, Draine for the diffuse phases.
- **Bin 4** (189–400 C) has no parent. The bins' in-bin size spread is not sampled: one parent per bin.
- **The super-hydrogenated ladder** is Andrews et al.'s, to $N_{\rm H,0}+4$. The P1 hydrocarbon network
  would extend it.
