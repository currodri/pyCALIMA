# One-zone collapse with evolving dust: progress (2026-10-10)

Working notes for `src/pycalima/collapse/` and the scripts in this directory.
The plan this follows has stages 1–7: (1) grid and coagulation, (2) Galactic
validation, (3) materials and optics, (4) minihalo tracks and accretion,
(5) full physics and science grid, (6) docs, (7) energy equation.

## Implemented

| Stage | What | Where |
|---|---|---|
| 1 | Dense log size grid with in-bin power laws; two-collector Smoluchowski coagulation; Marchand+21 M1/M5 benchmarks | `grid.py`, `coagulation.py`, `check_marchand21.py` |
| 2 | Brownian + Ormel & Cuzzi 07 turbulence, ambipolar drift, Coulomb factor; dark WDB06 charge table with the Marchand closure (M2); Guillet+20 | `velocities.py`, `charge.py`, `check_guillet20.py` |
| 3 | Materials, Planck-mean optics, per-bin T_d, gas–grain cooling (neutrals, uniform α_T) | `materials.py`, `optics.py`, `dust_temperature.py`, `gas_grain.py` |
| 4 | Omukai+05 and Grackle dust-free tracks; key-species accretion; D_cr search (criterion C) | `tracks.py`, `accretion.py`, `integrator.py` |
| 5 | N07 CCSN initial conditions and reverse shock (Chiaki+15 f_dep); ice mantles; science grid | `ics.py`, `science_grid_dust_evolution.py` |
| 7 | Energy equation and Schneider & Omukai 10 fragmentation criterion (`frag`) | `integrator.py` |

Environments of the gas track, from Grackle (`grackle_freefall.py`, `grackle_environments.py`):

- attenuated cosmic rays ζ0;
- a Lyman–Werner J21 (RT H2-dissociation field, Jeans-length self-shielding);
- the CMB at redshift z;
- a slowed collapse (`Track.t_col_factor`).

## Results so far

[Z/H]_cr, Z_sun = 0.02. Results files are under `results/collapse/`.

- **Coagulation and shattering.** Both move [Z_cr] by ≤ 0.02 dex.
  - Turbulent Δv stays below the Jones+96 shattering thresholds (`check_shattering_velocities.py`).
- **Accretion with charge.** Coulomb focusing of Mg+ and Fe+, taken as an upper bound, lowers [Z_cr] by 0–0.14 dex. Without cosmic rays the metals are neutral.
- **Cosmic rays** (`grackle_cosmic_rays.py`).
  - Static dust with `frag`: ζ0 ≥ 1e-19 stops fragmentation altogether (20 Msun, n1).
  - With accretion: no change, −5.23.
  - Criterion C barely moves.
- **Literature reproduced from their own dust inputs** (`check_literature_zcr.py`).
  - Within ~0.3 dex: O05, S03, S06, S12a, Ji14.
  - Chiaki+15 comes out 0.2–0.9 dex high, which is still unexplained.
- **Transition band** (`science_grid_transition.py`, `plot_transition.py`).
  - Classical assumptions (MW f_dep 0.5, MRN, static dust, no CRs or LW) give −5.47 (`frag`) and −6.09 (C).
  - Each input alone, `frag`:

    | Input | Range |
    |---|---|
    | f_dep | −5.47 to no fragmentation (n10) |
    | Progenitor | −5.80 to −5.19 |
    | Sizes | −5.34 to −5.03 |
    | α_T 0.3 | −5.02 |
    | ζ0 1e-16 | no fragmentation |
    | J21 100 | −6.30 |

  - All combinations (descent search, not a full grid): `frag` −6.88 to no fragmentation; C −7.78 to −3.18.

## Remaining physics

Ordered by how much each could change the answer.

1. **H2 formation on grains, fed back into the gas.**
   - The tracks are dust-free, so grain-catalysed H2, with its formation heating and the H2 cooling that follows, is missing.
   - This matters most for the J21 = 100 (atomic) case, which sets the low end of the band.
   - It needs a dust-aware track, either Grackle with `dust_chemistry` on or a gas network, or an H2 term in the energy equation.
2. **Composition-dependent thermal accommodation α_T(T, T_d, species)** (Burke & Hollenbach 83).
   - Only a single scalar is used now.
   - α_T = 0.3 alone moves `frag` by +0.45 dex.
   - The plan item was to extend `low_temp_cooling` to H2 and to each material.
3. **Grain sublimation** (`sublimation.py`, wrapping `dust_sublimation.py`).
   - Species should be removed once T_d exceeds their sublimation T. Not implemented: grains survive to n_end.
   - This sets where dust cooling ends, which matters for the late γ > 0.97 part of `frag`.
4. **Self-consistent charge in the run.**
   - Accretion takes a prescribed focusing factor (`accretion_focus`).
   - The dark table and the Marchand closure in `charge.py` are not solved step by step with the track's (cosmic-ray) ionisation.
   - As a result there is no Coulomb barrier in the collapse coagulation kernel.
   - Electron and ion energy exchange in Λ_gd is also omitted (x_e < 1e-6). That needs rechecking for ζ0 ≳ 1e-17.
5. **Shattering redistribution** (`shattering.py`, Jones+96 fragment matrix).
   - Not implemented, because Δv stays below threshold for Mach ≲ 1.
   - It would only matter for supersonic turbulence or shock-driven grain velocities.
6. **More progenitors.** Faint SNe and PISNe (Grackle pathways fsn13/15/50/80, pisn170/200) would widen the composition axis beyond the N07 CCSNe.
7. **Atomic-cooling and DCBH track proper.** Deferred; J21 = 100 is the proxy for now.
8. **The Chiaki+15 offset.** Compare Grackle and Omukai tracks with the same dust.

Not physics:
- `calima-collapse` CLI;
- `docs/physics/collapse.md`, with the prescribed-track caveat, the cross-species coagulation rule and the α_T sources;
- `refs.bib` entries (plan stage 6).

## Caveats

- **Descent search.** It brackets the combined range but does not guarantee the global extremes.
- **One bad track point.** The J21 = 100 track has one T = 0 point at [Z/H] = −3.5, which `Track` drops.
- **SVG fonts.** The SVG figures expect DejaVu Serif to be installed.

## Running the Grackle scripts

- They need a Grackle source tree and its Python: `--grackle-src`, `--grackle-python`.
- The working copy was in the scratchpad, which is lost on reboot. Rebuild it under `RAMSES_dev` before rerunning.
