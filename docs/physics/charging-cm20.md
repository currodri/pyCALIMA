(physics-charging-cm20)=
# Charging of THEMIS amorphous carbon (`cm20`)

The WDB06 charging tables (`models/dust_charge/wdb06_yields.py`,
`rt_group_charging.py`, `charging_isrf_tables.py`) take a third material
besides `graphite` and `silicate`: `cm20`, the H-poor amorphous carbon (a-C,
$E_g\simeq0.1$ eV) of the THEMIS CM20 grains (Jones et al. 2013, A&A 558, A62),
charged as DustEM v4.3 charges it (`CHRG_CM20.DAT`; `PE_CURRENT` and
`GAS_CURRENTS` in `DM_zdist.f90`; DustEM user guide, App. B). It is the benchmark;
the default for THEMIS carbon is `ac` (below), which departs from DustEM where
measurements say it should.

Everything WDB06 gives graphite and silicate exists for `cm20`: the RT-group
tables (`charging_model='WDB06rt'`), the uniform-ISRF tables
(`charging_model='WDB06isrf'`), photoelectric heating, recombination cooling,
Coulomb focusing factors and grain-assisted ion recombination. Only the pieces
below differ from graphite.

The constants need the DustEM release (`$CALIMA_DUSTEM_DIR`, dataset
`dustem-themis-cm20` in `data/registry.toml`), as the THEMIS optics do.

## Thresholds

With $W=4.75$ eV, the ionisation potential and the electron affinity of a grain
of radius $a$ and charge $Z$ are

$$
{\rm IP}(Z) = W + \Big(Z+\tfrac12\Big)\frac{e^2}{a} + A_C,\qquad
{\rm EA}(Z) = W - E_{\rm bg}(a) + \Big(Z-\tfrac12\Big)\frac{e^2}{a} - A_C
$$ (eq-cm20-ip)

(`ionisation_potential`, `cm20_electron_affinity`). The aromatic correction

$$
A_C = \frac{e^2}{2a}\Big(\frac{\pi a}{2a_R}-1\Big),\qquad a_R=\frac a2,
$$ (eq-cm20-ac)

is the energy to take the electron off the largest aromatic domain, a
conducting disk of radius $a_R$ (DustEM's `p_6`; `aromatic_correction`). It
replaces WD01's $(Z+2)(0.3\,\text{Å}/a)\,e^2/a$, and is large for small grains:
the IP of a neutral 0.4 nm grain is 10.4 eV, against 6.7 eV for WD01 graphite.
The band gap follows the size law of Jones (2012, A&A 542, A98) that DustEM uses,
$E_{\rm bg} = 0.2\,(5\,{\rm nm}/a-1)$ eV within $[0.1, 2.5]$ eV
(`cm20_band_gap`). Below $a\simeq1$ nm the EA of the lowest charges is negative.

$Z_{\rm min}$ uses DustEM's $U_{\rm ait}$, which is WD01's carbon one
(`most_negative_charge_cm20`); $Z_{\rm max}$ is WD01's with $W=4.75$ eV, as in
DustEM. Thresholds for inner shells take $W\to I_{i,s}$ in
{eq}`eq-cm20-ip`, as WDB06 do for graphite.

## Band photoemission

Below 20 eV the band yield is that of Kimura (2016, MNRAS 459, 2751), eq. A12
(`cm20_band_yield`):

$$
Y = y_2\,\min(y_0y_1, 1),\qquad
y_0 = \tfrac12\Big(1-\sqrt{1-e_2/e_t}\Big),
$$

$$
y_1 = \frac{(1+\sqrt{1-e_2/e_t})\,\alpha^2(1-e^{-\alpha})
 + w(\alpha)(\alpha-\beta)}{\alpha-\beta+1}\,
 \frac{(\beta/\alpha)^3}{w(\beta)},\qquad w(x) = x^2-2x+2(1-e^{-x}),
$$

with $y_2$ and the mean energy at infinity $e_2$ of the escaping electrons from
WD01's parabolic distribution, $e_t = e_2+h\nu-\Theta$ the electron's mean energy
inside the grain, $\beta = \tfrac34 Q_{\rm ext}$ (from the THEMIS optics,
`themis_optics`), and $\alpha = \beta + a/l_e(e_t)$ with
$l_e(E) = 143\,E^{-2} + 0.054\,E^{1/2}$ nm ($E$ in eV; DustEM's `p_le`).
$Q_{\rm abs}$ and $Q_{\rm ext}$ are tabulated once per radius from 1 eV to 20 keV
(`_cm20_efficiencies`): the DustEM tables, Mie on the Henke index below 0.04 µm,
and anomalous diffraction above $x = 2\times10^4$.

Between 20 and 50 eV the product $y_0y_1$ is blended, linearly in $\ln h\nu$, to
WDB06's atomic band yield (their eq. 9 with the Watson factor), as WDB06 blend
WD01's yield; above 50 eV the channels are WDB06's (inner shells, Auger and
secondary electrons). For these, carbon is treated as graphite (WDB06 eq. 12
$l_e$, $\epsilon=117$ eV) at the CM20 density of 1.6 g cm$^{-3}$ and its carbon
number density ($X_{\rm H} = E_g/4.3$, Jones 2012). These are graphite's
constants, used for want of a-C values.

Photodetachment uses WD01's cross-section with ${\rm EA}(Z+1)$ of
{eq}`eq-cm20-ip` and the van Hoof et al. (2004) $E_{\rm min}$. The UV-background
kernels of a `cm20` table start at 0.1 eV (graphite and silicate at 1 eV): the EA
falls below 1 eV, so the optical and IR part of the Mathis field photodetaches
negative grains.

## Collisions

The electron sticking coefficient is $s_e = \tfrac12(1-e^{-a/\bar l_e})$, with no
PAH factor and zero at $Z_{\rm min}$ (`cm20_sticking`). As in DustEM, $\bar l_e$ is
$\langle 1/l_e\rangle^{-1}$ over the band photoemission rate; it is computed once
per radius in the Mathis et al. (1983) field (the average depends on the field's
shape only), and, as DustEM's recursion has it, the sticking at charge $Z$ takes
the $\bar l_e$ of charge $Z-1$ (rounded to an integer). A `cm20` table stores
$s_e$ at its potential nodes (format 4).

Ions stick with $s_i = 1$ and the focusing factors are DS87's, as for the other
materials. There is no autoionisation cooling at $Z_{\rm min}$: DustEM's sticking
is zero there, so its term vanishes (and the EA of $Z_{\rm min}$ can be negative).
The hot-gas terms (gas-electron secondaries, electron pass-through, ion charge
transfer; Draine & Salpeter 1979) use graphite's constants at $\rho = 1.6$.

## Check against DustEM

`diagnostics/dust_charge/check_cm20_dustem.py` builds the table of each of the 25
CM20 radii of DustEM's `GRAIN_J13` (0.4 nm–4.9 µm), solves the cell RAMSES would
solve in DustEM's default gas ($T=8000$ K, $n_e=7.2\times10^{-3}$,
$n({\rm H^+})=7.08\times10^{-3}$, $n({\rm C^+})=1.2\times10^{-4}$ cm$^{-3}$) and
Mathis field ($G_0=1$, 1.123 Habing), and compares with DustEM's `ZDIST.RES`.

The `ZDIST.RES` shipped with the release is wrong above ~0.5 µm: it gives
$\langle Z\rangle\simeq0$ for 1–5 µm grains, where a fresh build gives −119 to
−2226. A fresh build also differs from the shipped file by ~4% at the smallest
radii. DustEM's $l_e$ in `PE_CURRENT` is not set when no photon exceeds the
threshold, so it depends on the build. `diagnostics/dust_charge/build_dustem.sh`
builds DustEM with `gfortran -O2` (optionally dumping the per-charge rates); the
comparison uses that build.

| $a$ [nm] | $\langle Z\rangle$ DustEM | $\langle Z\rangle$ | $\sigma_Z$ DustEM | $\sigma_Z$ | $J_{\rm pe}$ / DustEM | $H_{\rm pe}/P_{\rm abs}$ / DustEM |
|---|---|---|---|---|---|---|
| 0.4 | 0.133 | 0.135 | 0.361 | 0.362 | 0.995 | 1.005 |
| 1.3 | 1.61 | 1.75 | 1.00 | 1.12 | 0.955 | 0.945 |
| 4.2 | 7.60 | 7.91 | 1.85 | 1.96 | 0.974 | 0.958 |
| 13.6 | 24.5 | 25.2 | 3.20 | 3.27 | 0.998 | 0.984 |
| 44 | 67.9 | 69.8 | 5.51 | 5.58 | 1.006 | 0.997 |
| 144 | 119 | 124 | 9.32 | 9.42 | 1.018 | 1.015 |
| 1021 | −119 | −106 | 19.3 | 19.4 | 1.026 | 1.032 |
| 4900 | −2226 | −2172 | 40.9 | 44.7 | 1.030 | 1.032 |

$P_{\rm abs}$ agrees to 0.3% at every radius. A Python transcription of DustEM's
`PE_CURRENT` on DustEM's own grid reproduces its photoemission rates to 0.8%,
with DustEM's $Q$ or ours. The remaining differences are physics that DustEM does
not have, and are attributed as follows.

- **Photo-excited secondaries** (WDB06 sec. 6): +1–3% in $J_{\rm pe}$ above
  ~10 nm, and $\langle Z\rangle$ up by up to half a $\sigma_Z$. With them
  switched off, $\langle Z\rangle$ is within $0.1\,\sigma_Z$ and $J_{\rm pe}$
  within 1% (`tests/test_cm20_charging.py`).
- **Secondaries excited by gas electrons** (DS79): in this warm gas they make
  1–4 nm grains more positive ($\langle Z\rangle$ +8% at 1.3 nm) and raise the
  high-$Z$ tail of $P(Z)$.
- **$J_{\rm gas}$ below 2 nm** (+15–19%): DustEM prints the electron current
  with the $\bar l_e$ of charge $Z$, while its $P(Z)$, and ours, use that of
  $Z-1$.
- **The 2.2 µm point**: DustEM's `Q_CM20.DAT` is broken for $a\gtrsim0.9$ µm
  below 0.1 µm. $Q_{\rm sca}$ reaches 15–670 there, and it enters Kimura's yield
  through $Q_{\rm ext}$. Separately, $Q_{\rm abs}$ is exactly 0.840 over
  0.04–2.95 µm for $a\ge0.45$ µm. Only the top bin of a grid that reaches 1 µm
  sees either.

The field also differs below 5 eV: our Mathis field
(`tools/radiation_fields.py`) is 14–18% brighter than DustEM's `ISRF_MATHIS.DAT`
at 1–2 eV. This is in the 4000 K component, whose dilution is $1.65\times10^{-13}$
here and about $10^{-13}$ in DustEM. The UV agrees to 0.3%. Only the
photodetachment of sub-nm negative grains sees it (+7%).

## pyCALIMA's amorphous carbon (`ac`): the default

`cm20` is the faithful DustEM port, kept for benchmarks. The default for THEMIS
carbon in the GSD (`themis_ac` → `ac`; the `aromatic` carbon carrier) keeps its
densities and WDB06 channels, and departs from DustEM where measurements say it
should (a literature check of 2026-10-24):

- **Yield.** The extended Kimura model, eK16, of Hróðmarsson et al. (2025, A&A 698,
  A202, Sect. 4), validated on the photoionisation of neutral and cationic PAHs
  (`ek16_band_yield`). It departs from DustEM 4.3's Kimura yield in two ways. First,
  the electron's total energy in the grain is $E_t = E_{2i} + {\rm IE}_x$, where
  $E_{2i} = \Theta/2$ is the mean of the nascent parabolic distribution on
  $[0,\Theta]$, not of the escaping electrons. Second, the electron–grain interaction
  is $E_x = -(Z+1+c_x)e^2/a$ with $c_x$ = 0.15; it enters ${\rm IE}_x$, $\Theta$, the
  escape fraction $Y_2$ and the energy at infinity. For neutral perylene in the
  Mathis field, with the Draine & Li (2007) absorption and WD01's IE (W = 4.4 eV),
  as in the paper, the maximum photoelectric efficiency is 1.64%, against their
  ~1.5%; DustEM 4.3's form gives 1.04% (`tests/test_cm20_charging.py`).

- **Threshold.** DustEM's aromatic correction $A_C$ puts the ionisation potential
  of small grains far above measurement. WD01's eq. 2, calibrated on PAHs, with
  $W$ = 4.75 eV and radii at 1.6 g cm$^{-3}$, matches measured ionisation energies
  within 0.55 eV (below; `tests/test_cm20_charging.py`). C$_{60}$, a cage, is 1.3 eV
  above every formula. The $W$ is DustEM's own measured a-C value (Feuerbacher &
  Fitton 1972, via Kimura 2016), inside the 4.05–4.77 eV of soot (Ouf et al. 2016).

  | | $N_{\rm C}$, $Z$ | measured [eV] | `ac` | `cm20` (DustEM) |
  |---|---|---|---|---|
  | pyrene | 16, 0 | 7.43 | 7.39 | 10.99 |
  | coronene | 24, 0 | 7.29 | 6.99 | 10.20 |
  | (coronene)$_5$ (Joblin et al. 2017) | 120, 0 | 6.45 | 5.94 | 7.94 |
  | C$_{30}^+$ (Wenzel et al. 2020) | 30, 1 | 10.2 | 10.23 | 13.03 |
  | C$_{48}^+$ (Wenzel et al. 2020) | 48, 1 | 9.1 | 9.36 | 11.83 |

- **Electron affinity.** $W - E_{\rm bg}(a) + (Z-\tfrac12)e^2/a$: DustEM's without
  $A_C$. It is the closest of the published forms to measured small-PAH affinities,
  though still above them (pyrene 0.86 against 0.41 eV, coronene 1.37 against
  0.47 eV); WD01's eq. 4 gives 1.8–2.2 eV. The $E_{\rm bg}(a)$ law underestimates
  molecular gaps.
- **Sticking.** WD01's eqs. 27–30 ($l_e$ = 1 nm, with the PAH factor of eq. 28):
  the probability that an incoming electron thermalises inside the grain. DustEM's
  photoelectron escape length, averaged over the field, has no physical link to it.
  The anthracene measurement (Moustefaoui et al. 1998) is within an order of
  magnitude of eq. 28; C$_{60}$ (Viggiano et al. 2010) implies $s_e\approx0.8$,
  against WD01's ~0.2 and DustEM's ~0.1, which is left as an open question.
- **Optics.** The recomputed CM20 grains ({doc}`carbon-optics`); the UV background
  kernels start at 0.1 eV, as for `cm20`.
- Zmin and the autoionisation cooling as for WD01 graphite: an electron with
  negative affinity is held by the Coulomb barrier until field emission.

Against DustEM in its WNM gas (`check_cm20_dustem.py --material ac`):

| $a$ [nm] | ⟨Z⟩ DustEM | ⟨Z⟩ `ac` | $J_{\rm pe}$ / DustEM | $H_{\rm pe}/P_{\rm abs}$ / DustEM |
|---|---|---|---|---|
| 0.4 | 0.13 | 0.38 | 3.2 | 1.43 |
| 0.88 | 0.79 | 1.19 | 2.7 | 2.8 |
| 1.3 | 1.61 | 2.01 | 2.3 | 2.3 |
| 4.2 | 7.6 | 8.5 | 1.47 | 1.35 |
| 14 | 24.5 | 28.9 | 1.12 | 0.97 |
| 144 | 119 | 145 | 1.13 | 0.99 |
| 1021 | −119 | −151 | 0.93 | 1.01 |
| 4900 | −2226 | −3322 | 0.45 | 0.50 |

The sub-nm to few-nm grains are more positive and heat the gas 1.4–2.8 times more
efficiently, through the lower thresholds and the eK16 yield; the larger electron
sticking offsets part of it. Above ~10 nm the efficiency is DustEM's to a few per
cent. The largest grains follow the recomputed optics, which replace the broken
table.

## pyCALIMA's H-rich a-C:H (`ach`)

The `aliphatic` carbon carrier (a-C:H, $E_g$ = 2.5 eV, 1.33 g cm$^{-3}$) is charged as
`ach`. That is `ac` with the thresholds of polymeric a-C:H measured by Schäfer,
Ristein & Ley (1993, J. Non-Cryst. Solids 164–166, 1123) by photoelectron-yield
spectroscopy:

- work function (vacuum to Fermi level) 2.6 eV for their most polymeric film (~40%
  H, 14% sp$^2$), rising to graphite's ~4.5 eV as sp$^2$ grows to 37%;
- valence-band top 2 eV below the Fermi level, so a bulk ionisation energy of
  4.6 eV, which `ach` uses in WD01's eq. 2 in place of $W$;
- the yield onset following the optical absorption, i.e. a conduction band at the
  vacuum level ("incipient negative electron affinity"): $\chi$ = 0, and
  ${\rm EA}(Z) = \chi + (Z-\tfrac12)e^2/a$.

The 2.6 eV onset itself comes from two defect bands in the gap, of 10$^{20}$–10$^{21}$
cm$^{-3}$, about 10$^{-3}$ of the atoms. They are left out. Our carrier has more H
(58%) than their films, so it is taken at the polymeric end of their series. The
optics are the Jones (2012) a-C:H ($E_g$ = 2.5 eV) tables ({doc}`carbon-optics`),
and the yield is eK16.

Against `ac` at the same radii (one cell each: WNM 8000 K; CNM 100 K,
$n_e = 3\times10^{-2}$; dense 30 K, $n_e = 10^{-3}$, $G_0 = 10^{-2}$), `ach` grains are
10–20% more positive in the WNM, and their photoelectric heating is 1.1–2 times
higher, from the stronger far-UV absorption of a-C:H.

Open:

- the photoelectron kinetic-energy share: Hróðmarsson et al. measure 0.55 of
  $h\nu - {\rm IE}$, against the parabola's 0.5, and keep 0.5 for consistency with
  $Y_2$ (< 10% on the heating); their per-energy partition is in `pahchem.photo`
  (`PARTITION_NODES`);
- the C$_{60}$-like sticking of grains with $N_{\rm C}\gtrsim60$;
- a-C:H of intermediate H content: the work function between 2.6 and 4.5 eV is
  measured only at the ends of the Schäfer et al. series in their text.

## The PAH / grain seam

In the PAH-size bins (N_C 20–400) the aromatic carbon is charged with pahchem's
PAH tables (`pahchem.group_tables`); above N_C,max the same carbon is charged as
`ac`. `diagnostics/gsd/check_pah_grain_seam.py` builds both at each bin's mean N_C
(PAHdb v4.00 molecules) and compares them in fixed test cells. In every bin and cell
the PAH tables heat the gas 4.5–7.5 times more per C atom than `ac`, and charge it
less positively. In dense gas the PAHs stay neutral (0.1–0.5% anions), where `ac`
grains carry 25–37% anions.

Swapping one ingredient at a time from the PAH side into `ac`, in a four-state chain
that reproduces both models (N_C ≈ 63), multiplies the `ac` heating by:

| ingredient | WNM | CNM | PDR |
|---|---|---|---|
| electron capture | 2.8 | 1.7 | 2.2 |
| energy per photoelectron | 1.6 | 1.9 | 1.3 |
| absorption | 1.13 | 1.45 | 1.16 |
| yield per absorbed photon | 1.06 | 1.18 | 1.07 |
| ion and anion channels (associative detachment, mutual neutralisation, charge transfer) | 1.0 | 0.87 | 0.99 |

What the measurements say about each:

- **Thresholds.** PAHdb's DFT ionisation energies, which the PAH tables use, run
  0.4–0.6 eV below measured values for the same molecules: coronene 6.79 against
  7.29, ovalene 6.13–6.28 against 6.71, benzo[ghi]perylene 6.24–6.57 against 7.15,
  hexabenzocoronene 6.31 against 6.86. `ac`'s formula is within 0.1–0.3 eV for most.
  The PAH thresholds also feed the energy of their photoelectrons (0.46 of
  $h\nu - {\rm IP}$).
- **Cation recombination.** The measured rates at 300 K for N_C 6–16 (pahchem's
  `pah_electron_recombination.csv`) are 0.3–4.1×10$^{-6}$ cm$^3$ s$^{-1}$. `ac`, with
  WD01's sticking, gives 1.1–2.1×10$^{-6}$, within a factor of 2. The PAH default
  (BT94 with sticking 1) gives 0.7–1.1×10$^{-5}$, 3–30 times too high. pahchem's
  optional `lab_trend` follows the data. The measured/`ac` ratio rises with N_C
  (≈0.5 at N_C ≤ 12, 1.2–2 at 14–16).
- **Neutral attachment.** C$_{60}$ (Viggiano et al. 2010) gives $s_e\approx0.8$,
  close to pahchem's statistical value (≈1 above N_C ≈ 50) and 4 times WD01's 0.19.
- **Geometry.** pahchem captures on a planar disc of radius 0.9 √N_C Å
  (0.82 π a²); `ac` grains are compact spheres. The disc's area is 1.3 times the
  sphere's at N_C = 63 and 2.4 times at 400. For planar PAHs this is physical, and
  it is part of what distinguishes them from compact grains.
- **Associative detachment** of anions by H (7.8×10$^{-10}$ cm$^3$ s$^{-1}$,
  ±50%, Demarais et al. 2012) is measured on PAH anions. It sets the anion fraction
  in dense gas, and it is absent from the grain model.

## Gas-electron secondaries: energy threshold

While checking `cm20`, the DS79 gas-electron secondary yield
(`gas_secondary_yield_at`, shared by all materials) was found to count
secondaries that cannot escape. It required only $E_0 > W$ of the impinging
electron. A secondary taken from the Fermi level needs $W + E_s$ to escape over
the barrier $E_s = (Z+1)e^2/a$, while the primary brings $E_0 + W$. So $E_0 > E_s$
is also needed, and DS79's escape factor $(1+E_s^2/8)^{-1/2}$ does not enforce
it. The yield now requires $E_0 > \max(W, E_s)$, and the thermal quadrature starts
there (`_thermal_nodes`).

The second condition binds only when $E_s > W$: small, positive grains, e.g.
$Z\ge4$ at 1.3 nm. Only the `delta` and `E_bar` planes of the RT-group tables
change; the photon kernels are identical. For a 0.4 nm graphite grain at $Z=+5$
and $10^4$ K, $\delta$ falls 47-fold; at $10^6$ K and above the change is
negligible. Graphite equilibrium charges in warm gas (8000–$10^4$ K) fall by
1–7%, most at 1 nm, and the photoelectric heating rises by 1–20%: at 1 nm,
+20% for $G_0=10$, $n_e=7.2\times10^{-3}$ cm$^{-3}$ and +18% for $G_0=10^3$,
$n_e=1$ cm$^{-3}$. Tables exported before this change keep the old values until
they are regenerated (`calima-export --stages dust_charging_rtgroups,dust_charging_isrf`).

## Not included

- The a-C:H population ($E_g = 2.5$ eV) of the carbon GSD: it needs its own
  optics (the Jones 2012 $n$, $k$ on CDS) before it can be charged.
- The core/mantle structure of large CM20 grains: photoelectrons come from
  within $l_e$ of the surface, i.e. from the a-C mantle, so the whole grain takes
  the mantle's composition.
- RAMSES: material code 3 (IP, EA with $A_C$ and $E_{\rm bg}$, tabulated $s_e$)
  is not yet in `calima/dust_charging_rtgroups.f90`.
