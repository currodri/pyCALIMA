(physics-carbon-optics)=
# Optics and heat capacity of the carbon carriers

The carbon of the GSD solver lives on one grid in carbon mass
(`gsd/carbon.py`, `CarbonLayout`). Two optical carriers share its smallest
bins (`CARRIERS`):

| Carrier | Material | $E_g$ | $\rho$ [g cm$^{-3}$] | Source |
|---|---|---|---|---|
| `aromatic` | THEMIS CM20 structure: a-C below 20 nm, a-C:H core in a 20 nm a-C mantle above | 0.1 eV | 1.6 | DustEM `Q_CM20` below 20 nm; coated-sphere Mie with the Jones (2012) $n$, $k$ above (`themis_optics.mie_efficiencies`) |
| `aromatic_dustem` | the same | 0.1 eV | 1.6 | DustEM v4.3 `Q_CM20`/`G_CM20` throughout (`themis_optics.efficiencies`): the benchmark |
| `aliphatic` | a-C:H | 2.5 eV | 1.33 | Jones (2012) optEC(s)(a) $n$, $k$ (`jones2012_optics`) |

THEMIS and DustEM guide these choices; they do not fix them. Where a choice can be
made better founded, the default carrier takes it, and the DustEM version stays
available for benchmarks against DustEM (`aromatic_dustem`, and the material
`cm20` of the charging tables).

## Hydrogen bookkeeping

A bin's mass $M$ counts carbon only. Each carrier carries H at the fixed atomic
fraction of its band gap, $X_{\rm H}=E_g/4.3$ eV (Jones 2012, paper I), so a
grain of carbon mass $m$ has the total mass $m(1+f_{\rm H})$, with
$f_{\rm H} = X_{\rm H}m_{\rm H}/[(1-X_{\rm H})m_{\rm C}]$ (0.12 for a-C:H, 0.002
for a-C), and at its carrier's density its own radius. An a-C:H grain is 1.10
times the radius of an a-C grain with the same carbon. The optics see that
radius, and `per_gram_carbon` gives cross-sections per gram of carbon, the unit
of the bins' tables. The large CM20 grains' core H is not counted, so their
carbon is overestimated by up to ~10%.

## a-C:H optical constants (Jones 2012)

`jones2012_optics` reads the size-dependent tables of Jones (2012, A&A 542, A98;
corrigendum 545, C3), CDS catalogue J/A+A/542/A98, registered as the `fetch`
dataset `optical-jones-2012-size` (`calima-fetch-data fetch
optical-jones-2012-size`). They give $n$ and $k$ for 14 band gaps from −0.1 to
2.67 eV and radii of 0.33, 0.5, 1, 3, 10, 30 and 100 nm, at 1000 wavelengths from
0.02 µm to 1 m.

- **Column labels.** The ReadMe labels columns 7 and 8 as $E_g$ = 1.5 and
  1.15 eV; they are 1.25 and 1.5 eV (paper II, Table 1). The module reads them
  so, and a test checks that $k$ then falls with $E_g$ in the visible for
  $a\ge3$ nm. At smaller radii it does not: the size-limited aromatic clusters
  lower $k$ in the low-$E_g$ columns.
- **Density.** $\rho = 1.3+0.4\,e^{-(E_g+0.2)}$ g cm$^{-3}$ (paper II), 1.60
  for a-C and 1.33 for a-C:H.
- **Between radii.** $n$ and $\ln k$ are interpolated linearly in $\ln a$; outside
  0.33–100 nm the nearest table is used.
- **Below 0.02 µm** the index comes from the Henke atomic factors at the same
  composition and density. It is not rescaled, and at 0.02 µm it gives $Q_{\rm abs}$
  1.23–1.27 times the table for a-C and 0.53–0.70 times for a-C:H (`splice_ratio`).
- Grains are homogeneous Mie spheres.

### Against THEMIS

Below 20 nm the CM20 grains are a-C, so the $E_g$ = 0.1 eV column describes the
same grains as DustEM's `Q_CM20`, except where THEMIS changed the constants
(Jones et al. 2013, A&A 558, A62, App. C: band reassignments and widths, the UV
bump narrowed below 3 nm, the σ–σ* band shifted and strengthened below 10 nm).
`diagnostics/dust_radiation/check_jones2012_optics.py` gives, for
$Q_{\rm abs}$(Jones)/$Q_{\rm abs}$(CM20), the median over each band:

| $a$ [nm] | FUV | bump | optical | NIR | MIR | FIR |
|---|---|---|---|---|---|---|
| 0.35 | 1.22 | 1.08 | 1.72 | 1.37 | 0.81 | 0.71 |
| 1 | 1.18 | 1.13 | 1.10 | 1.10 | 1.04 | 1.00 |
| 2 | 1.15 | 1.12 | 1.00 | 0.87 | 0.71 | 0.43 |
| 3 | 1.03 | 1.09 | 1.08 | 1.08 | 1.08 | 1.09 |
| 5 | 0.99 | 1.07 | 1.01 | 0.96 | 0.89 | 0.74 |
| 10–20 | 1.000 | 1.000 | 1.000 | 1.000 | 1.00 | 1.003 |

At 10–20 nm the tables reproduce THEMIS to 0.3% (bands aside, ≤ 8%); THEMIS's
changes act below 10 nm. At the radii between tables (2 and 5 nm) the far-IR
ratio drops to 0.43–0.74: there the far-IR $Q_{\rm abs}/a$ of a-C rises 180-fold
from 1 to 3 nm, and not as a power law. That is the size-limited band gap of
small grains, which log-radius interpolation cannot follow on a 7-radius grid.
**a-C therefore keeps THEMIS's `Q_CM20`** (98 radii). The a-C:H column varies
smoothly and monotonically with radius (steps between tabulated radii within a
factor of 2.5, mostly 1.5), so the interpolation serves it. It has no THEMIS
counterpart, and below 10 nm it inherits the unmodified model.

### Per-bin tables

`diagnostics/gsd/check_carbon_carriers.py` builds both carriers' per-bin tables
$\kappa(\lambda, s)$ on the four PAH-size bins (N_C 20–400) with the A2 settings
of `gsd.optics` (128 nodes per bin, $s$ grids refined to $3\times10^{-3}$).
The $s$ grids need 17–25 points per bin for the aromatic carrier and 11–12 for
the aliphatic one. Per gram of carbon at $s=0.5$, aliphatic over aromatic:

| bin, $N_{\rm C}$ | 0.1 µm | 2175 Å | 0.55 µm | 3.4 µm | 6.2 µm | 11.3 µm | 100 µm |
|---|---|---|---|---|---|---|---|
| 20–42 | 1.68 | 0.33 | 0.50 | 24 | 0.16 | 0.023 | 0.41 |
| 189–400 | 1.69 | 0.39 | 0.053 | 2.4 | 0.064 | 0.036 | 0.060 |

The a-C:H carrier has no UV bump, weak visible absorption, a strong 3.4 µm band
(10 times stronger per radius at 0.33 nm than at 3 nm: H-passivated surfaces) and
essentially no aromatic 6.2 or 11.3 µm bands.

## Large grains recomputed

THEMIS's grains above 20 nm are an a-C:H core ($E_g$ = 2.5 eV) inside a 20 nm a-C
mantle ($E_g$ = 0.1 eV). `themis_optics.mie_efficiencies` computes them by
coated-sphere Mie (`tools/coated_mie.py`) with the Jones (2012) 100 nm $n$, $k$
(the bulk) for both layers, and Henke's volume-weighted index below 0.02 µm, as
for the tables.

The coated-sphere solver is Yang's (2003, Appl. Opt. 42, 1710) layered
recursion, written with logarithmic derivatives and ratios of Riccati–Bessel
functions only. Bohren & Huffman's BHCOAT recurs the Riccati–Neumann functions
upwards, and for 1 µm carbon grains in the far UV ($|{\rm Im}\,m|\,x\gtrsim50$)
it returns NaN or $Q_{\rm sca}$ of several times the geometric limit. That is the
same pattern as the broken region of `Q_CM20` (below). The solver reproduces
`miepython`'s homogeneous spheres to $10^{-5}$ when the layers are equal, and the
core sphere when the mantle vanishes.

For 30–300 nm the recomputed grains reproduce `Q_CM20` to 0.1% in the
continuum (median; $Q_{\rm sca}$ to 1.6% everywhere). The exceptions are the
6.9 and 7.7 µm C–H bending bands, 7–12% off, where THEMIS changed the band
intensities and widths (Jones et al. 2013, App. C). So THEMIS's large grains
are these constants in a coated sphere, and the recomputation replaces the
table where the table fails. That is from ~0.5 µm up: at 1–5 µm $Q_{\rm abs}$ in
the near-IR is 0.4–0.65 of the table's clamped 0.840, and the far-UV
$Q_{\rm sca}$ is ~1.2 instead of 12–41.

## Heat capacity

The default (`carbon_heat_capacity`, used by `CarbonCarrier.heat_capacity`) follows
THEMIS's recipe (Jones et al. 2013, App. B): a sum over the atoms. It keeps only
harmonic terms, so that no atom carries more than 3 k:

| Atoms | Modes | Source |
|---|---|---|
| sp² C | graphite: 2-D Debye spectra, $\Theta$ = 863 K (1 mode) and 2504 K (2 modes) | Draine & Li (2001) eq. 9, with $f_n$ normalised by $n$ (the printed $1/n$ does not reach 3 k) |
| sp³ C | polyethylene skeleton: 2 Tarasov modes ($\Theta_1$ = 519 K, $\Theta_3$ = 158 K) and the C–C stretch | Wunderlich (1995); ATHAS (Wunderlich 1990, Fig. 9) |
| H (on sp³ C) | half of polyethylene's six CH$_2$ group vibrations | ATHAS |
| H (aromatic, option) | C–H at 886, 1161, 3030 cm$^{-1}$ | Draine & Li (2001) |

The composition comes from the band gap through Jones (2012, paper I).
$X_{\rm H} = E_g/4.3$ eV. For the defected-graphite network,
$X_{\rm sp^3} = X_{\rm H}/2$ (valid for $0\le X_{\rm H}\le0.35$). For the random
covalent network, $X_{\rm sp^3}/X_{\rm sp^2} = (8X_{\rm H}-3)/(8-13X_{\rm H})$
(valid for $0.38\le X_{\rm H}\le0.62$). In both, the H is on sp³ C. This gives
$X_{\rm sp^3}$ = 0.012 for a-C and 0.33 for a-C:H; a gap between the networks raises.
A grain of $N$ atoms has $3N-6$ modes, and the CM20 structure (a-C:H core in an a-C
mantle above 20 nm) is summed layer by layer (`heat_capacity_cm20`).

Checks (`tests/test_carbon_heat_capacity.py`,
`diagnostics/dust_radiation/check_cm20_heat_capacity.py`):

- Every term tends to 3 k per mode.
- The graphite term is DustEM's `C_Gra.DAT` (DL01) to 4%.
- The polyethylene terms give the measured C$_p$ of amorphous polyethylene (Gaur &
  Wunderlich 1981) to 2% at 100 K, and 0.87–0.90 of it at 50 and 150 K. They fall
  below it at 10–20 K (0.27–0.53: the low-temperature excess of glasses) and
  towards its glass transition at ~237 K (0.80 at 200 K).

Against DustEM's `C_CM20.DAT`, per atom of a-C at 1.6 g cm$^{-3}$: the table is
1.08–1.20 times the harmonic model at 300–1000 K and 1.50–1.74 times at 5000 K. The
table is reproduced to 1–4% from 27 to 5000 K when its sp³ term is taken to be the
liquid-polyethylene C$_p$ of Gaur & Wunderlich (1981),
$0.04325\,T + 17.919$ J mol$^{-1}$ K$^{-1}$ per CH$_2$, extrapolated far beyond its
600 K data, with a fitted $X_{\rm sp^3}\approx0.06$. That extrapolation reaches 28R per
CH$_2$ at 5000 K, against 9R for its 9 modes, which suggests the excess comes from it. Below
30 K the table is 1.6–4.7 times the model. Amorphous solids do show a low-T excess
(two-level systems), but this model leaves it out until a measurement for a-C(:H)
is in hand. The excess matters for the photo-fragmentation of the smallest grains
(Jones & Ysard 2025: a rate $\propto e^{-6.4\,{\rm eV}/kT}$), whose peak
temperatures it sets.

DustEM's table stays available (`themis_optics.heat_capacity`; the
`aromatic_dustem` carrier) for benchmarks.

## Known problems of the DustEM tables

- `Q_CM20.DAT`: for $a\gtrsim0.9$ µm below 0.1 µm, $Q_{\rm sca}$ reaches 15–670.
  Separately, $Q_{\rm abs}$ is exactly 0.840 over 0.04–2.95 µm for $a\ge0.45$ µm.
  Only the top bin of a grid that reaches 1 µm sees either (see also
  {doc}`charging-cm20`).
- `C_CM20.DAT`: above the Dulong–Petit limit at high $T$, from an extrapolated
  liquid-polyethylene term (above).
