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
- **Between band gaps.** $n$ and $\ln k$ are linear in $E_g$ between the columns, so
  any $E_g$ from −0.1 to 2.67 eV can be used.
- **Between radii.** See "Size-limited band gap" below: by default the steep part of
  the size dependence is carried by an effective gap, and only the smooth remainder is
  interpolated in $\ln a$. Outside 0.33–100 nm the nearest table is used.
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

| $a$ [nm] | FUV | bump | optical | NIR | MIR | FIR | FIR, plain $\ln a$ |
|---|---|---|---|---|---|---|---|
| 0.35 | 1.22 | 1.08 | 1.72 | 1.37 | 0.81 | 0.71 | 0.71 |
| 1 | 1.18 | 1.13 | 1.10 | 1.10 | 1.04 | 1.00 | 1.00 |
| 1.5 | 1.17 | 1.17 | 1.06 | 1.05 | 1.03 | 0.98 | 0.35 |
| 2 | 1.15 | 1.16 | 1.08 | 1.08 | 1.06 | 1.04 | 0.43 |
| 3 | 1.03 | 1.09 | 1.08 | 1.08 | 1.08 | 1.09 | 1.09 |
| 5 | 0.99 | 1.09 | 1.08 | 1.07 | 1.04 | 0.96 | 0.74 |
| 7 | 0.95 | 1.10 | 1.05 | 1.04 | 1.03 | 1.01 | 0.79 |
| 10–20 | 1.000 | 1.000 | 1.000 | 1.000 | 1.00 | 1.003 | 1.003 |

At 10–20 nm the tables reproduce THEMIS to 0.3% (bands aside, ≤ 8%); THEMIS's
changes act below 10 nm. The last column is plain $\ln a$ interpolation of the
7 radii (`size_interp="lna"`): between the tables at 1, 3 and 10 nm it misses the
far-IR by factors up to 2.8, because the far-IR $Q_{\rm abs}/a$ of a-C rises
180-fold from 1 to 3 nm, and not as a power law.

#### Size-limited band gap

That steep rise is the size-limited band gap of paper III (Sect. 2.3, App. B): a
particle's aromatic clusters cannot be larger than the particle, so its gap cannot
fall below $E_{\rm lim}(a)$. At 1–10 nm the tables bear this out directly. The
low-$E_g$ columns of a radius match the bulk (100 nm) table at one effective gap,
$E_{\rm eff}=\max(E_g, E_{\rm lim}(a))$, to an rms of 0.01–0.03 in $\ln k$ in the
far-IR at 1 nm and 0.02 at 3 nm. An $E_g$ = 0.5 grain at 1 nm behaves like bulk at
0.76 eV, close to the 0.74 found for $E_g$ = 0.1, so one $E_{\rm lim}(a)$ serves every column.
Below ~0.5 nm no bulk gap describes the tables (surface CH$_n$ groups dominate).

`jones2012_optics.refractive_index` (default `size_interp="effective_gap"`)
therefore writes $m(a) = m_{\rm bulk}(E_{\rm eff}(a)) + R(a)$. The remainder
$R = m_{\rm table} - m_{\rm bulk}(E_{\rm eff})$ is known at the tabulated radii and
is interpolated in $\ln a$ (for $n$ and $\ln k$). The tables are reproduced at their
radii; where $E_g > E_{\rm lim}$ (a-C:H at every radius) the scheme equals plain
interpolation; below 1 nm it falls back to plain interpolation.

$E_{\rm lim}(a) = 0.75\ {\rm eV}\,(1\ {\rm nm}/a)^{1.06}$ (`size_limited_gap`)
is the power law through the gaps that make the a-C far-IR match `Q_CM20` at 1–5 nm
(scatter ±7%). `Q_CM20` was computed from Jones's continuous-in-radius model, so it
is the reference between the published radii. The law also matches the best-fitting
gaps of the tables themselves at 1 nm (0.74 eV) and 3 nm (0.24 eV). Paper III's
empirical limit, $1/a - 0.2$ eV, gives 0.80 and 0.13 there, and puts the 2 nm
far-IR at 1.8 times `Q_CM20`.

With the scheme, a-C follows THEMIS to within 0.95–1.18 in every band from 1 to
7 nm, the residue being THEMIS's own changes. The default `aromatic` carrier still
uses `Q_CM20` below 20 nm. The scheme matters for intermediate band gaps, which
THEMIS does not tabulate.

#### Hydrogen of a particle

A particle's hydrogen is the bulk $X_{\rm H}$ plus the H atoms passivating its
surface carbon (paper III, App. A, eqs. A.1–A.13). The surface layer is one atom
deep, with atomic radius $r_{\rm atom}=(3\bar A/4\pi\rho N_A)^{1/3}$ and
$\bar A = 12-11X_{\rm H}$. It holds the fraction
$F_s = 1-(1-2r_{\rm atom}/a)^3$ of the atoms, and each surface carbon carries
$m_C/2$ extra H, with $m_C$ the mean carbon coordination of the DG or eRCN
composition (paper I). `h_per_carbon(a, E_g)` is
$(X_{\rm H}+X_{\rm Hs}(a))/(1-X_{\rm H})$ H atoms per C atom:

- For $E_g$ = 0.1 eV it is 1.5 at 0.4 nm and 0.86 at 1.2 nm. Jones & Ysard
  (2025) quote ≈ 1.4 and 0.8.
- For H-rich grains the surface share is 3–6% at 30 nm, which paper III calls
  negligible.

`e_gap_of_h_per_carbon(a, h/C)` inverts it at fixed radius, mapping a tracked H/C
to the band gap of each radius. Within each network, H/C rises with $E_g$. Below
~1.5 nm the eRCN branch starts up to 2% below the end of the DG branch, because the
sp³ fraction jumps between the networks. There the DG branch is used to its end.

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

### Tables for tracked hydrogen (rung M1)

The covalent bins from $N_{\rm C,max}$ (400 C, 1.06 nm) to $a_{\rm H,max}$ (53 nm)
store $M_{\rm H}$ next to $M_{\rm C}$, so their optics depend on the hydrogen as well
as on the in-bin state $s$. `gsd.carbon.HydrogenatedCarbon` gives a grain's optics at
a bulk band gap (Jones's constants, effective-gap size interpolation). Its radius
follows from its carbon, its bulk H and the density of that gap; the surface H adds
mass but no volume. `gsd.optics.adaptive_kappa_tables_x` builds $\kappa(\lambda, s,
E_g)$ per gram of carbon for each bin, refining both grids until $\ln\kappa$, linear in
each, meets the tolerance at every wavelength
(`diagnostics/gsd/check_hydrogenated_tables.py`).

**The table axis is the band gap the bin's grains share.** RAMSES turns a bin's
$M_{\rm H}/M_{\rm C}$ into it with one small table per bin (`bin_h_per_c`, increasing
in $E_g$). The tracked H/C itself is a poor axis: at a common H/C every in-bin grain
reaches its own floor and saturation, and crosses each tabulated band gap, at a
different H/C. The bin's tables then carry a kink per node and need roughly four times
more points on each axis. On the 2.8–7.5 nm bin (32 nodes, tolerance $3\times10^{-3}$):

| axis | $s$ points | composition points |
|---|---|---|
| H/C | 76–80 | 190–216 |
| hydrogenation fraction (each grain from its floor to its saturation) | 37 | 108–120 |
| bulk band gap, starting from Jones's columns | 37 | 39 |

The tables keep all 241 wavelengths of the RAMSES optics grid, so RAMSES can average
them over any groups, in number and limits, as it does now at start-up. Refining on the
group and Planck means only (`--refine groups`, a reference group set given by its
edges) needed no fewer band-gap points.

At the plan's settings (128 nodes per bin, tolerance $3\times10^{-3}$, 7.4 min on one
core):

| bin | $a$ [nm] | $s$ points | $E_g$ points | 8-group table | per-wavelength table |
|---|---|---|---|---|---|
| 4 | 1.06–2.82 | 65 | 47 | 573 kB | 16.9 MB |
| 5 | 2.82–7.50 | 38 | 39 | 278 kB | 8.2 MB |
| 6 | 7.50–20.0 | 34 | 41 | 261 kB | 7.7 MB |
| 7 | 20.0–53.1 | 36 | 48 | 324 kB | 9.5 MB |

The per-wavelength tables (in double precision) are read at start-up and can be
dropped after the group averaging; the group tables, 1.4 MB for the four bins, stay.

The hydrogen matters: from $E_g$ = 0.1 to 2.5 eV the bin opacity at 100 µm falls about
4000-fold, the visible about 40-fold, and the 0.1 µm far-UV rises about 2-fold. At
saturation the tables equal the `aliphatic` carrier's to 0.1%. At the a-C floor they
match the `aromatic` carrier to 1% in bin 6. In bins 4–5 they differ by 0.83–1.49,
from THEMIS's band changes in `Q_CM20`, most at 11.3 µm. In bin 7 they differ by
0.91–1.21, where the aromatic carrier is a core/mantle grain and these are homogeneous.

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
