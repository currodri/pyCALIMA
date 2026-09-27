(physics-radiation)=
# Radiation: grain optics and radiation fields

This page covers how pyCALIMA builds the optical properties of each dust and
PAH bin, which radiation fields it uses, and how $G_0$ is defined. None of
this runs inside the solver. The tables go to
`model_data/[<model_name>/]optical_properties/`, and RAMSES-CALIMA and the
charging and heating exporters read them from there. Units are cgs unless
stated. Sizes in the grain-size JSON are in µm; the distribution classes take
cm.

## Bins and size distributions

`models/grain_size_distribution.json` lists the bins. Each bin has an `id`, a
`composition` (`graphite` or `silicate`), an `is_pah` flag and an integer
`bin_rank` (`models/grain_size_config.py:171-184`). Two parameter sets,
`basic` and `shattering`, each hold one array per key, indexed like the bins:
`a0`, `amin`, `amax`, `sigma` and `s` (`models/grain_size_config.py:207-218`).
Every exporter on this page uses `basic`. `a0` is the representative size and
`[amin, amax]` the size range, all in µm. `sigma` is the width, and `s` is the
material density in g cm$^{-3}$: `build_lognormal_distribution` passes it as
the fifth, `grain_density`, argument (`models/grain_size_config.py:298-305`).
The default config has two PAH bins, two graphite bins and two silicate bins,
with $s = 2.0$, $2.2$ and $3.3$ g cm$^{-3}$ respectively.

The default distribution is `LogNormal_Distribution`. For a bin of dust mass
density $\rho_d$ (g cm$^{-3}$) it gives

$$
\frac{dn}{da} = \frac{C}{a^{4}}\exp\!\left[-\frac{\ln^{2}(a/a_0)}{2\sigma^{2}}\right],
\qquad a_\mathrm{min}\le a\le a_\mathrm{max},
$$ (eq-lognormal)

and zero outside that range (`models/grain_distributions.py:35-40`). The
constant is $C=\rho_d/S$, with

$$
S = \frac{4\pi s}{3}\int a^{-1}\exp\!\left[-\frac{\ln^{2}(a/a_0)}{2\sigma^{2}}\right]da ,
$$

evaluated with the trapezoidal rule on a 1000-point grid
(`models/grain_distributions.py:26`, `models/grain_distributions.py:31-33`).
So $\int m(a)\,(dn/da)\,da=\rho_d$, where $m(a)=\tfrac{4}{3}\pi s a^{3}$. The
representative grain mass is $\tfrac{4}{3}\pi s a_0^{3}$
(`models/grain_distributions.py:29`).

The same module has drop-in alternatives with the same five-argument
signature, where `sigma` changes meaning: `Flat_Distribution`,
`PowerLaw_Distribution` ($dn/da\propto a^{-\alpha}$ with $\alpha$=`sigma`),
the power-law classes with dual, lower or upper exponential cutoffs,
`Exponential_Distribution` and `PowerLaw_ExpCutoff_Distribution`
(`models/grain_distributions.py:117-466`). You can pass them to the dust
optical exporter through `distribution_class` or `distribution_class_map`
(`models/dust_radiation/export_dust_optical_properties.py:54`).

:::{note}
Three quirks in `LogNormal_Distribution` affect results:

- **Grid lower edge.** The normalisation grid is
  `np.logspace(np.log(amin), np.log10(amax), 1000)`
  (`models/grain_distributions.py:26`). Its lower edge uses the natural log, so
  $S$ is integrated from about $10^{-17}$ cm rather than from $a_\mathrm{min}$,
  while `n_density` cuts off at $a_\mathrm{min}$. For the default bins, the
  mass inside $[a_\mathrm{min},a_\mathrm{max}]$ comes to 0.9552 of $\rho_d$ for
  `PAHBin_01` and within 0.5 per cent of $\rho_d$ for the other bins
  (evaluated numerically).
- **Log base in the averages.** `averaged_over`, `averaged_over_mass`,
  `averaged_over_column` and `averaged_over_number` use $\log_{10}$ in the
  exponent (`models/grain_distributions.py:42-82`), whereas
  {eq}`eq-lognormal` uses $\ln$. None of these averages is used by the
  optical exporters on this page.
- **Known bug in the cutoff class.**
  `PowerLaw_ExpCutoff_Distribution.averaged_over_number` weights by
  $X\,a^{+\alpha}$ rather than $X\,a^{-\alpha}$
  (`models/grain_distributions.py:463`). This is the bug recorded by the
  strict xfail in the test suite.
:::

## Dust bins: optical properties

### Two methods for $Q$

The dust exporter `export_dust_optical_properties` gets
$Q_\mathrm{abs}(a,\lambda)$, $Q_\mathrm{sca}(a,\lambda)$ and $g(a,\lambda)$ in
one of two ways.

**`precomputed`** is the function default
(`models/dust_radiation/export_dust_optical_properties.py:54`) and the method
`calima-export` uses, since it calls the function without `cabs_method`
(`models/export_all_grain_data.py:828`). It reads the tables
{cite:t}`DraineLee1984` and {cite:t}`LaorDraine1993` computed: `Gra_81` for
graphite and `suvSil_81` for smoothed-UV astronomical silicate
(`models/dust_radiation/dust_oppacity.py:1778-1783`). Each table has 81 radii
from $10^{-3}$ to 10 µm and 241 wavelengths from $10^{-3}$ to $10^{3}$ µm.

The code interpolates $\log Q$ linearly in $\log\lambda$
(`models/dust_radiation/dust_oppacity.py:1694-1697`), then linearly in
$\log a$ (`models/dust_radiation/dust_oppacity.py:1738-1741`). $g$ is
interpolated linearly in $\lambda$ and in $a$. `np.interp` clamps at the ends
of the tabulated range. The default silicate bin `DustBin_03` starts at
$a_\mathrm{min}=5\times10^{-4}$ µm, below the first tabulated radius, so for
its smallest grains $Q$ is held at its $a=10^{-3}$ µm value.

**`mie`** is the default when you run the module directly
(`models/dust_radiation/export_dust_optical_properties.py:239`). It computes
Mie efficiencies with `miepython` from the optical constants in
`draine_lee_1984/eps_suvSil`, `callindex.out_CpaD03_0.01` and
`callindex.out_CpeD03_0.01` (`models/dust_radiation/dust_oppacity.py:1856-1858`).
The refractive index is $m=n-ik$, interpolated linearly in $\lambda$
(`models/tools/mie_theory.py:128-130`), and
$Q_\mathrm{abs}=Q_\mathrm{ext}-Q_\mathrm{sca}$
(`models/tools/mie_theory.py:185-186`).

Graphite uses the 1/3–2/3 approximation (`models/tools/mie_theory.py:201-203`):

$$
Q = \tfrac13 Q_{\parallel} + \tfrac23 Q_{\perp},\qquad
g = \frac{\tfrac13 g_{\parallel}Q_{\mathrm{sca},\parallel}+\tfrac23 g_{\perp}Q_{\mathrm{sca},\perp}}{Q_\mathrm{sca}} .
$$

When $|m|x>1000$, with $x=2\pi a/\lambda$, a large-grain approximation
replaces the full Mie series (`models/tools/mie_theory.py:136-182`). A Henke
X-ray extension exists in the code {cite:p}`Henke1993` but never runs in this
path: the material labels `suvSil` and `graphite` are not keys of
`DUST_METADATA` (`models/tools/mie_theory.py:119-126`), and the tabulated
constants already cover the $10^{-3}$ µm lower edge of the output grid.

### Averaging over a bin

Both methods integrate over the bin's size distribution with 30 log-spaced
sizes between $a_\mathrm{min}$ and $a_\mathrm{max}$
(`models/dust_radiation/dust_oppacity.py:1810`). The weight is
{eq}`eq-lognormal` evaluated for $\rho_d=1$ g cm$^{-3}$
(`models/dust_radiation/dust_oppacity.py:1812-1814`). The result is a mass
opacity in cm$^{2}$ per gram of dust:

$$
\mathcal{C}_x(\lambda)=\int_{a_\mathrm{min}}^{a_\mathrm{max}}
\left.\frac{dn}{da}\right|_{\rho_d=1}\pi a^{2}\,Q_x(a,\lambda)\,da ,
\qquad
Q_\mathrm{rp}=Q_\mathrm{abs}+(1-g)\,Q_\mathrm{sca}.
$$ (eq-bin-opacity)

Here $x$ is `abs`, `sca` or `rp` (radiation pressure). This is implemented at
`models/dust_radiation/dust_oppacity.py:1744-1752`, and in the Mie path at
`models/dust_radiation/dust_oppacity.py:1927-1935`.

### Average over the Mathis ISRF

Each file also stores one average of each quantity over the
{cite:t}`Mathis1983` field, taken over $0.1\le E\le13.6$ eV:

$$
\langle\mathcal{C}_x\rangle=\frac{\int\mathcal{C}_x(E)\,u_E\,dE}{\int u_E\,dE},
\qquad E\,[\mathrm{eV}]=\frac{1.239841984\times10^{-4}}{\lambda\,[\mathrm{cm}]},
$$

with $u_E$ from `Mathis83_radiation_field`
(`models/dust_radiation/dust_oppacity.py:344`,
`models/dust_radiation/dust_oppacity.py:361-369`).

### Output files

| file | content | code |
|---|---|---|
| `averaged_cross_section_<bin_id>.txt` | header; `NWAV` = 241; one line of ISRF averages; then 241 rows of `lambda[Å] C_abs C_sca C_rp`. The wavelengths are log-spaced over $10^{-3}$–$10^{3}$ µm, and the values are cm$^2$ g$^{-1}$ | `models/dust_radiation/export_dust_optical_properties.py:123`, `:159`, `:174-196` |
| `averaged_cross_section_<bin_id>_quicklook.png` | plot of $\mathcal{C}_\mathrm{abs}$, $\mathcal{C}_\mathrm{sca}$ and $\mathcal{C}_\mathrm{ext}$ | `models/dust_radiation/export_dust_optical_properties.py:198` |
| `Im_n_<bin_id>` (silicate); `Im_n_<bin_id>_pe` and `Im_n_<bin_id>_pa` (graphite) | $\mathrm{Im}(n)$ against $\lambda$ [Å], resampled log-uniformly onto the source's own range and point count. The sources are `eps_suvSil` and `callindex.out_Cp{e,a}D03_0.10` | `models/dust_radiation/dust_oppacity.py:149-208` |

:::{note}
The ISRF line is labelled `ISRF_AVG_CROSS_SECTIONS_CM2`
(`models/dust_radiation/export_dust_optical_properties.py:187`), but for dust
bins the values are in cm$^2$ g$^{-1}$, like the spectral columns. The Mie path
reads the graphite constants at $a=0.01$ µm (`_0.01` files), while the
$\mathrm{Im}(n)$ tables are built from the $a=0.10$ µm files (`_0.10`).
:::

## PAH bins: optical properties

The PAH exporter `export_pah_optical_properties` uses the
{cite:t}`LiDraine2001` efficiency tables `PAHneu_30` (neutral) and `PAHion_30`
(ionised) (`models/PAH_radiation/pah_oppacity.py:176-180`). Each has 30 radii
from $3.548\times10^{-4}$ to $10^{-2}$ µm and 1201 wavelengths. $Q_\mathrm{abs}$
and $Q_\mathrm{sca}$ are interpolated log–log, first in $\lambda$ and then in
$a$. The code falls back to linear interpolation wherever a column has
non-positive entries (`models/PAH_radiation/pah_oppacity.py:233-272`).

Unlike the dust bins, a PAH bin gets no size average: the code evaluates it at
the single size $a_0$, per grain:

$$
C_x(\lambda)=\pi a_0^{2}\,Q_x(a_0,\lambda),\qquad
Q_\mathrm{rp}=Q_\mathrm{abs}+(1-g)\,Q_\mathrm{sca}
$$

in cm$^2$ (`models/PAH_radiation/pah_oppacity.py:275-290`).

The output uses the same file stem as the dust tables,
`averaged_cross_section_<bin_id>.txt`. It has 300 wavelengths log-spaced over
$3\times10^{-3}$–$10^{3}$ µm (`models/PAH_radiation/pah_oppacity.py:352`), and
each row reads `lambda C_abs C_sca C_rp | lambda C_abs C_sca C_rp`: neutral on
the left, ionised on the right. Two header lines hold the Mathis-ISRF averages
for the neutral and ionised states
(`models/PAH_radiation/pah_oppacity.py:383-425`).

:::{note}
PAH and dust tables share a name but not a normalisation. PAH tables hold cm$^2$
per grain at $a_0$; dust tables hold cm$^2$ per gram of dust, averaged over
the bin. The file header says `averaged_cross_section` for both.
:::

## Radiation fields

### Analytic fields

`models/tools/radiation_fields.py` defines two analytic fields.

The {cite:t}`Draine1978` field, with $\lambda$ in nm, gives a photon intensity
in photons cm$^{-2}$ s$^{-1}$ nm$^{-1}$ (`models/tools/radiation_fields.py:12`):

$$
I_\lambda = 3.2028\times10^{13}\lambda^{-3}-5.1542\times10^{15}\lambda^{-4}+2.0546\times10^{17}\lambda^{-5}.
$$

The {cite:t}`Mathis1983` field returns $u_E$ in erg cm$^{-3}$ eV$^{-1}$, with
$E$ in eV and $\nu=E/h$ (`models/tools/radiation_fields.py:64-83`):

$$
u_E=\frac{d\nu}{dE}\times
\begin{cases}
3.328\times10^{-9}E^{-4.4172}/\nu, & 11.2<E\le13.6\\
8.463\times10^{-13}E^{-1}/\nu, & 9.26<E\le11.2\\
2.055\times10^{-14}E^{0.6678}/\nu, & 5.04<E\le9.26\\
\dfrac{4\pi}{c}\left[10^{-14}B_\nu(7500\,\mathrm{K})+1.65\times10^{-13}B_\nu(4000\,\mathrm{K})+4\times10^{-13}B_\nu(3000\,\mathrm{K})\right], & E\le5.04
\end{cases}
$$

and $u_E=0$ for $E>13.6$ eV.

### Named fields

The charging and photoelectric-heating code selects a field by name through
`get_radiation_field` (`models/dust_charge/dust_photoelectric_heating.py:1620`).
Every field is then cut to $0.1\le E\le13.6$ eV
(`models/dust_charge/dust_photoelectric_heating.py:1796-1805`).

| name | definition | code |
|---|---|---|
| `Draine` | Draine (1978) formula on the `draine1978.dat` wavelengths; returns `[E, λ, I_E]` | `models/dust_charge/dust_photoelectric_heating.py:1640-1649` |
| `Habing` | the Draine field divided by 1.7, labelled {cite:t}`Habing1968` | `models/dust_charge/dust_photoelectric_heating.py:1650-1659` |
| `Mathis` | `Mathis83_radiation_field` converted to $I_\lambda$ in erg cm$^{-2}$ s$^{-1}$ nm$^{-1}$ sr$^{-1}$, using $I_\nu=c\,u_\nu/4\pi$; returns `[λ, I_λ]` | `models/dust_charge/dust_photoelectric_heating.py:1660-1695` |
| `BB<T>`, `O6V`, `B0V`, `A0`, `HD200775` | stellar spectra diluted by $(r_\star/d)^2$ | `models/dust_charge/dust_photoelectric_heating.py:1701-1753` |
| `BPASS_veryyoung_lowz`, `BPASS_young_midz`, `BPASS_old_highz` | BPASS v2.2.1 SED from `$CALIMA_SED_DIR`, at (10 Myr, $Z=0.0002$), (0.1 Gyr, 0.01) or (1 Gyr, 0.02) | `models/dust_charge/dust_photoelectric_heating.py:1754-1792`, `:48-72` |

`$CALIMA_SED_DIR` must point at a directory; if it is unset, the code raises
`RuntimeError`, and if it is not a directory, `FileNotFoundError`
(`models/dust_charge/dust_photoelectric_heating.py:58-71`). The default
`export_parameters` choose `Mathis` for both the dust and the PAH
photoelectric exports (`models/grain_size_distribution.json:162`,
`models/grain_size_distribution.json:173`).

### What $G_0$ means

$G_0$ is defined differently in different modules:

| where | definition of $G_0$ | code |
|---|---|---|
| dust charging and PE tables | a pure multiplier of the chosen base spectrum, $J_\nu = G_0\,J_\nu^\mathrm{base}$; the base spectrum is not renormalised | `models/dust_charge/dust_photoelectric_heating.py:2338` |
| `compute_G0_from_rad_field` | $\int_{6}^{13.6\,\mathrm{eV}}4\pi I_E\,dE\,/\,1.68\times10^{-3}$ erg cm$^{-2}$ s$^{-1}$ | `models/dust_charge/dust_charging.py:1568`, `:1619`, `:1630` |
| PAH photoelectric heating | $\int_{5.17}^{13.6\,\mathrm{eV}}2\pi I_E\,dE\,/\,1.68\times10^{-6}$ W m$^{-2}$ | `models/PAH_charge/PAH_photoelectric_heating.py:881-882` |
| PAH temperature (`compute_base_g0`) | $c\int_{6}^{13.6\,\mathrm{eV}}u_E\,dE\,/\,1.6\times10^{-3}$ | `models/PAH_photophysics/pah_temperature.py:103-111` |
| sublimation | a multiplier of the full Mathis field | `models/dust_radiation/dust_sublimation.py:362-368` |

So a table's "$G_0=1$" means the base spectrum exactly as tabulated, not a
fixed 6–13.6 eV flux. For reference, `compute_G0_from_rad_field` gives 1.07
for `Mathis` and `compute_base_g0` gives 1.13 for `Mathis83_radiation_field`
(evaluated numerically).

:::{note}
Two inconsistencies in `compute_G0_from_rad_field`
(`models/dust_charge/dust_charging.py:1568-1633`):

- **Solid angle.** The comment at line 1618 says "hemisphere (2π sr)", but the
  code multiplies by $4\pi$ (line 1619).
- **Energy-first fields.** For three-column fields it integrates $I_E$ in file
  order with no sort, and does not multiply by a solid angle
  (lines 1593-1599). `Draine` and `Habing` store energy in descending order,
  so they come out negative: $G_0=-1.59$ and $-0.94$.

The docstring of `get_radiation_field` promises three columns for every field;
`Mathis`, the stellar fields and BPASS return two.
:::

## Emission and temperature

For equilibrium dust temperatures, `dust_emission.py` balances absorption from
a field $F_\lambda$ (erg s$^{-1}$ cm$^{-2}$ cm$^{-1}$) against thermal emission
(`models/dust_radiation/dust_emission.py:646`,
`models/dust_radiation/dust_emission.py:675`):

$$
\int F_\lambda C_\mathrm{abs}\,d\lambda = 4\pi\int C_\mathrm{abs}B_\lambda(T_d)\,d\lambda .
$$

It solves for $T_d$ by root bracketing on $[2.7, 800]$ K
(`models/dust_radiation/dust_emission.py:695-696`). $B_\lambda$ is in
erg s$^{-1}$ cm$^{-2}$ cm$^{-1}$ sr$^{-1}$
(`models/dust_radiation/dust_emission.py:600-612`).

The `dust_band_luminosities` stage reads $\mathcal{C}_\mathrm{abs}$ back from
`averaged_cross_section_<bin_id>.txt`. For each of nine Spitzer/MIPS and
Herschel PACS/SPIRE filters (`models/dust_radiation/export_dust_band_luminosities.py:62-72`)
it computes the filter-weighted mean

$$
L_\mathrm{band}=\frac{\int 4\pi\,\mathcal{C}_\mathrm{abs}B_\lambda(T_d)\,R(\lambda)\,d\lambda}{\int R(\lambda)\,d\lambda}
$$

(`models/dust_radiation/dust_emission.py:2279-2284`). It then divides by
$m_0=\tfrac43\pi s a_0^{3}$
(`models/dust_radiation/export_dust_band_luminosities.py:105`,
`models/dust_radiation/export_dust_band_luminosities.py:134`) and writes
`band_luminosity_<bin_id>.txt` to `optical_properties/`. Each row holds
$\log_{10}T_d$ and nine $\log_{10}L$ values. $T_d$ runs over 500 log-spaced
points in 1–5000 K (`models/dust_radiation/export_dust_band_luminosities.py:81`,
`models/dust_radiation/export_dust_band_luminosities.py:124-141`).

:::{note}
These band luminosities are not in the units their header claims. The header
says erg s$^{-1}$ g$^{-1}$. But $\mathcal{C}_\mathrm{abs}$ is already per gram
of dust, and dividing by $m_0$ normalises by mass a second time. On top of
that, dividing by $\int R\,d\lambda$ gives a mean $L_\lambda$ (per unit
wavelength), not a band-integrated power.
:::

The `dust_sublimation` stage writes `model_data/[<model_name>/]dust_sublimation/sublimation_rate_<bin_id>.dat`.
It follows {cite:t}`GuhathakurtaDraine1989`, and each row holds $T_d$ [K] and
$\log_{10}(|\dot a|/a\,[\mathrm{s^{-1}}])$ on a 200–4000 K grid
(`models/dust_radiation/dust_sublimation.py:1308-1408`). The table depends on
$T_d$ only; the radiation field (Mathis scaled by $G_0$, or O6V) enters only
the diagnostic plots. `dust_ratd.py`, on radiative torque disruption, is not
part of any export stage.
