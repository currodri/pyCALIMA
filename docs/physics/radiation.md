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
(`models/export_all_grain_data.py:655`). It reads the tables
{cite:t}`DraineLee1984` and {cite:t}`LaorDraine1993` computed: `Gra_81` for
graphite and `suvSil_81` for smoothed-UV astronomical silicate
(`models/dust_radiation/dust_oppacity.py:1780-1785`). Each table has 81 radii
from $10^{-3}$ to 10 µm and 241 wavelengths from $10^{-3}$ to $10^{3}$ µm.

The code interpolates $\log Q$ linearly in $\log\lambda$
(`models/dust_radiation/dust_oppacity.py:1696-1699`), then linearly in
$\log a$ (`models/dust_radiation/dust_oppacity.py:1740-1743`). $g$ is
interpolated linearly in $\lambda$ and in $a$. `np.interp` clamps at the ends
of the tabulated range. The default silicate bin `DustBin_03` starts at
$a_\mathrm{min}=5\times10^{-4}$ µm, below the first tabulated radius, so for
its smallest grains $Q$ is held at its $a=10^{-3}$ µm value.

**`mie`** is the default when you run the module directly
(`models/dust_radiation/export_dust_optical_properties.py:239`). It computes
Mie efficiencies with `miepython` from the optical constants in
`draine_lee_1984/eps_suvSil`, `callindex.out_CpaD03_0.01` and
`callindex.out_CpeD03_0.01` (`models/dust_radiation/dust_oppacity.py:1858-1860`).
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
(`models/dust_radiation/dust_oppacity.py:1812`). The weight is
{eq}`eq-lognormal` evaluated for $\rho_d=1$ g cm$^{-3}$
(`models/dust_radiation/dust_oppacity.py:1814-1816`). The result is a mass
opacity in cm$^{2}$ per gram of dust:

$$
\mathcal{C}_x(\lambda)=\int_{a_\mathrm{min}}^{a_\mathrm{max}}
\left.\frac{dn}{da}\right|_{\rho_d=1}\pi a^{2}\,Q_x(a,\lambda)\,da ,
\qquad
Q_\mathrm{rp}=Q_\mathrm{abs}+(1-g)\,Q_\mathrm{sca}.
$$ (eq-bin-opacity)

Here $x$ is `abs`, `sca` or `rp` (radiation pressure). This is implemented at
`models/dust_radiation/dust_oppacity.py:1746-1754`, and in the Mie path at
`models/dust_radiation/dust_oppacity.py:1929-1937`.

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
`get_radiation_field` (`models/dust_charge/dust_photoelectric_heating.py:1575`).
Every field is then cut to $0.1\le E\le13.6$ eV
(`models/dust_charge/dust_photoelectric_heating.py:1751-1760`).

| name | definition | code |
|---|---|---|
| `Draine` | Draine (1978) formula on the `draine1978.dat` wavelengths; returns `[E, λ, I_E]` | `models/dust_charge/dust_photoelectric_heating.py:1595-1604` |
| `Habing` | the Draine field divided by 1.7, labelled {cite:t}`Habing1968` | `models/dust_charge/dust_photoelectric_heating.py:1605-1614` |
| `Mathis` | `Mathis83_radiation_field` converted to $I_\lambda$ in erg cm$^{-2}$ s$^{-1}$ nm$^{-1}$ sr$^{-1}$, using $I_\nu=c\,u_\nu/4\pi$; returns `[λ, I_λ]` | `models/dust_charge/dust_photoelectric_heating.py:1615-1650` |
| `BB<T>`, `O6V`, `B0V`, `A0`, `HD200775` | stellar spectra diluted by $(r_\star/d)^2$ | `models/dust_charge/dust_photoelectric_heating.py:1656-1708` |
| `BPASS_veryyoung_lowz`, `BPASS_young_midz`, `BPASS_old_highz` | BPASS v2.2.1 SED from `$CALIMA_SED_DIR`, at (10 Myr, $Z=0.0002$), (0.1 Gyr, 0.01) or (1 Gyr, 0.02) | `models/dust_charge/dust_photoelectric_heating.py:1709-1747`, `:47-71` |

`$CALIMA_SED_DIR` must point at a directory; if it is unset, the code raises
`RuntimeError`, and if it is not a directory, `FileNotFoundError`
(`models/dust_charge/dust_photoelectric_heating.py:57-70`). The default
`export_parameters` choose `Mathis` for the PAH photoelectric export
(`models/grain_size_distribution.json:154`).

### What $G_0$ means

$G_0$ is defined differently in different modules:

| where | definition of $G_0$ | code |
|---|---|---|
| full-spectrum dust charging solver (`dust_charging`) | a pure multiplier of the chosen base spectrum, $J_\nu = G_0\,J_\nu^\mathrm{base}$; the base spectrum is not renormalised | `models/dust_charge/dust_photoelectric_heating.py:2294` |
| uniform-ISRF charging tables | Habing: the Mathis field scaled to $1.6\times10^{-3}$ erg cm$^{-2}$ s$^{-1}$ in 6–13.6 eV, times $G_0$ | `models/dust_charge/rt_group_charging.py:221-226` |
| `compute_G0_from_rad_field` | $\int_{6}^{13.6\,\mathrm{eV}}4\pi I_E\,dE\,/\,1.68\times10^{-3}$ erg cm$^{-2}$ s$^{-1}$ | `models/dust_charge/dust_charging.py:1542`, `:1593`, `:1604` |
| PAH photoelectric heating | $\int_{5.17}^{13.6\,\mathrm{eV}}2\pi I_E\,dE\,/\,1.68\times10^{-6}$ W m$^{-2}$ | `models/PAH_charge/PAH_photoelectric_heating.py:881-882` |
| PAH temperature (`compute_base_g0`) | $c\int_{6}^{13.6\,\mathrm{eV}}u_E\,dE\,/\,1.6\times10^{-3}$ | `models/PAH_photophysics/pah_temperature.py:103-111` |
| sublimation | a multiplier of the full Mathis field | `models/dust_radiation/dust_sublimation.py:573-579` |

So, except in the uniform-ISRF charging tables, a table's "$G_0=1$" means the
base spectrum exactly as tabulated, not a fixed 6–13.6 eV flux. For reference, `compute_G0_from_rad_field` gives 1.07
for `Mathis` and `compute_base_g0` gives 1.13 for `Mathis83_radiation_field`
(evaluated numerically).

:::{note}
Two inconsistencies in `compute_G0_from_rad_field`
(`models/dust_charge/dust_charging.py:1542-1607`):

- **Solid angle.** The comment at line 1592 says "hemisphere (2π sr)", but the
  code multiplies by $4\pi$ (line 1593).
- **Energy-first fields.** For three-column fields it integrates $I_E$ in file
  order with no sort, and does not multiply by a solid angle
  (lines 1567-1573). `Draine` and `Habing` store energy in descending order,
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

(`models/dust_radiation/dust_emission.py:2283-2288`). It then divides by
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
(`models/dust_radiation/dust_sublimation.py:1519-1408`). The table depends on
$T_d$ only; the radiation field (Mathis scaled by $G_0$, or O6V) enters only
the diagnostic plots. `dust_ratd.py`, on radiative torque disruption, is not
part of any export stage.

## Hot dust around an AGN

`models/dust_radiation/agn_hot_dust.py` holds the pieces for optically thin
grain heating by a quasar:

- `skirtor_sed(lam_cm, L_bol)` (`agn_hot_dust.py:57`) is the SKIRTOR disk SED
  of {cite:t}`Stalevski2012`: $\lambda L_\lambda \propto \lambda^{1.2}$
  (1–10 nm), flat (10–100 nm), $\lambda^{-0.5}$ (0.1–5 µm), $\lambda^{-3}$
  beyond, normalised to $L_\mathrm{bol}$ over 1 nm–1 mm.
- `disk_anisotropy(cos_theta, law)` (`:69`) is the angular distribution
  $f(\theta)$ of the disk emission, with mean 1 over the sphere: `isotropic`,
  `cos` ($2|\cos\theta|$) or `skirtor`
  ($\tfrac{6}{7}|\cos\theta|(1+2|\cos\theta|)$).
- `HotGrain(material, a_um)` (`:90`) holds the Draine & Lee / Laor & Draine
  absorption cross section of one grain and tabulates its emitted power
  $4\pi\int C_\mathrm{abs}B_\lambda(T)\,d\lambda$ on 3–5000 K, so that the
  equilibrium temperature is a table look-up. It avoids the 800 K bracket of
  the `dust_emission.compute_equilibrium_temperature*` helpers and agrees
  with `dust_sublimation.compute_bin_temperature` to 0.2%.
  `luminosity_density` is the grain's emission $4\pi C_\mathrm{abs}B_\lambda(T)$.
- `AGNIllumination(L_bol, law)` (`:145`) gives the temperature of a grain at
  $(r,\theta)$ and its inverse, `radius_at_temperature`. With no attenuation
  the absorbed power scales as $f(\theta)/r^2$, so
  $R(T)\propto[f(\theta)L_\mathrm{bol}]^{1/2}$.

Sublimation kinetics for 28 materials (graphite, amorphous carbon,
astronomical silicate, forsterite, enstatite, fayalite, ferrosilite, silica,
SiO, Fe, Si, Cr, FeS, Al$_2$O$_3$, spinel, perovskite, TiO$_2$, MgO, FeO,
Fe$_3$O$_4$, Fe$_2$O$_3$, SiC, TiC, NaCl, KCl, MnS, Na$_2$S and water ice) are
in `models/dust_radiation/sublimation_kinetics.py`, read from
`data/external_data/sublimation_kinetics/sublimation_models.toml`. That file
is generated (`make_sublimation_models.py`) from a literature compilation in
the same directory (`compilation.csv`, 107 rows, and
`COMPILATION_NOTES.md`): vapour-pressure fits, evaporation coefficients
(temperature dependent where measured, e.g. Fe), vapour channels (C$_1$,
C$_2$, C$_3$ for graphite), surface energies and congruency flags, each with
its source. `get_sublimation_model(material, dataset)` returns the bulk flux
$J(T) = \sum_c n_c\alpha_c P_c/\sqrt{2\pi m_c kT}$ (or the GD89 form), the
condensation coefficient and the small-grain term
$\theta = (32\pi/3)^{1/3}(m/\rho)^{2/3}\sigma/k$. The `fiducial` dataset is
the default; `GD89` is kept for graphite and silicate. Incongruent materials
(FeS, enstatite, SiC, Fe oxides) carry a formal congruent row and a flag.

The sublimation temperature of a grain follows from these kinetics with
recondensation (`dust_sublimation.py:331`–`380`).
`saturated_key_density` is the key-species density (C for graphite, Si for
silicate) at which the kinetic condensation flux cancels evaporation, the same
flux balance as the `condensation_rate` kernel
({doc}`dust-growth-destruction`). `net_sublimation_timescale` divides the
vacuum lifetime by the undersaturation $1-n_X/n_\mathrm{sat}$, and
`sublimation_temperature(material, a_cm, t_exp_s, n_key, T_gas)` is the
temperature at which the grain survives an exposure $t_\mathrm{exp}$.
`write_sublimation_rate_table` (`dust_sublimation.py:1465`) writes the
solver table of a single grain, for runs that do not go through a grain-size
configuration.

`models/tools/spherex.py` models the 102 SPHEREx channels (six bands of 17
over 0.75–5.0 µm, $R$ = 35–41 below 3.82 µm and 110–130 above) as log-spaced
top-hats, and `spherex_photometry` returns the photon-weighted mean
$F_\lambda$ per channel. It approximates the real linear-variable-filter
responses, which should replace it once released.
