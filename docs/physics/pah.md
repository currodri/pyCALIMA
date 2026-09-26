(physics-pah)=
# PAH life cycle

PAH bins are the first `npah` entries of the solver's dust array, each holding
a mass density $\rho_{\rm PAH}$ [g cm$^{-3}$] of molecules of a fixed
representative mass $m_{\rm PAH} = N_{\rm C}\times 12.011$ amu
(`solvers/dust_init.py:490`) and radius
$a_{\rm PAH} = (3m_{\rm PAH}/4\pi s_{\rm PAH})^{1/3}$, with a default material
density $s_{\rm PAH} = 2.24$ g cm$^{-3}$ (`solvers/dust_init.py:489`,
`solvers/dust_init.py:493`). A bin may be flagged `is_cluster`, which marks it
as a PAH cluster rather than a free molecule. Six processes change
$\rho_{\rm PAH}$:

| Process | `physics` flag | Kernel | Mass goes |
|---|---|---|---|
| Accretion of gas-phase C | `pah_accretion` | `pah_accretion_rate` | gas C to PAH |
| UV photolysis | `pah_photolysis` | `pah_photolysis_rate` | PAH to gas C |
| Sputtering by ions and electrons | `pah_sputtering` | `pah_sputtering_rate` | PAH to gas C |
| Coalescence | `pah_coalescence` | `totton2012_pah_coalescence_rate` or `tielens2021_pah_coalescence_rate` | bin $p$ to bin $p+1$ |
| Cluster evaporation | `pah_cluster_evaporation` | `pah_cluster_evaporation_rate` | cluster bin $p$ to bin $p-1$ |
| Freezing onto grains | `pah_freezing` | `pah_freezing_rate` | PAH bin to dust bin |

All six flags default to `false` (`solvers/dust_init.py:588`–`593`), and
`build_process_list` registers the kernels (`solvers/rhs.py:167`–`216`).
Coalescence and cluster evaporation also need `npah > 1`, and freezing needs at
least one dust bin. Every kernel skips a bin whose density is at or below its
floor `smallr_pah`.

## Photophysics and photolysis

### What the solver does

`pah_photolysis_rate` reads a per-bin table of $\log_{10}k_{\rm diss}$
[s$^{-1}$] on a $(\log_{10}G_0, \log_{10}n_{\rm H})$ grid. Each event removes a
C$_2$ unit:

$$
\dot\rho_{\rm PAH} = -\dot\rho_{\rm C,gas}
 = -k_{\rm diss}(G_0,n_{\rm H})\,\frac{\rho_{\rm PAH}}{m_{\rm PAH}}\,2m_{\rm C}
$$ (eq-pah-photolysis)

(`solvers/dust_rates.py:961`). The kernel returns immediately if
$G_0 \le 0$, and it skips cluster bins and bins with no table
(`solvers/dust_rates.py:936`, `solvers/dust_rates.py:946`). $G_0$ comes from
`environment.radiation_field_G0` (`solvers/dust_init.py:376`). The table is
`model_data/PAH_dissociation_data/dissociation_<bin_id>.dat`
(`solvers/dust_init.py:244`, `solvers/dust_init.py:361`), and it is
interpolated bilinearly in log space (`solvers/table_io.py:208`).

:::{note}
The docstring of `build_pah_photolysis_interpolator` says out-of-range queries
are clamped to the table edge. The code does something else: any
$(G_0, n_{\rm H})$ outside the grid returns the *global minimum* of the table
(`fill_value = log_rate.min()`, `solvers/table_io.py:207`).
:::

:::{note}
The `models.photolysis_model` key (default `"RM2026"`) is parsed and stored
(`solvers/dust_init.py:600`), but no Python code reads it. There is only one
photolysis kernel. The RAMSES Fortran branches on the key, but both branches
call the same routine. `pah_sputtering_model` and `cluster_evaporation_model`
are in the same position.
:::

### How the table is built

`export_pah_dissociation_tables.py` calls
`PAH_photophysics.plot_acetylene_dissociation_rate` once per PAH bin in the
grain-size configuration (`models/PAH_photophysics/export_pah_dissociation_tables.py:96`).
That function writes a $100\times100$ grid spanning $G_0 = 10^{-2}$–$10^{6}$
and $n_{\rm H} = 10^{-2}$–$10^{6}$ cm$^{-3}$
(`models/PAH_photophysics/export_pah_dissociation_tables.py:19`–`24`,
`models/PAH_photophysics/PAH_photophysics.py:294`–`295`). The bin's
$N_{\rm C}$ comes from the lognormal $a_0$ of its size distribution, via
$N_{\rm C} = \lfloor 418\,(a/10)^3 \rfloor$ with $a$ in Å, from {cite:t}`Draine2021`
(`models/tools/utils.py:30`). This value is independent of the `nc` set in the
solver configuration.

The only channel is acetylene loss from a dehydrogenated PAH. It uses the
{cite:t}`Murga2020` parameters $E_0 = 4.6$ eV and
$\Delta S = 10$ cal K$^{-1}$ mol$^{-1}$
(`models/PAH_photophysics/PAH_photophysics.py:122`). For an absorbed photon of
energy $E$ [eV], the microcanonical effective temperature
{cite:p}`Tielens2005` is

$$
T_{\rm eff}(E) = 2000\,\mathrm{K}\,\left(\frac{E}{N_{\rm C}}\right)^{0.4}
\left(1 - 0.2\,\frac{E_0}{E}\right)
$$ (eq-pah-teff)

(`models/PAH_photophysics/PAH_photophysics.py:187`). The rate at that
temperature is

$$
k(T) = \frac{k_{\rm B}T}{h}\,e^{1+\Delta S/R}\,
\exp\!\left(-\frac{E_0}{k_{\rm B}T}\right),
$$ (eq-pah-gibbs)

with $R = 1.98720$ cal K$^{-1}$ mol$^{-1}$ and $k_{\rm B} = 8.617\times10^{-5}$
eV K$^{-1}$ in the exponent
(`models/PAH_photophysics/PAH_photophysics.py:81`, `194`, `197`).
The code follows {cite:t}`Micelotta2010b` and evaluates $k$ at
$T_{\rm av} = [T_{\rm eff}(E)\,T_{\rm eff}(E - n_{\max}\Delta\varepsilon)]^{1/2}$,
where $n_{\max} = \lfloor N_{\rm C}/5\rfloor$ and $\Delta\varepsilon = 0.16$ eV
(`models/PAH_photophysics/PAH_photophysics.py:255`, `272`–`275`, `85`). The
dissociation probability competes $k$ against IR cooling at
$k_{\rm IR} = 100$ s$^{-1}$ (`models/PAH_photophysics/PAH_photophysics.py:83`):

$$
P(E) = \frac{k(T_{\rm av})}{k(T_{\rm av}) + k_{\rm IR}/(n_{\max}+1)}
$$ (eq-pah-pdiss)

(`models/PAH_photophysics/PAH_photophysics.py:208`). This probability is
integrated over the far-UV spectrum:

$$
R_1 = \int_{E_0}^{13.6\,\rm eV} \frac{\sigma_{\rm abs}^{+}(E)\,I(E)\,P(E)}{E}\,\mathrm{d}E
$$ (eq-pah-r1)

(`models/PAH_photophysics/PAH_photophysics.py:266`, `280`, `282`). Here
$\sigma^{+}_{\rm abs}$ [m$^2$] is the {cite:t}`LiDraine2001` *cation*
cross-section, number-averaged over the bin's lognormal distribution
(`models/PAH_photophysics/PAH_photophysics.py:245`, `304`). $I(E)$ is the
{cite:t}`Draine1978` field divided by 1.7 and converted to W m$^{-2}$ eV$^{-1}$
(`models/PAH_photophysics/PAH_photophysics.py:260`–`263`), and the factor
$6.2415\times10^{18}$ eV J$^{-1}$ turns the integrand into photons, so $R_1$
is in s$^{-1}$.

$R_1$ is then scaled linearly with $G_0$ and multiplied by a smooth estimate of
the fraction of fully dehydrogenated molecules,
$k_{\rm diss} = R_1\,G_0\,f(G_0,n_{\rm H})$
(`models/PAH_photophysics/PAH_photophysics.py:299`, `307`). The function $f$
uses a straight line $\log n_{\rm H} = a\log G_0 + b$, fitted to the 50%
hydrogenation contour for C$_{54}$ from {cite:t}`Montillaud2013`
(`models/PAH_photophysics/PAH_photophysics.py:290`–`291`). With $d$ the
distance of a point from that line in the log plane:

$$
f = \tfrac12 \mp \frac{1/2}{1 + 0.1\,d^{-2}}, \qquad
d = \frac{|a\log G_0 + b - \log n_{\rm H}|}{\sqrt{1+a^2}}
$$ (eq-pah-fdehydro)

The minus sign applies on the dense side of the line
($\log n_{\rm H} \ge a\log G_0 + b$)
(`models/PAH_photophysics/PAH_photophysics.py:600`–`611`).

:::{note}
One constant is defined twice, and the second definition wins:
$\Delta\varepsilon$ is set to 0.145 eV at
`models/PAH_photophysics/PAH_photophysics.py:78` and to 0.16 eV at line `85`.
Nothing in eq. {eq}`eq-pah-teff` guards against
$E - n_{\max}\Delta\varepsilon \le 0$. For $N_{\rm C} = 418$ ($a_0 = 10$ Å),
$n_{\max}\Delta\varepsilon = 13.3$ eV, so the integrand of eq. {eq}`eq-pah-r1` is NaN at every $E$, and the
exported `dissociation_PAHBin_02.dat` in the default configuration contains
only `nan`. The solver never interpolates that table when bin 2 is a cluster
bin, but it would if the bin were a free PAH.
:::

### The RRKM research modules

`pah_mol_data.py`, `pah_temperature.py`, `pah_dissociation.py` and
`pah_h_state.py` implement a more detailed treatment of H- and H$_2$-loss. None
of them is used by the exporter or the solver. They compute:

- the canonical temperature $T_m$ from $U_{\rm QHO}(T_m) = E$ over the PAHdb
  vibrational modes (`models/PAH_photophysics/pah_mol_data.py:84`–`90`);
- the rate
  $K = e\,(k_{\rm B}T_e/h)\,e^{\Delta S/R}\,e^{-E_{\rm act}/k_{\rm B}T_e}$,
  with $T_e = T_m(1-0.2E_{\rm act}/E)$ and $\Delta S$ in J K$^{-1}$ mol$^{-1}$
  (`models/PAH_photophysics/pah_mol_data.py:212`–`219`);
- a vibrational temperature distribution $f(T)$ in the style of
  {cite:t}`GuhathakurtaDraine1989`
  (`models/PAH_photophysics/pah_temperature.py:118`);
- the cascade-corrected branching $Y_{\rm H} = \sum f(T)\,\Delta T\,
  k_{\rm H}W_{\rm down}/(k_{\rm H}+k_{\rm H_2}+W_{\rm down})$
  (`models/PAH_photophysics/pah_dissociation.py:349`–`355`).

The activation parameters are those of {cite:t}`Andrews2016`, for example
$E_{\rm act} = 4.60$ eV and $\Delta S = 44.8$ J K$^{-1}$ mol$^{-1}$ for H-loss
from even-$N_{\rm H}$ molecules, and 3.52 eV and $-53.1$ J K$^{-1}$ mol$^{-1}$
for H$_2$-loss (`models/PAH_photophysics/pah_h_state.py:30`–`45`).

### The NASA Ames PAHdb is optional

The PAHdb theoretical library (v4.00 XML, about 500 MB) cannot be
redistributed, and downloading it requires a free registration. Only
`models/PAH_photophysics/pah_db_lookup.py` reads it. It uses the library to
build a JSON species catalog and to extract missing vibrational-mode files
named `C<Nc>H<Nh>_<Z>.dat` into `model_data/PAH_states/`
(`models/PAH_photophysics/pah_db_lookup.py:54`, `371`–`373`). To use it, set
`$CALIMA_PAHDB_DIR` or run
`calima-fetch-data import pahdb-theoretical-v4-00 <file>`. The solvers, the
photolysis tables above, and existing `PAH_states` mode files all work without
it. `pah_mol_data.py` imports `amespahdbpythonsuite` only for a type
annotation (`models/PAH_photophysics/pah_mol_data.py:20`–`27`).

## Sputtering by gas collisions

For each non-cluster bin, the rate sums over electrons and the elements H, He,
C and O (`solvers/dust_init.py:151`–`157`):

$$
\mathcal{R} = n_e J_e(T) + \sum_i n_i J_i(T), \qquad
\dot\rho_{\rm PAH} = -\dot\rho_{\rm C,gas} = -\mathcal{R}\,\frac{\rho_{\rm PAH}}{m_{\rm PAH}}\,m_{\rm C}
$$ (eq-pah-sput)

(`solvers/dust_rates.py:1024`, `solvers/dust_rates.py:1029`). Here
$n_i = \rho_i/m_i$ is the *total* number density of element $i$, not its ion
density (`solvers/dust_rates.py:1016`), and $n_e$ is
`electron_number_density_cm3`, which defaults to $10^{-4}n_{\rm H}$
(`solvers/dust_init.py:375`). The tables
`pah_sputtering_data/sputtering_<bin_id>_Z_<Z>` hold $\log_{10}J$
[cm$^3$ s$^{-1}$] on 100 points spanning $T = 10^3$–$10^9$ K
(`models/PAH_gas_collisions/export_pah_sputtering_rates_bins.py:21`–`27`).
Queries outside that range take the edge value (`solvers/table_io.py:281`–`285`).

$J$ already counts carbon atoms lost: $J_e^{\rm tab} = 2J_e$ and
$J_i^{\rm tab} = 2J_i^{\rm el} + \tfrac12 N_{\rm C}J_i^{\rm nuc}$
(`models/PAH_gas_collisions/PAH_sputtering.py:1467`, `1509`, `1534`). The two
pieces are:

- **Electronic excitation** (the factor 2 is one C$_2$H$_2$ per event),
  following {cite:t}`Micelotta2010b`. It is a Maxwellian integral over
  $v\int_0^{\pi/2}\sigma(\theta)P\sin\theta\,\mathrm{d}\theta$ for a thick
  disc with $\sigma(\theta) = \pi R^2\cos\theta + 2Rd\sin\theta$ and
  $d = 4.31\times10^{-8}$ cm
  (`models/PAH_gas_collisions/PAH_sputtering.py:45`, `148`, `640`–`643`).
  The energy deposited is set by {cite:t}`Joy1995` electron stopping or by
  the {cite:t}`PuskaNieminen1983` friction for ions. $P$ is
  eq. {eq}`eq-pah-pdiss` with $E_0 = 4.6$ eV and $\Delta S = 10$ cal K$^{-1}$
  mol$^{-1}$ (`models/PAH_gas_collisions/PAH_sputtering.py:41`, `383`–`388`).
  It is set to zero when $E - n_{\max}\Delta\varepsilon \le 0.2E_0$
  (`models/PAH_gas_collisions/PAH_sputtering.py:633`).
- **Nuclear knock-on.** A ZBL energy-transfer cross-section with the
  {cite:t}`Ziegler1985` $m(\epsilon)$ and a threshold of 7.5 eV
  (`models/PAH_gas_collisions/PAH_sputtering.py:1355`, `1619`–`1652`).

All tables are computed for a neutral PAH with no Coulomb shift ($\phi = 0$)
(`models/PAH_gas_collisions/PAH_sputtering.py:1425`).

## Coalescence

Coalescence moves mass from each non-cluster bin $p < n_{\rm pah}-1$ into bin
$p+1$ at the rate
$\dot\rho_p = -\dot\rho_{p+1} = -k_1\rho_p$
(`solvers/dust_rates.py:1086`–`1090`). `coalescence_model` selects the rate
$k_1$ [s$^{-1}$] (default `Totton2012`, `solvers/dust_init.py:598`;
`solvers/rhs.py:186`–`201`).

**`Totton2012`** {cite:p}`Totton2012`:

$$
k_1 = 4\pi a_{\rm PAH}^2\,\sqrt{\frac{8k_{\rm B}T}{m_{\rm PAH}/2}}\;
\frac{1}{1 + 9.92807\times10^{-7}(\log_{10}T)^{13.7934}}\;
\frac{\rho_{\rm PAH}}{m_{\rm PAH}}
$$ (eq-pah-totton)

(`solvers/dust_rates.py:1064`, `1078`–`1082`).

:::{note}
The thermal speed in eq. {eq}`eq-pah-totton` is $\sqrt{8k_{\rm B}T/\mu}$, with
no $1/\pi$ inside the root. It is therefore $\sqrt{\pi}$ times the Maxwellian
mean relative speed. The RAMSES Fortran and
`models/PAH_collisions/PAH_coalescence.py` use the same expression.
:::

**`Tielens2021`**, in the neutral approximation {cite:p}`Tielens2021`:
$k_1 = 4\times10^{-11}\,(T/10\,{\rm K})^{1/2}(N_{\rm C}/50)^{1/2}
\;\rho_{\rm PAH}/m_{\rm PAH}$, with the coefficient in cm$^3$ s$^{-1}$
(`solvers/dust_rates.py:1130`–`1133`). The RAMSES version mixes in an ionized
(Langevin) term weighted by the PAH cation fraction. The Python solver does
not track PAH charge, so it omits that term.

## Cluster evaporation

Only bins with `is_cluster = true` evaporate, and each loses monomers to bin
$p-1$ {cite:p}`Montillaud2014`:

$$
k_{\rm ev} = \min\!\left[\frac{G_0}{0.19306},\;
10^{\,3.1692061\log_{10}G_0 - 13.5642486}\right]\ {\rm yr^{-1}},\qquad
\dot\rho_p = -\dot\rho_{p-1} = -\frac{k_{\rm ev}}{3.1536\times10^{7}\,{\rm s\,yr^{-1}}}\,\frac{\rho_p}{m_p}\,m_{p-1}
$$ (eq-pah-evap)

(`solvers/dust_rates.py:1171`–`1173`, `1189`). The first term is the
single-cluster timescale $0.19306/G_0$ yr
(`models/PAH_photophysics/PAH_photophysics.py:390`). The second is a log-log
line through digitised (C$_{54}$H$_{18}$)$_n$ curves
(`models/PAH_photophysics/PAH_photophysics.py:394`–`403`).

## Freezing onto grains

A PAH bin freezes onto every dust bin, or onto the range set by
`dust_index_interact` and `nd_bins_interact`, regardless of composition
(`solvers/dust_rates.py:1237`–`1242`). The rate is

$$
k_1 = \sqrt{\frac{8}{3\pi}}\,\pi(a_{\rm PAH}+a_d)^2\,v_{\rm rel}\,
\frac{\rho_d}{m_d}\,p_{\rm stick},\qquad
p_{\rm stick} = \frac{1}{1+\exp[4(v_{\rm rel}/v_{\rm coag}-1)]}
$$ (eq-pah-freeze)

(`solvers/dust_rates.py:1256`–`1266`, `solvers/grain_dynamics.py:119`–`124`),
and $\dot\rho_{\rm PAH} = -\dot\rho_d = -k_1\rho_{\rm PAH}$
(`solvers/dust_rates.py:1270`–`1274`). $v_{\rm rel}$ comes from
`grain_relative_velocity` with `dust_velocity_model` (default `Ormel2007`,
`solvers/dust_init.py:596`). $v_{\rm coag}$ is the dust bin's
`vthresh_coag_cm_s`, which defaults to $10^4$ cm s$^{-1}$
(`solvers/dust_init.py:465`). No Coulomb focusing is applied.

## Accretion of gas-phase carbon

PAHs grow by sticking gas-phase C atoms, with the {cite:t}`LeBourlot2012`
sticking factor:

$$
\dot\rho_{\rm PAH} = -\dot\rho_{\rm C,gas}
= \frac{\pi a_{\rm PAH}^2\sqrt{8k_{\rm B}T/\pi}}{m_{\rm PAH}}\,
\frac{1}{1+10^{-4}T^{1.5}}\,\frac{\rho_{\rm C}}{\sqrt{m_{\rm C}}}\,\rho_{\rm PAH}
$$ (eq-pah-accr)

(`solvers/dust_rates.py:834`, `852`–`855`). This kernel also acts on cluster
bins.

## Role of PAH charge

No PAH kernel in the Python solver uses the PAH charge. In each process,
charge is fixed or ignored:

- photolysis uses the cation cross-section throughout;
- sputtering tables are computed at $\phi = 0$;
- `Tielens2021` coalescence uses its neutral branch;
- freezing has no Coulomb factor.

`models/PAH_charge/` computes photoelectric heating tables. None of the PAH
mass-exchange rates on this page read them.
