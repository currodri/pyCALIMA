(physics-dust-growth-destruction)=
# Grain growth and destruction

This page describes the three processes that exchange mass directly between
the gas-phase metals and the dust bins: growth by accretion, thermal
sputtering and thermal sublimation. Each is a rate kernel in
`solvers/dust_rates.py` that adds to the derivatives of the dust mass density
$\rho_d$ of every bin and of the gas-phase element mass densities $\rho_e$
(both in g cm$^{-3}$). All three are first order in $\rho_d$: the kernel
computes a fractional rate $k$ [s$^{-1}$] per bin and then applies

$$
\frac{d\rho_d}{dt} = \pm k\,\rho_d, \qquad
\frac{d\rho_e}{dt} = \mp k\,\rho_d\,f_e ,
$$ (eq-dgd-bookkeeping)

where $f_e$ is the mass fraction of element $e$ in the grain material
(`solvers/dust_rates.py:321`–`325` for accretion, `:396`–`400` for the
tabulated sputtering, `:592`–`594` for sublimation). Mass is therefore
conserved element by element.

| Process | `physics` flag | `models` key | Kernel |
|---|---|---|---|
| Accretion | `dust_accretion` | none | `accretion_rate` |
| Thermal sputtering | `dust_sputtering` | `dust_sputtering_model` | `thermal_sputtering_rate`, `thermal_sputtering_rate_nozawa`, `thermal_sputtering_rate_nozawa_ramses` |
| Thermal sublimation | `dust_sublimation` | none | `sublimation_rate` |
| Recondensation | `dust_condensation` | `condensation_coefficient`, `condensation_balance`, `condensation_ion_fraction` | `condensation_rate` |
| Nucleation | `dust_nucleation` | `nucleation` list (top level) | `nucleation_rate` |

All five flags default to `false` (`solvers/dust_init.py:663`–`666`). The
processes are appended to the process list only if the flag is set and there
is at least one dust bin (`solvers/rhs.py:87`–`114`).

## Grain growth by accretion

### Rate

For each bin the accretion kernel (`accretion_rate`, following
{cite:t}`Dubois2024`) computes

$$
k_\mathrm{acc} = k_0\;\Pi(T)\;
\min_e\!\left[\frac{\rho_e}{f_e\sqrt{m_e}}\right]\;E_C\;C_\mathrm{turb}
\quad [\mathrm{s^{-1}}]
$$ (eq-dgd-kacc)

(`solvers/dust_rates.py:315`), with $m_e$ the atomic mass of element $e$ [g].
The rate constant is precomputed per bin from the grain radius $a$ [cm],
the bulk density $\rho_\mathrm{gr}$ [g cm$^{-3}$] (`grain_density_gcm3`) and
the configured `sticking_coefficient` $S$ (default 1):

$$
k_0 = \frac{\pi a^2\, S\,\sqrt{8 k_B/\pi}}{m_\mathrm{gr}},
\qquad m_\mathrm{gr} = \tfrac{4}{3}\pi a^3 \rho_\mathrm{gr}
$$ (eq-dgd-k0)

(`_k0_accretion`, `solvers/dust_init.py:175`–`177`), in
cm$^3$ g$^{-1/2}$ K$^{-1/2}$ s$^{-1}$. Combined with a factor $\sqrt{T}$ in
$\Pi$, equation {eq}`eq-dgd-kacc` is the geometric collision rate
$\pi a^2 n_e \langle v_e\rangle$ per unit grain mass, with
$\langle v_e\rangle = \sqrt{8 k_B T/\pi m_e}$, evaluated for the limiting
element.

**Limiting element.** The minimum in {eq}`eq-dgd-kacc` runs over the
elements of the bin; it picks the element whose supply, relative to its
share of the grain mass, is smallest. If any constituent has
$\rho_e \le 0$ or $f_e \le 0$, the bin does not grow
(`solvers/dust_rates.py:286`–`299`). The same $k_\mathrm{acc}$ then removes
every constituent in stoichiometric proportion, through
{eq}`eq-dgd-bookkeeping`.

**Global cut-offs.** Accretion is switched off entirely for
$T > 1.8\times10^{5}$ K, where the gas is taken to be fully ionised
(`solvers/dust_rates.py:209`–`211`). A bin is skipped when
$n_\mathrm{H} >$ `nhmax_acc` (`solvers/dust_rates.py:269`). This per-bin
ceiling is read from the bin's `nhmax_acc` key, defaulting to
$10^4$ cm$^{-3}$ (`solvers/dust_init.py:459`). The code comment gives
$10^3$ cm$^{-3}$ for carbon grains and $10^4$ cm$^{-3}$ for silicates in
{cite:t}`Dubois2024` (`solvers/dust_rates.py:268`). The lower ceiling for
carbon is how the docstring's "C is fully molecular (CO) for
$n_\mathrm{H} > 10^3$ cm$^{-3}$" is implemented; no separate CO check exists.

### Regime selection

A cell is treated as **sub-grid** when all of the following hold
(`solvers/dust_rates.py:219`–`226`):

- $n_\mathrm{H} \ge 0.1$ cm$^{-3}$;
- $T \le 10^4$ K;
- the Jeans length is not resolved, i.e. not
  $\lambda_J > 4\,\Delta x$, with

$$
\lambda_J = \sqrt{\frac{\pi k_B T}{G\, m_\mathrm{H}^2\, n_\mathrm{H}}}
$$ (eq-dgd-jeans)

(`solvers/dust_rates.py:221`), using $m_\mathrm{H} = 1.6726219\times10^{-24}$ g
and $G = 6.674\times10^{-8}$ cm$^3$ g$^{-1}$ s$^{-2}$
(`solvers/dust_rates.py:202`–`203`). The cell size $\Delta x$ comes from
`turbulence.local_dx_pc`, converted with 1 pc $= 3.085677581\times10^{18}$ cm
(`solvers/dust_init.py:566`–`568`). Every other cell is **resolved**.

:::{note}
`local_dx_pc` defaults to 0. With $\Delta x = 0$ the Jeans test is skipped
and treated as unresolved (`solvers/dust_rates.py:223`–`224`). So, unless a
cell size is given, every cell with $n_\mathrm{H}\ge0.1$ cm$^{-3}$ and
$T\le10^4$ K uses the sub-grid regime.
:::

The two regimes differ in the prefactor $\Pi$ [K$^{1/2}$] and in the
clumping boost:

| Regime | $\Pi(T)$ | $C_\mathrm{turb}$ | Code |
|---|---|---|---|
| Resolved | $\sqrt{T}\,/\,(1 + 10^{-4}\,T^{1.5})$, the sticking of {cite:t}`LeBourlot2012` | 1 | `solvers/dust_rates.py:251`, `:283` |
| Sub-grid | $\alpha_\mathrm{eff}\sqrt{T_\mathrm{acc}}$, with $\alpha_\mathrm{eff}=1/3$, $T_\mathrm{acc}=100$ K | Eq. {eq}`eq-dgd-boost` | `solvers/dust_rates.py:232`–`234` |

In both regimes $\Pi$ multiplies the configured $S$ already folded into $k_0$.

:::{note}
The `_k0_accretion` docstring says that "no further T-dependent suppression
is applied in accretion_rate()" (`solvers/dust_init.py:172`–`173`). That is
not what the code does: in resolved cells `accretion_rate` applies the
factor $1/(1+10^{-4}T^{1.5})$ (`solvers/dust_rates.py:251`).
:::

### Turbulent clumping boost

In sub-grid cells the rate is multiplied by the integral of a log-normal
density PDF up to the per-bin ceiling $n_\mathrm{max} =$ `nhmax_acc`. The
Mach number uses the *local* temperature (only the sticking prefactor is
fixed at $T_\mathrm{acc}$):

$$
c_s = \sqrt{\tfrac{5}{3}\,k_B T / m_\mathrm{H}},\qquad
\mathcal{M} = \max\!\left(10^{-5},\; \sigma/c_s\right),\qquad
w = \ln\!\left[1 + (0.4\,\mathcal{M})^2\right]
$$ (eq-dgd-mach)

(`solvers/dust_rates.py:236`, `:244`–`245`). Here $\sigma$ [cm s$^{-1}$] is
`turbulence.local_sigma_km_s` $\times10^5$ (`solvers/dust_init.py:567`),
default 0. Then

$$
C_\mathrm{turb} = \tfrac{1}{2}\,e^{w^2}\,
\mathrm{erfc}\!\left[\frac{1.5\,w^2 - s_\mathrm{max}}{\sqrt{2}\,w}\right],
\qquad
s_\mathrm{max} = \ln\frac{n_\mathrm{max}}{\min(n_\mathrm{H},\,n_\mathrm{max})}
$$ (eq-dgd-boost)

(`solvers/dust_rates.py:279`–`281`), evaluated separately for each bin
because $n_\mathrm{max}$ differs between bins.

:::{note}
This deliberately follows the ramses-yomp convention
(`solvers/dust_rates.py:237`–`246`): $w = \ln(1+b^2\mathcal{M}^2)$ with
$b = 0.4$ is used as the log-normal *width*, and $w^2$ as the variance. The
textbook log-normal has variance $\sigma_s^2 = \ln(1+b^2\mathcal{M}^2)$
instead. The docstring (`solvers/dust_rates.py:180`–`182`) states the same
convention. With $\sigma = 0$ the Mach floor gives $w \approx 1.6\times10^{-11}$
and $C_\mathrm{turb}\to 1$ whenever $n_\mathrm{H} < n_\mathrm{max}$.
:::

### Coulomb enhancement

$E_C$ is a fixed factor per bin, not a computed grain charge. It is applied
only when $T < 2\times10^4$ K **or** $n_\mathrm{H} > 10$ cm$^{-3}$
(`solvers/dust_rates.py:257`); otherwise $E_C = 1$. When applied
(`solvers/dust_rates.py:303`–`313`):

| Bin | Condition | $E_C$ |
|---|---|---|
| Large carbonaceous | `composition == "graphite"` and $a > 0.05$ µm | $10^{-5}$ |
| Small silicate | `composition == "silicate"` and $a < 0.05$ µm | 10 |
| All others | | 1 |

:::{note}
The `accretion_rate` docstring gives $E_C = 0$ for large carbonaceous grains
(`solvers/dust_rates.py:192`). The code uses $10^{-5}$
(`solvers/dust_rates.py:307`), matching the Fortran value quoted in the
comment. The solver loads per-bin grain-charge tables and defines a
point-charge factor `_coulomb_factor_WD99` (`solvers/dust_rates.py:88`), but
accretion never calls them. The fixed factors above are the only Coulomb
treatment.
:::

## Thermal sputtering

The variant is selected by `models.dust_sputtering_model`
(`solvers/rhs.py:94`–`99`). The default is `"kirchschlager"`
(`solvers/dust_init.py:602`), and any unrecognised string also falls through
to the tabulated kernel.

| `dust_sputtering_model` | Kernel | Tables read |
|---|---|---|
| `kirchschlager` (default) | `thermal_sputtering_rate` | `thermal_sputtering_data/sputtering_<bin_id>_Z_<Z>` |
| `nozawa2006` | `thermal_sputtering_rate_nozawa` | none |
| `nozawa2006_ramses` | `thermal_sputtering_rate_nozawa_ramses` | none |

### Tabulated rates (default)

For each bin the solver loads one table per projectile species: H, He, C, N,
O, Ne, Mg, Si, S and Fe (`solvers/dust_init.py:137`, `:226`–`233`). Missing
tables are skipped silently, and a bin with no tables is not sputtered at
all (`solvers/dust_rates.py:367`–`368`). Each table gives
$R_e(T,\phi) = n_e^{-1}\,|da/dt|$ in µm yr$^{-1}$ cm$^3$. The solver
evaluates it at $\phi = 0$ (uncharged grain; `solvers/dust_rates.py:385`) and
forms

$$
k_\mathrm{spu} = \frac{3}{a_{\mu\mathrm{m}}\;t_\mathrm{yr}}
\sum_e \frac{\rho_e}{m_e}\,R_e(T, 0)
$$ (eq-dgd-spu-table)

(`solvers/dust_rates.py:372`–`393`), with $a_{\mu\mathrm{m}}$ the bin radius in
µm and $t_\mathrm{yr} = 3.1536\times10^7$ s (`solvers/dust_rates.py:33`). The
projectile density $\rho_e/m_e$ is the *total* gas-phase density of element
$e$, whatever its ionisation state. Species with $\rho_e < 10^{-40}$ g cm$^{-3}$
are ignored (`solvers/dust_rates.py:378`).

The tables are written by `calima-export`
(`models/dust_gas_collisions/export_sputtering_rates_bins.py`). By default
they cover $10^3$–$10^9$ K (`export_sputtering_rates_bins.py:23`–`24`), with
the bin radius taken from the grain-size JSON
(`export_sputtering_rates_bins.py:128`). For each species, `export_rates_T_phi`
averages the yield over a Maxwell–Boltzmann speed distribution
$f_\mathrm{MB}(v)$ (`models/dust_gas_collisions/dust_sputtering.py:124`), with
the projectile energy $E = \tfrac12 m_i v^2 + \phi$ (`dust_sputtering.py:791`):

$$
R_e = \frac{m_d}{2\rho_\mathrm{gr}}\int f_\mathrm{MB}(v)\,v\,Y(E)\,dv
\times 10^4\,t_\mathrm{yr}
$$ (eq-dgd-spu-export)

(`dust_sputtering.py:812`, `:1056`). Here $m_d$ is the mean atomic mass of
the grain material, and the result is floored at $10^{-30}$
(`dust_sputtering.py:1057`). The yield $Y(E)$ is the semi-infinite-target
formula

$$
Y = \frac{3.56}{U_0}\,\frac{M_i}{M_i+M_d}\,
\frac{Z_i Z_d}{\sqrt{Z_i^{2/3}+Z_d^{2/3}}}\,
\frac{\alpha}{K\mu+1}\,s_n(\varepsilon)
\left[1-\left(\frac{E_\mathrm{th}}{E}\right)^{2/3}\right]
\left(1-\frac{E_\mathrm{th}}{E}\right)^{2}
$$ (eq-dgd-yield)

(`dust_sputtering.py:320`), with $E$ and $U_0$ in eV, $\mu = M_d/M_i$, and
$s_n$ the screened-Coulomb nuclear stopping function of {cite:t}`Matsunami1980`
(`dust_sputtering.py:217`). The threshold is
$E_\mathrm{th} = U_0/[g(1-g)]$ if $M_i/M_d \le 0.3$, and
$8U_0(M_i/M_d)^{1/3}$ otherwise, where $g = 4M_iM_d/(M_i+M_d)^2$
(`dust_sputtering.py:250`–`257`). The model uses $U_0 = 4.0$ and $5.7$ eV and
$K = 0.65$ and $0.1$ for carbon and silicate (`dust_sputtering.py:64`–`65`).
The exporter also turns on a finite-size correction
(`export_sputtering_rates_bins.py:162`). This multiplies $Y$ by the fitted
function `bocchio_correction` of $x = a/(0.7\,r_p)$, clipped at zero, where
$r_p$ is the ion penetration depth. $r_p$ is integrated from a Bethe–Bloch
stopping formula (`dust_sputtering.py:301`–`306`, `:1545`). Grain materials are graphite
($\rho_\mathrm{gr}=2.24$, $M_d=12.011$, $Z_d=6$) and a silicate of mean
atomic mass $(24.305+55.845+28.0855+4\times15.999)/7$ with
$\rho_\mathrm{gr}=3.3$ g cm$^{-3}$ (`dust_sputtering.py:662`–`679`).

:::{note}
The table interpolator is bilinear in $(\log_{10}T,\phi)$ on
$\log_{10}R$. Outside the tabulated range it returns the table's smallest
positive rate, not the boundary value (`solvers/table_io.py:118`–`120`),
contrary to its docstring. In practice sputtering is negligible below
$10^3$ K.
:::

### Polynomial yields (`nozawa2006`, `nozawa2006_ramses`)

Both variants use the {cite:t}`Hu2019` fifth-degree polynomial fit to the
{cite:t}`Nozawa2006` yields:

$$
\log_{10} Y_\mathrm{th} = \sum_{k=0}^{5} c_k\,(\log_{10}[0.60\,T])^k ,
$$ (eq-dgd-hu)

in µm yr$^{-1}$ cm$^3$ (`solvers/dust_rates.py:78`–`81`). The coefficients
for carbon and silicate are at `solvers/dust_rates.py:45`–`52`, and $0.60$
is the mean molecular weight of fully ionised gas. The rate is scaled with
$n_\mathrm{H}$ rather than with individual ion densities:

$$
k_\mathrm{spu} = \frac{3\,n_\mathrm{H}\,Y_\mathrm{th}(T)}{a_{\mu\mathrm{m}}\,t_\mathrm{yr}}
$$ (eq-dgd-spu-hu)

(`solvers/dust_rates.py:459`–`462`). A bin counts as carbonaceous if its
highest mass-fraction element is C (`solvers/dust_rates.py:450`–`452`).

- `nozawa2006` assigns the composition-matched yield
  (`solvers/dust_rates.py:454`).
- `nozawa2006_ramses` deliberately **swaps** the yields: carbon bins get the
  silicate yield and silicate bins get the carbon yield
  (`solvers/dust_rates.py:522`). This reproduces the bin-index-based dispatch
  of the RAMSES-CALIMA Fortran runs of {cite:t}`Dubois2024`, whose bin
  ordering is the reverse of pyCALIMA's. Use it only to compare against
  those simulations. Above roughly $7\times10^5$ K the silicate yield exceeds
  the carbon yield, so the swap speeds up carbon-grain destruction in hot
  gas.

:::{note}
The `thermal_sputtering_rate_nozawa` docstring gives $Y_\mathrm{th}$ in
cm$^4$ yr$^{-1}$ (`solvers/dust_rates.py:424`). The implementation treats it
as µm yr$^{-1}$ cm$^3$, as `_nozawa_yield` documents
(`solvers/dust_rates.py:69`). The polynomial has no temperature cut-off; it
simply becomes negligible in cold gas.
:::

## Thermal sublimation

`sublimation_rate` implements the grain-evaporation model of
{cite:t}`GuhathakurtaDraine1989`. It reads a per-bin table
`dust_sublimation/sublimation_rate_<bin_id>.dat` (falling back to
`erosion_rate_<bin_id>.dat`) of $\log_{10}\varepsilon$, with
$\varepsilon = |da/dt|/a$ [s$^{-1}$], against the *dust* temperature
(`solvers/dust_init.py:278`–`325`). The mass-loss rate is

$$
k_\mathrm{sub} = 3\,\varepsilon(T_d)
$$ (eq-dgd-sub)

(`solvers/dust_rates.py:582`–`587`), since $m\propto a^3$. $T_d$ is the bin's
`dust_temperature_K`, or the environment's when the bin sets none; loading a
configuration with `dust_sublimation` on and no dust temperature raises
(`solvers/dust_init.py:593`). The gas temperature plays no role. The reader
interpolates $\log_{10}\varepsilon$ linearly in $T_d$, returns 0 below the
table and holds the last row above it. Tables without the `log10(epsilon`
column header are older exports holding linear $\varepsilon$, and are read as
such. The loader warns when a table was exported for a grain radius more than
1% different from the solver bin's, since $\varepsilon \propto 1/a$.

The tables come from `write_sublimation_rate_tables` in
`models/dust_radiation/dust_sublimation.py`, run by `calima-export`. For the
representative radius of each bin in the grain-size JSON, and on 300
log-spaced dust temperatures $T_d$ from 200 to 4000 K
(`dust_sublimation.py:1352`), it evaluates:

$$
J = \alpha_N\,P\,\exp\!\left[-\frac{B_\infty - \sigma N^{-1/3}}{T_d}\right],\qquad
\frac{dN}{dt} = 4\pi a^2 J f_\mathrm{corr},\qquad
\left|\frac{da}{dt}\right| = \frac{m_\mathrm{mon}}{\rho_\mathrm{gr}}\,\frac{dN/dt}{4\pi a^2}
$$ (eq-dgd-gd89)

(`dust_sublimation.py:172`–`175`, `:277`, `:291`). The terms are:

- $\alpha_N = 0.1$ (`dust_sublimation.py:87`);
- $N$ is the number of monomers (C atoms for graphite, Mg$_2$SiO$_4$ units
  for silicate);
- $f_\mathrm{corr}$ is the microcanonical correction, a ratio of Gamma
  functions of the grain's internal energy (`dust_sublimation.py:215`–`265`).
  The lnΓ differences are evaluated with Stirling's series above argument 50,
  so that $f_\mathrm{corr}\to1$ for large grains instead of losing all
  precision.

The material constants (`dust_sublimation.py:88`–`104`) are:

| Material | $B_\infty/k_B$ [K] | $\sigma/k_B$ [K] | $P$ [cm$^{-2}$ s$^{-1}$] | $m_\mathrm{mon}$ [amu] | $\rho_\mathrm{gr}$ [g cm$^{-3}$] |
|---|---|---|---|---|---|
| graphite | 81200 | 20000 | $4.6\times10^{30}$ | 12.011 | 2.24 |
| silicate | 68100 | 20000 | $7\times10^{30}$ | 140.69 | 3.5 |

Only rows whose timescale $a/|da/dt|$ is at most 10 times
$1.38\times10^{10}$ yr are written (`dust_sublimation.py:1402`–`1417`).

:::{note}
For large grains this reduces to the bulk rate
$|da/dt| = (m/\rho)^{1/3}\nu_0 e^{-B/kT}$ of {cite:t}`WaxmanDraine2000`
(their eq. 10), with the GD89 constants they quote
(`wd00_sublimation_timescale`). Graphite agrees to 1%. Silicate lifetimes
are 1.7 times longer than that formula with $m$ = one Mg$_2$SiO$_4$ unit, and
10% shorter with $m$ = the mean atomic mass; WD00 do not say which. That is
$\lesssim 20$ K in $T_\mathrm{sub}$. `diagnostics/dust_radiation/check_sublimation_radius_agn.py`
compares the resulting sublimation radii in a quasar SED with Barvainis
(1987), Kishimoto et al. (2007), Nenkova et al. (2008), Mor & Netzer (2012)
and Baskin & Laor (2018).

On its own $\varepsilon$ is the evaporation rate into vacuum, so
$T_\mathrm{sub}$ does not depend on the gas density; `dust_condensation`
(below) adds the reverse flux.
:::

## Recondensation

`condensation_rate` is the reverse of sublimation: the gas-phase atoms that
hit a hot grain and stick. With `dust_condensation` and `dust_sublimation`
both on, the pair is the net evaporation–condensation exchange of a grain
with its vapour, so the dust survives only where the gas is supersaturated
and $T_\mathrm{sub}$ depends on the gas density {cite:p}`BaskinLaor2018`.

The growth rate is the kinetic (Hertz–Knudsen) flux of the key species $X$,

$$
k_\mathrm{cond} = \alpha\,\frac{\pi a^2}{m_\mathrm{gr}}\,
\sqrt{\frac{8k_BT}{\pi}}\;\min_e\frac{\rho_e}{f_e\sqrt{m_e}}
\times C_\mathrm{Coul}\times C_\mathrm{bal},
$$ (eq-dgd-cond)

(`solvers/dust_rates.py:681`–`682`), i.e. $dm/dt = \alpha\pi a^2 n_X v_X (m_X/f_X)$ per grain, with $v_X$ at the
*gas* temperature and $X$ the element whose arrival limits growth. It is the
accretion kernel {eq}`eq-dgd-kacc` without the cold-ISM regime switches: no
sub-grid branch, no $T_\mathrm{acc}=100$ K, no CO lock, no $1.8\times10^5$ K
cut-off, and Le Bourlot sticking replaced by the condensation coefficient
$\alpha$ (`condensation_coefficient`, default 0.1). Detailed balance needs
the $\alpha_N$ the tables were built with: the evaporation flux
$\alpha_N P e^{-B/kT_d}$ then cancels condensation at
$n_\mathrm{sat}(T_d) = 4Pe^{-B/kT_d}/v_X(T_d)$, whatever $\alpha$ is. The
two cannot be on together with `dust_accretion`, which would count the same
collisions twice; loading such a configuration raises
(`solvers/dust_init.py:614`–`621`).

$C_\mathrm{bal}$ selects the balance condition (`condensation_balance`):

- `flux` (default): $C_\mathrm{bal}=1$, so the balance is
  $n_X v_X(T) = n_\mathrm{sat} v_X(T_d)$, the kinetic condition for a grain
  at $T_d$ in gas at $T$;
- `pressure`: $C_\mathrm{bal}=\sqrt{T/T_d}$, which turns it into
  $n_X k_B T = n_\mathrm{sat} k_B T_d$, the equal-pressure condition of
  {cite:t}`BaskinLaor2018` (their eq. 27; `solvers/dust_rates.py:678`).

$C_\mathrm{Coul} = (1-x_i) + x_i D(Z)$, where $x_i$ is the fraction of $X$
that arrives singly ionised (`condensation_ion_fraction`, default 0) and $D$
is the {cite:t}`WeingartnerDraine1999` point-charge factor for a grain at
potential $U$ (`grain_potential_V`, $Z = Ua/e$); a positive grain repels the
ions as $e^{-eU/k_BT}$.

The tables hold the rate of a grain of the bin's representative radius, and
the bins have fixed sizes, so net growth or loss moves mass in and out of a
bin but does not move it between bins. For silicate the tables count
Mg$_2$SiO$_4$ units; a bin of another stoichiometry adds $m_X/f_X$ per $X$
atom, which shifts the balance by that mass ratio (a few K in
$T_\mathrm{sub}$).

Integrate this pair with `mprk22` or `sparse_newton` (or `rk4`/`rk54`, which
are accurate but need many steps). With the two in balance the stiff variable
is the gas; `anninos` treats it explicitly and `newton_krylov` cannot meet its
stationarity test, and `calima-run` warns about both
({ref}`solver-types`).

:::{note}
`diagnostics/dust_radiation/check_recondensation_baskin_laor.py` solves for
$T_\mathrm{sub}(n)$ with the solver kernels and runs closed boxes to
equilibrium. For 0.1 µm graphite with `pressure` balance at $T = 10^4$ K it
reproduces Baskin & Laor's eq. 27 to 1–2 K over $n = 10^2$–$10^{14}$ cm$^{-3}$;
silicate (Si as key species) is 14–44 K cooler, since the GD89 silicate
constants give a vapour pressure 5 times theirs. `flux` balance is 20–50 K
cooler than `pressure`. Their sublimation time (eq. 36) takes the
condensation flux as $n v 4\pi a^2$ with unit sticking; with the one-sided
flux $n v\pi a^2$ and $\alpha = 0.1$ the grains here live 40 times longer.
:::

## Nucleation

`nucleation_rate` (`solvers/dust_rates.py:696`) forms new grains from the gas
by steady-state classical nucleation in the key-species form of
{cite:t}`KozasaHasegawa1987` and {cite:t}`Nozawa2003` (their eq. 6):

$$
J_s = \alpha\,\Omega_0\left(\frac{2\sigma}{\pi m_X}\right)^{1/2}
(f_n n_X)^2 \exp\!\left[-\frac{4\mu^3}{27(\ln S)^2}\right],\qquad
n_* = \left(\frac{2\mu}{3\ln S}\right)^3,
$$ (eq-dgd-nuc)

with $\Omega_0$ the volume per key species $X$, $f_n$ its neutral fraction
and $\mu = 4\pi a_0^2\sigma/kT_c$. Nucleation shares the thermodynamics of
sublimation and recondensation:

- the supersaturation is the flux ratio those two balance at $S=1$,
  $S = f_n n_X v_X(T_g)/[4P_0 e^{-B/kT_c}]$;
- $\mu = \tfrac32\sigma_\mathrm{GD89}/T_c$, from the GD89 surface term of
  $B(N) = B_\infty - \sigma N^{-1/3}$, so that $n_*$ is the size at which the
  size-corrected evaporation of the sublimation tables balances growth. For
  graphite this gives $\sigma \simeq 2000$ erg cm$^{-2}$ (Nozawa et al. 2003
  list 1400).

$T_c$ is the cluster temperature. By default it is the target bin's dust
temperature, because near an AGN clusters are heated by the radiation field
({cite:t}`Lazzati2008`); $T_g$ is the temperature of the nucleating gas.
Where $n_*$ falls below ~2 the formula exceeds the rate at which monomers
meet, so $J_s$ is capped at the dimer collision rate
(`solvers/dust_rates.py:754`).

Each cluster is counted as one grain of a designated target bin, so grain
number is conserved and the bin gains $J_s m_\mathrm{grain}$ per unit volume
from the gas (`solvers/dust_rates.py:757`). Growth from $n_*$ to the bin
radius is assumed instantaneous; at $n_\mathrm{H}\sim10^{10}$ cm$^{-3}$ it
takes days for nm bins, so the target should be a small bin.

The `physics` flag is `dust_nucleation`. The constants come from a
`nucleation` list in the configuration (`solvers/dust_init.py:328`), one entry
per composition: `target_bin`, `key_element`, `evaporation_prefactor_cm2s`,
`binding_energy_K`, `surface_energy_K`, `monomer_mass_amu`, and optionally
`density_gcm3`, `alpha` (0.1), `neutral_fraction` (1), `gas_temperature_K`
and `cluster_temperature_K`. `dust_sublimation.nucleation_parameters`
(`models/dust_radiation/dust_sublimation.py:389`) returns them for graphite
and silicate.

:::{important}
Classical nucleation assumes neutral gas. At $T_g\sim10^4$ K the key species
(C, Si, Mg, Fe) are mostly singly ionised and the precursor molecules are
photodissociated, so near an AGN nucleation can only proceed in shielded,
neutral gas. `neutral_fraction` and `gas_temperature_K` make that an explicit
assumption of the model rather than something it predicts.
:::

:::{note}
`diagnostics/dust_radiation/check_nucleation_nozawa_kozasa2013.py` runs the
steady-state model of {cite:t}`NozawaKozasa2013` for carbon grains in
homologously cooling gas, with `classical_nucleation_rate`
(`dust_sublimation.py:427`) and their thermodynamics. For
$\Lambda_\mathrm{on} = \tau_\mathrm{sat}/\tau_\mathrm{coll}$ from 30 to
$3\times10^6$ and $\gamma$ = 1.25 and 1.5, the final mean radius follows
their eq. 64 to within 9%, and the condensation efficiency is 1 (their
fitting formula is good to 5% against their non-steady model). For
$c_{10}=10^8$ cm$^{-3}$ it gives $T_0 = 1994$ K, an onset at
$x_\mathrm{on}=1.077$ and $\tau_\mathrm{sat}=10.5$ d (theirs: ~2000 K,
1.08, ~10 d). Below $\Lambda_\mathrm{on}\simeq30$ the steady model gives
too-small grains, as they found.
:::
