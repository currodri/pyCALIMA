(physics-charging-heating)=
# Grain charging and gas heating

Grain charge controls three quantities that pyCALIMA exports as lookup tables:
photoelectric heating of the gas, cooling by electron recombination, and the
rate at which grains neutralise gas-phase ions. The charge itself is tabulated
too. Gas-grain collisional cooling depends on charge only through a Coulomb
energy shift, so it is tabulated separately. All of this runs offline in
`models/`. As the last section shows, the Python solver reads almost none of it.
Units are cgs, energies are in eV where noted, and $a$ is the grain radius in cm.

## Charge distribution of a dust grain

`compute_equilibrium_charge_distribution_vectorized`
(`models/dust_charge/dust_charging.py:809`) solves for the steady-state
distribution $P(Z)$ over integer charges, following {cite:t}`WeingartnerDraine2001`.
The low-level physics is in `models/dust_charge/shared_physics.py`.

### Charging currents

**Photoemission.** For each charge state, the photoemission rate [s$^{-1}$] is

$$
R_{\rm pe}(Z) = \int C_{\rm abs}(\nu)\,J_\nu\,Y(\nu,Z)\,d\nu
 \;+\; [Z<0]\int \sigma_{\rm pd}(\nu,Z)\,J_\nu\,d\nu
$$ (eq-rpe)

(`models/dust_charge/dust_charging.py:324`–`327`, `350`). Here $J_\nu$ is a
photon flux [photons cm$^{-2}$ s$^{-1}$ Hz$^{-1}$], obtained from the
field's $I_E$ [erg s$^{-1}$ cm$^{-2}$ eV$^{-1}$] as
$J_\nu = I_E/(\nu\,\mathrm{eV})$ (`models/dust_charge/dust_charging.py:1664`).
The yield is $Y = y_2\min(y_0y_1,1)$ (`models/dust_charge/dust_charging.py:181`), where:

- $y_0$ is the bulk yield, a function of $x=\theta/W$:
  $9\times10^{-3}x^5/(1+3.7\times10^{-2}x^5)$ for graphite and
  $0.5x/(1+5x)$ for silicate (`models/dust_charge/shared_physics.py:226`, `231`);
- $y_1$ is the small-particle enhancement, computed from $a/l_a$ and
  $a/l_e$ (`models/dust_charge/shared_physics.py:216`–`221`).
  The electron escape length is $l_e = 10^{-7}$ cm
  (`models/dust_charge/shared_physics.py:18`). The photon attenuation length is
  $l_a^{-1} = (4\pi/\lambda)(\tfrac23{\rm Im}\,m_\perp+\tfrac13{\rm Im}\,m_\parallel)$
  for graphite and $l_a = \lambda/(4\pi\,{\rm Im}\,m)$ for silicate
  (`models/dust_charge/shared_physics.py:208`, `213`);
- $y_2 = E_{\rm high}^2(E_{\rm high}-3E_{\rm low})/(E_{\rm high}-E_{\rm low})^3$
  for $Z\ge0$, with $E_{\rm low}=-(Z+1)e^2/a$ and $E_{\rm high}=h\nu-h\nu_{\rm pet}$;
  $y_2=1$ for $Z<0$ (`models/dust_charge/shared_physics.py:195`–`204`).

The energies involved are:

$$
{\rm IP_V} = W + \frac{e^2}{a}\Big[(Z+\tfrac12)+(Z+2)\frac{0.3\,\text{Å}}{a}\Big],\qquad
E_{\rm min} = -\frac{(Z+1)e^2/a}{1+(27\,\text{Å}/a)^{0.75}}\ (Z<0)
$$ (eq-ipv)

(`models/dust_charge/shared_physics.py:121`, `133`–`134`). The threshold is
$h\nu_{\rm pet}={\rm IP_V}$ for $Z\ge-1$ and ${\rm IP_V}+E_{\rm min}$ otherwise.
The parameter $\theta = h\nu-h\nu_{\rm pet}$ gains an extra $(Z+1)e^2/a$ when
$Z\ge0$ (`models/dust_charge/shared_physics.py:169`, `174`). The work functions
are $W=4.4$ eV (graphite) and $8.0$ eV (silicate), and the silicate band gap is
5 eV (`models/dust_charge/shared_physics.py:15`–`17`). Photodetachment uses the
cross-section

$$
\sigma_{\rm pd} = 1.2\times10^{-17}\,|Z|\,\frac{x}{(1+x^2/3)^2}\ {\rm cm^2},
\qquad x = \frac{h\nu-E_{\rm pd}}{3\ {\rm eV}}
$$ (eq-spd)

(`models/dust_charge/shared_physics.py:162`–`163`), with
$E_{\rm pd} = {\rm EA}(Z+1)+E_{\rm min}(Z)$. The electron affinity is
${\rm EA}=W+(e^2/a)[(Z-\tfrac12)-4\,\text{Å}/(a+7\,\text{Å})]$ for graphite and
$W-E_{\rm bg}+(e^2/a)(Z-\tfrac12)$ for silicate
(`models/dust_charge/shared_physics.py:125`, `129`, `146`).

**Collisional capture.** An electron or ion of charge $q$ is captured at the rate

$$
J_q(Z) = s_q\,\pi a^2\,n_q\left(\frac{8kT_q}{\pi m_q}\right)^{1/2}\tilde J(\nu=Z/q,\ \tau=akT_q/q^2e^2)
$$ (eq-jcoll)

(`models/dust_charge/shared_physics.py:375`–`379`, `420`–`422`), which is the
{cite:t}`DraineSutin1987` focusing factor $\tilde J$. The code evaluates
$1+(\pi/2\tau)^{1/2}$ for $\nu=0$ and $(1-\nu/\tau)[1+(2/(\tau-2\nu))^{1/2}]$ for
$\nu<0$. For $\nu>0$ it evaluates $[1+(4\tau+3\nu)^{-1/2}]^2e^{-\theta_\nu/\tau}$
with $\theta_\nu = 1/(1+\nu^{-1/2})$ (`models/dust_charge/shared_physics.py:33`–`45`).
The electron sticking coefficient is $s_e=\tfrac12(1-e^{-a/l_e})$ for $Z>0$. For
$Z=0$ and $Z_{\rm min}<Z<0$ it is multiplied by $[1+e^{20-N_{\rm C}}]^{-1}$, with
$N_{\rm C}=468\,(a/1\,{\rm nm})^3$, and it is zero at or below $Z_{\rm min}$
(`models/dust_charge/shared_physics.py:54`–`65`). Ions have no sticking factor
in the charging currents ($s_i = 1$).

:::{note}
The same $\theta_\nu$ expression appears in $\tilde\lambda$
(`models/dust_charge/shared_physics.py:104`). The module
`dust_photoelectric_heating.py` tabulates `DS87_theta_nu` against `DS87_nu`
(`models/dust_charge/dust_photoelectric_heating.py:96`–`97`) and labels that
table $\theta_\nu/\nu$ in `plot_DS87_thetanu`
(`models/dust_charge/dust_photoelectric_heating.py:1606`). The code's
$1/(1+\nu^{-1/2})$ matches those tabulated values, so what the code puts in the
exponent is $\theta_\nu/\nu$, not $\theta_\nu$. As a result the repulsive
($\nu>0$) branch is suppressed less strongly than a $\theta_\nu$ exponent would
give.
:::

### Charge balance

The allowed charges run from
$Z_{\rm min}=\lfloor -U_{\rm ait}\,(a/\text{Å})/14.4\rfloor$ to

$$
Z_{\rm max}=\Big\lfloor\frac{(h\nu_{\rm max}-W)(a/\text{Å})/14.4+0.5-0.3\,\text{Å}/a}{1+0.3\,\text{Å}/a}\Big\rfloor .
$$

$U_{\rm ait}$ is $3.9+0.12\,(a/\text{Å})+2/(a/\text{Å})$ eV for graphite and
$2.5+0.07\,(a/\text{Å})+8/(a/\text{Å})$ eV for silicate
(`models/dust_charge/shared_physics.py:284`–`306`). $h\nu_{\rm max}$ is the
highest photon energy in the field (`models/dust_charge/dust_charging.py:607`).
A bisection finds a reference charge $Z_{\rm ref}$
(`models/dust_charge/dust_charging.py:659`–`669`), and the window around it
doubles in width until $P<10^{-4}$ at both edges
(`models/dust_charge/dust_charging.py:789`). Detailed balance between adjacent
states is then solved in log space:

$$
\frac{P(Z+1)}{P(Z)} = \frac{R_{\rm pe}(Z)+J_{\rm ion}(Z)}{J_e(Z+1)}
$$ (eq-detbal)

(`models/dust_charge/dust_charging.py:978`–`1021`). The outputs are the moments
$\langle Z\rangle=\sum ZP$ and $\sigma_Z=[\sum(Z-\langle Z\rangle)^2P]^{1/2}$
(`models/dust_charge/dust_charging.py:1039`–`1041`).

Every exporter passes an empty ion list: the heating grid, the recombination
grid and the stand-alone charge scan
(`models/dust_charge/export_dust_photoelectric_heating.py:280`,
`models/dust_charge/export_dust_ion_recombination.py:92`,
`models/dust_charge/export_dust_charging_vs_gamma.py:393`). In every table
below, charge is therefore set by photoemission against electron capture alone.

## Dust photoelectric heating and recombination cooling

For each charge state, the heating power [erg s$^{-1}$] is

$$
\Gamma_{\rm pe}(Z) = \int Y\,\frac{I_E}{E}\,C_{\rm abs}\,\frac{\int E f_E\,dE}{y_2}\,dE
 \;+\;[Z<0]\int\sigma_{\rm pd}\,\frac{I_E}{E}\,(E-E_{\rm pd}+E_{\rm min})\,dE
$$ (eq-gpe)

(`models/dust_charge/dust_photoelectric_heating.py:362`–`364`, `389`). The
photoelectron energy distribution is parabolic,
$f_E=6(E-E_{\rm low})(E_{\rm high}-E)/(E_{\rm high}-E_{\rm low})^3$
(`models/dust_charge/dust_photoelectric_heating.py:578`). The code integrates
it analytically (`models/dust_charge/dust_photoelectric_heating.py:254`–`271`)
{cite:p}`WeingartnerDraine2001`. The cooling per charge state from electron
capture is

$$
\Lambda_{\rm rec}(Z) = \pi a^2 n_e s_e\left(\frac{8kT}{\pi m_e}\right)^{1/2}\tilde\lambda(Z/(-1),\tau)\,kT
$$ (eq-lrec)

(`models/dust_charge/dust_photoelectric_heating.py:1097`), with $\tilde\lambda$
from `models/dust_charge/shared_physics.py:96`–`105`. Grains at $Z_{\rm min}$
also lose energy by autoionisation, at the rate
$\pi a^2n_eP(Z_{\rm min})v_e\tilde J\,{\rm EA}(Z_{\rm min})$ with no $s_e$
(`models/dust_charge/dust_photoelectric_heating.py:1108`–`1109`). The totals are
the averages $\sum P\,\Gamma_{\rm pe}$ and $\sum P\,\Lambda_{\rm rec}$
(`models/dust_charge/dust_charging.py:1090`–`1091`). What is tabulated is:

- heating $=\sum P\,\Gamma_{\rm pe} - \Lambda_{\rm auto}$
  (`models/dust_charge/dust_photoelectric_heating.py:2348`);
- cooling $=\sum P\,\Lambda_{\rm rec}$
  (`models/dust_charge/dust_photoelectric_heating.py:2349`).

Any negative part of the cooling would be moved into the heating
(`models/dust_charge/dust_photoelectric_heating.py:2524`–`2526`). Non-positive
values are written as $-10^{30}$
(`models/dust_charge/dust_photoelectric_heating.py:2529`).

The grid (`make_rate_gamma_T_tables`,
`models/dust_charge/dust_photoelectric_heating.py:2396`–`2409`) is log-spaced in
$T$ and in the charging parameter $\gamma=G_0\sqrt T/n_e$ [K$^{1/2}$ cm$^3$].
With `mode = "fix_G0"`, $G_0$ is held at `fixed_value` and
$n_e=G_0\sqrt T/\gamma$ (floored at $10^{-20}$). $G_0$ scales the chosen field
(`radiation_model`, default `Mathis`) linearly
(`models/dust_charge/dust_photoelectric_heating.py:2338`). The shipped
configuration uses $T=10$–$10^5$ K and $\gamma=10^{-6}$–$10^6$, with 100 points
on each axis.

## Grain-assisted ion recombination

`compute_ion_recombination_coefficients`
(`models/dust_charge/dust_ion_recombination.py:79`) averages the collision rate
over $P(Z)$:

$$
\alpha_i = \sum_Z P(Z)\,\pi a^2\left(\frac{8kT}{\pi m_i}\right)^{1/2} s_i\,\tilde J(Z/z_i,\tau)\,\Theta_i(Z)\quad[{\rm cm^3\,s^{-1}\ per\ grain}]
$$ (eq-alpha)

(`models/dust_charge/dust_ion_recombination.py:178`), with $s_i=1$. The
exporter uses "case A" of {cite:t}`WeingartnerDraine2001`: $\Theta_i=1$ if
${\rm IP}(X)\ge {\rm IP}(a,Z)$ and 0 otherwise
(`models/dust_charge/dust_ion_recombination.py:163`–`164`,
`models/dust_charge/export_dust_ion_recombination.py:113`). The grain
ionisation potential ${\rm IP}(a,Z)$ is ${\rm IP_V}$ for $Z\ge0$ and
${\rm EA}(Z+1)$ for $Z<0$. The threshold is applied only after $P(Z)$ has been
solved; $P(Z)$ itself is found without ions. Eleven singly charged ions are
included (H, He, C, Na, Mg, Si, S, K, Ca, Mn, Fe), all at the gas temperature
(`models/dust_charge/export_dust_ion_recombination.py:104`–`108`). The grid is
the same $(T,\gamma)$ grid as the heating tables, from the exporter defaults;
the shipped configuration has no `dust_ion_recombination` block.

## Coulomb factor and what the solver uses

`models/dust_charge/Coulomb_enhancement.py` averages the
{cite:t}`WeingartnerDraine1999` point-charge factor over a charge distribution
(`cmp_D_WD99`, `models/dust_charge/Coulomb_enhancement.py:29`–`48`). It also
offers several Gaussian approximations of the same average. Per charge state
the factor is

$$
B = \begin{cases} e^{-Z Z_i e^2/akT} & ZZ_i>0\\ 1 - ZZ_ie^2/akT & ZZ_i<0\\ 1+(\pi Z_i^2e^2/2akT)^{1/2} & Z=0\end{cases}
$$ (eq-coulomb)

(`models/dust_charge/Coulomb_enhancement.py:39`–`43`). No export stage calls
this module.

On the solver side, `solvers/dust_init.py:419`–`431` loads
`dust_charge_Z_vs_T_<bin_id>` and `dust_charge_sigma_vs_T_<bin_id>` for each
bin through `build_charge_interpolator` (`solvers/table_io.py:301`). The mean
charge is clamped to $[-60,60]$ and the width to $[0,20]$. The resulting
`charge_Z_interp` is used only by `_get_coulomb_D`
(`solvers/dust_rates.py:131`–`159`), which applies Eq. {eq}`eq-coulomb` at
$\langle Z\rangle$ (`solvers/dust_rates.py:120`–`127`). Nothing calls
`_get_coulomb_D` or `_coulomb_factor_WD99`. Instead, `accretion_rate` multiplies
its rate by a fixed per-bin factor $E_C$ (`solvers/dust_rates.py:303`–`315`),
applied when $T<2\times10^4$ K or $n_{\rm H}>10$ cm$^{-3}$
(`solvers/dust_rates.py:257`):

| Bin | $E_C$ |
|---|---|
| graphite, $a>0.05$ µm | $10^{-5}$ |
| silicate, $a<0.05$ µm | 10 |
| all others | 1 |

These values follow {cite:t}`Dubois2024`; see {ref}`physics-dust-growth-destruction`.

:::{note}
The charge tables are written as $n_\gamma$ rows of $n_T$ values
(`models/dust_charge/export_dust_charging_vs_gamma.py:269`–`273`), but
`build_charge_interpolator` reads $n_T$ rows of $n_\gamma$ values
(`solvers/table_io.py:337`–`346`). On the default square $100\times100$ grid,
the table is therefore read transposed without any error. On a non-square grid
the read fails, and `dust_init` quietly sets the interpolator to `None`
(`solvers/dust_init.py:429`–`430`). Today this has no effect, because nothing
uses the interpolator.
:::

## PAH photoelectric heating

`models/PAH_charge/PAH_photoelectric_heating.py` follows {cite:t}`Berne2022`
and tracks four charge states ($Z=-1,0,1,2$) of a PAH with $N_{\rm C}$ carbon
atoms and radius $a=(N_{\rm C}/468)^{1/3}$ nm
(`models/PAH_charge/PAH_photoelectric_heating.py:890`). The ionisation
potential is ${\rm IP}=3.9\ {\rm eV}+(e/4\pi\epsilon_0)[(Z+\tfrac12)/a+(Z+2)(0.03\,{\rm nm})/a^2]$
for $Z\neq-1$, and 6.0 eV for $Z=-1$
(`models/PAH_charge/PAH_photoelectric_heating.py:187`–`189`)
{cite:p}`WeingartnerDraine2001,Wenzel2020`. The yields are piecewise linear
(`models/PAH_charge/PAH_photoelectric_heating.py:221`–`246`). For photodetachment,
photoionisation and absorption the code evaluates

$$
k = 2\pi\!\int_{\rm IP}^{13.6}\!Y\sigma\frac{I}{E}dE,\quad
P_{\rm inj}=2\pi\!\int_{\rm IP}^{13.6}\!(E-{\rm IP})Y\sigma\frac{I}{E}dE,\quad
P_{\rm abs}=2\pi\!\int^{13.6}\!\sigma I\,dE
$$ (eq-pah-rates)

(`models/PAH_charge/PAH_photoelectric_heating.py:548`, `565`, `583`,
`895`–`914`). Here $I$ is in W m$^{-2}$ eV$^{-1}$ and $\sigma$ in m$^2$; $k$
is converted to s$^{-1}$ with the factor $6.2415\times10^{18}$ eV J$^{-1}$
(`models/PAH_charge/PAH_photoelectric_heating.py:548`), and the powers to
erg s$^{-1}$ with $10^7$ (`models/PAH_charge/PAH_photoelectric_heating.py:994`–`1001`). The
neutral and cation $P_{\rm inj}$ are multiplied by a partition coefficient of
0.46 {cite:p}`Brechignac2014` (`models/PAH_charge/PAH_photoelectric_heating.py:81`,
`908`–`909`).

Two models set the electron-capture rates, selected by `attach_model`. In
`Berne`, attachment follows {cite:t}`Carelli2013`,
$2.74\times10^{-9}(T/300)^{0.11}e^{1.12/T}$ cm$^3$ s$^{-1}$, and recombination is
$1.28\times10^{-10}N_{\rm C}\sqrt T\,[1+\phi(1+Z)]$ with
$\phi=1.85\times10^5/(T\sqrt{N_{\rm C}})$. In `Tielens`, attachment is
$1.3\times10^{-7}\sqrt{N_{\rm C}}$ {cite:p}`Tielens2005` and recombination is
$1.3\times10^{-6}\sqrt{N_{\rm C}}\sqrt{300/T}$ {cite:p}`Tielens2021`
(`models/PAH_charge/PAH_photoelectric_heating.py:293`–`343`). The code solves
the four-state balance for the fractions $f_Z$ and forms three outputs:

- $P_{\rm inj}=\sum f_ZP_{{\rm inj},Z}$ and $P_{\rm abs}=\sum f_ZP_{{\rm abs},Z}$,
  in erg s$^{-1}$;
- the efficiency $\epsilon=P_{\rm inj}/P_{\rm abs}$;
- recombination cooling with $\tfrac32kT$ per captured electron.

(`models/PAH_charge/PAH_photoelectric_heating.py:978`–`1006`.)

:::{note}
The last term of $f_2$ divides by $k_{\rm det}k_{\rm pe,0}k_{\rm pe,0}$
(`models/PAH_charge/PAH_photoelectric_heating.py:989`). The other three
fractions imply $k_{\rm det}k_{\rm pe,0}k_{\rm pe,1}$. Renormalising afterwards
does not remove this inconsistency.
:::

The exporter does not scale the field. $G_0$ is the field's own integral,
$2\pi\int_{5.17}^{13.6}I\,dE/1.68\times10^{-6}$
(`models/PAH_charge/PAH_photoelectric_heating.py:882`). $\gamma$ is swept by
varying $n_e$ at one fixed $T$ (1000 K in the shipped configuration), and
$\gamma=G_0\sqrt T/n_e$ (`models/PAH_charge/PAH_photoelectric_heating.py:1658`).
The `G0` export parameter is recorded in `index.json` but is not passed to the
calculation.

## Gas-grain collisional cooling

`export_collisional_cooling` (`models/dust_gas_collisions/dust_collisional_cooling.py:1986`)
follows each projectile of energy $E$ through a path of at most $\tfrac43a$ and
records the energy $E_{\rm dep}$ it deposits
(`models/dust_gas_collisions/dust_collisional_cooling.py:505`). For ions the
stopping is nuclear, using the screened-Coulomb function of {cite:t}`Matsunami1980`,
plus electronic (`models/dust_gas_collisions/dust_collisional_cooling.py:510`–`513`).
For electrons it is a fit to tabulated stopping powers; for carbonaceous
grains the fitted data are those of {cite:t}`Joy1995`
(`models/dust_gas_collisions/dust_collisional_cooling.py:2116`–`2137`).
Charge enters only as a shift $E\to E+\phi$
(`models/dust_gas_collisions/dust_collisional_cooling.py:1071`, `1229`). The
cooling table is

$$
H(T,\phi) = \left(\frac{32}{\pi m}\right)^{1/2}\pi a^2(kT)^{1/2}k\cdot\frac12\int x^2e^{-x}\frac{E_{\rm dep}(xkT+\phi)}{xkT+\phi}\,dx
$$ (eq-hcoll)

(`models/dust_gas_collisions/dust_collisional_cooling.py:1063`, `1087`, `2178`,
`2237`). It is written per unit $(T-T_d)$: the reader forms the heating per
grain as $n\,H\,(T-T_d)$
(`models/dust_gas_collisions/dust_collisional_cooling.py:2451`–`2455`). $H$
therefore has units erg cm$^3$ s$^{-1}$ K$^{-1}$, even though the file header
says erg cm$^3$ s$^{-1}$ (`models/dust_gas_collisions/dust_collisional_cooling.py:2187`).

## Exported tables

| Directory / file | Exporter | Contents | Read by |
|---|---|---|---|
| `dust_photoelectric_heating_data/dust_rates_peh_<bin>.dat`, `dust_rates_rec_<bin>.dat` | `export_dust_photoelectric_heating` | 6 `#` lines; `nT nγ`; $\log T$; $\log\gamma$; $n_\gamma$ rows × $n_T$ of $\log_{10}$ rate [erg s$^{-1}$ per grain] | RAMSES Fortran only |
| `dust_photoelectric_heating_data/heating_<bin>.npz` | same | `T`, `gamma`, `peh_rate`, `rec_rate`, `G0_grid`, `ne_grid`, `Zmean_grid`, `Zsigma_grid` ($n_T\times n_\gamma$) | `export_dust_charging_vs_gamma` |
| `dust_charging_data/dust_charge_Z_vs_T_<bin>`, `dust_charge_sigma_vs_T_<bin>` | `export_dust_charging_vs_gamma` | same layout, $\langle Z\rangle$ and $\sigma_Z$ | `solvers/dust_init.py:419` (unused) |
| `dust_ion_recombination_data/dust_rates_ion_recomb_<bin>.dat` | `export_dust_ion_recombination` | `nT nγ`; axes; $n_Tn_\gamma$ rows ($T$ outer) × 11 ions of $\log_{10}\alpha$ [cm$^3$ s$^{-1}$] | none in pyCALIMA |
| `PAH_photoelectric_heating_data/peh_ISRF_<rad>_<opt>_<att>_<bin>.dat` | `export_PAH_photoelectric_heating_tables` | $\log\gamma$, $\log\epsilon$, $\log P_{\rm abs}$, $f_{-1}$, $f_0$, $f_1$, $f_2$ | none in the solver |
| `collisional_cooling_data/cooling_<bin>_Z_<Zi>` | `export_collisional_cooling_bins` | `nT nφ`; $\phi$ [eV]; rows of $\log T$, $\log_{10}H$. `Z_0` is electrons; the ions are H, He, C, N, O, Ne, Mg, Si, S, Fe | `models/dust_radiation/dust_emission.py:966` |

When run from `calima-export`, the charging stage reuses the heating NPZ
(`reuse_heating_data=True`, `models/export_all_grain_data.py:1255`). Run on its
own, it samples random $(T,n_e)$ pairs for each $\gamma$
(`models/dust_charge/dust_charging.py:1868`–`1882`). Each sample then gets its
own temperature column, so the written grid has many empty cells.
