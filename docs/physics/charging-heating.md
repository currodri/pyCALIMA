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
(`models/dust_charge/dust_charging.py:847`) solves for the steady-state
distribution $P(Z)$ over integer charges, following {cite:t}`WeingartnerDraine2001`.
The low-level physics is in `models/dust_charge/shared_physics.py`.

### Charging currents

**Photoemission.** For each charge state, the photoemission rate [s$^{-1}$] is

$$
R_{\rm pe}(Z) = \int C_{\rm abs}(\nu)\,J_\nu\,Y(\nu,Z)\,d\nu
 \;+\; [Z<0]\int \sigma_{\rm pd}(\nu,Z)\,J_\nu\,d\nu
$$ (eq-rpe)

(`models/dust_charge/dust_charging.py:318`–`321`, `341`). Here $J_\nu$ is a
photon flux [photons cm$^{-2}$ s$^{-1}$ Hz$^{-1}$], obtained from the
field's $I_E$ [erg s$^{-1}$ cm$^{-2}$ eV$^{-1}$] as
$J_\nu = I_E/(\nu\,\mathrm{eV})$ (`models/dust_charge/dust_charging.py:1638`).
The yield is $Y = y_2\min(y_0y_1,1)$ (`models/dust_charge/dust_charging.py:173`), where:

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
(`models/dust_charge/dust_photoelectric_heating.py:95`–`96`) and labels that
table $\theta_\nu/\nu$ in `plot_DS87_thetanu`
(`models/dust_charge/dust_photoelectric_heating.py:1561`). The code's
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
highest photon energy in the field (`models/dust_charge/dust_charging.py:604`).
A bisection finds a reference charge $Z_{\rm ref}$
(`models/dust_charge/dust_charging.py:656`–`666`), and the window around it
doubles in width until $P<10^{-4}$ at both edges
(`models/dust_charge/dust_charging.py:786`). Detailed balance between adjacent
states is then solved in log space:

$$
\frac{P(Z+1)}{P(Z)} = \frac{R_{\rm pe}(Z)+J_{\rm ion}(Z)}{J_e(Z+1)}
$$ (eq-detbal)

(`models/dust_charge/dust_charging.py:1023`–`1066`). The outputs are the moments
$\langle Z\rangle=\sum ZP$ and $\sigma_Z=[\sum(Z-\langle Z\rangle)^2P]^{1/2}$
(`models/dust_charge/dust_charging.py:1084`–`1086`).

The uniform-ISRF charging
tables below use the WDB06 charge balance of `rt_group_charging` instead, with
H$^+$ at $n({\rm H^+})=n_e$ (`models/dust_charge/charging_isrf_tables.py:61`).

## Dust photoelectric heating and recombination cooling

For each charge state, the heating power [erg s$^{-1}$] is

$$
\Gamma_{\rm pe}(Z) = \int Y\,\frac{I_E}{E}\,C_{\rm abs}\,\frac{\int E f_E\,dE}{y_2}\,dE
 \;+\;[Z<0]\int\sigma_{\rm pd}\,\frac{I_E}{E}\,(E-E_{\rm pd}+E_{\rm min})\,dE
$$ (eq-gpe)

(`models/dust_charge/dust_photoelectric_heating.py:317`–`319`, `344`). The
photoelectron energy distribution is parabolic,
$f_E=6(E-E_{\rm low})(E_{\rm high}-E)/(E_{\rm high}-E_{\rm low})^3$
(`models/dust_charge/dust_photoelectric_heating.py:533`). The code integrates
it analytically (`models/dust_charge/dust_photoelectric_heating.py:209`–`226`)
{cite:p}`WeingartnerDraine2001`. The cooling per charge state from electron
capture is

$$
\Lambda_{\rm rec}(Z) = \pi a^2 n_e s_e\left(\frac{8kT}{\pi m_e}\right)^{1/2}\tilde\lambda(Z/(-1),\tau)\,kT
$$ (eq-lrec)

(`models/dust_charge/dust_photoelectric_heating.py:1052`), with $\tilde\lambda$
from `models/dust_charge/shared_physics.py:96`–`105`. Grains at $Z_{\rm min}$
also lose energy by autoionisation, at the rate
$\pi a^2n_eP(Z_{\rm min})v_e\tilde J\,{\rm EA}(Z_{\rm min})$ with no $s_e$
(`models/dust_charge/dust_photoelectric_heating.py:1063`–`1064`). The totals are
the averages $\sum P\,\Gamma_{\rm pe}$ and $\sum P\,\Lambda_{\rm rec}$
(`models/dust_charge/dust_charging.py:1145`–`1146`).

These rates are not tabulated from this solver. The tables that carry them are
the uniform-ISRF charging tables, `dust_charging_isrf_data/dust_charging_isrf_<bin>.dat`,
written by `export_dust_charging_isrf` through
`models/dust_charge/charging_isrf_tables.py`. They are built from each bin's
RT-group table with the WDB06 charge balance of `rt_group_charging`, under the
Mathis ISRF scaled to $G_0$ Habing in 6–13.6 eV
(`models/dust_charge/rt_group_charging.py:221`–`226`) and with H$^+$ at
$n({\rm H^+})=n_e$. Every rate is linear in $G_0$ or in $n_e$, so one table at
$G_0=1$ serves any $G_0$ through $x=n_e/G_0$. On 161 points of $\log T$ from 1
to 9 and 161 of $\log x$ from $-8$ to 8
(`models/dust_charge/charging_isrf_tables.py:38`–`39`), each node stores
$\langle Z\rangle$, $\sigma_Z$, $\ln\Gamma_1$ with $\Gamma=G_0\Gamma_1$,
$\lambda_1$ with $\Lambda=n_e\sqrt T\,\lambda_1$ (the net recombination cooling
with autoionisation, which can be negative), and $\ln D_j$ for $j=-1,1,2,3$
(Eq. {eq}`eq-coulomb` averaged over $P(Z)$), all evaluated on the exact $P(Z)$
(`models/dust_charge/charging_isrf_tables.py:59`–`71`). `isrf_lookup`
interpolates bilinearly in $(\log T,\log x)$, clamped to the grid
(`models/dust_charge/charging_isrf_tables.py:133`–`145`). RAMSES reads the
tables with `charging_model='WDB06isrf'`.

## Grain-assisted ion recombination

The coefficients $\alpha_i$ [cm$^3$ s$^{-1}$ per grain] of the singly charged
ions in `RECOMB_IONS` come from the WDB06 charge balance of `rt_group_charging`.
`GroupChargingTable._recombination` sums
$P(Z)\,\pi a^2(8kT/\pi m_i)^{1/2}\tilde J(Z,\tau)$ over the charges at or below
the "case A" threshold of {cite:t}`WeingartnerDraine2001`, the largest $Z$ with
${\rm IP}(a,Z)\le{\rm IP}(X)$ (`recombination_thresholds`)
(`models/dust_charge/rt_group_charging.py:538`–`547`, `256`–`266`); for a wide
$P(Z)$, `_recombination_fit` evaluates the same sum. RAMSES computes them itself
for `dust_ion_recombination` with `charging_model='WDB06rt'`, and reads them
from the uniform-ISRF charging tables with `charging_model='WDB06isrf'`.

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

On the solver side, `solvers/dust_init.py:420`–`421` loads each bin's
uniform-ISRF charging table, when present, into `DustBinParams.isrf_table`,
with the solver's own reader (`solvers/table_io.py`, so that solvers/ does not
import models/). `_get_coulomb_D` (`solvers/dust_rates.py:132`–`164`) looks it
up at $(G_0,T,n_e)$, with $G_0$ in Habing units as in the table. For impactor
charges $j=-1,1,2,3$ it returns the table's $D_j$, Eq. {eq}`eq-coulomb` averaged
over the exact $P(Z)$; for any other charge, the point-charge factor at the
table's $\langle Z\rangle$ (`_coulomb_factor_WD99`). Nothing calls
`_get_coulomb_D`. Instead, `accretion_rate` multiplies
its rate by a fixed per-bin factor $E_C$ (`solvers/dust_rates.py:308`–`320`),
applied when $T<2\times10^4$ K or $n_{\rm H}>10$ cm$^{-3}$
(`solvers/dust_rates.py:262`):

| Bin | $E_C$ |
|---|---|
| graphite, $a>0.05$ µm | $10^{-5}$ |
| silicate, $a<0.05$ µm | 10 |
| all others | 1 |

These values follow {cite:t}`Dubois2024`; see {ref}`physics-dust-growth-destruction`.

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
| `dust_charging_isrf_data/dust_charging_isrf_<bin>.dat` | `export_dust_charging_isrf` | 4 `#` lines; `version nT nX nQ a` ($a$ the grain radius [cm], which RAMSES checks against the bin's); $\log T$; $\log(n_e/G_0)$; 8 planes of $n_T$ rows × $n_X$: $\langle Z\rangle$, $\sigma_Z$, $\ln\Gamma_1$, $\lambda_1$, $\ln D_{-1}$, $\ln D_{+1}$, $\ln D_{+2}$, $\ln D_{+3}$ | RAMSES (`charging_model='WDB06isrf'`); `solvers/dust_init.py:420` |
| `PAH_photoelectric_heating_data/peh_ISRF_<rad>_<opt>_<att>_<bin>.dat` | `export_PAH_photoelectric_heating_tables` | $\log\gamma$, $\log\epsilon$, $\log P_{\rm abs}$, $f_{-1}$, $f_0$, $f_1$, $f_2$ | none in the solver |
| `collisional_cooling_data/cooling_<bin>_Z_<Zi>` | `export_collisional_cooling_bins` | `nT nφ`; $\phi$ [eV]; rows of $\log T$, $\log_{10}H$. `Z_0` is electrons; the ions are H, He, C, N, O, Ne, Mg, Si, S, Fe | `models/dust_radiation/dust_emission.py:966` |
