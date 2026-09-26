(physics-grain-collisions)=
# Grain-grain collisions: coagulation and shattering

Grain-grain collisions move mass *between dust bins* without touching the gas,
except for shattering fragments that fall below the smallest tracked size. The
solver has four coagulation kernels and three shattering kernels in
`solvers/dust_rates.py`. All rates below are in cgs: mass densities
$\rho$ in g cm$^{-3}$, rates in s$^{-1}$, mass fluxes in g cm$^{-3}$ s$^{-1}$.
The code converts years with $1\,\mathrm{yr} = 3.1536\times10^{7}$ s
(`solvers/dust_rates.py:33`).

## Selecting a kernel

`build_process_list` turns the `physics` flags and `models` strings into a
list of kernels. It adds a collision process only when the flag is set and
there are at least two dust bins (`solvers/rhs.py:110`, `solvers/rhs.py:143`).

| `physics` flag | `models` key | value | kernel | code |
|---|---|---|---|---|
| `dust_coagulation` | `coagulation_model` | `Aoyama2017` (default) | `coagulation_rate` | `solvers/rhs.py:118` |
| | | `Dubois2024` | `dubois_coagulation_rate` | `solvers/rhs.py:111` |
| | | `turbulent` | `turbulent_coagulation_rate` | `solvers/rhs.py:122` |
| | | `turbulent_all` | `turbulent_all_coagulation_rate` | `solvers/rhs.py:129` |
| | | anything else | `coagulation_rate` | `solvers/rhs.py:136` |
| `dust_shattering` | `shattering_model` | `Dubois2024` | `dubois_shattering_rate` | `solvers/rhs.py:144` |
| | | `turbulent_all` | `turbulent_all_shattering_rate` | `solvers/rhs.py:151` |
| | | `turbulent` (default) or anything else | `turbulent_shattering_rate` | `solvers/rhs.py:158` |

The defaults are read in `solvers/dust_init.py:595-597`. For the turbulent
kernels, `dust_velocity_model` (default `Ormel2007`) picks the relative-velocity
model.

### Parameters used by the kernels

| key | block | default | meaning | code |
|---|---|---|---|---|
| `coagulation_partner` | dust bin | none | id of the bin that receives coagulated mass (`Aoyama2017`, `turbulent`) | `solvers/dust_init.py:468-476` |
| `nh_coa` | dust bin | 0.1 cm$^{-3}$ | minimum $n_\mathrm{H}$ for `Aoyama2017` | `solvers/dust_init.py:458` |
| `amin_micron`, `amax_micron` | dust bin | $0.5a$, $2a$ | size range that sets the mass interval $[m_\mathrm{min}, m_\mathrm{max}]$ a bin accepts fragments into | `solvers/dust_init.py:433-438` |
| `catastrophic_specific_energy_erg_g` | dust bin | $10^{7}$ erg g$^{-1}$ | $Q^\ast$ | `solvers/dust_init.py:459-461` |
| `interact_pah` | dust bin | `false` | whether fragments can go into PAH bins | `solvers/dust_init.py:464` |
| `vthresh_coag_cm_s` | dust bin | $10^{4}$ cm s$^{-1}$ | sticking threshold $v_\mathrm{coag}$ | `solvers/dust_init.py:465` |
| `slope_frag_func` | `models` | $1.3/3$ | fragment slope $\alpha$ | `solvers/dust_init.py:605` |
| `local_sigma_km_s`, `local_dx_pc` | `turbulence` | 0 | $\sigma$ and $\Delta x$ | `solvers/dust_init.py:564-565` |

## Composition groups

`_composition_groups` splits the ordered bin list into runs of consecutive
bins that share a `composition` string. It returns the first and last index,
$(i_1, i_2)$, of each run (`solvers/dust_rates.py:873-892`). The grouping
assumes bins of one composition are contiguous and does not check it. The
`Dubois2024` kernels only exchange mass between the first ("small") and last
("large") bin of each group, and skip any group with a single bin
(`solvers/dust_rates.py:692`, `solvers/dust_rates.py:779`). The turbulent
shattering kernels and `turbulent_all_coagulation_rate` loop over every bin in
a group. `coagulation_rate` and `turbulent_coagulation_rate` instead use the
explicit `coagulation_partner`.

## Dubois2024 kernels

These kernels follow {cite:t}`Dubois2024`, using the timescale form of
{cite:t}`Aoyama2017`. They are what the `ramses_*` configurations use.

### Shattering

The kernel takes the large bin $L$ of each group, with dust-to-gas ratio
$D_L = \rho_L/\rho_\mathrm{gas}$ (`solvers/dust_rates.py:704`). It computes

$$
\frac{1}{t_\mathrm{sha}} =
\frac{D_L\, n_\mathrm{H}^{\,p_\mathrm{sh}}}
{5.41\times10^{5}\,\mathrm{yr}\;
\left(\dfrac{a_L}{0.1\,\mu\mathrm{m}}\right)
\left(\dfrac{s_L}{3\,\mathrm{g\,cm^{-3}}}\right)}
\quad [\mathrm{s^{-1}}],
$$ (eq-dubois-sha)

(`solvers/dust_rates.py:709-710`). Since $5.41\times10^{5}\,\mathrm{yr}/0.01 = 54.1$ Myr, this is
$t_\mathrm{sha} = 54.1\,\mathrm{Myr}\,(a_L/0.1\,\mu\mathrm{m})(s_L/3)(D_L/0.01)^{-1} n_\mathrm{H}^{-p_\mathrm{sh}}$.

The density exponent is $p_\mathrm{sh}=1$ for $n_\mathrm{H}<1$ cm$^{-3}$ and
$p_\mathrm{sh}=1/3$ otherwise (`solvers/dust_rates.py:686`). Shattering is off
for $n_\mathrm{H}\ge 10^{3}$ cm$^{-3}$ (`solvers/dust_rates.py:683`). There is
no temperature cut. A bin with $\rho_L$ at or below `smallr_dust`
($10^{-40}$ g cm$^{-3}$) is skipped (`solvers/dust_rates.py:701`). The mass
flux $\rho_L/t_\mathrm{sha}$ goes from bin $L$ to the small bin $S$
(`solvers/dust_rates.py:716-719`).

### Coagulation

For the small bin $S$, with $D_S=\rho_S/\rho_\mathrm{gas}$, the kernel computes

$$
\frac{1}{t_\mathrm{coa}} =
\frac{D_S}
{5.42\times10^{3}\,\mathrm{yr}\;
\left(\dfrac{a_S}{0.005\,\mu\mathrm{m}}\right)
\left(\dfrac{s_S}{3\,\mathrm{g\,cm^{-3}}}\right)}
\quad [\mathrm{s^{-1}}],
$$ (eq-dubois-coa)

(`solvers/dust_rates.py:796-797`). This is $t_\mathrm{coa}=0.542\,\mathrm{Myr}\,(a_S/0.005\,\mu\mathrm{m})(s_S/3)(D_S/0.01)^{-1}$, independent of density. The mass flux $\rho_S/t_\mathrm{coa}$ goes from $S$ to $L$ (`solvers/dust_rates.py:805-806`).

Coagulation runs only if all three conditions hold:

1. $T\le10^{4}$ K.
2. $n_\mathrm{H}\ge0.1$ cm$^{-3}$ (`solvers/dust_rates.py:765`). This threshold
   is hard-coded; the per-bin `nh_coa` is not used here.
3. The Jeans length does not exceed four cell sizes:

   $$
   \lambda_\mathrm{J}=\sqrt{\frac{\pi k_\mathrm{B}T}{G\,m_\mathrm{H}^{2}\,n_\mathrm{H}}}\;\le\;4\,\Delta x,
   $$

   with $G=6.674\times10^{-8}$, $m_\mathrm{H}=1.6726219\times10^{-24}$ g, and
   $\Delta x$ = `local_dx` in cm (`solvers/dust_rates.py:758-774`). The test is
   skipped entirely when $\Delta x=0$ (`solvers/dust_rates.py:771`), which is
   the case whenever the `turbulence` block has no `local_dx_pc`.

:::{note}
**Grain-density normalisation.** In both Dubois2024 kernels the bulk density
$s$ (`sgrain`, g cm$^{-3}$) is divided by 3 exactly once, inside the kernel
(`solvers/dust_rates.py:709`, `solvers/dust_rates.py:796`). This matches the
RAMSES Fortran, which divides `sgrain` by 3 once when it reads the namelist.
Do not divide by 3 again when you prepare inputs.

The `dubois_coagulation_rate` docstring differs from the code in two places:

- It writes the timescale with a bare $s_i$ (`solvers/dust_rates.py:745`). The
  code uses $s_i/3$.
- An inline comment quotes $0.27/F$ Myr (`solvers/dust_rates.py:793`). The code
  uses $5.42\times10^{3}$ yr per unit $D_S/0.01$, which is 0.542 Myr.

Equation {eq}`eq-dubois-coa` follows the code.
:::

## Aoyama2017 coagulation (default)

Each bin that has a `coagulation_partner` loses mass to that partner at the rate

$$
\mathcal{R}=k_0^\mathrm{coa}\,\frac{\rho_i}{n_\mathrm{H}},\qquad
k_0^\mathrm{coa}=\sqrt{\frac{8}{3\pi}}\;\frac{\pi(2a_i)^{2}}{m_i},
$$

(`solvers/dust_rates.py:636`, `solvers/dust_init.py:189-190`). The mass flux
is $\mathcal{R}\rho_i$ (`solvers/dust_rates.py:641-644`), from bin $i$ to its
partner. The kernel is off for $T>10^{4}$ K (`solvers/dust_rates.py:624`) and
for $n_\mathrm{H}<$ `nh_coa` (`solvers/dust_rates.py:630`). It follows
{cite:t}`Aoyama2017`.

:::{note}
$k_0^\mathrm{coa}$ has units of cm$^{2}$ g$^{-1}$, and $\rho_i/n_\mathrm{H}$
has units of g. So $\mathcal{R}$ as coded has units of cm$^{2}$, not the
s$^{-1}$ that the comment claims: no velocity factor appears. The
`DustBinParams` docstring also gives $k_0^\mathrm{coa}$ the wrong units
(cm$^{3}$ g$^{-1}$ s$^{-1}$). Treat this kernel's absolute rate with caution.
:::

## Turbulent kernels

All four turbulent kernels need a velocity dispersion $\sigma$ and an
injection scale $L$, taken from `_effective_sigma`. If both `local_sigma` and
`local_dx` are positive, the code uses $(\sigma, L) = $ (`local_sigma`,
`local_dx`). Otherwise it falls back to a CNM estimate for both
(`solvers/dust_rates.py:907-913`):

$$
\sigma = 5.67\times10^{5}\left(\frac{n_\mathrm{H}}{100}\right)^{-1/4}\ \mathrm{cm\,s^{-1}},\qquad
L = 10\,\mathrm{pc}\left(\frac{n_\mathrm{H}}{100}\right)^{-1/3}.
$$

So `local_dx` serves as the Jeans-test cell size in `Dubois2024` coagulation
and as the turbulent injection scale here.

Both turbulent coagulation kernels are off for $T>10^{4}$ K
(`solvers/dust_rates.py:1632`, `solvers/dust_rates.py:1698`). They weight each
collision by a sticking probability (`solvers/grain_dynamics.py:119-124`):

$$
p_\mathrm{stick}=\left[1+\exp\!\left(4\left(\frac{v_\mathrm{rel}}{v_\mathrm{coag}}-1\right)\right)\right]^{-1}.
$$

### `turbulent_coagulation_rate`

Each bin collides only with itself, and the merged mass goes to its
`coagulation_partner`. The rate is

$$
\mathcal{R}_i = k_0^\mathrm{coa}\,v_\mathrm{rel}(i,i)\,\rho_i\,p_\mathrm{stick}
\quad[\mathrm{s^{-1}}]
$$

(`solvers/dust_rates.py:1660`), and the mass flux is $\mathcal{R}_i\rho_i$
(`solvers/dust_rates.py:1665-1669`). There is no $n_\mathrm{H}$ threshold.

### `turbulent_all_coagulation_rate`

This kernel loops over pairs $(i,k)$ with $k\ge i$ inside a group. Bin $i$
runs up to $i_2-1$ (`solvers/dust_rates.py:1715`), so the largest bin never
starts a pair. For each pair:

- $K_{ik}=\sqrt{8/(3\pi)}\,\pi(a_i+a_k)^{2}\,v_\mathrm{rel}$ (`solvers/dust_rates.py:1738-1742`).
- $v_\mathrm{coag} = \min(v_{\mathrm{coag},i}, v_{\mathrm{coag},k})$ (`solvers/dust_rates.py:1735`).
- A symmetry factor 1/2 applies for $i=k$ (`solvers/dust_rates.py:1743`).
- Bin $i$ loses $\tfrac12^{\delta_{ik}}K_{ik}\,(\rho_k/m_k)\,\rho_i\,p_\mathrm{stick}$, and bin $k$ loses the mirror term (`solvers/dust_rates.py:1746-1748`).

The coagulated mass goes to the bin in the group whose $m$ is closest to
$m_i+m_k$ (`solvers/dust_rates.py:1756-1770`).

### Fragment distribution

The shattering kernels share `_compute_shattered_fragments`. It first computes
the impact energy and ejected mass from the target mass $m_1$ and projectile
mass $m_2$ (`solvers/dust_rates.py:1323-1325`):

$$
E_\mathrm{imp}=\tfrac12\frac{m_1m_2}{m_1+m_2}v_\mathrm{rel}^{2},\qquad
\phi=\frac{E_\mathrm{imp}}{m_1Q^\ast},\qquad
m_\mathrm{ej}=\frac{\phi}{1+\phi}\,m_1.
$$

Fragments follow $\mathrm{d}M/\mathrm{d}m\propto m^{-\alpha}$ between
$m_\mathrm{min}=10^{-6}m_\mathrm{max}$ and $m_\mathrm{max}=0.02\,m_\mathrm{ej}$
(`solvers/dust_rates.py:1342-1346`). The normalisation integrates to
$m_\mathrm{ej}$ (`solvers/dust_rates.py:1349-1352`). The fragments are then
shared out as follows:

- Each bin in the group receives the integral over the overlap of its
  $[m_\mathrm{min}, m_\mathrm{max}]$ with the fragment range
  (`solvers/dust_rates.py:1356-1363`).
- If the target bin has `interact_pah`, PAH bins receive the same kind of
  integral over their mass ranges (`solvers/dust_rates.py:1366-1375`).
- Fragments below the smallest tracked mass go back to the gas, split by the
  target's elemental mass fractions (`solvers/dust_rates.py:1379-1388`,
  `solvers/dust_rates.py:1494-1497`).

The fractions are then rescaled (`solvers/dust_rates.py:1391-1396`). Whatever
is left over, $\chi_\mathrm{rem}=1-\sum\chi_\mathrm{frag}-\sum\chi_\mathrm{PAH}-\chi_\mathrm{dest}$,
is the remnant.

:::{note}
The rescaling adds a mass in g (the assigned fragment mass) to a dimensionless
leftover fraction, so the fractions do not sum to one in general. When the
bins do not fully cover the fragment range, $\sum\chi$ collapses to roughly
$m_\mathrm{ej}$ in g, and $\chi_\mathrm{rem}\approx1$.

We checked this with `all_processes_test.json`: a 0.1 µm graphite
self-collision gives $\sum\chi_\mathrm{frag}\approx10^{-14}$. The rate
coefficient also never uses the ejected fraction $m_\mathrm{ej}/m_1$; the whole
collided mass $\mathcal{R}\rho$ is redistributed.
:::

### `turbulent_shattering_rate` (default)

Each bin collides with itself at the rate

$$
\mathcal{R}_j=\sqrt{\frac{8}{3\pi}}\;4\pi a_j^{2}\,v_\mathrm{rel}(j,j)\,\frac{\rho_j}{m_j}
\quad[\mathrm{s^{-1}}]
$$

(`solvers/dust_rates.py:1453-1457`). It loses $\mathcal{R}_j\rho_j$. The
fragments are shared out as above. The remnant goes to the bin whose mass is
closest to $(1-\chi_\mathrm{rem})\,m_j$ (`solvers/dust_rates.py:1470-1481`).
With the normalisation issue above, that is usually the smallest bin in the
group.

### `turbulent_all_shattering_rate`

This kernel uses the same pair loop, $K_{ij}$ and 1/2 symmetry factor as
`turbulent_all_coagulation_rate` (`solvers/dust_rates.py:1556-1564`), with no
sticking factor. It calls `_compute_shattered_fragments` twice, once with each
bin as the target. The remnant goes back into the target bin itself
(`solvers/dust_rates.py:1577-1579`), not to the bin nearest in mass.

## Relative velocities

The helper `grain_relative_velocity` in `solvers/grain_dynamics.py` returns
$v_\mathrm{rel}$ in cm s$^{-1}$. Any `dust_velocity_model` string other than
the two below returns $\sigma$ unchanged (`solvers/grain_dynamics.py:97`).

**`Ormel2007`** ({cite:t}`OrmelCuzzi2007`, Appendix B, in the formulation of
{cite:t}`KawasakiMachida2023`) adds a Brownian and a turbulent part in
quadrature (`solvers/grain_dynamics.py:205`). The Brownian part is

$$
\Delta V_\mathrm{th}=\sqrt{8k_\mathrm{B}T\,(m_1+m_2)/(m_1m_2)}
$$

(`solvers/grain_dynamics.py:148`). The turbulent part uses these quantities:

- Sound speed $c_s=\sqrt{5k_\mathrm{B}T/(3\mu m_\mathrm{H})}$ and gas thermal speed $v_\mathrm{th}=\sqrt{8/\pi}\,c_s$ (`solvers/grain_dynamics.py:154-156`).
- Coulomb mean free path $\ell=1/(n_\mathrm{H}r_c^{2})$ with $r_c=e^{2}/(k_\mathrm{B}T)$ (`solvers/grain_dynamics.py:159-160`).
- Timescales $\tau_L=L/\sigma$, $\mathrm{Re}=\max(3\sigma L/(c_s\ell),1)$ and $\tau_\eta=\tau_L/\sqrt{\mathrm{Re}}$ (`solvers/grain_dynamics.py:163-166`).
- Epstein stopping times $t_s=s\,a/(\rho_\mathrm{gas}v_\mathrm{th})$, ordered so the target has the larger one (`solvers/grain_dynamics.py:171-176`).
- Stokes numbers $\mathrm{St}=t_s/\tau_L$ (`solvers/grain_dynamics.py:179-181`).

Which of three regimes applies depends on how $t_{s,1}$ compares with
$\tau_\eta$ and $\tau_L$ (`solvers/grain_dynamics.py:184-203`). If $\sigma$ or
$L$ is zero, only the Brownian term is returned.

**`Hirashita2019`** follows {cite:t}`HirashitaAoyama2019`, Appendix C. For
each grain

$$
v=1.1\times10^{5}\,\mathcal{M}^{3/2}\left(\frac{a}{10^{-5}\,\mathrm{cm}}\right)^{1/2}\left(\frac{T}{10^{4}\,\mathrm{K}}\right)^{1/4}n_\mathrm{H}^{-1/4}\left(\frac{s}{3.5}\right)^{1/2}\ \mathrm{cm\,s^{-1}},
$$

with $\mathcal{M}=\sigma/c_s$. The two speeds are combined as
$v_\mathrm{rel}=\sqrt{v_1^{2}+v_2^{2}}$ (`solvers/grain_dynamics.py:222-242`).

## Offline modules versus solver kernels

The solver does not import `models/dust_collisions/`. Those modules are
stand-alone exploration and plotting code: they are not exported to
`model_data/` and are not called by any kernel. They differ from the solver
in several ways:

- **`dust_dynamics.relative_velocity`.** It takes a Mach number and $L$ in pc,
  and fixes the Ormel viscosity through $\mathrm{Re}=\sigma L/8.7\times10^{15}$
  (`models/dust_collisions/dust_dynamics.py:44`). It combines the Hirashita
  speeds as $(v_1+v_2)/2$ (`models/dust_collisions/dust_dynamics.py:71`)
  instead of in quadrature.
- **`dust_shattering.t_shattering_Dubois2024`.** It uses
  $75.73\,\mathrm{Myr}/\mu$ with $\mu=1.4$, i.e. 54.1 Myr
  (`models/dust_collisions/dust_shattering.py:14-22`). That agrees with
  equation {eq}`eq-dubois-sha`.
- **`dust_coagulation.t_coagulation`.** It keeps the density scaling
  $10^{3}/(n_\mathrm{H}\,\mathrm{boost})$, a log-normal clumping boost, and an
  $n_\mathrm{H}\ge10^{2}$ cut (`models/dust_collisions/dust_coagulation.py:15-30`).
  `dubois_coagulation_rate` drops all three (power-law exponents set to 0).
