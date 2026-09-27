(configuration)=
# Configuration

Two kinds of JSON file drive pyCALIMA. The **grain-size configuration**
defines the grain bins and so what `calima-export` computes. A **solver
configuration** defines one gas cell — its environment, initial dust and gas
content, and which processes are active — for `calima-run` and `calima-grid`.

## Grain-size configuration

The default is `models/grain_size_distribution.json`; the other shipped
configurations sit next to it (`calima-paths` lists them). Pass a different one
with `--config` to any exporter, or call
{func}`~pycalima.models.grain_size_config.set_config_path` before importing any
other model module. {mod}`pycalima.models.grain_size_config` is the only
loader.

| Key | Contents |
|---|---|
| `bins` | One entry per bin: `id`, `composition` (`graphite` or `silicate`), `is_pah`, and `bin_rank` (the bin's position, smallest first, within its composition) |
| `basic` | Size-distribution parameters, one list entry per bin: `a0` (characteristic size), `amin`, `amax` (size range, µm), `sigma` (log-normal width) and `s` (material density, g cm⁻³) |
| `shattering` | The same parameters for the shattering-adapted distribution |
| `export_parameters` | Sampling grids for each exporter (temperature ranges, number of points, …), keyed by export stage |
| `model_name` | Optional. If present, generated tables go to `model_data/<model_name>/` |

After editing it, rerun `calima-export` so the tables match the bins.

## Solver configuration

The shipped solver configurations are in `solvers/configs/`; pass one by name
(`calima-run example_ic`) or by path. Every key below is read by
{func}`~pycalima.solvers.dust_init.load_initial_conditions`; a missing key
takes the default shown.

### `environment`

| Key | Default | Meaning |
|---|---|---|
| `gas_temperature_K` | required | Gas temperature [K] |
| `hydrogen_number_density_cm3` | required | $n_\mathrm{H}$ [cm⁻³] |
| `electron_number_density_cm3` | $10^{-4}\,n_\mathrm{H}$ | $n_e$ [cm⁻³] |
| `radiation_field_G0` | 1 | FUV field in Habing units |
| `radiation_field_model` | `habing` | Label of the field shape |
| `mean_molecular_weight` | 1.4 | Mass per H nucleus, in $m_\mathrm{H}$ |
| `gas_mass_density_gcm3` | $n_\mathrm{H}\,m_\mathrm{H}\,\mu$ | Gas mass density [g cm⁻³] |

### `elemental_abundances`

Mass fraction of each element in the gas, e.g. `"C": {"mass_fraction":
1.66e-3}`. These set the initial gas-phase element densities; the elements
locked in dust come on top of them through the bins' initial densities.

### `dust_bins`

| Key | Default | Meaning |
|---|---|---|
| `id` | required | Bin identifier, e.g. `DustBin_01`; generated tables are looked up by it |
| `grain_size_micron` | required | Representative grain radius [µm] |
| `grain_density_gcm3` | required | Material density [g cm⁻³] |
| `composition` | `graphite` | `graphite` or `silicate` |
| `elements` | `[]` | Element mass fractions of the grain material, e.g. `[{"name": "C", "mass_fraction": 1.0}]` |
| `initial_mass_density_gcm3` | 0 | Initial dust mass density in this bin [g cm⁻³] |
| `sticking_coefficient` | 1 | Accretion sticking coefficient |
| `nhmax_acc` | $10^4$ | Density above which accretion onto this bin stops [cm⁻³] |
| `nh_coa` | 0.1 | Minimum $n_\mathrm{H}$ for coagulation [cm⁻³] |
| `coagulation_partner` | — | `id` of the bin this one coagulates into |
| `amin_micron`, `amax_micron` | $a/2$, $2a$ | Size range of the bin, for the fragment distribution [µm] |
| `catastrophic_specific_energy_erg_g` | $10^7$ | Catastrophic specific energy $Q^*$ for shattering [erg g⁻¹] |
| `interact_pah` | `false` | Whether shattering fragments of this bin can populate PAH bins |
| `vthresh_coag_cm_s` | $10^4$ | Coagulation threshold velocity for self-collisions [cm s⁻¹] |

### `pah_bins`

| Key | Default | Meaning |
|---|---|---|
| `id` | required | Bin identifier, e.g. `PAHBin_01` |
| `nc` | 54 | Number of carbon atoms of the representative PAH |
| `nc_min`, `nc_max` | 24, $2\,n_C$ | Carbon-number range of the bin |
| `mpah_g` | $n_C \times 12.011$ u | PAH mass [g] |
| `grain_density_gcm3` | 2.24 | Material density [g cm⁻³] |
| `initial_mass_density_gcm3` | 0 | Initial PAH mass density [g cm⁻³] |
| `fpah` | 0.1 | Fraction of carbon locked in PAHs, used for the initial conditions when no initial density is given |
| `is_cluster` | `false` | Whether the bin represents PAH clusters rather than monomers |
| `dust_index_interact`, `nd_bins_interact` | −1, 0 | First dust bin and number of dust bins this PAH bin freezes onto |

### `physics`

One boolean per process, all `false` by default: `dust_accretion`,
`dust_sputtering`, `dust_sublimation`, `dust_coagulation`, `dust_shattering`,
`pah_accretion`, `pah_photolysis`, `pah_sputtering`, `pah_coalescence`,
`pah_cluster_evaporation`, `pah_freezing`. {doc}`/physics/index` describes each.

### `models`

Variant selectors for the active processes: `dust_sputtering_model`,
`coagulation_model`, `shattering_model`, `dust_velocity_model`,
`coalescence_model`, `photolysis_model`, `pah_sputtering_model`,
`cluster_evaporation_model`, plus `slope_frag_func` (power-law slope of the
fragment mass distribution, default 1.3/3). The table in {doc}`/physics/index`
lists the values each accepts and the defaults.

### `turbulence`

| Key | Default | Meaning |
|---|---|---|
| `local_sigma_km_s` | 0 | One-dimensional turbulent velocity dispersion [km s⁻¹] |
| `local_dx_pc` | 0 | Cell size [pc], used for the Jeans-length criteria; 0 means unknown |

### `solver`

`type` selects the integrator (`rk4`, `rk54`, `anninos`, `newton_krylov`,
`sparse_newton`); the other keys — `t_end_Myr`, step-size limits, tolerances
and iteration caps — depend on it. {doc}`solvers` documents each solver and its
keys.

### `model_data_dir`

Optional. Where to read the generated tables from; missing, `null` or `"auto"`
means the resolved `model_data/` ({doc}`/getting-started/data-locations`). A
relative path is resolved against the configuration file, not the working
directory.
