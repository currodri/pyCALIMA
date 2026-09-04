# `calima-export`

Regenerates every precomputed lookup table under the generated-data directory,
from the active grain-size configuration. The chemistry solver reads these
tables, so this is the step that has to run after any change to the bin
configuration or to the underlying physics.

## Stages and what they cost

A full export runs twelve stages. Two of them dominate: the table below is a
measured profile of the 25-bin `5PAH10C10Sil` configuration (20 dust + 5 PAH),
recorded in that model's `export_profile.json`.

| Stage | Writes | Seconds | Share |
|---|---|---:|---:|
| `dust_photoelectric_heating` | `dust_photoelectric_heating_data/` | 955.8 | 47.0% |
| `dust_ion_recombination` | `dust_ion_recombination_data/` | 925.5 | 45.5% |
| `sputtering_rates` | `thermal_sputtering_data/` | 64.6 | 3.2% |
| `collisional_cooling` | `collisional_cooling_data/` | 43.0 | 2.1% |
| `dust_charging` | `dust_charging_data/` | 22.6 | 1.1% |
| `dust_optical_properties` | `optical_properties/` | 5.7 | 0.3% |
| `pah_photoelectric_heating_tables` | `PAH_photoelectric_heating_data/` | 5.6 | 0.3% |
| `dust_sublimation` | `dust_sublimation/` | 4.8 | 0.2% |
| `pah_sputtering_rates` | `pah_sputtering_data/` | 3.7 | 0.2% |
| `pah_optical_properties` | `optical_properties/` | 2.7 | 0.1% |
| `pah_dissociation_tables` | `PAH_dissociation_data/` | 1.2 | 0.1% |
| `dust_band_luminosities` | `optical_properties/` | 0.0 | 0.0% |
| **total** | | **2035.3** | **33.9 min** |

Both dominant stages scale per dust bin, so the default configuration — 4 dust
and 2 PAH bins — is very much faster than this. Use `--stages` or
`--skip-stages` to regenerate only what you need:

```bash
# Everything (~34 min at 25 bins, minutes at the default 6)
calima-export

# Skip the single most expensive stage that nothing else reuses
calima-export --skip-stages dust_ion_recombination

# Only the PAH tables
calima-export --stages pah_optical_properties,pah_sputtering_rates,pah_dissociation_tables

calima-export --list-stages
```

:::{note}
`dust_charging` reuses the equilibrium charge solves already performed by
`dust_photoelectric_heating` — which is why it costs 22.6 s rather than
minutes. A selection that keeps `dust_charging` but drops
`dust_photoelectric_heating` is rejected rather than silently recomputing.
:::

Each stage is also runnable on its own, which is useful when iterating on one
piece of physics:

```bash
python -m pycalima.models.dust_gas_collisions.export_sputtering_rates_bins
```

Note that a standalone `export_dust_charging_vs_gamma` does not get the reuse
above and will re-solve from scratch.

```{eval-rst}
.. argparse::
   :module: pycalima.models.export_all_grain_data
   :func: _build_parser
   :prog: calima-export
```
