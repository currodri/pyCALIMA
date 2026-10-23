# `calima-export`

Regenerates every precomputed lookup table under the generated-data directory,
from the active grain-size configuration. The chemistry solver reads these
tables, so this is the step that has to run after any change to the bin
configuration or to the underlying physics.

## Stages and what they cost

A full export runs eleven stages. The table below is a measured profile of the
25-bin `5PAH10C10Sil` configuration (20 dust + 5 PAH), recorded in that model's
`export_profile.json`, less the stages since removed; the total and the shares
are of the stages listed, from that same measurement. It predates the two
charging stages for RAMSES, `dust_charging_rtgroups` (RT-group tables, a few
seconds per bin) and `dust_charging_isrf` (uniform-ISRF tables built from them,
about half a minute per dust bin). Of the stages measured, two take most of the
time:

| Stage | Writes | Seconds | Share |
|---|---|---:|---:|
| `sputtering_rates` | `thermal_sputtering_data/` | 64.6 | 49.2% |
| `collisional_cooling` | `collisional_cooling_data/` | 43.0 | 32.7% |
| `dust_optical_properties` | `optical_properties/` | 5.7 | 4.4% |
| `pah_photoelectric_heating_tables` | `PAH_photoelectric_heating_data/` | 5.6 | 4.3% |
| `dust_sublimation` | `dust_sublimation/` | 4.8 | 3.6% |
| `pah_sputtering_rates` | `pah_sputtering_data/` | 3.7 | 2.8% |
| `pah_optical_properties` | `optical_properties/` | 2.7 | 2.0% |
| `pah_dissociation_tables` | `PAH_dissociation_data/` | 1.2 | 0.9% |
| `dust_band_luminosities` | `optical_properties/` | 0.0 | 0.0% |
| **total** | | **131.3** | **2.2 min** |

It scales per dust bin, so the default configuration — 4 dust and 2 PAH bins —
is very much faster than this. Use `--stages` or `--skip-stages` to regenerate
only what you need:

```bash
# Everything (roughly a quarter of an hour at 25 bins, minutes at the default 6)
calima-export

# Skip a stage whose tables you do not need
calima-export --skip-stages dust_band_luminosities

# Only the PAH tables
calima-export --stages pah_optical_properties,pah_sputtering_rates,pah_dissociation_tables

calima-export --list-stages
```

:::{note}
`dust_charging_isrf` reads the tables that `dust_charging_rtgroups` writes, so
it runs after it; selected on its own, it uses the RT-group tables already on
disk.
:::

Each stage is also runnable on its own, which is useful when iterating on one
piece of physics:

```bash
python -m pycalima.models.dust_gas_collisions.export_sputtering_rates_bins
```

```{eval-rst}
.. argparse::
   :module: pycalima.models.export_all_grain_data
   :func: _build_parser
   :prog: calima-export
```
