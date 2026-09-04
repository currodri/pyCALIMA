(data-locations)=
# Data locations


Reference data ships inside the package and is read-only. Generated tables and
run output go somewhere writable, resolved in this order:

1. `$CALIMA_DATA` — a writable **root**; `model_data/`, `results/` and
   `datasets/` are created underneath it.
2. `./model_data` if it already exists, or the current directory if it is a
   pyCALIMA source checkout.
3. A per-user data directory (`platformdirs`).

Finer-grained overrides, each beating `$CALIMA_DATA`: `$CALIMA_MODEL_DATA`,
`$CALIMA_RESULTS`, `$CALIMA_DATASETS`. Also `$CALIMA_BUNDLED_DATA` to point an
installed copy at a checkout's reference data, and `$CALIMA_CONFIG` for the
default grain-size configuration.

To see every resolved location for your environment:

```bash
calima-paths
```

Four external data sources come from other projects and are not redistributed
here; set the corresponding variable if you need the routines that read them:

| Variable | Points at | Used by |
|---|---|---|
| `$CALIMA_SED_DIR` | BPASS SED tables (from Dusty-PRISM) | dust/PAH photoelectric heating with stellar SEDs |
| `$CALIMA_DUSTEM_FILE` | a DustEM heat-capacity table (`hcap/C_PAH0_DL07.DAT`) | the f(T) cross-check in `diagnose_temperature_distribution` |
| `$CALIMA_YIELD_DIR` | raw stellar-yield tables (Karakas, Limongi & Chieffi, Kobayashi) | `models/yields/build_tables.py` |
| `$CALIMA_SIM_DIR` | your own RAMSES snapshot directory | the two RAMSES post-processing notebooks — see {doc}`/guide/post-processing` |

None of these are bundled or downloadable; each accessor raises with the
variable's name if it is needed and unset.

The large PAHdb archives are registered but not bundled:

```bash
calima-fetch-data list                  # what is available and where
calima-fetch-data list --missing
calima-fetch-data import pahdb-theoretical-v4-00 /path/to/pahdb-...xml
```
