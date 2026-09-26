(quickstart)=
# Quickstart

Six commands, from nothing to an integrated dust chemistry run.

```bash
# 1. Install, with every optional extra
pip install -e ".[all]"

# 2. See where pyCALIMA will read and write
calima-paths

# 3. Confirm the bundled reference data is present
calima-fetch-data verify

# 4. Generate the lookup tables the solver reads.
#    Minutes at the default 4-dust + 2-PAH configuration.
calima-export

# 5. Integrate one environment for 10 kyr
calima-run example_ic --t_end_Myr 0.01 --output-dir /tmp/calima-check

# 6. Look at what it wrote
ls /tmp/calima-check
```

Step 5 is the end-to-end check: it exercises configuration loading, table
lookup, the ODE integration and the output writer, and it fails loudly if any
generated table is missing or malformed. It writes an `_evolution.txt`
trajectory and an `_evolution.png` summary figure.

If step 4 looks slow, it is: two of the twelve export stages account for about
92% of the total. {doc}`/cli/calima-export` breaks down the cost and shows how
to regenerate only part of it.

## Where to go next

- {doc}`/getting-started/data-locations` — how paths are resolved, and the
  environment variables that override them
- {doc}`/guide/workflows` — the full set of commands and the Python API
- {doc}`/guide/solvers` — the five solvers and when to prefer each
- {doc}`/tutorials/index` — worked notebooks
