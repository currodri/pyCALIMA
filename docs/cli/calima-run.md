# `calima-run`

Integrates the dust and PAH chemistry for a single environment, given an
initial-conditions JSON file. The `config` argument accepts a path or the bare
name of a bundled configuration (`example_ic`), so `calima-run example_ic`
works from any directory.

Five solvers are available via `--solver`; {ref}`solver-types` describes what
each is for and when to prefer it.

```{eval-rst}
.. argparse::
   :module: pycalima.solvers.run_chemistry
   :func: _build_parser
   :prog: calima-run
```
