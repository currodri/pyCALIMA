# `calima-grid`

Runs `calima-run`'s integration over a 2-D parameter grid, optionally in
parallel, and saves the result as a compressed `.npz`.

`--t-end-mode` is worth knowing about: rather than integrating every grid point
for the same wall-clock duration, it can scale the end time per cell to a local
physical timescale, so that each point is evolved for a comparable number of
dynamical times.

```{eval-rst}
.. argparse::
   :module: pycalima.solvers.run_grid
   :func: _build_parser
   :prog: calima-grid
```
