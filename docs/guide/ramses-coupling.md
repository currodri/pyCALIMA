(ramses-coupling)=
# The RAMSES-side Fortran module

pyCALIMA's `solvers/` package mirrors a Fortran implementation that runs inside
RAMSES. The Fortran source ships with the package under
`solvers/ramses_source/` for reference; the document below is its own layout
guide, describing how the modules fit together on the simulation side.

The Python solver reads the same precomputed tables, which is what makes the
two comparable — see {doc}`/guide/solvers`.

```{include} ../../src/pycalima/solvers/ramses_source/README.md
:start-line: 1
```
