# Provenance of the sublimation kinetics compilation

These are the scripts that produced the converted and derived rows of
`src/pycalima/data/external_data/sublimation_kinetics/compilation.csv`
(see `COMPILATION_NOTES.md` there, section 2):

- `conv.py`: converts the Nozawa et al. (2003) Table 2 Gibbs-energy fits to
  effective vapour pressures per formula unit;
- `parse_w18.py`, `freeevap.py`, `runfits.py`: parse the Woitke et al. (2018)
  tables, solve the congruent free-evaporation vapour over each solid, and fit
  log10 P_eff = A10 - B10/T;
- `build_base.py`, `build.py`: assemble the CSV.

They are a record, not a pipeline you can rerun as is: `freeevap.py` and
`parse_w18.py` read a `pdftotext -layout` dump of arXiv:1712.01010 and an
intermediate molecule table from the working directory in which the
compilation was made, which are not part of the repository. The JSON files
are their intermediate outputs.
