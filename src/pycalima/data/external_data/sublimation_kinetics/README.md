# Sublimation kinetics of dust materials

Read by `pycalima.models.dust_radiation.sublimation_kinetics`.

| File | Content |
|---|---|
| `sublimation_models.toml` | Curated datasets per material (`fiducial`, `GD89`, alternatives). **Generated**: do not edit by hand. |
| `make_sublimation_models.py` | Builds the TOML from `compilation.csv`. The `SELECTION` table in it records which rows make each dataset and why. |
| `compilation.csv` | Literature compilation: 107 rows (one per material and source), 32 materials. Vapour-pressure fits in the original form and converted to `log10 P[dyn cm^-2] = A10 - B10/T`, evaporation coefficients with their conditions, vapour species, congruency, unit masses and densities, GD89-form quantities, surface energies, condensation temperatures and a verification tag. |
| `COMPILATION_NOTES.md` | Conventions, every conversion, the mapping onto the GD89 framework (with the checks against GD89 graphite and silicate), which materials have laboratory kinetics, the evidence for incongruent Fe release, per-material caveats and the reference list. |
| (provenance) | The conversion and fitting scripts are in `diagnostics/dust_radiation/sublimation_compilation_provenance/`. |

To change a dataset, edit `SELECTION` in `make_sublimation_models.py` and
rerun it. `tests/test_models_physics.py::TestSublimationKinetics` checks that
the bundled TOML is what the builder writes.

Main points from the compilation:

- **Graphite.** GD89 (C1 only) is within a factor of 1.4–3 of the
  multi-species laboratory rate at 2000–2500 K, and increasingly too slow above.
- **Silicate.** GD89 is 7–40 times slower than free evaporation of forsterite
  into vacuum at 1500–2000 K. Its binding energy is 68,100 K, against
  62,000–65,300 K measured, because GD89 assumed a vapour in equilibrium with
  cosmic O abundances.
- **Fe-bearing silicates and FeS evaporate incongruently.** Olivine loses Fe
  first; FeS loses S and leaves metallic Fe.
- **No data** for amorphous carbon (graphite is the proxy), amorphous
  silicates or the intermediate Mg–Fe compositions.
- **Surface energies** from different sources disagree by up to a factor of 10.
