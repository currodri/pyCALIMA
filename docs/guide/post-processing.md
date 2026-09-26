(post-processing)=
# Post-processing RAMSES outputs


`notebooks/CALIMA_model_explorer.ipynb` is the reference for the
post-processing tools: it loads RAMSES snapshots cell by cell, runs the
pyCALIMA solvers over the same conditions, and quantifies how well each solver
and initial-condition prescription reproduces the simulated dust.

> **You must supply the simulation outputs.** RAMSES snapshots are not
> distributed with pyCALIMA — a single one is many GB. Nothing in the package,
> the wheel or the repository contains them, and there is no download that will
> fetch them. The notebook stops immediately with an actionable error if they
> are not configured.

## What you need

1. **`yt`**, for reading RAMSES output:

   ```bash
   pip install -e ".[sim]"
   ```

2. **Your own RAMSES snapshots**, from a build with the CALIMA dust module.
   The notebook reads the dust-bin scalars and `hydro_scalar_14` (turbulent
   velocity dispersion), so a stock RAMSES run will not have the fields it
   expects.

3. **Generated tables**, since the notebook runs the solvers:
   `calima-export`, or point `$CALIMA_MODEL_DATA` at an existing tree.

## Pointing it at your outputs

Set `$CALIMA_SIM_DIR` to the directory holding your snapshots:

```bash
export CALIMA_SIM_DIR=/path/to/your/ramses/outputs
```

then edit `SIM_SUBDIRS` in §1 of the notebook so the sub-paths match your run
names. As shipped it expects three galaxies at a common epoch:

```python
SIM_SUBDIRS = {
    'G8':  'CorrectSN_0.03Zsun_DTMini1d-3_MR/output_00081',
    'G9':  'CorrectSN_0.1Zsun_DTMini1d-3/output_00081',
    'G10': 'CorrectSN_MR/output_00081',
}
```

Three constants in §0 and §1 describe the snapshots rather than the code, so
they must be changed to match your own:

| Constant | Shipped value | Meaning |
|---|---|---|
| `T_SIM_MYR` | `399.6` | simulation age at the snapshot [Myr]; also the solver integration time |
| `UNIT_V_KMS` | `65.59` | RAMSES code velocity unit in km/s, used to scale `hydro_scalar_14` |
| `Z_ZSUN_SIM` | `{'G8': 0.03, 'G9': 0.1, 'G10': 0.3}` | nominal metallicities, for plot labels only — the true Z is read from the data |
| `SPHERE_KPC` | `{'G8': 5, 'G9': 10, 'G10': 15}` | analysis sphere radius around each galaxy centre |

`CONFIG_PATHS` maps each galaxy to a bundled solver config
(`ramses_G8_0.03Zsun`, `ramses_G8_0.1Zsun`, `ramses_G10_MWZsun`), so those
resolve automatically and need no editing.

If `$CALIMA_SIM_DIR` is unset, §1 raises:

```
This notebook post-processes RAMSES outputs, which are not shipped with pyCALIMA.
Set $CALIMA_SIM_DIR to the directory containing your snapshots:
    export CALIMA_SIM_DIR=/path/to/your/ramses/outputs
then adjust SIM_SUBDIRS below to match your run names.
```

and if the sub-paths do not resolve it lists exactly which ones were missing.

## What the notebook does

| Section | Content |
|---|---|
| §0 | Setup, unit constants, output directory (`$CALIMA_DATA/results/comparison`) |
| §1 | Loads each snapshot through `yt`, registers derived fields (`nH`, `T_gas`, `sigma_turb_kms`, per-bin dust densities, DTM/DTG projection weights) and extracts per-cell arrays |
| §2–§3 | ISM property histograms and T–nH phase diagrams per galaxy |
| §4 | Initial-condition prescriptions and the 4-D (Z, σ, T, nH) grid setup |
| §4b–§4d | Fits *learned* IC prescriptions from the outputs: `DTM(nH, Z, T)`, `f_small(nH, Z)`, `f_carb(nH, Z)`, with residual diagnostics |
| §5, §7 | Runs pyCALIMA equilibrium and time-integrated grids, cached as `grid_<key>.npz` so re-runs are cheap |
| §6, §8 | Interpolation and bias helpers; ranking of every solver / IC combination |
| §9–§10 | `f_small` and `f_carb` behaviour by density regime |
| §11 | The recommended configuration (Ann-tff with learned IC), including Newton–Krylov comparisons, bias vs `nH`, a phase map of the bias, and `yt` spatial projections |
| §12 | Summary and recommendations |

Figures and `.npz` caches are written to `get_results_dir('comparison')` — run
`calima-paths` to see where that resolves. Nothing is written into the
installed package or into your simulation directories.

`notebooks/ramses_equilibrium_tutorial.ipynb` is the shorter companion: a
single galaxy, one equilibrium grid, and the same `$CALIMA_SIM_DIR`
requirement.
