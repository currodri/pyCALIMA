# Berne 2022 PAH Cross-Section Data

Photoabsorption cross-section data for PAH molecules in four charge states,
used by `absorption_cross_section_Berne()` in
`models/PAH_charge/PAH_photoelectric_heating.py` when `optical_model='Malloci'`
is selected.

Source: Berne et al. (2022) — cross sections computed with the Malloci TD-DFT
database.

## Required directory structure

```
berne_2022/
├── anions/          # PAH anion (Z = -1) cross sections
├── cations/         # PAH cation (Z = +1) cross sections
├── dications/       # PAH dication (Z = +2) cross sections
└── neutrals/        # PAH neutral (Z = 0) cross sections  ← must be present
```

All four subdirectories must be present. Runs will fail with a `FileNotFoundError`
if `neutrals/` is missing.

## Files per subdirectory

Each subdirectory contains one `.txt` file per PAH species (two columns:
photon energy [eV], cross section [Mb]):

| Filename stem | Species | Nc |
|---|---|---|
| `ovalene` | Ovalene | 32 |
| `tetrabenzocoronene` | Tetrabenzocoronene | 36 |
| `circumbiphenyl` | Circumbiphenyl | 38 |
| `circumanthracene` | Circumanthracene | 40 |
| `circumpyrene` | Circumpyrene | 42 |
| `hexabenzocoronene` | Hexabenzocoronene | 42 |
| `dicoronylene` | Dicoronylene | 48 |
| `circumcoronene` | Circumcoronene | 54 |
| `circumovalene` | Circumovalene | 66 |

The `anions/` directory also includes `coronene_anion.txt` (C24), used only for
the anion average.

## Overriding the data path

By default the code looks for this directory at `optical_props/berne_2022/`
relative to the repository root. Set the `BERNEPATH` environment variable to
point to a different installation:

```bash
export BERNEPATH=/path/to/your/berne_2022
```
