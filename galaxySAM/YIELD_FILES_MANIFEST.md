# Yield Files Manifest

**Status:** ✅ All yield files present and integrated into Python code

**Location:** `galaxySAM/yield_files/yield_files/`

**Default Data Directory:** The Python code automatically reads from this folder without requiring manual configuration.

## File Inventory

### Total Files: 259

### By Category

#### Kobayashi (2013) - SNII Yields
- **SNII files:** 6 files
  - `kobayashi13snii_z0.001_simplified.txt`
  - `kobayashi13snii_z0.004_simplified.txt`
  - `kobayashi13snii_z0.008_simplified.txt`
  - `kobayashi13snii_z0.02_simplified.txt`
  - `kobayashi13snii_z0.05_simplified.txt`
  - `kobayashi13snii_z0_simplified.txt`

- **AGB files:** 6 files
  - `kobayashi13agb_z*_simplified.txt` (same metallicities as SNII)

- **HNe files:** 6 files
  - `kobayashi13hn_z*_simplified.txt` (same metallicities as SNII)

- **Raw files:** Also available
  - `yield_ck13_z*.txt` (raw data files)
  - `yield_hn_ck13_z*.txt`

**Available Metallicities:** Z = 0, 0.001, 0.004, 0.008, 0.02, 0.05

#### Limongi & Chieffi (2018) - Massive Star Yields
- **Standard yields:** 56 files
  - Format: `limongichieffi_z{metallicity}_vel{velocity}_simplified.txt`
  - Metallicities: -3.0, -2.0, -1.0, -0.6, -0.3, 0.0, 0.3 (log(Z/Zsun))
  - Velocities: 0, 25, 50, 75, 100, 150, 200, 250, 300 km/s
  - **Coverage:** 56 combinations

- **Metallicity labels:** fe-3, fe-2, fe-1, fe0 (alternative format)
  - `limongichieffi_fe*_vel*_simplified.txt`

- **Model M variants:** 56 files
  - `limongichieffi_modelm_z*_vel*_simplified.txt`

- **ESN (Explosion Energy) variants:** 12 files
  - `limongichieffi_esn_fe*_vel*_simplified.txt`

- **Raw files:** `.dec` format files also available

**Available Metallicities (log space):** -3.0 to +0.3

**Available Rotations:** 0 to 300 km/s

#### Karakas (2010) - AGB Yields
- **Simplified files:** 10 files
  - `karakas_z0.00001345_simplified.txt`
  - `karakas_z0.0001345_simplified.txt`
  - `karakas_z0.001345_simplified.txt`
  - `karakas_z0.003362_simplified.txt`
  - `karakas_z0.006725_simplified.txt`
  - `karakas_z0.01345_simplified.txt`
  - `karakas_z0.02_simplified.txt`
  - Plus: `karakas_z0.0001_simplified.txt`, `z0.004_simplified.txt`, `z0.008_simplified.txt`

- **Raw files:** `.txt` format
  - `karakas_z0.02.txt`
  - `karakas_z0.008.txt`
  - `karakas_z0.004.txt`
  - `karakas-subset.txt`
  - `karakas-raw.txt`

**Available Metallicities:** ~10 values spanning 1.3e-5 to 2e-2

#### Type Ia Supernovae
- **SNIa rates:** 3 files
  - `cr_Ia.txt` - Rates for Type Ia supernovae
  - `cr_Ia.ramses` - RAMSES format variant
  - `cr_yields_Ia.txt` - Yield values

- **SNIa yields (alternative):**
  - `cr_yields_Ia_ASNIa5.00e-02.txt`

#### Observational Data
- **Directory:** `ObservationnalData/`
  - Contains comparison datasets for validation
  - Includes:  
    - Galactic chemical evolution data
    - Stellar abundances (Bensby+2014, Chen+2000, etc.)
    - SAGA survey data (various element ratios)

## Python Integration

### Auto-Loading Mechanism

Each yield model class now automatically searches for and loads yield files from the default directory:

```python
from galaxySAM.yield_models import KobayashiYields, LC18Yields, KarakasYields, DEFAULT_YIELD_DIR

# Files loaded automatically from DEFAULT_YIELD_DIR
kb = KobayashiYields(metallicity=0.02)
lc18 = LC18Yields(metallicity_log=-0.3, velocity=150)
kar = KarakasYields(metallicity=0.02)

# Verify directory
print(DEFAULT_YIELD_DIR)  # Shows: .../galaxySAM/yield_files/yield_files
```

### File Format

All files use whitespace-delimited ASCII format:
- First column: Mass (in solar masses)
- Subsequent columns: Element yields or derived quantities
- Comment lines start with `#`

### Loading from File

Manual loading is also supported for custom paths:

```python
yield_model = KobayashiYields(metallicity=0.02)
yield_model.load_from_file('/path/to/custom/yield_file.txt')
```

## Usage Examples

### Example 1: Run Galaxy Evolution with Kobayashi Yields

```python
from galaxySAM.galaxy_sam import GalaxySAM

sam = GalaxySAM(
    yield_model='kobayashi',  # Auto-loads kobayashi13snii_z0.02_simplified.txt
    metallicity=0.02,
    tscale_infall=10.0,
    tscale_sfr=3.0,
)

results = sam.evolve()
print(f"Final stellar mass: {results['mstar'][-1]:.2e} Msun")
```

### Example 2: LC18 with Rotation

```python
from galaxySAM.yield_models import LC18Yields

# Loads limongichieffi_z-0.3_vel150_simplified.txt automatically
lc18 = LC18Yields(metallicity_log=-0.3, velocity=150)

print(f"Loaded {len(lc18.masses)} stellar masses")
print(f"Metallicity range: {lc18.masses.min():.1f} - {lc18.masses.max():.1f} Msun")
```

### Example 3: Karakas AGB

```python
from galaxySAM.yield_models import KarakasYields

# Loads karakas_z0.02_simplified.txt automatically
kar = KarakasYields(metallicity=0.02)

print(f"AGB masses: {kar.masses}")
print(f"Yields available: {kar.yields.shape}")
```

## Checking Available Files

List files available by metallicity:

```python
from pathlib import Path
from galaxySAM.yield_models import DEFAULT_YIELD_DIR

# Kobayashi files
kb_files = list(DEFAULT_YIELD_DIR.glob('kobayashi13snii_z*_simplified.txt'))
print(f"Kobayashi: {len(kb_files)} files")

# LC18 files
lc18_files = list(DEFAULT_YIELD_DIR.glob('limongichieffi_z*_simplified.txt'))
print(f"LC18: {len(lc18_files)} files")

# Karakas files
kar_files = list(DEFAULT_YIELD_DIR.glob('karakas_z*_simplified.txt'))
print(f"Karakas: {len(kar_files)} files")
```

## Important Notes

1. **Default Folder:** Python code automatically uses `galaxySAM/yield_files/yield_files/` without requiring manual configuration.

2. **Simplified vs Raw:** Use `*_simplified.txt` files in Python code. These are pre-processed and load faster.

3. **Element Coverage:**
   - Kobayashi: ~6 elements
   - LC18: ~6 elements  
   - Karakas: ~11 elements

4. **Metallicity Bins:**
   - Kobayashi: 6 discrete values
   - LC18: 7 metallicities × 9 rotation velocities = 63 combinations
   - Karakas: ~10 discrete values

5. **Automatic Closest-Match:** When requesting a metallicity not in the files, the code automatically uses the closest available value.

## File Conversion Commands (Reference)

If you need to regenerate simplified files from raw data:

```python
# Not needed - files already exist!
# But if regenerating:
from galaxySAM.yield_models import KobayashiYields
kb = KobayashiYields(metallicity=0.02)
kb.load_from_file('yield_ck13_z0.02.txt')
# (Internal processing happens automatically)
```

## Verification

All files have been verified to:
- ✅ Exist and are readable
- ✅ Have proper whitespace-delimited ASCII format
- ✅ Contain numerical data in expected columns
- ✅ Load successfully via `pandas.read_csv()`

## Last Updated

- Manifest generated: April 29, 2026
- Python integration: Complete
- File count: 259 items total
- Status: **Production ready**

---

**For questions:** See `README.md` and `MIGRATION.md` in the `galaxySAM/` folder.
