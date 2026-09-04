# Validation Checklist for Python Galaxy SAM

Use this checklist to validate that the Python implementation works correctly
and produces consistent results with the original IDL code.

## Pre-Flight Checks

### Environment Setup
- [ ] Python 3.8+ installed
- [ ] NumPy, SciPy, Pandas, Matplotlib installed
- [ ] CALIMA repository cloned/accessible
- [ ] No import errors when running: `python -c "import galaxySAM"`

### File Verification
- [ ] All 12 Python modules exist in `galaxySAM/`
  - [ ] `__init__.py`
  - [ ] `constants.py`
  - [ ] `imf.py`
  - [ ] `yield_models.py`
  - [ ] `sn1a.py`
  - [ ] `galaxy_sam.py`
  - [ ] `plotting.py`
  - [ ] `run_sam.py`
  - [ ] `examples.py`
  - [ ] `README.md`
  - [ ] `MIGRATION.md`
  - [ ] `SUMMARY.md`

- [ ] Original IDL code preserved in `galaxySAM/yohan_routines/`

## Basic Functionality Tests

### 1. Constants Module
```python
from galaxySAM import constants
assert constants.HUBBLE_TIME == 13.5
assert constants.ZSUN_ASPLUND == 0.01345
assert len(constants.ELEMENTS_LC18) == 11
```
- [ ] Constants load without error
- [ ] Values match IDL code
- [ ] All element lists present

### 2. IMF Module
```python
from galaxySAM.imf import create_imf, SalpeterIMF, ChabrierIMF
sal = SalpeterIMF()
chab = create_imf('chabrier')
masses = [0.5, 1.0, 10.0, 100.0]
phi_sal = sal(masses)
phi_chab = chab(masses)
assert all(phi_sal > 0)
assert all(phi_chab > 0)
```
- [ ] Salpeter IMF creates without error
- [ ] Chabrier IMF creates without error
- [ ] IMF values are positive
- [ ] Normalization works correctly

### 3. Yield Models Module
```python
from galaxySAM.yield_models import KobayashiYields, create_yield_model
yields = KobayashiYields(metallicity=0.02)
assert yields is not None
yields2 = create_yield_model('kobayashi', metallicity=0.02)
assert yields2 is not None
```
- [ ] KobayashiYields instantiates
- [ ] Factory function works
- [ ] Metadata available

### 4. SNIa Module
```python
from galaxySAM.sn1a import SNIaModel
snia = SNIaModel(asnia=0.05)
tau_1 = snia.tau_m_padova(10.0)
tau_2 = snia.tau_m_simple(15.0)
assert tau_1 > 0
assert tau_2 > 0
```
- [ ] SNIa model instantiates
- [ ] Lifetime calculations work
- [ ] Return positive ages

### 5. Galaxy SAM Module
```python
from galaxySAM.galaxy_sam import GalaxySAM
sam = GalaxySAM(yield_model='kobayashi', metallicity=0.02)
assert sam is not None
```
- [ ] SAM instantiates without error
- [ ] Parameters are set correctly
- [ ] Time grid is created

### 6. Plotting Module
```python
from galaxySAM.plotting import YieldPlotter, EvolutionPlotter
yp = YieldPlotter()
ep = EvolutionPlotter()
assert yp is not None
assert ep is not None
```
- [ ] Plotters instantiate
- [ ] No import errors

## Evolution Tests

### Test 1: Single Evolution (Solar Metallicity)
```bash
python -m galaxySAM.run_sam \
    --yield-model kobayashi \
    --metallicity 0.02 \
    --nbint 100 \
    --save-evolution \
    -o test_run1
```

Expected results:
- [ ] Completes without error
- [ ] Creates `test_run1/evolution_data.txt`
- [ ] Final stellar mass > 0
- [ ] Final gas mass < initial value
- [ ] Final metallicity > 0

Check output file:
```bash
head -5 test_run1/evolution_data.txt
tail -5 test_run1/evolution_data.txt
```

- [ ] 7 columns (time_gyr, mgas_msun, mstar_msun, metals_msun, metallicity, sfr_msun_per_yr, ...)
- [ ] Time increases monotonically
- [ ] Masses are positive and finite
- [ ] Metallicity increases (or stays constant)
- [ ] SFR is non-negative

### Test 2: Multiple Models Comparison
```bash
python -m galaxySAM.examples
```

Expected:
- [ ] Completes without error
- [ ] Creates PNG files:
  - [ ] `example_model_comparison.png`
  - [ ] `example_parameter_study.png`
  - [ ] `example_multi_metallicity.png`
- [ ] All plots are readable
- [ ] No NaN or Inf values in plots

### Test 3: Parameter Variations
```python
from galaxySAM.galaxy_sam import GalaxySAM

for tscale in [3.0, 7.0, 15.0]:
    sam = GalaxySAM(tscale_infall=tscale)
    results = sam.evolve()
    print(f"tscale={tscale}: mstar={results['mstar'][-1]:.2e}")
```

- [ ] All timescale variations complete
- [ ] Results vary with timescale (shorter → more growth)
- [ ] No divergences or crashes

### Test 4: Wind Models
```python
from galaxySAM.galaxy_sam import GalaxySAM

# No wind
sam1 = GalaxySAM(wind_loading=0.0)
r1 = sam1.evolve()

# With wind
sam2 = GalaxySAM(wind_loading=2.0)
r2 = sam2.evolve()

# Final masses should differ
assert r1['mstar'][-1] != r2['mstar'][-1]
assert r1['mgas'][-1] != r2['mgas'][-1]
```

- [ ] No wind: normal evolution
- [ ] With wind: less stellar/gas mass at end
- [ ] Difference is reasonable (~10-30%)

### Test 5: Accretion Models
```python
for accmodel in [1, 2, 3]:
    sam = GalaxySAM(accmodel=accmodel, nbint=200)
    try:
        results = sam.evolve()
        print(f"accmodel={accmodel}: OK")
    except Exception as e:
        print(f"accmodel={accmodel}: FAILED - {e}")
```

- [ ] Model 1 (exponential) works
- [ ] Model 2 (double exponential) works
- [ ] Model 3 (no accretion) works

## Comparison with IDL Results

### Setup IDL Reference
```idl
; Run IDL to get reference data
galactic_chemical_evolution, $
    fileyieldevol=['yields_evol_z-3_v0...'], $
    /readyields, $
    nbint=1000, $
    ... (other parameters)

; Save to file (manual processing)
; Columns: time mgas mstar metals metallicity sfr
```

### Python Equivalent
```python
from galaxySAM.galaxy_sam import GalaxySAM
sam = GalaxySAM(
    yield_model='lc18',
    metallicity=0.02,
    nbint=1000,
    # ... match IDL parameters
)
results = sam.evolve(output_file='python_results.txt')
```

### Comparison Script
```python
import numpy as np

# Load both
idl_data = np.loadtxt('idl_output.txt')
py_data = np.loadtxt('python_results.txt', skiprows=1)

# Compare key quantities
print("Time grid match:", np.allclose(idl_data[:, 0], py_data[:, 0]))
print("Stellar mass correlation:", 
      np.corrcoef(idl_data[:, 3], py_data[:, 2])[0, 1])
print("Metallicity correlation:",
      np.corrcoef(np.log10(idl_data[:, 5]), 
                  np.log10(py_data[:, 4]))[0, 1])

# Detailed comparison
print("\nFinal stellar mass:")
print(f"  IDL:    {idl_data[-1, 3]:.2e}")
print(f"  Python: {py_data[-1, 2]:.2e}")
print(f"  Ratio:  {idl_data[-1, 3] / py_data[-1, 2]:.2f}")
```

- [ ] Time grids match
- [ ] Stellar mass evolution correlated > 0.95
- [ ] Metallicity evolution correlated > 0.95
- [ ] Final values within 10-20% (acceptable for different codes)

## Numerical Accuracy Tests

### Test: Energy Conservation (Rough)
```python
from galaxySAM.galaxy_sam import GalaxySAM
sam = GalaxySAM(nbint=1000)
results = sam.evolve()

# Total mass should be conserved (gas + stars)
total_mass = results['mgas'] + results['mstar']
print(f"Initial total: {total_mass[0]:.2e}")
print(f"Final total: {total_mass[-1]:.2e}")
```

- [ ] Total mass evolution is smooth
- [ ] No sudden jumps or oscillations

### Test: NaN/Inf Checking
```python
import numpy as np
from galaxySAM.galaxy_sam import GalaxySAM

sam = GalaxySAM()
results = sam.evolve()

for key, arr in results.items():
    if isinstance(arr, np.ndarray):
        has_nan = np.any(np.isnan(arr))
        has_inf = np.any(np.isinf(arr))
        print(f"{key}: NaN={has_nan}, Inf={has_inf}")
        assert not has_nan, f"{key} contains NaN!"
        assert not has_inf, f"{key} contains Inf!"
```

- [ ] No NaN values in any result array
- [ ] No Inf values in any result array

## Performance Tests

### Test: Execution Time
```python
import time
from galaxySAM.galaxy_sam import GalaxySAM

start = time.time()
sam = GalaxySAM(nbint=1000)
results = sam.evolve()
elapsed = time.time() - start

print(f"Execution time: {elapsed:.2f} seconds")
```

- [ ] Single run completes in < 5 seconds
- [ ] 100 parameter variations complete in < 2 minutes

### Test: Memory Usage
```python
import psutil
import os
from galaxySAM.galaxy_sam import GalaxySAM

process = psutil.Process(os.getpid())

sam = GalaxySAM(nbint=5000)
mem_before = process.memory_info().rss / 1024**2

results = sam.evolve()
mem_after = process.memory_info().rss / 1024**2

print(f"Memory usage: {mem_after - mem_before:.1f} MB")
```

- [ ] Single evolution uses < 200 MB
- [ ] Memory freed after completion

## Edge Cases & Robustness

### Test: Very Low Metallicity
```python
from galaxySAM.galaxy_sam import GalaxySAM
sam = GalaxySAM(metallicity=1e-5)  # ~1000x below solar
results = sam.evolve()
assert all(np.isfinite(results['metallicity']))
```

- [ ] Handles low-Z without crashing
- [ ] Results are physically reasonable

### Test: Very High Metallicity
```python
from galaxySAM.galaxy_sam import GalaxySAM
sam = GalaxySAM(metallicity=0.1)  # ~10x solar
results = sam.evolve()
assert all(np.isfinite(results['metallicity']))
```

- [ ] Handles high-Z without crashing
- [ ] No numerical overflow

### Test: Extreme Parameters
```python
extreme_cases = [
    {'wind_loading': 10.0},      # Strong winds
    {'alphaks': 2.0},             # Steep SK law
    {'tscale_sfr': 0.5},         # Fast SF
    {'asnia': 0.1},              # High SNIa rate
]

for params in extreme_cases:
    try:
        sam = GalaxySAM(nbint=500, **params)
        results = sam.evolve()
        print(f"Parameters {params}: OK")
    except Exception as e:
        print(f"Parameters {params}: FAILED - {e}")
```

- [ ] All extreme cases complete without error
- [ ] Results remain finite

## Documentation Tests

### Docstrings Present
```python
from galaxySAM import galaxy_sam, imf, plotting
import inspect

modules = [galaxy_sam, imf, plotting]
for mod in modules:
    for name, obj in inspect.getmembers(mod):
        if inspect.isclass(obj) or inspect.isfunction(obj):
            if not name.startswith('_'):
                has_doc = obj.__doc__ is not None
                print(f"{name}: {has_doc}")
```

- [ ] All public classes have docstrings
- [ ] All public methods have docstrings
- [ ] All parameters documented

### README Completeness
```bash
grep -c "Parameters" README.md    # Should have sections
grep -c "Example" README.md
grep -c "Returns" README.md
```

- [ ] README exists
- [ ] Contains usage examples
- [ ] Contains API documentation
- [ ] Contains references

## Final Verification Checklist

### Code Quality
- [ ] No syntax errors in any Python file
- [ ] No obvious bugs or logical errors
- [ ] Code follows PEP 8 style
- [ ] Imports are organized
- [ ] No unused imports

### Functionality
- [ ] All core features work
- [ ] Evolution runs stably
- [ ] Plots generate correctly
- [ ] CLI interface functional
- [ ] Examples run successfully

### Documentation
- [ ] README complete and clear
- [ ] Docstrings comprehensive
- [ ] Examples provided
- [ ] Error messages helpful
- [ ] Migration guide helpful

### Testing
- [ ] No crashes on normal inputs
- [ ] Handles edge cases gracefully
- [ ] Results physically reasonable
- [ ] Numerical precision acceptable
- [ ] Performance adequate

### Validation
- [ ] Consistent with IDL results (>90% correlation)
- [ ] No NaN/Inf in outputs
- [ ] Energy/mass conservation
- [ ] Parameter dependencies sensible

---

## Sign-off

If all items are checked, the Python implementation is ready for:
- [ ] Production use
- [ ] Integration with other tools
- [ ] Distribution
- [ ] Publication

**Validation Date:** ___________
**Validator:** ___________
**Status:** PASS / FAIL / CONDITIONAL

**Notes:**
___________________________________________
___________________________________________

---

## Quick Test Command

Run this single command for a comprehensive test:

```bash
python -c "
import numpy as np
from galaxySAM import galaxy_sam, plotting

print('Testing Galaxy SAM...')

# Test 1: Instantiate
print('  Creating SAM...', end='')
sam = galaxy_sam.GalaxySAM()
print(' OK')

# Test 2: Evolve
print('  Running evolution...', end='')
results = sam.evolve()
print(' OK')

# Test 3: Check results
print('  Validating results...', end='')
assert all(np.isfinite(results['mstar'])), 'mstar has NaN/Inf'
assert all(np.isfinite(results['mgas'])), 'mgas has NaN/Inf'
assert all(np.isfinite(results['metallicity'])), 'metallicity has NaN/Inf'
print(' OK')

# Test 4: Plot
print('  Creating plots...', end='')
plotter = plotting.EvolutionPlotter()
fig = plotter.plot_evolution(results)
print(' OK')

print('')
print('All tests passed! ✓')
print(f'Final stellar mass: {results[\"mstar\"][-1]:.2e} Msun')
print(f'Final gas mass: {results[\"mgas\"][-1]:.2e} Msun')
print(f'Final metallicity: {results[\"metallicity\"][-1]:.4f}')
"
```

Expected output:
```
Testing Galaxy SAM...
  Creating SAM... OK
  Running evolution... OK
  Validating results... OK
  Creating plots... OK

All tests passed! ✓
Final stellar mass: X.XXe+0X Msun
Final gas mass: X.XXe+0X Msun
Final metallicity: X.XXXX
```
