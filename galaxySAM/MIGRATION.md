# IDL to Python Migration Guide

This document explains the mapping between the original IDL code in `yohan_routines/` and the new Python implementation in `galaxySAM/`.

## Overview of Conversion

### File Structure

| IDL File(s) | Python Module | Purpose |
|-------------|---------------|---------|
| `galactic_chemical_evolution.pro` | `galaxy_sam.py::GalaxySAM` | Main evolution engine |
| `cmp_yield_release.pro` | `yield_models.py` | Yield computation and interpolation |
| `rewrite_kobayashi.pro` | `yield_models.py::KobayashiYields` | Kobayashi yield processing |
| `interpolate_lc18.pro` | `yield_models.py::LC18Yields` | LC18 yield interpolation |
| `crunch_karakas.pro` | `yield_models.py::KarakasYields` | Karakas yield processing |
| `sn1a.pro` | `sn1a.py::SNIaModel` | Type Ia calculations |
| `plot_yields.pro` | `plotting.py::YieldPlotter` | Yield visualization |
| `plot_yields_ratio.pro` | `plotting.py::YieldPlotter` | Yield comparisons |
| `plots_yieldevolejecta.pro` | `plotting.py::EvolutionPlotter` | Evolution plotting |

## Key Conversions

### 1. Constants & Parameters

**IDL:**
```idl
thubble=13.5d0
XH=0.76
zsun=0.01345d0
```

**Python:**
```python
from galaxySAM import constants

constants.HUBBLE_TIME  # 13.5 Gyr
constants.ELEMENTS_KOBAYASHI  # Element list
constants.ZSUN_ASPLUND  # 0.01345
```

### 2. File I/O & Arrays

**IDL (readcol):**
```idl
readcol, filename, el, At, y1, y2, y3, ..., format='(A,)'
nlines = n_elements(el)
yield_tmp = dblarr(nlines, 36)
```

**Python:**
```python
import pandas as pd
import numpy as np

df = pd.read_csv(filename, delim_whitespace=True, comment='#')
yield_tmp = df.values  # NumPy array
nlines = len(df)
```

### 3. Yields Reading

**IDL (rewrite_kobayashi.pro):**
```idl
pro rewrite_kobayashi, zmet=zmet
  if(zmet eq 0.)then zmets='0'
  if(zmet eq 0.02)then zmets='0.02'
  ...
  fileintermediate='./yield_ck13_z'+zmets+'.txt'
  readcol, fileintermediate, el, At, y1, y2, ...
  ...
  yield_i(i,j) = total(yield_tmp(indel,i))
end
```

**Python:**
```python
from galaxySAM.yield_models import KobayashiYields

yields = KobayashiYields(metallicity=0.02)
yields.load_from_file('./yield_ck13_z0.02.txt')
yield_value = yields.get_yield(mass=15.0, element='Fe')
```

### 4. Differential Equations

**IDL (galactic_chemical_evolution.pro - simplified):**
```idl
; SFR using Schmidt-Kennicutt
sfr = (mgas^alphaks) / tscale_sfr

; Gas equation
dmgas_dt = accr - sfr + mret - outf

; Stellar mass
dmstar_dt = sfr - mret

; Metals
dmetal_dt = accr * zmet_init - outf * (metals/mgas) + metal_yield
```

**Python:**
```python
from galaxySAM.galaxy_sam import GalaxySAM

# Parameters set in constructor
sam = GalaxySAM(yield_model='kobayashi', ...)

# Equations in _dydt method
def _dydt(self, y, t, accmodel):
    mgas, mstar, metals = y
    sfr = self.star_formation_rate(mgas, mstar)
    accr = self.infall_rate_exponential(t)
    outf = self.outflow_rate(sfr, mstar)
    # ... compute returns ...
    dmgas_dt = accr - sfr + mret - outf
    dmstar_dt = sfr - mret
    dmetal_dt = accr * self.metallicity_init - outf * (metals/mgas) + metal_yields
    return [dmgas_dt, dmstar_dt, dmetal_dt]
```

### 5. Integration

**IDL (odeint equivalent):**
```idl
for i=0L, nbint-1L do begin
  ; Compute derivatives and update state
  y_new = y_old + dt * dydt(y_old, t_old)
endfor
```

**Python:**
```python
from scipy.integrate import odeint

solution = odeint(self._dydt, y_init, time_grid, args=(accmodel,))
```

### 6. Plotting

**IDL (plot_yields.pro):**
```idl
readcol, intermediate, mzas_i, ..., format='(f,f,f,a,f,f,f,f)'
plot_oi, mzas_i, alog10(mla_i), psym=-4, xr=[0.1,300.0], /xs
for j=0, nel2follow-1 do begin
  oplot, mzas_i, alog10(mlos_i(*,j)), psym=-4
endfor
```

**Python:**
```python
from galaxySAM.plotting import YieldPlotter

plotter = YieldPlotter()
fig = plotter.plot_yields_vs_mass(masses, yields_dict, elements=['O', 'Fe'])
fig.savefig('yields.png')
```

## Common IDL → Python Patterns

### Array Operations

| IDL | Python |
|-----|--------|
| `where(arr eq 42, count)` | `np.where(arr == 42)[0]` |
| `dblarr(n)` | `np.zeros(n)` |
| `fltarr(m,n)` | `np.zeros((m,n))` |
| `total(arr)` | `np.sum(arr)` |
| `sort(arr)` | `np.argsort(arr)` |
| `alog10(x)` | `np.log10(x)` |
| `exp(x)` | `np.exp(x)` |
| `size(arr)` / `n_elements(arr)` | `arr.size` / `len(arr)` |
| `(*,i)` column selection | `[:,i]` |

### Control Structures

| IDL | Python |
|-----|--------|
| `for i=0,n-1 do begin ... endfor` | `for i in range(n): ...` |
| `if cond then begin ... endif` | `if cond: ...` |
| `while cond do begin ... endwhile` | `while cond: ...` |

### Keywords/Parameters

| IDL | Python |
|-----|--------|
| `pro func, x, keyword=keyword` | `def func(x, keyword=None):` |
| `if keyword_set(key) then` | `if key is not None:` |
| `if not keyword_set(key) then val=10.0` | `if key is None: val = 10.0` |

## Type Conversions

### String Handling

**IDL:**
```idl
zmets = '0.02'
fileintermediate = './yield_ck13_z' + zmets + '.txt'
```

**Python:**
```python
zmets = '0.02'
fileintermediate = f'./yield_ck13_z{zmets}.txt'
# or
fileintermediate = './yield_ck13_z' + zmets + '.txt'
```

### Floating Point

**IDL:**
```idl
1.0d0    ; double
1.0      ; single
```

**Python:**
```python
1.0      ; default is double precision
np.float32(1.0)  ; explicit single precision
```

## Example: Complete Subroutine Conversion

### Original IDL (simplified)

```idl
pro calculate_yields, mzas, elements, yields_array
  nmass = n_elements(mzas)
  nel = n_elements(elements)
  yields_array = dblarr(nmass, nel)
  
  for i = 0, nmass-1 do begin
    for j = 0, nel-1 do begin
      elem = elements(j)
      m = mzas(i)
      if m lt 8.0 then begin
        yields_array(i,j) = yield_agb(m, elem)
      endif else begin
        yields_array(i,j) = yield_snii(m, elem)
      endelse
    endfor
  endfor
end
```

### Converted Python

```python
import numpy as np

def calculate_yields(masses, elements):
    """Calculate stellar yields for given masses and elements."""
    nmass = len(masses)
    nel = len(elements)
    yields_array = np.zeros((nmass, nel))
    
    for i in range(nmass):
        for j in range(nel):
            elem = elements[j]
            m = masses[i]
            if m < 8.0:
                yields_array[i, j] = yield_agb(m, elem)
            else:
                yields_array[i, j] = yield_snii(m, elem)
    
    return yields_array

# Or more Pythonically with vectorization:
def calculate_yields_vectorized(masses, elements):
    """Calculate yields using NumPy vectorization."""
    nmass = len(masses)
    nel = len(elements)
    yields_array = np.zeros((nmass, nel))
    
    agb_mask = masses < 8.0
    snii_mask = ~agb_mask
    
    for j, elem in enumerate(elements):
        if np.any(agb_mask):
            yields_array[agb_mask, j] = yield_agb(masses[agb_mask], elem)
        if np.any(snii_mask):
            yields_array[snii_mask, j] = yield_snii(masses[snii_mask], elem)
    
    return yields_array
```

## Performance Considerations

### Vectorization

IDL code often uses explicit loops. Python benefits from NumPy vectorization:

**Slow (loop):**
```python
for i in range(len(x)):
    y[i] = x[i]**2 + 2*x[i] + 1
```

**Fast (vectorized):**
```python
y = x**2 + 2*x + 1
```

### Array Preallocatio

Both IDL and Python benefit from preallocating arrays:

```python
# Good
result = np.zeros((1000, 100))
for i in range(1000):
    result[i, :] = expensive_calculation(i)

# Bad - slow due to resizing
result = []
for i in range(1000):
    result.append(expensive_calculation(i))
result = np.array(result)
```

## Debugging Tips

### IDL
- Use `print` for output
- `.compile_opt idl2` for strict checking
- `stop` to halt execution

### Python
```python
import pdb
pdb.set_trace()  # Breakpoint

# Or use logging
import logging
logging.debug('Debug message')

# Or simple print with context
print(f"Value of x at step {i}: {x}")
```

## Common Issues & Fixes

### Issue 1: Array Indexing

**IDL is 1-indexed**, Python is 0-indexed:
```idl
; IDL
arr(0) = 5  ; First element
for i=0, n-1  ; Goes from 0 to n-1
```

```python
# Python
arr[0] = 5  # First element
for i in range(n):  # Goes from 0 to n-1
```

### Issue 2: String Concatenation

**IDL:** `str = 'a' + 'b'` gives `'ab'`
**Python:** `str = 'a' + 'b'` gives `'ab'` (same, good)

But for numbers:
```idl
; IDL - auto conversion
str = 'value_' + 42  ; Gives 'value_42'
```

```python
# Python - must convert
str = f'value_{42}'  # or 'value_' + str(42)
```

### Issue 3: Comparison with Undefined

```idl
; IDL
if not keyword_set(var) then var = default_value
```

```python
# Python
if var is None:
    var = default_value
# Or use default argument value in function
```

## Testing & Validation

### Comparison Script

To validate Python conversion against IDL results:

```python
# Load IDL results
idl_results = np.loadtxt('idl_output.txt')

# Get Python results
python_sam = GalaxySAM(...)
python_results = python_sam.evolve()

# Compare
np.testing.assert_allclose(
    python_results['mstar'],
    idl_results[:, 1],
    rtol=1e-3,  # 0.1% tolerance
    atol=1e10   # Absolute tolerance for small numbers
)
```

## Resources

- NumPy documentation: https://numpy.org/doc/
- SciPy documentation: https://docs.scipy.org/
- Matplotlib documentation: https://matplotlib.org/
- Python IDL bridge (for hybrid code): https://www.scipy.org/

## Contributing Conversions

If you convert additional IDL routines:

1. Place Python code in appropriate module
2. Add docstrings with parameter descriptions
3. Include examples in docstrings
4. Add unit tests
5. Update this migration guide
6. Document any deviations from IDL original

## See Also

- [`README.md`](README.md) - Main module documentation
- [`examples.py`](examples.py) - Runnable examples
- Original IDL code in `yohan_routines/`
