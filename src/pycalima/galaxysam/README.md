# Galaxy SAM Python Module

A comprehensive Python implementation of a Semi-Analytic Model (SAM) for galactic chemical evolution and stellar nucleosynthesis. This module converts and extends the IDL routines from `yohan_routines/` into a modular, well-documented Python package.

## Overview

The `galaxySAM` module provides tools for:

- **Stellar Yield Models**: Multiple nucleosynthesis yield models (Kobayashi, Limongi & Chieffi, Karakas)
- **Galaxy Evolution**: Semi-analytic simulation of star formation, gas accretion, chemical enrichment, and feedback
- **Initial Mass Function**: Various IMF prescriptions (Salpeter, Chabrier)
- **Type Ia Supernovae**: SNIa rates, yields, and progenitor models
- **Analysis & Visualization**: Comprehensive plotting utilities for yields and evolution results

## Module Structure

### Core Modules

- **`constants.py`**: Physical constants, element definitions, and default parameters
- **`imf.py`**: Initial Mass Function implementations (Salpeter, Chabrier, broken power-law)
- **`yield_models.py`**: Stellar yield models and interpolation
- **`sn1a.py`**: Type Ia supernova rates and yields
- **`galaxy_sam.py`**: Main galaxy evolution engine
- **`plotting.py`**: Visualization tools
- **`run_sam.py`**: Command-line interface
- **`examples.py`**: Usage examples and demonstrations

## Installation & Setup

### Dependencies

```bash
pip install numpy scipy pandas matplotlib
```

### Python Environment

The module is already integrated into the pyCALIMA repository:

```bash
cd /Users/currodri/Documents/GitHub/CALIMA
python -m galaxySAM.examples  # Run examples
python -m galaxySAM.run_sam --help  # See command-line options
```

## Quick Start

### Basic Galaxy Evolution

```python
from galaxySAM import galaxy_sam

# Create a galaxy SAM
sam = galaxy_sam.GalaxySAM(
    yield_model='kobayashi',
    metallicity=0.02,  # Solar
    imf_type='chabrier',
    tscale_infall=7.0,  # Gyr
    tscale_sfr=2.2,      # Gyr
)

# Run evolution
results = sam.evolve()

# Access results
print(f"Final stellar mass: {results['mstar'][-1]:.2e} Msun")
print(f"Final metallicity: {results['metallicity'][-1]:.4f}")
```

### Plot Results

```python
from galaxySAM import plotting

plotter = plotting.EvolutionPlotter()
fig = plotter.plot_evolution(results, output_file='evolution.png')
```

### Command-Line Usage

```bash
# Basic run
python -m galaxySAM.run_sam --yield-model kobayashi --plot -o ./output

# With custom parameters
python -m galaxySAM.run_sam \
    --yield-model lc18 \
    --metallicity 0.01 \
    --tscale-infall 5.0 \
    --wind-loading 2.0 \
    --plot \
    -o ./results_wind
```

## Yield Models

### Available Models

| Model | Reference | Range | Notes |
|-------|-----------|-------|-------|
| `kobayashi` | Kobayashi et al. 2006 | Z=0 to Z=0.05 | SNII yields, HNe treatment |
| `lc18` | Limongi & Chieffi 2018 | Z=-3 to Z=0.3 (log scale) | Rotation-dependent, extensive grid |
| `karakas` | Karakas 2010 | Z=0.001 to Z=0.02 | AGB yields |

### Loading Yield Data

```python
from galaxySAM import yield_models

# Create Kobayashi model
yields = yield_models.KobayashiYields(metallicity=0.02)

# Load from file
yields.load_from_file('yield_ck13_z0.02.txt')

# Get yield for specific mass/element
yield_fe = yields.get_yield(mass=20.0, element='Fe')
```

## IMF Models

### Available IMF Types

```python
from galaxySAM import imf

# Salpeter (power-law)
imf_sal = imf.create_imf('salpeter', alpha=-2.35, mmin=0.1, mmax=100.0)

# Chabrier (lognormal + power-law)
imf_chab = imf.create_imf('chabrier')

# Broken power-law (custom)
imf_broken = imf.create_imf('broken_powerlaw',
                            alpha_slopes=[-1.3, -2.3],
                            mass_bounds=[0.1, 0.5, 100.0])

# Evaluate at specific mass
phi = imf_sal(15.0)  # IMF value at 15 Msun
```

## Galaxy Evolution Parameters

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tscale_infall` | 7.0 | Gas infall timescale (Gyr) |
| `tscale_sfr` | 2.2 | Star formation timescale (Gyr) |
| `alphaks` | 1.0 | Schmidt-Kennicutt power-law index |
| `asnia` | 0.05 | Fraction of stars becoming SNIa |
| `wind_loading` | 0.0 | Wind mass loading factor |
| `windmodel` | None | Wind model (1,2,3 for different implementations) |
| `accmodel` | 1 | Accretion model (1=exp, 2=double exp, 3=none) |
| `nbint` | 1000 | Number of integration time steps |

### Wind Models

```python
# No wind
sam = galaxy_sam.GalaxySAM(wind_loading=0.0)

# Fixed wind loading
sam = galaxy_sam.GalaxySAM(wind_loading=2.0)

# Halo-mass dependent wind
sam = galaxy_sam.GalaxySAM(windmodel=1, minmload=0.5, maxmload=10.0)

# Hayward & Hopkins 2017 model
sam = galaxy_sam.GalaxySAM(windmodel=2)
```

## Advanced Usage

### Multi-Metallicity Evolution

Track chemical evolution across a range of metallicities:

```python
sam_multi = galaxy_sam.MultiMetallicitySAM(
    nz_bins=10,
    yield_model='kobayashi',
    imf_type='chabrier'
)

results = sam_multi.evolve()
```

### Custom Yield Model

```python
class CustomYields(yield_models.YieldModel):
    def __init__(self, metallicity=0.02):
        super().__init__('custom', metallicity)
        # Load your custom data
        
    def get_yield(self, mass, element):
        # Implement your yield prescription
        return custom_yield_function(mass, element)

# Use with SAM
sam = galaxy_sam.GalaxySAM(yield_model=CustomYields())
```

### Parameter Studies

```python
# Scan over wind loading
wind_loads = [0.0, 1.0, 2.0, 5.0, 10.0]
results_scan = {}

for w in wind_loads:
    sam = galaxy_sam.GalaxySAM(
        yield_model='kobayashi',
        wind_loading=w
    )
    results_scan[w] = sam.evolve()
    
# Plot comparison
# ... (use plotting module)
```

## Plotting

### Evolution Plots

```python
from galaxySAM import plotting

# Main evolution quantities
plotter = plotting.EvolutionPlotter()
plotter.plot_evolution(results)

# Gas-metallicity phase space
plotter.plot_gas_metal_phase(results)

# Abundance ratios
plotter.plot_abundance_ratios(results)
```

### Yield Plots

```python
# Individual yields
yield_plotter = plotting.YieldPlotter()
yield_plotter.plot_yields_vs_mass(masses, yields_dict, elements=['O', 'Fe', 'Mg'])

# Mass loss
yield_plotter.plot_mass_loss_vs_mass(masses, mass_loss_dict)

# Model comparison
yield_plotter.plot_yields_comparison(
    [masses1, masses2],
    [yields1, yields2],
    ['Kobayashi', 'LC18'],
    element='Fe'
)
```

## Examples

Run the built-in examples:

```bash
python -m galaxySAM.examples
```

### Example 1: Basic Evolution
```python
from galaxySAM.examples import example_basic_evolution
results = example_basic_evolution()
```

### Example 2: Model Comparison
```python
from galaxySAM.examples import example_multiple_models
results = example_multiple_models()
```

### Example 3: Parameter Study
```python
from galaxySAM.examples import example_parameter_study
results = example_parameter_study()
```

### Example 4: Multi-Metallicity
```python
from galaxySAM.examples import example_multi_metallicity
results = example_multi_metallicity()
```

## Output

The module generates several types of output:

### Data Files

Evolution results are saved as tab-separated values:
```
time_gyr  mgas_msun  mstar_msun  metals_msun  metallicity  sfr_msun_per_yr
0.0       1.00e+10   0.00e+00   1.35e+07     1.350e-03    0.00e+00
...
```

### Plots

- `evolution.png`: Main evolution quantities (4-panel plot)
- `gas_metal_phase.png`: Phase space diagram
- `yields.png`: Stellar yield comparisons
- `model_comparison.png`: Results from different models

## Cosmological Accretion (Experimental)

The module includes support for cosmological accretion rates (Dekel et al. 2009):

```python
sam = galaxy_sam.GalaxySAM(
    cosmic=True,
    startz=20,      # Starting redshift
    startm=3.75e9,  # Starting halo mass (Msun)
)
```

## Constants & Abundances

### Solar Abundances

```python
from galaxySAM import constants

# Asplund et al. 2009
print(constants.ASPLUND_ABUNDANCES['Fe'])  # 3.1e-4

# Solar metallicity
print(constants.ZSUN_ASPLUND)  # 0.01345

# Element lists
print(constants.ELEMENTS_LC18)  # ['H', 'He', 'C', ...]
```

## Related IDL Code

The original IDL routines that were converted:

- `galactic_chemical_evolution.pro` → `galaxy_sam.GalaxySAM`
- `cmp_yield_release.pro` → `yield_models.py`
- `rewrite_kobayashi.pro` → `yield_models.KobayashiYields`
- `sn1a.pro` → `sn1a.SNIaModel`
- `plot_yields.pro` → `plotting.YieldPlotter`
- `crunch_karakas.pro` → Yield file processing

## Limitations & Future Improvements

### Current Limitations

1. Yield file loading requires proper formatting (see `yield_models.py` for expected format)
2. Some cosmological features are simplified/placeholder
3. Multi-zone models not yet fully implemented
4. Dust/grain growth not included in evolution

### Planned Enhancements

- [ ] Full cosmological accretion implementation
- [ ] Dust enrichment tracking in evolution
- [ ] Radial migration in disk models
- [ ] AGN feedback models
- [ ] Reionization modeling
- [ ] Integration with RAMSES simulations

## Contributing

To add new features or yield models:

1. Subclass the appropriate base class (`YieldModel`, `IMF`, etc.)
2. Implement required methods
3. Add tests and documentation
4. Submit pull request

## References

- Kobayashi et al. 2006: MNRAS 369, 1137
- Limongi & Chieffi 2018: ApJS 237, 13
- Karakas 2010: MNRAS 403, 1413
- Chabrier 2003: PASP 115, 763
- Dekel et al. 2009: Nature 457, 451

## License

Same as pyCALIMA repository

## Contact

For questions or issues, please contact the pyCALIMA maintainers.
