"""Quick Reference: IDL Routines to Python Classes/Functions

Maps every IDL routine in yohan_routines/ to its Python equivalent.
"""

# MAPPING: IDL Routine → Python Module.Class.method / function

ROUTINE_MAPPING = {
    # ============ MAIN EVOLUTION ENGINE ============
    "galactic_chemical_evolution.pro": {
        "description": "Main galaxy SAM evolution driver",
        "maps_to": "galaxySAM.galaxy_sam.GalaxySAM",
        "key_methods": [
            "evolve()",           # Run evolution
            "_dydt()",            # Differential equations  
            "star_formation_rate()",
            "infall_rate_exponential()",
            "outflow_rate()",
        ],
        "example": """
from galaxySAM.galaxy_sam import GalaxySAM
sam = GalaxySAM(
    yield_model='kobayashi',
    tscale_infall=7.0,
    tscale_sfr=2.2,
)
results = sam.evolve()
"""
    },

    # ============ YIELD CALCULATIONS ============
    "cmp_yield_release.pro": {
        "description": "Compute stellar yields from populations",
        "maps_to": "galaxySAM.yield_models",
        "key_classes": [
            "KobayashiYields",
            "LC18Yields", 
            "KarakasYields",
            "CombinedYieldModel",
        ],
        "example": """
from galaxySAM.yield_models import KobayashiYields
yields = KobayashiYields(metallicity=0.02)
yields.load_from_file('yield_ck13_z0.02.txt')
fe_yield = yields.get_yield(mass=20.0, element='Fe')
"""
    },

    "rewrite_kobayashi.pro": {
        "description": "Process Kobayashi yield files",
        "maps_to": "galaxySAM.yield_models.KobayashiYields",
        "key_methods": [
            "load_from_file()",
            "get_yield()",
            "interpolate_yield()",
        ],
        "example": """
from galaxySAM.yield_models import KobayashiYields
kb = KobayashiYields(metallicity=0.02)
kb.load_from_file('yield_ck13_z0.02.txt')
"""
    },

    "interpolate_lc18.pro": {
        "description": "Interpolate LC18 yields for arbitrary rotation",
        "maps_to": "galaxySAM.yield_models.LC18Yields",
        "key_methods": [
            "load_from_file()",
            "interpolate_yield()",
        ],
        "example": """
from galaxySAM.yield_models import LC18Yields
lc18 = LC18Yields(metallicity_log=-0.3, velocity=150)
lc18.load_from_file('lc18_yields.txt')
"""
    },

    "crunch_karakas.pro": {
        "description": "Process Karakas yield files",
        "maps_to": "galaxySAM.yield_models.KarakasYields",
        "key_methods": [
            "load_from_file()",
            "get_yield()",
        ],
        "example": """
from galaxySAM.yield_models import KarakasYields
kar = KarakasYields(metallicity=0.02)
kar.load_from_file('karakas_z0.02_simplified.txt')
"""
    },

    "prepare_lc18_yields_forcurro.pro": {
        "description": "Prepare LC18 yields for calculations",
        "maps_to": "galaxySAM.yield_models.LC18Yields.load_from_file()",
        "note": "Functionality integrated into LC18Yields class"
    },

    # ============ TYPE Ia SUPERNOVAE ============
    "sn1a.pro": {
        "description": "Type Ia supernova calculations",
        "maps_to": "galaxySAM.sn1a.SNIaModel",
        "key_methods": [
            "tau_m_padova()",       # Stellar lifetime
            "tau_m_simple()",
            "tau_m_rood()",
            "snia_rate_delay_time()",  # SNIa rate
            "snia_rate_progenitor_age()",
            "yields_snia()",        # SNIa yields
            "snia_mass_return()",
            "inverse_mass_sampler()",
        ],
        "example": """
from galaxySAM.sn1a import SNIaModel
snia = SNIaModel(asnia=0.05)
age_years = 1e9
rate = snia.snia_rate_delay_time(age_years)
yields = snia.yields_snia(model='nomoto84')
"""
    },

    # ============ IMF UTILITIES ============
    "imf.pro (implied)": {
        "description": "Initial Mass Function utilities",
        "maps_to": "galaxySAM.imf",
        "key_classes": [
            "SalpeterIMF",
            "ChabrierIMF",
            "BrokenPowerLawIMF",
        ],
        "key_functions": [
            "create_imf()",
            "imf_weighted_quantity()",
        ],
        "example": """
from galaxySAM.imf import create_imf
imf = create_imf('chabrier')
phi = imf(15.0)  # IMF at 15 Msun

from galaxySAM.imf import SalpeterIMF
sal = SalpeterIMF(alpha=-2.35)
"""
    },

    # ============ PLOTTING ============
    "plot_yields.pro": {
        "description": "Plot stellar yields vs mass",
        "maps_to": "galaxySAM.plotting.YieldPlotter.plot_yields_vs_mass()",
        "example": """
from galaxySAM.plotting import YieldPlotter
plotter = YieldPlotter()
fig = plotter.plot_yields_vs_mass(masses, yields_dict)
"""
    },

    "plot_yields_ratio.pro": {
        "description": "Compare yield models",
        "maps_to": "galaxySAM.plotting.YieldPlotter.plot_yields_comparison()",
        "example": """
from galaxySAM.plotting import YieldPlotter
plotter = YieldPlotter()
fig = plotter.plot_yields_comparison(
    [masses1, masses2],
    [yields1, yields2],
    ['Model1', 'Model2'],
    element='Fe'
)
"""
    },

    "plot_yieldsnia.pro": {
        "description": "Plot SNIa yields",
        "maps_to": "galaxySAM.sn1a.SNIaModel.yields_snia()",
        "note": "Yields can be plotted with YieldPlotter"
    },

    "plots_yieldevolejecta.pro": {
        "description": "Plot galaxy evolution quantities",
        "maps_to": "galaxySAM.plotting.EvolutionPlotter.plot_evolution()",
        "example": """
from galaxySAM.plotting import EvolutionPlotter
plotter = EvolutionPlotter()
fig = plotter.plot_evolution(results)
"""
    },

    "plot_allimf.pro": {
        "description": "Compare IMF models",
        "maps_to": "galaxySAM.imf module functions",
        "example": """
from galaxySAM.imf import create_imf
import numpy as np
masses = np.logspace(-1, 2, 100)

for imf_type in ['salpeter', 'chabrier']:
    imf = create_imf(imf_type)
    phi = imf(masses)
    # Plot phi vs masses
"""
    },

    "plot_wiersma.pro": {
        "description": "Plot gas cooling (not implemented)",
        "maps_to": "Not in scope - dust cooling in separate modules",
        "note": "Beyond scope of galaxy SAM conversion"
    },

    # ============ COMPARISON & ANALYSIS ============
    "cmp_typeia_yield_release.pro": {
        "description": "Compare Type Ia yields",
        "maps_to": "galaxySAM.plotting + galaxySAM.sn1a",
        "example": """
from galaxySAM.sn1a import SNIaModel
from galaxySAM.plotting import YieldPlotter
snia = SNIaModel()
yields1 = snia.yields_snia('nomoto84')
yields2 = snia.yields_snia('iwamoto99')
# Use YieldPlotter to compare
"""
    },

    "exemple_comparison_yield.pro": {
        "description": "Example yield comparison script",
        "maps_to": "galaxySAM.examples module",
        "example": """
from galaxySAM.examples import example_multiple_models
results = example_multiple_models()
"""
    },

    # ============ HELPER ROUTINES ============
    "command_idl_gce.pro": {
        "description": "Command wrapper for GCE",
        "maps_to": "galaxySAM.run_sam.main()",
        "usage": "python -m galaxySAM.run_sam --help"
    },

    "command_idl_run_all_yield.pro": {
        "description": "Batch yield calculations",
        "maps_to": "Custom loop over yield_models classes",
        "example": """
from galaxySAM.yield_models import create_yield_model
models = ['kobayashi', 'lc18', 'karakas']
for model in models:
    y = create_yield_model(model, metallicity=0.02)
    # Process...
"""
    },

    "run_all_crunch_karakas.pro": {
        "description": "Process all Karakas files",
        "maps_to": "galaxySAM.yield_models.KarakasYields + loop",
        "example": """
from galaxySAM.yield_models import KarakasYields
metallicities = [0.001, 0.004, 0.008, 0.02]
for z in metallicities:
    kar = KarakasYields(metallicity=z)
    kar.load_from_file(f'karakas_z{z}.txt')
"""
    },

    "run_all_crunch_lc18.pro": {
        "description": "Process all LC18 files",
        "maps_to": "galaxySAM.yield_models.LC18Yields + loop",
        "example": """
from galaxySAM.yield_models import LC18Yields
for z_log in [-3.0, -2.0, -1.0, -0.6, -0.3, 0.0, 0.3]:
    for vel in [0, 150, 300]:
        lc18 = LC18Yields(metallicity_log=z_log, velocity=vel)
"""
    },

    "run_all_interpolate_lc18.pro": {
        "description": "Interpolate LC18 for all parameters",
        "maps_to": "galaxySAM.yield_models.LC18Yields.interpolate_yield()",
    },

    "run_all_yield_release.pro": {
        "description": "Batch yield calculations",
        "maps_to": "Custom loop with GalaxySAM",
    },

    # ============ MISC ============
    "convert_myyieldsintoramses.pro": {
        "description": "Convert to RAMSES format (specialized)",
        "maps_to": "Custom function (see examples)",
        "note": "Would need to implement based on RAMSES format"
    },

    "convert_myyieldsintoramses2.pro": {
        "description": "Convert to RAMSES format v2",
        "maps_to": "Custom function (see examples)",
        "note": "Would need to implement based on RAMSES format"
    },

    "execute_convertintoramses.pro": {
        "description": "Execute RAMSES conversion",
        "maps_to": "Custom wrapper function",
    },

    "exemples_command_idl_gce.pro": {
        "description": "Examples of GCE calculations",
        "maps_to": "galaxySAM.examples module",
        "example": """
python -m galaxySAM.examples
# or
from galaxySAM.examples import *
example_basic_evolution()
"""
    },

    "Old/ directory": {
        "description": "Deprecated/old IDL code",
        "maps_to": "Not converted (archived)",
        "note": "Check git history if needed"
    },
}


def print_mapping():
    """Print the complete IDL→Python mapping."""
    for idl_file, info in ROUTINE_MAPPING.items():
        print(f"\n{'='*70}")
        print(f"IDL: {idl_file}")
        print(f"{'='*70}")
        
        if 'description' in info:
            print(f"Description: {info['description']}")
        
        if 'maps_to' in info:
            print(f"Maps to: {info['maps_to']}")
        
        if 'key_methods' in info:
            print("\nKey Methods:")
            for method in info['key_methods']:
                print(f"  - {method}")
        
        if 'key_classes' in info:
            print("\nKey Classes:")
            for cls in info['key_classes']:
                print(f"  - {cls}")
        
        if 'key_functions' in info:
            print("\nKey Functions:")
            for func in info['key_functions']:
                print(f"  - {func}")
        
        if 'example' in info:
            print(f"\nExample:\n{info['example']}")
        
        if 'usage' in info:
            print(f"Usage: {info['usage']}")
        
        if 'note' in info:
            print(f"Note: {info['note']}")


def find_python_equivalent(idl_routine_name):
    """
    Find Python equivalent for an IDL routine.
    
    Usage:
        find_python_equivalent('galactic_chemical_evolution.pro')
    """
    for idl, info in ROUTINE_MAPPING.items():
        if idl_routine_name.lower() in idl.lower():
            return info
    return None


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        routine = sys.argv[1]
        result = find_python_equivalent(routine)
        if result:
            print(f"\nIDL routine: {routine}")
            print(f"Python equivalent: {result['maps_to']}")
            if 'example' in result:
                print(f"\nExample:\n{result['example']}")
        else:
            print(f"Routine '{routine}' not found in mapping")
    else:
        print_mapping())
