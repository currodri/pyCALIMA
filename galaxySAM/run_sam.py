"""
Main execution script for galaxy SAM evolution calculations.

Usage:
    python -m galaxySAM.run_sam --help
    python -m galaxySAM.run_sam --yield-model kobayashi --metallicity 0.02
"""

import argparse
import numpy as np
from pathlib import Path
import sys

from . import galaxy_sam
from . import plotting
from . import constants


def main():
    """Main entry point for galaxy SAM execution."""
    
    parser = argparse.ArgumentParser(
        description='Run galaxy SAM evolution with different yield models'
    )
    
    # Yield model options
    parser.add_argument('--yield-model', type=str, default='lc18',
                       choices=['kobayashi', 'lc18', 'karakas'],
                       help='Stellar yield model')
    
    # Physical parameters
    parser.add_argument('--metallicity', type=float, default=0.02,
                       help='Initial metallicity (linear Z)')

    parser.add_argument('--metallicity-zsun', type=float, default=None,
                       help='Initial metallicity in units of Zsun (overrides --metallicity)')
    
    parser.add_argument('--imf', type=str, default='chabrier',
                       choices=['salpeter', 'chabrier'],
                       help='Initial Mass Function')
    
    # Galaxy evolution parameters
    parser.add_argument('--tscale-infall', type=float, default=7.0,
                       help='Infall timescale in Gyr')
    
    parser.add_argument('--tscale-sfr', type=float, default=2.2,
                       help='Star formation timescale in Gyr')
    
    parser.add_argument('--alphaks', type=float, default=1.0,
                       help='Schmidt-Kennicutt power law index')
    
    parser.add_argument('--asnia', type=float, default=0.05,
                       help='Fraction of stars becoming SNIa')
    
    parser.add_argument('--wind-loading', type=float, default=0.0,
                       help='Wind mass loading factor')
    
    parser.add_argument('--accmodel', type=int, default=1,
                       choices=[1, 2, 3],
                       help='Accretion model (1=exp, 2=double exp, 3=none)')
    
    parser.add_argument('--nbint', type=int, default=1000,
                       help='Number of integration time steps')
    
    # Output options
    parser.add_argument('-o', '--output', type=Path, default=None,
                       help='Output directory for results and plots')
    
    parser.add_argument('--plot', action='store_true',
                       help='Generate plots')
    
    parser.add_argument('--save-evolution', action='store_true',
                       help='Save evolution data to file')
    
    args = parser.parse_args()

    metallicity = (
        args.metallicity_zsun * constants.ZSUN_ASPLUND
        if args.metallicity_zsun is not None
        else args.metallicity
    )
    
    # Setup output directory
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Galaxy SAM Evolution")
    print("=" * 60)
    print(f"Yield model:      {args.yield_model}")
    print(f"Metallicity:      {metallicity:.4f} (Z/Zsun = {metallicity/constants.ZSUN_ASPLUND:.3f})")
    print(f"IMF:              {args.imf}")
    print(f"Infall timescale: {args.tscale_infall:.1f} Gyr")
    print(f"SFR timescale:    {args.tscale_sfr:.1f} Gyr")
    print(f"Schmidt-Kennicutt: alpha = {args.alphaks:.1f}")
    print(f"SNIa fraction:    {args.asnia:.3f}")
    print(f"Wind loading:     {args.wind_loading:.1f}")
    print(f"Accretion model:  {args.accmodel}")
    print(f"Integration steps:{args.nbint}")
    print("=" * 60)
    
    # Create SAM
    params = {
        'tscale_infall': args.tscale_infall,
        'tscale_sfr': args.tscale_sfr,
        'alphaks': args.alphaks,
        'asnia': args.asnia,
        'wind_loading': args.wind_loading,
        'accmodel': args.accmodel,
        'nbint': args.nbint,
    }
    
    sam = galaxy_sam.GalaxySAM(
        yield_model=args.yield_model,
        metallicity=metallicity,
        imf_type=args.imf,
        **params
    )
    
    # Run evolution
    print("\nRunning galaxy evolution...")
    evolution_file = None
    if args.save_evolution and args.output:
        evolution_file = args.output / 'evolution_data.txt'
    
    results = sam.evolve(output_file=evolution_file)
    
    # Print results summary
    print("\nEvolution Complete!")
    print(f"Final stellar mass:   {results['mstar'][-1]:.2e} Msun")
    print(f"Final gas mass:       {results['mgas'][-1]:.2e} Msun")
    print(f"Final metallicity:    {results['metallicity'][-1]:.4f}")
    z_sun = constants.ZSUN_ASPLUND
    final_logz = np.log10(np.clip(results['metallicity'][-1], 1e-5, 1.0) / z_sun)
    print(f"Final log(Z/Zsun):    {final_logz:.2f}")
    print(f"Mean SFR:             {np.mean(results['sfr']):.2e} Msun/yr")
    
    # Generate plots
    if args.plot:
        print("\nGenerating plots...")
        output_dir = args.output if args.output else Path('.')
        figs = plotting.create_all_plots(results, output_dir=output_dir)
        print(f"Plots saved to {output_dir}")
    
    print("\nDone!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
