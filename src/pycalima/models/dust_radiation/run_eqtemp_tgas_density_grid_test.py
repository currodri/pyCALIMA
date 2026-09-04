"""CLI helper to run the Tdust/Tgas grid plot over gas temperature and density.

This wraps `plot_eqtemp_tgas_density_grid` so the parameter-space map can be
re-generated from the command line.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from pycalima.models.dust_radiation.dust_emission import plot_eqtemp_tgas_density_grid


def build_parser():
    parser = argparse.ArgumentParser(
        description='Run the Tdust/Tgas map on a (Tgas, nH) grid with collisions (G0 fixed to 1).' 
    )
    parser.add_argument('--dust-bin', default='DustBin_01',
                        help='Bin id used for both optical properties and collisional tables, e.g. DustBin_01.')

    parser.add_argument('--Tgas-min', type=float, default=10.0)
    parser.add_argument('--Tgas-max', type=float, default=1e6)

    parser.add_argument('--nH-min', type=float, default=1e-4)
    parser.add_argument('--nH-max', type=float, default=1e4)

    parser.add_argument('--near-equilibrium-tol', type=float, default=0.1)
    parser.add_argument('--method', choices=['linearized', 'newton'], default='linearized')

    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--filename', default=None)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    plt.rcParams['text.usetex'] = False

    plot_eqtemp_tgas_density_grid(
        dust_bin=args.dust_bin,
        Tgas_min=args.Tgas_min,
        Tgas_max=args.Tgas_max,
        nH_min=args.nH_min,
        nH_max=args.nH_max,
        near_equilibrium_tol=args.near_equilibrium_tol,
        method=args.method,
        output_dir=args.output_dir,
        filename=args.filename,
    )


if __name__ == '__main__':
    main()
