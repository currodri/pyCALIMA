"""Small CLI helper to run the collisional equilibrium-temperature plot.

This is a thin wrapper around `plot_eqtemp_withcollision` so the test can be
re-run from the command line without opening a Python shell.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from pycalima.models.dust_radiation.dust_emission import plot_eqtemp_withcollision


def build_parser():
    parser = argparse.ArgumentParser(
        description='Run the dust equilibrium temperature test with collisions.'
    )
    parser.add_argument('--dust-type', default='silicate_bin_00')
    parser.add_argument('--collisional-dust-bin', default=None,
                        help='Collisional table bin, e.g. DustBin_00 or 00. Defaults to bin inferred from --dust-type.')
    parser.add_argument('--ne', type=float, default=1.0)
    parser.add_argument('--nH', type=float, default=1.0)
    parser.add_argument('--nHe', type=float, default=0.1)
    parser.add_argument('--nC', type=float, default=1e-4)
    parser.add_argument('--Tmin', type=float, default=10.0)
    parser.add_argument('--Tmax', type=float, default=1e6)
    parser.add_argument('--nG0', type=int, default=4)
    parser.add_argument('--nT', type=int, default=4)
    parser.add_argument('--G0min', type=float, default=1.0)
    parser.add_argument('--G0max', type=float, default=1e5)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    plt.rcParams['text.usetex'] = False

    plot_eqtemp_withcollision(
        args.dust_type,
        args.ne,
        args.nH,
        args.nHe,
        args.nC,
        args.Tmin,
        args.Tmax,
        nG0=args.nG0,
        nT=args.nT,
        G0min=args.G0min,
        G0max=args.G0max,
        collisional_dust_bin=args.collisional_dust_bin,
    )


if __name__ == '__main__':
    main()