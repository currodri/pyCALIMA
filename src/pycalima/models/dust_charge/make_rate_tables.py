#!/usr/bin/env python3
"""Example runner for make_rate_gamma_T_tables

Writes tables to the requested output directory. Usage:

python examples/make_rate_tables.py --grain graphite --a 0.01 --outdir examples/tables --nT 8 --n_gamma 16

"""
import argparse
import os

# Ensure the repository root is on sys.path so examples can import top-level modules

from pycalima.models.dust_charge.dust_photoelectric_heating import make_rate_gamma_T_tables


def parse_args():
    p = argparse.ArgumentParser(description='Build heating/cooling tables (log-log) for Fortran interpolation')
    p.add_argument('--grain', default='silicate', help='grain type: graphite or silicate')
    p.add_argument('--a-cm', type=float, default=1e-6, help='grain radius in cm')
    p.add_argument('--a-micron', type=float, default=None, help='legacy grain radius in microns (converted to cm)')
    p.add_argument('--mode', choices=['fix_G0','fix_ne'], default='fix_G0')
    p.add_argument('--fixed', type=float, default=1.0, help='fixed G0 or ne value depending on mode')
    p.add_argument('--Tmin', type=float, default=10.0)
    p.add_argument('--Tmax', type=float, default=1e5)
    p.add_argument('--nT', type=int, default=8)
    p.add_argument('--gamma_min', type=float, default=1e-4)
    p.add_argument('--gamma_max', type=float, default=1e6)
    p.add_argument('--n_gamma', type=int, default=16)
    p.add_argument('--outdir', default='./', help='output directory for tables')
    p.add_argument('--label', default=None, help='optional grain label for output filenames (e.g. pah_bin_0)')
    p.add_argument('--workers', type=int, default=None)
    p.add_argument('--debug', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    a_cm = float(args.a_cm)
    if args.a_micron is not None:
        a_cm = float(args.a_micron) * 1e-4
    res = make_rate_gamma_T_tables(
        grain_type=args.grain,
        a_cm=a_cm,
        radiation_model='Mathis',
        mode=args.mode,
        fixed_value=args.fixed,
        Tmin=args.Tmin,
        Tmax=args.Tmax,
        nT=args.nT,
        gamma_min=args.gamma_min,
        gamma_max=args.gamma_max,
        n_gamma=args.n_gamma,
        num_workers=args.workers,
        out_dir=args.outdir,
        debug=args.debug,
        grain_label=args.label,
    )
    print('Wrote tables to:', res['out_dir'])
    print('T shape:', res['log_peh'].shape)
    print('gamma length:', len(res['gamma_vals']))


if __name__ == '__main__':
    main()
