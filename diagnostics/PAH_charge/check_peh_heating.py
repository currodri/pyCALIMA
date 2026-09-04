import argparse
import logging
import multiprocessing as mp
import os
import traceback

# Ensure the repository root is on sys.path.

from pycalima.models.PAH_charge.PAH_photoelectric_heating import compute_tables_ISRF


def parse_args():
	p = argparse.ArgumentParser(description='Debug runner for PAH PEH table computation')
	p.add_argument('--T', type=float, default=500.0, help='Gas temperature [K]')
	p.add_argument('--ne-min', type=float, default=1e-5, help='Minimum electron density [cm^-3]')
	p.add_argument('--ne-max', type=float, default=5e5, help='Maximum electron density [cm^-3]')
	p.add_argument('--n-ne', type=int, default=100, help='Number of electron-density points')
	p.add_argument('--radiation-model', default='Draine')
	p.add_argument('--op-model', default='Malloci')
	p.add_argument('--attach-model', default='Berne')
	p.add_argument('--mp-log', action='store_true', help='Enable multiprocessing worker logging')
	return p.parse_args()


def main():
	args = parse_args()

	# Helps surface crashes from child processes.
	os.environ['PYTHONFAULTHANDLER'] = '1'
	if args.mp_log:
		mp.log_to_stderr(logging.INFO)

	print('[test_peh_heating] Running with:')
	print(f'  T={args.T}, ne_min={args.ne_min}, ne_max={args.ne_max}, n_ne={args.n_ne}')
	print(f'  radiation_model={args.radiation_model}, op_model={args.op_model}, attach_model={args.attach_model}')

	try:
		compute_tables_ISRF(
			args.T,
			args.ne_min,
			args.ne_max,
			n_ne=args.n_ne,
			radiation_model=args.radiation_model,
			op_model=args.op_model,
			attach_model=args.attach_model,
		)
	except Exception as exc:
		print('\n[test_peh_heating] compute_tables_ISRF failed with exception:')
		print(f'  {type(exc).__name__}: {exc}')
		print('[test_peh_heating] Full traceback follows:\n')
		traceback.print_exc()
		raise


if __name__ == '__main__':
	main()