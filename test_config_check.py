#!/usr/bin/env python
"""Quick test of test configuration."""

from models.grain_size_config import set_config_path, load_grain_size_config, get_bins, get_lognormal_parameters

# Set test config
test_config = 'models/test_grain_size_distribution.json'
set_config_path(test_config)

# Load and display
cfg = load_grain_size_config()
dust_bins = get_bins(is_pah=False)
pah_bins = get_bins(is_pah=True)

print(f'Test Configuration: {test_config}')
print(f'\nDust Bins: {len(dust_bins)}')
for bin_info in dust_bins:
    bin_id = bin_info['id']
    params = get_lognormal_parameters(bin_id)
    print(f'  {bin_id}: a0={params["a0"]:.4e} micron')

print(f'\nPAH Bins: {len(pah_bins)}')
for bin_info in pah_bins:
    bin_id = bin_info['id']
    params = get_lognormal_parameters(bin_id)
    a0_angstrom = params['a0'] * 1e4
    print(f'  {bin_id}: a0={a0_angstrom:.2f} Angstrom ({params["a0"]:.4e} micron)')

print('\n✓ Test configuration ready!')
