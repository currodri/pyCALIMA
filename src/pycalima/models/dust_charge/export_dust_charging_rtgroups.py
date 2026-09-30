"""Export per-bin RT-group charging tables (WDB06 yields) for RAMSES/CALIMA.

One file per dust bin, ``dust_charging_rtgroups_<bin_id>.dat``, with the
photoemission and heating rates per photon of every radiation group tabulated
in grain potential and group mean energy, plus the hot-gas tables; see
``rt_group_charging`` for the contents and the per-cell solve RAMSES does.

The group edges must match the RAMSES ``groupL0``/``groupL1`` namelist (RAMSES
checks them against the file header). Set them in the configuration under
``export_parameters.dust_charging_rtgroups`` or with ``--group-L0/--group-L1``.
"""
import argparse

from pycalima.models.dust_charge import rt_group_charging as rg
from pycalima.models.grain_size_config import (
    get_bins,
    get_export_parameters,
    get_lognormal_parameters,
    get_model_data_dir,
    set_config_path,
)

DEFAULT_EXPORT_PARAMS = {
    'group_L0': list(rg.DEFAULT_GROUP_L0),
    'group_L1': list(rg.DEFAULT_GROUP_L1),
}


def main(config_path=None, group_L0=None, group_L1=None):
    if config_path:
        set_config_path(config_path)
    params = get_export_parameters('dust_charging_rtgroups', defaults=DEFAULT_EXPORT_PARAMS)
    L0 = list(group_L0 if group_L0 is not None else params['group_L0'])
    L1 = list(group_L1 if group_L1 is not None else params['group_L1'])
    if len(L0) != len(L1):
        raise ValueError('group_L0 and group_L1 must have the same length')

    output_dir = get_model_data_dir() / 'dust_charging_rtgroups_data'
    output_dir.mkdir(parents=True, exist_ok=True)
    bins = sorted(get_bins(is_pah=False), key=lambda b: b['index'])
    if not bins:
        raise RuntimeError('No dust bins found in grain-size configuration.')

    print(f'Exporting RT-group charging tables to {output_dir}')
    print(f'  groups L0 = {L0}\n  groups L1 = {L1}')
    files = []
    for b in bins:
        a_cm = float(get_lognormal_parameters(b['id'], model_name='basic')['a0']) * 1e-4
        material = 'silicate' if b['composition'].lower() == 'silicate' else 'graphite'
        print(f"  [{b['id']}] {material}, a = {a_cm:.4e} cm", flush=True)
        tab = rg.build_bin_table(material, a_cm, L0, L1, label=b['id'])
        files.append(rg.write_bin_table(tab, output_dir / f"dust_charging_rtgroups_{b['id']}.dat"))
    return {'output_dir': str(output_dir), 'file_count': len(files), 'bins_processed': len(bins),
            'group_L0': L0, 'group_L1': L1}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--config', default=None, help='grain-size configuration JSON')
    ap.add_argument('--group-L0', type=float, nargs='+', default=None, help='group lower edges [eV]')
    ap.add_argument('--group-L1', type=float, nargs='+', default=None, help='group upper edges [eV]')
    args = ap.parse_args()
    main(args.config, args.group_L0, args.group_L1)
