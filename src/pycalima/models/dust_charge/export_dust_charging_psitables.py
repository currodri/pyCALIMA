"""Export per-bin WDB06 charging tables with an EUV dimension for RAMSES/CALIMA (charging_model='WDB06tab').

One file per dust bin, ``dust_charging_psitab_<bin_id>.dat``, built from that
bin's RT-group table (``export_dust_charging_rtgroups``, same groups), at the
run's group mean energies (``--egy``, as in RAMSES info_rt); see
``charging_psi_tables`` for the contents.
"""
import argparse
from pathlib import Path

from pycalima.models.dust_charge import charging_psi_tables as cpt
from pycalima.models.dust_charge import rt_group_charging as rg
from pycalima.models.grain_size_config import get_bins, get_model_data_dir, set_config_path


def main(config_path=None, rtgroups_dir=None, output_dir=None, T_bb=cpt.T_EUV, egy=None, workers=1):
    if config_path:
        set_config_path(config_path)
    base = get_model_data_dir()
    rtgroups_dir = Path(rtgroups_dir) if rtgroups_dir else base / 'dust_charging_rtgroups_data'
    output_dir = Path(output_dir) if output_dir else base / 'dust_charging_psitab_data'
    output_dir.mkdir(parents=True, exist_ok=True)
    bins = sorted(get_bins(is_pah=False), key=lambda b: b['index'])
    print(f'Exporting EUV-extended charging tables to {output_dir} (EUV shape: {T_bb:.3g} K blackbody)')
    files = []
    for b in bins:
        tab = rg.read_bin_table(rtgroups_dir / f"dust_charging_rtgroups_{b['id']}.dat")
        print(f"  [{b['id']}] {tab['material']}, a = {tab['a']:.4e} cm", flush=True)
        pt = cpt.build_psi_table(tab, T_bb=T_bb, egy=egy, workers=workers)
        files.append(cpt.write_psi_table(pt, output_dir / f"dust_charging_psitab_{b['id']}.dat", label=b['id']))
    return {'output_dir': str(output_dir), 'file_count': len(files)}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--config', default=None, help='grain-size configuration JSON')
    ap.add_argument('--rtgroups-dir', default=None, help='RT-group tables (default: model_data/dust_charging_rtgroups_data)')
    ap.add_argument('--output-dir', default=None)
    ap.add_argument('--T-bb', type=float, default=cpt.T_EUV, help='blackbody temperature of the EUV shape [K]')
    ap.add_argument('--egy', default=None,
                    help="the run's group mean energies [eV], comma-separated (RAMSES info_rt egy; default: the blackbody's)")
    ap.add_argument('--workers', type=int, default=1)
    a = ap.parse_args()
    egy = [float(x) for x in a.egy.split(',')] if a.egy else None
    print(main(a.config, a.rtgroups_dir, a.output_dir, a.T_bb, egy, a.workers))
