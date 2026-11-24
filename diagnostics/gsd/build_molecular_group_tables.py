"""PAH group tables (charge, photoelectric heating, absorption) of the GSD's molecular bins, on the GSD's photon groups.

``pahchem.group_tables`` builds one table per N_C range on the 8 RTZ groups; the GSD charges its grains on the 12 groups
of ``gsd.charging.AGN_GROUPS`` (the 8, with the last split up to 10 keV), and a cell's molecular and grain bins see
the same photon groups. This script builds the tables of ``CarbonLayout``'s molecular bins (N_C 20-400, 4 bins by
default) on those groups, with every PAHdb v4.00 molecule complete in the charges -1..+2 (``group_tables.complete``),
named as ``pahchem.hydrogen_tables`` names its bins (``bin_nc<lo>-<hi>``), for ``gsd.molecular.MolecularTables``.

Needs PAHdb ($CALIMA_PAHDB_DIR). About 1.3 s per molecule (712 molecules) and 1.8 GB for the parent.

    python diagnostics/gsd/build_molecular_group_tables.py --out results/gsd_a3c/tables/group_tables --workers 4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pycalima.gsd.carbon import M_C, CarbonLayout
from pycalima.gsd.charging import AGN_GROUP_L0, AGN_GROUP_L1
from pycalima.pahchem import group_tables as gt
from pycalima.pahchem import population as pop
from pycalima.pahchem.baseline import Params


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args(argv)
    lay = CarbonLayout()
    nc = lay.grid().m_edges[:lay.n_pah_bins + 1] / M_C
    lib = pop.library_v400()
    a.out.mkdir(parents=True, exist_ok=True)
    for lo, hi in zip(nc[:-1], nc[1:], strict=True):
        label = f"bin_nc{lo:.0f}-{hi:.0f}"
        w = gt.nc_range_weights(lib, lo, hi * (1 + 1e-9) if hi == nc[-1] else hi)
        tab = gt.build_bin_table(lib, w, params=Params(), group_L0=np.array(AGN_GROUP_L0),
                                 group_L1=np.array(AGN_GROUP_L1), workers=a.workers, label=label)
        path = gt.write_bin_table(tab, a.out / f"{label}_group.dat")
        print(path, f"({tab['n_mol']} molecules, <N_C> {tab['nc']:.1f})", flush=True)


if __name__ == "__main__":
    main()
