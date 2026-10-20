#!/usr/bin/env python3
"""PROTOTYPE: group-independent ("universal") WDB06 charging tables.

The format-5 tables (charging_psi_tables) are built on a run's own RT groups, so a run with
other group edges needs its own tables. Here the same table is built once on a fixed template:
fine sub-groups inside each band (SOFT 5.6-11.2, HARD 11.2-13.6, EUV 13.6-24.59, EUV_HE
24.59-54.42, EUV_HEP 54.42-500 eV) with a T_template blackbody inside each band. A run then
needs only its band intensities and hardnesses (cell_inputs), whatever its groups are.

    python prototype_universal_psitab.py --out DIR [--T-template 4e4] [--workers 8] [--bins 1 2 3 4] [--family]

--family: the five-parameter table (T, psi_F, h, h_E, psi_E). At each hardness node the shape
inside the FUV bands is that of the blackbody with that h, and inside the EUV bands (with the
split above 54.42 eV) that of the blackbody with that h_E = N(>= 24.59 eV)/N(>= 13.6 eV).

writes DIR/dust_charging_psitab_DustBin_0b.{dat,bin}, readable by charging_psi_tables.read_psi_table.
Not part of the package: it exists to measure the error of the template assumption.
"""
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from pycalima._paths import get_model_data_dir
from pycalima.models.dust_charge import charging_psi_tables as cpt
from pycalima.models.dust_charge import rt_group_charging as rg

# sub-group edges per band [eV]; consecutive bands share an edge
TEMPLATE_EDGES = {
    "SOFT": [5.6, 7.0, 8.6, 11.2],
    "HARD": [11.2, 12.4, 13.6],
    "EUV": [13.6, 15.2, 17.5, 20.5, 24.59],
    "EUV_HE": [24.59, 31.0, 39.0, 54.42],
    "EUV_HEP": [54.42, 80.0, 130.0, 250.0, 500.0],
}


def template_groups():
    L0, L1 = [], []
    for edges in TEMPLATE_EDGES.values():
        L0 += edges[:-1]
        L1 += edges[1:]
    return np.array(L0), np.array(L1)


FAMILY_T = np.geomspace(3e3, 1e8, 400)    # blackbody temperatures of the spectral family [K]


def family_temperatures(L0, L1):
    """h(T) of the FUV and h_E1(T) of the EUV for the family's blackbodies on the template
    groups, increasing with T: the inverses give the blackbody of a hardness node."""
    roles = cpt.group_roles(L0, L1)
    hF, hE = [], []
    for T in FAMILY_T:
        kT = rg.KB_CGS * T / rg.EV2ERG
        n = np.array([np.trapezoid(E ** 2 / np.expm1(np.minimum(E / kT, 700.0)), E)
                      for E in (np.geomspace(l0, l1, 400) for l0, l1 in zip(L0, L1))])
        s, h = n[roles == cpt.SOFT].sum(), n[roles == cpt.HARD].sum()
        hF.append(h / (s + h))
        hE.append(cpt._hardness(n, roles)[0])
    return np.array(hF), np.array(hE)


def _family_row(args):
    """All (psi_F, h, h_E, psi_E) points of one temperature: the in-band shape of the FUV
    bands from the blackbody of h, of the EUV bands (and h_E2) from the blackbody of h_E."""
    tab, egy, lt, grid, hF_T, hE_T = args
    roles = cpt.group_roles(tab["L0"], tab["L1"])
    gt = rg.GroupChargingTable(tab, egy=egy, kernels="pchip")
    T = 10.0 ** lt
    psi_E = np.concatenate([[0.0], 10.0 ** grid["lpsiE"]])
    fuv = (roles == cpt.SOFT) | (roles == cpt.HARD)
    out = np.zeros((len(cpt.QUANTITIES), grid["lpsiF"].size, grid["sh"].size, grid["hE1"].size, 1, psi_E.size))
    for m, sh in enumerate(grid["sh"]):
        TF = np.interp(sh ** 2, hF_T, FAMILY_T)
        wF = cpt.band_shapes(tab["L0"], tab["L1"], TF)[1]
        for e1, hE1 in enumerate(grid["hE1"]):
            TE = np.interp(hE1, hE_T, FAMILY_T)
            wE = cpt.band_shapes(tab["L0"], tab["L1"], TE)[1]
            w = np.where(fuv, wF, wE)
            hE2 = cpt.reference_hardness(tab["L0"], tab["L1"], TE)[1]
            for k, pe in enumerate(psi_E):
                zg = None
                for j, lpf in enumerate(grid["lpsiF"]):
                    out[:, j, m, e1, 0, k], zg = cpt._point(gt, roles, w, T, 10.0 ** lpf, sh ** 2, pe, hE1, hE2, zg)
    return out


def build_family(tab, workers, grid):
    """Five-parameter table (T, psi_F, h, h_E, psi_E): no template temperature, no h_E2 axis."""
    hF_T, hE_T = family_temperatures(tab["L0"], tab["L1"])
    egy = cpt.band_shapes(tab["L0"], tab["L1"], cpt.T_EUV)[2]   # sub-group energies (narrow sub-groups)
    args = [(tab, egy, lt, grid, hF_T, hE_T) for lt in grid["lT"]]
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(_family_row, args))
    pt = cpt.build_psi_table(tab, T_bb=cpt.T_EUV, egy=egy, workers=1,
                             grid=dict(grid, lT=grid["lT"][:1], lpsiF=grid["lpsiF"][:1], sh=grid["sh"][:1],
                                       hE1=grid["hE1"][:1], hE2=np.array([0.0]), lpsiE=grid["lpsiE"][:1]))
    pt.update(lT=np.array(grid["lT"]), lpsiF=np.array(grid["lpsiF"]), sh=np.array(grid["sh"]),
              hE1=np.array(grid["hE1"], float), hE2=np.array([0.0]),
              psiE=np.concatenate([[0.0], 10.0 ** np.asarray(grid["lpsiE"])]),
              q=np.stack(rows, axis=1).astype(np.float32).astype(np.float64))
    return pt


def build(b, T_template, out, workers, family=False):
    """Universal table of dust bin b (1-based), with the bin's material and size from the
    standard rt-group tables."""
    ref = rg.read_bin_table(f"{get_model_data_dir()}/dust_charging_rtgroups_data/dust_charging_rtgroups_DustBin_0{b}.dat")
    L0, L1 = template_groups()
    tab = rg.build_bin_table(ref["material"], ref["a"], group_L0=L0, group_L1=L1, label=f"DustBin_0{b}")
    t0 = time.time()
    if family:
        grid = dict(lT=cpt.LT, lpsiF=cpt.LPSI_F, sh=cpt.SQRT_H, hE1=cpt.H_E1, lpsiE=cpt.LPSI_E)
        pt = build_family(tab, workers, grid)
        label = "universal, blackbody family"
    else:
        pt = cpt.build_psi_table(tab, T_bb=T_template, egy=None, workers=workers, euv_hardness=True)
        label = f"universal T={T_template:g} K"
    os.makedirs(out, exist_ok=True)
    cpt.write_psi_table(pt, f"{out}/dust_charging_psitab_DustBin_0{b}.dat", label=label)
    return time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--T-template", type=float, default=cpt.T_EUV)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--bins", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--family", action="store_true",
                    help="in-band shapes from the blackbody of each hardness node (no template T, no h_E2)")
    a = ap.parse_args()
    for b in a.bins:
        print(f"bin {b}: {build(b, a.T_template, a.out, a.workers, a.family):.0f} s", flush=True)


if __name__ == "__main__":
    main()
