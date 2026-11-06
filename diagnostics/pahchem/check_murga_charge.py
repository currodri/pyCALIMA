#!/usr/bin/env python3
"""The WD01 charge of Murga+2020/2022 (pahchem.murga_charge) against their charges.

Prints:
1. Murga+2022 Figshare f(a, Z) at every location (158) and size (10) against
   ``f_wd01``, by region, for the MURGA2022 configuration and one-at-a-time
   variants of each choice the papers do not state;
2. Murga+2020 Fig. 3 (integer <Z> of C24H12, C66H20, C128H28 against A_V,
   Orion Bar isochoric and Horsehead), as rounded mean and as mode;
3. a cross-check against pahchem's own BT94 charge ladder at a few M22 points.

Measured (2026-10-05), M22, excluding the HII points (EUV not modelled):

| configuration | median dZ PDR 1 / PDR 2 / DF / MC | max abs(df) PDR 1 / PDR 2 / DF / MC | mode agreement |
|---|---|---|---|
| MURGA2022 (MMP83, s_e = 1, no ions, eq. 24, truncated) | -0.04 / -0.03 / -0.00 / -0.00 | 0.13 / 0.05 / 0.03 / 0.03 | 0.92-1.00 |
| WD01 sticking law | +0.47 / +0.77 / +0.28 / +0.05 | 0.46 / 0.43 / 0.33 / 0.32 | 0.26-0.97 |
| Draine 1978 shape | +0.03 / +0.18 / +0.05 / +0.01 | 0.13 / 0.12 / 0.12 / 0.12 | 0.80-1.00 |
| C+ collisions on | -0.04 / -0.02 / +0.03 / +0.08 | 0.13 / 0.05 / 0.08 / 0.30 | 0.92-1.00 |
| graphite Mie Q_abs | -0.19 / -0.37 / -0.06 / -0.02 | 0.26 / 0.35 / 0.24 / 0.22 | 0.54-1.00 |
| Z_min rounded down (pyCALIMA) | -0.04 / -0.06 / -0.00 / -0.00 | 0.13 / 0.80 / 0.98 / 1.00 | 0.80-0.93 |

- PDR 1 retains -0.1 to -0.15 at 12-20 A, and up to -0.6 on the n_e rise at
  126-128 arcsec. The Figshare chi has 2 significant figures, which smears
  the distance (hence n_e) assignment there.
- HII: median dZ -0.97, max abs(df) 0.57 (xfail: M22's added EUV).
- The pyCALIMA solver's bounds finder overflows at chi <~ 0.01 and T ~ 13 K and
  puts 8.7 A grains at Z = -2 (Murga: -1): f_wd01 solves the chain on the fixed
  WD01 range instead.

M20 Fig. 3 (66 points), hits of rounded mean / mode:
- MURGA2022 configuration: 48 / 47.
- WD01 sticking, C+: 55 / 56.
- Z_min rounded down, s_e = 0.5, C+: 57 / 57 (best of those tried).
- No variant gives Orion C128H28 +2 at A_V = 0.1 or +1 at A_V = 5. The
  Horsehead C66H20 anion needs Z_min rounded down, which the M22 files
  exclude, so M20 and M22 ran different charge conventions.
- Rounded mean and mode do equally well (within one point).

Cross-check, pahchem BT94 ladder (Li & Draine sigma, Jochims yield, s_e = 1)
against MURGA2022 <Z>: in PDR 1 +0.0 to +0.3 high; in the DF and MC the
ladder makes C24 and C60 anions (<Z> ~ -0.9), which WD01 eq. 24 forbids
below 6 A.

Needs the murga2022-figshare and murga2020-figshare datasets. Takes ~2 min.
"""
import argparse

import numpy as np
import pandas as pd

from pycalima._paths import get_external_data_path
from pycalima.pahchem import murga_charge as C
from pycalima.pahchem import murga_data as M

VARIANTS = {"MURGA2022": {},
            "WD01 sticking law": {"sticking": "wd01"},
            "Draine 1978 shape": {"field": "Draine"},
            "C+ collisions on": {"ions": True},
            "graphite Mie Q_abs": {"cabs": "graphite"},
            "Z_min rounded down": {"z_min": "floor"}}


def m22_table(overrides=None, model="b"):
    """One row per (location, size): region, <Z> ours and Murga, max abs(df), modes."""
    o = dict(overrides or {})
    ions = o.pop("ions", False)
    cfg = {**C.MURGA2022, **o}
    d = M.read_model_2022(model)
    p = pd.read_csv(get_external_data_path("pahchem/murga2022_fig6_profile.csv"), comment="#")
    order = np.argsort(d["z"])
    zg = d["z"][order].astype(int)
    rows = []
    for loc in range(M.N_LOC):
        for m, a in enumerate(d["a_cm"]):
            fm = d["f"][loc, :, 0, 0, m][order]
            Z, f = C.f_wd01(a, p.chi[loc], p.t_gas[loc], p.n_e[loc], p.n_cplus[loc] if ions else 0.0,
                            z_range=C.Z_GRID_2022, **cfg)
            rows.append((loc, C.region_of(p.distance_arcsec[loc]), a * 1e8, float(Z @ f), float(zg @ fm),
                         float(np.abs(f - fm).max()), int(Z[np.argmax(f)]), int(zg[np.argmax(fm)])))
    t = pd.DataFrame(rows, columns=["loc", "region", "a_A", "z_ours", "z_murga", "max_df", "mode_ours",
                                    "mode_murga"])
    t["dz"] = t.z_ours - t.z_murga
    return t


def summary(t):
    g = t.groupby("region").agg(median_dz=("dz", "median"), max_df=("max_df", "max"))
    g["mode_agreement"] = (t.mode_ours == t.mode_murga).groupby(t.region).mean()
    return g


def m20_fig3(sticking=1.0, ions=False, z_min="int"):
    """{(site, molecule): (theirs, rounded mean, mode)} over the 11 A_V of Fig. 3."""
    fig3 = pd.read_csv(get_external_data_path("pahchem/murga2020_fig3_mean_charge.csv"), comment="#")
    out = {}
    for site in ("orion_isochoric", "horsehead"):
        p = M.read_profile_2020(site)
        for mol, nc in (("C24H12", 24), ("C66H20", 66), ("C128H28", 128)):
            got_r, got_m = [], []
            for i in range(p["av"].size):
                ne = p["x_e"][i] * p["n_tot"][i]
                ncp = p["x_cplus"][i] * p["n_tot"][i] if ions else 0.0
                Z, f = C.f_wd01(C.radius_cm(nc), p["chi"][i], p["t_gas"][i], ne, ncp, field="Mathis",
                                sticking=sticking, z_min=z_min)
                zbar, zmode = C.mean_and_mode(Z, f)
                got_r.append(int(np.round(zbar)))
                got_m.append(zmode)
            theirs = fig3[(fig3.site == site) & (fig3.molecule == mol)].sort_values("av").mean_z.to_numpy()
            out[(site, mol)] = (theirs, np.array(got_r), np.array(got_m))
    return out


def main():
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    for name, o in VARIANTS.items():
        print(f"\n== M22, {name}\n{summary(m22_table(o)).round(3).to_string()}", flush=True)
    for label, kw in (("MURGA2022", {}), ("WD01 sticking, C+", dict(sticking="wd01", ions=True)),
                      ("Z_min down, s_e 0.5, C+", dict(sticking=0.5, ions=True, z_min="floor"))):
        res = m20_fig3(**kw)
        hr = sum(int((r == t).sum()) for t, r, _ in res.values())
        hm = sum(int((m == t).sum()) for t, _, m in res.values())
        print(f"\n== M20 Fig. 3, {label}: rounded {hr}/66, mode {hm}/66")
        for (site, mol), (t, r, m) in res.items():
            print(f"  {site:16s} {mol:8s} theirs {t.tolist()} round {r.tolist()} mode {m.tolist()}")
    p = pd.read_csv(get_external_data_path("pahchem/murga2022_fig6_profile.csv"), comment="#")
    print("\n== cross-check: <Z> of WD01 (MURGA2022) vs pahchem BT94 ladder, N_C = 24, 60, 128")
    for loc in (20, 64, 72, 100):
        cells = []
        for nc in (24, 60, 128):
            Zw, fw = C.f_murga2022(C.radius_cm(nc), p.chi[loc], p.t_gas[loc], p.n_e[loc])
            Zb, fb = C.f_bt94_ladder(nc, p.chi[loc], p.t_gas[loc], p.n_e[loc])
            cells.append(f"C{nc}: {Zw @ fw:+.2f} / {Zb @ fb:+.2f}")
        print(f"  loc {loc:3d} ({C.region_of(p.distance_arcsec[loc])}): " + "   ".join(cells))


if __name__ == "__main__":
    main()
