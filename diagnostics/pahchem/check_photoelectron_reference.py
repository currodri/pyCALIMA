#!/usr/bin/env python3
"""Where a charge state reads the measured photoelectron shares (``Params.photoelectron_reference``, gas-coupling section 2.2c).

``photo.photoelectron_energy`` holds Hrodmarsson+25's shares, measured on nine neutral C13-C20 PAHs at photon energies 7-22 eV.
"photon" (the default) reads them at the photon energy for every state; "neutral" (opt-in) reads a state of threshold T at
hv - (T - IP(0)), its molecule's neutral ladder shifted rigidly by the charge. Three checks:
1. The data: do the nine molecules (IE 7.00-7.91 eV) collapse better on hv or on the excess X = hv - IE?
2. The ladders (``--octopus-dir``, Octopus LDA ground states ``<dir>/results/dial3_library/<mol>_<Z>`` and
   ``<dir>/results/runs/coronene_<Z>_h0.25``, ``<dir>/dications_2026-10-30/coronene_2``): with the charge, do all occupied
   levels shift alike (the "neutral" reading), and among neutrals of different size, do they line up in absolute energy (the
   "photon" reading) or from the HOMO?
3. Gamma_PE and eps with both readings for C24H12-C216H36 (``check_charge_ladder``) in the fields of the photoelectric
   validations: V9 (Draine G0 1.7 with the old stars, T 100 K, gamma 1e2-1e6), ``check_dissociative_recombination``'s PDR,
   EUV and X-ray fields, and ``check_kshell_auger``'s F_nu ~ nu^-1 to 10 keV.
Nothing is fitted.

Result (2026-11-04):
1. Neither collapses cleanly: at hv 11.5-14 eV X is better (perylene - others: +0.003 on X, -0.03 on hv at 13.5-14 eV),
   at 14.5-19.5 eV hv is better (std across molecules 0.013-0.019 on hv, 0.016-0.024 on X). The dip in the share at
   hv ~ 14 eV (slow electrons: the sigma onset) sits at one photon energy in all nine, perylene included.
2. With the charge every level shifts alike: C54H18 -2.66 eV per charge, the same to +-0.03 eV from the HOMO to > 16 eV
   below it (C24 -3.68 +-0.05, C96 -2.09, C216 -1.46 +-0.01; the anion as much the other way), close to the DFT IP
   steps. Among neutrals the density of states per C lines up better in absolute energy (std 0.012 against 0.066 at
   -7 eV, 0.021 against 0.044 at -18 eV) while the HOMO rises 0.84 eV from C24 to C216: the deep (sigma) levels are anchored.
   So: across sizes the photon energy, across charges the excess over the state's own threshold, which is "neutral".
3. Gamma_PE "neutral"/"photon": V9 gamma 1e2 1.008-1.023 and 1e3 1.000-1.012 (only the anions' valence branch moves),
   >= 1e4 and the PDRs cut at 13.6 eV <= 1.003 (neutrals and cations unchanged, as the test asserts); EUV psi 1e4
   0.955-0.975, 1e5 0.77-0.79, 1e6 0.49-0.54; nu^-1 psi 1e4 0.93-0.95, 1e6 0.74-0.82; X-rays only >= 0.9994. <Z> is
   unchanged (yields untouched); the IR of ``ionisation_heating`` rises 0.3-1.7 %. "photon" gives states above IP 21.9 eV
   no hole, so their electron takes the whole excess, share 1 even 1 eV above threshold.

    check_photoelectron_reference.py [--octopus-dir DIR] [--workers 4] [--skip-heating]
"""
import argparse
import sys
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from pycalima.pahchem import hrodmarsson2025 as H


def collapse():
    """Std of the share across the molecules at matched points, on hv and on X = hv - 7.71 eV (their mean IE)."""
    data = {n: H.mean_share(n) for n in H.IE if n not in H.DEPLETED}
    ie_ref = np.mean([H.IE[n][0] for n in data])

    def smooth(x, y, grid, w=0.4):
        return np.array([np.mean(y[np.abs(x - g) <= w]) if (np.abs(x - g) <= w).sum() >= 3 else np.nan for g in grid])
    hv = np.arange(10.0, 22.01, 0.5)
    on_hv = np.array([smooth(*data[n], hv) for n in data])
    on_x = np.array([smooth(data[n][0] - H.IE[n][0], data[n][1], hv - ie_ref) for n in data])
    per = list(data).index("Perylene")
    rest = [i for i in range(len(data)) if i != per]
    print(f"1. Hrodmarsson+25 shares, {len(data)} molecules (mean IE {ie_ref:.2f} eV): std across them on hv / on X, "
          "perylene - the others on hv / on X")
    for j, g in enumerate(hv):
        print(f"   hv {g:5.1f}  X {g - ie_ref:5.2f}:  {np.nanstd(on_hv[:, j]):.4f} / {np.nanstd(on_x[:, j]):.4f}   "
              f"{on_hv[per, j] - np.nanmean(on_hv[rest, j]):+.3f} / {on_x[per, j] - np.nanmean(on_x[rest, j]):+.3f}")


def occupied(info):
    """Occupied Kohn-Sham levels of an Octopus ``static/info``, one per electron, highest first."""
    lines = info.read_text().splitlines()
    i = next(k for k, ln in enumerate(lines) if ln.startswith("Eigenvalues [eV]")) + 2
    eps = []
    for ln in lines[i:]:
        p = ln.split()
        if len(p) < 4:
            break
        eps += [float(p[2])] * int(round(float(p[3])))
    return np.sort(np.array(eps))[::-1]


def ladders(root):
    states = {"C24H12": {0: root / "results/runs/coronene_0_h0.25", 1: root / "results/runs/coronene_1_h0.25",
                         2: root / "dications_2026-10-30/coronene_2"}}
    for mol in ("C54H18", "C96H24", "C150H30", "C216H36"):
        states[mol] = {z: root / f"results/dial3_library/{mol}_{z:+d}" for z in (-1, 0, 1, 2)}
    nc = {"C24H12": 24, "C54H18": 54, "C96H24": 96, "C150H30": 150, "C216H36": 216}
    bins = [(0, 3), (3, 6), (6, 9), (9, 12), (12, 16), (16, 99)]
    print("\n2a. Octopus LDA: shift of each charge's occupied levels against the neutral's (matched from the bottom), by depth "
          "below the neutral HOMO: " + " | ".join(f"{a}-{b}" for a, b in bins[:-1]) + " | >16 eV")
    neutrals = {}
    for mol, d in states.items():
        lad = {z: occupied(p / "gs/static/info") for z, p in d.items() if (p / "gs/static/info").exists()}
        if 0 not in lad:
            print(f"   {mol}: no neutral ground state under {root}")
            continue
        neutrals[mol] = lad[0]
        e0 = lad[0][::-1]
        depth = lad[0][0] - e0
        for z, v in sorted(lad.items()):
            if z == 0:
                continue
            m = min(v.size, e0.size)
            shift = v[::-1][:m] - e0[:m]
            cells = [f"{np.mean(shift[(depth[:m] >= a) & (depth[:m] < b)]):+.2f}" if ((depth[:m] >= a) & (depth[:m] < b)).any()
                     else "  -  " for a, b in bins]
            print(f"   {mol} {z:+d}: " + " | ".join(cells))
    if len(neutrals) < 2:
        return
    grid = np.arange(-20.0, 0.01, 0.5)
    dos_ = lambda e, g: np.exp(-0.5 * ((g[:, None] - e[None, :]) / 0.5) ** 2).sum(axis=1) / (0.5 * np.sqrt(2 * np.pi))  # noqa: E731
    homo = {m: v[0] for m, v in neutrals.items()}
    ref = np.mean(list(homo.values()))
    rel = np.array([dos_(v - homo[m], grid) / nc[m] for m, v in neutrals.items()])
    ab = np.array([dos_(v - ref, grid) / nc[m] for m, v in neutrals.items()])
    print("2b. Neutrals: electrons per C per eV (0.5 eV Gaussian), std across molecules from the HOMO / in absolute energy; "
          "HOMO " + ", ".join(f"{m} {h:.2f}" for m, h in homo.items()) + " eV")
    for j in range(0, grid.size, 2):
        print(f"   {grid[j]:6.1f} eV: {rel[:, j].std():.3f} / {ab[:, j].std():.3f}")


def heating(workers):
    import check_charge_ladder as L
    import check_dissociative_recombination as DR
    import fit_pdrs4all_population as FP

    from pycalima.pahchem import population as P
    from pycalima.pahchem.baseline import Environment, Params

    envs = {f"V9 Draine gamma {g:.0e}": Environment("v9", 1.7, 50.0, 1.7 * np.sqrt(100.0) / g, 100.0, None)
            for g in (1e2, 1e3, 1e4, 1e5, 1e6)} | DR.envs()
    for psi in (1e4, 1e6):
        ne = 1e4 * 100.0 / psi
        envs[f"nu^-1 to 10 keV psi {psi:.0e}"] = Environment("pl", 1e4, 1e-3 * ne, ne, 1e4, None, field_e_max_ev=1e4,
                                                              ions=(("H+", ne),), field_powerlaw=1.0)
    pops = {r: P.Population(L.library(), {"all": (1.0, 1e4)}, params=replace(Params(), photoelectron_reference=r),
                            cache_dir=FP.BASIS_CACHE) for r in ("photon", "neutral")}
    zs = np.array(pops["photon"].zs, float)
    print("\n3. Gamma_PE neutral/photon, eps photon -> neutral, <Z> photon / neutral, IR power neutral/photon")
    for name, e in envs.items():
        shape = P.field_shape(e)
        for pop in pops.values():
            pop.prepare(shape, workers=workers)
            pop.prepare_responses(shape, workers=workers)
        g = {r: pop.photoelectric_heating(e) for r, pop in pops.items()}
        F = {r: pop.band_fluxes(e)[0][:, 0] for r, pop in pops.items()}
        zbar = {r: pop.charge_matrix(e) @ zs for r, pop in pops.items()}
        print(f"  {name}:")
        for s, mol in enumerate(pops["photon"].library):
            ir = F["neutral"][s] / F["photon"][s] if F["photon"][s] > 0 else np.nan
            print(f"    {mol.name:14s} Gamma x{g['neutral'][0][s] / g['photon'][0][s]:.4f}  eps {g['photon'][2][s]:.4f} -> "
                  f"{g['neutral'][2][s]:.4f}  <Z> {zbar['photon'][s]:6.3f} / {zbar['neutral'][s]:6.3f}  IR x{ir:.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--octopus-dir", type=Path, default=None, help="the pahspec run tree (Octopus ground states)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--skip-heating", action="store_true")
    a = ap.parse_args()
    warnings.simplefilter("ignore")
    collapse()
    if a.octopus_dir is not None:
        ladders(a.octopus_dir)
    else:
        print("\n2. skipped: no --octopus-dir")
    if not a.skip_heating:
        heating(a.workers)


if __name__ == "__main__":
    main()
