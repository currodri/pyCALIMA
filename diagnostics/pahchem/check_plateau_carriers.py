#!/usr/bin/env python3
"""Can PAH clusters, anharmonic bands or a broad population make the 10-15 um plateau?

Plateau metric, the same for templates and models: the share of the 10.9-14.45
um PAH power under the grey opening of the spectrum (F_nu against lambda,
structuring element 0.4 um, wider than any band). A morphological opening
keeps whatever stays up between bands and removes the bands. For the
templates, the PAH spectrum is the line-cleaned template minus the Chown+24
spline (as compare_pdrs4all); for models, the band spectrum on the same
lambda grid.

Emission: pahchem's exact cascade under the Kurucz 40 000 K field, Lorentzian
FWHM 15 cm^-1 shifted -15 cm^-1. Charge states as the population fit gives
them (least-ionising corner of the Atomic PDR); clusters are neutral (no
cluster IPs in the library) and are compared with their monomers run alike.

Cases:
1. single PAHdb v4.00 molecules (every charge state of ``library_v400``);
2. the whole library mixed: equal numbers of each molecule, charges from the
   conditions;
3. the best-fit population of the v4.00 scan (cache pahchem_population_scan_v400_base.pkl);
4. clusters of pyrene, coronene, ovalene, TBPP, mixed per monomer and in
   total, against their monomers;
5. anharmonic (0 K absorption) against harmonic v4.00, same bond graph and
   charge: the off-fundamental intensity share, the centroid shifts, and the
   metric on cold absorption spectra.

    check_plateau_carriers.py --templates .../orionbar.dat
"""
import argparse
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import grey_opening

import compare_pdrs4all as C
from pycalima import _paths
from pycalima.pahchem import ames_pahdb, conditions, emission, population
from pycalima.pahchem.baseline import E_PH

SHAPE = ("kurucz", 40000)
LAM = np.arange(10.5, 15.0, 0.002)
RANGE = (10.9, 14.45)
OPEN_UM = 0.4


def plateau_share(lam, fnu):
    n = int(round(OPEN_UM / np.median(np.diff(lam)))) | 1
    base = grey_opening(np.clip(fnu, 0.0, None), size=n)
    s = (lam >= RANGE[0]) & (lam <= RANGE[1])
    nu = 1e4 / lam[s]
    tot = np.trapezoid(fnu[s][::-1], nu[::-1])
    return float(np.trapezoid(base[s][::-1], nu[::-1]) / tot)


def model_fnu(freqs, strength, shift_cm=-15.0):
    """Band spectrum per cm^-1 (= F_nu up to a constant) on LAM."""
    return emission.spectrum(freqs, strength, 1e4 / LAM, fwhm_cm=15.0, shift_cm=shift_cm)


def neutral_emitted(sp):
    return population.build_state_basis(sp.freqs_cm, sp.einstein_a, sp.nc, sp.nh, 0, np.zeros_like(E_PH), SHAPE,
                                        100.0, 16.0 * 8065.544, (4.41, 12.0), 1.0).emitted


def pct(v):
    v = np.asarray(v)
    return f"median {np.median(v):.2f} [16-84 % {np.percentile(v, 16):.2f}, {np.percentile(v, 84):.2f}] (n {v.size})"


def centroid(f, i, lo, hi):
    m = (f >= lo) & (f < hi)
    return float((f[m] * i[m]).sum() / i[m].sum()) if i[m].sum() > 0 else np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--templates", type=Path, required=True)
    ap.add_argument("--open-um", type=float, default=OPEN_UM, help="structuring element of the opening [um]")
    args = ap.parse_args()
    globals()["OPEN_UM"] = args.open_um
    lam, tmpl = C.load_templates(args.templates)
    print(f"plateau share of the {RANGE[0]}-{RANGE[1]} um PAH power (grey opening, {OPEN_UM} um):")
    for t, f in tmpl.items():
        lm, fm = C._clean(lam[lam < 20], f[lam < 20], 0.025)
        cont = CubicSpline(C.ANCHORS, [np.median(fm[np.abs(lm - a) < 0.004]) for a in C.ANCHORS])(lm)
        print(f"  observed {t:6s} {plateau_share(LAM, np.interp(LAM, lm, fm - cont)):.2f}")
    lib = population.library_v400(cache_dir=_paths.get_model_data_dir() / "pahchem_pahdb_index")
    pop = population.Population(lib, {"x": (10.0, 15.0)}, cache_dir=_paths.get_model_data_dir() / "pahchem_population_basis")
    pop.prepare(SHAPE)
    env = conditions.bracketing_environments("orion_bar", "Atomic")["least ionising"]
    f = pop.charge_matrix(env)
    spec = np.zeros((len(lib), LAM.size))
    singles = []
    for s, m in enumerate(lib):
        for z in m.charges:
            b = pop.basis(m, z, SHAPE)
            sz = model_fnu(b.freqs_cm, b.emitted)
            spec[s] += f[s, z - pop.zmin] * sz
            singles.append(plateau_share(LAM, sz))
    print(f"  single PAHdb v4.00 states: {pct(singles)}")
    print(f"  whole v4.00 library, one of each molecule: {plateau_share(LAM, spec.sum(axis=0)):.2f}")
    cache = _paths.get_results_dir() / "pahchem_population_scan_v400_base.pkl"
    if cache.exists():
        with open(cache, "rb") as fh:
            res = pickle.load(fh)
        names = [x[0] for x in res["library"]]
        idx = {m.name: i for i, m in enumerate(lib)}
        for t in ("Atomic", "DF3"):
            best = min(res[t], key=lambda c: c["spline"]["chi2"])
            w = np.zeros(len(lib))
            for n, wt in zip(names, best["spline"]["weights"]):
                if n in idx:
                    w[idx[n]] = wt
            print(f"  best-fit population, {t} (v4.00 scan, spline): {plateau_share(LAM, w @ spec):.2f}")
    # -- clusters ------------------------------------------------------------------------------------
    cl = ames_pahdb.read_xml(ames_pahdb.xml_path("pahdb-clusters-theoretical-v1-00"))
    by_m = defaultdict(list)
    for r in cl:
        sp = ames_pahdb.species_from_record(r, f"cluster {r['monomers']} {r['type']}")
        by_m[r["monomers"]].append((r["type"], r["conformation"], sp.nc, model_fnu(sp.freqs_cm, neutral_emitted(sp))))
    print("\nclusters (neutral), per molecule:")
    allc = np.zeros(LAM.size)
    allm = np.zeros(LAM.size)
    for m, v in by_m.items():
        mono = next(x for x in v if x[0] == "monomer")
        cls = [x for x in v if x[0] != "monomer"]
        mix = sum(x[3] / x[3].sum() for x in cls)
        allc += mix / len(cls)
        allm += mono[3] / mono[3].sum()
        print(f"  monomer N_C {mono[2]:3d}: monomer {plateau_share(LAM, mono[3]):.2f}; single clusters "
              f"{pct([plateau_share(LAM, x[3]) for x in cls])}; mixed clusters {plateau_share(LAM, mix):.2f}")
    print(f"  all four molecules: monomers mixed {plateau_share(LAM, allm):.2f}, clusters mixed {plateau_share(LAM, allc):.2f}")
    # -- anharmonic ----------------------------------------------------------------------------------
    an = [r for r in ames_pahdb.read_xml(ames_pahdb.xml_path("pahdb-anharmonic-v1-00")) if r["pure_ch"]]
    th = ames_pahdb.xml_index(ames_pahdb.xml_path("pahdb-theoretical-v4-00"), _paths.get_model_data_dir() / "pahchem_pahdb_index")
    by = defaultdict(list)
    for r in th:
        by[(r.get("graph"), r["charge"])].append(r)
    print("\nanharmonic (0 K absorption) against harmonic v4.00, same bond graph and charge:")
    print(f"  {'formula':8s} {'Z':>2s}  lines anh/harm  off-fundamental  centroid shift oop / CC / CH [cm^-1]  plateau harm / anh")
    for r in an:
        h = by.get((r["graph"], r["charge"]))
        if not h:
            print(f"  {r['formula']:8s} {r['charge']:+2d}  no harmonic v4.00 match")
            continue
        h = h[0]
        fa, ia, fh, ih = r["freqs_cm"], r["intensity"], h["freqs_cm"], h["intensity"]
        strong = fh[ih >= 0.01 * ih.sum()]
        off = ia[~np.any(np.abs(fa[:, None] - strong[None, :]) < 30.0, axis=1)].sum() / ia.sum()
        sh = [centroid(fa, ia, lo, hi) - centroid(fh, ih, lo, hi) for lo, hi in ((650, 950), (1500, 1650), (2950, 3150))]
        print(f"  {r['formula']:8s} {r['charge']:+2d}  {fa.size:6d}/{fh.size:<6d}  {off:10.3f}      "
              f"{sh[0]:+7.1f} {sh[1]:+7.1f} {sh[2]:+7.1f}                  "
              f"{plateau_share(LAM, model_fnu(fh, ih, 0.0)):.2f} / {plateau_share(LAM, model_fnu(fa, ia, 0.0)):.2f}")


if __name__ == "__main__":
    main()
