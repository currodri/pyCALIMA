"""Do we recover the published critical metallicities with the published dust?

Each case rebuilds the dust of one paper (species, size distribution, dust per
metal) and finds our [Z_cr] by criteria ``C`` and ``frag`` (``critical_dust_to_gas``,
static dust), on Omukai+05's dust-free track interpolated in [Z/H] (the gas model
of all but Chiaki+15, who used the Z = 0 track's physics with SN yields: ours on
the Z = 0 track, as ``science_grid_dust_evolution.py``'s n0 ICs).

* O05: Omukai+05 (astro-ph/0503010; grains of Omukai 2000 Sect. 2.2.2, Pollack+94):
  refractory part only, ice and volatile organics evaporated: olivine 2.51e-3
  (as Mg2SiO4), orthopyroxene 7.33e-4 (MgSiO3), troilite 5.69e-4 (FeS), iron
  2.53e-4 (Fe), refractory organics 3.53e-3 (as amorphous C) per gas mass at
  Z = 0.02; a^-3.5 from 5 nm to 1 um, a^-5.5 to 5 um. Fragments appear at -6 and
  dominate from -5 (their elongation criterion). Clark+08 used their equation of
  state, and Dopcke+11,13 and Chon+21 Milky Way dust scaled with Z: no extra case.
* S03: Schneider+03 (astro-ph/0304254): the same grains with f_dep = 0.2
  fragment at [Z/H] = -5.1.
* S06: Schneider+06 (astro-ph/0603766) Table 1 species of the 195 Msun PISN
  (f_dep 0.65, S = 5.9e4) and the 22 Msun SN II (f_dep 0.24, S = 7.6e5), one
  lognormal (sigma 0.7) of that S for every species; Z_cr = 1e-6 Z_sun (gamma = 1).
* S12a: Schneider+12a (1109.2900) Table 1 (20 Msun, Z = 0) without reverse shock:
  amorphous C only, f_dep 0.91, S = 4.71e5; lowest Z that fragments 1e-6 (their
  grid is in steps of 1 dex: between -7 and -6), and D_cr = 1.4e-3 cm^2/g / S
  = 3.0e-9 by their Eq. 16.
* Ji14: Ji, Frebel & Bromm 2014 (1307.2239): silicate dust holding all the Si
  (SiO2 0.45, Mg2SiO4 0.30, Si 0.15, FeS 0.10 by mass, within their Table 1),
  ``standard`` (Pollack sizes) or ``shock`` (a^-5.5 above 5 nm) sizes, T_CMB = 50 K;
  [Si/H]_cr from our D_cr by their Eq. 16 (Asplund+09 Si): they find -4.39 to
  -4.63 and -5.17 to -5.37 over their compositions.
* C15: Chiaki+15 (1410.8384) Table 3 (n_amb = 0) and 4: the N07 CCSN dust
  (``nozawa_ccsn``) with the carbon and Mg-silicate grains reshaped to their S_C,
  S_Sil; their [Z_cr] without grain growth.

    python diagnostics/collapse/check_literature_zcr.py [--workers 8]

Writes ``collapse/check_literature_zcr.txt`` under the results directory.
"""

from __future__ import annotations

import argparse
import math
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from pycalima import _paths
from pycalima.collapse.grid import SizeGrid
from pycalima.collapse.ics import RHO_PER_H, Z_SUN, lognormal_from_moments, nozawa_ccsn
from pycalima.collapse.integrator import Population, critical_dust_to_gas
from pycalima.collapse.materials import get_material
from pycalima.collapse.tracks import TRACK_COLUMNS, Track

UM = 1e-4
A_MIN, A_MAX, N_BINS = 1e-8, 1e-3, 50
SI_SUN = -4.49                      # log n_Si / n_H, Asplund+09
POLLACK = [(-3.5, 0.005 * UM, 1.0 * UM), (-5.5, 1.0 * UM, 5.0 * UM)]
SHOCK = [(-5.5, 0.005 * UM, 5.0 * UM)]
O05 = {"Mg2SiO4": 2.51e-3, "MgSiO3": 7.33e-4, "FeS": 5.69e-4, "Fe": 2.53e-4, "C": 3.53e-3}
JI14 = {"SiO2": 0.45, "Mg2SiO4": 0.30, "Si": 0.15, "FeS": 0.10}
S06 = {"PISN195": ({"Al2O3": 5.82e-4, "Fe3O4": 1.33e-2, "Mg2SiO4": 0.217, "SiO2": 0.705, "C": 6.41e-2}, 0.65, 5.9e4),
       "SNII22": ({"Al2O3": 5.82e-3, "Fe3O4": 0.288, "MgSiO3": 5.24e-2, "Mg2SiO4": 0.343, "C": 0.311}, 0.24, 7.6e5)}
SI_MASS = {"SiO2": 28.086 / 60.08, "Mg2SiO4": 28.086 / 140.69, "Si": 1.0, "FeS": 0.0}


def omukai_track(lz):
    """Omukai+05 dust-free track at [Z/H] = lz, linear in [Z/H] between their -6, -4, -2, 0
    (-6 below), with their Z = 0 H2 fraction."""
    d = np.loadtxt(_paths.get_external_data_path("collapse_tracks", "omukai2005_dustfree.txt"))
    zs = np.array([-6.0, -4.0, -2.0, 0.0])
    cols = [d[:, TRACK_COLUMNS[k]] for k in ("m6", "m4", "m2", "0")]
    x = min(max(lz, zs[0]), zs[-1])
    i = min(int(np.searchsorted(zs, x, side="right")) - 1, len(zs) - 2)
    w = (x - zs[i]) / (zs[i + 1] - zs[i])
    a, b = cols[i], cols[i + 1]
    logT = np.where(np.isfinite(a) & np.isfinite(b), (1 - w) * a + w * b, np.where(np.isfinite(a), a, b))
    return Track(d[:, 0], logT, d[:, 6], name=f"Omukai+05 [Z/H] = {lz:.2f}")


def powerlaw(segments):
    """Mass fraction per bin of dn/da ∝ a^q over the segments (q, a_lo, a_hi), continuous."""
    def w(grid):
        ae = grid.a_edges
        out, A = np.zeros(len(ae) - 1), 1.0
        for k, (q, lo, hi) in enumerate(segments):
            if k:
                A *= lo ** (segments[k - 1][0] - q)
            l, h = np.clip(ae[:-1], lo, hi), np.clip(ae[1:], lo, hi)
            out += A * (h ** (4 + q) - l ** (4 + q)) / (4 + q)
        return out / out.sum()
    return w


def lognormal_S(S, sigma=0.7):
    """Mass fraction per bin of a lognormal of width sigma with cross-section per mass S."""
    def w(grid):
        a_S = 3.0 / (4.0 * grid.rho_s * S)
        mu = math.log(a_S) - 2.5 * sigma ** 2
        f = lognormal_from_moments(grid, math.exp(2 * mu + 2 * sigma ** 2), math.exp(3 * mu + 4.5 * sigma ** 2))
        return f / f.sum()
    return w


def dust(D, parts):
    """Populations holding dust-to-gas mass ratio D, split as {species: (mass fraction, weights)}."""
    tot = sum(f for f, _ in parts.values())
    pops = []
    for sp, (f, w) in parts.items():
        g = SizeGrid.from_radii(A_MIN, A_MAX, N_BINS, get_material(sp).rho_s)
        pops.append(Population(sp, g, D * RHO_PER_H * f / tot * w(g)))
    return pops


def reshaped_chiaki(M, n_amb, Z, S_C, S_Sil):
    pops = nozawa_ccsn(M, Z, n_amb=n_amb or None)[0]
    out = []
    for p in pops:
        S = S_C if p.material == "C" else S_Sil if p.material in ("Mg2SiO4", "MgSiO3") else None
        x = p.x if S is None or not np.isfinite(S) else p.x.sum() * lognormal_S(S)(p.grid)
        out.append(Population(p.material, p.grid, x, p.table))
    return out


def chiaki_models():
    d = np.loadtxt(_paths.get_external_data_path("collapse_ics", "chiaki2015_n07_reverse_shock.txt"))
    return {(int(r[0]), float(r[1])): dict(S_C=r[3] * 1e4, S_Sil=r[5] * 1e4, f_dep=r[6], Zcr=r[9])
            for r in d if r[0] <= 30}


def cases():
    out = []     # (name, quoted, make(x) -> pops, track(x), x is "Z" (Z/Z_sun) or "D", T_rad)
    pw = powerlaw(POLLACK)
    o05 = {sp: (f, pw) for sp, f in O05.items()}
    f_o05 = sum(O05.values()) / Z_SUN
    out.append(("O05 Pollack (f_dep %.2f)" % f_o05, "-6 onset, -5 dominant",
                lambda Z: dust(Z * Z_SUN * f_o05, o05), "Z", 2.725))
    out.append(("S03 Pollack, f_dep 0.2", "-5.1", lambda Z: dust(Z * Z_SUN * 0.2, o05), "Z", 2.725))
    for k, (fr, fd, S) in S06.items():
        parts = {sp: (f, lognormal_S(S)) for sp, f in fr.items()}
        out.append((f"S06 {k} (f_dep {fd}, S {S:.1e})", "-6", lambda Z, p=parts, fd=fd: dust(Z * Z_SUN * fd, p),
                    "Z", 2.725))
    out.append(("S12a 20 Msun Z=0 norev (C, f_dep 0.91, S 4.7e5)", "-7 to -6",
                lambda Z: dust(Z * Z_SUN * 0.91, {"C": (1.0, lognormal_S(4.71e5))}), "Z", 2.725))
    out.append(("S12a same, D_cr", "D_cr 3.0e-9", lambda D: dust(D, {"C": (1.0, lognormal_S(4.71e5))}), "D", 2.725))
    for k, seg, q in (("standard", POLLACK, "-4.39 to -4.63"), ("shock", SHOCK, "-5.17 to -5.37")):
        parts = {sp: (f, powerlaw(seg)) for sp, f in JI14.items()}
        out.append((f"Ji14 silicates, {k} sizes: [Si/H]", q, lambda D, p=parts: dust(D, p), "D", 50.0))
    for (M, na), m in chiaki_models().items():
        out.append((f"C15 N07M{M}n{na:g}", f"{m['Zcr']:.1f}",
                    lambda Z, M=M, na=na, m=m: reshaped_chiaki(M, na, Z, m["S_C"], m["S_Sil"]), "C15", 2.725))
    return out


def one(job):
    i, crit = job
    name, quoted, make, kind, T_rad = cases()[i]
    if kind == "C15":
        track = Track.omukai2005("Z0")
    elif kind == "Z":
        track = lambda Z: omukai_track(math.log10(Z))
    else:
        track = omukai_track(-5.0)
    mk = lambda x: (make(x), None)
    kw = dict(accretion=False, dlnn=0.1, n0=1e-1, T_rad=T_rad)
    if kind == "D":
        lo, hi = 1e-12, 1e-4
    else:
        lo, hi = 1e-8, 10 ** -2.5
    if crit == "frag":
        x = critical_dust_to_gas(mk, track, criterion="frag", lo=lo, hi=hi, scan_dex=0.25, rtol=0.05, **kw)
    else:
        x = critical_dust_to_gas(mk, track, criterion="C", lo=lo, hi=hi, rtol=0.05, **kw)
    return job, x


def value(name, kind, x):
    if math.isinf(x):
        return "none"
    if kind != "D":
        return f"{math.log10(x):.2f}"
    if name.startswith("Ji14"):
        si = sum(f * SI_MASS[sp] for sp, f in JI14.items()) / sum(JI14.values())
        return f"{math.log10(x * RHO_PER_H * si / (28.086 * 1.6605e-24)) - SI_SUN:.2f}"
    return f"{x:.1e}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    cs = cases()
    jobs = [(i, c) for i in range(len(cs)) for c in ("C", "frag")]
    with ProcessPoolExecutor(a.workers) as ex:
        res = dict(ex.map(one, jobs))
    lines = ["# Published critical metallicities against ours with the same dust, "
             "diagnostics/collapse/check_literature_zcr.py",
             "# case | quoted | ours criterion C | ours criterion frag   ([Z/H], Z_sun = 0.02; Ji14: [Si/H]; D: D_cr)"]
    for i, (name, quoted, _, kind, _) in enumerate(cs):
        lines.append(f"{name:52s} | {quoted:16s} | {value(name, kind, res[(i, 'C')]):>8s} | "
                     f"{value(name, kind, res[(i, 'frag')]):>8s}")
    out = _paths.get_results_dir() / "collapse" / "check_literature_zcr.txt"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
