#!/usr/bin/env python3
"""The cosmic-ray-induced UV field (``pahchem.cr_uv``, gas-coupling section 2.2d): its photons, the atoms' rates against
Heays+17's (RAMSES-RTZ's), and what it does to PAHs in a dark cloud.

1. The field at zeta(H2) = 1e-16, x(H2) = 0.5 (Heays+17's dark cloud): photons generated per ionisation by band, the
   4 pi photon flux at 1-13.6 eV and 6-13.6 eV, its Habing-equivalent G0; photon conservation on the population grid.
2. With ``--leiden-zip`` (the Leiden database's all_cross_sections_text.zip, read in place): photoionisation rates of
   C, Mg, Si, S, Fe in this field, int F sigma dE, against Heays+17 Table 20, the values RTZ uses
   (rtz/cosmic_ray_ionization_module.f90). This field has dust alone in the denominator, Heays+17 dust and gas: H2's
   lines and continuum below 110 nm, which shield C (ionised below 110 nm) and S (120 nm) most.
3. PAHs in a dark cloud (n_H = 1e4 with n(H) = 1, x_e = 1e-7, T = 10 K; only this field) at zeta_H = 1e-17, 1e-16 and
   1e-15 (zeta(H2) = 2 zeta_H (1 + phi_s)), and 1e-19 as an almost dark reference: photoionisation and capture rates
   and <Z>, and the absorbed power per C.

Result (2026-11-04):
- 0.36 photons per ionisation (0.002 of them below 91.2 nm). At zeta(H2) = 1e-16, x(H2) = 0.5: 2.2e4 photons cm^-2 s^-1
  at 1-13.6 eV, a Habing-equivalent G0 of 1.8e-4. Photon conservation exact.
- Atoms against Heays+17 (RTZ): Si 1.13x, Mg 1.18x; C 1.38x (H2's shielding below 110 nm, not followed here); S 3.0x
  and Fe 5.7x (their narrow autoionising resonances against the lines, integrated at 0.001 nm by Heays+17; a smooth
  absorber such as a PAH does not see it). With Hensley & Draine (2023) dust instead, Si and Mg came out 1.4-1.6x.
- PAHs, dark cloud: with almost no field (zeta_H 1e-19) all are anions; the induced field detaches C24H12's, <Z> -0.69,
  -0.18, -0.02 at zeta_H 1e-17, 1e-16, 1e-15; C54H18 and C96H24 stay anions (-0.84 to -1.00). They absorb 3e-13 eV
  s^-1 per C at 1e-17 (x10 per decade).

    check_cr_uv.py [--leiden-zip PATH] [--workers 4]
"""
import argparse
import sys
import warnings
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import check_charge_ladder as L
import fit_pdrs4all_population as FP

from pycalima.pahchem import cr_uv, photo
from pycalima.pahchem import population as P

#: Heays+17 Table 20 photoionisation rates at zeta(H2) = 1e-16 [s^-1], as in RTZ, and the Leiden file of each
RTZ_RATES = {"C": (2.60e-14, "C/c.pi"), "Mg": (1.12e-14, "Mg/mg.pi"), "Si": (4.16e-13, "Si/si.pi"),
             "S": (7.91e-14, "S/s.pi"), "Fe": (4.81e-14, "Fe/fe.pi")}
NAMES = ("C24H12 (18)", "C54H18 (37)", "C96H24 (108)")


def leiden_sigma(zf, member):
    """(E [eV] ascending, sigma [cm^2]) of a Leiden .pi table (wavelengths in Angstrom)."""
    def numeric(tokens):
        try:
            [float(t) for t in tokens]
        except ValueError:
            return False
        return len(tokens) >= 3

    rows = [ln.split() for ln in zf.read(member).decode().splitlines()]
    rows = [r for r in rows if numeric(r)]
    lam = np.array([float(r[1]) for r in rows])
    sig = np.array([float(r[-1]) for r in rows])
    E = 12398.41984 / lam
    o = np.argsort(E)
    return E[o], sig[o]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--leiden-zip", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    warnings.simplefilter("ignore")
    lam, cum = cr_uv.heays17()
    print(f"1. Heays+17 P(lambda), digitised: {cum[-1]:.4f} photons per H2 ionisation to {lam[-1]:.0f} nm")
    for lo, hi in ((78, 91.2), (91.2, 110), (110, 121.5), (121.5, 121.6), (121.6, 150), (150, 200), (200, 300)):
        n = np.interp(hi, lam, cum) - np.interp(lo, lam, cum)
        print(f"   {lo:6.1f}-{hi:5.1f} nm: {n:.4f}")
    f = cr_uv.field(cr_uv.ZETA_REF, energy_ev=P.E_PH)
    E, F = f.energy_ev, f.photon_flux
    band = lambda lo, hi: float(np.trapezoid(np.where((E >= lo) & (E <= hi), F, 0.0), E))      # noqa: E731
    sig = cr_uv.dust_absorption_per_h(E)
    gen = cr_uv.ZETA_REF * cr_uv.X_H2_REF * float(np.interp(cr_uv.HC_EV_NM / P.E_PH_MIN_EV, lam, cum)
                                                    - np.interp(cr_uv.HC_EV_NM / 13.6, lam, cum))
    print(f"   field at zeta(H2) = 1e-16, x(H2) = 0.5: {band(1.0, 13.6):.3e} photons cm^-2 s^-1 at 1-13.6 eV, "
          f"{band(6.0, 13.6):.3e} at 6-13.6 eV; Habing-equivalent G0 {f.g0():.2e}")
    print(f"   photon conservation: int F sigma_abs dE {np.trapezoid(F * sig, E):.4e} = generated {gen:.4e} per H s^-1")
    print(f"   RTZ scale factor: zeta_H 1e-16 at x_e 1e-7 -> {cr_uv.scale(1e-16, 1e-7):.3f}; at x_e 1e-2 -> "
          f"{cr_uv.scale(1e-16, 1e-2):.3f}")
    if a.leiden_zip is not None:
        print("\n2. Photoionisation rates at zeta(H2) = 1e-16 [s^-1]: this field (dust only) / Heays+17 (dust + gas) = ratio")
        with zipfile.ZipFile(a.leiden_zip) as zf:
            names = {n.split("cross_sections/")[-1]: n for n in zf.namelist() if "cross_sections/" in n}
            for el, (ref, member) in RTZ_RATES.items():
                e, s = leiden_sigma(zf, names[member])
                k = float(np.trapezoid(F * np.interp(E, e, s, left=0.0, right=0.0), E))
                print(f"   {el:2s}: {k:.3e} / {ref:.3e} = {k / ref:.2f}   (threshold {e[0]:.2f} eV)")
    print("\n3. PAHs in a dark cloud (n_H 1e4, n(H) 1, x_e 1e-7, T 10 K), lit by the induced field only:")
    lib = [m for m in L.library() if m.name in NAMES]
    pop = P.Population(lib, {"all": (1.0, 1e4)}, cache_dir=FP.BASIS_CACHE)
    zs = np.array(pop.zs, float)
    for zeta in (1e-19, 1e-17, 1e-16, 1e-15):                              # 1e-19: almost no field, the reference
        env = cr_uv.environment("dark", zeta, 1e-7, 1.0, 1e-3, 10.0)
        shape = P.field_shape(env)
        pop.prepare(shape, workers=a.workers)
        pop.prepare_responses(shape, workers=a.workers)
        f_z = pop.charge_matrix(env)
        up = env.g0 * pop._stack(shape)[0]
        cap = pop._capture_rates(env.temperature_k) * env.n_e
        print(f"   zeta_H {zeta:.0e} (scale {env.g0:.3f}, G0-equivalent {env.field().g0():.1e}):")
        for s, m in enumerate(pop.library):
            i0 = 0 - pop.zmin
            absorbed = env.g0 * float(np.trapezoid(P._unit_field(shape).photon_flux * E
                                                   * photo.sigma_uv(m.nc, 0, E, pop.p.uv_cross_section), E))
            print(f"     {m.name}: <Z> {f_z[s] @ zs:+.3f}; neutral's photoionisation {up[s, i0]:.2e} s^-1 against the "
                  f"cation's capture {cap[s, i0 + 1]:.2e}; absorbed {absorbed / m.nc:.2e} eV s^-1 per C")


if __name__ == "__main__":
    main()
