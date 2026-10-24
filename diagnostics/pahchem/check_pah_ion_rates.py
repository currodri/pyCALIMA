#!/usr/bin/env python3
"""Ion and electron collision rates on PAHs: Draine & Sutin (1987) against Wolfire+08, the lab and Langevin.

The derived numbers behind docs/physics/pahchem-gas-coupling.md (2026-10-21). DS87 is pyCALIMA's
``J_function_DS87``, and pahchem's geometry is ``chemistry.electron_capture`` (0.82 pi a^2, a = 0.9 sqrt(N_C) A,
sticking 1).

A. **Wolfire+08 Table 2** (N_C = 35, Phi = 1) reproduced by DS87 for a sphere of radius (N_C/1.222)^1/2 A with
   sticking 1 and no disc factor; and pahchem's geometry against it.
B. **Mutual neutralisation, X+ + PAH- (C50H18).** Two size-independent laws against DS87 with s = 1: the
   polyatomic law of Shuman+14, 2.7e-7 mu^-1/2 (T/300)^-0.9, and Hickman (1979, eq. 28),
   5.35e-7 mu^-1/2 (T/300)^-1/2 EA^-0.4 (mu in amu, EA in eV; 1.5-2.5 eV here); and Tielens (2021, eq. 8.112),
   3e-8 N_C^1/2 (300/T)^1/2 mu^-1/2. Their ratio is the effective P_MN.
C. **Ion-neutral reactions, X+ + PAH0.** Langevin, k_L = 2 pi e (alpha/mu)^1/2 with alpha = 1.75 N_C A^3, times
   the total efficiency 0.9 measured on anthracene (Canosa+95: He+ 0.90, C+ 0.88; for C+ 60 % of it is charge
   transfer and 40 % carbon accretion), against DS87 for a neutral sphere (which is Langevin with alpha = a^3).
D. **Electron + PAH+ at 300 K:** the FALP data (``labdata.electron_recombination``) against pahchem's capture.
E. **Wolfire+08 against WD01b alpha_g** at W08's CNM point (G0 = 1.7, n = 30, T = 100 K, x_e = 4.4e-4,
   psi ~ 1.3e3): W08's three-state balance at Phi = 0.4, x_PAH = 6e-7 (``check_wolfire08_cplus.wolfire08``)
   over WD01b's fit (``check_grain_size_distribution``, R_V = 3.1, b_C = 6e-5) for C+ and H+. WD01b's H+/C+
   follows the DS87 mass scaling sqrt(m_C/m_H) = 3.45, as W08's does, so the two ratios agree (1.6). The
   ~6 for H+ quoted until 2026-10-23 is reproduced only with C2 = 1.578 for H+, a transcription of WD01b's
   1.378 (which pyCALIMA, 3D-PDR and DESPOTIC/Gong+17 all print). The C+ row differs between codes: DESPOTIC's
   GOW network has C4 = 0.4845 where the others have 4.845e-2, which is 5x lower here.

    check_pah_ion_rates.py
"""
import sys
from pathlib import Path

import numpy as np

from pycalima.models.PAH_photophysics.pah_charge_utils import J_function_DS87
from pycalima.pahchem import labdata
from pycalima.pahchem.chemistry import AMU_G, ME_G, disc_radius_cm, electron_capture, thermal_speed

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[1] / "dust_charge"))
import check_grain_size_distribution as WD01B
import check_wolfire08_cplus as W08_CNM

E_STATC = 4.80320e-10
#: Wolfire+08 Table 2 at T = 100 K, Phi = 1 [cm^3 s^-1]: (label, PAH charge, projectile charge, mass [amu], value)
W08 = [("e + PAH+", 1, -1, ME_G / AMU_G, 3.5e-5), ("e + PAH0", 0, -1, ME_G / AMU_G, 1.3e-6),
       ("C+ + PAH-", -1, 1, 12.011, 2.4e-7), ("C+ + PAH0", 0, 1, 12.011, 8.8e-9),
       ("H+ + PAH-", -1, 1, 1.008, 8.3e-7), ("H+ + PAH0", 0, 1, 1.008, 3.1e-8)]
IONS = {"H+": 1.008, "C+": 12.011, "O+": 15.999, "S+": 32.06, "Si+": 28.086, "Mg+": 24.305, "Fe+": 55.845,
        "HCO+": 29.018}


def ds87(nc_or_a, z_pah, z_proj, mass_amu, t, radius_cm=None, disc=1.0):
    """DS87 collision rate [cm^3 s^-1]: disc pi a^2 v J~(Z, z); v from the projectile mass."""
    a = disc_radius_cm(nc_or_a) if radius_cm is None else radius_cm
    return disc * np.pi * a**2 * thermal_speed(t, mass_amu * AMU_G) * J_function_DS87(z_pah, z_proj, a, float(t))


def main():
    print("A. Wolfire+08 Table 2 (N_C = 35, T = 100 K, Phi = 1)")
    a_w08 = np.sqrt(35 / 1.222) * 1e-8
    for lab, z, q, m, w in W08:
        k = ds87(35, z, q, m, 100.0, radius_cm=a_w08)
        kp = ds87(35, z, q, m, 100.0, disc=0.82)
        print(f"  {lab:10s} W08 {w:.2e}  DS87 sphere {k:.2e} ({k / w:.3f})  pahchem geometry {kp:.2e} ({kp / w:.2f})")

    nc, m_pah = 50, 50 * 12.011 + 18 * 1.008
    print("\nB. Mutual neutralisation on C50H18-, T = 300 K: Shuman+14, Hickman 1979 and Tielens 2021 laws / DS87 (pahchem geometry)")
    for ion, m in IONS.items():
        mu = m * m_pah / (m + m_pah)
        shuman = 2.7e-7 * mu**-0.5
        hickman = [5.35e-7 * mu**-0.5 * ea**-0.4 for ea in (2.5, 1.5)]
        tielens = 3e-8 * np.sqrt(nc) * mu**-0.5
        k = ds87(nc, -1, 1, mu, 300.0, disc=0.82)
        print(f"  {ion:5s} DS87 {k:.2e} | Shuman {shuman:.2e} ({shuman / k:.2f}) | Hickman EA 2.5-1.5 eV "
              f"{hickman[0]:.2e}-{hickman[1]:.2e} ({hickman[0] / k:.2f}-{hickman[1] / k:.2f}) | Tielens 8.112 "
              f"{tielens:.2e} ({tielens / k:.2f})")

    print("\nC. Ion-neutral reactions on neutral PAHs, T = 100 K: 0.9 Langevin (alpha = 1.75 N_C A^3) / DS87")
    for nc_ in (24, 50, 100):
        for ion in ("H+", "C+", "S+"):
            m = IONS[ion]
            mu = m * (nc_ * 12.011) / (m + nc_ * 12.011) * AMU_G
            k_l = 0.9 * 2 * np.pi * E_STATC * np.sqrt(1.75 * nc_ * 1e-24 / mu)
            k = ds87(nc_, 0, 1, m, 100.0, disc=0.82)
            print(f"  N_C {nc_:3d} {ion:3s} 0.9 k_L {k_l:.2e}  DS87 {k:.2e}  ratio {k_l / k:.2f}")

    print("\nD. e + PAH+ at 300 K: FALP / pahchem capture (s = 1)")
    for r in labdata.electron_recombination().itertuples():
        k = electron_capture(r.n_c, 1, 300.0)
        print(f"  {r.species:13s} N_C {r.n_c:2d}  measured {r.k_cm3_s:.2e}  pahchem {k:.2e}  ratio {r.k_cm3_s / k:.2f}")

    ne = (W08_CNM.XC + W08_CNM.XH) * W08_CNM.N
    g0, t = W08_CNM.G0, W08_CNM.T
    print(f"\nE. Wolfire+08 (Phi = 0.4, x_PAH = 6e-7) / WD01b alpha_g, G0 = {g0}, T = {t:.0f} K, n_e = {ne:.4f} "
          f"(psi = {g0 * np.sqrt(t) / ne:.0f})")
    wd = {}
    for ion in ("C+", "H+"):
        w08 = W08_CNM.wolfire08(0.4, ion=ion)[0]
        wd[ion] = WD01B.get_analytical_recomb_coefficient(ion, g0, ne, t)
        print(f"  {ion}  W08 {w08:.2e}  WD01b {wd[ion]:.2e}  ratio {w08 / wd[ion]:.2f}")
    print(f"  WD01b H+/C+ {wd['H+'] / wd['C+']:.2f}, sqrt(m_C/m_H) {np.sqrt(IONS['C+'] / IONS['H+']):.2f}")
    c = list(WD01B.TABLE_2_COEFFS["H+"])
    WD01B.TABLE_2_COEFFS["H+"] = [*c[:2], 1.578, *c[3:]]
    print(f"  H+ with C2 = 1.578 in place of 1.378: WD01b "
          f"{WD01B.get_analytical_recomb_coefficient('H+', g0, ne, t):.2e} (9.1e-15 at the x_e = 4.5e-4 used until 2026-10-23)")
    WD01B.TABLE_2_COEFFS["H+"] = c
    c = list(WD01B.TABLE_2_COEFFS["C+"])
    WD01B.TABLE_2_COEFFS["C+"] = [*c[:4], 0.4845, *c[5:]]
    print(f"  C+ with C4 = 0.4845 (DESPOTIC GOW.py): WD01b "
          f"{WD01B.get_analytical_recomb_coefficient('C+', g0, ne, t):.2e}")
    WD01B.TABLE_2_COEFFS["C+"] = c


if __name__ == "__main__":
    main()
