#!/usr/bin/env python3
"""Ion and electron collision rates on PAHs: Draine & Sutin (1987) against Wolfire+08, the lab and Langevin.

The derived numbers behind docs/physics/pahchem-gas-coupling.md (2026-10-21). DS87 is pyCALIMA's
``J_function_DS87``, and pahchem's geometry is ``chemistry.electron_capture`` (0.82 pi a^2, a = 0.9 sqrt(N_C) A,
sticking 1).

A. **Wolfire+08 Table 2** (N_C = 35, Phi = 1) reproduced by DS87 for a sphere of radius (N_C/1.222)^1/2 A with
   sticking 1 and no disc factor; and pahchem's geometry against it.
B. **Mutual neutralisation, X+ + PAH- (C50H18).** The polyatomic law of Shuman+14,
   2.7e-7 mu^-1/2 (T/300)^-0.9, against DS87 with s = 1. Their ratio is the effective P_MN.
C. **Ion-neutral reactions, X+ + PAH0.** Langevin, k_L = 2 pi e (alpha/mu)^1/2 with alpha = 1.75 N_C A^3, times
   the total efficiency 0.9 measured on anthracene (Canosa+95: He+ 0.90, C+ 0.88; for C+ 60 % of it is charge
   transfer and 40 % carbon accretion), against DS87 for a neutral sphere (which is Langevin with alpha = a^3).
D. **Electron + PAH+ at 300 K:** the FALP data (``labdata.electron_recombination``) against pahchem's capture.

    check_pah_ion_rates.py
"""
import numpy as np

from pycalima.models.PAH_photophysics.pah_charge_utils import J_function_DS87
from pycalima.pahchem import labdata
from pycalima.pahchem.chemistry import AMU_G, ME_G, disc_radius_cm, electron_capture, thermal_speed

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
    print("\nB. Mutual neutralisation on C50H18-, T = 300 K: Shuman+14 law / DS87 (pahchem geometry)")
    for ion, m in IONS.items():
        mu = m * m_pah / (m + m_pah)
        shuman = 2.7e-7 * mu**-0.5
        k = ds87(nc, -1, 1, mu, 300.0, disc=0.82)
        print(f"  {ion:5s} Shuman {shuman:.2e}  DS87 {k:.2e}  ratio {shuman / k:.2f}")

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


if __name__ == "__main__":
    main()
