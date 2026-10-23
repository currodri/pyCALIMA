#!/usr/bin/env python3
"""pahchem's neutral photoionisation against Hrodmarsson+25 (A&A 698, A202): nine PAHs, C13-C20, 7-22 eV.

A. The photoelectron share <KE>/(hv - IE) per photon energy, against ``photo.photoelectron_energy`` (0.46 up to
   13.6 eV, after Berné+22 / Bréchignac+14 coronene; ``PARTITION_NODES`` above, which come from these data).
B. The ionisation cross section (total ion yield scaled to 22 Mb/C on the 17 eV peak, as the paper) against
   sigma x Y_Jochims with Li & Draine (pahchem's default) and Malloci+07 absorption: the Mathis-weighted 6-13.6 eV
   rate, model over data, and the same divided by each curve's 17 eV peak (independent of the assumed scale).
C. Perylene's maximum photoelectric efficiency in the Mathis ISRF (their eq. 5, no recombination): pahchem's
   ingredients, and the measured sigma_ion and <KE> over model absorption. The paper's eK16 gives 1.5 % with the WD01
   free-space factor Y2; its premise, that the lab's extraction field removes the photoelectron-ion attraction, is
   checked here: a 440 V/cm field matches the ion's Coulomb field only at ~1800 A, and lowers the threshold by
   2 (e^3 F)^1/2 ~ 0.016 eV, so the measured yields and energies are free-space values already.

    calima-fetch-data fetch hrodmarsson2025-pes malloci2007-spectra
    check_hrodmarsson25.py
"""
import numpy as np

from pycalima.pahchem import fields, photo
from pycalima.pahchem import hrodmarsson2025 as H
from pycalima.pahchem import malloci2007 as M

HV = np.array([9.0, 10.0, 11.0, 12.0, 13.0, 13.6, 15.0, 18.0, 20.0, 21.5])
MALLOCI = {"Fluorene": "fluorene", "Phenanthrene": "phenanthrene", "Triphenylene": "triphenylene",
           "Corannulene": "corannulene", "Perylene": "perylene"}
E2_EV_A = 14.399645                       # e^2 [eV A]


def main():
    print("A. photoelectron share <KE>/(hv - IE); columns hv =", " ".join(f"{h:g}" for h in HV))
    table = {}
    for name, (ie, nc) in H.IE.items():
        hv, s = H.mean_share(name)
        table[name] = np.array([np.median(s[np.abs(hv - h) <= 0.25]) for h in HV])
        print(f"  {name:22s} C{nc}: " + " ".join(f"{x:.2f}" for x in table[name]))
    ours = [photo.photoelectron_energy(np.array([h]), 7.8)[0][0] / (h - 7.8) for h in HV]
    print("  median                  : " + " ".join(f"{x:.2f}" for x in np.median(list(table.values()), axis=0)))
    print("  pahchem                 : " + " ".join(f"{x:.2f}" for x in ours))

    ma = fields.mathis1983(1.0, e_min=0.5)
    mal = M.spectra()
    print("\nB. ionisation cross section, model / data (Mathis 6-13.6 eV rate | rate per 17 eV peak)")
    rows = []
    for name, (ie, nc) in H.IE.items():
        if name in H.DEPLETED:
            continue
        hv, s_data = H.ionisation_cross_section(name)
        k = (hv > 15.0) & (hv < 19.0)
        band = (hv >= 6.0) & (hv <= 13.6)
        rate = lambda s: ma.photon_rate(s[band], hv[band])
        y = photo.yield_jochims1996(hv, ie)
        models = {"LD01": photo.sigma_li_draine(nc, 0, hv) * y}
        if MALLOCI.get(name) in mal:
            models["Malloci"] = np.interp(hv, *mal[MALLOCI[name]]["sigma"][0]) * y
        r = {key: (rate(v) / rate(s_data), (rate(v) / v[k].max()) / (rate(s_data) / s_data[k].max()))
             for key, v in models.items()}
        rows.append([r["LD01"][0], r.get("Malloci", (np.nan,))[0], r["LD01"][1], r.get("Malloci", (np.nan, np.nan))[1]])
        print(f"  {name:22s} C{nc}: " + ", ".join(f"{key} {a:.2f} | {b:.2f}" for key, (a, b) in r.items()))
    med = np.nanmedian(np.array(rows), axis=0)
    print(f"  median: LD01 {med[0]:.2f} | {med[2]:.2f}, Malloci {med[1]:.2f} | {med[3]:.2f}")

    print("\nC. perylene, maximum photoelectric efficiency in the Mathis ISRF")
    ie, nc = H.IE["Perylene"]
    E = np.linspace(0.5, 13.6, 4000)
    F = ma.flux_at(E)
    eps = lambda s_abs, s_ion, ke: 100 * np.trapezoid(s_ion * ke * F, E) / np.trapezoid(s_abs * E * F, E)
    absorption = {"LD01": photo.sigma_li_draine(nc, 0, E), "Malloci": np.interp(E, *mal["perylene"]["sigma"][0])}
    hv, s_data = H.ionisation_cross_section("Perylene")
    m = H.molecule("Perylene")
    P = np.clip(m["matrix"], 0.0, None)
    mean_ke = np.interp(E, m["hv_pes"], (P @ m["ke"]) / np.maximum(P.sum(axis=1), 1e-300), left=0.0)
    s_meas = np.interp(E, hv, s_data, left=0.0)
    for key, s_abs in absorption.items():
        print(f"  {key:8s} absorption: pahchem (Jochims yield, its share) "
              f"{eps(s_abs, s_abs * photo.yield_jochims1996(E, ie), photo.photoelectron_energy(E, ie)[0]):.2f} %, "
              f"measured sigma_ion and <KE> {eps(s_abs, s_meas, mean_ke):.2f} %")
    print("  paper: eK16 1.5 % (with Y2), WD01 2 %")
    field = 440.0 * 1e-8                                   # V/A
    print(f"  lab field 440 V/cm equals the ion's Coulomb field at {np.sqrt(E2_EV_A / field):.0f} A "
          f"(potential there {E2_EV_A / np.sqrt(E2_EV_A / field) * 1e3:.1f} meV); threshold lowering "
          f"{2 * np.sqrt(E2_EV_A * field) * 1e3:.0f} meV")


if __name__ == "__main__":
    main()
