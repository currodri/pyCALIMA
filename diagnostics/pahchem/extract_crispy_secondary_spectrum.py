#!/usr/bin/env python3
"""The secondary-electron spectrum per unit zeta in H2, from CRISpy's Spencer-Fano degradation.

CRISpy (crispy_cr) builds the electrons ejected by every primary ionisation
(Rudd's distribution for CR protons, times its heavy-nuclei factor 1.51; Opal,
Beaty & Peterson for primary electrons) from the Padovani+18 spectra attenuated
by CSDA. It then degrades them in H2 with the MCCC cross sections, elastic and
Coulomb losses. Its ``DegradationResult.track`` is the column traversed per
energy bin, defined so that the rate of any process is sum_i track_i sigma_i
= int 4 pi j(E) sigma(E) dE. So 4 pi j_sec(E) = track / dE.

The 0.02 eV bins below 40 eV carry a comb from the discrete losses (bin rms
50-80 %), so the table is rebinned, conserving the flux, onto 10 bins per
decade from 1 eV to 100 keV.

Primary electrons are not in the table: pahchem.cosmic_rays already carries
them. zeta in the table is the total (protons x 1.51, primary electrons and
secondaries) per H2, for CRs from one side (CRISpy's convention).

Checks printed:
1. CSDA balance at keV: 4 pi j(eps) L_ESTAR(eps) / S(>eps), with S the rate of
   secondaries born above eps (an independent loss function).
2. zeta_sec / zeta_primary against Ivlev+21 (0.8-1.1 for 1e20-1e25 cm^-2).

Needs crispy_cr and the MCCC tree; run with CRISpy's interpreter:
  ~/Documents/GitHub/CRISpy/.venv/bin/python diagnostics/pahchem/extract_crispy_secondary_spectrum.py \
      --mccc ~/Documents/GitHub/MCCC_H2 --out results/pahchem/cr_secondary_electrons
"""
import argparse
import subprocess
from pathlib import Path

import numpy as np

import crispy_cr
from crispy_cr.cosmicrays import spectra
from crispy_cr.cosmicrays.degradation import Degradation
from crispy_cr.cosmicrays.excitation import M_E_OVER_M_P, CRExcitation
from crispy_cr.cosmicrays.losses import electron_loss
from crispy_cr.cosmicrays.transport import HEAVY_NUCLEI_FACTOR, IonisationModel, ionisation_rate
from crispy_cr.io.mccc import read_h2_elastic, read_h2_processes

LOG_N = np.arange(20.0, 24.01, 0.5)
X_E = (0.0, 1e-6, 1e-5, 1e-4, 1e-3)
SPECTRA = {"L": spectra.PROTONS_L, "H": spectra.PROTONS_H}
EDGES = np.logspace(0.0, 5.0, 51)  # eV


def crispy_version() -> str:
    root = Path(crispy_cr.__file__).resolve().parents[2]
    git = lambda *a: subprocess.run(["git", "-C", str(root), *a], capture_output=True,  # noqa: E731
                                    text=True).stdout.strip()
    dirty = " (working tree modified)" if git("status", "--porcelain", "src/crispy_cr/cosmicrays") else ""
    return f"{git('rev-parse', '--short', 'HEAD')}{dirty}"


def rebin(E, flux_per_ev):
    """Flux-conserving average of 4 pi j over EDGES (bin centres assigned by midpoint)."""
    dE = np.gradient(E)
    k = np.digitize(E, EDGES) - 1
    ok = (k >= 0) & (k < EDGES.size - 1)
    out = np.bincount(k[ok], weights=(flux_per_ev * dE)[ok], minlength=EDGES.size - 1)
    return out / np.diff(EDGES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mccc", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    procs, elastic = read_h2_processes(args.mccc), read_h2_elastic(args.mccc)
    rows, zrows = [], []
    for tag, prot in SPECTRA.items():
        model = IonisationModel.build(prot, spectra.ELECTRONS)
        for x_e in X_E:
            deg = Degradation(procs, elastic=elastic, x_e=x_e, n_e=max(x_e, 1e-30) * 100.0, T_e=100.0)
            cr = CRExcitation(model, deg)
            dE = np.gradient(deg.E)
            for lg in LOG_N:
                N = 10.0**lg
                jp = model.csda_p.attenuated(model.protons, model.E, N)
                je = model.csda_e.attenuated(model.electrons, model.E, N)
                src = (cr._secondary_source(jp, model.sigma_p, M_E_OVER_M_P, HEAVY_NUCLEI_FACTOR)
                       + cr._secondary_source(je, model.sigma_e, None, 1.0))
                sec = deg.run(src)
                z_p = HEAVY_NUCLEI_FACTOR * ionisation_rate(jp, model.E, model.sigma_p)
                z_e = ionisation_rate(je, model.E, model.sigma_e)
                zeta = z_p + z_e + sec.n_ion
                F = sec.track / dE
                rows += [(tag, x_e, lg, EDGES[b], EDGES[b + 1], f)
                         for b, f in enumerate(rebin(deg.E, F / zeta))]
                zrows.append((tag, x_e, lg, zeta, z_p, z_e, sec.n_ion))
                if x_e == 0.0 and lg in (21.0, 23.0):
                    S_above = np.cumsum(src[::-1])[::-1]
                    for eps in (1e3, 1e4, 3e4):
                        i = int(np.argmin(abs(deg.E - eps)))
                        r = F[i] * electron_loss(deg.E[i:i + 1])[0] / S_above[i]
                        print(f"CSDA balance {tag} logN={lg} eps={deg.E[i]:.0f} eV: {r:.3f}")
            if x_e == 0.0:
                z = np.array([r[6] / (r[4] + r[5]) for r in zrows if r[0] == tag and r[1] == 0.0])
                print(f"zeta_sec/zeta_prim {tag}, logN {LOG_N[0]}-{LOG_N[-1]}: "
                      + " ".join(f"{v:.2f}" for v in z))

    head = (f"# Secondary-electron flux per unit zeta in H2, from CRISpy {crispy_version()}\n"
            "# (crispy_cr.cosmicrays.degradation, Spencer-Fano; MCCC e- + H2 cross sections;\n"
            "# Padovani+18 L/H protons x 1.51 heavy nuclei + Padovani+18 electrons, CSDA).\n"
            "# Made by diagnostics/pahchem/extract_crispy_secondary_spectrum.py; T_e = 100 K,\n"
            "# n(H2) = 100 cm^-3 for the Coulomb loss. Primary electrons are NOT included.\n")
    with open(args.out / "secondary_flux_per_zeta.csv", "w") as f:
        f.write(head + "# F = 4 pi j_sec / zeta, cm^-2 eV^-1, averaged over [E_lo, E_hi) eV\n"
                "spectrum,x_e,log_N_H2,E_lo_eV,E_hi_eV,F_per_zeta\n")
        f.writelines(f"{t},{x:.0e},{lg:.1f},{a:.6g},{b:.6g},{v:.5e}\n" for t, x, lg, a, b, v in rows)
    with open(args.out / "zeta.csv", "w") as f:
        f.write(head + "# ionisation rates per H2, s^-1 (one-sided CRs): total, protons x 1.51,"
                " primary electrons, secondaries\nspectrum,x_e,log_N_H2,zeta,zeta_p,zeta_e,zeta_sec\n")
        f.writelines(f"{t},{x:.0e},{lg:.1f},{z:.5e},{a:.5e},{b:.5e},{c:.5e}\n"
                     for t, x, lg, z, a, b, c in zrows)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
