#!/usr/bin/env python3
"""pahchem's pyrene cascades against measured IR emission of UV-excited pyrene (Saykally's laboratory).

Gas-phase pyrene after one 5 eV photon: the 3.3 um (Wagner, Kim & Saykally 2000) and 12 um (Cook et al. 1996)
emission bands, as replotted by de Bentzmann+26 (arXiv:2609.01696) with their own cascade
(saykally_pyrene_emission.csv). Models, each fitted with a free scale and constant offset (the lab data sit on a
baseline):

- **lab laws**: ``demyk2026.band_profile`` with the CosmicPAH-IRDB pyrene laws of their comparison (dataset ID 2,
  ``cosmicpah`` key ``pyrene-mono``), the B3LYP/4-31G pyrene modes of ``demyk2026_modes.csv``, PAHdb v4.00
  UID 334 intensities by frequency rank (CH stretches times the lab stretch scale);
- **VPT2**: ``mackie2018`` from the Mackie+18 pyrene force field (polyads, exact shift distributions), the same
  intensities mapped to the nearest VPT2 fundamental, strongest first.

Figure ``saykally_pyrene`` (plots/pahchem); printed: rms over the lab points / lab peak.

    check_saykally_pyrene.py     (needs PAHdb v4.00, cosmicpah-irdb and mackie2018-supplement)
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pycalima import _paths
from pycalima.pahchem import ames_suite, cosmicpah, demyk2026 as D, mackie2018 as M, population, vpt2
from pycalima.pahchem.molecule import _EINSTEIN_A_PER_KM_MOL

BANDS = {"3.3": (2750.0, 3200.0, (2900.0, 3200.0), 3051.3), "12": (760.0, 920.0, (800.0, 880.0), 838.5)}
E_UV = 5.0


def pahdb_intensities(nu, how):
    recs = ames_suite.xml_index(ames_suite.xml_path("pahdb-theoretical-v4-00"),
                                _paths.get_model_data_dir() / "pahchem_pahdb_index")
    r = next(x for x in recs if x["uid"] == 334)
    pf, pi = np.asarray(r["freqs_cm"], float), np.asarray(r["intensity"], float)
    out = np.zeros(nu.size)
    if how == "rank":
        for sf, sn in ((pf > 2500, nu > 2500), (pf <= 2500, nu <= 2500)):
            out[np.where(sn)[0][np.argsort(-nu[sn])]] = pi[np.where(sf)[0][np.argsort(-pf[sf])]]
    else:
        taken = np.zeros(nu.size, bool)
        for j in np.argsort(-pi):
            cand = np.where(((nu > 2500) == (pf[j] > 2500)) & ~taken)[0]
            m = cand[np.argmin(np.abs(nu[cand] - pf[j]))]
            taken[m], out[m] = True, pi[j]
    out[nu > 2500] *= population.LAB_STRETCH_SCALE
    return out


def models(nu_by_band):
    f = D.modes("pyrene")
    a = _EINSTEIN_A_PER_KM_MOL * f**2 * pahdb_intensities(f, "rank")
    laws = {b.label_cm: b for b in cosmicpah.laws("pyrene-mono")}
    ff = vpt2.mackie2018_force_field("pyrene")
    nuv = vpt2.fundamentals(ff.omega, vpt2.anharmonic_constants(ff))
    cs = M.run(ff, E_UV, pahdb_intensities(nuv, "nearest"))
    out = {}
    for band, (lo, hi, sel, lab_band) in BANDS.items():
        nu = nu_by_band[band]
        out[band] = {"lab laws": D.band_profile(f, a, (f > sel[0]) & (f < sel[1]), laws[lab_band], E_UV, nu),
                     "VPT2 + polyads": cs.spectrum_distribution(nu, lo_cm=lo - 50, hi_cm=hi + 50)}
    return out


def scored(model, nu, lab_nu, lab_f):
    B = np.column_stack([np.interp(lab_nu, nu, model), np.ones(lab_nu.size)])
    coef, *_ = np.linalg.lstsq(B, lab_f, rcond=None)
    return float(np.sqrt(np.mean((B @ coef - lab_f) ** 2)) / lab_f.max()), coef


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    data = pd.read_csv(_paths.get_external_data_path("pahchem", "saykally_pyrene_emission.csv"), comment="#")
    grids = {b: np.linspace(v[0], v[1], 1500) for b, v in BANDS.items()}
    mods = models(grids)
    from pycalima.plotting_style import use_calima_style

    use_calima_style(usetex=False)
    plt.rcParams.update({"font.serif": ["cmr10", "DejaVu Serif", "serif"], "mathtext.fontset": "cm",
                         "axes.formatter.use_mathtext": True, "legend.frameon": False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, band in zip(axes, BANDS):
        d = data[np.isclose(data.band_um, float(band))]
        lab, theirs = d[d.kind == "lab"], d[d.kind == "debentzmann_model"]
        ax.plot(lab.nu_cm, lab.intensity, "o", color="k", ms=3.5, label="Saykally lab")
        ax.plot(theirs.nu_cm, theirs.intensity, color="0.6", lw=2.2, alpha=0.6, label="de Bentzmann+26 model")
        r_theirs = float(np.sqrt(np.mean((np.interp(lab.nu_cm, theirs.nu_cm, theirs.intensity) - lab.intensity) ** 2))
                         / lab.intensity.max())
        print(f"{band} um: de Bentzmann+26 model rms {r_theirs:.3f}")
        for (name, m), col in zip(mods[band].items(), ("C3", "C0")):
            rms, coef = scored(m, grids[band], lab.nu_cm.to_numpy(), lab.intensity.to_numpy())
            ax.plot(grids[band], coef[0] * m + coef[1], color=col, lw=1.2, label=f"pahchem, {name} (rms {rms:.3f})")
            print(f"{band} um: pahchem {name} rms {rms:.3f}, peak {grids[band][np.argmax(m)]:.1f} cm^-1")
        ax.set_xlim(BANDS[band][1], BANDS[band][0])
        ax.set_xlabel(r"$\nu$ [cm$^{-1}$]")
        ax.set_ylabel("intensity [arbitrary]")
        ax.set_title(f"pyrene, {band} $\\mu$m band, one 5 eV photon")
        ax.legend(fontsize=7)
        ax.minorticks_on()
        ax.tick_params(which="both", direction="in")
    out = args.out or _paths.get_plots_dir("pahchem")
    fig.tight_layout()
    fig.savefig(Path(out) / "saykally_pyrene.png", dpi=200, bbox_inches="tight")
    print(f"wrote {Path(out) / 'saykally_pyrene.png'}")


if __name__ == "__main__":
    main()
