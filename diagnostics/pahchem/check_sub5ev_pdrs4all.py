#!/usr/bin/env python3
"""How much photons below 5 eV change the PDRs4All Orion Bar fits (fit_pdrs4all_spectra --fit-shape).

The population's photon grid starts at 1 eV since BASIS_VERSION 6; before, it started at 5 eV. Nothing else in the
fit changed, so the two cached results differ only in that light (``field_e_min_ev = 5`` reproduces the old rates
to 1e-6, tests/test_pahchem_population.py). The Kurucz 40 000 K field keeps its whole stellar spectrum from 1 eV.
Compared per template and band profile:
- refitted: the best cell of each run (conditions, rms, emitted-power charge shares, band ratios);
- at fixed conditions: the old best cell in both runs, which isolates the direct effect from the re-inference.

The cross section below 5 eV is still Li & Draine 2001's, which absorbs 2-4x less than Malloci+07 there; for the
cations, whose visible bands are real, the effect computed here is therefore a lower bound.

Result (2026-10-24, Kurucz 40 000 K, the five templates, both profiles): a small effect.
- The inferred conditions do not move: the same best cell in 9 of 10 fits (DF3 Lorentzian moves to the adjacent
  n_e, rms 0.0601 -> 0.0597); gamma ~1e4 throughout.
- rms improves by ~1 % (e.g. HII 0.0484 -> 0.0478).
- At fixed conditions, emitted-power charge shares move by <= 0.006 (neutral up, cation down, anion +0.000 to +0.002);
  <N_C> by carbon falls by 0.4-1.1.
- Model band ratios: 3.3/11.2 -1.3 to -3.2 %, 6.2/11.2 -1.3 to -2.5 %, 7.7/11.2 -0.5 to -1.8 %, 12.7/11.2
  +0.8 to +2.3 %; 11.0/11.2 within 0.7 %. The extra photons are of low energy, so the emission is slightly cooler.
- The continuum share of the 5-15 um power falls by 0.1-0.7 percentage points.
If the effect scales with the absorbed sub-5 eV power, the 2-4x Malloci/Li & Draine ratio there would make the
band-ratio changes ~3-13 %; the conditions inferred from the Orion Bar would still hardly move. The hot star is why: little of its absorbed
power is below 5 eV. Cooler fields (the ISRF, B-star nebulae such as NGC 7023) are where it matters.

    check_sub5ev_pdrs4all.py [--old results/pahchem_spectral_shape_4154c0c24afa.pkl] [--new <pkl>]
"""
import argparse
import pathlib
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from pycalima import _paths  # noqa: E402
from pycalima.plotting_style import use_calima_style  # noqa: E402

sys.modules.setdefault("pathlib._local", pathlib)          # pickles written under Python 3.13
TEMPLATES = ("HII", "Atomic", "DF1", "DF2", "DF3")
PROFILES = ("lorentzian", "gaussian")
RATIOS = ("3.3", "6.2", "7.7", "8.6", "11.0", "12.7")
SHARES = ("e_anion", "e_neutral", "e_cation", "e_dication")
OLD = _paths.get_results_dir() / "pahchem_spectral_shape_4154c0c24afa.pkl"


def load(path):
    with Path(path).open("rb") as fh:
        return pickle.load(fh)


def summary(cell):
    out = {k: float(cell[k]) for k in ("rms", "gamma", "g0", "n_e", "mean_nc_carbon", "cont_share")}
    out |= {k: float(cell[k]) for k in ("e_anion", "e_cation", "e_dication")}
    out["e_neutral"] = 1.0 - out["e_anion"] - out["e_cation"] - out["e_dication"]
    out |= {f"r{k}": float(cell["ratios"][k]) for k in RATIOS if k in cell["ratios"]}
    return out


def best(cells):
    i = int(np.argmin([c["sse"] for c in cells]))
    return i, cells[i]


def new_result():
    """The current fit_pdrs4all_spectra result for the standard arguments (computed if not cached)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import fit_pdrs4all_spectra as F

    args = argparse.Namespace(templates=Path("results/pahchem/data/vdp25/orionbar.dat"), library="v400",
                              ip_shift="-1:0.37,0:0.6,1:0.64,2:0.64", capture_scale=1.0, capture_model="bt94",
                              stretch_scale="lab", intact_min=0.5, aliphatic="none", plot=False, workers=5,
                              fit_shape=True)
    return F.run_shape(args)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--old", type=Path, default=OLD, help="result with the 5 eV grid")
    ap.add_argument("--new", type=Path, default=None, help="result with the 1 eV grid (default: run or load it)")
    ap.add_argument("--out", type=Path, default=_paths.get_plots_dir("pahchem"))
    a = ap.parse_args()
    old, new = load(a.old), (load(a.new) if a.new else new_result())
    keys = ("rms", "gamma", "g0", "n_e", "mean_nc_carbon", "cont_share", *SHARES, *(f"r{k}" for k in RATIOS))
    rows = {}
    for t in TEMPLATES:
        for p in PROFILES:
            co, cn = old[t][p]["cells"], new[t][p]["cells"]
            i_o, b_o = best(co)
            i_n, b_n = best(cn)
            rows[t, p] = {"old": summary(b_o), "new": summary(b_n), "fixed": summary(cn[i_o]), "moved": i_o != i_n}
    for label, which in (("refitted (best cell of each run)", "new"), ("at the old best cell", "fixed")):
        print(f"\n== {label}: 5 eV -> 1 eV")
        print("template  profile     " + "  ".join(f"{k:>14s}" for k in keys))
        for (t, p), r in rows.items():
            cells = []
            for k in keys:
                o, n = r["old"].get(k, np.nan), r[which].get(k, np.nan)
                cells.append(f"{o:6.3g}->{n:<6.3g}" if k not in SHARES else f"{o:6.3f}->{n:<6.3f}")
            flag = " (best cell moved)" if which == "new" and r["moved"] else ""
            print(f"{t:8s}  {p:10s}  " + "  ".join(f"{c:>14s}" for c in cells) + flag)

    use_calima_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"wspace": 0.28})
    x = np.arange(len(TEMPLATES))
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#e87ba4"]
    ax = axes[0]
    for j, k in enumerate(SHARES):
        o = [rows[t, "lorentzian"]["old"][k] for t in TEMPLATES]
        n = [rows[t, "lorentzian"]["fixed"][k] for t in TEMPLATES]
        ax.plot(x + 0.06 * (j - 1.5), o, "o", ms=8, mfc="white", color=colors[j])
        ax.plot(x + 0.06 * (j - 1.5), n, "o", ms=8, color=colors[j], label=k.split("_")[1])
    ax.set_xticks(x, TEMPLATES)
    ax.set_ylabel("share of the emitted PAH power", fontsize=14)
    ax.set_title("charge, at the old best cell (open: 5 eV, filled: 1 eV)", fontsize=12)
    ax.legend(frameon=False, fontsize=10, ncol=2)
    ax = axes[1]
    for j, k in enumerate(("r3.3", "r6.2", "r7.7", "r11.0")):
        rel = [rows[t, "lorentzian"]["fixed"][k] / rows[t, "lorentzian"]["old"][k] - 1 for t in TEMPLATES]
        ax.plot(x + 0.06 * (j - 1.5), np.array(rel) * 100, "s", ms=8, color=colors[j],
                label=k[1:] + r"/11.2")
    ax.axhline(0, color="0.6", lw=1)
    ax.set_xticks(x, TEMPLATES)
    ax.set_ylabel(r"band ratio change [\%]" if plt.rcParams["text.usetex"] else "band ratio change [%]", fontsize=14)
    ax.set_title("model band ratios, at the old best cell", fontsize=12)
    ax.legend(frameon=False, fontsize=10, ncol=2, loc="upper center")
    for ax in axes:
        ax.minorticks_on()
        ax.tick_params(axis="x", which="minor", bottom=False, top=False)
        ax.tick_params(which="both", direction="in", top=True, right=True)
    a.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out / "pahchem_sub5ev_pdrs4all.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nfigure: {a.out / 'pahchem_sub5ev_pdrs4all.png'}")


if __name__ == "__main__":
    main()
