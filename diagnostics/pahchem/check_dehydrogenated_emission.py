#!/usr/bin/env python3
"""What letting the dehydrogenated states emit changes in the PDRs4All Orion Bar fits (fit_pdrs4all_spectra --fit-shape).

Two runs of the same fit, which differ only in ``--hydrogenation``:
- ``intact``: the intact molecule alone emits; a molecule is kept where its intact fraction is >= 0.5 (as before
  2026-10-25);
- ``emit``: N_H0 ... N_H0-3 emit in the chain's proportions, on stripped modes (``Population.emission_weights``);
  a molecule is kept where those states hold >= 0.5 of it; the share below N_H0-3 is not modelled.
Both use the atomic n(H) of the conditions card for the H chemistry (``conditions.atomic_h``).

Compared per template and band profile, at each run's best cell: rms, gamma, <N_C> by carbon, charge shares of the
emitted power, the share emitted by N_H0-1 ... N_H0-3 (``e_dehydro``) and the unmodelled share (``deep_share``),
and the band ratios of the fitted PAH spectrum.

Result (2026-10-25, Kurucz 40 000 K, the five templates, both profiles): a null result, as expected for N_C >~ 54
(Andrews+16 Sect. 6; Hony+01; PDRs4All).
- N_H0-1 ... N_H0-3 emit 0.1-1.6 % of the PAH power at the best cells (most in the Atomic template); the
  unmodelled share below N_H0-3 is 0.01-1.2 %.
- rms, gamma (~1e4) and the best cells do not move (2 of 10 move to an adjacent cell); <N_C> by carbon falls by
  <= 0.7; charge shares change by <= 0.011.
- Band ratios move by <= 2 % (3.3/11.2 +1.9 % in the Atomic template, Lorentzian).
- The atomic n(H) alone (same runs, intact only, against the cache before it): the DF best cells go from
  n(H) = 1.9e4 to 1.3e4, the anion share rises by 5-8 % of itself; rms, gamma and <N_C> unchanged.
The chain is nearly all-or-nothing: where the PDRs4All conditions dehydrogenate a molecule, most of it is below
N_H0-3, so the unmodelled share, not the stripped states, is where a bigger effect would show; here it is small
because the fits favour N_C ~ 80-85 molecules, which stay intact.

    check_dehydrogenated_emission.py [--emit <pkl>] [--intact <pkl>]
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
RATIOS = ("3.3", "6.2", "7.7", "11.0", "12.7")
KEYS = ("rms", "gamma", "n_h", "mean_nc_carbon", "e_cation", "e_anion", "e_dehydro", "deep_share")


def load(path):
    with Path(path).open("rb") as fh:
        return pickle.load(fh)


def result(hydrogenation):
    """The fit_pdrs4all_spectra --fit-shape result with the standard arguments (computed if not cached)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import fit_pdrs4all_spectra as F

    args = argparse.Namespace(templates=Path("results/pahchem/data/vdp25/orionbar.dat"), library="v400",
                              ip_shift="-1:0.37,0:0.6,1:0.64,2:0.64", capture_scale=1.0, capture_model="bt94",
                              stretch_scale="lab", intact_min=0.5, aliphatic="none", plot=False, workers=5,
                              fit_shape=True, hydrogenation=hydrogenation)
    return F.run_shape(args)


def summary(cell):
    out = {k: float(cell.get(k, np.nan)) for k in KEYS}
    out |= {f"r{k}": float(cell["ratios"][k]) for k in RATIOS if k in cell["ratios"]}
    return out


def best(cells):
    i = int(np.argmin([c["sse"] for c in cells]))
    return i, cells[i]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--emit", type=Path, default=None, help="result with --hydrogenation emit (default: run or load it)")
    ap.add_argument("--intact", type=Path, default=None, help="result with --hydrogenation intact")
    ap.add_argument("--out", type=Path, default=_paths.get_plots_dir("pahchem"))
    a = ap.parse_args()
    new = load(a.emit) if a.emit else result("emit")
    old = load(a.intact) if a.intact else result("intact")
    keys = (*KEYS, *(f"r{k}" for k in RATIOS))
    rows = {}
    for t in TEMPLATES:
        for p in PROFILES:
            co, cn = old[t][p]["cells"], new[t][p]["cells"]
            i_o, b_o = best(co)
            i_n, b_n = best(cn)
            rows[t, p] = {"old": summary(b_o), "new": summary(b_n), "fixed": summary(cn[i_o]), "moved": i_o != i_n}
    for label, which in (("refitted (best cell of each run)", "new"), ("at the intact run's best cell", "fixed")):
        print(f"\n== {label}: intact -> emit")
        print("template  profile     " + "  ".join(f"{k:>15s}" for k in keys))
        for (t, p), r in rows.items():
            cells = [f"{r['old'][k]:6.3g}->{r[which][k]:<6.3g}" for k in keys]
            flag = " (best cell moved)" if which == "new" and r["moved"] else ""
            print(f"{t:8s}  {p:10s}  " + "  ".join(f"{c:>15s}" for c in cells) + flag)

    use_calima_style()
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), gridspec_kw={"wspace": 0.3})
    x = np.arange(len(TEMPLATES))
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#e87ba4", "#8a5cc2"]
    pick = {t: min(PROFILES, key=lambda p: min(c["sse"] for c in new[t][p]["cells"])) for t in TEMPLATES}
    ax = axes[0]
    for j, (k, lab) in enumerate((("e_dehydro", r"emitted by $N_{\rm H0}-1 \dots -3$"),
                                  ("deep_share", r"unmodelled (below $N_{\rm H0}-3$)"))):
        ax.plot(x + 0.08 * (j - 0.5), [rows[t, pick[t]]["new"][k] for t in TEMPLATES], "o", ms=9, color=colors[j],
                label=lab)
    ax.set_xticks(x, TEMPLATES)
    ax.set_ylabel("share of the PAH power", fontsize=14)
    ax.set_title("dehydrogenated states, best cell", fontsize=12)
    ax.legend(frameon=False, fontsize=10)
    ax = axes[1]
    for j, k in enumerate(RATIOS):
        rel = [rows[t, pick[t]]["fixed"][f"r{k}"] / rows[t, pick[t]]["old"][f"r{k}"] - 1 for t in TEMPLATES]
        ax.plot(x + 0.06 * (j - 2), np.array(rel) * 100, "s", ms=8, color=colors[j], label=k + r"/11.2")
    ax.axhline(0, color="0.6", lw=1)
    ax.set_xticks(x, TEMPLATES)
    pct = r"[\%]" if plt.rcParams["text.usetex"] else "[%]"
    ax.set_ylabel(f"band ratio change {pct}", fontsize=14)
    ax.set_title("band ratios at the intact run's best cell", fontsize=12)
    ax.legend(frameon=False, fontsize=9, ncol=2)
    ax = axes[2]
    for j, (k, lab) in enumerate((("rms", "rms"), ("mean_nc_carbon", r"$\langle N_{\rm C}\rangle$"))):
        rel = [rows[t, pick[t]]["new"][k] / rows[t, pick[t]]["old"][k] - 1 for t in TEMPLATES]
        ax.plot(x + 0.08 * (j - 0.5), np.array(rel) * 100, "D", ms=8, color=colors[j + 2], label=lab)
    ax.axhline(0, color="0.6", lw=1)
    ax.set_xticks(x, TEMPLATES)
    ax.set_ylabel(f"change, emit vs intact {pct}", fontsize=14)
    ax.set_title("refitted", fontsize=12)
    ax.legend(frameon=False, fontsize=10)
    for ax in axes:
        ax.minorticks_on()
        ax.tick_params(axis="x", which="minor", bottom=False, top=False)
        ax.tick_params(which="both", direction="in", top=True, right=True)
    a.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out / "pahchem_dehydrogenated_emission.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nfigure: {a.out / 'pahchem_dehydrogenated_emission.png'}")


if __name__ == "__main__":
    main()
