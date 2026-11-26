#!/usr/bin/env python3
"""Figure of ``check_odd_parent_thresholds.py``: the change the odd-parent law makes (odd law - even laws) against G0,
for each molecule (by size) and n_H, in the joint default and deep chains.

Usage: plot_odd_parent_thresholds.py [--csv results/pahchem/odd_parent_thresholds/compare.csv]
                                     [--out results/plots/pahchem/odd_parent_thresholds/odd_parent_effect.png]
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from pycalima import _paths  # noqa: E402
from pycalima.plotting_style import use_calima_style  # noqa: E402

ROWS = (("p1_anion", r"$\Delta$ anion share" "\n" r"of $N_\mathrm{H0}+1$"),
        ("superh", r"$\Delta$ super-H share"),
        ("z_mean", r"$\Delta\langle Z\rangle$"))
STYLE = {1e4: "-", 1e2: ":"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", type=Path,
                    default=_paths.get_results_dir() / "pahchem" / "odd_parent_thresholds" / "compare.csv")
    ap.add_argument("--out", type=Path,
                    default=_paths.get_plots_dir() / "pahchem" / "odd_parent_thresholds" / "odd_parent_effect.png")
    args = ap.parse_args()
    rows = list(csv.DictReader(open(args.csv)))
    val = {(r["chain"], r["law"], r["molecule"], float(r["g0"]), float(r["n_h"])): r for r in rows}
    chains = list(dict.fromkeys(r["chain"] for r in rows))
    mols = sorted({(int(r["nc"]), r["molecule"]) for r in rows})
    colours = plt.get_cmap("Blues")(np.linspace(0.62, 1.0, len(mols)))
    series = defaultdict(list)
    for (chain, law, name, g0, n), r in val.items():
        if law == "odd":
            series[(chain, name, n)].append(g0)

    use_calima_style()
    fig, axes = plt.subplots(len(ROWS), len(chains), figsize=(4.0 * len(chains), 2.2 * len(ROWS)), sharex=True,
                             sharey="row", gridspec_kw={"hspace": 0, "wspace": 0}, squeeze=False)
    for j, chain in enumerate(chains):
        for i, (key, label) in enumerate(ROWS):
            ax = axes[i, j]
            ax.axhline(0.0, color="0.75", lw=0.6, zorder=0)
            for c, (nc, name) in zip(colours, mols):
                for n, ls in STYLE.items():
                    g = sorted(series.get((chain, name, n), []))
                    if not g:
                        continue
                    d = [float(val[(chain, "odd", name, x, n)][key]) - float(val[(chain, "even", name, x, n)][key])
                         for x in g]
                    ax.plot(g, d, ls, color=c, lw=1.6, marker="o", ms=3,
                            label=name.split(" ")[0] if (i == 0 and j == 0 and n == 1e4) else None)
            ax.set_xscale("log")
            ax.minorticks_on()
            ax.tick_params(which="both", axis="both", direction="in", bottom=True, left=True, top=True, right=True)
            ax.tick_params(which="major", length=5)
            ax.tick_params(which="minor", length=2.6)
            if j == 0:
                ax.set_ylabel(label, fontsize=12)
        axes[0, j].set_title({"joint": "joint chain (4 levels)", "deep": "deep chain (24 levels)"}.get(chain, chain),
                             fontsize=12)
        axes[-1, j].set_xlabel(r"$G_0$", fontsize=14)
    axes[0, 0].plot([], [], "-", color="0.3", label=r"$n_\mathrm{H} = 10^4$")
    axes[0, 0].plot([], [], ":", color="0.3", label=r"$n_\mathrm{H} = 10^2$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, 0.02),
               ncol=min(len(labels), 4), title=r"odd-parent law $-$ even-parent laws; Draine field, 100 K, "
               r"$n_e = 1.4\times10^{-4}\,n_\mathrm{H}$", title_fontsize=9)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=220, bbox_inches="tight")
    print(args.out)


if __name__ == "__main__":
    main()
