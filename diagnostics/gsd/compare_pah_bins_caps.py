"""Does the super-hydrogenation cap change the molecular bins' observables? Runs of ``check_pah_bins_hydrogen.py`` at
several caps (``--superh-levels``), each scored against the deepest one.

The truth is the deepest run's evolution along each path. Every other run is scored against it on the bin observables
(``check_pah_bins_hydrogen.score_obs``: factor errors max(a/b, b/a) - 1, worst cell and path budget), twice:

- truth: the run's own evolution, i.e. how far the cap alone moves the true observables;
- E: the run's equilibrium, i.e. what a table built at that cap gives against the deepest chain.

The deepest run is also scored against its own evolution (E). The report also gives the deepest run's largest share at
its own cap along each path; where that share exceeds a few per cent, the deepest cap is not converged either.

    python diagnostics/gsd/compare_pah_bins_caps.py K4.pkl K8.pkl K12.pkl K24.pkl --murga2020 DIR --out caps.txt
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_pah_bins_hydrogen as C   # noqa: E402

AT_CAP = C.COLS.index("at_cap")
CAP_SHOWN = 0.05                      # count the paths where the deepest run holds more than this at its cap
CAP_LISTED = 6                        # and list the largest of them


def load(pkl):
    """(stored run, {name: (shard result, index)}, label, cap) of a ``check_pah_bins_hydrogen`` .pkl."""
    with open(pkl, "rb") as fh:
        d = pickle.load(fh)
    if any("nc" not in r for r in d["results"]):               # runs before 2026-11-15 did not store it
        nc = {m.name: m.nc for m in C.library()}
        for r in d["results"]:
            r.setdefault("nc", np.array([nc[n] for n in r["names"]]))
    sh = d.get("superh", {})
    cap = int(sh.get("superh_levels", 4))
    label = f"{sh.get('superh_chemistry', 'default')} K={cap}"
    return d, {n: (r, i) for r in d["results"] for i, n in enumerate(r["names"])}, label, cap


def path_rates(by_name, names, p_i):
    """``_bin_rates`` of the bin's evolution along path p_i."""
    return C._bin_rates(np.stack([by_name[n][0]["path_obs"][p_i][:, by_name[n][1]] for n in names], axis=1))


def eq_rates(by_name, names, cs):
    """``_bin_rates`` of the bin's equilibrium on the path's conditions cs."""
    return C._bin_rates(np.array([[by_name[n][0]["obs_eq"][c][by_name[n][1]] for n in names] for c in cs]))


def compare(runs, paths):
    """{bin edges: {"n", "cap": [(path, share)], label: {"truth"|"E": {obs: [(path, cell, budget)]}}}}, runs as
    ``load``'s, the deepest last."""
    ref_d, ref_by, ref_label, _ = runs[-1]
    common = set.intersection(*(set(by) for _, by, _, _ in runs))
    out = {}
    for edges, names in ref_d["bins"]:
        names = [n for n in names if n in common]
        if not names:
            continue
        n_c = sum(ref_by[n][0]["nc"][ref_by[n][1]] for n in names)
        entry = {"n": len(names), "cap": []}
        for p_i, (pname, cs, ds) in enumerate(paths):
            truth = path_rates(ref_by, names, p_i)
            share = np.mean([ref_by[n][0]["paths"][p_i][:, ref_by[n][1], AT_CAP] for n in names], axis=0)
            if share.max() > CAP_SHOWN:
                entry["cap"].append((pname, float(share.max())))
            for d, by, label, _ in runs:
                kinds = {"E": eq_rates(by, names, cs)}
                if by is not ref_by:
                    kinds["truth"] = path_rates(by, names, p_i)
                for kind, rates in kinds.items():
                    for k, (cell, budget) in C.score_obs(rates, truth, ds, n_c).items():
                        entry.setdefault(label, {}).setdefault(kind, {}).setdefault(k, []).append((pname, cell, budget))
        out[edges] = entry
    return out, ref_label


def _worst(rows, col):
    vals = [(r[col], r[0]) for r in rows if np.isfinite(r[col])]
    return max(vals) if vals else (np.nan, "-")


def report(res, ref_label, out_txt):
    lines = [f"Observables against the deepest run's evolution ({ref_label}); factor errors, worst over paths "
             "(cell | budget, and the path of the worst budget)."]
    for (lo, hi), e in res.items():
        lines.append(f"\nbin N_C {lo:.0f}-{hi:.0f} ({e['n']} molecules)")
        for label in (k for k in e if k not in ("n", "cap")):
            for kind in ("truth", "E"):
                if kind not in e[label]:
                    continue
                cells = []
                for k, rows in e[label][kind].items():
                    c, _ = _worst(rows, 1)
                    b, p = _worst(rows, 2)
                    cells.append(f"{k} {c:6.3f} | {b:6.3f} ({p})")
                lines.append(f"  {label:14s} {kind:5s}: " + "; ".join(cells))
        if e["cap"]:
            top = sorted(e["cap"], key=lambda t: -t[1])
            lines.append(f"  at {ref_label}'s cap (> {CAP_SHOWN}, {len(top)} paths; the first {CAP_LISTED}): "
                         + ", ".join(f"{p} {s:.2f}" for p, s in top[:CAP_LISTED]))
    Path(out_txt).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("pkl", nargs="+", help="check_pah_bins_hydrogen runs (.pkl) at different caps")
    ap.add_argument("--murga2020", default=str(Path.home() / "Library/Caches/calima/datasets/pahchem/murga2020"))
    ap.add_argument("--out", default="results/carbon_plan/compare_pah_bins_caps.txt")
    args = ap.parse_args(argv)
    runs = sorted((load(p) for p in args.pkl), key=lambda r: r[3])
    _, paths = C.conditions(args.murga2020)
    for d, _, label, _ in runs:
        if [(n, len(c)) for n, c, _ in paths] != d["paths"]:
            raise ValueError(f"{label}: its paths differ from conditions()'s")
    res, ref_label = compare(runs, paths)
    report(res, ref_label, args.out)


if __name__ == "__main__":
    main()
