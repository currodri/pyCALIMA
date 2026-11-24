"""Compact per-bin path data from the cap runs of check_pah_bins_hydrogen.py (run on megatron, where the pickles are).

For every run and bin, and for a few paths: cumulative time, the conditions (G0, n(H)), the bin rates of the truth
(evolved) and of E (equilibrium on each condition), and the truth's <N_H>/N_H0, super-H and at-cap shares.
Writes paths.npz next to --out.
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "pycalima_deep/diagnostics/gsd"))
import check_pah_bins_hydrogen as C  # noqa: E402

PATHS = ("NGC 7023 in  1 km/s", "Orion Bar in  1 km/s", "Orion Bar out 1 km/s", "Horsehead in  1 km/s",
         "Horsehead out 1 km/s", "CNM<->WNM, period 1e+04 yr", "CNM<->WNM, period 1e+03 yr",
         "G0 0.1<->1 at n(H)=10, period 1e+03 yr", "G0 100<->1000 at n(H)=10, period 1e+03 yr")
RUNS = ("andrews16_k4", "deep_k8", "deep_k12", "deep_k24")
RATES = ("carbon", "heating", "cooling", "h2", "h_uptake", "absorbed", "uv_aro", "uv_ali")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(Path.home() / "pycalima_runs/pah_bins_obs"))
    ap.add_argument("--out", default=str(Path.home() / "pycalima_runs/pah_bins_obs/paths.npz"))
    args = ap.parse_args()
    _, paths = C.conditions(str(Path.home() / "pycalima_data/murga2020"))
    names = [n for n, _, _ in paths]
    out = {"paths": np.array(PATHS), "runs": np.array(RUNS), "rates": np.array(RATES)}
    idx = [C.OBS_RATES.index(k) for k in RATES]
    for p_name in PATHS:
        p = names.index(p_name)
        _, cs, ds = paths[p]
        key = f"p{PATHS.index(p_name)}"
        out[key + "_t"] = np.cumsum(ds) / C.YR
        out[key + "_g0"] = np.array([c[0] for c in cs])
        out[key + "_nh"] = np.array([c[1] for c in cs])
    for run in RUNS:
        d = pickle.load(open(Path(args.runs) / f"{run}.pkl", "rb"))
        by = {n: (r, i) for r, in zip(d["results"]) for i, n in enumerate(r["names"])}
        for b, (edges, bn) in enumerate(d["bins"]):
            bn = [n for n in bn if n in by]
            nh0 = np.array([by[n][0]["nh0"][by[n][1]] for n in bn])
            nc = np.array([by[n][0]["nc"][by[n][1]] for n in bn])
            out[f"bin{b}_edges"], out[f"bin{b}_nc"] = np.array(edges), np.array(nc.sum())
            for p_name in PATHS:
                p = names.index(p_name)
                cs = paths[p][1]
                key = f"{run}_b{b}_p{PATHS.index(p_name)}"
                tr = np.stack([by[n][0]["path_obs"][p][:, by[n][1]] for n in bn], axis=1).sum(axis=1)
                eq = np.array([[by[n][0]["obs_eq"][c][by[n][1]] for n in bn] for c in cs]).sum(axis=1)
                out[key + "_truth"], out[key + "_eq"] = tr[:, idx], eq[:, idx]
                sh = np.stack([by[n][0]["paths"][p][:, by[n][1]] for n in bn], axis=1)       # [seg, mol, COLS]
                out[key + "_x"] = sh[..., 0].sum(axis=1) / nh0.sum()
                out[key + "_superh"] = sh[..., C.COLS.index("f_superh")].mean(axis=1)
                out[key + "_atcap"] = sh[..., C.COLS.index("at_cap")].mean(axis=1)
        print(run, "done", flush=True)
    np.savez_compressed(args.out, **out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
