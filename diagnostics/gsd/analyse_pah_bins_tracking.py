"""Which molecular bins need their hydrogen tracked? Schemes scored on the whole PAH population.

Reads the per-molecule results of ``check_pah_bins_hydrogen.py`` (its .pkl: every molecule's chain along every path,
and its one-scalar closure family at every path condition) and assembles, without new physics, the PAH population
under each scheme:

- "all E": every bin at its equilibrium;
- "bin 1 M": bin 1 (N_C 20-42) tracked (its one-scalar closure, exact per cell), the others at equilibrium;
- "bin 1 M, steps": the same with RAMSES-like steps of 1e3 yr (rate frozen at each step);
- "all M": every bin tracked;
- "bin 1 split, M": bin 1 cut at ``--split-nc`` (default its median N_C) into two tracked bins (recombined with bin 1's own weights), the others
  at equilibrium.

Against the true mixture (every molecule's own chain), it scores the population's shares (intact, emitting
N_H0 ... N_H0-3, bare; by number within a bin, as the bins are built, and by carbon across bins) and its H content (sum over bins of the carbon weight times x = <N_H>/N_H0), as
the worst absolute error along each path, by the G0 class of the cell. The carbon per bin is an input:

- "equal": the same carbon in each of the four bins;
- "DL07-like": a lognormal in radius, a0 = 0.4 nm, sigma = 0.4 (the main PAH component of Draine & Li 2007),
  integrated over the bins' radius edges (rho = 1.6).

    python diagnostics/gsd/analyse_pah_bins_tracking.py --pkl ~/pycalima_runs/pah_bins_full.pkl --murga2020 DIR
"""
from __future__ import annotations

import argparse
import pickle
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_pah_bins_hydrogen as B    # noqa: E402
import check_pah_hydrogen_tracking as P2   # noqa: E402

OBS = ("x", "f_intact", "f_emit", "f_bare", "f_superh")   # f_superh: runs with the joint chain (2026-11-13 on)


def nc_of(name):
    return int(re.match(r"C(\d+)H", name).group(1))


def carbon_weights(kind):
    from pycalima.gsd.carbon import CarbonLayout

    a = CarbonLayout().grid().a_edges[:5] * 1e7            # nm
    if kind == "equal":
        return np.full(4, 0.25)
    a0, sig = 0.4, 0.4
    lo, hi = np.log(a[:-1] / a0), np.log(a[1:] / a0)
    grid = [np.linspace(l, h, 200) for l, h in zip(lo, hi)]
    # number per ln a ~ exp(-u^2 / 2 sigma^2), mass ~ a^3 = a0^3 e^(3u)
    m = np.array([np.trapezoid(np.exp(-u ** 2 / (2 * sig ** 2) + 3 * u), u) for u in grid])
    return m / m.sum()


class Bin:
    """One bin's series along a path: the truth, its equilibrium and its tracked closure."""

    def __init__(self, names, by_name, p_i, cs, ds):
        nh0 = np.array([by_name[n][0]["nh0"][by_name[n][1]] for n in names])
        pick = lambda get: np.stack([get(*by_name[n]) for n in names], axis=-2)        # noqa: E731
        self.ref = B.bin_obs(pick(lambda r, k: r["paths"][p_i][:, k]), nh0)
        self.eq = B.bin_obs(np.array([pick(lambda r, k: r["eq"][c][k]) for c in cs]), nh0)
        cl = {c: B.BinClosure(np.stack([by_name[n][0]["family"][c][by_name[n][1]] for n in names]), nh0)
              for c in set(cs)}
        x0 = float(B.bin_obs(pick(lambda r, k: r["eq"][cs[0]][k]), nh0)["x"])
        self.m = {}
        for tag, dt_yr in (("C", None), ("exp 1e+03", 1e3)):
            x, o = x0, {}
            for c, dt in zip(cs, ds):
                if dt_yr is None:
                    x = cl[c].step(x, dt)
                else:
                    n = max(int(np.ceil(dt / (dt_yr * P2.YR))), 1)
                    for _ in range(n):
                        x = cl[c].exp_step(x, dt / n)
                for k, v in cl[c].observables(x).items():
                    o.setdefault(k, []).append(v)
            self.m[tag] = {k: np.array(v) for k, v in o.items()}


def combine(parts):
    """Carbon-weighted population observables from [(weight, obs dict)]."""
    return {k: sum(w * o[k] for w, o in parts) for k in OBS if all(k in o for _, o in parts)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pkl", default="results/carbon_plan/check_pah_bins_hydrogen.pkl")
    ap.add_argument("--murga2020", default=str(Path.home() / "Library/Caches/calima/datasets/pahchem/murga2020"))
    ap.add_argument("--split-nc", type=int, default=None, help="N_C at which bin 1 is split (default: its median)")
    ap.add_argument("--out", default="results/carbon_plan/analyse_pah_bins_tracking.txt")
    args = ap.parse_args(argv)
    with open(args.pkl, "rb") as fh:
        d = pickle.load(fh)
    by_name = {n: (r, k) for r in d["results"] for k, n in enumerate(r["names"])}
    bins = [[n for n in names if n in by_name] for _, names in d["bins"]]
    _, paths = B.conditions(args.murga2020)
    SPLIT_NC = args.split_nc if args.split_nc is not None else int(np.median([nc_of(n) for n in bins[0]]))
    lines = [f"bins: {[len(b) for b in bins]} molecules; bin 1 split at N_C {SPLIT_NC}: "
             f"{sum(nc_of(n) < SPLIT_NC for n in bins[0])} + {sum(nc_of(n) >= SPLIT_NC for n in bins[0])}"]
    weights = {k: carbon_weights(k) for k in ("equal", "DL07-like")}
    for k, w in weights.items():
        lines.append(f"carbon per bin, {k}: " + ", ".join(f"{v:.3f}" for v in w))
    schemes = ("all E", "bin 1 M", "bin 1 M, steps", "all M", "bin 1 split, M")
    worst = {(wk, s, c[0]): (0.0, "") for wk in weights for s in schemes for c in P2.G0_CLASSES}
    rows = []
    for p_i, (name, cs, ds) in enumerate(paths):
        bs = [Bin(b, by_name, p_i, cs, ds) for b in bins]
        lo_names = [n for n in bins[0] if nc_of(n) < SPLIT_NC]
        hi_names = [n for n in bins[0] if nc_of(n) >= SPLIT_NC]
        halves = [Bin(h, by_name, p_i, cs, ds) for h in (lo_names, hi_names)]
        # bin 1 again from its halves, with its own weights: shares by number of molecules, x by N_H0
        n_lo = len(lo_names) / len(bins[0])
        nh = lambda ns: sum(by_name[n][0]["nh0"][by_name[n][1]] for n in ns)        # noqa: E731
        h_lo = nh(lo_names) / nh(bins[0])
        merged = {k: (h_lo if k == "x" else n_lo) * halves[0].m["C"][k]
                  + (1 - (h_lo if k == "x" else n_lo)) * halves[1].m["C"][k] for k in OBS}
        masks = P2.classes_of(cs)
        row = [name]
        for wk, w in weights.items():
            truth = combine([(w[b], bs[b].ref) for b in range(4)])
            est = {"all E": combine([(w[b], bs[b].eq) for b in range(4)]),
                   "bin 1 M": combine([(w[0], bs[0].m["C"])] + [(w[b], bs[b].eq) for b in range(1, 4)]),
                   "bin 1 M, steps": combine([(w[0], bs[0].m["exp 1e+03"])] + [(w[b], bs[b].eq) for b in range(1, 4)]),
                   "all M": combine([(w[b], bs[b].m["C"]) for b in range(4)]),
                   "bin 1 split, M": combine([(w[0], merged)] + [(w[b], bs[b].eq) for b in range(1, 4)])}
            cells = []
            for s in schemes:
                sc = B.score(est[s], truth, masks)
                cells.append("/".join(f"{sc[c[0]][0]:.2f}" if np.isfinite(sc[c[0]][0]) else "  - "
                                      for c in P2.G0_CLASSES))
                for c in P2.G0_CLASSES:
                    v = sc[c[0]]
                    if np.isfinite(v[0]) and v[0] > worst[(wk, s, c[0])][0]:
                        worst[(wk, s, c[0])] = (v[0], f"{v[1]}, {name}")
            row.append(cells)
        rows.append(row)
    for i, wk in enumerate(weights):
        lines.append(f"\n=== carbon per bin: {wk}; worst |error| of the population's x and carbon shares, by G0 class "
                     "(>=100 / 1-100 / <1) ===")
        lines.append(f"{'':52s} " + " ".join(f"{s:>17s}" for s in schemes))
        for row in rows:
            lines.append(f"{row[0]:52s} " + " ".join(f"{c:>17s}" for c in row[1 + i]))
        lines.append("worst over the paths:")
        for s in schemes:
            lines.append(f"   {s:15s} " + " | ".join(f"{c[0]}: {worst[(wk, s, c[0])][0]:.3f} ({worst[(wk, s, c[0])][1]})"
                                                  for c in P2.G0_CLASSES))
    text = "\n".join(lines)
    print(text)
    Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
