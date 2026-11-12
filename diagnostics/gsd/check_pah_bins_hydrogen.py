"""Tabulate or track the hydrogen of the molecular bins, with the full pahchem model (P2, PAH half, by bin).

Each molecular bin of the default ``CarbonLayout`` (N_C 20-42, 42-89, 89-189, 189-400) is the set of PAHdb v4.00
molecules in its range, built as pahchem builds them: ``population.library_v400`` (pure aromatic CH, from 21 C)
with the calibrated threshold shifts (``population.CALIBRATED_SHIFTS``, pahchem's default), the molecules with charges
-1..+2 (``group_tables.complete``), equal weights (``group_tables.nc_range_weights``), default ``Params``. Their
hydrogenation is ``Population``'s chain, N_H0 ... 0 with the intact molecule's charge
(``hydrogen_tracking.population_chains``; no super-hydrogenated states).

The molecules are sharded over ``--workers`` processes. Each prepares its molecules' bases for the four field shapes
and returns, per molecule and condition: the steady state, the one-scalar closure family (H addition scaled by a
common lambda), the evolution after the T1 steps and along every path. The bins are then assembled from those:
x = sum <N_H> / sum N_H0, and the number shares intact (N_H0), emitting (N_H0 ... N_H0-3, what ``Population``
emits) and bare (N_H = 0, where carbon is lost). It prints:

- T1 per bin: the steady state and the settling time (90 %) of the bin after a x10 step in G0, up and down, on a
  (G0, n(H)) grid (Draine shape without the old-star light, 100 K, n_e = 1.4e-4 n(H)), with a figure;
- T3/T4/T6 per bin: the paths of ``check_pah_hydrogen_tracking.py`` (NGC 7023 NW, Orion Bar, Horsehead at 0.3, 1,
  3 km/s in and out; CNM <-> WNM) and cycles of G0 x10 at n(H) = 10 across G0/n(H) = 1e-2 ... 1e4 (every bin's
  transition), periods 1e3, 1e4, 1e5 yr. The bin's mixture (each molecule's chain, conditions held per cell) is the
  reference for: E, the equilibrium; C, the bin's one-scalar closure (R = sum of the molecules' rates on the common
  lambda family); exp, its RAMSES-like step (rate frozen at each step of 1e2-1e4 yr). Worst absolute error over x
  and the three shares, by the G0 class of the cell;
- the treatment map per bin.

Run on megatron (``--workers`` 16-24 at nice 19: each worker holds up to ~3 GB while it builds a deposit
response). Writes ``--out``.npz/.txt/_T1.png. ``--per-bin N`` keeps N molecules per bin (spread in N_C) for a test.

    nice -n 19 python diagnostics/gsd/check_pah_bins_hydrogen.py --workers 20 --murga2020 DIR
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import scipy.linalg

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_pah_hydrogen_tracking as P2   # noqa: E402  (paths, profiles, scoring)

YR = P2.YR
LOG_LAM = np.linspace(-30.0, 30.0, 61)
OBS = ("x", "f_intact", "f_emit", "f_bare")
N_EMITTING = 4
G0_T1 = 10.0 ** np.arange(-3.0, 5.01, 0.5)
NH_T1 = 10.0 ** np.arange(-1.0, 6.01, 0.5)
CYCLE_G0 = (0.1, 1.0, 10.0, 1e2, 1e3, 1e4)                 # lower G0 of each x10 cycle, at n(H) = 10
CYCLE_PERIODS = (1e3, 1e4, 1e5)


def library():
    from pycalima import _paths
    from pycalima.pahchem import group_tables as G
    from pycalima.pahchem import population

    lib = population.library_v400(min_nc=21, cache_dir=_paths.get_model_data_dir() / "pahchem_pahdb_index")
    return [m for m in population.shift_thresholds(lib, population.CALIBRATED_SHIFTS) if G.complete(m)]


def bins_of(lib, per_bin=None):
    from pycalima.gsd.carbon import M_C, CarbonLayout

    e = CarbonLayout().grid().m_edges[:5] / M_C
    out = []
    for lo, hi in zip(e[:-1], e[1:]):
        sel = sorted((m for m in lib if lo <= m.nc < hi), key=lambda m: (m.nc, m.name))
        if per_bin and len(sel) > per_bin:
            sel = [sel[int(i)] for i in np.linspace(0, len(sel) - 1, per_bin)]
        out.append(((lo, hi), [m.name for m in sel]))
    return out


def environment(cond):
    from pycalima.pahchem.baseline import Environment

    g0, n_h, n_e, t, teff = cond
    return Environment("cell", g0, n_h, n_e, t, teff, optical_isrf=0.0)


def conditions(murga_root):
    """(T1 steps, paths): T1 as {(i_g0, j_nh): (cond, cond_from_below, cond_from_above)}, paths as [(name, conds, durs)]."""
    t1 = {}
    pad = 10.0 ** np.arange(np.log10(G0_T1[0]) - 1, np.log10(G0_T1[-1]) + 1 + 1e-9, 0.5)
    for j, nh in enumerate(NH_T1):
        c = {g: P2.cond_key(g, nh, 1.4e-4 * nh, 100.0, None) for g in pad}
        keys = list(c)
        for i, g in enumerate(G0_T1):
            k = keys.index(min(keys, key=lambda v: abs(np.log(v / g))))
            t1[(i, j)] = (c[keys[k]], c[keys[k - 2]], c[keys[k + 2]])
    nodes = {"NGC 7023": P2.ngc7023_nodes(), "Orion Bar": P2.murga_nodes("Orion Bar", murga_root),
             "Horsehead": P2.murga_nodes("Horsehead", murga_root)}
    paths = []
    for site, (av, q) in nodes.items():
        for v in P2.SPEEDS_KMS:
            for inward in (True, False):
                cs, ds, _ = P2.profile_path(av, q, P2.SITES[site], v, inward)
                paths.append((f"{site} {'in ' if inward else 'out'} {v:g} km/s", cs, ds))
    for period in (1e3, 1e4):
        cs, ds, _ = P2.cycle_path(period)
        paths.append((f"CNM<->WNM, period {period:.0e} yr", cs, ds))
    for g in CYCLE_G0:
        lo, hi = (g, 10.0, 1.4e-3, 100.0, None), (10.0 * g, 10.0, 1.4e-3, 100.0, None)
        for period in CYCLE_PERIODS:
            cs, ds, _ = P2.cycle_path(period, lo, hi)
            paths.append((f"G0 {g:g}<->{10 * g:g} at n(H)=10, period {period:.0e} yr", cs, ds))
    return t1, paths


# -- per molecule (workers) ------------------------------------------------------------------------
def _chain_obs(p, nh0):
    nh = np.arange(nh0 + 1)
    return np.array([p @ nh, p[nh0], p[max(nh0 - N_EMITTING + 1, 0):].sum(), p[0]])     # <N_H>, intact, emit, bare


def _family(chain, nh0):
    """[n_lam, 1 + len(OBS)]: rate d<N_H>/dt [s^-1] and the observables of the lambda-scaled steady states."""
    out = []
    for ll in LOG_LAM:
        net = P2.ht._scaled(chain, np.exp(ll))
        try:
            p = net.steady_state()[0]
        except ValueError:
            out.append(np.full(1 + len(OBS), np.nan))
            continue
        r = sum(q.dnh * (p * q.rate[0]).sum() for q in chain.processes)
        out.append(np.r_[r, _chain_obs(p, nh0)])
    return np.array(out)


def work(job):
    names, t1, paths, cache_dir = job
    import os

    os.environ.setdefault("CALIMA_WORKERS", "1")
    from pycalima.pahchem import population
    from pycalima.pahchem import hydrogen_tracking as ht

    lib = {m.name: m for m in library()}
    mols = [lib[n] for n in names]
    pop = population.Population(mols, {"all": (1.0, 1e4)}, cache_dir=cache_dir)
    conds = sorted({c for v in t1.values() for c in v} | {c for _, cs, _ in paths for c in cs},
                   key=lambda c: (str(c[4]), c))
    chains, prepared, dropped = {}, set(), []
    k = 0
    while k < len(conds):
        c = conds[k]
        env = environment(c)
        shape = population.field_shape(env)
        try:
            if shape not in prepared:
                pop.prepare(shape, workers=1)
                prepared.add(shape)
            chains[c] = ht.population_chains(pop, env)
        except ValueError as exc:
            # a molecule whose own physics raises (e.g. population._round_off): drop it from this shard, record it
            bad = next((i for i, m in enumerate(mols) if str(exc).startswith(m.name + " ")), None)
            if bad is None:
                raise
            dropped.append((mols[bad].name, f"at {c}: {exc}"))
            del mols[bad]
            for v in chains.values():
                del v[bad]
            if not mols:
                return {"names": [], "nh0": np.zeros(0, int), "eq": {}, "family": {}, "t1": {}, "paths": [],
                        "dropped": dropped}
            pop = population.Population(mols, {"all": (1.0, 1e4)}, cache_dir=cache_dir)
            prepared = set()
            continue
        k += 1
    names = [m.name for m in mols]
    nh0 = np.array([m.nh for m in mols])
    steady = {c: [ch.steady_state()[0] for ch in chains[c]] for c in conds}
    path_conds = {c for _, cs, _ in paths for c in cs}
    family = {c: np.array([_family(ch, n) for ch, n in zip(chains[c], nh0)]) for c in path_conds}
    eq = {c: np.array([_chain_obs(p, n) for p, n in zip(steady[c], nh0)]) for c in conds}
    ts = P2.T_EVOLVE_YR * YR
    t1_out = {}
    for key, (c, below, above) in t1.items():
        Q = [ch.generator().toarray() for ch in chains[c]]
        series = []
        for start in (below, above):
            s = []
            for q, p0, n in zip(Q, steady[start], nh0):
                f, prev, row = p0, 0.0, []
                for t in ts:
                    f = np.clip(scipy.linalg.expm(q * (t - prev)) @ f, 0.0, None)
                    prev = t
                    row.append(_chain_obs(f / f.sum(), n))
                s.append(row)
            series.append(np.array(s))                   # [mol, t, obs]
        t1_out[key] = series
    path_out = []
    for _, cs, ds in paths:
        f = [p.copy() for p in steady[cs[0]]]
        rows = []
        for c, dt in zip(cs, ds):
            for k, ch in enumerate(chains[c]):
                g = np.clip(scipy.linalg.expm(ch.generator().toarray() * dt) @ f[k], 0.0, None)
                f[k] = g / g.sum()
            rows.append([_chain_obs(p, n) for p, n in zip(f, nh0)])
        path_out.append(np.array(rows))                  # [segment, mol, obs]
    return {"names": names, "nh0": nh0, "eq": eq, "family": family, "t1": t1_out, "paths": path_out, "dropped": dropped}


# -- bins (master) ---------------------------------------------------------------------------------
def bin_obs(per_mol, nh0):
    """Bin observables from per-molecule [..., mol, (<N_H>, intact, emit, bare)]."""
    a = np.asarray(per_mol)
    return {"x": a[..., 0].sum(axis=-1) / nh0.sum(), "f_intact": a[..., 1].mean(axis=-1),
            "f_emit": a[..., 2].mean(axis=-1), "f_bare": a[..., 3].mean(axis=-1)}


class BinClosure:
    """The bin's one-scalar closure from the molecules' lambda families at one condition."""

    def __init__(self, fam, nh0):
        i_eq = int(np.argmin(np.abs(LOG_LAM)))
        self.x_eq = float(fam[:, i_eq, 1].sum() / nh0.sum())
        f = fam[:, ~np.isnan(fam).any(axis=(0, 2))]              # [mol, lam, 1 + obs]: lambdas every molecule has
        x = f[:, :, 1].sum(axis=0) / nh0.sum()
        rate = f[:, :, 0].sum(axis=0) / nh0.sum()
        obs = {"x": x, "f_intact": f[:, :, 2].mean(axis=0), "f_emit": f[:, :, 3].mean(axis=0),
               "f_bare": f[:, :, 4].mean(axis=0)}
        keep = [0]
        for i in range(1, x.size):
            if x[i] > x[keep[-1]] + 1e-9:
                keep.append(i)
        self.x, self.rate = x[keep], rate[keep]
        self.obs = {k: v[keep] for k, v in obs.items()}

    def dxdt(self, x):
        return float(np.interp(x, self.x, self.rate))

    def observables(self, x):
        return {k: float(np.interp(x, self.x, v)) for k, v in self.obs.items()}

    def step(self, x, dt):
        from scipy.integrate import solve_ivp

        if dt <= 0 or self.dxdt(x) == 0.0 or self.x.size < 2:
            return x
        y = float(solve_ivp(lambda t, y: [self.dxdt(y[0])], (0.0, dt), [x], method="LSODA", rtol=1e-8,
                            atol=1e-10).y[0, -1])
        lo, hi = sorted((x, self.x_eq))
        return float(np.clip(y, lo, hi))

    def exp_step(self, x, dt):
        if abs(x - self.x_eq) < 1e-9 or self.x.size < 2:
            return x
        k = max(-self.dxdt(x) / (x - self.x_eq), 0.0)
        return self.x_eq + (x - self.x_eq) * np.exp(-k * dt)


def score(a, b, masks):
    out = {}
    for name, m in masks.items():
        if not m.any():
            out[name] = (np.nan, "")
            continue
        errs = {k: float(np.max(np.abs(np.asarray(a[k])[m] - np.asarray(b[k])[m]))) for k in OBS}
        k = max(errs, key=errs.get)
        out[name] = (errs[k], k)
    return out


def assemble(results, bins, t1, paths):
    ts = P2.T_EVOLVE_YR * YR
    by_name = {}
    for r in results:
        for k, n in enumerate(r["names"]):
            by_name[n] = (r, k)
    report = {}
    for (lo, hi), names in bins:
        if not names:
            continue
        nh0 = np.array([by_name[n][0]["nh0"][by_name[n][1]] for n in names])
        pick = lambda get: np.stack([get(*by_name[n]) for n in names], axis=-2)        # noqa: E731
        maps = {k: np.zeros((G0_T1.size, NH_T1.size)) for k in ("x", "f_intact", "f_bare", "t_up", "t_down")}
        for (i, j), (c, _, _) in t1.items():
            e = bin_obs(pick(lambda r, k: r["eq"][c][k]), nh0)
            for k in ("x", "f_intact", "f_bare"):
                maps[k][i, j] = e[k]
            for tag, s in zip(("t_up", "t_down"), range(2)):
                o = bin_obs(pick(lambda r, k: r["t1"][(i, j)][s][k]), nh0)        # [t, mol, obs] -> per t
                tt = [P2.ht.settling_time(ts, o[q], y_inf=e[q]) for q in OBS if abs(o[q][0] - e[q]) > 0.01]
                maps[tag][i, j] = max(tt) / YR if tt else 0.0
        res_paths = []
        for p_i, (name, cs, ds) in enumerate(paths):
            ref = bin_obs(pick(lambda r, k: r["paths"][p_i][:, k]), nh0)
            eq = bin_obs(np.array([pick(lambda r, k: r["eq"][c][k]) for c in cs]), nh0)
            masks = P2.classes_of(cs)
            res = {"E": score(eq, ref, masks)}
            cl = {c: BinClosure(np.stack([by_name[n][0]["family"][c][by_name[n][1]] for n in names]), nh0)
                  for c in set(cs)}
            x0 = float(bin_obs(pick(lambda r, k: r["eq"][cs[0]][k]), nh0)["x"])
            for tag, dt_yr in (("C", None), ("exp 1e+02", 1e2), ("exp 1e+03", 1e3), ("exp 1e+04", 1e4)):
                x, o = x0, {k: [] for k in OBS}
                for c, dt in zip(cs, ds):
                    if dt_yr is None:
                        x = cl[c].step(x, dt)
                    else:
                        n = max(int(np.ceil(dt / (dt_yr * YR))), 1)
                        for _ in range(n):
                            x = cl[c].exp_step(x, dt / n)
                    for k, v in cl[c].observables(x).items():
                        o[k].append(v)
                res[tag] = score({k: np.array(v) for k, v in o.items()}, ref, masks)
            res_paths.append((name, res, (float(ref["x"].min()), float(ref["x"].max()))))
        report[(lo, hi)] = {"n": len(names), "maps": maps, "paths": res_paths}
    return report


def plot_t1(report, out_png):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    from pycalima.plotting_style import use_calima_style

    use_calima_style()
    fig, axs = plt.subplots(1, len(report), figsize=(4.6 * len(report), 4.4), squeeze=False)
    X, Y = np.meshgrid(np.log10(NH_T1), np.log10(G0_T1))
    for ax, ((lo, hi), r) in zip(axs[0], report.items()):
        mp = r["maps"]
        t = np.clip(np.maximum(mp["t_up"], mp["t_down"]), 1e-6, 1e6)
        im = ax.pcolormesh(X, Y, t, norm=LogNorm(1e-4, 1e6), cmap="viridis", shading="nearest")
        fig.colorbar(im, ax=ax, label="H settling (90 %) [yr]")
        cs = ax.contour(X, Y, mp["f_intact"], levels=[0.1, 0.5, 0.9], colors="w", linewidths=1.0)
        ax.clabel(cs, fmt="intact %.1f", fontsize=8)
        ax.set_title(f"bin N$_{{\\rm C}}$ {lo:.0f}-{hi:.0f} ({r['n']} molecules)", fontsize=12)
        ax.set_xlabel(r"$\log\,n$(H) [cm$^{-3}$]", fontsize=13)
        ax.set_ylabel(r"$\log\,G_0$ [Habing]", fontsize=13)
        ax.minorticks_on()
        ax.tick_params(which="both", axis="both", direction="in")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    print(f"wrote {out_png}")


def print_report(report, out_txt):
    lines = []
    show = ("E", "C", "exp 1e+03", "exp 1e+04")
    for (lo, hi), r in report.items():
        mp = r["maps"]
        lines.append(f"\n=== bin N_C {lo:.0f}-{hi:.0f}: {r['n']} molecules ===")
        lines.append("T1 (rows G0; at n(H) = 1e1 / 1e3 / 1e5): x, f_intact, f_bare | settling up, down [yr]")
        for i, g0 in enumerate(G0_T1):
            cells = []
            for nh in (1e1, 1e3, 1e5):
                j = int(np.argmin(np.abs(np.log(NH_T1 / nh))))
                cells.append(f"{mp['x'][i, j]:4.2f} {mp['f_intact'][i, j]:4.2f} {mp['f_bare'][i, j]:4.2f} | "
                             f"{mp['t_up'][i, j]:8.2g} {mp['t_down'][i, j]:8.2g}")
            lines.append(f"  G0 {g0:8.2g}: " + "  ||  ".join(cells))
        lines.append("paths: worst |error| over x, f_intact, f_emit, f_bare by G0 class (>=100 / 1-100 / <1)")
        lines.append(f"{'':52s} " + " ".join(f"{s:>17s}" for s in show))
        worst = {(s, c[0]): (0.0, "") for s in show for c in P2.G0_CLASSES}
        for name, res, (xlo, xhi) in r["paths"]:
            cells = ["/".join(f"{res[s][c[0]][0]:.2f}" if np.isfinite(res[s][c[0]][0]) else "  - "
                              for c in P2.G0_CLASSES) for s in show]
            lines.append(f"{name:52s} " + " ".join(f"{c:>17s}" for c in cells) + f"  x {xlo:.2f}-{xhi:.2f}")
            for s in show:
                for c in P2.G0_CLASSES:
                    v = res[s][c[0]]
                    if np.isfinite(v[0]) and v[0] > worst[(s, c[0])][0]:
                        worst[(s, c[0])] = (v[0], f"{v[1]}, {name}")
        lines.append("treatment map (worst over the paths):")
        for s in show:
            lines.append(f"   {s:10s} " + " | ".join(f"{c[0]}: {worst[(s, c[0])][0]:.3f} ({worst[(s, c[0])][1]})"
                                                 for c in P2.G0_CLASSES))
    text = "\n".join(lines)
    print(text)
    Path(out_txt).write_text(text + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--per-bin", type=int, default=None, help="molecules per bin, spread in N_C (default: all)")
    ap.add_argument("--murga2020", default=str(Path.home() / "Library/Caches/calima/datasets/pahchem/murga2020"))
    ap.add_argument("--out", default="results/carbon_plan/check_pah_bins_hydrogen")
    args = ap.parse_args(argv)
    from concurrent.futures import ProcessPoolExecutor

    from pycalima import _paths

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    lib = library()
    bins = bins_of(lib, args.per_bin)
    nc = {m.name: m.nc for m in lib}
    for (lo, hi), names in bins:
        print(f"bin N_C {lo:.0f}-{hi:.0f}: {len(names)} molecules" + (f", N_C {nc[names[0]]}-{nc[names[-1]]}" if names else ""))
    t1, paths = conditions(args.murga2020)
    every = sorted((n for _, names in bins for n in names), key=lambda n: -nc[n])
    shards = [every[k::args.workers] for k in range(args.workers)]
    shards = [s for s in shards if s]
    cache = _paths.get_model_data_dir() / "pahchem_population_basis"
    print(f"{len(every)} molecules on {len(shards)} workers; {len(t1)} T1 cells, {len(paths)} paths", flush=True)
    from concurrent.futures import as_completed

    results = []
    with ProcessPoolExecutor(len(shards)) as ex:
        futures = [ex.submit(work, (s, t1, paths, cache)) for s in shards]
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            for name, why in r["dropped"]:
                print(f"  dropped {name}: {why}", flush=True)
            print(f"  shard {len(results)}/{len(shards)} done ({time.time() - t0:.0f} s)", flush=True)
            with open(out.with_suffix(".pkl"), "wb") as fh:      # every shard as it comes: a failure keeps the rest
                pickle.dump({"bins": bins, "results": results, "paths": [(n, len(c)) for n, c, _ in paths]}, fh)
    kept = {n for r in results for n in r["names"]}
    bins = [(edges, [n for n in names if n in kept]) for edges, names in bins]
    report = assemble(results, bins, t1, paths)
    print_report(report, out.with_suffix(".txt"))
    plot_t1(report, out.with_name(out.name + "_T1.png"))
    print(f"total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
