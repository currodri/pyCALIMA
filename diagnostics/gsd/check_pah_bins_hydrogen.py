"""Tabulate or track the hydrogen of the molecular bins, with the full pahchem model (P2, PAH half, by bin).

Each molecular bin of the default ``CarbonLayout`` (N_C 20-42, 42-89, 89-189, 189-400) is the set of PAHdb v4.00
molecules in its range, built as pahchem builds them: ``population.library_v400`` (pure aromatic CH, from 21 C)
with the calibrated threshold shifts (``population.CALIBRATED_SHIFTS``, pahchem's default), the molecules with charges
-1..+2 (``group_tables.complete``), equal weights (``group_tables.nc_range_weights``), default ``Params``. Their
hydrogenation is ``Population``'s joint (Z, N_H) chain (``--chain joint``, the default: ``Population.joint_networks``
with ``Params.hydrogen_chain = "joint"``, super-hydrogenated states to N_H0 + ``superh_levels``, 4 unless
``--superh-levels``, their chemistry ``--superh-chemistry``, and a charge per N_H), or its
charge-averaged chain (``--chain averaged``: N_H0 ... 0 with the intact molecule's charge,
``hydrogen_tracking.population_chains``; the 2026-11-08 run). Steady states are ``steady_state_closed``.

The molecules are sharded over ``--workers`` processes. Each prepares its molecules' bases for the four field shapes
and returns, per molecule and condition: the steady state, the one-scalar closure family (H addition scaled by a
common lambda), the evolution after the T1 steps and along every path. The bins are then assembled from those:
x = sum <N_H> / sum N_H0, and the number shares intact (N_H0), emitting (N_H0 ... N_H0-3 and the super-hydrogenated
states, which emit as the parent: what ``Population`` emits), bare (N_H = 0, where carbon is lost) and
super-hydrogenated (> N_H0); and, reported but not scored, the share at the cap N_H0 + K (where it holds more than a
few per cent the cap, not the chemistry, sets the super-hydrogenation). It prints:

- T1 per bin: the steady state and the settling time (90 %) of the bin after a x10 step in G0, up and down, on a
  (G0, n(H)) grid (Draine shape without the old-star light, 100 K, n_e = 1.4e-4 n(H)), with a figure;
- T3/T4/T6 per bin: the paths of ``check_pah_hydrogen_tracking.py`` (NGC 7023 NW, Orion Bar, Horsehead at 0.3, 1,
  3 km/s in and out; CNM <-> WNM) and cycles of G0 x10 at n(H) = 10 across G0/n(H) = 1e-2 ... 1e4 (every bin's
  transition), periods 1e3, 1e4, 1e5 yr. The bin's mixture (each molecule's chain, conditions held per cell) is the
  reference for: E, the equilibrium; C, the bin's one-scalar closure (R = sum of the molecules' rates on the common
  lambda family); exp, its RAMSES-like step (rate frozen at each step of 1e2-1e4 yr). Worst absolute error over x
  and the four shares, by the G0 class of the cell;
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
OBS = ("x", "f_intact", "f_emit", "f_bare", "f_superh")                     # scored
COLS = ("n_h", "f_intact", "f_emit", "f_bare", "f_superh", "at_cap")         # per molecule (_chain_obs)
N_EMITTING = 4
G0_T1 = 10.0 ** np.arange(-3.0, 5.01, 0.5)
NH_T1 = 10.0 ** np.arange(-1.0, 6.01, 0.5)
CYCLE_G0 = (0.1, 1.0, 10.0, 1e2, 1e3, 1e4)                 # lower G0 of each x10 cycle, at n(H) = 10
CYCLE_PERIODS = (1e3, 1e4, 1e5)
# per molecule, joint chain only (pahchem.hydrogen_observables): what the hydrogenation changes, scored for E
OBS_RATES = ("carbon", "heating", "cooling", "h2", "h_uptake", "absorbed", "uv_aro", "uv_ali")
OBS_SHOW = ("carbon", "heating", "cooling", "h2", "h_uptake", "r_ali")      # r_ali = sum uv_ali / sum uv_aro
OBS_SIGNIFICANT = 1e-3            # cells scored: above this share of the path maximum (score_obs)
CARBON_FLOOR_S = 1.0 / (1e9 * YR)  # carbon scored only where the bin loses its carbon within 1 Gyr (a reporting threshold)


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
def _chain_obs(f, nh0):
    """COLS of f[Z, N_H] (or f[N_H]): <N_H>, intact, emitting (N_H0 ... N_H0-3 and the super-hydrogenated states, which
    emit as the parent, as in ``Population.hydrogenation_fractions``), bare, super-hydrogenated, at the cap."""
    p = f.sum(axis=0) if f.ndim == 2 else f
    nh = np.arange(p.size)
    return np.array([p @ nh, p[nh0], p[max(nh0 - N_EMITTING + 1, 0):].sum(), p[0], p[nh0 + 1:].sum(),
                     p[-1] if p.size > nh0 + 1 else 0.0])


def _family(chain, nh0):
    """[n_lam, 1 + len(COLS)]: rate d<N_H>/dt [s^-1] and the observables of the lambda-scaled steady states."""
    out = []
    for ll in LOG_LAM:
        net = P2.ht._scaled(chain, np.exp(ll))
        try:
            g = net.steady_state_closed()
        except ValueError:
            out.append(np.full(1 + len(COLS), np.nan))
            continue
        r = sum(q.dnh * (g * q.rate).sum() for q in chain.processes)
        out.append(np.r_[r, _chain_obs(g, nh0)])
    return np.array(out)


def _flat(f, net):
    """f[Z, N_H] as the state vector of ``net.generator`` (Z-major); the conditions of one path share the field shape,
    and so the live charges."""
    if f.shape != net.shape:
        raise ValueError(f"a distribution of shape {f.shape} carried into a network of shape {net.shape}")
    return f.ravel()


def _obs_row(w, f):
    """OBS_RATES of one molecule's weights (``hydrogen_observables.state_weights``) and state f[Z, N_H]."""
    return [float((w[k] * f).sum()) for k in OBS_RATES]


def work(job):
    names, t1, paths, cache_dir, chain, superh = job
    import os

    os.environ.setdefault("CALIMA_WORKERS", "1")
    from pycalima.pahchem import population
    from pycalima.pahchem import hydrogen_tracking as ht

    lib = {m.name: m for m in library()}
    mols = [lib[n] for n in names]
    params = population.Params(hydrogen_chain="joint", **superh) if chain == "joint" else population.Params()
    pop = population.Population(mols, {"all": (1.0, 1e4)}, cache_dir=cache_dir, params=params)
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
            chains[c] = pop.joint_networks(env) if chain == "joint" else ht.population_chains(pop, env)
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
                return {"names": [], "nh0": np.zeros(0, int), "nc": np.zeros(0, int), "eq": {}, "family": {}, "t1": {},
                        "paths": [], "dropped": dropped, "obs_eq": {}, "path_obs": []}
            pop = population.Population(mols, {"all": (1.0, 1e4)}, cache_dir=cache_dir, params=params)
            prepared = set()
            continue
        k += 1
    names = [m.name for m in mols]
    nh0 = np.array([m.nh for m in mols])
    steady = {c: [ch.steady_state_closed() for ch in chains[c]] for c in conds}
    path_conds = {c for _, cs, _ in paths for c in cs}
    family = {c: np.array([_family(ch, n) for ch, n in zip(chains[c], nh0)]) for c in path_conds}
    eq = {c: np.array([_chain_obs(p, n) for p, n in zip(steady[c], nh0)]) for c in conds}
    W, obs_eq = {}, {}
    if chain == "joint":                       # the observables' weights on each path condition, with the final library
        from pycalima.pahchem import hydrogen_observables as ho

        for c in sorted(path_conds, key=lambda c: (str(c[4]), c)):
            W[c] = ho.state_weights(pop, environment(c), chains[c])
            obs_eq[c] = np.array([_obs_row(w, p) for w, p in zip(W[c], steady[c])])
    ts = P2.T_EVOLVE_YR * YR
    t1_out = {}
    for key, (c, below, above) in t1.items():
        Q = [ch.generator().toarray() for ch in chains[c]]
        series = []
        for start in (below, above):
            s = []
            for q, p0, n, ch in zip(Q, steady[start], nh0, chains[c]):
                f, prev, row = _flat(p0, ch), 0.0, []
                for t in ts:
                    f = np.clip(scipy.linalg.expm(q * (t - prev)) @ f, 0.0, None)
                    prev = t
                    row.append(_chain_obs((f / f.sum()).reshape(ch.shape), n))
                s.append(row)
            series.append(np.array(s))                   # [mol, t, obs]
        t1_out[key] = series
    path_out, path_obs = [], []
    for _, cs, ds in paths:
        f = [p.copy() for p in steady[cs[0]]]
        rows, orows = [], []
        for c, dt in zip(cs, ds):
            for k, ch in enumerate(chains[c]):
                g = np.clip(scipy.linalg.expm(ch.generator().toarray() * dt) @ _flat(f[k], ch), 0.0, None)
                f[k] = (g / g.sum()).reshape(ch.shape)
            rows.append([_chain_obs(p, n) for p, n in zip(f, nh0)])
            if W:
                orows.append([_obs_row(w, p) for w, p in zip(W[c], f)])
        path_out.append(np.array(rows))                  # [segment, mol, obs]
        path_obs.append(np.array(orows))                 # [segment, mol, OBS_RATES] (empty: averaged chain)
    return {"names": names, "nh0": nh0, "nc": np.array([m.nc for m in mols]), "eq": eq, "family": family, "t1": t1_out,
            "paths": path_out, "dropped": dropped, "obs_eq": obs_eq, "path_obs": path_obs}


# -- bins (master) ---------------------------------------------------------------------------------
def bin_obs(per_mol, nh0):
    """Bin observables from per-molecule [..., mol, (<N_H>, intact, emit, bare)]."""
    a = np.asarray(per_mol)
    out = {"x": a[..., 0].sum(axis=-1) / nh0.sum(), "f_intact": a[..., 1].mean(axis=-1),
           "f_emit": a[..., 2].mean(axis=-1), "f_bare": a[..., 3].mean(axis=-1)}
    if a.shape[-1] > 4:                              # runs before 2026-11-13 kept the first four
        out["f_superh"], out["at_cap"] = a[..., 4].mean(axis=-1), a[..., 5].mean(axis=-1)
    return out


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
        if f.shape[-1] > 5:
            obs["f_superh"], obs["at_cap"] = f[:, :, 5].mean(axis=0), f[:, :, 6].mean(axis=0)
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
        errs = {k: float(np.max(np.abs(np.asarray(a[k])[m] - np.asarray(b[k])[m]))) for k in OBS if k in a and k in b}
        k = max(errs, key=errs.get)
        out[name] = (errs[k], k)
    return out


def _bin_rates(a):
    """{OBS_SHOW name: [segment]} of the bin from per-molecule OBS_RATES [segment, mol, rate] (equal weights: sums)."""
    tot = np.asarray(a).sum(axis=-2)
    out = {k: tot[..., OBS_RATES.index(k)] for k in OBS_SHOW if k in OBS_RATES}
    out["r_ali"] = tot[..., OBS_RATES.index("uv_ali")] / np.maximum(tot[..., OBS_RATES.index("uv_aro")], 1e-300)
    out["uv_ali"], out["uv_aro"] = tot[..., OBS_RATES.index("uv_ali")], tot[..., OBS_RATES.index("uv_aro")]
    return out


def _factor(a, b):
    """max(a/b, b/a) - 1, element-wise (0 where both are 0, inf where one is)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    hi, lo = np.maximum(a, b), np.minimum(a, b)
    return np.where(hi == 0.0, 0.0, np.divide(hi, lo, out=np.full_like(hi, np.inf), where=lo > 0) - 1.0)


def score_obs(e, ref, ds, n_c=None):
    """{name: (cell, budget)} of E against the truth on a path's observables (``_bin_rates``, durations ``ds``), each the
    factor error max(E/truth, truth/E) - 1:

    - cell: the worst over the cells where E or the truth holds more than OBS_SIGNIFICANT of the path maximum; carbon
      only where it is also lost within 1 Gyr (CARBON_FLOOR_S per C atom, ``n_c`` the bin's carbon atoms); r_ali where
      the truth's uv_aro (the emission) holds more than OBS_SIGNIFICANT of its path maximum and r_ali exceeds
      OBS_SIGNIFICANT;
    - budget: of the path integrals (r_ali: the integrated uv_ali over uv_aro, the time-averaged emission ratio, scored
      where it exceeds OBS_SIGNIFICANT; carbon where the path-averaged loss passes the floor).

    NaN where nothing qualifies."""
    ds = np.asarray(ds, float)
    out = {}
    for k in OBS_SHOW:
        a, b = np.asarray(e[k], float), np.asarray(ref[k], float)
        top = np.maximum(a, b)
        if k == "r_ali":
            aro = np.asarray(ref["uv_aro"], float)
            sig = (aro > OBS_SIGNIFICANT * aro.max()) & (top > OBS_SIGNIFICANT)
            ratio = lambda d: (d["uv_ali"] * ds).sum() / max((d["uv_aro"] * ds).sum(), 1e-300)      # noqa: E731
            bi, bt = ratio(e), ratio(ref)
            keep_budget = max(bi, bt) > OBS_SIGNIFICANT
        else:
            sig = top > OBS_SIGNIFICANT * top.max()
            bi, bt = (a * ds).sum(), (b * ds).sum()
            keep_budget = max(bi, bt) > 0.0
            if k == "carbon" and n_c:
                sig &= top / n_c > CARBON_FLOOR_S
                keep_budget &= max(bi, bt) / ds.sum() / n_c > CARBON_FLOOR_S
        cell = float(np.max(_factor(a[sig], b[sig]))) if sig.any() else np.nan
        out[k] = (cell, float(_factor(bi, bt)) if keep_budget else np.nan)
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
        maps = {k: np.zeros((G0_T1.size, NH_T1.size))
                for k in ("x", "f_intact", "f_bare", "f_superh", "at_cap", "t_up", "t_down")}
        for (i, j), (c, _, _) in t1.items():
            e = bin_obs(pick(lambda r, k: r["eq"][c][k]), nh0)
            for k in ("x", "f_intact", "f_bare", "f_superh", "at_cap"):
                maps[k][i, j] = e.get(k, 0.0)
            for tag, s in zip(("t_up", "t_down"), range(2)):
                o = bin_obs(pick(lambda r, k: r["t1"][(i, j)][s][k]), nh0)        # [t, mol, obs] -> per t
                tt = [P2.ht.settling_time(ts, o[q], y_inf=e[q]) for q in OBS if q in o and abs(o[q][0] - e[q]) > 0.01]
                maps[tag][i, j] = max(tt) / YR if tt else 0.0
        res_paths, res_obs = [], []
        for p_i, (name, cs, ds) in enumerate(paths):
            ref = bin_obs(pick(lambda r, k: r["paths"][p_i][:, k]), nh0)
            eq = bin_obs(np.array([pick(lambda r, k: r["eq"][c][k]) for c in cs]), nh0)
            masks = P2.classes_of(cs)
            res = {"E": score(eq, ref, masks)}
            cl = {c: BinClosure(np.stack([by_name[n][0]["family"][c][by_name[n][1]] for n in names]), nh0)
                  for c in set(cs)}
            x0 = float(bin_obs(pick(lambda r, k: r["eq"][cs[0]][k]), nh0)["x"])
            for tag, dt_yr in (("C", None), ("exp 1e+02", 1e2), ("exp 1e+03", 1e3), ("exp 1e+04", 1e4)):
                x, o = x0, {}
                for c, dt in zip(cs, ds):
                    if dt_yr is None:
                        x = cl[c].step(x, dt)
                    else:
                        n = max(int(np.ceil(dt / (dt_yr * YR))), 1)
                        for _ in range(n):
                            x = cl[c].exp_step(x, dt / n)
                    for k, v in cl[c].observables(x).items():
                        o.setdefault(k, []).append(v)
                res[tag] = score({k: np.array(v) for k, v in o.items()}, ref, masks)
            res_paths.append((name, res, (float(ref["x"].min()), float(ref["x"].max()),
                                          float(np.max(ref.get("at_cap", 0.0))))))
            if all(len(by_name[n][0].get("path_obs", [])) for n in names):      # joint chain: the observables
                ref_o = _bin_rates(pick(lambda r, k: r["path_obs"][p_i][:, k]))
                eq_o = _bin_rates(np.array([pick(lambda r, k: r["obs_eq"][c][k]) for c in cs]))
                n_c = sum(by_name[n][0]["nc"][by_name[n][1]] for n in names) if "nc" in results[0] else None
                res_obs.append((name, score_obs(eq_o, ref_o, ds, n_c)))
        report[(lo, hi)] = {"n": len(names), "maps": maps, "paths": res_paths, "obs": res_obs}
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
        lines.append("T1 (rows G0; at n(H) = 1e1 / 1e3 / 1e5): x, f_intact, f_bare, f_superh, at_cap | settling up, "
                     "down [yr]")
        for i, g0 in enumerate(G0_T1):
            cells = []
            for nh in (1e1, 1e3, 1e5):
                j = int(np.argmin(np.abs(np.log(NH_T1 / nh))))
                cells.append(f"{mp['x'][i, j]:4.2f} {mp['f_intact'][i, j]:4.2f} {mp['f_bare'][i, j]:4.2f} "
                             f"{mp['f_superh'][i, j]:4.2f} {mp['at_cap'][i, j]:4.2f} | "
                             f"{mp['t_up'][i, j]:8.2g} {mp['t_down'][i, j]:8.2g}")
            lines.append(f"  G0 {g0:8.2g}: " + "  ||  ".join(cells))
        lines.append("paths: worst |error| over " + ", ".join(OBS) + " by G0 class (>=100 / 1-100 / <1); the most at "
                     "the cap along the path")
        lines.append(f"{'':52s} " + " ".join(f"{s:>17s}" for s in show))
        worst = {(s, c[0]): (0.0, "") for s in show for c in P2.G0_CLASSES}
        for name, res, (xlo, xhi, cap) in r["paths"]:
            cells = ["/".join(f"{res[s][c[0]][0]:.2f}" if np.isfinite(res[s][c[0]][0]) else "  - "
                              for c in P2.G0_CLASSES) for s in show]
            lines.append(f"{name:52s} " + " ".join(f"{c:>17s}" for c in cells) + f"  x {xlo:.2f}-{xhi:.2f}  at cap {cap:.2f}")
            for s in show:
                for c in P2.G0_CLASSES:
                    v = res[s][c[0]]
                    if np.isfinite(v[0]) and v[0] > worst[(s, c[0])][0]:
                        worst[(s, c[0])] = (v[0], f"{v[1]}, {name}")
        lines.append("treatment map (worst over the paths):")
        for s in show:
            lines.append(f"   {s:10s} " + " | ".join(f"{c[0]}: {worst[(s, c[0])][0]:.3f} ({worst[(s, c[0])][1]})"
                                                 for c in P2.G0_CLASSES))
        if r.get("obs"):
            lines.append("observables, E against the truth: factor error max(E/T, T/E) - 1, worst cell / path integral "
                         f"(cells above {OBS_SIGNIFICANT:g} of the path maximum; carbon lost within 1 Gyr; r_ali = "
                         "uv_ali/uv_aro where it and the emission are significant; score_obs)")
            lines.append(f"{'':52s} " + " ".join(f"{k:>13s}" for k in OBS_SHOW))
            worst_o = {k: (0.0, "", 0.0, "") for k in OBS_SHOW}
            fmt = lambda v: f"{v:5.2f}" if np.isfinite(v) and v < 99.5 else ("  -  " if not np.isfinite(v) else " >99 ")  # noqa: E731
            for name, so in r["obs"]:
                lines.append(f"{name:52s} " + " ".join(f"{fmt(so[k][0])}/{fmt(so[k][1])}".rjust(13) for k in OBS_SHOW))
                for k in OBS_SHOW:
                    c, b = so[k]
                    w = list(worst_o[k])
                    if np.isfinite(c) and c > w[0]:
                        w[0], w[1] = c, name
                    if np.isfinite(b) and b > w[2]:
                        w[2], w[3] = b, name
                    worst_o[k] = tuple(w)
            lines.append("worst over the paths (cell; budget):")
            for k in OBS_SHOW:
                c, cn, b, bn = worst_o[k]
                lines.append(f"   {k:9s} cell {c:.3f} ({cn}); budget {b:.3f} ({bn})")
    text = "\n".join(lines)
    print(text)
    Path(out_txt).write_text(text + "\n")


def rescore(pkl, murga_root, out):
    """``assemble`` and the report of a stored run (``main``'s .pkl), e.g. after a change to the scoring."""
    with open(pkl, "rb") as fh:
        d = pickle.load(fh)
    t1, paths = conditions(murga_root)
    if [(n, len(c)) for n, c, _ in paths] != d["paths"]:
        raise ValueError(f"{pkl}: its paths differ from conditions()'s")
    nc = {m.name: m.nc for m in library()}
    for r in d["results"]:
        r.setdefault("nc", np.array([nc[n] for n in r["names"]]))          # runs before 2026-11-15 did not store it
    kept = {n for r in d["results"] for n in r["names"]}
    bins = [(edges, [n for n in names if n in kept]) for edges, names in d["bins"]]
    out = Path(out)
    report = assemble(d["results"], bins, t1, paths)
    print_report(report, out.with_suffix(".txt"))
    plot_t1(report, out.with_name(out.name + "_T1.png"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--per-bin", type=int, default=None, help="molecules per bin, spread in N_C (default: all)")
    ap.add_argument("--murga2020", default=str(Path.home() / "Library/Caches/calima/datasets/pahchem/murga2020"))
    ap.add_argument("--out", default="results/carbon_plan/check_pah_bins_hydrogen")
    ap.add_argument("--chain", choices=("joint", "averaged"), default="joint",
                    help="Population's joint (Z, N_H) chain, or its charge-averaged one (the 2026-11-08 run)")
    ap.add_argument("--superh-chemistry", default=None, choices=("andrews16", "deep"),
                    help="joint chain: Params.superh_chemistry (default: pahchem's)")
    ap.add_argument("--superh-levels", type=int, default=None, help="joint chain: Params.superh_levels (the cap)")
    ap.add_argument("--superh-magic", default=None, choices=("cazaux16", "boschman15"),
                    help='"deep": Params.superh_magic_barriers (default: pahchem\'s)')
    ap.add_argument("--rescore", default=None, metavar="PKL",
                    help="score the stored results of an earlier run (its .pkl) again, without solving; writes --out")
    args = ap.parse_args(argv)
    if args.rescore:
        return rescore(args.rescore, args.murga2020, args.out)
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
    superh = {k: v for k, v in (("superh_chemistry", args.superh_chemistry), ("superh_levels", args.superh_levels),
                                ("superh_magic_barriers", args.superh_magic)) if v is not None}
    print(f"{len(every)} molecules on {len(shards)} workers; {len(t1)} T1 cells, {len(paths)} paths; "
          f"{args.chain} chain {superh}", flush=True)
    from concurrent.futures import as_completed

    results = []
    with ProcessPoolExecutor(len(shards)) as ex:
        futures = [ex.submit(work, (s, t1, paths, cache, args.chain, superh)) for s in shards]
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            for name, why in r["dropped"]:
                print(f"  dropped {name}: {why}", flush=True)
            print(f"  shard {len(results)}/{len(shards)} done ({time.time() - t0:.0f} s)", flush=True)
            with open(out.with_suffix(".pkl"), "wb") as fh:      # every shard as it comes: a failure keeps the rest
                pickle.dump({"bins": bins, "results": results, "paths": [(n, len(c)) for n, c, _ in paths],
                             "chain": args.chain, "superh": superh}, fh)
    kept = {n for r in results for n in r["names"]}
    bins = [(edges, [n for n in names if n in kept]) for edges, names in bins]
    report = assemble(results, bins, t1, paths)
    print_report(report, out.with_suffix(".txt"))
    plot_t1(report, out.with_name(out.name + "_T1.png"))
    print(f"total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
