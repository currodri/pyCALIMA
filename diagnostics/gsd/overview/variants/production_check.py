"""Production path (pycalima.gsd with method=...) against the variants study (diagnostics/gsd/pl2m_variants.py).

Run from the worktree root after the A0 runs into results/gsd_overview/variants/a0_prod:
    PYTHONPATH=$PWD/src:$PWD/diagnostics/gsd python results/gsd_overview/variants/production_check.py
Writes results/gsd_overview/variants/production_check.txt.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

OUT = Path("results/gsd_overview/variants")
A0P = OUT / "a0_prod" / "gsd_bin_convergence"
REF = Path("results/binconv_v4/gsd_bin_convergence")
CASES = ("sn_diffuse", "mrn_accretion", "mrn_coagulation", "mrn_shattering", "mrn_sputtering")


def growth(lines):
    import check_mckinnon18_tests as MK
    import pl2m_variants as V
    from pycalima.gsd.remap import remap

    var = pickle.load(open(OUT / "growth.pkl", "rb"))
    a_cut = MK.A_MIN * (MK.A_MAX / MK.A_MIN) ** 0.25
    da = 0.005 * MK.UM / MK.GYR * 5.0 * MK.GYR
    exact = np.pi * MK.RHO / 3.0 * ((a_cut + da) ** 4 - (MK.A_MIN + da) ** 4)
    ns = (8, 16, 32, 64)
    lines.append("1. McKinnon+18 growth convergence (boxcar 1-5.6 nm, 25 nm in 100 steps): |M/M_exact - 1|, production "
                 "remap(method=...) / variants module")
    lines.append(f"   {'method':6s} " + " ".join(f"{n:>21d}" for n in ns))
    spec = {"pl2m": "pl2m", "ea": "ea:1e-6", "eaf": "eaf:1e-6:-3.5"}
    for name in ("pl2m", "ea", "eaf"):
        cells = []
        for n in ns:
            a, g = MK.grid_of(n)
            N, M = MK.boxcar_moments(a, MK.A_MIN, a_cut)
            for _ in range(100):
                N, M, *_ = remap(g, N, M, lambda x: x + da / 100, method=name)
            prod = abs(M.sum() / exact - 1.0)
            k = spec[name]
            if k in var["conv_bins"]:
                v = var["conv_bins"][k][list(var["conv_bins"]["N"]).index(n)]
            elif n in (8, 16, 32):
                v = var["ea_scan"][k][(8, 16, 32).index(n)]
            else:                                            # not in the study's tables: the variants module now
                cl = V.make_closure(k)
                N, M = MK.boxcar_moments(a, MK.A_MIN, a_cut)
                for _ in range(100):
                    N, M, *_ = V.remap(g, N, M, lambda x: x + da / 100, cl)
                v = abs(M.sum() / exact - 1.0)
            cells.append(f"{prod:9.3e} / {v:9.3e}")
        lines.append(f"   {name:6s} " + " ".join(f"{c:>21s}" for c in cells))


def a0_errors(path, refs):
    import check_bin_convergence as cbc
    meta, t, names, v = cbc._load(path)
    mr, tr, nr, vr = refs[meta["case"]]
    err = np.array([[cbc.errors(dict(zip(names, vi)), dict(zip(nr, vri)))[q] for q in nr]
                    for vi, vri in zip(v[1:], vr[1:])])
    w = err.max(axis=0)
    return float(w.max()), nr[int(w.argmax())]


def a0(lines):
    import check_bin_convergence as cbc
    refs = {}
    for case in CASES:
        f = REF / f"{case}_exact_ref_amin3.50A.npz"
        refs[case] = cbc._load(f if f.exists() else REF / f"{case}_pl2m_ref_amin3.50A.npz")
    var = pickle.load(open(OUT / "a0.pkl", "rb"))
    tag = {"pl2m": "stored", "eaf": "eaf6", "ea": "ea6"}
    lines.append("")
    lines.append("2. A0 worst observable (10/30/100 % snapshots, against the stored references), production "
                 "check_bin_convergence --schemes / variants module (pl2m: the stored binconv_v4 runs)")
    for case in CASES:
        lines.append(f"   {case}")
        for name in ("pl2m", "eaf", "ea"):
            cells = []
            for nb in (4, 6, 8, 12):
                f = A0P / f"{case}_{name}_N{nb:02d}_amin3.50A.npz"
                prod = a0_errors(f, refs)[0] if f.exists() else np.nan
                v = var.get((tag[name], case, nb), {}).get("err", np.nan)
                cells.append(f"{prod:.4f} / {v:.4f}")
            lines.append(f"     {name:5s} " + "  ".join(f"N={nb:2d} {c}" for nb, c in zip((4, 6, 8, 12), cells)))
    worst = {}
    for name in ("pl2m", "eaf", "ea"):
        for nb in (4, 6, 8, 12):
            vals = [a0_errors(A0P / f"{c}_{name}_N{nb:02d}_amin3.50A.npz", refs)[0] for c in CASES
                    if (A0P / f"{c}_{name}_N{nb:02d}_amin3.50A.npz").exists()]
            worst[(name, nb)] = max(vals) if len(vals) == len(CASES) else np.nan
    lines.append("   worst over the cases (production): " + "; ".join(
        f"{name} " + "/".join(f"{worst[(name, nb)]:.3f}" for nb in (4, 6, 8, 12)) for name in ("pl2m", "eaf", "ea"))
        + "  (N = 4/6/8/12)")
    # pl2m reruns against the stored files
    same, close = [], []
    for case in CASES:
        for nb in (4, 6, 8, 12):
            f, g = A0P / f"{case}_pl2m_N{nb:02d}_amin3.50A.npz", REF / f"{case}_pl2m_N{nb:02d}_amin3.50A.npz"
            if f.exists():
                x, y = np.load(f)["values"], np.load(g)["values"]
                if np.array_equal(x, y):
                    same.append(f"{case}/{nb}")
                else:
                    close.append(f"{case}/{nb}: {np.nanmax(np.abs(x - y) / np.maximum(np.abs(y), 1e-300)):.1e}")
    lines.append(f"   pl2m production reruns identical to the stored runs: {len(same)} of {len(same) + len(close)}; "
                 "the others differ by round-off (max rel. difference of any observable): " + ", ".join(close))
    lines.append("   (the HEAD code before the port gives the same round-off difference on sn_diffuse/8: the stored "
                 "binconv_v4 files predate the last collision commits; production pl2m == HEAD code bit for bit)")
    full = {}
    for case in CASES:
        for nb in (2, 3, 4, 6, 8, 12, 16, 24, 32):
            f = A0P / f"{case}_eaf_N{nb:02d}_amin3.50A.npz"
            if f.exists():
                full[(case, nb)] = a0_errors(f, refs)
    lines.append("")
    lines.append("3. A0 with the default (eaf) at the full bin list: worst observable [metric]; second row: the stored "
                 "PL2M runs (binconv_v4)")
    bins = (2, 3, 4, 6, 8, 12, 16, 24, 32)
    for case in CASES:
        lines.append(f"   {case:16s} eaf  " + "  ".join(
            f"{nb}: {full[(case, nb)][0]:.4f} [{full[(case, nb)][1].split(':')[-1]}]" for nb in bins
            if (case, nb) in full))
        lines.append(f"   {'':16s} pl2m " + "  ".join(
            f"{nb}: {a0_errors(REF / f'{case}_pl2m_N{nb:02d}_amin3.50A.npz', refs)[0]:.4f}" for nb in bins))
    worst_e = [max(full[(c, nb)][0] for c in CASES) for nb in bins]
    worst_p = [max(a0_errors(REF / f"{c}_pl2m_N{nb:02d}_amin3.50A.npz", refs)[0] for c in CASES) for nb in bins]
    lines.append("   worst over the cases, eaf / pl2m: " + "  ".join(
        f"{nb}: {e:.3f}/{q:.3f}" for nb, e, q in zip(bins, worst_e, worst_p)))


def rates(lines):
    import check_rates_convergence as crc
    from pycalima.gsd.closures import samples
    from pycalima.gsd.collisions import bin_nodes
    from pycalima.gsd.grid import MassGrid

    var = pickle.load(open(OUT / "rates.pkl", "rb"))
    cache = {}
    ref = {}
    for case in CASES:
        meta, pops = crc.load(REF / f"{case}_pl2m_ref_amin3.50A.npz")
        for p, (me, N, M) in pops.items():
            grid = MassGrid(me, crc.rho_of(me, meta["a_min_c"] if p == "graphite" else meta["a_min_si"]))
            key = (p, grid.rho)
            if key not in cache:
                cache[key] = crc.single_grain(crc.POPS[p], grid.a_edges[0], 5e-4)
            a, qs = cache[key]
            for cell, qd in qs.items():
                for q, gq in qd.items():
                    ref[(case, cell, q)] = ref.get((case, cell, q), 0.0) + crc.totals(grid, N, M, a, gq)["exact"]
    lines.append("")
    lines.append("4. Rates on the production eaf A0 end states (PE heating, recombination cooling, surface; worst over "
                 "the cells, rates and cases): |run/ref - 1| with the eaf exact in-bin integral / Gauss rate nodes "
                 "(the default rate_nodes); variants module: eaf exact / its median nodes, and EA + Gauss nodes")
    for nb in (4, 6, 8, 12):
        ex, tw = [], []
        for case in CASES:
            f = A0P / f"{case}_eaf_N{nb:02d}_amin3.50A.npz"
            if not f.exists():
                continue
            meta, pops = crc.load(f)
            acc = {}
            for p, (me, N, M) in pops.items():
                grid = MassGrid(me, crc.rho_of(me, meta["a_min_c"] if p == "graphite" else meta["a_min_si"]))
                a, qs = cache[(p, grid.rho)]
                a_s, dN = samples(grid, N, M, "eaf", step=0.005)
                m, w = bin_nodes(grid, N, M, method="eaf", rule="gauss")
                f_ = N > 0
                for cell, qd in qs.items():
                    for q, gq in qd.items():
                        at = lambda r: np.exp(np.interp(np.log(r), np.log(a), np.log(gq)))   # noqa: E731
                        x = acc.setdefault((cell, q), [0.0, 0.0])
                        x[0] += float(np.sum(dN * at(a_s)))
                        x[1] += float(np.sum(w[f_] * at(grid.radius(m[f_]))))
            for (cell, q), (e, t2) in acc.items():
                r = ref[(case, cell, q)]
                ex.append(abs(e / r - 1))
                tw.append(abs(t2 / r - 1))

        def vmax(tag, k):
            vals = [v[k] for kk, v in var.items() if kk[0] == tag and kk[2] == nb]
            return max(vals) if vals else np.nan
        lines.append(f"   N={nb:2d}: production eaf {max(ex):.4f} / gauss {max(tw):.4f};  variants eaf {vmax('eaf6', 0):.4f} "
                     f"/ median {vmax('eaf6', 1):.4f};  variants EA+Gauss {vmax('ea6_gauss', 0):.4f} / "
                     f"{vmax('ea6_gauss', 1):.4f}")


def main():
    lines = ["Production path (pycalima.gsd, method=...) versus the variants study (diagnostics/gsd/pl2m_variants.py). "
             "Differences between the two come from the ea/eaf base: production uses PL2M's BinProfile (spikes beyond "
             "PL2M's slope cap), the study a truncated power law beyond that cap; elsewhere the formulas are the same."]
    growth(lines)
    a0(lines)
    rates(lines)
    (OUT / "production_check.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
