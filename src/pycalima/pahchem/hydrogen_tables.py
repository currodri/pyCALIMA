"""Equilibrium tables of the PAH hydrogenation state, per molecular bin, from pahchem's joint (Z, N_H) chain.

RAMSES tabulates the hydrogenation of the molecular bins rather than tracking it (the user's decision of 2026-11-15,
after the literature review in results/carbon_plan/pah_hydrogenation_review/; docs/physics/pah-hydrogen.md). A bin is
a set of PAHdb molecules with weights (``group_tables.nc_range_weights``, or any other). At each node of the
pahchem-nd v1 axes (``export_pah_tables``: log10 G0 [Habing], log10 n_H [cm^-3], log10 T [K], log10 x_e = n_e / n_H),
every molecule's joint chain is solved for its steady state (``Params.hydrogen_chain = "joint"``), and the observables
of ``hydrogen_observables`` are averaged over the bin with the weights, per molecule:

- ``<label>_carbon.dat`` (pahchem-nd v1, ``export_pah_tables.write_nd_table``): log10 of the C atoms lost per molecule
  per second / 2, ``export_pah_tables``' unit (floor ``FLOOR``: no carbon loss). It is computed from the bin's molecules
  and their whole joint chain: the bare skeletons, the C2H2 of the emitting states, and the explosions. The
  ``export_pah_tables`` file comes from one parent molecule's BaselineModel network;
- ``<label>.npz``: every ``hydrogen_observables.OBSERVABLES`` per molecule on the same grid (rates per second, powers in
  erg s^-1, shares), the axes, the weights' mean N_C and N_H0, and the build settings.

The field is the Draine (1978) shape scaled by G0, plus ``optical_isrf`` times the solar circle's old-star light
(unscaled; 0 by default, as in ``check_pah_bins_hydrogen.py``). n(H) is the atomic H density. The tables hold
what the chain holds: super-hydrogenation to N_H0 + ``Params.superh_levels``, with ``Params.superh_chemistry``. The npz's
at_cap share flags the nodes where that cap, not the chemistry, sets the super-hydrogenated share.

``--hot-gas`` adds the thermal electrons' and ions' passages (``Params.hot_gas``: ``population.hot_passages``,
``hot_ion_passages``, the C and H knock-outs) for tables reaching above 1e4 K. The ions are inputs of each node
(labelled): H+ at n_e, and above ``HE_ION_T`` K helium as He+ at ``HE_ABUNDANCE`` n_H (pahchem has no He2+).

    python -m pycalima.pahchem.hydrogen_tables --bin 20 42 --workers 16 --out results/carbon_plan/hydrogen_tables
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from pycalima.pahchem import hydrogen_observables as ho

AXES = ("log_g0", "log_nh", "log_t", "log_xe")
FLOOR = -100.0
HE_ABUNDANCE = 0.1           # n(He) / n_H
HE_ION_T = 2e4               # K: above it the hot-gas tables count helium as He+ (labelled)


def default_axes():
    """{axis: values} of ``export_pah_tables.DEFAULT_AXES`` (the pahchem-nd v1 grid RAMSES reads)."""
    from pycalima.pahchem.export_pah_tables import DEFAULT_AXES

    return {k: np.round(np.arange(lo, hi + 0.5 * step, step), 10) for k, (lo, hi, step) in DEFAULT_AXES.items()}


def environment(lg0, lnh, lt, lxe, optical_isrf=0.0, hot_gas=False):
    from pycalima.pahchem.baseline import Environment

    n_h, n_e, t = 10.0 ** lnh, 10.0 ** (lnh + lxe), 10.0 ** lt
    ions = ()
    if hot_gas:
        ions = (("H+", n_e),) + ((("He+", HE_ABUNDANCE * n_h),) if t >= HE_ION_T else ())
    return Environment("table", 10.0 ** lg0, n_h, n_e, t, None, optical_isrf=optical_isrf, ions=ions)


def library(nc_min=21, nc_max=384):
    """PAHdb v4.00 molecules (``population.library_v400``, pure aromatic CH) with the calibrated thresholds and every
    charge -1 ... +2 (``group_tables.complete``): the molecules of the GSD bin checks."""
    from pycalima import _paths
    from pycalima.pahchem import group_tables as G
    from pycalima.pahchem import population

    lib = population.library_v400(min_nc=nc_min, max_nc=nc_max,
                                  cache_dir=_paths.get_model_data_dir() / "pahchem_pahdb_index")
    return [m for m in population.shift_thresholds(lib, population.CALIBRATED_SHIFTS) if G.complete(m)]


def molecule_grid(job):
    """[node..., mol, obs]: ``OBSERVABLES`` of each molecule of ``names`` at every node of ``axes`` (a worker)."""
    names, axes, params_kw, optical_isrf, cache_dir = job
    import os

    os.environ.setdefault("CALIMA_WORKERS", "1")
    from pycalima.pahchem import population

    lib = {m.name: m for m in library()}
    mols = [lib[n] for n in names]
    pop = population.Population(mols, {"all": (1.0, 1e4)}, cache_dir=cache_dir,
                                params=population.Params(hydrogen_chain="joint", **params_kw))
    shape = tuple(len(axes[a]) for a in AXES)
    out = np.zeros(shape + (len(mols), len(ho.OBSERVABLES)))
    prepared = set()
    for idx in np.ndindex(*shape):
        env = environment(*(axes[a][i] for a, i in zip(AXES, idx)), optical_isrf=optical_isrf,
                          hot_gas=params_kw.get("hot_gas", False))
        sh = population.field_shape(env)
        if sh not in prepared:
            pop.prepare(sh, workers=1)
            prepared.add(sh)
        o = ho.steady_observables(pop, env)
        out[idx] = np.stack([o[k] for k in ho.OBSERVABLES], axis=-1)
    return out


def bin_tables(per_mol, weights):
    """{name: grid}: the weighted mean over the molecules (axis -2 of ``per_mol``) of every observable."""
    w = np.asarray(weights, float)
    w = w / w.sum()
    mean = np.einsum("...mo,m->...o", per_mol, w)
    return {k: mean[..., i] for i, k in enumerate(ho.OBSERVABLES)}


def write_bin(out, label, axes, tables, mols, weights, settings):
    """``<label>_carbon.dat`` and ``<label>.npz`` (module docstring)."""
    from pycalima.pahchem.export_pah_tables import write_nd_table

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    w = np.asarray(weights, float) / np.sum(weights)
    nc, nh0 = float(w @ [m.nc for m in mols]), float(w @ [m.nh for m in mols])
    k = 0.5 * tables["carbon"]
    log_k = np.where(k > 10.0 ** FLOOR, np.log10(np.maximum(k, 1e-300)), FLOOR)
    header = [f"# pahchem joint-chain carbon-loss rate per molecule, bin {label}: {len(mols)} PAHdb v4.00 molecules, "
              f"N_C {min(m.nc for m in mols)}-{max(m.nc for m in mols)} (weighted mean {nc:.1f}, N_H0 {nh0:.1f})",
              "# Source: pycalima.pahchem.hydrogen_tables (Population, Params.hydrogen_chain = 'joint'; "
              + ", ".join(f"{a} = {b}" for a, b in settings.items()) + ")",
              "# Values: log10(k [s^-1]), carbon atoms lost per molecule per second / 2, averaged over the bin; "
              f"floor {FLOOR:g} = no carbon loss"]
    write_nd_table(out / f"{label}_carbon.dat", axes, log_k, header)
    np.savez_compressed(out / f"{label}.npz", **{a: axes[a] for a in AXES}, **tables,
                        names=np.array([m.name for m in mols]), weights=w, nc_mean=nc, nh0_mean=nh0,
                        settings=np.array(repr(settings)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bin", type=float, nargs=2, action="append", required=True, metavar=("NC_MIN", "NC_MAX"),
                    help="a bin's N_C range [min, max), repeatable")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--per-bin", type=int, default=None, help="molecules per bin, spread in N_C (a test)")
    ap.add_argument("--optical-isrf", type=float, default=0.0)
    ap.add_argument("--superh-levels", type=int, default=None, help="Params.superh_levels (default: Params')")
    ap.add_argument("--superh-chemistry", default=None, help="Params.superh_chemistry (default: Params')")
    ap.add_argument("--axes", default=None, help="'lo:hi:step' for each of log_g0, log_nh, log_t, log_xe, comma-separated")
    ap.add_argument("--hot-gas", action="store_true", help="thermal electron and ion passages (Params.hot_gas)")
    ap.add_argument("--out", default="results/carbon_plan/hydrogen_tables")
    args = ap.parse_args(argv)
    from concurrent.futures import ProcessPoolExecutor

    from pycalima import _paths
    from pycalima.pahchem import group_tables as G

    axes = default_axes()
    if args.axes:
        for a, spec in zip(AXES, args.axes.split(",")):
            lo, hi, step = (float(v) for v in spec.split(":"))
            axes[a] = np.round(np.arange(lo, hi + 0.5 * step, step), 10)
    params_kw = {k: v for k, v in (("superh_levels", args.superh_levels), ("superh_chemistry", args.superh_chemistry),
                                   ("hot_gas", True if args.hot_gas else None)) if v is not None}
    settings = {"optical_isrf": args.optical_isrf, "field": "Draine 1978", **params_kw}
    lib = library()
    bins = []
    for lo, hi in args.bin:
        sel = sorted((m for m in lib if lo <= m.nc < hi), key=lambda m: (m.nc, m.name))
        if args.per_bin and len(sel) > args.per_bin:
            sel = [sel[int(i)] for i in np.linspace(0, len(sel) - 1, args.per_bin)]
        bins.append(((lo, hi), sel))
    every = sorted({m.name: m for _, sel in bins for m in sel}.values(), key=lambda m: -m.nc)
    shards = [s for s in (every[k::args.workers] for k in range(args.workers)) if s]
    n_nodes = int(np.prod([len(axes[a]) for a in AXES]))
    print(f"{len(every)} molecules on {len(shards)} workers, {n_nodes} nodes", flush=True)
    t0 = time.time()
    cache = _paths.get_model_data_dir() / "pahchem_population_basis"
    per = {}
    with ProcessPoolExecutor(len(shards)) as ex:
        for shard, res in zip(shards, ex.map(molecule_grid, [([m.name for m in s], axes, params_kw, args.optical_isrf,
                                                              cache) for s in shards])):
            for k, m in enumerate(shard):
                per[m.name] = res[..., k, :]
    print(f"solved in {time.time() - t0:.0f} s", flush=True)
    for (lo, hi), sel in bins:
        w = G.nc_range_weights(sel, lo, hi)
        tables = bin_tables(np.stack([per[m.name] for m in sel], axis=-2), w)
        label = f"bin_nc{lo:.0f}-{hi:.0f}"
        write_bin(args.out, label, axes, tables, sel, w, settings)
        print(f"wrote {args.out}/{label}_carbon.dat, .npz ({len(sel)} molecules)")


if __name__ == "__main__":
    main()
