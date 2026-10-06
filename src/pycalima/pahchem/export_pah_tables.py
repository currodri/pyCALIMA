#!/usr/bin/env python
"""Export pahchem PAH carbon-loss (photolysis) tables for RAMSES-CALIMA.

For each PAH bin whose N_C maps onto a parent with PAHdb modes (C24H12,
C54H18, C96H24 in ``model_data/PAH_states``), this writes
``model_data/<model>/PAH_dissociation_data/pahchem_dissociation_<bin>.dat``,
the rate of C2-unit loss per molecule, log10 k [s^-1], on four axes:
log10 G0 (Habing; Draine 1978 shape), log10 nH [cm^-3], log10 T [K] and
log10 x_e = n_e/nH.

The physics is ``pahchem.baseline``:

    k = sum_Z f(Z, N_H = 0) k_C2(Z),

- f comes from the (Z, N_H) network at steady state, with the H-bearing
  states' physical (sink) photo-rates;
- k_C2 is RRKM C2 loss from the bare skeleton, which compact PAHs reach
  before losing carbon (Zhen+14; Montillaud+13), using the bare skeleton's
  own PAHdb modes where available;
- the atomic H density is taken as nH.

Carbon loss does not feed back on f: it is slow compared with the
hydrogenation exchange.

A companion ``pahchem_state_<bin>.npz`` holds, on the same grid:
- the cation and anion fractions;
- the fraction dehydrogenated (N_H < N_H0) and the fully dehydrogenated
  fraction;
- the H2 formation rate per PAH.

File format (``read_pah_nd_table`` in ``solvers.table_io``):

    lines starting with '#'   header, including "# Format: pahchem-nd v1"
    ndim                      (4)
    n_1 ... n_ndim            axis lengths, in the order G0, nH, T, x_e
    axis values               n_1 + ... + n_ndim lines, one value each
    table values              prod(n_i) lines, C order (x_e fastest)

Bins without a parent (N_C outside 16-150, e.g. cluster bins) are skipped.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from pycalima import _paths
from pycalima.models.grain_size_config import (get_bins, get_export_parameters, get_header_lines,
                                               get_lognormal_parameters, get_model_data_dir,
                                               set_config_path)

DEFAULT_AXES = {"log_g0": (-1.0, 6.0, 0.5), "log_nh": (-1.0, 6.0, 0.5),
                "log_t": (1.0, 4.0, 0.5), "log_xe": (-5.0, -1.0, 1.0)}
PARENT_FILES = {24: ("C24", "C24H12_0.dat", "C24_0.dat"),
                54: ("C54", "C54H18_0.dat", "C54_0.dat"),
                96: ("C96", "C96H24_0.dat", "C96_0.dat")}
FLOOR = -100.0


def axes_from_config():
    cfg = get_export_parameters("pahchem_dissociation", defaults={})
    out = {}
    for name, (lo, hi, step) in DEFAULT_AXES.items():
        lo, hi, step = cfg.get(name, (lo, hi, step))
        out[name] = np.round(np.arange(lo, hi + 0.5 * step, step), 10)
    return out


def parent_for(nc):
    """The parent key closest to N_C, or None if N_C is outside 16-150."""
    if not 16 <= nc <= 150:
        return None
    return min(PARENT_FILES, key=lambda p: abs(np.log(p / nc)))


def compute_bin_table(model, bare, axes, progress=print):
    """log10 k_C2 and the companion fractions on the four axes."""
    from pycalima.pahchem.baseline import Environment

    shape = tuple(len(axes[k]) for k in ("log_g0", "log_nh", "log_t", "log_xe"))
    log_k = np.full(shape, FLOOR)
    aux = {k: np.zeros(shape) for k in ("f_cation", "f_anion", "f_dehydrogenated",
                                        "f_bare", "h2_per_pah_per_s")}
    nh_vals = np.array(model.nh_values)
    for i, lg in enumerate(axes["log_g0"]):
        t0 = time.time()
        env0 = Environment("table", 10**lg, 1.0, 1.0, 100.0, teff_k=None)
        pr = model.photo_rates(env0)
        k_c2 = model.carbon_loss_rates(env0, bare)
        for j, ln in enumerate(axes["log_nh"]):
            for t, lt in enumerate(axes["log_t"]):
                for x, lx in enumerate(axes["log_xe"]):
                    env = Environment("table", 10**lg, 10**ln, 10**(ln + lx), 10**lt, teff_k=None)
                    net = model.network(pr, env)
                    f = net.steady_state()
                    bare_z = f[:, 0]
                    k = float(sum(bare_z[iz] * k_c2[z] for iz, z in enumerate(model.charges)))
                    idx = (i, j, t, x)
                    log_k[idx] = np.log10(k) if k > 10**FLOOR else FLOOR
                    fz = f.sum(axis=1)
                    aux["f_cation"][idx] = fz[np.array(model.charges) > 0].sum()
                    aux["f_anion"][idx] = fz[np.array(model.charges) < 0].sum()
                    aux["f_dehydrogenated"][idx] = f[:, nh_vals < model.nh0].sum()
                    aux["f_bare"][idx] = bare_z.sum()
                    aux["h2_per_pah_per_s"][idx] = net.h2_formation_rate(f)
        progress(f"  log G0 = {lg:+.2f}: {time.time() - t0:.0f} s")
    return log_k, aux


def write_nd_table(path, axes, values, header_lines):
    names = ("log_g0", "log_nh", "log_t", "log_xe")
    with open(path, "w", encoding="utf-8") as fh:
        for line in header_lines:
            fh.write(line + "\n")
        fh.write("# Format: pahchem-nd v1; axes log10 G0 [Habing], log10 nH [cm^-3], "
                 "log10 T [K], log10 x_e; values C order (last axis fastest)\n")
        fh.write(f"{len(names)}\n")
        fh.write(" ".join(str(len(axes[n])) for n in names) + "\n")
        for n in names:
            for v in axes[n]:
                fh.write(f"{v:.6f}\n")
        for v in np.asarray(values).ravel(order="C"):
            fh.write(f"{v:.6f}\n")


def _bin_table(task):
    """compute_bin_table for one bin, from picklable inputs (a process-pool task)."""
    from pycalima.pahchem.baseline import BaselineModel, Params
    from pycalima.pahchem.molecule import PAHSpecies

    key, parent_file, bare_file, params, axes = task
    species = PAHSpecies.from_states_file(parent_file)
    bare = PAHSpecies.from_states_file(bare_file) if bare_file else None
    model = BaselineModel(key, species, params or Params())
    return compute_bin_table(model, bare, axes, progress=lambda message: None)


def main(config_path=None, output_dir=None, axes=None, states_dir=None, params=None, workers=None):
    """Write the tables of every free-PAH bin; bins run in parallel.

    ``workers``: processes for the bins (default: one per bin, up to the CPU
    count; 1 runs them in this process).
    """
    import os
    from concurrent.futures import ProcessPoolExecutor

    from pycalima.pahchem.molecule import PAHSpecies

    if config_path:
        set_config_path(config_path)
    axes = axes_from_config() if axes is None else axes
    out_dir = Path(output_dir) if output_dir else get_model_data_dir(create=True) / "PAH_dissociation_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    states = Path(states_dir) if states_dir else _paths.get_model_data_dir() / "PAH_states"
    jobs = []
    for b in get_bins(is_pah=True):
        a0_um = get_lognormal_parameters(b["id"], model_name="basic")["a0"]
        nc = int(418 * (a0_um * 1e4 / 10.0) ** 3)              # Draine+21 eq. 8
        parent = parent_for(nc)
        if parent is None:
            print(f"[{b['id']}] N_C = {nc}: no PAHdb parent; skipped (cluster bins: see pahchem-spec.md)")
            continue
        key, fparent, fbare = PARENT_FILES[parent]
        if not (states / fparent).exists():
            print(f"[{b['id']}] missing {states / fparent}; skipped")
            continue
        species = PAHSpecies.from_states_file(states / fparent)
        bare = PAHSpecies.from_states_file(states / fbare) if (states / fbare).exists() else None
        print(f"[{b['id']}] N_C = {nc} -> {species.formula} (bare: {bare.formula if bare else 'stripped'})")
        task = (key, states / fparent, states / fbare if bare else None, params, axes)
        jobs.append((b, nc, species, bare, task))

    n_workers = min(len(jobs), os.cpu_count() or 1) if workers is None else max(1, int(workers))
    if n_workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            tables = list(pool.map(_bin_table, [job[-1] for job in jobs]))
    else:
        tables = [_bin_table(job[-1]) for job in jobs]

    written = []
    for (b, nc, species, bare, _), (log_k, aux) in zip(jobs, tables):
        header = get_header_lines(
            "pahchem C2-loss (photolysis) rate per molecule", "pahchem/export_pah_tables.py",
            bin_info=f"PAH Bin: {b['id']} (N_C = {nc}; molecule {species.formula}; "
                     f"bare skeleton {bare.formula if bare else 'stripped modes'})",
            val_desc="Values: log10(k_C2 [s^-1]); floor -100 = no carbon loss")
        header += ["# Physics: pahchem.baseline defaults (method_cards/pahchem_baseline.toml); "
                   "Draine 1978 field shape; n(H) = nH"]
        path = out_dir / f"pahchem_dissociation_{b['id']}.dat"
        write_nd_table(path, axes, log_k, header)
        np.savez(out_dir / f"pahchem_state_{b['id']}.npz", **axes, **aux)
        written.append(path)
        print(f"  wrote {path}")
    return {"output_dir": str(out_dir), "file_count": len(written), "tables_generated": len(written)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--workers", type=int, default=None, help="processes for the bins (default: one per bin)")
    args = ap.parse_args()
    main(config_path=args.config, output_dir=args.output_dir, workers=args.workers)
