"""pahchem.hydrogen_tables: the bin mixture and the pahchem-nd v1 round trip."""
from types import SimpleNamespace

import numpy as np
import pytest

from pycalima.pahchem import hydrogen_observables as ho
from pycalima.pahchem import hydrogen_tables as htab


def test_bin_tables_is_the_weighted_mean():
    rng = np.random.default_rng(0)
    per_mol = rng.random((3, 2, 4, len(ho.OBSERVABLES)))          # [g0, nh, mol, obs]
    w = np.array([1.0, 0.0, 2.0, 1.0])
    t = htab.bin_tables(per_mol, w)
    for i, k in enumerate(ho.OBSERVABLES):
        assert np.allclose(t[k], (per_mol[..., i] * w).sum(axis=-1) / w.sum())


def test_written_carbon_table_reads_back(isolated_env):
    from pycalima.solvers.table_io import read_pah_nd_table

    axes = {"log_g0": np.array([0.0, 1.0]), "log_nh": np.array([1.0, 2.0, 3.0]), "log_t": np.array([2.0]),
            "log_xe": np.array([-4.0, -3.0])}
    shape = tuple(len(axes[a]) for a in htab.AXES)
    carbon = np.geomspace(1e-14, 1e-8, int(np.prod(shape))).reshape(shape)
    carbon[0, 0, 0, 0] = 0.0                                       # no carbon loss: the floor
    tables = {k: np.zeros(shape) for k in ho.OBSERVABLES} | {"carbon": carbon}
    mols = [SimpleNamespace(name="a", nc=24, nh=12), SimpleNamespace(name="b", nc=32, nh=14)]
    htab.write_bin(isolated_env, "bin_test", axes, tables, mols, [1.0, 1.0], {"field": "Draine 1978"})
    ax, values = read_pah_nd_table(isolated_env / "bin_test_carbon.dat")
    for a, v in zip(htab.AXES, ax):
        assert np.allclose(v, axes[a])
    expect = np.where(carbon > 0, np.log10(np.maximum(0.5 * carbon, 1e-300)), htab.FLOOR)
    assert np.allclose(values, expect, atol=1e-6)
    d = np.load(isolated_env / "bin_test.npz")
    assert d["nc_mean"] == pytest.approx(28.0) and np.allclose(d["carbon"], carbon)
