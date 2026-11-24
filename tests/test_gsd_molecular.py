"""The molecular bins' tables (``pycalima.gsd.molecular``) and their carbon loss in the operator."""
from __future__ import annotations

import numpy as np
import pytest
from test_gsd_cell import YR, carbon, inputs

from pycalima.gsd import cell
from pycalima.gsd.charging import PhotonGroups
from pycalima.gsd.molecular import EV, U_HABING, MolecularTables, habing_g0
from pycalima.pahchem.export_pah_tables import write_nd_table

pytestmark = pytest.mark.gsd_quick

M_C = 12.011 * 1.66053906660e-24
AXES = {"log_g0": np.array([-2.0, 0.0, 2.0, 4.0]), "log_nh": np.array([-1.0, 2.0, 5.0]),
        "log_t": np.array([1.0, 2.0, 4.0]), "log_xe": np.array([-6.0, -3.0, 0.0])}


@pytest.fixture(autouse=True)
def _clean_env(isolated_env):
    yield


def table(path, value):
    """A pahchem-nd table of log10(rate / 2); ``value`` a constant or a function of the axes."""
    g = np.meshgrid(*(AXES[k] for k in ("log_g0", "log_nh", "log_t", "log_xe")), indexing="ij")
    v = value(*g) if callable(value) else np.full(g[0].shape, float(value))
    write_nd_table(path, AXES, v, ["# test table"])
    return path


def tables(tmp_path, values, tag="cold"):
    d = tmp_path / tag
    d.mkdir(exist_ok=True)
    edges = (20, 42, 89, 189, 400)
    return [table(d / f"bin_nc{lo}-{hi}_carbon.dat", v) for lo, hi, v in zip(edges[:-1], edges[1:], values, strict=True)]


def test_habing_g0_counts_the_6_to_13_6_ev_energy_density():
    u = U_HABING / EV                                          # eV cm^-3 at G0 = 1
    ph = PhotonGroups(L0=np.array([1.0, 6.0, 13.6]), L1=np.array([6.0, 13.6, 24.6]),
                      N=np.array([1.0, u / 10.0, 1.0]), egy=np.array([3.0, 10.0, 18.0]))
    assert habing_g0(ph) == pytest.approx(1.0, rel=1e-12)
    half = PhotonGroups(L0=np.array([2.0]), L1=np.array([10.0]), N=np.array([u / 6.0]), egy=np.array([6.0]))
    assert habing_g0(half) == pytest.approx(0.5, rel=1e-12)                 # half the group's width is inside
    assert habing_g0(None) == 0.0


def test_tables_read_back_in_bin_order_with_the_pahchem_unit(tmp_path):
    tables(tmp_path, [-10.0, -11.0, -12.0, lambda lg, ln, lt, lx: lg - 13.0])
    mt = MolecularTables.from_directories(tmp_path / "cold")
    assert mt.n_bins == 4
    k = mt.carbon_loss(10.0, 100.0, 50.0, 0.1)
    np.testing.assert_allclose(k[:3], 2.0 * np.array([1e-10, 1e-11, 1e-12]), rtol=1e-12)
    assert k[3] == pytest.approx(2.0 * 10.0 ** (1.0 - 13.0), rel=1e-10)        # linear in log G0
    assert mt.carbon_loss(1e8, 100.0, 50.0, 0.1)[3] == pytest.approx(2e-9, rel=1e-10)   # clamped at the edge


def test_the_hot_tables_take_over_above_t_join(tmp_path):
    cold = tables(tmp_path, [-10.0] * 4)
    hot = tables(tmp_path, [-8.0] * 4, tag="hot")
    mt = MolecularTables(cold, hot, t_join=1e4)
    np.testing.assert_allclose(mt.carbon_loss(1.0, 1.0, 1e4, 1.0), 2e-10, rtol=1e-12)
    np.testing.assert_allclose(mt.carbon_loss(1.0, 1.0, 1.01e4, 1.0), 2e-8, rtol=1e-12)
    with pytest.raises(ValueError):
        MolecularTables(cold, hot[:3])


def run(tmp_path, values, procs=frozenset({"carbon_loss"}), dt=10 * YR, T=100.0):
    c = carbon(n_H=1.0)
    c.molecular_bins = 4
    c.molecular_tables = MolecularTables(tables(tmp_path, values))
    N0, M0 = c.N.copy(), c.M.copy()
    res = cell.update([c], inputs(n_H=1.0, T=T, gas_C=1e-4), None, dt, cell.Options(processes=procs))
    return c, N0, M0, res


def test_carbon_loss_erodes_the_molecular_bins_and_conserves_carbon(tmp_path):
    """Bins 1-3 lose k m_C per molecule per second (bin 0 holds still, so nothing leaves the grid): for k dt much
    less than a bin's carbon, the dust loses sum N_i k m_C dt, the gas gains it, and the grain bins do not move."""
    k = 2e-9
    c, N0, M0, res = run(tmp_path, [-40.0, np.log10(k / 2)] + [np.log10(k / 2)] * 2)
    dt = 10 * YR
    lost = M0.sum() - c.M.sum()
    assert lost == pytest.approx(np.sum(N0[1:4]) * k * M_C * dt, rel=1e-3)
    assert abs(res.conservation["C"]) < 1e-12
    np.testing.assert_allclose(c.M[4:], M0[4:], rtol=1e-13)          # a still remap: round-off only
    np.testing.assert_allclose(c.N[4:], N0[4:], rtol=1e-13)
    assert c.N.sum() == pytest.approx(N0.sum(), rel=1e-12)                     # nothing left the grid


def test_the_bottom_bin_loses_its_eroded_molecules_to_the_gas(tmp_path):
    c, N0, _, res = run(tmp_path, [np.log10(1e-6)] * 4, dt=1e3 * YR)
    assert c.N.sum() < N0.sum()
    assert abs(res.conservation["C"]) < 1e-12
    assert res.gas["C"] > 1e-4


def test_no_carbon_loss_without_the_process_or_the_tables(tmp_path):
    """Tables present but 'carbon_loss' not chosen: the same as without tables, bit for bit; chosen without
    tables: nothing happens."""
    c, _, _, _ = run(tmp_path, [-8.0] * 4, procs=frozenset({"sputtering"}), T=1e6)
    ref = carbon(n_H=1.0)
    ref.molecular_bins = 4
    cell.update([ref], inputs(n_H=1.0, T=1e6, gas_C=1e-4), None, 10 * YR, cell.Options(processes=frozenset({"sputtering"})))
    np.testing.assert_array_equal(c.M, ref.M)
    np.testing.assert_array_equal(c.N, ref.N)
    c = carbon(n_H=1.0)
    c.molecular_bins = 4
    M0 = c.M.copy()
    cell.update([c], inputs(n_H=1.0, gas_C=1e-4), None, 10 * YR, cell.Options(processes=frozenset({"carbon_loss"})))
    np.testing.assert_array_equal(c.M, M0)


def test_sputtering_and_carbon_loss_share_one_erosion_step(tmp_path):
    """In hot gas both act in one remap: the grains erode down to N_C,max, the molecules by their tables."""
    both = run(tmp_path, [-40.0, -10.0, -10.0, -10.0], procs=frozenset({"sputtering", "carbon_loss"}), T=1e6)
    gr = run(tmp_path, [-40.0, -10.0, -10.0, -10.0], procs=frozenset({"sputtering"}), T=1e6)
    assert abs(both[3].conservation["C"]) < 1e-12
    np.testing.assert_allclose(both[0].M[5:], gr[0].M[5:], rtol=1e-12)     # the grain bins above don't see the tables
    assert both[0].M[1:4].sum() < gr[0].M[1:4].sum()
