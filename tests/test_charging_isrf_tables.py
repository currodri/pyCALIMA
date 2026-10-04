"""Invariants of the uniform-ISRF charging tables
(pycalima.models.dust_charge.charging_isrf_tables, read by RAMSES charging_model='WDB06isrf')."""
import numpy as np
import pytest

from pycalima.models.dust_charge import charging_isrf_tables as cit


def _synthetic(seed=0):
    """Random node values on a small grid; magnitudes kept away from zero so that relative
    tolerances hold at any interpolation weight."""
    rng = np.random.default_rng(seed)
    lT, lX = np.linspace(1.0, 3.0, 5), np.linspace(-2.0, 2.0, 5)
    shape = (len(cit.QUANTITIES), lT.size, lX.size)
    q = rng.uniform(0.5, 3.0, shape) * rng.choice([-1.0, 1.0], shape)
    q[1] = np.abs(q[1])                 # sigma_Z
    q[2] -= 60.0                        # ln Gamma_1 [erg/s]
    q[3] = np.abs(q[3]) * 1e-20         # lambda_1
    return dict(version=cit.FORMAT_VERSION, material="graphite", a=1e-6, lT=lT, lX=lX, q=q)


def test_file_round_trip(isolated_env):
    t = _synthetic()
    back = cit.read_isrf_table(cit.write_isrf_table(t, isolated_env / "isrf.dat", label="DustBin_01"))
    assert back["version"] == cit.FORMAT_VERSION
    assert back["a"] == pytest.approx(t["a"], rel=1e-12)
    np.testing.assert_array_equal(back["lT"], t["lT"])
    np.testing.assert_array_equal(back["lX"], t["lX"])
    np.testing.assert_allclose(back["q"], t["q"], rtol=1e-10, atol=0)


def test_lookup_at_a_node_returns_the_node():
    t = _synthetic()
    G0 = 4.0
    for i, lt in enumerate(t["lT"]):
        for j, lx in enumerate(t["lX"]):
            T, ne = 10.0 ** lt, G0 * 10.0 ** lx
            q = t["q"][:, i, j]
            lk = cit.isrf_lookup(t, G0, T, ne)
            assert lk["Zmean"] == pytest.approx(q[0], rel=1e-12)
            assert lk["Zsigma"] == pytest.approx(q[1], rel=1e-12)
            assert lk["Gamma"] == pytest.approx(G0 * np.exp(q[2]), rel=1e-12)
            assert lk["Lambda"] == pytest.approx(ne * np.sqrt(T) * q[3], rel=1e-12)
            np.testing.assert_allclose(lk["D"], np.exp(q[4:]), rtol=1e-12)


def test_solver_reader_agrees_with_the_exporter(isolated_env):
    # solvers/ reads the tables itself (it does not import models/): same values, same lookup
    from pycalima.solvers import table_io

    t = _synthetic(1)
    path = cit.write_isrf_table(t, isolated_env / "isrf.dat", label="DustBin_01")
    ref, sol = cit.read_isrf_table(path), table_io.read_isrf_charging_table(path)
    assert sol["a"] == ref["a"]
    np.testing.assert_array_equal(sol["q"], ref["q"])
    assert table_io.ISRF_D_CHARGES == cit.D_CHARGES
    for G0, T, ne in [(1.0, 10.0 ** 2.3, 10.0 ** 0.7), (40.0, 30.0, 1e-3), (1e-2, 1e5, 10.0)]:
        lk = cit.isrf_lookup(ref, G0, T, ne)
        Zmean, Zsigma, D = table_io.isrf_charging_lookup(sol, G0, T, ne)
        assert (Zmean, Zsigma) == pytest.approx((lk["Zmean"], lk["Zsigma"]), rel=1e-14)
        np.testing.assert_allclose(D, lk["D"], rtol=1e-14)


@pytest.mark.parametrize("G0", [1e-3, 0.37, 7.0, 2e4])
def test_g0_scaling_at_fixed_ne_over_g0(G0):
    # every rate is linear in G0 or in n_e: one table at G0 = 1 serves any G0 through x = n_e/G0
    t = _synthetic()
    T, x = 10.0 ** 2.3, 10.0 ** 0.7     # off the nodes
    ref = cit.isrf_lookup(t, 1.0, T, x)
    lk = cit.isrf_lookup(t, G0, T, G0 * x)
    assert lk["Zmean"] == pytest.approx(ref["Zmean"], rel=1e-12, abs=1e-12)
    assert lk["Zsigma"] == pytest.approx(ref["Zsigma"], rel=1e-12)
    np.testing.assert_allclose(lk["D"], ref["D"], rtol=1e-12)
    assert lk["Gamma"] / ref["Gamma"] == pytest.approx(G0, rel=1e-12)
    assert lk["Lambda"] / ref["Lambda"] == pytest.approx(G0, rel=1e-12)     # n_e = G0 x


def test_build_matches_its_nodes(isolated_env, model_data):
    """A tiny build from a real RT-group table: every node is the node solved on its own
    (cold), so neither the warm start along x nor the row assembly changes a value."""
    from pycalima.models.dust_charge import rt_group_charging as rg

    files = sorted((model_data / "dust_charging_rtgroups_data").glob("dust_charging_rtgroups_*.dat"))
    if not files:
        pytest.skip("no RT-group charging tables; run `calima-export --stages dust_charging_rtgroups`")
    tab = rg.read_bin_table(files[0])
    lT, lX = [2.0, 2.5], [-1.0, 0.0]
    t = cit.build_isrf_table(tab, lT=lT, lX=lX)
    assert t["q"].shape == (len(cit.QUANTITIES), len(lT), len(lX))
    gt = rg.GroupChargingTable(tab, kernels="pchip")
    for i, lt in enumerate(lT):
        for j, lx in enumerate(lX):
            node, _ = cit._node(gt, 10.0 ** lt, 10.0 ** lx, None)
            np.testing.assert_allclose(t["q"][:, i, j], node, rtol=1e-10, atol=0)
