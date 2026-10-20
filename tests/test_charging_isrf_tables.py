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
            nD = len(cit.D_CHARGES)
            np.testing.assert_allclose(lk["D"], np.exp(q[4:4 + nD]), rtol=1e-12)
            np.testing.assert_allclose(lk["alpha"], np.exp(q[4 + nD:]), rtol=1e-12)


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
    np.testing.assert_allclose(lk["alpha"], ref["alpha"], rtol=1e-12)     # per grain: a function of P(Z) and T
    assert lk["Gamma"] / ref["Gamma"] == pytest.approx(G0, rel=1e-12)
    assert lk["Lambda"] / ref["Lambda"] == pytest.approx(G0, rel=1e-12)     # n_e = G0 x


def test_build_matches_its_nodes(isolated_env, model_data):
    """A tiny build from a real RT-group table: every node is the node solved on its own
    (cold), so neither the warm start along x nor the row assembly changes a value. And the
    recombination planes order as the case-A thresholds do: an ion of higher ionisation
    potential can take an electron from more grain charges, so alpha_i sqrt(m_i) (the same
    capture kernel summed over more charges) cannot decrease with it."""
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
    k0 = cit.QUANTITIES.index(f"lnalpha_{rg.RECOMB_IONS[0][0]}")
    order = np.argsort([ip for _, _, ip in rg.RECOMB_IONS])
    lnm = 0.5 * np.log([m for _, m, _ in rg.RECOMB_IONS])
    for i in range(len(lT)):
        for j in range(len(lX)):
            scaled = (t["q"][k0:k0 + len(rg.RECOMB_IONS), i, j] + lnm)[order]
            assert np.all(np.diff(scaled) >= -1e-12), (lT[i], lX[j], scaled)


# --------------------------------------------------------------------------- memory bounds
# A grain of radius a has ~1e10 a [cm] allowed charges, so nothing may build an array over all of
# them (a 1 mm grain exhausted the memory of a laptop: one array per worker of ~4e8 charges).

@pytest.fixture(scope="module")
def bins(tmp_path_factory):
    from pycalima.models.dust_charge import rt_group_charging as rg

    out = {}
    for a in (1e-5, 0.1):                   # 0.1 um (the shipped DustBin_02) and 1 mm graphite
        tab = rg.build_bin_table("graphite", a, U=rg.u_grid(n=241), Ts=np.geomspace(3e3, 3e6, 13),
                                 T_hot=np.geomspace(1e3, 1e9, 13), label=f"g{a:g}")
        out[a] = rg.read_bin_table(rg.write_bin_table(tab, tmp_path_factory.mktemp("isrf") / f"g{a:g}.dat"))
    return out


@pytest.mark.parametrize("material", ["graphite", "silicate"])
def test_charge_edges_by_bisection_match_every_charge(material):
    from pycalima.models.dust_charge import rt_group_charging as rg
    from pycalima.models.dust_charge import wdb06_yields as wy

    W = wy.MATERIALS[material]["W"]
    for a in (5e-8, 1e-6, 1e-5, 1e-4):
        Zmin = float(rg.most_negative_allowed_charge_graphite(a) if material == "graphite"
                     else rg.most_negative_allowed_charge_silicate(a))
        Zmax = float(rg.most_positive_allowed_charge(a, W, 13.6))
        Z = np.arange(Zmin, Zmax + 1.0)
        ipz = rg.grain_ionisation_potential(material, a, W, Z)
        assert np.all(np.diff(ipz) > 0)                     # what the bisection relies on
        for E in [0.1, 5.6, 11.2, 13.6, 24.6, 54.4] + [ip for _, _, ip in rg.RECOMB_IONS]:
            k = np.nonzero(ipz <= E)[0]
            assert rg.last_charge_with_ip_below(material, a, W, Zmin, Zmax, E) == (float(Z[k[-1]]) if k.size else None)


def _nodes(gt, lts=(1.0, 3.0, 5.0, 9.0), lxs=(-8.0, -2.0, 2.0, 8.0)):
    return np.array([cit._node(gt, 10.0 ** lt, 10.0 ** lx, None)[0] for lt in lts for lx in lxs])


def test_piecewise_distribution_reproduces_the_contiguous_one(bins, monkeypatch):
    """The path of wide or far-from-neutral grains (from the window out to the floor, CHUNK at a
    time) gives the contiguous result where both apply."""
    from pycalima.models.dust_charge import rt_group_charging as rg

    gt = rg.GroupChargingTable(bins[1e-5], kernels="pchip")
    ref = _nodes(gt)
    monkeypatch.setattr(cit, "MAX_CONTIGUOUS", 0)
    monkeypatch.setattr(cit, "CHUNK", 997)                  # many chunks, not a multiple of anything
    # ln P differs by rounding (~1e-13) where the cumulative sum starts; sigma_Z^2 = <Z^2> - <Z>^2 of a
    # narrow P(Z) far from neutral cancels ~10 digits of it (sigma 0.17 at <Z> = 625: 4e-6)
    np.testing.assert_allclose(_nodes(gt), ref, rtol=1e-5, atol=1e-12)


def test_a_millimetre_grain_is_tabulated_on_a_bounded_number_of_charges(bins, monkeypatch):
    from pycalima.models.dust_charge import rt_group_charging as rg

    gt = rg.GroupChargingTable(bins[0.1], kernels="pchip")
    assert gt.t["Zmax"] - gt.t["Zmin"] > 1e8
    kept, strides = [], []
    orig = cit.full_distribution

    def spy(*args):
        Z, P, s = orig(*args)
        kept.append(Z.size)
        strides.append(s)
        return Z, P, s

    monkeypatch.setattr(cit, "full_distribution", spy)
    lts, lxs = (1.0, 4.0, 6.0, 9.0), (-8.0, 0.0, 4.0, 8.0)
    v = _nodes(gt, lts, lxs)
    assert np.all(np.isfinite(v))
    assert max(kept) <= max(cit.MAX_CONTIGUOUS, 3 * cit.MAX_CHARGES)
    assert max(strides) > 1                                 # hot gas: P(Z) wider than MAX_CHARGES
    # the stride does not show in the sums over a P(Z) that wide
    monkeypatch.setattr(cit, "MAX_CHARGES", 8 * cit.MAX_CHARGES)
    np.testing.assert_allclose(v, _nodes(gt, lts, lxs), rtol=1e-4, atol=1e-12)
