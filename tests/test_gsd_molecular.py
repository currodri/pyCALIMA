"""The molecular bins' tables (``pycalima.gsd.molecular``) and their carbon loss in the operator."""
from __future__ import annotations

import os

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


_DUSTEM_DIR = os.environ.get("CALIMA_DUSTEM_DIR")


@pytest.fixture(autouse=True)
def _clean_env(isolated_env):
    yield


@pytest.fixture
def dustem(monkeypatch):
    """DustEM's THEMIS data (the release term's IR cooling) inside isolated_env: the ambient $CALIMA_DUSTEM_DIR,
    or skip."""
    from pycalima._datasets import get_dataset

    if _DUSTEM_DIR:
        monkeypatch.setenv("CALIMA_DUSTEM_DIR", _DUSTEM_DIR)
    if get_dataset("dustem-themis-cm20").locate() is None:
        pytest.skip("DustEM THEMIS data not available; set $CALIMA_DUSTEM_DIR")


def table(path, value, h2=1e-12):
    """A pahchem-nd table of log10(rate / 2) (``value`` a constant or a function of the axes) and its npz of
    observables, with 'h2' (a constant or a function of the axes, per molecule per second)."""
    g = np.meshgrid(*(AXES[k] for k in ("log_g0", "log_nh", "log_t", "log_xe")), indexing="ij")
    v = value(*g) if callable(value) else np.full(g[0].shape, float(value))
    write_nd_table(path, AXES, v, ["# test table"])
    hv = h2(*g) if callable(h2) else np.full(g[0].shape, float(h2))
    np.savez(str(path).replace("_carbon.dat", ".npz"), h2=hv, **AXES)
    return path


def tables(tmp_path, values, tag="cold"):
    d = tmp_path / tag
    d.mkdir(parents=True, exist_ok=True)
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


def fake_group_table(sigma=1e-16, k_up=1e-17, alpha=1e-7, a_bg=1e-9):
    """A ``pahchem.group_tables`` bin table with flat kernels: absorption sigma, photo-up k_up (out of -1, 0, +1;
    none out of +2), 2 eV per photoelectron, capture alpha for 0..+2 and a_bg background absorptions per s."""
    from pycalima.gsd.charging import AGN_GROUP_L0, AGN_GROUP_L1
    from pycalima.pahchem import group_tables as gt

    L0, L1 = np.array(AGN_GROUP_L0), np.array(AGN_GROUP_L1)
    nodes = gt._group_nodes(L0, L1, gt.N_E_PER_GROUP)
    shape = (4, len(nodes), gt.N_E_PER_GROUP)
    K = np.full(shape, k_up)
    K[3] = 0.0
    nT, ni = gt.LT.size, len(gt.ION_NAMES)
    kern = {"A": np.full(shape, sigma), "K": K, "H": 2.0 * K, "A_bg": np.full(4, a_bg), "K_bg": np.zeros(4),
            "H_bg": np.zeros(4), "alpha": np.full((3, nT), alpha), "L": np.full((3, nT), 0.1 * alpha),
            "k_mn": np.zeros((ni, nT)), "k_ct": np.zeros(ni), "k_ad": 0.0, "nc": 30.0, "nh": 12.0}
    return gt.PAHGroupTable(gt.combine([kern], [1.0], L0, L1))


def agn_photons(N):
    from pycalima.gsd.charging import AGN_GROUP_L0, AGN_GROUP_L1

    L0, L1 = np.array(AGN_GROUP_L0), np.array(AGN_GROUP_L1)
    return PhotonGroups(L0, L1, np.full(L0.size, float(N)), np.sqrt(L0 * L1))


def test_molecular_charge_is_the_detailed_balance_and_absorption_of_the_group_tables(tmp_path):
    from pycalima.gsd.molecular import C_CGS

    gt_ = [fake_group_table() for _ in range(4)]
    mt = MolecularTables(tables(tmp_path, [-10.0] * 4), groups=gt_)
    ph = agn_photons(1e-6)
    n_e, G0 = 0.01, 3.0
    st = mt.charge(ph, 100.0, n_e, None, G0)
    F = C_CGS * ph.N
    r = F.sum() * 1e-17 / (n_e * 1e-7)
    f = np.array([1.0, r, r * r, r ** 3])
    np.testing.assert_allclose(st.f[0], f / f.sum(), rtol=1e-9)
    np.testing.assert_allclose(st.r_abs, 1e-16 * F.sum() + G0 * 1e-9, rtol=1e-9)
    np.testing.assert_allclose(st.absorbed.sum(axis=1), st.r_abs, rtol=1e-9)
    assert st.energy.size == 2 * ph.L0.size and st.valid
    ref = fake_group_table().solve(F, 100.0, n_e, 0.0, (("H+", n_e),), G0)
    assert st.gamma[2] == pytest.approx(ref["gamma"], rel=1e-12) and st.lam[2] == pytest.approx(ref["lam"], rel=1e-12)
    assert st.mean[0] == pytest.approx(float(st.f[0] @ np.array([-1, 0, 1, 2])), rel=1e-12)
    bad = PhotonGroups(ph.L0[:8], ph.L1[:8], ph.N[:8], ph.egy[:8])
    with pytest.raises(ValueError):
        mt.charge(bad, 100.0, n_e)


def test_pah_ions_names_the_singly_charged_ions_by_mass():
    from pycalima.gsd.molecular import AMU, pah_ions

    assert pah_ions(None, 0.5) == (("H+", 0.5),)
    out = pah_ions([(1.0, 1, 4.0026 * AMU), (2.0, 1.0, 1.008 * AMU), (3.0, 2, 4.0026 * AMU), (4.0, 1, 12.0 * AMU)], 1.0)
    assert out == (("He+", 1.0), ("H+", 2.0), ("C+", 4.0))


def molecular_carbon(tmp_path, pair_rule=False, **kw):
    from pycalima.gsd.carbon import CarbonLayout
    from pycalima.gsd.pairs import PairRule

    c = carbon(n_H=kw.pop("n_H", 30.0))
    c.molecular_bins = 4
    c.molecular_tables = MolecularTables(tables(tmp_path, [-40.0] * 4), groups=[fake_group_table(**kw)] * 4)
    if pair_rule:
        c.pair_rule = PairRule.from_layout(CarbonLayout(), c.grid)
    return c


def test_precool_takes_the_molecular_bins_heating_and_cooling_from_their_tables(tmp_path):
    from dataclasses import replace

    c = molecular_carbon(tmp_path)
    inp = replace(inputs(), photons=agn_photons(1e-8), G0_bg=1.0)
    pre = cell.precool([c], inp)
    st = c.molecular_tables.charge(inp.photons, inp.T, inp.n_e, None, 1.0)
    np.testing.assert_allclose(pre.heating[0][:4], c.N[:4] * st.gamma, rtol=1e-12)
    np.testing.assert_allclose(pre.cooling[0][:4], c.N[:4] * st.lam, rtol=1e-12)
    assert np.all(pre.heating[0][4:] == 0.0)                  # no grain charging model: neutral grains
    assert pre.release == [None]                              # no pair rule
    assert pre.molecular[0].f.shape == (4, 4)


def test_s_eff_is_one_in_the_dark_and_zero_without_burial(tmp_path, dustem):
    """No light: S_eff = 1. Light and no growth (burial capped at 1 Gyr): small molecules on the smallest hosts are
    released (S ~ 0), none on 5-20 nm hosts; growth by accretion buries them, raising S; Y_pd releases on any
    host."""
    from dataclasses import replace

    c = molecular_carbon(tmp_path, pair_rule=True)
    dark = cell.precool([c], inputs(), cell.Options())
    assert np.all(dark.release[0] == 1.0) and dark.release[0].shape == (4, 3)
    lit = replace(inputs(), photons=agn_photons(1e-6), G0_bg=1e3)
    no_burial = cell.precool([c], lit, cell.Options(processes=frozenset({"coagulation"}))).release[0]
    assert no_burial[0, 0] < 1e-4                             # small molecules on ~1.5 nm hosts are released
    assert np.all(no_burial[:, -1] > 1 - 1e-6)                  # 5-20 nm hosts: no thermal release
    burial = cell.precool([c], lit, cell.Options()).release[0]
    assert np.all(burial >= no_burial) and 0.0 < burial[0, 0] < 1.0
    pd = cell.precool([c], lit, cell.Options(processes=frozenset({"coagulation"}), y_pd=1e-6)).release[0]
    assert np.all(pd < 1e-2)                                  # photodesorption with no burial releases everything


def test_the_molecules_charges_and_s_eff_reach_the_collisions(tmp_path, dustem):
    """S_eff = 0 on the nano-grains keeps molecules off them, so fewer freeze out; the collisions see the molecular
    bins' charges below N_C,max and the grains' (here neutral) above."""
    from dataclasses import replace

    lit = replace(inputs(n_H=1e3, T=50.0), photons=agn_photons(1e-6), G0_bg=1e3)
    opt = cell.Options(processes=frozenset({"coagulation"}), table_ppd=None)
    kept = {}
    for name in ("released", "stuck"):
        c = molecular_carbon(tmp_path / name, pair_rule=True, n_H=1e3)
        pre = cell.precool([c], lit, opt)
        if name == "stuck":
            pre.release[0] = np.ones_like(pre.release[0])
        M0 = c.M[:4].sum()
        res = cell.update([c], lit, pre, 3e4 * YR, opt)
        assert abs(res.conservation["C"]) < 1e-12
        kept[name] = c.M[:4].sum() / M0
    assert kept["released"] > kept["stuck"]
    g = c.grid
    pts = cell._charge_points(None, pre.molecular[0], g.a_edges, 4)
    Z, w = pts(g.a_centres)
    np.testing.assert_allclose(Z[:4] @ w, pre.molecular[0].mean, rtol=1e-12)
    assert np.all(Z[4:] == 0.0)


def test_observables_read_the_npz_and_precool_gives_the_molecular_h2(tmp_path):
    from dataclasses import replace

    tables(tmp_path, [-10.0] * 4)
    d = tmp_path / "cold"
    table(d / "bin_nc189-400_carbon.dat", -10.0, h2=lambda lg, ln, lt, lx: 10.0 ** (lg - 12.0))
    mt = MolecularTables.from_directories(d)
    h2 = mt.observable("h2", 100.0, 10.0, 50.0, 0.01)
    np.testing.assert_allclose(h2, [1e-12, 1e-12, 1e-12, 1e-10], rtol=1e-9)
    assert mt.observable("h2", 1e9, 10.0, 50.0, 0.01)[3] == pytest.approx(1e-8, rel=1e-9)     # clamped at log G0 4
    c = carbon(n_H=30.0)
    c.molecular_bins = 4
    c.molecular_tables = mt
    pre = cell.precool([c], replace(inputs(), G0_bg=100.0))
    np.testing.assert_allclose(pre.h2[0], c.N[:4] * h2, rtol=1e-9)
    assert pre.molecular == [None] and pre.release == [None]          # no group tables: no charge, no S_eff
