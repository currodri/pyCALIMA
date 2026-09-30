"""RT-group charging tables (rt_group_charging) and their per-cell solver."""

from __future__ import annotations

import numpy as np
import pytest

C = 2.99792458e10
EV = 1.602176634e-12
KB_EV = 8.617333262e-5
M_P = 1.6726219e-24


def _bb_cell(Tsrc, G0, L0, L1):
    """Group photon densities and true group mean energies of a blackbody
    normalised to G0 Habing units in 6-13.6 eV; also the full spectrum."""
    E = np.geomspace(0.1, 500.0, 6000)
    uE = E ** 3 / np.expm1(E / (KB_EV * Tsrc))
    b = (E >= 6) & (E <= 13.6)
    uE *= G0 * 5.33e-14 / np.trapezoid(uE[b], E[b])
    N, egy = [], []
    for l0, l1 in zip(L0, L1):
        m = (E >= l0) & (E < l1)
        n = np.trapezoid(uE[m] / (E[m] * EV), E[m])
        N.append(n)
        egy.append(np.trapezoid(uE[m], E[m]) / EV / n)
    return np.array(N), np.array(egy), np.column_stack([E, 1239.84193 / E, C * uE])


@pytest.fixture(scope="module")
def tables(tmp_path_factory):
    from pycalima.models.dust_charge import rt_group_charging as rg

    out = {}
    for mat, a in (("graphite", 1e-6), ("graphite", 1e-5)):
        tab = rg.build_bin_table(mat, a, U=rg.u_grid(n=241), Ts=np.geomspace(3e3, 3e6, 13),
                                 T_hot=np.geomspace(1e3, 1e9, 13), label=f"{mat}_{a:g}")
        path = rg.write_bin_table(tab, tmp_path_factory.mktemp("rtg") / f"{mat}_{a:g}.dat")
        out[mat, a] = (tab, path)
    return out


def test_file_round_trip(tables):
    from pycalima.models.dust_charge import rt_group_charging as rg

    tab, path = tables["graphite", 1e-5]
    back = rg.read_bin_table(path)
    for key in ("L0", "L1", "U", "log10_Ts", "egy", "K", "H", "delta", "P_tr", "f_cool", "E_bar"):
        np.testing.assert_allclose(back[key], tab[key], rtol=1e-11, atol=0)
    assert back["material"] == "graphite" and back["Zmin"] == tab["Zmin"]


def test_group_collapse_hits_table_rows(tables):
    from pycalima.models.dust_charge import rt_group_charging as rg

    _, path = tables["graphite", 1e-5]
    back = rg.read_bin_table(path)
    gt = rg.GroupChargingTable(back, egy=back["egy"][:, 4])
    np.testing.assert_array_equal(gt.Kg, back["K"][:, 4])
    np.testing.assert_array_equal(gt.Hg, back["H"][:, 4])


def test_kernels_are_physical(tables):
    tab, _ = tables["graphite", 1e-5]
    assert np.all(tab["K"] >= 0) and np.all(tab["H"] >= 0)
    # a non-negative grain cannot emit below the 4.4 eV work function (negative
    # grains can: lower ionisation threshold and photodetachment)
    assert np.all(tab["K"][0][:, tab["U"] >= 0] == 0)
    # more positive grains emit less, per photon
    assert np.all(np.diff(tab["K"][5], axis=-1)[:, tab["U"][1:] > 1] <= 1e-30)
    assert np.all((tab["P_tr"] >= 0) & (tab["P_tr"] <= 1) & (tab["f_cool"] >= 0) & (tab["f_cool"] <= 1))


@pytest.mark.parametrize("a, tol_Z", [(1e-5, 0.005), (1e-6, 0.03)])
def test_solver_matches_the_full_solve_for_a_blackbody(tables, a, tol_Z):
    """A source whose within-group shape is in the table family: only the
    interpolation and the Gaussian/discrete closure separate the two."""
    from pycalima.models.dust_charge import rt_group_charging as rg
    from pycalima.models.dust_charge.dust_charging import equilibrium_charge_for_grain

    tab, path = tables["graphite", a]
    N, egy, field = _bb_cell(1e5, 1e4, tab["L0"], tab["L1"])
    ne, T = 100.0, 8e3
    r = rg.GroupChargingTable(rg.read_bin_table(path), egy=egy).solve(N, C, T, ne, [(ne, 1.0, M_P)])
    _, _, rates, Zm, Zs = equilibrium_charge_for_grain(
        1.0, ne, T, "graphite", a, rad_field=field,
        ion_species=[{'n': ne, 'T': T, 'm': M_P * 1e-3, 'z': 1}],
        yield_params={'yield_model': 'WDB06', 'include_ion_transmission': False})
    assert r["Zmean"] == pytest.approx(Zm, rel=tol_Z)
    assert r["Zsigma"] == pytest.approx(Zs, rel=0.05)
    assert r["Gamma"] == pytest.approx(rates["Gamma_total"], rel=0.02)
    assert r["Lambda"] == pytest.approx(rates["Recomb_total"], rel=0.02)


def test_discrete_charge_distribution_matches_the_full_solve(tables):
    """A 100 A grain in a weak field has sigma_Z < 3: P(Z) is built exactly."""
    from pycalima.models.dust_charge import rt_group_charging as rg
    from pycalima.models.dust_charge.dust_charging import equilibrium_charge_for_grain

    tab, path = tables["graphite", 1e-6]
    N, egy, field = _bb_cell(4e4, 3.0, tab["L0"], tab["L1"])
    ne, T = 30.0, 8e3
    r = rg.GroupChargingTable(rg.read_bin_table(path), egy=egy).solve(N, C, T, ne, [(ne, 1.0, M_P)])
    _, _, rates, Zm, Zs = equilibrium_charge_for_grain(
        1.0, ne, T, "graphite", 1e-6, rad_field=field,
        ion_species=[{'n': ne, 'T': T, 'm': M_P * 1e-3, 'z': 1}],
        yield_params={'yield_model': 'WDB06', 'include_ion_transmission': False})
    assert r["discrete"]
    assert r["Zmean"] == pytest.approx(Zm, abs=0.05 * max(1.0, abs(Zm)))
    assert r["Zsigma"] == pytest.approx(Zs, rel=0.05)
    assert r["Gamma"] == pytest.approx(rates["Gamma_total"], rel=0.03)
    assert r["Lambda"] == pytest.approx(rates["Recomb_total"], rel=0.03)


def test_no_field_gives_a_negative_collisional_potential(tables):
    from pycalima.models.dust_charge import rt_group_charging as rg

    tab, path = tables["graphite", 1e-5]
    gt = rg.GroupChargingTable(rg.read_bin_table(path))
    T = 1e4
    r = gt.solve(np.zeros(tab["L0"].size), C, T, 1.0, [(1.0, 1.0, M_P)])
    phi = r["Zmean"] * 1.43996e-7 / 1e-5 / (KB_EV * T)   # eU/kT
    assert -3.0 < phi < -1.5   # Spitzer -2.51 at s_e = 1; WD01 s_e = 0.5 gives ~ -2.1
    assert r["Gamma"] == 0.0 and r["Lambda"] > 0.0
