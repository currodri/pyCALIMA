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
    for mat, a in (("graphite", 1e-6), ("graphite", 1e-5), ("silicate", 5e-7)):
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


def test_uv_background_matches_the_full_solve_in_the_mathis_field(tables):
    """G0_bg * K_bg reproduces equilibrium_charge_for_grain in the Mathis ISRF
    (normalised to G0 = 1 Habing flux) with no RT photons."""
    from pycalima.models.dust_charge import rt_group_charging as rg
    from pycalima.models.dust_charge.dust_charging import equilibrium_charge_for_grain
    from pycalima.models.tools.radiation_fields import Mathis83_radiation_field

    tab, path = tables["graphite", 1e-5]
    E = np.geomspace(1.0, 13.6, 3000)
    uE = np.array([Mathis83_radiation_field(e) for e in E])
    b = E >= 6.0
    G0 = 30.0
    uE *= G0 * rg.HABING_FLUX / (C * np.trapezoid(uE[b], E[b]))
    field = np.column_stack([E, 1239.84193 / E, C * uE])
    ne, T = 0.03, 6000.0
    r = rg.GroupChargingTable(rg.read_bin_table(path)).solve(
        np.zeros(tab["L0"].size), C, T, ne, [(ne, 1.0, M_P)], G0_bg=G0)
    _, _, rates, Zm, _ = equilibrium_charge_for_grain(
        1.0, ne, T, "graphite", 1e-5, rad_field=field,
        ion_species=[{'n': ne, 'T': T, 'm': M_P * 1e-3, 'z': 1}],
        yield_params={'yield_model': 'WDB06', 'include_ion_transmission': False})
    assert Zm > 10
    assert r["Zmean"] == pytest.approx(Zm, rel=0.01)
    assert r["Gamma"] == pytest.approx(rates["Gamma_total"], rel=0.02)


def test_uniform_lookup_reproduces_solves_at_the_nodes(tables):
    from pycalima.models.dust_charge import rt_group_charging as rg

    tab, path = tables["graphite", 1e-6]
    gt = rg.GroupChargingTable(rg.read_bin_table(path))
    # a 2x2 patch of the grid is enough to pin the node values and indexing
    sub = np.zeros((4, rg.UNIF_LT.size, rg.UNIF_LNE.size))
    for i in (40, 41):
        for j in (50, 51):
            ne = 10.0 ** rg.UNIF_LNE[j]
            r = gt.solve(np.zeros(tab["L0"].size), 0.0, 10.0 ** rg.UNIF_LT[i], ne, [(ne, 1.0, M_P)], G0_bg=1.0)
            sub[:, i, j] = (r["Zmean"], r["Zsigma"], r["Gamma"], r["Lambda"] + r["Lambda_auto"])
    got = rg.uniform_lookup(sub, 10.0 ** rg.UNIF_LT[41], 10.0 ** rg.UNIF_LNE[50])
    np.testing.assert_allclose(got, sub[:, 41, 50], rtol=1e-12)


@pytest.mark.parametrize("a", [1e-6, 1e-5])
def test_warm_start_finds_the_same_root_from_any_guess(tables, a):
    """The bracket grown from Z_guess (RAMSES: the bin's previous Z*) reaches
    the cold-start root, wherever the guess lies, including pinned edges."""
    from pycalima.models.dust_charge import rt_group_charging as rg

    tab, path = tables["graphite", a]
    N, egy, _ = _bb_cell(4e4, 3.0, tab["L0"], tab["L1"])
    gt = rg.GroupChargingTable(rg.read_bin_table(path), egy=egy)
    Zlo, Zhi = tab["Zmin"], tab["Zmax"] - 1.0
    for scale, ne, T in ((1.0, 30.0, 8e3), (1e4, 1e-2, 1e4), (0.0, 1.0, 1e4), (1e8, 1e-8, 1e2)):
        cold = gt.solve(N * scale, C, T, ne, [(ne, 1.0, M_P)])
        for Zg in (Zlo + 0.5, cold["Zstar"] - 7.3, cold["Zstar"], cold["Zstar"] + 0.4, Zhi - 0.5):
            if not Zlo < Zg < Zhi:
                continue
            warm = gt.solve(N * scale, C, T, ne, [(ne, 1.0, M_P)], Z_guess=Zg)
            assert warm["Zstar"] == pytest.approx(cold["Zstar"], abs=1e-5 * max(1.0, abs(cold["Zstar"])))
            for key in ("Zmean", "Zsigma", "Gamma", "Lambda", "Lambda_auto"):
                assert warm[key] == pytest.approx(cold[key], rel=1e-5, abs=1e-40)


def test_recombination_is_the_case_a_sum_over_the_charge_distribution(tables):
    """_recombination over a discrete P(Z)'s window, and on charges across the ion thresholds
    read from the table file, is the case-A sum written from its definition (WD01b):
    sum_Z P(Z) pi a^2 (8kT / pi m_i)^1/2 J~(Z, 1) [IP(a, Z) <= IP(X)]; alpha_rec, which adds
    the charges below the window, is the sum over the exact P(Z)."""
    from pycalima.models.dust_charge import rt_group_charging as rg
    from pycalima.models.dust_charge.shared_physics import (
        DS87_J_function_vec, electron_affinity_graphite_vec, ionisation_potential_valence_vec)

    tab, path = tables["graphite", 1e-6]
    N, egy, _ = _bb_cell(4e4, 3.0, tab["L0"], tab["L1"])
    gt = rg.GroupChargingTable(rg.read_bin_table(path), egy=egy)
    ne, T, a = 30.0, 8e3, tab["a"]
    r = gt.solve(N, C, T, ne, [(ne, 1.0, M_P)])
    assert r["discrete"]

    def case_a(Z, P):
        ip = np.where(Z >= 0, ionisation_potential_valence_vec(tab["W"], Z, a), electron_affinity_graphite_vec(Z + 1.0, a))
        PA = P * np.pi * a ** 2 * DS87_J_function_vec(Z, 1.0, a, T)
        return np.array([np.sum(PA * (ip <= ip_x)) * np.sqrt(8 * rg.KB_CGS * T / (np.pi * m * rg.M_U))
                         for _, m, ip_x in rg.RECOMB_IONS])

    np.testing.assert_allclose(gt._recombination(r["nodes"], r["weights"]), case_a(r["nodes"], r["weights"]),
                               rtol=1e-10)
    np.testing.assert_allclose(r["alpha_rec"], gt._recombination_exact(r["Zstar"], max(r["Zsigma"], 0.5)), rtol=5e-5)
    Z = np.arange(30.0, 150.0)          # Mg+, Fe+, Si+ thresholds below these charges, the others among them
    P = np.random.default_rng(1).random(Z.size)
    alpha = gt._recombination(Z, P / P.sum())
    assert 0 < np.count_nonzero(alpha) < alpha.size
    np.testing.assert_allclose(alpha, case_a(Z, P / P.sum()), rtol=1e-10)


@pytest.mark.parametrize("state", ["cold", "fuv_lit"])
def test_recombination_of_a_discrete_distribution_includes_the_charges_below_its_window(tables, state):
    """On a positive small grain in cold gas the capture rate rises by ~exp(e^2/akT) per charge
    towards lower Z (e^2/akT ~ 110 for 5 nm at 30 K, ~20 at 160 K), so the recombination of a
    discrete P(Z) is set by charges below its window, near Z <= 0. alpha_rec is the exact sum,
    also far below the geometric rate (C+ on the grain in FUV-lit neutral gas: < 1e-20 of it),
    and it varies smoothly where the window's lowest charge moves. There a sum over the window
    alone jumps by up to ~e^500, and one stopped at 1e-8 of the geometric rate by ~e^13."""
    from pycalima.models.dust_charge import rt_group_charging as rg

    tab, path = tables["silicate", 5e-7]
    iC = [ion for ion, _, _ in rg.RECOMB_IONS].index("C")
    if state == "cold":                 # 30 K, n_e swept
        N, egy, _ = _bb_cell(4e4, 1.0, tab["L0"], tab["L1"])
        T, cells, jump = 30.0, [(N, ne) for ne in np.geomspace(2.5e-3, 3.5e-3, 21)], 0.5
    else:                               # 160 K neutral gas, no photons above 13.6 eV, G0 swept
        N, egy, _ = _bb_cell(4.29e4, 1.0, tab["L0"], tab["L1"])
        N = np.where(tab["L1"] <= 13.6 + 1e-9, N, 0.0)
        T, cells, jump = 160.0, [(G0 * N, 0.057) for G0 in np.geomspace(1e3, 1e4, 25)], 3.0
    gt = rg.GroupChargingTable(rg.read_bin_table(path), egy=egy)
    ln_alpha, lowest = [], set()
    for Ng, ne in cells:
        r = gt.solve(Ng, C, T, ne, [(ne, 1.0, M_P)])
        assert r["discrete"]
        exact = gt._recombination_exact(r["Zstar"], max(r["Zsigma"], 0.5))
        ok = exact > 1e-280
        assert ok[iC]
        np.testing.assert_allclose(r["alpha_rec"][ok], exact[ok], rtol=5e-5)
        ln_alpha.append(np.log(r["alpha_rec"][iC]))
        lowest.add(r["nodes"][0])
    assert len(lowest) > 1
    assert np.max(np.abs(np.diff(ln_alpha))) < jump
    if state == "fuv_lit":
        geo = np.pi * tab["a"] ** 2 * np.sqrt(8.0 * rg.KB_CGS * T / (np.pi * rg.M_U))
        assert np.exp(max(ln_alpha)) < 1e-20 * geo


@pytest.mark.parametrize("scale, ne, T", [(0.0, 0.03, 100.0), (0.0, 0.03, 6000.0), (1.0, 30.0, 8e3)])
def test_exact_recombination_of_a_wide_distribution_is_the_integer_sum(tables, scale, ne, T):
    """recomb="exact" is the sum over every integer charge of P(Z) from detailed balance."""
    from pycalima.models.dust_charge import rt_group_charging as rg

    tab, path = tables["graphite", 1e-5]
    N, egy, _ = _bb_cell(4e4, 3.0, tab["L0"], tab["L1"])
    gt = rg.GroupChargingTable(rg.read_bin_table(path), egy=egy)
    r = gt.solve(N * scale, C, T, ne, [(ne, 1.0, M_P)], G0_bg=1.0, recomb="exact")
    assert not r["discrete"]
    w = int(80 * r["Zsigma"])
    Z = np.arange(max(tab["Zmin"], np.floor(r["Zstar"]) - w), min(tab["Zmax"] - 1, np.floor(r["Zstar"]) + w) + 1)
    lnf = np.concatenate([[0.0], np.cumsum([gt._g(z) for z in Z[:-1]])])
    P = np.exp(lnf - lnf.max())
    np.testing.assert_allclose(r["alpha_rec"], gt._recombination(Z, P / P.sum()), rtol=1e-6, atol=1e-300)


@pytest.mark.parametrize("a", [1e-6, 1e-5])
@pytest.mark.parametrize("Ts, G0, T, ne", [(1e5, 1.0, 30.0, 1e-3), (3e4, 1e2, 500.0, 1e-2),
                                           (4e4, 1e3, 8e3, 1.0), (3e4, 1.0, 1e6, 1e-2)])
def test_fitted_recombination_of_a_wide_distribution_is_within_ten_percent(tables, a, Ts, G0, T, ne):
    """recomb="fit" (PCHIP kernels) against the exact sum, per ion, where the
    rate is resolved (> 1e-6 of the geometric rate, see RECOMB_FLOOR); neutral
    gas has no ionising photons."""
    from pycalima.models.dust_charge import rt_group_charging as rg

    tab, path = tables["graphite", a]
    N, egy, _ = _bb_cell(Ts, G0, tab["L0"], tab["L1"])
    if T < 5e3:
        N = np.where(tab["L0"] >= 13.6, 0.0, N)
    gt = rg.GroupChargingTable(rg.read_bin_table(path), egy=egy, kernels="pchip")
    fit = gt.solve(N, C, T, ne, [(ne, 1.0, M_P)])
    if fit["discrete"]:
        pytest.skip("narrow P(Z): exact sum")
    ex = gt.solve(N, C, T, ne, [(ne, 1.0, M_P)], recomb="exact")
    geo = np.pi * a ** 2 * np.sqrt(8 * 1.380649e-16 * T / (np.pi * np.array([m for _, m, _ in rg.RECOMB_IONS]) * rg.M_U))
    ok = ex["alpha_rec"] > 1e-6 * geo
    np.testing.assert_allclose(fit["alpha_rec"][ok], ex["alpha_rec"][ok], rtol=0.1)
    assert np.all(fit["alpha_rec"][~ok] < 1e-5 * geo[~ok])    # unresolved rates stay negligible


def test_recombination_thresholds_follow_the_ionisation_potentials(tables):
    """Z_th rises with IP(X); silicate (W = 8 eV) must be negative to neutralise Mg+ and Fe+."""
    from pycalima.models.dust_charge import rt_group_charging as rg

    ip = np.array([p for _, _, p in rg.RECOMB_IONS])
    for mat, a in (("graphite", 1e-6), ("silicate", 1e-5)):
        tab = rg.read_bin_table(tables["graphite", 1e-6][1]) if mat == "graphite" else None
        if tab is None:
            tab = {"recomb_Zth": rg.recombination_thresholds("silicate", a, 8.0, -5000.0, 30000.0)}
        order = np.argsort(ip)
        assert np.all(np.diff(tab["recomb_Zth"][order]) >= 0)
    zth = dict(zip([e for e, _, _ in rg.RECOMB_IONS], rg.recombination_thresholds("silicate", 1e-5, 8.0, -5000.0, 30000.0)))
    assert zth["Mg"] < 0 and zth["Fe"] < 0 and zth["H"] > 0


def _exact_moments(gt, Zs, width):
    """<Z> and sigma_Z of the exact P(Z), from g on every integer around Z*."""
    Z = np.arange(max(gt.t["Zmin"], np.floor(Zs - 15 * width - 10)), min(gt.t["Zmax"], np.ceil(Zs + 15 * width + 10)) + 1.0)
    lnf = np.concatenate([[0.0], np.cumsum([gt._g(z) for z in Z[:-1]])])
    P = np.exp(lnf - lnf.max())
    P /= P.sum()
    m = P @ Z
    return m, np.sqrt(P @ Z ** 2 - m ** 2)


def test_charge_width_holds_across_the_kink_of_g_at_zero(tables):
    """A 10 nm graphite grain lit by soft FUV only, through the fields where its charge
    crosses Z = 0: there g(Z) has a kink, and the secant at Z* +- 1/2 read sigma ~3x
    too wide, flipping a P(Z) of sigma ~2.8 to the Gaussian branch. The chord width
    (_chord) keeps sigma and <Z> on the exact P(Z)."""
    from pycalima.models.dust_charge import rt_group_charging as rg

    tab, path = tables["graphite", 1e-6]
    gt = rg.GroupChargingTable(rg.read_bin_table(path))
    soft = np.nonzero((np.asarray(tab["L0"]) >= 5.6) & (np.asarray(tab["L1"]) <= 11.2))[0]
    T, ne = 1e4, 1.0
    crossed = False
    for x in np.geomspace(1e-2, 3e-1, 61):
        N = np.zeros(len(tab["L0"]))
        N[soft] = x
        r = gt.solve(N, C, T, ne, [(ne, 1.0, M_P)], recomb=None)
        m, sig = _exact_moments(gt, r["Zstar"], max(r["Zsigma"], 0.5))
        crossed |= abs(m) < 1.0
        assert r["Zsigma"] == pytest.approx(sig, rel=0.06)
        assert r["Zmean"] == pytest.approx(m, abs=0.15 * sig)   # wide: mean at Z* + 1/2
    assert crossed


@pytest.mark.parametrize("a", [1e-6, 1e-5])
def test_warm_started_solve_matches_a_cold_one(tables, a):
    """The next sub-step, warm-started from the previous result, gives the cold result."""
    from pycalima.models.dust_charge import rt_group_charging as rg

    tab, path = tables["graphite", a]
    N, egy, _ = _bb_cell(4e4, 3.0, tab["L0"], tab["L1"])
    gt = rg.GroupChargingTable(rg.read_bin_table(path), egy=egy)
    prev = gt.solve(N, C, 8e3, 30.0, [(30.0, 1.0, M_P)], recomb=None)
    warm = gt.solve(N * 1.001, C, 8e3 * 1.0005, 30.0, [(30.0, 1.0, M_P)], recomb=None, warm=prev)
    cold = gt.solve(N * 1.001, C, 8e3 * 1.0005, 30.0, [(30.0, 1.0, M_P)], recomb=None)
    rel = 1e-5 if cold["discrete"] else 1e-3       # wide: the slope of the warm pair is not re-centred
    for key in ("Zmean", "Zsigma", "Gamma", "Lambda"):
        assert warm[key] == pytest.approx(cold[key], rel=rel, abs=1e-6)


@pytest.mark.parametrize("a", [1e-6, 1e-5])
def test_temperature_derivatives_match_central_differences(tables, a):
    """dlnT gives the d/d ln T that the cooling solver takes from solves at T (1 +- 1e-5)."""
    from pycalima.models.dust_charge import rt_group_charging as rg

    tab, path = tables["graphite", a]
    N, egy, _ = _bb_cell(4e4, 3.0, tab["L0"], tab["L1"])
    gt = rg.GroupChargingTable(rg.read_bin_table(path), egy=egy)
    T, ne, d = 8e3, 30.0, 1e-5
    r = gt.solve(N, C, T, ne, [(ne, 1.0, M_P)], recomb=None, dlnT=d)
    up, dn = (gt.solve(N, C, T * (1 + s), ne, [(ne, 1.0, M_P)], recomb=None) for s in (d, -d))
    h = np.log1p(d) - np.log1p(-d)
    for key in ("Gamma", "Lambda"):
        f = (lambda q: q[key] + (q["Lambda_auto"] if key == "Lambda" else 0.0))
        rel = 1e-6 if r["discrete"] else 1e-2          # discrete: analytic; wide: forward difference
        assert r["d" + key + "_dlnT"] == pytest.approx((f(up) - f(dn)) / h, rel=rel, abs=rel * 1e-1 * abs(f(r)))


def test_warm_solves_reuse_the_recombination_fit_only_within_the_tolerance(tables):
    from pycalima.models.dust_charge import rt_group_charging as rg

    tab, path = tables["graphite", 1e-5]
    N, egy, _ = _bb_cell(4e4, 3.0, tab["L0"], tab["L1"])
    gt = rg.GroupChargingTable(rg.read_bin_table(path), egy=egy, kernels="pchip")
    args = (C, 8e3, 30.0, [(30.0, 1.0, M_P)])
    r0 = gt.solve(N, *args)
    assert not r0["discrete"]
    f = 1 + 0.2 * rg.RECOMB_REUSE_TOL
    near = gt.solve(N * f, *args, warm=r0)
    assert near["recomb_ref"] is r0["recomb_ref"]                       # reused, carried by the estimate
    fresh = gt.solve(N * f, *args)["alpha_rec"]
    ok = fresh > rg.RECOMB_RESOLVED * gt._geometric_rates()
    np.testing.assert_allclose(near["alpha_rec"][ok], fresh[ok], rtol=1e-2)
    f = 1 + 3 * rg.RECOMB_REUSE_TOL
    far = gt.solve(N * f, *args, warm=near)
    assert far["recomb_ref"] is not r0["recomb_ref"]                    # re-fitted: drift from the fitted state
    np.testing.assert_allclose(far["alpha_rec"], gt.solve(N * f, *args)["alpha_rec"], rtol=1e-6)


def test_a_warm_discrete_window_follows_a_large_change_of_the_cell(tables):
    """The reused window regrows where the charge moved: the cold result, and P ends below DISCRETE_TAIL."""
    from pycalima.models.dust_charge import rt_group_charging as rg

    tab, path = tables["graphite", 1e-6]
    N, egy, _ = _bb_cell(4e4, 3.0, tab["L0"], tab["L1"])
    gt = rg.GroupChargingTable(rg.read_bin_table(path), egy=egy)
    prev = gt.solve(N, C, 8e3, 30.0, [(30.0, 1.0, M_P)], recomb=None)
    assert prev["discrete"]
    args = (N, C, 2e3, 300.0, [(300.0, 1.0, M_P)])
    warm, cold = gt.solve(*args, warm=prev), gt.solve(*args)
    assert warm["discrete"] and abs(warm["Zmean"] - prev["Zmean"]) > 3.0
    for key in ("Zmean", "Zsigma", "Gamma", "Lambda"):
        assert warm[key] == pytest.approx(cold[key], rel=1e-9)
    ends = warm["weights"][[0, -1]] / warm["weights"].max()
    at_limit = warm["nodes"][[0, -1]] == (tab["Zmin"], tab["Zmax"])
    assert np.all((ends < rg.DISCRETE_TAIL) | at_limit)


def test_solve_cell_fits_only_the_bins_that_carry_the_recombination(tables):
    """Wide bins keep the Gaussian estimate unless they carry more than RECOMB_IMPORTANT of an ion's rate."""
    from pycalima.models.dust_charge import rt_group_charging as rg

    gts = []
    for a in (1e-6, 1e-5):
        tab, path = tables["graphite", a]
        N, egy, _ = _bb_cell(4e4, 3.0, tab["L0"], tab["L1"])
        gts.append(rg.GroupChargingTable(rg.read_bin_table(path), egy=egy, kernels="pchip"))
    args = (N, C, 8e3, 30.0, [(30.0, 1.0, M_P)])
    for n_big, expect in ((1e-12, False), (1e12, True)):
        res = rg.solve_cell(gts, [1.0, n_big], *args)
        assert res[0]["discrete"] and not res[0]["refined"] and not res[1]["discrete"]
        assert res[1]["refined"] is expect
        if expect:
            np.testing.assert_allclose(res[1]["alpha_rec"], gts[1].solve(*args)["alpha_rec"], rtol=1e-12)
