"""THEMIS a-C grain charging ('cm20' in wdb06_yields and rt_group_charging).

All tests need the DustEM THEMIS dataset (the optics, and the CHRG_CM20 constants
and charge distributions it ships) and skip without it.
"""

from __future__ import annotations

import numpy as np
import pytest

AMU = 1.66053906660e-24


def _chrg_numbers(root):
    rows = [l.split() for l in (root / "data" / "CHRG_CM20.DAT").read_text().splitlines()
            if l.strip() and not l.startswith("#")]
    return [[float(x) for x in r] for r in rows]


def test_constants_are_dustems(themis_cm20):
    from pycalima.models.dust_charge import wdb06_yields as wy

    l1, l2, l3, l4, _ = _chrg_numbers(themis_cm20)
    assert wy.CM20["W"] == l1[0] and wy.CM20["uait"] == tuple(l1[1:4])
    assert wy.CM20["aromatic"] == tuple(l2[0:2]) and wy.CM20["stick"] == tuple(l2[2:5])
    assert l3[0] == 0.0 and wy.CM20["le_nm"] == tuple(l3[1:5])     # p_ke = 0: no forced KE
    assert l4[4] == -1                                              # E_bg(a): the size law
    assert wy.MATERIALS["cm20"]["rho"] == 1.6


def test_band_yield_is_a_probability_that_starts_at_the_threshold(themis_cm20):
    """(Not monotonic in Z: Kimura's y0 rises with e2/e_t, by ~1% between charges.)"""
    from pycalima.models.dust_charge import wdb06_yields as wy

    for a in (4e-8, 1e-6, 1e-5):
        E = np.linspace(5.0, 13.6, 60)
        Zs = np.arange(-3.0, 6.0) if a < 1e-7 else np.arange(-20.0, 60.0, 5.0)
        Y, _, le = wy.cm20_band_yield(E, Zs, a, 0.75 * wy.cm20_extinction_efficiency(E, a))
        assert np.all((Y >= 0) & (Y <= 1)) and np.all(le > 0)
        thr = wy.threshold_energy(wy.CM20["W"], Zs[None, :], a, "cm20")
        assert np.all(Y[E[:, None] <= thr] == 0)
        assert np.all(Y[E[:, None] > thr + 0.5] > 0)


def test_thresholds_carry_the_aromatic_correction_and_the_band_gap(themis_cm20):
    from pycalima.models.dust_charge import wdb06_yields as wy

    a = np.array([4e-8, 1e-7, 1e-6])
    e2a = 1.43996448e-7 / a
    np.testing.assert_allclose(wy.aromatic_correction(a), (np.pi - 1.0) / 2.0 * e2a, rtol=1e-6)
    np.testing.assert_allclose(wy.cm20_band_gap(a), [2.3, 0.8, 0.1])
    ip = wy.ionisation_potential(wy.CM20["W"], 0.0, a, "cm20")
    ea = wy.cm20_electron_affinity(1.0, a)            # the electron of Z = 0 -> Z = 1
    np.testing.assert_allclose(ip - ea, wy.cm20_band_gap(a) + 2.0 * wy.aromatic_correction(a), rtol=1e-12)


def test_table_round_trip_carries_the_sticking_and_graphite_is_unchanged(themis_cm20, tmp_path):
    from pycalima.models.dust_charge import rt_group_charging as rg

    kw = dict(U=rg.u_grid(n=121), Ts=np.geomspace(3e3, 3e6, 5), T_hot=np.geomspace(1e3, 1e9, 5))
    tab = rg.build_bin_table("cm20", 1e-6, (5.6,), (13.6,), **kw)
    assert tab["version"] == rg.FORMAT_VERSION_STICKING
    back = rg.read_bin_table(rg.write_bin_table(tab, tmp_path / "cm20.dat"))
    assert back["material"] == "cm20" and back["W"] == 4.75
    np.testing.assert_allclose(back["s_e"], tab["s_e"], rtol=1e-11)
    Z = tab["U"] * 1e-6 / rg.E2_EV_CM
    s = tab["s_e"][Z > tab["Zmin"]]
    assert np.all((s > 0) & (s <= 0.5)) and np.all(tab["s_e"][Z <= tab["Zmin"]] == 0)
    gra = rg.build_bin_table("graphite", 1e-6, (5.6,), (13.6,), **kw)
    assert gra["version"] == rg.FORMAT_VERSION and "s_e" not in gra


@pytest.mark.parametrize("a_nm", [9.22, 44.3, 144.0])
def test_charge_matches_dustems_own_distribution(themis_cm20, monkeypatch, a_nm):
    """Against out/ZDIST.RES of the release (WNM gas, Mathis ISRF), at radii where it
    agrees with a fresh build of DustEM (below ~0.5 um; above, the shipped file is
    an artefact), with WDB06's photo-excited secondaries off, as DustEM has none
    (with them: photoemission ~1-2% higher, <Z> up by up to half a sigma_Z)."""
    import functools
    import importlib.util
    from pathlib import Path

    from pycalima.models.dust_charge import wdb06_yields as wy

    for name in ("total_yield", "yield_energy_product"):
        monkeypatch.setattr(wy, name, functools.partial(getattr(wy, name), include_secondaries=False))
    spec = importlib.util.spec_from_file_location(
        "check_cm20_dustem",
        Path(__file__).parents[1] / "diagnostics" / "dust_charge" / "check_cm20_dustem.py")
    ck = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ck)

    recs, gas = ck.read_zdist(themis_cm20 / "out" / "ZDIST.RES")
    rec = min(recs, key=lambda r: abs(r["a"] * 1e7 - a_nm))
    lam, fnu = ck.dustem_isrf(themis_cm20, gas["G0"])
    gt, Z, P = ck.solve("cm20", rec["a"], gas, ck.habing_g0(lam, fnu))
    zm = P @ Z
    sd = np.sqrt(P @ Z ** 2 - zm ** 2)
    assert abs(zm - rec["Zmean"]) < 0.2 * rec["sdZ"]
    assert sd == pytest.approx(rec["sdZ"], rel=0.03)
    jpe, _, _ = ck.charge_rates(gt, Z, P)
    # 2%: the gas-electron secondaries of warm gas (DS79 for E0 > max(W, E_s)), which
    # DustEM lacks, make a 10 nm grain slightly more positive
    assert jpe / rec["Jpe"] == pytest.approx(1.0, abs=0.02)


# measured ionisation energies [eV] (NIST WebBook; Joblin et al. 2017 for the cluster;
# Wenzel et al. 2020 for the second IEs of cations): (N_C, Z, IE)
MEASURED_IE = ((16, 0, 7.43), (24, 0, 7.29), (120, 0, 6.45), (30, 1, 10.2), (48, 1, 9.1))


def _radius_cm(n_c, rho=1.6):
    return (3 * n_c * 12.011 * AMU / (4 * np.pi * rho)) ** (1 / 3)


def test_ac_thresholds_follow_the_measured_ionisation_energies(themis_cm20):
    """pyCALIMA's a-C (WD01 eq. 2, W = 4.75 eV, radii at 1.6 g/cm^3) is within 0.55 eV
    of measured IEs from pyrene to a coronene pentamer and C30-C48 cations; DustEM's
    aromatic correction puts them 2.6-3.6 eV too high."""
    from pycalima.models.dust_charge import wdb06_yields as wy

    for n_c, z, ie in MEASURED_IE:
        a = _radius_cm(n_c)
        assert abs(float(wy.ionisation_potential(4.75, z, a, "ac")) - ie) < 0.55
        assert float(wy.ionisation_potential(4.75, z, a, "cm20")) - ie > 1.4
    a = _radius_cm(24)
    np.testing.assert_allclose(wy.electron_affinity(0.0, a, "ac") - wy.electron_affinity(0.0, a, "cm20"),
                               wy.aromatic_correction(a))


def test_ac_tables_use_wd01_sticking_and_the_recomputed_optics(themis_cm20, tmp_path):
    from pycalima._datasets import get_dataset
    from pycalima.models.dust_charge import rt_group_charging as rg

    if get_dataset("optical-jones-2012-size").locate() is None:
        pytest.skip("Jones (2012) tables not fetched")
    kw = dict(U=rg.u_grid(n=121), Ts=np.geomspace(3e3, 3e6, 5), T_hot=np.geomspace(1e3, 1e9, 5))
    tab = rg.build_bin_table("ac", 1e-6, (5.6,), (13.6,), **kw)
    assert tab["version"] == rg.FORMAT_VERSION and "s_e" not in tab
    back = rg.read_bin_table(rg.write_bin_table(tab, tmp_path / "ac.dat"))
    assert back["material"] == "ac" and back["W"] == 4.75
    gt = rg.GroupChargingTable(back)
    assert gt._sticking(5.0) == pytest.approx(0.5 * (1 - np.exp(-1e-6 / 1e-7)))


def test_ek16_reproduces_the_perylene_photoelectric_efficiency_of_hrodmarsson2025():
    """Neutral perylene (N_C = 20, a = 3.5 A) in the Mathis field, with the Draine & Li
    (2007) absorption and WD01's IE (W = 4.4 eV), as in Hrodmarsson et al. (2025,
    Sect. 4, eq. 5): eps_PE ~ 1.5%."""
    from pycalima.models.dust_charge import wdb06_yields as wy
    from pycalima.models.tools.radiation_fields import Mathis83_radiation_field
    from pycalima.pahchem import photo

    a = 3.5e-8
    E = np.linspace(1.0, 13.6, 2000)
    I = np.array([Mathis83_radiation_field(e) for e in E])
    sig = photo.sigma_amespahdb(20, 0, E)
    wy.MATERIALS["_perylene"] = dict(wy.MATERIALS["ac"], W=4.4)
    try:
        Y, e2, _ = wy.ek16_band_yield(E, [0.0], a, 0.75 * sig / (np.pi * a * a), "_perylene")
    finally:
        del wy.MATERIALS["_perylene"]
    eps = np.trapezoid(I * sig * Y[:, 0] * e2[:, 0] / E, E) / np.trapezoid(I * sig, E)
    assert 0.013 < eps < 0.018


def test_ach_thresholds_are_the_polymeric_a_ch_of_schafer_1993(themis_cm20, tmp_path):
    """Bulk IE 4.6 eV (work function 2.6 eV + valence-band top 2 eV below E_F) and
    electron affinity 0, in WD01's size and charge terms."""
    from pycalima._datasets import get_dataset
    from pycalima.models.dust_charge import rt_group_charging as rg
    from pycalima.models.dust_charge import wdb06_yields as wy

    a = 1e-4                                           # 1 um: the bulk values plus small corrections
    e2a = 1.43996448e-7 / a
    assert float(wy.ionisation_potential(wy.MATERIALS["ach"]["W"], 0.0, a, "ach")) == pytest.approx(4.6 + 0.5 * e2a, abs=1e-4)
    assert float(wy.electron_affinity(0.0, a, "ach")) == pytest.approx(-0.5 * e2a)
    if get_dataset("optical-jones-2012-size").locate() is None:
        pytest.skip("Jones (2012) tables not fetched")
    kw = dict(U=rg.u_grid(n=121), Ts=np.geomspace(3e3, 3e6, 5), T_hot=np.geomspace(1e3, 1e9, 5))
    tab = rg.build_bin_table("ach", 1e-7, (5.6,), (13.6,), **kw)
    back = rg.read_bin_table(rg.write_bin_table(tab, tmp_path / "ach.dat"))
    assert back["material"] == "ach" and back["W"] == pytest.approx(4.6)
    assert np.all(back["K_bg"] >= 0) and np.all(back["K"] >= 0)
