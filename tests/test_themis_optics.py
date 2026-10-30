"""THEMIS a-C(:H) optics (pycalima.models.dust_radiation.themis_optics).

All tests need the DustEM THEMIS dataset and skip without it.
"""
import numpy as np
import pytest

from pycalima.models.dust_radiation import themis_optics as th


def test_tables_are_reproduced_at_their_nodes(themis_cm20):
    t = th.load_cm20()
    ia = np.array([0, 17, 50, 97])
    il = np.array([0, 200, 500, 799])
    qabs, qsca, g = th.efficiencies(t.a_um[ia], t.lam_um[il])
    np.testing.assert_allclose(qabs, t.qabs[np.ix_(il, ia)], rtol=1e-10)
    np.testing.assert_allclose(qsca, t.qsca[np.ix_(il, ia)], rtol=1e-10)
    np.testing.assert_allclose(g, t.g[np.ix_(il, ia)], rtol=1e-10, atol=1e-14)


def test_efficiencies_are_physical_on_both_sides_of_the_splice(themis_cm20):
    a = np.geomspace(3.5e-4, 1.0, 12)
    lam = np.geomspace(2.5e-3, 1e3, 40)            # 500 eV to 1 mm
    qabs, qsca, g = th.efficiencies(a, lam)
    assert np.all(np.isfinite(qabs)) and np.all(qabs > 0)
    assert np.all(qsca >= 0) and np.all(np.abs(g) <= 1)


def test_xray_absorption_per_volume_is_size_independent(themis_cm20):
    """At 500 eV |m - 1| ~ 1e-3: absorption is volumetric for grains that are
    optically thin, so C_abs / V does not depend on a."""
    a = np.array([3.5e-4, 1e-3, 3e-3, 1e-2])         # all a-C, below the mantle
    lam = np.array([1.23984193 / 500.0])
    c = th.cross_sections(a, lam)["C_abs"][0]
    per_volume = c / (4.0 / 3.0 * np.pi * (a * 1e-4) ** 3)
    np.testing.assert_allclose(per_volume, per_volume[0], rtol=0.02)


def test_out_of_table_sizes_are_refused(themis_cm20):
    with pytest.raises(ValueError, match="radius"):
        th.efficiencies([1e-4], [0.5])
    with pytest.raises(ValueError, match="wavelength"):
        th.efficiencies([0.1], [2e5])


def _dustem_population(line):
    """a [cm], a^4 dn/da and rho on DustEM's own size grid for one
    GRAIN_J13.DAT row (DM_inout.f90: ln-spaced, PLAW[-ED] or LOGN)."""
    p = line.split()
    nsize, kind = int(p[1]), p[2].lower()
    mprop, rho, amin, amax = (float(x) for x in p[3:7])
    par = [float(x) for x in p[7:]]
    lna = np.linspace(np.log(amin), np.log(amax), nsize)
    a = np.exp(lna)
    if kind.startswith("plaw"):
        ava = a ** (4.0 + par[0])
        if "-ed" in kind:
            at, ac, gam = par[1:4]
            ava = np.where(a >= at, ava * np.exp(-(((a - at) / ac) ** gam)), ava)
    else:
        ava = a ** 3 * np.exp(-0.5 * ((lna - np.log(par[0])) / par[1]) ** 2)
    return lna, a, ava, rho, mprop


def test_j13_extinction_matches_dustem(themis_cm20):
    """The CM20 columns of DustEM's EXT_J13.RES, rebuilt from GRAIN_J13.DAT
    with DustEM's own size grid and trapezoid rule."""
    rows = [r for r in (themis_cm20 / "data" / "GRAIN_J13.DAT").read_text().splitlines()
            if r.strip() and not r.startswith("#")]
    pops = [_dustem_population(r) for r in rows if r.split()[0] == "CM20"]
    ext = np.array([r.split() for r in (themis_cm20 / "out" / "EXT_J13.RES").read_text().splitlines()
                    if r.strip() and not r.startswith("#")][1:], float)
    lam = ext[:, 0]
    sel = (lam >= 0.04) & (lam <= 1e3)
    xmp = 1.67262158e-24                          # DustEM's proton mass
    for k, (lna, a, ava, rho, mprop) in enumerate(pops):
        qabs, qsca, _ = th.efficiencies(a * 1e4, lam[sel])
        dn = ava / a ** 3                        # dn / d ln a
        mass = np.trapezoid(ava * rho * 4.0 * np.pi / 3.0, lna)
        for q, col in ((qabs, 1 + k), (qsca, 5 + k)):   # ABS(CM20 #k), SCA(CM20 #k)
            per_gram = np.trapezoid(q * np.pi * a ** 2 * dn, lna, axis=1) / mass
            ours = per_gram * mprop * xmp * 1e21
            np.testing.assert_allclose(ours, ext[sel, col], rtol=0.015)


def test_heat_capacity_reproduces_the_table_and_rises_with_temperature(themis_cm20):
    a_t, lT, lC = th.load_heat_capacity()
    ia = np.array([0, 41, 42, 97])                 # either side of the 20 nm core/mantle step
    T = 10.0 ** lT[[0, 12, 29]]
    C = th.heat_capacity(a_t[ia], T)
    V = 4.0 / 3.0 * np.pi * (a_t[ia] * 1e-4) ** 3
    np.testing.assert_allclose(C / V, 10.0 ** lC[np.ix_([0, 12, 29], ia)], rtol=1e-10)
    Cs = th.heat_capacity([1e-3, 0.1], np.geomspace(0.1, 5000.0, 50))
    assert np.all(np.diff(Cs, axis=0) > 0)
    with pytest.raises(ValueError, match="temperature"):
        th.heat_capacity([1e-3], [1e4])


def test_recomputed_large_grains_match_the_tables_where_they_are_sound(themis_cm20):
    from pycalima._datasets import get_dataset

    if get_dataset("optical-jones-2012-size").locate() is None:
        pytest.skip("Jones (2012) tables not fetched")
    a = np.array([0.03, 0.1, 0.3])
    lam = np.concatenate([np.geomspace(0.1, 5.0, 30), np.geomspace(15.0, 1000.0, 15)])  # outside the 6-12 um bands
    q, s, g = th.mie_efficiencies(a, lam)
    qt, st, gt = th.efficiencies(a, lam)
    np.testing.assert_allclose(q, qt, rtol=0.02)
    np.testing.assert_allclose(s, st, rtol=0.02)
    # where Q_CM20 is broken (Q_sca up to ~700 in the far UV above ~0.9 um) these stay physical
    q, s, g = th.mie_efficiencies([2.24, 5.0], [0.0912, 0.1, 0.12])
    assert np.all(q + s < 2.6) and np.all(q > 0.5)
