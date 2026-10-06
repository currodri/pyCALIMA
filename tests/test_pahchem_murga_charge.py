"""The WD01 charge of Murga+2020/2022 (pahchem.murga_charge) against their charges.

Gate 1 compares f(a, Z) with the Murga+2022 Figshare tables at every location
and size, where chi, T and n_e are tabulated per location; it fixes the
choices the papers do not state (module docstring). Gate 2 is the integer
<Z> of Murga+2020 Fig. 3. Gate 3 holds invariants. Measured numbers are in
diagnostics/pahchem/check_murga_charge.py.
"""

import numpy as np
import pytest

from pycalima.pahchem import murga_charge as C
from pycalima.pahchem import murga_data as M


def _dataset(name):
    from pycalima._datasets import DatasetUnavailable, ensure_dataset

    try:
        return ensure_dataset(name)
    except DatasetUnavailable:
        pytest.skip(f"{name} not fetched (calima-fetch-data fetch {name})")


_TABLES = {}


def _m22(name="MURGA2022", **overrides):
    import sys
    from pathlib import Path

    _dataset("murga2022-figshare")
    if name not in _TABLES:
        sys.path.insert(0, str(Path(__file__).parents[1] / "diagnostics" / "pahchem"))
        from check_murga_charge import m22_table

        _TABLES[name] = m22_table(overrides)
    return _TABLES[name]


# --- gate 3: invariants -------------------------------------------------------

def test_f_is_normalised_and_respects_wd01_zmin():
    for a in (3.3e-8, 6e-8, 8.7e-8, 2e-7):
        Z, f = C.f_wd01(a, 1e3, 300.0, 1.0, **{k: v for k, v in C.MURGA2022.items() if k != "fold"})
        assert f.sum() == pytest.approx(1.0) and np.all(f >= 0)
        assert Z[0] == C.z_min_wd01(a)
    # eq. 24 as printed: no anions below 6 A, Z >= -1 to 7.8 A, -2 at 8.7 A (as in the Figshare files)
    assert [int(C.z_min_wd01(a * 1e-8)) for a in (3.3, 5.0, 6.0, 7.8, 8.7)] == [0, 0, -1, -1, -2]


@pytest.mark.parametrize("a", [5e-8, 1.2e-7])
def test_mean_charge_rises_with_chi_at_fixed_t_and_ne(a):
    z = [C.mean_and_mode(*C.f_murga2022(a, chi, 300.0, 1.0))[0] for chi in (1e-2, 1e0, 1e2, 1e4)]
    assert np.all(np.diff(z) > 0), z


def test_deep_cloud_is_solved_without_the_solver_overflow():
    """chi = 2e-6, T = 13 K, n_e = 3e-3 (M22 deep MC): 8.7 A grains are Z = -1, as
    in the Figshare file (pyCALIMA's bounds finder returns Z = -2 there)."""
    Z, f = C.f_murga2022(8.7e-8, 2e-6, 13.0, 2.9e-3)
    assert Z[np.argmax(f)] == -1 and f[Z == -1][0] > 0.99


# --- gate 1: Murga+2022 Figshare f(a, Z) ---------------------------------------

@pytest.mark.parametrize("region, max_df, max_dz", [("PDR 1", 0.15, 0.06), ("PDR 2", 0.06, 0.04),
                                                     ("DF", 0.04, 0.01), ("MC", 0.04, 0.01)])
def test_m22_charge_fractions(region, max_df, max_dz):
    """MURGA2022 (MMP83, s_e = 1, no ions, eq. 24, truncated grid) against the
    Figshare f(a, Z), 10 sizes x every location of the region. Measured max
    abs(df) 0.13 / 0.05 / 0.03 / 0.03 and median dZ -0.04 / -0.03 / -0.00 /
    -0.00 (PDR 1 / PDR 2 / DF / MC); modal Z agrees at 92-100 %."""
    t = _m22()
    t = t[t.region == region]
    assert t.max_df.max() < max_df
    assert abs(t.dz.median()) < max_dz
    assert (t.mode_ours == t.mode_murga).mean() > 0.9


def test_m22_requires_unit_electron_sticking():
    """The WD01 sticking law instead of s_e = 1 puts <Z> +0.47 (PDR 1) and +0.77
    (PDR 2) too high: s_e is the decisive unstated input."""
    t = _m22("wd01", sticking="wd01")
    assert t[t.region == "PDR 1"].dz.median() > 0.3 and t[t.region == "PDR 2"].dz.median() > 0.5


@pytest.mark.xfail(strict=True, reason=(
    "M22 HII points without EUV: <Z> -0.97 median below the Figshare f. M22 add photons above 13.6 eV "
    "'with photon flux corresponding to an O6-type star' there (Sect. 3.2), spectrum and normalisation "
    "not stated; with them the HII f is reproduced (test_m22_hii_region_with_o6_euv)."))
def test_m22_hii_region():
    t = _m22()
    assert abs(t[t.region == "HII"].dz.median()) < 0.1


def test_m22_hii_region_with_o6_euv():
    """With the EUV of a 40 000 K Kurucz atmosphere above 13.6 eV, matched to the field
    at 6-13.6 eV and scaled by EUV_FACTOR_HII = 4 (the one inferred number), every size
    up to 14.2 A has median <Z> within 0.06 of the Figshare f in the HII region
    (19.9 A +0.20, 50 A +0.52; max |df| 0.11). One factor fitting all sizes is the
    test: the sizes respond to the EUV very differently. A lower n_e cannot do it
    (n_e/8 leaves 19.9 A at -1.1 while n_e/16 puts 6-12 A at +0.3-0.4)."""
    import pandas as pd

    from pycalima._paths import get_external_data_path

    _dataset("murga2022-figshare")
    d = M.read_model_2022("b")
    p = pd.read_csv(get_external_data_path("pahchem/murga2022_fig6_profile.csv"), comment="#")
    order = np.argsort(d["z"])
    zg = d["z"][order].astype(int)
    hii = np.where(p.chi.to_numpy() >= 3.99e4)[0]
    for m, a in enumerate(d["a_cm"]):
        if a > 1.5e-7:
            continue
        dz = []
        for loc in hii:
            Z, f = C.f_wd01(a, p.chi[loc], p.t_gas[loc], p.n_e[loc], z_range=C.Z_GRID_2022,
                            euv_factor=C.EUV_FACTOR_HII, **C.MURGA2022)
            fm = d["f"][loc, :, 0, 0, m][order]
            dz.append(float(Z @ f - zg @ fm))
            assert np.abs(f - fm).max() < 0.15
        assert abs(np.median(dz)) < 0.1, (a, np.median(dz))


# --- gate 2: Murga+2020 Fig. 3 --------------------------------------------------

def _fig3(**kw):
    import sys
    from pathlib import Path

    _dataset("murga2020-figshare")
    sys.path.insert(0, str(Path(__file__).parents[1] / "diagnostics" / "pahchem"))
    from check_murga_charge import m20_fig3

    res = m20_fig3(**kw)
    return (sum(int((r == t).sum()) for t, r, _ in res.values()),
            sum(int((m == t).sum()) for t, _, m in res.values()))


def test_m20_fig3_regression():
    """Integer <Z> of C24H12, C66H20, C128H28 at 11 A_V, Orion and Horsehead (66
    points). With the M22 configuration, measured 48 (rounded mean) and 47 (mode);
    with Z_min rounded down, s_e = 0.5 and C+, 57 and 57. Rounded mean and mode
    are indistinguishable."""
    assert _fig3() == (48, 47)
    assert min(_fig3(sticking=0.5, ions=True, z_min="floor")) >= 57


@pytest.mark.xfail(strict=True, reason=(
    "M20 Fig. 3 is not reproduced with M22's configuration (48/66): the Horsehead C66H20 anion needs "
    "Z_min rounded down, which the M22 files exclude, so M20 and M22 ran different charge conventions; "
    "no variant tried gives Orion C128H28 +2 at A_V = 0.1 or +1 at A_V = 5 (best 57/66)."))
def test_m20_fig3_agreement():
    assert min(_fig3(sticking=0.5, ions=True, z_min="floor")) >= 63
