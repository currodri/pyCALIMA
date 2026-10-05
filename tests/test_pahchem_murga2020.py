"""Murga+2020 per-molecule evolution (pahchem.murga2020) against Figs. 4-5.

Gates, from the cleanest to the most composite:
- the H-addition inputs (Goumans 2011 per site, Fig. 1 ion rates);
- the integrator: the stiff solver agrees with M20's explicit Euler where
  Euler is stable;
- Fig. 4 at Orion A_V = 0.1, Z = 0: full destruction (N_C/2 C2H2 per
  molecule) for the 20 parents, and the destruction time against the end of
  PAH C2H2 production in the Figshare rates (1e4 yr);
- the partial-destruction points of Fig. 4, which test the rate magnitudes.

The rate tables need the Figshare profiles (dataset murga2020-figshare).
"""

import numpy as np
import pandas as pd
import pytest

from pycalima._paths import get_external_data_path
from pycalima.pahchem import murga2020 as T


@pytest.fixture(scope="module")
def profiles():
    from pycalima._datasets import DatasetUnavailable, ensure_dataset

    try:
        ensure_dataset("murga2020-figshare")
    except DatasetUnavailable:
        pytest.skip("murga2020-figshare not fetched (calima-fetch-data fetch murga2020-figshare)")


def _fig4(site, av, z, parent):
    t = pd.read_csv(get_external_data_path("pahchem/murga2020_fig4_c2h2_per_molecule.csv"), comment="#")
    return float(t[(t.site == site) & (t.av == av) & (t.z == z) & (t.molecule == parent)].n_c2h2.iloc[0])


# --- inputs -----------------------------------------------------------------

def test_neutral_h_addition_follows_goumans_edge_and_core_sites():
    """M20 Sect. 2.2: 0.5 (N_edge k_edge + N_core k_core). For coronene at 100 K,
    edge sites dominate (k_edge / k_core > 1e4), and the rate 'sharply
    increases with temperature, when the temperature exceeds ~200 K'."""
    k = lambda T_: T.h_addition_neutral(24.0, 24, 12, T_)
    assert T.goumans_rate("1", 100.0) / T.goumans_rate("3a1", 100.0) > 1e4
    assert k(400.0) / k(200.0) > 10.0 and k(200.0) / k(100.0) > 3.0


def test_goumans_rates_are_held_below_40_k():
    """Goumans 2011: the fits must not be used below 40 K (the low-T fit diverges)."""
    assert T.goumans_rate("1", 15.0) == T.goumans_rate("1", 40.0)
    assert T.goumans_rate("1", 15.0, "extrapolate") > 1e3 * T.goumans_rate("1", 40.0)


def test_ion_h_addition_regimes():
    """Fig. 1: normal 1.4e-10 N_H/N_C, dehydrogenated 0.5 (1.4e-10 N_H/N_C + 5e-11),
    super-H 1e-12 cm^3 s^-1."""
    assert T.h_addition_ion(24.0, 12.0, 12.0) == pytest.approx(1.4e-10 * 0.5)
    assert T.h_addition_ion(24.0, 6.0, 12.0) == pytest.approx(0.5 * (1.4e-10 * 0.25 + 5e-11))
    assert T.h_addition_ion(24.0, 16.0, 12.0) == pytest.approx(1e-12)


# --- the integrator -----------------------------------------------------------

def test_stiff_solver_matches_m20_explicit_euler_where_euler_is_stable(profiles):
    """M20 integrate eq. 1 with explicit Euler, dt = 1 yr. Where every rate is << 1
    yr^-1 (Horsehead A_V = 2, C24H12, Z = 0) Euler and the stiff solver agree:
    measured 1e-8 relative in N_C2H2 after 1e5 yr. (Where H exchange is fast,
    ~1e4 yr^-1 at Orion A_V = 0.1, Euler with dt = 1 yr is unstable; M20's
    statement that a smaller dt changes nothing implies slower rates there.)"""
    env = T.environment("horsehead", 2.0)
    y = T.euler("C24H12", env, 0, t_end_yr=2e4, dt_yr=1.0)
    tr = T.evolve("C24H12", env, 0, t_end_yr=2e4)
    assert tr.total_c2h2 == pytest.approx(y[2], rel=1e-4)
    assert tr.nh[-1] == pytest.approx(y[1], rel=1e-6)


# --- Fig. 4: full destruction in the Orion Bar ----------------------------------

@pytest.fixture(scope="module")
def orion_av01_neutrals(profiles):
    env = T.environment("orion_isochoric", 0.1)
    return {p: T.evolve(p, env, 0) for p in T.PARENTS}


def test_fig4_orion_av01_neutrals_are_fully_destroyed(orion_av01_neutrals):
    """M20 Fig. 4 / Sect. 4.1: every parent is destroyed at A_V = 0.1 (chi = 1.1e4,
    T = 4400 K, n(H) = 2e5): N_C2H2 = N_C/2 for all 20. Measured to 0.1 %: the
    neutrals are super-hydrogenated by Goumans addition (~2e4 yr^-1) and lose
    carbon by the 2.0 eV super-H C2H2 channel."""
    for p, tr in orion_av01_neutrals.items():
        assert tr.destroyed_at_yr is not None, p
        assert tr.total_c2h2 == pytest.approx(T.parse(p)[0] / 2.0, rel=1e-3), p


def test_orion_destruction_time_against_the_end_of_c2h2_production(orion_av01_neutrals):
    """The Figshare PAH C2H2 rate at Orion A_V = 0.1 (WD01-weighted over 4 bins up to
    C150H30) is last non-zero at 1.0e4 yr (M20 Sect. 4.1: 'production stops after
    1e4 yr'), so the slowest parent dies at ~1e4 yr. With DL01 T(E): C150H30
    neutral at 1.6e3 yr (-0.8 dex); with Tielens T(E) it would be 24 yr (-2.6
    dex), which this gate rejects."""
    slowest = max(tr.destroyed_at_yr for tr in orion_av01_neutrals.values())
    assert abs(np.log10(slowest / 1.0e4)) < 1.0


def test_tielens_temperature_destroys_c150_too_fast(profiles):
    env = T.environment("orion_isochoric", 0.1)
    tr = T.evolve("C150H30", env, 0, T.Options(temperature="tielens"))
    assert tr.destroyed_at_yr < 1.0e2          # measured 24 yr: 2.6 dex before the Figshare 1e4 yr


# --- Fig. 4: partial destruction --------------------------------------------------

@pytest.mark.xfail(strict=True, reason=(
    "Where destruction is partial (Orion A_V = 2, Horsehead, Orion cations) our C2H2 per molecule is "
    "1-4 dex above Fig. 4 for every combination of the open choices tried (T(E), p(E), super-H threshold, "
    "ion H2 addition, channel competition): the real-valued N_H drifts above X_H0 under any H addition and "
    "opens the 2.0 eV super-H C2H2 channel. With the super-H path switched off the normal-state rates alone "
    "are 1-2 dex high for C16-C24 and 8-20 dex low for N_C >= 66, so Fig. 4's nearly size-independent "
    "1e-4-1e-3 for large PAHs needs a super-H contribution, 1e2-1e4 x smaller than ours "
    "(check_murga2020.py). The threshold handling and the p(E) tail of SHIVA's Monte Carlo are not published."))
@pytest.mark.parametrize("site, av, z, parent", [("horsehead", 0.1, 0, "C24H12"), ("orion_isochoric", 2.0, 0, "C66H20"),
                                                 ("orion_isochoric", 0.1, 1, "C24H12")])
def test_fig4_partial_destruction(profiles, site, av, z, parent):
    tr = T.evolve(parent, T.environment(site, av), z)
    assert abs(np.log10(tr.total_c2h2 / _fig4(site, av, z, parent))) < 1.0
