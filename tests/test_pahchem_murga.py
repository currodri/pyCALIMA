"""Murga et al. (2020, 2022): the public model outputs and the vector-figure targets.

Gates on the data themselves come first: the Figshare files are parsed into
the layouts the method cards record, and the figure extractions are checked
against statements of the papers and against the Figshare tables. The
Figshare data sets are fetched on demand (``calima-fetch-data fetch
murga2020-figshare``, ``murga2022-figshare``); their gates skip without them.
"""

import numpy as np
import pandas as pd
import pytest

from pycalima._paths import get_external_data_path
from pycalima.pahchem import murga_data as M


def _dataset(name):
    from pycalima._datasets import DatasetUnavailable, ensure_dataset

    try:
        return ensure_dataset(name)
    except DatasetUnavailable:
        pytest.skip(f"{name} not fetched (calima-fetch-data fetch {name})")


def _csv(name):
    return pd.read_csv(get_external_data_path(f"pahchem/{name}"), comment="#")


# --- the extracted figures ---------------------------------------------------

def test_fig4_orion_neutrals_at_av01_are_fully_destroyed():
    """M20 Sect. 4.1: every parent is destroyed at A_V = 0.1 in the Orion Bar, so the
    C2H2 per neutral molecule is N_C/2 (measured to 0.01 % for all 20)."""
    t = _csv("murga2020_fig4_c2h2_per_molecule.csv")
    t = t[(t.site == "orion_isochoric") & (t.av == 0.1) & (t.z == 0)]
    assert len(t) == 20
    assert np.allclose(t.n_c2h2, t.n_c / 2, rtol=1e-3)


def test_fig3_charges_are_integers_and_match_the_text():
    """M20 Sect. 3: in the Orion Bar 'large PAHs are ionized twice'; in the Horsehead
    'larger ones can have also a negative charge number in 0.1 < A_V < 2'."""
    t = _csv("murga2020_fig3_mean_charge.csv")
    orion = t[t.site == "orion_isochoric"].set_index(["molecule", "av"]).mean_z
    horse = t[t.site == "horsehead"].set_index(["molecule", "av"]).mean_z
    assert orion["C128H28", 0.1] == 2 and (orion.loc["C24H12"] == 0).all()
    assert horse["C66H20"].loc[0.1:1.5].eq(-1).all() and horse["C128H28"].loc[0.5:1.5].eq(-1).all()
    assert (horse.loc[(slice(None), slice(2.0, 5.0))] == 0).all()


def test_fig6_profile_reproduces_the_figshare_headers():
    """Each Figshare location, placed on the Fig. 6 distance axis by its chi, has the
    T_gas of Fig. 6 at that distance (median |dex| 0.001); distances increase."""
    p = _csv("murga2022_fig6_profile.csv")
    c = _csv("murga2022_fig6_curves.csv")
    t = c[c.quantity == "t_gas"].sort_values("distance_arcsec")
    model = np.interp(p.distance_arcsec, t.distance_arcsec, np.log10(t.value))
    assert len(p) == M.N_LOC and np.all(np.diff(p.distance_arcsec) >= 0)
    assert np.median(np.abs(np.log10(p.t_gas) - model)) < 0.01
    assert p.n_e.min() > 0 and p.loc[p.chi >= 3.99e4, "n_e"].min() > 1e3      # the HII region is ionised


# --- the Figshare data -------------------------------------------------------

def test_murga2020_profiles_and_rates_parse():
    root = _dataset("murga2020-figshare")
    p = M.read_profile_2020("orion_isochoric", root)
    av, t, rate = M.read_c2h2_rates_2020("orion_isochoric", root)
    assert p["av"].size == 11 and p["chi"][0] == pytest.approx(1.1e4)
    assert rate.shape == (11, 100) and t[0] == 1.0 and t[-1] == pytest.approx(1e6)
    # 'At A_V = 0.1 the PAH production stops after 10^4 yr' (Sect. 4.1)
    last = t[np.nonzero(rate[0] > 0)[0].max()]
    assert last == pytest.approx(1e4, rel=0.05)


def test_murga2022_model_b_layout():
    """The card's layout: 158 locations x 12 Z x 11 times x 5 X_H x 10 sizes; f(a, Z)
    independent of time and X_H and normalised over Z; the normal X_H bin
    (1.385) holds everything at t = 0."""
    _dataset("murga2022-figshare")
    d = M.read_model_2022("b")
    assert d["n"].shape == (M.N_LOC, M.N_Z, M.N_T, M.N_XH, M.N_SIZE)
    f = d["f"]
    assert np.allclose(f, f[:, :, :1, :1, :])
    assert np.allclose(f[:, :, 0, 0, :].sum(axis=1), 1.0, atol=1e-3)
    n0 = d["n"][:, :, 0]
    assert np.allclose(n0[:, :, [0, 1, 3, 4]], 0.0) and n0[:, :, 2].sum() > 0
    assert d["x_h"][2] == pytest.approx(1.385)
