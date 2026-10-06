"""Maragkoudakis, Peeters & Ricca (2020) recovered from their stated inputs (M7a).

Gates 1-2 run their model (the thermal cascade) and recover the paper. Gate 3
runs pahchem's exact cascade on the same inputs and records the physical
difference. The 308 species come from PAHdb v3.20 ($CALIMA_PAHDB_DIR,
manual dataset ``pahdb-theoretical-v3-20``); without it these tests skip.
"""

import numpy as np
import pandas as pd
import pytest

from pycalima._paths import get_external_data_path
from pycalima.pahchem import maragkoudakis2020 as M

_CACHE = {}


def _species():
    from pycalima._datasets import DatasetUnavailable
    from pycalima.pahchem import ames_pahdb as pahdb

    if "species" not in _CACHE:
        try:
            path = pahdb.ascii_path()
        except DatasetUnavailable:
            pytest.skip("PAHdb v3.20 not installed (calima-fetch-data import pahdb-theoretical-v3-20 ...)")
        _CACHE["species"] = pahdb.read_ascii(path, M.sample().uid)
    return _CACHE["species"]


def _thermal():
    if "thermal" not in _CACHE:
        rows = []
        for uid, sp in _species().items():
            for field, b in M.band_fluxes(sp, "thermal").items():
                rows.append(dict(uid=uid, nc=sp.nc, charge=sp.charge, field=str(field), **b))
        _CACHE["thermal"] = pd.DataFrame(rows)
    return _CACHE["thermal"]


def _match_fig2(ours):
    """log10(ours / paper) for every N_C whose number of plotted points equals ours, per charge."""
    fig = pd.read_csv(get_external_data_path("pahchem/maragkoudakis2020_fig2.csv"), comment="#")
    out = {}
    for z, group in ((0, "neutral_11.2/3.3"), (1, "cation_CHoop/3.3"), (-1, "anion_CHoop/3.3")):
        f = fig[fig.group == group]
        dex = []
        for nc, g in f.groupby(f.nc.round().astype(int)):
            o = np.sort(ours[(ours.charge == z) & (ours.nc == nc)].ratio.to_numpy())
            if o.size == len(g):
                dex += list(np.log10(o / np.sort(g.ratio.to_numpy())))
        out[z] = np.array(dex)
    return out


def test_sample_is_their_tables_h1_h3():
    s = M.sample()
    assert len(s) == 308 and (s.charge == -1).sum() == 42
    assert (s.charge == 0).sum() == (s.charge == 1).sum() == 133
    assert s.nc.min() == 22 and s.nc.max() == 216


def test_fig2_is_recovered_molecule_by_molecule():
    """Their model (thermal cascade, Mathis ISRF x their cross section) against every
    marker of Fig. 2: 120 neutrals, 116 cations, 35 anions match in number per N_C.
    Measured median log10(ours/paper) +0.001 for each charge, 0.002-0.004 at most
    except one cation (N_C = 22, plotted at 2.23 where every C22 cation gives
    0.9-1.2; the figure also has a cation at N_C = 28, which is not in the sample)."""
    t = _thermal()
    t = t[t.field == "isrf"].assign(ratio=lambda d: d["11.2"] / d["3.3"])
    dex = _match_fig2(t)
    assert len(dex[0]) >= 120 and len(dex[1]) >= 116 and len(dex[-1]) == 35
    for z, d in dex.items():
        assert abs(np.median(d)) < 0.003, z
        assert np.mean(np.abs(d) < 0.01) > 0.99, z


@pytest.mark.parametrize("energy", [6.0, 8.0, 10.0, 12.0])
def test_fixed_energy_size_fits_are_recovered(energy):
    """Their eq. 3 fits (Tables F9-F12) against our per-molecule 11.2/3.3 (neutral) and
    11.0/3.3 (cation) at N_C >= 50: measured median residual -0.007 to -0.028 dex. The
    fits are of binned data, so the scatter (~0.05 dex) is theirs."""
    t = _thermal()
    t = t[(t.field == str(energy)) & (t.nc >= 50)]
    for z, name in ((0, "Neutral"), (1, "Cation")):
        g = t[t.charge == z]
        res = np.log10(g["11.2"] / g["3.3"]) - M.size_fit(g.nc, name, energy)
        assert abs(np.median(res)) < 0.035, (z, np.median(res))


def test_exact_cascade_raises_11p2_over_3p3_by_a_few_percent():
    """pahchem's microcanonical cascade on the same inputs, for 12 species across the
    sample: under the ISRF, 11.2/3.3 is 0.016-0.020 dex (4-5 %) above the thermal
    model (all 308: median +0.016 neutral, +0.020 cation, +0.019 anion). The thermal
    approximation over-emits at 3.3 um from the modest energies that dominate ISRF
    heating. At N_C ~ 100 this moves the eq. 3 size by ~2-3 C."""
    species = _species()
    uids = M.sample().sort_values("nc").uid.to_numpy()[::26]
    for uid in uids:
        sp = species[int(uid)]
        th = M.band_fluxes(sp, "thermal", energies_ev=())["isrf"]
        ex = M.band_fluxes(sp, "exact", energies_ev=())["isrf"]
        d = np.log10(ex["11.2"] / ex["3.3"]) - np.log10(th["11.2"] / th["3.3"])
        assert 0.0 < d < 0.05, (uid, sp.formula, d)
