"""The cation -> dication ionisation of pahchem against Wenzel et al. (2020).

Wenzel+20 measured absolute ionisation cross sections sigma_I of C30H14+,
C32H14+, C36H18+ and C48H20+ at SOLEIL, and fitted a yield law (eqs. 4-5)
to them. The data are their Zenodo record (wenzel2020_cation_ionisation.csv);
the fit curves are read from their vector Fig. 9 (wenzel2020_fig9_yield_fit.csv).
"""

import numpy as np
import pandas as pd
import pytest

from pycalima._paths import get_external_data_path
from pycalima.pahchem import fields, photo


def _csv(name):
    return pd.read_csv(get_external_data_path(f"pahchem/{name}"), comment="#")


def test_eq4_reproduces_their_fig9():
    """Every vertex of the three Fig. 9 Eq. (4) curves (N_C = 34, 48, 60), with
    IP2 read from the curve's first vertex, to 2e-3."""
    t = _csv("wenzel2020_fig9_yield_fit.csv")
    for nc, g in t[t.curve == "wenzel_eq4"].groupby("nc"):
        ip2 = g.hnu_ev.iloc[0]
        assert photo.yield_wenzel2020(g.hnu_ev.to_numpy() + 1e-6, ip2, nc) == pytest.approx(g.y.to_numpy(), abs=2e-3)


def test_jochims_ramp_is_the_one_they_compare_with():
    """Their Jochims+96 curves, shifted to IP2 (the Andrews+16 recipe), run from 0
    at IP2 to 1 at IP2 + 9.2 eV, as photo.yield_jochims1996."""
    t = _csv("wenzel2020_fig9_yield_fit.csv")
    for _, g in t[t.curve == "jochims96_shifted"].groupby("nc"):
        x, y = g.hnu_ev.to_numpy(), g.y.to_numpy()
        assert photo.yield_jochims1996(x, x[0]) == pytest.approx(y, abs=2e-3)


def test_beta_is_the_20ev_branching_ratio():
    """eq. 5 beta against the measured BR = I/(I+F) at 20 eV (their definition), to 0.03."""
    t = _csv("wenzel2020_cation_ionisation.csv")
    for nc, g in t[t.hnu_ev == 20.0].groupby("nc"):
        assert photo.yield_wenzel2020(20.0, g.ip2_ev.iloc[0], nc) == pytest.approx(g.br.iloc[0], abs=0.03)


def test_eq4_domain():
    with pytest.raises(ValueError):
        photo.yield_wenzel2020(12.0, 9.5, 24)
    with pytest.raises(ValueError):
        photo.yield_wenzel2020(12.0, 11.5, 54)


def _rates(field):
    t = _csv("wenzel2020_cation_ionisation.csv").dropna(subset=["sigma_i_per_c"])
    out = {}
    for nc, g in t[t.hnu_ev <= 13.6].groupby("nc"):
        E, ip2 = g.hnu_ev.to_numpy(), g.ip2_ev.iloc[0]
        sld = photo.sigma_li_draine(nc, 1, E)
        k = lambda s: np.trapezoid(s * field.flux_at(E), E)
        meas = k(g.sigma_i_per_c.to_numpy() * 1e-18 * nc)
        wen = k(sld * photo.yield_wenzel2020(E, ip2, nc)) / meas if nc >= 32 else None
        out[nc] = (k(sld * photo.yield_jochims1996(E, ip2)) / meas, wen)
    return out


@pytest.mark.parametrize("field", [fields.draine1978(g0=1.0), fields.kurucz(15000, 1.0), fields.kurucz(40000, 1.0)],
                         ids=["Draine78", "Kurucz15kK", "Kurucz40kK"])
def test_cation_ionisation_rate_against_measured_sigma_I(field):
    """k_ion = int sigma Y F dE (9.5-13.6 eV) with the Li & Draine ionised sigma,
    against int sigma_I F dE from the measured sigma_I. Measured ratios: Jochims
    0.56-0.97 (0.14 dex rms), Wenzel eq. 4 0.66-1.30 (0.09 dex rms), over the
    three fields. LD01 sigma is 1.1-1.5x their TD-DFT sigma_abs here, and eq. 4
    integrates 8-18 % below their Fig. 8 yields under 13.6 eV, so the agreement
    is partly compensating. Jochims (the neutral law) is low for C36H18+."""
    for nc, (joch, wen) in _rates(field).items():
        assert abs(np.log10(joch)) < 0.27, (nc, joch)
        if wen is not None:
            assert abs(np.log10(wen)) < 0.2, (nc, wen)
