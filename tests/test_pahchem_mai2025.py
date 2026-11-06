"""Mai+25 ML-MD spectra reader (pahchem.mai2025): the band measure on a synthetic band, and the size trend of
the CH-stretch shift in their data (skipped until ``calima-fetch-data fetch mai2025-mlmd``)."""

import numpy as np
import pytest

from pycalima.pahchem import mai2025


def test_ch_band_of_a_gaussian():
    nu = np.arange(2900.0, 3200.0, 1.0)
    y = np.exp(-0.5 * ((nu - 3060.0) / 8.0) ** 2)
    peak, centroid, red, blue = mai2025.ch_band(nu, y, smooth_cm=0.5)
    sigma = np.hypot(8.0, 0.5)
    assert peak == pytest.approx(3060.0, abs=1.0) and centroid == pytest.approx(3060.0, abs=0.5)
    assert red == pytest.approx(blue, abs=1.0) and red + blue == pytest.approx(2.3548 * sigma, abs=2.0)


def test_large_pahs_shift_their_ch_peak_further():
    from pycalima._datasets import get_dataset

    if get_dataset(mai2025.DATASET).locate() is None:
        pytest.skip("mai2025-mlmd not fetched")
    peak = {}
    for T in (50, 600):
        for name, nc, nh, uid, nu, inten in mai2025.spectra(T):
            if 24 <= nc < 40 or nc >= 60:
                peak.setdefault(name, {"nc": nc})[T] = mai2025.ch_band(nu, inten)[0]
    shift = lambda sel: np.median([v[600] - v[50] for v in peak.values() if 50 in v and 600 in v and sel(v["nc"])])  # noqa: E731
    small, large = shift(lambda n: n < 40), shift(lambda n: n >= 60)
    assert small < 0 and large < 1.25 * small
