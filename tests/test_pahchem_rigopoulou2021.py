"""Rigopoulou et al. (2021) Fig. 1 recovered from its stated inputs (M7a).

PAHdb UID 742 (C210H36) after a 5 and a 12.4 eV photon, against the curves read
from the vector figure. Needs PAHdb v3.20 ($CALIMA_PAHDB_DIR); skips otherwise.
"""

import numpy as np
import pytest

from pycalima.pahchem import rigopoulou2021 as R


@pytest.fixture(scope="module")
def uid742():
    from pycalima._datasets import DatasetUnavailable
    from pycalima.pahchem import ames_pahdb as pahdb

    try:
        path = pahdb.ascii_path()
    except DatasetUnavailable:
        pytest.skip("PAHdb v3.20 not installed (calima-fetch-data import pahdb-theoretical-v3-20 ...)")
    return pahdb.read_ascii(path, [R.FIG1_UID])[R.FIG1_UID]


def _rms(peaks):
    d = R.peak_deviation(peaks)
    return np.sqrt(np.mean(d**2)), np.abs(d).max()


def test_fig1_is_the_thermal_cascade_with_15_cm_bands(uid742):
    """The 12 band peaks (3.3-12.7 um, both curves) after one common scale: measured
    1.6 % rms (4.9 % max) with FWHM 15 cm^-1, 12.8 % with the 30 cm^-1 the text
    states. The 12.4/5 eV peak ratios are 4.51, 3.47, 2.74, 2.12, 1.82 at 6.2-12.7 um
    against 4.49, 3.45, 2.73, 2.12, 1.83 in the figure."""
    assert uid742.formula == "C210H36" and uid742.charge == 0
    rms15, max15 = _rms(R.fig1_peaks(uid742, 15.0))
    rms30, _ = _rms(R.fig1_peaks(uid742, 30.0))
    assert rms15 < 0.025 and max15 < 0.06
    assert rms30 > 0.10


def test_exact_cascade_on_fig1(uid742):
    """pahchem's microcanonical cascade: 4.8 % rms. The difference is the 3.3 um band
    at 5 eV, which the exact cascade puts lower (12.4/5 eV ratio 38 against 34 thermal),
    as it does for every molecule at low energy (test_pahchem_emission.py)."""
    th = R.fig1_peaks(uid742, 15.0, "thermal")
    ex = R.fig1_peaks(uid742, 15.0, "exact")
    assert _rms(ex)[0] < 0.06
    assert ex["5eV"]["3.3"] / ex["5eV"]["11.3"] < 0.95 * th["5eV"]["3.3"] / th["5eV"]["11.3"]
