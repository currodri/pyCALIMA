"""photo.PARTITION_NODES follow from the Hrodmarsson+25 photoelectron matrices.

Skip without ``calima-fetch-data fetch hrodmarsson2025-pes``.
"""
import numpy as np
import pytest

from pycalima.pahchem import hrodmarsson2025 as H
from pycalima.pahchem import photo


def _fetched(read):
    from pycalima._datasets import DatasetUnavailable

    try:
        return read()
    except DatasetUnavailable:
        pytest.skip("Hrodmarsson+25 data not fetched (calima-fetch-data fetch hrodmarsson2025-pes)")


def _shares():
    return _fetched(lambda: {n: H.mean_share(n) for n in H.IE if n not in H.DEPLETED})


def test_partition_nodes_are_the_measured_mean_shares():
    shares = _shares()
    for hv, node in photo.PARTITION_NODES[1:]:          # 13.6 eV is the join with the share below it
        med = np.median([np.median(s[np.abs(e - hv) <= 0.25]) for e, s in shares.values()])
        assert med == pytest.approx(node, abs=0.02), hv


def test_scaled_ion_yield_peaks_at_the_requested_value_per_carbon():
    hv, s = _fetched(lambda: H.ionisation_cross_section("Perylene", peak_mb_per_c=22.0))
    k = (hv > 15) & (hv < 19)
    assert s[k].max() / 1e-18 / H.IE["Perylene"][1] == pytest.approx(22.0)
