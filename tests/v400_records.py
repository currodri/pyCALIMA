"""PAHdb v4.00's records (``ames_suite.xml_index``), parsed once per test session for every module that needs them: the
parse takes ~33 s, and each module used to repeat it."""

import pytest

_RECORDS = {}


def records():
    """The v4.00 records, or a skip when the dataset is not installed (no cache_dir: tests do not write to model data)."""
    if "v400" not in _RECORDS:
        from pycalima._datasets import DatasetUnavailable
        from pycalima.pahchem import ames_suite

        try:
            path = ames_suite.xml_path("pahdb-theoretical-v4-00")
        except DatasetUnavailable:
            pytest.skip("PAHdb v4.00 not installed")
        _RECORDS["v400"] = ames_suite.xml_index(path)
    return _RECORDS["v400"]
