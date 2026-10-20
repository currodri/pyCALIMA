"""Grain optical tables are per gram of dust, and what is derived from them stays per gram.

RAMSES reads averaged_cross_section_DustBin_NN.txt as cm^2 per gram of dust (it converts to per grain with the
bin's grain mass), and band_luminosity_DustBin_NN.txt as erg/s per gram. Grains much smaller than the wavelength
absorb in proportion to their mass, so per gram the far-IR emission of two graphite bins of different size must be
nearly the same; a per-grain/per-gram mix-up shows up as the ratio of their grain masses (1e3 here).
"""

from pathlib import Path

import numpy as np

from pycalima.models.grain_size_config import get_model_data_dir


def _band_table(path):
    lines = Path(path).read_text().splitlines()
    rows = [line.split() for line in lines if not line.startswith("#") and line.strip()]
    return np.array([[float(x) for x in r] for r in rows if len(r) > 2])


def test_exported_optics_and_band_luminosities_are_per_gram(isolated_env):
    from pycalima.models.dust_radiation.export_dust_band_luminosities import (
        export_band_luminosities,
    )
    from pycalima.models.dust_radiation.export_dust_optical_properties import (
        export_dust_optical_properties,
    )

    export_dust_optical_properties(cabs_method="mie")
    export_band_luminosities(num_temperatures=40, T_min=5.0, T_max=200.0)
    out = get_model_data_dir() / "optical_properties"

    header = (out / "averaged_cross_section_DustBin_01.txt").read_text()
    assert "C_abs[cm^2/g_dust]" in header

    small, large = (_band_table(out / f"band_luminosity_DustBin_0{i}.txt") for i in (1, 2))  # graphite 0.01, 0.1 um
    j = np.argmin(abs(small[:, 0] - np.log10(20.0)))  # T_dust = 20 K
    log_ratio = small[j, 1:] - large[j, 1:]  # log10 of the per-gram luminosity ratio, every band
    assert np.all(np.abs(log_ratio) < 0.5), log_ratio
    # and the specific luminosity is of the right order: 4 pi kappa B with kappa ~ 1e2 cm^2/g at 250 um, 20 K
    assert 0.0 < small[j, 1:].max() < 10.0
