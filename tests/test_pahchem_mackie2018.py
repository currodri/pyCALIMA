"""Anharmonic cascade spectra after Mackie+18 (pahchem.mackie2018): the shift distribution's moments, the
polyad's uncoupled limit, and 6H-pyrene's CH stretches against their supplement (skipped without the data)."""

import numpy as np
import pandas as pd
import pytest

from pycalima import _paths
from pycalima.pahchem import mackie2018 as M


def test_shift_distribution_has_the_bose_moments():
    omega = np.array([300.0, 800.0, 1500.0])
    rows = np.array([[-1.5, -4.0, -9.0], [0.5, -2.0, 0.0]])
    T = 900.0
    g, d = M.shift_distribution(rows, omega, T, step_cm=0.25, n=4096, smooth_cm=0.5)
    q = np.exp(-omega / (0.6950348 * T))
    nbar = q / (1 - q)
    for r, dens in zip(rows, d):
        mean = np.sum(g * dens) * 0.25
        var = np.sum((g - mean) ** 2 * dens) * 0.25
        assert mean == pytest.approx(r @ nbar, abs=1e-3)
        assert var == pytest.approx((r**2) @ (nbar * (nbar + 1)) + 0.5**2, rel=1e-3)


def test_an_uncoupled_polyad_keeps_its_states():
    poly = M.Polyad([("1", 0), ("11", 1, 2)], np.array([3000.0, 2950.0]), np.zeros((2, 2)), np.array([0, -1]))
    chi = np.zeros((3, 3))
    e, shares, var = M.polyad_lines(poly, chi, np.array([3100.0, 1500.0, 1450.0]), 500.0)
    assert sorted(e) == pytest.approx([2950.0, 3000.0])
    assert shares[np.argmax(e), 0] == pytest.approx(1.0) and shares[np.argmin(e), 0] == pytest.approx(0.0)


def test_6h_pyrene_ch_stretches_are_mackie18s_cascade():
    """Regression gate at the 2026-10-21 level: rms of the peak-normalised 2800-3150 cm^-1 spectrum."""
    from pycalima._datasets import get_dataset
    from pycalima.pahchem import vpt2

    if get_dataset("mackie2018-supplement").locate() is None or get_dataset("pahdb-theoretical-v4-00").locate() is None:
        pytest.skip("needs the Mackie+18 supplement and PAHdb v4.00 (intensities)")
    d = pd.read_csv(_paths.get_external_data_path("pahchem", "mackie2018_ch_cascades.csv"), comment="#")
    ff = vpt2.mackie2018_force_field("123678-hexahydropyrene")
    nu = vpt2.fundamentals(ff.omega, vpt2.anharmonic_constants(ff))
    inten = M.intensities(ff, nu, 336)
    grid = np.linspace(2800, 3150, 1401)
    for ev, limit in ((3, 0.13), (9, 0.15)):
        s = M.run(ff, float(ev), inten).spectrum_distribution(grid)
        m = d[(d.molecule == "123678-hexahydropyrene") & (d.e_uv_ev == ev)]
        assert np.sqrt(np.mean((np.interp(m.nu_cm, grid, s / s.max()) - m.intensity) ** 2)) < limit, ev
