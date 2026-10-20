"""Demyk+26 single-photon emission (pahchem.demyk2026) against their Figs. 5, 10 and 11 (digitised, bundled).

The figure gates need the CosmicPAH-IRDB laws (``calima-fetch-data fetch cosmicpah-irdb``) and skip without them.
"""

import numpy as np
import pandas as pd
import pytest

from pycalima import _paths
from pycalima.pahchem import demyk2026 as D, dos


def _csv(name):
    return pd.read_csv(_paths.get_external_data_path("pahchem", name), comment="#")


@pytest.fixture(scope="module")
def laws():
    from pycalima._datasets import get_dataset
    from pycalima.pahchem import cosmicpah

    if get_dataset(cosmicpah.DATASET).locate() is None:
        pytest.skip("CosmicPAH-IRDB not fetched (calima-fetch-data fetch cosmicpah-irdb)")


def test_every_band_gets_its_own_mode_of_its_class():
    for key in ("6h-pyrene", "me-pyrene"):
        f = D.modes(key)
        t5 = D.table5(key)
        a, index = D.assign_intensities(f, t5.band_cm.to_numpy(), t5.intensity_km_mol.to_numpy())
        assert len(set(index)) == len(t5) and np.count_nonzero(a) == len(t5)
        assert np.all((f[index] > D.CH_STRETCH_CM) == (t5.band_cm > D.CH_STRETCH_CM))
        # the strong bands sit on their fundamentals; Me-pyrene's weak 2956.9 and 2908.4 cm^-1 bands (Fermi
        # resonances: six CH3 bands, three CH3 stretch modes) take aromatic CH stretch modes 90-140 cm^-1 away
        strong = t5.intensity_km_mol.to_numpy() > 15.0
        assert np.all(np.abs(f[index] - t5.band_cm)[strong] < 30.0)


def test_mode_lists_have_the_molecules_dof():
    for key, atoms in (("pyrene", 26), ("6h-pyrene", 32), ("me-pyrene", 29)):
        f = D.modes(key)
        assert f.size == 3 * atoms - 6 and f.min() > 50.0


def test_cascade_radiates_the_photon_energy(laws):
    em = D.run("6h-pyrene", 4.0)
    radiated = sum(em.energy(b.band_cm) for b in em.bands) / (4.0 * dos.CM_PER_EV)
    assert 0.95 < radiated < 1.0


@pytest.mark.parametrize("key,peak_um", [("6h-pyrene", 3.404), ("me-pyrene", 3.288)])
@pytest.mark.parametrize("ev", [4.0, 6.0])
def test_3um_profiles_are_their_fig5(laws, key, peak_um, ev):
    """Main CH band within 1.5 cm^-1 in position and width, and the 3 um spectrum within 8 % of the peak,
    with one scale per molecule (the 4 eV peak; their units are arbitrary)."""
    d = _csv("demyk2026_fig5_insets.csv")
    lam = np.linspace(3.12, 3.58, 4601)
    nu = 1e4 / lam
    f = {e: D.run(key, e).spectrum(nu) * nu**2 / 1e4 for e in {4.0, ev}}
    th4 = d[(d.molecule == key) & (d.e_uv_ev == 4.0)]
    th = d[(d.molecule == key) & (d.e_uv_ev == ev)]
    near = np.abs(lam - peak_um) < 0.01
    s = th4.f_lambda_au[np.abs(th4.lam_um - peak_um) < 0.01].max() / f[4.0][near].max()
    ours = s * np.interp(th.lam_um, lam, f[ev])
    assert np.max(np.abs(ours - th.f_lambda_au)) < 0.08 * th.f_lambda_au.max()
    i = np.argmax(np.where(np.abs(th.lam_um - peak_um) < 0.02, th.f_lambda_au, 0))
    j = np.argmax(np.where(near, f[ev], 0))
    assert 1e4 / lam[j] == pytest.approx(1e4 / th.lam_um.iloc[i], abs=1.5 + 0.5 * 1e4 * 0.002 / peak_um**2)


def test_34um_photon_temperatures_are_their_fig11(laws):
    h = _csv("demyk2026_fig11.csv")
    em = D.run("6h-pyrene", 4.0)
    b = em.band(2942.7)
    assert b.photons.sum() == pytest.approx(h.counts.sum() / 1e4, rel=0.02)
    edges = np.arange(300, 1301, 100)
    ours, _ = np.histogram(em.temperature_k, bins=edges, weights=b.photons)
    ours = dict(zip(edges[:-1], 1e4 * ours))
    for row in h.itertuples():
        lo = int(round(row.t_lo_k, -2))
        if row.counts > 1000:
            assert ours[lo] == pytest.approx(row.counts, rel=0.10), lo


def test_temperature_is_their_fig10_harmonic_curve():
    t = _csv("demyk2026_fig10.csv")
    u = t.u_ev.to_numpy()[t.u_ev >= 1.0]
    ours = D.temperature(D.modes("6h-pyrene"), u * dos.CM_PER_EV)
    assert np.max(np.abs(ours / t.t_harmonic_k[t.u_ev >= 1.0].to_numpy() - 1)) < 0.01


@pytest.mark.xfail(strict=True, reason=(
    "D26 Sect. 4.3: 6.9/3.3 um = 1.08 (6H-pyrene, 4 eV) and 0.19 (Me-pyrene, 6 eV); pahchem gives 0.90 and 0.16 "
    "from band energies. The 3 um profiles, the 6.27 and 7.92 um peaks and the 3.4 um photon temperatures match "
    "their figures, so the cascade is not the cause: their 6.9 um (CH2/CH3 deformation) band carries 15-20 % more "
    "energy, and for 6H-pyrene is wider (FWHM ~33 vs 24 cm^-1). Likely how they spread the measured intensity "
    "over the CH2 scissoring modes (mode lists 'adjusted', not published)."))
def test_69_33_ratios_are_their_sect43_values(laws):
    for key, b69, b33, ev, target in (("6h-pyrene", 1448.8, 3028.0, 4.0, 1.08), ("me-pyrene", 1448.6, 3049.9, 6.0, 0.19)):
        em = D.run(key, ev)
        assert em.energy(b69) / em.energy(b33) == pytest.approx(target, rel=0.05)
