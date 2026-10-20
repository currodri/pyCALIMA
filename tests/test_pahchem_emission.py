"""IR emission from the microcanonical cascade (pahchem.emission), M7a.

The gates:
- energy closure;
- the AmesPAHdb suite's own cascade model on the same PAHdb v4.00 modes, as band
  fractions (ames_suite_cascade_band_fractions.csv, from
  diagnostics/pahchem/make_pahdb_cascade_reference.py);
- the Monte Carlo of pahchem.montecarlo;
- the steady state's radiated power.
"""

import numpy as np
import pandas as pd
import pytest

from pycalima._paths import get_external_data_path
from pycalima.pahchem import dos, emission, energy_me, fields, photo
from pycalima.pahchem.molecule import PAHSpecies

EV = dos.CM_PER_EV
BANDS = {"3.3": (2950, 3200), "6.2": (1500, 1700), "7.7": (1150, 1500), "8.6": (1000, 1150),
         "11.2": (850, 930), "12.7": (700, 850), "far-IR": (0, 700)}


def _species(model_data, formula):
    path = model_data / "PAH_states" / f"{formula}_0.dat"
    if not path.exists():
        pytest.skip(f"model_data/PAH_states/ has no {path.name}")
    return PAHSpecies.from_states_file(path)


def _cascade(sp, e_max_ev=13.0, de_cm=20.0):
    return energy_me.CascadeModel(sp.freqs_cm, sp.einstein_a, e_max_cm=e_max_ev * EV, de_cm=de_cm)


def _bands(freqs, energy):
    return {b: energy[(freqs >= lo) & (freqs < hi)].sum() for b, (lo, hi) in BANDS.items()}


@pytest.mark.parametrize("formula", ["C24H12", "C54H18"])
def test_cascade_photons_conserve_energy(model_data, formula):
    """sum_i h nu_i N_i(E) = E less what is left when no mode can emit (< 150 cm^-1;
    measured 66 cm^-1 for C24H12 at 8 eV)."""
    sp = _species(model_data, formula)
    cm = _cascade(sp)
    N = emission.cascade_photons(cm)
    for e in (2.0, 8.0, 12.0):
        j = int(round(e * EV / cm.de_cm))
        assert 0.0 <= cm.energy_cm[j] - sp.freqs_cm @ N[:, j] < 150.0


@pytest.mark.parametrize("formula", ["C24H12", "C54H18"])
def test_cascade_against_the_pahdb_thermal_cascade(model_data, formula):
    """Fraction of E emitted per band, against Transitions.cascade(E) of the AmesPAHdb
    suite (thermal approximation) on the same modes (equal to 0.005 cm^-1; here the
    mode count per band is checked).

    Measured at 4-12 eV: every band but 3.3 um within 1 % (far-IR 4 %). At 3.3 um the
    exact cascade is below the thermal one and converges to it as E grows, as it
    should: ours/PAHdb = 0.96, 0.99, 0.995 (C24H12) and 0.90, 0.97, 0.99 (C54H18) at 4,
    8 and 12 eV. At 2 eV it is 0.84 / 0.75: the thermal approximation overestimates
    high-frequency emission from a small energy content."""
    sp = _species(model_data, formula)
    ref = pd.read_csv(get_external_data_path("pahchem/ames_suite_cascade_band_fractions.csv"), comment="#")
    ref = ref[ref.species == formula]
    r8 = ref[ref.e_ev == 8.0].set_index("band")
    for b, (lo, hi) in BANDS.items():
        assert (lo, hi) == (r8.lo_cm[b], r8.hi_cm[b])
        assert int(((sp.freqs_cm >= lo) & (sp.freqs_cm < hi)).sum()) == r8.n_modes[b], b
    cm = _cascade(sp)
    N = emission.cascade_photons(cm)
    r33 = []
    for e in (4.0, 8.0, 12.0):
        j = int(round(e * EV / cm.de_cm))
        ours = sp.freqs_cm * N[:, j]
        ours = _bands(sp.freqs_cm, ours / ours.sum())
        theirs = ref[ref.e_ev == e].set_index("band").energy_fraction
        for b in BANDS:
            if b != "3.3":
                assert ours[b] == pytest.approx(theirs[b], rel=0.05), (e, b)
        r33.append(ours["3.3"] / theirs["3.3"])
    assert np.all(np.diff(r33) > 0) and r33[-1] > 0.98 and max(r33) < 1.0, r33


@pytest.mark.parametrize("formula", ["C24H12", "C54H18"])
def test_thermal_cascade_is_the_pahdb_suite_model(model_data, formula):
    """emission.ThermalCascade (the suites' model, Maragkoudakis+20 eqs. A1-A4) against
    Transitions.cascade at 2-12 eV: every band with > 0.5 % of E within 1e-3 (measured
    5e-4), and the energy closes."""
    sp = _species(model_data, formula)
    ref = pd.read_csv(get_external_data_path("pahchem/ames_suite_cascade_band_fractions.csv"), comment="#")
    th = emission.ThermalCascade(sp.freqs_cm, sp.einstein_a)
    for e, r in ref[ref.species == formula].groupby("e_ev"):
        em = th.energy_per_mode(e * EV)
        assert em.sum() == pytest.approx(e * EV, rel=1e-9)
        ours = _bands(sp.freqs_cm, em / em.sum())
        for _, row in r[r.energy_fraction > 0.005].iterrows():
            assert ours[row.band] == pytest.approx(row.energy_fraction, rel=1e-3), (e, row.band)


def test_cascade_photons_match_the_monte_carlo(model_data):
    """C24H12 from 6 eV, 3000 Gillespie cascades on the same rates: every band's
    photon count within 3 % (measured <= 1.5 %)."""
    from pycalima.pahchem.montecarlo import MonteCarlo

    sp = _species(model_data, "C24H12")
    cm = _cascade(sp, e_max_ev=8.0)
    j = int(round(6.0 * EV / cm.de_cm))
    me = _bands(sp.freqs_cm, emission.cascade_photons(cm)[:, j])
    mc = _bands(sp.freqs_cm, MonteCarlo(cm, {}, seed=3).cascade_photons(cm.energy_cm[j], 3000))
    for b in BANDS:
        assert mc[b] == pytest.approx(me[b], rel=0.03), (b, mc[b], me[b])


def test_steady_state_emission_is_the_radiated_power(model_data):
    """Multiphoton steady state of C54H18 at G0 = 1e4: sum_i h nu_i R_i = radiated power
    = absorbed power."""
    sp = _species(model_data, "C54H18")
    cm = _cascade(sp, e_max_ev=30.0, de_cm=100.0)
    e = np.linspace(5.0, 13.6, 345)
    h = photo.heating_spectrum(fields.draine1978(g0=1e4), photo.sigma_li_draine(54, 0, e),
                               np.zeros_like(e), e)
    ss = cm.steady_state(e, h, {})
    power = sp.freqs_cm @ emission.steady_state_emission(cm, ss)
    assert power == pytest.approx(ss.radiated_power, rel=1e-9)
    assert power == pytest.approx(ss.absorbed_power, rel=1e-3)


@pytest.mark.parametrize("profile", ["gaussian", "lorentzian"])
def test_spectrum_bands_carry_their_strength(profile):
    grid = np.linspace(0.0, 4000.0, 400001)
    s = emission.spectrum([800.0, 1600.0, 3050.0], [1.0, 2.0, 0.5], grid, fwhm_cm=15.0, profile=profile)
    tol = 1e-6 if profile == "gaussian" else 5e-3          # Lorentzian wings beyond the grid
    assert np.trapezoid(s, grid) == pytest.approx(3.5, rel=tol)
    assert grid[np.argmax(s)] == pytest.approx(1600.0 - 15.0, abs=0.05)


def test_emission_by_energy_splits_the_cascade_photons():
    """Summed over emission bins, the energy-resolved photons are the cascade's, for any start distribution;
    photons come only from bins at or below the start."""
    rng = np.random.default_rng(3)
    freqs = np.sort(rng.uniform(150.0, 3100.0, 40))
    model = energy_me.CascadeModel(freqs, rng.uniform(0.1, 20.0, 40), e_max_cm=6.0 * EV, de_cm=50.0)
    start = np.zeros(model.n)
    start[[60, 200, 400]] = (0.2, 0.5, 0.3)
    X = emission.emission_by_energy(model, start)
    N = emission.cascade_photons(model)
    assert np.allclose(X.sum(axis=1), N @ start, rtol=1e-10, atol=1e-14)
    assert not X[:, 401:].any()
